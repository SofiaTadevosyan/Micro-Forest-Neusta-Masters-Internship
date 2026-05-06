#!/usr/bin/env python3
"""
train_rgbn.py

Train YOLOv8 on the urban tree dataset using 4-channel RGB+NIR images.

Strategy:
  - Subclasses Ultralytics DetectionTrainer to inject 4-channel support
    while inheriting the full training infrastructure:
    EMA, warm-up scheduling, TaskAlignedAssigner, DetectionValidator,
    cosine LR scheduler, gradient clipping, checkpointing.
  - Only 3 methods are overridden: get_model(), build_dataset(), get_dataloader()
  - .npy RGBN files (H,W,4 float32 [0,1]) are loaded by RGBNDataset.
  - YOLOv8s first Conv2d is patched from 3→4 channels after model load.
    NIR filter initialised as mean of pretrained RGB filters.
  - Weights saved in full Ultralytics format → load with YOLO(path)

Usage:
    python3 train_rgbn.py \
        --data    /path/to/yolo_dataset/dataset_rgbn.yaml \
        --epochs  50 \
        --batch   16 \
        --name    yolov8_rgbn \
        --project runs/train \
        --device  cuda

Trained weights → runs/train/yolov8_rgbn/weights/best.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RGBNDataset(Dataset):
    """
    Loads .npy RGBN tiles (H,W,4) float32 [0,1] and YOLO .txt labels.

    Returns a dict per sample — matching the format expected by the
    Ultralytics trainer's collate_fn and preprocess_batch:
        img       : tensor (4, H, W) float32  [0, 1]
        cls       : tensor (N,)      int64
        bboxes    : tensor (N, 4)    float32  [cx, cy, w, h] normalised
        batch_idx : tensor (N,)      float32  (filled with 0; collate adds img idx)
    """

    def __init__(self, img_dir, label_dir, augment=False):
        self.augment   = augment
        self.img_paths = sorted(
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.endswith(".npy")
        )
        self.label_dir = label_dir

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(path))[0]

        img = np.load(path)   # (H, W, 4) float32 [0, 1]

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()   # horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()   # vertical flip
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k=k, axes=(0, 1)).copy()

        img = torch.from_numpy(img).permute(2, 0, 1)   # (4, H, W)

        lp = os.path.join(self.label_dir, f"{stem}.txt")
        cls_list, box_list = [], []
        if os.path.exists(lp):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_list.append(float(parts[0]))
                        box_list.append([float(p) for p in parts[1:]])

        n = len(cls_list)
        cls     = torch.tensor(cls_list, dtype=torch.int64)
        bboxes  = torch.tensor(box_list, dtype=torch.float32) if n else torch.zeros((0, 4))
        batch_idx = torch.zeros(n, dtype=torch.float32)

        return {
            "img":       img,
            "cls":       cls,
            "bboxes":    bboxes,
            "batch_idx": batch_idx,
            "im_file":   path,   # required by plot_training_samples
        }

    @staticmethod
    def collate_fn(batch):
        """Collate list of sample dicts into a batched dict."""
        imgs      = torch.stack([b["img"]   for b in batch])
        cls       = torch.cat ([b["cls"]    for b in batch])
        bboxes    = torch.cat ([b["bboxes"] for b in batch])
        im_files  = [b["im_file"] for b in batch]
        # batch_idx: which image in the batch each box belongs to
        batch_idx = torch.cat([
            torch.full((len(b["cls"]),), i, dtype=torch.float32)
            for i, b in enumerate(batch)
        ])
        # ori_shape and resized_shape required by DetectionValidator._prepare_batch
        # All images are 256x256 with no letterboxing (ratio_pad = identity)
        H, W = imgs.shape[2], imgs.shape[3]
        ori_shape     = [(H, W)] * len(batch)
        resized_shape = [(H, W)] * len(batch)
        ratio_pad     = [((1.0, 1.0), (0.0, 0.0))] * len(batch)
        return {
            "img":          imgs,
            "cls":          cls,
            "bboxes":       bboxes,
            "batch_idx":    batch_idx,
            "im_file":      im_files,
            "ori_shape":    ori_shape,
            "resized_shape": resized_shape,
            "ratio_pad":    ratio_pad,
        }


# ---------------------------------------------------------------------------
# Patch first Conv2d: 3 → 4 input channels
# ---------------------------------------------------------------------------

def patch_model_to_4ch(model):
    """Replace first Conv2d (3-ch) with a 4-channel version. NIR = mean(RGB)."""
    first_conv = model.model[0].conv
    old_w  = first_conv.weight.data   # (out_ch, 3, kH, kW)
    out_ch = old_w.shape[0]

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_ch,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_w
        new_conv.weight[:, 3:4] = old_w.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    model.model[0].conv = new_conv
    print("Patched first Conv2d: 3 → 4 channels (NIR filter = mean of RGB filters)")
    return model


# ---------------------------------------------------------------------------
# Trainer subclass — injects 4-channel support into full Ultralytics pipeline
# ---------------------------------------------------------------------------

class RGBNDetectionTrainer(DetectionTrainer):
    """
    DetectionTrainer subclass for 4-channel RGB+NIR training.

    Inherits: EMA, warm-up scheduling, TaskAlignedAssigner loss,
    DetectionValidator (real mAP/Recall each epoch), checkpointing, logging.

    Overrides only: get_model(), build_dataset(), get_dataloader()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load YOLOv8s and patch first Conv2d to 4-channel input."""
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK

        # Build standard model with 3 channels, then patch
        model = DetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=3,   # start as 3-ch; we patch immediately after
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)

        patch_model_to_4ch(model)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Return RGBNDataset instead of standard YOLODataset."""
        # Labels are always at <dataset_root>/labels/<split>
        # img_path is e.g. /content/.../images/rgbn/train
        # Derive label_dir: replace /images/rgbn/ → /labels/
        label_dir = img_path.replace("/images/rgbn/", "/labels/").replace(
            "\\images\\rgbn\\", "\\labels\\"
        )
        return RGBNDataset(img_path, label_dir, augment=(mode == "train"))

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build DataLoader using RGBNDataset with custom collate_fn."""
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = (mode == "train")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers,
            collate_fn=RGBNDataset.collate_fn,
            pin_memory=False,
            drop_last=(mode == "train"),
        )

    def preprocess_batch(self, batch):
        """Move batch to device. Images are already float32 [0,1] — skip /255."""
        device = self.device
        batch["img"]       = batch["img"].to(device, non_blocking=True)
        batch["cls"]       = batch["cls"].to(device)
        batch["bboxes"]    = batch["bboxes"].to(device)
        batch["batch_idx"] = batch["batch_idx"].to(device)
        # Do NOT divide by 255 — NPY images are already [0, 1]
        # (parent would do /255 and break our data)
        return batch

    def plot_training_samples(self, batch, ni):
        """Skip plotting — our batch uses NPY paths, not standard image files."""
        pass

    def plot_training_labels(self):
        """Skip — RGBNDataset has no .labels attribute (not a YOLODataset)."""
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RGB+NIR urban tree data")
    parser.add_argument("--data",    required=True,        help="Path to dataset_rgbn.yaml")
    parser.add_argument("--model",   default="yolov8s.pt", help="YOLOv8 model variant")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--lr",      type=float, default=0.01)
    parser.add_argument("--name",    default="yolov8_rgbn")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--device",  default="cuda",
                        help="cuda, mps, or cpu")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    trainer = RGBNDetectionTrainer(overrides={
        "model":    args.model,
        "data":     args.data,
        "epochs":   args.epochs,
        "batch":    args.batch,
        "imgsz":    256,
        "name":     args.name,
        "project":  args.project,
        "device":   args.device,
        "workers":  args.workers,
        "lr0":      args.lr,
        # Same augmentation as RGB baseline (aerial-appropriate)
        "flipud":   0.5,
        "fliplr":   0.5,
        "degrees":  90,
        "mosaic":   0.0,   # mosaic disabled — incompatible with NPY files
        "plots":    True,
        "save":     True,
        "verbose":  True,
    })

    trainer.train()
    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
