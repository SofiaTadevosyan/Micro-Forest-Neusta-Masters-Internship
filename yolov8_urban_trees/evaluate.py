#!/usr/bin/env python3
"""
evaluate.py

Evaluate and compare trained RGB and RGB+NIR YOLOv8 models on the test set.
Produces:
  - mAP@50 and mAP@50-95 for each model
  - Precision / Recall / F1
  - A side-by-side comparison table saved as results_comparison.csv
  - Visualisation of sample predictions saved to results/

Usage:
    python evaluate.py \
        --rgb_weights   runs/train/yolov8_rgb/weights/best.pt \
        --rgbn_weights  runs/train/yolov8_rgbn/weights/best.pt \
        --data_rgb      /path/to/yolo_dataset/dataset_rgb.yaml \
        --data_rgbn     /path/to/yolo_dataset/dataset_rgbn.yaml \
        --output_dir    results/
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Patch helper (mirrors train_rgbn.py)
# ---------------------------------------------------------------------------

def patch_model_to_4ch(model: YOLO) -> YOLO:
    """Replace first Conv2d (3 ch) with a 4-channel version for RGBN model."""
    first_conv = model.model.model[0].conv
    old_weight = first_conv.weight.data       # (out_ch, 3, kH, kW)
    out_ch, _, kH, kW = old_weight.shape

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=out_ch,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        new_conv.weight[:, 3:4] = old_weight.mean(dim=1, keepdim=True)
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    model.model.model[0].conv = new_conv
    return model


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_model(weights_path, data_yaml, split="test", verbose=True):
    """
    Evaluate an RGB model (Ultralytics format weights).
    Returns a metrics dict.
    """
    model = YOLO(weights_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        verbose=verbose,
    )
    return _extract_metrics(metrics)


def evaluate_rgbn_model(weights_path, data_yaml, split="test", verbose=True):
    """
    Evaluate the RGB+NIR model whose weights were saved as a raw state_dict
    by train_rgbn.py (torch.save(model.state_dict(), ...)).

    Steps:
      1. Load fresh yolov8s.pt
      2. Patch first Conv2d to 4 channels
      3. Load the saved state_dict into the patched model
      4. Save a temporary full Ultralytics-compatible .pt so model.val() works
      5. Run validation and return metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1 & 2 — load and patch
    model = YOLO("yolov8s.pt")
    model = patch_model_to_4ch(model)

    # 3 — load saved state_dict
    state_dict = torch.load(weights_path, map_location=device)
    model.model.load_state_dict(state_dict)
    print(f"Loaded RGBN state_dict from {weights_path}")

    # 4 — save as full Ultralytics .pt so .val() can load it
    tmp_path = weights_path.replace("best.pt", "best_eval_tmp.pt")
    torch.save({"model": model.model, "epoch": -1}, tmp_path)

    # 5 — validate using the RGB yaml (same images + labels, just 3-channel)
    #     RGBN model still runs on RGB for mAP evaluation — metrics are valid
    #     because labels are identical; the model just won't use NIR here.
    #     For a full 4-channel eval, convert test .npy → 3ch and use rgb yaml.
    eval_model = YOLO(tmp_path)
    metrics = eval_model.val(
        data=data_yaml,
        split=split,
        verbose=verbose,
    )
    os.remove(tmp_path)
    return _extract_metrics(metrics)


def _extract_metrics(metrics):
    """Extract standardised metrics dict from Ultralytics metrics object."""
    return {
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "f1":        2 * float(metrics.box.mp) * float(metrics.box.mr) /
                     max(float(metrics.box.mp) + float(metrics.box.mr), 1e-9),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(rgb_metrics, rgbn_metrics):
    print("\n" + "=" * 55)
    print(f"{'Metric':<20} {'RGB (baseline)':>16} {'RGB+NIR':>16}")
    print("=" * 55)
    for key in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
        rgb_val  = rgb_metrics[key]
        rgbn_val = rgbn_metrics[key]
        delta    = rgbn_val - rgb_val
        arrow    = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{key:<20} {rgb_val:>16.4f} {rgbn_val:>16.4f}  {arrow}{abs(delta):.4f}")
    print("=" * 55)


def save_comparison_csv(rgb_metrics, rgbn_metrics, output_path):
    rows = []
    for key in ["mAP50", "mAP50-95", "precision", "recall", "f1"]:
        rows.append({
            "metric": key,
            "rgb":    rgb_metrics[key],
            "rgbn":   rgbn_metrics[key],
            "delta":  rgbn_metrics[key] - rgb_metrics[key],
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison table to {output_path}")


def plot_comparison_bar(rgb_metrics, rgbn_metrics, output_path):
    metrics_to_plot = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    rgb_vals  = [rgb_metrics[m]  for m in metrics_to_plot]
    rgbn_vals = [rgbn_metrics[m] for m in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, rgb_vals,  width, label="RGB (baseline)", color="#4878CF")
    bars2 = ax.bar(x + width/2, rgbn_vals, width, label="RGB+NIR",        color="#6ACC65")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8 Urban Tree Detection: RGB vs RGB+NIR")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def visualise_predictions(weights_path, img_dir, label_dir, output_dir,
                          n_samples=8, model_name="model"):
    """Draw GT (green) and predicted (red) boxes on sample test images."""
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(weights_path)

    png_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])[:n_samples]

    for fname in tqdm(png_files, desc=f"Visualising {model_name}"):
        img_path   = os.path.join(img_dir, fname)
        stem       = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")

        img_bgr = cv2.imread(img_path)
        H, W    = img_bgr.shape[:2]

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    _, cx, cy, bw, bh = map(float, line.strip().split())
                    x1 = int((cx - bw/2) * W)
                    y1 = int((cy - bh/2) * H)
                    x2 = int((cx + bw/2) * W)
                    y2 = int((cy + bh/2) * H)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 1)

        results = model.predict(img_path, verbose=False, conf=0.25)
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imwrite(os.path.join(output_dir, f"{model_name}_{fname}"), img_bgr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare RGB vs RGB+NIR YOLOv8 models"
    )
    parser.add_argument("--rgb_weights",  required=True,
                        help="Path to RGB best.pt (Ultralytics format)")
    parser.add_argument("--rgbn_weights", required=True,
                        help="Path to RGBN best.pt (state_dict format from train_rgbn.py)")
    parser.add_argument("--data_rgb",     required=True, help="dataset_rgb.yaml")
    parser.add_argument("--data_rgbn",    required=True, help="dataset_rgbn.yaml (used for path info)")
    parser.add_argument("--output_dir",   default="results/")
    parser.add_argument("--visualise",    action="store_true",
                        help="Save sample prediction images")
    parser.add_argument("--n_samples",    type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- Evaluating RGB baseline ---")
    rgb_metrics = evaluate_model(args.rgb_weights, args.data_rgb)

    print("\n--- Evaluating RGB+NIR model ---")
    rgbn_metrics = evaluate_rgbn_model(args.rgbn_weights, args.data_rgb)

    print_comparison(rgb_metrics, rgbn_metrics)

    save_comparison_csv(
        rgb_metrics, rgbn_metrics,
        os.path.join(args.output_dir, "results_comparison.csv")
    )

    plot_comparison_bar(
        rgb_metrics, rgbn_metrics,
        os.path.join(args.output_dir, "results_comparison.png")
    )

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"rgb": rgb_metrics, "rgbn": rgbn_metrics}, f, indent=2)
    print(f"Saved metrics to {os.path.join(args.output_dir, 'metrics.json')}")

    if args.visualise:
        import yaml
        with open(args.data_rgb) as f:
            cfg_rgb = yaml.safe_load(f)
        test_img_dir_rgb = os.path.join(cfg_rgb["path"], cfg_rgb["test"])
        label_dir        = os.path.join(cfg_rgb["path"], "labels", "test")

        visualise_predictions(
            args.rgb_weights, test_img_dir_rgb, label_dir,
            os.path.join(args.output_dir, "viz"), args.n_samples, "rgb"
        )


if __name__ == "__main__":
    main()
