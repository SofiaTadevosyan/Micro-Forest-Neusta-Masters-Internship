"""
train_rgb.py

Train YOLOv8 on the urban tree dataset using standard 3-channel RGB images.
This is the baseline experiment.

Usage:
    python train_rgb.py \
        --data    /path/to/yolo_dataset/dataset_rgb.yaml \
        --model   yolov8s.pt \
        --epochs  100 \
        --imgsz   256 \
        --batch   16 \
        --name    yolov8_rgb \
        --project runs/train

The trained weights will be saved to:
    runs/train/yolov8_rgb/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on RGB urban tree data")
    parser.add_argument("--data",    required=True,      help="Path to dataset_rgb.yaml")
    parser.add_argument("--model",   default="yolov8s.pt", help="YOLOv8 model variant (n/s/m/l/x)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=256)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--name",    default="yolov8_rgb")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--device",  default="",         help="cuda device (0,1,..) or cpu")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        device=args.device,
        # Augmentation — important for a small-object aerial dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,    # vertical flip (valid for aerial views)
        fliplr=0.5,
        degrees=90,    # rotation (trees look the same from any angle)
        translate=0.1,
        scale=0.2,
        # Mosaic helps with dense small-object detection
        mosaic=1.0,
        # Anchor-free, so no anchor tuning needed
        close_mosaic=10,
        # Logging
        plots=True,
        save=True,
        verbose=True,
    )

    print(f"\nTraining complete. Best weights: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
