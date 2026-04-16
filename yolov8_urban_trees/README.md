# YOLOv8 Urban Tree Detection — RGB vs RGB+NIR

This module trains and evaluates YOLOv8 for urban tree detection from aerial imagery,
comparing a standard 3-channel RGB baseline against a 4-channel RGB+NIR model.

## Scientific motivation

The dataset (Ventura et al., 2024) contains 4-band NAIP aerial imagery: Red, Green,
Blue, and **Near-Infrared (NIR)**. NIR is the spectral band used to compute NDVI
(Normalized Difference Vegetation Index), the standard measure of vegetation health.
Trees reflect strongly in NIR when healthy and reflect less when stressed or dying.

The previous team's YOLO work used only ground-level RGB photos from a tiny
hand-labelled dataset. This work:
1. Uses the full 1,651-image / 95,972-tree aerial dataset
2. Trains on RGB (baseline) and RGB+NIR (improved) with the same architecture
3. Produces quantitative mAP metrics for direct comparison

## Dataset

The `urban-tree-detection-data-main` dataset must be downloaded separately.
- 1,651 images (256×256 px, 60 cm/pixel NAIP aerial)
- 95,972 annotated trees across 8 Californian cities
- Annotations: point (x, y) per tree in CSV files
- Pre-defined train/val/test splits (1336/149/166)
- Source: Ventura et al. (2024), *International Journal of Applied Earth Observation and Geoinformation*

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Convert annotations

Point annotations → YOLO bounding boxes (30×30 px boxes, i.e. radius=15).
Also exports RGB PNGs and RGBN numpy arrays.

```bash
python convert_annotations.py \
    --dataset_dir /path/to/urban-tree-detection-data-main \
    --output_dir  /path/to/yolo_dataset \
    --radius      15
```

## Step 2 — Train RGB baseline

```bash
python train_rgb.py \
    --data    /path/to/yolo_dataset/dataset_rgb.yaml \
    --model   yolov8s.pt \
    --epochs  100 \
    --batch   16 \
    --name    yolov8_rgb
```

## Step 3 — Train RGB+NIR model

```bash
python train_rgbn.py \
    --data    /path/to/yolo_dataset/dataset_rgbn.yaml \
    --model   yolov8s.pt \
    --epochs  100 \
    --batch   16 \
    --name    yolov8_rgbn
```

## Step 4 — Evaluate and compare

```bash
python evaluate.py \
    --rgb_weights   runs/train/yolov8_rgb/weights/best.pt \
    --rgbn_weights  runs/train/yolov8_rgbn/weights/best.pt \
    --data_rgb      /path/to/yolo_dataset/dataset_rgb.yaml \
    --data_rgbn     /path/to/yolo_dataset/dataset_rgbn.yaml \
    --output_dir    results/ \
    --visualise
```

This produces:
- `results/results_comparison.csv` — metrics table
- `results/results_comparison.png` — bar chart (RGB vs RGB+NIR)
- `results/metrics.json` — raw scores for reporting
- `results/viz/` — sample images with GT (green) and predicted (red) boxes

## Expected output format

```
=======================================================
Metric               RGB (baseline)          RGB+NIR
=======================================================
mAP50                        0.XXX            0.XXX  ↑X.XXX
mAP50-95                     0.XXX            0.XXX  ↑X.XXX
precision                    0.XXX            0.XXX  ↑X.XXX
recall                       0.XXX            0.XXX  ↑X.XXX
f1                           0.XXX            0.XXX  ↑X.XXX
=======================================================
```

## Model architecture note

YOLOv8's first convolutional layer is patched to accept 4 input channels.
The NIR filter weights are initialised as the mean of the pretrained RGB
filters — this avoids random initialisation and gives the model a sensible
starting point for the new channel.

## Citation

```
Ventura, J. et al. (2024). Individual Tree Detection in Large-Scale Urban Environments
using High-Resolution Multispectral Imagery. International Journal of Applied Earth
Observation and Geoinformation, 130, 103848.
```
