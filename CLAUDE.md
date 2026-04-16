# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Who is working on this

**Sofya Tadevosyan** — Masters internship student at NEUSTA France. Working on the computer vision / machine learning part of the micro-forest health monitoring project. Not the same team as the previous students (ISAE-SUPAERO PIE project). Sofya is continuing and improving their work independently.

- GitHub: `SofiaTadevosyan`
- Repo: `git@github.com:SofiaTadevosyan/Micro-Forest-Neusta-Masters-Internship.git`
- Local repo path: `~/Documents/NEUSTA/Micro-Forest-Neusta-Masters-Internship/`
- On macOS — use `python3` and `pip3`, NOT `python`/`pip`

---

## Project context

**Client:** NEUSTA France (Damien Appert)
**Site:** Urban micro-forest planted March 2025 on ISAE-SUPAERO campus, Toulouse (Miyawaki method)
**Goal:** Automated health monitoring system combining IoT environmental sensors + computer vision AI

The previous student team (PIE ETE-03, 5 students, Oct 2025 – Mar 2026) built:
1. ClimaTrack IoT sensor pipeline → collects temperature, humidity, PM1.0/2.5/10, sound
2. Python data pipeline (`extract_data.py`, `transform_data.py`, `filter_alarm.py`) computing REW, VPD, ISHS stress indicators with 60-min rolling alarm filter
3. Explored YOLO segmentation on a tiny hand-labelled ground-level dataset — got good visual results but **no quantitative metrics, no mAP scores**
4. Tested DeepForest on Google Maps aerials — failed (domain mismatch)

**Gaps left by previous team (what Sofya is solving):**
- No proper training dataset (only a few hand-labelled ground photos)
- No quantitative evaluation (no mAP, precision, recall)
- Never used the Near-Infrared band (NIR) from aerial imagery — key for vegetation health
- No reproducible, scalable training pipeline

---

## Sofya's problematic and solution

**Problematic:** Find a more powerful and accurate computer vision model for urban tree detection than what the previous team used, train it on a real dataset, and demonstrate measurable improvement.

**Solution chosen: YOLOv8 with RGB+NIR (4-channel input)**

Instead of using only standard 3-channel RGB images, we exploit the 4th Near-Infrared band present in the NAIP aerial dataset. NIR is the basis of NDVI (Normalized Difference Vegetation Index) — the standard remote sensing measure of vegetation health. Healthy trees reflect strongly in NIR; stressed trees reflect less. This band was never used by the previous team.

**Experiment design:**
- **Baseline:** YOLOv8s trained on RGB only (3 channels)
- **Improvement:** YOLOv8s trained on RGB+NIR (4 channels) — first conv layer patched to accept 4 channels, NIR filter initialised as mean of RGB filters
- **Comparison:** mAP@50, mAP@50-95, Precision, Recall, F1 on the held-out test set
- This gives a concrete, quantitative, scientifically justified improvement story

---

## Full folder structure on this machine

```
~/Documents/NEUSTA/                                        ← parent folder (NOT a git repo)
│
├── Micro-Forest-Neusta-Masters-Internship/                ← THIS GIT REPO
│   ├── CLAUDE.md                                          ← this file
│   ├── README.md
│   ├── .gitignore
│   └── yolov8_urban_trees/                               ← all ML code (Sofya's work)
│       ├── convert_annotations.py
│       ├── train_rgb.py
│       ├── train_rgbn.py
│       ├── evaluate.py
│       ├── requirements.txt
│       └── README.md
│
├── Codes/                                                 ← previous team's code (NOT in git)
│   └── L_3_2_1-SysColl-Python_pipeline_physical_data/
│       ├── main.py
│       ├── extract_data.py
│       ├── transform_data.py
│       ├── filter_alarm.py
│       └── plot_indicators.py
│
├── Dataset/                                               ← dataset (NOT in git — too large)
│   └── urban-tree-detection-data-main/
│       ├── images/     (1,651 × .tif files, 256×256 px, 4-band NAIP aerial)
│       ├── csv/        (1,645 × point annotations: x,y pixel coords per tree)
│       ├── json/       (1,645 × GeoJSON georeferenced annotations)
│       ├── scripts/deepforest/   (DeepForest scripts from original paper)
│       ├── train.txt   (1,336 image names)
│       ├── val.txt     (149 image names)
│       └── test.txt    (166 image names)
│
└── Livrable/                                              ← project reports (NOT in git)
    └── (PDFs, PPTX, DOCX from previous team)
```

---

## Dataset details

**Source:** Ventura et al. (2024), *Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery*, Int. J. of Applied Earth Observation and Geoinformation, 130, 103848.

| Property | Value |
|---|---|
| Total images | 1,651 TIFF files |
| Image size | 256 × 256 px, 60 cm/pixel ground resolution |
| Bands | Band 0=Red, Band 1=Green, Band 2=Blue, Band 3=NIR |
| Data type | uint8 (0–255 per band) |
| Total annotated trees | 96,547 |
| Annotation type | Point (x, y) pixel coordinates — NOT bounding boxes |
| Images with no trees | 6 (expected, treated as empty labels) |
| Cities | 8 cities in California (Chico, Claremont, Long Beach, Palm Springs, Riverside, Santa Monica, Bishop, Eureka) |
| Years | 2016, 2018, 2020 |
| Train/val/test | 1,336 / 149 / 166 |

**Key detail:** annotations are points, not boxes. `convert_annotations.py` converts them to YOLO bounding boxes using a fixed radius (default 15 px → 30×30 px boxes). This is the same approach used in the original paper's scripts (`make_deepforest_data.py`).

---

## What has been done so far (in order)

### 1. CLAUDE.md created for the parent NEUSTA folder
File: `~/Documents/NEUSTA/CLAUDE.md`
Covers the previous team's sensor pipeline code.

### 2. All ML code written and committed to GitHub
Four Python scripts in `yolov8_urban_trees/`:

**`convert_annotations.py`**
- Reads all 1,651 NAIP TIFF files using `rasterio`
- Extracts RGB → saves as PNG (for standard YOLOv8)
- Extracts RGBN → saves as `.npy` float32 normalised to [0,1] (for 4-channel model)
- Converts point annotations → YOLO format bounding boxes (`class cx cy w h`, normalised)
- Writes `dataset_rgb.yaml` and `dataset_rgbn.yaml` for Ultralytics
- Run once before training; output goes to `yolo_dataset/` (gitignored)

**`train_rgb.py`**
- Trains `yolov8s.pt` on 3-channel RGB PNG images
- Standard Ultralytics `.train()` call with aerial-appropriate augmentation (90° rotation, vertical flip, mosaic)
- Output: `runs/train/yolov8_rgb/weights/best.pt`

**`train_rgbn.py`**
- Loads `yolov8s.pt`, patches first Conv2d from 3→4 input channels
- NIR filter initialised as mean of pretrained RGB filters (not random)
- Trains on 4-channel `.npy` images
- Output: `runs/train/yolov8_rgbn/weights/best.pt`

**`evaluate.py`**
- Runs Ultralytics `.val()` on both models against test set
- Prints side-by-side comparison: mAP@50, mAP@50-95, Precision, Recall, F1
- Saves `results/results_comparison.csv`, `results/results_comparison.png`, `results/metrics.json`
- Optional `--visualise` flag: draws GT boxes (green) and predicted boxes (red) on sample images

### 3. Dataset explored and confirmed ready
All 1,651 images verified present. All split files verified correct. CSV format confirmed compatible with `convert_annotations.py`. Dataset is at `~/Documents/NEUSTA/Dataset/urban-tree-detection-data-main/` and does NOT need to be downloaded.

### 4. macOS Python issue resolved
Sofya's Mac uses `python3` and `pip3` — NOT `python`/`pip`. All scripts have `#!/usr/bin/env python3` shebang added.

---

## Current status — what still needs to be done

- [ ] **Step 1:** Install dependencies → `pip3 install -r yolov8_urban_trees/requirements.txt`
- [ ] **Step 2:** Run annotation converter → converts dataset to YOLO format (takes ~5 min, CPU only)
- [ ] **Step 3:** Train RGB baseline → needs GPU (Google Colab recommended)
- [ ] **Step 4:** Train RGB+NIR model → needs GPU (Google Colab recommended)
- [ ] **Step 5:** Run evaluation → compare mAP scores between both models
- [ ] **Step 6 (optional):** Create Colab notebook for training (Claude can write this)

---

## How to run everything (macOS terminal, from repo root)

```bash
# Step 1 — install
pip3 install -r yolov8_urban_trees/requirements.txt

# Step 2 — convert dataset (run once, ~5 min)
python3 yolov8_urban_trees/convert_annotations.py --dataset_dir ~/Documents/NEUSTA/Dataset/urban-tree-detection-data-main --output_dir yolo_dataset --radius 15

# Step 3 — train RGB baseline
python3 yolov8_urban_trees/train_rgb.py --data yolo_dataset/dataset_rgb.yaml --epochs 100 --batch 16 --name yolov8_rgb

# Step 4 — train RGB+NIR
python3 yolov8_urban_trees/train_rgbn.py --data yolo_dataset/dataset_rgbn.yaml --epochs 100 --batch 16 --name yolov8_rgbn

# Step 5 — evaluate and compare
python3 yolov8_urban_trees/evaluate.py --rgb_weights runs/train/yolov8_rgb/weights/best.pt --rgbn_weights runs/train/yolov8_rgbn/weights/best.pt --data_rgb yolo_dataset/dataset_rgb.yaml --data_rgbn yolo_dataset/dataset_rgbn.yaml --output_dir results/ --visualise
```

---

## Key technical decisions and why

| Decision | Reason |
|---|---|
| YOLOv8 not DeepForest | DeepForest failed on this project (domain mismatch). YOLOv8 was already validated by the previous team on ground-level photos. |
| Aerial dataset not ground photos | 1,651 images / 96,547 trees gives proper quantitative evaluation. Previous team had no real metrics. |
| YOLOv8s (small) variant | Good balance of speed and accuracy; can be trained on Colab free tier in ~1-2h |
| Radius=15 for box conversion | 30×30 px box at 60 cm/pixel = ~18m diameter — reasonable for urban tree crowns at this scale. Same approach as the original paper's scripts. |
| NIR filter init = mean(RGB) | Avoids random initialisation for the new channel; gives model a sensible starting point for the NIR signal |
| `yolo_dataset/` gitignored | 1,651 PNGs + 1,651 NPYs would be ~1 GB — should not be in git |
| `runs/` gitignored | Model weights (`.pt` files) are large binary files — should not be in git |
