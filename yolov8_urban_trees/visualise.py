#!/usr/bin/env python3
"""
visualise.py

Generates static visualisations for the F13-UC2 report:
  1. rgb_detections_grid.png  — 2x4 grid of 8 test images with GT (green)
                                and predicted (red) bounding boxes
  2. evolution_table.png      — styled table showing model evolution:
                                RGBN v1 (broken) → RGBN v2 (fixed) → RGB baseline

Usage:
    python3 visualise.py \
        --rgb_weights  results/weights/best_rgb.pt \
        --img_dir      yolo_dataset/images/rgb/test \
        --label_dir    yolo_dataset/labels/test \
        --output_dir   results/viz
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gt_boxes(label_path, H, W):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    _, cx, cy, bw, bh = map(float, p)
                    x1 = int((cx - bw / 2) * W)
                    y1 = int((cy - bh / 2) * H)
                    x2 = int((cx + bw / 2) * W)
                    y2 = int((cy + bh / 2) * H)
                    boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(img_bgr, gt_boxes, pred_boxes):
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
    for (x1, y1, x2, y2) in pred_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 1)
    return out


# ---------------------------------------------------------------------------
# Visualisation 1 — Detection grid
# ---------------------------------------------------------------------------

def make_detection_grid(model, img_dir, label_dir, output_path, n=8, conf=0.25):
    png_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    # Pick evenly spaced samples for variety
    step = max(1, len(png_files) // n)
    selected = [png_files[i * step] for i in range(n)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, fname in zip(axes.flatten(), selected):
        img_path = os.path.join(img_dir, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(label_dir, f"{stem}.txt")

        img_bgr = cv2.imread(img_path)
        H, W = img_bgr.shape[:2]

        gt_boxes = load_gt_boxes(label_path, H, W)

        results = model.predict(img_path, verbose=False, conf=conf)
        pred_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy():
                pred_boxes.append(tuple(map(int, box)))

        annotated = draw_boxes(img_bgr, gt_boxes, pred_boxes)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        city = stem.split('_')[0].replace('_', ' ').title()
        year = stem.split('_')[1] if len(stem.split('_')) > 1 else ''
        ax.set_title(f"{city} {year}\nGT: {len(gt_boxes)}  Pred: {len(pred_boxes)}",
                     color='white', fontsize=8, pad=4)
        ax.axis('off')

    # Legend
    gt_patch   = mpatches.Patch(color='#00c800', label='Ground Truth')
    pred_patch = mpatches.Patch(color='#0000dc', label='Predicted')
    fig.legend(handles=[gt_patch, pred_patch], loc='lower center',
               ncol=2, fontsize=11, facecolor='#1a1a2e', labelcolor='white',
               framealpha=0.8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle('YOLOv8 Urban Tree Detection — RGB Model\nTest Set Samples',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualisation 2 — Model evolution table
# ---------------------------------------------------------------------------

def make_evolution_table(output_path):
    data = [
        # Model,          mAP@50,  Precision, Recall,  F1,     Status
        ['RGBN v1\n(custom loop)', '~0.000',  '0.001',   '0.000', '0.000', 'FAILED'],
        ['RGBN v2\n(fixed)',        '0.061',   '0.072',   '0.842', '0.133', 'PARTIAL'],
        ['RGB baseline',            '0.439',   '0.567',   '0.773', '0.654', 'BEST'],
    ]
    columns = ['Model', 'mAP@50', 'Precision', 'Recall', 'F1', 'Status']

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')

    row_colors = [
        ['#3d0000'] * 6,   # v1 — red (failed)
        ['#1a2d1a'] * 6,   # v2 — dark green (partial)
        ['#0d3d0d'] * 6,   # RGB — bright green (best)
    ]
    # Status cell colors
    status_colors = ['#cc0000', '#cc8800', '#00aa00']
    for i, color in enumerate(status_colors):
        row_colors[i][5] = color

    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#2d2d5e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style all cells
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            table[i, j].set_text_props(color='white', fontsize=10)

    ax.set_title('Model Evolution: RGBN v1 → RGBN v2 (fixed) → RGB Baseline',
                 color='white', fontsize=13, fontweight='bold', pad=20)

    # Add arrows showing improvement
    fig.text(0.5, 0.08,
             'Recall improved from 0.000 → 0.842 (+42,000%)   |   '
             'Fix: replaced custom loop with DetectionTrainer subclass',
             ha='center', color='#aaaaaa', fontsize=9)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_weights', required=True)
    parser.add_argument('--img_dir',     required=True)
    parser.add_argument('--label_dir',   required=True)
    parser.add_argument('--output_dir',  default='results/viz')
    args = parser.parse_args()

    print("Loading RGB model...")
    model = YOLO(args.rgb_weights)

    print("Generating detection grid...")
    make_detection_grid(
        model, args.img_dir, args.label_dir,
        os.path.join(args.output_dir, 'rgb_detections_grid.png')
    )

    print("Generating evolution table...")
    make_evolution_table(
        os.path.join(args.output_dir, 'evolution_table.png')
    )

    print("\nDone. Visualisations saved to:", args.output_dir)


if __name__ == '__main__':
    main()
