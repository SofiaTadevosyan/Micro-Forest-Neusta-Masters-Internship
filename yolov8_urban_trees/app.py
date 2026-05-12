#!/usr/bin/env python3
"""
app.py — Urban Micro-Forest Tree Detection Demo
Run: streamlit run yolov8_urban_trees/app.py
"""

import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from health_analysis import (
    load_rgbn, compute_ndvi, classify_health,
    draw_health_boxes, health_summary, parse_filename
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌳 Urban Tree Detection",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

/* ── Sidebar toggle arrow ── */
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-iiif1v,
section[data-testid="stSidebarCollapsedControl"] {
    background-color: #4a7a4a !important;
    border-radius: 0 8px 8px 0 !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] path {
    fill: white !important; stroke: white !important; color: white !important;
}

/* ══════════════════════════════════════════════
   LIGHT THEME — muted sage green
══════════════════════════════════════════════ */
.stApp, body                { background: #eef5ee !important; color: #1a3a1a !important; }
[data-testid="stSidebar"]   { background: linear-gradient(180deg,#c8dfc8 0%,#bfd8bf 100%) !important; border-right: 1px solid #7aa07a !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #1a3a1a !important; }

.sidebar-section  { font-size:0.7rem; font-weight:700; color:#2a5a2a !important; text-transform:uppercase; letter-spacing:1.2px; margin:16px 0 8px 0; white-space:nowrap; }
.sb-metric        { display:flex; justify-content:space-between; align-items:center; background:#b8d0b8; border:1px solid #7aa07a; border-radius:8px; padding:7px 12px; margin-bottom:6px; gap:8px; }
.sb-metric-label  { font-size:0.78rem; color:#1a4a1a; white-space:nowrap; }
.sb-metric-value  { font-size:0.9rem; font-weight:700; color:#1a5a1a; white-space:nowrap; flex-shrink:0; }

.page-header      { display:flex; align-items:center; justify-content:space-between; padding:18px 32px 14px 32px; border-bottom:1px solid #7aa07a; margin-bottom:20px; flex-wrap:nowrap; gap:16px; }
.page-header-left { display:flex; align-items:center; gap:12px; min-width:0; }
.page-header-icon { font-size:1.6rem; flex-shrink:0; }
.page-header-title { font-size:1.25rem; font-weight:600; color:#1a4a1a; white-space:nowrap; letter-spacing:-0.2px; }
.page-header-sub  { font-size:0.78rem; color:#1a4a1a; white-space:nowrap; margin-top:2px; }
.page-header-stats { display:flex; gap:28px; align-items:center; flex-shrink:0; }
.page-header-stat { text-align:center; }
.page-header-stat-value { font-size:1rem; font-weight:700; color:#1a5a1a; white-space:nowrap; }
.page-header-stat-label { font-size:0.62rem; color:#1a4a1a; text-transform:uppercase; letter-spacing:0.5px; white-space:nowrap; }
.page-header-badge { background:rgba(40,100,40,0.2); border:1px solid #4a8a4a; border-radius:20px; padding:4px 12px; font-size:0.72rem; color:#1a5a1a; font-weight:500; white-space:nowrap; flex-shrink:0; }

.metric-card      { flex:1; background:linear-gradient(135deg,#c4d8c4,#b8d0b8); border:1px solid #7aa07a; border-radius:12px; padding:16px; text-align:center; }
.metric-number    { font-size:2rem; font-weight:700; color:#1a5a1a; line-height:1; white-space:nowrap; }
.metric-unit      { font-size:0.75rem; color:#2a5a2a; margin-top:6px; text-transform:uppercase; letter-spacing:0.5px; white-space:nowrap; }
.section-title    { font-size:0.8rem; font-weight:600; color:#2a5a2a; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; white-space:nowrap; }
.divider          { border:none; border-top:1px solid #7aa07a; margin:28px 0; }

[data-testid="stFileUploaderDropzone"] { background:#c8dfc8 !important; border:2px dashed #4a8a4a !important; border-radius:12px !important; }

/* Deploy button — matches YOLOv8s · RGB badge style */
[data-testid="stToolbarActionButton"],
[data-testid="stActionButton"],
button[kind="headerNoPadding"],
.stDeployButton button,
header button[data-testid] {
    background-color: rgba(40,100,40,0.2) !important;
    color: #1a5a1a !important;
    border: 1px solid #4a8a4a !important;
    border-radius: 20px !important;
}
/* Slider track bar — colored green */
.stSlider > div > div > div > div[role="slider"] + div,
.stSlider [data-baseweb="slider"] > div > div { background:#3a7a3a !important; }

/* Hide Streamlit's built-in tick bar labels (replaced with static HTML below) */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}
/* Always show the thumb value tooltip */
.stSlider [role="slider"] > div,
.stSlider [data-baseweb="tooltip"],
.stSlider [data-baseweb="slider"] [role="slider"] + div {
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}

/* Reset ALL stSlider divs, then re-apply track color only */
.stSlider > div > div > div { background: transparent !important; }
.stSlider [data-baseweb="slider"] [role="progressbar"] { background: #3a7a3a !important; }
.stImage > div > div                  { color:#2a5a2a !important; font-size:0.8rem !important; }
p, span, li { color: #1a3a1a; }
[data-testid="stMarkdownContainer"] p { color: #1a3a1a !important; }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT     = os.path.dirname(SCRIPT_DIR)
RGB_WEIGHTS   = os.path.join(REPO_ROOT, 'results', 'weights', 'best_rgb.pt')
TEST_DIR      = os.path.join(REPO_ROOT, 'yolo_dataset', 'images', 'rgb', 'test')
TEST_DIR_RGBN = os.path.join(REPO_ROOT, 'yolo_dataset', 'images', 'rgbn', 'test')

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def run_detection(img_array, model, conf):
    img_bgr  = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results  = model.predict(img_bgr, verbose=False, conf=conf)
    boxes    = results[0].boxes
    n        = 0
    avg_conf = 0.0
    out      = img_bgr.copy()
    box_list = []

    if boxes is not None and len(boxes) > 0:
        n        = len(boxes)
        avg_conf = float(boxes.conf.mean().item())
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            box_list.append((x1, y1, x2, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80), 2)
            cv2.rectangle(out, (x1, y1), (x1+5, y1+5), (0, 220, 80), -1)
            cv2.rectangle(out, (x2-5, y2-5), (x2, y2), (0, 220, 80), -1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), n, avg_conf, box_list

# ---------------------------------------------------------------------------
# Hero + stats bar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <div class="page-header-left">
        <div class="page-header-icon">🌳</div>
        <div>
            <div class="page-header-title">Urban Micro-Forest Tree Detection</div>
            <div class="page-header-sub">YOLOv8 · NDVI Health Analysis · NEUSTA Monitoring System</div>
        </div>
    </div>
    <div class="page-header-stats">
        <div class="page-header-stat">
            <div class="page-header-stat-value">0.439</div>
            <div class="page-header-stat-label">mAP@50</div>
        </div>
        <div class="page-header-stat">
            <div class="page-header-stat-value">77.3%</div>
            <div class="page-header-stat-label">Recall</div>
        </div>
        <div class="page-header-stat">
            <div class="page-header-stat-value">56.7%</div>
            <div class="page-header-stat-label">Precision</div>
        </div>
        <div class="page-header-stat">
            <div class="page-header-stat-value">96,547</div>
            <div class="page-header-stat-label">Annotated Trees</div>
        </div>
    </div>
    <div class="page-header-badge">YOLOv8s · RGB</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Logo / title ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 4px 0 10px 0;">
        <div style="font-size:2.8rem;">🌳</div>
        <div style="font-size:1rem; font-weight:700; color:#1a4a1a; margin-top:6px;">
            Tree Detection
        </div>
        <div style="font-size:0.72rem; color:#1a4a1a; margin-top:2px;">
            NEUSTA · YOLOv8s · RGB
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #1e3a1e; margin:8px 0 12px 0;">
    """, unsafe_allow_html=True)

    # ── 1. Detection settings ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🎯 Detection Settings</div>', unsafe_allow_html=True)

    conf_pct = st.slider(
        "Confidence Threshold",
        min_value=1, max_value=90,
        value=25, step=1,
        format="%d%%",
        help="Minimum certainty (%) for a detection to be shown. Lower = more boxes. Higher = only certain trees.",
        label_visibility="visible",
    )
    conf_threshold = conf_pct / 100.0

    # Live feedback on threshold choice
    if conf_threshold < 0.15:
        st.markdown("<div style='font-size:0.75rem;color:#ddaa33;'>⚠️ Very low — many false detections</div>", unsafe_allow_html=True)
    elif conf_threshold > 0.60:
        st.markdown("<div style='font-size:0.75rem;color:#dd8844;'>⚠️ Very high — may miss trees</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.75rem;color:#5aaa5a;'>✓ Good range for this model</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.73rem;color:#1a4a1a;line-height:1.7;margin-top:8px;">
        The model scores each detection 0–100% — how certain it is that box contains a real tree.<br>
        Only detections <b style="color:#2a5a2a;">at or above this %</b> are shown.<br><br>
        <b style="color:#2a5a2a;">Lower %</b> → more boxes, more false positives<br>
        <b style="color:#2a5a2a;">Higher %</b> → fewer boxes, only high-certainty trees
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2. Model performance ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-metric"><span class="sb-metric-label">mAP@50</span><span class="sb-metric-value">0.439</span></div>
    <div class="sb-metric"><span class="sb-metric-label">Recall</span><span class="sb-metric-value">77.3%</span></div>
    <div class="sb-metric"><span class="sb-metric-label">Precision</span><span class="sb-metric-value">56.7%</span></div>
    <div class="sb-metric"><span class="sb-metric-label">F1 Score</span><span class="sb-metric-value">0.654</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3. Dataset info ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🗂️ Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-metric"><span class="sb-metric-label">Training images</span><span class="sb-metric-value">1,651</span></div>
    <div class="sb-metric"><span class="sb-metric-label">Annotated trees</span><span class="sb-metric-value">96,547</span></div>
    <div class="sb-metric"><span class="sb-metric-label">Image size</span><span class="sb-metric-value">256 px</span></div>
    <div class="sb-metric"><span class="sb-metric-label">Resolution</span><span class="sb-metric-value">60 cm/px</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4. How to use ─────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">💡 How to Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#2a5a2a; line-height:1.7;">
    1. Upload PNG/JPG → tree detection<br>
    2. Upload .npy → detection + NDVI health<br>
    3. Upload 2 .npy → temporal comparison<br>
    4. Or pick a demo sample below<br>
    5. Adjust confidence threshold as needed
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style="border:none; border-top:1px solid #1e3a1e; margin:20px 0 10px 0;">
    <div style="font-size:0.7rem; color:#2a5a2a; text-align:center;">
        NEUSTA Micro-Forest Monitoring<br>
        Sofya Tadevosyan · May 2026
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Check weights
# ---------------------------------------------------------------------------
if not os.path.exists(RGB_WEIGHTS):
    st.error(f"Model weights not found: `{RGB_WEIGHTS}`")
    st.stop()

model = load_model(RGB_WEIGHTS)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
import pandas as pd
from collections import defaultdict

def analyse_single(arr, fname):
    rgb = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    _, _, _, boxes = run_detection(rgb, model, conf_threshold)
    ndvi_vals  = compute_ndvi(arr, boxes)
    annotated  = draw_health_boxes(rgb, boxes, ndvi_vals)
    summ       = health_summary(ndvi_vals)
    meta       = parse_filename(fname)
    return annotated, boxes, ndvi_vals, summ, meta

def show_single_result(annotated, boxes, ndvi_vals, summ, meta, label=None):
    title = label or meta.get('label', '')
    if title:
        st.markdown(f"<div style='font-size:0.8rem;color:#2a5a2a;margin-bottom:4px;'>{title}</div>",
                    unsafe_allow_html=True)
    st.image(annotated, use_container_width=True)
    st.markdown("<div style='font-size:0.75rem;color:#2a5a2a;margin-bottom:8px;'>"
                "🟢 Healthy &nbsp; 🟡 Moderate &nbsp; 🔴 Stressed &nbsp;·&nbsp; NDVI per tree</div>",
                unsafe_allow_html=True)
    if len(boxes) == 0:
        st.warning("No trees detected. Try lowering the confidence threshold.")
        return
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{len(boxes)}</div>
            <div class="metric-unit">Trees</div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#1a5a1a">{summ['pct_healthy']}%</div>
            <div class="metric-unit">🟢 Healthy</div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#ddcc44">{summ['pct_moderate']}%</div>
            <div class="metric-unit">🟡 Moderate</div></div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="color:#dd6644">{summ['pct_stressed']}%</div>
            <div class="metric-unit">🔴 Stressed</div></div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:10px;
                padding:10px 14px;margin-top:10px;display:flex;align-items:baseline;
                gap:10px;flex-wrap:nowrap;">
        <span style="font-size:0.72rem;color:#2a5a2a;text-transform:uppercase;white-space:nowrap;">Mean NDVI</span>
        <span style="font-size:1.4rem;font-weight:700;color:#1a5a1a;white-space:nowrap;">{summ['mean_ndvi']}</span>
        <span style="font-size:0.75rem;color:#1a4a1a;white-space:nowrap;">min {summ['min_ndvi']} · max {summ['max_ndvi']}</span>
    </div>""", unsafe_allow_html=True)

def show_comparison_panel(annotated, boxes, ndvi_vals, summ, meta, label=None):
    """Compact single-result view for side-by-side comparison columns."""
    title = label or meta.get('label', '')
    st.markdown(
        f"<div style='font-size:0.9rem;font-weight:600;color:#1a5a1a;margin-bottom:6px;'>{title}</div>",
        unsafe_allow_html=True)
    st.image(annotated, use_container_width=True)
    st.markdown("<div style='font-size:0.72rem;color:#2a5a2a;margin-bottom:8px;'>"
                "🟢 Healthy &nbsp; 🟡 Moderate &nbsp; 🔴 Stressed</div>",
                unsafe_allow_html=True)
    if len(boxes) == 0:
        st.warning("No trees detected.")
        return
    st.markdown(f"""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:10px;
                padding:10px 14px;margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;color:#2a5a2a;text-transform:uppercase;white-space:nowrap;">Trees</span>
            <span style="font-size:1.1rem;font-weight:700;color:#1a5a1a;white-space:nowrap;">{len(boxes)}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-size:0.7rem;color:#2a5a2a;text-transform:uppercase;white-space:nowrap;">Mean NDVI</span>
            <span style="font-size:1.1rem;font-weight:700;color:#1a5a1a;white-space:nowrap;">{summ['mean_ndvi']}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:0.7rem;color:#1a5a1a;white-space:nowrap;">🟢 Healthy</span>
            <span style="font-size:0.9rem;font-weight:600;color:#1a5a1a;white-space:nowrap;">{summ['pct_healthy']}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <span style="font-size:0.7rem;color:#ddcc44;white-space:nowrap;">🟡 Moderate</span>
            <span style="font-size:0.9rem;font-weight:600;color:#ddcc44;white-space:nowrap;">{summ['pct_moderate']}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.7rem;color:#dd6644;white-space:nowrap;">🔴 Stressed</span>
            <span style="font-size:0.9rem;font-weight:600;color:#dd6644;white-space:nowrap;">{summ['pct_stressed']}%</span>
        </div>
    </div>""", unsafe_allow_html=True)

def show_detection_only(annotated_rgb, n_trees, avg_conf):
    st.image(annotated_rgb, caption="🟩 Detected trees", use_container_width=True)
    if n_trees == 0:
        st.warning("No trees detected. Try lowering the confidence threshold.")
        return
    quality = "Excellent" if avg_conf >= 0.7 else ("Good" if avg_conf >= 0.45 else "Fair")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{n_trees}</div>
            <div class="metric-unit">Trees Found</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number">{avg_conf:.0%}</div>
            <div class="metric-unit">Avg Confidence</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-number" style="font-size:1.2rem">{quality}</div>
            <div class="metric-unit">Detection Quality</div></div>""", unsafe_allow_html=True)

def make_comparison_section(s1_, s2_, b1, b2, m1, m2, label_a=None, label_b=None):
    label_a = label_a or m1.get('label', 'File A')
    label_b = label_b or m2.get('label', 'File B')
    delta_ndvi  = round(s2_['mean_ndvi']  - s1_['mean_ndvi'], 3)
    delta_h     = s2_['pct_healthy']  - s1_['pct_healthy']
    delta_s     = s2_['pct_stressed'] - s1_['pct_stressed']
    delta_trees = len(b2) - len(b1)
    def arrow(v): return f"+{v} ↑" if v > 0 else (f"{v} ↓" if v < 0 else "0 =")
    st.markdown('<div class="section-title" style="margin-top:20px;">Temporal Comparison</div>',
                unsafe_allow_html=True)
    cmp_df = pd.DataFrame([
        {"Metric": "Mean NDVI",      label_a: s1_['mean_ndvi'],  label_b: s2_['mean_ndvi'],  "Change": arrow(delta_ndvi)},
        {"Metric": "% Healthy 🟢",   label_a: f"{s1_['pct_healthy']}%",  label_b: f"{s2_['pct_healthy']}%",  "Change": arrow(delta_h)},
        {"Metric": "% Moderate 🟡",  label_a: f"{s1_['pct_moderate']}%", label_b: f"{s2_['pct_moderate']}%", "Change": arrow(s2_['pct_moderate']-s1_['pct_moderate'])},
        {"Metric": "% Stressed 🔴",  label_a: f"{s1_['pct_stressed']}%", label_b: f"{s2_['pct_stressed']}%", "Change": arrow(delta_s)},
        {"Metric": "Trees Detected", label_a: len(b1),            label_b: len(b2),            "Change": arrow(delta_trees)},
    ])
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    if delta_ndvi > 0.02:
        conclusion = f"Between {label_a} and {label_b}, mean NDVI improved by +{delta_ndvi} — suggesting vegetation recovery."
        icon = "🟢"
    elif delta_ndvi < -0.02:
        conclusion = f"Between {label_a} and {label_b}, mean NDVI declined by {delta_ndvi} — suggesting increasing stress."
        icon = "🔴"
    else:
        conclusion = f"Between {label_a} and {label_b}, vegetation health remained stable (NDVI change: {delta_ndvi})."
        icon = "🟡"
    st.markdown(f"""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:10px;
                padding:12px 16px;margin-top:8px;font-size:0.85rem;color:#2a6a2a;">
        {icon} {conclusion}
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📤  Upload Your Image",
    "🖼️  Demo — Single Image",
    "📅  Demo — Compare Years",
])

# ── TAB 1: Upload ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:12px;
                padding:16px 24px;margin-bottom:20px;">
        <div style="font-size:1rem;font-weight:600;color:#0f2e0f;margin-bottom:4px;">
            📤 Analyse Your Own Aerial Image
        </div>
        <div style="font-size:0.82rem;color:#2a5a2a;line-height:1.7;">
            Upload an aerial image of your site and the AI will automatically detect every tree and assess its health.
            You get a count of detected trees, their health breakdown (healthy / moderate / stressed),
            and the mean NDVI score — a standard measure of vegetation vigour.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload Your Aerial Image</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.75rem;color:#1a4a1a;margin-bottom:8px;">
            PNG / JPG → tree detection only<br>
            .npy (4-band R,G,B,NIR) → detection + NDVI health<br>
            2 × .npy → temporal comparison<br>
            <span style="color:#2a5a2a;">Name format: <code style="color:#2a5a2a;">city_year_tile.npy</code>
            &nbsp; e.g. <code style="color:#2a5a2a;">toulouse_2026_tile1.npy</code></span>
        </div>""", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload",
            type=["png", "jpg", "jpeg", "npy"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="PNG/JPG = detection only · .npy = detection + NDVI · 2 .npy files = temporal comparison"
        )

    with col_right:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        npy_uploads, img_uploads = [], []
        if uploaded_files:
            for f in uploaded_files:
                (npy_uploads if f.name.lower().endswith('.npy') else img_uploads).append(f)

        if img_uploads:
            f = img_uploads[0]
            img = Image.open(f).convert("RGB")
            img_array = np.array(img)
            with st.spinner("Detecting trees..."):
                annotated_rgb, n_trees, avg_conf, _ = run_detection(img_array, model, conf_threshold)
            show_detection_only(annotated_rgb, n_trees, avg_conf)

        elif len(npy_uploads) == 1:
            f = npy_uploads[0]
            arr = np.load(f).astype(np.float32)
            if arr.max() > 1.0: arr /= 255.0
            with st.spinner("Analysing..."):
                annotated, boxes, ndvi_vals, summ, meta = analyse_single(arr, f.name)
            show_single_result(annotated, boxes, ndvi_vals, summ, meta)
            if boxes:
                st.markdown('<div class="section-title" style="margin-top:16px;">Per-Tree Table</div>',
                            unsafe_allow_html=True)
                rows = [{"Tree #": i+1, "NDVI": round(v,3),
                         "Health": f"{classify_health(v)[1]} {classify_health(v)[0]}"}
                        for i, v in enumerate(ndvi_vals)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        elif len(npy_uploads) >= 2:
            f1, f2 = npy_uploads[0], npy_uploads[1]
            arr1 = np.load(f1).astype(np.float32); arr1 = arr1/255 if arr1.max()>1 else arr1
            arr2 = np.load(f2).astype(np.float32); arr2 = arr2/255 if arr2.max()>1 else arr2
            with st.spinner("Running temporal comparison..."):
                a1, b1, n1, s1_, m1 = analyse_single(arr1, f1.name)
                a2, b2, n2, s2_, m2 = analyse_single(arr2, f2.name)
            tc1, tc2 = st.columns(2)
            with tc1: show_comparison_panel(a1, b1, n1, s1_, m1, label=f1.name)
            with tc2: show_comparison_panel(a2, b2, n2, s2_, m2, label=f2.name)
            make_comparison_section(s1_, s2_, b1, b2, m1, m2)

        else:
            st.markdown("""
            <div style="background:#c4dac4;border:1px solid #7aa07a;border-radius:16px;
                        padding:60px 20px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">🌿</div>
                <div style="color:#1a4a1a;font-size:1rem;font-weight:500;">
                    Upload an image to get started
                </div>
                <div style="color:#2a5a2a;font-size:0.82rem;margin-top:6px;">
                    PNG/JPG → detection only &nbsp;·&nbsp; .npy → detection + NDVI health
                </div>
            </div>""", unsafe_allow_html=True)

# ── TAB 2: Demo — Single Image ────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:12px;
                padding:16px 24px;margin-bottom:20px;">
        <div style="font-size:1rem;font-weight:600;color:#0f2e0f;margin-bottom:4px;">
            🖼️ See the Model in Action — Single Image
        </div>
        <div style="font-size:0.82rem;color:#2a5a2a;line-height:1.7;">
            Not ready to upload your own data yet? Pick one of our example aerial images from California
            to see exactly what the model produces — detected tree crowns, colour-coded health status,
            and a per-tree NDVI table. This is the same output you will get on your own site.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_left2, col_right2 = st.columns([1, 2], gap="large")

    with col_left2:
        st.markdown('<div class="section-title">Select a Sample Image</div>',
                    unsafe_allow_html=True)

        selected_sample = "— select —"
        if os.path.exists(TEST_DIR_RGBN):
            npy_files    = sorted([f for f in os.listdir(TEST_DIR_RGBN) if f.endswith('.npy')])
            sample_names = [npy_files[i] for i in [0, 20, 50, 80, 120, 150] if i < len(npy_files)]
            selected_sample = st.selectbox(
                "Sample image",
                ["— select —"] + sample_names,
                label_visibility="collapsed"
            )
        else:
            st.warning("Demo images not found.")

        st.markdown("""
        <hr style="border:none;border-top:1px solid #1e3a1e;margin:16px 0 12px 0;">
        <div style="font-size:0.7rem;font-weight:700;color:#1a4a1a;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:10px;">How to read the results</div>

        <div style="font-size:0.78rem;color:#2a5a2a;line-height:1.9;">
            <b style="color:#1a5a1a;">Bounding boxes</b> — each box is one detected tree crown.<br>
            Box colour shows its health status based on NDVI.
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">Health colours</div>
        <div style="font-size:0.78rem;color:#2a5a2a;line-height:2;">
            🟢 <b style="color:#1a5a1a;">Healthy</b> — NDVI &gt; 0.2 · active photosynthesis<br>
            🟡 <b style="color:#ddcc44;">Moderate</b> — NDVI 0.0–0.2 · mild stress or young growth<br>
            🔴 <b style="color:#dd6644;">Stressed</b> — NDVI &lt; 0.0 · hydric or disease stress
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">What is NDVI?</div>
        <div style="font-size:0.78rem;color:#2a5a2a;line-height:1.8;">
            NDVI = (NIR − Red) / (NIR + Red)<br>
            Measures how much near-infrared light a tree reflects — healthy vegetation reflects strongly in NIR.
            Range: −1 to +1. Urban trees typically score 0.1–0.6.
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">Data source</div>
        <div style="font-size:0.75rem;color:#1a4a1a;line-height:1.7;">
            California NAIP aerial imagery · 60 cm/pixel · 4-band (R, G, B, NIR)<br>
            Used for model training and testing only — not your production site.
        </div>
        """, unsafe_allow_html=True)

    with col_right2:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        if selected_sample != "— select —":
            arr = load_rgbn(os.path.join(TEST_DIR_RGBN, selected_sample))
            with st.spinner("Analysing..."):
                annotated, boxes, ndvi_vals, summ, meta = analyse_single(arr, selected_sample)
            show_single_result(annotated, boxes, ndvi_vals, summ, meta)
            st.markdown('<div style="font-size:0.72rem;color:#2a5a2a;margin-top:8px;">Demo data — California NAIP dataset</div>',
                        unsafe_allow_html=True)
            if boxes:
                st.markdown('<div class="section-title" style="margin-top:16px;">Per-Tree Table</div>',
                            unsafe_allow_html=True)
                rows = [{"Tree #": i+1, "NDVI": round(v,3),
                         "Health": f"{classify_health(v)[1]} {classify_health(v)[0]}"}
                        for i, v in enumerate(ndvi_vals)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background:#c4dac4;border:1px solid #7aa07a;border-radius:16px;
                        padding:60px 20px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">🖼️</div>
                <div style="color:#1a4a1a;font-size:1rem;font-weight:500;">
                    Select a sample image on the left
                </div>
            </div>""", unsafe_allow_html=True)

# ── TAB 3: Demo — Compare Years ───────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="background:#b8d4b8;border:1px solid #7aa07a;border-radius:12px;
                padding:16px 24px;margin-bottom:20px;">
        <div style="font-size:1rem;font-weight:600;color:#0f2e0f;margin-bottom:4px;">
            📅 Track Forest Health Over Time
        </div>
        <div style="font-size:0.82rem;color:#2a5a2a;line-height:1.7;">
            Select a location photographed in two different years and the model analyses both images side by side.
            You get a comparison table showing exactly how tree count, NDVI, and health percentages
            changed between the two dates — so you can tell whether the urban forest is recovering, stable, or declining.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col_left3, col_right3 = st.columns([1, 2], gap="large")

    with col_left3:
        st.markdown('<div class="section-title">Select a Location & Year Pair</div>',
                    unsafe_allow_html=True)

        temporal_pairs = {}
        if os.path.exists(TEST_DIR_RGBN):
            npy_files_t = sorted([f for f in os.listdir(TEST_DIR_RGBN) if f.endswith('.npy')])
            groups = defaultdict(list)
            for f in npy_files_t:
                meta = parse_filename(f)
                if meta['year']:
                    key = f"{meta['city']}_{meta['tile']}"
                    groups[key].append((meta['year'], f))
            for k, v in groups.items():
                if len(v) >= 2:
                    temporal_pairs[k] = sorted(v)

        pair_labels = ["— select —"] + [
            f"{k.replace('_', ' ').title()} ({' vs '.join(y for y, _ in v)})"
            for k, v in temporal_pairs.items()
        ]
        selected_pair_label = st.selectbox(
            "Location & year pair",
            pair_labels,
            label_visibility="collapsed"
        )

        if not temporal_pairs:
            st.warning("No multi-year demo pairs found.")

        st.markdown("""
        <hr style="border:none;border-top:1px solid #1e3a1e;margin:16px 0 12px 0;">
        <div style="font-size:0.7rem;font-weight:700;color:#1a4a1a;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:10px;">How to read the results</div>

        <div style="font-size:0.78rem;color:#2a5a2a;line-height:1.9;">
            The same location is analysed at two different years side by side.<br>
            Each panel shows detected trees coloured by health status.
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">Health colours</div>
        <div style="font-size:0.78rem;color:#2a5a2a;line-height:2;">
            🟢 <b style="color:#1a5a1a;">Healthy</b> — NDVI &gt; 0.2<br>
            🟡 <b style="color:#ddcc44;">Moderate</b> — NDVI 0.0–0.2<br>
            🔴 <b style="color:#dd6644;">Stressed</b> — NDVI &lt; 0.0
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">Comparison table</div>
        <div style="font-size:0.78rem;color:#2a5a2a;line-height:1.9;">
            <b style="color:#1a5a1a;">Mean NDVI</b> — average vegetation health across all detected trees.<br>
            <b style="color:#1a5a1a;">Change column</b> — ↑ improvement · ↓ decline · = stable<br>
            A change of &gt; 0.02 NDVI is considered significant.
        </div>

        <div style="margin:10px 0 6px 0;font-size:0.7rem;font-weight:700;color:#1a4a1a;
                    text-transform:uppercase;letter-spacing:1px;">Data source</div>
        <div style="font-size:0.75rem;color:#1a4a1a;line-height:1.7;">
            California NAIP aerial imagery · 60 cm/pixel · 4-band (R, G, B, NIR)<br>
            Years available: 2016, 2018, 2020.
        </div>
        """, unsafe_allow_html=True)

    with col_right3:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        if selected_pair_label != "— select —" and selected_pair_label in pair_labels[1:]:
            pair_key    = list(temporal_pairs.keys())[pair_labels[1:].index(selected_pair_label)]
            years_files = temporal_pairs[pair_key]
            ann_list, box_list2, ndvi_list, sum_list, meta_list = [], [], [], [], []
            with st.spinner("Running temporal comparison..."):
                for year, fname in years_files[:2]:
                    arr = load_rgbn(os.path.join(TEST_DIR_RGBN, fname))
                    ann, bxs, nvs, smm, mta = analyse_single(arr, fname)
                    ann_list.append(ann); box_list2.append(bxs)
                    ndvi_list.append(nvs); sum_list.append(smm); meta_list.append(mta)
            year_labels = [y for y, _ in years_files[:2]]
            tc1, tc2 = st.columns(2)
            with tc1: show_comparison_panel(ann_list[0], box_list2[0], ndvi_list[0], sum_list[0], meta_list[0], label=year_labels[0])
            with tc2: show_comparison_panel(ann_list[1], box_list2[1], ndvi_list[1], sum_list[1], meta_list[1], label=year_labels[1])
            make_comparison_section(sum_list[0], sum_list[1], box_list2[0], box_list2[1],
                                    meta_list[0], meta_list[1])
            st.markdown('<div style="font-size:0.72rem;color:#2a5a2a;margin-top:4px;">Demo data — California NAIP dataset</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#c4dac4;border:1px solid #7aa07a;border-radius:16px;
                        padding:60px 20px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">📅</div>
                <div style="color:#1a4a1a;font-size:1rem;font-weight:500;">
                    Select a location on the left
                </div>
                <div style="color:#2a5a2a;font-size:0.82rem;margin-top:6px;">
                    Only locations with images from 2+ years are shown
                </div>
            </div>""", unsafe_allow_html=True)
