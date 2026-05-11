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

/* Full dark background */
.stApp {
    background: #0a0f0a;
}

/* Hide only the Streamlit branding, NOT the header itself */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1rem !important; max-width: 100% !important; }

/* Sidebar toggle arrow — target every possible selector */
[data-testid="collapsedControl"],
button[kind="header"],
.st-emotion-cache-iiif1v,
section[data-testid="stSidebarCollapsedControl"] {
    background-color: #e65c00 !important;
    border-radius: 0 8px 8px 0 !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] path {
    fill: white !important;
    stroke: white !important;
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1a0b 0%, #081208 100%) !important;
    border-right: 1px solid #1e3a1e !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #c8e6c8 !important; }

/* Sidebar section headers */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    color: #5a8a5a !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 16px 0 8px 0;
}
/* Sidebar metric pill */
.sb-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0d2b0d;
    border: 1px solid #1e3a1e;
    border-radius: 8px;
    padding: 7px 12px;
    margin-bottom: 6px;
}
.sb-metric-label { font-size: 0.78rem; color: #7aaa7a; }
.sb-metric-value { font-size: 0.9rem; font-weight: 700; color: #7ddb7d; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d2b0d 0%, #1a3d1a 40%, #0d2b0d 100%);
    border-bottom: 1px solid #2d5a2d;
    padding: 28px 40px 24px 40px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 0;
}
.hero-icon { font-size: 3rem; }
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #7ddb7d;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8ab88a;
    margin: 4px 0 0 0;
}
.hero-badge {
    margin-left: auto;
    background: #1e4a1e;
    border: 1px solid #3d7a3d;
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.8rem;
    color: #7ddb7d;
    font-weight: 500;
}

/* Stats bar */
.stats-bar {
    background: #0d1f0d;
    border-bottom: 1px solid #1e3a1e;
    padding: 14px 40px;
    display: flex;
    gap: 48px;
    margin-bottom: 24px;
}
.stat-item { text-align: center; }
.stat-value { font-size: 1.3rem; font-weight: 700; color: #7ddb7d; }
.stat-label { font-size: 0.72rem; color: #5a8a5a; text-transform: uppercase; letter-spacing: 0.5px; }

/* Upload zone */
.upload-zone {
    background: #0d1f0d;
    border: 2px dashed #2d5a2d;
    border-radius: 16px;
    padding: 40px 20px;
    text-align: center;
    transition: all 0.2s;
}
.upload-zone:hover { border-color: #5aaa5a; }

/* Result panel */
.result-panel {
    background: #0d1f0d;
    border: 1px solid #1e3a1e;
    border-radius: 16px;
    padding: 20px;
    min-height: 300px;
}

/* Metric cards row */
.metrics-row {
    display: flex;
    gap: 16px;
    margin-top: 16px;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #0d2b0d, #162816);
    border: 1px solid #2d5a2d;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-number { font-size: 2rem; font-weight: 700; color: #7ddb7d; line-height: 1; }
.metric-unit   { font-size: 0.75rem; color: #5a8a5a; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

/* Section title */
.section-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #5a8a5a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

/* Sample grid */
.sample-card {
    background: #0d1f0d;
    border: 1px solid #1e3a1e;
    border-radius: 10px;
    overflow: hidden;
    transition: border-color 0.2s;
    cursor: pointer;
}
.sample-card:hover { border-color: #5aaa5a; }
.sample-info {
    padding: 8px 10px;
    font-size: 0.78rem;
    color: #5a8a5a;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e3a1e;
    margin: 28px 0;
}

/* Streamlit overrides */
.stFileUploader > div { background: transparent !important; border: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #0d1f0d !important;
    border: 2px dashed #2d5a2d !important;
    border-radius: 12px !important;
}
.stSlider > div > div > div { background: #2d5a2d !important; }

/* Image caption */
.stImage > div > div { color: #5a8a5a !important; font-size: 0.8rem !important; }
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
<div class="hero">
    <div class="hero-icon">🌳</div>
    <div>
        <div class="hero-title">Urban Micro-Forest Tree Detection</div>
        <div class="hero-sub">AI-powered aerial tree detection using YOLOv8 · NEUSTA Monitoring System</div>
    </div>
    <div class="hero-badge">YOLOv8s · RGB Model</div>
</div>

<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value">1,651</div>
        <div class="stat-label">Training Images</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">96,547</div>
        <div class="stat-label">Annotated Trees</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">0.439</div>
        <div class="stat-label">mAP@50</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">77.3%</div>
        <div class="stat-label">Recall</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">56.7%</div>
        <div class="stat-label">Precision</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">256px</div>
        <div class="stat-label">Input Size</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Logo / title ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.8rem;">🌳</div>
        <div style="font-size:1rem; font-weight:700; color:#7ddb7d; margin-top:6px;">
            Tree Detection
        </div>
        <div style="font-size:0.72rem; color:#3a6a3a; margin-top:2px;">
            NEUSTA · YOLOv8s · RGB
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #1e3a1e; margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    # ── 1. Detection settings ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">🎯 Detection Settings</div>', unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01, max_value=0.90,
        value=0.25, step=0.01,
        help="Lower = more detections (but more false positives). Higher = fewer but more certain."
    )

    # Live feedback on threshold choice
    if conf_threshold < 0.15:
        st.markdown("<div style='font-size:0.75rem;color:#ddaa33;'>⚠️ Very low — many false detections</div>", unsafe_allow_html=True)
    elif conf_threshold > 0.60:
        st.markdown("<div style='font-size:0.75rem;color:#dd8844;'>⚠️ Very high — may miss trees</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.75rem;color:#5aaa5a;'>✓ Good range for this model</div>", unsafe_allow_html=True)

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
    <div style="font-size:0.78rem; color:#5a8a5a; line-height:1.7;">
    1. Upload an aerial PNG image<br>
    2. Adjust the confidence slider<br>
    3. Green boxes = detected trees<br>
    4. Try the sample images below
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr style="border:none; border-top:1px solid #1e3a1e; margin:20px 0 10px 0;">
    <div style="font-size:0.7rem; color:#2a4a2a; text-align:center;">
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
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🌳 Tree Detection", "🌿 Health Analysis (NDVI)"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Tree Detection (existing flow)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload Aerial Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            help="Upload a 256×256 aerial PNG image"
        )

        if uploaded:
            img_pil   = Image.open(uploaded).convert("RGB")
            img_array = np.array(img_pil)
            st.image(img_array, caption="Input image", use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#0d1f0d;border:2px dashed #1e3a1e;border-radius:16px;
                        padding:80px 20px;text-align:center;">
                <div style="font-size:3.5rem;margin-bottom:12px;">🛸</div>
                <div style="color:#3a6a3a;font-size:1rem;font-weight:500;">Drop your aerial image here</div>
                <div style="color:#2a4a2a;font-size:0.82rem;margin-top:6px;">PNG or JPG · 256×256 px recommended</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Detecting trees..."):
                annotated, n_trees, avg_conf, _ = run_detection(img_array, model, conf_threshold)

            st.image(annotated, caption="Detected trees (green boxes)", use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number">{n_trees}</div>
                    <div class="metric-unit">Trees Found</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                val = f"{avg_conf*100:.0f}%" if n_trees > 0 else "—"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number">{val}</div>
                    <div class="metric-unit">Avg Confidence</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                if n_trees == 0:
                    level, color = "None", "#dd4444"
                elif avg_conf > 0.5:
                    level, color = "High", "#7ddb7d"
                elif avg_conf > 0.3:
                    level, color = "Med", "#ddbb44"
                else:
                    level, color = "Low", "#dd8844"
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-number" style="color:{color}">{level}</div>
                    <div class="metric-unit">Quality</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:16px;
                        padding:80px 20px;text-align:center;">
                <div style="font-size:3.5rem;margin-bottom:12px;">🌲</div>
                <div style="color:#3a6a3a;font-size:1rem;font-weight:500;">Results will appear here</div>
                <div style="color:#2a4a2a;font-size:0.82rem;margin-top:6px;">Upload an image on the left to run detection</div>
            </div>""", unsafe_allow_html=True)

    # Sample images
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Test Images — Try these</div>', unsafe_allow_html=True)

    if os.path.exists(TEST_DIR):
        all_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.png')])
        indices   = [0, 25, 55, 90, 130, 155]
        samples   = [all_files[i] for i in indices if i < len(all_files)]
        cols = st.columns(len(samples))
        for col, fname in zip(cols, samples):
            img  = np.array(Image.open(os.path.join(TEST_DIR, fname)).convert("RGB"))
            city = fname.split('_')[0].title()
            year = fname.split('_')[1] if '_' in fname else ''
            with col:
                st.image(img, use_container_width=True)
                st.markdown(f"<div style='text-align:center;color:#4a7a4a;font-size:0.75rem;'>{city} {year}</div>",
                            unsafe_allow_html=True)
        st.markdown("""<div style="color:#2a4a2a;font-size:0.8rem;margin-top:10px;text-align:center;">
        Find these images at: <code style="color:#3a6a3a">yolo_dataset/images/rgb/test/</code></div>""",
        unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Health Analysis (NDVI) — location-agnostic + temporal comparison
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    import pandas as pd

    st.markdown("""
    <div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:12px;
                padding:16px 20px;margin-bottom:20px;">
        <div style="font-size:0.95rem;font-weight:600;color:#7ddb7d;margin-bottom:6px;">
            NDVI Tree Health Analysis
        </div>
        <div style="font-size:0.82rem;color:#5a8a5a;line-height:1.8;">
            NDVI = (NIR − Red) / (NIR + Red) &nbsp;·&nbsp;
            Computed per detected tree from the 4-channel aerial image.<br>
            <b style="color:#7ddb7d;">🟢 Healthy</b> NDVI &gt; 0.2 &nbsp;·&nbsp;
            <b style="color:#ddcc44;">🟡 Moderate</b> NDVI 0.0–0.2 &nbsp;·&nbsp;
            <b style="color:#dd6644;">🔴 Stressed</b> NDVI &lt; 0.0<br>
            <span style="color:#3a6a3a;font-size:0.78rem;">
            Upload <b>1 file</b> for health analysis &nbsp;·&nbsp;
            Upload <b>2 files</b> of the same location (different years) for temporal comparison
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([1, 2], gap="large")

    with col_h1:
        # ── Upload your own files ────────────────────────────────────────────
        st.markdown('<div class="section-title">Upload Your Aerial Image(s)</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.75rem;color:#3a6a3a;margin-bottom:8px;">
            4-channel .npy files (R, G, B, NIR)<br>
            Name format: <code style="color:#5a8a5a;">city_year_tile.npy</code><br>
            e.g. <code style="color:#5a8a5a;">toulouse_2026_tile1.npy</code>
        </div>""", unsafe_allow_html=True)

        uploaded_npys = st.file_uploader(
            "Upload",
            type=["npy"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload 1 file for health analysis or 2 files for temporal comparison"
        )

        # ── Demo samples ─────────────────────────────────────────────────────
        st.markdown("""
        <hr style="border:none;border-top:1px solid #1e3a1e;margin:16px 0 10px 0;">
        <div style="font-size:0.7rem;font-weight:700;color:#3a6a3a;text-transform:uppercase;
                    letter-spacing:1px;margin-bottom:4px;">
            Demo Samples — California NAIP (testing only)
        </div>
        <div style="font-size:0.72rem;color:#2a4a2a;margin-bottom:8px;">
            These are example images from California. For your own site, upload files above.
        </div>""", unsafe_allow_html=True)

        selected_sample = "— select —"
        if os.path.exists(TEST_DIR_RGBN):
            npy_files    = sorted([f for f in os.listdir(TEST_DIR_RGBN) if f.endswith('.npy')])
            sample_names = [npy_files[i] for i in [0, 20, 50, 80, 120, 150] if i < len(npy_files)]
            selected_sample = st.selectbox(
                "Demo sample",
                ["— select —"] + sample_names,
                label_visibility="collapsed"
            )

        # ── Temporal demo picker ─────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:0.7rem;font-weight:700;color:#3a6a3a;text-transform:uppercase;
                    letter-spacing:1px;margin:12px 0 4px 0;">
            Demo Temporal Comparison
        </div>
        <div style="font-size:0.72rem;color:#2a4a2a;margin-bottom:8px;">
            Pick a location that has images from 2 different years.
        </div>""", unsafe_allow_html=True)

        # Build list of multi-year pairs from test dir
        temporal_pairs = {}
        if os.path.exists(TEST_DIR_RGBN):
            from collections import defaultdict
            groups = defaultdict(list)
            for f in npy_files:
                meta = parse_filename(f)
                if meta['year']:
                    key = f"{meta['city']}_{meta['tile']}"
                    groups[key].append((meta['year'], f))
            for k, v in groups.items():
                if len(v) >= 2:
                    v_sorted = sorted(v)
                    temporal_pairs[k] = v_sorted

        pair_labels = ["— select —"] + [
            f"{k.replace('_', ' ').title()} ({' vs '.join(y for y,_ in v)})"
            for k, v in temporal_pairs.items()
        ]
        selected_pair_label = st.selectbox(
            "Temporal pair",
            pair_labels,
            label_visibility="collapsed"
        )

    # ── Determine what to analyse ────────────────────────────────────────────
    with col_h2:
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        def analyse_single(arr, fname):
            """Run detection + NDVI on one rgbn array. Returns annotated img + summary + table."""
            rgb = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
            _, _, _, boxes = run_detection(rgb, model, conf_threshold)
            ndvi_vals      = compute_ndvi(arr, boxes)
            annotated      = draw_health_boxes(rgb, boxes, ndvi_vals)
            summ           = health_summary(ndvi_vals)
            meta           = parse_filename(fname)
            return annotated, boxes, ndvi_vals, summ, meta

        def show_single_result(annotated, boxes, ndvi_vals, summ, meta, label=None):
            title = label or meta.get('label', '')
            if title:
                st.markdown(f"<div style='font-size:0.8rem;color:#5a8a5a;margin-bottom:4px;'>{title}</div>",
                            unsafe_allow_html=True)
            st.image(annotated,
                     caption="🟢 Healthy  🟡 Moderate  🔴 Stressed  · NDVI per tree",
                     use_container_width=True)
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
                    <div class="metric-number" style="color:#7ddb7d">{summ['pct_healthy']}%</div>
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
            <div style="background:#0d2b0d;border:1px solid #1e3a1e;border-radius:10px;
                        padding:10px 14px;margin-top:10px;">
                <span style="font-size:0.72rem;color:#5a8a5a;text-transform:uppercase;">Mean NDVI</span>
                <span style="font-size:1.4rem;font-weight:700;color:#7ddb7d;margin-left:12px;">{summ['mean_ndvi']}</span>
                <span style="font-size:0.75rem;color:#3a6a3a;margin-left:10px;">
                    min {summ['min_ndvi']} · max {summ['max_ndvi']}</span>
            </div>""", unsafe_allow_html=True)

        # ── CASE 1: user uploaded files ──────────────────────────────────────
        if uploaded_npys and len(uploaded_npys) >= 1:
            if len(uploaded_npys) == 1:
                f = uploaded_npys[0]
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

            elif len(uploaded_npys) >= 2:
                f1, f2 = uploaded_npys[0], uploaded_npys[1]
                arr1 = np.load(f1).astype(np.float32); arr1 = arr1/255 if arr1.max()>1 else arr1
                arr2 = np.load(f2).astype(np.float32); arr2 = arr2/255 if arr2.max()>1 else arr2
                with st.spinner("Running temporal comparison..."):
                    a1, b1, n1, s1_, m1 = analyse_single(arr1, f1.name)
                    a2, b2, n2, s2_, m2 = analyse_single(arr2, f2.name)
                tc1, tc2 = st.columns(2)
                with tc1: show_single_result(a1, b1, n1, s1_, m1)
                with tc2: show_single_result(a2, b2, n2, s2_, m2)
                # Comparison table
                label_a = m1.get('label', f1.name)
                label_b = m2.get('label', f2.name)
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
                # Auto conclusion
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
                <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:10px;
                            padding:12px 16px;margin-top:8px;font-size:0.85rem;color:#9ddb9d;">
                    {icon} {conclusion}
                </div>""", unsafe_allow_html=True)

        # ── CASE 2: demo temporal pair ───────────────────────────────────────
        elif selected_pair_label != "— select —" and selected_pair_label in pair_labels[1:]:
            pair_key = list(temporal_pairs.keys())[pair_labels[1:].index(selected_pair_label)]
            years_files = temporal_pairs[pair_key]
            arr_list, meta_list, ann_list, box_list2, ndvi_list, sum_list = [], [], [], [], [], []
            with st.spinner("Running temporal comparison..."):
                for year, fname in years_files[:2]:
                    arr = load_rgbn(os.path.join(TEST_DIR_RGBN, fname))
                    ann, bxs, nvs, smm, mta = analyse_single(arr, fname)
                    arr_list.append(arr); ann_list.append(ann); box_list2.append(bxs)
                    ndvi_list.append(nvs); sum_list.append(smm); meta_list.append(mta)
            tc1, tc2 = st.columns(2)
            with tc1: show_single_result(ann_list[0], box_list2[0], ndvi_list[0], sum_list[0], meta_list[0])
            with tc2: show_single_result(ann_list[1], box_list2[1], ndvi_list[1], sum_list[1], meta_list[1])
            s1_, s2_ = sum_list[0], sum_list[1]
            b1,  b2  = box_list2[0], box_list2[1]
            m1,  m2  = meta_list[0], meta_list[1]
            label_a, label_b = m1['label'], m2['label']
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
            st.markdown('<div style="font-size:0.72rem;color:#2a4a2a;margin-top:4px;">Demo data — California NAIP dataset</div>',
                        unsafe_allow_html=True)
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
            <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:10px;
                        padding:12px 16px;margin-top:8px;font-size:0.85rem;color:#9ddb9d;">
                {icon} {conclusion}
            </div>""", unsafe_allow_html=True)

        # ── CASE 3: demo single sample ───────────────────────────────────────
        elif selected_sample != "— select —":
            arr = load_rgbn(os.path.join(TEST_DIR_RGBN, selected_sample))
            with st.spinner("Analysing..."):
                annotated, boxes, ndvi_vals, summ, meta = analyse_single(arr, selected_sample)
            show_single_result(annotated, boxes, ndvi_vals, summ, meta)
            st.markdown('<div style="font-size:0.72rem;color:#2a4a2a;margin-top:8px;">Demo data — California NAIP dataset</div>',
                        unsafe_allow_html=True)
            if boxes:
                st.markdown('<div class="section-title" style="margin-top:16px;">Per-Tree Table</div>',
                            unsafe_allow_html=True)
                rows = [{"Tree #": i+1, "NDVI": round(v,3),
                         "Health": f"{classify_health(v)[1]} {classify_health(v)[0]}"}
                        for i, v in enumerate(ndvi_vals)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Empty state ──────────────────────────────────────────────────────
        else:
            st.markdown("""
            <div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:16px;
                        padding:60px 20px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">🌿</div>
                <div style="color:#3a6a3a;font-size:1rem;font-weight:500;">
                    Upload a .npy file or select a demo sample
                </div>
                <div style="color:#2a4a2a;font-size:0.82rem;margin-top:6px;">
                    1 file → health analysis &nbsp;·&nbsp; 2 files → temporal comparison
                </div>
            </div>""", unsafe_allow_html=True)
