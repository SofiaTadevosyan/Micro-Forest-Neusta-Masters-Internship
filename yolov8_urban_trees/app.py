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
    draw_health_boxes, health_summary
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
# TAB 2 — Health Analysis (NDVI)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:12px;padding:16px 20px;margin-bottom:20px;">
        <div style="font-size:0.95rem;font-weight:600;color:#7ddb7d;margin-bottom:6px;">What is NDVI Health Analysis?</div>
        <div style="font-size:0.82rem;color:#5a8a5a;line-height:1.7;">
        After detecting trees, this tab computes <b style="color:#9ddb9d;">NDVI (Normalized Difference Vegetation Index)</b>
        for each tree using the Near-Infrared band of the NAIP aerial imagery.<br>
        NDVI = (NIR − Red) / (NIR + Red) &nbsp;·&nbsp; Range: −1 to +1<br><br>
        <b style="color:#7ddb7d;">🟢 Healthy</b> &nbsp; NDVI &gt; 0.4 &nbsp;·&nbsp;
        <b style="color:#ddcc44;">🟡 Moderate</b> &nbsp; NDVI 0.2–0.4 &nbsp;·&nbsp;
        <b style="color:#dd6644;">🔴 Stressed</b> &nbsp; NDVI &lt; 0.2
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([1, 1], gap="large")

    with col_h1:
        st.markdown('<div class="section-title">Upload 4-Channel .npy Image</div>', unsafe_allow_html=True)
        uploaded_npy = st.file_uploader(
            "Upload NPY",
            type=["npy"],
            label_visibility="collapsed",
            help="Upload a .npy file from yolo_dataset/images/rgbn/test/"
        )

        st.markdown('<div class="section-title" style="margin-top:16px;">Or pick a sample</div>', unsafe_allow_html=True)
        if os.path.exists(TEST_DIR_RGBN):
            npy_files = sorted([f for f in os.listdir(TEST_DIR_RGBN) if f.endswith('.npy')])
            sample_names = [npy_files[i] for i in [0, 20, 50, 80, 120, 150] if i < len(npy_files)]
            selected_sample = st.selectbox(
                "Sample image",
                ["— select —"] + sample_names,
                label_visibility="collapsed"
            )
        else:
            selected_sample = "— select —"
            st.info("RGBN test directory not found.")

    with col_h2:
        st.markdown('<div class="section-title">Health Analysis Result</div>', unsafe_allow_html=True)

        # Determine source: uploaded file or sample
        rgbn_array = None
        source_name = None

        if uploaded_npy is not None:
            rgbn_array  = np.load(uploaded_npy).astype(np.float32)
            if rgbn_array.max() > 1.0:
                rgbn_array = rgbn_array / 255.0
            source_name = uploaded_npy.name

        elif selected_sample != "— select —":
            path = os.path.join(TEST_DIR_RGBN, selected_sample)
            rgbn_array  = load_rgbn(path)
            source_name = selected_sample

        if rgbn_array is not None:
            # RGB preview from first 3 channels
            rgb_preview = (rgbn_array[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

            with st.spinner("Running detection + NDVI analysis..."):
                _, _, _, boxes = run_detection(rgb_preview, model, conf_threshold)
                ndvi_vals = compute_ndvi(rgbn_array, boxes)
                annotated_health = draw_health_boxes(rgb_preview, boxes, ndvi_vals)
                summary = health_summary(ndvi_vals)

            st.image(annotated_health,
                     caption="🟢 Healthy  🟡 Moderate  🔴 Stressed  ·  NDVI values shown per tree",
                     use_container_width=True)

            if len(boxes) == 0:
                st.warning("No trees detected. Try lowering the confidence threshold in the sidebar.")
            else:
                # Summary cards
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number">{len(boxes)}</div>
                        <div class="metric-unit">Trees Detected</div>
                    </div>""", unsafe_allow_html=True)
                with s2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number" style="color:#7ddb7d">{summary['pct_healthy']}%</div>
                        <div class="metric-unit">🟢 Healthy</div>
                    </div>""", unsafe_allow_html=True)
                with s3:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number" style="color:#ddcc44">{summary['pct_moderate']}%</div>
                        <div class="metric-unit">🟡 Moderate</div>
                    </div>""", unsafe_allow_html=True)
                with s4:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-number" style="color:#dd6644">{summary['pct_stressed']}%</div>
                        <div class="metric-unit">🔴 Stressed</div>
                    </div>""", unsafe_allow_html=True)

                # Mean NDVI bar
                st.markdown(f"""
                <div style="background:#0d2b0d;border:1px solid #1e3a1e;border-radius:10px;
                            padding:12px 16px;margin-top:12px;">
                    <div style="font-size:0.72rem;color:#5a8a5a;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
                        Mean NDVI across all detected trees
                    </div>
                    <div style="font-size:1.6rem;font-weight:700;color:#7ddb7d;">{summary['mean_ndvi']}</div>
                    <div style="font-size:0.75rem;color:#3a6a3a;margin-top:2px;">
                        min {summary['min_ndvi']} &nbsp;·&nbsp; max {summary['max_ndvi']}
                    </div>
                </div>""", unsafe_allow_html=True)

                # Per-tree table
                st.markdown('<div class="section-title" style="margin-top:20px;">Per-Tree NDVI Table</div>',
                            unsafe_allow_html=True)
                import pandas as pd
                rows = []
                for i, (box, ndvi) in enumerate(zip(boxes, ndvi_vals)):
                    label, emoji, _ = classify_health(ndvi)
                    rows.append({
                        "Tree #": i + 1,
                        "NDVI": round(ndvi, 3),
                        "Health": f"{emoji} {label}",
                        "Box (x1,y1,x2,y2)": f"{box[0]},{box[1]},{box[2]},{box[3]}"
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div style="background:#0d1f0d;border:1px solid #1e3a1e;border-radius:16px;
                        padding:60px 20px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">🌿</div>
                <div style="color:#3a6a3a;font-size:1rem;font-weight:500;">Upload a .npy file or select a sample</div>
                <div style="color:#2a4a2a;font-size:0.82rem;margin-top:6px;">
                    Files are in: yolo_dataset/images/rgbn/test/
                </div>
            </div>""", unsafe_allow_html=True)
