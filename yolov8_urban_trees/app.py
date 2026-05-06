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

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f0d 0%, #0a150a 100%);
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] * { color: #c8e6c8 !important; }

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
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
RGB_WEIGHTS = os.path.join(REPO_ROOT, 'results', 'weights', 'best_rgb.pt')
TEST_DIR    = os.path.join(REPO_ROOT, 'yolo_dataset', 'images', 'rgb', 'test')

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
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results  = model.predict(img_bgr, verbose=False, conf=conf)
    boxes    = results[0].boxes
    n        = 0
    avg_conf = 0.0
    out      = img_bgr.copy()

    if boxes is not None and len(boxes) > 0:
        n        = len(boxes)
        avg_conf = float(boxes.conf.mean().item())
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80), 2)
            cv2.rectangle(out, (x1, y1), (x1+5, y1+5), (0, 220, 80), -1)
            cv2.rectangle(out, (x2-5, y2-5), (x2, y2), (0, 220, 80), -1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), n, avg_conf

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
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01, max_value=0.90,
        value=0.25, step=0.01,
        help="Lower = more detections. Higher = fewer but more certain."
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This demo uses **YOLOv8s** trained on NAIP aerial imagery.

    The model detects individual tree canopies from top-down aerial photos at 60 cm/pixel resolution.

    **Dataset:** 8 California cities
    **Input:** 256×256 px PNG
    **Bands:** RGB (3-channel)
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | mAP@50 | 0.439 |
    | Precision | 0.567 |
    | Recall | 0.773 |
    | F1 | 0.654 |
    """)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3a6a3a;'>
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
# Main layout
# ---------------------------------------------------------------------------
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
        <div style="
            background: #0d1f0d;
            border: 2px dashed #1e3a1e;
            border-radius: 16px;
            padding: 80px 20px;
            text-align: center;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 12px;">🛸</div>
            <div style="color: #3a6a3a; font-size: 1rem; font-weight: 500;">
                Drop your aerial image here
            </div>
            <div style="color: #2a4a2a; font-size: 0.82rem; margin-top: 6px;">
                PNG or JPG · 256×256 px recommended
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)

    if uploaded:
        with st.spinner("Detecting trees..."):
            annotated, n_trees, avg_conf = run_detection(img_array, model, conf_threshold)

        st.image(annotated, caption="Detected trees (green boxes)", use_container_width=True)

        # Metric cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{n_trees}</div>
                <div class="metric-unit">Trees Found</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            val = f"{avg_conf*100:.0f}%" if n_trees > 0 else "—"
            st.markdown(f"""
            <div class="metric-card">
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
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number" style="color:{color}">{level}</div>
                <div class="metric-unit">Quality</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: #0d1f0d;
            border: 1px solid #1e3a1e;
            border-radius: 16px;
            padding: 80px 20px;
            text-align: center;
        ">
            <div style="font-size: 3.5rem; margin-bottom: 12px;">🌲</div>
            <div style="color: #3a6a3a; font-size: 1rem; font-weight: 500;">
                Results will appear here
            </div>
            <div style="color: #2a4a2a; font-size: 0.82rem; margin-top: 6px;">
                Upload an image on the left to run detection
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------
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
            st.markdown(
                f"<div style='text-align:center;color:#4a7a4a;font-size:0.75rem;'>"
                f"{city} {year}</div>",
                unsafe_allow_html=True
            )

    st.markdown("""
    <div style="color:#2a4a2a; font-size:0.8rem; margin-top:10px; text-align:center;">
    Find these images at: <code style="color:#3a6a3a">yolo_dataset/images/rgb/test/</code>
    </div>
    """, unsafe_allow_html=True)
