#!/usr/bin/env python3
"""
app.py — Urban Micro-Forest Tree Detection Demo

Streamlit app for interactive tree detection on aerial imagery.

Run:
    streamlit run yolov8_urban_trees/app.py

Requirements:
    pip3 install streamlit ultralytics opencv-python-headless numpy pillow
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
    page_title="Urban Tree Detection",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark green forest theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1a0f; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #162416;
        border-right: 1px solid #2d4a2d;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a3a1a 0%, #0d2d0d 100%);
        padding: 24px 32px;
        border-radius: 12px;
        border: 1px solid #2d5a2d;
        margin-bottom: 24px;
    }
    .main-header h1 {
        color: #7ddb7d;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #a0c8a0;
        margin: 6px 0 0 0;
        font-size: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: #162416;
        border: 1px solid #2d5a2d;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7ddb7d;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0c8a0;
        margin-top: 4px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: #162416;
        border: 2px dashed #2d5a2d;
        border-radius: 10px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2d7a2d, #1a5a1a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3d8a3d, #2a6a2a);
        transform: translateY(-1px);
    }

    /* Section headers */
    h2, h3 { color: #7ddb7d !important; }

    /* Info box */
    .info-box {
        background: #1a2d1a;
        border-left: 4px solid #7ddb7d;
        border-radius: 6px;
        padding: 12px 16px;
        color: #c0d8c0;
        font-size: 0.9rem;
        margin-bottom: 16px;
    }

    /* Sample image labels */
    .sample-label {
        color: #a0c8a0;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resolve model path relative to this script
# ---------------------------------------------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
RGB_WEIGHTS = os.path.join(REPO_ROOT, 'results', 'weights', 'best_rgb.pt')

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_detection(image_array, model, conf):
    """Run YOLO detection on a numpy RGB image array. Returns annotated image + stats."""
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    results = model.predict(img_bgr, verbose=False, conf=conf)
    boxes      = results[0].boxes
    n_trees    = 0
    avg_conf   = 0.0
    annotated  = img_bgr.copy()

    if boxes is not None and len(boxes) > 0:
        n_trees  = len(boxes)
        avg_conf = float(boxes.conf.mean().item())
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            # Small filled corner for style
            cv2.rectangle(annotated, (x1, y1), (x1+6, y1+6), (0, 220, 0), -1)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, n_trees, avg_conf

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌳 Configuration")
    st.markdown("---")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01, max_value=0.90,
        value=0.25, step=0.01,
        help="Lower = more detections (may include false positives). Higher = fewer but more certain."
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <div style='color: #a0c8a0; font-size: 0.85rem;'>
    This demo uses a <b style='color:#7ddb7d'>YOLOv8s</b> model trained on
    1,651 NAIP aerial images with 96,547 annotated urban trees across
    8 California cities.<br><br>
    Input: 256×256 px aerial image<br>
    Resolution: 60 cm/pixel<br>
    Model: YOLOv8s (RGB)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Performance")
    st.markdown("""
    <div style='color: #a0c8a0; font-size: 0.85rem;'>
    mAP@50: <b style='color:#7ddb7d'>0.439</b><br>
    Precision: <b style='color:#7ddb7d'>0.567</b><br>
    Recall: <b style='color:#7ddb7d'>0.773</b><br>
    F1: <b style='color:#7ddb7d'>0.654</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color: #607060; font-size: 0.75rem;'>
    NEUSTA Micro-Forest Monitoring<br>
    Sofya Tadevosyan — May 2026
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🌳 Urban Micro-Forest Tree Detection</h1>
    <p>Upload an aerial image to detect and count urban trees using YOLOv8 AI</p>
</div>
""", unsafe_allow_html=True)

# Check weights exist
if not os.path.exists(RGB_WEIGHTS):
    st.error(f"Model weights not found at: `{RGB_WEIGHTS}`\n\nPlease run the training pipeline first.")
    st.stop()

model = load_model(RGB_WEIGHTS)

# ---------------------------------------------------------------------------
# Upload + detection
# ---------------------------------------------------------------------------

col_upload, col_result = st.columns(2, gap="large")

with col_upload:
    st.markdown("### Upload Aerial Image")
    st.markdown("""
    <div class="info-box">
    Upload a <b>256×256 px aerial image</b> (PNG or JPG).<br>
    The model expects a top-down view at ~60 cm/pixel resolution.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop image here or click to browse",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded:
        img_pil   = Image.open(uploaded).convert("RGB")
        img_array = np.array(img_pil)
        st.image(img_array, caption="Uploaded image", use_container_width=True)

with col_result:
    st.markdown("### Detection Result")

    if uploaded:
        with st.spinner("Running tree detection..."):
            annotated, n_trees, avg_conf = run_detection(img_array, model, conf_threshold)

        st.image(annotated, caption=f"{n_trees} trees detected", use_container_width=True)

        # Metric cards
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_trees}</div>
                <div class="metric-label">Trees Detected</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            conf_pct = f"{avg_conf*100:.0f}%" if avg_conf > 0 else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{conf_pct}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            level = "High" if avg_conf > 0.5 else "Medium" if avg_conf > 0.3 else "Low"
            color = "#7ddb7d" if avg_conf > 0.5 else "#ddbb7d" if avg_conf > 0.3 else "#dd7d7d"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{level}</div>
                <div class="metric-label">Confidence Level</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background: #162416;
            border: 2px dashed #2d4a2d;
            border-radius: 10px;
            padding: 80px 20px;
            text-align: center;
            color: #4a6a4a;
        '>
            <div style='font-size: 3rem;'>🌳</div>
            <div style='margin-top: 12px; font-size: 1rem;'>
                Upload an image to see detections
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample images section
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Sample Test Images")
st.markdown("<div style='color: #a0c8a0; font-size: 0.9rem; margin-bottom: 16px;'>Click any sample to use it for detection.</div>", unsafe_allow_html=True)

# Show 4 sample images from test set
test_dir = os.path.join(REPO_ROOT, 'yolo_dataset', 'images', 'rgb', 'test')
if os.path.exists(test_dir):
    samples = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
    # Pick 4 varied samples
    indices = [0, 20, 50, 100]
    chosen  = [samples[i] for i in indices if i < len(samples)]

    cols = st.columns(len(chosen))
    for col, fname in zip(cols, chosen):
        img_path = os.path.join(test_dir, fname)
        img      = np.array(Image.open(img_path).convert("RGB"))
        city     = fname.split('_')[0].title()
        year     = fname.split('_')[1] if len(fname.split('_')) > 1 else ''
        with col:
            st.image(img, use_container_width=True)
            st.markdown(f"<div class='sample-label'>{city} {year}</div>", unsafe_allow_html=True)
            if st.button(f"Use this", key=fname):
                st.session_state['sample'] = img_path
                st.info(f"Download `{fname}` from `yolo_dataset/images/rgb/test/` and upload it above.")
