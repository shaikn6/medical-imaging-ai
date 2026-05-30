"""
Streamlit dashboard — Medical Imaging AI.

Features:
  - Upload a chest X-ray or select a synthetic test image
  - Run prediction + Grad-CAM
  - Display original, Grad-CAM overlay, and per-class probabilities
  - Show confidence bar chart

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CHECKPOINT = os.path.join(_ROOT, "model", "xray_model.pth")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Imaging AI | X-Ray Pathology Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-themed custom CSS
st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .class-badge {
        display: inline-block;
        background: #2a4a7f;
        color: #90cdf4;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    h1 { color: #63b3ed; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Lazy load pipeline (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    from model.inference import InferencePipeline
    if not os.path.exists(CHECKPOINT):
        return None
    return InferencePipeline(CHECKPOINT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_COLORS = {
    "Normal":           "#48bb78",
    "Pneumonia":        "#fc8181",
    "Cardiomegaly":     "#f6ad55",
    "Pleural Effusion": "#76e4f7",
}
CLASSES = list(CLASS_COLORS.keys())


def make_probability_chart(probabilities: dict[str, float]) -> go.Figure:
    classes = list(probabilities.keys())
    values  = [probabilities[c] * 100 for c in classes]
    colors  = [CLASS_COLORS.get(c, "#a0aec0") for c in classes]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#e0e0e0",
        xaxis=dict(range=[0, 100], title="Confidence (%)", gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    input_mode = st.radio(
        "Image source",
        ["Upload image", "Use synthetic test image"],
        index=1,
    )

    if input_mode == "Use synthetic test image":
        selected_class = st.selectbox("Pathology class", CLASSES, index=1)

    st.markdown("---")
    st.markdown(
        """
        **Model:** Custom 4-block CNN
        **Input:** 224×224 grayscale
        **Classes:** Normal · Pneumonia · Cardiomegaly · Pleural Effusion
        **Explainability:** Grad-CAM (last Conv2d layer)
        """
    )
    st.markdown("---")
    st.caption("⚠️ For demonstration only — not a clinical diagnostic tool.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("🫁 Medical Imaging AI")
st.subheader("Chest X-Ray Pathology Detection with Grad-CAM Explainability")
st.caption(
    "Upload a chest X-ray or use a synthetic test image to detect pathologies "
    "and visualise the anatomical regions driving the prediction."
)

# Model status
pipeline = load_pipeline()
if pipeline is None:
    st.error(
        "Model checkpoint not found. "
        "Run `python -m model.trainer` first, then refresh this page."
    )
    st.stop()

st.success("Model loaded — ready for inference.")
st.divider()

# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------

pil_input: Image.Image | None = None

if input_mode == "Upload image":
    uploaded = st.file_uploader(
        "Upload a chest X-ray (PNG / JPEG)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        pil_input = Image.open(uploaded).convert("L")
else:
    # Synthetic test image
    from data.synthetic_xray import GENERATORS
    gen_fn = GENERATORS[selected_class]
    arr = gen_fn(224, seed=99)
    pil_input = Image.fromarray(arr, mode="L")

# ---------------------------------------------------------------------------
# Run inference when image is ready
# ---------------------------------------------------------------------------

if pil_input is not None:
    col_img, col_cam, col_chart = st.columns([1, 1, 1.4])

    with col_img:
        st.subheader("Input X-Ray")
        st.image(pil_input, caption="Original (grayscale)", use_column_width=True)

    # Run
    with st.spinner("Running inference + Grad-CAM …"):
        result = pipeline.run(pil_input)

    with col_cam:
        st.subheader("Grad-CAM Overlay")
        st.image(
            result.overlay,
            caption="Regions driving the prediction (red = highest activation)",
            use_column_width=True,
        )

    with col_chart:
        st.subheader("Prediction")
        st.markdown(
            f"<div class='metric-card'>"
            f"Predicted class:&nbsp;&nbsp;"
            f"<span class='class-badge'>{result.predicted_class}</span><br/>"
            f"Confidence:&nbsp;&nbsp;<strong>{result.confidence:.1%}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_probability_chart(result.probabilities),
            use_container_width=True,
        )

    # Expander with raw probabilities
    with st.expander("Raw probabilities"):
        for cls, p in result.probabilities.items():
            st.write(f"**{cls}**: {p:.4f}")

else:
    st.info("Select an image source in the sidebar and an image will appear here.")
