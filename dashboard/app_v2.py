"""
Medical Imaging AI — V2 Streamlit Dashboard.

Tabs:
  1. Multi-label Classifier    — 5 pathology predictions with per-label sigmoid
  2. U-Net Segmentation        — pixel-level lesion mask overlay
  3. Grad-CAM / ScoreCAM       — side-by-side explainability comparison
  4. DICOM Metadata Explorer   — synthetic DICOM metadata viewer + windowing

Run:
    streamlit run dashboard/app_v2.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Imaging AI V2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body { background-color: #0d1117; color: #c9d1d9; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important;
        color: #ffffff !important;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .label-chip {
        display: inline-block;
        border-radius: 12px;
        padding: 2px 12px;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 2px;
    }
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached model loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_multilabel_model():
    from models.multilabel_classifier import build_multilabel_model
    model = build_multilabel_model()
    model.eval()
    return model


@st.cache_resource
def load_seg_model():
    from models.unet import build_unet
    model = build_unet()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "Pneumonia":    "#f85149",
    "Cardiomegaly": "#f0883e",
    "Effusion":     "#d2a8ff",
    "Atelectasis":  "#56d364",
    "Nodule":       "#58a6ff",
}
MULTILABEL_CLASSES = list(LABEL_COLORS.keys())


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert greyscale PIL image to (1, 1, 224, 224) normalised tensor."""
    arr = np.array(pil_img.convert("L").resize((224, 224)), dtype=np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def make_label_bar_chart(probs: dict[str, float]) -> go.Figure:
    classes = list(probs.keys())
    values  = [probs[c] * 100 for c in classes]
    colors  = [LABEL_COLORS.get(c, "#8b949e") for c in classes]

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
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font_color="#c9d1d9",
        xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#30363d"),
        yaxis=dict(gridcolor="#30363d"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar — shared image source
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    img_source = st.radio("Image source", ["Synthetic test image", "Upload image"], index=0)

    if img_source == "Synthetic test image":
        from data.synthetic_xray import GENERATORS
        syn_cls = st.selectbox("Pathology class", list(GENERATORS.keys()), index=1)
        syn_seed = st.slider("Seed", 0, 99, 7)
        arr = GENERATORS[syn_cls](224, seed=syn_seed)
        shared_pil = Image.fromarray(arr, mode="L")
    else:
        uploaded = st.file_uploader("Upload X-ray (PNG/JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded:
            _MAX_UPLOAD_MB = 10
            if uploaded.size > _MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File too large. Maximum allowed size is {_MAX_UPLOAD_MB} MB.")
                st.stop()
            try:
                shared_pil = Image.open(uploaded).convert("L")
            except Exception:
                st.error("Could not decode the uploaded image. Please upload a valid PNG or JPEG.")
                st.stop()
        else:
            shared_pil = None

    st.divider()
    st.caption("⚠️ Demo only — not a clinical tool.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔬 Medical Imaging AI — V2")
st.caption(
    "Multi-label classification · U-Net segmentation · "
    "Grad-CAM + ScoreCAM · DICOM explorer"
)
st.divider()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🏷️ Multi-label Classifier",
    "🗺️ U-Net Segmentation",
    "🔥 Grad-CAM / ScoreCAM",
    "📋 DICOM Explorer",
])


# ================================================================ TAB 1 ====
with tab1:
    st.header("Multi-label Pathology Classification")
    st.markdown(
        "Simultaneously detects **5 independent pathologies** using sigmoid "
        "outputs (not softmax). Each label is classified independently."
    )

    if shared_pil is None:
        st.info("Select or upload an image in the sidebar.")
        st.stop()

    tensor = pil_to_tensor(shared_pil)
    model = load_multilabel_model()

    with st.spinner("Running multi-label inference …"):
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    prob_dict = {cls: float(p) for cls, p in zip(MULTILABEL_CLASSES, probs)}

    col_img, col_chart = st.columns([1, 1.6])
    with col_img:
        st.subheader("Input X-Ray")
        st.image(shared_pil, caption="Grayscale (224×224)", use_column_width=True)

    with col_chart:
        st.subheader("Per-Label Probabilities")
        st.plotly_chart(make_label_bar_chart(prob_dict), use_container_width=True)

        st.subheader("Detected Pathologies (p ≥ 0.5)")
        import html as _html
        chips = ""
        found_any = False
        for cls, p in prob_dict.items():
            if p >= 0.5:
                # cls comes from a fixed MULTILABEL_CLASSES list (not user input),
                # but escape defensively in case the list ever includes external data.
                color = LABEL_COLORS.get(cls, "#8b949e")
                safe_cls = _html.escape(str(cls))
                safe_pct = _html.escape(f"{p:.0%}")
                chips += (
                    f"<span class='label-chip' style='background:{color}20;"
                    f"color:{color};border:1px solid {color}'>{safe_cls} {safe_pct}</span> "
                )
                found_any = True
        if not found_any:
            chips = "<span style='color:#8b949e'>No pathology detected above threshold</span>"
        st.markdown(chips, unsafe_allow_html=True)

    with st.expander("Raw probabilities"):
        for cls, p in prob_dict.items():
            st.write(f"**{cls}**: {p:.4f}")


# ================================================================ TAB 2 ====
with tab2:
    st.header("U-Net Pixel-Level Segmentation")
    st.markdown(
        "U-Net with 4-block encoder/decoder + skip connections predicts a "
        "**binary lesion mask** (pixel-wise) overlaid on the original X-ray."
    )

    if shared_pil is None:
        st.info("Select or upload an image in the sidebar.")
        st.stop()

    seg_model = load_seg_model()
    tensor = pil_to_tensor(shared_pil)

    with st.spinner("Running U-Net segmentation …"):
        inp = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.no_grad():
            logit = seg_model(inp)
            prob_map = torch.sigmoid(logit).squeeze().cpu().numpy()
            binary   = (prob_map >= 0.5).astype(np.uint8) * 255

    # Overlay mask on original
    original_arr = np.array(shared_pil.resize((256, 256)).convert("RGB"))
    mask_rgb = np.zeros_like(original_arr)
    mask_rgb[:, :, 0] = binary  # Red channel = mask

    blended = (original_arr.astype(np.float32) * 0.6 + mask_rgb.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Original")
        st.image(shared_pil.resize((256, 256)), caption="Input", use_column_width=True)
    with col_b:
        st.subheader("Segmentation Mask")
        st.image(binary, caption="Predicted lesion (white)", use_column_width=True)
    with col_c:
        st.subheader("Overlay")
        st.image(blended, caption="Mask overlaid (red = lesion)", use_column_width=True)

    coverage = float(binary.sum()) / (256 * 256 * 255) * 100
    st.metric("Lesion coverage", f"{coverage:.1f}%")


# ================================================================ TAB 3 ====
with tab3:
    st.header("Grad-CAM vs ScoreCAM Comparison")
    st.markdown(
        "**Grad-CAM** uses backward-pass gradients. "
        "**ScoreCAM** scores each feature map as an independent mask — "
        "no backward pass required (more faithful attribution)."
    )

    if shared_pil is None:
        st.info("Select or upload an image in the sidebar.")
        st.stop()

    from model.gradcam import GradCAM, overlay_heatmap
    from explainability.gradcam_video import ScoreCAM

    cls_model = load_multilabel_model()
    tensor = pil_to_tensor(shared_pil)

    target_cls_name = st.selectbox(
        "Explain class", MULTILABEL_CLASSES, index=0, key="explain_cls"
    )
    target_cls_idx = MULTILABEL_CLASSES.index(target_cls_name)

    with st.spinner("Computing Grad-CAM …"):
        with GradCAM(cls_model) as gcam:
            gcam_map = gcam.compute(tensor, target_class=target_cls_idx)

    with st.spinner("Computing ScoreCAM (slower — no backward pass) …"):
        with ScoreCAM(cls_model, batch_size=8) as scam:
            scam_map = scam.compute(tensor, target_class=target_cls_idx)

    original_arr = np.array(shared_pil.resize((224, 224)).convert("L"))
    gcam_overlay = Image.fromarray(overlay_heatmap(original_arr, gcam_map))
    scam_overlay = Image.fromarray(overlay_heatmap(original_arr, scam_map))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(shared_pil, use_column_width=True)
    with col2:
        st.subheader("Grad-CAM")
        st.image(gcam_overlay, caption=f"Gradient-based ({target_cls_name})", use_column_width=True)
    with col3:
        st.subheader("ScoreCAM")
        st.image(scam_overlay, caption=f"Perturbation-based ({target_cls_name})", use_column_width=True)

    st.info(
        "Tip: ScoreCAM can differ substantially from Grad-CAM — "
        "it is less susceptible to gradient saturation artefacts."
    )


# ================================================================ TAB 4 ====
with tab4:
    st.header("DICOM Metadata Explorer")
    st.markdown(
        "Generates synthetic DICOM-like metadata and demonstrates "
        "**DICOM windowing** (CT HU and X-ray) for model preprocessing."
    )

    from preprocessing.dicom_handler import (
        generate_synthetic_dicom,
        apply_windowing,
        apply_hu_windowing,
        preprocess_dicom,
        _CT_WINDOW_PRESETS,
    )

    col_ctrl, col_view = st.columns([1, 2])

    with col_ctrl:
        dicom_modality = st.selectbox("Modality", ["CT", "CR", "DX"], index=0)
        dicom_seed = st.slider("Patient seed", 0, 99, 42)

        if dicom_modality == "CT":
            preset = st.selectbox("CT Window Preset", list(_CT_WINDOW_PRESETS.keys()), index=0)
        else:
            preset = None

    with st.spinner("Generating synthetic DICOM …"):
        ds = generate_synthetic_dicom(modality=dicom_modality, size=256, seed=dicom_seed)

    with col_ctrl:
        st.markdown("### Metadata")
        # PHI fields (PatientName, PatientID) are intentionally excluded from
        # display even for synthetic data — the pattern must be safe for real DICOMs.
        # If you need to show patient identity for clinical review, use a fully
        # de-identified display layer with role-based access control.
        meta_items = {
            "StudyDate":     ds.StudyDate,
            "Modality":      ds.Modality,
            "SOPInstanceUID": ds.SOPInstanceUID[:30] + "…",
            "Rows×Cols":     f"{ds.Rows}×{ds.Columns}",
            "BitsAllocated": ds.BitsAllocated,
            "PixelSpacing":  f"{ds.PixelSpacing[0]:.3f} mm",
            "WindowCenter":  ds.WindowCenter,
            "WindowWidth":   ds.WindowWidth,
        }
        if dicom_modality == "CT":
            meta_items["RescaleIntercept"] = ds.RescaleIntercept
            meta_items["RescaleSlope"]     = ds.RescaleSlope

        import html as _html
        for k, v in meta_items.items():
            safe_k = _html.escape(str(k))
            safe_v = _html.escape(str(v))
            st.write(f"**{safe_k}**: `{safe_v}`")

    with col_view:
        st.subheader("Pixel Arrays")

        if dicom_modality == "CT":
            hu = ds.hu_array
            raw_display = np.clip(
                (hu - hu.min()) / (hu.max() - hu.min() + 1e-8) * 255, 0, 255
            ).astype(np.uint8)
            windowed = apply_hu_windowing(hu, preset=preset or "lung")
        else:
            raw_display = ds.pixel_array
            windowed = apply_windowing(ds.pixel_array, ds.WindowCenter, ds.WindowWidth)

        model_input = preprocess_dicom(ds)
        model_display = np.clip((model_input.squeeze() + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(raw_display, caption="Raw stored pixels", use_column_width=True)
        with c2:
            st.image(windowed, caption=f"After windowing ({preset or 'standard'})", use_column_width=True)
        with c3:
            st.image(model_display, caption="Model input (normalised [-1,1])", use_column_width=True)

        if dicom_modality == "CT":
            st.markdown(
                f"**HU range**: [{float(ds.hu_array.min()):.0f}, {float(ds.hu_array.max()):.0f}] HU  |  "
                f"**Window**: center={ds.WindowCenter}, width={ds.WindowWidth}"
            )
