import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
from xgboost import XGBClassifier
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WasteVision | ADVANZ Team 7",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    .main { background-color: #0d0d0d; }

    /* Header */
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        color: #00ff87;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        font-size: 1rem;
        color: #888;
        letter-spacing: 0.05em;
        margin-bottom: 2rem;
    }
    .team-badge {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #00ff87;
        color: #00ff87;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 4px 12px;
        border-radius: 2px;
        margin-bottom: 2rem;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #333;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: #111;
        transition: border-color 0.3s;
    }

    /* Result cards */
    .result-card {
        background: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card.svm { border-left: 4px solid #00ff87; }
    .result-card.xgb { border-left: 4px solid #00b4ff; }

    .model-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .predicted-class {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .svm .predicted-class { color: #00ff87; }
    .xgb .predicted-class { color: #00b4ff; }

    .confidence-val {
        font-size: 0.95rem;
        color: #888;
        margin-top: 0.3rem;
    }

    /* Agreement badge */
    .badge-agree {
        background: #0a2e1a;
        border: 1px solid #00ff87;
        color: #00ff87;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        padding: 6px 16px;
        border-radius: 2px;
        display: inline-block;
        margin: 1rem 0;
    }
    .badge-disagree {
        background: #2e1a00;
        border: 1px solid #ff8c00;
        color: #ff8c00;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        padding: 6px 16px;
        border-radius: 2px;
        display: inline-block;
        margin: 1rem 0;
    }

    /* Tip card */
    .tip-card {
        background: #111;
        border: 1px solid #1e3a2a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .tip-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #00ff87;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .tip-text { font-size: 0.9rem; color: #bbb; }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 2rem 0;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: #00ff87;
        color: #0d0d0d;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .stFileUploader { background: #111; border-radius: 8px; }
    div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WASTE HANDLING TIPS
# ─────────────────────────────────────────────
WASTE_TIPS = {
    "cardboard": "Flatten and bundle before placing in recycling. Keep dry — wet cardboard is not recyclable.",
    "glass":     "Rinse thoroughly and sort by color if required. Do not mix with ceramics or broken glass.",
    "metal":     "Rinse cans and foil. Metal is infinitely recyclable — always separate from general waste.",
    "paper":     "Keep dry and clean. Soiled or greasy paper (e.g. pizza boxes) goes to organic waste.",
    "plastic":   "Check resin code (1–7). Compress bottles to save space. Avoid single-use when possible.",
    "trash":     "General/residual waste. Minimize by reducing consumption and choosing recyclable packaging.",
}

def get_tip(class_name: str) -> str:
    key = class_name.lower().strip()
    for k, v in WASTE_TIPS.items():
        if k in key:
            return v
    return "Consult your local waste management authority for proper disposal guidance."

# ─────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B0...")
def load_extractor():
    base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    return Model(inputs=base.input, outputs=base.output)

@st.cache_resource(show_spinner="Loading scaler...")
def load_scaler():
    return joblib.load("Model/scaler.joblib")

@st.cache_resource(show_spinner="Loading SVM model...")
def load_svm():
    return joblib.load("Model/svm_model.joblib")

@st.cache_resource(show_spinner="Loading XGBoost model...")
def load_xgboost():
    model = XGBClassifier()
    model.load_model("Model/xgboost_model.json")
    return model

@st.cache_resource(show_spinner="Loading class names...")
def load_class_names():
    return np.load("features/class_names.npy", allow_pickle=True).tolist()

# ─────────────────────────────────────────────
# INFERENCE PIPELINE
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def predict(image: Image.Image):
    extractor   = load_extractor()
    scaler      = load_scaler()
    svm         = load_svm()
    xgb         = load_xgboost()
    class_names = load_class_names()

    processed = preprocess_image(image)
    features  = extractor.predict(processed, verbose=0)
    scaled    = scaler.transform(features)

    # SVM
    svm_pred  = svm.predict(scaled)[0]
    svm_proba = svm.predict_proba(scaled)[0] if hasattr(svm, "predict_proba") else None

    # XGBoost
    xgb_pred  = xgb.predict(scaled)[0]
    xgb_proba = xgb.predict_proba(scaled)[0]

    return {
        "class_names": class_names,
        "svm_class":   class_names[svm_pred] if isinstance(svm_pred, (int, np.integer)) else str(svm_pred),
        "xgb_class":   class_names[xgb_pred] if isinstance(xgb_pred, (int, np.integer)) else str(xgb_pred),
        "svm_proba":   svm_proba,
        "xgb_proba":   xgb_proba,
    }

# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">♻️ WasteVision</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Image-based waste classification · EfficientNet-B0 + SVM & XGBoost</div>', unsafe_allow_html=True)
st.markdown('<div class="team-badge">ADVANZ · TEAM 7 · UNESA</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown("#### Upload Waste Image")
    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the waste item"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)
        classify_btn = st.button("🔍 Classify Waste")
    else:
        st.markdown(
            '<div class="upload-zone">📂 Drag & drop or browse<br><small style="color:#555">JPG · JPEG · PNG</small></div>',
            unsafe_allow_html=True
        )
        classify_btn = False

with col_right:
    if uploaded and classify_btn:
        with st.spinner("Extracting features and classifying..."):
            try:
                results = predict(image)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        cn          = results["class_names"]
        svm_class   = results["svm_class"]
        xgb_class   = results["xgb_class"]
        svm_proba   = results["svm_proba"]
        xgb_proba   = results["xgb_proba"]
        agree       = svm_class.lower() == xgb_class.lower()

        # ── Result cards ──
        st.markdown("#### Classification Results")

        card_col1, card_col2 = st.columns(2)

        with card_col1:
            svm_conf = f"{max(svm_proba)*100:.1f}%" if svm_proba is not None else "N/A"
            st.markdown(f"""
            <div class="result-card svm">
                <div class="model-label">SVM</div>
                <div class="predicted-class">{svm_class}</div>
                <div class="confidence-val">Confidence: {svm_conf}</div>
            </div>
            """, unsafe_allow_html=True)

        with card_col2:
            xgb_conf = f"{max(xgb_proba)*100:.1f}%"
            st.markdown(f"""
            <div class="result-card xgb">
                <div class="model-label">XGBoost</div>
                <div class="predicted-class">{xgb_class}</div>
                <div class="confidence-val">Confidence: {xgb_conf}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Agreement badge ──
        if agree:
            st.markdown(f'<div class="badge-agree">✅ Both models agree → {svm_class.upper()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-disagree">⚠️ Models disagree — review image quality</div>', unsafe_allow_html=True)

        # ── Probability chart ──
        st.markdown("#### Confidence per Class")

        fig = go.Figure()

        if svm_proba is not None:
            fig.add_trace(go.Bar(
                name="SVM",
                x=cn,
                y=[round(p * 100, 2) for p in svm_proba],
                marker_color="#00ff87",
                opacity=0.85
            ))

        fig.add_trace(go.Bar(
            name="XGBoost",
            x=cn,
            y=[round(p * 100, 2) for p in xgb_proba],
            marker_color="#00b4ff",
            opacity=0.85
        ))

        fig.update_layout(
            barmode="group",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(color="#aaa", family="DM Sans"),
            legend=dict(bgcolor="#111", bordercolor="#222"),
            xaxis=dict(gridcolor="#1a1a1a", tickfont=dict(size=11)),
            yaxis=dict(gridcolor="#1a1a1a", title="Confidence (%)", range=[0, 100]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Handling tip ──
        final_class = svm_class if agree else xgb_class
        tip = get_tip(final_class)
        st.markdown(f"""
        <div class="tip-card">
            <div class="tip-label">♻️ Handling Tip</div>
            <div class="tip-text">{tip}</div>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="color:#444; font-family:'Space Mono',monospace; font-size:0.85rem; margin-top:4rem; text-align:center;">
            ← Upload an image to begin classification
        </div>
        """, unsafe_allow_html=True)