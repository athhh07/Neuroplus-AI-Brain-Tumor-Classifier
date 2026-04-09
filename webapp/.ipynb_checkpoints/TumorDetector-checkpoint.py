import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPlus AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

/* ── Root palette ─────────────────────────────── */
:root {
    --navy:       #050d1f;
    --navy-mid:   #0a1628;
    --navy-card:  #0e1e38;
    --navy-border:#152342;
    --teal:       #00c9b1;
    --teal-glow:  rgba(0,201,177,0.18);
    --teal-soft:  rgba(0,201,177,0.08);
    --cyan:       #38bdf8;
    --amber:      #f59e0b;
    --rose:       #f43f5e;
    --green:      #10b981;
    --text-pri:   #e2eaf8;
    --text-sec:   #7a91b8;
    --text-muted: #3d5378;
    --shadow-lg:  0 24px 60px rgba(0,0,0,0.55);
    --shadow-teal:0 0 32px rgba(0,201,177,0.22);
}

/* ── Base ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy) !important;
    color: var(--text-pri) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%,
        rgba(0,201,177,0.08) 0%, transparent 65%),
        var(--navy);
}

[data-testid="stHeader"]        { background: transparent !important; }
[data-testid="stSidebar"]       { background: var(--navy-mid) !important; }
.block-container                { padding: 2rem 3rem 4rem !important; max-width: 1300px; }

/* ── Hero header ──────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 3.2rem 1rem 2.4rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: var(--teal-soft);
    border: 1px solid rgba(0,201,177,0.35);
    border-radius: 999px;
    padding: 0.35rem 1.1rem;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 1.1rem;
    font-weight: 600;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    color: var(--text-pri);
    line-height: 1.15;
    margin: 0 0 0.6rem;
}
.hero-title span { color: var(--teal); }
.hero-subtitle {
    color: var(--text-sec);
    font-size: 1rem;
    font-weight: 400;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Stat pills ───────────────────────────────── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin: 1.8rem 0 2.4rem;
}
.stat-pill {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 0.6rem 1.3rem;
    text-align: center;
    min-width: 130px;
}
.stat-pill .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--teal);
}
.stat-pill .lbl {
    font-size: 0.72rem;
    color: var(--text-sec);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1px;
}

/* ── Cards ────────────────────────────────────── */
.card {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
        rgba(0,201,177,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--teal);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* ── Upload zone ──────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--navy-border) !important;
    border-radius: 16px !important;
    background: var(--teal-soft) !important;
    transition: border-color 0.25s;
    padding: 1.2rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--teal) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-sec) !important;
    font-size: 0.9rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Result badge ─────────────────────────────── */
.result-badge {
    background: linear-gradient(135deg, rgba(0,201,177,0.15), rgba(56,189,248,0.1));
    border: 1.5px solid rgba(0,201,177,0.45);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.result-icon { font-size: 2rem; line-height: 1; }
.result-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--teal);
    font-weight: 600;
    margin-bottom: 2px;
}
.result-class {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-pri);
}
.result-conf {
    font-size: 0.85rem;
    color: var(--text-sec);
    margin-top: 2px;
}

/* ── Confidence bar ───────────────────────────── */
.conf-bar-wrap { margin-bottom: 0.9rem; }
.conf-bar-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-sec);
    margin-bottom: 5px;
}
.conf-bar-header span:last-child {
    color: var(--teal);
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
}
.conf-bar-bg {
    height: 7px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Info grid ────────────────────────────────── */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
}
.info-tile {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--navy-border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.info-tile .it-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 3px;
}
.info-tile .it-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-pri);
}

/* ── Section heading ──────────────────────────── */
.section-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: var(--teal);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--navy-border);
    margin: 2rem 0 1.2rem;
}

/* ── Warning box ──────────────────────────────── */
.warn-box {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    color: #fcd34d;
    font-size: 0.85rem;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    line-height: 1.55;
}

/* ── Streamlit overrides ──────────────────────── */
h1,h2,h3 { color: var(--text-pri) !important; font-family: 'Space Grotesk', sans-serif !important; }
p, label, div { color: var(--text-pri); }

/* spinner */
[data-testid="stSpinner"] { color: var(--teal) !important; }

/* image */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--navy-border);
}

/* footer */
.footer {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--navy-border);
    margin-top: 3rem;
}
.footer span { color: var(--teal); }

/* hide default streamlit elements */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── IMAGE SIZE CONSTANT ─────────────────────────────────────────────────────────
IMG_SIZE = 224


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/brain_tumor_classifier_model.h5")

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# Class config — label, bar color, emoji
CLASSES = [
    ("Glioma Tumor",      "#f43f5e", "🔴"),
    ("Meningioma Tumor",  "#f59e0b", "🟠"),
    ("Pituitary Tumor",   "#38bdf8", "🔵"),
    ("No Tumor",          "#10b981", "🟢"),
]
CLASS_NAMES   = [c[0] for c in CLASSES]
CLASS_COLORS  = [c[1] for c in CLASSES]
CLASS_EMOJIS  = [c[2] for c in CLASSES]

BAR_GRADIENT = {
    "Glioma Tumor":     "linear-gradient(90deg,#f43f5e,#fb7185)",
    "Meningioma Tumor": "linear-gradient(90deg,#f59e0b,#fbbf24)",
    "Pituitary Tumor":  "linear-gradient(90deg,#0ea5e9,#38bdf8)",
    "No Tumor":         "linear-gradient(90deg,#059669,#10b981)",
}


# ─── HERO ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-badge">🧠 AI-Powered Neuroimaging</div>
  <h1 class="hero-title">NeuroPlus <span>AI</span></h1>
  <p class="hero-subtitle">
    Upload a brain MRI scan and receive instant AI-assisted classification
    across four tumor categories with confidence analysis.
  </p>
</div>

<div class="stats-row">
  <div class="stat-pill"><div class="val">4</div><div class="lbl">Tumor Classes</div></div>
  <div class="stat-pill"><div class="val">{IMG_SIZE}×{IMG_SIZE}</div><div class="lbl">Input Resolution</div></div>
  <div class="stat-pill"><div class="val">CNN</div><div class="lbl">Architecture</div></div>
  <div class="stat-pill"><div class="val">Real-time</div><div class="lbl">Inference</div></div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.markdown("""<div class="warn-box">⚠️ &nbsp;
    <div>Model file <code>brain_tumor_classifier_model.h5</code> not found.
    Place the trained model in the <code>models/</code> directory and restart the app.</div></div>""",
    unsafe_allow_html=True)
    st.stop()


# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 &nbsp;MRI Image Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded MRI Scan")
        st.markdown(f"""
        <div class="info-grid">
          <div class="info-tile">
            <div class="it-label">Filename</div>
            <div class="it-val" style="font-size:0.82rem;word-break:break-all">{uploaded_file.name}</div>
          </div>
          <div class="info-tile">
            <div class="it-label">Dimensions</div>
            <div class="it-val">{image.size[0]} × {image.size[1]} px</div>
          </div>
          <div class="info-tile">
            <div class="it-label">Mode</div>
            <div class="it-val">{image.mode}</div>
          </div>
          <div class="info-tile">
            <div class="it-label">Size</div>
            <div class="it-val">{uploaded_file.size / 1024:.1f} KB</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#3d5378;">
          <div style="font-size:3rem;margin-bottom:0.6rem;opacity:0.5">🧠</div>
          <div style="font-size:0.88rem">Supported formats: JPG · JPEG · PNG</div>
          <div style="font-size:0.78rem;margin-top:6px;color:#2a3e60">Grayscale or RGB MRI scans accepted</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔬 &nbsp;Analysis Results</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center;padding:3.5rem 1rem;color:#3d5378;">
          <div style="font-size:3rem;margin-bottom:0.8rem;opacity:0.4">📊</div>
          <div style="font-size:0.88rem;color:#4a6080">
            Upload an MRI image to view<br>the classification results here
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing MRI scan…"):
            time.sleep(0.4)          # brief pause for UX feel
            prediction = model.predict(img_array, verbose=0)

        pred_index = int(np.argmax(prediction))
        pred_class = CLASS_NAMES[pred_index]
        pred_emoji = CLASS_EMOJIS[pred_index]
        confidence = prediction[0][pred_index] * 100
        is_tumor   = pred_class != "No Tumor"

        # ── Result badge ──────────────────────────────────────
        status_color = "#f43f5e" if is_tumor else "#10b981"
        status_label = "TUMOR DETECTED" if is_tumor else "NO TUMOR DETECTED"
        st.markdown(f"""
        <div class="result-badge">
          <div class="result-icon">{pred_emoji}</div>
          <div>
            <div class="result-label">{status_label}</div>
            <div class="result-class">{pred_class}</div>
            <div class="result-conf">Confidence: <strong style="color:{CLASS_COLORS[pred_index]}">{confidence:.2f}%</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Per-class probability bars ─────────────────────────
        st.markdown('<div style="margin-top:1.2rem"></div>', unsafe_allow_html=True)
        for i, (cname, ccolor, _) in enumerate(CLASSES):
            pct = prediction[0][i] * 100
            gradient = BAR_GRADIENT[cname]
            bold = "font-weight:700;" if i == pred_index else ""
            st.markdown(f"""
            <div class="conf-bar-wrap">
              <div class="conf-bar-header">
                <span style="{bold}color:{'var(--text-pri)' if i==pred_index else 'var(--text-sec)'}">{cname}</span>
                <span>{pct:.1f}%</span>
              </div>
              <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{gradient};
                {'box-shadow:0 0 8px ' + ccolor + '88;' if i==pred_index else ''}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── CHART SECTION (only when prediction is done) ───────────────────────────────
if uploaded_file is not None:
    st.markdown('<div class="section-head">📈 &nbsp;Probability Distribution</div>',
                unsafe_allow_html=True)

    chart_col, table_col = st.columns([3, 2], gap="large")

    with chart_col:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        fig.patch.set_facecolor("#0e1e38")
        ax.set_facecolor("#0e1e38")

        bars = ax.bar(CLASS_NAMES, prediction[0],
                      color=CLASS_COLORS, width=0.55,
                      zorder=3, edgecolor="none")

        # highlight winner
        bars[pred_index].set_edgecolor(CLASS_COLORS[pred_index])
        bars[pred_index].set_linewidth(2)

        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Probability", color="#7a91b8", fontsize=9, labelpad=8)
        ax.tick_params(axis="x", colors="#7a91b8", labelsize=8.5, rotation=15)
        ax.tick_params(axis="y", colors="#4a6080", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.yaxis.grid(True, color="#152342", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        # value labels on bars
        for bar, val in zip(bars, prediction[0]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.025, f"{val*100:.1f}%",
                    ha="center", va="bottom",
                    color="#e2eaf8", fontsize=8.5, fontweight="bold")

        plt.tight_layout(pad=0.4)
        st.pyplot(fig)
        plt.close()

    with table_col:
        st.markdown('<div class="card" style="padding:1.2rem 1.5rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗂 &nbsp;Detailed Breakdown</div>', unsafe_allow_html=True)

        for i, (cname, ccolor, cemoji) in enumerate(CLASSES):
            pct = prediction[0][i] * 100
            highlight = f"border-left:3px solid {ccolor};" if i == pred_index else \
                        "border-left:3px solid transparent;"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.55rem 0.8rem;margin-bottom:0.45rem;
                        background:rgba(255,255,255,0.03);border-radius:9px;{highlight}">
              <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.84rem;
                          color:{'var(--text-pri)' if i==pred_index else 'var(--text-sec)'}">
                {cemoji} &nbsp;{cname}
              </div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;
                          font-weight:700;color:{ccolor}">{pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Clinical disclaimer ───────────────────────────────────
    st.markdown("""
    <div class="warn-box" style="margin-top:2rem;">
      ⚕️ &nbsp;
      <div><strong>Clinical Disclaimer</strong> — This AI tool is intended for
      research and educational purposes only. Results should not be used as a substitute
      for professional medical diagnosis. Always consult a qualified radiologist or
      neurologist for clinical decisions.</div>
    </div>
    """, unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  NeuroPlus AI &nbsp;·&nbsp; Brain Tumor MRI Classification &nbsp;·&nbsp;
  Built with <span>TensorFlow</span> &amp; <span>Streamlit</span><br>
  <span style="color:#2a3e60;margin-top:4px;display:block">
    For research use only — not a certified medical device
  </span>
</div>
""", unsafe_allow_html=True)