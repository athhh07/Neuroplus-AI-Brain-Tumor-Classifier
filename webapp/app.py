import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── CONFIG ─────────────────────────────────────
st.set_page_config(
    page_title="NeuroPlus AI",
    page_icon="🧠",
    layout="wide"
)

IMG_SIZE = 224

# ─── LOAD MODEL ─────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../models/brain_tumor_classifier_model.h5")

try:
    model = load_model()
except:
    st.error("⚠️ Model not found. Please add model file.")
    st.stop()

# ─── CLASSES ────────────────────────────────────
CLASSES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# ─── HEADER ─────────────────────────────────────
st.title("🧠 NeuroPlus AI")
st.write("AI-based Brain Tumor Detection from MRI")

st.divider()

# ─── MAIN LAYOUT ────────────────────────────────
col1, col2 = st.columns(2)

# ─── LEFT SIDE ──────────────────────────────────
with col1:
    st.subheader("📤 Upload MRI Image")

    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        st.write("### Image Details")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Filename:**", uploaded_file.name)
            st.write("**Width:**", image.size[0])

        with c2:
            st.write("**Height:**", image.size[1])
            st.write("**Mode:**", image.mode)

# ─── RIGHT SIDE ─────────────────────────────────
with col2:
    st.subheader("🔬 Analysis Results")

    if uploaded_file:
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing..."):
            prediction = model.predict(img_array)

        pred_index = np.argmax(prediction)
        pred_class = CLASSES[pred_index]
        confidence = prediction[0][pred_index] * 100

        # Result
        if pred_class == "No Tumor":
            st.success(f"🟢 {pred_class}")
        else:
            st.error(f"🔴 {pred_class}")

        st.write(f"**Confidence:** {confidence:.2f}%")

        st.write("### Class Probabilities")

        for i, cname in enumerate(CLASSES):
            pct = prediction[0][i]
            st.write(cname)
            st.progress(float(pct))

    else:
        st.info("Upload an image to see results")

# ─── CHART SECTION ──────────────────────────────
if uploaded_file:
    st.divider()
    st.subheader("📊 Probability Distribution")

    col_chart, col_table = st.columns([2, 1])

    # Chart
    with col_chart:
        fig, ax = plt.subplots()

        ax.bar(CLASSES, prediction[0])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        st.pyplot(fig)

    # Table
    with col_table:
        st.write("### Detailed Breakdown")

        for i, cname in enumerate(CLASSES):
            st.write(f"{cname}: {prediction[0][i]*100:.2f}%")

# ─── DISCLAIMER ─────────────────────────────────
st.divider()
st.warning("⚕️ This is not a medical diagnosis. Consult a doctor.")