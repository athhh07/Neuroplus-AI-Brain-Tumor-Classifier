<div align="center">

# 🧠 NeuroPlus AI — Brain Tumor Classifier

### Deep Learning–powered MRI classification for glioma, meningioma, pituitary tumors, and healthy scans

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV2-blue)](#-model-details)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-89.06%25-success)](#-performance)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

**[🚀 Live Demo](#-live-demo) · [📦 Installation](#-installation) · [📊 Performance](#-performance) · [🧩 How It Works](#-how-it-works)**

</div>

---

## 📋 Overview

**NeuroPlus AI** is an end-to-end deep learning pipeline that classifies brain MRI scans into four categories — **glioma**, **meningioma**, **pituitary tumor**, or **no tumor** — using transfer learning on **MobileNetV2**. The trained model is deployed as an interactive **Streamlit** web app, allowing anyone to upload an MRI scan and get instant predictions with confidence scores.

> ⚠️ **Disclaimer:** This project is built for educational and research purposes only. It is **not** a certified medical diagnostic tool and should never be used as a substitute for professional medical evaluation.

---

## 🚀 Live Demo

🔗 **[Try the app here](#)** *https://neuroplus.streamlit.app/*

<div align="center">

| Upload MRI Scan | Get Instant Prediction |
|:---:|:---:|
| Drag & drop or browse an MRI image | View tumor type + confidence score |

</div>

---

## ✨ Features

- 🖼️ **Real-time MRI classification** — upload a scan and get results in seconds
- 📊 **Confidence score visualization** — see exactly how certain the model is
- 📈 **Probability distribution charts** across all four classes
- 🎨 **Clean, intuitive Streamlit UI** — no technical knowledge required
- ⚡ **Lightweight model (MobileNetV2)** — fast inference, easy to deploy

---

## 🧩 How It Works?

```
MRI Image Input
      │
      ▼
Preprocessing (Resize → 224×224, Normalize)
      │
      ▼
MobileNetV2 (Transfer Learning Backbone)
      │
      ▼
Custom Classification Head
      │
      ▼
Softmax Output → [Glioma | Meningioma | No Tumor | Pituitary]
      │
      ▼
Streamlit UI → Prediction + Confidence Chart
```

---

## 🏗️ Model Details

| Attribute | Detail |
|---|---|
| **Architecture** | Transfer Learning — MobileNetV2 |
| **Framework** | TensorFlow / Keras |
| **Input Size** | 224 × 224 × 3 |
| **Classes** | 4 — Glioma, Meningioma, No Tumor, Pituitary |
| **Task Type** | Multi-class image classification |
| **Test Accuracy** | **89.06%** |

---

## 📊 Performance

### Overall Test Accuracy: **89.06%**

### 📈 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| 🔴 Glioma | 0.90 | 0.79 | 0.84 | 400 |
| 🟠 Meningioma | 0.80 | 0.84 | 0.82 | 400 |
| 🟢 No Tumor | 0.95 | 0.97 | 0.96 | 400 |
| 🔵 Pituitary | 0.92 | 0.96 | 0.94 | 400 |
| **Macro Avg** | **0.89** | **0.89** | **0.89** | 1600 |
| **Weighted Avg** | **0.89** | **0.89** | **0.89** | 1600 |

**Key takeaways:**
- The model performs best on **no-tumor** and **pituitary** classes (F1 > 0.94).
- **Glioma** has the lowest recall (0.79), indicating some gliomas are misclassified — a good candidate for future improvement (e.g., more augmentation, fine-tuning deeper layers).
- Overall, the model generalizes well across a balanced 1,600-image test set (400 per class).

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/athhh07/Neuroplus-AI-Brain-Tumor-Classifier.git
cd Neuroplus-AI-Brain-Tumor-Classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
cd webapp
streamlit run app.py
```

The app will launch in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Neuroplus-AI-Brain-Tumor-Classifier/
├── Samples/                # Sample MRI images for quick testing
├── brain_tumor_dataset/    # Training/testing dataset
├── models/                 # Saved trained model(s)
├── outputs/                # Evaluation outputs (plots, reports)
├── source_code/            # Model training & evaluation scripts
├── webapp/                 # Streamlit application
│   └── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model:** MobileNetV2 (Transfer Learning)
- **Web Framework:** Streamlit
- **Visualization:** Matplotlib / Seaborn

---

## 🔭 Future Improvements

- [ ] Improve glioma recall with targeted data augmentation
- [ ] Add Grad-CAM visualizations for model explainability
- [ ] Experiment with EfficientNet / ResNet backbones for comparison
- [ ] Deploy on cloud (Hugging Face Spaces / Render / AWS)
- [ ] Add batch prediction support
- [ ] Optimize inference speed with TensorFlow Lite

---

## ⚠️ Disclaimer

This project is intended **strictly for educational and research purposes**. It is **not** a certified medical device and must not be used for actual clinical diagnosis. Always consult a qualified medical professional for health-related decisions.

---

## 👤 Author

**Atharva Desai**

[![GitHub](https://img.shields.io/badge/GitHub-athhh07-181717?style=flat&logo=github)](https://github.com/athhh07)

---

## ⭐ Support

If you found this project useful, please consider **starring** ⭐ the repository — it helps a lot and motivates further development!

</div>
