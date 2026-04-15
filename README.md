# NeuroPlus AI — Brain Tumor Classifier

An end-to-end deep learning project for brain tumor classification using MRI images. The project uses transfer learning and provides an interactive web interface for real-time predictions.

---

## Project Overview

NeuroPlus AI classifies brain MRI scans into four categories:
- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

The model achieves ~89.06% accuracy and is deployed using a Streamlit web application.

---

## Model Details

- Architecture: Transfer Learning (MobileNetV2 / CNN)  
- Framework: TensorFlow / Keras  
- Input Size: 224 × 224  
- Classes: 4 (Multi-class classification)  
- Accuracy: ~89.06%  

---

## Features

- Real-time MRI image classification  
- Interactive interface built with Streamlit  
- Confidence score visualization  
- Probability distribution charts  
- Clean and user-friendly UI  

---

## Installation

1. Clone the repository:
   git clone https://github.com/athhh07/Neuroplus-AI.git
   cd Neuroplus-AI

3. Install dependencies:
   pip install -r requirements.txt
   
---

## Run the Web App
cd webapp
streamlit run app.py

---

## Model Evaluation

- Accuracy: ~89.06%  
- Evaluation metrics:
  - Confusion Matrix  
  - Classification Report  
  - Accuracy and Loss graphs  

---

## Disclaimer

This project is intended for educational and research purposes only. It is not a certified medical tool and should not be used for clinical diagnosis.

---

## Author

Atharva Desai

---

## Future Improvements

- Improve model accuracy with advanced architectures  
- Add explainability (Grad-CAM)  
- Deploy on cloud platforms  
- Optimize inference performance  

---

## Support

If you found this project useful, consider starring the repository.

Thank You!!!
