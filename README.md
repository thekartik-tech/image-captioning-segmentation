# 🧠 Image Captioning and Segmentation 
 
📸 **Image Captioning** — generating textual descriptions of images  
🖼️ **Image Segmentation** — identifying and labeling regions in an image

This project demonstrates a deep learning pipeline that integrates both systems and deploys them via a Streamlit web interface.

---

## 🎯 Project Objectives

- Understand and implement deep learning models for image captioning and segmentation.
- Perform semantic segmentation using U-Net (or similar).
- Generate natural language captions using CNN + LSTM (or Transformer-based).
- Integrate both models into a unified Streamlit app.
- Train and evaluate models using datasets like **MS COCO** or **Pascal VOC**.

---

## 🔧 Tech Stack & Tools

- `Python`
- `PyTorch` / `TensorFlow`
- `OpenCV` for image preprocessing
- `NLTK` / `spaCy` for language preprocessing
- `Streamlit` for deployment
- `Jupyter Notebook` for experimentation

---

## 📦 Dataset

- Trained on a **small subset of the MS COCO dataset** for both captioning and segmentation.
- Includes **dummy data** for segmentation training with synthetic masks.
- Supports future use of Pascal VOC or full COCO datasets.

---

## 📁 Key Components

- `CNNEncoder` + `LSTMDecoder` for image captioning
- `TinyUNet` for image segmentation
- Streamlit interface for image upload, caption generation, and mask visualization
- Script-based training, inference, and integration modules

---


