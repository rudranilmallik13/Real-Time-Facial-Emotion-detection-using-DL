# 🎭 Real-Time Facial Emotion Detection

A computer vision and deep learning project that detects **human facial emotions in real time** using a webcam.  
The system analyzes live video frames and classifies emotions such as **Happy, Sad, Angry, Surprise, Fear, and Neutral**.

---

## 🚀 Project Overview

This project combines **computer vision** and **deep learning** to understand facial expressions from live video input.  
A trained **CNN model** classifies emotions after detecting faces frame by frame.

---

## ✨ Features

- 🎥 Real-time emotion detection using a webcam  
- 🧠 Deep learning–based **CNN model**  
- 🙂 Detects multiple emotions:
  - Happy  
  - Sad  
  - Angry  
  - Surprise  
  - Fear  
  - Neutral  
- 🧑 Face detection + emotion classification  
- ⚡ Fast, lightweight, and suitable for real-time use  

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Haar Cascade / DNN Face Detector**

---


---

## 📊 Dataset

- **FER-2013 Facial Emotion Dataset**
- Grayscale facial expression images
- Labels mapped to emotion classes:
  - Happy, Sad, Angry, Surprise, Fear, Neutral

---

## 🧠 Model Architecture

The emotion classifier is built using a **Convolutional Neural Network (CNN)**:

- Convolution layers + ReLU activation  
- MaxPooling layers  
- Dropout layers to reduce overfitting  
- Fully connected (Dense) layers  
- **Softmax output layer** for multi-class emotion prediction  

---

## ▶️ How It Works

1. Webcam captures live video frames  
2. Face detected using Haar Cascade / DNN  
3. Face region is preprocessed and passed to CNN  
4. Model predicts emotion for each frame  
5. Emotion label is displayed in real time  

---

## 🎯 Use Cases

- Human–computer interaction  
- Emotion-aware applications  
- AI-based surveillance systems  
- Learning project for **Computer Vision & Deep Learning**



