# Real-Time Emotion Recognition System

A real-time facial emotion recognition system built with **PyTorch**, **OpenCV**, and a custom **Convolutional Neural Network (CNN)** trained on the FER-2013 dataset.

## 🚀 Features

- **Real-Time Processing**: Captures video frames from the webcam and detects faces in real time.
- **Robust Fallback Detection**: Uses OpenCV Haar Cascade for stable face-region (ROI) extraction.
- **Custom CNN Model**: A 3-block convolutional neural network that classifies faces into 7 basic emotions:
  `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

## 🛠️ Tech Stack

- **Python 3.x**
- **PyTorch & Torchvision**
- **OpenCV**
- **NumPy**

## ⚙️ Installation & Running

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RealTime-Emotion-Recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (if model weights are not present)

```bash
python train.py
```

### 4. Run the real-time emotion detector

```bash
python main.py
```

## 📊 Dataset

This project uses the **FER-2013** facial expression dataset to train the emotion classification model.

## 📝 Notes

- Ensure your webcam is connected and accessible before running the real-time detector.
- Replace `<your-repo-url>` with the actual URL of your repository.
- If trained model weights are already available in the expected location, you can skip the training step.
