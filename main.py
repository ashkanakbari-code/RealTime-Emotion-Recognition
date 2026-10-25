import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

DEVICE = torch.device("cpu")
MODEL_PATH = "emotion_cnn_model.pth"
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def ensure_haar_cascade_exists():
    cascade_filename = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_filename):
        print("Downloading Haar Cascade XML file...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
            print("Haar Cascade downloaded successfully!")
        except Exception as e:
            print(f"Error downloading cascade file: {e}")
    return cascade_filename

def start_emotion_detector():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights file '{MODEL_PATH}' not found! Please run train.py first.")
        return

    print("Loading trained emotion model...")
    model = EmotionCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully")

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"

    use_dnn = os.path.exists(modelFile) and os.path.exists(configFile)
    net = None
    face_cascade = None

    if use_dnn:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        print("Using OpenCV DNN Face Detector.")
    else:
        print("DNN files not found. Falling back to robust Haar Cascade...")
        cascade_path = ensure_haar_cascade_exists()
        face_cascade = cv2.CascadeClassifier(cascade_path)

    transform_pipeline = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    cap = None
    for index in [0, 1, 2, -1]:
        temp_cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"Successfully connected and verified camera index {index}")
                break
            else:
                temp_cap.release()
                
    if cap is None:
        print("Error: Could not find any working webcam that streams frames.")
        return

    print("Real-time emotion detector started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        h, w, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        if use_dnn and net is not None:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence_det = detections[0, 0, i, 2]
                if confidence_det > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x and y2 > y:
                        faces.append((x, y, x2 - x, y2 - y))
        else:
            detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = [(x, y, w_box, h_box) for (x, y, w_box, h_box) in detected_faces]

        for (x, y, box_w, box_h) in faces:
            x2, y2 = min(w, x + box_w), min(h, y + box_h)
            face_roi = gray_frame[y:y2, x:x2]

            if face_roi.size > 0:
                try:
                    tensor_face = transform_pipeline(face_roi).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(tensor_face)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, dim=1)

                        emotion_text = CLASSES[predicted_idx.item()]
                        confidence_score = confidence.item() * 100
                    
                    label = f"{emotion_text} ({confidence_score:.1f}%)"
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    pass

        cv2.imshow("Real-time Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    start_emotion_detector()