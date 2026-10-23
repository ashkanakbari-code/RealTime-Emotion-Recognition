import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import urllib.request

def ensure_haar_cascade_exists():
    cascade_filename = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_filename):
        print("Downloading Haar Cascade file...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Error downloading cascade file: {e}")
    return cascade_filename

def start_emotion_detector():
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

    cascade_path = ensure_haar_cascade_exists()
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Error: Failed to load Haar Cascade classifier.")
        return

    print("Emotion detector started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    start_emotion_detector()