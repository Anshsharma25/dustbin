from flask import Flask, Response
import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import time

# Initialize Flask app
app = Flask(__name__)

# Load model paths from pickle file
with open('model_paths.pkl', 'rb') as f:
    model_paths = pickle.load(f)

# Load YOLO models
garbage_model = YOLO(model_paths['garbage_model_path'])
dry_wet_model = YOLO(model_paths['dry_wet_model_path'])

# ESP32-CAM stream URL
ESP32_CAM_URL = "http://192.168.1.18/cam-hi.jpg"  # Replace with your ESP32-CAM URL

# Frame settings
fps_limit = 10  # Limit FPS to reduce processing load
last_frame_time = 0

# Function to fetch frames from ESP32-CAM
def fetch_frame():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Function to generate video feed
def generate_video():
    global last_frame_time

    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1 / fps_limit:
            time.sleep(0.01)
            continue

        # Fetch frame from ESP32-CAM
        frame = fetch_frame()
        if frame is None:
            print("Failed to fetch frame from ESP32-CAM.")
            time.sleep(0.5)
            continue

        # Process frame with YOLO model
        garbage_results = garbage_model(frame)

        # Draw bounding boxes and labels on frame
        for result in garbage_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = box.cls[0]

                # Draw bounding box and label
                color = (0, 255, 0) if conf >= 0.5 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{garbage_model.names[int(class_id)]} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        # Update last frame time
        last_frame_time = current_time

# Flask route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)