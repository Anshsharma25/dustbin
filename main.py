from flask import Flask, Response
from ultralytics import YOLO
import cv2
import time
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from pymongo import MongoClient
from datetime import datetime, timezone
from urllib.parse import quote_plus
import requests

# Initialize Flask app
app = Flask(__name__)

# MongoDB Setup
username = quote_plus("dustbin")  # Replace with your MongoDB username
password = quote_plus("Dustbin@123")  # Replace with your MongoDB password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.fmudd.mongodb.net/"
DATABASE_NAME = "garbage_detection"
COLLECTION_NAME = "detections"

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    detections_collection = db[COLLECTION_NAME]
    detections_collection.create_index("timestamp", expireAfterSeconds=7200)
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Function to insert detection data into MongoDB
def insert_detection_data(data):
    try:
        detections_collection.insert_one(data)
        print("Detection data saved to MongoDB.")
    except Exception as e:
        print(f"Error saving data to MongoDB: {e}")

# Load model paths from pickle file
with open('models_pickle.pkl', 'rb') as f:
    models = pickle.load(f)

# Load YOLO models using the paths from the pickle metadata
garbage_model = YOLO(models['garbage_model']['model_path'])
dry_wet_model = YOLO(models['drywet_model']['model_path'])
cover_noncover_model = YOLO(models['cover_uncover_model']['model_path'])
polythene_nonpoly_model = YOLO(models['polythene_nonpoly_model']['model_path'])

# KNN for distance estimation
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])
distance = np.array([3, 2, 1.5, 1, 0.5])
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)

# Constants
boundary_distance = 1.0
ESP32_CAM_URL = "http://192.168.1.100/cam-hi.jpg"
fps_limit = 40
last_frame_time = 0

# Fetch frame from ESP32-CAM
def fetch_frame():
    try:
        response = requests.get(ESP32_CAM_URL, stream=True, timeout=5)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error fetching frame: {e}")
    return None

# Draw a curved boundary line
def draw_curved_boundary(frame):
    frame_height, frame_width = frame.shape[:2]
    curve_depth = 30
    curve_center_y = frame_height // 2

    num_points = frame_width
    curve_points = []

    for x in range(num_points):
        t = x / (frame_width - 1)
        y = curve_center_y + int(curve_depth * np.sin(np.pi * t))
        curve_points.append((x, y))

    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)

    return frame

# Process and generate video feed
def generate_video():
    global last_frame_time

    while True:
        current_time = time.time()
        if current_time - last_frame_time < 1 / fps_limit:
            time.sleep(0.01)
            continue

        frame = fetch_frame()
        if frame is None:
            print("Failed to fetch frame from ESP32-CAM.")
            time.sleep(0.5)
            continue

        # Detect garbage
        garbage_results = garbage_model.predict(source=frame, conf=0.5, show=False)
        for result in garbage_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                class_id = box.cls[0].item()

                if conf < 0.5:
                    continue

                object_name = garbage_model.names[int(class_id)]
                class_name_with_confidence = f"{object_name} ({int(conf * 100)}%)"
                cropped_garbage = frame[y1:y2, x1:x2]

                # Detect cover or uncover status
                cover_results = cover_noncover_model.predict(source=cropped_garbage, conf=0.5, show=False)
                cover_class = None
                if cover_results and cover_results[0].boxes:
                    cover_class = cover_noncover_model.names[int(cover_results[0].boxes[0].cls[0])]

                # Detect polythene or non-polythene
                poly_results = polythene_nonpoly_model.predict(source=cropped_garbage, conf=0.7, show=False)
                poly_class = None
                if poly_results and poly_results[0].boxes:
                    poly_class = polythene_nonpoly_model.names[int(poly_results[0].boxes[0].cls[0])]

                # Detect dry or wet
                dry_wet_results = dry_wet_model.predict(source=cropped_garbage, conf=0.5, show=False)
                dry_wet_class = None
                if dry_wet_results and dry_wet_results[0].boxes:
                    dry_wet_class = dry_wet_model.names[int(dry_wet_results[0].boxes[0].cls[0])]

                # Estimate distance using KNN
                bbox_width, bbox_height = x2 - x1, y2 - y1
                estimated_distance = knn.predict([[bbox_width, bbox_height]])[0]

                # Save detection data to MongoDB
                detection_data = {
                    "object": object_name,
                    "cover_status": cover_class if cover_class else "unknown",
                    "poly_status": poly_class if poly_class else "unknown",
                    "dry_wet_status": dry_wet_class if dry_wet_class else "unknown",
                    "distance": estimated_distance,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                insert_detection_data(detection_data)

                # Draw bounding box and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, class_name_with_confidence, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if cover_class:
                    cv2.putText(frame, f"Cover: {cover_class}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                if poly_class:
                    cv2.putText(frame, f"Poly: {poly_class}", (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if dry_wet_class:
                    cv2.putText(frame, f"{dry_wet_class}", (x1, y2 + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Distance: {estimated_distance:.2f}m", (x1, y2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw boundary curve
        frame = draw_curved_boundary(frame)

        # Encode frame as JPEG and yield for video stream
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# Video stream route
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
