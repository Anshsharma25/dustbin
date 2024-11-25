from flask import Flask, Response
import cv2
from ultralytics import YOLO
import pickle
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsRegressor

# Initialize Flask app
app = Flask(__name__)

# Load model paths from pickle file
with open('models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load YOLO models
garbage_model = YOLO(models['garbage_model_path'])
dry_wet_model = YOLO(models['dry_wet_model_path'])

# KNN for distance estimation
bbox_size = np.array([[50, 50], [100, 100], [150, 150], [200, 200], [250, 250]])  # Example values
distance = np.array([3, 2, 1.5, 1, 0.5])  # Corresponding distances in meters
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(bbox_size, distance)  # Train the KNN model

# Define boundary parameters
boundary_distance = 1.0  # 1 meter

# ESP32-CAM stream URL
ESP32_CAM_URL = "http://192.168.1.9/cam-hi.jpg"  # Replace with your ESP32-CAM URL

# Frame settings
fps_limit = 10  # Limit FPS to reduce processing load
last_frame_time = 0

# Function to fetch frames from ESP32-CAM
def fetch_frame():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Function to draw a curved boundary line
def draw_curved_boundary(frame):
    frame_height, frame_width = frame.shape[:2]
    curve_depth = 30  # Adjust depth of the curve
    curve_center_y = frame_height // 2  # Midpoint height

    num_points = frame_width
    curve_points = []

    for x in range(num_points):
        t = x / (frame_width - 1)  # Parameter from 0 to 1
        y = curve_center_y + int(curve_depth * np.sin(np.pi * t))
        curve_points.append((x, y))

    for i in range(len(curve_points) - 1):
        cv2.line(frame, curve_points[i], curve_points[i + 1], (0, 255, 0), 2)

    return frame

# Function to process and generate video feed
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

        # Perform garbage detection
        garbage_results = garbage_model(frame)
        garbage_count = 0
        closest_outside_distance = float('inf')
        closest_label = ""

        for result in garbage_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                class_id = box.cls[0]

                if conf < 0.5:  # Filter low-confidence detections
                    continue

                garbage_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f'{garbage_model.names[int(class_id)]} {conf:.2f}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cropped_garbage = frame[y1:y2, x1:x2]

                # Perform dry/wet classification
                dry_wet_results = dry_wet_model(cropped_garbage)
                dry_wet_label = 'Common'
                dry_wet_confidence = 0.0

                for dw_result in dry_wet_results:
                    if len(dw_result.boxes) > 0:
                        dw_class_id = dw_result.boxes[0].cls[0]
                        dry_wet_label = dry_wet_model.names[int(dw_class_id)]
                        dry_wet_confidence = dw_result.boxes[0].conf[0]

                cv2.putText(frame, f"Dry/Wet: {dry_wet_label} ({dry_wet_confidence:.2f})", 
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Predict distance
                box_width = x2 - x1
                box_height = y2 - y1
                predicted_distance = knn.predict([[box_width, box_height]])[0]

                # Check boundary condition
                if predicted_distance <= boundary_distance:
                    cv2.putText(frame, f"Inside 1m ({predicted_distance:.2f}m)", 
                                (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Outside 1m ({predicted_distance:.2f}m)", 
                                (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    if predicted_distance < closest_outside_distance:
                        closest_outside_distance = predicted_distance
                        closest_label = dry_wet_label

        # Draw the boundary line
        frame = draw_curved_boundary(frame)

        # Add garbage count
        cv2.putText(frame, f"Garbage Count: {garbage_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    # Dynamically set the port
    port = int(os.environ.get('PORT', 5000))  # Use the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
 