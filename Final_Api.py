from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime
from urllib.parse import quote_plus
from bson import ObjectId

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# MongoDB Setup
username = quote_plus("dustbin")  # Replace with your MongoDB username
password = quote_plus("Dustbin@123")  # Replace with your MongoDB password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.fmudd.mongodb.net/"

DATABASE_NAME = "garbage_detection"
COLLECTION_NAME = "sensor_data"

# Connect to MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    detections_collection = db['detection']  # Change to 'detection' collection
    sensor_collection = db[COLLECTION_NAME]
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# Helper function to convert ObjectId to string
def convert_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

@app.route('/final-result', methods=['GET'])
def final_result():
    try:
        # Fetch detection data and sensor data
        detection_data = list(detections_collection.find())
        sensor_data = list(sensor_collection.find())

        final_results = []

        # Print all values of both collections for debugging or reference
        print("=== Detection Collection ===")
        for det in detection_data:
            print(det)  # Print each detection record

        print("\n=== Sensor Collection ===")
        for sensor in sensor_data:
            print(sensor)  # Print each sensor record

        # Iterate through each detection record
        for det in detection_data:
            det_time = datetime.datetime.strptime(det['timestamp'], "%Y-%m-%d %H:%M:%S")
            det_class = det['classification']  # Detected classification (dry or wet)
            det_accuracy = det['accuracy']

            # Find closest sensor data based on timestamp
            matching_sensor = None
            for sensor in sensor_data:
                sensor_time = datetime.datetime.strptime(sensor['timestamp'], "%Y-%m-%d %H:%M:%S")
                if abs(det_time - sensor_time) <= datetime.timedelta(seconds=5):
                    matching_sensor = sensor
                    break

            if matching_sensor:
                # Extract sensor data
                humidity = matching_sensor['humidity']
                moisture = matching_sensor['moisture']
                dryness_percentage = matching_sensor['drynessPercentage']
                wetness_percentage = matching_sensor['wetnessPercentage']

                # Classification logic based on thresholds
                if humidity < 50 and moisture < 30:
                    sensor_class = "Dry"
                elif humidity >= 50 and moisture >= 30:
                    sensor_class = "Wet"
                else:
                    if dryness_percentage > wetness_percentage:
                        sensor_class = "Dry"
                    elif wetness_percentage > dryness_percentage:
                        sensor_class = "Wet"
                    else:
                        sensor_class = "Common"

                # Determine final classification (from both sensor and detection data)
                if det_class == sensor_class:
                    final_results.append({
                        "timestamp": det['timestamp'],
                        "final_object": det_class,
                        "source": "Both",
                        "confidence": det_accuracy if det_accuracy >= 50 else "Sensor"
                    })
                else:
                    final_results.append({
                        "timestamp": det['timestamp'],
                        "final_object": "Common",
                        "reason": "Classification mismatch between Detection and Sensor"
                    })

        # Convert ObjectId to string before returning data
        detection_data = [convert_objectid(det) for det in detection_data]
        sensor_data = [convert_objectid(sensor) for sensor in sensor_data]

        # Return the final results with full data from both collections
        return jsonify({
            "detection_data": detection_data,
            "sensor_data": sensor_data,
            "final_results": final_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)
