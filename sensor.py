from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from pymongo import MongoClient # type: ignore
import datetime
from urllib.parse import quote_plus

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# MongoDB Setup
username = quote_plus("dustbin")  # Replace with your MongoDB username
password = quote_plus("Dustbin@123")  # Replace with your MongoDB password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.fmudd.mongodb.net/"

DATABASE_NAME = "garbage_detection"
COLLECTION_NAME = "sensor_data"

# Connect to MongoDB and create TTL index for automatic deletion
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DATABASE_NAME]
    detections_collection = db[COLLECTION_NAME]
    # Create TTL index on the 'timestamp' field to automatically delete documents after 2 hours (7200 seconds)
    detections_collection.create_index("timestamp", expireAfterSeconds=7200)  # 2 hours
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

@app.route('/store-data', methods=['POST'])
def store_data():
    try:
        # Check if the Content-Type is application/json
        if request.content_type != 'application/json':
            return jsonify({"error": "Content-Type must be application/json"}), 415
        
        # Get JSON data from the request
        data = request.json

        # Validate data: Check if all required fields are present
        required_fields = ["moisture", "weight", "temperature", "humidity", "drynessPercentage", "wetnessPercentage", "classification", "distance"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required data fields"}), 400

        # Extract distance from data (to determine sensor status)
        distance = data["distance"]

        # Add timestamp to the data in the format 'YYYY-MM-DD HH:MM:SS'
        data["timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Determine sensor state based on distance
        if distance <= 0.5:
            # Activate sensor if the distance is less than or equal to 0.5 meters
            data["sensor_status"] = "Activated"
            data["sensor_message"] = f"Distance: {distance} meters. Object detected within 0.5 meters. Activating sensors..."
        else:
            # Deactivate sensor if the distance is greater than 0.5 meters
            data["sensor_status"] = "Deactivated"
            data["sensor_message"] = f"Distance: {distance} meters. No object within 0.5 meters. Sensors deactivated."

        # Add distance to the "distances" array in the database
        if "distances" not in data:
            data["distances"] = []
        data["distances"].append(distance)

        # Insert data into MongoDB only if sensor is activated
        if data["sensor_status"] == "Activated":
            result = detections_collection.insert_one(data)
            return jsonify({
                "message": "Data stored successfully", 
                "id": str(result.inserted_id), 
                "sensor_status": data["sensor_status"],
                "sensor_message": data["sensor_message"],
                "distances": data["distances"]
            }), 201
        else:
            return jsonify({
                "message": "No object detected within range. No data stored.",
                "sensor_status": data["sensor_status"],
                "sensor_message": data["sensor_message"]
            }), 200

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='192.168.1.13', port=5003, debug=True)  # Replace '192.168.1.13' with your desired IP
