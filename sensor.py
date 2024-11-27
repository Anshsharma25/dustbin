from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
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
        required_fields = ["moisture", "weight", "temperature", "humidity", "drynessPercentage", "wetnessPercentage", "classification"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required data fields"}), 400

        # Add timestamp to the data (format as YYYY-MM-DD HH:MM:SS)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format as year-month-day hour:minute:second
        data["timestamp"] = timestamp  # Store the formatted timestamp

        # Insert data into MongoDB
        result = detections_collection.insert_one(data)

        # Respond with success
        return jsonify({"message": "Data stored successfully", "id": str(result.inserted_id)}), 201

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
