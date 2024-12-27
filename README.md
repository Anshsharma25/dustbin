# 🤖 Automate Dustbin: Smart Garbage Detection with YOLOv8m & KNN
# 🚀 Project Overview
This project combines YOLOv8m for real-time garbage detection and KNN (K-Nearest Neighbors) for proximity-based identification of the nearest garbage to a dustbin. The system classifies objects into garbage and dry wet, and automatically sorts waste to promote cleanliness and organization. 🌍

# 📌 Key Features
YOLOv8m Detection: Detects and classifies garbage objects, such as garbage and dry wet. 🗑️
KNN Proximity Detection: Identifies and classifies the nearest garbage objects to the dustbin. 📍
Custom Dataset: Trained with images of various waste objects for improved accuracy. ♻️

# 🛠️ Installation & Usage
Clone the repo:
git clone https://github.com/Anshsharma25/dustbin.git
Install dependencies:

pip install -r requirements.txt
Train the model:

python train.py --data data.yaml --weights yolov8m.pt --batch-size 16 --epochs 50
Run object detection:

python detect.py --weights best.pt --source input_image_or_video
# 🔍 Example Output
Garbage Detection: Identifies garbage and dry wet objects with bounding boxes and labels. 📦
Nearest Garbage: Uses KNN to find the closest detected garbage objects to the dustbin. 📍

# 🌱 Contributing
Feel free to fork, create pull requests, or contribute to the project to make it even smarter and cleaner! Together, we can create innovative solutions for waste management. ♻️
