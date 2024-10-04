# Parking Space Detection using YOLOv8 and OpenCV
This project detects parked cars and tracks parking space availability using YOLOv8 and OpenCV. The system processes video streams to determine if spaces are occupied or free, with results displayed in real-time.

## Features
Detects cars in real-time from video streams or files.
Counts and displays available and occupied parking spaces.
Interactive tool for selecting parking space areas.
## Installation
Clone the repo:
git clone https://github.com/MYahya3/parking-space-detection.git
cd parking-space-detection
Install dependencies:
pip install -r requirements.txt
Download YOLOv8 model weights (e.g., yolov8n.pt) and place them in the root directory.
## Usage
Select Parking Spaces:
python parkingspaceslection.py
Left-click to draw polygons for parking spaces.
Right-click to remove polygons. Data is saved in carParkPos.
## Run Detection:

python main.py
Video source can be an RTMP stream or local file (adjust in the code).
Detects cars and shows free/occupied parking spaces in real-time.
## Files
main.py: Runs the parking space detection.
parkingspaceslection.py: Interactive tool for selecting parking space areas.
utils.py: Contains utility functions for detection, labeling, and drawing polygons.
## Requirements
Python 3.8+
OpenCV, YOLOv8, NumPy, Torch
