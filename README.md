# Parking Management System

## Overview
The **Parking Management System** is a computer vision project designed to monitor and manage parking spaces in real-time. Using YOLOv8 for object detection and OpenCV for image processing, the system can detect parking occupancy, update parking status dynamically, and display information on available and occupied spots.

## Features
- **Real-Time Parking Spot Detection**: Utilizes YOLOv8 for detecting vehicles in predefined parking spots.
- **Interactive User Interface**: Displays the video stream with parking spots highlighted as occupied or available.
- **Data Storage**: Saves parking status on user command for later analysis.
- **Scalability**: Easily extendable to multiple video feeds, suitable for large parking facilities.

## Installation
To set up the Parking Management System, follow these steps:

1. Clone the repository:
   git clone <repository-url>
   cd Parking-Management-System

##  Install the required packages:
pip install -r requirements.txt

If you're using the SAHI algorithm for faraway object detection, ensure your OpenCV version is compatible:

Use opencv-python==4.9.0.80 if needed to avoid conflicts.

## 1. Run the detection script:
python main.py
## 2. View the parking status:

The display will show the heading "Parking Management System" at the top, with real-time updates on each parking spot.
## 3. Save Parking Status:

Press the s button while the system is running to save the current parking occupancy status.
## Configuration
Video Feed URLs: The system retrieves video feeds from URLs specified in a JSON file (paths_and_idx.json).
Parking Spot Positions: Pre-defined parking spot coordinates are loaded from pickle files.
## Project Structure
main.py: Core script to start the detection system.
utils.py: Utility functions, including face detection (if required).
paths_and_idx.json: JSON file specifying the video feed URLs and camera indices.
