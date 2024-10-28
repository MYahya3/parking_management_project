import cv2  # OpenCV library for image and video processing
import numpy as np  # NumPy for numerical operations, used here for handling polygons
import json  # To load and save data in JSON format
import time  # For timestamp generation
import os  # To check and manipulate file paths
from sahi import AutoDetectionModel  # SAHI's auto-detection model for object detection
from sahi.predict import get_sliced_prediction  # Function for slicing input images for detection

# Load a JSON file containing paths, indices, and URLs for cameras or resources
def load_json(json_file):
    """Load paths, indices, and URLs from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Initialize the SAHI (Slicing Aided Hyper Inference) object detection model using YOLOv8
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # YOLOv8 model type
    model_path="yolov8s.pt",  # Path to the pre-trained YOLOv8 model
    confidence_threshold=0.1,  # Minimum confidence threshold for detection
)

# Perform object detection on the given frame using the SAHI model
def sahi_detection(frame):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=520,  # Height of each slice to split the image for detection
        slice_width=520,  # Width of each slice
        overlap_height_ratio=0.1,  # Overlap ratio between slices (height)
        overlap_width_ratio=0.1,  # Overlap ratio between slices (width)
    )
    return result  # Return the detection result

# Extract bounding boxes, class IDs, and names from the detection result
def get_detction_info(result):
    boxes = []  # List to hold bounding box coordinates
    classes = []  # List to hold detected class IDs
    names = []  # List to hold names of detected objects
    for object_prediction in result.object_prediction_list:
        # Filter detections for vehicles (cars, buses, trucks, motorcycles)
        if object_prediction.category.name in ["car", "bus", "truck", "motorcycle"]:
            box = object_prediction.bbox.to_xyxy()  # Get the bounding box in XYXY format
            boxes.append(box)  # Append the bounding box to the list
            classes.append(object_prediction.category.id)  # Append the class ID
            names.append(object_prediction.category.name)  # Append the object name (class)
    return boxes, classes, names  # Return the lists of boxes, classes, and names

# Save parking status (occupied and available slots) to a JSON file
def save_parking_status(occupied_count, available_count, filepath='parking_status.json'):
    # Create a new entry with the current parking status
    parking_status_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),  # Add a timestamp for the current status
        "occupied_slots": occupied_count,  # Number of occupied slots
        "available_slots": available_count,  # Number of available slots
        "total_slots": occupied_count + available_count  # Total number of slots
    }

    # Check if the file exists, and if so, load existing data
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            try:
                parking_status_data = json.load(json_file)  # Load existing parking status data
                if not isinstance(parking_status_data, list):  # Check if the data is a list
                    print(
                        f"Warning: JSON file contained a {type(parking_status_data).__name__}. Resetting to an empty list.")
                    parking_status_data = []  # If not, reset to an empty list
            except (json.JSONDecodeError, ValueError):  # Handle any JSON decoding errors
                print("Error reading JSON file. Starting with an empty list.")
                parking_status_data = []
    else:
        # If the file does not exist, create a new list for storing data
        parking_status_data = []
        print(f"File not found. Creating new file: {filepath}")

    # Add the new parking status entry to the list
    parking_status_data.append(parking_status_entry)

    # Save the updated parking status data back to the file
    with open(filepath, 'w') as json_file:
        json.dump(parking_status_data, json_file, indent=4)  # Save data with indentation for readability

    # Print the new entry for debugging purposes
    print(json.dumps(parking_status_entry, indent=4))

# Draw bounding boxes and labels for detected objects on the video frame
def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    # Draw the bounding box around the detected object
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    # Calculate text size to display the object name inside or near the box
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w, text_h = textSize[0]
    y_adjust = 10  # Adjust the position of the text relative to the bounding box
    # Draw a filled rectangle behind the text (for readability)
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    # Put the object name text on the frame
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)

# Draw parking spot polygons on the frame and check if any detection points are inside those polygons
def drawPolygons(frame, points_list, detection_points_1=None, detection_points_2=None,
                 polygon_color_inside=(30, 205, 50), polygon_color_outside=(30, 50, 180), alpha=0.5):
    # Create a copy of the frame for drawing the overlay (to blend later)
    overlay = frame.copy()

    occupied_polygons = 0  # Counter to track the number of occupied parking spots

    # Iterate over each parking spot polygon in the points_list
    for area in points_list:
        area_np = np.array(area, np.int32)  # Convert the polygon points to a NumPy array
        is_inside = False  # Flag to check if the polygon is occupied

        # Check if any detection points from detection_points_1 are inside the polygon
        if detection_points_1:
            inside_1 = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points_1)
        else:
            inside_1 = False  # If no detection points are provided, assume no detection inside

        # Check if any detection points from detection_points_2 are inside the polygon
        if detection_points_2:
            inside_2 = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points_2)
        else:
            inside_2 = False  # If no detection points are provided, assume no detection inside

        # If either set of detection points is inside the polygon, mark it as occupied
        if inside_1 or inside_2:
            is_inside = True
            occupied_polygons += 1  # Increment the counter for occupied spots

        # Set the color for the polygon (inside/occupied or outside/available)
        color = polygon_color_inside if is_inside else polygon_color_outside

        # Fill the polygon with the selected color on the overlay
        cv2.fillPoly(overlay, [area_np], color)

    # Blend the overlay with the original frame using the alpha transparency value
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame, occupied_polygons  # Return the frame and the count of occupied polygons
