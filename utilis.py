import cv2
import numpy as np
import json
import time
import os


# To make detections and get required outputs
def YOLO_Detection(model, frame, conf=0.35):
    # Perform inference on an image
    results = model.predict(frame, conf=conf, classes = [0,2])
    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    return boxes, classes, names


def save_parking_status(occupied_count, available_count, filepath='parking_status.json'):
    parking_status_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "occupied_slots": occupied_count,
        "available_slots": available_count,
        "total_slots": occupied_count + available_count
    }
    
    # If the file exists, load the existing content, otherwise start with an empty list
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            try:
                parking_status_data = json.load(json_file)
                
                # If the loaded data is a dict (or something else), reset to an empty list
                if not isinstance(parking_status_data, list):
                    print(f"Warning: JSON file contained a {type(parking_status_data).__name__}. Resetting to an empty list.")
                    parking_status_data = []
                    
            except (json.JSONDecodeError, ValueError):
                # If the file is corrupted or empty, we reset it with an empty list
                print("Error reading JSON file. Starting with an empty list.")
                parking_status_data = []
    else:
        parking_status_data = []

    # Append the new entry
    parking_status_data.append(parking_status_entry)

    # Save the updated data back to the JSON file
    with open(filepath, 'w') as json_file:
        json.dump(parking_status_data, json_file, indent=4)
    
    # Optionally print to display the JSON format (for debugging)
    print(json.dumps(parking_status_entry, indent=4))

## Draw YOLOv8 detections function
def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    # Draw Bounding Box
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    # Draw and Label Text
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w = textSize[0][0]
    text_h = textSize[0][1]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)

def drawPolygons(frame, points_list, detection_points=None, polygon_color_inside=(30, 205, 50),
                 polygon_color_outside=(30, 50, 180), alpha=0.5):
    # Create a transparent overlay for the polygons
    overlay = frame.copy()

    occupied_polygons = 0
    for area in points_list:
        # Reshape the flat tuple to an array of shape (4, 1, 2)
        area_np = np.array(area, np.int32)
        if detection_points:
            is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points)
        else:
            is_inside = False
        color = polygon_color_inside if is_inside else polygon_color_outside
        if is_inside:
            occupied_polygons += 1

        # Draw filled polygons on the overlay
        cv2.fillPoly(overlay, [area_np], color)

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame, occupied_polygons