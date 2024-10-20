import cv2
import numpy as np
import json
import time
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov9s.pt",
    confidence_threshold=0.1,
)


def sahi_detection(frame):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=520,
        slice_width=520,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
    )
    return result

def get_detction_info(result):
    boxes = []
    classes = []
    names = []
    for object_prediction in result.object_prediction_list:
        # print(object_prediction.category.name)
        if object_prediction.category.name in ["car", "bus", "truck","motorcycle"]:
            box = object_prediction.bbox.to_xyxy()
            boxes.append(box)
            classes.append(object_prediction.category.id)
            names.append(object_prediction.category.name)
    return boxes, classes, names


def save_parking_status(occupied_count, available_count, filepath='parking_status.json'):
    # Create a new parking status entry
    parking_status_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "occupied_slots": occupied_count,
        "available_slots": available_count,
        "total_slots": occupied_count + available_count
    }

    # Initialize or load existing parking status data
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            try:
                parking_status_data = json.load(json_file)
                if not isinstance(parking_status_data, list):
                    print(
                        f"Warning: JSON file contained a {type(parking_status_data).__name__}. Resetting to an empty list.")
                    parking_status_data = []
            except (json.JSONDecodeError, ValueError):
                print("Error reading JSON file. Starting with an empty list.")
                parking_status_data = []
    else:
        # If the file does not exist, create a new list for parking status data
        parking_status_data = []
        print(f"File not found. Creating new file: {filepath}")

    # Append the new parking status entry
    parking_status_data.append(parking_status_entry)

    # Write the updated parking status data back to the file
    with open(filepath, 'w') as json_file:
        json.dump(parking_status_data, json_file, indent=4)

    # Print the new entry for debugging
    print(json.dumps(parking_status_entry, indent=4))

def label_detection(frame, text, left, top, bottom, right, tbox_color=(30, 155, 50), fontFace=1, fontScale=0.8,
                    fontThickness=1):
    cv2.rectangle(frame, (int(left), int(top)), (int(bottom), int(right)), tbox_color, 2)
    textSize = cv2.getTextSize(text, fontFace, fontScale, fontThickness)
    text_w, text_h = textSize[0]
    y_adjust = 10
    cv2.rectangle(frame, (int(left), int(top) - text_h - y_adjust), (int(left) + text_w + y_adjust, int(top)),
                  tbox_color, -1)
    cv2.putText(frame, text, (int(left) + 5, int(top) - 5), fontFace, fontScale, (255, 255, 255), fontThickness,
                cv2.LINE_AA)


def drawPolygons(frame, points_list, detection_points_1=None, detection_points_2=None,
                 polygon_color_inside=(30, 205, 50), polygon_color_outside=(30, 50, 180), alpha=0.5):
    overlay = frame.copy()

    occupied_polygons = 0  # Counter for occupied polygons

    for area in points_list:
        area_np = np.array(area, np.int32)  # Convert points to NumPy array
        is_inside = False  # Reset for each polygon

        # Check detection_points_1 if provided
        if detection_points_1:
            inside_1 = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points_1)
        else:
            inside_1 = False  # If no points provided, assume no detection inside

        # Check detection_points_2 if provided
        if detection_points_2:
            inside_2 = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points_2)
        else:
            inside_2 = False  # If no points provided, assume no detection inside

        # If either detection point is inside, mark the polygon as occupied
        if inside_1 or inside_2:
            is_inside = True
            occupied_polygons += 1

        # Set the color based on whether the polygon is occupied
        color = polygon_color_inside if is_inside else polygon_color_outside

        # Fill the polygon on the overlay
        cv2.fillPoly(overlay, [area_np], color)

    # Blend the overlay with the original frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame, occupied_polygons
