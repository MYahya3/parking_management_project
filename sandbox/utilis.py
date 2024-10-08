import cv2
import numpy as np
import json
import time
import os
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="yolov8n.pt",
    confidence_threshold=0.3,
    device=device
)

def SAHI_Detection(frame):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    boxes = []
    classes = []
    names = []
    for object_prediction in result.object_prediction_list:
        box = object_prediction.bbox.to_xyxy()
        boxes.append(box)
        classes.append(object_prediction.category.id)
        names.append(object_prediction.category.name)
    return boxes, classes, names

def save_parking_status(occupied_count, available_count, filepath='parking_status.json'):
    parking_status_entry = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "occupied_slots": occupied_count,
        "available_slots": available_count,
        "total_slots": occupied_count + available_count
    }
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as json_file:
            try:
                parking_status_data = json.load(json_file)
                if not isinstance(parking_status_data, list):
                    print(f"Warning: JSON file contained a {type(parking_status_data).__name__}. Resetting to an empty list.")
                    parking_status_data = []
            except (json.JSONDecodeError, ValueError):
                print("Error reading JSON file. Starting with an empty list.")
                parking_status_data = []
    else:
        parking_status_data = []

    parking_status_data.append(parking_status_entry)

    with open(filepath, 'w') as json_file:
        json.dump(parking_status_data, json_file, indent=4)
    
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

def drawPolygons(frame, points_list, detection_points=None, polygon_color_inside=(30, 205, 50),
                 polygon_color_outside=(30, 50, 180), alpha=0.5):
    overlay = frame.copy()

    occupied_polygons = 0
    for area in points_list:
        area_np = np.array(area, np.int32)
        if detection_points:
            is_inside = any(cv2.pointPolygonTest(area_np, pt, False) >= 0 for pt in detection_points)
        else:
            is_inside = False
        color = polygon_color_inside if is_inside else polygon_color_outside
        if is_inside:
            occupied_polygons += 1

        cv2.fillPoly(overlay, [area_np], color)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame, occupied_polygons