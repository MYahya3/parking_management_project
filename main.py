import cv2
import torch
import numpy as np
import json
import time
from ultralytics import YOLO
from utilis import YOLO_Detection, drawPolygons, label_detection, save_parking_status
import pickle
import os


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load YOLO model and move it to the appropriate device
model = YOLO("yolov8n.pt")
model.to(device)
os.makedirs("output", exist_ok=True)
# Load the positions from the pickle file
with open(r'carParkPos', 'rb') as f:
    posList = pickle.load(f)
# url = "rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream"
url_1 = "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"
# Capture from camera or video
# cap = cv2.VideoCapture("C:\\Users\\sohail\\Videos\\parking.mp4")  # Change to the appropriate source if not using a webcam
cap = cv2.VideoCapture(url_1)
# get vcap property
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file format
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (640, 480))

last_save_time = time.time()  # Get the initial time
save_interval = 10  # Save every 10 seconds                                      #### Main Loop ####
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection
        boxes, classes, names = YOLO_Detection(model, frame)

        # Collect points to determine if any detection is inside polygons
        detection_points = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = ((x1 + x2) / 2)
            center_y = ((y1 + y2) / 2)
            detection_points.append((int(center_x), int(center_y)))

        # Draw polygons with updated color based on detection status and make them transparent
        frame, occupied_count = drawPolygons(frame, posList, detection_points=detection_points)
        # Calculate available polygons
        available_count = len(posList) - occupied_count
        # Display the counts on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (int((width/2) - 200), 5), (int((width/2) - 40), 40), (250, 250, 250), -1)  # Rectangle dimensions and color
        # Put the current time on top of the black rectangle
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width/2) - 190), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
                    (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), (250, 250, 250), -1)  # Rectangle dimensions and color
        # Put the current time on top of the black rectangle
        cv2.putText(frame, f"Free Slots: {available_count}", (int((width/2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
                    (50, 50, 50), 1, cv2.LINE_AA)
        # cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), (250, 250, 250), -1)  # Rectangle dimensions and color
        # # Put the current time on top of the black rectangle
        # cv2.putText(frame, f"Total Slots: {available_count + occupied_count}", (int((width/2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95,
        #             (50, 50, 50), 1, cv2.LINE_AA)

        # Check if X seconds have passed, if so, save the status
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_parking_status(occupied_count, available_count)  # Save parking status to JSON
            last_save_time = current_time  # Reset the timer

        # Check if 's' key is pressed to save the parking status manually
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_parking_status(occupied_count, available_count)  # Save parking status to JSON
            print("Parking status saved manually.")

        # Iterate through the results
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            detected_class = cls
            name = names[int(cls)]

            # Calculate the center point of the bounding box
            center_x = ((x1 + x2) / 2)
            center_y = ((y1 + y2) / 2)
            center_point = (int(center_x), int(center_y))

            # Define the color of the circle (BGR format)
            circle_color = (0, 120, 0)  # Green color in BGR
            cv2.circle(frame, center_point, 1, (255, 255, 255), thickness=2)

            # Determine the color of the bounding box based on detection location
            detection_in_polygon = False
            for pos in posList:
                matching_result = cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False)
                if matching_result >= 0:
                    detection_in_polygon = True
                    break

            if detection_in_polygon:
                label_detection(frame=frame, text=str(name), tbox_color=(50, 50, 50), left=x1, top=y1, bottom=x2, right=y2)
            else:
                label_detection(frame=frame, text=str(name), tbox_color=(100, 25, 50), left=x1, top=y1, bottom=x2, right=y2)
        frame_resized = cv2.resize(frame, (640, 480))        
        cv2.imshow("Frame", frame_resized)
        out.write(frame_resized)
        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

except:
    raise NotImplementedError