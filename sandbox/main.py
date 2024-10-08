import cv2
import torch
import numpy as np
import time
import pickle
from ultralytics import YOLO
from utilis import drawPolygons, label_detection, save_parking_status, SAHI_Detection

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO model and move it to the appropriate device
model = YOLO("yolov8n.pt")
model.to(device)

# Load the positions from the pickle file
with open('carParkPos', 'rb') as f:
    posList = pickle.load(f)

url_1 = "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"
cap = cv2.VideoCapture(url_1)

# Get video properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (640, 480))

last_save_time = time.time()
save_interval = 10  # Save every 10 seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform SAHI detection
        boxes, classes, names = SAHI_Detection(frame)

        # Collect points to determine if any detection is inside polygons
        detection_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]

        # Draw polygons and get occupied count
        frame, occupied_count = drawPolygons(frame, posList, detection_points=detection_points)
        
        # Calculate available polygons
        available_count = len(posList) - occupied_count

        # Display the counts on the frame
        cv2.rectangle(frame, (int((width/2) - 200), 5), (int((width/2) - 40), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width/2) - 190), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame, (int(width/2), 5), (int((width/2) + 175), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Free Slots: {available_count}", (int((width/2) + 10), 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)

        # Save parking status every X seconds
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_parking_status(occupied_count, available_count)
            last_save_time = current_time

        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_parking_status(occupied_count, available_count)
            print("Parking status saved manually.")
        elif key == ord('q'):
            break

        # Draw bounding boxes and labels
        for box, cls, name in zip(boxes, classes, names):
            x1, y1, x2, y2 = box
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(frame, center_point, 1, (255, 255, 255), thickness=2)

            detection_in_polygon = any(cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False) >= 0 for pos in posList)
            tbox_color = (50, 50, 50) if detection_in_polygon else (100, 25, 50)
            label_detection(frame=frame, text=str(name), tbox_color=tbox_color, left=x1, top=y1, bottom=x2, right=y2)

        frame_resized = cv2.resize(frame, (640, 480))        
        cv2.imshow("Frame", frame_resized)
        out.write(frame_resized)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()