import cv2
import numpy as np
import time
import pickle
import threading
from flask import Flask, jsonify
<<<<<<< HEAD
from utilis import drawPolygons, label_detection, save_parking_status, SAHI_Detection
=======
from utilis import drawPolygons, label_detection, save_parking_status, sahi_detection, get_detction_info
>>>>>>> 99f1cdd3e803d5bc01c6ca53628bc3122352b6a1


# Load the positions from the pickle file (positions of parking spots)
with open('output/0/carParkPos', 'rb') as f:
    posList = pickle.load(f)


# Define save interval for parking status and frame processing interval (20 seconds)
<<<<<<< HEAD
process_interval = 1 * 60  # 20 seconds in seconds
=======
process_interval = 1 * 10  # 20 seconds in seconds
>>>>>>> 99f1cdd3e803d5bc01c6ca53628bc3122352b6a1


# Global variables for counting parking slots
occupied_count = 0
available_count = 0


last_process_time = time.time()  # Last time a frame was processed
last_save_time = time.time()     # Last time the status was saved


# Global flag to trigger frame processing via API
api_triggered = False


# Flask app setup
app = Flask(__name__)

@app.route('/trigger-detection', methods=['POST'])
def process_frame_trigger():
    global api_triggered
    api_triggered = True  # Set flag to true to process frame
    return jsonify({"status": "success", "message": "Frame processing triggered"}), 200

def process_frame(cap):
    global last_process_time, last_save_time, api_triggered, occupied_count, available_count

    current_time = time.time()

    # Process frame every 20 seconds or when API is triggered
    if current_time - last_process_time >= process_interval or api_triggered:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read from video stream.")
            return

        # Perform SAHI detection on the frame
<<<<<<< HEAD
        boxes, classes, names = SAHI_Detection(frame)
=======
        results = sahi_detection(frame)
        boxes, classes, names = get_detction_info(results)
>>>>>>> 99f1cdd3e803d5bc01c6ca53628bc3122352b6a1

        # Collect detection points to check if they are inside the parking polygons
        detection_points = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in boxes]

        # Draw polygons and count occupied spots
        frame, occupied_count = drawPolygons(frame, posList, detection_points=detection_points)

        # Calculate available parking slots
        available_count = len(posList) - occupied_count

        # Display the counts on the frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.rectangle(frame, (int((width / 2) - 200), 5), (int((width / 2) - 40), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width / 2) - 190), 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (int(width / 2), 5), (int((width / 2) + 175), 40), (250, 250, 250), -1)
        cv2.putText(frame, f"Free Slots: {available_count}", (int((width / 2) + 10), 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)

        # Save parking status periodically
        save_parking_status(occupied_count, available_count)
        last_save_time = current_time

        # Draw bounding boxes and labels for detected objects
        for box, cls, name in zip(boxes, classes, names):
            x1, y1, x2, y2 = box
            center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(frame, center_point, 1, (255, 255, 255), thickness=2)

            # Check if the detection is inside any parking polygons
            detection_in_polygon = any(cv2.pointPolygonTest(np.array(pos, dtype=np.int32), center_point, False) >= 0 for pos in posList)
            tbox_color = (50, 50, 50) if detection_in_polygon else (100, 25, 50)
            label_detection(frame=frame, text=str(name), tbox_color=tbox_color, left=x1, top=y1, bottom=x2, right=y2)

        # Resize and show the processed frame
        frame_resized = cv2.resize(frame, (920, 640))
        cv2.imshow("Frame", frame_resized)

        # Reset API trigger flag after processing
        api_triggered = False
        last_process_time = current_time

def start_flask_app():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
<<<<<<< HEAD
    url_1 = "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"
=======
    url_1 = "rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream"
>>>>>>> 99f1cdd3e803d5bc01c6ca53628bc3122352b6a1
    cap = cv2.VideoCapture(url_1)

    # Start the Flask API server in a separate thread
    flask_thread = threading.Thread(target=start_flask_app, daemon=True)
    flask_thread.start()

    try:
        while True:
            # Process frames at intervals or via API trigger
            process_frame(cap)

            # Handle keypress events (manual save or quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                save_parking_status(occupied_count, available_count)
                print("Parking status saved manually.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")
        cap.release()
        cv2.destroyAllWindows()
