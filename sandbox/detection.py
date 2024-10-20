import cv2
import pickle
import os
import json
from utilis import drawPolygons, label_detection, save_parking_status, sahi_detection, get_detction_info

def load_json(json_file):
    """Load paths, indices, and URLs from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['url_list'], data['idx_list']

def run_detection(json_file = "paths_and_idx.json"):
    # Load paths, idx_list, and url_list from the JSON file
    url_list, idx_list  = load_json(json_file)

    for idx in idx_list:
    # Choose the index (e.g., index 0 as mentioned in your question)
    # idx = 0  # You can modify this to handle different indices dynamically if needed

        # Get the corresponding folder path and URL for the selected idx
        folder_path = os.path.join(f"output/{idx}/{idx}")
        video_url = url_list[idx]  # Use the URL corresponding to the selected index

        # Load the positions from the pickle file (positions of parking spots)
        with open(folder_path, 'rb') as f:
            posList = pickle.load(f)

        # Open the video stream using the URL
        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform SAHI detection on the frame
                results = sahi_detection(frame)
                boxes, classes, names = get_detction_info(results)

                # Collect detection points to check if they are inside the parking polygons
                detection_points_1 = [(int((x1 + x2 -10) / 2), int((y1 + y2 -10) / 2)) for x1, y1, x2, y2 in boxes]
                detection_points_2 = [(int((x1 + x2 - 10) / 2), int((y1 + y2 - 10)/2.5)) for x1, y1, x2, y2 in boxes]

                # Draw polygons and count occupied spots
                frame, occupied_count = drawPolygons(frame, posList, detection_points_1=detection_points_1,
                                                     detection_points_2=detection_points_2)

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

                # Save parking status
                save_parking_status(occupied_count, available_count, filepath=f"output/{idx}//{idx}_slots_info.json")

                for box, cls, name in zip(boxes, classes, names):
                    x1, y1, x2, y2 = box
                    center_point = (int(((x1 - 5) + x2 - 10) / 2), int(((y1 - 5) + y2 - 10) / 2))
                    cv2.circle(frame, center_point, 3, (255, 255, 255), thickness=2)
                    tbox_color = (50, 50, 50)
                    label_detection(frame=frame, text=str(name), tbox_color=tbox_color, left=x1, top=y1, bottom=x2 -10, right=y2 - 10)

                # Resize and show the processed frame
                frame_resized = cv2.resize(frame, (920, 640))
                cv2.imshow("Frame", frame_resized)

                cv2.waitKey(0)

                # Clean up
                cap.release()
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"An error occurred: {e}")
            cap.release()
            cv2.destroyAllWindows()


# # Run the detection function
# run_detection()
