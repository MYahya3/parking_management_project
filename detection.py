import cv2  # OpenCV for video and image processing
import pickle  # For saving/loading data in binary format (used for parking spots data)
import os  # For handling file and directory operations
from utilis import drawPolygons, label_detection, save_parking_status, sahi_detection, get_detction_info, load_json  # Custom utility functions


def run_detection(json_file="paths_and_idx.json"):
    """
    Main function to run parking lot detection. Loads video streams and pre-defined 
    parking spot positions, performs detection, and displays the occupancy status.
    
    Args:
    json_file (str): Path to the JSON file containing video URLs and index data.
    """
    # Load paths, idx_list (camera indices), and url_list (video URLs) from the JSON file
    data = load_json(json_file)
    idx_list = data["parking_monitoring_cameras"]  # List of camera indices to monitor
    url_list = data["url_list"]  # List of video feed URLs

    # Iterate over each camera index from the idx_list
    for idx in idx_list:
        # Construct the path to the pickle file containing the parking spot positions for this camera
        folder_path = os.path.join(f"reference_data/{idx}/{idx}")

        # Get the corresponding video URL for the current index
        video_url = url_list[idx]  # Access the URL corresponding to this camera index

        # Load the pre-defined parking spot positions from the pickle file
        with open(folder_path, 'rb') as f:
            posList = pickle.load(f)

        # Open the video stream using the specified URL
        cap = cv2.VideoCapture(video_url)

        # Check if the video stream was successfully opened
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return  # Exit the function if the video stream fails to open

        try:
            # Loop to process frames from the video stream
            while True:
                ret, frame = cap.read()  # Read a frame from the video stream
                if not ret:  # If no frame is retrieved (end of stream or error), exit the loop
                    break

                # Perform object detection on the current frame using SAHI (Slicing Aided Hyper Inference)
                results = sahi_detection(frame)
                boxes, classes, names = get_detction_info(results)  # Extract detection information (bounding boxes, classes, and names)

                # Generate points based on bounding boxes to check if detections are inside parking spots
                detection_points_1 = [(int((x1 + x2 - 10) / 2), int((y1 + y2 - 10) / 2)) for x1, y1, x2, y2 in boxes]
                detection_points_2 = [(int((x1 + x2 - 10) / 2), int((y1 + y2 - 10) / 2)) for x1, y1, x2, y2 in boxes]

                # Draw the parking spot polygons on the frame and count the occupied spots
                frame, occupied_count = drawPolygons(frame, posList, detection_points_1=detection_points_1,
                                                     detection_points_2=detection_points_2)

                # Calculate the number of available parking spots
                available_count = len(posList) - occupied_count

                # Display the number of occupied and available parking slots on the frame
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video frame
                cv2.rectangle(frame, (int((width / 2) - 200), 5), (int((width / 2) - 40), 40), (250, 250, 250), -1)  # Draw a white box
                cv2.putText(frame, f"Fill Slots: {occupied_count}", (int((width / 2) - 190), 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)  # Add occupied slots count
                cv2.rectangle(frame, (int(width / 2), 5), (int((width / 2) + 175), 40), (250, 250, 250), -1)  # Draw another white box
                cv2.putText(frame, f"Free Slots: {available_count}", (int((width / 2) + 10), 30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.95, (50, 50, 50), 1, cv2.LINE_AA)  # Add available slots count

                # Save the parking status (occupied and available slots) to a JSON file
                save_parking_status(occupied_count, available_count, filepath=f"reference_data/{idx}//{idx}_slots_info.json")

                # Draw detection information on the frame (bounding boxes, labels)
                for box, cls, name in zip(boxes, classes, names):
                    x1, y1, x2, y2 = box
                    center_point = (int(((x1 - 5) + x2 - 10) / 2), int(((y1 - 5) + y2 - 10) / 2))  # Calculate the center of the bounding box
                    cv2.circle(frame, center_point, 3, (255, 255, 255), thickness=2)  # Draw a small circle at the center of the detection
                    tbox_color = (50, 50, 50)  # Set color for the label background
                    # Draw the label for the detection
                    label_detection(frame=frame, text=str(name), tbox_color=tbox_color, left=x1, top=y1, bottom=x2 - 10, right=y2 - 10)

                # Resize the frame for display (optional, based on your window size)
                frame_resized = cv2.resize(frame, (920, 640))  # Resize the frame to 920x640 for better display
                cv2.imshow("Frame", frame_resized)  # Show the processed frame in a window

                # Wait for a key press (or continue if no key is pressed)
                cv2.waitKey(0)  # Modify this if you want real-time display instead of waiting for a key press

                # Clean up: release the video stream and close OpenCV windows
                cap.release()
                cv2.destroyAllWindows()

        except Exception as e:
            # Handle any exceptions that occur during the process
            print(f"An error occurred: {e}")
            cap.release()  # Release the video stream on error
            cv2.destroyAllWindows()  # Close OpenCV windows

# Uncomment the line below to run the detection function when needed
# run_detection()
