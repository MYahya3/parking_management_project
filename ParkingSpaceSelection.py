import os
import cv2
import pickle
import numpy as np
from rtmp_video_feeds import check_video_path, process_video
from utilis import load_json

# Global variables
polygon_points = []  # List to store points of the polygon
posList = []  # List to store existing ROIs
roi_save_path = ""  # Path to save ROIs


def load_existing_rois(save_dir, file_name):
    """Load existing ROIs from the specified directory."""
    print(save_dir)
    print(os.path.join(save_dir, f'{file_name}'))
    try:
        with open(os.path.join(save_dir, f'{file_name}'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []  # Return empty list if the file does not exist


def mouseClick(event, x, y, flags, params):
    global polygon_points, posList, roi_save_path

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))

        if len(polygon_points) == 6:
            posList.append(polygon_points.copy())
            with open(roi_save_path, 'wb') as f:  # Save ROIs in the same directory
                pickle.dump(posList, f)
            polygon_points.clear()  # Clear after saving

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0:
                posList.pop(i)
                with open(roi_save_path, 'wb') as f:  # Save updated ROIs
                    pickle.dump(posList, f)
                break


def process_roi_image(image_path, save_dir, file_name):
    """Process the saved image to mark polygons and handle mouse clicks."""
    global roi_save_path, polygon_points, posList  # Declare the variables as global
    roi_save_path = os.path.join(save_dir, f'{file_name}')  # Save ROIs in the same folder

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    while True:
        img = cv2.imread(image_path)  # Load the saved image
        if img is None:
            print(f"Error: Cannot load image from {image_path}")
            break

        # Draw existing polygons
        for polygon in posList:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)

        # Draw points for the current selection
        for point in polygon_points:
            cv2.circle(img, point, 5, (0, 255, 0), -1)

        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", mouseClick)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def select_slots(select_new_image=True, new_pos=True):
    """Main execution function to process videos and select ROIs."""
    global posList, polygon_points  # Declare global variables here to access them

    # Load paths and indices from the JSON file
    json_file = "parameters.json"  # Path to your JSON file
    json_data = load_json(json_file)
    path_list = json_data["url_list"]
    idx_list = json_data["parking_selection_camera"]


    # Loop through all the indices in the idx_list
    for idx in idx_list:
        path = path_list[idx]  # Use the idx to select the corresponding path

        if select_new_image and new_pos:
            cap = check_video_path(path)
            if cap:
                saved_image, save_dir = process_video(cap, idx)  # Capture and save the frame
                posList = []  # Reset ROIs for new selection
                polygon_points.clear()  # Clear polygon points for new selection
                print(f"Image saved at: {saved_image}")  # Debug statement
                process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the saved image and allow ROI selection

        elif select_new_image and not new_pos:
            cap = check_video_path(path)
            if cap:
                saved_image, save_dir = process_video(cap, idx)  # Capture and save the frame
                posList = load_existing_rois(save_dir, os.path.basename(save_dir))  # Load existing ROIs
                # Check if existing ROIs were loaded
                if not posList:
                    print("No existing ROIs found. Starting with an empty list.")
                polygon_points.clear()  # Clear current polygon points to avoid confusion
                process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the loaded image and ROIs

        else:
            save_dir = f"reference_data/{idx}"  # Access the existing directory
            saved_image = os.path.join(f"{save_dir}/", f"{idx}.png")
            print(f"Looking for image at: {saved_image}")  # Debug statement
            posList = load_existing_rois(save_dir, os.path.basename(save_dir))  # Load existing ROIs

            # Check if existing ROIs were loaded
            if not posList:
                print("No existing ROIs found. Starting with an empty list.")
            polygon_points.clear()  # Clear current polygon points to avoid confusion
            process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the loaded image and ROIs


if __name__ == "__main__":
    # Call the main function
    select_slots(select_new_image=True, new_pos=False)  # Change to True to capture new images and ROIs
