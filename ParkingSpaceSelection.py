import os  # Module for interacting with the file system
import cv2  # OpenCV for handling images and video processing
import pickle  # For saving and loading ROI data as binary files
import numpy as np  # For array manipulation and mathematical operations
from rtmp_video_feeds import check_video_path, process_video  # Import custom video handling functions
from utilis import load_json  # Import custom function for loading JSON data

# Global variables
polygon_points = []  # List to store points of the polygon for ROI
posList = []  # List to store existing ROIs (polygons)
roi_save_path = ""  # Path to save ROIs as a file


def load_existing_rois(save_dir, file_name):
    """
    Load existing ROIs from the specified directory if they exist.
    
    Args:
    save_dir (str): Directory where ROI files are saved.
    file_name (str): Name of the ROI file to load.
    
    Returns:
    list: List of polygons (ROIs) if the file exists, otherwise an empty list.
    """
    print(save_dir)
    print(os.path.join(save_dir, f'{file_name}'))  # Display the full path for debugging
    try:
        # Try to load the ROI file
        with open(os.path.join(save_dir, f'{file_name}'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []  # Return an empty list if the file is not found


def mouseClick(event, x, y, flags, params):
    """
    Handle mouse click events for defining and removing ROIs.
    
    Args:
    event: Mouse event (e.g., left click, right click).
    x, y: Coordinates of the mouse event.
    """
    global polygon_points, posList, roi_save_path  # Use global variables to track the state

    # Left mouse button to add points to a polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))  # Add the clicked point to the polygon

        # If 6 points have been added, finalize the polygon and save it
        if len(polygon_points) == 6:
            posList.append(polygon_points.copy())  # Add the polygon to the list of ROIs
            with open(roi_save_path, 'wb') as f:  # Save the updated ROI list to a file
                pickle.dump(posList, f)
            polygon_points.clear()  # Clear the polygon points after saving

    # Right mouse button to remove a polygon
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Check if the clicked point is inside any of the saved polygons
        for i, polygon in enumerate(posList):
            if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0:
                posList.pop(i)  # Remove the polygon if the point is inside
                with open(roi_save_path, 'wb') as f:  # Save the updated ROI list to a file
                    pickle.dump(posList, f)
                break  # Exit the loop after removing the polygon


def process_roi_image(image_path, save_dir, file_name):
    """
    Process the saved image to mark polygons (ROIs) and handle mouse clicks for ROI selection.
    
    Args:
    image_path (str): Path to the image to be processed.
    save_dir (str): Directory where ROI data will be saved.
    file_name (str): Name of the ROI file to be saved.
    """
    global roi_save_path, polygon_points, posList  # Access global variables
    roi_save_path = os.path.join(save_dir, f'{file_name}')  # Set the path for saving ROIs

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a resizable window for displaying the image
    while True:
        img = cv2.imread(image_path)  # Load the image from the specified path
        if img is None:
            print(f"Error: Cannot load image from {image_path}")
            break

        # Draw existing polygons (ROIs) on the image
        for polygon in posList:
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 255), 2)  # Red color for polygons

        # Draw points for the current polygon being selected
        for point in polygon_points:
            cv2.circle(img, point, 5, (0, 255, 0), -1)  # Green color for points

        cv2.imshow("Image", img)  # Show the image with the drawn polygons and points
        cv2.setMouseCallback("Image", mouseClick)  # Set the mouse callback to handle clicks

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows after exiting the loop


def select_slots(select_new_image=True, new_pos=True):
    """
    Main function to capture frames, process them, and allow ROI selection or loading.
    
    Args:
    select_new_image (bool): If True, capture a new image from the video.
    new_pos (bool): If True, start fresh with new ROIs; otherwise, load existing ROIs.
    """
    global posList, polygon_points  # Access global variables

    # Load paths and indices from the JSON file
    json_file = "parameters.json"  # Path to the JSON configuration file
    json_data = load_json(json_file)  # Load JSON data
    path_list = json_data["url_list"]  # List of video feed URLs
    idx_list = json_data["parking_selection_camera"]  # List of indices to process

    # Loop through all the indices in idx_list
    for idx in idx_list:
        path = path_list[idx]  # Get the video path based on the index

        if select_new_image and new_pos:
            # Capture a new frame and process it for ROI selection
            cap = check_video_path(path)
            if cap:
                saved_image, save_dir = process_video(cap, idx)  # Capture and save a frame from the video
                posList = []  # Clear the existing ROIs
                polygon_points.clear()  # Clear the points for a new selection
                print(f"Image saved at: {saved_image}")  # Debug message
                process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the saved image and allow ROI selection

        elif select_new_image and not new_pos:
            # Capture a new frame, but load existing ROIs if available
            cap = check_video_path(path)
            if cap:
                saved_image, save_dir = process_video(cap, idx)  # Capture and save a frame from the video
                posList = load_existing_rois(save_dir, os.path.basename(save_dir))  # Load ROIs if they exist
                if not posList:
                    print("No existing ROIs found. Starting with an empty list.")
                polygon_points.clear()  # Clear points for new selection
                process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the image and allow ROI selection

        else:
            # Load existing image and ROIs without capturing a new frame
            save_dir = f"reference_data/{idx}"  # Use the directory based on the index
            saved_image = os.path.join(f"{save_dir}/", f"{idx}.png")  # Construct the path to the saved image
            print(f"Looking for image at: {saved_image}")  # Debug message
            posList = load_existing_rois(save_dir, os.path.basename(save_dir))  # Load existing ROIs
            if not posList:
                print("No existing ROIs found. Starting with an empty list.")
            polygon_points.clear()  # Clear points for new selection
            process_roi_image(saved_image, save_dir, os.path.basename(save_dir))  # Process the image and ROIs


if __name__ == "__main__":
    # Run the main function, defaulting to load an existing image and ROI
    select_slots(select_new_image=True, new_pos=False)  # Set flags to True for capturing new images/ROIs
