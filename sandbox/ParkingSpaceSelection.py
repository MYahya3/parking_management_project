import os
import cv2
import pickle
import numpy as np
from rtmp_video_feeds import check_video_path, process_video


# Global variables
polygon_points = []  # List to store points of the polygon
posList = []  # List to store existing ROIs
roi_save_path = ""  # Path to save ROIs


def load_existing_rois(save_dir):
    """Load existing ROIs from the specified directory."""
    try:
        with open(os.path.join(save_dir, 'carParkPos'), 'rb') as f:
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

def process_roi_image(image_path, save_dir):
    """Process the saved image to mark polygons and handle mouse clicks."""
    global roi_save_path, polygon_points, posList  # Declare the variables as global
    roi_save_path = os.path.join(save_dir, 'carParkPos')  # Save ROIs in the same folder

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

def main(select_new_image=True):
    """Main execution function to process video and select ROIs."""
    global posList, polygon_points  # Declare global variables here to access them

    pth_list = ["rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream","rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"]

    idx = 1
    path = pth_list[idx]

    if select_new_image:
        cap = check_video_path(path)
        if cap:
            saved_image, save_dir = process_video(cap, idx)  # Capture and save the frame
            posList = []  # Reset ROIs for new selection
            polygon_points.clear()  # Clear polygon points for new selection
            print(f"Image saved at: {saved_image}")  # Debug statement
            process_roi_image(saved_image, save_dir)  # Process the saved image and allow ROI selection
    else:
        save_dir = f"output/{idx}/"  # Access the existing directory
        saved_image = os.path.join(save_dir, f"{idx}.png")
        print(f"Looking for image at: {saved_image}")  # Debug statement
        posList = load_existing_rois(save_dir)  # Load existing ROIs

        # Check if existing ROIs were loaded
        if not posList:
            print("No existing ROIs found. Starting with an empty list.")
        polygon_points.clear()  # Clear current polygon points to avoid confusion
        process_roi_image(saved_image, save_dir)  # Process the loaded image and ROIs



# Call the main function
main(select_new_image=False)  # Change to True to capture new images and ROIs
