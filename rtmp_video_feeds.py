import os  # Module for interacting with the operating system
import cv2  # OpenCV module for handling video and image processing

def check_video_path(path):
    """
    Check if the video path is valid and return a VideoCapture object.
    
    Args:
    path (str): Path or URL to the video file or stream.
    
    Returns:
    cap (cv2.VideoCapture): VideoCapture object if the video path is valid, otherwise None.
    """
    # Create a VideoCapture object to read the video from the given path
    cap = cv2.VideoCapture(path)
    
    # Check if the video file/stream opened successfully
    if not cap.isOpened():
        print(f"Error: Cannot open video at {path}")
        return None
    
    return cap  # Return the VideoCapture object if the video is successfully opened

def process_video(cap, idx):
    """
    Process the video, extract, and save a specific frame.
    
    Args:
    cap (cv2.VideoCapture): OpenCV VideoCapture object for reading the video frames.
    idx (int): Index or identifier for naming the saved frame.
    
    Returns:
    saved_image_path (str): Path to the saved image.
    save_dir (str): Directory where the image was saved.
    """
    # Create a directory to save the frame, if it doesn't already exist
    save_dir = f"reference_data/{idx}"
    os.makedirs(save_dir, exist_ok=True)

    saved_image_path = None  # Variable to store the saved image path

    while True:
        # Read a frame from the video capture object
        ret, frame = cap.read()
        
        # If no frame is read (end of video or error), exit the loop
        if not ret:
            print("Frame not found or end of video reached.")
            break
        
        # Generate the file path for saving the frame image
        saved_image_path = f"{save_dir}/{idx}.png"
        
        # Save the current frame as a PNG image in the specified directory
        cv2.imwrite(saved_image_path, frame)
        
        # Display the frame in a resizable window (currently commented out)
        # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame", frame)

        # Wait briefly for any key press before moving to the next frame
        cv2.waitKey(1)

        # Break immediately after processing the first frame (for demonstration purposes)
        break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Return the path of the saved image and the directory where it was saved
    return saved_image_path, save_dir


# Example usage
# pth_list = ["rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream",
#             "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"]
#
# idx = 1  # Index to select the desired video from the list
# path = pth_list[idx]  # Get the video path using the selected index
#
# cap = check_video_path(path)  # Validate and capture the video
# if cap:
#     saved_image, save_dir = process_video(cap, idx)  # Capture and save the first frame
