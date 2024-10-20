import os
import cv2

def check_video_path(path):
    """Check if the video path is valid and return a VideoCapture object."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Cannot open video at {path}")
        return None
    return cap

def process_video(cap, idx):
    """Process video, extract and save a specific frame."""
    # Create directory if it doesn't exist
    save_dir = f"output/{idx}"
    os.makedirs(save_dir, exist_ok=True)

    saved_image_path = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not found or end of video reached.")
            break
        # time.sleep(2)

        saved_image_path = f"{save_dir}/{idx}.png"
        cv2.imwrite(saved_image_path, frame)


        # Display the frame in a resizable window
        # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        # cv2.imshow("Frame", frame)

        cv2.waitKey(1)
        break  # Break immediately (for demo)

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return saved_image_path, save_dir  # Return the saved image path and directory


# Example usage
# pth_list = ["rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream","rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"]
#
# idx = 1
# path = pth_list[idx]
#
# cap = check_video_path(path)
# if cap:
#     saved_image, save_dir = process_video(cap, idx)  # Capture and save the frame