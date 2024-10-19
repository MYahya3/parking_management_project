import cv2

url = "rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream"
url_1 = "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"

cap = cv2.VideoCapture(url_1)


# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()


# Get frame width, height, and FPS from the video stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if not available

# Define the codec and create a VideoWriter object (use 'XVID' codec, or 'mp4v' for .mp4 files)
fourcc = cv2.VideoWriter_fourcc(*'mp4')  # or 'mp4v' for MP4 format
output_file = "output_video.mp4"  # Output file name
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Not Found")
        break  # Exit the loop if the video stream is not available

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.imshow("Frame", frame)  # Display the frame

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()