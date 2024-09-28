import cv2

url = "rtmp://stream.dpctechstudios.com/stream/1fba2adb-6e08-448a-b4cd-809f5fb18313.stream"
url_1 = "rtmp://stream.dpctechstudios.com/stream/1264b0aa-de17-4ac5-997e-17388bfc6cbf.stream"
cap = cv2.VideoCapture(url_1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Not Found")
        exit()

    cv2.imshow("Frame", cv2.resize(frame, (640,640)))

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()