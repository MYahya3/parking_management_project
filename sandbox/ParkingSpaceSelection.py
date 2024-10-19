import os
import cv2
import pickle
import numpy as np
from rtmp_video_feeds import check_video_path, process_video


# Global variables
polygon_points = []  # List to store points of the polygon

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
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
while True:
    img = cv2.imread("roi_ref.PNG")
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
