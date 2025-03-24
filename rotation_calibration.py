import numpy as np
import cv2
from finished import kamera
import serial
import time

# Parameters
b = 0.071589  # m
p = 0.21215  # m
l_1 = 0.16200  # m
l_2 = 0.2525  # m

mtx = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])



CV = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=mtx, distortion_coefficients=dist)


time.sleep(0.5)

coordinates = np.array([[0, 0], [0, 0]])
callback_output = np.array([0, 0])
middle_point = np.array([0, 0])

# Zoom & Pan Variables
scale = 1.0
pan_x, pan_y = 0, 0
dragging = False
start_x, start_y = 0, 0

WINDOW_SIZE = (640, 480)  # Fixed window size

def mouse_callback(event, x, y, flags, param):
    global callback_output, dragging, start_x, start_y, scale, pan_x, pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert clicked coordinates to original scale
        real_x = int((x - pan_x) / scale)
        real_y = int((y - pan_y) / scale)
        print(f'Clicked on {real_x, real_y}')
        callback_output = np.array([real_x, real_y])

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start dragging for panning
        dragging = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Adjust pan position
        pan_x += x - start_x
        pan_y += y - start_y
        start_x, start_y = x, y

    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        # Zoom in or out
        zoom_factor = 1.2 if flags > 0 else 0.8
        scale *= zoom_factor
        scale = max(0.2, min(5.0, scale))  # Limit zoom level

def transform_point(pt):
    """Transforms a point according to pan and zoom."""
    return (int(pt[0] * scale + pan_x), int(pt[1] * scale + pan_y))

CV.exposure_calibration()

CV.create_window()
cv2.setMouseCallback(CV.window_name, mouse_callback)

a, b, c = False, False, False  # Flags to track points

while True:
    ret, frame = CV.cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Resize frame according to zoom level
    resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    undistorted_frame = cv2.undistort(resized_frame, mtx, dist)
    #max_zoom_frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)


    # Process clicks
    if np.any(callback_output != 0):
        if coordinates[0][0] == 0:
            coordinates[0] = callback_output
            callback_output = np.array([0, 0])
            a = True
        else:
            coordinates[1] = callback_output
            callback_output = np.array([0, 0])
            b = True

        if coordinates[-1][-1] != 0:
            delta_x = coordinates[1][0] - coordinates[0][0]
            delta_y = coordinates[1][1] - coordinates[0][1]
            phi = np.rad2deg(np.arctan(delta_y / delta_x)) + 30
            middle_point = np.array([coordinates[0][0] + delta_x / 2, coordinates[0][1] + delta_y / 2])
            print(f'Angle = {phi}, center coordinates = {middle_point}')
            c = True

    # Draw selected points (adjust for zoom/pan)
    if a:
        cv2.circle(undistorted_frame, transform_point(coordinates[0]), 2, (0, 0, 255), -1)
    if b:
        cv2.circle(undistorted_frame, transform_point(coordinates[1]), 2, (0, 0, 255), -1)
    if c:
        cv2.circle(undistorted_frame, transform_point(middle_point), 2, (0, 0, 255), -1)    

    cv2.imshow("Webcam feed", undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
