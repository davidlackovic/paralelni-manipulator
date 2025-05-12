import cv2
import numpy as np

# Image resolution
width, height = 1920, 1080  # Adjust if needed

# Refined Camera Matrix (Intrinsics)
camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])


def mouse_callback(event, x, y, flags, param):
    global callback_output, dragging, start_x, start_y, scale, pan_x, pan_y

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Clicked on {x, y}')

def undistort_frame(frame, camera_matrix, dist_coeffs):
    """Undistort a given frame using a predefined camera matrix and distortion coefficients."""
    h, w = frame.shape[:2]

    # Get the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Optional: Crop the image to remove black borders
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    return undistorted

# Example Usage
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
ret, frame = cap.read()
cv2.namedWindow("Undistorted")
cv2.setMouseCallback("Undistorted", mouse_callback)

if ret:
    corrected_frame = undistort_frame(frame, camera_matrix, dist_coeffs)

    # Display results
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", corrected_frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
