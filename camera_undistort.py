import cv2
import numpy as np

# Image resolution
width, height = 1920, 1080  # Adjust if needed

# Refined Camera Matrix (Intrinsics)
camera_matrix = np.array([
    [1000, 0, width / 2],   # fx,  0, cx
    [0, 1000, height / 2],  #  0, fy, cy
    [0, 0, 1]               #  0,  0,  1
], dtype=np.float32)

# Improved Distortion Coefficients for wide-angle correction
# k1, k2 (radial), p1, p2 (tangential), k3 (higher-order radial)
dist_coeffs = np.array([-0.35, 0.15, 0, 0, -0.05], dtype=np.float32)



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

if ret:
    corrected_frame = undistort_frame(frame, camera_matrix, dist_coeffs)

    # Display results
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", corrected_frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
