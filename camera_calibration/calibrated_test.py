import cv2
import numpy as np

# Camera calibration parameters (replace these with the ones obtained from calibration)
mtx = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])


# Open a connection to the camera (index 1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Undistort the frame
    undistorted_frame = cv2.undistort(processed_frame, mtx, dist)

    # Display the original and undistorted frames side by side
    cv2.namedWindow('original')
    cv2.namedWindow('undistorted')

    # Show the combined frame
    cv2.imshow("original", processed_frame)
    cv2.imshow("undistorted", undistorted_frame)

    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
