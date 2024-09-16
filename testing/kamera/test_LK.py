import cv2
import numpy as np
import sys

def open_webcam(camera_index=0, open_window=False):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    # Lucas-Kanade parameters for optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame to initialize optical flow
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Detect the green ball in the first frame
    hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    # Updated HSV range for a green ball with hex color #4a803e
    lower_green = np.array([50, 50, 50])   # Adjusted lower HSV bound
    upper_green = np.array([90, 255, 255]) # Adjusted upper HSV bound

    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found, get the bounding box and initial position of the green ball
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx, cy = x + w // 2, y + h // 2  # Center of the green ball

        # Initialize the point to track
        prev_points = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
    else:
        print("Error: No green ball detected in the first frame.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow to track the green ball
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_points, None, **lk_params)

        # Only proceed if the optical flow status is good (i.e., the point was tracked successfully)
        if status[0] == 1:
            # Get the new position of the green ball
            new_center = next_points[0].ravel()

            # Draw a rectangle and line indicating the ball's position and movement
            cv2.circle(frame, (int(new_center[0]), int(new_center[1])), 10, (0, 0, 255), -1)  # Red circle for tracking
            cv2.putText(frame, f"Coordinates: ({int(new_center[0])}, {int(new_center[1])})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update the points and previous frame for the next iteration
            prev_points = next_points
            prev_gray = gray_frame.copy()

        # Display the frame
        if open_window:
            cv2.imshow('USB Camera Feed - Green Object Tracking (Optical Flow)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    open_webcam(1, True)
