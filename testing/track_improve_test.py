import cv2
import numpy as np

# Capture video
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # or the path to your video file
cap.set(cv2.CAP_PROP_EXPOSURE, -8.1)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of orange color in HSV
    lower_orange = np.array([0, 169, 146])     # Adjust the values if needed
    upper_orange = np.array([38, 255, 246])

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply morphological operations to clean up the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are found
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        (center_x,center_y), radius = cv2.minEnclosingCircle(largest_contour)

        # Get the center of the contour (ball)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Draw the circle and the center on the original frame
            cv2.circle(frame, (int(cX), int(cY)), 2, (0, 0, 0), 2)
            cv2.circle(frame, (int(cX), int(cY)), int(radius), (0, 255, 0), 1)

    # Show the original frame and the mask
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
