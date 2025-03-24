import cv2
import numpy as np

def adjust_exposure(value):
    global cap
    # Map the slider value (0-200) to the range -20.0 to 0.0
    exposure_value = value / 10 - 20
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    print(f"Exposure set to: {exposure_value:.1f}")  # Print current exposure value for feedback

def open_webcam_with_slider(camera_index=1):
    global cap
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return
    
    # Initialize previous positions for velocity calculations
    prev_positions = {'cx': [0, 0, 0], 'cy': [0, 0, 0]}

    # Set default resolution and frame rate for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create a window and a trackbar for exposure adjustment
    cv2.namedWindow("Webcam Feed")
    cv2.createTrackbar("Exposure", "Webcam Feed", 100, 200, adjust_exposure)  # Slider from 0 to 200 (mapped to -20.0 to 0.0)

    # Set initial exposure value (slider midpoint corresponds to -10.0 exposure)
    adjust_exposure(100)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Resize the frame to 1440x1080 (stretched to fit screen)
        resized_frame = cv2.resize(frame, (1440, 1080), interpolation=cv2.INTER_LINEAR)

        # Convert the frame to HSV for color segmentation
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for green color (adjustable for specific lighting)
        lower_green = np.array([32, 71, 84])    # Adjusted lower HSV bound
        upper_green = np.array([32, 71, 84])  # Adjusted upper HSV bound

        # Create a mask for green objects
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = -1, -1  # Default values if no contour is found

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate bounding rectangle and center of the object
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx, cy = x + w // 2, y + h // 2

            # Update previous positions for velocity calculation
            prev_positions['cx'] = [cx] + prev_positions['cx'][:2]
            prev_positions['cy'] = [cy] + prev_positions['cy'][:2]

            # Calculate velocity using the gradient of the last three positions
            vx = int(np.gradient(prev_positions['cx'])[0])
            vy = int(np.gradient(prev_positions['cy'])[0])

            print(f"Position: ({cx}, {cy})   Velocity: ({vx}, {vy})")

            # Draw bounding rectangle and center point
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(resized_frame, (cx, cy), 10, (0, 0, 255), 2)

            # Draw velocity vector
            cv2.line(resized_frame, (cx, cy), (cx - 2 * vx, cy - 2 * vy), (255, 0, 0), 2)

            # Add label above the object
            cv2.putText(resized_frame, "Zogica", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the processed frame
        cv2.imshow("Webcam Feed", resized_frame)

        cv2.imshow("Webcam Feed", mask)
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function
open_webcam_with_slider(1)
