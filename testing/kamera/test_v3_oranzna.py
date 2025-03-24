import cv2
import numpy as np

alpha = 0.5

smooth_x, smooth_y, smooth_w, smooth_h = None, None, None, None

def smooth_value(new_value, smooth_value, alpha):
    """Function to smooth the bounding box values using exponential moving average."""
    if smooth_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * smooth_value

def adjust_exposure(value):
    global cap
    exposure_value = 30*value / 200 - 15
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
    print(f"Exposure set to: {exposure_value:.1f}")  # Print current exposure value for feedback

def open_webcam_with_slider(camera_index=1):
    global cap
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return


    # Set default resolution and frame rate for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Ensure full sensor usage
    cap.set(cv2.CAP_PROP_ZOOM, 0)  # Ensure no zoom
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Auto-focus may change FOV
    cap.set(cv2.CAP_PROP_ZOOM, 0)  # Set zoom level to zero

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Ensure MJPEG for full resolution
    cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure high frame rate




    # Create a window and a trackbar for exposure adjustment
    cv2.namedWindow("Webcam Feed")
    cv2.createTrackbar("Exposure", "Webcam Feed", 100, 200, adjust_exposure)  # Slider from 0 to 200 (mapped to -20.0 to 0.0)

    # Set initial exposure value (slider midpoint corresponds to -10.0 exposure)
    #adjust_exposure(-8)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7.3)

    cx_i, cx_i1, cx_i2 = 0.0, 0.0, 0.0
    cy_i, cy_i1, cy_i2 = 0.0, 0.0, 0.0

    global smooth_x, smooth_y, smooth_w, smooth_h

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Resize the frame to 1440x1080 (stretched to fit screen)
        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Convert the frame to HSV for color segmentation
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for green color (adjustable for specific lighting)
        lower_green = np.array([0, 141, 155])    # Adjusted lower HSV bound
        upper_green = np.array([38, 241, 255])  # Adjusted upper HSV bound

        # Create a mask for green objects
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = -1, -1  # Default values if no contour is found

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            x, y, w, h = cv2.boundingRect(largest_contour)

            smooth_x = smooth_value(x, smooth_x, alpha)
            smooth_y = smooth_value(y, smooth_y, alpha)
            smooth_w = smooth_value(w, smooth_w, alpha)
            smooth_h = smooth_value(h, smooth_h, alpha)

            # Calculate bounding rectangle and center of the object
           
            cx, cy = int(smooth_x + smooth_w // 2), int(smooth_y + smooth_h // 2)
            cx_i, cx_i1, cx_i2 = cx, cx_i, cx_i1
            cy_i, cy_i1, cy_i2 = cy, cy_i, cy_i1

            cx_arr = np.array([cx_i, cx_i1, cx_i2])
            cy_arr = np.array([cy_i, cy_i1, cy_i2])

            vx = int(np.gradient(cx_arr)[0])
            vy = int(np.gradient(cy_arr)[0])


            print(f"Position: ({cx}, {cy})   Velocity: ({vx}, {vy})")

            # Draw bounding rectangle and center point
            cv2.rectangle(resized_frame, (int(smooth_x), int(smooth_y)), (int(smooth_x + smooth_w), int(smooth_y + smooth_h)), (0, 255, 0), 2)
            cv2.circle(resized_frame, (cx, cy), 10, (0, 0, 255), 2)

            # Draw velocity vector
            cv2.line(resized_frame, (cx, cy), (cx - 2 * vx, cy - 2 * vy), (255, 0, 0), 2)

            # Add label above the object
            cv2.putText(resized_frame, "Zogica", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the processed frame
        cv2.imshow("Webcam Feed", resized_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function
open_webcam_with_slider(1)
