import cv2
import numpy as np
import sys

alpha = 0.7

smooth_x, smooth_y, smooth_w, smooth_h = None, None, None, None

def smooth_value(new_value, smooth_value, alpha):
    """Function to smooth the bounding box values using exponential moving average."""
    if smooth_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * smooth_value

def find_black_line(image):
    """
    Find a black line on a white background, determine its orientation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    blurred = cv2.GaussianBlur(binary_thresh, (5, 5), 0)

    edges = cv2.Canny(blurred, 80, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=1)

    pixel_length_min = 40  
    pixel_length_max = 80  

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if pixel_length_min <= line_length <= pixel_length_max:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(image, f"Angle: {angle:.2f} deg", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Detected line: length={line_length:.2f}px, angle={angle:.2f} degrees")
                break
                
    else:
        print("No suitable line detected")

    return image
def open_webcam(camera_index=0, open_window=False):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    cx_i, cx_i1, cx_i2 = 0.0, 0.0, 0.0
    cy_i, cy_i1, cy_i2 = 0.0, 0.0, 0.0

    global smooth_x, smooth_y, smooth_w, smooth_h
    ret, frame = cap.read()

    x, y, w, h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

    roi = frame[y:y+h, x:x+w]

    
    frame_with_line = find_black_line(roi)
    cv2.imshow('line detection', frame_with_line)


    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 50, 30])  
        upper_green = np.array([80, 255, 255]) 

        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = -1, -1

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)

            smooth_x = smooth_value(x, smooth_x, alpha)
            smooth_y = smooth_value(y, smooth_y, alpha)
            smooth_w = smooth_value(w, smooth_w, alpha)
            smooth_h = smooth_value(h, smooth_h, alpha)

            cx, cy = int(smooth_x + smooth_w // 2), int(smooth_y + smooth_h // 2)
            cx_i, cx_i1, cx_i2 = cx, cx_i, cx_i1
            cy_i, cy_i1, cy_i2 = cy, cy_i, cy_i1

            cx_arr = np.array([cx_i, cx_i1, cx_i2])
            cy_arr = np.array([cy_i, cy_i1, cy_i2])

            vx = int(np.gradient(cx_arr)[0])
            vy = int(np.gradient(cy_arr)[0])

            print(cx, cy, '   ', vx, vy)

            cv2.rectangle(frame, (int(smooth_x), int(smooth_y)),
                          (int(smooth_x + smooth_w), int(smooth_y + smooth_h)),
                          (0, 255, 0), 1)

            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 1)
            cv2.line(frame, (cx, cy), (cx - 2 * vx, cy - 2 * vy), (255, 0, 0), 2)

            cv2.putText(frame, f"Zogica", (int(smooth_x + 20), int(smooth_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if open_window:
            cv2.imshow('USB Camera Feed - Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if sys.argv[0] == '-c':
    var1, var2 = sys.argv[1], sys.argv[2]
    open_webcam(var1, var2)
else:
    open_webcam(1, True)
