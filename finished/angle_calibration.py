import cv2
import numpy as np

#za kalibracijo kota med sliko in plosco

def find_black_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    blurred = cv2.GaussianBlur(binary_thresh, (5, 5), 0)

    edges = cv2.Canny(blurred, 80, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=1)

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

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {1}")
    

ret, frame = cap.read()


x, y, w, h = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

roi = frame[y:y+h, x:x+w]
frame_with_line = find_black_line(roi)
while True:
    cv2.imshow('line detection', frame_with_line)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()