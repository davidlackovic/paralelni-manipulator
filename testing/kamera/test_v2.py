import cv2
import numpy as np
import sys



def open_webcam(camera_index=0, open_window=False):
    cap = cv2.VideoCapture(camera_index)


    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    cx_i, cx_i1, cx_i2 = 0.0,0.0,0.0
    cy_i, cy_i1, cy_i2 = 0.0,0.0,0.0

    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #lower_blue = np.array([100, 150, 50])
        #upper_blue = np.array([140, 255, 255])
        #mask = cv2.inRange(hsv, lower_blue, upper_blue)

        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        lower_green = np.array([40, 50, 30])   # Adjusted lower HSV bound
        upper_green = np.array([80, 255, 255]) # Adjusted upper HSV bound
            # Create a mask for green objects
        mask = cv2.inRange(hsv, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = -1, -1

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest_contour)


            cx, cy = x + w // 2, y + h // 2 
            cx_i, cx_i1, cx_i2 = cx, cx_i, cx_i1
            cy_i, cy_i1, cy_i2 = cy, cy_i, cy_i1

            cx_arr = np.array([cx_i, cx_i1, cx_i2])
            cy_arr = np.array([cy_i, cy_i1, cy_i2])

            vx = int(np.gradient(cx_arr)[0])
            vy = int(np.gradient(cy_arr)[0])


            print(cx, cy, '   ', vx, vy)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.circle(frame, (cx,cy), 10, (0, 0, 255), 1)
            cv2.line(frame, (cx, cy), (cx-2*vx,cy-2*vy), (255, 0, 0), 2)




            cv2.putText(frame, f"Zogica", (x + 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if(open_window==True):
            cv2.imshow('USB Camera Feed - Red Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if sys.argv[0] == '-c':
    var1, var2 = sys.argv[1], sys.argv[2]
    open_webcam(var1, var2)
else:
    open_webcam(1, True)
