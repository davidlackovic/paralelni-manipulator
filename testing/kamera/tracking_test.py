import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

x_offset = 0  # x-coordinate of the top-left corner of the crop
y_offset = 0  # y-coordinate of the top-left corner of the crop
crop_width = 640  # width of the crop (480p)
crop_height = 480  # height of the crop (480p)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Max Resolution: {width} x {height}")

mtx = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 210, 251])

lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])

if not cap.isOpened():
    print('ni odpru capa')
    exit()
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x: {x}, y: {y}')

cv2.namedWindow('Webcam Feed')
#cv2.namedWindow('original')

old_x, old_y = 0, 0
while True:
    ret, frame = cap.read()
    undistorted = cv2.undistort(frame, mtx, dist)
    processed_frame = cv2.resize(undistorted, (640, 480), interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
    #blurred = cv2.GaussianBlur(gray, (9,9), 2)
    blurred = gray


    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.8, minDist=30, param1=40, param2=40, minRadius=10, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
           center_color = np.mean(hsv[y-8:y+8, x-8:x+8], axis=(0,1))
           print(f'{lower_color, center_color, upper_color}, koordinate: {x, y}')
           if (lower_color[0] <= center_color[0] <= upper_color[0] and
                lower_color[1] <= center_color[1] <= upper_color[1] and
                lower_color[2] <= center_color[2]<= upper_color[2]):
                if x-old_x>=3 and y-old_y>=3:
                    cv2.circle(processed_frame, (x, y), r, (0, 255, 0), 1)  # Draw detected circle
                    cv2.circle(processed_frame, (x, y), 2, (0, 0, 255), -1) # Draw center
                    old_x=x
                    old_y=y

    else:
        print('no ball was detected!')
    #cropped_frame = frame[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]
    if not ret:
        print('ni zajel framea')
        break
    cv2.imshow('Webcam Feed', processed_frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break