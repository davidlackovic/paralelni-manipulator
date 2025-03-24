import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

x_offset = 0  # x-coordinate of the top-left corner of the crop
y_offset = 0  # y-coordinate of the top-left corner of the crop
crop_width = 640  # width of the crop (480p)
crop_height = 480  # height of the crop (480p)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Max Resolution: {width} x {height}")

mtx = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

if not cap.isOpened():
    print('ni odpru capa')
    exit()
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x: {x}, y: {y}')

cv2.namedWindow('Webcam Feed')
cv2.namedWindow('original')


while True:
    ret, frame = cap.read()
    undistorted = cv2.undistort(frame, mtx, dist)
    processed_frame = cv2.resize(undistorted, (640, 480), interpolation=cv2.INTER_LINEAR)
    processed_frame_original = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    #cropped_frame = frame[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]
    if not ret:
        print('ni zajel framea')
        break
    cv2.imshow('Webcam Feed', processed_frame)
    cv2.imshow('original', processed_frame_original)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break