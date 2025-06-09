import numpy as np
import cv2
from finished import kamera


# Description:
# measure pixel coordinates, click the mouse to measure
# Parameters
b = 0.071589 # m
p = 0.116 # m
l_1 = 0.08254 # m
l_2 = 0.1775 # m

mtx = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

CV = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=mtx, distortion_coefficients=dist)
CV.create_window()
CV.adjust_raw_exposure(-8.7)
cv2.setMouseCallback(CV.window_name, mouse_callback)
while True:
    ret, frame = CV.cap.read()
    
    cv2.imshow(CV.window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()