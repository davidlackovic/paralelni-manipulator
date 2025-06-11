import numpy as np
import cv2
from finished import kamera
from finished import communicator_v2
from finished import kinematika
import serial
import time

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

#Dejanski parametri skrajšano
b = 0.071589 # m
p = 0.116 # m
l_1 = 0.08254 # m
l_2 = 0.1775 # m

# PID konstante za počasnejše premikanje
K_p = 0.0001
K_i = 0.000000
K_d = 0.0003
acceleration = 500
feedrate = 10000
delay = 0.06

camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])
alpha = 0.7

serial_port = 'COM5'
ser = serial.Serial(serial_port, 115200, timeout=1)
PID = kinematika.PID_controller(np.array([0,0]), np.array([0,0]), np.deg2rad(16), np.deg2rad(45), K_p, K_i, K_d)
kamera_object = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
communication_object = communicator_v2.SerialCommunication(ser=ser, normal_acceleration=acceleration)

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked at: {clicked_point}")

CV = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=mtx, distortion_coefficients=dist)
CV.create_window()
CV.adjust_raw_exposure(-8.7)
cv2.setMouseCallback(CV.window_name, mouse_callback)

lift_postion = kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 0, 0)
lift_steps = kinematika.deg2steps(lift_postion)
communication_object.enable_steppers()
time.sleep(1)
communication_object.move_to_position(lift_steps)
time.sleep(1)

while True:
    ret, frame = CV.cap.read()
    processed_frame, current_pos, current_vel = CV.process_frame(frame, lower_color, upper_color, alpha)
    cv2.imshow(CV.window_name, processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.15, 0, 0)
flat_steps=kinematika.deg2steps(flat_position)
communication_object.move_to_position(flat_steps, feedrate=1000)
time.sleep(7)
communication_object.disable_steppers()

cv2.destroyAllWindows()