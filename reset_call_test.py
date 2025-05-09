from finished import kinematika
from finished import communicator_v2
from finished import kamera
from finished import environment
import numpy as np
import serial
import time
import cv2

from IPython.display import display, clear_output
from PIL import Image


#Dejanski parametri skrajšano
b = 0.071589 # m
p = 0.116 # m
l_1 = 0.08254 # m
l_2 = 0.1775 # m

# PID konstante za počasnejše premikanje
K_p = 0.0001
K_i = 0.000000
K_d = 0.0003
acceleration = 100
feedrate = 1200
delay = 0.02

camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])


serial_port = 'COM5'
ser = serial.Serial(serial_port, 115200, timeout=1)
PID = kinematika.PID_controller(np.array([0,0]), np.array([0,0]), np.deg2rad(16), np.deg2rad(45), K_p, K_i, K_d)
kamera_object = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
communication_object = communicator_v2.SerialCommunication(ser=ser, normal_acceleration=acceleration)
kamera_object.adjust_raw_exposure(-8.55) # value according to exposure_calibration_SCRIPT.py

env_object = environment.ManipulatorEnv(kamera_obj=kamera_object, comm_obj=communication_object, PID_obj=PID, feedrate=feedrate, delay=delay, show_feed=True)

# Lift plošče
flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 0, 0) # hardcoded 0.19
flat_steps=kinematika.deg2steps(flat_position)
communication_object.enable_steppers()
time.sleep(0.5)
communication_object.move_to_position(flat_steps, feedrate=1000)

env_object.reset()

flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.15, 0, 0) # hardcoded 0.19
flat_steps=kinematika.deg2steps(flat_position)
communication_object.move_to_position(flat_steps, feedrate=1000)
time.sleep(3)
communication_object.disable_steppers()