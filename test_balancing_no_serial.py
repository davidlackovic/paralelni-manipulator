import numpy as np
from finished import kinematika
from finished import communicator_v2
from finished import kamera
import serial
import cv2
import time

#Dejanski parametri
b = 0.071589 # m
p = 0.21215 # m
l_1 = 0.16200 # m
l_2 = 0.2525 # m

alpha = 0.6
theta = np.deg2rad(30.75)

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])


lower_color = np.array([0, 141, 155])    
upper_color = np.array([38, 241, 255])

start_pos = np.array([0,0])
target_pos = np.array([0,0])

start_config = np.array([0.32, 0, 0])
current_angle = start_config[1:]


max_change_rate = np.deg2rad(45) #deg/s
max_tilt_angle = np.deg2rad(50000) #deg

#PID konstante
K_p = 0.02
K_i = 0.006
K_d = 0.0

time_step = 1/30 # 1/FPS



CV = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
PID = kinematika.PID_controller(start_pos, target_pos, max_tilt_angle, max_change_rate, K_p, K_i, K_d)

#TODO premik motorjev na start_config
CV.exposure_calibration(lower_color=lower_color, upper_color=upper_color)

CV.create_window()
CV.set_mouse_callback()

starting_position=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], 0, 0)
starting_steps=kinematika.deg2steps(starting_position)


while True:
    ret, frame = CV.cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    processed_frame, current_position, current_velocity = CV.process_frame(frame, lower_color, upper_color, alpha)

    if np.all(current_position != None):
        new_angle = PID.calculate(current_position=current_position, target_position=CV.target_pos, time_step=time_step)
        clipped_new_angle = PID.clip_angle(new_angle, current_angle, time_step)
        psi_x, psi_y = clipped_new_angle
        calculated_angles=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], psi_x, psi_y)
        calculated_steps=kinematika.deg2steps(calculated_angles)
        limited_steps = kinematika.limit_steps(calculated_steps)    # samo kot safety measure, drgac nima veze



        
        




        current_angle = clipped_new_angle
        time.sleep(0.01)

        print(f'[{time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}], {current_position}, {CV.target_pos}, {np.rad2deg(clipped_new_angle)}, {np.rad2deg(clipped_new_angle) @ R}, {calculated_angles}, {calculated_steps}, {limited_steps}')

    cv2.imshow("Webcam feed", processed_frame)

    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


starting_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.255, 0, 0)
starting_steps=kinematika.deg2steps(starting_position)
time.sleep(2)

CV.cap.release()
cv2.destroyAllWindows() 