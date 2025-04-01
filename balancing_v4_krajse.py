import numpy as np
from finished import kinematika
from finished import communicator_v2
from finished import kamera
import serial
import cv2
import time

#Dejanski parametri skrajšano
b = 0.071589 # m
p = 0.116 # m
l_1 = 0.08254 # m
l_2 = 0.1775 # m

alpha = 0.7
theta = np.deg2rad(30.75)

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

#za oranzno
lower_color = np.array([0, 141, 155])    
upper_color = np.array([38, 241, 255])

lower_color = np.array([0, 169, 146])    
upper_color = np.array([38, 255, 246])

# za circle track
lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])

#za pingpong oranzno podnevi
#lower_color = np.array([0, 111, 151])    
#upper_color = np.array([36, 211, 251])

#za zeleno
#lower_color = np.array([27, 146,  91])
#upper_color = np.array([67, 246, 191])

#lower_color = np.array([33, 72,  115])
#upper_color = np.array([73, 172, 215])

#nogometna zogica crno-bela:
#lower_color = np.array([64, 66,  49])
#upper_color = np.array([104, 166, 149])

#golf rumena:
#lower_color = np.array([11, 189,  85])
#upper_color = np.array([51, 255, 185])

target_pos = np.array([0,0])

start_config = np.array([0.19, 0, 0])
current_angle = start_config[1:]


max_change_rate = np.deg2rad(45) #deg/s
max_tilt_angle = np.deg2rad(16) #deg

#PID konstante in movement parameters za oranzno pingpong 
K_p = 0.00038
K_i = 0.00009
K_d = 0.00011
acceleration = 100
feedrate = 1200
delay = 0.07

# set 2

#acceleration = 140
#feedrate = 1200
#delay = 0.0



#za velik error
#K_p = 0.0001
#K_i = 0.0000
#K_d = 0.000008

#PID konstante za zeleno
#K_p = 0.00045
#K_i = 0.00012
#K_d = 0.00017

#PID konstante golf 
#K_p = 0.0004
#K_i = 0.00014
#K_d = 0.00011





print(K_p, K_i, K_d)
time_step = 1/30 # 1/FPS

serial_port = 'COM5'
ser = serial.Serial(serial_port, 115200, timeout=1)



com = communicator_v2.SerialCommunication(ser=ser, normal_acceleration=acceleration)  # communication object iz communicator_v2
CV = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)

CV.exposure_calibration(lower_color=lower_color, upper_color=upper_color)

CV.create_window()
CV.set_mouse_callback()

starting_position=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], 0, 0)
starting_steps=kinematika.deg2steps(starting_position)
limited_steps = starting_steps
com.enable_steppers()
time.sleep(1)
com.move_to_position(starting_steps, feedrate=1500)
time.sleep(5)
print('Searching for ball')
time.sleep(1)
ret, frame = CV.cap.read()
processed_frame, current_position, current_velocity = CV.process_frame(frame, lower_color, upper_color, alpha)

if np.any(current_position==None):
    print('No contour detected on startup, searching again...')
    time.sleep(1)
    ret, frame = CV.cap.read()
    processed_frame, current_position, current_velocity = CV.process_frame(frame, lower_color, upper_color, alpha)

if np.all(current_position!=None):
    CV.target_pos = current_position
    no_ball_startup = False
    print('Ball detected on startup.')
    
else:    
    print('No contour detected on startup, starting with no ball.')
    resting_position=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], 0, 0)
    resting_steps=kinematika.deg2steps(resting_position)
    com.move_to_position(resting_steps, feedrate=feedrate)
    limited_steps = resting_steps
    
    current_position = np.array([0,0])
    CV.target_pos = np.array([0,0])
    no_ball_startup = True


PID = kinematika.PID_controller(current_position, target_pos, max_tilt_angle, max_change_rate, K_p, K_i, K_d)

print(f'before while loop: curent_posisiton: {current_position}, target_pos: {CV.target_pos}')
ret, frame = CV.cap.read()
processed_frame, current_position, current_velocity = CV.process_frame(frame, lower_color, upper_color, alpha)
if np.all(current_position!=None):
    CV.target_pos = current_position
    PID.reset_old_pos(current_position)
else:
    print('couldn\'t set target position')

time_to_move = 0
old_move_time = time.time_ns()
old_time = time.time()

while True:
    ret, frame = CV.cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    old_position = current_position
    processed_frame, current_position, current_velocity = CV.process_frame(frame, lower_color, upper_color, alpha)
    print(f'old position: {old_position}, current position: {current_position}, target position: {CV.target_pos}, integral: {PID.integral}')  

    if np.all(current_position != None):
        time_step=time.time()-old_time
        old_time=time.time()

        if np.all(old_position != None):
            if np.linalg.norm(current_position - old_position) <= 4: # check if error <= 3
                current_position = old_position
            #TODO: to se lahko vse dela v kamera.py, da ni tukaj raztreseno
            #print(f'napaka je: {np.linalg.norm(current_position - old_position)}')
        
        
        new_angle = PID.calculate(current_position=current_position, target_position=CV.target_pos, time_step=time_step)
        #print(f'integral: {PID.integral}')

        #if np.linalg.norm(PID.current_error) < 150:
            #new_angle += (1-(PID.current_error/150))*0.0027*np.sin(5*2*np.pi*time.time())
            
        
        rotated_angle = new_angle @ R
        #clipped_new_angle = PID.clip_angle(rotated_angle, current_angle, time_step)
        clipped_new_angle = rotated_angle
        psi_x, psi_y = clipped_new_angle
        calculated_angles=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], np.rad2deg(psi_x), np.rad2deg(psi_y))

        if np.all(calculated_angles!=0):
            #print(f'delta t =  {time.time_ns()-old_move_time}')
            old_steps = limited_steps
            calculated_steps=kinematika.deg2steps(calculated_angles)
            limited_steps = kinematika.limit_steps(calculated_steps, -6.2, 23)    # samo kot safety measure
            #print(f'koraki: {limited_steps}, delta korakov: {limited_steps-old_steps}, hitrost: {int(np.max(np.abs(60*(limited_steps-old_steps)/0.045)))}')
        
            #print(f'Clamped feedrate: {clamped_feedrate}')

            if time.time_ns() - old_move_time > 3*time_to_move*1e9:
                com.move_to_position(limited_steps, feedrate=feedrate)
                time.sleep(delay)
                time_to_move = np.max(limited_steps - old_steps)/(feedrate/60)

                #print(f'moved {time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}')         

                current_angle = clipped_new_angle
                old_move_time = time.time_ns()

    else:
        resting_position=kinematika.izracun_kotov(b, p, l_1, l_2, start_config[0], 0, 0)
        resting_steps=kinematika.deg2steps(resting_position)
        PID.integral=0
        print(f'limited steps: {limited_steps}, resting steps: {resting_steps}')
        if np.all(limited_steps != resting_steps):  
            com.move_to_position(resting_steps, feedrate=feedrate)
            time_to_move = np.max(resting_steps - old_steps)/(feedrate/60)
            old_move_time = time.time_ns()
            limited_steps = resting_steps
            print(f'Moving to resting position.')






    cv2.imshow("Webcam feed", processed_frame)

    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


starting_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.15, 0, 0)
starting_steps=kinematika.deg2steps(starting_position)
com.move_to_position(starting_steps, feedrate=1500)
time.sleep(4)
com.disable_steppers()
ser.close()

CV.cap.release()
cv2.destroyAllWindows() 