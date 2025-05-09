
import os
import gymnasium as gym
import numpy as np
import cv2
from finished import kinematika
from finished import communicator_v2
import serial
import time



class ManipulatorEnv(gym.Env):
    def __init__(self, kamera_obj, comm_obj, PID_obj, feedrate, delay, show_feed=False):

        super(ManipulatorEnv, self).__init__()
        self.CV = kamera_obj
        self.com = comm_obj
        self.PID = PID_obj

        # za kamero
        self.lower_color = np.array([0, 50, 120])    
        self.upper_color = np.array([36, 255, 251])
        self.alpha = 0.7

        # speed parameters
        self.feedrate = feedrate
        self.delay = delay


        # window za prikazovanje
        self.show_feed = show_feed
        if self.show_feed == True:
            self.CV.create_window() 
            self.CV.set_mouse_callback() 

        # rotacijska matrika
        theta = np.deg2rad(30.75)
        self.R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])


        # observation space: x, y kordinati žogice
        self.observation_space = gym.spaces.Box(low=-500, high=500, shape=(2,), dtype=np.float32) # TODO lahko dodam vx, vy in time_since_last_observation

        # action space: nakloni 3 rok manipulatorja
        #self.action_space = gym.spaces.Box(low=-2, high=8, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        self.ball_pos = np.array([0.0, 0.0])

    def step(self, action):
        termination_circle_radius = 140 # pikslov
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        print(f"Step {self._step_counter}")


        # apply action
        self.com.move_to_position(action, feedrate=self.feedrate) # TODO: speed in acceleration?

        # wait
        time.sleep(0.07) 

        # observation
        ret, frame = self.CV.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return None, None, True, {}
        
        processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)

        if np.all(current_position == None):
            print("No ball detected, trying again...")
            processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
        
        if np.all(current_position != None):
            print(f'Current error: {np.linalg.norm(current_position)}')

        
        # reward
        #reward = 50 - np.linalg.norm(current_position) # TODO reward je lahko tudi odvisen od hitrosti žogice
        reward = 1 / (1 + np.linalg.norm(current_position)) 

        # termination
        terminated = np.linalg.norm(current_position) > termination_circle_radius 
        if terminated:
            print('Termination condition met in step().')
        observation = current_position

        truncated = False # TODO: implement truncation if needed
        time.sleep(0.14) 

        return observation, reward, terminated, truncated, {}
    
    def reset(self, *, seed=None, options=None):
        #Dejanski parametri skrajšano
        b = 0.071589 # m
        p = 0.116 # m
        l_1 = 0.08254 # m
        l_2 = 0.1775 # m


        print('Resetting environment...')
        # izravna ploščo
        flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 0, 0) # hardcoded 0.19
        flat_steps=kinematika.deg2steps(flat_position)
        self.com.enable_steppers()
        time.sleep(0.5)
        self.com.move_to_position(flat_steps, feedrate=1000)

        # poišče žogico
        first_frame = True
        ret, frame = self.CV.cap.read()
        processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
        
        # 3 zaporedne frame z žogico
        i = 0
        while i < 3:
            

            if np.all(current_position != None):
                #cv2.imshow("Webcam feed", processed_frame)

                # notebook alternativa da ne šteka
                
                ret, frame = self.CV.cap.read()
                processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                i += 1
            else:
                print('No ball detected, searching again...')
                time.sleep(0.5)
                i = 0
                ret, frame = self.CV.cap.read()
                processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        #ko jo najde izravna ploščo in začne balansirat        
        print('3 consective frames with ball detected. Balancing started.')
        timer = time.time()
        old_time = time.time()
        while True:
            ret, frame = self.CV.cap.read()    
            if not ret:
                print("Error: Could not read frame")
                break
            
            processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            if first_frame == True:
                if np.all(current_position != None):
                    self.PID.reset_old_pos(current_position)
                first_frame = False
            else:
                if np.all(self.CV.pos_cam_old != None) and np.all(current_position != None):

                    # termination check
                    if np.linalg.norm(current_velocity) > 4 or np.linalg.norm(current_position) > 90:
                        timer = time.time()



                    if time.time() - timer > 2:
                        print('Ball is centered, exiting reset().')
                        observation = current_position
                        return observation, {}
                    
                    # check if error <= 4, TODO lahko dam stran če bo treba
                    if np.linalg.norm(current_position - self.CV.pos_cam_old) <= 4:
                        current_position = self.CV.pos_cam_old
                        
                if np.all(current_position != None):
                    time_step = time.time() - old_time
                    old_time = time.time()
                    new_angle = self.PID.calculate(current_position=current_position, target_position=np.array([0,0]), time_step=time_step)
                    rotated_angle = new_angle @ self.R 
                    psi_x, psi_y = rotated_angle

                    

                    #Dejanski parametri skrajšano
                    b = 0.071589 # m
                    p = 0.116 # m
                    l_1 = 0.08254 # m
                    l_2 = 0.1775 # m
                    calculated_angles=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, np.rad2deg(psi_x), np.rad2deg(psi_y))
                    #calculated_angles=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 0, 0)

                    if np.all(calculated_angles!=0):
                        calculated_steps=kinematika.deg2steps(calculated_angles)
                        limited_steps = kinematika.limit_steps(calculated_steps, -6.2, 23)

                        self.com.move_to_position(limited_steps, feedrate=self.feedrate)
                        #print(f'moved {time.strftime('%Y-%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}')
                        time.sleep(self.delay)
                else:
                    flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 1.5, 1.5) # hardcoded 0.19
                    flat_steps=kinematika.deg2steps(flat_position)
                    self.com.move_to_position(flat_steps, feedrate=1000)
                    time.sleep(0.5)

            if self.show_feed == True:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Webcam feed", processed_frame) # prikaze sliko


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    
    
    def close(self):
        #Dejanski parametri skrajšano
        b = 0.071589 # m
        p = 0.116 # m
        l_1 = 0.08254 # m
        l_2 = 0.1775 # m
        print('Closing environment...')
        flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.15, 0, 0)
        flat_steps=kinematika.deg2steps(flat_position)
        self.com.move_to_position(flat_steps, feedrate=1000)
        time.sleep(7)
        self.com.disable_steppers()
        self.CV.cap.release()
        cv2.destroyAllWindows()

       
     
    
