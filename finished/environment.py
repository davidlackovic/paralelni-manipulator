
import os
import gymnasium as gym
import numpy as np
import cv2
from finished import kinematika
from finished import communicator_v2
import serial
import time
from datetime import datetime
from stable_baselines3.common.callbacks import EventCallback
import keyboard


class ManipulatorEnv(gym.Env):
    def __init__(self, kamera_obj, comm_obj, PID_obj, feedrate, delay, show_feed=False, verbose=False):

        super(ManipulatorEnv, self).__init__()
        self.CV = kamera_obj
        self.com = comm_obj
        self.PID = PID_obj

        # za oranžno ping pong
        self.lower_color = np.array([0, 50, 120])    
        self.upper_color = np.array([36, 255, 251])
        self.alpha = 0.7
 
        # za zeleno gumijasto
        #self.lower_color = np.array([36, 50, 70])  
        #self.upper_color = np.array([89, 255, 255])


        # speed parameters
        self.feedrate = feedrate
        self.delay = delay

        # scaling matrix 
        self.scaling_matrix = np.array([[7.6923e-4, 0],
                                        [0, 7.6923e-4]]) # popravljeno


        # window za prikazovanje
        self.show_feed = show_feed
        if self.show_feed == True:
            self.CV.create_window() 
            self.CV.set_mouse_callback() 

        # rotacijska matrika
        theta = np.deg2rad(30.75)
        self.R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        
        # debugging
        self.verbose = verbose

        # for clipping max tilt per frame
        self.max_tilt_per_frame = np.array([0.05, 0.05]) # maximum tilt per frame in radians
        self.previous_action = np.zeros(2) # old tilt values for X and Y axes

      



        self.observation_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(8,), dtype=np.float32) # dodane še vx, vy, nakloni plošče thetaX, thetaY
        
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(2,), dtype=np.float32) # naklon v X in Y smeri
        
        self.ball_pos = np.array([0.0, 0.0])

    def step(self, action):
        # print time
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") + f".{int(now.microsecond / 1000):03d}"
        print(formatted_time)

        termination_circle_radius = 0.15
        self._step_counter = getattr(self, "_step_counter", 0) + 1

        action = np.array([action[1], -action[0]]) # pretvori iz naklon_okoli_osi v naklon_v_smeri_osi, ampak še vedno v K.S. kamere
        #action = np.array([-action[0], action[1]]) # obrnjeno ker se nagiba v napačno smer - zakaj? nevem 
        # zdej dela ok v X smeri, v Y smeri sploh ne spreminja actiona
        action = action @ self.R # rotiraj akcijo z rotacijsko matriko - v K.S. robota


        #clip action
        clipped_action = np.clip(action, self.previous_action-self.max_tilt_per_frame, self.previous_action+self.max_tilt_per_frame)
        self.action_delta = clipped_action - self.previous_action
        self.previous_action = clipped_action # update previous action

        # convert action to steps
        # Dejanski parametri skrajšano
        b = 0.071589 # m
        p = 0.116 # m
        l_1 = 0.08254 # m
        l_2 = 0.1775 # m
        angles = kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, np.rad2deg(clipped_action[0]), np.rad2deg(clipped_action[1]))
        steps = kinematika.deg2steps(angles)

        # apply action
        self.com.move_to_position(steps, feedrate=self.feedrate) # TODO: speed in acceleration?

        # wait
        time.sleep(self.delay) 

        # observation
        ret, frame = self.CV.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return None, None, True, {}
        
        processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)

        ball_tracking_attempts = 1
        while np.all(current_position == None):
            print(f"No ball detected, trying again in a while loop, try: {ball_tracking_attempts}")
            ret, frame = self.CV.cap.read()
            processed_frame, current_position, current_velocity = self.CV.process_frame(frame, self.lower_color, self.upper_color, self.alpha)
            ball_tracking_attempts += 1

        
        self.frame = processed_frame

        #if np.all(current_position != None):
            #print(f'Current error: {np.linalg.norm(current_position)}')

        if np.all(current_position != None):
            current_position_scaled = self.scaling_matrix @ current_position # scaled to meters
            current_velocity_scaled = self.scaling_matrix @ current_velocity # scaled to m/s
            x, y = current_position_scaled # x, y position of the ball in meters
            vx, vy = current_velocity_scaled # x, y velocity of the ball in m/s
            thetax, thetay = action # nakloni plošče v radianih

        # reward
        #reward = 10*self._step_counter + self.gauss_reward_function(current_position_scaled, 10, 0.04)*self.gauss_reward_function(current_velocity_scaled, 50, 0.02) za 4.0
        reward = 10*self._step_counter + self.gauss_reward_function(current_position_scaled, 40, 0.02)*self.gauss_reward_function(current_velocity_scaled, 20, 0.006) # za 4.1

        #reward = 10*self._step_counter + self.gauss_reward_function(current_position_scaled, 10, 0.03)*self.gauss_reward_function(current_velocity_scaled, 50, 0.006)


        #if np.any(abs(self.action_delta) > 0.04):
        #    reward *= 0.5

        if np.linalg.norm(current_position_scaled) < 0.04 and np.linalg.norm(current_velocity_scaled) < 0.01:
           reward += 50  # Small bonus for staying centered

        # termination
        terminated = np.linalg.norm(current_position_scaled) > termination_circle_radius 
        if terminated:
            print('Termination condition met in step().')
            reward = reward - 1000 # give a penalty for termination
        
        observation = [x, y, vx, vy, thetax, thetay, self.action_delta[0], self.action_delta[1]]
        

        if self.verbose:
            print(f'Position: {current_position_scaled}, Radius: {np.linalg.norm(current_position_scaled)},  Velocity: {current_velocity_scaled}, action: {action}, clipped action: {clipped_action}, Reward: {reward}')
        

        truncated = False # obsolete
        #time.sleep(0.14) 
    
        
        return observation, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        #Dejanski parametri skrajšano
        b = 0.071589 # m
        p = 0.116 # m
        l_1 = 0.08254 # m
        l_2 = 0.1775 # m

        self.previous_action = np.zeros(2) # reset old tilt values for X and Y axes

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
                cv2.imshow("Webcam feed", processed_frame)
                
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
        limited_steps = np.array([None, None, None])
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
                    # steady state check
                    #if np.linalg.norm(current_velocity) < 0.5 and np.all(limited_steps != None):
                        #limited_steps[0] += 1
                        #limited_steps[1] += -1
                        #limited_steps[2] += 1
                        #self.com.move_to_position(limited_steps, feedrate=self.feedrate)
                        #time.sleep(self.delay)

                    x, y = self.scaling_matrix @ current_position # scaled to meters
                    vx, vy = self.scaling_matrix @ current_velocity # scaled to m/s
                    scaled_position = np.array([x, y])
                    scaled_velocity = np.array([vx, vy])
                    
                    # termination check
                    if np.linalg.norm(scaled_velocity) > 0.2 or np.linalg.norm(scaled_position) > 0.08:
                        timer = time.time()

                



                    if time.time() - timer > 0.4:
                        print('Ball is centered, exiting reset().')
                        observation = [x, y, vx, vy, 0, 0, 0, 0]
                        self._step_counter = 0 # reset step counter
                        return observation, {}
                    
                    # check if error <= 4, TODO lahko dam stran če bo treba
                    if np.linalg.norm(current_position - self.CV.pos_cam_old) <= 10:
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
                    flat_position=kinematika.izracun_kotov(b, p, l_1, l_2, 0.19, 1.5, 1.5) # ce je v vogalu in je ne vidi, nagne plosco nasproti
                    flat_steps=kinematika.deg2steps(flat_position)
                    self.com.move_to_position(flat_steps, feedrate=1000)
                    time.sleep(0.5)

            if self.show_feed == True:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Webcam feed", processed_frame) # prikaze sliko


            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #   break



    
    
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
    
    def gauss_reward_function(self, ball_position, a, sigma):
        '''Gauss reward function type a * exp(-error^2/(2*sigma^2)) \n
        
        NOTE: position is passed to np.linalg.norm() within this function'''
        reward = a * np.exp(-np.linalg.norm(ball_position[0:2])**2 / (2 * sigma**2))
        return reward
    
class RolloutEndCallback(EventCallback):
    '''Class for creating a callback.
    '''
    def __init__(self, simEnv, end_after_n_episodes = None):
        super(RolloutEndCallback, self).__init__()
        self.rollout_rewards = []
        self.learning_rewards = []
        self.simEnv = simEnv
        self.save = False
    


    
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.rollout_rewards.append(reward)

        print('listening for keyboard input...')
        
        if keyboard.is_pressed('q'):
            print("Stopping training without saving.")
            self.save = False
            return False
    
        if keyboard.is_pressed('s'):
            print("Stopping training and saving the model.")
            self.save = True
            return False
    

    
        return True

    def _on_rollout_end(self) -> None:
        if len(self.rollout_rewards) > 0:
            rew_sum = np.sum(self.rollout_rewards)
            print(f"Rollout ended, reward sum: {rew_sum}")
            self.learning_rewards.append(rew_sum)
            self.rollout_rewards = []
            
            self.simEnv.reset()

       
     
    
