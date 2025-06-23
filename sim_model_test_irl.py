import gymnasium as gym
import numpy as np
import cv2
import csv

from finished import kinematika
from finished import communicator_v2
from finished import kamera
from finished import environment
import serial
import time
import os
from datetime import datetime
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

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

# za oranzno ping pong
lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])

# za zeleno gumijasto
#lower_color = np.array([36, 50, 70])  
#upper_color = np.array([89, 255, 255])

alpha = 0.7




# set name of experiment
name = 'v3.0_TD3'
measurement_number = '1'





training_data_path = 'pybullet/training_data'
folder_path = os.path.join(training_data_path, name)
experiments_path = os.path.join(folder_path, 'experiments')

if not os.path.exists(experiments_path):
    os.makedirs(experiments_path)
vec_file = os.path.join(folder_path, f'{name}_vec.pkl')
model_file = os.path.join(folder_path, f'{name}_model.zip')
csv_file = os.path.join(experiments_path, f'{name}_irl_experiment_{measurement_number}.csv')


serial_port = 'COM5'
ser = serial.Serial(serial_port, 115200, timeout=1)
PID = kinematika.PID_controller(np.array([0,0]), np.array([0,0]), np.deg2rad(16), np.deg2rad(45), K_p, K_i, K_d)
kamera_object = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
communication_object = communicator_v2.SerialCommunication(ser=ser, normal_acceleration=acceleration)
kamera_object.adjust_raw_exposure(-8.55) # value according to exposure_calibration_SCRIPT.py

env = environment.ManipulatorEnv(kamera_obj=kamera_object, comm_obj=communication_object, PID_obj=PID, feedrate=feedrate, delay=delay, show_feed=True, verbose=True)
env = DummyVecEnv([lambda: env])

# Load the VecNormalize object
env = VecNormalize.load(vec_file, env)

env.training = False # Ne normalnizira več na nove podatke
env.norm_reward = True
env.norm_obs = True

# Load the trained model
#model = PPO("MlpPolicy", env, verbose=1)
model = TD3.load(model_file, env=env)

obs = env.reset()
done = False
i = 0
reward_sum = 0 

info = []
start_time = datetime.now()

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated = env.step(action)
    
    time = (datetime.now() - start_time).total_seconds()

    if np.all(obs != None):
        # unnormalize the observation
        mean = env.obs_rms.mean
        var = env.obs_rms.var
        unnormalized_obs = obs * np.sqrt(var) + mean

        radius = np.sqrt(unnormalized_obs[0][0]**2 + unnormalized_obs[0][1]**2)

        data = np.hstack([np.array(time), np.array(unnormalized_obs[0]), np.array(radius)])
        info.append(data)

    #ret, frame = kamera_object.cap.read()
    #processed_frame, current_position, current_velocity = kamera_object.process_frame(frame, lower_color, upper_color, alpha)
    #cv2.line(processed_frame, (334, 287), (334+int(action[0][0]*700), 287+int(action[0][1]*700)), (0, 255, 0), 1)  # Draw the action line
    #cv2.imshow(kamera_object.window_name, processed_frame)
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #            break
    done = terminated
    reward_sum += reward
    i += 1
 
print(f'Finished first episode after {i} steps with total reward: {reward_sum}')

# shrani observation v CSV
columns = ['time', 'x', 'y', 'vx', 'vy', 'theta_x', 'theta_y', 'action_x', 'action_y', 'radius']
df = pd.DataFrame(info, columns=columns)
df.to_csv(csv_file, index=False, float_format="%.6f")

'''
obs = env.reset()
done = False
i = 0
reward_sum = 0  

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated = env.step(action)
    done = terminated
    reward_sum += reward
    i += 1
print(f'Finished second episode after {i} steps with total reward: {reward_sum}')
'''

env.close()