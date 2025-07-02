import gymnasium as gym
import numpy as np
import cv2

from finished import kinematika
from finished import communicator_v2
from finished import kamera
from finished import environment_v2
import serial
import time
import os
import pickle
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

import pandas as pd

#Dejanski parametri skrajšano
b = 0.071589 # m
p = 0.116 # m
l_1 = 0.08254 # m
l_2 = 0.1775 # m

# PID konstante za počasnejše premikanje
K_p = 0.00014
K_i = 0.000000
K_d = 0.0003
acceleration = 500
feedrate = 10000
delay = 0.08

camera_matrix = np.array([[649.84070017, 0.00000000e+00, 326.70849136],
                [0.00000000e+00, 650.79575464, 306.13746377],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-0.4586199,  0.20583847,   0.00120806,  0.00507029,  -0.0083358])

lower_color = np.array([0, 50, 120])    
upper_color = np.array([36, 255, 251])
alpha = 0.5

# name of pre-trained model
name_pre = 'v4.0_TD3'
# set name of post-trained model
name_post = 'v4.1_TD3' 

training_data_path = 'pybullet/training_data'
folder_path_pre = os.path.join(training_data_path, name_pre)
vec_file_pre = os.path.join(folder_path_pre, f'{name_pre}_vec.pkl')
model_file_pre = os.path.join(folder_path_pre, f'{name_pre}_model.zip')

folder_path_post = os.path.join(training_data_path, name_post)
if not os.path.exists(folder_path_post):
    os.makedirs(folder_path_post)
vec_file_post = os.path.join(folder_path_post, f'{name_post}_vec.pkl')
model_file_post = os.path.join(folder_path_post, f'{name_post}_model.zip')
data_file_post = os.path.join(folder_path_post, f'{name_post}_data.pkl')


serial_port = 'COM5'
ser = serial.Serial(serial_port, 115200, timeout=1)
PID = kinematika.PID_controller(np.array([0,0]), np.array([0,0]), np.deg2rad(16), np.deg2rad(45), K_p, K_i, K_d)
kamera_object = kamera.CV2Wrapper(camera_index=1, window_name='Webcam feed', camera_matrix=camera_matrix, distortion_coefficients=distortion_coefficients)
communication_object = communicator_v2.SerialCommunication(ser=ser, normal_acceleration=acceleration)
kamera_object.adjust_raw_exposure(-8.85) # value according to exposure_calibration_SCRIPT.py

env = environment_v2.ManipulatorEnv(kamera_obj=kamera_object, comm_obj=communication_object, PID_obj=PID, feedrate=feedrate, delay=delay, show_feed=True, verbose=True)
env = DummyVecEnv([lambda: env])

# Load/create the VecNormalize object
env = VecNormalize.load(vec_file_pre, env)
#env = VecNormalize(env, norm_obs=True, norm_reward=True)


env.training = True 
env.norm_reward = True
env.norm_obs = True

# Load the trained model
#model = PPO("MlpPolicy", env, verbose=1)
model = TD3.load(model_file_pre, env=env)
'''model = TD3("MlpPolicy", env, 
            buffer_size=50_000,          # Smaller buffer for faster updates (if short episodes)
            learning_starts=500,         # Start training earlier
            train_freq=(100, "step"),    # More frequent updates
            gradient_steps=64,           # More updates per train call
            batch_size=256,              # Larger batches
            policy_kwargs=dict(net_arch=[256, 256]))'''

obs = env.reset()
done = False
i = 0
reward_sum = 0  

reward_logger_callback = environment_v2.RolloutEndCallback(env)


model.learn(total_timesteps=40_000_000, progress_bar=True, callback=reward_logger_callback,)


if reward_logger_callback.save == True:
    model.save(model_file_post)  # save the trained model
    env.save(vec_file_post) # save vectorize data

    columns = ['episode_rewards', 'episode_lengths']
    data = np.stack((reward_logger_callback.episode_rewards, reward_logger_callback.episode_lengths), axis=1)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(data_file_post, index=False, float_format="%.6f")
env.close()




s

plt.figure(figsize=(10, 5))
plt.plot(reward_logger_callback.episode_rewards, label='Episode rewards', color='blue')
plt.title('Training Progress')
plt.xlabel('Rollout Number')
plt.ylabel('Mean Episode Reward')
plt.grid(True)
plt.legend()
plt.show()