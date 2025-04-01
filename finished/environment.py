
import os
import gymnasium as gym
import numpy as np
import cv2
import kinematika
import communicator_v2
import serial
import time




class ManipulatorEnv(gym.Env):
    def __init__(self):
        super(ManipulatorEnv, self).__init__()

        # observation space: x, y kordinati žogice
        self.observation_space = gym.spaces.Box(low=-500, high=500, shape=(2,), dtype=np.float32) # TODO lahko dodam vx, vy in time_since_last_observation

        # action space: nakloni 3 rok manipulatorja
        self.action_space = gym.spaces.Box(low=-2, high=8, shape=(3,), dtype=np.float32)

        self.ball_pos = np.array([0.0, 0.0])

    def step(self, action):
        self.ball_pos # iz kamere

        # TODO premik kotov glede na action in communicator_v2

        reward = -np.linalg.norm(self.ball_pos) # TODO reward je lahko tudi odvisen od hitrosti žogice

        # če je žogica izven mize, igra je končana
        terminated = np.linalg.norm(self.ball_pos) > 500

        return self.ball_pos, reward, terminated, {}
    
    def reset(self):
        return
    
