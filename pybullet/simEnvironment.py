import os
import gymnasium as gym
import numpy as np
import time
import pybullet as p
from stable_baselines3.common.callbacks import EventCallback
from collections import deque
from datetime import datetime


class ManipulatorSimEnv(gym.Env):
    '''Class for creating a simulated Parallel Manipulator environment.
        

        robot_id: Pybullet unique id for platform \n
        ball_id: Pybullet unique id for ball \n
        max_RTF: if True, maximum achievable real-time factor will be used (for speed-learning)
        '''
    def __init__(self, robot_id, ball_id, max_RTF = False, steps_per_frame = 1, verbose = False):
        super(ManipulatorSimEnv, self).__init__()
        self.ball_id = ball_id
        self.robot_id = robot_id
        self.joint_X = p.getJointInfo(robot_id, 0)[0]
        self.joint_Y = p.getJointInfo(robot_id, 1)[0]
        
        self.max_RTF = max_RTF
        self.steps_per_frame = steps_per_frame

        self.episode_rewards = [] 
        self.episode_count = 0   

        self.verbose = verbose

        self.max_tilt_per_frame = np.array([0.02, 0.02]) # maximum tilt per frame in radians
        self.previous_action = np.zeros(2) # old tilt values for X and Y axes
        

        #self.observation_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(2,), dtype=np.float32) # TODO lahko dodam time_since_last_observation
        self.observation_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(6,), dtype=np.float32) # dodane vx, vy, nakloni plošče thetaX, thetaY
        
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32) # naklon v X in Y smeri
        

    def step(self, action):
        # print time

        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S") + f".{int(now.microsecond / 1000):03d}"
        print(formatted_time)
        # clamp action to maximum tilt per frame
        clipped_action = np.clip(action, self.previous_action-self.max_tilt_per_frame, self.previous_action+self.max_tilt_per_frame)
        self.previous_action = clipped_action # update previous action

        # apply action
        p.setJointMotorControl2(self.robot_id, self.joint_X, p.POSITION_CONTROL, targetPosition=clipped_action[0], force=1e30, positionGain=0.8, velocityGain=1.0)
        p.setJointMotorControl2(self.robot_id, self.joint_Y, p.POSITION_CONTROL, targetPosition=clipped_action[1], force=1e30, positionGain=0.8, velocityGain=1.0)


        termination_circle_radius = 0.17 # m
        self._step_counter = getattr(self, "_step_counter", 0) + 1

        for _ in range(self.steps_per_frame):
            p.stepSimulation()
            if self.max_RTF == False:
                time.sleep(1/240.0)

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.ball_id)
        thetaX = p.getJointState(self.robot_id, self.joint_X)[0]
        thetaY = p.getJointState(self.robot_id, self.joint_Y)[0]
        # Old reward system
        #reward = 1 / (1 + np.linalg.norm(ball_position[0:2])*1e3 + np.linalg.norm(linear_velocity)*1e3)
        #reward = 10 - 100*np.linalg.norm(ball_position[0:2])**2 - 100*np.linalg.norm(linear_velocity)**2 # ful dobraaa!!
        #reward = 10 - 100 * np.linalg.norm(ball_position[0:2]) * np.linalg.norm(linear_velocity)
        #reward = 1 - np.linalg.norm(ball_position[0:2]) - np.linalg.norm(linear_velocity)
        #reward = 1 - (1 / (1 + np.exp(10 * np.linalg.norm(ball_position[0:2])))) - (1 / (1 + np.exp(10 * np.linalg.norm(linear_velocity))))
        
        # Gauss reward system
        #reward = self.gauss_reward_function(ball_position[0:2], 10, 0.09)*self.gauss_reward_function(linear_velocity, 10, 0.09) 
        #reward = self.gauss_reward_function(ball_position[0:2], 1000, 0.007)*self.gauss_reward_function(linear_velocity, 10, 0.01) 
        
        reward = self._step_counter + self.gauss_reward_function(ball_position[0:2], 100, 0.03)*self.gauss_reward_function(linear_velocity, 50, 0.01) 

        #re-worked reward system
        #reward = self._step_counter - 2000*np.linalg.norm(ball_position[0:2])**2 * np.linalg.norm(linear_velocity)**2

        if np.linalg.norm(ball_position[0:2]) < 0.03 and np.linalg.norm(linear_velocity) < 0.01:
           reward += 50  # Small bonus for staying centered

        # real plate angle
        
        plate_orientation = np.array([p.getJointState(self.robot_id, self.joint_X)[0], p.getJointState(self.robot_id, self.joint_Y)[0]])
        if self.verbose:
            print(f'Position: {ball_position}, Velocity: {linear_velocity}, Action: {clipped_action}, Plate angle: {plate_orientation[0:2]} Reward: {reward}')
        # termination
        terminated = np.linalg.norm(ball_position[0:2]) > termination_circle_radius 
        if terminated:
            print(f'Termination condition met in step() after {self._step_counter} steps.')
            reward = reward - 100 # give a penalty for termination
            

        observation = [ball_position[0], ball_position[1], linear_velocity[0], linear_velocity[1], clipped_action[0], clipped_action[1]]

        truncated = False # TODO: implement truncation if needed

    

        return observation, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        print('Resetting environment...')
        # Get ball position from URDF definition
        #ball_start_pos = [0.05, 0, 0.2]
        # randoom start position
        rng = np.random.default_rng(seed)
        
        r = 0.1
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.05, r)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ball_start_pos = [x, y, 0.23]
        ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        max_start_velocity = 0.12
        start_linear_velocity_x = rng.uniform(-max_start_velocity, max_start_velocity)
        start_linear_velocity_y = rng.uniform(-max_start_velocity, max_start_velocity)

        p.resetJointState(self.robot_id, self.joint_X, targetValue=0, targetVelocity=0)
        p.resetJointState(self.robot_id, self.joint_Y, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.ball_id, ball_start_pos, ball_start_orientation)
        p.resetBaseVelocity(self.ball_id, [start_linear_velocity_x, start_linear_velocity_y, 0], [0, 0, 0])

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id) 
        thetaX = p.getJointState(self.robot_id, self.joint_X)[0]
        thetaY = p.getJointState(self.robot_id, self.joint_Y)[0]

        observation = [x, y, start_linear_velocity_x, start_linear_velocity_y, thetaX, thetaY]
        self._step_counter = 0 # reset step counter

        return observation, {}
    
    def gauss_reward_function(self, ball_position, a, sigma):
        '''Gauss reward function type a * exp(-error^2/(2*sigma^2)) \n'''
        reward = a * np.exp(-np.linalg.norm(ball_position[0:2])**2 / (2 * sigma**2))
        return reward
    
    def set_RTF(self, RTF_flag):
        if self.max_RTF != RTF_flag:
            self.max_RTF = RTF_flag
            print(f'Setting max_RTF to {RTF_flag}')


class RolloutEndCallback(EventCallback):
    '''Class for creating a callback.
    '''
    def __init__(self, simEnv, end_after_n_episodes = None):
        super(RolloutEndCallback, self).__init__()
        self.rollout_rewards = []
        self.learning_rewards = []
        self.simEnv = simEnv
    


    
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.rollout_rewards.append(reward)
        
        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("Training terminated by user.")
            return False

    
        return True

    def _on_rollout_end(self) -> None:
        if len(self.rollout_rewards) > 0:
            rew_sum = np.sum(self.rollout_rewards)
            print(f"Rollout ended, reward sum: {rew_sum}")
            self.learning_rewards.append(rew_sum)
            self.rollout_rewards = []
            self.simEnv.reset()


            
        
            

