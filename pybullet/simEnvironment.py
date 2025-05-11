import os
import gymnasium as gym
import numpy as np
import time
import pybullet as p
from stable_baselines3.common.callbacks import EventCallback
from collections import deque

class ManipulatorSimEnv(gym.Env):
    '''Class for creating a simulated Parallel Manipulator environment.
        

        robot_id: Pybullet unique id for platform \n
        ball_id: Pybullet unique id for ball \n
        max_RTF: if True, maximum achievable real-time factor will be used (for speed-learning)
        '''
    def __init__(self, robot_id, ball_id, max_RTF = False, steps_per_frame = 1):
        super(ManipulatorSimEnv, self).__init__()
        self.ball_id = ball_id
        self.robot_id = robot_id
        self.joint_X = p.getJointInfo(robot_id, 0)[0]
        self.joint_Y = p.getJointInfo(robot_id, 1)[0]
        
        self.max_RTF = max_RTF
        self.steps_per_frame = steps_per_frame

        self.episode_rewards = [] 
        self.episode_count = 0   
        

        self.observation_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(2,), dtype=np.float32) # TODO lahko dodam vx, vy in time_since_last_observation
        self.action_space = gym.spaces.Box(low=-0.13, high=0.13, shape=(2,), dtype=np.float32) # naklon v X in Y smeri
        

    def step(self, action):
        # apply action
        p.setJointMotorControl2(self.robot_id, self.joint_X, p.POSITION_CONTROL, targetPosition=action[0], force=1)
        p.setJointMotorControl2(self.robot_id, self.joint_Y, p.POSITION_CONTROL, targetPosition=action[1], force=1)


        termination_circle_radius = 0.17 # m
        self._step_counter = getattr(self, "_step_counter", 0) + 1

        for _ in range(self.steps_per_frame):
            p.stepSimulation()
            if self.max_RTF == False:
                time.sleep(1/240.0)

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.ball_id)
        #reward = 1 / (1 + np.linalg.norm(ball_position[0:2])*1e3 + np.linalg.norm(linear_velocity)*1e3)
        # reward = 10 - 100*np.linalg.norm(ball_position[0:2])**2 - 100*np.linalg.norm(linear_velocity)**2 # ful dobraaa!!
        reward = 10 - 5000*np.linalg.norm(ball_position[0:2])**2 - 500*np.linalg.norm(linear_velocity)**2
        #reward = 1 - np.linalg.norm(ball_position[0:2]) - np.linalg.norm(linear_velocity)
        #reward = 1 - (1 / (1 + np.exp(10 * np.linalg.norm(ball_position[0:2])))) - (1 / (1 + np.exp(10 * np.linalg.norm(linear_velocity))))

        if np.linalg.norm(ball_position[0:2]) < 0.04:
            reward += 100  # Small bonus for staying centered


        # termination
        terminated = np.linalg.norm(ball_position[0:2]) > termination_circle_radius 
        if terminated:
            print('Termination condition met in step().')
            reward = reward - 1000 # give a penalty for termination
        observation = ball_position[0:2]

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
        radius = rng.uniform(0, r)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        ball_start_pos = [x, y, 0.2]
        ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        p.resetJointState(self.robot_id, self.joint_X, targetValue=0, targetVelocity=0)
        p.resetJointState(self.robot_id, self.joint_Y, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.ball_id, ball_start_pos, ball_start_orientation)
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id)
        observation = ball_position[0:2]

        return observation, {}


class RolloutEndCallback(EventCallback):
    '''Class for creating a callback.
    
    end_after_n_episodes: if not None, training will be terminated after n episodes with the same reward. \n
    '''
    def __init__(self, end_after_n_episodes = None):
        super(RolloutEndCallback, self).__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.smoothed_rewards = [] 

        self.termination_list = None
        if end_after_n_episodes is not None:
            self.termination_list = deque(maxlen=end_after_n_episodes) # Last 10 episode rewards

        self.termination_flag = False

    
    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.current_episode_reward += reward
        
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        if self.termination_flag:
            print("Training terminated due to 5 consecutive episodes with the same reward.")
            return False
        else:
            return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) > 0:
            window_size = min(100, len(self.episode_rewards))
            ep_rew_mean = sum(self.episode_rewards[-window_size:]) / window_size
            print(f"Rollout ended, ep_rew_mean (smoothed, last {window_size} episodes): {ep_rew_mean}")
            self.smoothed_rewards.append(ep_rew_mean)  # Store the average
            self.termination_list.append(ep_rew_mean)
            if self.termination_list is not None:
                if len(self.termination_list) == self.termination_list.maxlen and len(set(self.termination_list)) == 1 and self.termination==True:
                    self.termination_flag = True
            

