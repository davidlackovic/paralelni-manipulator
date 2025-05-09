import os
import gymnasium as gym
import numpy as np
import time
import pybullet as p

class ManipulatorSimEnv(gym.Env):
    def __init__(self, robot_id, ball_id):
        super(ManipulatorSimEnv, self).__init__()
        self.ball_id = ball_id
        self.robot_id = robot_id
        self.joint_X = p.getJointInfo(robot_id, 0)[0]
        self.joint_Y = p.getJointInfo(robot_id, 1)[0]

        self.observation_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(2,), dtype=np.float32) # TODO lahko dodam vx, vy in time_since_last_observation
        self.action_space = gym.spaces.Box(low=-0.13, high=0.13, shape=(2,), dtype=np.float32) # naklon v X in Y smeri
        

    def step(self, action):
        # apply action
        p.setJointMotorControl2(self.robot_id, self.joint_X, p.POSITION_CONTROL, targetPosition=action[0], force=100)
        p.setJointMotorControl2(self.robot_id, self.joint_Y, p.POSITION_CONTROL, targetPosition=action[1], force=100)


        termination_circle_radius = 0.15 # m
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        p.stepSimulation()
        #time.sleep(1/240.0)

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.ball_id)
        #reward = 1 / (1 + np.linalg.norm(ball_position[0:2])*1e3 + np.linalg.norm(linear_velocity)*1e3)
        reward = 1 - 100*np.linalg.norm(ball_position[0:2])**2 - 0.1*np.linalg.norm(linear_velocity)**2
        # termination
        terminated = np.linalg.norm(ball_position[0:2]) > termination_circle_radius 
        if terminated:
            print('Termination condition met in step().')
            reward = reward - 20 # give a penalty for termination
        observation = ball_position[0:2]

        truncated = False # TODO: implement truncation if needed

        return observation, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        print('Resetting environment...')
        # Get ball position from URDF definition
        ball_start_pos = [0.05, 0, 0.2]
        ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        p.resetJointState(self.robot_id, self.joint_X, targetValue=0, targetVelocity=0)
        p.resetJointState(self.robot_id, self.joint_Y, targetValue=0, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.ball_id, ball_start_pos, ball_start_orientation)
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        ball_position, _ = p.getBasePositionAndOrientation(self.ball_id)
        observation = ball_position[0:2]

        return observation, {}
