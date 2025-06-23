import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import simEnvironment
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import csv
import pandas as pd
from datetime import datetime

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Camera parameters
width, height = 640, 480
fov = 30
aspect = width / height
near = 0.02
far = 5

# Camera position and orientation (looking straight down)
camera_eye = [0, 0, 1]
camera_target = [0, 0, 0]
camera_up = [1, 0, 0]  # Up vector along the X-axis

# Compute matrices
view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Load URDF
plane_id = p.loadURDF('plane.urdf')
robot_id = p.loadURDF('pybullet/plosca.urdf', [0, 0, 0], useFixedBase=True)

# Get joint indices
joint_X = p.getJointInfo(robot_id, 0)[0]
joint_Y = p.getJointInfo(robot_id, 1)[0]

# Set simulation parameters
p.setRealTimeSimulation(0)
p.changeDynamics(robot_id, -1, linearDamping=0, angularDamping=0)

# Set the camera's default position, target, and FOV (zoom)
p.resetDebugVisualizerCamera(
    cameraDistance=1,    # Distance from the target (zoom level)
    cameraYaw=50,          # Horizontal angle
    cameraPitch=-30,       # Vertical angle
    cameraTargetPosition=[0, 0, 0]  # Target position (focus point)
)


# Get ball position from URDF definition
ball_start_pos = [0, 0, 0.23]
ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Spawn the ball as a separate dynamic object
ball_id = p.loadURDF('pybullet/ball.urdf', ball_start_pos, ball_start_orientation, useFixedBase=False)
p.changeDynamics(ball_id, -1, lateralFriction=0.2, rollingFriction=0.0002, restitution=0.2)


# set name of experiment
name = 'v2.3_TD3'
measurement_number = '0'

training_data_path = 'pybullet/training_data'
folder_path = os.path.join(training_data_path, name)
experiments_path = os.path.join(folder_path, 'experiments')

vec_file = os.path.join(folder_path, f'{name}_vec.pkl')
model_file = os.path.join(folder_path, f'{name}_model.zip')
csv_file = os.path.join(experiments_path, f'{name}_sim_experiment_{measurement_number}.csv')



simEnv = simEnvironment.ManipulatorSimEnv(robot_id, ball_id, steps_per_frame=16, verbose=True, wait_to_finish_moves=False)
simEnv = DummyVecEnv([lambda: simEnv])

# Load the VecNormalize object
simEnv = VecNormalize.load(vec_file, simEnv)
#simEnv = VecNormalize(simEnv, norm_obs=True, norm_reward=False)

simEnv.training = False # Ne normalnizira več na nove podatke
simEnv.norm_reward = True
simEnv.norm_obs = True


#model = PPO("MlpPolicy", simEnv, verbose=1)
model = TD3.load(model_file, simEnv)  # Load the trained model
#model = TD3("MlpPolicy", simEnv, verbose=1, train_freq=2048)

#model.policy.set_training_mode(False)

model.set_env(simEnv)

# 0 = end after termination, 1 = repeat after termination
test_mode = 1



obs = simEnv.reset()
done = False
i = 0
reward_sum = 0  

info = []
start_time = datetime.now()

while not done:
    action, _ = model.predict(obs, deterministic=True) # True ne dodaja šuma
    obs, reward, terminated, truncated = simEnv.step(action)
    print(f'ibservation in scritpt: {obs}')
    time = (datetime.now() - start_time).total_seconds()
    print(time, obs)

    # unnormalize the observation
    mean = simEnv.obs_rms.mean
    var = simEnv.obs_rms.var
    unnormalized_obs = obs * np.sqrt(var) + mean
    radius = np.sqrt(unnormalized_obs[0][0]**2 + unnormalized_obs[0][1]**2)

    data = np.hstack([np.array(time), np.array(unnormalized_obs[0]), np.array(radius)])
    print(f'data: {data}')
    if np.all(obs != None):
        info.append(data)
    keys = p.getKeyboardEvents()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        print("Resetting...")
        obs = simEnv.reset()
    
    if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        break

    #print(f"Step: {i}, Action: {action}, Reward: {reward}")
        
    #time.sleep(0.1)
    #for vecEnv
     # Unwrap the values from the vectorized form
    terminated = terminated[0]
    truncated = truncated[0]
    reward = reward[0]
    if terminated and test_mode==0:
        done = True
    elif terminated and test_mode==1:
        obs = simEnv.reset()
        i=0
    reward_sum += reward
    i += 1


columns = ['time', 'x', 'y', 'vx', 'vy', 'theta_x', 'theta_y', 'action_x', 'action_y', 'radius']
df = pd.DataFrame(info, columns=columns)
df.to_csv(csv_file, index=False, float_format="%.6f")
p.disconnect()