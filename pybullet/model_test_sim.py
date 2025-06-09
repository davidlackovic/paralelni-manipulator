import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import simEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

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
p.changeDynamics(ball_id, -1, lateralFriction=1e-5, rollingFriction=1e-8, restitution=0.002)


# set name of experiment
name = 'v1.7'

training_data_path = 'pybullet/training_data'
folder_path = os.path.join(training_data_path, name)
vec_file = os.path.join(folder_path, f'{name}_vec.pkl')
model_file = os.path.join(folder_path, f'{name}_model.zip')


simEnv = simEnvironment.ManipulatorSimEnv(robot_id, ball_id, steps_per_frame=16, verbose=True)
simEnv = DummyVecEnv([lambda: simEnv])

# Load the VecNormalize object
simEnv = VecNormalize.load(vec_file, simEnv)
#simEnv = VecNormalize(simEnv, norm_obs=True, norm_reward=False)

simEnv.training = False # Ne normalnizira več na nove podatke
simEnv.norm_reward = False


#model = PPO("MlpPolicy", simEnv, verbose=1)
model = PPO.load(model_file, simEnv)  # Load the trained model

#model.policy.set_training_mode(False)

model.set_env(simEnv)

# 0 = end after termination, 1 = repeat after termination
test_mode = 1



obs = simEnv.reset()
done = False
i = 0
reward_sum = 0  



while not done:
    action, _ = model.predict(obs, deterministic=True) # True ne dodaja šuma
    obs, reward, terminated, truncated = simEnv.step(action)
    keys = p.getKeyboardEvents()
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        print("Resetting...")
        obs = simEnv.reset()

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



p.disconnect()