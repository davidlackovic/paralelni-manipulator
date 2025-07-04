import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import simEnvironment
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import TD3
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
import pickle


# Initialize PyBullet
#p.connect(p.DIRECT)
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


# Get ball position from URDF 
ball_start_pos = [0, 0, 0.23]
ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Spawn the ball as a separate dynamic object
ball_id = p.loadURDF('pybullet/ball.urdf', ball_start_pos, ball_start_orientation, useFixedBase=False)
p.changeDynamics(ball_id, -1, lateralFriction=1e-5, rollingFriction=1e-8, restitution=0.002)

# set name of experiment
name = 'v2.4_TD3'

training_data_path = 'pybullet/training_data'
folder_path = os.path.join(training_data_path, name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

vec_file = os.path.join(folder_path, f'{name}_vec.pkl')
model_file = os.path.join(folder_path, f'{name}_model.zip')
data_file = os.path.join(folder_path, f'{name}_data.pkl')



simEnv = simEnvironment.ManipulatorSimEnv(robot_id, ball_id, max_RTF=True, steps_per_frame=17, verbose=True, wait_to_finish_moves=False)
simEnv = DummyVecEnv([lambda: simEnv]) 
#simEnv = VecNormalize.load("pybullet/training_data/vec_normalize_v8.pkl", simEnv)
simEnv = VecNormalize(simEnv, norm_obs=True, norm_reward=True)



#model = PPO("MlpPolicy", simEnv, n_steps=2048, verbose=0)
model = TD3("MlpPolicy", simEnv, 
            buffer_size=50_000,          # Smaller buffer for faster updates (if short episodes)
            learning_starts=5_000,       # Start training earlier
            train_freq=(256, "step"),    # More frequent updates
            gradient_steps=64,           # More updates per train call
            batch_size=256,              # Larger batches
            policy_kwargs=dict(net_arch=[256, 256]))
#model = PPO.load("pybullet/training_data/v3/v3_model.zip", simEnv)  # Load the trained model
reward_logger_callback = simEnvironment.RolloutEndCallback(simEnv)


model.learn(total_timesteps=40_000_000, callback=reward_logger_callback, progress_bar=True)
model.save(model_file)  # save the trained model
simEnv.save(vec_file) # save vectorize data

p.disconnect()




with open(data_file, 'wb') as f:
    pickle.dump(reward_logger_callback.learning_rewards, f)


plt.figure(figsize=(10, 5))
plt.plot(reward_logger_callback.learning_rewards, label='Mean Reward', color='blue')
plt.title('Training Progress')
plt.xlabel('Rollout Number')
plt.ylabel('Mean Episode Reward')
plt.grid(True)
plt.legend()
plt.show()

'''
simEnv.envs[0].max_RTF=False

obs = simEnv.reset()
done = False
i = 0
reward_sum = 0  


# 0 = end after termination, 1 = repeat after termination
test_mode = 1
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
'''
