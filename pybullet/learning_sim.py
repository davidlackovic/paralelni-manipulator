import pybullet as p
import pybullet_data
import numpy as np
import time
import simEnvironment
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv


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


# Get ball position from URDF definition
ball_start_pos = [0, 0, 0.2]
ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Spawn the ball as a separate dynamic object
ball_id = p.loadURDF('pybullet/ball.urdf', ball_start_pos, ball_start_orientation, useFixedBase=False)
p.changeDynamics(ball_id, -1, lateralFriction=0.01, rollingFriction=0.0001, restitution=0.002)

simEnv = simEnvironment.ManipulatorSimEnv(robot_id, ball_id, max_RTF=True, steps_per_frame=8)
simEnv = DummyVecEnv([lambda: simEnv]) 
simEnv = VecNormalize(simEnv, norm_obs=True, norm_reward=False)

#model = PPO("MlpPolicy", simEnv, n_steps=2048, verbose=0)
model = PPO.load("8_spf_2.7M.zip", simEnv)  # Load the trained model
reward_logger_callback = simEnvironment.RolloutEndCallback(end_after_n_episodes=25)


model.learn(total_timesteps=2700000, callback=reward_logger_callback, progress_bar=True)
model.save("8_spf_2.7M_x2.1.zip")  # Save the trained model



p.disconnect()


plt.figure(figsize=(10, 5))
plt.plot(reward_logger_callback.smoothed_rewards, label='Mean Reward', color='blue')
plt.title('Training Progress')
plt.xlabel('Rollout Number')
plt.ylabel('Mean Episode Reward')
plt.grid(True)
plt.legend()
plt.show()