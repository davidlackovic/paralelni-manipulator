import pybullet as p
import pybullet_data
import numpy as np
import time

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
ball_start_pos = [0, 0, 0.2]
ball_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Spawn the ball as a separate dynamic object
ball_id = p.loadURDF('pybullet/ball.urdf', ball_start_pos, ball_start_orientation, useFixedBase=False)
p.changeDynamics(ball_id, -1, lateralFriction=0.01, rollingFriction=0.0001, restitution=0.002)

def reset_ball_position():
    # Reset the ball's position and velocity
    p.resetJointState(robot_id, joint_X, targetValue=0, targetVelocity=0)
    p.resetJointState(robot_id, joint_Y, targetValue=0, targetVelocity=0)
    p.resetBasePositionAndOrientation(ball_id, ball_start_pos, ball_start_orientation)
    p.resetBaseVelocity(ball_id, [0, 0, 0], [0, 0, 0])

# Main loop
try:
    while True:
        p.stepSimulation()
        #images = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        plate_orientation = p.getJointState(robot_id, joint_X)[0], p.getJointState(robot_id, joint_Y)[0]
        ball_position, _ = p.getBasePositionAndOrientation(ball_id)
        print(f"Plate Orientation X: {plate_orientation[0]}, Y: {plate_orientation[1]}, ball Position: {ball_position}")
        time.sleep(1/240.0)

        if np.linalg.norm(ball_position) > 0.35:
            print("Ball out of bounds, resetting...")
            reset_ball_position()


except KeyboardInterrupt:
    pass

p.disconnect()