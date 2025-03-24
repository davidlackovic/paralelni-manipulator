import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)

# Transition matrix: Models movement physics
kalman.transitionMatrix = np.array([[1, 0, 1, 0],  
                                    [0, 1, 0, 1],  
                                    [0, 0, 1, 0],  
                                    [0, 0, 0, 1]], np.float32)

# Measurement matrix: We only observe (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],  
                                     [0, 1, 0, 0]], np.float32)

# Increase process noise slightly (allows smooth adjustments)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  

# **STRONGLY decrease measurement noise** (almost fully trusts predictions)
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01  

# **Reduce initial error covariance** (so it barely adjusts to bad readings)
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.01  

# Simulated movement
true_positions = []
measured_positions = []
kalman_positions = []

x, y = 100, 100  # True ball position

for i in range(50):  # Simulate 50 frames
    # Move ball in a straight line
    x += 5
    y += 5
    
    # Simulate measurement noise (sometimes a big error)
    measured_x = x + random.gauss(0, 3)  # Small noise
    measured_y = y + random.gauss(0, 3)
    
    if random.random() < 0.1:  # 10% chance of a big error
        measured_x += random.choice([-25, 25])
        measured_y += random.choice([-25, 25])

    # Predict using Kalman
    prediction = kalman.predict()
    kalman_x, kalman_y = int(prediction[0]), int(prediction[1])

    # Correct using measurement
    measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])
    kalman.correct(measurement)
    
    # Store positions
    true_positions.append((x, y))
    measured_positions.append((measured_x, measured_y))
    kalman_positions.append((kalman_x, kalman_y))

# Plot Results
plt.figure(figsize=(8, 6))
plt.plot(*zip(*true_positions), label="True Position", marker='o', color='green', linestyle='dashed')
plt.plot(*zip(*measured_positions), label="Measured Position (Noisy)", marker='x', color='red')
plt.plot(*zip(*kalman_positions), label="Kalman Filtered (Very Smooth)", marker='s', color='blue')
plt.legend()
plt.title("Kalman Filter with Extreme Prediction Trust")
plt.show()
