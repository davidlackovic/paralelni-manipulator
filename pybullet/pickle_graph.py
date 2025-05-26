import pickle
import matplotlib.pyplot as plt

with open('pybullet/training_data/8_spf_10M_v2.pkl', 'rb') as f:
    data = pickle.load(f)

plt.figure(figsize=(10, 5))
plt.plot(data, label='Mean Reward', color='blue')
plt.title('Training Progress')
plt.xlabel('Rollout Number')
plt.ylabel('Mean Episode Reward')
plt.grid(True)
plt.legend()
plt.show()