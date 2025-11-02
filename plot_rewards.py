# plot_rewards.py
import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("./logs/episode_rewards.npy", allow_pickle=True)

plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("SAC on hover-aviary-v0")
plt.grid(True)
plt.show()
