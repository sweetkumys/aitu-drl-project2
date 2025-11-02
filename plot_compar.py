import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "./logs/"

def load(name):
    path = os.path.join(LOG_DIR, name)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None

def smooth(x, k=50):
    if x is None:
        return None
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k)/k, mode="valid")

sac  = load("sac_rewards.npy")
td3  = load("td3_rewards.npy")
ddpg = load("ddpg_rewards.npy")
ppo  = load("ppo_rewards.npy")

plt.figure()

if sac is not None:
    plt.plot(smooth(sac), label="SAC")
if td3 is not None:
    plt.plot(smooth(td3), label="TD3")
if ddpg is not None:
    plt.plot(smooth(ddpg), label="DDPG")
if ppo is not None:
    plt.plot(smooth(ppo), label="PPO")

plt.xlabel("Episode (logged)")
plt.ylabel("Return")
plt.title("Comparison on hover-aviary-v0")
plt.legend()
plt.grid(True)
plt.show()