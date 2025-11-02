import time
import numpy as np
import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import SAC

def main():
    env = gym.make("hover-aviary-v0", gui=True)
    model = SAC.load("sac_hover_aviary")
    obs, info = env.reset()

    for _ in range(2000):
        if isinstance(obs, np.ndarray) and obs.dtype != np.float32:
            obs = obs.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(1/10)
        if terminated or truncated:
            obs, info = env.reset()

    input("Press Enter to close...")
    env.close()

if __name__ == "__main__":
    main()
