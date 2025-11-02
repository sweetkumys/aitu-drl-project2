# train_td3.py
import gymnasium as gym
import gym_pybullet_drones
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from callbacks import RewardLoggerCallback

def main():
    env = gym.make("hover-aviary-v0")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        action_noise=action_noise,
        tensorboard_log="./tb_logs_td3/",
    )

    callback = RewardLoggerCallback(save_path="./logs/", filename="td3_rewards.npy")

    total_steps = 50_000
    model.learn(total_timesteps=total_steps, callback=callback)

    model.save("td3_hover_aviary")
    env.close()
    print("TD3 done")

if __name__ == "__main__":
    main()
