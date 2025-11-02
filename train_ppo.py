import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from callbacks import RewardLoggerCallback

def main():
    env = gym.make("hover-aviary-v0")

    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./tb_logs_ppo/",
    )

    callback = RewardLoggerCallback(save_path="./logs/", filename="ppo_rewards.npy")

    total_steps = 50_000
    model.learn(total_timesteps=total_steps, callback=callback)

    model.save("ppo_hover_aviary")
    env.close()
    print("PPO done")

if __name__ == "__main__":
    main()
