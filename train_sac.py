import gymnasium as gym
import gym_pybullet_drones
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from callbacks import RewardLoggerCallback

def main():
    env = gym.make("hover-aviary-v0")

    model = SAC(MlpPolicy, env, verbose=1, learning_rate=1e-3, buffer_size = 100_000, batch_size=256, gamma = 0.99, tau=0.02,tensorboard_log="./tb_logs/")
    total_steps= 50_000
    callback = RewardLoggerCallback(save_path="./logs/")
    model.learn(total_timesteps=total_steps, callback = callback)
    model.save("sac_hover_aviary2")
    env.close()
    print(f"Training completed for {total_steps} timesteps and model saved as sac_hover_aviary.zip")    

if __name__ == "__main__":
    main()