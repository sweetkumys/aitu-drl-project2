import gymnasium as gym
import gym_pybullet_drones

def make_env(gui: bool = False):
    env = gym.make('hover-aviary-v0', gui=gui)
    return env

if __name__ == "__main__":
    e = make_env(gui=True)
    obs = e.reset()
    print("Initial Observation:", obs.shape)
    e.close()