# callbacks.py
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, save_path: str = "./logs/", filename: str = "episode_rewards.npy", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.filename = filename
        os.makedirs(self.save_path, exist_ok=True)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
        return True

    def _on_training_end(self) -> None:
        path = os.path.join(self.save_path, self.filename)
        np.save(path, np.array(self.episode_rewards, dtype=float))
