from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import deque


class CurriculumCallback(BaseCallback):
    def __init__(self, threshold: float, max_difficulty: int, verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold
        self.max_difficulty = max_difficulty
        self.window_size = 100
        self.recent_rewards = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.recent_rewards.append(info["episode"]["r"])

        if len(self.recent_rewards) >= self.window_size:
            mean_reward = np.mean(self.recent_rewards)
            if mean_reward >= self.threshold:
                current_diff = self.training_env.get_attr("current_difficulty")[0]

                if current_diff < self.max_difficulty:
                    new_diff = current_diff + 1
                    self.training_env.set_attr("current_difficulty", new_diff)
                    self.recent_rewards.clear()

                    if self.verbose > 0:
                        print(
                            f"\nDifficulty increased to {new_diff} (Mean Reward: {mean_reward:.2f})"
                        )

        return True
