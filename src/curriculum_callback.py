from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, threshold: float, max_difficulty: int, eval_env: ActionMasker, verbose: int = 0):
        super().__init__(verbose)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be in the interval [0, 1].")

        self.threshold = threshold
        self.max_difficulty = max_difficulty
        self.eval_env = eval_env
        self.ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            current_diff = self.training_env.get_attr("current_difficulty")[0]
            if (
                "episode" in info
            ):
                self.ep_count += 1
                if not (self.ep_count % 2048 == 0):
                    continue

                if (current_diff < self.max_difficulty
                    and (performance := self._eval_episode_performance(current_diff)) > self.threshold
                ):
                    current_diff += 1
                    self.training_env.set_attr("current_difficulty", current_diff)
                    if self.verbose > 0:
                        print(
                            f"\nDifficulty increased to {current_diff} (Episode performance: {performance:.2f})"
                        )
        return True

    def _eval_episode_performance(self, current_diff: int) -> float:
        success = 0
        self.eval_env.set_wrapper_attr("current_difficulty", current_diff)
        for _ in range(100):
            obs, _ = self.eval_env.reset()
            is_success = False
            is_done = False
            while not is_done:
                action, _ = self.model.predict(obs, deterministic=True, action_masks=self.eval_env.action_masks()) # pyrefly: ignore
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                is_done = terminated or truncated
                is_success = terminated
            success += int(is_success)
        return success / 100
