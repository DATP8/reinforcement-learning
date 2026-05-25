import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(
        self, threshold: float, use_fast_curriculum: bool = True, verbose: int = 0
    ):
        super().__init__(verbose)
        self.threshold = threshold
        self.rollout_successes = []
        self.use_fast_curriculum = use_fast_curriculum
        self.exponent = 0

    def _on_training_start(self) -> None:
        self.max_difficulty = self.training_env.get_attr("_max_difficulty")[0]

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                is_truncated = info.get("TimeLimit.truncated", False)
                self.rollout_successes.append(not is_truncated)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.rollout_successes) == 0:
            return

        success_rate = np.mean(self.rollout_successes)
        current_diff = self.training_env.env_method("get_difficulty")[0]

        if current_diff < self.max_difficulty and self.verbose > 0:
            print(
                f"\n[Curriculum] Rollout success rate: {success_rate} (Difficulty {current_diff})"
            )
        if success_rate >= self.threshold and current_diff < self.max_difficulty:
            if success_rate >= 1.0 and self.use_fast_curriculum:
                self.exponent += 1
            else:
                self.exponent = 0

            current_diff = min(
                current_diff + np.floor(1.999**self.exponent), self.max_difficulty
            )
            self.training_env.env_method("set_difficulty", current_diff)
            if self.verbose > 0:
                print(f"[Curriculum] Difficulty increased to {current_diff}!")

        self.rollout_successes.clear()

        current_diff = self.training_env.env_method("get_difficulty")[0]
        self.logger.record("curriculum/difficulty", current_diff)
        self.logger.record("curriculum/success_rate", success_rate)
