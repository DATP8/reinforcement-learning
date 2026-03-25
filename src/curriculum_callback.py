from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, threshold: float, max_difficulty: int, verbose: int = 0):
        super().__init__(verbose)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be in the interval [0, 1].")

        self.threshold = threshold
        self.max_difficulty = max_difficulty

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            current_diff = self.training_env.get_attr("current_difficulty")[0]
            if (
                "episode" in info
                and current_diff < self.max_difficulty
                and (performance := self._eval_episode_performance()) > self.threshold
            ):
                current_diff += 1
                self.training_env.set_attr("current_difficulty", current_diff)
                if self.verbose > 0:
                    print(
                        f"\nDifficulty increased to {current_diff} (Episode performance: {performance:.2f})"
                    )
        return True

    def _eval_episode_performance(self) -> float:
        # TODO: implement eval logic
        return 1.0
