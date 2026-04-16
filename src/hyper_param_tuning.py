from ray.tune.schedulers import ASHAScheduler
from numpy import random
from ray.tune.search import Repeater
from ray import train
from stable_baselines3.common.callbacks import BaseCallback
from ray.tune.search.optuna.optuna_search import OptunaSearch
from qiskit.transpiler import CouplingMap
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from src.curriculum_callback import CurriculumCallback
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from src.ppo_util import make_env
from ray import tune
import multiprocessing as mp
import torch


class RayTuneCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_callback: MaskableEvalCallback,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        seed,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._eval_callback = eval_callback
        self._curriculum_callback = curriculum_callback
        self._eval_freq = eval_freq
        self._last_mean_reward = -float("inf")
        self._seed = seed
        self._post_curriculum_evals = 0

    def _init_callback(self) -> None:
        self._eval_callback.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq == 0:
            current_diff = self.training_env.env_method("get_difficulty")[0]
            if current_diff >= self._curriculum_callback.max_difficulty:
                self._eval_callback.on_step()
                self._last_mean_reward = self._eval_callback.last_mean_reward
                self._post_curriculum_evals += 1

            train.report(
                {
                    "mean_reward": self._last_mean_reward,
                    "difficulty": current_diff,
                    "seed": self._seed,
                    "post_curriculum_evals": self._post_curriculum_evals,
                }
            )
        return True


def objective(config):
    seed = random.randint(0, 2**31 - 1)
    config["coupling_map"] = CouplingMap.from_line(config["num_qubits"])

    train_env = make_vec_env(
        lambda: make_env(
            config["num_qubits"],
            config["coupling_map"],
            config["horizon"],
            initial_difficulty=config["initial_difficulty"],
            max_difficulty=config["max_difficulty"],
            depth_slope=config["depth_slope"],
            max_depth=config["max_depth"],
        ),
        n_envs=config["n_envs"],
        seed=seed,
    )

    eval_env = make_env(
        config["num_qubits"],
        config["coupling_map"],
        config["horizon"],
        config["initial_difficulty"],
        config["max_difficulty"],
    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        batch_size=config["batch_size"],
        seed=seed,
    )

    curriculum_callback = CurriculumCallback(config["threshold"])

    eval_freq = max(config["base_eval_freq"] // config["n_envs"], 1)

    eval_callback = MaskableEvalCallback(eval_env, eval_freq=eval_freq, verbose=0)

    ray_tune_eval = RayTuneCurriculumCallback(
        eval_callback, curriculum_callback, eval_freq, seed
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[curriculum_callback, ray_tune_eval],
    )


if __name__ == "__main__":
    cpus_per_trial = 4
    num_unique_samples = 100
    repeats_per_config = 3
    grace_period = 5

    total_cpus = mp.cpu_count()
    num_concurrent_trials = max(1, total_cpus // cpus_per_trial)

    num_samples = repeats_per_config * num_unique_samples

    gpus_per_trial = 1.0 / num_concurrent_trials if torch.cuda.is_available() else 0.0

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "gamma": tune.uniform(0.8, 1.0),
        "gae_lambda": tune.uniform(0.9, 1.0),
        "batch_size": tune.choice([1024, 2048]),
        "horizon": tune.randint(1, 101),
        "num_qubits": 6,
        "initial_difficulty": 1,
        "max_difficulty": 100,
        "depth_slope": 2,
        "max_depth": 128,
        "threshold": 0.85,
        "base_eval_freq": 100_000,
        "total_timesteps": 5_000_000,
        "n_envs": cpus_per_trial,
    }

    algo = OptunaSearch(metric="mean_reward", mode="max")
    repeated_algo = Repeater(algo, repeat=repeats_per_config)

    max_evals = search_space["total_timesteps"] // search_space["base_eval_freq"]

    scheduler = ASHAScheduler(
        time_attr="post_curriculum_evals",
        metric="mean_reward",
        mode="max",
        max_t=max_evals,
        grace_period=grace_period,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            objective, resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples, search_alg=repeated_algo, scheduler=scheduler
        ),
    )

    results = tuner.fit()

    df = results.get_dataframe()

    config_cols = [col for col in df.columns if col.startswith("config/")]

    agg_df = (
        df.groupby(config_cols)
        .agg(
            avg_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            seeds_used=("seed", lambda x: list(x)),
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values("avg_reward", ascending=False)

    print(f"\n--- Top Hyperparameters (Averaged over {repeats_per_config} seeds) ---")
    print(agg_df.to_string(index=False))
