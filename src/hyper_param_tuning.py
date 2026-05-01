import multiprocessing as mp
import os
import sys
import tempfile

import torch
from numpy import random
from qiskit.transpiler import CouplingMap
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna.optuna_search import OptunaSearch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from circuit_generator import CircuitGenerator
from src.curriculum_callback import CurriculumCallback
from src.policy_types import ActorCriticPolicyType
from src.ppo_util import make_env
from src.routing_env import RoutingEnv


class RayTuneCurriculumCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Monitor,
        curriculum_callback: CurriculumCallback,
        eval_freq: int,
        n_eval_episodes: int,
        num_qubits: int,
        seed: int,
        eval_set_seed: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._eval_env: RoutingEnv = eval_env.unwrapped  # pyrefly: ignore
        self._eval_freq = eval_freq
        self._curriculum_callback = curriculum_callback
        self._seed = seed
        self._post_curriculum_evals = 0
        self._best_avg_num_swaps = sys.float_info.max
        self._eval_circuits = CircuitGenerator.generate_n_random_cx_circuits(
            n=n_eval_episodes,
            num_qubits=num_qubits,
            num_gates=[i * 8 for i in range(1, 26)],
            seed=eval_set_seed,
        )

    def _on_step(self) -> bool:
        current_diff = self.training_env.env_method("get_difficulty")[0]
        curriculum_done = current_diff >= self._curriculum_callback.max_difficulty

        if not curriculum_done:
            return True

        if self._eval_freq > 0 and self.n_calls % self._eval_freq == 0:
            self._post_curriculum_evals += 1

            avg_num_swaps = self._compute_num_avg_swaps()

            metrics = {
                "avg_swaps": avg_num_swaps,
                "diff": current_diff,
                "seed": self._seed,
                "pc_evals": self._post_curriculum_evals,
            }

            if avg_num_swaps < self._best_avg_num_swaps:
                self._best_avg_num_swaps = avg_num_swaps
                metrics["best_swaps"] = self._best_avg_num_swaps
                with tempfile.TemporaryDirectory() as ckpt_dir:
                    self.model.save(os.path.join(ckpt_dir, "model"))
                    checkpoint = tune.Checkpoint.from_directory(ckpt_dir)
                    tune.report(metrics, checkpoint=checkpoint)
            else:
                metrics["best_swaps"] = self._best_avg_num_swaps
                tune.report(metrics)

        return True

    def _compute_num_avg_swaps(self) -> float:
        if not isinstance(self.model, MaskablePPO):
            raise ValueError("Must be maskable PPO")

        num_swaps = 0
        for circuit in self._eval_circuits:
            obs, info = self._eval_env.reset(options={"circuit": circuit})
            done = False
            while not done:
                mask = self._eval_env.valid_action_mask()
                action, _ = self.model.predict(
                    obs, action_masks=mask, deterministic=True
                )

                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                done = terminated

            routed_circuit = self._eval_env.routed_circuit
            ops = routed_circuit.count_ops()
            num_swaps += ops.get("swap", 0)

        return num_swaps / len(self._eval_circuits)


def maskable_ppo_obj(config):
    seed = random.randint(0, 2**31 - 1)
    coupling_map = CouplingMap.from_line(config["num_qubits"])

    train_env = make_vec_env(
        lambda: make_env(
            coupling_map=coupling_map,
            num_active_swaps=config["num_active_swaps"],
            horizon=config["horizon"],
            diff_slope=config["diff_slope"],
            layout_exponent=config["layout_exponent"],
            initial_difficulty=config["initial_difficulty"],
            max_difficulty=config["max_difficulty"],
            policy_type=config["policy_type"],
        ),
        n_envs=config["num_envs"],
        seed=seed,
    )

    eval_env = make_env(
        coupling_map=coupling_map,
        num_active_swaps=config["num_active_swaps"],
        horizon=config["horizon"],
        diff_slope=config["diff_slope"],
        layout_exponent=config["layout_exponent"],
        initial_difficulty=config["max_difficulty"],
        max_difficulty=config["max_difficulty"],
        policy_type=config["policy_type"],
    )
    eval_env = Monitor(eval_env)

    model = MaskablePPO(
        policy=config["policy_type"].get_sb3_policy(),
        policy_kwargs=config["policy_type"].get_policy_kwargs(),
        env=train_env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        seed=seed,
        ent_coef=config["ent_coef"],
    )

    curriculum_callback = CurriculumCallback(config["threshold"])

    eval_freq = max(config["base_eval_freq"] // config["num_envs"], 1)
    ray_tune_eval = RayTuneCurriculumCallback(
        eval_env,
        curriculum_callback,
        eval_freq,
        config["n_eval_episodes"],
        config["num_qubits"],
        seed,
        eval_set_seed=EVAL_SEED,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[curriculum_callback, ray_tune_eval],
    )


if __name__ == "__main__":
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    EVAL_SEED = 2026
    CPUS_PER_TRIAL = 4
    NUM_UNIQUE_SAMPLES = 128
    REPEATS_PER_CONFIG = 1
    GRACE_PERIOD = 1

    total_cpus = mp.cpu_count()
    num_concurrent_trials = max(1, total_cpus // CPUS_PER_TRIAL)

    # num_samples = repeats_per_config * num_unique_samples # when using Repeater
    num_samples = NUM_UNIQUE_SAMPLES

    gpus_per_trial = 1.0 / num_concurrent_trials if torch.cuda.is_available() else 0.0

    num_qubits = 3

    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "gamma": tune.uniform(0.8, 1.0),
        "gae_lambda": tune.uniform(0.9, 1.0),
        "batch_size": tune.choice([512, 1024, 2048, 4096]),
        "horizon": tune.randint(4, 64),
        "policy_type": ActorCriticPolicyType.BASIC,  # tune.choice([e for e in ActorCriticPolicyType]),
        "n_steps": tune.choice([256, 512, 1024, 2048]),
        "num_active_swaps": tune.randint(1, num_qubits + 1),
        "ent_coef": tune.loguniform(1e-5, 0.05),
        "num_qubits": num_qubits,
        "initial_difficulty": 1,
        "max_difficulty": 256,
        "diff_slope": 1,
        "layout_exponent": 1.0,
        "threshold": 0.85,
        "base_eval_freq": 100_000,
        "n_eval_episodes": 100,
        "total_timesteps": 10_000_000,
        "num_envs": CPUS_PER_TRIAL,
        "n_epochs": 10,
    }

    algo = OptunaSearch(metric="best_swaps", mode="min")
    # repeated_algo = Repeater(algo, repeat=repeats_per_config) #! Can cause problems when using scheduler

    max_evals = search_space["total_timesteps"] // search_space["base_eval_freq"]

    scheduler = ASHAScheduler(
        time_attr="pc_evals",
        metric="avg_swaps",
        mode="min",
        max_t=max_evals,
        grace_period=GRACE_PERIOD,
    )

    reporter = CLIReporter(
        infer_limit=10,
        print_intermediate_tables=True,
        metric="best_swaps",
        mode="min",
        sort_by_metric=True,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            maskable_ppo_obj, resources={"cpu": CPUS_PER_TRIAL, "gpu": gpus_per_trial}
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples, search_alg=algo, scheduler=scheduler
        ),
        run_config=tune.RunConfig(progress_reporter=reporter),
    )

    results = tuner.fit()

    df = results.get_dataframe()

    config_cols = [col for col in df.columns if col.startswith("config/")]

    agg_df = (
        df.groupby(config_cols)
        .agg(
            best_swaps=("best_swaps", "min"),
            final_avg_swaps=("avg_swaps", "last"),
            seeds_used=("seed", lambda x: list(x)),
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values("best_swaps", ascending=True)

    print(f"\n--- Top Hyperparameters (Averaged over {REPEATS_PER_CONFIG} seeds) ---")
    print(agg_df.to_string(index=False))

    agg_df.to_csv("results.csv", index=False)
