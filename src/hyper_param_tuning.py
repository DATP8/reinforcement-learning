import random

from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.schedulers import PopulationBasedTraining

import pprint
from ray.rllib.algorithms.algorithm import Algorithm

from model import Model, RetardModel, BiCircuitGNN, ValueModel
from routing_env import RoutingEnv
from topology import get_topology_from_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
            config["train_batch_size_per_learner"] = config["minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_epochs"] < 1:
            config["num_epochs"] = 1
        return config

    hyperparam_mutations = {
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size_per_learner": lambda: random.randint(2000, 160000),
    }
    
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )
    
    config = (
        PPOConfig()
            .environment(
                RoutingEnv,
                env_config={
                    "cmap": get_topology_from_file("./src/topologies/torino_topology.txt", 10),
                    "horizon": tune.choice([1, 5, 10, 50, 100]),
                    "initial_difficulty": 1, 
                },    
            )
            .env_runners(num_env_runners=1)
            .training(
                kl_coeff=1.0,
                lambda_=0.95,
                clip_param=0.2,
                lr=1e-4,
                num_epochs=tune.choice([10, 20, 30]),
                minibatch_size=tune.choice([128, 512, 2048]),
                train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
            )
            .rl_module(
                # Model selected once at trial start, never mutated mid-training
                model_config=tune.choice([
                    DefaultModelConfig(Model),
                    DefaultModelConfig(ValueModel),
                    DefaultModelConfig(BiCircuitGNN),
                    DefaultModelConfig(RetardModel),
                ])
            )
    )

    stopping_criteria = {"training_iteration": 100, "env_runners/episode_return_mean": 300}
    
    tuner = tune.Tuner(
        "PPO", 
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1 if args.smoke_test else 8,  # enough to sample all models
        ),
        param_space=config,
        run_config=tune.RunConfig(stop=stopping_criteria),
    )
    results = tuner.fit()


    best_result = results.get_best_result()

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint(
        {k: v for k, v in best_result.config.items() if k in hyperparam_mutations}
    )

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})


    loaded_ppo = Algorithm.from_checkpoint(best_result.checkpoint)
    loaded_policy = loaded_ppo.get_module()

    # See your trained policy in action
    # loaded_policy.compute_single_action(...)