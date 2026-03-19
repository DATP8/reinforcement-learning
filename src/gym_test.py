from gym_extractor import HybridExtractor, SimpleExtractor
from curriculum_callback import CurriculumCallback
import gymnasium
from qiskit.transpiler import CouplingMap
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import multiprocessing as mp
import numpy as np

from routing_env import RoutingEnv

cmap = CouplingMap.from_line(10)
n_envs = mp.cpu_count() - 1
print(f"Using {n_envs} envs")

def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()  # pyrefly: ignore


def make_env(cmap: CouplingMap, horizon: int, render_mode: str, initial_difficulty = 1):
    env = RoutingEnv(cmap, horizon, render_mode, initial_difficulty=initial_difficulty)
    env = ActionMasker(env, mask_fn)
    return env

train_env = make_vec_env(
    lambda: make_env(cmap, horizon=6, render_mode="human"),
    n_envs=n_envs,
)

# Simple Extractor
policy_kwargs = dict(
    features_extractor_class=SimpleExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)

# Hybrid Graph Extractor
policy_kwargs = dict(
    features_extractor_class=HybridExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)

# model = MaskablePPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1)
model = MaskablePPO("MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1)

curriculum_callback = CurriculumCallback(threshold=11.0, max_difficulty=20, verbose=1)

model.learn(total_timesteps=100000, progress_bar=True, callback=curriculum_callback)

eval_env = make_env(cmap, horizon=6, render_mode="human", initial_difficulty=20)

for _ in range(10):
    obs, _ = eval_env.reset()
    flag = True
    while flag:
        action_masks = eval_env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if terminated:
            eval_env.render()
            flag = False
