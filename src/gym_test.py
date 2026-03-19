from curriculum_callback import CurriculumCallback
import gymnasium
from qiskit.transpiler import CouplingMap
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import multiprocessing as mp
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from routing_env import RoutingEnv

cmap = CouplingMap.from_line(4)
n_envs = mp.cpu_count() - 1
print(f"Using {n_envs} envs")

# class FeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim=128):
#         super().__init__(observation_space, features_dim)
#
#         N, H = observation_space.shape
#
#         print(observation_space)
#
#         # TODO: This is just a random network
#         self.net = nn.Sequential(
#             nn.Conv2d(H, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * N * N, features_dim),
#             nn.ReLU(),
#         )
#
#     def forward(self, obs):
#         # convert (batch, N, N, H) → (batch, H, N, N)
#         # obs = obs.permute(0, 2, 1)
#         # Ensure batch dimension
#         if obs.dim() == 3:
#             obs = obs.unsqueeze(0)
#
#         # Reorder for Conv2d
#         obs = obs.permute(0, 3, 1, 2)
#         # obs = obs.permute(0, 3, 1, 2)
#         return self.net(obs)

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0] * observation_space.shape[1]

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)

def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()  # pyrefly: ignore


def make_env(cmap: CouplingMap, horizon: int, render_mode: str):
    env = RoutingEnv(cmap, horizon, render_mode)
    env = ActionMasker(env, mask_fn)
    return env

train_env = make_vec_env(
    lambda: make_env(cmap, horizon=6, render_mode="human"),
    n_envs=n_envs,
)

policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
)

model = MaskablePPO("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1)

curriculum_callback = CurriculumCallback(threshold=11.0, max_difficulty=20, verbose=1)

model.learn(total_timesteps=100000, progress_bar=True, callback=curriculum_callback)

eval_env = make_env(cmap, horizon=6, render_mode="human")

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
