from src.ppo_util import make_env, PostCurriculumEvalCallback, mask_fn
from stable_baselines3.common.monitor import Monitor
from src.gym_extractor import HybridExtractor, SimpleExtractor
from src.curriculum_callback import CurriculumCallback
from qiskit.transpiler import CouplingMap
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
import multiprocessing as mp

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

### INFO
### When reporting results, take mean and standard deviation
### of at least 5 runs. Report the seeds for reproducability.


HORIZON = 16
MAX_DIFF = 100
NUM_QUBITS = 6

if __name__ == "__main__":
    coupling_map = CouplingMap.from_line(NUM_QUBITS)
    n_envs = mp.cpu_count() - 1
    print(f"Using {n_envs} envs")

    train_env = make_vec_env(
        lambda: make_env(
            num_qubits=NUM_QUBITS,
            coupling_map=coupling_map,
            horizon=HORIZON,
            initial_difficulty=1,
            max_difficulty=MAX_DIFF,
        ),
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

    model = MaskablePPO("MlpPolicy", train_env, verbose=1, batch_size=2048)
    # model = MaskablePPO(
    #    "MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1
    # )

    eval_env = make_env(
        num_qubits=NUM_QUBITS,
        coupling_map=coupling_map,
        horizon=HORIZON,
        render_mode="ansi",
        initial_difficulty=MAX_DIFF,
        max_difficulty=MAX_DIFF,
    )
    eval_env = Monitor(eval_env)

    curriculum_callback = CurriculumCallback(threshold=0.85, verbose=1)

    eval_freq = max(100000 // n_envs, 1)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1,
    )

    conditional_eval = PostCurriculumEvalCallback(
        eval_callback, curriculum_callback, eval_freq
    )

    model.learn(
        total_timesteps=25_000_000,
        progress_bar=True,
        callback=[curriculum_callback, conditional_eval],
    )
    model.save("test_model")

    for _ in range(5):
        obs, _ = eval_env.reset()
        flag = True
        while flag:
            action_masks = mask_fn(eval_env)  # pyrefly: ignore
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated:
                eval_env.render()
                flag = False
