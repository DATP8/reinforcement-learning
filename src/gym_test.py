import multiprocessing as mp

from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.curriculum_callback import CurriculumCallback
from src.gym_extractor import SimpleExtractor
from src.policy_types import ActorCriticPolicyType
from src.ppo_util import PostCurriculumEvalCallback, make_env, mask_fn

### INFO
### When reporting results, take mean and standard deviation
### of at least 5 runs. Report the seeds for reproducability.


HORIZON = 64
MAX_DIFF = 256
SLOPE = 1
TEST_SAMPLES = 3
TOTAL_STEPS = 10_000_000
EVAL_FREQ = 100_000
N_EVAL_EPISODES = 10
THRESHOLD = 0.85
BATCH_SIZE = 2048
N_STEPS = 512
EPOCHS = 10
LAYOUT_EXPONENT = 1.0
NUM_QUBITS = 6
NUM_ACTIVE_SWAPS = 6
INITIAL_DIFFICULTY = 1
POLICY_TYPE: ActorCriticPolicyType = ActorCriticPolicyType.BASIC

if __name__ == "__main__":
    coupling_map = CouplingMap.from_line(NUM_QUBITS)
    n_envs = mp.cpu_count() - 1
    print(f"Using {n_envs} envs")

    train_env = make_vec_env(
        lambda: make_env(
            coupling_map=coupling_map,
            num_active_swaps=NUM_ACTIVE_SWAPS,
            horizon=HORIZON,
            initial_difficulty=INITIAL_DIFFICULTY,
            max_difficulty=MAX_DIFF,
            diff_slope=SLOPE,
            layout_exponent=LAYOUT_EXPONENT,
            policy_type=POLICY_TYPE,
        ),
        n_envs=n_envs,
    )

    # Simple Extractor
    policy_kwargs: dict[
        str, type[SimpleExtractor] | dict[str, int] | dict[str, list[int]]
    ] = dict(
        features_extractor_class=SimpleExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    # Hybrid Graph Extractor
    # policy_kwargs = dict(
    #    features_extractor_class=HybridExtractor,
    #    features_extractor_kwargs=dict(features_dim=128),
    #    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    # )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=EPOCHS,
    )
    # model = MaskablePPO(
    #    "MultiInputPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1
    # )

    eval_env = make_env(
        coupling_map=coupling_map,
        num_active_swaps=NUM_ACTIVE_SWAPS,
        horizon=HORIZON,
        render_mode="ansi",
        initial_difficulty=MAX_DIFF,
        max_difficulty=MAX_DIFF,
        diff_slope=SLOPE,
        layout_exponent=LAYOUT_EXPONENT,
        policy_type=POLICY_TYPE,
    )
    eval_env = Monitor(eval_env)

    curriculum_callback = CurriculumCallback(threshold=THRESHOLD, verbose=1)

    eval_freq = max(EVAL_FREQ // n_envs, 1)
    conditional_eval = PostCurriculumEvalCallback(
        eval_env=eval_env,
        curriculum_callback=curriculum_callback,
        eval_freq=eval_freq,
        n_eval_episodes=N_EVAL_EPISODES,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        callback=[curriculum_callback, conditional_eval],
    )
    model.save("test_model")

    for _ in range(TEST_SAMPLES):
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
