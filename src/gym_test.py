import multiprocessing as mp

from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.curriculum_callback import CurriculumCallback
from src.policy_types import ActorCriticPolicyType
from src.ppo_util import PostCurriculumEvalCallback, make_env, mask_fn

### INFO
### When reporting results, take mean and standard deviation
### of at least 5 runs. Report the seeds for reproducability.

HORIZON = 6
MAX_DIFF = 256
SLOPE = 0.9
TEST_SAMPLES = 3
TOTAL_STEPS = 35_000_000
EVAL_FREQ = 256_000
N_EVAL_EPISODES = 256
THRESHOLD = 0.85
BATCH_DIVISOR = 14 * 4
N_STEPS = 2048
EPOCHS = 4
LAYOUT_EXPONENT = 1.0
NUM_QUBITS = 4
NUM_EVAL_QUBITS = 4
NUM_ACTIVE_SWAPS = 24
INITIAL_DIFFICULTY = 1
POLICY_TYPE: ActorCriticPolicyType = ActorCriticPolicyType.BASIC_CANCEL
TENSORBOARD_LOG_DIR = "./logs/tensorboard/"
SAMPLE_DIFF = True
FAST_CURRICULUM = True
LOG_AVG_S_CX = False
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
LEARNING_RATE = 1e-4
GAMMA = 0.99
SHAPING_COEF = 0.00

if __name__ == "__main__":
    # backend = FakeTorino()
    # coupling_map = backend.coupling_map
    coupling_map = CouplingMap.from_grid(NUM_QUBITS, NUM_QUBITS)
    eval_coupling_map = CouplingMap.from_grid(NUM_EVAL_QUBITS, NUM_EVAL_QUBITS)
    n_envs = mp.cpu_count()
    buffer_size = N_STEPS * n_envs
    batch_size = 256  # max(1, buffer_size // BATCH_DIVISOR)
    print(f"Using {n_envs} envs")
    print(f"Batch size: {batch_size}")

    train_env = make_vec_env(
        lambda: make_env(
            coupling_map=coupling_map,
            num_active_swaps=NUM_ACTIVE_SWAPS,
            horizon=HORIZON,
            initial_difficulty=INITIAL_DIFFICULTY,
            max_difficulty=MAX_DIFF,
            diff_slope=SLOPE,
            gamma=GAMMA,
            layout_exponent=LAYOUT_EXPONENT,
            policy_type=POLICY_TYPE,
            shaping_coef=SHAPING_COEF,
            sample_diff=SAMPLE_DIFF,
        ),
        n_envs=n_envs,
    )

    model = MaskablePPO(
        POLICY_TYPE.get_sb3_policy(),
        train_env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        batch_size=batch_size,
        n_steps=N_STEPS,
        n_epochs=EPOCHS,
        policy_kwargs=POLICY_TYPE.get_policy_kwargs(),
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        learning_rate=LEARNING_RATE,
    )

    eval_env = make_env(
        coupling_map=eval_coupling_map,
        num_active_swaps=NUM_ACTIVE_SWAPS,
        horizon=HORIZON,
        render_mode="ansi",
        initial_difficulty=MAX_DIFF,
        max_difficulty=MAX_DIFF,
        diff_slope=SLOPE,
        layout_exponent=LAYOUT_EXPONENT,
        gamma=GAMMA,
        shaping_coef=SHAPING_COEF,
        policy_type=POLICY_TYPE,
        sample_diff=SAMPLE_DIFF,
    )
    eval_env = Monitor(eval_env)

    curriculum_callback = CurriculumCallback(
        threshold=THRESHOLD, use_fast_curriculum=FAST_CURRICULUM, verbose=1
    )

    eval_freq = max(EVAL_FREQ // n_envs, 1)
    conditional_eval = PostCurriculumEvalCallback(
        eval_env=eval_env,
        curriculum_callback=curriculum_callback,
        eval_freq=eval_freq,
        n_eval_episodes=N_EVAL_EPISODES,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        num_qubits=NUM_EVAL_QUBITS,
        log_avg_s_cx=LOG_AVG_S_CX,
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
            action_masks = mask_fn(eval_env)
            action, _ = model.predict(
                obs, deterministic=True, action_masks=action_masks
            )
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated:
                eval_env.render()
                flag = False
