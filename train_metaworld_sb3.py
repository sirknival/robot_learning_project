"""
Meta-World MT10 Training Script with Stable Baselines3 (SAC)

- Lernt eine einzige Multi-Task-Policy über die 10 offiziellen MT10-Tasks
- Verwendet die offizielle MT10-Umgebung:
    envs = gym.make("Meta-World/MT10", vector_strategy="sync" oder "async", seed=SEED)
- Zwei-Phasen-Training:
    Phase 1: 0 -> 5 Mio Schritte (Modell + Replay-Buffer speichern)
    Phase 2: Fortsetzung von 5 -> 10 Mio Schritte (Modell + Replay-Buffer laden)
- Speichert zusätzlich regelmäßige Checkpoints:
    ./metaworld_models/checkpoints_MT10/        (Modelle, .zip)
    ./metaworld_models/checkpoints_MT10_buffer/ (Replay-Buffer, .pkl)

Paper: "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
Meta-World+: "An Improved, Standardized, RL Benchmark"
"""

import os

import gymnasium as gym
import metaworld
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, \
    StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from gymnasium.envs.registration import register

import metaworld_tasks
from ReplayBufferCheckpointCallback import *
from OneHotTaskWrapper import *
from env_generator import *
from debug_printer import *
from algorithms import *
from custom_multitask import *

# from callbacks import checkpoint_callback, eval_callback

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    EXPERIMENT = "MT10"  # "MT1 (then, task name has to be specified too.), MT3, MT10
    TASK_NAME = "reach-v3"  # Change to other MT1 tasks like "push-v3", "pick-place-v3", etc.
    ALGORITHM = "SAC"  # "TD3" or "DDPG" - SAC recommended for Meta-World

    # Environment Settings
    # USE_TRUE_PARALLEL = False  # Set to False for dummy vec environment
    SEED = 42

    # Training Settings
    # Training Mode [False = Start New Training (Phase 1) / True  = Continue training  (Phase 2 --> loads buffer model)]
    CONTINUE_TRAINING = False
    SEL_TRAIN_PHASE = 1

    # Training phases in steps (times factor 1e6)
    TRAIN_PHASES = {
        1: {"start": 5, "end": 10},
        2: {"start": 10, "end": 25},
        3: {"start": 25, "end": 40},
    }

    MODEL_BASENAME = EXPERIMENT + "_" + ALGORITHM
    paths_dict = {
        "first": {
            "model": f"./metaworld_models/{MODEL_BASENAME}_{TRAIN_PHASES[SEL_TRAIN_PHASE]['start']}M",
            "buffer": f"./metaworld_models/{MODEL_BASENAME}_{TRAIN_PHASES[SEL_TRAIN_PHASE]['start']}M_replay.pkl"},
        "second": {
            "model": f"./metaworld_models/{MODEL_BASENAME}_{TRAIN_PHASES[SEL_TRAIN_PHASE]['end']}M",
            "buffer": f"./metaworld_models/{MODEL_BASENAME}_{TRAIN_PHASES[SEL_TRAIN_PHASE]['end']}M_replay.pkl"}
    }

    model_phase = "start" if not CONTINUE_TRAINING else "end"
    total_timesteps = TRAIN_PHASES[SEL_TRAIN_PHASE][model_phase] * 1e6

    # TOTAL_TIMESTEPS = 1_000_000  # Increased for better convergence
    MAX_EPISODE_STEPS = 500  # Maximum steps per episode
    NORMALIZE_REWARD = False  # Set to True if experiencing training instability

    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 25000  # Save checkpoint every N steps

    # Debug Settings
    DEBUG = True
    # ======================================================
    # Register custom multitask
    """register(
        id='Meta-World/custom-mt-envs',
        entry_point='custom_multitask.CustomMTEnv',
    )"""

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}_buffer", exist_ok=True)

    if DEBUG:
        print_start_setup(EXPERIMENT, ALGORITHM, CONTINUE_TRAINING)

    # Define Tasks
    if EXPERIMENT == 'MT1':
        ENV_SPEC = TASK_NAME
    elif EXPERIMENT == 'MT3':
        ENV_SPEC = metaworld_tasks.MT3_TASKS
    elif EXPERIMENT == 'MT10':
        ENV_SPEC = metaworld_tasks.MT10_TASKS
    else:
        raise ValueError("Invalid EXPERIMENT setting.")

    # ------------------ MT3/10-Umgebungen ------------------

    print(f"Creating {EXPERIMENT} training and evaluation environments ...")
    if EXPERIMENT == 'MT3':
        env = make_mt3_env(0, SEED, MAX_EPISODE_STEPS)
        eval_env = make_mt3_env(0, SEED + 1000, MAX_EPISODE_STEPS)

    if EXPERIMENT == 'MT10':
        env = make_mt10_env(0, SEED, MAX_EPISODE_STEPS)
        eval_env = make_mt10_env(0, SEED + 1000, MAX_EPISODE_STEPS)

    #env = VecMonitor(env)
    #eval_env = VecMonitor(env)

    # Custom Wrapper Class for Hot Vector Encoding
    env = OneHotTaskWrapper(env, ENV_SPEC)
    eval_env = OneHotTaskWrapper(eval_env, ENV_SPEC)

    num_envs = getattr(env, "num_envs", 1)
    print(f" -> {EXPERIMENT} num_envs = {num_envs}")

    """
    if USE_TRUE_PARALLEL:
        print("Not implemented yet")
        env = SubprocVecEnv(
            [lambda i=i: make_mt_env(ENV_SPEC, i, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS)
             for i in range(N_ENVS)],
            start_method='spawn'
        )

    else:
        # Create training environment
        print("Creating training environment...")
        train_env = SubprocVecEnv(
            [make_env(task_name, seed=SEED + i)
             for i, task_name in enumerate(ENV_SPEC)],
            start_method="forkserver"
        )

        env = VecMonitor(train_env)  # sb3 wrapper (Gymnasium VectorEnv  -> Stable-Baselines3 VecEnv)

        # Create evaluation environment
        print("Creating evaluation environment...")
        eval_env = SubprocVecEnv(
            [make_env(task_name, seed=SEED + 1000 + i, max_episode_steps=MAX_EPISODE_STEPS)
             for i, task_name in enumerate(ENV_SPEC)],
            start_method="forkserver"
        )
        eval_env = VecMonitor(eval_env)
    """

    # Get action space dimensions
    if hasattr(env, "single_action_space"):
        action_space = env.single_action_space
    else:
        action_space = env.action_space
    n_actions = action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")
    if ALGORITHM == "TD3":
        model = model_factory_TD3(n_actions, env, ALGORITHM, SEED)

    elif ALGORITHM == "DDPG":
        model = model_factory_DDPG(n_actions, env, ALGORITHM, SEED)

    elif ALGORITHM == "SAC":
        model = model_factory_SAC(env, ALGORITHM, SEED, CONTINUE_TRAINING, paths_dict)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{TASK_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{TASK_NAME}",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{TASK_NAME}/",
        log_path=f"./metaworld_logs/eval_{TASK_NAME}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,  # More episodes for robust evaluation
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    buffer_checkpoint_callback = ReplayBufferCheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{EXPERIMENT}_buffer/",
        name_prefix=f"{ALGORITHM.lower()}_{EXPERIMENT}_buffer",
        verbose=1,
    )

    if DEBUG:
        print_training_start(model, TASK_NAME, ALGORITHM, total_timesteps, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD,
                             EVAL_FREQ, N_EVAL_EPISODES, CHECKPOINT_FREQ, SEL_TRAIN_PHASE, num_envs, action_space)

    callbacks = [eval_callback, checkpoint_callback]
    if ALGORITHM in ["SAC", "TD3", "DDPG"]:
        callbacks.append(buffer_checkpoint_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=not CONTINUE_TRAINING
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(paths_dict[model_phase]["model"])
    model.save_replay_buffer(paths_dict[model_phase]["buffer"])
    print(f"Model saved to: {paths_dict[model_phase]['model']}.zip")
    print(f"Replay buffer saved to: {paths_dict[model_phase]['buffer']}")

    # model.save(f"./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final")

    if DEBUG:
        print_training_finished(TASK_NAME, ALGORITHM)

    # Cleanup
    env.close()
    eval_env.close()
