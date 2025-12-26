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

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from training_setup_multitask.utilities import MetaworldTasks
# from training_setup_multitask.utilities.ReplayBufferCheckpointCallback import *
from training_setup_multitask.utilities.DebugPrinter import *

from training_setup_multitask.WrapperClasses.OneHotTaskWrapper import OneHotTaskWrapper
from training_setup_multitask.utilities.CurriculumConfig import CurriculumConfig

from training_setup_multitask.utilities.MetaWorldEnvFactory import *
from training_setup_multitask.utilities.algorithms import *
from training_setup_multitask.utilities.TransferLearningManager import TransferLearningManager
from training_setup_multitask.Callbacks.ProgressiveTaskCallback import ProgressiveTaskCallback

# from callbacks import checkpoint_callback, eval_callback

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Training Strategy
    TRAINING_MODE = "PROGRESSIVE"  # "SEQUENTIAL", "PROGRESSIVE", or "MIXED"
    USE_TRANSFER_LEARNING = True  # Nutze vortrainiertes Modell als Basis
    USE_CURRICULUM = True  # Nutze Curriculum Learning

    # Pretrained Model (für Transfer Learning)
    PRETRAINED_MODEL_PATH = None  # "./metaworld_models/MT10_SAC_5M.zip"

    EXPERIMENT = "MT10"  # "MT1 (then, task name has to be specified too.), MT3, MT10
    TASK_NAME = "reach-v3"  # Change to other MT1 tasks like "push-v3", "pick-place-v3", etc.
    ALGORITHM = "SAC"  # "TD3" or "DDPG" - SAC recommended for Meta-World

    # Environment Settings
    SEED = 42

    # Curriculum Learning Settings
    CURRICULUM_STAGE = 0  # Start mit Stage 0 (einfachste Tasks)
    MIN_STEPS_PER_STAGE = 200000  # Minimum Steps pro Curriculum Stage
    STAGE_EVAL_FREQ = 10000

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

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}_buffer", exist_ok=True)
    os.makedirs(f"./metaworld_models/transfer_checkpoints", exist_ok=True)

    if DEBUG:
        print_start_setup(EXPERIMENT, ALGORITHM, CONTINUE_TRAINING)

    # Initialize Transfer Learning Manager
    transfer_manager = TransferLearningManager(PRETRAINED_MODEL_PATH)

    # Get curriculum configuration
    curriculum_config = CurriculumConfig()

    # Determine task list based on training mode
    if TRAINING_MODE == "SEQUENTIAL":
        # Train tasks one by one in order of difficulty
        sorted_tasks = sorted(
            MT10_TASKS,
            key=lambda x: curriculum_config.TASK_DIFFICULTY[x]
        )
        current_tasks = [sorted_tasks[CURRICULUM_STAGE]]
        print(f"Sequential Training - Current Task: {current_tasks[0]}")

    elif TRAINING_MODE == "PROGRESSIVE":
        # Use curriculum stages
        current_tasks = curriculum_config.CURRICULUM_STAGES[CURRICULUM_STAGE]
        print(f"Progressive Training - Stage {CURRICULUM_STAGE + 1}/{len(curriculum_config.CURRICULUM_STAGES)}")
        print(f"Tasks: {current_tasks}")

    else:  # MIXED
        # Start with easier tasks, gradually add harder ones
        current_tasks = MT10_TASKS[:3 + CURRICULUM_STAGE]
        print(f"Mixed Training - {len(current_tasks)} tasks")

    # Create environments based on current curriculum stage
    print(f"\nCreating environments for {len(current_tasks)} task(s)...")

    if len(current_tasks) == 1:
        # Single task environment
        env = make_mt1_env(current_tasks[0], SEED, MAX_EPISODE_STEPS)
        eval_env = make_mt1_env(current_tasks[0], SEED + 1000, MAX_EPISODE_STEPS)
    elif len(current_tasks) <= 3:
        # Small multi-task (MT3-like)
        env = make_mt3_env(0, SEED, MAX_EPISODE_STEPS)
        eval_env = make_mt3_env(0, SEED + 1000, MAX_EPISODE_STEPS)
    else:
        # Full MT10
        env = make_mt10_env(0, SEED, MAX_EPISODE_STEPS)
        eval_env = make_mt10_env(0, SEED + 1000, MAX_EPISODE_STEPS)

    # Apply task wrappers
    env = OneHotTaskWrapper(env, current_tasks)
    eval_env = OneHotTaskWrapper(eval_env, current_tasks)

    num_envs = getattr(env, "num_envs", 1)
    print(f"✓ Created {num_envs} parallel environment(s)")

    # Initialize or load model
    print(f"\nInitializing {ALGORITHM} agent...")

    if USE_TRANSFER_LEARNING and PRETRAINED_MODEL_PATH:
        # Load pretrained model for transfer learning
        from stable_baselines3 import SAC, TD3, DDPG

        algorithm_class = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}[ALGORITHM]
        model = transfer_manager.load_pretrained_model(
            algorithm_class,
            env,
            PRETRAINED_MODEL_PATH
        )

        if model:
            # Fine-tune for new tasks
            model = transfer_manager.fine_tune_for_new_tasks(
                model,
                current_tasks,
                learning_rate_multiplier=0.3
            )
    else:
        # Create new model from scratch
        if ALGORITHM == "SAC":
            model = model_factory_SAC(env, ALGORITHM, SEED, CONTINUE_TRAINING, paths_dict)
        elif ALGORITHM == "TD3":
            action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
            n_actions = action_space.shape[0]
            model = model_factory_TD3(n_actions, env, ALGORITHM, SEED)
        elif ALGORITHM == "DDPG":
            action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
            n_actions = action_space.shape[0]
            model = model_factory_DDPG(n_actions, env, ALGORITHM, SEED)
        else:
            raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{EXPERIMENT}/",
        name_prefix=f"{ALGORITHM.lower()}_stage{CURRICULUM_STAGE}",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{EXPERIMENT}/",
        log_path=f"./metaworld_logs/eval_{EXPERIMENT}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    callbacks = [eval_callback, checkpoint_callback]

    # Add curriculum callback if using progressive training
    if USE_CURRICULUM and TRAINING_MODE == "PROGRESSIVE":
        curriculum_callback = ProgressiveTaskCallback(
            curriculum_stages=curriculum_config.CURRICULUM_STAGES,
            stage_thresholds=curriculum_config.STAGE_THRESHOLDS,
            eval_freq=STAGE_EVAL_FREQ,
            min_steps_per_stage=MIN_STEPS_PER_STAGE,
            verbose=1
        )
        callbacks.append(curriculum_callback)

    if DEBUG:
        print_training_start(model, TASK_NAME, ALGORITHM, total_timesteps, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD,
                             EVAL_FREQ, N_EVAL_EPISODES, CHECKPOINT_FREQ, SEL_TRAIN_PHASE, num_envs, env.action_space)

    callbacks = [eval_callback, checkpoint_callback]
    # if ALGORITHM in ["SAC", "TD3", "DDPG"]:
    #    callbacks.append(buffer_checkpoint_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=not CONTINUE_TRAINING
    )

    # Save final model
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    model.save(paths_dict["second"]["model"])
    model.save_replay_buffer(paths_dict["second"]["buffer"])
    print(f"✓ Model saved to: {paths_dict['second']['model']}.zip")
    print(f"✓ Replay buffer saved to: {paths_dict['second']['buffer']}")

    # Save transfer checkpoint
    transfer_manager.save_transfer_checkpoint(
        model,
        stage=CURRICULUM_STAGE,
        tasks=current_tasks,
        save_path="./metaworld_models/transfer_checkpoints"
    )

    if DEBUG:
        print_training_finished(TASK_NAME, ALGORITHM)

    # Cleanup
    env.close()
    eval_env.close()
