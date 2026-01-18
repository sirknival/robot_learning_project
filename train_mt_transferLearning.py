"""
Meta-World Multi-Task Training Script with Curriculum & Transfer Learning

Features:
- Multi-task training (MT1, MT3, MT10)
- Curriculum Learning: Progressive task introduction with automatic stage transitions
- Transfer Learning: Fine-tune pretrained models for new tasks
- Three training strategies: SEQUENTIAL, PROGRESSIVE, MIXED
- Supports SAC, TD3, DDPG algorithms
- Automatic checkpoint saving and evaluation

Usage:
1. Configure training parameters in the CONFIGURATION section
2. Choose training mode (SEQUENTIAL/PROGRESSIVE/MIXED)
3. Set curriculum stage and enable transfer learning if desired
4. Run script to start training

Papers:
- "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
- "Meta-World+: An Improved, Standardized, RL Benchmark"
"""

import os
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from helper_classes_transferLearning.utilities.MetaworldTasks import MT10_TASKS, MT3_TASKS
from helper_classes_transferLearning.utilities.DebugPrinter import DebugPrinter
from helper_classes_transferLearning.utilities.CurriculumConfig import CurriculumConfig
from helper_classes_transferLearning.utilities.TransferLearningManager import TransferLearningManager
from helper_classes_transferLearning.utilities.MetaWorldEnvFactory import MetaWorldEnvFactory
from helper_classes_transferLearning.utilities.algorithms import model_factory_SAC, model_factory_TD3, model_factory_DDPG
from helper_classes_transferLearning.utilities.TaskEvaluator import TaskEvaluator, FinalModelEvaluator

from helper_classes_transferLearning.WrapperClasses.OneHotTaskWrapper import OneHotTaskWrapper
from helper_classes_transferLearning.Callbacks.ProgressiveTaskCallback import ProgressiveTaskCallback

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================

    # -------------------- Training Strategy --------------------
    TRAINING_MODE = "PROGRESSIVE"    # Options: "SEQUENTIAL", "PROGRESSIVE", "MIXED"
    USE_TRANSFER_LEARNING = False    # Use pretrained model as starting point
    USE_CURRICULUM = True           # Enable curriculum learning with automatic stage transitions
    PRETRAINED_MODEL_PATH = None    # None path in case of initial training
    # PRETRAINED_MODEL_PATH = './metaworld_models/MT10k_SAC_5M'   # Path to pretrained model without .zip
    USE_SUBPROC_VEC_ENV = True      # Use SubprocVecEnv (True) for faster multi-processing or DummyVecEnv (False)

    # -------------------- Experiment Setup --------------------
    EXPERIMENT = "MT10_CURRICULUM"  # Options: "MT1", "MT3", "MT10", "MT10_CURRICULUM"
    TASK_NAME = "reach-v3"          # Required for MT1 (e.g., "reach-v3", "push-v3")
    ALGORITHM = "SAC"               # Options: "SAC", "TD3", "DDPG" (SAC standard)
    SEED = 42                       # Random seed for reproducibility
    N_PARALLEL_ENVS = 1             # Number of parallel environments (1-10, higher = faster training)
    MAX_TASKS = len(MT10_TASKS)     # MT10 Project; Do not change
    TOTAL_TIMESTEPS = 5 * 1e6       # Number of total time-steps while training

    # -------------------- Curriculum Settings --------------------
    CURRICULUM_STAGE = 0           # Starting curriculum stage (0 = easiest tasks)
    MIN_STEPS_PER_STAGE = 200000   # Minimum training steps before stage transition
    STAGE_EVAL_FREQ = 10000        # Evaluate performance every N steps for stage transitions

    # ----------------- Transfer Learning Settings -----------------
    LEARN_RATE_MULTIPLIER = 0.3     # Reduces learning rate by factor to avoid forgetting

    # -------------------- Environment Settings --------------------
    MAX_EPISODE_STEPS = 500        # Maximum steps per episode
    NORMALIZE_REWARD = False       # Normalize rewards (enable if training is unstable)

    # -------------------- Evaluation & Checkpointing --------------------
    EVAL_FREQ = 10000              # Evaluate model every N steps
    N_EVAL_EPISODES = 10           # Number of episodes for evaluation
    CHECKPOINT_FREQ = 50000        # Save checkpoint every N steps
    RUN_FINAL_EVAL = True
    FINAL_EVAL_EPISODES = 10

    # -------------------- Debug Settings --------------------
    DEBUG = True                   # Enable verbose debug output

    # ======================================================

    # -------------------- Setup Paths --------------------
    MODEL_BASENAME = f"{EXPERIMENT}_{ALGORITHM}_{int(TOTAL_TIMESTEPS / 1e6)}M"

    paths_dict = {
            "model": f"./metaworld_models/{MODEL_BASENAME}",
            "buffer": f"./metaworld_models/{MODEL_BASENAME}_replay.pkl"
    }

    # -------------------- Create Directories --------------------
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{EXPERIMENT}_buffer", exist_ok=True)
    os.makedirs("./metaworld_models/transfer_checkpoints", exist_ok=True)

    # -------------------- Initialize Components --------------------

    # Initialize debug printer
    printer = DebugPrinter(verbose=DEBUG, line_width=70)

    # Print initial setup
    printer.print_start_setup(
        experiment=EXPERIMENT,
        algorithm=ALGORITHM,
        training_mode=TRAINING_MODE,
        use_transfer=USE_TRANSFER_LEARNING,
        use_curriculum=USE_CURRICULUM
    )

    # Initialize curriculum configuration
    curriculum_config = CurriculumConfig()

    # Initialize transfer learning manager
    transfer_manager = TransferLearningManager(PRETRAINED_MODEL_PATH)

    # Initialize environment factory
    env_factory = MetaWorldEnvFactory(
        reward_function_version='v3',
        vector_strategy='sync',
        terminate_on_success=False,
        use_subproc=USE_SUBPROC_VEC_ENV,
        verbose=DEBUG
    )

    # -------------------- Determine Task List --------------------

    # Simple mode selection based on EXPERIMENT
    if EXPERIMENT == "MT1":
        # Single task training
        if not TASK_NAME:
            raise ValueError("TASK_NAME must be specified for MT1 training")
        current_tasks = [TASK_NAME]

        if DEBUG:
            printer.print_section(f"MT1 Training - Single Task")
            print(f"Task: {TASK_NAME}")
            print(f"Parallel Environments: {N_PARALLEL_ENVS}")
            difficulty = curriculum_config.TASK_DIFFICULTY.get(TASK_NAME, "Unknown")
            print(f"Difficulty Level: {difficulty}")

    elif EXPERIMENT == "MT3":
        # Fixed 3-task training (default MT3 tasks)
        current_tasks = MT3_TASKS

        if DEBUG:
            printer.print_section("MT3 Training - Three Tasks")
            for i, task in enumerate(current_tasks, 1):
                difficulty = curriculum_config.TASK_DIFFICULTY.get(task, "Unknown")
                print(f"  {i}. {task} (difficulty: {difficulty})")

    elif EXPERIMENT == "MT10":
        # All 10 tasks training
        current_tasks = MT10_TASKS

        if DEBUG:
            printer.print_section("MT10 Training - All Tasks")
            print(f"Training with all {len(current_tasks)} MT10 tasks")

    elif "CURRICULUM" in EXPERIMENT:
        # Curriculum learning mode
        if TRAINING_MODE == "SEQUENTIAL":
            # Train tasks one by one in order of difficulty
            sorted_tasks = sorted(
                MT10_TASKS,
                key=lambda x: curriculum_config.TASK_DIFFICULTY[x]
            )
            current_tasks = [sorted_tasks[CURRICULUM_STAGE]]

            if DEBUG:
                printer.print_section(f"Sequential Training - Task {CURRICULUM_STAGE + 1}/{len(sorted_tasks)}")
                print(f"Current Task: {current_tasks[0]}")
                print(f"Difficulty Level: {curriculum_config.TASK_DIFFICULTY[current_tasks[0]]}")

        elif TRAINING_MODE == "PROGRESSIVE":
            # Use predefined curriculum stages
            current_tasks = curriculum_config.CURRICULUM_STAGES_AUG[CURRICULUM_STAGE]

            if DEBUG:
                printer.print_curriculum_info(
                    stage=CURRICULUM_STAGE,
                    total_stages=len(curriculum_config.CURRICULUM_STAGES),
                    tasks=current_tasks,
                    stage_thresholds=curriculum_config.STAGE_THRESHOLDS
                )

        else:  # MIXED
            # Start with easier tasks, gradually add harder ones
            current_tasks = MT10_TASKS[:3 + CURRICULUM_STAGE]

            if DEBUG:
                printer.print_section(f"Mixed Training - Stage {CURRICULUM_STAGE + 1}")
                print(f"Training with {len(current_tasks)} tasks")
                for i, task in enumerate(current_tasks, 1):
                    difficulty = curriculum_config.TASK_DIFFICULTY.get(task, "Unknown")
                    print(f"  {i}. {task} (difficulty: {difficulty})")

    else:
        raise ValueError(f"Unknown EXPERIMENT type: {EXPERIMENT}. "
                         f"Choose from 'MT1', 'MT3', 'MT10', or include 'CURRICULUM' in name")

    # -------------------- Create Environments --------------------

    if DEBUG:
        print(f"\nCreating environments for {len(current_tasks)} task(s)...")

    try:
        # Create training and evaluation environments using factory
        train_env, eval_env = env_factory.create_train_eval_pair(
            tasks=current_tasks,
            train_seed=SEED,
            eval_seed=SEED + 1000,
            max_episode_steps=MAX_EPISODE_STEPS,
            n_parallel_envs=N_PARALLEL_ENVS  # Only used for MT1
        )

        if DEBUG:
            printer.print_success(f"Train environment created: {type(train_env).__name__}")
            printer.print_success(f"Eval environment created: {type(eval_env).__name__}")
            # print(f"  Train env observation space: {train_env.observation_space}")
            # print(f"  Train env action space: {train_env.action_space}")

        # Apply one-hot task encoding wrapper
        train_env = OneHotTaskWrapper(train_env, current_tasks, MAX_TASKS)
        eval_env = OneHotTaskWrapper(eval_env, current_tasks, MAX_TASKS)

        train_env = VecNormalize(
            venv=train_env,
            norm_obs=True,
            norm_reward=True,)

        eval_env = VecNormalize(
            venv=eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False)

        if DEBUG:
            printer.print_success("OneHotTaskWrapper applied")
            # print(f"  New observation space: {train_env.observation_space}")

        num_envs = getattr(train_env, "num_envs", 1)

        if DEBUG:
            printer.print_success(f"Created {num_envs} parallel environment(s)")

            # Test environment reset
            print("\nTesting environment reset...")
            obs = train_env.reset()
            printer.print_success(f"Reset successful, observation shape: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")

    except Exception as e:
        if DEBUG:
            printer.print_error("Failed to create environments", exception=e)
        raise
    # -------------------- Initialize or Load Model --------------------

    if DEBUG:
        print(f"\nInitializing {ALGORITHM} agent...")

    model = None

    if USE_TRANSFER_LEARNING and PRETRAINED_MODEL_PATH:
        # Load pretrained model for transfer learning
        algorithm_class = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}[ALGORITHM]

        model = transfer_manager.load_pretrained_model(
            algorithm_class,
            train_env,
            PRETRAINED_MODEL_PATH
        )

        if model:
            # Fine-tune for new tasks with reduced learning rate
            model = transfer_manager.fine_tune_for_new_tasks(
                model,
                current_tasks,
                learning_rate_multiplier=LEARN_RATE_MULTIPLIER
            )

            if DEBUG:
                printer.print_transfer_info(
                    pretrained_model=PRETRAINED_MODEL_PATH,
                    target_tasks=current_tasks,
                    lr_multiplier=LEARN_RATE_MULTIPLIER
                )
        else:
            # Fallback to new model if loading failed
            printer.print_warning("Failed to load pretrained model, creating new model instead")
            USE_TRANSFER_LEARNING = False
    try:
        if not USE_TRANSFER_LEARNING or not model:
            # Create new model from scratch
            if DEBUG:
                print(f"Creating new {ALGORITHM} model from scratch...")

            if ALGORITHM == "SAC":
                model = model_factory_SAC(train_env, ALGORITHM, SEED)
            elif ALGORITHM == "TD3":
                action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") \
                    else train_env.action_space
                n_actions = action_space.shape[0]
                model = model_factory_TD3(n_actions, train_env, ALGORITHM, SEED)
            elif ALGORITHM == "DDPG":
                action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") \
                    else train_env.action_space
                n_actions = action_space.shape[0]
                model = model_factory_DDPG(n_actions, train_env, ALGORITHM, SEED)
            else:
                raise ValueError(f"Unknown algorithm: {ALGORITHM}. Choose from 'SAC', 'TD3', or 'DDPG'")

        if model is None:
            raise RuntimeError("Failed to create or load model")

        if DEBUG:
            printer.print_success(f"Model initialized successfully")
            print(f"  Model type: {type(model).__name__}")
            print(f"  Policy type: {type(model.policy).__name__}")

    except Exception as e:
        printer.print_error("Failed to initialize model", exception=e)
        raise
    # -------------------- Setup Callbacks --------------------

    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{EXPERIMENT}/",
        name_prefix=f"{ALGORITHM.lower()}_stage{CURRICULUM_STAGE}",
        verbose=1
    )

    # Evaluation callback - evaluates model and saves best version
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

    # Add curriculum callback for progressive training
    if USE_CURRICULUM and TRAINING_MODE == "PROGRESSIVE":
        # Create task evaluator for curriculum callback
        task_evaluator = TaskEvaluator(
            env_factory=env_factory,
            n_eval_episodes=N_EVAL_EPISODES,
            max_episode_steps=MAX_EPISODE_STEPS,
            seed=SEED + 2000,  # Different seed for curriculum eval
            verbose=DEBUG
        )

        curriculum_callback = ProgressiveTaskCallback(
            curriculum_stages=curriculum_config.CURRICULUM_STAGES_AUG,
            stage_thresholds=curriculum_config.STAGE_THRESHOLDS,
            task_evaluator=task_evaluator,
            current_stage=CURRICULUM_STAGE,
            eval_freq=STAGE_EVAL_FREQ,
            min_steps_per_stage=MIN_STEPS_PER_STAGE,
            verbose=1,
            one_hot_dim=MAX_TASKS,
            env_factory=env_factory,
            eval_callback=eval_callback
        )
        callbacks.append(curriculum_callback)

        if DEBUG:
            printer.print_success(f"Progressive curriculum callback enabled")

    # -------------------- Print Training Configuration --------------------

    if DEBUG:
        action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") \
            else train_env.action_space

        printer.print_training_start(
            model=model,
            task_name=EXPERIMENT,
            algorithm=ALGORITHM,
            time_steps=int(TOTAL_TIMESTEPS),
            seed=SEED,
            max_eps_steps=MAX_EPISODE_STEPS,
            norm_reward=NORMALIZE_REWARD,
            eval_freq=EVAL_FREQ,
            n_eval_eps=N_EVAL_EPISODES,
            checkpoint_freq=CHECKPOINT_FREQ,
            num_envs=num_envs,
            action_space=action_space,
            current_tasks=current_tasks
        )

    # -------------------- Start Training --------------------

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        printer.print_warning("Training interrupted by user")
    except Exception as e:
        printer.print_error("Training failed", exception=e)
        import traceback
        traceback.print_exc()
        raise

    # -------------------- Save Final Model --------------------

    if DEBUG:
        printer.print_header("SAVING FINAL MODEL")

    try:
        # Save model and replay buffer
        model.save(paths_dict["model"])

        if hasattr(model, 'save_replay_buffer'):
            model.save_replay_buffer(paths_dict["buffer"])
            if DEBUG:
                printer.print_success(f"Replay buffer saved to: {paths_dict['buffer']}")

        if DEBUG:
            printer.print_success(f"Model saved to: {paths_dict['model']}.zip")

        # Save transfer checkpoint for future use
        transfer_checkpoint_path = transfer_manager.save_transfer_checkpoint(
            model,
            stage=CURRICULUM_STAGE,
            tasks=current_tasks,
            save_path="./metaworld_models/transfer_checkpoints"
        )

        if DEBUG:
            printer.print_training_finished(
                task_name=EXPERIMENT,
                algorithm=ALGORITHM,
                final_model_path=f"{paths_dict['model']}.zip",
                best_model_path=f"./metaworld_models/best_{EXPERIMENT}/best_model.zip",
                checkpoint_path=f"./metaworld_models/checkpoints_{EXPERIMENT}/",
                transfer_checkpoint_path=transfer_checkpoint_path
            )

        # Print next steps for curriculum progression
        if USE_CURRICULUM and CURRICULUM_STAGE < len(curriculum_config.CURRICULUM_STAGES) - 1:
            next_stage = CURRICULUM_STAGE + 1
            next_tasks = curriculum_config.CURRICULUM_STAGES[next_stage]

            printer.print_curriculum_stage(next_stage, transfer_checkpoint_path, next_tasks)

    except Exception as e:
        printer.print_error("Failed to save model", exception=e)
        raise

    if DEBUG:
        printer.print_success("Training completed successfully!")

    # -------------------- Evaluation --------------------
    if RUN_FINAL_EVAL:
        if DEBUG:
            printer.print_header("RUNNING FINAL COMPREHENSIVE EVALUATION")

        try:
            # Create final evaluator
            final_evaluator = FinalModelEvaluator(
                env_factory=env_factory,
                tasks=current_tasks,
                n_eval_episodes=FINAL_EVAL_EPISODES,
                max_episode_steps=MAX_EPISODE_STEPS,
                seed=SEED + 3000,  # Different seed for final eval
                one_hot_dim=MAX_TASKS,
                save_results=True,
                results_dir="./evaluation_results",
                verbose=DEBUG
            )

            # Evaluate the final model
            final_results = final_evaluator.evaluate_model(
                model_path=f"{paths_dict['model']}.zip",
                algorithm=ALGORITHM,
                experiment_name=f"{EXPERIMENT}_stage{CURRICULUM_STAGE}_final"
            )

            if DEBUG:
                printer.print_success(f"Final evaluation completed")
                printer.print_success(f"Results saved to: ./evaluation_results/")

        except Exception as e:
            printer.print_error("Final evaluation failed", exception=e)

    # -------------------- Cleanup --------------------
    train_env.close()
    eval_env.close()


