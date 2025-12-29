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

from training_setup_multitask.utilities.MetaworldTasks import MT10_TASKS
from training_setup_multitask.utilities.DebugPrinter import DebugPrinter
from training_setup_multitask.utilities.CurriculumConfig import CurriculumConfig
from training_setup_multitask.utilities.TransferLearningManager import TransferLearningManager
from training_setup_multitask.utilities.MetaWorldEnvFactory import MetaWorldEnvFactory
from training_setup_multitask.utilities.MetaWorldEvaluator import MetaWorldEvaluator
from training_setup_multitask.utilities.algorithms import model_factory_SAC, model_factory_TD3, model_factory_DDPG
from training_setup_multitask.WrapperClasses.OneHotTaskWrapper import OneHotTaskWrapper
from training_setup_multitask.Callbacks.ProgressiveTaskCallback import ProgressiveTaskCallback


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================

    # -------------------- Training Strategy --------------------
    TRAINING_MODE = "SEQUENTIAL"  # Options: "SEQUENTIAL", "PROGRESSIVE", "MIXED"
    USE_TRANSFER_LEARNING = True  # Use pretrained model as starting point
    USE_CURRICULUM = True          # Enable curriculum learning with automatic stage transitions
    PRETRAINED_MODEL_PATH = './metaworld_models/MT10k_SAC_5M'   # Path to pretrained model (e.g., "<./metaworld_models/MT10_SAC_5M.zip")
    USE_SUBPROC_VEC_ENV = True     # Use SubprocVecEnv (True) for faster multi-processing or DummyVecEnv (False)

    # -------------------- Experiment Setup --------------------
    EXPERIMENT = "MT10"             # Options: "MT1", "MT3", "MT10", "MT10_CURRICULUM"
    TASK_NAME = "reach-v3"          # Required for MT1 (e.g., "reach-v3", "push-v3")
    ALGORITHM = "SAC"               # Options: "SAC", "TD3", "DDPG" (SAC recommended)
    SEED = 42                       # Random seed for reproducibility
    N_PARALLEL_ENVS = 1             # Number of parallel environments (1-10, higher = faster training)
    MAX_TASKS_IN_PROJECT = 10       # Für MT10 Projekt; Do not touch!
    TOTAL_TIMESTEPS = 5 * 1e4       # Number of total time-steps while training

    # -------------------- Curriculum Settings --------------------
    CURRICULUM_STAGE = 1           # Starting curriculum stage (0 = easiest tasks)
    MIN_STEPS_PER_STAGE = 2000 # 200000   # Minimum training steps before stage transition
    STAGE_EVAL_FREQ = 1000        # Evaluate performance every N steps for stage transitions

    # -------------------- Environment Settings --------------------
    MAX_EPISODE_STEPS = 500        # Maximum steps per episode
    NORMALIZE_REWARD = False       # Normalize rewards (enable if training is unstable)

    # -------------------- Evaluation & Checkpointing --------------------
    EVAL_FREQ = 1000              # Evaluate model every N steps
    N_EVAL_EPISODES = 20           # Number of episodes for evaluation
    CHECKPOINT_FREQ = 50000        # Save checkpoint every N steps

    # -------------------- Debug Settings --------------------
    DEBUG = True                   # Enable verbose debug output

    # ======================================================

    # -------------------- Setup Paths --------------------
    MODEL_BASENAME = f"{EXPERIMENT}_{ALGORITHM}"
    # ToDo remove hardcoding 1e4, dev Pur
    paths_dict = {
            "model": f"./metaworld_models/{MODEL_BASENAME}_{TOTAL_TIMESTEPS / 1e4}M",
            "buffer": f"./metaworld_models/{MODEL_BASENAME}_{TOTAL_TIMESTEPS / 1e4}M_replay.pkl"
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
        current_tasks = env_factory.MT3_TASKS

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
            current_tasks = curriculum_config.CURRICULUM_STAGES[CURRICULUM_STAGE]

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
        raise ValueError(f"Unknown EXPERIMENT type: {EXPERIMENT}. Choose from 'MT1', 'MT3', 'MT10', or include 'CURRICULUM' in name")

    # -------------------- Create Environments --------------------

    if DEBUG:
        print(f"\nCreating environments for {len(current_tasks)} task(s)...")

    # Create training and evaluation environments using factory
    train_env, eval_env = env_factory.create_train_eval_pair(
        tasks=current_tasks,
        train_seed=SEED,
        eval_seed=SEED + 1000,
        max_episode_steps=MAX_EPISODE_STEPS,
        n_parallel_envs=N_PARALLEL_ENVS  # Only used for MT1
    )

    # Apply one-hot task encoding wrapper
    train_env = OneHotTaskWrapper(train_env, current_tasks, MAX_TASKS_IN_PROJECT)
    eval_env = OneHotTaskWrapper(eval_env, current_tasks, MAX_TASKS_IN_PROJECT)

    num_envs = getattr(train_env, "num_envs", 1)

    if DEBUG:
        print(f"✓ Created {num_envs} parallel environment(s)")

    # -------------------- Initialize or Load Model --------------------

    if DEBUG:
        print(f"\nInitializing {ALGORITHM} agent...")

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
                learning_rate_multiplier=0.3
            )

            if DEBUG:
                printer.print_transfer_info(
                    pretrained_model=PRETRAINED_MODEL_PATH,
                    target_tasks=current_tasks,
                    lr_multiplier=0.3
                )
        else:
            # Fallback to new model if loading failed
            printer.print_warning("Failed to load pretrained model, creating new model instead")
            USE_TRANSFER_LEARNING = False

    if not USE_TRANSFER_LEARNING or not model:
        # Create new model from scratch
        if ALGORITHM == "SAC":
            model = model_factory_SAC(train_env, ALGORITHM, SEED)
        elif ALGORITHM == "TD3":
            action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") else train_env.action_space
            n_actions = action_space.shape[0]
            model = model_factory_TD3(n_actions, train_env, ALGORITHM, SEED)
        elif ALGORITHM == "DDPG":
            action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") else train_env.action_space
            n_actions = action_space.shape[0]
            model = model_factory_DDPG(n_actions, train_env, ALGORITHM, SEED)
        else:
            raise ValueError(f"Unknown algorithm: {ALGORITHM}. Choose from 'SAC', 'TD3', or 'DDPG'")

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
        curriculum_callback = ProgressiveTaskCallback(
            curriculum_stages=curriculum_config.CURRICULUM_STAGES,
            stage_thresholds=curriculum_config.STAGE_THRESHOLDS,
            eval_freq=STAGE_EVAL_FREQ,
            min_steps_per_stage=MIN_STEPS_PER_STAGE,
            verbose=1
        )
        callbacks.append(curriculum_callback)

        if DEBUG:
            print("\n✓ Progressive curriculum callback enabled")

    # -------------------- Print Training Configuration --------------------

    if DEBUG:
        action_space = train_env.single_action_space if hasattr(train_env, "single_action_space") else train_env.action_space

        printer.print_training_start(
            model=model,
            task_name=EXPERIMENT,
            algorithm=ALGORITHM,
            time_steps=TOTAL_TIMESTEPS,
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
        raise

    # -------------------- Save Final Model --------------------

    if DEBUG:
        print("\n" + "=" * 70)
        print("SAVING FINAL MODEL")

    try:
        # Save model and replay buffer
        model.save(paths_dict["model"])

        if hasattr(model, 'save_replay_buffer'):
            model.save_replay_buffer(paths_dict["buffer"])
            if DEBUG:
                print(f"✓ Replay buffer saved to: {paths_dict['buffer']}")

        if DEBUG:
            print(f"✓ Model saved to: {paths_dict['model']}.zip")

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

            print("\n" + "=" * 70)
            print("NEXT CURRICULUM STAGE")
            print(f"To continue training with the next stage:")
            print(f"  1. Set CURRICULUM_STAGE = {next_stage}")
            print(f"  2. Set USE_TRANSFER_LEARNING = True")
            print(f"  3. Set PRETRAINED_MODEL_PATH = '{transfer_checkpoint_path}.zip'")
            print(f"\nNext stage will include {len(next_tasks)} tasks:")
            for task in next_tasks:
                print(f"  • {task}")
            print("=" * 70)

    except Exception as e:
        printer.print_error("Failed to save model", exception=e)
        raise

    # -------------------- Cleanup --------------------

    train_env.close()
    eval_env.close()

    final_evaluator = MetaWorldEvaluator(
        task_list=current_tasks,
        max_episode_steps=200
    )

    mean_reward, success_rate, details = final_evaluator.evaluate(
        model=model,
        num_episodes_per_task=20  # Mehr Episoden für Genauigkeit
    )

    print(f"Gesamt Success Rate: {success_rate * 100:.2f}%")
    # Details anzeigen, wenn gewünscht
    for task, stats in details.items():
         print(f"{task}: {stats['success']*100}%")

    if DEBUG:
        printer.print_success("Training completed successfully!")