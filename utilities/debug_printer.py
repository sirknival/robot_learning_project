import numpy as np


def print_start_setup(experiment: str, algorithm: str, train_mode: bool):
    print(f"=" * 60)
    print(f"Meta-World {experiment} Training")
    print(f"Algorithm: {algorithm}")
    print(f"=" * 60)
    print(
        "Mode: CONTINUE_TRAINING = "
        f"{train_mode}"
        f"({'Resume' if train_mode else 'Initial training'})"
    )
    return


def print_training_start(model, task_name: str, algorithm: str, time_steps: int, seed: int,  max_eps_steps: int,
                         norm_reward: bool, eval_freq: int, n_eval_eps: int, checkpoint_freq: int, train_phase: bool,
                         num_envs: int, action_space: np.array):
    # Train the agent
    print(f"\nStarting training for {time_steps} time steps...")
    print("=" * 60)
    print("Training configuration:")
    print(f" - Seed: {seed}")
    print(f" - Algorithm: {algorithm}")
    print(f" - Tasks: {task_name}")
    print(f" - Mode: {'Resume from previous training' if train_phase else 'Initial run'}")
    print(f" - Num_envs: {num_envs}")
    if hasattr(model, 'learning_rate'):
        print(f" - Learning rate: {model.learning_rate}")
    if hasattr(model, 'learning_starts'):
        print(f" - Learning starts: {model.learning_starts}")
    if hasattr(model, 'batch_size'):
        print(f" - Batch size: {model.batch_size}")
    if hasattr(model, 'buffer_size'):
        print(f" - Buffer size: {model.buffer_size}")
    if hasattr(model, 'gamma'):
        print(f" - Gamma: {model.gamma}")
    if hasattr(model, 'gradient_steps'):
        print(f" - Gradient steps: {model.gradient_steps}")
    if hasattr(model, 'policy_kwargs'):
        print(f" - Network architecture: {model.policy_kwargs['net_arch']}")
    """
    if algorithm == "TD3":
        print(f" - Exploration noise: σ=0.1")
        print(f" - Target policy noise: 0.1 (clip: 0.3)")
    """
    if algorithm == "SAC":
        print(f" - Entropy tuning: Automatic")
        print(f" - Target entropy: {-action_space.shape[0]}")

    print(f" - Max episode steps: {max_eps_steps}")
    print(f" - Normalize reward: {norm_reward}")
    print(f" - Eval frequency: {eval_freq} steps")
    print(f" - Eval episodes: {n_eval_eps}")
    print(f" - Checkpoint frequency: {checkpoint_freq} steps")
    print(f" - Reward function: v3 (more stable)")
    print("=" * 60)
    return


def print_training_finished(task_name: str, algorithm: str):
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{algorithm.lower()}_{task_name}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{task_name}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{task_name}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)
    return
