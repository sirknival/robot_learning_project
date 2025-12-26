from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC, TD3, DDPG, PPO
import numpy as np
import torch
import os


def model_factory_TD3(n_actions, env, algorithm: str, seed: int):
    # TD3 - Optimized for Meta-World manipulation tasks
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=10_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=-1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.1,
        target_noise_clip=0.3,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=f"./metaworld_logs/{algorithm}/",
        verbose=1,
        device="auto",
        seed=seed,
    )
    return model


def model_factory_DDPG(n_actions, env, algorithm: str, seed : int ):
    # DDPG - Recommended for Meta-World (better exploration)
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)  # Reduced noise for fine manipulation
    )
    model = DDPG(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,  # Lower LR for stability
        buffer_size=1_000_000,
        learning_starts=5000,  # Start training sooner with parallel envs
        batch_size=256,
        tau=0.005,
        gamma=0.99,  # Higher gamma for longer horizon tasks
        train_freq=(1, "step"),
        gradient_steps=-1,  # Train on all available data at each step
        action_noise=action_noise,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],  # Deeper network for complex policies
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=f"./metaworld_logs/{algorithm}/",
        verbose=1,
        device="auto",
        seed=seed,
    )
    return model


def model_factory_SAC(env, algorithm: str, seed : int, train_mode: bool, paths):
    # SAC - Recommended for Meta-World (better exploration)
    if not train_mode:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=0,  # Start training sooner
            batch_size=500,
            tau=0.005,
            gamma=0.99,  # Higher gamma for multi-step tasks
            train_freq=1,
            gradient_steps=1,  # Train on all available data
            ent_coef='auto',  # Automatic entropy tuning - crucial for SAC
            target_entropy='auto',  # Automatically set target entropy
            use_sde=False,  # State-dependent exploration (can be enabled for more exploration)
            policy_kwargs=dict(
                net_arch=[400, 400],  # Deeper network
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,  # Initial exploration level
            ),
            tensorboard_log=f"./metaworld_logs/{algorithm}/",
            verbose=1,
            device="auto",
            seed=seed,
        )
    else:
        # Continue with training in case modell and replay-buffer is available
        if not os.path.exists(paths['first']['model'] + ".zip"):
            raise FileNotFoundError(
                f"Can't find model in order to continue training: {paths['first']['model']}.zip"
            )
        if not os.path.exists(paths['first']['buffer']):
            raise FileNotFoundError(
                f"Can't find replay buffer in order to continue training: {paths['first']['buffer']}"
            )

        print(f"Loading model from {paths['first']['model']}.zip ...")
        model = SAC.load(paths['first']['model'] + ".zip", env=env)

        print(f"Loading replay buffer from {paths['first']['buffer']} ...")
        model.load_replay_buffer(paths['first']['buffer'])

    return model

