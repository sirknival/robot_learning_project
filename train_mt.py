
"""
Meta-World MT1/MT3 Training Script with Stable Baselines3

- Unterstützt MT3/MT10 Multi-Task Training (reach / push / pick-place / ...)
- Verwendet gymnasium.make('Meta-World/MT1', ...)
- Nutzt Reward-Funktion v3 (Standard Meta-World)
- Task-ID One-Hot Conditioning (Policy weiß, welche Task aktiv ist)
- VecNormalize für Observation- und Reward-Normalisierung
- Optional: Checkpointing, Evaluation, Replay-Buffer-Save

Für Doku siehe: METAWORLD_README.md
"""

import os
import warnings

import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import TD3, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize


# === Task-Sets ===
MT3 = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
]

MT10 = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]


# === Task-ID Mapping für Conditioning ===
TASK_TO_ID = {
    "reach-v3": 0,
    "push-v3": 1,
    "pick-place-v3": 2,
    # Wenn du MT10 verwendest, hier erweitern:
    # "door-open-v3": 3,
    # ...
}
N_TASKS = len(TASK_TO_ID)


class TaskIdObsWrapper(gym.ObservationWrapper):
    """
    Fügt der Observation eine One-Hot Task-ID hinzu.
    Dadurch weiß die Policy explizit, welche Task sie gerade lösen soll.
    """
    def __init__(self, env, task_id, n_tasks):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks

        orig = env.observation_space #ursprüngliche Observation Space der inneren Env
        #Bei Meta-World ist das normalerweise ein gym.spaces.Box (ein Vektor aus floats mit min/max Grenzen).
        assert isinstance(orig, gym.spaces.Box) #check: ob Observations ein “Box”-Space ist.

        #obere und untere Grenze des observation space erweitern -> um n_tasks ids
        low = np.concatenate([orig.low, np.zeros(n_tasks, dtype=np.float32)])
        high = np.concatenate([orig.high, np.ones(n_tasks, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(self.n_tasks, dtype=np.float32)
        vec[self.task_id] = 1.0
        return np.concatenate([obs, vec], axis=-1)
    #für MT3, sieht die Policy immer:
      #Reach: obs_extended = [orig_obs..., 1,0,0]
      #Push: obs_extended = [orig_obs..., 0,1,0] 
      #PickPlace:obs_extended = [orig_obs..., 0,0,1]

def make_env(task_name='reach-v3', rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    """
    Create and wrap the Meta-World MT1 environment.

    - reward_function_version='v3'
    - Monitor für Episodenstatistiken
    - TaskIdObsWrapper: One-Hot Task-ID wird an Observation angehängt
    """
    def _init():
        # Create Meta-World MT1 environment
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,  # Different seed for each env
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        )

        # Wie im MT1-Code: pro Env optional reward normalisieren
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)

        # Monitor wrapper for logging episode statistics
        env = Monitor(env)

        # Task-ID anhängen
        if task_name not in TASK_TO_ID:
            raise ValueError(f"Task {task_name} nicht in TASK_TO_ID mapping definiert!")
        task_id = TASK_TO_ID[task_name]
        env = TaskIdObsWrapper(env, task_id, N_TASKS)

        return env

    return _init


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task Selection
    MODEL_BASENAME = "MT3"  # savename
    MT_TASKS = MT3          # MT3 or MT10

    # Algorithm Selection
    ALGORITHM = "SAC"  # "TD3" or "DDPG" or "SAC"

    # Environment Settings
    SEED = 42

    # Training Settings
    TOTAL_TIMESTEPS = 2_000_000
    MAX_EPISODE_STEPS = 200
    NORMALIZE_REWARD = True

    # Evaluation Settings
    EVAL_FREQ = 10_000
    N_EVAL_EPISODES = 20
    CHECKPOINT_FREQ = 25_000
    # ======================================================

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World Multi-Task Training: {MODEL_BASENAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"Tasks: {MT_TASKS}")
    print(f"=" * 60)

    # ==================== ENVIRONMENTS ====================

    print("Creating vectorized training environment (DummyVecEnv + VecNormalize)...")
    # Multi-Task: eine Env pro Task
    #train_env = DummyVecEnv(
    #    [make_env(task_name, seed=SEED + i, max_episode_steps=MAX_EPISODE_STEPS)
    #     for i, task_name in enumerate(MT_TASKS)]
    #)

    train_env = DummyVecEnv(
        [make_env(task_name, rank=i, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=NORMALIZE_REWARD,)
         for i, task_name in enumerate(MT_TASKS)]
    )

    # Optional: VecMonitor (nicht zwingend notwendig, da VecNormalize auch loggt,
    # aber schadet nicht – hier können wir es weglassen oder einkommentieren)
    # train_env = VecMonitor(train_env)

    #if NORMALIZE_REWARD:
    #    train_env = VecNormalize(
    #        train_env,
    #        norm_obs=True,
    #        norm_reward=True,
    #        clip_obs=10.0,
    #        clip_reward=50.0, #10.0,
    #    )

    env = train_env

    print("Creating evaluation environment (DummyVecEnv + VecNormalize for obs only)...")
    #eval_env_raw = DummyVecEnv(
    #    [make_env(task_name, seed=SEED + 1000 + i, max_episode_steps=MAX_EPISODE_STEPS)
    #     for i, task_name in enumerate(MT_TASKS)]
    #)

    eval_env = DummyVecEnv(
        [make_env(task_name, rank=i, seed=SEED + 1000, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=False,)  # Eval ohne Reward-Normalisierung
         for i, task_name in enumerate(MT_TASKS)]
    )

    #if NORMALIZE_REWARD:
    #    eval_env = VecNormalize(
    #        eval_env_raw,
    #        training=False,
    #        norm_obs=True,
    #        norm_reward=False,  # Reward im Eval nicht normalisieren
    #    )
        # gleiche Obs-Statistiken wie Training verwenden
    #    eval_env.obs_rms = train_env.obs_rms
    #else:
    #    eval_env = eval_env_raw

    # Get action space dimensions
    n_actions = env.action_space.shape[0]

    # ==================== MODEL ====================

    print(f"\nInitializing {ALGORITHM} agent...")

    if ALGORITHM == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=10_000,
            learning_starts=5_000,
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
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )

    elif ALGORITHM == "DDPG":
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=-1,
            action_noise=action_noise,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )

    elif ALGORITHM == "SAC":
        # SAC - mit gradient_steps=2
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,#3e-4,
            buffer_size=1_500_000,#1_000_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=-1,  # <--- WICHTIG: mehr Updates pro Schritt
            ent_coef='auto',
            target_entropy='auto',
            use_sde=False,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],#net_arch=[400, 400],  # wie in deinem ursprünglichen Code
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # ==================== CALLBACKS ====================

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{MODEL_BASENAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{MODEL_BASENAME}",
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{MODEL_BASENAME}/",
        log_path=f"./metaworld_logs/eval_{MODEL_BASENAME}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    # ==================== TRAINING INFO ====================

    total_timesteps = TOTAL_TIMESTEPS
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    print("Training configuration:")
    print(f"  - Tasks: {MODEL_BASENAME} -> {MT_TASKS}")
    print(f"  - Algorithm: {ALGORITHM}")
    print(f"  - Parallel environments: {env.num_envs}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma: {model.gamma}")
    print(f"  - Learning starts: {model.learning_starts}")
    print(f"  - Buffer size: {model.buffer_size:,}")
    print(f"  - Network architecture: [256, 256, 256]")
    print(f"  - Gradient steps: {model.gradient_steps}")
    print(f"  - Seed: {SEED}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  - Reward function: v3")
    print(f"  - Normalize reward (NormalizeReward wrapper): {NORMALIZE_REWARD}")
    print(f"  - Eval frequency: {EVAL_FREQ} steps")
    print(f"  - Eval episodes: {N_EVAL_EPISODES}")
    print(f"  - Checkpoint frequency: {CHECKPOINT_FREQ} steps")
    if ALGORITHM == "TD3":
        print(f"  - Exploration noise: σ=0.1")
        print(f"  - Target policy noise: 0.1 (clip: 0.3)")
    elif ALGORITHM == "SAC":
        print(f"  - Entropy coef: {model.ent_coef}")
        print(f"  - Target entropy: {model.target_entropy}")
    print("=" * 60)

    # ==================== TRAINING ====================

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # ==================== SAVE MODEL & NORMALIZER ====================

    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{MODEL_BASENAME}_final")
    model.save_replay_buffer(f"./metaworld_models/{ALGORITHM.lower()}_{MODEL_BASENAME}_replay_buffer_final")

    # VecNormalize-Statistiken speichern (wichtig für späteres Eval / Fine-Tuning)
    #if NORMALIZE_REWARD and isinstance(env, VecNormalize):
    #    env.save(f"./metaworld_models/vecnormalize_{MODEL_BASENAME}.pkl")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{MODEL_BASENAME}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{MODEL_BASENAME}/best_model.zip")
    print(f"Replay buffer saved to: ./metaworld_models/{ALGORITHM.lower()}_{MODEL_BASENAME}_replay_buffer_final")
    print(f"VecNormalize stats saved to: ./metaworld_models/vecnormalize_{MODEL_BASENAME}.pkl")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{MODEL_BASENAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()
