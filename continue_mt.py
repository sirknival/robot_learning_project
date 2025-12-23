
"""
Meta-World MT3 Continue-Training Script (SAC, Multi-Task)

- Lädt ein existierendes MT3-Modell (z.B. aus train_mt.py)
- Rekonstruiert dieselben Envs (DummyVecEnv + NormalizeReward + TaskIdObsWrapper)
- Führt weiteres Training für ADDITIONAL_TIMESTEPS durch
- Speichert neues "continued"-Modell + Replay-Buffer

WICHTIG:
- TASK-SET, TASK_TO_ID, MAX_EPISODE_STEPS, NORMALIZE_REWARD
  müssen zu deinem ursprünglichen train_mt.py passen.
"""

import os
import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# ==================== TASK LISTS ====================

MT3 = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
]

# Falls du später MT10 weitertrainieren willst:
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

# === Task-ID Mapping (muss identisch zu train_mt.py sein!) ===
TASK_TO_ID = {
    "reach-v3": 0,
    "push-v3": 1,
    "pick-place-v3": 2,
    # ggf. erweitern, falls MT10
}
N_TASKS = len(TASK_TO_ID)


class TaskIdObsWrapper(gym.ObservationWrapper):
    """
    Fügt der Observation eine One-Hot Task-ID hinzu.
    Muss 1:1 so sein wie im Training, sonst passt die Policy nicht mehr.
    """
    def __init__(self, env, task_id, n_tasks):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks

        orig = env.observation_space
        assert isinstance(orig, gym.spaces.Box)

        low = np.concatenate([orig.low, np.zeros(n_tasks, dtype=np.float32)])
        high = np.concatenate([orig.high, np.ones(n_tasks, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(self.n_tasks, dtype=np.float32)
        vec[self.task_id] = 1.0
        return np.concatenate([obs, vec], axis=-1)


def make_env(task_name='reach-v3', rank=0, seed=0,
             max_episode_steps=200, normalize_reward=True):
    """
    Erzeugt eine einzelne MT1-Umgebung mit:
    - reward_function_version='v3'
    - optional NormalizeReward
    - Monitor
    - TaskIdObsWrapper
    """
    def _init():
        if task_name not in TASK_TO_ID:
            raise ValueError(f"Task {task_name} nicht in TASK_TO_ID definiert!")

        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=seed + rank,
            reward_function_version="v3",
            max_episode_steps=max_episode_steps,
            terminate_on_success=False,
        )

        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)

        env = Monitor(env)

        task_id = TASK_TO_ID[task_name]
        env = TaskIdObsWrapper(env, task_id, N_TASKS)

        return env

    return _init


if __name__ == "__main__":
    # ==================== CONFIG (ANPASSBAR) ====================

    MODEL_BASENAME = "MT3"       # muss zu deinem ursprünglichen Run passen
    MT_TASKS = MT3               # gleiche Task-Liste wie beim Training
    ALGORITHM = "SAC"            # hier: SAC

    SEED = 42
    MAX_EPISODE_STEPS = 200      # wie im Training
    NORMALIZE_REWARD = True      # wie im Training

    # Wie viele zusätzliche Schritte du trainieren willst
    ADDITIONAL_TIMESTEPS = 4_000_000

    # Welches Modell laden?
    LOAD_BEST_MODEL = False       # True = best_MT3/best_model.zip, False = sac_MT3_final.zip

    # Pfade
    MODELS_DIR = "./metaworld_models"
    LOGS_DIR = "./metaworld_logs"

    best_model_path = f"{MODELS_DIR}/best_{MODEL_BASENAME}/best_model.zip"
    final_model_path = f"{MODELS_DIR}/sac_{MODEL_BASENAME}_final.zip"

    # Replay-Buffer (optional)
    REPLAY_BUFFER_PATH = f"{MODELS_DIR}/sac_{MODEL_BASENAME}_replay_buffer_final.pkl"

    # Neue Speichernamen
    CONT_SUFFIX = "continued5"
    CONT_MODEL_PATH = f"{MODELS_DIR}/sac_{MODEL_BASENAME}_{CONT_SUFFIX}"
    CONT_REPLAY_PATH = f"{MODELS_DIR}/sac_{MODEL_BASENAME}_replay_buffer_{CONT_SUFFIX}"

    # Eval-Settings
    EVAL_FREQ = 10_000
    N_EVAL_EPISODES = 20
    CHECKPOINT_FREQ = 25_000

    # ==================== MODELL LADEN ====================

    print("=" * 60)
    print(f"Weitertraining: {MODEL_BASENAME} ({ALGORITHM})")
    print(f"Zusätzliche Timesteps: {ADDITIONAL_TIMESTEPS:,}")
    print("=" * 60)

    if LOAD_BEST_MODEL and os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Lade BEST model: {model_path}")
    elif os.path.exists(final_model_path):
        model_path = final_model_path
        print(f"Best model nicht gefunden, lade FINAL model: {model_path}")
    else:
        raise FileNotFoundError(
            f"Weder best model ({best_model_path}) noch final model ({final_model_path}) gefunden."
        )

    if ALGORITHM == "SAC":
        model = SAC.load(model_path)
    elif ALGORITHM == "TD3":
        model = TD3.load(model_path)
    elif ALGORITHM == "DDPG":
        model = DDPG.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Optional: Replay-Buffer laden (wenn vorhanden)
    if os.path.exists(REPLAY_BUFFER_PATH):
        try:
            model.load_replay_buffer(REPLAY_BUFFER_PATH)
            print(f"Replay-Buffer geladen von: {REPLAY_BUFFER_PATH}")
        except Exception as e:
            print(f"[WARNUNG] Replay-Buffer konnte nicht geladen werden: {e}")
    else:
        print("Kein Replay-Buffer gefunden, starte mit leerem Buffer weiter.")

    # ==================== ENVIRONMENTS REKONSTRUIEREN ====================

    print("\nErzeuge Trainings-VecEnv (DummyVecEnv) mit denselben Tasks wie im ursprünglichen Training...")
    train_env = DummyVecEnv(
        [
            make_env(
                task_name,
                rank=i,
                seed=SEED,
                max_episode_steps=MAX_EPISODE_STEPS,
                normalize_reward=NORMALIZE_REWARD,
            )
            for i, task_name in enumerate(MT_TASKS)
        ]
    )

    print("Erzeuge Evaluations-VecEnv...")
    eval_env = DummyVecEnv(
        [
            make_env(
                task_name,
                rank=i,
                seed=SEED + 1000,
                max_episode_steps=MAX_EPISODE_STEPS,
                normalize_reward=False,  # Eval ohne Reward-Normalisierung
            )
            for i, task_name in enumerate(MT_TASKS)
        ]
    )

    # Env mit dem geladenen Modell verbinden
    model.set_env(train_env)

    # ==================== CALLBACKS ====================

    os.makedirs(f"{MODELS_DIR}/checkpoints_{MODEL_BASENAME}", exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"{MODELS_DIR}/checkpoints_{MODEL_BASENAME}/",
        name_prefix=f"sac_{MODEL_BASENAME}_{CONT_SUFFIX}",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODELS_DIR}/best_{MODEL_BASENAME}_{CONT_SUFFIX}/",
        log_path=f"{LOGS_DIR}/eval_{MODEL_BASENAME}_{CONT_SUFFIX}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False,
    )

    # ==================== INFO ====================

    print("\nStarte WEITER-Training...")
    print("=" * 60)
    print(f"  - Algorithmus: {ALGORITHM}")
    print(f"  - Tasks: {MT_TASKS}")
    print(f"  - VecEnv num_envs: {train_env.num_envs}")
    print(f"  - Additional timesteps: {ADDITIONAL_TIMESTEPS:,}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  - NormalizeReward (pro Env): {NORMALIZE_REWARD}")
    print("=" * 60)

    # ==================== WEITER-TRAINING ====================

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    # ==================== SPEICHERN ====================

    print("\nSpeichere weitertrainiertes Modell...")
    model.save(CONT_MODEL_PATH)
    model.save_replay_buffer(CONT_REPLAY_PATH)

    print("\n" + "=" * 60)
    print("Weitertraining abgeschlossen!")
    print(f"Continued model gespeichert unter: {CONT_MODEL_PATH}.zip")
    print(f"Replay-Buffer gespeichert unter: {CONT_REPLAY_PATH}.pkl")
    print("=" * 60)

    train_env.close()
    eval_env.close()
