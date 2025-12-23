
"""
Meta-World MT3/MT10 Evaluation Script with Stable Baselines3

- Lädt ein trainiertes Multi-Task-Modell (z.B. MT3: reach/push/pick-place)
- Nutzt dieselbe Task-ID-One-Hot-Erweiterung wie im Training
- Eval ohne Reward-Normalisierung (echte Rewards)
"""

import os
import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3 import TD3, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==================== TASK LISTS ====================

# Für die Auswertung brauchen wir jede Task nur EINMAL:
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

# === Task-ID Mapping (muss zu train_mt.py passen!) ===
TASK_TO_ID = {
    "reach-v3": 0,
    "push-v3": 1,
    "pick-place-v3": 2,
    # Wenn du später MT10 nimmst: hier erweitern
    # "door-open-v3": 3,
    # ...
}
N_TASKS = len(TASK_TO_ID)


class TaskIdObsWrapper(gym.ObservationWrapper):
    """
    Fügt der Observation eine One-Hot Task-ID hinzu.
    Muss identisch zum Wrapper im Training sein.
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


def make_eval_env(task_name, seed, max_episode_steps, render_mode="rgb_array"):
    """
    Erzeugt eine einzelne MT1-Umgebung für einen Task mit:
    - reward_function_version='v3'
    - TaskIdObsWrapper
    (kein NormalizeReward, wir wollen echte Rewards im Eval)
    """
    if task_name not in TASK_TO_ID:
        raise ValueError(f"Task {task_name} nicht in TASK_TO_ID definiert!")

    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        seed=seed,
        render_mode=render_mode,
        reward_function_version="v3",
        max_episode_steps=max_episode_steps,
        terminate_on_success=False,
    )

    task_id = TASK_TO_ID[task_name]
    env = TaskIdObsWrapper(env, task_id, N_TASKS)
    return env


if __name__ == "__main__":
    # ==================== CONFIG ====================

    MODEL_BASENAME = "MT10"   # muss zu train_mt.py passen
    MT_TASKS = MT3           # MT3 oder MT10
    ALGORITHM = "SAC"        # "TD3", "DDPG" oder "SAC"
    SEED = 42
    MAX_EPISODE_STEPS = 200  # wie im Training
    NUM_EPISODES = 300

    # In deinem aktuellen train_mt.py speicherst du KEIN VecNormalize.
    # Deshalb hier standardmäßig: False
    USE_VECNORMALIZE = False
    VECNORM_PATH = f"./metaworld_models/vecnormalize_{MODEL_BASENAME}.pkl"

    # ==================== MODELL LADEN ====================

    print(f"Evaluating model '{MODEL_BASENAME}' with algorithm {ALGORITHM}")
    model_path = f"./metaworld_models/best_{MODEL_BASENAME}/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Best model not found at {model_path}")
        alt_path = f"./metaworld_models/{ALGORITHM.lower()}_{MODEL_BASENAME}_final.zip"
        print(f"Trying final model instead: {alt_path}")
        model_path = alt_path

        if not os.path.exists(model_path):
            print("Kein passendes Modell gefunden. Bitte zuerst trainieren.")
            raise SystemExit

    print(f"Loading model from: {model_path}")
    if ALGORITHM == "DDPG":
        model = DDPG.load(model_path)
    elif ALGORITHM == "TD3":
        model = TD3.load(model_path)
    elif ALGORITHM == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # ==================== STATISTIK-STRUCTS ====================

    total_rewards = []
    success_count = 0

    task_rewards = {name: [] for name in MT_TASKS}
    task_success = {name: [] for name in MT_TASKS}

    print(f"\nRunning {NUM_EPISODES} evaluation episodes across tasks {MT_TASKS}...")
    print("=" * 60)

    # ==================== EPISODISCHES EVAL ====================

    for episode in range(1, NUM_EPISODES + 1):
        task_name = MT_TASKS[(episode - 1) % len(MT_TASKS)]
        print(f"\n--- Episode {episode}/{NUM_EPISODES} ---")
        print(f"Task in this episode: {task_name}")

        # Basis-Umgebung mit Task-ID
        base_env = make_eval_env(
            task_name=task_name,
            seed=SEED + episode,
            max_episode_steps=MAX_EPISODE_STEPS,
            render_mode="rgb_array",
        )

        # Für SB3-Policies brauchen wir ein VecEnv (auch bei nur 1 Env)
        vec_env = DummyVecEnv([lambda: base_env])

        # Optional: VecNormalize-Statistiken laden (nur falls du später wieder VecNormalize nutzt)
        if USE_VECNORMALIZE and os.path.exists(VECNORM_PATH):
            vec_env = VecNormalize.load(VECNORM_PATH, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        elif USE_VECNORMALIZE:
            print(f"  [Warnung] VecNormalize-Datei {VECNORM_PATH} nicht gefunden, fahre ohne Normalisierung fort.")

        obs = vec_env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        episode_success = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]

            total_reward += reward
            steps += 1

            if "success" in info and info["success"]:
                episode_success = True

        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {episode_success}")

        total_rewards.append(total_reward)
        if episode_success:
            success_count += 1

        task_rewards[task_name].append(total_reward)
        task_success[task_name].append(1 if episode_success else 0)

        vec_env.close()

    # ==================== SUMMARY (GESAMT) ====================

    print("\n" + "=" * 60)
    print(f"=== {MODEL_BASENAME} Evaluation Complete ===")
    print(f"Episodes (total): {NUM_EPISODES}")
    print(f"Average reward (overall): {np.mean(total_rewards):.2f}")
    print(f"Std reward (overall): {np.std(total_rewards):.2f}")
    print(f"Min reward (overall): {np.min(total_rewards):.2f}")
    print(f"Max reward (overall): {np.max(total_rewards):.2f}")
    print(f"Success rate (overall): {success_count}/{NUM_EPISODES} "
          f"({100 * success_count / NUM_EPISODES:.1f}%)")
    print("=" * 60)

    # ==================== SUMMARY (PER TASK) ====================

    print("\nPer-task statistics:")
    for task_name in MT_TASKS:
        rewards = task_rewards[task_name]
        successes = task_success[task_name]
        if len(rewards) == 0:
            continue

        avg_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        min_r = float(np.min(rewards))
        max_r = float(np.max(rewards))
        succ_rate = 100.0 * (np.sum(successes) / len(successes))

        print(f"\n  Task: {task_name}")
        print(f"    Episodes: {len(rewards)}")
        print(f"    Avg reward: {avg_r:.2f}")
        print(f"    Std reward: {std_r:.2f}")
        print(f"    Min / Max: {min_r:.2f} / {max_r:.2f}")
        print(f"    Successes: {int(np.sum(successes))}/{len(successes)} ({succ_rate:.1f}%)")

    print("=" * 60)
