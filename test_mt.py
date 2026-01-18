import os
import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3 import TD3, SAC, DDPG, PPO
import imageio

from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


from train_mt import MT10_TASKS, MT3_TASKS  
from train_mt import MultiHeadSACPolicy  

# -------------------------------
# Eval Wrapper -> append task one-hot to obs
# -------------------------------
class AppendTaskIdWrapper(gym.Wrapper):
    """
    Appends a fixed task one-hot vector to every observation.
    This makes MT1 env observations compatible with MT training that expects task_id in obs.
    """
    def __init__(self, env, task_index: int, n_tasks: int):
        super().__init__(env)
        self.task_index = int(task_index)
        self.n_tasks = int(n_tasks)
        self.task_onehot = np.zeros((self.n_tasks,), dtype=np.float32)
        self.task_onehot[self.task_index] = 1.0

        # Expand observation space
        assert isinstance(env.observation_space, spaces.Box)
        low = env.observation_space.low
        high = env.observation_space.high
        low = np.concatenate([low, -np.ones(self.n_tasks, dtype=low.dtype)], axis=0)
        high = np.concatenate([high, np.ones(self.n_tasks, dtype=high.dtype)], axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.task_onehot], axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info


if __name__ == "__main__":
    ALGORITHM = "SAC"
    SEED = 42
    MT_TASKS = MT10_TASKS #task list

    MULTI_HEAD = False
    MT_N = "MT10" #MT3 or MT10

    NUM_EPISODES = 100
    VIDEO_EPISODES = []
    MAX_EPISODE_STEPS = 500

    # IMPORTANT: point to your MH critic trained model
    MODEL_CANDIDATES = [
        f"./metaworld_models/best_{MT_N}/best_model.zip",
        #f"./metaworld_models/SAC_{MT_N}_9M.zip",
        #f"./metaworld_models/checkpoints_{MT_N}/SAC_{MT_N}_28000000_steps.zip",
    ]

    model_path = None
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            model_path = candidate
            break
    if model_path is None:
        print("Kein Modell gefunden! Erwartet eines von:")
        for c in MODEL_CANDIDATES:
            print("  -", c)
        raise SystemExit(1)

    print(f"Loading model from: {model_path}")

    # Dummy env for load (must match obs dim = MT1 obs + task_onehot)
    dummy_task = "reach-v3"
    dummy_task_idx = MT_TASKS.index(dummy_task)

    dummy_env = gym.make(
        "Meta-World/MT1",
        env_name=dummy_task,
        seed=SEED,
        render_mode="rgb_array",
        reward_function_version="v3",
        max_episode_steps=MAX_EPISODE_STEPS,
        terminate_on_success=False,
    )
    dummy_env = AppendTaskIdWrapper(dummy_env, task_index=dummy_task_idx, n_tasks=len(MT_TASKS))
    dummy_env = Monitor(dummy_env)

    if ALGORITHM == "SAC":
        if not MULTI_HEAD:
            model = SAC.load(
                model_path,
            )
            
        else:
            model = SAC.load(
                model_path,
                custom_objects={"policy_class": MultiHeadSACPolicy},
            )
    else:
        raise ValueError("This eval script is for SAC + MultiHeadSACPolicy.")

    total_rewards = []
    success_count = 0
    task_rewards = {name: [] for name in MT_TASKS}
    task_success = {name: [] for name in MT_TASKS}
    video_frames = {ep: [] for ep in VIDEO_EPISODES}

    print(f"\nRunning {NUM_EPISODES} {MT_N} evaluation episodes...")
    print("=" * 60)

    global_episode_idx = 0
    for episode in range(1, NUM_EPISODES + 1):
        task_name = MT_TASKS[(episode - 1) % len(MT_TASKS)]
        task_idx = MT_TASKS.index(task_name)

        print(f"\n--- Episode {episode}/{NUM_EPISODES} --- Task: {task_name}")

        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=SEED + episode,
            render_mode="rgb_array",
            reward_function_version="v3",
            max_episode_steps=MAX_EPISODE_STEPS,
            terminate_on_success=False,
        )
        env = AppendTaskIdWrapper(env, task_index=task_idx, n_tasks=len(MT_TASKS))
        env = Monitor(env)

        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        episode_success = False

        global_episode_idx += 1
        save_video = global_episode_idx in VIDEO_EPISODES

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            steps += 1

            if "success" in info and info["success"]:
                episode_success = True

            if save_video:
                frame = env.render()
                if frame is not None:
                    video_frames[global_episode_idx].append(frame)

        print(f"Episode finished after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {episode_success}")

        total_rewards.append(total_reward)
        if episode_success:
            success_count += 1

        task_rewards[task_name].append(total_reward)
        task_success[task_name].append(1 if episode_success else 0)

        env.close()

    print("\n" + "=" * 60)
    print(f"=== {MT_N} Evaluation Complete ===")
    print(f"Episodes (total): {NUM_EPISODES}")
    print(f"Average reward (overall): {np.mean(total_rewards):.2f}")
    print(f"Std reward (overall): {np.std(total_rewards):.2f}")
    print(f"Min reward (overall): {np.min(total_rewards):.2f}")
    print(f"Max reward (overall): {np.max(total_rewards):.2f}")
    print(f"Success rate (overall): {success_count}/{NUM_EPISODES} ({100*success_count/NUM_EPISODES:.1f}%)")
    print("=" * 60)

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

    for ep, frames in video_frames.items():
        if len(frames) > 0:
            filename = f"{ALGORITHM.lower()}_{MT_N}_eval_ep{ep}.mp4"
            print(f"Saving Episode {ep} Video ({len(frames)} frames) to: {filename}")
            imageio.mimsave(filename, frames, fps=30)
        else:
            print(f"No frames captured for episode {ep}")

    dummy_env.close()
