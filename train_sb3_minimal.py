# ==============================================
# sb3_metaworld_multitask.py
# ==============================================

import gymnasium as gym
import numpy as np
from typing import List, Optional, Union
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# -----------------------------
# 1️⃣ CurriculumTaskWrapper
# -----------------------------
class CurriculumTaskWrapper(gym.Wrapper):
    """
    Unified curriculum interface for:
    - MT1 (set_task)
    - MT3 / MT10 / custom multitask (set_tasks / envs_list)
    """

    def __init__(self, env, initial_tasks: List[str]):
        super().__init__(env)
        self.set_tasks(initial_tasks)

    def set_tasks(self, tasks: List[str]):
        if not tasks:
            raise ValueError("tasks must not be empty")

        self.tasks = list(tasks)
        base_env = self.env.unwrapped

        # MT1
        if hasattr(base_env, "set_task"):
            if len(tasks) != 1:
                raise ValueError(
                    f"MT1 env received multiple tasks: {tasks}. Use multitask env instead."
                )
            base_env.set_task(tasks[0])

        # Multi-task
        elif hasattr(base_env, "envs_list"):
            base_env.envs_list = self.tasks

        else:
            raise RuntimeError(
                f"Underlying env {type(base_env)} does not support task switching"
            )


# -----------------------------
# 2️⃣ TaskOneHotWrapper
# -----------------------------
from gymnasium import spaces

class TaskOneHotWrapper(gym.ObservationWrapper):
    """
    Adds one-hot task encoding to the observation
    """

    def __init__(self, env, all_tasks: List[str]):
        super().__init__(env)
        self.all_tasks = all_tasks
        self.task_to_idx = {t: i for i, t in enumerate(all_tasks)}

        orig_space = env.observation_space
        self.task_dim = len(all_tasks)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(orig_space.shape[0] + self.task_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        one_hot = np.zeros(self.task_dim, dtype=np.float32)
        base_env = self.env.unwrapped
        task_name = getattr(base_env, "env_name", None)
        if task_name in self.task_to_idx:
            one_hot[self.task_to_idx[task_name]] = 1.0
        return np.concatenate([obs, one_hot], axis=0)


# -----------------------------
# 3️⃣ CurriculumCallback
# -----------------------------
class CurriculumCallback(BaseCallback):
    """
    Stage-based Curriculum for SB3 VecEnvs
    """

    def __init__(
        self,
        stages,
        eval_env,
        min_steps_per_stage: int = 100_000,
        success_threshold: float = 0.8,
        eval_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.eval_env = eval_env
        self.min_steps = min_steps_per_stage
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq

        self.stage_idx = 0
        self.stage_step_start = 0

    def _on_training_start(self):
        self._set_stage(0)

    def _set_stage(self, idx: int):
        self.stage_idx = idx
        self.stage_step_start = self.num_timesteps
        tasks = self.stages[idx]
        self.training_env.env_method("set_tasks", tasks)
        if self.verbose:
            print(f"\n🚀 Curriculum → Stage {idx}: {tasks}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.stage_step_start < self.min_steps:
            return True
        if self.n_calls % self.eval_freq != 0:
            return True
        mean_reward, success_rate = self._evaluate()
        if self.verbose:
            print(
                f"[Curriculum] Stage {self.stage_idx} | "
                f"reward={mean_reward:.2f} | success={success_rate:.2f}"
            )
        if success_rate >= self.success_threshold:
            if self.stage_idx + 1 < len(self.stages):
                self._set_stage(self.stage_idx + 1)
            else:
                if self.verbose:
                    print("🏁 Curriculum completed")
        return True

    def _evaluate(self, n_episodes: int = 10):
        rewards, successes = [], []
        obs = self.eval_env.reset()
        for _ in range(n_episodes):
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = self.eval_env.step(action)
                ep_reward += reward[0]
                if done:
                    info = infos[0]
                    successes.append(info.get("success", 0.0))
            rewards.append(ep_reward)
            obs = self.eval_env.reset()
        return np.mean(rewards), np.mean(successes)


# -----------------------------
# 4️⃣ Taskwise Evaluation
# -----------------------------
def evaluate_per_task(model, env, tasks, n_episodes=5):
    results = {}
    for task in tasks:
        env.env_method("set_tasks", [task])
        rewards, successes = [], []
        obs = env.reset()
        for _ in range(n_episodes):
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                ep_reward += reward[0]
                if done:
                    info = infos[0]
                    successes.append(info.get("success", 0.0))
            rewards.append(ep_reward)
            obs = env.reset()
        results[task] = {"reward": float(np.mean(rewards)), "success": float(np.mean(successes))}
    return results


# -----------------------------
# 5️⃣ Factory
# -----------------------------
class SB3MetaWorldEnvFactory:
    """
    SB3-only MetaWorld Env Factory
    Supports MT1, MT3, MT10, custom multi-task
    """

    MT10_TASKS = [
        "reach-v3", "push-v3", "pick-place-v3",
        "plate-slide-v3", "pick-out-of-hole-v3",
        "push-wall-v3", "plate-slide-side-v3",
        "hammer-v3", "door-open-v3", "drawer-close-v3"
    ]

    MT3_TASKS = ["reach-v3", "push-v3", "pick-place-v3"]

    def __init__(self, vector_strategy="sync", verbose=True, use_subproc=False):
        self.vector_strategy = vector_strategy
        self.verbose = verbose
        self.use_subproc = use_subproc

    def _log(self, msg):
        if self.verbose:
            print(f"[EnvFactory] {msg}")

    def _make_mt1_env_fn(self, task, seed, max_episode_steps, rank):
        print(task)
        def _init():
            env = gym.make(
                'Meta-World/MT1',
                env_name=task,
                seed=seed + rank,
                max_episode_steps=max_episode_steps,
            )
            env = CurriculumTaskWrapper(env, [task])
            env = TaskOneHotWrapper(env, all_tasks=self.MT10_TASKS)
            return env
        return _init

    def _make_multitask_env_fn(self, tasks, seed, max_episode_steps, rank):
        def _init():
            env = gym.make(
                "Meta-World/custom-mt-envs",
                envs_list=tasks,
                seed=seed + rank,
                max_episode_steps=max_episode_steps,
            )
            env = CurriculumTaskWrapper(env, tasks)
            env = TaskOneHotWrapper(env, all_tasks=self.MT10_TASKS)
            return env
        return _init

    def make_mt1_env(self, task_name, seed=0, max_episode_steps=500, n_envs=1):
        env_fns = [self._make_mt1_env_fn(task_name, seed, max_episode_steps, i) for i in range(n_envs)]
        if n_envs > 1 and self.use_subproc:
            vec_env = SubprocVecEnv(env_fns)
        else:
            print("Dummy VEc")
            vec_env = DummyVecEnv(env_fns)
        return VecMonitor(vec_env)

    def make_multitask_env(self, tasks: List[str], seed=0, max_episode_steps=500):
        env_fns = [self._make_multitask_env_fn(tasks, seed, max_episode_steps, 0)]
        vec_env = DummyVecEnv(env_fns)
        return VecMonitor(vec_env)

    def create_train_eval_pair(
        self, tasks: Union[str, List[str]], train_seed=0, n_parallel_envs=1
    ):
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            print("Here")

            train_env = self.make_mt1_env(tasks[0], seed=train_seed, n_envs=n_parallel_envs)
            eval_env = self.make_mt1_env(tasks[0], seed=train_seed + 1000, n_envs=1)
        else:
            train_env = self.make_multitask_env(tasks, seed=train_seed)
            eval_env = self.make_multitask_env(tasks, seed=train_seed + 1000)
        print("There")
        train_env = VecNormalize(train_env)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        eval_env.obs_rms = train_env.obs_rms

        return train_env, eval_env


# -----------------------------
# 6️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    factory = SB3MetaWorldEnvFactory(verbose=True, use_subproc=True)

    stages = [
        ["reach-v3"],
        ["reach-v3", "push-v3"],
        ["reach-v3", "push-v3", "pick-place-v3"],
    ]
    train_env, eval_env = factory.create_train_eval_pair(
        tasks=stages[0],
        train_seed=42,
        n_parallel_envs=2,

      )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    model = SAC("MlpPolicy", train_env, policy_kwargs=policy_kwargs, verbose=1)

    callback = CurriculumCallback(stages=stages, eval_env=eval_env, min_steps_per_stage=50_000, eval_freq=5_000)
    model.learn(200_000, callback=callback)

    results = evaluate_per_task(model, eval_env, ["reach-v3", "push-v3", "pick-place-v3"])
    print("Evaluation per task:")
    for task, metrics in results.items():
        print(task, metrics)
