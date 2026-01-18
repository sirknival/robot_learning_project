
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ------------------------------------------------------------
# Deterministic Task One-Hot Wrapper for vectorized Environments
# ------------------------------------------------------------
class TaskIdVecEnvWrapper(VecEnvWrapper):
    """
    Appends one-hot task id to obs based on envs_list:
    sub-env i corresponds to envs_list[i] (including duplicates).
    """

    def __init__(self, venv: VecEnv, envs_list: list[str], task_name_to_id: dict, n_tasks: int):
        super().__init__(venv)
        self.envs_list = list(envs_list)
        self.task_name_to_id = dict(task_name_to_id)
        self.n_tasks = int(n_tasks)

        if len(self.envs_list) != self.num_envs:
            raise RuntimeError(
                f"AppendFixedTaskIdVecWrapper: len(envs_list)={len(self.envs_list)} "
                f"but venv.num_envs={self.num_envs}. Must match."
            )

        idx = []
        for name in self.envs_list:
            if name not in self.task_name_to_id:
                raise RuntimeError(f"Unknown task '{name}'. Known: {list(self.task_name_to_id.keys())}")
            idx.append(self.task_name_to_id[name])
        self.task_idx_per_env = np.array(idx, dtype=np.int64)

        obs_space = venv.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise RuntimeError("AppendFixedTaskIdVecWrapper expects Box observation_space")

        low = np.asarray(obs_space.low, dtype=np.float32)
        high = np.asarray(obs_space.high, dtype=np.float32)
        low = np.concatenate([low, -np.ones(self.n_tasks, dtype=np.float32)], axis=0)
        high = np.concatenate([high, np.ones(self.n_tasks, dtype=np.float32)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _onehot(self) -> np.ndarray:
        oh = np.zeros((self.num_envs, self.n_tasks), dtype=np.float32)
        oh[np.arange(self.num_envs), self.task_idx_per_env] = 1.0
        return oh

    def reset(self):
        obs = self.venv.reset()
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._onehot()], axis=1)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._onehot()], axis=1), rewards, dones, infos


# -------------------------------
# Eval Wrapper -> append task one-hot to obs
# -------------------------------
class TaskIdDummyEnvWrapper(gym.Wrapper):
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
