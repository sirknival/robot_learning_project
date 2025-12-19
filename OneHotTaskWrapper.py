import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class OneHotTaskWrapper(VecEnv):
    """
    Erweitert die Observation um eine One-Hot-Kodierung des aktuellen Tasks.
    """

    def __init__(self, venv: VecEnv, task_names):
        self.venv = venv
        self.task_names = task_names
        self.n_tasks = len(task_names)

        self.num_envs = venv.num_envs
        self.action_space = venv.action_space

        # Original Observation Space
        orig_space = venv.observation_space
        assert isinstance(orig_space, spaces.Box)

        low = np.concatenate([orig_space.low, np.zeros(self.n_tasks)])
        high = np.concatenate([orig_space.high, np.ones(self.n_tasks)])

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

        # pro Env merken wir uns die Task-ID
        self._task_ids = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        obs = self.venv.reset()

        # Task-ID aus reset_infos lesen (Meta-World legt sie dort ab)
        for i in range(self.num_envs):
            info = self.venv.reset_infos[i]
            task_name = info.get("task_name", None)

            if task_name is None:
                raise RuntimeError(
                    "Meta-World liefert keinen task_name im reset info."
                )

            self._task_ids[i] = self.task_names.index(task_name)

        return self._augment_obs(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        for i, done in enumerate(dones):
            if done:
                task_name = infos[i].get("task_name", None)
                if task_name is not None:
                    self._task_ids[i] = self.task_names.index(task_name)

        return self._augment_obs(obs), rewards, dones, infos

    def close(self):
        return self.venv.close()

    def _augment_obs(self, obs):
        one_hot = np.zeros((self.num_envs, self.n_tasks), dtype=np.float32)
        one_hot[np.arange(self.num_envs), self._task_ids] = 1.0
        return np.concatenate([obs, one_hot], axis=1)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, name, values, indices=None):
        pass
