import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class OneHotTaskWrapper(VecEnv):
    """
    Erweitert die Observation um eine One-Hot-Kodierung des aktuellen Tasks.
    """

    def __init__(self, venv: VecEnv, task_names, one_hot_dim=None):
        # super().__init__()
        self.venv = venv
        self.task_names = task_names

        if one_hot_dim is not None:
            self.n_tasks = one_hot_dim
        else:
            self.n_tasks = len(task_names)

        self.num_envs = venv.num_envs
        self.action_space = venv.action_space

        orig_space = venv.observation_space
        assert isinstance(orig_space, spaces.Box)

        # Space erweitern
        low = np.concatenate([orig_space.low, np.zeros(self.n_tasks)]).astype(np.float32)
        high = np.concatenate([orig_space.high, np.ones(self.n_tasks)]).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._task_ids = np.zeros(self.num_envs, dtype=int)

    def reset(self):
        reset_result = self.venv.reset()

        if isinstance(reset_result, tuple):
            obs, infos = reset_result
        else:
            # Fallback für alte SB3 VecEnv
            obs = reset_result
            # Versuche infos aus reset_infos zu holen
            if hasattr(self.venv, 'reset_infos'):
                infos = self.venv.reset_infos
            else:
                infos = [{} for _ in range(self.num_envs)]

            # Task-ID Extraktion
        for i in range(self.num_envs):
            task_name = self._extract_task_name(i, infos[i])
            if task_name and task_name in self.task_names:
                self._task_ids[i] = self.task_names.index(task_name)


        return self._augment_obs(obs)

    def _extract_task_name(self, env_idx, info):
        """Robuste Task-Name Extraktion mit Fallbacks"""
        # Versuch 1: Aus info dict
        task_name = info.get("task_name")
        if task_name:
            return task_name

        # Versuch 2: aus dem Environment selbst
        try:
            if hasattr(self.venv, 'envs'):
                env = self.venv.envs[env_idx]
                if hasattr(env, 'task_name'):
                    return env.task_name
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'task_name'):
                    return env.unwrapped.task_name
        except:
            pass

        # Versuch 3: aus gym_vec_env (bei GymnasiumVecEnvAdapter)
        try:
            if hasattr(self.venv, 'gym_vec_env'):
                gym_env = self.venv.gym_vec_env
                if hasattr(gym_env, 'envs'):
                    env = gym_env.envs[env_idx]
                    if hasattr(env, 'task_name'):
                        return env.task_name
        except:
            pass

        return None

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
        """Fügt One-Hot Task Encoding hinzu"""

        one_hot = np.zeros((self.num_envs, self.n_tasks), dtype=np.float32)
        safe_ids = np.clip(self._task_ids, 0, self.n_tasks - 1)

        one_hot[np.arange(self.num_envs), safe_ids] = 1.0
        return np.concatenate([obs, one_hot], axis=1)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.venv.env_is_wrapped(wrapper_class, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_attr(self, name, indices=None):
        return self.venv.get_attr(name, indices=indices)

    def set_attr(self, name, values, indices=None):
        return self.venv.set_attr(name, values, indices=indices)

    def get_images(self):
        return self.venv.get_images()

    def seed(self, seed=None):
        return self.venv.seed(seed)

