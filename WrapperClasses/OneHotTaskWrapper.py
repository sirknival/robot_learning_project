import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from training_setup_multitask.utilities.MetaworldTasks import MT10_TASKS


class OneHotTaskWrapper(VecEnv):
    """
    Wraps a VecEnv to add one-hot task encoding to observations.

    This wrapper extends the observation space by concatenating a one-hot
    encoded task ID to each observation. Uses a fixed dimension for the
    one-hot encoding to ensure consistency across different numbers of tasks.

    The wrapper properly handles terminal_observation from both DummyVecEnv
    and GymnasiumVecEnvAdapter to ensure correct shapes for SB3.

    Args:
        venv: Vectorized environment to wrap
        task_names: List of task names in this environment
        one_hot_dim: Fixed dimension for one-hot encoding (default: 10 for MT10)

    """

    def __init__(self, venv: VecEnv, task_names, one_hot_dim=10):
        """
        Args:
            venv: Vectorized environment to wrap
            task_names: List of task names in this environment
            one_hot_dim: Fixed dimension for one-hot encoding (default: 10 for MT10)
        """
        self.venv = venv
        self.task_names = task_names
        self.n_tasks = one_hot_dim  # Use fixed dimension

        self.num_envs = venv.num_envs
        self.action_space = venv.action_space

        orig_space = venv.observation_space
        assert isinstance(orig_space, spaces.Box), \
            f"Expected Box observation space, got {type(orig_space)}"

        # Extend observation space with one-hot encoding
        low = np.concatenate([
            orig_space.low,
            np.full(self.n_tasks, -np.inf, dtype=np.float32)
        ]).astype(np.float32)

        high = np.concatenate([
            orig_space.high,
            np.full(self.n_tasks, np.inf, dtype=np.float32)
        ]).astype(np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        # Track task IDs for each environment
        self._task_ids = np.zeros(self.num_envs, dtype=int)

        # Create mapping from task names to IDs
        self._task_name_to_id = {name: i for i, name in enumerate(MT10_TASKS)}

    def reset(self):
        """Reset environments and return augmented observations"""
        reset_result = self.venv.reset()

        # Handle both old and new Gymnasium API
        if isinstance(reset_result, tuple):
            obs, infos = reset_result
        else:
            # Fallback for older SB3 VecEnv
            obs = reset_result
            if hasattr(self.venv, 'reset_infos'):
                infos = self.venv.reset_infos
            else:
                infos = [{} for _ in range(self.num_envs)]

        # IMPORTANT: Ensure obs is 2D array (num_envs, obs_dim)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Extract task IDs from infos
        for i in range(self.num_envs):
            task_name = self._extract_task_name(i, infos[i])
            if task_name and task_name in self._task_name_to_id:
                self._task_ids[i] = self._task_name_to_id[task_name]
            else:
                # If task not found, use environment index % num_tasks
                self._task_ids[i] = i % len(self.task_names)

        augmented_obs = self._augment_obs(obs)

        # Always return just the observations for SB3 compatibility
        return augmented_obs

    def _extract_task_name(self, env_idx, info):
        """
        Robust task name extraction with multiple fallback strategies

        Args:
            env_idx: Environment index
            info: Info dictionary from environment

        Returns:
            Task name string or None
        """
        # Strategy 1: From info dict
        task_name = info.get("task_name")
        if task_name:
            return task_name

        # Strategy 2: From environment itself
        try:
            if hasattr(self.venv, 'envs'):
                env = self.venv.envs[env_idx]
                if hasattr(env, 'task_name'):
                    return env.task_name
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'task_name'):
                    return env.unwrapped.task_name
        except (AttributeError, IndexError):
            pass

        # Strategy 3: From gym_vec_env (for GymnasiumVecEnvAdapter)
        try:
            if hasattr(self.venv, 'gym_vec_env'):
                gym_env = self.venv.gym_vec_env
                if hasattr(gym_env, 'envs'):
                    env = gym_env.envs[env_idx]
                    if hasattr(env, 'task_name'):
                        return env.task_name
        except (AttributeError, IndexError):
            pass

        # Strategy 4: Use round-robin based on environment index
        # This works for MT3/MT10 where tasks are assigned in order
        if len(self.task_names) > 0:
            return self.task_names[env_idx % len(self.task_names)]

        return None

    def step_async(self, actions):
        """Submit actions to environments (async step part 1)"""
        self.venv.step_async(actions)

    def step_wait(self):
        """Wait for step results and return augmented observations (async step part 2)"""
        obs, rewards, dones, infos = self.venv.step_wait()

        # Update task IDs when environments reset
        for i, done in enumerate(dones):
            if done:
                task_name = self._extract_task_name(i, infos[i])
                if task_name and task_name in self._task_name_to_id:
                    self._task_ids[i] = self._task_name_to_id[task_name]

                if "terminal_observation" in infos[i]:
                    terminal_obs = infos[i]["terminal_observation"]
                    expected_base_dim = self.observation_space.shape[0] - self.n_tasks

                    # Check if terminal_obs needs one-hot encoding
                    if terminal_obs.shape[-1] == expected_base_dim:

                        if terminal_obs.ndim == 1:
                            terminal_obs = terminal_obs.reshape(1, -1)

                        # Create one-hot encoding for this specific environment
                        one_hot = np.zeros((1, self.n_tasks), dtype=np.float32)
                        task_id = np.clip(self._task_ids[i], 0, self.n_tasks - 1)
                        one_hot[0, task_id] = 1.0

                        # Concatenate base observation with one-hot encoding
                        augmented_terminal = np.concatenate([terminal_obs, one_hot], axis=1).astype(np.float32)

                        # Store back as 1D array (SB3 expects 1D terminal_observation)
                        infos[i]["terminal_observation"] = augmented_terminal.flatten()

                    elif terminal_obs.shape[-1] != self.observation_space.shape[0]:
                        # Unexpected shape - remove to avoid errors
                        del infos[i]["terminal_observation"]

        return self._augment_obs(obs), rewards, dones, infos

    def close(self):
        """Close wrapped environment"""
        return self.venv.close()

    def _augment_obs(self, obs):
        """
        Add one-hot task encoding to observations

        Args:
            obs: Base observations from environment (num_envs, obs_dim)

        Returns:
            Augmented observations with one-hot encoding (num_envs, obs_dim + n_tasks)
        """
        # Ensure obs is 2D
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Validate shape
        assert obs.shape[0] == self.num_envs, \
            f"Expected {self.num_envs} observations, got {obs.shape[0]}"

        # Create one-hot encoding (num_envs, n_tasks)
        one_hot = np.zeros((self.num_envs, self.n_tasks), dtype=np.float32)

        # Ensure task IDs are valid
        safe_ids = np.clip(self._task_ids, 0, self.n_tasks - 1)

        # Set one-hot values
        one_hot[np.arange(self.num_envs), safe_ids] = 1.0

        # Concatenate along feature dimension (axis=1)
        augmented = np.concatenate([obs, one_hot], axis=1).astype(np.float32)

        # Validate output shape
        expected_dim = obs.shape[1] + self.n_tasks
        assert augmented.shape == (self.num_envs, expected_dim), \
            f"Expected shape ({self.num_envs}, {expected_dim}), got {augmented.shape}"

        return augmented

    # VecEnv interface methods

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environments are wrapped with specific wrapper"""
        return self.venv.env_is_wrapped(wrapper_class, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on wrapped environments"""
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_attr(self, name, indices=None):
        """Get attribute from wrapped environments"""
        return self.venv.get_attr(name, indices=indices)

    def set_attr(self, name, values, indices=None):
        """Set attribute on wrapped environments"""
        return self.venv.set_attr(name, values, indices=indices)

    def get_images(self):
        """Get images from wrapped environments"""
        return self.venv.get_images()

    def seed(self, seed=None):
        """Set random seed for wrapped environments"""
        return self.venv.seed(seed)

    def __repr__(self):
        """String representation"""
        return f"OneHotTaskWrapper({self.venv}, tasks={self.task_names}, dim={self.n_tasks})"


