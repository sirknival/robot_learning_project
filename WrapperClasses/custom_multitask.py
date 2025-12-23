import gymnasium as gym
import numpy as np

class CustomMTEnv(gym.Env):
    def __init__(self, env_list, seed=0, max_episode_steps=500, reward_function_version='v3'):
        super().__init__()
        self.env_list = env_list
        self.seed_value = seed
        self.max_episode_steps = max_episode_steps
        self.reward_version = reward_function_version

        # We dynamically create ML1 instances for each task
        self.task_classes = {
            name: metaworld.ML1(name).train_classes[name]
            for name in env_list
        }

        self.current_env = None
        self._set_new_env()

        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def _set_new_env(self):
        """Pick a random task."""
        task_name = np.random.choice(self.env_list)
        env_class = self.task_classes[task_name]
        self.current_env = env_class(seed=self.seed_value)

    def reset(self, **kwargs):
        self._set_new_env()
        return self.current_env.reset(**kwargs)

    def step(self, action):
        return self.current_env.step(action)


