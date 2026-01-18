from stable_baselines3.common.vec_env import VecEnv
import numpy as np

# ------------------------------------------------------------
# Adapter: Gymnasium VectorEnv -> SB3 VecEnv 
# ------------------------------------------------------------
class GymnasiumVecEnvAdapter(VecEnv):
    def __init__(self, gym_vec_env):
        self.gym_vec_env = gym_vec_env

        num_envs = gym_vec_env.num_envs
        observation_space = gym_vec_env.single_observation_space
        action_space = gym_vec_env.single_action_space

        super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)

        self.num_envs = num_envs
        self._actions = None
        self.reset_infos = [dict() for _ in range(self.num_envs)]

    def close(self):
        return self.gym_vec_env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        if not hasattr(self.gym_vec_env, name):
            return [None for _ in indices]

        value = getattr(self.gym_vec_env, name)

        try:
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_envs:
                return [value[i] for i in indices]
        except TypeError:
            pass

        return [value for _ in indices]

    def render(self, mode="human"):
        return self.gym_vec_env.render()

    def reset(self):
        obs, info = self.gym_vec_env.reset()

        if isinstance(info, dict):
            list_infos = [dict() for _ in range(self.num_envs)]
            for k, v in info.items():
                v = np.array(v)
                if v.ndim > 0 and v.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v[i]
                else:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v.item() if v.ndim == 0 else v
            self.reset_infos = list_infos
        elif isinstance(info, list):
            self.reset_infos = info
        else:
            self.reset_infos = [dict() for _ in range(self.num_envs)]

        return obs

    def seed(self, seed=None):
        try:
            self.gym_vec_env.reset(seed=seed)
        except TypeError:
            pass

    def set_attr(self, name, values, indices=None):
        if hasattr(self.gym_vec_env, name):
            setattr(self.gym_vec_env, name, values)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.gym_vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)

        if isinstance(infos, dict):
            list_infos = [dict() for _ in range(self.num_envs)]
            for k, v in infos.items():
                v = np.array(v)
                if v.ndim > 0 and v.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v[i]
                else:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v.item() if v.ndim == 0 else v
            infos = list_infos

        if isinstance(infos, list) and len(infos) == self.num_envs:
            for i in range(self.num_envs):
                if dones[i]:
                    fin = infos[i].get("final_observation", None)
                    if fin is not None and "terminal_observation" not in infos[i]:
                        infos[i]["terminal_observation"] = fin
                    if "final_info" in infos[i] and "terminal_info" not in infos[i]:
                        infos[i]["terminal_info"] = infos[i]["final_info"]

        return obs, rewards, dones, infos
    
