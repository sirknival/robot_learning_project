from stable_baselines3.common.vec_env import VecEnv
import numpy as np
# ------------------------------------------------------------
#  Adapter: Gymnasium VectorEnv (Metaworld)  -> Stable-Baselines3 VecEnv
# ------------------------------------------------------------


class GymnasiumVecEnvAdapter(VecEnv):
    """
    Adapter, der eine gymnasium.vector.VectorEnv (z.B. MT10 via gym.make_vec)
    in eine Stable-Baselines3-VecEnv übersetzt.
    """

    def __init__(self, gym_vec_env, stage_tasks=None):
        self.gym_vec_env = gym_vec_env

        self.num_envs = int(gym_vec_env.num_envs)
        observation_space = gym_vec_env.single_observation_space
        action_space = gym_vec_env.single_action_space

        super().__init__(
            num_envs=self.num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._actions = None

        # For SB3 compatibility: store reset infos here
        self.reset_infos = [dict() for _ in range(self.num_envs)]

        # Optional: curriculum task list (can include duplicates)
        self.stage_tasks = list(stage_tasks) if stage_tasks is not None else None

    # -------------------- helpers --------------------

    def _task_for_env_index(self, i: int):
        if not self.stage_tasks:
            return None
        return self.stage_tasks[i % len(self.stage_tasks)]

    def _infos_dict_to_list(self, info_dict):
        """
        Convert Gymnasium dict-of-arrays infos into list-of-dicts (per env).
        """
        list_infos = [dict() for _ in range(self.num_envs)]
        for k, v in info_dict.items():
            v = np.array(v)
            if v.ndim > 0 and v.shape[0] == self.num_envs:
                for i in range(self.num_envs):
                    list_infos[i][k] = v[i]
            else:
                val = v.item() if v.ndim == 0 else v
                for i in range(self.num_envs):
                    list_infos[i][k] = val
        return list_infos

    def _ensure_list_infos(self, infos):
        """
        Ensure infos is list[dict] length num_envs.
        """
        if isinstance(infos, dict):
            infos = self._infos_dict_to_list(infos)
        elif isinstance(infos, list):
            # ensure length
            if len(infos) != self.num_envs:
                infos = [dict() for _ in range(self.num_envs)]
            else:
                # ensure dicts
                for i in range(self.num_envs):
                    if infos[i] is None:
                        infos[i] = {}
        else:
            infos = [dict() for _ in range(self.num_envs)]
        return infos

    def _inject_task_name(self, infos):
        """
        Make sure infos[i]["task_name"] exists when stage_tasks is known.
        If env already provides task_name, keep it.
        """
        if self.stage_tasks is None:
            return infos

        for i in range(self.num_envs):
            if "task_name" not in infos[i] or infos[i]["task_name"] in (None, "", b""):
                infos[i]["task_name"] = self._task_for_env_index(i)
        return infos

    def _inject_terminal_observation(self, obs, dones, infos):
        """
        Gymnasium vector env autoreset: final observation may appear in infos["final_observation"].
        SB3 expects infos[i]["terminal_observation"] when done.
        """
        if not (isinstance(infos, list) and len(infos) == self.num_envs):
            return infos

        for i in range(self.num_envs):
            if bool(dones[i]):
                # final_observation -> terminal_observation
                if "final_observation" in infos[i] and "terminal_observation" not in infos[i]:
                    infos[i]["terminal_observation"] = infos[i]["final_observation"]
                # final_info -> terminal_info (optional)
                if "final_info" in infos[i] and "terminal_info" not in infos[i]:
                    infos[i]["terminal_info"] = infos[i]["final_info"]
        return infos

    # -------------------- SB3 VecEnv API --------------------

    def close(self):
        return self.gym_vec_env.close()

    def render(self, mode="human"):
        # many gymnasium vec envs ignore mode
        try:
            return self.gym_vec_env.render()
        except TypeError:
            return self.gym_vec_env.render(mode=mode)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # Forward vectorized method calls (important for curriculum updates)
        if indices is None:
            return self.gym_vec_env.call(method_name, *method_args, **method_kwargs)

        results = self.gym_vec_env.call(method_name, *method_args, **method_kwargs)
        if isinstance(indices, int):
            return [results[indices]]
        return [results[i] for i in indices]

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

    def set_attr(self, name, values, indices=None):
        # Minimal: set on the vector env object
        if hasattr(self.gym_vec_env, name):
            setattr(self.gym_vec_env, name, values)

    def seed(self, seed=None):
        try:
            self.gym_vec_env.reset(seed=seed)
        except TypeError:
            pass

    def reset(self):
        """
        SB3 expects: return obs only.
        infos are stored in self.reset_infos.
        """
        res = self.gym_vec_env.reset()
        # Gymnasium reset returns (obs, infos)
        if isinstance(res, tuple) and len(res) == 2:
            obs, infos = res
        else:
            # fallback (shouldn't happen)
            obs, infos = res, [dict() for _ in range(self.num_envs)]

        infos = self._ensure_list_infos(infos)
        infos = self._inject_task_name(infos)

        self.reset_infos = infos
        return obs  # IMPORTANT: obs only

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.gym_vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)

        infos = self._ensure_list_infos(infos)
        infos = self._inject_terminal_observation(obs, dones, infos)
        infos = self._inject_task_name(infos)

        return obs, rewards, dones, infos


