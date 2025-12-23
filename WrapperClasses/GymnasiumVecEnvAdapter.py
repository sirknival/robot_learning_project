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

    def __init__(self, gym_vec_env):
        self.gym_vec_env = gym_vec_env

        # Gymnasium-VectorEnv hat single_observation_space / single_action_space
        num_envs = gym_vec_env.num_envs
        observation_space = gym_vec_env.single_observation_space
        action_space = gym_vec_env.single_action_space

        # super() ruft intern get_attr("render_mode") -> self.gym_vec_env MUSS vorher gesetzt sein
        super().__init__(
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
        )

        self.num_envs = num_envs
        self._actions = None

        # SB3 erwartet, dass reset() Infos (falls vorhanden) hier abgelegt werden können
        # (SB3 VecEnv API: reset() gibt nur obs zurück, infos separat)
        self.reset_infos = [dict() for _ in range(self.num_envs)]

    def close(self):
        return self.gym_vec_env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Für unsere Nutzung nicht relevant. Falls später pro-Env-Methoden
        gebraucht werden, kann man das noch sauber implementieren.
        """
        raise NotImplementedError

    def get_attr(self, name, indices=None):
        """
        SB3 ruft z.B. get_attr("render_mode") im Konstruktor.
        Wir geben entweder den entsprechenden Attributwert der
        zugrunde liegenden VectorEnv zurück oder None.
        """
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        # Falls die VectorEnv das Attribut gar nicht hat → None
        if not hasattr(self.gym_vec_env, name):
            return [None for _ in indices]

        value = getattr(self.gym_vec_env, name)

        # Falls es schon eine Liste/Array pro Env ist, indexieren
        try:
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_envs:
                return [value[i] for i in indices]
        except TypeError:
            pass

        # Sonst denselben Wert für alle gewünschten Envs zurückgeben
        return [value for _ in indices]

    def render(self, mode="human"):
        # Gymnasium-VectorEnv hat meist .render(), evtl. ohne mode
        return self.gym_vec_env.render()

    def reset(self):
        obs, info = self.gym_vec_env.reset()

        # Gymnasium VectorEnv kann info als dict-of-arrays liefern.
        # SB3 speichert reset infos typischerweise separat, reset() liefert nur obs. :contentReference[oaicite:2]{index=2}
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
            info = list_infos
        elif not isinstance(info, list):
            info = [dict() for _ in range(self.num_envs)]

        self.reset_infos = info

        # SB3-VecEnv.reset() gibt nur obs zurück
        return obs, info

    def seed(self, seed=None):
        # Gymnasium-VectorEnv nutzt reset(seed=...)
        # Wir triggern ein reset mit seed, damit das auch wirklich wirkt (statt pass).
        # (kein obs-return erwartet hier; SB3 ruft seed meist optional)
        try:
            self.gym_vec_env.reset(seed=seed)
        except TypeError:
            # falls reset(seed=...) nicht unterstützt wird
            pass

    def set_attr(self, name, values, indices=None):
        """
        Minimal-Implementierung: Versuch Attribut auf der
        gesamten VectorEnv zu setzen. In der Praxis wird das von
        SB3 für MT10 normalerweise nicht benutzt.
        """
        if hasattr(self.gym_vec_env, name):
            setattr(self.gym_vec_env, name, values)
        # kein Return nötig

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.gym_vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)

        # Gymnasium-VectorEnv gibt infos meist als dict mit Arrays/Listern/Skalaren zurück
        # SB3 erwartet: Liste von dicts (pro Env ein dict)
        if isinstance(infos, dict):
            list_infos = [dict() for _ in range(self.num_envs)]
            for k, v in infos.items():
                v = np.array(v)

                # Fall 1: v ist z.B. shape (num_envs, ...) -> pro Env indexieren
                if v.ndim > 0 and v.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v[i]
                else:
                    # Fall 2: Skalar oder "globaler" Wert -> für alle Envs gleich setzen
                    for i in range(self.num_envs):
                        list_infos[i][k] = v.item() if v.ndim == 0 else v

            infos = list_infos

        # Gymnasium VectorEnv autoreset: obs/infos nach done können schon vom "neuen" Episode-Start sein.
        # Die echte letzte Observation steckt in infos["final_observation"] (und ggf. final_info). :contentReference[oaicite:3]{index=3}
        # SB3 erwartet bei done oft 'terminal_observation' im info (ein Dict pro Env).
        if isinstance(infos, list) and len(infos) == self.num_envs:
            for i in range(self.num_envs):
                if dones[i]:
                    fin = None
                    if "final_observation" in infos[i]:
                        fin = infos[i]["final_observation"]
                    if fin is not None and "terminal_observation" not in infos[i]:
                        infos[i]["terminal_observation"] = fin

                    # Optional: Gymnasium liefert manchmal "final_info"
                    if "final_info" in infos[i] and "terminal_info" not in infos[i]:
                        infos[i]["terminal_info"] = infos[i]["final_info"]

        return obs, rewards, dones, infos
