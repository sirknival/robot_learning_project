import os
from stable_baselines3.common.callbacks import BaseCallback


class ReplayBufferCheckpointCallback(BaseCallback):
    """
    Saves the Replay-Buffer in regelmäßigen Abständen als .pkl,
    parallel zu den Modell-Checkpoints.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.model is None:
            return True

        # num_timesteps ist die globale Schrittzahl (alle Vektor-Env-Instanzen zusammen)
        if self.model.num_timesteps > 0 and self.model.num_timesteps % self.save_freq == 0:
            filename = f"{self.name_prefix}_{self.model.num_timesteps}_steps_replay.pkl"
            path = os.path.join(self.save_path, filename)
            if self.verbose > 0:
                print(f"Saving replay buffer checkpoint to: {path}")
            self.model.save_replay_buffer(path)
        return True
