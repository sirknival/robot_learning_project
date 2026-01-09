from stable_baselines3.common.callbacks import BaseCallback

class PrintOptimizerCallback(BaseCallback):
    """
    Callback zum Auslesen der Optimizer von Actor und Critic
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        actor_optim = self.model.actor.optimizer
        critic_optim = self.model.critic.optimizer

        print("Actor Optimizer:", type(actor_optim).__name__)
        print("Actor LR:", actor_optim.param_groups[0]["lr"])
        print("Critic Optimizer:", type(critic_optim).__name__)
        print("Critic LR:", critic_optim.param_groups[0]["lr"])

        return True
