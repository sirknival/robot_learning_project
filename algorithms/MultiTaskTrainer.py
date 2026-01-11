"""
Custom Trainer für Multi-Task SAC
"""
import numpy as np
from typing import List, Optional


class MultiTaskTrainer:
    def __init__(self, model, env, tasks: List[str], task_sampler="uniform"):
        self.model = model
        self.env = env
        self.tasks = tasks
        self.task_sampler = task_sampler
        self.step_count = 0

    def sample_task(self) -> str:
        """Sample welcher Task als nächstes trainiert wird"""
        if self.task_sampler == "uniform":
            return np.random.choice(self.tasks)
        elif self.task_sampler == "cyclic":
            task = self.tasks[self.step_count % len(self.tasks)]
            self.step_count += 1
            return task
        else:
            raise ValueError(f"Unknown sampler: {self.task_sampler}")

    def collect_rollout(self, n_episodes: int = 10):
        """Sammle Rollouts von allen Tasks"""
        for episode in range(n_episodes):
            # Sample Task
            task_id = self.sample_task()

            # Reset Environment für diesen Task
            obs = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Predict Action mit task-spezifischem Head
                action, _ = self.model.predict(
                    obs,
                    task_id=task_id,
                    deterministic=False
                )

                # Environment Step
                next_obs, reward, done, info = self.env.step(action)
                episode_reward += reward

                # Speichere in task-spezifischem Buffer
                self.model.add_to_buffer(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    task_id=task_id
                )

                obs = next_obs

            if episode % 10 == 0:
                print(f"Task {task_id}: Episode Reward = {episode_reward:.2f}")

    def train(
            self,
            total_timesteps: int,
            episodes_per_update: int = 10,
            gradient_steps_per_update: int = 50,
            active_tasks: Optional[List[str]] = None
    ):
        """
        Haupt Training Loop

        Args:
            total_timesteps: Gesamt Steps
            episodes_per_update: Episoden sammeln vor Update
            gradient_steps_per_update: Gradient steps pro Task
            active_tasks: Welche Tasks trainieren (für Curriculum)
        """
        if active_tasks is None:
            active_tasks = self.tasks

        timesteps_done = 0
        update_count = 0

        while timesteps_done < total_timesteps:
            # Sammle Rollouts
            self.collect_rollout(n_episodes=episodes_per_update)

            # Train auf allen aktiven Tasks
            losses = self.model.train_all_tasks(
                gradient_steps=gradient_steps_per_update,
                active_tasks=active_tasks
            )

            # Logging
            update_count += 1
            timesteps_done = update_count * episodes_per_update * 500  # ~500 steps/episode

            if update_count % 10 == 0:
                print(f"\n{'=' * 60}")
                print(f"Update {update_count} | Steps: {timesteps_done}/{total_timesteps}")
                for task, task_losses in losses.items():
                    if task_losses:
                        print(f"  {task}:")
                        print(f"    Actor Loss:  {task_losses['actor_loss']:.4f}")
                        print(f"    Critic Loss: {task_losses['critic_loss']:.4f}")
                print(f"{'=' * 60}")

            # Checkpoint
            if update_count % 100 == 0:
                self.model.save(f"checkpoint_{timesteps_done}.zip")

        return self.model