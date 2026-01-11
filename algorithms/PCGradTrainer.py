"""
Custom Training Loop für PCGrad-SAC
"""
import numpy as np
from typing import List, Optional


class PCGradTrainer:
    def __init__(self, model, env, tasks: List[str]):
        self.model = model
        self.env = env
        self.tasks = tasks
        self.current_task_idx = 0

    def collect_rollout(self, n_steps: int = 2048):
        """Fixed for VecEnv with multiple parallel environments"""

        # Detect VecEnv
        is_vecenv = hasattr(self.env, 'num_envs')
        n_envs = self.env.num_envs if is_vecenv else 1

        obs = self.env.reset()
        episode_count = 0

        for step in range(n_steps):
            # Sample task ID
            task_id = self.tasks[self.current_task_idx % len(self.tasks)]

            # Predict action
            action, _ = self.model.predict(obs, deterministic=False)

            # Environment step
            next_obs, reward, done, info = self.env.step(action)

            # Handle VecEnv vs Single Env
            if is_vecenv:
                # VecEnv: Loop through each parallel env
                for env_idx in range(n_envs):
                    # Get task_id for this specific env
                    # Option A: Alle envs haben gleichen Task
                    current_task = task_id

                    # Option B: Jeder env hat anderen Task (falls du das willst)
                    # current_task = self.tasks[env_idx % len(self.tasks)]

                    self.model.add_to_buffer(
                        obs=obs[env_idx],
                        next_obs=next_obs[env_idx],
                        action=action[env_idx],
                        reward=reward[env_idx],
                        done=done[env_idx],
                        task_id=current_task
                    )

                    # Count episodes
                    if done[env_idx]:
                        episode_count += 1
            else:
                # Single Env
                self.model.add_to_buffer(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    task_id=task_id
                )

                if done:
                    episode_count += 1
                    obs = self.env.reset()

            # Update obs (VecEnv does auto-reset)
            obs = next_obs

            # Switch task every N episodes
            if episode_count > 0 and episode_count % 5 == 0:
                self.current_task_idx += 1

        print(f"Collected {episode_count} episodes in rollout")

    def train(self, total_timesteps: int, rollout_steps: int = 2048):
        """
        Haupt-Training Loop
        """
        n_rollouts = total_timesteps // rollout_steps

        for rollout in range(int(n_rollouts)):
            # Sammle Daten
            self.collect_rollout(rollout_steps)

            # Multi-Task Training mit PCGrad
            losses = self.model.train_step_multitask(
                gradient_steps=rollout_steps,
                active_tasks=self.tasks
            )

            # Logging
            if rollout % 10 == 0:
                print(f"Rollout {rollout}/{n_rollouts}")
                print(f"  Actor Loss: {losses['actor_loss']:.4f}")
                print(f"  Critic Loss: {losses['critic_loss']:.4f}")