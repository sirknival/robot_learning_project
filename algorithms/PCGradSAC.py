"""
PCGrad-Enhanced SAC für Multi-Task Learning
Integration in dein bestehendes Stable-Baselines3 Setup
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.callbacks import BaseCallback
import copy


class PCGradSAC(SAC):
    """
    SAC mit Project Conflicting Gradients (PCGrad) für Multi-Task Learning

    Erweitert normale SAC um:
    - Task-spezifische Replay Buffers
    - Gradient Projection bei Konflikten
    - Task-balanced Sampling
    """

    def __init__(
            self,
            policy,
            env: GymEnv,
            tasks: List[str],
            buffer_size_per_task: int = 100000,
            use_pcgrad: bool = True,
            **kwargs
    ):
        # Initialisiere basis SAC
        super().__init__(policy, env, **kwargs)

        self.tasks = tasks
        self.use_pcgrad = use_pcgrad

        # Erstelle separate Replay Buffers pro Task
        self.task_buffers = {}
        for task in tasks:
            self.task_buffers[task] = ReplayBuffer(
                buffer_size_per_task,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=1
            )

        print(f"[PCGrad-SAC] Initialized with {len(tasks)} tasks")
        print(f"[PCGrad-SAC] PCGrad: {'ENABLED' if use_pcgrad else 'DISABLED'}")

    def add_to_buffer(self,
                      obs: np.ndarray,
                      next_obs: np.ndarray,
                      action: np.ndarray,
                      reward: np.ndarray,
                      done: np.ndarray,
                      task_id: str):
        """Add transition - handles both batched and unbatched data"""

        # Detect if batched (from VecEnv)
        if isinstance(action, np.ndarray) and action.ndim == 2:
            # Batched: Loop through each env
            batch_size = action.shape[0]
            for i in range(batch_size):
                self.task_buffers[task_id].add(
                    obs[i:i + 1] if obs.ndim > 1 else obs.reshape(1, -1),
                    next_obs[i:i + 1] if next_obs.ndim > 1 else next_obs.reshape(1, -1),
                    action[i:i + 1],
                    np.array([reward[i]]) if hasattr(reward, '__len__') else np.array([reward]),
                    np.array([done[i]]) if hasattr(done, '__len__') else np.array([done]),
                    [{}]
                )
        else:
            # Unbatched: Add directly
            obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
            next_obs = next_obs.reshape(1, -1) if next_obs.ndim == 1 else next_obs
            action = action.reshape(1, -1) if action.ndim == 1 else action
            reward = np.array([reward]) if np.isscalar(reward) else reward
            done = np.array([done]) if np.isscalar(done) else done

            self.task_buffers[task_id].add(
                obs, next_obs, action, reward, done, [{}]
            )

    def sample_multitask_batch(self,
                               batch_size: int,
                               active_tasks: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Sample gleichmäßig aus allen aktiven Task-Buffern

        Returns:
            Dict mit {task_id: replay_data} für jeden Task
        """
        if active_tasks is None:
            active_tasks = self.tasks

        # Filter nur Tasks mit genug Daten
        available_tasks = [
            task for task in active_tasks
            if self.task_buffers[task].size() >= batch_size // len(active_tasks)
        ]

        if len(available_tasks) == 0:
            return {}

        samples_per_task = batch_size // len(available_tasks)
        task_batches = {}

        for task in available_tasks:
            task_batches[task] = self.task_buffers[task].sample(
                samples_per_task,
                env=self._vec_normalize_env
            )

        return task_batches

    def project_conflicting_gradients(self,
                                      task_gradients: Dict[str, List[torch.Tensor]]
                                      ) -> Dict[str, List[torch.Tensor]]:
        """
        PCGrad: Projiziere Gradienten, die in Konflikt stehen

        Wenn grad_i · grad_j < 0:
            grad_i = grad_i - proj(grad_i onto grad_j)
        """
        tasks = list(task_gradients.keys())

        # Erstelle Kopien, um Original nicht zu modifizieren
        projected_grads = {
            task: [g.clone() if g is not None else None for g in grads]
            for task, grads in task_gradients.items()
        }

        # Für jedes Task-Paar
        for i, task_i in enumerate(tasks):
            for j, task_j in enumerate(tasks):
                if i >= j:  # Nur eine Richtung prüfen
                    continue

                grads_i = projected_grads[task_i]
                grads_j = task_gradients[task_j]  # Original für Projektion

                # Für jedes Parameter-Paar
                for idx, (g_i, g_j) in enumerate(zip(grads_i, grads_j)):
                    if g_i is None or g_j is None:
                        continue

                    # Flatten für Dot Product
                    g_i_flat = g_i.flatten()
                    g_j_flat = g_j.flatten()

                    # Berechne Cosine Similarity
                    dot_product = torch.dot(g_i_flat, g_j_flat)

                    # Wenn konfliktär (negatives Dot Product)
                    if dot_product < 0:
                        # Projiziere g_i weg von g_j
                        # proj = (g_i · g_j / ||g_j||²) * g_j
                        g_j_norm_sq = torch.dot(g_j_flat, g_j_flat)
                        projection = (dot_product / (g_j_norm_sq + 1e-8)) * g_j

                        # g_i = g_i - projection
                        grads_i[idx].sub_(projection.view_as(g_i))

        return projected_grads

    def compute_task_actor_loss(self,
                                replay_data,
                                task_id: str) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Berechne Actor Loss und Gradienten für einen Task"""
        # Sample actions from current policy
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        # Min Q-value über beide Critics
        qf1_pi = self.critic(replay_data.observations, actions_pi)
        qf2_pi = self.critic_target(replay_data.observations, actions_pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Actor Loss: α * log π - Q
        actor_loss = (self.ent_coef * log_prob - min_qf_pi).mean()

        # Berechne Gradienten
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        # Extrahiere Gradienten
        gradients = [
            param.grad.clone() if param.grad is not None else None
            for param in self.actor.parameters()
        ]

        # Setze Gradienten zurück (wir applyen sie später)
        self.actor.optimizer.zero_grad()

        return actor_loss, gradients

    def compute_task_critic_loss(self, replay_data) -> torch.Tensor:
        """Standard SAC Critic Loss"""
        with torch.no_grad():
            # Sample actions from current policy
            next_actions, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )

            # Compute target Q-value
            target_q1 = self.critic_target(
                replay_data.next_observations, next_actions
            )
            target_q2 = self.critic_target(
                replay_data.next_observations, next_actions
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.ent_coef * next_log_prob.reshape(-1, 1)

            # TD target
            target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

        # Current Q estimates
        current_q1 = self.critic(replay_data.observations, replay_data.actions)
        current_q2 = self.critic_target(replay_data.observations, replay_data.actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        return critic_loss

    def train_step_multitask(self,
                             gradient_steps: int,
                             active_tasks: Optional[List[str]] = None):
        """
        Multi-Task Training mit optionalem PCGrad
        """
        if active_tasks is None:
            active_tasks = self.tasks

        actor_losses = []
        critic_losses = []

        for gradient_step in range(gradient_steps):
            # Sample Batches von allen Tasks
            task_batches = self.sample_multitask_batch(
                self.batch_size,
                active_tasks
            )

            if len(task_batches) == 0:
                continue

            # ============ CRITIC UPDATE ============
            # Critic wird normal auf allen Tasks trainiert
            total_critic_loss = 0
            for task_id, replay_data in task_batches.items():
                critic_loss = self.compute_task_critic_loss(replay_data)
                total_critic_loss += critic_loss

            # Optimizer step für Critics
            self.critic.optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(total_critic_loss.item())

            # ============ ACTOR UPDATE mit PCGrad ============
            task_actor_losses = {}
            task_gradients = {}

            # Berechne Gradienten pro Task
            for task_id, replay_data in task_batches.items():
                actor_loss, gradients = self.compute_task_actor_loss(
                    replay_data, task_id
                )
                task_actor_losses[task_id] = actor_loss
                task_gradients[task_id] = gradients

            # PCGrad: Projiziere konfliktäre Gradienten
            if self.use_pcgrad:
                task_gradients = self.project_conflicting_gradients(task_gradients)

            # Mittele Gradienten über alle Tasks
            avg_gradients = self._average_gradients(task_gradients)

            # Apply gemittelte Gradienten
            self._apply_gradients(avg_gradients)

            # Logging
            avg_actor_loss = sum(task_actor_losses.values()) / len(task_actor_losses)
            actor_losses.append(avg_actor_loss.item())

            # ============ UPDATE TARGET NETWORKS ============
            if gradient_step % self.target_update_interval == 0:
                self._update_target_networks()

        # Update Entropy Coefficient
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        return {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'critic_loss': np.mean(critic_losses) if critic_losses else 0
        }

    def _average_gradients(self,
                           task_gradients: Dict[str, List[torch.Tensor]]
                           ) -> List[torch.Tensor]:
        """Mittele Gradienten über alle Tasks"""
        n_tasks = len(task_gradients)

        # Initialisiere mit None
        avg_grads = None

        for task_id, gradients in task_gradients.items():
            if avg_grads is None:
                avg_grads = [
                    g.clone() / n_tasks if g is not None else None
                    for g in gradients
                ]
            else:
                for i, g in enumerate(gradients):
                    if g is not None and avg_grads[i] is not None:
                        avg_grads[i] += g / n_tasks

        return avg_grads

    def _apply_gradients(self, gradients: List[torch.Tensor]):
        """Setze berechnete Gradienten und führe Optimizer Step aus"""
        for param, grad in zip(self.actor.parameters(), gradients):
            if grad is not None:
                param.grad = grad

        # Optimizer step
        self.actor.optimizer.step()
        self.actor.optimizer.zero_grad()

    def _update_target_networks(self):
        """Soft update der Target Networks"""
        for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class PCGradCallback(BaseCallback):
    """
    Callback für PCGrad-SAC Training mit Multi-Task Environments
    """

    def __init__(
            self,
            tasks: List[str],
            gradient_steps_per_rollout: int = 1,
            verbose: int = 0
    ):
        super().__init__(verbose)
        self.tasks = tasks
        self.gradient_steps = gradient_steps_per_rollout
        self.current_task_idx = 0

    def _on_step(self) -> bool:
        """Called at each environment step"""
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Hole aktive Tasks für dieses Stage
        active_tasks = self._get_active_tasks()

        # Führe Multi-Task Training aus
        if isinstance(self.model, PCGradSAC):
            losses = self.model.train_step_multitask(
                gradient_steps=self.gradient_steps,
                active_tasks=active_tasks
            )

            # Log losses
            self.logger.record("train/actor_loss", losses['actor_loss'])
            self.logger.record("train/critic_loss", losses['critic_loss'])

    def _get_active_tasks(self) -> List[str]:
        """Bestimme welche Tasks aktuell aktiv sind (für Curriculum)"""
        # Hier kannst du Curriculum Logic einfügen
        # Beispiel: Alle Tasks
        return self.tasks


class PCGradTrainer:
    def __init__(self, model, env, tasks: List[str]):
        self.model = model
        self.env = env
        self.tasks = tasks
        self.current_task_idx = 0

    def collect_rollout(self, n_steps: int = 2048):
        """
        Sammle Rollouts von allen aktiven Tasks
        """
        obs = self.env.reset()

        for step in range(n_steps):
            # Wähle Task (rotiere durch Tasks)
            task_id = self.tasks[self.current_task_idx % len(self.tasks)]

            # Sample Action
            action, _ = self.model.predict(obs, deterministic=False)

            # Environment Step
            next_obs, reward, done, info = self.env.step(action)

            # Speichere in task-spezifischem Buffer
            self.model.add_to_buffer(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=np.array([reward]),
                done=np.array([done]),
                task_id=task_id
            )

            obs = next_obs

            if done:
                obs = self.env.reset()
                self.current_task_idx += 1

    def train(self, total_timesteps: int, rollout_steps: int = 2048):
        """
        Haupt-Training Loop
        """
        n_rollouts = total_timesteps // rollout_steps

        for rollout in range(n_rollouts):
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

# ============ USAGE EXAMPLE ============
def create_pcgrad_sac_model(env, tasks, use_pcgrad=True):
    """
    Erstelle PCGrad-enhanced SAC Model

    Args:
        env: Gymnasium/MetaWorld Environment
        tasks: Liste von Task-Namen
        use_pcgrad: Ob PCGrad verwendet werden soll
    """
    model = PCGradSAC(
        policy="MlpPolicy",
        env=env,
        tasks=tasks,
        buffer_size_per_task=100000,
        use_pcgrad=use_pcgrad,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="./pcgrad_logs/"
    )

    return model


def train_with_pcgrad(env, tasks, total_timesteps=1000000):
    """
    Trainiere mit PCGrad
    """
    # Erstelle Model
    model = create_pcgrad_sac_model(env, tasks, use_pcgrad=True)

    # Erstelle Callback
    callback = PCGradCallback(
        tasks=tasks,
        gradient_steps_per_rollout=1,
        verbose=1
    )

    # Training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=4
    )

    return model


if __name__ == "__main__":
    # Beispiel-Verwendung
    print("PCGrad-SAC Implementation für Multi-Task Learning")
    print("=" * 60)
    print("\nIntegrations-Schritte:")
    print("1. Ersetze 'from stable_baselines3 import SAC' mit PCGradSAC")
    print("2. Übergebe tasks Liste beim Erstellen des Models")
    print("3. Nutze PCGradCallback für automatisches Multi-Task Training")
    print("4. Verwende model.add_to_buffer() mit task_id beim Sammeln von Daten")