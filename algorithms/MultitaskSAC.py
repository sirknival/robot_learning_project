"""
Multi-Task SAC mit Shared Encoder + Separate Policy Heads
Bessere Alternative zu PCGrad für klar separierbare Tasks
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Optional, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor
from gymnasium import spaces
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    """
    Shared Feature Encoder für alle Tasks
    Lernt gemeinsame Repräsentationen
    """

    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        input_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class TaskSpecificHead(nn.Module):
    """
    Task-spezifischer Policy Head
    Erzeugt mean und log_std für Gaussian Policy
    """

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()

        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

        # Initialize weights
        self.mean.weight.data.uniform_(-3e-3, 3e-3)
        self.mean.bias.data.uniform_(-3e-3, 3e-3)
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)
        self.log_std.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilität
        return mean, log_std


class MultiHeadActor(nn.Module):
    """
    Multi-Task Actor mit Shared Encoder + Separate Heads

    Architecture:
        Observation → Shared Encoder → Task-Specific Head → Action
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            tasks: List[str],
            shared_hidden_dims: List[int] = [256, 256],
            action_space=None
    ):
        super().__init__()

        self.tasks = tasks
        self.action_dim = action_dim
        self.action_space = action_space

        # Shared Encoder
        self.shared_encoder = SharedEncoder(obs_dim, shared_hidden_dims)

        # Task-Specific Heads (ein Head pro Task)
        self.task_heads = nn.ModuleDict({
            task: TaskSpecificHead(self.shared_encoder.output_dim, action_dim)
            for task in tasks
        })

        print(f"[MultiHeadActor] Created with {len(tasks)} task-specific heads")
        print(f"[MultiHeadActor] Shared encoder: {obs_dim} → {shared_hidden_dims}")
        print(f"[MultiHeadActor] Each head: {shared_hidden_dims[-1]} → {action_dim}")

    def forward(
            self,
            obs: torch.Tensor,
            task_id: str,
            deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: Observations [batch_size, obs_dim]
            task_id: Which task head to use
            deterministic: If True, return mean action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        # Shared feature extraction
        features = self.shared_encoder(obs)

        # Task-specific head
        if task_id not in self.task_heads:
            raise ValueError(f"Unknown task: {task_id}")

        mean, log_std = self.task_heads[task_id](features)
        std = log_std.exp()

        # Sample action
        if deterministic:
            action = mean
        else:
            normal = Normal(mean, std)
            action = normal.rsample()  # Reparameterization trick

        # Compute log probability
        log_prob = Normal(mean, std).log_prob(action).sum(dim=-1, keepdim=True)

        # Apply tanh squashing
        if self.action_space is not None:
            action = torch.tanh(action)
            # Correct log_prob for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action_dist(self, obs: torch.Tensor, task_id: str):
        """Get action distribution for a specific task"""
        features = self.shared_encoder(obs)
        mean, log_std = self.task_heads[task_id](features)
        return Normal(mean, log_std.exp())

    def freeze_shared_encoder(self):
        """Freeze shared encoder (useful for fine-tuning)"""
        for param in self.shared_encoder.parameters():
            param.requires_grad = False
        print("[MultiHeadActor] Shared encoder frozen")

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("[MultiHeadActor] All parameters unfrozen")


class MultiHeadCritic(nn.Module):
    """
    Multi-Task Critic mit Shared Encoder + Separate Q-Heads
    Schätzt Q(s,a) für jeden Task separat
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            tasks: List[str],
            shared_hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()

        self.tasks = tasks

        # Shared Encoder für (obs, action)
        self.shared_encoder = SharedEncoder(
            obs_dim + action_dim,
            shared_hidden_dims
        )

        # Task-Specific Q-Value Heads
        self.task_q_heads = nn.ModuleDict({
            task: nn.Linear(self.shared_encoder.output_dim, 1)
            for task in tasks
        })

        print(f"[MultiHeadCritic] Created with {len(tasks)} Q-value heads")

    def forward(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            task_id: str
    ) -> torch.Tensor:
        """
        Compute Q(s,a) for specific task
        """
        # Concatenate obs and action
        obs_action = torch.cat([obs, action], dim=1)

        # Shared features
        features = self.shared_encoder(obs_action)

        # Task-specific Q-value
        if task_id not in self.task_q_heads:
            raise ValueError(f"Unknown task: {task_id}")

        q_value = self.task_q_heads[task_id](features)
        return q_value


class MultiTaskSACPolicy(SACPolicy):
    """
    Custom SAC Policy mit Multi-Task Heads

    Ersetzt den Standard Actor/Critic durch Multi-Head Varianten
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule,
            tasks: List[str],
            shared_encoder_dims: List[int] = [256, 256],
            **kwargs
    ):
        self.tasks = tasks
        self.shared_encoder_dims = shared_encoder_dims

        # Initialize parent (creates optimizers etc.)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

    def make_actor(self, features_extractor=None) -> MultiHeadActor:
        """Override to create Multi-Head Actor"""
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        actor = MultiHeadActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            tasks=self.tasks,
            shared_hidden_dims=self.shared_encoder_dims,
            action_space=self.action_space
        )
        return actor

    def make_critic(self, features_extractor=None) -> MultiHeadCritic:
        """Override to create Multi-Head Critic"""
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        critic = MultiHeadCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            tasks=self.tasks,
            shared_hidden_dims=self.shared_encoder_dims
        )
        return critic

    def forward(self, obs: torch.Tensor, task_id: str, deterministic: bool = False):
        """Forward pass through actor for specific task"""
        return self.actor(obs, task_id, deterministic)


class MultiTaskSAC(SAC):
    """
    SAC mit Multi-Task Policy Heads

    Features:
    - Shared encoder für gemeinsame Features
    - Separate policy/Q heads pro Task
    - Task-spezifische Replay Buffers
    - Automatisches Task-Routing
    """

    def __init__(
            self,
            policy,
            env,
            tasks: List[str],
            buffer_size_per_task: int = 100000,
            shared_encoder_dims: List[int] = [256, 256],
            **kwargs
    ):
        self.tasks = tasks
        self.shared_encoder_dims = shared_encoder_dims

        # Task-specific Replay Buffers
        self.task_buffers = {}

        # Initialize SAC
        super().__init__(policy, env, **kwargs)

        # Create task buffers after initialization
        for task in tasks:
            from stable_baselines3.common.buffers import ReplayBuffer
            self.task_buffers[task] = ReplayBuffer(
                buffer_size_per_task,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=1
            )

        print(f"[MultiTaskSAC] Initialized with {len(tasks)} tasks")
        print(f"[MultiTaskSAC] Shared encoder dims: {shared_encoder_dims}")

    def _setup_model(self):
        """Setup model with custom policy"""
        super()._setup_model()

        # Replace policy with MultiTask version
        self.policy = MultiTaskSACPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            tasks=self.tasks,
            shared_encoder_dims=self.shared_encoder_dims,
        ).to(self.device)

        # Setup optimizers
        self.policy.actor.optimizer = self.policy.optimizer_class(
            self.policy.actor.parameters(),
            lr=self.lr_schedule(1),
            **self.policy.optimizer_kwargs
        )

        self.policy.critic.optimizer = self.policy.optimizer_class(
            self.policy.critic.parameters(),
            lr=self.lr_schedule(1),
            **self.policy.optimizer_kwargs
        )

    def predict(
            self,
            observation: np.ndarray,
            task_id: str,
            state=None,
            episode_start=None,
            deterministic: bool = False
    ):
        """
        Predict action for specific task

        Args:
            observation: Current observation
            task_id: Which task head to use
            deterministic: If True, use mean action
        """

        observation = torch.as_tensor(observation).to(self.device)

        with torch.no_grad():
            action, _ = self.policy.actor(
                observation.unsqueeze(0),
                task_id,
                deterministic
            )

        action = action.cpu().numpy()[0]
        return action, state

    def add_to_buffer(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            task_id: str
    ):
        """Add transition to task-specific buffer"""
        if task_id not in self.task_buffers:
            raise ValueError(f"Unknown task: {task_id}")

        self.task_buffers[task_id].add(
            obs, next_obs, action,
            np.array([reward]),
            np.array([done]),
            [{}]
        )

    def train_on_task(self, task_id: str, gradient_steps: int):
        """
        Train on specific task

        Args:
            task_id: Task to train on
            gradient_steps: Number of gradient steps
        """
        if self.task_buffers[task_id].size() < self.batch_size:
            return {}

        actor_losses = []
        critic_losses = []

        for _ in range(gradient_steps):
            # Sample from task buffer
            replay_data = self.task_buffers[task_id].sample(
                self.batch_size,
                env=self._vec_normalize_env
            )

            # ===== CRITIC UPDATE =====
            with torch.no_grad():
                # Sample next actions
                next_actions, next_log_prob = self.policy.actor(
                    replay_data.next_observations,
                    task_id,
                    deterministic=False
                )

                # Compute target Q
                target_q1 = self.critic_target(
                    replay_data.next_observations,
                    next_actions,
                    task_id
                )
                target_q2 = self.critic_target(
                    replay_data.next_observations,
                    next_actions,
                    task_id
                )
                target_q = torch.min(target_q1, target_q2)
                target_q = target_q - self.ent_coef * next_log_prob

                # TD target
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Current Q estimates
            current_q1 = self.policy.critic(
                replay_data.observations,
                replay_data.actions,
                task_id
            )
            current_q2 = self.critic_target(
                replay_data.observations,
                replay_data.actions,
                task_id
            )

            # Critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            critic_losses.append(critic_loss.item())

            # Optimize critic
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.policy.critic.optimizer.step()

            # ===== ACTOR UPDATE =====
            actions_pi, log_prob = self.policy.actor(
                replay_data.observations,
                task_id,
                deterministic=False
            )

            q1_pi = self.policy.critic(
                replay_data.observations,
                actions_pi,
                task_id
            )
            q2_pi = self.critic_target(
                replay_data.observations,
                actions_pi,
                task_id
            )
            min_q_pi = torch.min(q1_pi, q2_pi)

            # Actor loss
            actor_loss = (self.ent_coef * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize actor
            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.actor.optimizer.step()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }

    def train_all_tasks(self, gradient_steps: int, active_tasks: Optional[List[str]] = None):
        """
        Train on all active tasks

        Args:
            gradient_steps: Steps per task
            active_tasks: List of tasks to train on (default: all)
        """
        if active_tasks is None:
            active_tasks = self.tasks

        all_losses = {}
        for task in active_tasks:
            losses = self.train_on_task(task, gradient_steps)
            all_losses[task] = losses

        return all_losses


# ============ USAGE EXAMPLE ============

def create_multitask_sac(env, tasks: List[str]):
    """
    Erstelle Multi-Task SAC Model

    Args:
        env: Gymnasium Environment
        tasks: Liste von Task-Namen
    """
    model = MultiTaskSAC(
        policy="MlpPolicy",  # Wird durch MultiTaskSACPolicy ersetzt
        env=env,
        tasks=tasks,
        buffer_size_per_task=100000,
        shared_encoder_dims=[256, 256],
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./multitask_logs/"
    )

    return model


def train_multitask_curriculum(env, tasks_by_stage: Dict[int, List[str]]):
    """
    Curriculum Training mit Multi-Task Heads

    Args:
        env: Environment
        tasks_by_stage: {stage: [task1, task2, ...]}
    """
    # Alle Tasks für die wir Heads brauchen
    all_tasks = list(set(sum(tasks_by_stage.values(), [])))

    # Erstelle Model mit Heads für ALLE Tasks
    model = create_multitask_sac(env, all_tasks)

    # Trainiere Stage für Stage
    for stage, active_tasks in tasks_by_stage.items():
        print(f"\n{'=' * 60}")
        print(f"Stage {stage}: Training on {active_tasks}")
        print(f"{'=' * 60}")

        for timestep in range(100000):  # 100k steps per stage
            # Sample Task aus active_tasks
            task_id = np.random.choice(active_tasks)

            # Collect rollout für diesen Task
            obs = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, task_id, deterministic=False)
                next_obs, reward, done, info = env.step(action)

                # Speichere in task buffer
                model.add_to_buffer(obs, next_obs, action, reward, done, task_id)

                obs = next_obs

            # Train on all active tasks
            if timestep % 100 == 0:
                losses = model.train_all_tasks(
                    gradient_steps=10,
                    active_tasks=active_tasks
                )

        # Save checkpoint
        model.save(f"stage_{stage}_checkpoint.zip")

    return model