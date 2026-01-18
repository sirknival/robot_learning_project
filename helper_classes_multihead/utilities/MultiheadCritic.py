import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import metaworld

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp


# ------------------------------------------------------------
# Multi-Head Critic
# ------------------------------------------------------------
class MultiHeadContinuousCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch=(256, 256, 256),
        activation_fn=nn.ReLU,
        n_tasks: int = 3,
        task_id_slice=slice(-3, None),
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = int(features_dim)

        self.n_tasks = int(n_tasks)
        self.task_id_slice = task_id_slice

        action_dim = action_space.shape[0]
        in_dim = self.features_dim + action_dim

        self.q1_heads = nn.ModuleList()
        self.q2_heads = nn.ModuleList()
        for _ in range(self.n_tasks):
            self.q1_heads.append(nn.Sequential(*create_mlp(in_dim, 1, list(net_arch), activation_fn)))
            self.q2_heads.append(nn.Sequential(*create_mlp(in_dim, 1, list(net_arch), activation_fn)))

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def _task_indices_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        onehot = obs[..., self.task_id_slice]
        if onehot.shape[-1] != self.n_tasks:
            raise RuntimeError(f"Task one-hot dim={onehot.shape[-1]} but expected {self.n_tasks}")
        return torch.argmax(onehot, dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.features_extractor(obs)
        x = torch.cat([features, actions], dim=1)
        task_idx = self._task_indices_from_obs(obs)

        q1 = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        q2 = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

        for t in range(self.n_tasks):
            mask = task_idx == t
            if torch.any(mask):
                xt = x[mask]
                q1[mask] = self.q1_heads[t](xt)
                q2[mask] = self.q2_heads[t](xt)
        return q1, q2

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, _ = self.forward(obs, actions)
        return q1

    def q2_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _, q2 = self.forward(obs, actions)
        return q2


class MultiHeadSACPolicy(SACPolicy):
    def __init__(self, *args, n_tasks=3, task_id_slice=slice(-3, None), **kwargs):
        self._mh_n_tasks = int(n_tasks)
        self._mh_task_id_slice = task_id_slice
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor = None):
        if features_extractor is None:
            features_extractor = self.make_features_extractor()

        return MultiHeadContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            n_tasks=self._mh_n_tasks,
            task_id_slice=self._mh_task_id_slice,
        )
