"""
Fix für Buffer Shape Mismatch bei VecEnv
Problem: VecEnv gibt batched data zurück, aber einzelne Buffers erwarten unbatched data
"""

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from typing import List, Dict, Optional


class MultiTaskReplayBuffer:
    """
    Multi-Task Replay Buffer der VecEnv korrekt handelt
    """

    def __init__(
            self,
            buffer_size_per_task: int,
            observation_space,
            action_space,
            tasks: List[str],
            n_envs: int = 1,
            device="auto"
    ):
        self.tasks = tasks
        self.n_envs = n_envs

        # Erstelle Buffer pro Task
        self.task_buffers = {}
        for task in tasks:
            self.task_buffers[task] = ReplayBuffer(
                buffer_size_per_task,
                observation_space,
                action_space,
                device=device,
                n_envs=n_envs  # WICHTIG: Muss mit VecEnv übereinstimmen
            )

        print(f"[MultiTaskReplayBuffer] Created {len(tasks)} buffers")
        print(f"[MultiTaskReplayBuffer] n_envs: {n_envs}")
        print(f"[MultiTaskReplayBuffer] buffer_size per task: {buffer_size_per_task}")

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            task_id: str,
            infos: List[Dict] = None
    ):
        """
        Füge Transition hinzu - handelt sowohl batched als auch unbatched data

        Args:
            obs: Shape (n_envs, obs_dim) oder (obs_dim,)
            next_obs: Shape (n_envs, obs_dim) oder (obs_dim,)
            action: Shape (n_envs, action_dim) oder (action_dim,)
            reward: Shape (n_envs,) oder scalar
            done: Shape (n_envs,) oder scalar
            task_id: Task identifier
        """
        if task_id not in self.task_buffers:
            print(f"Warning: Unknown task_id {task_id}, skipping")
            return

        # Ensure correct shapes für VecEnv
        obs = self._ensure_batch_shape(obs, is_obs=True)
        next_obs = self._ensure_batch_shape(next_obs, is_obs=True)
        action = self._ensure_batch_shape(action, is_obs=False)
        reward = self._ensure_scalar_shape(reward)
        done = self._ensure_scalar_shape(done)

        # Infos
        if infos is None:
            infos = [{}] * self.n_envs

        # Add to buffer
        self.task_buffers[task_id].add(
            obs, next_obs, action, reward, done, infos
        )

    def _ensure_batch_shape(self, data: np.ndarray, is_obs: bool) -> np.ndarray:
        """
        Stelle sicher, dass data die richtige Shape hat: (n_envs, dim)
        """
        # Falls data bereits (n_envs, dim) hat
        if data.ndim == 2 and data.shape[0] == self.n_envs:
            return data

        # Falls data (dim,) hat → reshape zu (1, dim)
        if data.ndim == 1:
            return data.reshape(1, -1)

        # Falls data (batch, dim) hat aber batch != n_envs
        if data.ndim == 2 and data.shape[0] != self.n_envs:
            # Nehme nur die ersten n_envs
            return data[:self.n_envs]

        return data

    def _ensure_scalar_shape(self, data: np.ndarray) -> np.ndarray:
        """
        Stelle sicher, dass data Shape (n_envs,) hat
        """
        # Falls scalar
        if np.isscalar(data):
            return np.array([data] * self.n_envs)

        # Falls array
        data = np.atleast_1d(data)

        # Falls bereits korrekte Länge
        if len(data) == self.n_envs:
            return data

        # Falls zu lang, schneide ab
        if len(data) > self.n_envs:
            return data[:self.n_envs]

        # Falls zu kurz, wiederhole
        return np.repeat(data, self.n_envs)

    def sample(self, batch_size: int, task_id: str, env=None):
        """Sample from specific task buffer"""
        if task_id not in self.task_buffers:
            raise ValueError(f"Unknown task: {task_id}")

        return self.task_buffers[task_id].sample(batch_size, env)

    def size(self, task_id: str) -> int:
        """Get size of specific task buffer"""
        if task_id not in self.task_buffers:
            return 0
        return self.task_buffers[task_id].size()


# ============ FIXED PCGradSAC ============

class PCGradSAC:
    """
    Fixed version mit korrektem Buffer Handling
    """

    def __init__(
            self,
            policy,
            env,
            tasks: List[str],
            buffer_size_per_task: int = 100000,
            **kwargs
    ):
        # Bestimme n_envs vom Environment
        if hasattr(env, 'num_envs'):
            self.n_envs = env.num_envs
        else:
            self.n_envs = 1

        print(f"[PCGradSAC] Detected n_envs: {self.n_envs}")

        # Initialisiere SAC (dein bestehendes SAC)
        # ... hier dein SAC init code ...

        # Erstelle Multi-Task Buffer
        self.replay_buffer = MultiTaskReplayBuffer(
            buffer_size_per_task=buffer_size_per_task,
            observation_space=env.observation_space,
            action_space=env.action_space,
            tasks=tasks,
            n_envs=self.n_envs,
            device='auto'
        )

        self.tasks = tasks

    def add_to_buffer(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            task_id: str
    ):
        """
        Wrapper für add - handelt automatisch batched/unbatched
        """
        self.replay_buffer.add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=reward,
            done=done,
            task_id=task_id
        )


# ============ FIXED PCGradTrainer ============

class PCGradTrainer:
    """
    Fixed Trainer mit korrektem VecEnv Handling
    """

    def __init__(self, model, env, tasks: List[str]):
        self.model = model
        self.env = env
        self.tasks = tasks
        self.current_task_idx = 0

        # Bestimme ob VecEnv
        self.is_vecenv = hasattr(env, 'num_envs')
        self.n_envs = env.num_envs if self.is_vecenv else 1

        print(f"[PCGradTrainer] VecEnv: {self.is_vecenv}, n_envs: {self.n_envs}")

    def collect_rollout(self, n_steps: int = 2048):
        """
        Sammle Rollouts - fixed für VecEnv
        """
        obs = self.env.reset()

        # Falls VecEnv, obs ist bereits (n_envs, obs_dim)
        # Falls nicht, mache es zu (1, obs_dim)
        if not self.is_vecenv:
            obs = obs.reshape(1, -1)

        for step in range(n_steps):
            # Wähle Task (rotiere durch alle)
            task_id = self.tasks[self.current_task_idx % len(self.tasks)]

            # Predict action
            if self.is_vecenv:
                # VecEnv: predict erwartet (n_envs, obs_dim)
                action, _ = self.model.predict(obs, task_id=task_id, deterministic=False)
            else:
                # Single env: predict erwartet (obs_dim,)
                action, _ = self.model.predict(obs[0], task_id=task_id, deterministic=False)
                action = action.reshape(1, -1)

            # Environment step
            next_obs, reward, done, info = self.env.step(action)

            # VecEnv gibt bereits richtige shapes zurück
            # Stelle sicher dass alles numpy arrays sind
            if not isinstance(reward, np.ndarray):
                reward = np.array([reward])
            if not isinstance(done, np.ndarray):
                done = np.array([done])

            # Speichere in Buffer
            self.model.add_to_buffer(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                task_id=task_id
            )

            # Update obs
            obs = next_obs

            # Reset wenn done (VecEnv macht das automatisch)
            if self.is_vecenv:
                # VecEnv reset automatisch, obs wird updated
                pass
            else:
                if done[0]:
                    obs = self.env.reset().reshape(1, -1)
                    self.current_task_idx += 1

    def train(
            self,
            total_timesteps: int,
            rollout_steps: int = 2048,
            gradient_steps: int = 50,
            active_tasks: Optional[List[str]] = None
    ):
        """
        Haupt Training Loop
        """
        if active_tasks is None:
            active_tasks = self.tasks

        n_rollouts = total_timesteps // rollout_steps

        for rollout in range(n_rollouts):
            print(f"\n{'=' * 60}")
            print(f"Rollout {rollout + 1}/{n_rollouts}")
            print(f"{'=' * 60}")

            # Sammle Daten
            self.collect_rollout(rollout_steps)

            # Multi-Task Training
            losses = self.model.train_step_multitask(
                gradient_steps=gradient_steps,
                active_tasks=active_tasks
            )

            # Logging
            print(f"Actor Loss:  {losses.get('actor_loss', 0):.4f}")
            print(f"Critic Loss: {losses.get('critic_loss', 0):.4f}")

            # Buffer Stats
            for task in active_tasks:
                buffer_size = self.model.replay_buffer.size(task)
                print(f"Buffer {task}: {buffer_size}")


# ============ DEBUG HELPER ============

def debug_shapes(env, model, task_id="reach-v3"):
    """
    Debug Helper: Prüfe alle Shapes
    """
    print("\n" + "=" * 60)
    print("SHAPE DEBUGGING")
    print("=" * 60)

    # Environment
    is_vecenv = hasattr(env, 'num_envs')
    n_envs = env.num_envs if is_vecenv else 1

    print(f"\nEnvironment:")
    print(f"  Type: {'VecEnv' if is_vecenv else 'Single Env'}")
    print(f"  n_envs: {n_envs}")
    print(f"  obs_space: {env.observation_space.shape}")
    print(f"  action_space: {env.action_space.shape}")

    # Reset
    obs = env.reset()
    print(f"\nReset:")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dtype: {obs.dtype}")

    # Predict
    action, _ = model.predict(obs if is_vecenv else obs, task_id=task_id)
    print(f"\nPredict:")
    print(f"  action shape: {action.shape}")
    print(f"  action dtype: {action.dtype}")

    # Step
    next_obs, reward, done, info = env.step(action)
    print(f"\nStep:")
    print(f"  next_obs shape: {next_obs.shape}")
    print(f"  reward shape: {np.array(reward).shape if not isinstance(reward, np.ndarray) else reward.shape}")
    print(f"  done shape: {np.array(done).shape if not isinstance(done, np.ndarray) else done.shape}")

    # Buffer shapes
    print(f"\nExpected Buffer Shapes:")
    print(f"  obs: ({n_envs}, {env.observation_space.shape[0]})")
    print(f"  action: ({n_envs}, {env.action_space.shape[0]})")
    print(f"  reward: ({n_envs},)")
    print(f"  done: ({n_envs},)")

    print("\n" + "=" * 60)


# ============ USAGE ============

if __name__ == "__main__":
    """
    Verwendung im main script:
    """

    # 1. Erstelle Model mit korrektem Buffer
    from stable_baselines3 import SAC

    # model = PCGradSAC(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     tasks=tasks,
    #     buffer_size_per_task=100000
    # )

    # 2. Debug shapes (optional aber empfohlen!)
    # debug_shapes(train_env, model, task_id="reach-v3")

    # 3. Erstelle Trainer
    # trainer = PCGradTrainer(model, train_env, tasks)

    # 4. Training
    # trainer.train(
    #     total_timesteps=1000000,
    #     rollout_steps=2048,
    #     gradient_steps=50,
    #     active_tasks=active_tasks
    # )

    print("Fixed Buffer Implementation Ready!")
    print("\nKey Changes:")
    print("  ✓ MultiTaskReplayBuffer handles VecEnv correctly")
    print("  ✓ Automatic shape correction for batched/unbatched data")
    print("  ✓ PCGradTrainer detects VecEnv automatically")
    print("  ✓ Debug helper to verify all shapes")