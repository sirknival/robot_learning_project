import gymnasium as gym
import metaworld
from typing import List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from training_setup_multitask.utilities.MetaworldTasks import MT3_TASKS, MT10_TASKS


class SB3MetaWorldEnvFactory:
    """
    SB3-only Meta-World Environment Factory
    - MT1 / MT3 / MT10 / Custom / Curriculum
    """

    def __init__(
        self,
        reward_function_version: str = "v3",
        terminate_on_success: bool = False,
        use_subproc: bool = True,
        verbose: bool = True,
    ):
        self.reward_function_version = reward_function_version
        self.terminate_on_success = terminate_on_success
        self.use_subproc = use_subproc
        self.verbose = verbose

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EnvFactory] {msg}")

    def _vec_cls(self, n_envs: int):
        if self.use_subproc and n_envs > 1:
            return SubprocVecEnv
        return DummyVecEnv

    # ------------------------------------------------------------------
    # base env builders
    # ------------------------------------------------------------------

    def _make_mt1_env_fn(
        self,
        task: str,
        seed: int,
        max_episode_steps: int,
        rank: int,
    ):
        def _init():
            env = gym.make(
                "Meta-World/MT1",
                env_name=task,
                seed=seed + rank,
                reward_function_version=self.reward_function_version,
                max_episode_steps=max_episode_steps,
                terminate_on_success=self.terminate_on_success,
            )
            return env
        return _init

    def _make_multitask_env_fn(
        self,
        tasks: List[str],
        seed: int,
        max_episode_steps: int,
        rank: int,
    ):
        def _init():
            env = gym.make(
                "Meta-World/custom-mt-envs",
                envs_list=tasks,
                seed=seed + rank,
                reward_function_version=self.reward_function_version,
                max_episode_steps=max_episode_steps,
                terminate_on_success=self.terminate_on_success,
            )
            return env
        return _init

    # ------------------------------------------------------------------
    # MT1
    # ------------------------------------------------------------------

    def make_mt1_env(
        self,
        task: str,
        seed: int = 0,
        max_episode_steps: int = 500,
        n_envs: int = 1,
    ):
        self._log(f"Creating MT1 env: {task} ({n_envs} envs)")

        env_fns = [
            self._make_mt1_env_fn(task, seed, max_episode_steps, i)
            for i in range(n_envs)
        ]

        env = self._vec_cls(n_envs)(env_fns)
        return VecMonitor(env)

    # ------------------------------------------------------------------
    # MT3 / MT10 / Custom
    # ------------------------------------------------------------------

    def make_multitask_env(
        self,
        tasks: List[str],
        seed: int = 0,
        max_episode_steps: int = 500,
        n_envs: int = 1,
    ):
        self._log(f"Creating multitask env ({len(tasks)} tasks, {n_envs} envs)")
        self._log(f"Tasks: {tasks}")

        env_fns = [
            self._make_multitask_env_fn(tasks, seed, max_episode_steps, i)
            for i in range(n_envs)
        ]

        env = self._vec_cls(n_envs)(env_fns)
        return VecMonitor(env)

    def make_mt3_env(
        self,
        seed: int = 0,
        max_episode_steps: int = 500,
        tasks: Optional[List[str]] = None,
        n_envs: int = 1,
    ):
        return self.make_multitask_env(
            tasks=tasks if tasks else MT3_TASKS,
            seed=seed,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
        )

    def make_mt10_env(
        self,
        seed: int = 0,
        max_episode_steps: int = 500,
        n_envs: int = 1,
    ):
        return self.make_multitask_env(
            tasks=MT10_TASKS,
            seed=seed,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
        )

    # ------------------------------------------------------------------
    # curriculum
    # ------------------------------------------------------------------

    def make_curriculum_env(
        self,
        stage_tasks: List[str],
        seed: int = 0,
        max_episode_steps: int = 500,
        n_envs: int = 1,
    ):
        if not stage_tasks:
            raise ValueError("stage_tasks must not be empty")

        self._log(f"Creating curriculum env ({len(stage_tasks)} tasks)")

        if len(stage_tasks) == 1:
            return self.make_mt1_env(
                task=stage_tasks[0],
                seed=seed,
                max_episode_steps=max_episode_steps,
                n_envs=n_envs,
            )

        return self.make_multitask_env(
            tasks=stage_tasks,
            seed=seed,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
        )

    # ------------------------------------------------------------------
    # train / eval
    # ------------------------------------------------------------------

    def create_train_eval_pair(
        self,
        tasks: Union[str, List[str]],
        train_seed: int = 0,
        eval_seed: Optional[int] = None,
        max_episode_steps: int = 500,
        n_envs: int = 1,
        seed_offset: int = 1000,
    ):
        if isinstance(tasks, str):
            tasks = [tasks]

        if eval_seed is None:
            eval_seed = train_seed + seed_offset

        self._log("Creating train/eval env pair")
        self._log(f"Train seed: {train_seed}, Eval seed: {eval_seed}")

        train_env = self.make_curriculum_env(
            stage_tasks=tasks,
            seed=train_seed,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
        )

        eval_env = self.make_curriculum_env(
            stage_tasks=tasks,
            seed=eval_seed,
            max_episode_steps=max_episode_steps,
            n_envs=1,
        )

        return train_env, eval_env
