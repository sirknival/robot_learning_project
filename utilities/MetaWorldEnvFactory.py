import gymnasium as gym
import metaworld
from typing import List, Dict, Optional, Union
from training_setup_multitask.WrapperClasses.GymnasiumVecEnvAdapter import GymnasiumVecEnvAdapter
from training_setup_multitask.utilities.MetaworldTasks import MT3_TASKS, MT10_TASKS


class MetaWorldEnvFactory:
    """
    Factory-Klasse für Meta-World Environments.
    Unterstützt MT1, MT3, MT10 und custom Curriculum Stages.
    """

    def __init__(
            self,
            reward_function_version: str = 'v3',
            vector_strategy: str = 'sync',
            terminate_on_success: bool = False,
            use_subproc: bool = True,
            verbose: bool = True
    ):
        """
        Args:
            reward_function_version: Version der Reward-Funktion ('v2' oder 'v3')
            vector_strategy: Vectorization-Strategie ('sync' oder 'async')
            terminate_on_success: Ob Episode bei Erfolg beendet werden soll
            use_subproc: Nutze SubprocVecEnv (True) oder DummyVecEnv (False)
            verbose: Ausgaben aktivieren/deaktivieren
        """
        self.reward_function_version = reward_function_version
        self.vector_strategy = vector_strategy
        self.terminate_on_success = terminate_on_success
        self.use_subproc = use_subproc
        self.verbose = verbose

        self.MT10_TASKS = MT10_TASKS
        self.MT3_TASKS = MT3_TASKS

    def _log(self, message: str):
        """Interne Logging-Funktion"""
        if self.verbose:
            print(f"[EnvFactory] {message}")

    def make_parallel_mt1_envs(
            self,
            task_name: str,
            n_envs: int = 4,
            seed: int = 0,
            max_episode_steps: int = 500
    ):
        """
        Erstelle mehrere parallele MT1 Umgebungen für schnelleres Training

        Args:
            task_name: Name des Tasks
            n_envs: Anzahl paralleler Environments (empfohlen: 4-8)
            seed: Base random seed
            max_episode_steps: maximale Schritte pro Episode

        Returns:
            DummyVecEnv mit n_envs parallelen Environments
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        if task_name not in self.MT10_TASKS:
            self._log(f"⚠️  Warning: '{task_name}' not in standard MT10 tasks")

        self._log(f"Creating {n_envs} parallel MT1 environments: {task_name}")

        # Create list of environment creation functions
        env_fns = []
        for i in range(n_envs):
            def _make_env(rank=i):
                env = gym.make(
                    'Meta-World/MT1',
                    env_name=task_name,
                    seed=seed + rank,
                    reward_function_version=self.reward_function_version,
                    max_episode_steps=max_episode_steps,
                    terminate_on_success=self.terminate_on_success,
                )
                return env

            env_fns.append(_make_env)

        # Create vectorized environment
        vec_env = DummyVecEnv(env_fns)

        self._log(f"✓ {n_envs} parallel MT1 environments created")

        return vec_env

    def make_mt1_env(
            self,
            task_name: str,
            seed: int = 0,
            max_episode_steps: int = 500,
            rank: int = 0,
            n_envs: int = 1
    ):
        """
        Erstelle Single-Task Umgebung (MT1) mit optionaler Parallelisierung

        Args:
            task_name: Name des Tasks (z.B. "reach-v3", "push-v3")
            seed: Random seed
            max_episode_steps: Maximale Schritte pro Episode
            rank: Environment rank (für parallele Umgebungen)
            n_envs: Anzahl paralleler Environments (1 = kein Parallel, >1 = parallel)

        Returns:
            DummyVecEnv mit einem oder mehreren Environments
        """
        if n_envs > 1:
            return self.make_parallel_mt1_envs(
                task_name=task_name,
                n_envs=n_envs,
                seed=seed,
                max_episode_steps=max_episode_steps
            )

        from stable_baselines3.common.vec_env import DummyVecEnv

        if task_name not in self.MT10_TASKS:
            self._log(f"⚠️  Warning: '{task_name}' not in standard MT10 tasks")

        self._log(f"Creating MT1 environment: {task_name}")

        # gym.make gibt ein einzelnes Environment zurück, kein VecEnv
        def _make_env():
            env = gym.make(
                'Meta-World/MT1',
                env_name=task_name,
                seed=seed + rank,
                reward_function_version=self.reward_function_version,
                max_episode_steps=max_episode_steps,
                terminate_on_success=self.terminate_on_success,
            )
            return env

        # Wrapping in DummyVecEnv für Kompatibilität mit VecEnv-Interface
        vec_env = DummyVecEnv([_make_env])

        self._log(f"✓ MT1 environment created (seed={seed + rank})")

        return vec_env

    def make_mt3_env(
            self,
            seed: int = 0,
            max_episode_steps: int = 500,
            rank: int = 0,
            custom_tasks: Optional[List[str]] = None
    ) -> GymnasiumVecEnvAdapter:
        """
        Erstelle MT3 Umgebung (3 Tasks)

        Args:
            seed: Random seed
            max_episode_steps: Maximale Schritte pro Episode
            rank: Environment rank
            custom_tasks: Optional - Custom Task-Liste (sonst DEFAULT_MT3_TASKS)

        Returns:
            Wrapped Gymnasium VecEnv
        """
        tasks = custom_tasks if custom_tasks else self.MT3_TASKS

        if len(tasks) != 3:
            self._log(f"⚠️  Warning: MT3 expects 3 tasks, got {len(tasks)}")

        self._log(f"Creating MT3 environment with tasks: {tasks}")

        raw_env = gym.make_vec(
            'Meta-World/custom-mt-envs',
            vector_strategy=self.vector_strategy,
            seed=seed + rank,
            envs_list=tasks,
            reward_function_version=self.reward_function_version,
            max_episode_steps=max_episode_steps,
            terminate_on_success=self.terminate_on_success
        )

        env = GymnasiumVecEnvAdapter(raw_env)
        self._log(f"✓ MT3 environment created (seed={seed + rank})")

        return env

    def make_mt10_env(
            self,
            seed: int = 0,
            max_episode_steps: int = 500,
            rank: int = 0
    ) -> GymnasiumVecEnvAdapter:
        """
        Erstelle MT10 Umgebung (alle 10 Standard-Tasks)

        Args:
            seed: Random seed
            max_episode_steps: Maximale Schritte pro Episode
            rank: Environment rank

        Returns:
            Wrapped Gymnasium VecEnv
        """
        self._log("Creating MT10 environment (10 tasks)")

        raw_env = gym.make_vec(
            'Meta-World/MT10',
            vector_strategy=self.vector_strategy,
            seed=seed + rank,
            reward_function_version=self.reward_function_version,
            max_episode_steps=max_episode_steps,
            terminate_on_success=self.terminate_on_success
        )

        env = GymnasiumVecEnvAdapter(raw_env)
        self._log(f"✓ MT10 environment created (seed={seed + rank})")

        return env

    def make_custom_multi_task_env(
            self,
            tasks: List[str],
            seed: int = 0,
            max_episode_steps: int = 500,
            rank: int = 0
    ) -> GymnasiumVecEnvAdapter:
        """
        Erstelle Custom Multi-Task Umgebung mit beliebiger Task-Liste

        Args:
            tasks: Liste von Task-Namen
            seed: Random seed
            max_episode_steps: Maximale Schritte pro Episode
            rank: Environment rank

        Returns:
            Wrapped Gymnasium VecEnv
        """
        if not tasks:
            raise ValueError("Task list cannot be empty")

        # Validiere Tasks
        invalid_tasks = [t for t in tasks if t not in self.MT10_TASKS]
        if invalid_tasks:
            self._log(f"⚠️  Warning: Unknown tasks: {invalid_tasks}")

        self._log(f"Creating custom multi-task environment with {len(tasks)} tasks")

        raw_env = gym.make_vec(
            'Meta-World/custom-mt-envs',
            vector_strategy=self.vector_strategy,
            seed=seed + rank,
            envs_list=tasks,
            reward_function_version=self.reward_function_version,
            max_episode_steps=max_episode_steps,
            terminate_on_success=self.terminate_on_success
        )

        env = GymnasiumVecEnvAdapter(raw_env)
        self._log(f"✓ Custom multi-task environment created (seed={seed + rank})")

        return env

    def make_curriculum_env(
            self,
            stage_tasks: List[str],
            seed: int = 0,
            max_episode_steps: int = 500,
            rank: int = 0
    ) -> GymnasiumVecEnvAdapter:
        """
        Erstelle Umgebung für spezifische Curriculum Stage
        Automatische Auswahl zwischen MT1, MT3, MT10 oder custom basierend auf Task-Anzahl

        Args:
            stage_tasks: Liste der Tasks für diese Stage
            seed: Random seed
            max_episode_steps: Maximale Schritte pro Episode
            rank: Environment rank

        Returns:
            Wrapped Gymnasium VecEnv
        """
        num_tasks = len(stage_tasks)
        self._log(f"Creating curriculum environment with {num_tasks} task(s)")

        if num_tasks == 0:
            raise ValueError("stage_tasks cannot be empty")

        elif num_tasks == 1:
            # Single task - MT1
            return self.make_mt1_env(
                task_name=stage_tasks[0],
                seed=seed,
                max_episode_steps=max_episode_steps,
                rank=rank
            )

        elif num_tasks == 3:
            # Three tasks - MT3
            return self.make_mt3_env(
                seed=seed,
                max_episode_steps=max_episode_steps,
                rank=rank,
                custom_tasks=stage_tasks
            )

        elif num_tasks == 10 and set(stage_tasks) == set(self.MT10_TASKS):
            # All standard MT10 tasks
            return self.make_mt10_env(
                seed=seed,
                max_episode_steps=max_episode_steps,
                rank=rank
            )

        else:
            # Custom number of tasks
            return self.make_custom_multi_task_env(
                tasks=stage_tasks,
                seed=seed,
                max_episode_steps=max_episode_steps,
                rank=rank
            )

    def create_train_eval_pair(
            self,
            tasks: Union[str, List[str]],
            train_seed: int = 0,
            eval_seed: Optional[int] = None,
            max_episode_steps: int = 500,
            seed_offset: int = 1000,
            n_parallel_envs: int = 1
    ) -> tuple:
        """
        Erstelle Train- und Eval-Environment Paar mit unterschiedlichen Seeds

        Args:
            tasks: Einzelner Task-Name oder Task-Liste
            train_seed: Seed für Training Environment
            eval_seed: Optional - Seed für Eval Environment (sonst train_seed + seed_offset)
            max_episode_steps: Maximale Schritte pro Episode
            seed_offset: Offset zwischen Train und Eval Seed
            n_parallel_envs: Anzahl paralleler Training-Environments (nur für MT1)

        Returns:
            Tuple (train_env, eval_env)
        """
        if eval_seed is None:
            eval_seed = train_seed + seed_offset

        # Konvertiere einzelnen Task zu Liste
        if isinstance(tasks, str):
            tasks = [tasks]

        self._log(f"Creating train/eval environment pair")
        self._log(f"  Train seed: {train_seed}, Eval seed: {eval_seed}")
        if n_parallel_envs > 1 and len(tasks) == 1:
            self._log(f"  Using {n_parallel_envs} parallel training environments")

        # Spezielle Behandlung für MT1 mit Parallelisierung
        if len(tasks) == 1 and n_parallel_envs > 1:
            train_env = self.make_mt1_env(
                task_name=tasks[0],
                seed=train_seed,
                max_episode_steps=max_episode_steps,
                n_envs=n_parallel_envs
            )
            # Eval bleibt einzeln
            eval_env = self.make_mt1_env(
                task_name=tasks[0],
                seed=eval_seed,
                max_episode_steps=max_episode_steps,
                n_envs=1
            )
        else:
            # Standard: Nutze curriculum_env für automatische Auswahl
            train_env = self.make_curriculum_env(
                stage_tasks=tasks,
                seed=train_seed,
                max_episode_steps=max_episode_steps,
                rank=0
            )

            eval_env = self.make_curriculum_env(
                stage_tasks=tasks,
                seed=eval_seed,
                max_episode_steps=max_episode_steps,
                rank=0
            )

        self._log("✓ Train/Eval pair created")

        return train_env, eval_env

    def get_task_info(self, task_name: str) -> Dict:
        """
        Hole Informationen über einen spezifischen Task

        Args:
            task_name: Name des Tasks

        Returns:
            Dictionary mit Task-Informationen
        """
        if task_name not in self.MT10_TASKS:
            return {
                "name": task_name,
                "valid": False,
                "in_mt10": False
            }

        # Erstelle temporäre Umgebung für Info
        temp_env = self.make_mt1_env(task_name, seed=0, max_episode_steps=100)

        info = {
            "name": task_name,
            "valid": True,
            "in_mt10": True,
            "observation_space": temp_env.observation_space,
            "action_space": temp_env.action_space,
            "num_envs": getattr(temp_env, "num_envs", 1)
        }

        temp_env.close()

        return info

    def validate_tasks(self, tasks: List[str]) -> tuple:
        """
        Validiere eine Liste von Tasks

        Args:
            tasks: Liste von Task-Namen

        Returns:
            Tuple (valid_tasks, invalid_tasks)
        """
        valid_tasks = [t for t in tasks if t in self.MT10_TASKS]
        invalid_tasks = [t for t in tasks if t not in self.MT10_TASKS]

        if invalid_tasks:
            self._log(f"⚠️  Invalid tasks found: {invalid_tasks}")

        return valid_tasks, invalid_tasks


# ==================== BACKWARD COMPATIBILITY FUNCTIONS ====================

# Global factory instance für backward compatibility
_default_factory = MetaWorldEnvFactory()


def make_mt1_env(task_name: str, seed: int = 0, max_episode_steps: int = 500, rank: int = 0):
    """Backward compatibility wrapper für make_mt1_env"""
    return _default_factory.make_mt1_env(task_name, seed, max_episode_steps, rank)


def make_mt3_env(rank: int = 0, seed: int = 0, max_episode_steps: int = 500):
    """Backward compatibility wrapper für make_mt3_env"""
    return _default_factory.make_mt3_env(seed, max_episode_steps, rank)


def make_mt10_env(rank: int = 0, seed: int = 0, max_episode_steps: int = 500):
    """Backward compatibility wrapper für make_mt10_env"""
    return _default_factory.make_mt10_env(seed, max_episode_steps, rank)


def make_curriculum_env(
        stage_tasks: List[str],
        seed: int,
        max_episode_steps: int = 500
):
    """Backward compatibility wrapper für make_curriculum_env"""
    return _default_factory.make_curriculum_env(stage_tasks, seed, max_episode_steps)


# ==================== USAGE EXAMPLES ====================
"""
if __name__ == "__main__":
    # Example 1: Basic usage mit Factory-Klasse
    factory = MetaWorldEnvFactory(verbose=True)

    # Erstelle MT1 Umgebung
    env_mt1 = factory.make_mt1_env("reach-v3", seed=42)
    print(f"MT1 Observation Space: {env_mt1.observation_space}")
    print(f"MT1 Action Space: {env_mt1.action_space}")
    env_mt1.close()

    # Example 2: Curriculum Environment
    curriculum_tasks = ["reach-v3", "push-v3", "pick-place-v3"]
    env_curriculum = factory.make_curriculum_env(
        stage_tasks=curriculum_tasks,
        seed=42,
        max_episode_steps=500
    )
    env_curriculum.close()

    # Example 3: Train/Eval Pair
    train_env, eval_env = factory.create_train_eval_pair(
        tasks=["reach-v3", "push-v3"],
        train_seed=42
    )
    train_env.close()
    eval_env.close()

    # Example 4: Task Validation
    all_tasks = ["reach-v3", "invalid-task", "push-v3"]
    valid, invalid = factory.validate_tasks(all_tasks)
    print(f"Valid tasks: {valid}")
    print(f"Invalid tasks: {invalid}")

    # Example 5: Backward compatibility
    env_old_style = make_mt1_env("reach-v3", seed=42)
    env_old_style.close()
"""
