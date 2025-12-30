import numpy as np
from typing import Optional, List, Dict, Any


class DebugPrinter:
    """
    Klasse für formatierte Debug-Ausgaben während des Meta-World Trainings.
    Unterstützt sowohl Standard- als auch Curriculum/Transfer-Learning Setups.
    """

    def __init__(self, verbose: bool = True, line_width: int = 70):
        """
        Args:
            verbose: Wenn False, werden keine Ausgaben gemacht
            line_width: Breite der Trennlinien
        """
        self.verbose = verbose
        self.line_width = line_width
        self.separator = "=" * line_width
        self.mini_separator = "-" * line_width

    def _print(self, *args, **kwargs):
        """Internal print routine with verbose check"""
        if self.verbose:
            print(*args, **kwargs)

    def print_header(self, text: str, use_mini: bool = False):
        """Druckt einen formatierten Header"""
        separator = self.mini_separator if use_mini else self.separator
        self._print(f"\n{separator}")
        self._print(text)
        self._print(f"{separator}\n")

    def print_section(self, title: str):
        """Druckt einen Section-Titel"""
        self._print(f"\n{title}")
        self._print(self.mini_separator)

    def print_start_setup(
            self,
            experiment: str,
            algorithm: str,
            training_mode: Optional[str] = None,
            use_transfer: bool = False,
            use_curriculum: bool = False
    ):
        """
        Druckt die initiale Setup-Konfiguration

        Args:
            experiment: Name des Experiments (z.B. "MT10", "MT10_CURRICULUM")
            algorithm: Verwendeter Algorithmus (SAC, TD3, DDPG)
            training_mode: Optional - "SEQUENTIAL", "PROGRESSIVE", "MIXED"
            use_transfer: Ob Transfer Learning verwendet wird
            use_curriculum: Ob Curriculum Learning verwendet wird
        """
        self.print_header(f"META-WORLD {experiment.upper()} TRAINING")

        self._print(f"Algorithm: {algorithm}")

        if training_mode:
            self._print(f"Training Strategy: {training_mode}")

        if use_transfer or use_curriculum:
            features = []
            if use_transfer:
                features.append("Transfer Learning")
            if use_curriculum:
                features.append("Curriculum Learning")
            self._print(f"Features: {', '.join(features)}")

        self._print(self.separator)

    def print_curriculum_info(
            self,
            stage: int,
            total_stages: int,
            tasks: List[str],
            stage_thresholds: Optional[Dict[int, float]] = None
    ):
        """
        Druckt Curriculum Learning Informationen

        Args:
            stage: Aktuelle Stage (0-indexed)
            total_stages: Gesamtanzahl der Stages
            tasks: Liste der Tasks in dieser Stage
            stage_thresholds: Optional - Performance-Schwellenwerte
        """
        self.print_section("CURRICULUM LEARNING CONFIGURATION")

        self._print(f"Current Stage: {stage + 1}/{total_stages}")
        self._print(f"Tasks in this stage ({len(tasks)}): ")
        for i, task in enumerate(tasks, 1):
            self._print(f"  {i}. {task}")

        if stage_thresholds and stage < total_stages - 1:
            next_threshold = stage_thresholds.get(stage + 1, 0.5)
            self._print(f"\nThreshold for next stage: {next_threshold: .1%} success rate")

    def print_transfer_info(
            self,
            pretrained_model: Optional[str],
            source_tasks: Optional[List[str]] = None,
            target_tasks: Optional[List[str]] = None,
            lr_multiplier: float = 0.1
    ):
        """
        Druckt Transfer Learning Informationen

        Args:
            pretrained_model: Pfad zum vortrainierten Modell
            source_tasks: Tasks auf denen vortrainiert wurde
            target_tasks: neue Tasks für Transfer
            lr_multiplier: Learning Rate Multiplikator fürs Fine-Tuning
        """
        self.print_section("TRANSFER LEARNING CONFIGURATION")

        if pretrained_model:
            self._print(f"Pretrained Model: {pretrained_model}")
            self._print(f"Fine-Tuning LR Multiplier: {lr_multiplier}")

            if source_tasks:
                self._print(f"\nSource Tasks ({len(source_tasks)}): ")
                for task in source_tasks:
                    self._print(f"  • {task}")

            if target_tasks:
                self._print(f"\nTarget Tasks ({len(target_tasks)}): ")
                for task in target_tasks:
                    self._print(f"  • {task}")
        else:
            self._print("No pretrained model - training from scratch")

    def print_training_start(
            self,
            model: Any,
            task_name: str,
            algorithm: str,
            time_steps: int,
            seed: int,
            max_eps_steps: int,
            norm_reward: bool,
            eval_freq: int,
            n_eval_eps: int,
            checkpoint_freq: int,
            num_envs: int,
            action_space: np.ndarray,
            current_tasks: Optional[List[str]] = None
    ):
        """
        Druckt detaillierte Training-Konfiguration

        Args:
            model: Das RL-Modell (SAC, TD3, etc.)
            task_name: Name des Tasks/Experiments
            algorithm: Algorithmus-Name
            time_steps: Gesamtanzahl der Trainingsschritte
            seed: Random Seed
            max_eps_steps: Max. Schritte pro Episode
            norm_reward: Ob Rewards normalisiert werden
            eval_freq: Evaluations-Frequenz
            n_eval_eps: Anzahl Evaluations-Episoden
            checkpoint_freq: Checkpoint-Frequenz
            num_envs: Anzahl paralleler Umgebungen
            action_space: Action Space des Environments
            current_tasks: Optional - Liste aktueller Tasks
        """
        self.print_header("STARTING TRAINING")

        # Basic Configuration
        self._print("Basic Configuration:")
        self._print(f"  Total time-steps: {time_steps}")
        self._print(f"  Seed: {seed}")
        self._print(f"  Algorithm: {algorithm}")
        self._print(f"  Parallel Environments: {num_envs}")

        # Task Information
        if current_tasks and len(current_tasks) > 1:
            self._print(f"\nMulti-Task Setup ({len(current_tasks)} tasks): ")
            for task in current_tasks:
                self._print(f"  • {task}")
        else:
            self._print(f"\nTask: {task_name}")

        # Model Hyperparameters
        self._print("\nModel Hyperparameters:")

        if hasattr(model, 'learning_rate'):
            lr = model.learning_rate
            if callable(lr):
                self._print(f"  Learning Rate: {lr(1): .2e} (initial)")
            else:
                self._print(f"  Learning Rate: {lr: .2e}")

        if hasattr(model, 'learning_starts'):
            self._print(f"  Learning Starts: {model.learning_starts}")

        if hasattr(model, 'batch_size'):
            self._print(f"  Batch Size: {model.batch_size}")

        if hasattr(model, 'buffer_size'):
            self._print(f"  Buffer Size: {model.buffer_size}")

        if hasattr(model, 'gamma'):
            self._print(f"  Gamma (Discount): {model.gamma}")

        if hasattr(model, 'gradient_steps'):
            self._print(f"  Gradient Steps: {model.gradient_steps}")

        if hasattr(model, 'tau'):
            self._print(f"  Tau (Target Update): {model.tau}")

        # Network Architecture
        if hasattr(model, 'policy_kwargs') and model.policy_kwargs:
            if 'net_arch' in model.policy_kwargs:
                self._print(f"  Network Architecture: {model.policy_kwargs['net_arch']}")

        # Algorithm-Specific
        if algorithm == "SAC":
            self._print(f"\nSAC-Specific: ")
            self._print(f"  Entropy Tuning: Automatic")
            self._print(f"  Target Entropy: {-action_space.shape[0]}")

        elif algorithm == "TD3":
            self._print(f"\nTD3-Specific: ")
            if hasattr(model, 'policy_kwargs'):
                self._print(f"  Exploration Noise: σ=0.1")
                self._print(f"  Target Policy Noise: 0.1 (clip: 0.3)")

        # Environment Settings
        self._print("\nEnvironment Settings:")
        self._print(f"  Max Episode Steps: {max_eps_steps}")
        self._print(f"  Normalize Reward: {norm_reward}")
        self._print(f"  Reward Function: v3 (stable)")

        # Evaluation & Checkpointing
        self._print("\nEvaluation & Checkpointing:")
        self._print(f"  Eval Frequency: {eval_freq} steps")
        self._print(f"  Eval Episodes: {n_eval_eps}")
        self._print(f"  Checkpoint Frequency: {checkpoint_freq} steps")

        self._print(self.separator)

    def print_training_finished(
            self,
            task_name: str,
            algorithm: str,
            final_model_path: Optional[str] = None,
            best_model_path: Optional[str] = None,
            checkpoint_path: Optional[str] = None,
            transfer_checkpoint_path: Optional[str] = None
    ):
        """
        Druckt Zusammenfassung nach Training

        Args:
            task_name: Name des Tasks/Experiments
            algorithm: Algorithmus-Name
            final_model_path: Pfad zum finalen Modell
            best_model_path: Pfad zum besten Modell
            checkpoint_path: Pfad zu den Checkpoints
            transfer_checkpoint_path: Optional - Pfad zu Transfer Checkpoints
        """
        self.print_header("TRAINING COMPLETED ✓")

        self._print("Saved Models:")

        if final_model_path:
            self._print(f"  Final Model: {final_model_path}")
        else:
            self._print(f"  Final Model: ./metaworld_models/{algorithm.lower()}_{task_name}_final.zip")

        if best_model_path:
            self._print(f"  Best Model: {best_model_path}")
        else:
            self._print(f"  Best Model: ./metaworld_models/best_{task_name}/best_model.zip")

        if checkpoint_path:
            self._print(f"  Checkpoints: {checkpoint_path}")
        else:
            self._print(f"  Checkpoints: ./metaworld_models/checkpoints_{task_name}/")

        if transfer_checkpoint_path:
            self._print(f"  Transfer Checkpoint: {transfer_checkpoint_path}")

        self._print("\nNext Steps:")
        self._print("  • Evaluate model performance")
        self._print("  • Monitor training: tensorboard --logdir=./metaworld_logs/")
        self._print("  • Use model for transfer learning if applicable")

        self._print(self.separator)

    def print_stage_transition(
            self,
            old_stage: int,
            new_stage: int,
            old_tasks: List[str],
            new_tasks: List[str],
            performance: float
    ):
        """
        Druckt Information über Curriculum Stage Übergang

        Args:
            old_stage: Vorherige Stage (0-indexed)
            new_stage: Neue Stage (0-indexed)
            old_tasks: Tasks der alten Stage
            new_tasks: Tasks der neuen Stage
            performance: Performance die zum Übergang führte
        """
        self.print_header(f"CURRICULUM STAGE TRANSITION: {old_stage + 1} → {new_stage + 1}", use_mini=True)

        self._print(f"Performance achieved: {performance: .1%}")

        self._print(f"\nOld Tasks ({len(old_tasks)}): ")
        for task in old_tasks:
            self._print(f"  ✓ {task}")

        new_added = [t for t in new_tasks if t not in old_tasks]
        if new_added:
            self._print(f"\nNewly Added Tasks ({len(new_added)}): ")
            for task in new_added:
                self._print(f" + {task}")

        self._print(self.mini_separator)

    def print_model_info(self, model: Any):
        """Druckt detaillierte Model-Informationen"""
        self.print_section("MODEL INFORMATION")

        self._print(f"Policy: {type(model.policy).__name__}")

        if hasattr(model, 'observation_space'):
            obs_space = model.observation_space
            self._print(f"Observation Space: {obs_space.shape}")

        if hasattr(model, 'action_space'):
            act_space = model.action_space
            self._print(f"Action Space: {act_space.shape}")

        # Count parameters
        if hasattr(model, 'policy'):
            total_params = sum(p.numel() for p in model.policy.parameters())
            trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
            self._print(f"Total Parameters: {total_params}")
            self._print(f"Trainable Parameters: {trainable_params}")

    def print_curriculum_stage(self, next_stage: int, transfer_checkpoint_path: str, next_tasks: List[str]):
        """

        Args:
            next_stage:
            transfer_checkpoint_path:
            next_tasks:

        Returns:

        """
        self.print_section("NEXT CURRICULUM STAGE")
        self._print(f"To continue training with the next stage: ")
        self._print(f"  1. Set CURRICULUM_STAGE = {next_stage}")
        self._print(f"  2. Set USE_TRANSFER_LEARNING = True")
        self._print(f"  3. Set PRETRAINED_MODEL_PATH = '{transfer_checkpoint_path}.zip'")
        self._print(f"\nNext stage will include {len(next_tasks)} tasks: ")
        for task in next_tasks:
            self._print(f"  • {task}")
        self._print("=" * 70)

    def print_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Druckt Fehler-Informationen"""
        self._print(f"\n{'!' * self.line_width}")
        self._print(f"ERROR: {error_msg}")
        if exception:
            self._print(f"Exception: {type(exception).__name__}: {str(exception)}")
        self._print(f"{'!' * self.line_width}\n")

    def print_warning(self, warning_msg: str):
        """Druckt Warnung"""
        self._print(f"\n⚠️  WARNING: {warning_msg}\n")

    def print_success(self, success_msg: str):
        """Druckt Erfolgs-Nachricht"""
        self._print(f"\n✓ {success_msg}\n")


# Convenience functions für Backward Compatibility
def print_start_setup(experiment: str, algorithm: str, train_mode: str):
    """Backward compatibility wrapper"""
    printer = DebugPrinter()
    printer.print_start_setup(experiment, algorithm, train_mode)


def print_training_start(model, task_name: str, algorithm: str, time_steps: int, seed: int,
                         max_eps_steps: int, norm_reward: bool, eval_freq: int, n_eval_eps: int,
                         checkpoint_freq: int, train_phase: int, num_envs: int, action_space: np.ndarray):
    """Backward compatibility wrapper"""
    printer = DebugPrinter()
    printer.print_training_start(
        model, task_name, algorithm, time_steps, seed, max_eps_steps,
        norm_reward, eval_freq, n_eval_eps, checkpoint_freq, train_phase,
        num_envs, action_space
    )


def print_training_finished(task_name: str, algorithm: str):
    """Backward compatibility wrapper"""
    printer = DebugPrinter()
    printer.print_training_finished(task_name, algorithm)
