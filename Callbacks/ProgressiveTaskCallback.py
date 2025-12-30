from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict


class ProgressiveTaskCallback(BaseCallback):
    """
    Callback for progressive task introduction in curriculum learning.
    Monitors performance and automatically advances to the next curriculum stage
    when performance thresholds are met.

    Usage:
        callback = ProgressiveTaskCallback(
            curriculum_stages=curriculum_config.CURRICULUM_STAGES,
            stage_thresholds=curriculum_config.STAGE_THRESHOLDS,
            task_evaluator=task_evaluator,
            eval_freq=10000,
            min_steps_per_stage=200000,
            verbose=1
        )
    """

    def __init__(
            self,
            curriculum_stages: List[List[str]],
            stage_thresholds: Dict[int, float],
            task_evaluator,
            eval_freq: int = 10000,
            min_steps_per_stage: int = 100000,
            verbose: int = 1
    ):
        """
        Args:
            curriculum_stages: List of curriculum stages (each stage is a list of tasks)
            stage_thresholds: Performance thresholds for stage transitions
            task_evaluator: TaskEvaluator instance for evaluating performance
            eval_freq: Evaluate performance every N steps
            min_steps_per_stage: Minimum steps before allowing stage transition
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)

        self.curriculum_stages = curriculum_stages
        self.stage_thresholds = stage_thresholds
        self.task_evaluator = task_evaluator
        self.eval_freq = eval_freq
        self.min_steps_per_stage = min_steps_per_stage

        self.current_stage = 0
        self.steps_in_current_stage = 0
        self.stage_performances = []
        self.last_eval_step = 0

        # Track best performance
        self.best_performance = 0.0
        self.stages_completed = []

    def _on_training_start(self):
        """Called at the beginning of training"""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("CURRICULUM LEARNING INITIALIZED")
            print("=" * 70)
            print(f"Starting Stage: {self.current_stage + 1}/{len(self.curriculum_stages)}")
            print(f"Tasks in Stage: {self.curriculum_stages[self.current_stage]}")
            print(f"Evaluation Frequency: {self.eval_freq} steps")
            print(f"Minimum Steps per Stage: {self.min_steps_per_stage}")
            print("=" * 70 + "\n")

    def _on_step(self) -> bool:
        """
        Called at each training step.
        Returns True to continue training, False to stop.
        """
        self.steps_in_current_stage += 1

        # Check if it's time to evaluate
        if self.n_calls - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.n_calls

            # Only consider stage transition after minimum steps
            if self.steps_in_current_stage >= self.min_steps_per_stage:
                performance = self._evaluate_current_stage()
                self.stage_performances.append(performance)

                if self.verbose > 0:
                    print(f"\n[Step {self.n_calls}] Current Performance: {performance:.1%}")

                # Check if should advance to next stage
                if self._should_advance_stage(performance):
                    self._advance_to_next_stage()

        return True

    def _evaluate_current_stage(self) -> float:
        """
        Evaluate performance on current curriculum stage tasks.

        Returns:
            Overall performance score (average success rate across tasks)
        """
        if self.verbose > 1:
            print(f"\n[Curriculum Eval] Evaluating stage {self.current_stage + 1}...")

        try:
            # Get current stage tasks
            current_tasks = self.curriculum_stages[self.current_stage]

            # Evaluate on current tasks
            results = self.task_evaluator.evaluate_tasks(
                model=self.model,
                tasks=current_tasks,
                deterministic=True
            )

            # Calculate overall performance
            performance = self.task_evaluator.get_overall_performance(results)

            # Update best performance
            if performance > self.best_performance:
                self.best_performance = performance

            if self.verbose > 0:
                print(f"\n[Curriculum Eval] Stage {self.current_stage + 1} Performance: {performance:.1%}")
                print(f"  Best Performance: {self.best_performance:.1%}")

            return performance

        except Exception as e:
            if self.verbose > 0:
                print(f"\n[Curriculum Eval] Evaluation failed: {e}")
            return 0.0

    def _should_advance_stage(self, performance: float) -> bool:
        """
        Determine if should advance to next curriculum stage.

        Args:
            performance: Current performance score

        Returns:
            True if should advance, False otherwise
        """
        # Already at final stage
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False

        # Get threshold for next stage
        next_stage = self.current_stage + 1
        threshold = self.stage_thresholds.get(next_stage, 0.5)

        # Check if performance meets threshold
        should_advance = performance >= threshold

        if self.verbose > 0:
            print(f"  Threshold for Stage {next_stage + 1}: {threshold:.1%}")
            print(f"  Should advance: {should_advance}")

        return should_advance

    def _advance_to_next_stage(self):
        """Advance to the next curriculum stage"""
        old_stage = self.current_stage
        self.current_stage += 1
        self.steps_in_current_stage = 0

        # Record completed stage
        self.stages_completed.append({
            "stage": old_stage,
            "final_performance": self.stage_performances[-1] if self.stage_performances else 0.0,
            "total_steps": self.n_calls
        })

        # Reset performance tracking
        self.stage_performances = []

        if self.verbose > 0:
            print("\n" + "=" * 70)
            print(f"ADVANCING TO CURRICULUM STAGE {self.current_stage + 1}/{len(self.curriculum_stages)}")
            print("=" * 70)

            old_tasks = self.curriculum_stages[old_stage]
            new_tasks = self.curriculum_stages[self.current_stage]

            print(f"\nCompleted Stage {old_stage + 1}:")
            print(f"  Tasks: {old_tasks}")
            print(f"  Final Performance: {self.stages_completed[-1]['final_performance']:.1%}")

            print(f"\nNew Stage {self.current_stage + 1}:")
            print(f"  Tasks: {new_tasks}")

            # Show newly added tasks
            new_added = [t for t in new_tasks if t not in old_tasks]
            if new_added:
                print(f"  Newly Added: {new_added}")

            print("=" * 70 + "\n")

    def _on_training_end(self):
        """Called at the end of training"""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("CURRICULUM LEARNING SUMMARY")
            print("=" * 70)

            print(f"\nTotal Stages Completed: {len(self.stages_completed)}")
            print(f"Final Stage: {self.current_stage + 1}/{len(self.curriculum_stages)}")
            print(f"Best Performance: {self.best_performance:.1%}")

            if self.stages_completed:
                print("\nStage History:")
                for stage_info in self.stages_completed:
                    stage_num = stage_info['stage'] + 1
                    perf = stage_info['final_performance']
                    steps = stage_info['total_steps']
                    print(f"  Stage {stage_num}: {perf:.1%} (completed at step {steps:,})")

            print("=" * 70 + "\n")

    def get_curriculum_progress(self) -> Dict:
        """
        Get current curriculum progress information.

        Returns:
            Dictionary with curriculum progress details
        """
        return {
            "current_stage": self.current_stage,
            "total_stages": len(self.curriculum_stages),
            "current_tasks": self.curriculum_stages[self.current_stage],
            "steps_in_stage": self.steps_in_current_stage,
            "best_performance": self.best_performance,
            "stages_completed": self.stages_completed,
            "recent_performances": self.stage_performances[-10:] if len(self.stage_performances) > 0 else []
        }