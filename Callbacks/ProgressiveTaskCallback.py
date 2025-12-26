from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict

class ProgressiveTaskCallback(BaseCallback):
    """
    Callback für progressive Task-Einführung im Curriculum Learning
    Überwacht Performance und wechselt automatisch zur nächsten Stage
    """

    def __init__(
            self,
            curriculum_stages: List[List[str]],
            stage_thresholds: Dict[int, float],
            eval_freq: int = 10000,
            min_steps_per_stage: int = 100000,
            verbose: int = 1
    ):
        super().__init__(verbose)
        self.curriculum_stages = curriculum_stages
        self.stage_thresholds = stage_thresholds
        self.eval_freq = eval_freq
        self.min_steps_per_stage = min_steps_per_stage

        self.current_stage = 0
        self.steps_in_current_stage = 0
        self.stage_performances = []

    def _on_step(self) -> bool:
        self.steps_in_current_stage += 1

        # Überprüfe alle eval_freq Schritte
        if self.n_calls % self.eval_freq == 0:
            if self.steps_in_current_stage >= self.min_steps_per_stage:
                avg_performance = self._evaluate_current_stage()

                # Prüfe ob Stage-Übergang möglich
                if self._should_advance_stage(avg_performance):
                    self._advance_to_next_stage()

        return True

    def _evaluate_current_stage(self) -> float:
        """Evaluiere Performance auf aktuellen Tasks"""
        # Hier würde man die tatsächliche Evaluation durchführen
        # Für jetzt: Placeholder mit letzten Evaluationsergebnissen
        if hasattr(self.model, 'last_eval_results'):
            return np.mean(self.model.last_eval_results)
        return 0.0

    def _should_advance_stage(self, performance: float) -> bool:
        """Entscheide ob nächste Stage erreicht werden soll"""
        if self.current_stage >= len(self.curriculum_stages) - 1:
            return False

        threshold = self.stage_thresholds.get(self.current_stage + 1, 0.5)
        return performance >= threshold

    def _advance_to_next_stage(self):
        """Wechsle zur nächsten Curriculum Stage"""
        self.current_stage += 1
        self.steps_in_current_stage = 0

        if self.verbose > 0:
            print(f"\n{'=' * 60}")
            print(f"ADVANCING TO CURRICULUM STAGE {self.current_stage + 1}")
            print(f"New tasks: {self.curriculum_stages[self.current_stage]}")
            print(f"{'=' * 60}\n")