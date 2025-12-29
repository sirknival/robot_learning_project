from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict
from training_setup_multitask.utilities.MetaWorldEvaluator import MetaWorldEvaluator


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
        print("Debug on_step")
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
        print("Debug eval_cur_stage")

        if self.model is None:
            return 0.0

        # Task Liste holen (entweder aus der Klasse oder global MT3/MT10)
        current_tasks = getattr(self, 'tasks', MT3)

        # Evaluator mit der aktuellen Task-Liste initialisieren
        evaluator = MetaWorldEvaluator(
            task_list=current_tasks,  # Dies wird an deinen OneHotTaskWrapper übergeben
            max_episode_steps=200
        )

        # Evaluation starten (3 Episoden reichen für einen schnellen Check)
        mean_reward, success_rate, _ = evaluator.evaluate(
            model=self.model,
            num_episodes_per_task=3,
            deterministic=True
        )

        # Ergebnisse speichern (optional)
        self.model.last_eval_results = mean_reward
        self.model.last_success_rate = success_rate

        print(f"Stage Evaluation: Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate * 100:.1f}%")

        return mean_reward

    def _should_advance_stage(self, performance: float) -> bool:
        print("Debug should_adv")
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