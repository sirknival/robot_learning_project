from typing import List, Optional
from stable_baselines3.common.utils import ConstantSchedule, FloatSchedule

class TransferLearningManager:
    """Manager for Transfer Learning between tasks and stages"""

    def __init__(self, base_model_path: Optional[str] = None):
        self.base_model_path = base_model_path
        self.transfer_history = []

    def load_pretrained_model(self, algorithm_class, env, model_path: str):
        """Lade vortrainiertes Modell für Transfer Learning"""
        print(f"\n[Transfer Learning] Loading pretrained model from: {model_path}")
        try:
            model = algorithm_class.load(
                model_path,
                env=env,
                print_system_info=False
            )
            print("[Transfer Learning] ✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"[Transfer Learning] ✗ Failed to load model: {e}")
            return None

    def fine_tune_for_new_tasks(
        self,
        model,
        new_tasks: List[str],
        learning_rate_multiplier: float = 0.1
    ):
        """
        Fine-tune Modell für neue Tasks mit reduzierter Learning Rate
        (Verhindert catastrophic forgetting)
        """
        print(f"\n[Transfer Learning] Fine-tuning for new tasks: {new_tasks}")

        # Reduziere Learning Rate für Fine-Tuning
        original_lr = model.learning_rate
        new_lr = original_lr * learning_rate_multiplier
        model.learning_rate = new_lr

        for optim in [model.actor.optimizer, model.critic.optimizer]:
            for pg in optim.param_groups:
                pg["lr"] *= learning_rate_multiplier
        
        model.lr_schedule = FloatSchedule(ConstantSchedule(new_lr))
            
        print(f"[Transfer Learning] Reduced LR: {original_lr} -> {model.learning_rate}")

        return model

    def save_transfer_checkpoint(
        self,
        model,
        stage: int,
        tasks: List[str],
        save_path: str
    ):
        """Speichere Checkpoint für Transfer Learning"""
        checkpoint_name = f"{save_path}/transfer_stage{stage}"
        model.save(checkpoint_name)

        self.transfer_history.append({
            "stage": stage,
            "tasks": tasks,
            "path": checkpoint_name
        })

        print(f"[Transfer Learning] Saved checkpoint: {checkpoint_name}")
