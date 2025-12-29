import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from training_setup_multitask.WrapperClasses.OneHotTaskWrapper import OneHotTaskWrapper


# ... Hier steht deine OneHotTaskWrapper Klasse ...

class MetaWorldEvaluator:
    def __init__(self, task_list, max_episode_steps=200, seed=42):
        """
        Evaluator, der den existierenden OneHotTaskWrapper (VecEnv) nutzt.

        :param task_list: Liste aller Task-Namen (wird an OneHotTaskWrapper übergeben)
        """
        self.task_list = task_list
        self.max_episode_steps = max_episode_steps
        self.seed = seed

    def _make_env_fn(self, task_name):
        """
        Gibt eine Funktion zurück, die das Environment erstellt.
        Wichtig: Setzt 'task_name' im Environment, damit dein Wrapper es findet.
        """

        def _init():
            env = gym.make(
                "Meta-World/MT1",
                env_name=task_name,
                seed=self.seed,
                render_mode="rgb_array",
                reward_function_version="v3",
                max_episode_steps=self.max_episode_steps,
                terminate_on_success=False,
            )
            # WICHTIG: Damit dein Wrapper den Namen findet (siehe _extract_task_name Versuch 2)
            env.unwrapped.task_name = task_name
            return env

        return _init

    def evaluate(self, model, num_episodes_per_task=5, deterministic=True):
        """
        Führt die Evaluation durch.
        """
        all_rewards = []
        all_successes = []
        detailed_results = {}

        print(f"Evaluiere {len(self.task_list)} Tasks ({num_episodes_per_task} Episoden/Task)...")

        for task_name in self.task_list:
            task_rewards = []
            task_successes = []

            # 1. Basis VecEnv erstellen (mit 1 Environment für den aktuellen Task)
            # Wir nutzen DummyVecEnv, da dein Wrapper ein VecEnv erwartet
            base_venv = DummyVecEnv([self._make_env_fn(task_name)])

            # 2. Deinen Wrapper anwenden
            # Er sorgt für das One-Hot Encoding basierend auf self.task_list
            eval_env = OneHotTaskWrapper(base_venv, self.task_list)

            # Loop über Episoden
            for i in range(num_episodes_per_task):
                # Reset (Dein Wrapper kümmert sich um die Obs-Erweiterung)
                obs = eval_env.reset()

                done = False
                total_reward = 0.0
                is_success = False

                while not done:
                    # Modell vorhersage
                    action, _ = model.predict(obs, deterministic=deterministic)

                    # Step (Dein Wrapper kümmert sich um Obs-Erweiterung)
                    obs, rewards, dones, infos = eval_env.step(action)

                    total_reward += float(rewards[0])
                    done = bool(dones[0])

                    # Erfolg prüfen
                    if "success" in infos[0] and infos[0]["success"]:
                        is_success = True

                task_rewards.append(total_reward)
                task_successes.append(1.0 if is_success else 0.0)

            # Wichtig: Environment schließen, um Speicherlecks zu vermeiden
            eval_env.close()

            # Statistik berechnen
            mean_r = np.mean(task_rewards)
            mean_s = np.mean(task_successes)
            detailed_results[task_name] = {"reward": mean_r, "success": mean_s}

            all_rewards.extend(task_rewards)
            all_successes.extend(task_successes)

            print(f"  -> {task_name:20s}: Reward={mean_r:8.2f}, Success={mean_s * 100:5.1f}%")

        overall_mean_reward = np.mean(all_rewards)
        overall_success_rate = np.mean(all_successes)

        return overall_mean_reward, overall_success_rate, detailed_results
