import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional, Tuple
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3 import SAC, TD3, DDPG
import os


class TaskEvaluator:
    """
    Evaluates model performance on specific tasks during training.
    Used by ProgressiveTaskCallback to determine curriculum stage transitions.
    """

    def __init__(
            self,
            env_factory,
            n_eval_episodes: int = 10,
            max_episode_steps: int = 500,
            seed: int = 42,
            one_hot_dim: int = 10,  # Fixed dimension for all tasks
            verbose: bool = True
    ):
        """
        Args:
            env_factory: MetaWorldEnvFactory instance for creating eval envs
            n_eval_episodes: Number of episodes to run per task
            max_episode_steps: Maximum steps per episode
            seed: Random seed for evaluation
            one_hot_dim: Fixed dimension for one-hot encoding (default: 10 for MT10)
            verbose: Print evaluation progress
        """
        self.env_factory = env_factory
        self.n_eval_episodes = n_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.one_hot_dim = one_hot_dim
        self.verbose = verbose

        # Cache for evaluation results
        self.last_results = {}

    def evaluate_tasks(
            self,
            model,
            tasks: List[str],
            deterministic: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on a list of tasks

        Args:
            model: Trained RL model (SAC, TD3, DDPG)
            tasks: List of task names to evaluate
            deterministic: Use deterministic policy

        Returns:
            Dictionary with results per task:
            {
                "task_name": {
                    "mean_reward": float,
                    "std_reward": float,
                    "success_rate": float,
                    "mean_episode_length": float
                }
            }
        """
        results = {}

        for task in tasks:
            if self.verbose:
                print(f"  Evaluating task: {task}...")

            task_results = self._evaluate_single_task(model, task, deterministic)
            results[task] = task_results

            if self.verbose:
                print(f"    Success rate: {task_results['success_rate']:.1%}")
                print(f"    Mean reward: {task_results['mean_reward']:.2f}")

        # Cache results
        self.last_results = results

        return results

    def _evaluate_single_task(
            self,
            model,
            task_name: str,
            deterministic: bool
    ) -> Dict[str, float]:
        """
        Evaluate model on a single task

        Returns:
            Dictionary with evaluation metrics
        """
        # Create evaluation environment
        eval_env = self.env_factory.make_mt1_env(
            task_name=task_name,
            seed=self.seed,
            max_episode_steps=self.max_episode_steps,
            n_envs=1  # Single environment for evaluation
        )

        # Wrap with task encoding - use fixed one_hot_dim
        from training_setup_multitask.WrapperClasses.OneHotTaskWrapper import OneHotTaskWrapper

        # Create task list that matches the training setup
        # For evaluation, we need all MT10 tasks in the same order
        from training_setup_multitask.utilities.MetaworldTasks import MT10_TASKS
        eval_env = OneHotTaskWrapper(eval_env, MT10_TASKS, one_hot_dim=self.one_hot_dim)

        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        # Run evaluation episodes
        for episode in range(self.n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_success = False

            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)

                # Step environment
                obs, reward, done, info = eval_env.step(action)

                # Extract values (handle VecEnv format)
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(info, list):
                    info = info[0]

                episode_reward += reward
                episode_length += 1

                # Check for success
                if isinstance(info, dict) and "success" in info:
                    if info["success"]:
                        episode_success = True

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(1.0 if episode_success else 0.0)

        # Close environment
        eval_env.close()

        # Calculate statistics
        results = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "success_rate": float(np.mean(episode_successes)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "n_episodes": self.n_eval_episodes
        }

        return results

    def get_overall_performance(self, results: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate overall performance metric across all tasks

        Args:
            results: Dictionary of per-task results

        Returns:
            Overall performance score (average success rate)
        """
        if not results:
            return 0.0

        success_rates = [task_results["success_rate"] for task_results in results.values()]
        return float(np.mean(success_rates))


class FinalModelEvaluator:
    """
    Comprehensive evaluation of trained models with detailed statistics.
    Used for final benchmarking after training is complete.
    """

    def __init__(
            self,
            env_factory,
            tasks: List[str],
            n_eval_episodes: int = 100,
            max_episode_steps: int = 500,
            seed: int = 42,
            one_hot_dim: int = 10,  # Fixed dimension for consistency
            save_results: bool = True,
            results_dir: str = "./evaluation_results",
            verbose: bool = True
    ):
        """
        Args:
            env_factory: MetaWorldEnvFactory instance
            tasks: List of tasks to evaluate
            n_eval_episodes: Episodes per task
            max_episode_steps: Max steps per episode
            seed: Random seed
            one_hot_dim: Fixed dimension for one-hot encoding
            save_results: Save results to file
            results_dir: Directory for saving results
            verbose: Print detailed output
        """
        self.env_factory = env_factory
        self.tasks = tasks
        self.n_eval_episodes = n_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.one_hot_dim = one_hot_dim
        self.save_results = save_results
        self.results_dir = results_dir
        self.verbose = verbose

        if save_results:
            os.makedirs(results_dir, exist_ok=True)

    def evaluate_model(
            self,
            model_path: str,
            algorithm: str = "SAC",
            experiment_name: str = "evaluation"
    ) -> Dict:
        """
        Run comprehensive evaluation on a trained model

        Args:
            model_path: Path to saved model (.zip file)
            algorithm: Algorithm type ("SAC", "TD3", "DDPG")
            experiment_name: Name for this evaluation run

        Returns:
            Dictionary with complete evaluation results
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"FINAL MODEL EVALUATION: {experiment_name}")
            print("=" * 70)
            print(f"Model: {model_path}")
            print(f"Algorithm: {algorithm}")
            print(f"Tasks: {len(self.tasks)}")
            print(f"Episodes per task: {self.n_eval_episodes}")
            print("=" * 70 + "\n")

        # Load model
        model = self._load_model(model_path, algorithm)

        # Evaluate all tasks
        all_results = {
            "experiment_name": experiment_name,
            "model_path": model_path,
            "algorithm": algorithm,
            "tasks": self.tasks,
            "n_eval_episodes": self.n_eval_episodes,
            "per_task_results": {},
            "overall_statistics": {}
        }

        task_evaluator = TaskEvaluator(
            env_factory=self.env_factory,
            n_eval_episodes=self.n_eval_episodes,
            max_episode_steps=self.max_episode_steps,
            seed=self.seed,
            one_hot_dim=self.one_hot_dim,
            verbose=self.verbose
        )

        # Evaluate each task
        per_task_results = task_evaluator.evaluate_tasks(
            model=model,
            tasks=self.tasks,
            deterministic=True
        )

        all_results["per_task_results"] = per_task_results

        # Calculate overall statistics
        all_results["overall_statistics"] = self._calculate_overall_stats(per_task_results)

        # Print summary
        if self.verbose:
            self._print_summary(all_results)

        # Save results
        if self.save_results:
            self._save_results(all_results, experiment_name)

        return all_results

    def _load_model(self, model_path: str, algorithm: str):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.verbose:
            print(f"Loading model from: {model_path}")

        algorithm_class = {
            "SAC": SAC,
            "TD3": TD3,
            "DDPG": DDPG
        }

        if algorithm not in algorithm_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model = algorithm_class[algorithm].load(model_path)

        if self.verbose:
            print("✓ Model loaded successfully\n")

        return model

    def _calculate_overall_stats(self, per_task_results: Dict) -> Dict:
        """Calculate overall statistics across all tasks"""
        all_rewards = []
        all_success_rates = []
        all_episode_lengths = []

        for task_results in per_task_results.values():
            all_rewards.append(task_results["mean_reward"])
            all_success_rates.append(task_results["success_rate"])
            all_episode_lengths.append(task_results["mean_episode_length"])

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_success_rate": float(np.mean(all_success_rates)),
            "std_success_rate": float(np.std(all_success_rates)),
            "mean_episode_length": float(np.mean(all_episode_lengths)),
            "total_episodes": len(self.tasks) * self.n_eval_episodes
        }

    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Overall statistics
        overall = results["overall_statistics"]
        print("\nOverall Performance:")
        print(f"  Mean Success Rate: {overall['mean_success_rate']:.1%} (±{overall['std_success_rate']:.1%})")
        print(f"  Mean Reward: {overall['mean_reward']:.2f} (±{overall['std_reward']:.2f})")
        print(f"  Mean Episode Length: {overall['mean_episode_length']:.1f} steps")
        print(f"  Total Episodes: {overall['total_episodes']}")

        # Per-task results
        print("\nPer-Task Performance:")
        print(f"{'Task':<30} {'Success Rate':<15} {'Mean Reward':<15} {'Episodes':<10}")
        print("-" * 70)

        for task_name, task_results in results["per_task_results"].items():
            success_rate = f"{task_results['success_rate']:.1%}"
            mean_reward = f"{task_results['mean_reward']:.2f}"
            n_episodes = f"{task_results['n_episodes']}"

            print(f"{task_name:<30} {success_rate:<15} {mean_reward:<15} {n_episodes:<10}")

        print("=" * 70 + "\n")

    def _save_results(self, results: Dict, experiment_name: str):
        """Save results to JSON file"""
        import json

        output_path = os.path.join(self.results_dir, f"{experiment_name}_results.json")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"✓ Results saved to: {output_path}\n")

    def compare_models(
            self,
            model_configs: List[Dict[str, str]],
            output_file: str = "model_comparison.txt"
    ):
        """
        Compare multiple models

        Args:
            model_configs: List of dicts with keys: 'name', 'path', 'algorithm'
            output_file: File to save comparison results
        """
        all_results = []

        for config in model_configs:
            name = config.get("name", "Model")
            path = config["path"]
            algorithm = config.get("algorithm", "SAC")

            results = self.evaluate_model(path, algorithm, name)
            all_results.append({
                "name": name,
                "results": results
            })

        # Print comparison
        self._print_comparison(all_results)

        # Save comparison
        if self.save_results:
            self._save_comparison(all_results, output_file)

    def _print_comparison(self, all_results: List[Dict]):
        """Print comparison table"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(f"\n{'Model':<25} {'Success Rate':<20} {'Mean Reward':<20}")
        print("-" * 70)

        for result in all_results:
            name = result["name"]
            overall = result["results"]["overall_statistics"]
            success = f"{overall['mean_success_rate']:.1%}"
            reward = f"{overall['mean_reward']:.2f}"

            print(f"{name:<25} {success:<20} {reward:<20}")

        print("=" * 70 + "\n")

    def _save_comparison(self, all_results: List[Dict], output_file: str):
        """Save comparison to text file"""
        output_path = os.path.join(self.results_dir, output_file)

        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            for result in all_results:
                name = result["name"]
                overall = result["results"]["overall_statistics"]
                per_task = result["results"]["per_task_results"]

                f.write(f"Model: {name}\n")
                f.write(f"  Overall Success Rate: {overall['mean_success_rate']:.1%}\n")
                f.write(f"  Overall Mean Reward: {overall['mean_reward']:.2f}\n\n")

                f.write("  Per-Task Results:\n")
                for task, task_results in per_task.items():
                    f.write(f"    {task}: {task_results['success_rate']:.1%} success, "
                            f"{task_results['mean_reward']:.2f} reward\n")

                f.write("\n" + "-" * 70 + "\n\n")

        if self.verbose:
            print(f"✓ Comparison saved to: {output_path}\n")
