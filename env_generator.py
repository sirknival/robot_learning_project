import gymnasium as gym
import metaworld
from stable_baselines3.common.monitor import Monitor
from GymnasiumVecEnvAdapter import *


"""
def make_env(task_name='reach-v3', rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    
    Create and wrap the Meta-World MTX environment.

    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3', 'push-v3')
        rank: Index of the subprocess (for parallel envs)
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default: 500)
        normalize_reward: Whether to normalize rewards (optional, can improve learning)
    
    def _init():
        # Create Meta-World MT1 environment
        import gymnasium as gym
        import metaworld

        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,  # Different seed for each parallel env
            reward_function_version='v3',  # Use v2 reward (default, more stable)
            max_episode_steps=max_episode_steps,  # Episode length
            terminate_on_success=False,  # Don't terminate early on success (for training)
        )

        # Optional: Normalize rewards for more stable learning
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)

        # Monitor wrapper for logging episode statistics
        env = Monitor(env)

        return env

    return _init

def make_mt10_env(seed):
    def _init():
        mt10 = metaworld.MT10(seed=seed)
        # Wähle zufällig einen Task pro Episode
        task_name = list(mt10.train_classes.keys())[0]  # z. B. erster Task
        env = mt10.train_classes[task_name]
        return env
    return _init

def make_mt_env(env_spec, rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    
    Erstellt die korrekte Meta-World Environment-Instanz.
    

    current_seed = seed + rank

    # --- MT1 - Single Task ---
    if isinstance(env_spec, str) and env_spec.endswith('-v3'):
        env = gym.make(
            'Meta-World/MT1',
            env_name=env_spec,
            seed=current_seed,
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,
            terminate_on_success=False)

    # --- MT10 / MT50 ---
    elif env_spec == 'MT10' or env_spec == 'MT50':
        env = gym.make(
            f'Meta-World/{env_spec}',
            seed=current_seed,
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,
        )

    # --- Custom MT3  ---
    elif isinstance(env_spec, list):
        env = gym.make(
            'Meta-World/custom-mt-envs',
            seed=current_seed,
            envs_list=env_spec,
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,
            terminate_on_success=False)

    else:
        raise ValueError(f"Ungültige env_spec: {env_spec}")
    env = Monitor(env)
    return env
"""

def make_mt3_env(rank=0, seed=0, max_episode_steps=500):
    raw_env = gym.make_vec(
        'Meta-World/custom-mt-envs',
        vector_strategy='sync',
        seed=seed + rank,
        envs_list=['reach-v3', 'push-v3', 'pick-place-v3'],
        reward_function_version='v3',
        max_episode_steps=max_episode_steps,
        terminate_on_success=False)
    env = GymnasiumVecEnvAdapter(raw_env)
    return env


def make_mt10_env(rank=0, seed=0, max_episode_steps=500):
    raw_env = gym.make_vec(
        'Meta-World/MT10',
        vector_strategy='sync',
        seed=seed + rank,
        reward_function_version='v3',
        max_episode_steps=max_episode_steps,
        terminate_on_success=False)
    env = GymnasiumVecEnvAdapter(raw_env)
    return env
