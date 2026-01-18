"""
Meta-World MT3 and MT10 Training Script with Stable Baselines3 (SAC) + Multi-Head Critic + Curriculum

Same structure as your original script:
- CONTINUE_TRAINING False -> Phase 1 (fresh training, save model + replay buffer)
- CONTINUE_TRAINING True  -> Phase 2 (load model + replay buffer, continue, save to SECOND_*)

Manual curriculum:
- You select PHASE manually, which defines env_list
- Append deterministic one-hot task id based on env_list

Reward function: reward_function_version='v3'
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import metaworld

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.sac.policies import SACPolicy

from helper_classes_multihead.Callbacks.ReplayBufferCheckpointCallback import ReplayBufferCheckpointCallback
from helper_classes_multihead.utilities.MultiheadCritic import MultiHeadSACPolicy
from helper_classes_multihead.WrapperClasses.TaskIdWrapper import TaskIdVecEnvWrapper
from helper_classes_multihead.WrapperClasses.GymnasiumVecEnvAdapter import GymnasiumVecEnvAdapter
from helper_classes_multihead.utilities.MetaworldTasks import MT10_TASKS, MT3_TASKS 
from helper_classes_multihead.utilities.AlgorithmFactory import make_factory_SAC 


# ------------------------------------------------------------
# Env builders
# ------------------------------------------------------------
def make_custom_mt_env(envs_list, rank=0, seed=0, max_episode_steps=500, terminate_on_success=False):
    raw_env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        vector_strategy="sync",
        seed=seed + rank,
        envs_list=envs_list,
        reward_function_version="v3",
        max_episode_steps=max_episode_steps,
        terminate_on_success=terminate_on_success,
    )
    return GymnasiumVecEnvAdapter(raw_env)

def make_mt3_env(rank=0, seed=0, max_episode_steps=500, terminate_on_success=False):
    return make_custom_mt_env(["reach-v3", "push-v3", "pick-place-v3"], rank, seed, max_episode_steps, terminate_on_success)

def make_mt10_env(rank=0, seed=0, max_episode_steps=500, terminate_on_success=False):
    raw_env = gym.make_vec(
        'Meta-World/MT10',
        vector_strategy='sync',
        seed=seed + rank,
        reward_function_version='v3',
        max_episode_steps=max_episode_steps,
        terminate_on_success=terminate_on_success,
    )
    env = GymnasiumVecEnvAdapter(raw_env)
    return env

# ------------------------------------------------------------
# Curriculum-Learning
# ------------------------------------------------------------

# Manual phases (define new phases if needed)
def mt3_curriculum_phases():
    phase0 = ["push-v3"] * 15 + ["reach-v3"] * 15 + ["pick-place-v3"] * 0
    #phase1 = ["push-v3"] * 15 + ["reach-v3"] * 15 
    return [phase0]

def mt10_curriculum_phases():

    phase0 = (
        ["reach-v3"] * 6 +
        ["push-v3"] * 6 +
        ["button-press-topdown-v3"] * 3 +
        ["drawer-open-v3"] * 3 +
        ["drawer-close-v3"] * 3 +
        ["window-open-v3"] * 3 +
        ["window-close-v3"] * 3 +
        ["door-open-v3"] * 3 +
        ["peg-insert-side-v3"] * 0 +
        ["pick-place-v3"] * 0
    )

    #phase1 = (
    #    ["reach-v3"] * 2 +
    #    ["push-v3"] * 2 +
    #    ["button-press-topdown-v3"] * 7 +
    #    ["drawer-open-v3"] * 2 +
    #    ["drawer-close-v3"] * 2 +
    #    ["window-open-v3"] * 2 +
    #    ["window-close-v3"] * 2 +
    #    ["door-open-v3"] * 7 +
    #    ["peg-insert-side-v3"] * 2 +
    #    ["pick-place-v3"] * 2
    #)
  
    return [phase0]

# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# main
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------

if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # -------------------- Experiment Setup --------------------
    ALGORITHM = "SAC" #Only SAC implemented
    MT_N = "MT10" #MT3 or MT10

    # -------------------- Training Strategy --------------------
    CURRICULUM = False # False = standard MT10 or MT3 list, True = use curriculum_phases()
    MULTI_HEAD = True # False = "MlpPolicy" (standard), True = MultiHeadSACPolicy

    CONTINUE_TRAINING = False    # False = start new (FIRST_PHASE), True = load model (SECOND_PHASE)
    USE_REPLAY_BUFFER = False    # False = train without replay buffer, True = load model+replay ---> only SECOND_PHASE
    TERMINATE_ON_SUCCESS = False
    
    # -------------------- Curriculum Settings --------------------
    PHASE = 0  # <-- Curriculum phase set manually
    if MT_N == "MT10":
        PHASE_LISTS = mt10_curriculum_phases()
    else:
        PHASE_LISTS = mt3_curriculum_phases()
    FIRST_PHASE_STEPS = 9_000_000     # <-- steps for new run
    SECOND_PHASE_STEPS = 1_000_000     # <-- steps for load model(+replay)

    # -------------------- Environment Settings --------------------
    SEED = 42
    MAX_EPISODE_STEPS = 500

    # -------------------- Evaluation & Checkpointing --------------------
    EVAL_FREQ = 10_000 
    N_EVAL_EPISODES = 20 
    CHECKPOINT_FREQ = 25_000 
    # ======================================================

    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{MT_N}", exist_ok=True)
    os.makedirs(f"./metaworld_models/checkpoints_{MT_N}_buffer", exist_ok=True)
    os.makedirs(f"./metaworld_models/best_{MT_N}", exist_ok=True)

    # --- Set Paths and Names for the Final Model and Replay Buffer ---
    FIRST_MODEL_PATH = f"./metaworld_models/SAC_{MT_N}_9M"
    FIRST_BUFFER_PATH = f"./metaworld_models/SAC_{MT_N}_9M_replay.pkl"
    SECOND_MODEL_PATH = f"./metaworld_models/SAC_{MT_N}_10M"
    SECOND_BUFFER_PATH = f"./metaworld_models/SAC_{MT_N}_10M_replay.pkl"

    print("=" * 60)
    print(f"Meta-World {MT_N} Training (Manual phases)")
    print(f"Algorithm: {ALGORITHM}")
    print(f"MULTI_HEAD = {MULTI_HEAD}")
    print(f"CONTINUE_TRAINING = {CONTINUE_TRAINING}")
    print(f"USE_REPLAY_BUFFER = {USE_REPLAY_BUFFER}")
    print(f"CURRICULUM = {CURRICULUM}")
    if CURRICULUM:
      print(f"CURRICULUM PHASE = {PHASE}")
    print("=" * 60)

    # ------------------ make env ------------------
    if MT_N == "MT10":
        TASK_NAME_TO_ID = {name: i for i, name in enumerate(MT10_TASKS)}
        N_TASKS = len(MT10_TASKS)
    else:
        TASK_NAME_TO_ID = {name: i for i, name in enumerate(MT3_TASKS)}
        N_TASKS = len(MT3_TASKS)


    if not CURRICULUM:
          # ------------------ Train env ------------------
          print(f"Creating {MT_N} training environment ...")
          if MT_N == "MT10":
              ENV_LIST = MT10_TASKS
              env = make_mt10_env(0, SEED, MAX_EPISODE_STEPS, False)
          else:
              ENV_LIST = MT3_TASKS
              env = make_mt3_env(0, SEED, MAX_EPISODE_STEPS, False)
          env = VecMonitor(env)
          env = TaskIdVecEnvWrapper(env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)

    else:
          ENV_LIST = PHASE_LISTS[PHASE]
          # ------------------ Train curriculum env ------------------
          print(f"Creating {MT_N} curriculum training environment ...")
          env = make_custom_mt_env(ENV_LIST, 0, SEED, MAX_EPISODE_STEPS, TERMINATE_ON_SUCCESS)
          env = VecMonitor(env)
          env = TaskIdVecEnvWrapper(env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)
          
          # ------------------ Eval curriculum env ------------------ #not used
          #print(f"Creating {MT_N} curriculum evaluation environment ...")
          #eval_env = make_custom_mt_env(ENV_LIST, 0, SEED + 1000, MAX_EPISODE_STEPS, False)
          #eval_env = VecMonitor(eval_env)
          #eval_env = TaskIdVecEnvWrapper(eval_env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)

    # ------------------ Eval env ------------------
    print(f"Creating {MT_N} evaluation environment ...")
    if MT_N == "MT10":
        ENV_LIST = MT10_TASKS
        eval_env = make_mt10_env(0, SEED + 1000, MAX_EPISODE_STEPS, False)
    else:
        ENV_LIST = MT3_TASKS
        eval_env = make_mt3_env(0, SEED + 1000, MAX_EPISODE_STEPS, False)
    eval_env = VecMonitor(eval_env)
    eval_env = TaskIdVecEnvWrapper(eval_env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)


    num_envs = getattr(env, "num_envs", 1)
    print(f"  -> {MT_N} num_envs = {num_envs}")
    
    o = env.reset()
    print("obs shape:", o.shape)
    if MT_N == "MT10":
        print("onehot counts:", np.sum(o[:, -10:], axis=0))
    else:
        print("onehot counts:", np.sum(o[:, -3:], axis=0))

    # Action dim
    action_space = env.action_space
    n_actions = action_space.shape[0]

    # steps for this run
    total_timesteps = FIRST_PHASE_STEPS if not CONTINUE_TRAINING else SECOND_PHASE_STEPS
    TASK_ID_SLICE = slice(-N_TASKS, None)


    # ------------------ Model create/load ------------------
    if ALGORITHM != "SAC":
        raise ValueError("Only SAC implemented.")

    model = make_factory_SAC(env, MULTI_HEAD, CONTINUE_TRAINING, MT_N, N_TASKS, TASK_ID_SLICE, SEED, FIRST_MODEL_PATH, USE_REPLAY_BUFFER, FIRST_BUFFER_PATH)

    # ------------------ Callbacks ------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{MT_N}/",
        name_prefix=f"sac_{MT_N}",
        verbose=1,
    )

    buffer_checkpoint_callback = ReplayBufferCheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{MT_N}_buffer/",
        name_prefix=f"sac_{MT_N}",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{MT_N}/",
        log_path=f"./metaworld_logs/eval_{MT_N}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False,
    )

    callbacks = [eval_callback, checkpoint_callback, buffer_checkpoint_callback]

    # ------------------ Train ------------------
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=not CONTINUE_TRAINING,
    )

    # ------------------ Save ------------------
    print("\nSaving final model...")
    if not CONTINUE_TRAINING:
        model.save(FIRST_MODEL_PATH)
        model.save_replay_buffer(FIRST_BUFFER_PATH)
        print(f"Saved: {FIRST_MODEL_PATH}.zip")
        print(f"Saved replay: {FIRST_BUFFER_PATH}")
    else:
        model.save(SECOND_MODEL_PATH)
        model.save_replay_buffer(SECOND_BUFFER_PATH)
        print(f"Saved: {SECOND_MODEL_PATH}.zip")
        print(f"Saved replay: {SECOND_BUFFER_PATH}")

    env.close()
    eval_env.close()
