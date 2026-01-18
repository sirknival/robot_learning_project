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

from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.sac.policies import SACPolicy


# ------------------------------------------------------------
# Tasks
# ------------------------------------------------------------

MT10_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
]

MT3_TASKS = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    ]


# ------------------------------------------------------------
# Adapter: Gymnasium VectorEnv -> SB3 VecEnv 
# ------------------------------------------------------------
class GymnasiumVecEnvAdapter(VecEnv):
    def __init__(self, gym_vec_env):
        self.gym_vec_env = gym_vec_env

        num_envs = gym_vec_env.num_envs
        observation_space = gym_vec_env.single_observation_space
        action_space = gym_vec_env.single_action_space

        super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)

        self.num_envs = num_envs
        self._actions = None
        self.reset_infos = [dict() for _ in range(self.num_envs)]

    def close(self):
        return self.gym_vec_env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def get_attr(self, name, indices=None):
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]

        if not hasattr(self.gym_vec_env, name):
            return [None for _ in indices]

        value = getattr(self.gym_vec_env, name)

        try:
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.num_envs:
                return [value[i] for i in indices]
        except TypeError:
            pass

        return [value for _ in indices]

    def render(self, mode="human"):
        return self.gym_vec_env.render()

    def reset(self):
        obs, info = self.gym_vec_env.reset()

        if isinstance(info, dict):
            list_infos = [dict() for _ in range(self.num_envs)]
            for k, v in info.items():
                v = np.array(v)
                if v.ndim > 0 and v.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v[i]
                else:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v.item() if v.ndim == 0 else v
            self.reset_infos = list_infos
        elif isinstance(info, list):
            self.reset_infos = info
        else:
            self.reset_infos = [dict() for _ in range(self.num_envs)]

        return obs

    def seed(self, seed=None):
        try:
            self.gym_vec_env.reset(seed=seed)
        except TypeError:
            pass

    def set_attr(self, name, values, indices=None):
        if hasattr(self.gym_vec_env, name):
            setattr(self.gym_vec_env, name, values)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.gym_vec_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)

        if isinstance(infos, dict):
            list_infos = [dict() for _ in range(self.num_envs)]
            for k, v in infos.items():
                v = np.array(v)
                if v.ndim > 0 and v.shape[0] == self.num_envs:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v[i]
                else:
                    for i in range(self.num_envs):
                        list_infos[i][k] = v.item() if v.ndim == 0 else v
            infos = list_infos

        if isinstance(infos, list) and len(infos) == self.num_envs:
            for i in range(self.num_envs):
                if dones[i]:
                    fin = infos[i].get("final_observation", None)
                    if fin is not None and "terminal_observation" not in infos[i]:
                        infos[i]["terminal_observation"] = fin
                    if "final_info" in infos[i] and "terminal_info" not in infos[i]:
                        infos[i]["terminal_info"] = infos[i]["final_info"]

        return obs, rewards, dones, infos


# ------------------------------------------------------------
# Replay Buffer Checkpoint
# ------------------------------------------------------------
class ReplayBufferCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.model is None:
            return True
        if self.model.num_timesteps > 0 and self.model.num_timesteps % (self.save_freq * 10) == 0:
            filename = f"{self.name_prefix}_{self.model.num_timesteps}_steps_replay.pkl"
            path = os.path.join(self.save_path, filename)
            if self.verbose > 0:
                print(f"Saving replay buffer checkpoint to: {path}")
            self.model.save_replay_buffer(path)
        return True


# ------------------------------------------------------------
# Deterministic Task One-Hot Wrapper
# ------------------------------------------------------------
class AppendFixedTaskIdVecWrapper(VecEnvWrapper):
    """
    Appends one-hot task id to obs based on envs_list:
    sub-env i corresponds to envs_list[i] (including duplicates).
    """

    def __init__(self, venv: VecEnv, envs_list: list[str], task_name_to_id: dict, n_tasks: int):
        super().__init__(venv)
        self.envs_list = list(envs_list)
        self.task_name_to_id = dict(task_name_to_id)
        self.n_tasks = int(n_tasks)

        if len(self.envs_list) != self.num_envs:
            raise RuntimeError(
                f"AppendFixedTaskIdVecWrapper: len(envs_list)={len(self.envs_list)} "
                f"but venv.num_envs={self.num_envs}. Must match."
            )

        idx = []
        for name in self.envs_list:
            if name not in self.task_name_to_id:
                raise RuntimeError(f"Unknown task '{name}'. Known: {list(self.task_name_to_id.keys())}")
            idx.append(self.task_name_to_id[name])
        self.task_idx_per_env = np.array(idx, dtype=np.int64)

        obs_space = venv.observation_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise RuntimeError("AppendFixedTaskIdVecWrapper expects Box observation_space")

        low = np.asarray(obs_space.low, dtype=np.float32)
        high = np.asarray(obs_space.high, dtype=np.float32)
        low = np.concatenate([low, -np.ones(self.n_tasks, dtype=np.float32)], axis=0)
        high = np.concatenate([high, np.ones(self.n_tasks, dtype=np.float32)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _onehot(self) -> np.ndarray:
        oh = np.zeros((self.num_envs, self.n_tasks), dtype=np.float32)
        oh[np.arange(self.num_envs), self.task_idx_per_env] = 1.0
        return oh

    def reset(self):
        obs = self.venv.reset()
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._onehot()], axis=1)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._onehot()], axis=1), rewards, dones, infos


# ------------------------------------------------------------
# Multi-Head Critic
# ------------------------------------------------------------
class MultiHeadContinuousCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch=(256, 256, 256),
        activation_fn=nn.ReLU,
        n_tasks: int = 3,
        task_id_slice=slice(-3, None),
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = int(features_dim)

        self.n_tasks = int(n_tasks)
        self.task_id_slice = task_id_slice

        action_dim = action_space.shape[0]
        in_dim = self.features_dim + action_dim

        self.q1_heads = nn.ModuleList()
        self.q2_heads = nn.ModuleList()
        for _ in range(self.n_tasks):
            self.q1_heads.append(nn.Sequential(*create_mlp(in_dim, 1, list(net_arch), activation_fn)))
            self.q2_heads.append(nn.Sequential(*create_mlp(in_dim, 1, list(net_arch), activation_fn)))

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def _task_indices_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        onehot = obs[..., self.task_id_slice]
        if onehot.shape[-1] != self.n_tasks:
            raise RuntimeError(f"Task one-hot dim={onehot.shape[-1]} but expected {self.n_tasks}")
        return torch.argmax(onehot, dim=-1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.features_extractor(obs)
        x = torch.cat([features, actions], dim=1)
        task_idx = self._task_indices_from_obs(obs)

        q1 = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        q2 = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)

        for t in range(self.n_tasks):
            mask = task_idx == t
            if torch.any(mask):
                xt = x[mask]
                q1[mask] = self.q1_heads[t](xt)
                q2[mask] = self.q2_heads[t](xt)
        return q1, q2

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1, _ = self.forward(obs, actions)
        return q1

    def q2_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _, q2 = self.forward(obs, actions)
        return q2


class MultiHeadSACPolicy(SACPolicy):
    def __init__(self, *args, n_tasks=3, task_id_slice=slice(-3, None), **kwargs):
        self._mh_n_tasks = int(n_tasks)
        self._mh_task_id_slice = task_id_slice
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor: BaseFeaturesExtractor = None):
        if features_extractor is None:
            features_extractor = self.make_features_extractor()

        return MultiHeadContinuousCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=features_extractor.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            n_tasks=self._mh_n_tasks,
            task_id_slice=self._mh_task_id_slice,
        )


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
    ALGORITHM = "SAC" #Only SAC implemented

    MT_N = "MT10" #MT3 or MT10

    #advanced methods
    CURRICULUM = False # False = standard MT10 or MT3 list, True = use curriculum_phases()
    MULTI_HEAD = False # False = "MlpPolicy" (standard), True = MultiHeadSACPolicy

    CONTINUE_TRAINING = False    # False = start new (FIRST_PHASE), True = load model (SECOND_PHASE)
    USE_REPLAY_BUFFER = False    # False = train without replay buffer, True = load model+replay ---> only SECOND_PHASE
    TERMINATE_ON_SUCCESS = False


    PHASE = 0  # <-- Curriculum phase set manually
    if MT_N == "MT10":
        PHASE_LISTS = mt10_curriculum_phases()
    else:
        PHASE_LISTS = mt3_curriculum_phases()
    

    FIRST_PHASE_STEPS = 9_000_000     # <-- steps for new run
    SECOND_PHASE_STEPS = 1_000_000     # <-- steps for load model(+replay)

    SEED = 42
    MAX_EPISODE_STEPS = 500

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
          env = AppendFixedTaskIdVecWrapper(env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)

    else:
          ENV_LIST = PHASE_LISTS[PHASE]
          # ------------------ Train curriculum env ------------------
          print(f"Creating {MT_N} curriculum training environment ...")
          env = make_custom_mt_env(ENV_LIST, 0, SEED, MAX_EPISODE_STEPS, TERMINATE_ON_SUCCESS)
          env = VecMonitor(env)
          env = AppendFixedTaskIdVecWrapper(env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)
          
          # ------------------ Eval curriculum env ------------------ #not used
          #print(f"Creating {MT_N} curriculum evaluation environment ...")
          #eval_env = make_custom_mt_env(ENV_LIST, 0, SEED + 1000, MAX_EPISODE_STEPS, False)
          #eval_env = VecMonitor(eval_env)
          #eval_env = AppendFixedTaskIdVecWrapper(eval_env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)

    # ------------------ Eval env ------------------
    print(f"Creating {MT_N} evaluation environment ...")
    if MT_N == "MT10":
        ENV_LIST = MT10_TASKS
        eval_env = make_mt10_env(0, SEED + 1000, MAX_EPISODE_STEPS, False)
    else:
        ENV_LIST = MT3_TASKS
        eval_env = make_mt3_env(0, SEED + 1000, MAX_EPISODE_STEPS, False)
    eval_env = VecMonitor(eval_env)
    eval_env = AppendFixedTaskIdVecWrapper(eval_env, envs_list=ENV_LIST, task_name_to_id=TASK_NAME_TO_ID, n_tasks=N_TASKS)


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

    if not MULTI_HEAD:
        POLICY="MlpPolicy"

        POLICY_KWARGS=dict(
                net_arch=[512, 1024, 1024, 512],
                activation_fn=torch.nn.ReLU,
                log_std_init=-3.0,
        )
    else:
        POLICY=MultiHeadSACPolicy

        POLICY_KWARGS=dict(
                net_arch=[512, 1024, 1024, 512],
                activation_fn=torch.nn.ReLU,
                log_std_init=-3.0,
                n_tasks=N_TASKS,
                task_id_slice=TASK_ID_SLICE,
        )

    if not CONTINUE_TRAINING:
        model = SAC(
            policy=POLICY,
            env=env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5_000, 
            batch_size=256, 
            tau=0.005, 
            gamma=0.99,
            train_freq=1,
            gradient_steps=1, 
            ent_coef="auto",
            target_entropy="auto",
            use_sde=False,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=f"./metaworld_logs/{MT_N}_SAC/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        if not os.path.exists(FIRST_MODEL_PATH + ".zip"):
            raise FileNotFoundError(f"Cannot find model: {FIRST_MODEL_PATH}.zip")

        print(f"Loading model from {FIRST_MODEL_PATH}.zip ...")
        model = SAC.load(FIRST_MODEL_PATH + ".zip", env=env)
      
        if USE_REPLAY_BUFFER:
            if not os.path.exists(FIRST_BUFFER_PATH):
              raise FileNotFoundError(f"Cannot find replay buffer: {FIRST_BUFFER_PATH}")

            print(f"Loading replay buffer from {FIRST_BUFFER_PATH} ...")
            model.load_replay_buffer(FIRST_BUFFER_PATH)
        
        #override parameter for loaded model (test)
        #model.ent_coef = 0.5
        #model.learning_rate=5e-5
        #model.target_entropy = -0.5 * n_actions
        #model.gradient_steps=1

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
