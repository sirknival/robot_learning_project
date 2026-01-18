import torch
import os

from stable_baselines3 import SAC

from helper_classes_multihead.utilities.MultiheadCritic import MultiHeadSACPolicy

def make_factory_SAC(env, MULTI_HEAD, CONTINUE_TRAINING, MT_N, N_TASKS, TASK_ID_SLICE, SEED, FIRST_MODEL_PATH, USE_REPLAY_BUFFER, FIRST_BUFFER_PATH):
    
    # ------------------ Adjust Network Hyperparameters ------------------
    POLICY_KWARGS=dict(
                net_arch=[512, 1024, 1024, 512],
                activation_fn=torch.nn.ReLU,
                log_std_init=-3.0,
        )
    
    if not MULTI_HEAD:
        POLICY="MlpPolicy"
    else:
        POLICY=MultiHeadSACPolicy

        POLICY_KWARGS.update({
        "n_tasks": N_TASKS,
        "task_id_slice": TASK_ID_SLICE,
    })

    if not CONTINUE_TRAINING:
        
        # ------------------ Adjust SAC Hyperparameters ------------------
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
    
    return model