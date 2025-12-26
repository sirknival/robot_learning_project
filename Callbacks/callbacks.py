checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path="../metaworld_models/checkpoints_MT10/",
    name_prefix=f"{ALGORITHM.lower()}_mt10",
    verbose=1,
)

buffer_checkpoint_callback = ReplayBufferCheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path="../metaworld_models/checkpoints_MT10_buffer/",
    name_prefix=f"{ALGORITHM.lower()}_mt10_buffer",
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="../metaworld_models/best_MT10/",
    log_path="./metaworld_logs/eval_MT10/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1,
    warn=False,
)