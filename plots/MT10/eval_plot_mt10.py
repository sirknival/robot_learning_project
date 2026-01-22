import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


SHOW_STD = True   #std on/off
STEP = 2           # polot each n. point


#MT10
CURVES = [
    ("./metaworld_logs/1_evaluations_MT10_default.npz", "MT10-SAC"),
    ("./metaworld_logs/2_evaluations_MT10_TaskCond.npz", "MT10-SAC with One-hot Encoding"),
    ("./metaworld_logs/3_evaluations_MT10_TaskCond_Multihead.npz", "MT10-SAC with One-hot Encoding + Multi-head Critic"),
]

plt.figure()

for path, label in CURVES:
    data = np.load(path)
    timesteps = data["timesteps"][::STEP]

    # -------- determine format -------- 
    if "mean_reward" in data.files:
        # reconstructed / combined format
        mean_r = data["mean_reward"][::STEP]
        std_r  = data["std_reward"][::STEP] if "std_reward" in data.files else None

    elif "results" in data.files:
        # original SB3 evaluations.npz
        results = data["results"]
        mean_r = results.mean(axis=1)[::STEP]
        std_r  = results.std(axis=1)[::STEP]

    else:
        raise KeyError(f"Unknown eval format in {path}")

    # -------- plot mean curve --------
    line, = plt.plot(timesteps, mean_r, label=label)

    if SHOW_STD and std_r is not None and np.isfinite(std_r).any():
        plt.fill_between(
            timesteps,
            mean_r - std_r,
            mean_r + std_r,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0
        )

plt.xlabel("Timesteps")
plt.ylabel("Eval reward")
plt.xlim(0, 35e6)
plt.ylim(0, 800_000)
plt.grid(True)
plt.legend(loc='upper left')

plt.savefig("./metaworld_logs/eval_curve_MT10.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
