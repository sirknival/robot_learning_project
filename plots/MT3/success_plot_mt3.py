import os, glob, json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

MT_N = "MT3"
RESULT_DIR = f"./metaworld_success/"
files = [
    f for f in os.listdir(RESULT_DIR)
    if os.path.isfile(os.path.join(RESULT_DIR, f))
]

CUSTOM_LABELS = ["MT3-SAC", "MT3-SAC with One-hot Encoding", "MT3-SAC with One-hot Encoding + Transfer Learning", "MT3-SAC with One-hot Encoding + Multi-head Critic + Curriculum Learning V1"]

if not files:
    raise FileNotFoundError(f"No result json files found in {RESULT_DIR}")

runs = []
for fp in files:
    full_path = os.path.join(RESULT_DIR, fp)
    with open(full_path, "r") as f:
        runs.append(json.load(f))

tasks = list(runs[0]["per_task_success_rate"].keys())
tasks_order = tasks[:]  
tasks_plot = ["Average"] + tasks_order

labels = []
vals = [] 
for r in runs:
    labels = CUSTOM_LABELS[:len(runs)]

    per_task = r["per_task_success_rate"]
    task_rates = np.array([per_task.get(t, np.nan) for t in tasks_order], dtype=float) * 100.0
    avg = np.nanmean(task_rates) 

    vals.append(np.concatenate([[avg], task_rates]))

vals = np.vstack(vals)  

n_runs = len(labels)
n_items = len(tasks_plot)

y = np.arange(n_items)
bar_h = 0.8 / n_runs  

plt.figure(figsize=(6.4, max(6, 0.6 * n_items)))


for i, label in enumerate(labels):
    y_i = y + (i - (n_runs - 1) / 2) * bar_h
    plt.barh(y_i, vals[i], height=bar_h, label=label)



plt.yticks(y, tasks_plot)
plt.gca().invert_yaxis()
plt.xlabel("Success Rate (%)")
plt.ylabel("Environment")
plt.xlim(0, 100)
plt.grid(True, axis="x", alpha=0.3)
ax = plt.gca()
pad = 1.0 
ax.set_ylim(n_items - 0.2, -0.2 - pad)
plt.legend(
    loc="upper right",
    bbox_to_anchor=(1, 1), 
    bbox_transform=plt.gca().transAxes,
    frameon=True,
    fontsize=8
)
plt.title(f"{MT_N} Per-Task Success Rates (N={runs[0].get('per_task_episodes', {}).get(tasks_order[0], '?')})")
plt.tight_layout()
plt.savefig(f"./metaworld_logs/{MT_N}_per_task_success.png", dpi=300, bbox_inches="tight")
plt.show()

