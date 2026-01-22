# Meta-World Multi-Task Training Framework

During the development of the Meta-World Multi-Task Training Framework, two different variants were developed in parallel. The first variant is denoted as **v1** and implemented in `train_mt_transferLearning.py`, while the second variant is denoted as **v2** and implemented in `train_mt_multihead.py`.

If a section header is labeled with **v1**, the described content refers exclusively to the first variant. Likewise, headers labeled with **v2** refer exclusively to the second variant. Sections marked with **v1 & v2** indicate that the described concepts or implementations apply to both variants.


## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Modes](#training-modes)
- [Configuration Options](#configuration-options)
- [Training Strategies](#training-strategies)
- [Performance Optimization](#performance-optimization)
- [Advanced Features](#advanced-features)
- [Examples](#examples)

---

## Overview

Training framework **v1** for Meta-World robotic manipulation tasks. It supports:

- **Multiple Training Modes**: MT1 (single task), MT3 (3 tasks), MT10 (all 10 tasks)
- **Curriculum Learning**: Progressive task introduction with automatic stage transitions
- **Transfer Learning**: Fine-tune pretrained models for new tasks
- **Parallel Training**: Multi-process environments for faster training
- **Multiple Algorithms**: SAC, TD3, DDPG

Training framework **v2** for Meta-World robotic manipulation tasks. It supports:

- **Multiple Training Modes**: MT1 (single task), MT3 (3 tasks), MT10 (all 10 tasks)
- **Curriculum Learning**: Progressive task introduction with automatic stage transitions
- **Multihead Policy**: Individual Q-Funciton for each task separately
- **Parallel Training**: Multi-process environments for faster training
- **One Algorithm**: SAC


### Supported Tasks (MT10) (**v1 & v2**)

1. `reach-v3` - Simple reaching task
2. `push-v3` - Push object to target
3. `pick-place-v3` - Pick and place object
4. `door-open-v3` - Open a door
5. `drawer-open-v3` - Open a drawer
6. `drawer-close-v3` - Close a drawer
7. `button-press-topdown-v3` - Press a button
8. `peg-insert-side-v3` - Insert peg into hole
9. `window-open-v3` - Open a window
10. `window-close-v3` - Close a window

---

## Installation (**v1 & v2**)

```bash
# Clone the repository
git clone https://github.com/sirknival/robot_learning_project.git

# Install dependencies
pip install stable-baselines3[extra]
pip install gymnasium
pip install metaworld
pip install tensorboard

#run code
python3 train_metaworld_sb3.py

```

---

## Quick Start

### 1. Single Task Training (MT1)

Train on a single task with default settings:

```python
# In main script
EXPERIMENT = "MT1"
TASK_NAME = "reach-v3"
ALGORITHM = "SAC"
N_PARALLEL_ENVS = 1
```

```bash
python train_metaworld_sb3.py
```

### 2. Multi-Task Training (MT10)

Train on all 10 tasks simultaneously:

```python
EXPERIMENT = "MT10"
ALGORITHM = "SAC"
```

### 3. Curriculum Learning

Train with progressive task introduction:

```python
EXPERIMENT = "MT10_CURRICULUM"
TRAINING_MODE = "PROGRESSIVE"
CURRICULUM_STAGE = 0  # Start from easiest tasks
USE_CURRICULUM = True
```

---

## Training Modes

### MT1 - Single Task Training

**Use Case**: Focus on mastering one specific task.

**Configuration**:
```python
EXPERIMENT = "MT1"
TASK_NAME = "push-v3"  # Any MT10 task
N_PARALLEL_ENVS = 1    # Number of parallel environments
```

**Designed For**:
- Initial experimentation
- Task-specific optimization
- Transfer learning source models

---

### MT3 - Three Task Training

**Use Case**: Train on a subset of related tasks, defined in task description.

**Configuration**:
```python
EXPERIMENT = "MT3"
# Default tasks: ["reach-v3", "push-v3", "pick-place-v3"]
# Tasks are selected automatically
```

---

### MT10 - Full Multi-Task Training

**Use Case**: Train a general-purpose policy across all tasks.

**Configuration**:
```python
EXPERIMENT = "MT10"
# Standard MetaWorld MT10-task set
# Tasks are selected automatically
```

---

## Configuration Options

### Basic Settings

```python
# -------------------- Experiment Setup --------------------
EXPERIMENT = "MT1"              # "MT1", "MT3", "MT10", "MT10_CURRICULUM"
TASK_NAME = "reach-v3"          # Required for MT1
ALGORITHM = "SAC"               # "SAC", "TD3", "DDPG"
SEED = 42                       # Random seed
N_PARALLEL_ENVS = 1             # Parallel environments (MT1 only)
```

### Training Parameters

```python
# -------------------- Training Phases --------------------
CONTINUE_TRAINING = False         # Resume from checkpoint
SEL_TRAIN_PHASE = 1               # Training phase (1, 2, or 3)

TRAIN_PHASES = {
    1: {"start": 0, "end": 5},    # 0-5M steps
    2: {"start": 5, "end": 10},   # 5-10M steps
    3: {"start": 10, "end": 20},  # 10-20M steps
}

# -------------------- Environment Settings --------------------
MAX_EPISODE_STEPS = 500          # Steps per episode
NORMALIZE_REWARD = False         # Reward normalization
```

### Evaluation & Checkpointing

```python
# -------------------- Evaluation & Checkpointing --------------------
EVAL_FREQ = 10000              # Evaluate every N steps
N_EVAL_EPISODES = 20           # Episodes per evaluation
CHECKPOINT_FREQ = 50000        # Save checkpoint every N steps
```

### Performance Settings

```python
# -------------------- Performance --------------------
USE_SUBPROC_VEC_ENV = True     # Use multi-process vectorization
N_PARALLEL_ENVS = 8            # Number of parallel environments
```

---

## Training Strategies

### 1. Sequential Training (Curriculum)

**Strategy**: Train tasks one by one, ordered by difficulty.

**Configuration**:
```python
EXPERIMENT = "MT10_CURRICULUM"
TRAINING_MODE = "SEQUENTIAL"
CURRICULUM_STAGE = 0  # Current task index
USE_CURRICULUM = True
```

**Task Order** (easiest to hardest):

| #  | Environment                 | Difficulty |
|----|-----------------------------|------------|
| 1  | reach-v3                    | 1          |
| 2  | push-v3                     | 2          |
| 3  | button-press-topdown-v3     | 2          |
| 4  | door-open-v3                | 3          |
| 5  | drawer-open-v3              | 3          |
| 6  | drawer-close-v3             | 3          |
| 7  | window-open-v3              | 4          |
| 8  | window-close-v3             | 4          |
| 9  | pick-place-v3               | 4          |
| 10 | peg-insert-side-v3          | 5          |


**Workflow**:
```bash
# Stage 0: Train on reach-v3
CURRICULUM_STAGE = 0
python train_metaworld.py

# Stage 1: Transfer to push-v3
CURRICULUM_STAGE = 1
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/transfer_checkpoints/transfer_stage0.zip"
python train_metaworld.py

# Continue for all stages...
```

---

### 2. Progressive Training (Curriculum)

**Strategy**: Gradually add more tasks to the training set.

**Configuration**:
```python
EXPERIMENT = "MT10_CURRICULUM"
TRAINING_MODE = "PROGRESSIVE"
CURRICULUM_STAGE = 0  # Current stage
USE_CURRICULUM = True
```

**Curriculum Stages**:

| Stage | Tasks | Purpose |
|-------|-------|---------|
| 0 | `reach-v3` | Learn basic reaching |
| 1 | `reach-v3`, `push-v3`, `button-press-topdown-v3` | Simple manipulation |
| 2 | Stage 1 + `door-open-v3`, `drawer-open-v3`, `drawer-close-v3` | Opening/closing |
| 3 | Stage 2 + `window-open-v3`, `window-close-v3` | Complex manipulation |
| 4 | All MT10 tasks | Full multi-task |

**Automatic Stage Transition**:
```python
# Performance thresholds for automatic progression
STAGE_THRESHOLDS = {
    1: 0.7,  # Need 70% success rate to move to stage 2
    2: 0.6,  # Need 60% success rate to move to stage 3
    3: 0.5,  # Need 50% success rate to move to stage 4
    4: 0.4,  # Need 40% success rate to complete
}

MIN_STEPS_PER_STAGE = 200000  # Minimum steps before transition
STAGE_EVAL_FREQ = 10000       # Check performance every N steps
```

**Workflow**:
```bash
# Stage 0: Start with basics
CURRICULUM_STAGE = 0
python train_metaworld.py

# Monitor logs for automatic stage transition
# Or manually advance when ready:
CURRICULUM_STAGE = 1
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/transfer_checkpoints/transfer_stage0.zip"
python train_metaworld.py
```

---

### 3. Mixed Training (Curriculum)

**Strategy**: Start with easy tasks, gradually add harder ones.

**Configuration**:
```python
EXPERIMENT = "MT10_CURRICULUM"
TRAINING_MODE = "MIXED"
CURRICULUM_STAGE = 0
```

**Stage Progression**:
- Stage 0: First 3 tasks (easiest)
- Stage 1: First 4 tasks
- Stage 2: First 5 tasks
- ...
- Stage 7: All 10 tasks


---

### 4. Transfer Learning

**Strategy**: Use pretrained models to speed up learning on new tasks.

**Configuration**:
```python
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/MT1_SAC_5M.zip"
```

**Fine-Tuning Parameters**:
```python
# In TransferLearningManager
learning_rate_multiplier = 0.3  # Reduce LR to 30% of original
```

**Workflow**:

**Step 1**: Train base model
```python
EXPERIMENT = "MT1"
TASK_NAME = "reach-v3"
USE_TRANSFER_LEARNING = False
# Train for 5M steps
```

**Step 2**: Transfer to related task
```python
EXPERIMENT = "MT1"
TASK_NAME = "push-v3"  # Related task
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/MT1_SAC_reach-v3_5M.zip"
# Train for 2M steps (converges faster!)
```

**Best Practices**:
- Transfer from easier to harder tasks
- Use lower learning rate (multiplier: 0.1-0.3)
- Train for fewer steps than from scratch
- Monitor for negative transfer

**Task Relationships** (good transfer pairs):
- reach-v3 → push-v3 (both require reaching)
- door-open-v3 → drawer-open-v3 (similar mechanics)
- window-open-v3 ↔ window-close-v3 (inverse tasks)
- drawer-open-v3 ↔ drawer-close-v3 (inverse tasks)

---

## Performance Optimization

### Parallel Training (MT1 Only)

**Single Environment** (Baseline):
```python
N_PARALLEL_ENVS = 1
USE_SUBPROC_VEC_ENV = False
```

**Multi-Environment (DummyVecEnv)**:
```python
N_PARALLEL_ENVS = 8
USE_SUBPROC_VEC_ENV = False
```

**Multi-Environment (SubprocVecEnv)**:
```python
N_PARALLEL_ENVS = 8
USE_SUBPROC_VEC_ENV = True
```

### Recommendations

**For Fast Experimentation**:
```python
EXPERIMENT = "MT1"
N_PARALLEL_ENVS = 8
USE_SUBPROC_VEC_ENV = True
SEL_TRAIN_PHASE = 1  # Only 5M steps
```

**For Final Training**:
```python
EXPERIMENT = "MT10"
N_PARALLEL_ENVS = 1  # Already vectorized internally
SEL_TRAIN_PHASE = 3  # Full 20M steps
```

**For Limited Resources**:
```python
N_PARALLEL_ENVS = 4
USE_SUBPROC_VEC_ENV = True
```

### Optimal CPU Configuration

```python
import os

# Use 75% of available cores
N_PARALLEL_ENVS = max(1, int(os.cpu_count() * 0.75))
```

---

## Advanced Features

### Custom Curriculum Stages

Edit `CurriculumConfig` to define custom stages:

```python
# In CurriculumConfig class
CUSTOM_CURRICULUM_STAGES = [
    ["reach-v3"],
    ["reach-v3", "push-v3"],
    ["reach-v3", "push-v3", "door-open-v3"],
    # ... your custom progression
]
```

### Custom Task Difficulties

Adjust task difficulty ratings:

```python
# In CurriculumConfig class
TASK_DIFFICULTY = {
    "reach-v3": 1,
    "push-v3": 2,
    "your-custom-task-v3": 3,
    # ...
}
```

### Replay Buffer Management

```python
# Save replay buffer
model.save_replay_buffer("./buffer.pkl")

# Load replay buffer (when continuing training)
CONTINUE_TRAINING = True
# Buffer is automatically loaded from paths_dict
```

### Custom Evaluation

```python
# Modify evaluation frequency
EVAL_FREQ = 5000  # More frequent evaluation

# Modify number of evaluation episodes
N_EVAL_EPISODES = 50  # More robust evaluation
```

---

## Examples

### Example 1: Quick MT1 Training

**Goal**: Train a reaching agent as fast as possible.

```python
EXPERIMENT = "MT1"
TASK_NAME = "reach-v3"
ALGORITHM = "SAC"
N_PARALLEL_ENVS = 8
USE_SUBPROC_VEC_ENV = True
SEL_TRAIN_PHASE = 1  # 5M steps
CHECKPOINT_FREQ = 25000
```

**Expected Result**: ~95% success rate in ~20 minutes.

---

### Example 2: Full Curriculum Training

**Goal**: Train a general policy using curriculum learning.

```bash
# Stage 0: Basic reaching
EXPERIMENT = "MT10_CURRICULUM"
TRAINING_MODE = "PROGRESSIVE"
CURRICULUM_STAGE = 0
USE_TRANSFER_LEARNING = False
python train_metaworld.py

# Stage 1: Simple manipulation
CURRICULUM_STAGE = 1
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/transfer_checkpoints/transfer_stage0.zip"
python train_metaworld.py

# ... continue through all stages
```

**Expected Result**: Better final performance on all tasks compared to direct MT10 training.

---

### Example 3: Transfer Learning Experiment

**Goal**: Compare training with and without transfer learning.

**Baseline** (No Transfer):
```python
EXPERIMENT = "MT1"
TASK_NAME = "push-v3"
USE_TRANSFER_LEARNING = False
SEL_TRAIN_PHASE = 1  # 5M steps
```

**With Transfer**:
```python
EXPERIMENT = "MT1"
TASK_NAME = "push-v3"
USE_TRANSFER_LEARNING = True
PRETRAINED_MODEL_PATH = "./metaworld_models/MT1_SAC_reach-v3_5M.zip"
SEL_TRAIN_PHASE = 1  # Same 5M steps
```

**Compare**: Check evaluation logs to see if transfer learning achieves higher success rate.

---

### Example 4: Benchmarking MT10

**Goal**: Train and benchmark on all MT10 tasks.

```python
EXPERIMENT = "MT10"
ALGORITHM = "SAC"
SEL_TRAIN_PHASE = 3  # 20M steps
EVAL_FREQ = 10000
N_EVAL_EPISODES = 50  # Robust evaluation
DEBUG = True
```

**Monitor**: Check `./metaworld_logs/` for TensorBoard logs.

```bash
tensorboard --logdir=./metaworld_logs/
```

---

### Example 5: Resume Training

**Goal**: Continue training from a checkpoint.

```python
CONTINUE_TRAINING = True
SEL_TRAIN_PHASE = 2  # Phase 2: 5M → 10M steps

# Model and buffer automatically loaded from:
# ./metaworld_models/MT10_SAC_5M.zip
# ./metaworld_models/MT10_SAC_5M_replay.pkl
```

---


## File Structure

```
meta-world-training/
├── train_metaworld.py                    # Main training script
├── training_setup_multitask/
│   ├── utilities/
│   │   ├── MetaWorldEnvFactory.py       # Environment creation
│   │   ├── DebugPrinter.py              # Logging utilities
│   │   ├── CurriculumConfig.py          # Curriculum configuration
│   │   ├── TransferLearningManager.py   # Transfer learning
│   │   ├── MetaworldTasks.py            # Task definitions
│   │   ├── TaskEvaluator.py             # Model evalutaion
│   │   └── algorithms.py                # Model factories
│   ├── WrapperClasses/
│   │   ├── OneHotTaskWrapper.py         # Task encoding wrapper
│   │   └── GymnasiumVecEnvAdapter.py    # Gymnasium adapter
│   └── Callbacks/
│       ├── ReplayBufferCheckpointCallback.py   # Legacy Code
│       └── ProgressiveTaskCallback.py   # Curriculum callback
├── metaworld_models/                    # Saved models
│   ├── checkpoints_*/                   # Training checkpoints
│   ├── best_*/                          # Best models
│   └── transfer_checkpoints/            # Transfer learning checkpoints
├── evaluation_results/                  # Stores final assements
└── metaworld_logs/                      # TensorBoard logs
```

---



