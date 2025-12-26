from  training_setup_multitask.utilities.MetaworldTasks import MT3_TASKS, MT10_TASKS


class CurriculumConfig:
    """Configuration for Curriculum Learning"""

    # Task difficulties based on metaworld paper
    TASK_DIFFICULTY = {
        "reach-v3": 1,                  # Einfach: nur Endposition erreichen
        "push-v3": 2,                   # Mittel: Objekt bewegen
        "button-press-topdown-v3": 2,   # Mittel: Präzise Positionierung
        "door-open-v3": 3,              # Mittel-Schwer: Türgriff + Bewegung
        "drawer-open-v3": 3,            # Mittel-Schwer: Ziehen erforderlich
        "drawer-close-v3": 3,           # Mittel-Schwer: Umgekehrt zu drawer-open
        "window-open-v3": 4,            # Schwer: Komplexe Manipulation
        "window-close-v3": 4,           # Schwer: Komplexe Manipulation
        "pick-place-v3": 4,             # Schwer: Greifen + Platzieren
        "peg-insert-side-v3": 5,        # Sehr Schwer: Präzises Alignment
    }

    # Task clusters for curriculum stages
    CURRICULUM_STAGES = [
        # Stage 1: Basics
        ["reach-v3"],

        # Stage 2: Simple manipulation
        ["reach-v3", "push-v3", "button-press-topdown-v3"],

        # Stage 3: Opening/Closing
        ["reach-v3", "push-v3", "button-press-topdown-v3",
         "door-open-v3", "drawer-open-v3", "drawer-close-v3"],

        # Stage 4: Complex manipulation
        ["reach-v3", "push-v3", "button-press-topdown-v3",
         "door-open-v3", "drawer-open-v3", "drawer-close-v3",
         "window-open-v3", "window-close-v3"],

        # Stage 5: All tasks
        MT10_TASKS,
    ]

    # Performance thresholds for stage-transitions
    STAGE_THRESHOLDS = {
        1: 0.7,
        2: 0.6,
        3: 0.5,
        4: 0.4
    }
