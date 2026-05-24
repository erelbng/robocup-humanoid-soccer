# Gemini CLI Project Instructions

## Overview
This project focuses on reinforcement learning for RoboCup Humanoid Soccer League (HSL) using the Booster K1 robot. It utilizes a two-phase training approach with different simulators for training (Genesis) and evaluation/video (MuJoCo).

## Workflow & Conventions
- **Simulator Strategy:** 
    - **Training:** Genesis (must be used for all training/curriculum tasks).
    - **Evaluation/Video:** MuJoCo 3.x (must be used for simulation-to-simulation transfer validation and video generation).
- **Configuration:** All hyperparameters and settings are in `configs/config.py` using Python dataclasses.
- **Project Structure:** Adhere to the established structure defined in `CLAUDE.md`.
- **GameController:** The Python reimplementation in `gamecontroller/sim_game_controller.py` must be used for all HSL-related state management in Phase 2.
- **W&B Logging:** All training runs must be logged to `robocup-humanoid-soccer`.
- **Dependencies:** Install via `./scripts/run.sh setup` or `pip install -e .[dev]`. Use the project-managed virtual environment.

## Task Lifecycle
1. **Research:** Analyze existing environment code (`envs/`) and rewards (`envs/rewards.py`) before attempting modifications.
2. **Strategy:** Formulate plans referencing the existing curriculum or reward structures.
3. **Execution:** Apply targeted changes, ensuring proper testing (e.g., verifying reward logic, checking field generation).
4. **Validation:** Run evaluation with `evaluation/evaluate.py` using the appropriate simulator, and verify against benchmarks or existing checkpoints.

## Prototyping/New Features
- Use `enter_plan_mode` for significant architecture or new system designs.
- Follow the design constraints for prototypes as specified in the global system prompt.
- Prioritize Vanilla CSS and standard web/Python stacks.

## Communication
- Keep responses concise, high-signal, and professional.
- Utilize the Topic Model (`update_topic`) for multi-step tasks.
- Avoid repetition; provide summaries only when transitioning chapters/topics.
