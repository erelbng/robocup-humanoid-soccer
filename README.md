# RoboCup Humanoid Soccer RL

End-to-end reinforcement learning pipeline for training humanoid robots (Booster K1) to play soccer, targeting the RoboCup Humanoid Soccer League (HSL).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                            │
│                                                                 │
│  Phase 1: Single-Robot Skills          Phase 2: Match Play      │
│  ┌───────────────────────────┐        ┌──────────────────────┐  │
│  │  Genesis Simulator (GPU)  │        │  Genesis Simulator   │  │
│  │                           │        │                      │  │
│  │  Curriculum Learning:     │───────>│  4v4 Self-Play       │  │
│  │  Stand → Walk → Dribble   │        │  GameController      │  │
│  │  → Shoot → Full           │        │  Tactical Rewards    │  │
│  │                           │        │                      │  │
│  │  200M timesteps           │        │  100M timesteps      │  │
│  └───────────────────────────┘        └──────────────────────┘  │
│                                                │                │
│                                                ▼                │
│                                       ┌──────────────────────┐  │
│                                       │  MuJoCo Evaluation   │  │
│                                       │  Sim2Sim Transfer     │  │
│                                       │  Video Recording      │  │
│                                       │  Metric Logging       │  │
│                                       └──────────────────────┘  │
│                                                │                │
│                                                ▼                │
│                                       ┌──────────────────────┐  │
│                                       │  W&B Dashboard       │  │
│                                       │  Training Curves      │  │
│                                       │  Match Videos         │  │
│                                       │  Reward Breakdown     │  │
│                                       └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
robocup_humanoid_soccer/
├── configs/
│   ├── config.py                  # All configuration dataclasses
│   └── field_hsl_2026.json        # HSL field dimensions (9m × 6m)
├── models/
│   ├── field_generator.py         # JSON → MuJoCo XML field generator
│   ├── field/                     # Generated field models (XML, JSON)
│   └── robot/K1/                  # Booster K1 URDF/MJCF assets
├── envs/
│   ├── rewards.py                 # Multi-objective reward function
│   ├── phase1_dribble_shoot.py    # Single-robot Genesis environment
│   └── phase2_match.py           # Multi-agent match environment
├── gamecontroller/
│   └── sim_game_controller.py     # Python reimplementation of HSL GameController
├── training/
│   └── train.py                   # PPO training loop with W&B logging
├── evaluation/
│   └── evaluate.py                # MuJoCo evaluation & video recording
├── scripts/
│   └── run.sh                     # Setup & orchestration script
├── checkpoints/                   # Saved model weights
├── logs/                          # Training logs
├── videos/                        # Recorded evaluation videos
└── pyproject.toml
```

## Quick Start

### 1. Setup

```bash
cd robocup_humanoid_soccer

# Creates .venv, installs project via pyproject.toml, downloads K1 robot, generates field
chmod +x scripts/run.sh
./scripts/run.sh setup
```

The setup script creates a virtual environment at `.venv/`, installs all dependencies from `pyproject.toml` in editable mode, and prepares the field and robot assets. All subsequent commands automatically activate the venv.

### 2. Phase 1 — Single-Robot Skill Training

Train the K1 robot to stand, walk, dribble, and shoot through curriculum learning:

```bash
# Train with default settings (200M timesteps, curriculum)
./scripts/run.sh train-phase1

# Custom settings
python -m training.train --phase 1 \
    --total-timesteps 50000000 \
    --lr 3e-4 \
    --aggressiveness 0.7 \
    --wandb-project robocup-k1
```

### 3. Phase 2 — Match Training

Fine-tune with 4v4 simulated matches using the GameController:

```bash
# Requires a Phase 1 checkpoint
./scripts/run.sh train-phase2 --checkpoint checkpoints/phase1_best.pt

# Custom match settings
python -m training.train --phase 2 \
    --checkpoint checkpoints/phase1_best.pt \
    --total-timesteps 100000000 \
    --num-agents 4 \
    --wandb-project robocup-k1-match
```

### 4. Evaluation

Evaluate trained policies in MuJoCo (sim2sim transfer validation):

```bash
./scripts/run.sh eval checkpoints/phase2_best.pt --phase phase2

# With video recording
python -m evaluation.evaluate checkpoints/phase2_best.pt \
    --phase phase2 \
    --record-video \
    --num-episodes 10
```

## Key Components

### Reward Function

The reward system is multi-objective with configurable aggressiveness (0.0–1.0):

| Component        | Description                                  | Aggressiveness Effect  |
|-------------------|----------------------------------------------|------------------------|
| `alive`           | Survival bonus                               | —                      |
| `upright`         | Torso orientation penalty                    | —                      |
| `forward_vel`     | Walking speed toward ball                    | ↑ weight at high aggr. |
| `ball_tracking`   | Head orientation toward ball                 | —                      |
| `ball_to_goal`    | Ball proximity to opponent goal              | ↑ weight at high aggr. |
| `kick`            | Contact-based kick detection                 | ↑ weight at high aggr. |
| `dribble`         | Ball control while moving                    | —                      |
| `energy`          | Joint torque penalty                         | —                      |
| `smoothness`      | Action jerk penalty                          | —                      |
| `possession`      | Team ball control (Phase 2)                  | —                      |
| `positioning`     | Tactical field coverage (Phase 2)            | —                      |
| `defensive`       | Defensive engagement (Phase 2)               | ↑ weight at high aggr. |

### Curriculum Stages (Phase 1)

1. **Stand** — balance upright for increasing durations
2. **Walk** — move toward targets with stable gait
3. **Dribble** — maintain ball proximity while walking
4. **Shoot** — kick ball toward goal from various positions
5. **Full** — combined dribble + shoot with full reward

### GameController

Python reimplementation of the RoboCup HSL GameController state machine:

- States: `INITIAL → READY → SET → PLAYING → FINISHED`
- Manages: goals, penalties, set plays, halves, substitutions
- Provides: 24-dimensional state vector for RL observations
- Detects: goals, out-of-bounds, fouls (robot proximity)

### Training Details

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: MLP (512→256→128) with LayerNorm + ELU
- **Observations**: 78-dim (Phase 1) / 156-dim (Phase 2)
- **Actions**: 22-dim continuous (joint position targets)
- **Domain Randomization**: friction, mass, actuator gains, ball restitution, IMU noise

## Robot: Booster K1

- **Height**: 95 cm | **Weight**: ~20 kg
- **DoF**: 22 joints (head×2, arms×4×2, legs×6×2)
- **Source**: [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1)

## Field: RoboCup HSL 2026

- **Dimensions**: 9m × 6m (with 1m border)
- **Goals**: 2.6m wide × 0.8m tall
- **Source**: [HSL-Rules FieldGenerator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/tree/main/figs/FieldGenerator)

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Genesis Simulator 0.4+ (GPU required for training)
- MuJoCo 3.0+ (evaluation)
- Weights & Biases (logging)

## References

- [RoboCup HSL Rules](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules)
- [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Booster Robotics K1](https://github.com/BoosterRobotics/booster_assets)

## License

Research use only. Robot assets are subject to Booster Robotics' license terms.
