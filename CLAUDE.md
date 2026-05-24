# RoboCup Humanoid Soccer RL — Project Guide

## Vision

Train Booster K1 humanoid robots to play soccer in the RoboCup Humanoid Soccer League (HSL) using deep reinforcement learning. The pipeline has two phases:

1. **Phase 1** — Single robot learns to stand, walk, dribble, and shoot through curriculum learning (Genesis simulator, GPU-accelerated)
2. **Phase 2** — Fine-tune with 4v4 self-play matches under RoboCup HSL GameController rules (Genesis → evaluate in MuJoCo)

## Simulator Strategy

| Task | Simulator | Reason |
|------|-----------|--------|
| Training (Phase 1 & 2) | Genesis (`genesis-world`) | Massively parallel GPU envs (4096 Phase 1, 256 Phase 2) |
| Evaluation / video | MuJoCo 3.x | Sim2sim transfer validation; physics fidelity for reporting |

**Never swap these.** Genesis is fast but lacks MuJoCo's mature rendering; MuJoCo is used only for eval and video.

## External Assets (do not re-download unless missing)

- **Booster K1 robot** — `models/robot/K1/` (from [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1))
  - `K1_22dof.xml` — MJCF (MuJoCo), primary eval format
  - `K1_22dof.urdf` — URDF, used by Genesis
  - 22 DOF: head×2, arms×8, legs×12
- **Field spec** — `configs/field_hsl_2026.json` (from [HSL-Rules FieldGenerator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/main/figs/FieldGenerator/generateField.py))
  - 9 m × 6 m, 1 m border, 2.6 m × 0.8 m goals
- **GameController reference** — [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController/tree/master) — Python reimplementation in `gamecontroller/sim_game_controller.py`

## Project Structure

```
robocup_humanoid_soccer/
├── configs/
│   ├── config.py                  # All dataclass configs (robot, Phase1/2, eval, W&B)
│   ├── field_hsl_2026.json        # HSL field dimensions
│   └── checkpoints/               # Eval result JSON files
├── models/
│   ├── field_generator.py         # JSON → MuJoCo XML field builder (+ Genesis builder)
│   ├── field/
│   │   ├── field_robocup.xml      # Generated MuJoCo field (do not hand-edit)
│   │   ├── match_scene.xml        # Multi-robot match scene
│   │   ├── field_genesis_builder.py  # Auto-generated Genesis scene helper
│   │   └── field_info.json        # Derived field metadata
│   └── robot/K1/                  # Booster K1 URDF/MJCF + meshes
├── envs/
│   ├── rewards.py                 # SoccerRewardFunction (12+ components, aggressiveness)
│   ├── phase1_dribble_shoot.py    # K1DribbleShootEnv — Genesis single-robot env
│   └── phase2_match.py           # K1SoccerMatchEnv — 4v4 multi-agent Genesis env
├── gamecontroller/
│   └── sim_game_controller.py     # HSL GameController state machine (Python)
├── training/
│   └── train.py                   # PPO loop, curriculum, W&B logging, checkpoint save/load
├── evaluation/
│   └── evaluate.py                # MuJoCo eval, sim2sim, video recording, metric logging
├── scripts/
│   └── run.sh                     # Setup + train/eval entry point
├── checkpoints/                   # Saved PyTorch model weights (phase1_<stage>_step*.pt)
├── logs/                          # Training stdout/tensorboard logs
└── videos/                        # MuJoCo eval recordings
```

## Common Commands

```bash
# First-time setup (creates .venv, installs deps, downloads K1, generates field)
./scripts/run.sh setup

# Phase 1 training — curriculum: stand → walk → dribble → shoot → full
./scripts/run.sh train-phase1
# or
python -m training.train --phase 1 --total-timesteps 200000000 --aggressiveness 0.7

# Phase 2 training — 4v4 self-play match fine-tuning
./scripts/run.sh train-phase2 --checkpoint checkpoints/phase1_best.pt
# or
python -m training.train --phase 2 --checkpoint checkpoints/phase1_best.pt

# Evaluate in MuJoCo with video recording
python -m evaluation.evaluate checkpoints/phase2_best.pt --phase phase2 --record-video --num-episodes 10

# Regenerate field model from JSON spec
python -m models.field_generator
```

## Key Architecture Decisions

### PPO Training (both phases)
- **Network**: MLP 512→256→128, LayerNorm, ELU activations
- **Phase 1**: 78-dim obs, 22-dim action (joint position targets), 4096 parallel envs
- **Phase 2**: 156-dim obs per agent (self + teammates + opponents + ball + field), 256 envs
- **Control**: 50 Hz policy (dt=0.02s), 500 Hz physics (sim_dt=0.002s), action_repeat=10
- **PD gains**: kp=50, kd=5 for all joints

### Curriculum (Phase 1, 5 stages)
1. **stand** — balance upright for increasing durations
2. **walk** — move toward targets with stable gait
3. **dribble** — maintain ball proximity while walking
4. **shoot** — kick ball toward goal from varied positions
5. **full** — combined dribble + shoot with full reward

Stage promotion is based on mean episode reward thresholds.

### Reward System (`envs/rewards.py`)
Configurable via `RewardWeights` in `configs/config.py`. Aggressiveness (0.0–1.0) scales:
- `forward_velocity` ×(1 + 0.5×a), `kick_reward` ×(1+a), `ball_to_goal` ×(1+0.5×a)
- `defensive_coverage` ×(1−0.5×a)

Phase 2 adds: `team_ball_possession`, `goal_scored` (+50), `goal_conceded` (−50), `positioning`, `passing`, `defensive_coverage`, `offsides_penalty`.

### GameController (`gamecontroller/sim_game_controller.py`)
Python reimplementation of the HSL GameController state machine:
- States: `INITIAL → READY → SET → PLAYING → FINISHED`
- Manages goals, penalties, set plays, halves (5 min each in sim time)
- Provides 24-dim state vector injected into Phase 2 observations
- Detects goals, out-of-bounds, fouls (robot-proximity checks)

### Domain Randomization
Applied during training to improve sim2real transfer:
- Friction: [0.6, 1.2]
- Robot mass, actuator gains, ball restitution
- IMU noise injection

### W&B Logging
Project: `robocup-humanoid-soccer`. Logs per iteration:
- All reward component breakdowns
- Episode stats (survival, ball speed, goal rate)
- Videos every 50 iterations
- Model checkpoints every 100 iterations

## Configuration

All config lives in `configs/config.py` as dataclasses. Key tunables:

```python
Phase1Config.num_envs = 4096          # reduce if GPU OOM
Phase1Config.total_timesteps = 200M
Phase2Config.players_per_team = 4     # 4v4
Phase2Config.half_duration = 300.0    # 5 min sim time per half
RewardWeights.aggressiveness = 0.0    # 0.0 = balanced, 1.0 = aggressive
```

## Checkpoints

Checkpoints are saved as `checkpoints/phase1_{stage}_step{N}.pt`. Phase 2 loads a Phase 1 checkpoint for initialization. Best models are symlinked as `phase1_best.pt` / `phase2_best.pt`.

## Dependencies

```
Python 3.10+
torch >= 2.0
genesis-world >= 0.4  (GPU training — CUDA required)
mujoco >= 3.0         (evaluation + video)
wandb >= 0.16
imageio + imageio-ffmpeg  (video)
```

Install via `./scripts/run.sh setup` or `pip install -e .[dev]`.

## References

- [HSL Rules & Field Generator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/main/figs/FieldGenerator/generateField.py)
- [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController/tree/master)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Booster K1 Assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1)
