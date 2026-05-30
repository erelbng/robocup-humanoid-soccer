# RoboCup Humanoid Soccer RL
hallo
End-to-end reinforcement learning pipeline for training humanoid robots (Booster K1) to play soccer, targeting the RoboCup Humanoid Soccer League (HSL).

The codebase is built around a **skill library**: instead of a monolithic policy that learns everything at once, four specialised policies (`standup`, `walk`, `dribble`, `shoot`) are trained independently, then a Phase-2 **orchestrator** selects between them in real time during 4v4 matches.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase 1 — Skill library                                         │
│                                                                  │
│   train_skill.py --skill walk    ─┐                              │
│   train_skill.py --skill standup  │   each in its own process    │
│   train_skill.py --skill dribble  │   (Genesis allocation        │
│   train_skill.py --skill shoot   ─┘    bounded per skill)        │
│                                                                  │
│   Each skill: 78-dim base obs + continuous command vector        │
│   Algorithm: PPO (default) or FlashSAC                           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (frozen skill checkpoints)
┌──────────────────────────────────────────────────────────────────┐
│  Phase 2 — Discrete-skill orchestrator                           │
│                                                                  │
│   train_orchestrator.py                                          │
│                                                                  │
│   Each control step (10 Hz):                                     │
│     orchestrator → (skill_idx, command_vec_7d) per agent         │
│     frozen skill → joint targets (50 Hz inner)                   │
│                                                                  │
│   4v4 self-play; opponent pool of past snapshots                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Evaluation — MuJoCo (sim2sim) + W&B / TensorBoard logging       │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
robocup_humanoid_soccer/
├── configs/                       # ProjectConfig, K1RobotConfig, field JSON
├── models/
│   ├── field_generator.py         # JSON → MuJoCo XML + Genesis builder
│   ├── field/                     # Auto-generated; do not hand-edit
│   └── robot/K1/                  # Booster K1 URDF/MJCF + meshes
├── envs/
│   └── standup.py                 # Fallen-pose definitions (reused by
│                                  # skills/standup)
├── skills/                        # Phase 1 skill library
│   ├── base.py                    # SkillEnv ABC, CommandSpec, SkillSpec
│   ├── common_obs.py              # 78-dim shared base obs builder
│   ├── walk/                      # K1WalkEnv  — 5-dim cmd (vx,vy,vyaw,fh,sf)
│   ├── standup/                   # K1StandupEnv — no command, fallen-pose spawn
│   ├── dribble/                   # K1DribbleEnv — walk+ball_offset (7-dim cmd)
│   └── shoot/                     # K1ShootEnv — [aim_angle, power, foot_pref]
├── orchestrator/                  # Phase 2 — discrete-skill controller
│   ├── policy.py                  # Hybrid Categorical+Normal actor-critic
│   ├── skill_router.py            # Frozen-skill loader + batched routing
│   ├── env.py                     # K1MatchEnv (4v4 self-play scene)
│   ├── self_play.py               # Opponent pool, team-action splitting
│   └── rewards.py                 # Team-aware match reward
├── training/
│   ├── train_skill.py             # Single-skill PPO/FlashSAC entry point
│   ├── train_orchestrator.py      # 4v4 self-play PPO entry point
│   ├── common.py                  # Shared logger/checkpoint/device helpers
│   └── algorithms/
│       ├── ppo.py                 # train_ppo_vec
│       ├── flashsac.py            # train_flashsac_vec (off-policy SAC + twin Q)
│       └── networks.py            # PPOActorCritic, SACActor, TwinQNetwork
├── gamecontroller/                # HSL GameController state machine (Python)
├── evaluation/
│   └── evaluate.py                # MuJoCo eval, sim2sim, video
├── scripts/run.sh                 # Setup + train-skill / train-orchestrator
└── checkpoints/                   # skill_<name>/...  and  orchestrator/...
```

## Quick Start

### 1. Setup

```bash
cd robocup_humanoid_soccer
chmod +x scripts/run.sh
./scripts/run.sh setup
```

Creates `.venv/`, installs everything from `pyproject.toml` (editable mode), downloads the K1 robot assets, and generates the field models. All subsequent commands auto-activate the venv.

### 2. Train the skills (Phase 1)

Each skill lives in its own process — Genesis re-allocates its GPU memory per skill, so you never hit the cross-stage OOM that a monolithic curriculum runs into.

```bash
# One skill at a time
./scripts/run.sh train-skill walk    --device gpu --vec-num-envs 1024 --wandb
./scripts/run.sh train-skill standup --device gpu
./scripts/run.sh train-skill dribble --device gpu \
    --init-from checkpoints/skill_walk/skill_walk_step50000000.pt
./scripts/run.sh train-skill shoot   --device gpu

# Or train every skill sequentially
./scripts/run.sh train-skills-all --device gpu --vec-num-envs 1024
```

**Algorithm choice** — PPO (on-policy, default) or FlashSAC (off-policy SAC with twin Q + GPU replay buffer + automatic temperature):

```bash
./scripts/run.sh train-skill walk --algorithm flashsac --device gpu
```

### 3. Train the orchestrator (Phase 2)

Once you have a checkpoint for each skill under `checkpoints/skill_<name>/`, the orchestrator picks them up automatically (latest by mtime) and trains via 4v4 self-play:

```bash
./scripts/run.sh train-orchestrator --num-envs 256 --wandb
```

The orchestrator runs PPO on team-0 transitions only; team 1 is controlled by a snapshot from the opponent pool (cold-starts with mirror-self play).

### 4. Evaluate

Evaluate any checkpoint in MuJoCo (sim2sim transfer validation, video recording):

```bash
./scripts/run.sh eval checkpoints/orchestrator/orchestrator_step100000000.pt --phase phase2

python -m evaluation.evaluate checkpoints/orchestrator/orchestrator_best.pt \
    --phase phase2 --record-video --num-episodes 10
```

## Key Concepts

### Skill design

| Skill | Command vec | Obs add-ons | obs_dim |
|-------|-------------|-------------|---------|
| **standup** | none | contact (foot/hand z + bool, 8) | 86 |
| **walk** | `[vx, vy, vyaw, foot_clearance, step_freq]` | none | 83 |
| **dribble** | walk + `[ball_off_x, ball_off_y]` | ball pos/vel in body frame | 91 |
| **shoot** | `[aim_angle, power, foot_pref]` | ball pos/vel + target pos in body frame | 90 |

All skills share a 78-dim base observation: root height, projected gravity (orientation proxy), body-frame linear & angular velocity, joint positions (relative to default pose), joint velocities, last action, and a gait clock signal.

### Orchestrator design

- **Obs (156-dim per agent):** shared base (78) + ball (6) + 3 teammates (18) + 4 opponents (24) + GameController state (24) + role one-hot GK/DEF/MID/ATK (4) + score_diff/time_remaining (2).
- **Action (8-dim):** discrete skill index ∈ `(standup, walk, dribble, shoot)` + 7-dim continuous command (per-skill prefix-sliced).
- **Timing:** 10 Hz orchestrator decisions, 50 Hz inner skill control (decisions latched across 5 physics steps).
- **Self-play:** opponent pool of past snapshots (default capacity 10), `latest_prob=0.5` samples the freshest snapshot, the rest uniform over older entries.

### Algorithms

| | PPO | FlashSAC |
|--|------|----------|
| On/off policy | On-policy | Off-policy |
| Networks | Single Gaussian actor + critic | Squashed-Gaussian actor + twin Q + targets |
| Best for | Walk, standup (dense rewards) | Shoot, dribble (sparser rewards) |
| `--resume` / `--init-from` | ✓ | ✗ (different network architecture) |
| Used by orchestrator | ✓ | — |

**Standup** additionally uses three mechanisms from the 2025 humanoid get-up literature (HoST / HumanUP), all standup-only and on by default: a decaying **assist-force curriculum** (upward trunk support that weans to zero), a **two-stage reward** (`discovery` → `deploy`), and **multi-critic PPO** (one value head per reward group). See `skills/standup/README.md`.

Both share `MLP 512→256→128 + LayerNorm + ELU`. Hyperparameter device presets (`training/common.DEVICE_PRESETS`) auto-pick reasonable defaults: GPU → `vec_num_envs=1024`, FlashSAC `buffer_capacity=2M / batch_size=2048`.

### GameController

`gamecontroller/sim_game_controller.py` — Python reimplementation of the HSL GameController state machine. States `INITIAL → READY → SET → PLAYING → FINISHED`, manages goals / penalties / set plays / halves, exposed via `get_state_vector()` (padded into a 24-dim obs slot). Goal attribution in the orchestrator uses score-delta from the previous step (the GC's `goal_just_scored` is a pulse without team info).

## Robot: Booster K1

- **Height**: 95 cm — **Weight**: ~20 kg
- **DoF**: 22 joints (head ×2, arms 8, legs 12)
- **Control**: 50 Hz policy, 500 Hz physics, PD gains kp=50 / kd=5
- **Source**: [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1)

## Field: RoboCup HSL 2026

- **Dimensions**: 9 m × 6 m playing area (1 m border)
- **Goals**: 2.6 m wide × 0.8 m tall, centred at `(±4.5, 0, 0.4)`
- **Source**: [HSL-Rules FieldGenerator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/tree/main/figs/FieldGenerator)

## Logging

TensorBoard is always on (`logs/<run_name>/`); pass `--wandb` to also stream to Weights & Biases (project `robocup-humanoid-soccer`). Per-iteration metrics include all reward components, PPO/SAC diagnostics (KL, clip fraction, explained variance, entropy, learning rate), and rollout videos every 50 iterations.

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Genesis Simulator 0.4+ (GPU required for training)
- MuJoCo 3.0+ (evaluation)
- Weights & Biases (logging, optional)

Install everything via `./scripts/run.sh setup` (or `pip install -e .[dev]` if you'd rather manage assets manually).

## References

- [RoboCup HSL Rules](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules)
- [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Booster Robotics K1](https://github.com/BoosterRobotics/booster_assets)

## License

Research use only. Robot assets are subject to Booster Robotics' license terms.
