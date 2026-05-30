# RoboCup Humanoid Soccer RL — Project Guide

## Vision

Train Booster K1 humanoid robots to play soccer in the RoboCup Humanoid Soccer League (HSL) using deep reinforcement learning. Two phases:

1. **Phase 1 — Skill library.** Independent policies for `standup`, `walk`, `dribble`, `shoot`. Each is trained in its own Python process (own Genesis allocation → bounded GPU memory, no cross-stage scene rebuilds). Each skill takes a continuous command vector (e.g. walk: `[vx, vy, vyaw, foot_clearance, step_freq]`).
2. **Phase 2 — Discrete orchestrator** over the frozen skill library. Each control step the orchestrator emits `(skill_idx, command_vec_7d)` per agent; the chosen skill's frozen policy then drives PD joint targets at 50 Hz. Orchestrator decisions are taken at 10 Hz. Trained with 4v4 self-play under HSL GameController rules. Roles (GK/DEF/MID/ATK) are passed via obs one-hot.

## Status

Skill-library refactor complete. Codebase is now organised around `skills/<name>/` (env+rewards+config per skill), `orchestrator/` (Phase 2 policy + env + self-play), and `training/{train_skill,train_orchestrator}.py`. The old monolithic Phase 1 / Phase 2 envs have been deleted (see "Refactor log" at the bottom).

**Validated:** module imports, CLI parsing, all per-skill reward shapes via mocked-state smoke tests (walk velocity tracking, standup upright bonus, dribble ball-offset/lost, shoot kick gate + power/aim, orchestrator hybrid action + skill routing).

**Not yet validated end-to-end:** real Genesis training runs. Everything below the orchestration logic (multi-robot scene build, physics stepping, GameController integration) needs GPU to validate.

## Simulator Strategy

| Task | Simulator | Reason |
|------|-----------|--------|
| Training (Phase 1 & 2) | Genesis (`genesis-world`) | Massively parallel GPU envs |
| Evaluation / video | MuJoCo 3.x | Sim2sim transfer validation; physics fidelity for reporting |

**Never swap these.** Genesis is fast but lacks MuJoCo's mature rendering; MuJoCo is used only for eval and video.

## External Assets (do not re-download unless missing)

- **Booster K1 robot** — `models/robot/K1/` (from [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1))
  - `K1_22dof.xml` — MJCF (MuJoCo), primary eval format
  - `K1_22dof.urdf` — URDF, used by Genesis
  - 22 DOF: head×2, arms×8, legs×12 (foot links: `left_foot_link`, `right_foot_link`)
- **Field spec** — `configs/field_hsl_2026.json` (from [HSL-Rules FieldGenerator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/main/figs/FieldGenerator/generateField.py))
  - 9 m × 6 m, 1 m border, 2.6 m × 0.8 m goals (centre at `(±4.5, 0, 0.4)`)
- **GameController reference** — [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController/tree/master) — Python reimplementation in `gamecontroller/sim_game_controller.py`

## Project Structure

```
robocup_humanoid_soccer/
├── configs/
│   ├── config.py                  # ProjectConfig, K1RobotConfig, WandbConfig
│   │                              # (Phase1Config/Phase2Config kept as zombies
│   │                              #  for evaluation/evaluate.py — see step 9)
│   └── field_hsl_2026.json        # HSL field dimensions
├── models/
│   ├── field_generator.py         # JSON → MuJoCo XML + Genesis builder
│   ├── field/                     # Auto-generated; do not hand-edit
│   └── robot/K1/                  # Booster K1 URDF/MJCF + meshes
├── envs/
│   └── standup.py                 # Fallen-pose definitions reused by
│                                  # skills/standup/env.py
├── skills/
│   ├── base.py                    # SkillEnv ABC, CommandSpec, SkillSpec
│   ├── common_obs.py              # 78-dim shared base obs builder
│   ├── walk/                      # K1WalkEnv — 5-dim command (vx,vy,vyaw,fh,sf)
│   ├── standup/                   # K1StandupEnv — no command, fallen-pose spawn
│   ├── dribble/                   # K1DribbleEnv — walk+ball_offset (7-dim cmd)
│   └── shoot/                     # K1ShootEnv — [aim_angle, power, foot_pref]
├── orchestrator/
│   ├── config.py                  # OrchestratorConfig, SKILL_ORDER, cmd dims
│   ├── policy.py                  # OrchestratorActorCritic (Categorical+Normal)
│   ├── skill_router.py            # Frozen-skill loader + group-by-skill router
│   ├── rewards.py                 # Team-aware match reward
│   ├── env.py                     # K1MatchEnv (4v4 self-play scene)
│   └── self_play.py               # OpponentPool + split_team_action
├── training/
│   ├── common.py                  # Shared logger/checkpoint/device helpers
│   ├── normalizers.py             # RunningMeanStd, ReturnNormalizer
│   ├── logger.py                  # TensorBoard + W&B
│   ├── algorithms/
│   │   ├── ppo.py                 # train_ppo_vec / train_ppo (single-env)
│   │   ├── flashsac.py            # train_flashsac_vec
│   │   ├── networks.py            # PPOActorCritic, SACActor, TwinQ
│   │   └── replay_buffer.py
│   ├── train_skill.py             # Single-skill PPO/FlashSAC entry point
│   └── train_orchestrator.py      # 4v4 self-play PPO entry point
├── gamecontroller/
│   └── sim_game_controller.py     # HSL GameController state machine
├── evaluation/
│   └── evaluate.py                # MuJoCo eval, sim2sim, video
├── scripts/
│   ├── run.sh                     # Setup + train-skill / train-orchestrator
│   ├── debug_genesis_spawn.py
│   └── debug_mujoco_scene.py
├── checkpoints/                   # skill_<name>/skill_<name>_step*.pt
│                                  # orchestrator/orchestrator_step*.pt
├── logs/                          # TensorBoard runs
└── videos/                        # MuJoCo eval recordings
```

## Common Commands

```bash
# First-time setup (creates .venv, installs deps, downloads K1, generates field)
./scripts/run.sh setup

# Train a single skill (own process — bounded GPU memory)
./scripts/run.sh train-skill walk --device gpu --vec-num-envs 1024 --wandb
./scripts/run.sh train-skill standup
./scripts/run.sh train-skill dribble \
    --init-from checkpoints/skill_walk/skill_walk_step50000000.pt
./scripts/run.sh train-skill shoot

# Or train every skill sequentially
./scripts/run.sh train-skills-all --device gpu --vec-num-envs 1024

# Train the Phase-2 orchestrator (needs all 4 skill checkpoints in
# checkpoints/skill_<name>/; latest by mtime is picked automatically).
./scripts/run.sh train-orchestrator --num-envs 256 --wandb

# Algorithm: PPO (default) or FlashSAC (off-policy SAC with twin Q)
./scripts/run.sh train-skill walk --algorithm flashsac

# Evaluate in MuJoCo with video recording
python -m evaluation.evaluate checkpoints/orchestrator/orchestrator_best.pt \
    --phase phase2 --record-video --num-episodes 10

# Regenerate field model from JSON spec
./scripts/run.sh generate-field
```

## Skill design

| Skill | Command vec (continuous, dim) | Obs add-ons (dim) | obs_dim | Reward focus |
|-------|-------------------------------|-------------------|---------|--------------|
| standup | none (0) | contact 8 (foot/hand z + bool) | 86 | upright + height + progress + feet-grounded + standing-tall + time-scaled success bonus + post-success standing. **Assist-force curriculum** (decaying upward trunk support), **two-stage reward** (discovery→deploy), **multi-critic PPO** (task/reg/success). Timeout-only termination |
| walk | `[vx, vy, vyaw, foot_clearance, step_freq]` (5) | none (0) | 83 | exp-shaped lin/ang vel tracking + posture + foot clearance vs cmd swing + regularizers |
| dribble | walk(5) + `[ball_off_x, ball_off_y]` (7) | ball pos/vel in body frame (6) | 91 | walk shaping + ball_offset (exp) + ball_velocity + ball_lost penalty/terminate at >2 m |
| shoot | `[aim_angle, power, foot_pref]` (3) | ball pos/vel body + target body (9) | 90 | dense approach + ball→target velocity + sparse kick pulse (speed>1.5 m/s, aim<45°) + power/aim match |

All skills share a 78-dim base obs from `skills/common_obs.py`:
- root height (1)
- projected gravity in body frame (3) — orientation proxy without quat sign ambiguity
- body-frame linear velocity (3)
- body-frame angular velocity (3)
- joint pos − default (22)
- joint vel (22)
- last action (22)
- clock sin/cos at `gait_freq_hz` for periodic-gait conditioning (2)

## Orchestrator design

**Obs (156-dim per agent):** shared base obs (78) + ball pos/vel body (6) + 3 teammates (3×6=18) + 4 opponents (4×6=24) + GameController state (24) + role one-hot GK/DEF/MID/ATK (4) + score_diff/time_remaining (2).

**Action (8-dim packed):** discrete skill index ∈ `SKILL_ORDER = (standup, walk, dribble, shoot)` (1) + 7-dim continuous command (per-skill prefix-sliced to `{0, 5, 7, 3}` dims).

**Timing:** 10 Hz orchestrator decisions / 50 Hz inner skill control. Decisions are latched across `inner_steps_per_decision=5` physics calls to reduce switch thrash.

**Self-play:** `OpponentPool` is a CPU state-dict deque (capacity 10 by default). Each iteration, with prob `latest_prob=0.5` we use the most recent snapshot (forces self-tracking); else uniformly over older entries (anti-forgetting). Snapshot every `opponent_update_freq=50` iters. Cold-starts with mirror-self play until the pool populates.

**Reward (team-aware):** sparse goals ±50 + dense per-team possession / ball-toward-goal / defensive coverage + per-agent posture (upright/alive/fall). Goal attribution comes from score-delta (the Python GC's `goal_just_scored` is a single-step pulse without team info).

## Algorithms

Both PPO and FlashSAC are wired through `training/train_skill.py` (FlashSAC = off-policy SAC with twin Q + GPU replay buffer + automatic temperature tuning). Orchestrator training is PPO-only currently (off-policy with self-play has extra design subtleties).

- **Network**: MLP 512→256→128, LayerNorm, ELU activations.
- **Control**: 50 Hz policy (`dt=0.02`), 500 Hz physics (`sim_dt=0.002`), `action_repeat=10`.
- **PD gains**: `kp=50`, `kd=5` for all joints.
- **Device presets** (`training/common.DEVICE_PRESETS`): GPU defaults `vec_num_envs=1024`, FlashSAC `buffer_capacity=2M / batch_size=2048 / gradient_steps=4`. CPU defaults are smaller for smoke tests.

### Standup-specific RL (HoST / HumanUP, 2025)

The standup skill was the hard case (it wouldn't learn under plain reward shaping). Three mechanisms from the 2025 get-up literature, all standup-only and on by default — the other skills route to plain single-critic PPO unchanged:

- **Assist-force curriculum** (HoST, [arXiv:2502.08378](https://arxiv.org/abs/2502.08378)) — a decaying upward "support" force on the trunk (spring-shaped on height deficit, ~67% body weight when fully fallen), weaning 1.0→0.0 over `assist_curriculum_env_steps`, performance-gated by success EMA. Generic `_assist_wrench()` hook in `skills/base.py` step(), summed with push-DR; standup override in `skills/standup/env.py`.
- **Two-stage reward** (HumanUP, [arXiv:2502.12152](https://arxiv.org/abs/2502.12152)) — `StandupConfig.reward_stage`: `"discovery"` zeroes the motion regularizers (via `config.discovery_weights`) so the policy can find any standup; `"deploy"` re-enables them for a smooth motion. Train discovery → `--init-from` the checkpoint for deploy.
- **Multi-critic PPO** (HoST) — one value head per reward group (`rewards.STANDUP_CRITIC_GROUPS = (task, reg, success)`), per-group GAE + return-norm, normalized-advantage aggregation. `PPOActorCritic(n_critics=G)` + `ppo.train_ppo_multicritic_vec`. `train_skill` routes here when the env exposes `CRITIC_GROUP_NAMES` and `cfg.use_multi_critic`. `n_critics=1` keeps the original param names, so existing single-critic checkpoints load unchanged.

## GameController

`gamecontroller/sim_game_controller.py` — Python reimplementation of the HSL GameController state machine. States: `INITIAL → READY → SET → PLAYING → FINISHED`. Manages goals, penalties, set plays, halves (5 min sim time per half, fast-forwarded 10×). Provides `get_state_vector()` (padded into a 24-dim slot in the orchestrator obs). Detects goals via ball position vs goal mouth geometry.

## Domain Randomization

Active. Lives in `envs/domain_randomization.py`. Ranges aligned with Booster's T1 walking config (`booster_gym/envs/T1.yaml`) — closest published reference for a similar biped. Sampled at scene-build:
- **Ground friction** 0.5–1.5
- **Motor gain scaling** kp_scale, kd_scale each ∈ [0.95, 1.05]
- **Joint friction** additive ∈ [0, 0.4] N·m/(rad/s)
- **Base / link mass scaling** [0.8, 1.2] / [0.98, 1.02]
- **COM offset** ±0.02 m on the trunk

Per-step:
- **Random base pushes** every ~5 s, magnitude up to 12 N / 3 N·m, held for 5 control steps. Applied via Genesis' external-wrench API; falls back silently if the build doesn't expose it.

Observation noise (Gaussian σ): root_quat 0.01, lin_vel 0.05, ang_vel 0.10, dof_pos 0.01, dof_vel 0.10 — applied to raw sensor readings before the common obs builder runs, so the projected_gravity / body-frame velocity downstream are computed from noisy inputs (closer to the real sensor model).

## Teacher-Student (sim-to-real)

Two-stage training pipeline for sim-to-real. Implemented via the `--mode` flag on `train_skill.py`:

```bash
# Stage 1: train an oracle teacher with PRIVILEGED obs (DR sample appended).
./scripts/run.sh train-skill walk --mode teacher --device gpu --wandb

# Stage 2: distill into a student that uses PROPRIO obs only (real-robot deployable).
./scripts/run.sh train-skill walk --mode student \
    --teacher-ckpt checkpoints/skill_walk/skill_walk_step100000000.pt
```

**Teacher.** Identical PPO loop as `--mode single`, but `include_privileged=True` appends 8 DR dims to the observation: `[ground_friction, kp_scale, kd_scale, joint_friction, base_mass_scale, com_offset_xyz]`. Teacher learns optimal control given oracle knowledge of the dynamics.

**Student.** Same actor-critic architecture but narrower input (no privileged dims). Trained via DAgger-lite behaviour cloning (`training/algorithms/distillation.py`): MSE between student action and teacher's deterministic action, mixed-policy rollouts with a linear β schedule from 1.0 (teacher only) → 0.0 (student only). The student sees the same noisy proprio + the same DR-perturbed dynamics — what matters for sim-to-real is that it learns invariance to the DR axis, not that it has a clean signal.

**Per-joint PD gains.** T1-style: `kp_hip=kp_knee=200`, `kp_ankle=50`, `kp_arm=50`, `kp_head=20`, with matching `kd`. Per-env scaling by the DR sample is applied at scene build. The legged_gym wisdom is that ankles need lower gains (less mechanical authority) and a stiff hip/knee keeps the base stable.

## W&B Logging

Project: `robocup-humanoid-soccer`. Logger is `training/logger.py` (TensorBoard always on; W&B opt-in via `--wandb`). Per-iteration metrics:
- All reward component breakdowns (one scalar per component).
- Episode stats (mean reward, length, fall rate).
- PPO diagnostics: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction`, `explained_variance`, `learning_rate`.

## Configuration

Per-skill configs are dataclasses inside each skill package: `skills/walk/config.WalkConfig`, `skills/standup/config.StandupConfig`, `skills/dribble/config.DribbleConfig`, `skills/shoot/config.ShootConfig`. Each exposes `num_envs`, `total_timesteps`, PPO hyperparams (`learning_rate`, `gamma`, `clip_range`, `n_steps`, `n_epochs`, `entropy_coef`, …), command vector ranges, and reward-weight subclasses.

Orchestrator config: `orchestrator/config.OrchestratorConfig`. Key tunables:

```python
OrchestratorConfig.num_envs = 256
OrchestratorConfig.players_per_team = 4         # 4v4
OrchestratorConfig.half_duration = 300.0        # 5 min sim time per half
OrchestratorConfig.inner_steps_per_decision = 5 # 50 Hz inner / 10 Hz orchestrator
OrchestratorConfig.opponent_pool_size = 10
OrchestratorConfig.opponent_update_freq = 50    # snapshot every N iterations
```

Top-level `configs/config.ProjectConfig` carries `K1RobotConfig`, `WandbConfig`, and zombie `Phase1Config`/`Phase2Config` (still read by `evaluation/evaluate.py`; refactor pending).

## Checkpoints

- **Per-skill**: `checkpoints/skill_<name>/skill_<name>_step<N>.pt` — `train_skill.py` writes these every 100 iterations + at the end of training. Latest by mtime is what the orchestrator loads.
- **Orchestrator**: `checkpoints/orchestrator/orchestrator_step<N>.pt`.

Checkpoint dicts carry `step`, `phase`, `algorithm`, `policy_state_dict` (or `actor_state_dict` for SAC), `optimizer_state_dict`, optional `obs_norm` and `ret_norm` state.

`--init-from PATH` does a shape-tolerant partial load (any layer whose shape doesn't match the target net is skipped) — useful for `walk → dribble` warm-start since they share the same backbone but differ in input width.

## Dependencies

```
Python 3.10+
torch >= 2.0
genesis-world >= 0.4   (GPU training — CUDA required)
mujoco >= 3.0          (evaluation + video)
wandb >= 0.16
imageio + imageio-ffmpeg  (video)
```

Install via `./scripts/run.sh setup` or `pip install -e .[dev]`.

## References

- [HSL Rules & Field Generator](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/main/figs/FieldGenerator/generateField.py)
- [HSL GameController](https://github.com/RoboCup-HumanoidSoccerLeague/GameController/tree/master)
- [Genesis Simulator](https://github.com/Genesis-Embodied-AI/genesis-world)
- [Booster K1 Assets](https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1)

---

## Refactor log

For history. The skill-library refactor was completed across 10 incremental steps, summarised here.

- **Step 1** — Extracted shared helpers (`training/common.py`): logger setup, checkpoint loader, device resolution, policy factory, eval-video writer. Fixed pre-existing broken `create_policy` import in `evaluation/evaluate.py`.
- **Step 2** — `skills/base.py` (SkillEnv ABC, CommandSpec, SkillSpec) + `skills/common_obs.py` (78-dim base obs builder with projected_gravity, body-frame velocities, clock).
- **Step 3** — Walk skill ported (`skills/walk/`) + generic `training/train_skill.py` with `_SKILL_REGISTRY`.
- **Step 4** — Standup skill ported, validates `CommandSpec.empty()` no-command path. `--algorithm {ppo,flashsac,sac}` flag added.
- **Step 5** — Dribble skill ported. Ball entity in `_add_scene_extras`, reward composes walk helpers + ball-tracking terms.
- **Step 6** — Shoot skill ported. Per-env world target derived from `aim_angle` at reset; reward gates the kick bonus on speed>1.5 m/s + aim<45°.
- **Step 7** — Orchestrator package: hybrid `OrchestratorActorCritic` (Categorical+Normal), `SkillRouter` with group-by-skill batched inference, team-aware `compute_match_reward`, `K1MatchEnv` obs/action contract.
- **Step 8** — Wired `K1MatchEnv` to real Genesis (multi-robot scene, batched state read, joint-target application, per-env GC). Added `OpponentPool` + `split_team_action` for self-play. `training/train_orchestrator.py` runs PPO on team-0 transitions with snapshot opponents.
- **Step 9** — Deleted legacy monolith: `envs/{phase1_dribble_shoot,phase1_vec,style_command,perturbations,gait_rewards,domain_randomization,phase2_match,rewards}.py` and `training/train.py`.
- **Step 10** — Updated `scripts/run.sh` (`train-skill <name>`, `train-skills-all`, `train-orchestrator`) and rewrote this guide.
