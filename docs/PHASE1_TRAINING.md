# Phase 1 Training Guide

How to drive the robust Phase 1 pipeline added in this iteration.

## Curriculum

Order: `stand → standup → walk → dribble → shoot → full`

- **stand** — pure balance. Ball is parked off-field. No pushes.
- **standup** — robot resets in supine/prone/side poses. Reward is
  upright orientation + trunk height + smoothness; episode terminates
  on success so the policy learns a clear "I made it" signal.
- **walk → full** — style-conditioned locomotion + ball pursuit. Each
  reset samples a 5-dim style command (target vx, vy, yaw rate,
  aggressiveness, defensiveness). Reward includes commanded-velocity
  tracking + gait shaping (foot alternation, foot clearance,
  joint-limit penalty, action smoothness).

The standup stage sits BEFORE walk so the policy can recover from
falls in later stages instead of timing out the episode.

## Style command

Defined in `envs/style_command.py`. Five dimensions:

| Index | Meaning              | Default range |
| ----- | -------------------- | ------------- |
| 0     | target_vx (body m/s) | -0.4 to 1.0   |
| 1     | target_vy (body m/s) | -0.4 to 0.4   |
| 2     | target_yaw_rate      | -1.0 to 1.0   |
| 3     | aggressiveness       | 0.0 to 1.0    |
| 4     | defensiveness        | 0.0 to 1.0    |

Sampled per reset, optionally resampled mid-episode
(`StyleCommandRanges.resample_prob_per_step`). The observation grows
from 78-dim base → 83-dim (78 + 5).

At deploy time, freeze the command with `sampler.fix(vx=0.8, aggressiveness=1.0)`
to play a striker, or `sampler.fix(vx=0.4, defensiveness=1.0)` for a
defender. One policy, multiple play styles.

## Pushes

`envs/perturbations.py` applies external horizontal forces to the
trunk at random intervals. Force / probability ramps with curriculum
stage — see `PerturbationSchedule.for_stage()`:

| Stage    | p(push)/step | force range (N) |
| -------- | ------------ | --------------- |
| stand    | 0            | —               |
| standup  | 0.001        | 10–40 (+vertical) |
| walk     | 0.003        | 5–25            |
| dribble  | 0.005        | 10–40           |
| shoot    | 0.004        | 10–35           |
| full     | 0.006        | 15–50           |

K1 trunk mass is ~6.5 kg, so a 30 N push for ~150 ms changes velocity
by ~0.7 m/s. Tune these in `PerturbationSchedule.for_stage` if you see
the policy never tipping or always tipping.

## Vectorised training

Set in `configs/config.py`:
```python
Phase1Config.use_vec_env = True
Phase1Config.vec_num_envs = 1024   # bump to 4096+ on a strong GPU
```

The vec env (`envs/phase1_vec.py`) builds Genesis with `n_envs=N` so
all N rollouts execute as batched GPU ops. Throughput on a single
A100 typically goes from ~200 sim steps/s (single env) to ~150–250k
sim steps/s (4096 envs). The single-env path stays available for
debug + evaluation.

Caveats:
- Standup reward + the full gait-shaping bundle are scalar-only in
  this iteration. The vec env uses a simpler batched reward
  (upright + height + velocity tracking + ball distance). Policies
  trained vectorised are compatible with the single-env eval path.
- Pusher is single-env only for now (the link-external-force API needs
  a (n_envs, 3) batch path).

## PPO improvements

In `training/normalizers.py`:
- **Observation running mean/std** — applied before every policy /
  value forward pass. The same stats are saved with the checkpoint so
  eval and Phase 2 fine-tuning use matching scaling.
- **Reward normalisation** — divides rewards by the running std of
  discounted returns so value loss stays well-scaled regardless of
  reward magnitude (the standup +50 success bonus, for instance,
  would otherwise dominate).
- **log_std schedule** — action noise decays linearly from log_std=-0.5
  (std≈0.61) at the start of each stage to log_std=-1.5 (std≈0.22)
  at the end. Faster convergence on the late "exploit" phase.

## Quick recipes

```bash
# Smoke-test the single-env stack end-to-end (CPU OK)
python -m training.train --phase 1 --no-curriculum

# Curriculum on CPU (slow, but works)
python -m training.train --phase 1

# Full vectorised curriculum on GPU
python - <<'PY'
from configs.config import ProjectConfig
from training.train import train_phase1
cfg = ProjectConfig()
cfg.phase1.use_vec_env = True
cfg.phase1.vec_num_envs = 4096
train_phase1(cfg)
PY

# Resume a stage
python -m training.train --phase 1 --resume checkpoints/phase1_walk_step12345678.pt
```

## Logging

TensorBoard is on by default. Logs land in `logs/tb/<run_name>/`. Open with:

```bash
tensorboard --logdir logs/tb
```

To also push to W&B, pass `--wandb` to the trainer (or `--wandb-project NAME`
to override the project). W&B is in the `[wandb]` extra:

```bash
pip install -e .[wandb]
python -m training.train --phase 1 --wandb
```

Without W&B installed, `--wandb` warns and keeps logging to TB only — the
run never crashes over telemetry.

## Domain randomization (motors + body + ball)

`envs/domain_randomization.py` — `MotorRandomizer` samples fresh values
per reset for every actuated joint:

| Parameter      | Range (× baseline) | Why it matters for sim2real |
| -------------- | ------------------ | --------------------------- |
| kp             | 0.80 – 1.20        | motor gain drift unit-to-unit |
| kv             | 0.70 – 1.30        | gearbox damping varies with temp |
| damping        | 0.5 – 1.5          | passive joint friction |
| frictionloss   | 0.5 – 1.5          | static stiction |
| armature       | 0.8 – 1.2          | effective rotor inertia |
| torque limit   | 0.80 – 1.10        | hot motors saturate earlier |
| link mass      | 0.90 – 1.10        | parts replaced over time |
| ball mass      | 0.16 – 0.24 kg     | absolute (different ball sets) |
| action delay   | 0 – 2 control steps | controller-to-actuator latency |
| target noise   | σ=0.005 rad        | encoder quantisation |

Enabled by default via `Phase1Config.use_domain_randomization = True`.
Disable for ablation studies with `cfg.phase1.use_domain_randomization = False`.

The action-delay buffer is implemented in the env's step path
(`MotorRandomizer.delay_action(...)`): the policy's output goes into
a ring buffer, and the action applied to the PD controller is the one
that was generated `delay` control steps ago.

## Diagnostics when training looks broken

- **Reward plateau near 0** — check `obs_norm` is updating: print
  `obs_norm.mean[:8]` mid-training. If still all zeros, the
  obs callback is skipping (the env `_get_obs` raised and returned
  zeros).
- **Robot flies off** — see `scripts/debug_genesis_spawn.py`. A spawn
  height bug or PD-gain mismatch will show up there before training
  burns hours of compute.
- **Sim2real gap on the real K1** — increase
  `DomainRandomizer.ranges.robot_mass_mult` and
  `actuator_delay_steps`, and turn on texture randomisation in the
  next iteration.
