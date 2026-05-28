# Standup skill — training guide

Recovering the Booster K1 from any fallen pose to a stable upright stance,
optimised for speed (sub-second is the target, hold for 1 s confirms
stability) and minimal arm use. This document covers the recommended
training pipeline end-to-end, including sim2real distillation.

## TL;DR — recommended pipeline

```bash
# 1. Train a teacher with all the privileged signals (contact + DR).
#    Fast convergence; not directly deployable on hardware.
./scripts/run.sh train-skill standup \
    --algorithm ppo --mode teacher \
    --device gpu --vec-num-envs 2048 \
    --total-timesteps 100_000_000 --wandb

# 2. Distill into a proprio-only student that IS deployable.
./scripts/run.sh train-skill standup \
    --algorithm ppo --mode student \
    --teacher-ckpt checkpoints/skill_standup/skill_standup_step100000000.pt \
    --device gpu --vec-num-envs 2048 \
    --total-timesteps 20_000_000 --wandb
```

If sim2real is not a current concern, just run `--mode single` and stop
after step 1 (using a teacher checkpoint is fine — it's a regular PPO
policy and works in eval).

---

## Design at a glance

| Component | Purpose | Where it lives |
|-----------|---------|----------------|
| Settle pool | Physically-realistic fallen starts (no hand-coded poses) | `env._build_settle_pool` |
| Contact obs (8 dims) | Foot/hand z + contact bool — tells the policy what's on the floor. **Privileged in sim2real.** | `env._read_contact_state` |
| `upright_progress` reward | Pays per step for `Δup > 0` — breaks the side-plank attractor | `rewards.compute_standup_reward` |
| `near_upright_gate` | Motion penalties only fire in the final balancing zone (up ∈ [0.7, 0.95]) | `rewards.near_upright_gate` |
| Sustained-success terminal bonus | `success_bonus × exp(-t_first / τ)` paid once on hold completion | `rewards.compute_standup_reward` |
| Arm-pose deviation penalty | Discourages heavy arm use without forbidding it | `rewards.compute_standup_reward` |

### Observation layout (proprio_only=False, include_privileged=True)

```
[0  : 78]   base proprio          ← deployable
[78 : 86]   contact addons        ← sim-only (foot/hand z + contact)
[86 : 94]   DR privileged         ← sim-only (friction, kp/kd, mass, COM)
```

Both non-deployable blocks sit at the tail, so a leading-prefix slice
`obs[:, :78]` gives exactly the real-robot proprio. The
`env.non_deployable_dim` property reports `8 + 8 = 16` and is what the
distillation pipeline reads to size the student.

### Reward components (`StandupRewardWeights`)

| Term | Weight | Notes |
|------|--------|-------|
| `upright` | 3.0 | `(cos(tilt)+1)/2` — smooth in [0, 1] everywhere, monotonic from upside-down to upright |
| `height` | 2.0 | Gaussian on trunk z, σ=0.3 (gradient is non-flat from z=0.15 upward) |
| `upright_progress` | 5.0 | `max(0, Δup)` — pays for active uprightening, not for being-in-state. Kills the side-plank attractor |
| `arm_pose_dev` | 0.05 | `Σ(q_arm − rest)²` — light arm use is essentially free, sustained flailing accrues cost |
| `base_ang_vel_sway` | 0.05 | ωx² + ωy², gated |
| `base_lin_vel_drift` | 0.5 | ‖v‖², gated |
| `joint_vel_quiet` | 0.001 | Σ q̇², gated |
| `action_smoothness` | 0.1 | (Δa)², gated |
| `action_jerk` | 0.1 | (Δ²a)², gated |
| `time_penalty` | 1.0 | Dense −1/step until sustained-success |
| `success_persistence` | 5.0 | +5/step during the hold window |
| `success_bonus` | 400.0 | Terminal, scaled `× exp(−t_first / 40)` (τ=0.8 s) — 0.5 s stand pays ~214, 2 s pays ~33 |

All "gated" penalties are scaled by `near_upright_gate(up)` which ramps
from 0 at up=0.7 to 1 at up=0.95. The intent: the recovery itself is
motion-free, stability shaping only activates in the final balancing
range.

---

## Training workflow

### Stage 1 — teacher (with contact + DR)

The teacher sees everything: proprio + contact obs + DR sample. This
gives the fastest possible convergence because the policy doesn't have
to *infer* contact from indirect signals; it's told directly.

```bash
./scripts/run.sh train-skill standup \
    --algorithm ppo --mode teacher \
    --device gpu --vec-num-envs 2048 \
    --total-timesteps 100_000_000 --wandb
```

What this does:
- `--mode teacher` sets `include_privileged=True` → DR sample appended to obs.
- `StandupConfig.proprio_only` stays `False` (default) → contact obs is in.
- Net obs dim: 94 (78 proprio + 8 contact + 8 DR).
- Checkpoints land in `checkpoints/skill_standup/skill_standup_step*.pt`.

### Stage 2 — student (proprio-only, deployable)

The student is a behaviour-cloning network that learns to emulate the
teacher's actions using **only what the real robot can measure**: the
78-dim base proprio (joint state, IMU, last action, clock). It learns
to implicitly estimate contact from joint velocities/torques and the
IMU — exactly the sim2real recipe used by RMA, DreamWaQ, and the
Concurrent Teacher-Student family.

```bash
./scripts/run.sh train-skill standup \
    --algorithm ppo --mode student \
    --teacher-ckpt checkpoints/skill_standup/skill_standup_step100000000.pt \
    --device gpu --vec-num-envs 2048 \
    --total-timesteps 20_000_000 --wandb
```

What this does internally (`training/algorithms/distillation.py`):
1. Builds the env with `include_privileged=True` (so the teacher gets the full obs).
2. Computes `student_obs_dim = env.obs_dim − env.non_deployable_dim = 94 − 16 = 78`.
3. Each iteration:
   - Rolls the env forward `n_steps`, mixing teacher and student actions per a DAgger β-schedule (starts at 1.0 = "always teacher", linearly decays to 0.0 = "always student").
   - At each step, records `(obs[:, :78], teacher_deterministic_action)`.
4. Trains the student to MSE the teacher's mean action.
5. Saves to `checkpoints/student_standup_step*.pt`.

The student inherits the same network architecture as the teacher,
just with a narrower input layer (78 vs 94). At deployment time you
feed it the 78-dim proprio vector and read the 22-dim joint targets.

### Alternative — single-stage proprio-only

If you don't want the two-stage overhead:

```python
# Edit skills/standup/config.py:
@dataclass
class StandupConfig:
    proprio_only: bool = True   # ← flip this
```

Then train normally:

```bash
./scripts/run.sh train-skill standup --algorithm ppo \
    --device gpu --vec-num-envs 2048 --total-timesteps 100_000_000 --wandb
```

This trains a policy directly on 78-dim proprio. Slower convergence
(no contact signal during training), but directly deployable.
Typically takes 2–3× the env-steps of the teacher path.

### Algorithm choice

Both PPO and FlashSAC work. Recommendations:
- **PPO** — default. Stable, predictable, well-behaved on the dense
  reward landscape we have here. Use this for production runs.
- **FlashSAC** — better sample efficiency on sparser reward shapes.
  We added enough dense shaping that PPO's sample efficiency is OK,
  but if you're VRAM-constrained and want fewer env-steps for the
  same wall clock, SAC's 4–8 gradient steps per env step can help.

---

## What to watch in TensorBoard

Key signals, in order of "if this isn't trending right, something is broken":

1. **`rewards/upright_progress`** — average `max(0, Δup)` per step. Should be **clearly positive and rising** in the first ~5M env-steps. If it stays near zero, the policy is stuck in a stable basin (lying still, side-plank). This is the canary for the side-plank failure mode.

2. **`rewards/sustained_rate`** — fraction of envs that completed a sustained-success this iteration. Should start at zero and ramp to 0.5+ as standup becomes reliable.

3. **`rewards/mean_robot_z`** — average trunk height across envs. Useful smoke-check: should rise from ~0.15 (fallen pool average) toward 0.55 (target standing).

4. **`mean_reward`** / **`R̄`** — per-episode total. Should climb monotonically once `upright_progress` and `sustained_rate` are moving. Numerical landmarks:
   - Pool-start "lie still" baseline: ~−500 to −650 per episode.
   - Side-plank attractor (if you hit it): ~+200 to +500.
   - First successful standups: ~+1500 to +2500 (depending on speed).

5. **`rewards/arm_pose_dev`** — average arm displacement². Should plateau low (~0.5–2.0) once the policy has converged; spikes above ~5 indicate the policy is leaning hard on arms.

6. **`rewards/near_upright_gate`** — average gate activation. Rises with `mean_robot_z`; useful to confirm motion penalties are actually firing once balancing is reached.

7. **PPO diagnostics** — `approx_kl` should stay under ~0.05, `clip_fraction` under ~0.3, `explained_variance` climbing toward 1.0.

---

## Hyperparameters worth tuning (and ones not to)

### Don't touch unless you have a strong reason
- `success_hold_steps=50` (1 s hold) — defines what "stable" means. Lowering it makes standups easier to mark successful but reduces deployment quality.
- `upright_threshold=0.92` (cos, ≈ 23° tilt) — defines what "upright" means.
- `time_to_stand_tau_steps=40` (0.8 s) — sets the shape of the speed bonus.
- Settle-pool params (`spawn_height_*`, `settle_steps`, `settle_pool_rounds`) — already tuned to give ~120 diverse fallen states at `vec_num_envs=32` and ~3800 at 1024.

### Things you might reasonably tune
- `upright_progress` (default 5.0) — bump higher (8–10) if the policy is still getting stuck in side-plank-like local minima.
- `arm_pose_dev` (default 0.05) — bump up to 0.1 if the policy is using arms more than you'd like; reduce to 0.02 if it can't get up at all on bad starts.
- `pool_max_upright=0.7` — lower (e.g. 0.5) to keep only HARDER fallen starts, harder (e.g. 0.85) for an easier curriculum.
- `entropy_coef` (default 0.005) — raise to 0.01 if the policy is committing to bad strategies too fast.

---

## Troubleshooting

### "The policy is stuck at side-plank / lying still"

Check `rewards/upright_progress`:
- If it's near zero → the policy never moves upward. Either the reward gradient is too weak (bump `upright_progress` to 8–10) or exploration is too narrow (bump entropy to 0.01).
- If it's positive but the policy isn't completing standups → it lurches forward in one step then falls back; lower `success_hold_steps` to 30 temporarily as a curriculum to give the policy partial credit, then raise it back.

### "Training crashes during the settle-pool build"

The settle-pool build does ~4 × `settle_steps` = ~1200 scene-steps before the first iteration. If Genesis OOMs here, lower `settle_pool_rounds` to 2 (still gives ~2000 states at 1024 envs) or `vec_num_envs`.

If the pool ends up empty (`RuntimeError: settle pool is empty after filtering`), loosen the filter: raise `pool_max_upright` to 0.85 or `pool_max_height` to 0.5.

### "The student diverges during distillation"

DAgger β-schedule starts at 1.0 (always teacher) — student should see fully teacher-driven rollouts at the start. If BC loss climbs over time:
- Lower `learning_rate` (try 1e-4 instead of 3e-4)
- Increase `n_epochs` per iteration (default 3, try 5)
- Raise the obs normalizer warm-up by training the teacher for longer.

### "Real-robot deployment misbehaves even though sim looked great"

Diagnosis order:
1. Confirm you're loading a `student_*.pt` checkpoint, not a `skill_standup_*.pt` teacher. Teachers expect 94-dim obs; the real robot can't fill the contact + DR slots.
2. Verify the 78-dim obs ordering on the real robot matches `skills/common_obs.py` byte-for-byte (root z first, then projected gravity, body-frame velocities, joint pos – default, joint vel, last action, clock sin/cos).
3. Check that the PD gains on the real controller match `K1RobotConfig` (`kp_hip=200`, `kp_knee=200`, `kp_ankle=50`, `kp_arm=50`, `kp_head=20`).
4. Domain randomization may not have covered your physical robot's parameters. Re-train the teacher with wider `DomainRandConfig` ranges if your friction/mass is outside [0.5, 1.5] × nominal.

---

## File map

```
skills/standup/
├── README.md      ← you are here
├── __init__.py
├── config.py      ← StandupConfig + StandupRewardWeights
├── env.py         ← K1StandupEnv, settle-pool builder, contact-obs reader
└── rewards.py     ← compute_standup_reward + per-component helpers
```

Related:
- `training/train_skill.py` — single entry point for `--mode {single,teacher,student}`
- `training/algorithms/distillation.py` — DAgger-lite BC loop
- `training/algorithms/ppo.py` — `train_ppo_vec`
- `training/algorithms/flashsac.py` — `train_flashsac_vec`
- `envs/domain_randomization.py` — DR sampler + push scheduler
