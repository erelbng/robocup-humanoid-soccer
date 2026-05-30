# Standup skill — training guide

Recovering the Booster K1 from any fallen pose to a stable upright stance,
optimised for speed (sub-second is the target, hold for 1 s confirms
stability) and minimal arm use. This document covers the recommended
training pipeline end-to-end, including sim2real distillation.

## TL;DR — recommended pipeline

Getting a robust standup is a **two-stage discovery → deploy** flow. The
defaults (assist-force curriculum, `reward_stage="discovery"`, multi-critic
PPO) are tuned for stage 1; you flip two config fields for stage 2.

```bash
# Stage 1 — DISCOVERY. Find ANY standup. Assist force on, motion
# regularizers off, one critic per reward group. Defaults are already set
# for this, so no extra flags are needed.
./scripts/run.sh train-skill standup \
    --device gpu --vec-num-envs 2048 \
    --total-timesteps 50_000_000 --wandb

# Stage 2 — DEPLOY. Refine into a smooth, sim2real-ready motion. Edit
# skills/standup/config.py: set reward_stage="deploy" (re-enables the
# motion regularizers), then warm-start from the discovery checkpoint
# (actor weights transfer; the per-group critics restart fresh).
DISCOVERY=$(ls -t checkpoints/skill_standup/skill_standup_step*.pt | head -1)
./scripts/run.sh train-skill standup \
    --device gpu --vec-num-envs 2048 \
    --init-from "$DISCOVERY" \
    --total-timesteps 50_000_000 --wandb
```

### Sim2real (teacher → student), once standup is working

```bash
# 1. Teacher: PPO with privileged DR + contact obs. Fast, not deployable.
./scripts/run.sh train-skill standup --mode teacher \
    --device gpu --vec-num-envs 2048 --total-timesteps 100_000_000 --wandb

# 2. Student: distill into a proprio-only deployable policy.
TEACHER=$(ls -t checkpoints/skill_standup/skill_standup_step*.pt | head -1)
./scripts/run.sh train-skill standup --mode student \
    --teacher-ckpt "$TEACHER" \
    --device gpu --vec-num-envs 2048 --total-timesteps 20_000_000 --wandb
```

`--mode teacher`/`student` work with all the above on by default. If
sim2real is not a current concern, a `--mode single` (or teacher) checkpoint
is a regular PPO policy and works fine in eval.

---

## Design at a glance

| Component | Purpose | Where it lives |
|-----------|---------|----------------|
| Settle pool | Physically-realistic fallen starts (no hand-coded poses) | `env._build_settle_pool` |
| **Assist-force curriculum** | Decaying upward "support" force on the trunk (HoST's infant-support trick) — the primary exploration aid. Lets the policy reach the standing pose while assisted, then weans to zero | `env._assist_wrench`, `env._current_assist_fraction` |
| **Two-stage reward** | `reward_stage="discovery"` zeroes motion regularizers so the policy can find ANY standup; `"deploy"` re-enables them for a smooth motion (HumanUP: early regularization blocks discovery) | `config.discovery_weights`, `env._reward_weights` |
| **Multi-critic PPO** | One value head per reward group (task/reg/success) — single critic over the mixed reward fails (HoST) | `rewards.STANDUP_CRITIC_GROUPS`, `ppo.train_ppo_multicritic_vec` |
| Easy pool + reverse curriculum | Near-standing starts that ramp out as training progresses. **Off by default** (`easy_pool_enabled=False`) — superseded by the assist-force curriculum, which changes task difficulty rather than just start state. Running both confounds the experiment | `env._build_easy_pool`, `env._current_easy_fraction` |
| Contact obs (8 dims) | Foot/hand z + contact bool — tells the policy what's on the floor. **Privileged in sim2real.** | `env._read_contact_state` |
| `upright_progress` reward | Pays per step for `Δup > 0` — breaks the side-plank attractor | `rewards.compute_standup_reward` |
| `near_upright_gate` | Motion penalties only fire in the final balancing zone (up ∈ [0.7, 0.95]) | `rewards.near_upright_gate` |
| Sustained-success terminal bonus | `success_bonus × exp(-t_first / τ)` paid once on hold completion | `rewards.compute_standup_reward` |
| Arm-pose deviation penalty | Discourages heavy arm use without forbidding it | `rewards.compute_standup_reward` |

### Why these three (the 2025 get-up literature)

Earlier iterations relied on reward shaping + a reverse pose curriculum and
got stuck (side-plank / kneel / squat attractors). The two papers that
solved real-humanoid get-up — [HoST](https://arxiv.org/abs/2502.08378) and
[HumanUP](https://arxiv.org/abs/2502.12152) — both use **pure RL, no
imitation/AMP**, and pin the wins on three things we now implement:

1. **Assist-force curriculum (HoST).** A decaying upward force on the
   trunk, like helping an infant stand. This is the exploration
   breakthrough: the policy reaches the upright pose while supported and
   learns the whole fallen→standing trajectory, then the force weans to 0
   so the final policy stands unaided.
   - Spring-shaped on height deficit: strongest when fully fallen
     (~131 N ≈ 67 % of the 20 kg body weight at z=0.10), ~0 near standing.
   - World-up, upward-only, applied at the trunk; summed onto the push-DR
     wrench via the generic `_assist_wrench` hook in `skills/base.py`.
   - Fraction decays 1.0 → 0.0 over `assist_curriculum_env_steps`,
     **performance-gated** by the success EMA so support isn't pulled out
     from under a still-failing policy (`assist_min_success`/`assist_min_frac`).

2. **Two-stage reward (HumanUP).** Early regularization blocks task
   discovery. `reward_stage="discovery"` (default) zeroes the six motion
   regularizers (smoothness, jerk, arm-pose, sway, drift, joint-quiet) so
   the policy can find *any* standup; `"deploy"` turns them back on to
   refine the motion. Train discovery → warm-start a deploy run from its
   checkpoint with `--init-from`.

3. **Multi-critic PPO (HoST).** A single critic over the full reward
   (a +400 terminal pulse mixed with dense [0,1] shaping and small
   penalties) achieved *zero* success in HoST's ablation — the value net
   can't fit such a heterogeneous return. We split the reward into
   `STANDUP_CRITIC_GROUPS = (task, reg, success)`, give each its own value
   head + return-normalizer + GAE, normalize advantages per group, then
   aggregate as a `critic_group_weights`-weighted sum for the policy
   surrogate. The three groups sum exactly to the single-critic reward, so
   nothing about the reward *magnitude* changes — only how it's valued.
   Watch the per-group `explained_variance_{task,reg,success}` in
   TensorBoard; each should climb toward ~1.

All three are **standup-only** and on by default (`assist_force_enabled`,
`reward_stage="discovery"`, `use_multi_critic` in `StandupConfig`). The
other skills are untouched: they expose no `CRITIC_GROUP_NAMES`, so
`train_skill` transparently routes them to single-critic PPO.

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
| `upright_progress` | 10.0 | `max(0, Δup)` — pays for active uprightening, not for being-in-state. Kills the side-plank / kneel attractor |
| `arm_pose_dev` | 0.2 | `Σ(q_arm − rest)²` × `arm_gate(up; 0.5 → 0.85)` — pushes the final standing pose to arms-at-the-sides instead of T-pose. Recovery (up < 0.5) is completely free, so the policy can still use arms to push off the floor; penalty ramps in as the robot approaches upright and saturates by up=0.85. |
| `base_ang_vel_sway` | 0.05 | ωx² + ωy², gated |
| `base_lin_vel_drift` | 0.5 | ‖v‖², gated |
| `joint_vel_quiet` | 0.001 | Σ q̇², gated |
| `action_smoothness` | 0.1 | (Δa)², gated |
| `action_jerk` | 0.1 | (Δ²a)², gated |
| `time_penalty` | 3.0 | Dense −3/step until sustained-success (≥2.5 keeps the fallen floor net-negative) |
| `success_persistence` | 5.0 | +5/step during the hold window |
| `success_bonus` | 400.0 | One-shot pulse on streak completion, scaled `× exp(−t_first / 150)` (τ=3.0 s) — 1 s stand pays ~330, 2 s ~150, 3 s ~100 |
| `post_success_standing` | 10.0 | +10/step for every frame still standing AFTER the episode's first sustained success. Episode runs to MAX_EPISODE_STEPS so a fast standup that falls forfeits ~1500–2000 of opportunity cost — the dominant gradient for "stand fast AND stay up" |
| `foot_grounded_up` | 5.0 | Anti-gaming: pays only when BOTH feet z < `foot_grounded_max_z` (0.10 m) AND trunk z > `trunk_up_min_z` (0.30 m) AND the trunk is roughly vertical. Smooth multiplicative gate (`feet × trunk_lift × max(0, cos(tilt))`). Closes the bridge / shoulder-stand / sprawled-on-back AND side-plank local optima — anything that's not actually standing on its feet evaluates to ~0 here. Saturates at the squat. |
| `standing_tall` | 5.0 | Continues where `foot_grounded_up` saturates: same feet-grounded × upright gate × trunk ramp on [`standing_tall_min_z`, `standing_tall_max_z`] = [0.30, 0.55]. 0 at squat, 1.0 at full K1 standing height. Stacks ADDITIVELY on top of `foot_grounded_up` so the squat reward is unchanged but full extension pays ~5/step extra (~1250 over the post-squat trajectory). Pulls the policy out of the squat local optimum. Like `foot_grounded_up`, gated by orientation so a side-plank can't game it. |

All "gated" penalties are scaled by `near_upright_gate(up)` which ramps
from 0 at up=0.7 to 1 at up=0.95. The intent: the recovery itself is
motion-free, stability shaping only activates in the final balancing
range.

**Critic groups / discovery zeroing.** For multi-critic PPO the terms map
to `STANDUP_CRITIC_GROUPS`: `task` = upright + height + upright_progress +
foot_grounded_up + standing_tall; `reg` = arm_pose_dev + sway + drift +
joint_vel_quiet + smoothness + jerk; `success` = success_persistence +
time_penalty + success_bonus + post_success_standing. In the **discovery**
stage the entire `reg` group is zeroed (`config.discovery_weights`), so its
critic just learns ≈0 and contributes ≈0 advantage — by design.

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
TEACHER=$(ls -t checkpoints/skill_standup/skill_standup_step*.pt | head -1)
./scripts/run.sh train-skill standup \
    --algorithm ppo --mode student \
    --teacher-ckpt "$TEACHER" \
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

2. **`rewards/sustained_rate`** — fraction of envs that completed a sustained-success this iteration (the one-step pulse). Should start at zero and ramp to ~0.01 with the curriculum, then settle: because envs no longer terminate on success, sustained_now fires AT MOST ONCE per episode, so high success rates show up as `achieved_sustained_rate ≈ 1.0` with a much smaller `sustained_rate` (sustained_now / episode_length).

   **`rewards/achieved_sustained_rate`** — fraction of envs that have completed at least one sustained success this episode. This is the better signal once the curriculum starts paying off: should climb from 0 → 0.5+ → 1.0 as the policy reliably stands up. If it tops out below 0.8 the policy is failing on a subset of fallen poses.

   **`rewards/post_success_standing`** — average per-step "still standing after success" fraction. Climbs as `achieved_sustained_rate` × (post-success fraction of the episode spent upright). The single most informative scalar for "robot stood up AND stayed up".

3. **`rewards/mean_robot_z`** — average trunk height across envs. Useful smoke-check: should rise from ~0.15 (fallen pool average) toward 0.55 (target standing).

4. **`mean_reward`** / **`R̄`** — per-episode total. Should climb monotonically once `upright_progress` and `achieved_sustained_rate` are moving. Numerical landmarks with the post-success standing reward:
   - Pool-start "lie still" baseline: ~−500 to −650 per episode.
   - Side-plank attractor (if you hit it): ~+200 to +500.
   - First successful standups (1 s held, then collapse): ~+500 to +900 — bonus paid but most post-success steps forfeited.
   - Fast-and-stable standup (1.5 s to upright, holds the rest of the 5 s episode): ~+2000 to +2500.
   - Sub-second standup that holds: ~+2500 to +3000.

5. **`rewards/arm_pose_dev`** — average arm displacement². Should plateau low (~0.5–2.0) once the policy has converged; spikes above ~5 indicate the policy is leaning hard on arms.

6. **`rewards/near_upright_gate`** — average gate activation. Rises with `mean_robot_z`; useful to confirm motion penalties are actually firing once balancing is reached.

7. **`rewards/hold_steps_current`** / **`rewards/upright_threshold_current`** / **`rewards/target_height_current`** / **`rewards/easy_start_fraction`** — the four curriculum knobs. Should ramp linearly 15→50, 0.80→0.92, 0.40→0.55, and 1.0→0.0 over the first ~25M env-steps. Use them to confirm the curricula are advancing as expected. Early in training `mean_robot_z` should be HIGH (~0.55) because most episodes start standing — this is the expected curriculum effect, not the policy succeeding from fallen starts. Watch for `mean_robot_z` and `upright_raw` STAYING high as `easy_start_fraction` drops; that's the signal the policy has learned to stand up.

8. **`rewards/foot_grounded_up`** / **`rewards/mean_foot_z`** — anti-gaming signal in [0, 1]. Should climb from 0 toward 1.0 as the policy learns to put feet down with trunk lifted. If it stays near 0 while `upright` and `height` rise, the policy is gaming the dense terms with a bridge / sprawled pose — exactly the failure this term is designed to close. `mean_foot_z` is the mean of both feet z; should drop toward `foot_grounded_max_z` (0.10) and stay there once standing.

9. **`rewards/standing_tall`** — full-extension signal in [0, 1]. Lags behind `foot_grounded_up` because it only kicks in past the squat (trunk_z > 0.30). Should climb later in training as the policy learns to extend out of the squat into full standing. If `foot_grounded_up` saturates high (~0.8+) but `standing_tall` stays near 0, the policy is stuck in the squat attractor — train longer or bump `standing_tall` weight to 8–10.

10. **`rewards/assist_fraction`** / **`rewards/assist_force_mean`** — the assist-force curriculum. `assist_fraction` should track 1.0 → 0.0 over `assist_curriculum_env_steps` (held at `assist_min_frac` while the success EMA is below `assist_min_success`). The key thing to watch: `achieved_sustained_rate` should be climbing **while `assist_fraction` is still high** — that's the policy learning the trajectory under support. If success only appears at high assist and **collapses as the force weans off**, lengthen `assist_curriculum_env_steps` or raise `assist_min_success` so the wean-off is slower / better gated.

11. **`explained_variance_task`** / **`_reg`** / **`_success`** (multi-critic only) — per-group critic fit. Each should climb toward ~1.0. `explained_variance_reg` will sit near 0 in the discovery stage (its reward is zeroed) — expected. If `_task` or `_success` stays low/negative, that critic isn't fitting its return; check the group's reward scale or lower the LR.

### The hovering failure mode (new with the assist force)

The assist supports trunk height, so a new way to game the reward is to
**float horizontally at moderate height** without uprighting. Tell-tale:
`mean_robot_z` rises but `rewards/upright_raw` stays low and
`foot_grounded_up` stays near 0. The orientation gates on the feet/tall
terms make this unrewarding, but if you see it, lower `assist_force_max`
(less free lift) — 100–130 N is a reasonable next try.

12. **PPO diagnostics** — `approx_kl` should stay under ~0.05, `clip_fraction` under ~0.3. For multi-critic runs the value fit is the per-group `explained_variance_{task,reg,success}` above rather than a single `explained_variance`.

---

## Hyperparameters worth tuning (and ones not to)

### Don't touch unless you have a strong reason

Three independent curricula tighten the success criteria from "reachable from a kneel" to "deployment quality" over ~25M env-steps. End values are the deployment criteria; `_start` values define what the policy is judged against early in training.

- `success_hold_steps=50` (1 s hold) — END value: defines what "stable" means.
- `success_hold_steps_start=15` (0.3 s) — START value: discoverable hold length for a fresh policy.
- `upright_threshold=0.92` (cos, ≈ 23° tilt) — END value: defines what "upright" means.
- `upright_threshold_start=0.80` (cos, ≈ 37° tilt) — START value: roughly the orientation the policy reaches in a kneel.
- `target_height=0.55` — END value: K1 standing trunk height. `frame_success` requires `z > target_h − 0.10`.
- `target_height_start=0.40` — START value: `frame_success` triggers at `z > 0.30`, just out of reach of the kneel attractor at z ≈ 0.25.
- `hold_curriculum_env_steps=25_000_000` and `threshold_curriculum_env_steps=25_000_000` — horizons over which each curriculum tightens. Lengthen if you have plenty of compute, shorten for faster convergence to full strictness.
- `time_to_stand_tau_steps=150` (3.0 s) — sets the shape of the terminal speed bonus. With τ=150 a 1 s stand pays ~330, 2 s ~150, 3 s ~100. The previous τ=40 decayed so fast that 3 s standups paid only ~9, less than the side-plank attractor.
- Settle-pool params (`spawn_height_*`, `settle_steps`, `settle_pool_rounds`) — already tuned to give ~120 diverse fallen states at `vec_num_envs=32` and ~3800 at 1024.

### Reverse curriculum on initial pose distribution

- `easy_pool_enabled=True` — toggles the second initial-pose pool.
- `start_curriculum_env_steps=25_000_000` — env-steps over which the easy-start fraction ramps from 1.0 → 0.0. Tune up (e.g. 50M) if you want the policy to spend more time learning "maintain standing" before recovering from harder falls.
- `easy_pool_height=0.60` — spawn height for the easy pool; let it settle to standing.
- `easy_pool_tilt_max=0.15` (rad, ~8°) — max initial trunk tilt.
- `easy_pool_joint_jitter=0.10` (rad) — joint perturbation around the default standing pose.
- `easy_pool_min_height=0.40` / `easy_pool_min_upright=0.70` — pool filter: only keep states that DIDN'T fall during the brief settle. Loosen if too many states get filtered out.

### Things you might reasonably tune
- `upright_progress` (default 5.0) — bump higher (8–10) if the policy is still getting stuck in side-plank-like local minima.
- `arm_pose_dev` (default 0.5) — bump up to 1.0 if the policy is still converging on a T-pose or wide-arm stance during the hold; reduce to 0.2 if the policy can't get up at all on hard starts (the arm penalty is interfering with recovery push-offs even though it's gated). You can also widen the `arm_gate` range to `[0.5, 0.85]` to give the policy more time to use arms before the penalty kicks in.
- `pool_max_upright=0.7` — lower (e.g. 0.5) to keep only HARDER fallen starts, harder (e.g. 0.85) for an easier curriculum.
- `entropy_coef` (default 0.005) — raise to 0.01 if the policy is committing to bad strategies too fast.

---

## Troubleshooting

### "The policy is stuck at side-plank (body horizontal but elevated, one arm propping it up)"

The `foot_grounded_up` and `standing_tall` rewards both include an `upright_factor = max(0, cos(tilt))` multiplier so side-plank earns only ~half of what a vertical stand would earn from these terms. If the policy still converges here, the gate may be too gentle for your DR / spawn distribution — switch to a sharper ramp:

```python
# In _upright_factor (skills/standup/rewards.py)
return np.clip((upright - 0.5) / 0.5, 0.0, 1.0).astype(np.float32)
```

This zeroes out the term entirely for any pose with cos(tilt) < 0.5 (~60° tilt) and gives full credit only when nearly vertical.

### "The policy is stuck at side-plank / lying still" (legacy diagnosis)

Check `rewards/upright_progress` AND `rewards/mean_robot_z`:
- If `upright_progress` rises early then **decays back toward 0** AND `mean_robot_z` plateaus below ~0.3 → the policy is stuck in a sit/kneel partial pose. The threshold curricula are designed to break this: at t=0 the policy can succeed at z>0.30, up>0.80 (reachable from a kneel). If you still see no `frame_success_rate` rise by ~5M env-steps, lower `target_height_start` to 0.30 or `upright_threshold_start` to 0.75.
- If `upright_progress` is near zero from the start → the policy never moves upward. Bump `upright_progress` further (to 15–20) or raise entropy to 0.01.
- If `frame_success_rate` rises but `achieved_sustained_rate` stays low → the policy is wobbling through the success window without sustaining. Lower `success_hold_steps_start` to 10 or lengthen `hold_curriculum_env_steps`.

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
