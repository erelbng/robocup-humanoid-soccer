# Walking skill — design, current state, how to train

This documents the `walk` skill as of the `walking-skill` branch: a
command-conditioned K1 locomotion policy trained with PPO, plus an
**Adversarial Motion Prior (AMP)** driven by **real human walking mocap
(LAFAN1) retargeted to the K1** for natural gait style.

## TL;DR current state

- **Confirmed working:** a **stable, upright, forward-locomoting** controller
  that transfers Genesis→MuJoCo without falling. Command vector
  `[vx, vy, vyaw, foot_clearance, step_freq]` (+ optional head look).
- **Open quality gap:** the AMP-only policy *glides* (feet slide rather than
  cleanly step). The **hybrid** config (AMP style + explicit foot-lift reward,
  below) is the current approach to force real stepping; it is still training
  / being tuned at the time of writing.
- The full **mocap→AMP pipeline is in-repo and reproducible** (see
  `scripts/retarget/`).

## Command + obs

- Command (5-dim): `vx, vy` (m/s, body frame), `vyaw` (rad/s), `foot_clearance`
  (m), `step_freq` (Hz). Optional head-look command (yaw/pitch) appended.
- Obs (85-dim): 78-dim shared base obs (`skills/common_obs`) + 5 command + 2
  head command.

## Reward (`skills/walk/rewards.py`, weights in `config.py`)

Hard-won lessons are baked into the weights — see the inline comments for the
full history. Key terms:

- **`track_lin_vel` (2.5) + `forward_progress` (2.0, linear) + `track_ang_vel`
  (0.5)** — velocity tracking. `forward_progress` is a *linear* fraction-of-
  commanded-speed reward with a constant gradient; it was added because the
  exp-shaped `track_lin_vel` gradient vanishes far from target, leaving the
  policy marching in place. With it the robot actually translates.
- **Posture as PENALTIES** — `upright` (5) and `height` (20) are applied as
  `(up−1)` / `(h−1)` (≤0), i.e. standing earns ZERO, not a bonus. This was the
  fix for a stand-still local optimum: when posture was a positive bonus the
  policy collected ~+2/step doing nothing and never risked a step. Booster's
  T1 recipe uses the same penalty form.
- **`feet_swing` (3.0)** — Booster T1's anti-shuffle term: +1 per foot airborne
  during its clock-phased swing window. **`swing_height` (2.0)** — dense
  phase-conditioned foot-LIFT gradient (rewards raising the swing foot even
  while planted, so there's a gradient to *initiate* a lift). **`feet_slip`
  (0.3)** — anti-skate penalty on planted-foot horizontal speed.
- **`arm_swing` (0.2)** + relaxed `arm_pose` — a little natural arm swing.

Critical gotcha (documented in code): the foot **contact threshold** must match
the *measured* Genesis foot-link standing height (~0.065 m); an earlier 0.04 m
threshold made contact never trigger, silently disabling the gait/slip rewards.

## AMP (`training/algorithms/amp.py`)

- **Reference:** real LAFAN1 human walking, retargeted to K1 via **GMR**
  (`scripts/retarget/`), converted to a 53-dim AMP feature stream:
  `[root_height, projected_gravity(3), body angular vel(3), joint pos(22),
  joint vel(22), per-foot clearance(2)]`. Root *linear* velocity is excluded
  (speed is the task's job, not style). **Per-foot clearance is included** so
  the discriminator can tell gliding from stepping.
- **Discriminator:** LSGAN on `(s, s')` transitions + zero-centred gradient
  penalty. GAN balance is delicate here (too strong → `disc_acc`→1.0 saturated,
  no gradient; too weak → 0.5 random, no gradient).
- **Style reward** blended with the task reward: `r = task_coef·r_task +
  style_coef·r_style`.

### Hybrid (current recommended config)

Pure AMP gave a stable glider (the discriminator couldn't, on its own, force
foot-lift within the workable GAN-balance window). The hybrid keeps AMP for
style but lets the **explicit foot-lift task reward force stepping**:

```
--amp --amp-motion-file data/motions/k1_lafan1_walk_amp.npz \
--amp-task-coef 0.6 --amp-style-coef 0.4
```

(task-dominant: `feet_swing`/`swing_height` force the feet up; AMP style keeps
it natural). `arm_pose` is auto-zeroed under AMP so arms can match the
reference's swing.

## PPO correctness fixes (`training/algorithms/ppo.py`, also PR'd separately)

Six episode-boundary/PPO bugs were fixed and are relied on by every run: GAE
terminal-mask off-by-one; auto-reset returning the terminal (not reset) obs;
timeouts treated as true terminals (now truncation-bootstrapped); a dead
`entropy_coef` (std was overwritten by a wall-clock schedule); multi-critic
advantage not re-standardised; and a mislogged `mean_robot_z`. See
`skills/base.py` `step()` (now returns `terminated`/`truncated`/`terminal_obs`).

## How to train / evaluate

```bash
# plain PPO (reward-shaped) walk
python -m training.train_skill --skill walk --device gpu --vec-num-envs 1024

# hybrid AMP + mocap (recommended for natural gait)
python -m training.train_skill --skill walk --amp \
    --amp-motion-file data/motions/k1_lafan1_walk_amp.npz \
    --amp-task-coef 0.6 --amp-style-coef 0.4 --device gpu --vec-num-envs 1024

# MuJoCo eval + video (renders to the given path; prints a gait diagnosis)
python -m evaluation.eval_walk_mujoco CKPT.pt --vx 0.4 --record-video \
    --video-path ~/walk_eval.mp4 --strip-path /tmp/strip.png
```

Regenerating the AMP reference from LAFAN1 mocap: see `scripts/retarget/README.md`.
