# standup-host-recipe — notes

Experimental branch off `standup-stable-pose` adopting the **HoST recipe**
inside our existing Genesis pipeline (no framework switch — HoST trains in
Isaac Gym and does *not* do sim2sim; their robustness is a reward+DR story that
ports cleanly, and we keep our MuJoCo sim2sim eval that HoST never had).

## What changed (vs standup-stable-pose)

| Knob | old | new | why |
|---|---|---|---|
| `use_multi_critic` | False | **True** | HoST multi-critic: one value head per reward group (task/reg/success). Already audited-correct. `--init-from` partial-loads the actor, starts 3 critics fresh. |
| `reward_stage` | discovery | **deploy** | reg (motion-quality) weights go live… |
| `reg_success_ramp` | — (new) | **True** | …but × `pose_scale = clip(success_ema/0.5, 0, 1)`, so reg is ~0 during the get-up (discovery stays free) and folds in only once it reliably stands. "Light style in one run." |
| `on_spot` | flat | **success-ramped** | no field-travel at the end, without taxing the big motion during discovery. |

The reg group = action smoothness + jerk + joint-velocity + base sway/drift +
arm-pose. Ramping it in is both the **deployability** lever (smooth, arms-to-
sides, implicit motion-speed bound) and the **sim2sim** lever (a gentler, lower-
impact motion exploits fewer Genesis-specific contact quirks → transfers).

## Deferred levers (need care — not applied)

- **Action-scale cut** (`ACTION_DELTA_MAX` 0.5→~0.35, HoST "action rescaler"):
  must be matched in the MuJoCo eval (`render_presentation._ACTION_DELTA_MAX`)
  or it breaks the audited gain/action-pipeline match. Wire as a config field +
  eval CLI arg together.
- **Knee/hip `kp` ×1.33–1.5** (HoST sim2real tip): `K1RobotConfig` is read by
  BOTH Genesis and the MuJoCo eval, so change it there to stay matched — but it
  also affects walk/dribble/shoot. Our kp are the NaoHTWK-authoritative K1
  values, so verify it doesn't destabilize.
- **Contact-solver DR** (stiffness/restitution randomization) + discrete-point
  foot/hand collisions: the most direct fix for the contact-overfit sim2sim gap,
  but needs a Genesis per-env contact-param API spike first.

## Run it

```bash
# warm-start the actor from the working (but non-transferring) get-up:
./scripts/run.sh train-skill standup --device gpu --vec-num-envs 1024 --wandb \
    --init-from checkpoints/skill_standup/skill_standup_step<LATEST>.pt
# or fresh (no --init-from) to let it find a different, maybe-more-robust basin.

# GATE on the MuJoCo sim2sim eval, NOT the Genesis z:
python scripts/eval_standup_sim2sim.py checkpoints/skill_standup --trials 3
```
