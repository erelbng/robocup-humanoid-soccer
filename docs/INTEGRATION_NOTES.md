# Integration Notes: NaoHTWK Reference Repositories

This file consolidates what we want to borrow from the two upstream
HTWK projects as this RL setup matures. Treat it as a living planning
doc — point at, copy from, or replace as needed.

## 1. `NaoHTWK/htwk-gym` — Phase 1 walking reference

URL: https://github.com/NaoHTWK/htwk-gym

**What it is:** Isaac/MuJoCo-style RL gym for humanoid locomotion that
already targets the Booster K1 (and the older T1) platform. Used by the
team for training base walking + parameterized walking + kicking
policies that are then deployed to the real robot.

**Relevant directories:**
| Path | Use for us |
| --- | --- |
| `envs/` | Per-task env definitions — mirror in our `envs/` (BaseWalk → curriculum stage "walk", ParameterWalk → "dribble", Kicking → "shoot") |
| `deploy/models/*.pt` | Reference checkpoints we can warm-start Phase 1 from |
| `resources/` | URDF + physical specs — sanity-check ours against theirs (e.g. inertias, joint limits) |

**Task / observation breakdown:**
| Task | Obs dim | Action dim | Command dim | Notes |
| --- | --- | --- | --- | --- |
| BaseWalk | 47 | 12 (legs only) | 3 (vx, vy, ω) | velocity tracking |
| ParameterWalk | 54 | 12 | 10 (vel + gait freq + foot yaw + body pitch/roll + feet offsets) | "fine-grained control" |
| Kicking | 44 | 12 | 0 | target-based ball velocity reward |

**Implications for our code:**
- Our `Phase1Config.act_dim = 22` (all DoFs) is much wider than htwk-gym's
  12 (legs only). For early curriculum (`stand`, `walk`), we should mask
  arm/head action outputs or train a leg-only policy first.
- Their command vector model (velocity + gait params) is a clean way to
  expose "go faster" / "lean left" without retraining. We should add a
  `command_dim` slot to `Phase1Config` once `walk` works end-to-end.
- Their "kicking" task already uses ball-velocity rewards similar to our
  `kick_reward` + `ball_to_goal`. Their reward weighting is worth
  cross-checking once we have stand → walk → kick working.

## 2. `NaoHTWK/HTWKVision` — perception for matches + digital twin

URL: https://github.com/NaoHTWK/HTWKVision

**What it is:** Production C++ vision stack used on the Naos in matches.
Mix of classical CV + TensorFlow Lite classifiers.

**Modules to mirror in sim (Phase 2 / digital twin):**
| File / module | Domain | Sim2real bridge |
| --- | --- | --- |
| `ball_detector` + `ball_classifier_upper_cam` | Ball localisation | Render Genesis ball through the policy's camera; train a sim2real distillation that matches their classifier outputs |
| `field_color_detector` / `field_border_detector` | Field segmentation | Useful for domain-randomised field rendering — we should jitter green hues during training (already wired in `envs/domain_randomization.py`) |
| `line_detector` + `lc_centercirclepoints_detector` | Self-localisation | Our generated field already has correct line geometry; perfect ground-truth for training a line-detection head |
| `object_detector_lowercam` + `jersey_detection` + `lc_obstacle_detection` | Other robots | For Phase 2 multi-robot scenes, expose a "render-from-camera-pose" hook and treat detected boxes as input observations |

**Implications for our code:**
- Add an off-screen camera per robot in Phase 2 (we already render one
  global camera for video; each robot needs its own). Genesis's
  `scene.add_camera(parent=robot_link)` is the path.
- For the digital twin, expose calibrated camera intrinsics + extrinsics
  matching the real K1 head camera. The MJCF should grow a `<camera>`
  attached to the head link.
- Their classifiers are TFLite — we can run them on synthetic frames
  during eval to measure sim2real detection gap as a metric.

## 3. Open follow-ups

- [ ] Mirror htwk-gym's BaseWalk observation layout into our `walk`
      curriculum stage so the policy can warm-start from `base_walk.pt`.
- [ ] Add per-robot head camera entities in `envs/phase2_match.py` so we
      can later plug HTWKVision detectors in for the digital twin.
- [ ] Texture-domain randomisation hook in
      `envs/domain_randomization.py` is wired but disabled — once we
      have CI rendering at a fixed seed, flip `swap_textures=True` and
      verify it doesn't break training stability.
