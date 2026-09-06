# wbc_ballet

Train Unitree G1 (29 DoF) to combine locomotion with masked whole-body pose
commands. The simulator is MJLab (MuJoCo + Warp), and PPO is provided by
`rsl_rl`. Isaac Lab and Isaac Sim are not dependencies.

The task ports the command/policy contract from `execbat/Orbit`. The complete
G1 model needed by this repository is copied into `src/wbc_ballet/robots/g1`:
both MJCF variants, their scene files, every STL mesh, the actuator model,
torque-speed envelope, home pose, collision configuration, and MJLab
articulation. Runtime does not load the robot from the `wbc-mjlab` package.

## Install

```bash
uv sync
```

The lockfile pins the CUDA 12.8 PyTorch build used by this project.

## Train

```bash
uv run train Mjlab-Ballet-Flat-Unitree-G1-29DoF
```

Minimal smoke run:

```bash
uv run train Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --env.scene.num-envs 1 \
  --agent.max-iterations 5
```

### NaN/Inf protection

The task rejects non-finite values at every environment boundary used by PPO:

- a physics state or raw policy action containing NaN/Inf terminates and resets
  only the affected environment;
- actor and critic observations are checked after concatenation, reported in
  the console, and sanitized before they are returned to `rsl_rl`;
- MJLab 1.5.3 sanitizes every weighted reward term before accumulating it.

Numerical failures are logged independently from falls and timeouts. Look for
`Episode_Termination/non_finite_state` and the `Episode_Metrics/nan_*`,
`Episode_Metrics/inf_*`, and `Episode_Metrics/nonfinite_*` series in
TensorBoard. The per-component series distinguish `qpos`, `qvel`, `qacc`,
`qacc_warmstart`, sensor data, and policy actions.

For a short diagnostic run, enable MJLab's rolling state dump:

```bash
uv run train Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --env.scene.num-envs 512 \
  --enable-nan-guard True
```

The first detected failure writes the preceding 32 physics states and the
compiled model to `logs/nan_dumps`. Inspect it with:

```bash
uv run viz-nan logs/nan_dumps/nan_dump_latest.npz
```

The rolling dump is deliberately opt-in because it synchronizes the GPU every
physics substep. Episode metrics and automatic resets remain enabled in normal
4096-environment training. Do not resume from a checkpoint whose actor, critic,
or optimizer parameters already contain NaN; start from the last finite
checkpoint after installing this protection.

## UDP play environment

The registered play configuration listens on `127.0.0.1:55001` and consumes
the same packet shape as Orbit, sized for G1-29DoF: 61 little-endian
`float32` values:

1. 29 normalized joint targets in `[-1, 1]`;
2. 29 enable-mask values (values `>= 0.5` become one);
3. `[vx, vy, yaw_rate]`.

Joint axis order (both directions) is the robot's canonical joint order --
the MJCF declaration order, identical to the order used by the
`joint_pos`/`axis_actual_normalized` observations and by the action space
(see `tests/test_g1_asset.py::test_model_joint_order_matches_canonical_order`
for the locked, verified order). Axis *i* means the same joint everywhere in
the stack: the model, the articulation/action space, the UDP packet, the
observation, and the reward.

Start the policy using MJLab's `play` command and send a test command from a
second terminal:

```bash
uv run play Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file /path/to/model_650.pt \
  --viewer native \
  --num-envs 1
uv run wbc-ballet-teleop --joint 15 --target 0.4 --vx 0.2
```

Malformed or non-finite UDP packets are ignored. The last valid command stays
active, matching Orbit's non-blocking last-value behavior.

Only the `play` configuration prints the joint-observation table. Every five
seconds of simulation it prints environment 0 in canonical 29-joint order:
the actual signed angle in radians (`q_actual`), model HOME value (`q_home`),
their difference (`delta`), normalized value in `[-1, 1]` (`q_norm`), binary
mask, and normalized target actually visible to the policy (`target_obs`).
Training does not install this event, so terminal I/O cannot reduce
4096-environment training throughput.

## Policy observation ABI

The actor term order is intentionally fixed:

`base_lin_vel, base_ang_vel, imu_lin_acc, projected_gravity,
velocity_commands, joint_pos, joint_vel, actions, axis_actual_normalized,
axis_target_normalized, axis_mask, whole_body_com_xy`.

The pelvis-mounted MuJoCo gyro and velocimeter were already used by
`base_ang_vel` and `base_lin_vel`. `imu_lin_acc` adds the physical
accelerometer channel. `whole_body_com_xy` is the ground projection of the
complete articulated robot's center of mass relative to the floating base,
expressed in the base yaw frame.

The critic inherits all actor observations and additionally receives
`support_center_xy`. It is the mask-aware center of the foot support points in
the same base-relative XY frame. If exactly one leg has any active joint mask,
only the opposite foot is support; if neither or both legs have active masks,
the midpoint of both feet is used.

`axis_target_normalized` is hard-gated by the binary mask. Consequently an
inactive joint always contributes exactly zero to the target observation,
even if an old or malformed sender puts a nonzero value in that UDP field.

The actor is now 191D and the critic is 205D. Checkpoints trained with the
previous observation ABI are not shape-compatible; train a new policy from
scratch after this change.

## Training curriculum

Training starts with a locomotion-only phase. For the first 300 PPO iterations
every mask is zero and `target_scale` is zero. From iteration 300 through
5,300, `mask_probability` grows linearly from `0.0` to `0.15` while
`target_scale` grows linearly from `0.0` to `0.8`. After that both stay at
their final values (4.35 active joints on average). The schedule values are
defined at the top of `ballet_curriculum_cfg.py`.

The locomotion reward now contains mask-aware human-posture priors: standing
pelvis height, sagittal-plane hip/ankle alignment, forward foot heading during
near-straight motion, and whole-body CoM alignment with the support-center
projection. A mask anywhere on one leg disables that leg's posture prior;
pitch joints remain free for gait. Safety and CoM/support terms remain active.
If any axis of a leg is masked, that whole leg is treated as commanded and
contact between its foot and the ground incurs a per-leg penalty. No leg mask
means no such penalty, so normal locomotion support is unaffected during the
walking-only curriculum phase.

`target_scale` controls how far the sampled command moves from the robot's
current normalized pose toward a full-range random goal. At curriculum
progress `p`, `target = lerp(current, Uniform(-0.8, 0.8), p)`. Thus scale zero
creates no initial pose error, scale `0.4` applies half of the displacement,
and scale `0.8` applies the full goal. Here `-1` and `+1` correspond to the
lower and upper joint limits. This changes target difficulty, not motor speed
or policy action scale.

Training episodes last 24 seconds. The complete 61D training command is
resampled at a random interval of 6–10 seconds, so the policy experiences
multiple command changes within a successful episode. UDP play keeps its
effectively infinite manager resampling interval because incoming packets are
polled every policy step instead.

The complete reward/penalty audit, new weights, and rationale are documented
in [`REWARD_DESIGN.md`](REWARD_DESIGN.md).

## Structured task configuration

The ballet task follows the `microduck_rl/feat/structured_cfg` pattern. Manager
terms are declarative `@configclass` fields and are converted to MJLab's native
dictionaries only by `BalletEnvCfg.to_mjlab_cfg()`:

```text
src/wbc_ballet/tasks/ballet/
├── ballet_actions_cfg.py
├── ballet_commands_cfg.py
├── ballet_curriculum_cfg.py
├── ballet_env_cfg.py          # assembly only
├── ballet_events_cfg.py
├── ballet_metrics_cfg.py
├── ballet_observations_cfg.py
├── ballet_rewards_cfg.py
├── ballet_rl_cfg.py
├── ballet_scene_cfg.py
└── ballet_terminations_cfg.py
```

An inherited term can be overridden by redeclaring the same field, or disabled
with `term_name: TermCfg | None = None`. `configclass` deep-copies mutable term
defaults, so changing a play/rough instance cannot mutate a training/flat one.

## Provenance

- Environment semantics: `execbat/Orbit` at `98e2b749b5f025079f3529b7cff568f2f8eca274`.
- Packaging pattern: `execbat/microduck_rl`, branch `feat/structured_cfg`, at
  `373ecaa1512ef5e939d5daf6c586ca4338dd1977`.
- G1 asset/model: `wbc-mjlab/wbc-mjlab` 0.0.5, source revision
  `9d3255828088839ab087f96ac64fd2a8b10c9343`.

## 23DoF -> 29DoF port

The task, protocol, and teleop layers now target the full G1-29DoF model
(`robots/g1/xmls/g1.xml`) instead of the 23DoF variant
(`robots/g1/xmls/g1_23dof.xml`). The 29DoF model adds `waist_roll`/
`waist_pitch` and `wrist_pitch`/`wrist_yaw` (both arms) on top of the 23DoF
joint set. `robots/g1/actuators.py`'s articulation groups already targeted
these joints before this port (`G1_ACTUATOR_WAIST`, `G1_ACTUATOR_4010`) --
loading them against the 23DoF model raised `ValueError: Not all regular
expressions are matched!` at entity-construction time, since those joints
don't exist there. That mismatch is what made this port necessary.

Every axis-ordered surface (`mdp/commands.py`'s 61D command tensor,
`mdp/observations.py`'s target/mask slices, `teleop/protocol.py`'s UDP
packet, `gamepad/game_emulator_run_v1.py`'s sliders) now uses the SAME
29-joint order: the robot's canonical MJCF declaration order (locked by
`tests/test_g1_asset.py::test_model_joint_order_matches_canonical_order`).
This is also the order mjlab resolves observations/actions into internally
(an unfiltered `SceneEntityCfg`/`JointPositionActionCfg(actuator_names=
(".*",))` both resolve to natural joint order, not actuator-group order or
pattern-list order -- verified against mjlab 1.5.3's source).

**Bug found and fixed along the way**: the pre-port `gamepad/
game_emulator_run_v1.py`'s `JOINT_NAMES` list used Orbit's own axis order
(interleaved left/right, legs mixed with arms), which did not match the
order the trained policy actually used. That mismatch meant the tool's
slider labels did not correspond to the axes they actually drove. Fixed as
part of this port; the corrected order is documented in that file.

The 23DoF model/loader (`robots/g1/xmls/g1_23dof.xml`,
`assets/g1_23dof.py::get_g1_23dof_robot_cfg`) are left in the repository as
reference and are still covered by a structural test, but are no longer
used by the active ballet task or its registered gym task IDs.
  
# Launch scene via MuJoCO viewer
```text  
uv run python -m mujoco.viewer \
  --mjcf=src/wbc_ballet/robots/g1/xmls/scene_g1.xml  
```  
  
# Launch trainng
```text
uv run train Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --env.scene.num-envs 4096 \
  --agent.save-interval 500 \
  --agent.logger tensorboard  
```  
  
# Launch Play 
```text
uv run play Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./logs/rsl_rl/g1_29dof_ballet/2026-09-05_23-52-26_ballet/model_15500.pt \
  --viewer native \
  --num-envs 1
```

# Launch Command Window
```text
uv run python ./gamepad/game_emulator_run_v1.py
```

# EXPORT INTO ONNX
uv run python scripts/export.py \
  Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./logs/rsl_rl/g1_29dof_ballet/2026-09-05_23-52-26_ballet/model_15500.pt \
  --onnx-file g1_29dof_wbc_ballet.onnx

# RUN TENSORBOARD
```text
uv run tensorboard \
  --logdir logs/rsl_rl \
  --port 6006
```
