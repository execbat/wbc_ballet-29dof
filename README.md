# wbc_ballet

Train Unitree G1 (29 DoF) to combine locomotion with masked whole-body pose
commands. The simulator is MJLab (MuJoCo + Warp), and PPO is provided by
`rsl_rl`. Isaac Lab and Isaac Sim are not dependencies.


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
```text
uv run python scripts/export.py \
  Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./logs/rsl_rl/g1_29dof_ballet/2026-09-05_23-52-26_ballet/model_15500.pt \
  --onnx-file g1_29dof_wbc_ballet.onnx
```  

# RUN TENSORBOARD
```text
uv run tensorboard \
  --logdir logs/rsl_rl \
  --port 6006
```
