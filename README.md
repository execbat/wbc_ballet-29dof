# Ballet + Flip — полный репозиторий

Оба таска зарегистрированы напрямую в `src/wbc_ballet/tasks/__init__.py`, одним и
тем же `register_mjlab_task(...)`. Второй таск назывался `break`; переименован в
`flip`, чтобы `wbc_ballet.tasks.flip` подключался обычным импортом, без обходного
`importlib.import_module(...)`, необходимого из-за того, что `break` — зарезервированное
слово Python. Исправление yaw в ballet уже применено напрямую в
`wbc_ballet/tasks/ballet/mdp/`.

Структура модульная и одинаковая для обоих тасков: один cfg-файл на менеджер
(`*_commands_cfg.py`, `*_events_cfg.py`, ...) плюс сборочный `*_env_cfg.py`, а
реализация command/event/reward-термов — в собственной папке `mdp/` каждого таска
(`tasks/ballet/mdp/`, `tasks/flip/mdp/`), а не в общей папке верхнего уровня.
Подробнее — в разделе [Structured task configuration](#structured-task-configuration).

```bash
uv run train Mjlab-Ballet-Flat-Unitree-G1-29DoF
uv run train Mjlab-Flip-Flat-Unitree-G1-29DoF
```

Это альтернативные команды запуска обучения. Rough-вариант ballet также сохранён.

Геймпады:

```bash
uv run python gamepad/game_emulator_run_v1.py
uv run python gamepad/game_emulator_run_v2.py
```

v1 — для ballet (порт 55001); v2 — для flip, с чекбоксом flip-режима (порт 55002).
Подробности flip: [src/wbc_ballet/tasks/flip/README.md](src/wbc_ballet/tasks/flip/README.md).

---

# wbc_ballet

Train Unitree G1 (29 DoF) to combine locomotion with masked whole-body pose
commands. The simulator is MJLab (MuJoCo + Warp), and PPO is provided by
`rsl_rl`. Isaac Lab and Isaac Sim are not dependencies.

Deployment repository is here: [wbc_ballet-29dof_deploy](https://github.com/execbat/wbc_ballet-29dof_deploy)


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
axis_target_normalized, axis_mask`.

The pelvis-mounted MuJoCo gyro and velocimeter are used by `base_ang_vel` and
`base_lin_vel`. `imu_lin_acc` is the physical accelerometer channel.
`projected_gravity` stays in the actor because it can be reconstructed from the
robot IMU/state estimator at deployment time.

The critic inherits all actor observations and additionally receives
`whole_body_com_xy`, `support_center_xy`, `foot_height`, `foot_air_time`,
`foot_contact`, and `foot_contact_forces`. `whole_body_com_xy` is the ground
projection of the complete articulated robot's center of mass relative to the
floating base, expressed in the base yaw frame, and is intentionally privileged
(critic-only). `support_center_xy` is the mask-aware center of the foot support
points in the same base-relative XY frame. If exactly one leg has any active
joint mask, only the opposite foot is support; if neither or both legs have
active masks, the midpoint of both feet is used.

`axis_target_normalized` is hard-gated by the binary mask. Consequently an
inactive joint always contributes exactly zero to the target observation,
even if an old or malformed sender puts a nonzero value in that UDP field.

The actor is now 186D and the critic is 205D. Checkpoints trained with the
previous observation ABI are not shape-compatible; train a new policy from
scratch after this change.


## Structured task configuration

Both tasks follow the `microduck_rl/feat/structured_cfg` pattern. Manager terms
are declarative `@configclass` fields and are converted to MJLab's native
dictionaries only by `<Task>EnvCfg.to_mjlab_cfg()`. Each task owns two things:
one `*_cfg.py` file per manager (the declarative wiring) and its own `mdp/`
subpackage (the term *implementations* those cfg files reference) — nothing
task-specific lives in a shared top-level location.

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
├── ballet_terminations_cfg.py
└── mdp/                         # command/event/reward/curriculum/metric/
    ├── __init__.py               # termination term implementations
    ├── commands.py
    ├── curriculums.py
    ├── events.py
    ├── metrics.py
    ├── observations.py
    ├── rewards.py
    └── terminations.py
```

(`torque_envelope.py`, previously also in this folder, moved to
`wbc_ballet/robots/g1/` instead — it's generic actuator torque/speed-limit
math used by the robot's actuator model, not a ballet-specific mdp term, and
keeping it here would have made the shared robot definition depend on a
task package.)

An inherited term can be overridden by redeclaring the same field, or disabled
with `term_name: TermCfg | None = None`. `configclass` deep-copies mutable term
defaults, so changing a play/rough instance cannot mutate a training/flat one.

The flip task lives next to ballet and follows the identical layout without a
Rough variant. It overrides actions and terminations and inherits metrics:

```text
src/wbc_ballet/tasks/flip/
├── __init__.py
├── flip_actions_cfg.py
├── flip_commands_cfg.py
├── flip_curriculum_cfg.py
├── flip_env_cfg.py             # assembly only
├── flip_events_cfg.py
├── flip_observations_cfg.py
├── flip_rewards_cfg.py
├── flip_rl_cfg.py
├── flip_terminations_cfg.py
└── mdp/                         # command/event/reward term implementations,
    ├── __init__.py               # only for this task's own terms
    ├── commands.py
    ├── events.py
    ├── rewards.py
    └── terminations.py
```

`wbc_ballet/teleop/flip_protocol.py` holds flip's UDP wire format, sibling of
`teleop/protocol.py` (ballet's). Both tasks are registered the same way, with
plain `register_mjlab_task` calls in `tasks/__init__.py` -- the task used to
be called `break`, which forced an `importlib.import_module` workaround since
`break` is a reserved Python keyword; renaming it to `flip` removed the need
for that entirely. See [src/wbc_ballet/tasks/flip/README.md](src/wbc_ballet/tasks/flip/README.md)
for the task's own details, including how its command classes are named to
avoid colliding with the boolean flip-mode flag they share a name with.

  
# Launch scene via MuJoCO viewer
```text  
uv run python -m mujoco.viewer \
  --mjcf=src/wbc_ballet/robots/g1/xmls/scene_g1.xml  
```  
  
# Launch trainng
```text
uv run train Mjlab-Flip-Flat-Unitree-G1-29DoF \
  --env.scene.num-envs 4096 \
  --agent.save-interval 100 \
  --agent.logger tensorboard  
```  
  
# Launch Play 
```text
uv run play Mjlab-Flip-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./logs/rsl_rl/g1_29dof_flip/2026-09-12_00-01-12_flip/model_25200.pt \
  --viewer native \
  --num-envs 1
```

# Launch Command Window
```text
uv run python ./gamepad/game_emulator_run_v2.py
```

# EXPORT INTO ONNX
```text
uv run python scripts/export.py \
  Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./logs/rsl_rl/g1_29dof_flip/2026-09-10_17-34-17_flip/model_2000.pt \
  --onnx-file g1_29dof_wbc_ballet.onnx
```  

# RUN TENSORBOARD
```text
uv run tensorboard \
  --logdir logs/rsl_rl \
  --port 6006
```




# Launch Play Ballet
```text
uv run play Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./model_21500.pt \
  --viewer native \
  --num-envs 1
```

# EXPORT Ballet INTO ONNX
```text
uv run python scripts/export.py \
  Mjlab-Ballet-Flat-Unitree-G1-29DoF \
  --checkpoint-file ./model_21500.pt \
  --onnx-file policy.onnx
```  
