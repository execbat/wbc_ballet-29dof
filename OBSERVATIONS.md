# Actor and Critic Observations

This document describes the observation ABI for the 29-DoF ballet policy.
The order of terms matters because the concatenated observation vector is the
input ABI of the neural network.

## Actor observations — 189D

The actor receives only observations intended to be available or reconstructable
on the real robot.

| # | Observation | Dim | Description |
|---:|---|---:|---|
| 1 | `base_ang_vel` | 3 | Pelvis/base angular velocity from the IMU gyroscope: `[wx, wy, wz]`. |
| 2 | `imu_lin_acc` | 3 | Pelvis IMU linear acceleration: `[ax, ay, az]`. |
| 3 | `projected_gravity` | 3 | Gravity direction projected into the robot base frame; reconstructable from IMU orientation/state estimation. |
| 4 | `velocity_commands` | 3 | Commanded planar motion: `[vx_cmd, vy_cmd, yaw_rate_cmd]`. |
| 5 | `joint_pos` | 29 | Joint positions relative to the default/home configuration. |
| 6 | `joint_vel` | 29 | Joint velocities. |
| 7 | `actions` | 29 | Previous policy action. |
| 8 | `axis_actual_normalized` | 29 | Current joint positions normalized to `[-1, 1]`. |
| 9 | `axis_target_normalized` | 29 | Ballet target joint positions normalized to `[-1, 1]`; hard-gated by `axis_mask`. |
| 10 | `axis_mask` | 29 | Binary mask indicating which joints are actively commanded by the ballet target. |
|  | **Total** | **186** | |

Dimension check:

```text
3 + 3 + 3 + 3 + 29 + 29 + 29 + 29 + 29 + 29 = 186
```

`whole_body_com_xy` is deliberately **not** included in the actor observation.

## Critic observations — 205D

The critic receives all **189D actor observations** plus **16D privileged
simulation observations** used only during training.

### Actor observations inherited by the critic — 189D

1. `base_ang_vel` — 3
2. `imu_lin_acc` — 3
3. `projected_gravity` — 3
4. `velocity_commands` — 3
5. `joint_pos` — 29
6. `joint_vel` — 29
7. `actions` — 29
8. `axis_actual_normalized` — 29
9. `axis_target_normalized` — 29
10. `axis_mask` — 29

### Additional privileged critic observations — 16D

| # | Observation | Dim | Description |
|---:|---|---:|---|
| 12 | `whole_body_com_xy` | 2 | Whole-body center-of-mass XY position relative to the floating base, expressed in the base yaw frame. Computed from the full articulated simulation state. |
| 13 | `support_center_xy` | 2 | Mask-aware center of the active foot support region relative to the base. |
| 14 | `foot_height` | 2 | Left/right foot height. |
| 15 | `foot_air_time` | 2 | Left/right foot air time. |
| 16 | `foot_contact` | 2 | Left/right foot contact state. |
| 17 | `foot_contact_forces` | 6 | 3D contact force for each foot: `2 x [Fx, Fy, Fz]`. |
|  | **Privileged subtotal** | **16** | |
|  | **Critic total** | **202** | |

Dimension check:

```text
186 + 2 + 2 + 2 + 2 + 2 + 6 = 202
```

## Final ABI

```text
Actor:  189D
Critic: 205D
```

`projected_gravity` is present in both actor and critic because the critic
inherits the actor observation group. `whole_body_com_xy` is critic-only.
