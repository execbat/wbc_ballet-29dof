"""Reward manager terms ported from the Orbit ballet task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from wbc_ballet import mdp as ballet_mdp
from wbc_ballet.utils.configclass import configclass


@configclass
class BalletRewardsCfg:
    track_linear_velocity: RewTerm | None = RewTerm(
        func=ballet_mdp.track_ballet_linear_velocity,
        weight=2.0,
        params={"std": math.sqrt(0.25)},
    )
    track_angular_velocity: RewTerm | None = RewTerm(
        func=ballet_mdp.track_ballet_angular_velocity,
        weight=2.0,
        params={"std": math.sqrt(0.5)},
    )
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        },
    )
    masked_pose_tracking: RewTerm | None = RewTerm(
        func=ballet_mdp.masked_pose_tracking,
        weight=2.5,
        params={"std": 0.35, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    unmasked_home_tracking: RewTerm | None = RewTerm(
        func=ballet_mdp.unmasked_home_tracking,
        weight=0.5,
        params={
            "std": 0.4,
            # Do not pull unmasked leg joints toward home: the locomotion
            # policy needs them for its gait. Stabilize only waist and arms.
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(r"waist_.*", r".*_(shoulder|elbow|wrist)_.*"),
            ),
        },
    )
    # Human-like locomotion priors. A mask anywhere on one leg disables the
    # leg-specific prior for that leg, while the other leg remains regularized.
    pelvis_height_tracking: RewTerm | None = RewTerm(
        func=ballet_mdp.pelvis_height_tracking,
        weight=1.0,
        params={
            "target_height": 0.8,
            "std": 0.18,
            "left_leg_cfg": SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
            "right_leg_cfg": SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
        },
    )
    leg_lateral_alignment: RewTerm | None = RewTerm(
        func=ballet_mdp.unmasked_leg_lateral_alignment,
        weight=0.75,
        params={
            "std": 0.35,
            "left_leg_cfg": SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
            "right_leg_cfg": SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
            "left_alignment_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_ankle_roll_joint",
                ),
                preserve_order=True,
            ),
            "right_alignment_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_ankle_roll_joint",
                ),
                preserve_order=True,
            ),
        },
    )
    foot_heading_alignment: RewTerm | None = RewTerm(
        func=ballet_mdp.unmasked_foot_heading_alignment,
        weight=0.35,
        params={
            "std": 0.45,
            "turn_rate_threshold": 0.15,
            "feet_cfg": SceneEntityCfg(
                "robot",
                site_names=("left_foot", "right_foot"),
                preserve_order=True,
            ),
            "left_leg_cfg": SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
            "right_leg_cfg": SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
        },
    )
    com_support_projection: RewTerm | None = RewTerm(
        func=ballet_mdp.com_support_projection_tracking,
        weight=1.0,
        params={
            "std": 0.12,
            "feet_cfg": SceneEntityCfg(
                "robot",
                site_names=("left_foot", "right_foot"),
                preserve_order=True,
            ),
            "left_leg_cfg": SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
            "right_leg_cfg": SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
        },
    )
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    )
    # Whole-body angular momentum conflicts with deliberate arm/waist ballet
    # targets, so it is not part of this task's objective.
    angular_momentum: RewTerm | None = None
    dof_pos_limits: RewTerm | None = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # action_rate_l2 sums over all 29 actions; at -0.1 it can contribute
    # several negative reward units during transitions and suppress movement.
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = RewTerm(
        func=mdp.feet_clearance,
        weight=-1.0,
        params={
            "target_height": 0.1,
            "height_sensor_name": "foot_height_scan",
            "command_name": "ballet",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )
    foot_swing_height: RewTerm | None = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "ballet",
            "command_threshold": 0.05,
        },
    )
    foot_slip: RewTerm | None = RewTerm(
        func=mdp.feet_slip,
        weight=-0.2,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "ballet",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )
    soft_landing: RewTerm | None = RewTerm(
        func=mdp.soft_landing,
        weight=-1.0e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "ballet",
            "command_threshold": 0.05,
        },
    )
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost,
        weight=-0.25,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )
    commanded_leg_ground_contact: RewTerm | None = RewTerm(
        func=ballet_mdp.commanded_leg_ground_contact,
        weight=-1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "left_leg_cfg": SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
            "right_leg_cfg": SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
        },
    )
    fell_over_penalty: RewTerm | None = RewTerm(
        func=ballet_mdp.fell_over_penalty,
        weight=-100.0,
    )
    non_finite_state_penalty: RewTerm | None = RewTerm(
        func=ballet_mdp.non_finite_state_penalty,
        weight=-100.0,
    )
