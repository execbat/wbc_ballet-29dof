"""Reward manager terms: flip-mode-aware overrides of the ballet rewards,
plus new terms for the hand-support (inverted/handstand) half of the task.
"""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wbc_ballet.utils.configclass import configclass
from mjlab.tasks.velocity import mdp as velocity_mdp

from . import mdp as flip_mdp

_FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
_HANDS = SceneEntityCfg("robot", site_names=("left_palm", "right_palm"), preserve_order=True)
_LEGS = (
    SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)),
    SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)),
)
_ARMS = (
    SceneEntityCfg("robot", joint_names=(r"left_(shoulder|elbow|wrist)_.*",)),
    SceneEntityCfg("robot", joint_names=(r"right_(shoulder|elbow|wrist)_.*",)),
)
_WAIST = SceneEntityCfg(
    "robot",
    joint_names=("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
    preserve_order=True,
)
_SUPPORT_PARAMS = {
    "feet_cfg": _FEET,
    "hands_cfg": _HANDS,
    "left_leg_cfg": _LEGS[0],
    "right_leg_cfg": _LEGS[1],
    "left_arm_cfg": _ARMS[0],
    "right_arm_cfg": _ARMS[1],
}


@configclass
class FlipRewardsCfg:
    masked_pose_tracking: RewTerm | None = RewTerm(
        func=flip_mdp.masked_pose_tracking,
        weight=2.5,
        params={"std": .35, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    # --- same params as ballet, flip-mode-aware function -----------------
    track_linear_velocity: RewTerm | None = RewTerm(
        func=flip_mdp.track_linear_velocity,
        weight=3.0,
        params={"std": 0.22},
    )
    linear_velocity_error_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.linear_velocity_error_penalty,
        weight=-1.5,
    )
    track_angular_velocity: RewTerm | None = RewTerm(
        func=flip_mdp.track_angular_velocity,
        weight=3.0,
        params={"std": math.sqrt(0.5)},
    )
    upright: RewTerm | None = RewTerm(
        func=flip_mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        },
    )
    pelvis_height_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.pelvis_height_penalty,
        weight=-1.0,
        params={"target_height": 0.77, "inverted_height": 0.58, "std": 0.18},
    )
    # Neutral standing quality.  These are penalties (positive error with
    # negative weight), not bounded rewards, so increasingly crooked poses
    # remain increasingly expensive.  They are gated inside the functions to
    # upright + near-zero velocity and therefore do not fight normal walking.
    feet_stand_pose_hold: RewTerm | None = RewTerm(
        func=flip_mdp.feet_stand_pose_hold,
        weight=-0.25,
        params={
            "velocity_epsilon": 0.05,
            "left_leg_cfg": _LEGS[0],
            "right_leg_cfg": _LEGS[1],
        },
    )
    feet_flatness_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.feet_flatness_penalty,
        weight=-0.10,
        params={
            "velocity_epsilon": 0.05,
            "feet_cfg": _FEET,
            "left_leg_cfg": _LEGS[0],
            "right_leg_cfg": _LEGS[1],
        },
    )
    pelvis_orientation_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.pelvis_orientation_penalty,
        weight=-0.50,
    )
    waist_zero_pose_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.waist_zero_pose_penalty,
        weight=-1.0,
        params={"asset_cfg": _WAIST},
    )
    handstand_leg_pose_penalty: RewTerm | None = RewTerm(
        func=flip_mdp.handstand_leg_pose_penalty,
        weight=-0.25,
        params={"left_leg_cfg": _LEGS[0], "right_leg_cfg": _LEGS[1]},
    )
    leg_lateral_alignment: RewTerm | None = RewTerm(
        func=flip_mdp.feet_alignment,
        weight=0.75,
        params={
            "std": 0.35,
            "left_leg_cfg": _LEGS[0],
            "right_leg_cfg": _LEGS[1],
            "left_alignment_cfg": SceneEntityCfg(
                "robot",
                joint_names=("left_hip_roll_joint", "left_hip_yaw_joint", "left_ankle_roll_joint"),
                preserve_order=True,
            ),
            "right_alignment_cfg": SceneEntityCfg(
                "robot",
                joint_names=("right_hip_roll_joint", "right_hip_yaw_joint", "right_ankle_roll_joint"),
                preserve_order=True,
            ),
        },
    )
    foot_heading_alignment: RewTerm | None = RewTerm(
        func=flip_mdp.foot_heading,
        weight=0.35,
        params={
            "std": 0.45,
            "turn_rate_threshold": 0.15,
            "feet_cfg": _FEET,
            "left_leg_cfg": _LEGS[0],
            "right_leg_cfg": _LEGS[1],
        },
    )

    # --- new params, flip-mode-aware function -----------------------------
    unmasked_home_tracking: RewTerm | None = RewTerm(
        func=flip_mdp.home_tracking,
        weight=0.5,
        params={
            "std": 0.4,
            # Waist is controlled separately. Upright unmasked arms retain a
            # light default-pose prior; inverted legs are handled by the
            # HANDSTAND_LEGS penalty instead of being pulled to standing home.
            "upright_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*_(shoulder|elbow|wrist)_.*",)
            ),
            "inverted_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*_(hip|knee|ankle)_.*",)
            ),
        },
    )
    com_support_projection: RewTerm | None = RewTerm(
        func=flip_mdp.com_support_projection,
        weight=1.0,
        params={"std": 0.12, **_SUPPORT_PARAMS},
    )
    commanded_leg_ground_contact: RewTerm | None = RewTerm(
        func=flip_mdp.commanded_contact,
        weight=-1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
            "left_cfg": _LEGS[0],
            "right_cfg": _LEGS[1],
            "mode": 0,
        },
    )
    upright_foot_clearance: RewTerm | None = RewTerm(
        func=flip_mdp.upright_feet_clearance,
        weight=-0.5,
        params={
            "target_height": 0.08,
            "height_sensor_name": "foot_height_scan",
            "command_name": "ballet",
            "command_threshold": 0.08,
            "asset_cfg": _FEET,
        },
    )
    upright_support_switch: RewTerm | None = RewTerm(
        func=flip_mdp.upright_support_switch_reward,
        weight=0.30,
        params={"sensor_name": "feet_ground_contact", "command_threshold": 0.08},
    )
    foot_slip: RewTerm | None = RewTerm(
        func=flip_mdp.slip,
        weight=-0.2,
        params={
            "sensor_name": "feet_ground_contact",
            "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
            "asset_cfg": _FEET,
            "mode": 0,
        },
    )
    soft_landing: RewTerm | None = RewTerm(
        func=flip_mdp.soft_landing,
        weight=-1.0e-5,
        params={"mode": 0, "sensor_name": "feet_ground_contact", "command_name": "ballet", "command_threshold": 0.05},
    )

    # --- new fields: the hand-support (inverted) mirror of the above ------
    arm_lateral_alignment: RewTerm | None = RewTerm(
        func=flip_mdp.arm_alignment,
        weight=0.35,
        params={
            "left_arm_cfg": _ARMS[0],
            "right_arm_cfg": _ARMS[1],
            "left_alignment_cfg": SceneEntityCfg(
                "robot", joint_names=("left_shoulder_roll_joint", "left_shoulder_yaw_joint"), preserve_order=True
            ),
            "right_alignment_cfg": SceneEntityCfg(
                "robot", joint_names=("right_shoulder_roll_joint", "right_shoulder_yaw_joint"), preserve_order=True
            ),
        },
    )
    commanded_arm_ground_contact: RewTerm | None = RewTerm(
        func=flip_mdp.commanded_contact,
        weight=-1.0,
        params={
            "sensor_name": "hands_ground_contact",
            "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            "left_cfg": _ARMS[0],
            "right_cfg": _ARMS[1],
            "mode": 1,
        },
    )
    hand_slip: RewTerm | None = RewTerm(
        func=flip_mdp.slip,
        weight=-0.2,
        params={
            "sensor_name": "hands_ground_contact",
            "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            "asset_cfg": _HANDS,
            "mode": 1,
        },
    )
    hand_soft_landing: RewTerm | None = RewTerm(
        func=flip_mdp.soft_landing,
        weight=-1.0e-5,
        params={"mode": 1, "sensor_name": "hands_ground_contact", "command_name": "ballet", "command_threshold": 0.05},
    )

    handstand_pose_hold: RewTerm | None = RewTerm(
        func=flip_mdp.handstand_pose_hold,
        weight=.3,
        params={"std": .5},
    )

    # --- new fields: apply regardless of mode ------------------------------
    forbidden_support: RewTerm | None = RewTerm(func=flip_mdp.forbidden_support, weight=-20.0)
    head_ground_contact: RewTerm | None = RewTerm(func=flip_mdp.head_ground_contact, weight=-20.0)
    
    body_ang_vel: RewTerm | None = RewTerm(
        func=velocity_mdp.body_angular_velocity_penalty,
        weight=-.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    )
    angular_momentum: RewTerm | None = None
    dof_pos_limits: RewTerm | None = RewTerm(func=velocity_mdp.joint_pos_limits, weight=-5.0)
    action_rate_l2: RewTerm | None = RewTerm(func=velocity_mdp.action_rate_l2, weight=-.02)
    air_time: RewTerm | None = None
    self_collisions: RewTerm | None = RewTerm(
        func=velocity_mdp.self_collision_cost,
        weight=-.25,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )
    #fell_over_penalty: RewTerm | None = RewTerm(func=flip_mdp.fell_over_penalty, weight=-100.0)
    forbidden_contact_termination_penalty: RewTerm | None = RewTerm(func=flip_mdp.forbidden_contact_termination_penalty, weight=-100.0,)
    non_finite_state_penalty: RewTerm | None = RewTerm(func=flip_mdp.non_finite_state_penalty, weight=-100.0)
    
    forbidden_body_contact: RewTerm | None = RewTerm(
        func=flip_mdp.forbidden_body_contact_penalty,
        weight=-10.0,
        params={
            "sensor_name": "forbidden_ground_contact",
        },
    )        


@configclass
class FlipPlayRewardsCfg(FlipRewardsCfg):
    """Play/eval uses the curriculum end-state weights immediately."""

    def __post_init__(self):
        self.feet_stand_pose_hold.weight = -1.0
        self.feet_flatness_penalty.weight = -0.5
        self.pelvis_orientation_penalty.weight = -2.0
        self.waist_zero_pose_penalty.weight = -4.0
        self.handstand_leg_pose_penalty.weight = -1.5
        
        self.forbidden_contact_termination_penalty = None

