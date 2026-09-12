"""Observation configuration; field order is the policy ABI."""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from wbc_ballet.robots.g1.constants import G1_IMU_ANG_VEL_SENSOR, G1_IMU_LIN_ACC_SENSOR, G1_IMU_LIN_VEL_SENSOR
from wbc_ballet.utils.configclass import configclass
from wbc_ballet.utils.manager_compat import ObsGroup
from . import mdp

_FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True)
_HANDS = SceneEntityCfg("robot", site_names=("left_palm", "right_palm"), preserve_order=True)
_LEGS = (SceneEntityCfg("robot", joint_names=(r"left_(hip|knee|ankle)_.*",)), SceneEntityCfg("robot", joint_names=(r"right_(hip|knee|ankle)_.*",)))
_ARMS = (SceneEntityCfg("robot", joint_names=(r"left_(shoulder|elbow|wrist)_.*",)), SceneEntityCfg("robot", joint_names=(r"right_(shoulder|elbow|wrist)_.*",)))
_SUPPORT = dict(feet_cfg=_FEET, hands_cfg=_HANDS, left_leg_cfg=_LEGS[0], right_leg_cfg=_LEGS[1], left_arm_cfg=_ARMS[0], right_arm_cfg=_ARMS[1])


@configclass
class FlipPolicyCfg(ObsGroup):
    base_ang_vel: ObsTerm | None = ObsTerm(func=velocity_mdp.builtin_sensor, params={"sensor_name": G1_IMU_ANG_VEL_SENSOR}, noise=Unoise(n_min=-.2, n_max=.2))
    imu_lin_acc: ObsTerm | None = ObsTerm(func=velocity_mdp.builtin_sensor, params={"sensor_name": G1_IMU_LIN_ACC_SENSOR}, noise=Unoise(n_min=-.2, n_max=.2), clip=(-30., 30.), scale=.1)
    projected_gravity: ObsTerm | None = ObsTerm(func=velocity_mdp.projected_gravity, noise=Unoise(n_min=-.05, n_max=.05))
    velocity_commands: ObsTerm | None = ObsTerm(func=mdp.ballet_velocity)
    joint_pos: ObsTerm | None = ObsTerm(func=velocity_mdp.joint_pos_rel, noise=Unoise(n_min=-.01, n_max=.01))
    joint_vel: ObsTerm | None = ObsTerm(func=velocity_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions: ObsTerm | None = ObsTerm(func=velocity_mdp.last_action)
    axis_actual_normalized: ObsTerm | None = ObsTerm(func=mdp.joint_pos_normalized)
    axis_target_normalized: ObsTerm | None = ObsTerm(func=mdp.masked_ballet_targets)
    axis_mask: ObsTerm | None = ObsTerm(func=mdp.ballet_mask)
    flip: ObsTerm | None = ObsTerm(func=mdp.flip)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.nan_policy = "warn"
        self.nan_check_per_term = False


@configclass
class FlipCriticCfg(FlipPolicyCfg):
    base_lin_vel: ObsTerm | None = ObsTerm(func=velocity_mdp.builtin_sensor, params={"sensor_name": G1_IMU_LIN_VEL_SENSOR}, noise=Unoise(n_min=-.1, n_max=.1))
    whole_body_com_xy: ObsTerm | None = ObsTerm(func=mdp.whole_body_com_xy_b)
    support_center_xy: ObsTerm | None = ObsTerm(func=mdp.support_center, params=_SUPPORT)
    foot_height: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_height, params={"sensor_name": "foot_height_scan"})
    foot_air_time: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"})
    foot_contact: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"})
    foot_contact_forces: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"})
    hand_air_time: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_air_time, params={"sensor_name": "hands_ground_contact"})
    hand_contact: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_contact, params={"sensor_name": "hands_ground_contact"})
    hand_contact_forces: ObsTerm | None = ObsTerm(func=velocity_mdp.foot_contact_forces, params={"sensor_name": "hands_ground_contact"})

    def __post_init__(self):
        super().__post_init__()
        self.enable_corruption = False


@configclass
class FlipObservationsCfg:
    actor: FlipPolicyCfg = FlipPolicyCfg()
    critic: FlipCriticCfg = FlipCriticCfg()
