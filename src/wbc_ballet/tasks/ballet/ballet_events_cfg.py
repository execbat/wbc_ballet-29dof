"""Event manager terms for reset and domain randomization."""

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from wbc_ballet.tasks.ballet import mdp as ballet_mdp
from wbc_ballet.utils.configclass import configclass

_FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))
_JOINT_TABLE_INTERVAL_S = 5.0
VELOCITY_PUSH_RANGE = (-0.15, 0.15)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)


@configclass
class BalletEventsCfg:
    # Kept out of training: copying and printing tensors would distort throughput.
    joint_observation_table: EventTerm | None = None
    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )
    reset_robot_joints: EventTerm | None = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
#    push_robot: EventTerm | None = EventTerm(
#        func=mdp.push_by_setting_velocity,
#        mode="interval",
#        interval_range_s=VELOCITY_PUSH_INTERVAL_S,
#        params={
#            "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
#            "asset_cfg": SceneEntityCfg("robot"),
#        },
#    )
    foot_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    )
    encoder_bias: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    )
    base_com: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
        },
    )


@configclass
class BalletPlayEventsCfg(BalletEventsCfg):
    joint_observation_table: EventTerm | None = EventTerm(
        func=ballet_mdp.print_joint_observation_table,
        mode="interval",
        interval_range_s=(_JOINT_TABLE_INTERVAL_S, _JOINT_TABLE_INTERVAL_S),
        is_global_time=True,
        params={
            "env_index": 0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    push_robot: EventTerm | None = None
