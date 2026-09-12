"""Event manager terms for reset and domain randomization."""

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wbc_ballet.utils.configclass import configclass

from . import mdp as flip_mdp

_HAND_GEOMS = ("left_hand_collision", "right_hand_collision")
_FOOT_GEOMS = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))


@configclass
class FlipEventsCfg:
    # One atomic reset sets pose, joints and the flip mode together, before
    # command resampling -- replaces ballet's separate base/joint resets.
    joint_observation_table: EventTerm | None = None
    reset_robot: EventTerm | None = EventTerm(
        func=flip_mdp.reset_robot,
        mode="reset",
        params={
            "flip_probability": 0.5,
            "randomize_pose": True,
            "joint_range_fraction": 0.7,
            "spawn_height_range": (1.0, 1.2),
        },
    )
    hand_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=_HAND_GEOMS),
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    )
    foot_friction: EventTerm | None = EventTerm(
        mode="startup", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS), "operation": "abs", "ranges": (.3, 1.2), "shared_random": True},
    )
    encoder_bias: EventTerm | None = EventTerm(
        mode="startup", func=dr.encoder_bias,
        params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-.015, .015)},
    )
    base_com: EventTerm | None = EventTerm(
        mode="startup", func=dr.body_com_offset,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": {0: (-.025, .025), 1: (-.025, .025), 2: (-.03, .03)}},
    )


@configclass
class FlipPlayEventsCfg(FlipEventsCfg):
    joint_observation_table: EventTerm | None = EventTerm(
        func=flip_mdp.print_joint_observation_table,
        mode="interval", interval_range_s=(5., 5.), is_global_time=True,
        params={"env_index": 0, "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    reset_robot: EventTerm | None = EventTerm(
        func=flip_mdp.reset_robot,
        mode="reset",
        # The goal follows the UDP checkbox, while physical spawn state uses
        # the same SO(3) and joint randomization as training.
        params={
            "flip_probability": None,
            "randomize_pose": True,
            "joint_range_fraction": 0.7,
            "spawn_height_range": (1.0, 1.2),
        },
    )
