"""Self-contained flat scene for foot and hand support."""

from copy import deepcopy

from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from wbc_ballet.robots.g1 import get_g1_robot_cfg

FOOT_HEIGHT_SCAN_SENSOR = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=n, entity="robot") for n in ("left_foot", "right_foot")),
    pattern=RingPatternCfg.single_ring(radius=.03, num_samples=6),
    ray_alignment="yaw", max_distance=1., exclude_parent_body=True,
    include_geom_groups=(0,), debug_vis=True,
)

FEET_GROUND_SENSOR = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
)

SELF_COLLISION_SENSOR = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
)

HANDS_GROUND_CONTACT_SENSOR = ContactSensorCfg(
    name="hands_ground_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(left_wrist_yaw_link|right_wrist_yaw_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

HEAD_GROUND_CONTACT_SENSOR = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="geom", pattern="head_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
)

_HAND_COLLISION_PATTERN = r"^(left|right)_hand_collision$"

FORBIDDEN_GROUND_CONTACT_SENSOR = ContactSensorCfg(
    name="forbidden_ground_contact",
    primary=ContactMatch(
        mode="geom",
        pattern=(
            r"^(?:"
            r"pelvis_collision|"
            r"torso_collision|"
            r"head_collision|"
            r"(?:left|right)_(?:"
            r"hip|"
            r"thigh|"
            r"shin|"
            r"linkage_brace|"
            r"shoulder_yaw|"
            r"elbow_yaw"
            r")_collision"
            r")$"
        ),
        entity="robot",
    ),
    secondary=ContactMatch(
        mode="body",
        pattern="terrain",
    ),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
)

def make_flip_scene_cfg(*, play: bool) -> SceneCfg:
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_g1_robot_cfg()},
        sensors=(
        FOOT_HEIGHT_SCAN_SENSOR,
        FEET_GROUND_SENSOR,
        SELF_COLLISION_SENSOR,
        HANDS_GROUND_CONTACT_SENSOR,
        HEAD_GROUND_CONTACT_SENSOR,
        FORBIDDEN_GROUND_CONTACT_SENSOR,
        ),
        num_envs=1 if play else 4096,
        extent=2.5,
    )

    # Hands need tangential friction to act as a support surface; this rule
    # must come before the general `.*_collision` rule below it, or the
    # model's default condim=1 for that rule wins instead.
    collision = deepcopy(scene.entities["robot"].collisions[0])
    collision.condim = {_HAND_COLLISION_PATTERN: 3, **collision.condim}
    collision.priority = {_HAND_COLLISION_PATTERN: 1, **collision.priority}
    collision.friction = {_HAND_COLLISION_PATTERN: (0.6, 0.005, 0.0001), **collision.friction}
    scene.entities["robot"].collisions = (collision,)
    return scene
