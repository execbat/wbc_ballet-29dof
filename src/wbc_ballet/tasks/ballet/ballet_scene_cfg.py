"""Scene and sensor configuration for the repository-owned G1-29DoF model."""

from dataclasses import replace

from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from wbc_ballet.robots.g1 import get_g1_robot_cfg

TERRAIN_SCAN_SENSOR = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
)

FOOT_HEIGHT_SCAN_SENSOR = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
        ObjRef(type="site", name=name, entity="robot") for name in ("left_foot", "right_foot")
    ),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
)

FEET_GROUND_SENSOR = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

SELF_COLLISION_SENSOR = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)


def make_ballet_scene_cfg(*, rough: bool, play: bool) -> SceneCfg:
    terrain = (
        TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=replace(ROUGH_TERRAINS_CFG),
            max_init_terrain_level=5,
        )
        if rough
        else TerrainEntityCfg(terrain_type="plane")
    )
    if rough and terrain.terrain_generator is not None:
        terrain.terrain_generator.curriculum = not play
        if play:
            terrain.terrain_generator.num_cols = 5
            terrain.terrain_generator.num_rows = 5
            terrain.terrain_generator.border_width = 10.0

    sensors = (
        (TERRAIN_SCAN_SENSOR, FOOT_HEIGHT_SCAN_SENSOR, FEET_GROUND_SENSOR, SELF_COLLISION_SENSOR)
        if rough
        else (FOOT_HEIGHT_SCAN_SENSOR, FEET_GROUND_SENSOR, SELF_COLLISION_SENSOR)
    )
    return SceneCfg(
        terrain=terrain,
        entities={"robot": get_g1_robot_cfg()},
        sensors=sensors,
        num_envs=1 if play else 4096,
        extent=2.5,
    )
