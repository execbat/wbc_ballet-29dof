"""Self-contained environment assembly for the flip task."""

from dataclasses import field
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.viewer import ViewerConfig
from wbc_ballet.utils.configclass import configclass
from wbc_ballet.utils.manager_compat import group_to_dict, observations_to_dict
from .flip_actions_cfg import FlipActionsCfg
from .flip_commands_cfg import FlipCommandsCfg, FlipUdpCommandsCfg
from .flip_curriculum_cfg import FlipCurriculumCfg, FlipPlayCurriculumCfg
from .flip_events_cfg import FlipEventsCfg, FlipPlayEventsCfg
from .flip_metrics_cfg import FlipMetricsCfg
from .flip_observations_cfg import FlipObservationsCfg
from .flip_rewards_cfg import FlipRewardsCfg
from .flip_scene_cfg import make_flip_scene_cfg
from .flip_terminations_cfg import FlipPlayTerminationsCfg, FlipTerminationsCfg


def _sim_cfg():
    return SimulationCfg(
        njmax=300,
        contact_sensor_maxmatch=1024,
        nan_guard=NanGuardCfg(enabled=False, buffer_size=32, output_dir="logs/nan_dumps", max_envs_to_dump=5),
        mujoco=MujocoCfg(timestep=.005, iterations=10, ls_iterations=20, ccd_iterations=50),
    )


def _viewer_cfg():
    return ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=3., elevation=-5., azimuth=90.)


def _to_mjlab_cfg(cfg):
    return ManagerBasedRlEnvCfg(
        scene=cfg.scene, observations=observations_to_dict(cfg.observations), actions=group_to_dict(cfg.actions),
        commands=group_to_dict(cfg.commands), events=group_to_dict(cfg.events), rewards=group_to_dict(cfg.rewards),
        terminations=group_to_dict(cfg.terminations), curriculum=group_to_dict(cfg.curriculum), metrics=group_to_dict(cfg.metrics),
        sim=cfg.sim, viewer=cfg.viewer, decimation=cfg.decimation, episode_length_s=cfg.episode_length_s,
    )


@configclass
class FlipEnvCfg:
    scene: SceneCfg = field(default_factory=lambda: make_flip_scene_cfg(play=False))
    observations: FlipObservationsCfg = FlipObservationsCfg()
    actions: FlipActionsCfg = FlipActionsCfg()
    commands: FlipCommandsCfg = FlipCommandsCfg()
    events: FlipEventsCfg = FlipEventsCfg()
    rewards: FlipRewardsCfg = FlipRewardsCfg()
    terminations: FlipTerminationsCfg = FlipTerminationsCfg()
    curriculum: FlipCurriculumCfg = FlipCurriculumCfg()
    metrics: FlipMetricsCfg = FlipMetricsCfg()
    decimation: int = 4
    episode_length_s: float = 24.
    sim: SimulationCfg = field(default_factory=_sim_cfg)
    viewer: ViewerConfig = field(default_factory=_viewer_cfg)

    def to_mjlab_cfg(self):
        return _to_mjlab_cfg(self)


@configclass
class FlipEnvCfg_PLAY:
    scene: SceneCfg = field(default_factory=lambda: make_flip_scene_cfg(play=True))
    observations: FlipObservationsCfg = FlipObservationsCfg()
    actions: FlipActionsCfg = FlipActionsCfg()
    commands: FlipUdpCommandsCfg = FlipUdpCommandsCfg()
    events: FlipPlayEventsCfg = FlipPlayEventsCfg()
    rewards: FlipRewardsCfg = FlipRewardsCfg()
    terminations: FlipPlayTerminationsCfg = FlipPlayTerminationsCfg()
    curriculum: FlipPlayCurriculumCfg = FlipPlayCurriculumCfg()
    metrics: FlipMetricsCfg = FlipMetricsCfg()
    decimation: int = 4
    episode_length_s: float = 1.e9
    sim: SimulationCfg = field(default_factory=_sim_cfg)
    viewer: ViewerConfig = field(default_factory=_viewer_cfg)

    def __post_init__(self):
        self.observations.actor.enable_corruption = False

    def to_mjlab_cfg(self):
        return _to_mjlab_cfg(self)


def make_flip_env_cfg(*, play=False) -> ManagerBasedRlEnvCfg:
    return (FlipEnvCfg_PLAY() if play else FlipEnvCfg()).to_mjlab_cfg()
