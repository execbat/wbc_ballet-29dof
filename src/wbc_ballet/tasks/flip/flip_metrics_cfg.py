"""Metrics manager configuration."""

from mjlab.managers.metrics_manager import MetricsTermCfg as MetricTerm
from wbc_ballet.utils.configclass import configclass
from . import mdp


@configclass
class FlipMetricsCfg:
    nan_physics_state: MetricTerm | None = MetricTerm(func=mdp.invalid_physics_value_type, params={"value_type": "nan"}, reduce="max")
    inf_physics_state: MetricTerm | None = MetricTerm(func=mdp.invalid_physics_value_type, params={"value_type": "inf"}, reduce="max")
    nonfinite_qpos: MetricTerm | None = MetricTerm(func=mdp.nonfinite_physics_component, params={"component": "qpos"}, reduce="max")
    nonfinite_qvel: MetricTerm | None = MetricTerm(func=mdp.nonfinite_physics_component, params={"component": "qvel"}, reduce="max")
    nonfinite_qacc: MetricTerm | None = MetricTerm(func=mdp.nonfinite_physics_component, params={"component": "qacc"}, reduce="max")
    nonfinite_qacc_warmstart: MetricTerm | None = MetricTerm(func=mdp.nonfinite_physics_component, params={"component": "qacc_warmstart"}, reduce="max")
    nonfinite_sensordata: MetricTerm | None = MetricTerm(func=mdp.nonfinite_physics_component, params={"component": "sensordata"}, reduce="max")
    nonfinite_policy_action: MetricTerm | None = MetricTerm(func=mdp.nonfinite_policy_action, reduce="max")
