"""Termination manager configuration for train and play."""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.tasks.velocity import mdp as velocity_mdp
from wbc_ballet.utils.configclass import configclass
from . import mdp


def _fell_over():
    return None #DoneTerm(func=mdp.pelvis_height_below_for_duration, params={"minimum_height": .2, "duration_s": 2., "recovery_height": .35, "recovery_duration_s": 2.})


def _forbidden():
    return DoneTerm(
        func=mdp.forbidden_contact_count_exceeded,
        params={
            "sensor_name": "forbidden_ground_contact",
            "maximum_contacts": 1,
            "contact_interval_s": 0.5,
            "grace_s": 2.0,
        },
    )


@configclass
class FlipTerminationsCfg:
    time_out: DoneTerm | None = DoneTerm(func=velocity_mdp.time_out, time_out=True)
    fell_over: DoneTerm | None = _fell_over()
    forbidden_ground_contact: DoneTerm | None = _forbidden()
    non_finite_state: DoneTerm | None = DoneTerm(func=mdp.non_finite_state_or_action)
    out_of_terrain_bounds: DoneTerm | None = None


@configclass
class FlipPlayTerminationsCfg:
    time_out: DoneTerm | None = None
    fell_over: DoneTerm | None = _fell_over()
    forbidden_ground_contact: DoneTerm | None = _forbidden()
    non_finite_state: DoneTerm | None = DoneTerm(func=mdp.non_finite_state_or_action)
    out_of_terrain_bounds: DoneTerm | None = None
