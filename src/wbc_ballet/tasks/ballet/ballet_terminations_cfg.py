"""Termination manager terms for G1 ballet."""

from mjlab.managers.termination_manager import TerminationTermCfg as DoneTerm
from mjlab.tasks.velocity import mdp

from wbc_ballet.tasks.ballet import mdp as ballet_mdp
from wbc_ballet.utils.configclass import configclass


@configclass
class BalletTerminationsCfg:
    time_out: DoneTerm | None = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over: DoneTerm | None = DoneTerm(
        func=ballet_mdp.pelvis_height_below,
        params={"minimum_height": 0.2},
    )
    non_finite_state: DoneTerm | None = DoneTerm(
        func=ballet_mdp.non_finite_state_or_action,
    )
    out_of_terrain_bounds: DoneTerm | None = None


@configclass
class BalletRoughTerminationsCfg(BalletTerminationsCfg):
    out_of_terrain_bounds: DoneTerm | None = DoneTerm(
        func=mdp.out_of_terrain_bounds,
        time_out=True,
    )


@configclass
class BalletPlayTerminationsCfg(BalletTerminationsCfg):
    time_out: DoneTerm | None = None


@configclass
class BalletRoughPlayTerminationsCfg(BalletRoughTerminationsCfg):
    time_out: DoneTerm | None = None
    out_of_terrain_bounds: DoneTerm | None = None
