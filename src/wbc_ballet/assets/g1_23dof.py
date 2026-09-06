"""Compatibility exports for the repository-owned G1-23DoF model."""

from wbc_ballet.robots.g1.constants import (
    G1_23DOF_XML,
    get_g1_23dof_robot_cfg,
)
from wbc_ballet.robots.g1.constants import (
    get_23dof_spec as get_spec,
)

__all__ = ["G1_23DOF_XML", "get_g1_23dof_robot_cfg", "get_spec"]
