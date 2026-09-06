"""Local copy of the wbc-mjlab Unitree G1 model and articulation."""

from .actuators import G1_ACTION_SCALE, G1_ARTICULATION
from .constants import G1_23DOF_XML, G1_XML, get_g1_23dof_robot_cfg, get_g1_robot_cfg

__all__ = [
    "G1_23DOF_XML",
    "G1_XML",
    "G1_ACTION_SCALE",
    "G1_ARTICULATION",
    "get_g1_23dof_robot_cfg",
    "get_g1_robot_cfg",
]
