import os

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("WBC_MJLAB_SKIP_MJLAB_REGISTER", "1")

import mujoco

from wbc_ballet.assets.g1_23dof import get_spec as get_23dof_spec
from wbc_ballet.robots.g1 import G1_ARTICULATION, get_g1_robot_cfg
from wbc_ballet.robots.g1.constants import get_spec

# Canonical G1-29DoF joint order: MJCF declaration order (= kinematic-tree
# DFS order), which is what mjlab resolves an unfiltered SceneEntityCfg's
# joints into (see mjlab.managers.scene_entity_config.SceneEntityCfg.resolve
# and mjlab.entity.Entity.find_joints_by_actuator_names). This is the SAME
# order used for observations (joint_pos, joint_vel, axis_actual_normalized),
# actions, and the ballet command's targets/mask -- see
# wbc_ballet.teleop.protocol / wbc_ballet.tasks.ballet.mdp.commands / .observations.
CANONICAL_JOINT_ORDER_29 = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _joint_names(model: mujoco.MjModel) -> list[str]:
    # index 0 is the free (floating base) joint; hinge joints start at 1.
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]


def test_model_has_exactly_29_hinge_joints() -> None:
    model = get_spec().compile()
    # One free joint contributes 7 qpos / 6 qvel; the remaining 29 are actuated hinges.
    assert model.nq - 7 == 29
    assert model.nv - 6 == 29
    names = _joint_names(model)
    assert len(names) == 29


def test_model_joint_order_matches_canonical_order() -> None:
    """Locks the exact axis order every other module assumes.

    If this ever fails after an MJCF edit, every hardcoded axis-order
    assumption in tasks/ballet/mdp/commands.py, tasks/ballet/mdp/observations.py,
    teleop/protocol.py, and gamepad/game_emulator_run_v1.py needs revisiting.
    """
    model = get_spec().compile()
    assert tuple(_joint_names(model)) == CANONICAL_JOINT_ORDER_29


def test_entity_uses_repository_articulation() -> None:
    entity_cfg = get_g1_robot_cfg()
    assert entity_cfg.articulation is G1_ARTICULATION
    assert len(G1_ARTICULATION.actuators) == 6


def test_all_29_joints_get_exactly_one_actuator() -> None:
    """Guards the ``resolve_matching_names`` failure mode found while
    porting to 29DoF: a G1_ACTUATOR_* group whose target_names_expr
    contains a pattern with zero matches makes entity construction raise.
    This also confirms no joint is claimed by two groups (which would
    raise separately, "multiple matches")."""
    entity_cfg = get_g1_robot_cfg()
    from mjlab.entity.entity import Entity

    robot = Entity(entity_cfg)
    matched = set()
    for actuator in G1_ARTICULATION.actuators:
        import re

        for pattern in actuator.target_names_expr:
            hits = [n for n in robot.joint_names if re.fullmatch(pattern, n)]
            assert hits, f"pattern {pattern!r} matched no joints in the 29DoF model"
            matched.update(hits)
    assert matched == set(CANONICAL_JOINT_ORDER_29)


##
# Legacy 23DoF reference model: kept in the repository, no longer used by
# the active ballet task (see ``wbc_ballet.tasks.ballet.ballet_scene_cfg``,
# which now loads the 29DoF model). These checks only confirm the leftover
# asset itself is still structurally intact, not that anything active still
# depends on it.
##


def test_legacy_23dof_model_still_has_23_hinge_joints() -> None:
    model = get_23dof_spec().compile()
    assert model.nq - 7 == 23
    assert model.nv - 6 == 23
    names = _joint_names(model)
    assert len(names) == 23
    assert not any("wrist_pitch" in name or "wrist_yaw" in name for name in names)
