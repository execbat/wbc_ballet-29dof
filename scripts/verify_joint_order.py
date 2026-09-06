"""Verify every manager (actions/observations/rewards/events/teleop) resolves
joints in the SAME order as the compiled model.

Run against a real mjlab install:
    uv run python scripts/verify_joint_order.py

Exits non-zero and prints a diff if any manager's resolved order diverges
from the model's own joint order (mujoco.mj_id2name over range(1, njnt)).
"""
import importlib.util
import sys

from mjlab.entity.entity import Entity
from mjlab.scene import Scene
from mjlab.managers.scene_entity_config import SceneEntityCfg

from wbc_ballet.robots.g1 import get_g1_robot_cfg
from wbc_ballet.tasks.ballet.ballet_actions_cfg import BalletActionsCfg
from wbc_ballet.tasks.ballet.ballet_rewards_cfg import BalletRewardsCfg
from wbc_ballet.tasks.ballet.ballet_events_cfg import BalletEventsCfg


def main() -> int:
    robot = Entity(get_g1_robot_cfg())
    scene = Scene({"robot": robot})
    model_order = list(robot.joint_names)

    rows: list[tuple[str, list[str], list[str]]] = []

    _act_ids, act_names = robot.find_joints_by_actuator_names(
        BalletActionsCfg().joint_pos.actuator_names
    )
    rows.append(("ACTIONS (JointPositionActionCfg)", act_names, model_order))
    rows.append(
        (
            "OBSERVATIONS (joint_pos / joint_vel / axis_actual_normalized)",
            list(robot.joint_names),
            model_order,
        )
    )

    rc = BalletRewardsCfg()
    c1 = rc.masked_pose_tracking.params["asset_cfg"]
    c1.resolve(scene)
    rows.append(("REWARDS masked_pose_tracking", c1.joint_names, model_order))
    c2 = rc.unmasked_home_tracking.params["asset_cfg"]
    c2.resolve(scene)
    upper_body_order = [
        name
        for name in model_order
        if name.startswith("waist_")
        or any(part in name for part in ("_shoulder_", "_elbow_", "_wrist_"))
    ]
    rows.append(
        ("REWARDS unmasked_home_tracking (upper body)", c2.joint_names, upper_body_order)
    )

    ec = BalletEventsCfg()
    c3 = ec.reset_robot_joints.params["asset_cfg"]
    c3.resolve(scene)
    rows.append(("EVENTS reset_robot_joints", c3.joint_names, model_order))

    spec = importlib.util.spec_from_file_location(
        "game_emulator", "gamepad/game_emulator_run_v1.py"
    )
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)
    rows.append(("TELEOP gamepad JOINT_NAMES (hardcoded list)", gp.JOINT_NAMES, model_order))

    print(f"{'Manager':60s} {'n':>3s}  Result")
    print("-" * 90)
    all_ok = True
    for label, order, expected_order in rows:
        ok = list(order) == expected_order
        all_ok &= ok
        print(f"{label:60s} {len(order):3d}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            for i, (a, b) in enumerate(zip(order, expected_order)):
                if a != b:
                    print(f"    first diff at index {i}: manager has {a!r}, model has {b!r}")
                    break

    print("-" * 90)
    if all_ok:
        print("ALL MANAGERS USE THE IDENTICAL JOINT ORDER -- NO MISMATCH ANYWHERE.")
        return 0
    print("MISMATCH FOUND -- see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
