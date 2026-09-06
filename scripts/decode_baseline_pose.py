"""Decode gamepad/game_emulator_run_v1.py's INIT_BASELINE (normalized
[-1, 1] per axis) back into absolute joint angles (radians), against the
real compiled model's joint limits. Useful when hand-tuning the baseline
pose (e.g. picking a non-zero neutral for waist_roll/waist_pitch/wrist
axes) -- move a slider in the tool, note the printed normalized value,
then re-run this to see the resulting absolute angle, or vice versa.

Run against a real mjlab/mujoco install:
    uv run python scripts/decode_baseline_pose.py
"""

import importlib.util

import mujoco

_HERE = __file__.rsplit("/", 1)[0]


def to_absolute(norm: float, lo: float, hi: float) -> float:
    return lo + (norm + 1.0) * (hi - lo) / 2.0


def main() -> None:
    spec = mujoco.MjSpec.from_file(f"{_HERE}/../src/wbc_ballet/robots/g1/xmls/g1.xml")
    model = spec.compile()
    limits = {}
    for i in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        limits[name] = tuple(model.jnt_range[jid])

    spec2 = importlib.util.spec_from_file_location(
        "game_emulator", f"{_HERE}/../gamepad/game_emulator_run_v1.py"
    )
    gp = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(gp)

    print(f"{'idx':>3} {'joint':28s} {'norm':>8} {'-> abs(rad)':>11} {'[lower, upper]':>18}")
    print("-" * 76)
    for i, name in enumerate(gp.JOINT_NAMES):
        lo, hi = limits[name]
        norm = float(gp.INIT_BASELINE[i])
        abs_angle = to_absolute(norm, lo, hi)
        print(f"{i:3d} {name:28s} {norm:8.4f} {abs_angle:11.5f} [{lo:7.4f}, {hi:7.4f}]")


if __name__ == "__main__":
    main()
