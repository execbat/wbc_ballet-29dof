"""Reset and diagnostic event terms owned by the flip task."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .observations import ballet_mask, joint_pos_normalized, masked_ballet_targets

_ROBOT_CFG = SceneEntityCfg("robot")


def print_joint_observation_table(env, env_ids, *, env_index=0, asset_cfg=_ROBOT_CFG):
    del env_ids
    if not 0 <= env_index < env.num_envs:
        raise IndexError(f"env_index={env_index} is outside [0, {env.num_envs})")
    robot = env.scene[asset_cfg.name]
    ids = list(range(len(robot.joint_names)))[asset_cfg.joint_ids] if isinstance(asset_cfg.joint_ids, slice) else list(asset_cfg.joint_ids)
    rows = zip(
        [robot.joint_names[i] for i in ids],
        robot.data.joint_pos[env_index, ids].detach().cpu().tolist(),
        robot.data.default_joint_pos[env_index, ids].detach().cpu().tolist(),
        joint_pos_normalized(env, asset_cfg)[env_index].detach().cpu().tolist(),
        ballet_mask(env)[env_index, ids].detach().cpu().tolist(),
        masked_ballet_targets(env)[env_index, ids].detach().cpu().tolist(),
        strict=True,
    )
    lines = [f"\n[G1 flip observations | env={env_index} | step={env.common_step_counter}]",
             f"{'#':>2}  {'joint':34} {'q_actual':>10} {'q_home':>9} {'q_norm':>9} {'mask':>5} {'target':>9}"]
    for i, (name, q, home, norm, mask, target) in enumerate(rows):
        lines.append(f"{i:2d}  {name:34} {q:10.4f} {home:9.4f} {norm:9.4f} {mask:5.0f} {target:9.4f}")
    print("\n".join(lines), flush=True)

# Fitted to this repository's g1.xml using MuJoCo forward kinematics.
# Left-side values; mirrored for the right side by reset_robot/arm_alignment
# (roll/yaw negated, pitch kept, matching the existing mirror rule).
HANDSTAND_LEFT = {
    'shoulder_pitch': -2.2707281707, 'shoulder_roll': 1.3338481535,
    'shoulder_yaw': -.7025248294, 'elbow': .1181938947,
    'wrist_roll': .2660466197, 'wrist_pitch': -1.0711314079,
    'wrist_yaw': 1.45,
}
# Compact, slightly bent stance, mirrored the same way as HANDSTAND_LEFT.
HANDSTAND_LEGS = {
    'hip_pitch': .5445615503, 'hip_roll': 0., 'hip_yaw': 0.,
    'knee': .4492760091, 'ankle_pitch': .0006505147, 'ankle_roll': 0.,
}
HANDSTAND_WAIST = {'waist_yaw': 0., 'waist_roll': 0., 'waist_pitch': -.0438119983}
# Measured against the flat hand-support boxes in this repository's g1.xml.
HANDSTAND_HEIGHT = .58


def handstand_value(name):
    """Look up a joint/actuator name (e.g. ``'left_knee_joint'``) in the
    handstand pose tables, applying the same left/right mirror rule as
    ``reset_robot``. Returns ``None`` if the name isn't part of the
    specified pose (caller should keep it at its own default in that case).
    Shared with the reward code that needs the same reference pose (e.g.
    ``handstand_pose_hold``), so the two can never silently drift apart.
    """
    for side in ('left', 'right'):
        for joint, angle in {**HANDSTAND_LEFT, **HANDSTAND_LEGS}.items():
            if name == f'{side}_{joint}_joint':
                mirror = side == 'right' and ('roll' in joint or 'yaw' in joint)
                return -angle if mirror else angle
    for joint, angle in HANDSTAND_WAIST.items():
        if name == f'{joint}_joint':
            return angle
    return None


def _uniform_quaternions(n, *, device, dtype):
    """Sample rotations uniformly from SO(3), returned as MuJoCo wxyz."""
    u1, u2, u3 = torch.rand(3, n, device=device, dtype=dtype)
    a = (1.0 - u1).sqrt()
    b = u1.sqrt()
    two_pi = 2.0 * torch.pi
    return torch.stack(
        (b * torch.cos(two_pi * u3), a * torch.sin(two_pi * u2),
         a * torch.cos(two_pi * u2), b * torch.sin(two_pi * u3)), dim=1
    )


def reset_robot(
    env,
    env_ids,
    flip_probability=.5,
    randomize_pose=True,
    joint_range_fraction=.7,
    spawn_height_range=(1.0, 1.2),
):
    robot = env.scene['robot']
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if not hasattr(env, '_flip_mode'):
        env._flip_mode = torch.zeros(env.num_envs, device=env.device)
    if flip_probability is None:
        # Play reset follows the latest checkbox; ordinary command changes do not reset.
        receiver = getattr(env, '_flip_udp_receiver', None)
        value = receiver.poll().flip if receiver is not None else 0
        f = torch.full((n,), bool(value), device=env.device)
    else:
        f = torch.rand(n, device=env.device) < flip_probability
    env._flip_mode[env_ids] = f.float()
    q = robot.data.default_joint_pos[env_ids].clone()
    if randomize_pose:
        if not 0.0 <= joint_range_fraction <= 1.0:
            raise ValueError("joint_range_fraction must be in [0, 1]")
        limits = robot.data.joint_pos_limits[env_ids]
        lo, hi = limits[..., 0], limits[..., 1]
        valid = torch.isfinite(lo) & torch.isfinite(hi) & (hi > lo)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * joint_range_fraction
        sampled = centre + (2.0 * torch.rand_like(q) - 1.0) * half
        fallback = q + (2.0 * torch.rand_like(q) - 1.0) * .35
        q = torch.where(valid, sampled, fallback)
    else:
        for i, name in enumerate(robot.joint_names):
            value = handstand_value(name)
            if value is not None:
                q[:, i] = torch.where(f, value, q[:, i])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    if randomize_pose:
        pose[:, :2] += (2.0 * torch.rand(n, 2, device=env.device) - 1.0) * .25
        low, high = spawn_height_range
        pose[:, 2] += low + torch.rand(n, device=env.device) * (high - low)
        pose[:, 3:7] = _uniform_quaternions(n, device=env.device, dtype=pose.dtype)
    else:
        pose[:, 2] += torch.where(f, HANDSTAND_HEIGHT + .01, .81)
        yaw = (2.0 * torch.rand(n, device=env.device) - 1.0) * torch.pi
        c, s = torch.cos(yaw / 2), torch.sin(yaw / 2)
        pose[:, 3] = torch.where(f, 0., c)
        pose[:, 4] = torch.where(f, c, 0.)
        pose[:, 5] = torch.where(f, s, 0.)
        pose[:, 6] = torch.where(f, 0., s)
    robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
