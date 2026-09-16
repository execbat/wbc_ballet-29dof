"""All reward terms owned by the standalone flip task."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from mjlab.tasks.velocity import mdp as velocity_mdp
from .observations import (
    _group_has_active_mask, _world_xy_to_base_heading,
    ballet_mask, ballet_targets, joint_pos_normalized,
    whole_body_com_xy_b, support_center_xy_b,
)
from .events import HANDSTAND_LEFT, HANDSTAND_LEGS, handstand_value

_ROBOT_CFG = SceneEntityCfg("robot")


def _selected_command_axes(values, asset_cfg):
    return values[:, asset_cfg.joint_ids]


def masked_pose_tracking(env, std=.5, asset_cfg=_ROBOT_CFG):
    targets = _selected_command_axes(ballet_targets(env), asset_cfg)
    mask = _selected_command_axes(ballet_mask(env), asset_cfg)
    error = (joint_pos_normalized(env, asset_cfg) - targets).square()
    count = mask.sum(1)
    score = torch.exp(-((error * mask).sum(1) / count.clamp_min(1.0)) / std**2)
    return torch.where(count > 0, score, torch.zeros_like(score))


def unmasked_home_tracking(env, std=.5, asset_cfg=_ROBOT_CFG):
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    scale = (0.5 * (robot.data.joint_pos_limits[:, ids, 1] - robot.data.joint_pos_limits[:, ids, 0])).clamp_min(1e-6)
    error = ((robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]) / scale).square()
    inactive = 1.0 - _selected_command_axes(ballet_mask(env), asset_cfg)
    count = inactive.sum(1)
    score = torch.exp(-((error * inactive).sum(1) / count.clamp_min(1.0)) / std**2)
    return torch.where(count > 0, score, torch.zeros_like(score))


def fell_over_penalty(env):
    return env.termination_manager.get_term("fell_over").float()

def forbidden_contact_termination_penalty(env) -> torch.Tensor:
    return env.termination_manager.get_term( "forbidden_ground_contact").float()


def non_finite_state_penalty(env):
    return env.termination_manager.get_term("non_finite_state").float()


def unmasked_leg_lateral_alignment(
    env, std=.35, asset_cfg=_ROBOT_CFG, *, left_leg_cfg, right_leg_cfg,
    left_alignment_cfg, right_alignment_cfg,
):
    robot = env.scene[asset_cfg.name]

    def score(cfg):
        ids = cfg.joint_ids
        half = (0.5 * (robot.data.joint_pos_limits[:, ids, 1] - robot.data.joint_pos_limits[:, ids, 0])).clamp_min(1e-6)
        mse = ((robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]) / half).square().mean(1)
        return torch.exp(-mse / std**2)

    left_ok = ~_group_has_active_mask(env, left_leg_cfg)
    right_ok = ~_group_has_active_mask(env, right_leg_cfg)
    count = left_ok.float() + right_ok.float()
    value = (score(left_alignment_cfg) * left_ok + score(right_alignment_cfg) * right_ok) / count.clamp_min(1.0)
    turning = env.command_manager.get_command("ballet")[:, 2].abs() > .15
    return torch.where((count > 0) & ~turning, value, torch.zeros_like(value))


def unmasked_foot_heading_alignment(
    env, std=.45, turn_rate_threshold=.15, asset_cfg=_ROBOT_CFG, *,
    feet_cfg, left_leg_cfg, right_leg_cfg,
):
    robot = env.scene[asset_cfg.name]
    quat = robot.data.site_quat_w[:, feet_cfg.site_ids]
    forward = torch.zeros_like(quat[..., :3]); forward[..., 0] = 1.0
    foot_forward = quat_apply(quat, forward)[..., :2]
    foot_forward = foot_forward / foot_forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    heading = robot.data.heading_w
    base_forward = torch.stack((heading.cos(), heading.sin()), dim=1)
    dot = (foot_forward * base_forward[:, None]).sum(2).clamp(-1, 1)
    cross = base_forward[:, None, 0] * foot_forward[:, :, 1] - base_forward[:, None, 1] * foot_forward[:, :, 0]
    scores = torch.exp(-torch.atan2(cross, dot).square() / std**2)
    eligible = torch.stack((~_group_has_active_mask(env, left_leg_cfg), ~_group_has_active_mask(env, right_leg_cfg)), dim=1)
    eligible &= (env.command_manager.get_command("ballet")[:, 2].abs() <= turn_rate_threshold)[:, None]
    count = eligible.sum(1)
    value = (scores * eligible.float()).sum(1) / count.clamp_min(1)
    return torch.where(count > 0, value, torch.zeros_like(value))


def flip(env):
    return env.command_manager.get_command('flip')


def inverted(env):
    return flip(env)[:, 0] >= .5


def upright(env, std, asset_cfg):
    q = env.scene[asset_cfg.name].data.body_link_quat_w[:, asset_cfg.body_ids]
    z = torch.zeros_like(q[..., :3]); z[..., 2] = 1
    world_z = quat_apply(q, z)[..., 2].mean(1)
    desired = torch.where(inverted(env), -1.0, 1.0)
    return torch.exp(-(world_z - desired).square() / std**2)


def pelvis_height_penalty(env, target_height=.77, inverted_height=.58, std=.18):
    z = env.scene['robot'].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target = torch.where(inverted(env), inverted_height, target_height)
    return ((z - target) / std).square()


def _stationary_ballet(env, velocity_epsilon=.05):
    """Return True where the commanded planar velocity/yaw rate is near zero."""
    return env.command_manager.get_command("ballet")[:, :3].abs().amax(1) < velocity_epsilon


def _joint_tolerance(name: str) -> float:
    """Desired neutral-pose tolerance in radians for posture penalties."""
    if "hip_pitch" in name:
        return 0.25
    if "hip_roll" in name or "hip_yaw" in name:
        return 0.15
    if "knee" in name:
        return 0.25
    if "ankle_pitch" in name:
        return 0.15
    if "ankle_roll" in name:
        return 0.12
    return 0.20


def _huber_normalized(error: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
    """Huber error after scaling each axis by a physically meaningful tolerance."""
    x = error / tolerance.clamp_min(1.0e-6)
    ax = x.abs()
    return torch.where(ax < 1.0, 0.5 * x.square(), ax - 0.5)


def feet_stand_pose_hold(
    env,
    velocity_epsilon=.05,
    asset_cfg=_ROBOT_CFG,
    *,
    left_leg_cfg,
    right_leg_cfg,
):
    """Strong neutral-standing prior for all 12 leg joints.

    Only active for upright, near-zero commanded velocity, and when neither
    leg has any ballet mask.  Unlike the old version this does *not* normalize
    by the huge mechanical joint ranges: every axis has a small posture
    tolerance in radians, so a visibly crooked stand remains expensive.
    """
    robot = env.scene[asset_cfg.name]
    leg_ids = list(left_leg_cfg.joint_ids) + list(right_leg_cfg.joint_ids)
    names = [robot.joint_names[i] for i in leg_ids]
    tol = torch.tensor([_joint_tolerance(n) for n in names], device=env.device, dtype=robot.data.joint_pos.dtype)
    err = robot.data.joint_pos[:, leg_ids] - robot.data.default_joint_pos[:, leg_ids]
    error = _huber_normalized(err, tol).mean(1)

    no_leg_masks = (~_group_has_active_mask(env, left_leg_cfg) & ~_group_has_active_mask(env, right_leg_cfg))
    active = (~inverted(env)) & _stationary_ballet(env, velocity_epsilon) & no_leg_masks
    return error * active.float()


def handstand_leg_pose_penalty(
    env,
    asset_cfg=_ROBOT_CFG,
    *,
    left_leg_cfg,
    right_leg_cfg,
):
    """Keep unmasked legs near the symmetric HANDSTAND_LEGS reference.

    This replaces the old inverted ``home_tracking`` conflict: in handstand
    mode the action reference and this penalty now agree on the same leg pose.
    Masked axes are individually exempted.
    """
    robot = env.scene[asset_cfg.name]
    leg_ids = list(left_leg_cfg.joint_ids) + list(right_leg_cfg.joint_ids)
    names = [robot.joint_names[i] for i in leg_ids]
    target = torch.tensor([handstand_value(n) for n in names], device=env.device, dtype=robot.data.joint_pos.dtype)
    tol = torch.tensor([_joint_tolerance(n) for n in names], device=env.device, dtype=robot.data.joint_pos.dtype)
    err = _huber_normalized(robot.data.joint_pos[:, leg_ids] - target, tol)
    mask = ballet_mask(env)[:, leg_ids]
    unmasked = 1.0 - mask
    count = unmasked.sum(1)
    value = (err * unmasked).sum(1) / count.clamp_min(1.0)
    value = torch.where(count > 0, value, torch.zeros_like(value))
    return value * inverted(env).float()


def feet_flatness_penalty(
    env,
    velocity_epsilon=.05,
    asset_cfg=_ROBOT_CFG,
    *,
    feet_cfg,
    left_leg_cfg,
    right_leg_cfg,
):
    """Penalize toe/heel/edge standing while stationary and upright."""
    robot = env.scene[asset_cfg.name]
    quat = robot.data.site_quat_w[:, feet_cfg.site_ids]
    local_z = torch.zeros_like(quat[..., :3]); local_z[..., 2] = 1.0
    foot_z_w = quat_apply(quat, local_z)
    side_error = foot_z_w[..., :2].square().sum(2)
    eligible = torch.stack((~_group_has_active_mask(env, left_leg_cfg), ~_group_has_active_mask(env, right_leg_cfg)), dim=1)
    eligible &= ((~inverted(env)) & _stationary_ballet(env, velocity_epsilon))[:, None]
    count = eligible.sum(1)
    value = (side_error * eligible.float()).sum(1) / count.clamp_min(1)
    return torch.where(count > 0, value, torch.zeros_like(value))


def pelvis_orientation_penalty(env, asset_cfg=_ROBOT_CFG):
    """Mode-aware pelvis orientation penalty, active in both support modes.

    Upright wants the pelvis local +Z aligned with world +Z; handstand wants it
    aligned with world -Z.  Yaw is unconstrained.  This closes the exploit in
    which pelvis tilt is cancelled by an extreme waist bend while torso_link
    still receives a good ``upright`` reward.
    """
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_link_quat_w
    local_z = torch.zeros_like(quat[..., :3]); local_z[..., 2] = 1.0
    pelvis_z_w = quat_apply(quat, local_z)
    desired_sign = torch.where(inverted(env), -1.0, 1.0)
    # 1-cos(theta) behaves quadratically for small tilt and stays informative
    # all the way to the wrong orientation.
    return 1.0 - desired_sign * pelvis_z_w[:, 2]


def waist_zero_pose_penalty(env, asset_cfg):
    """Strong mask-aware waist regularizer in raw radians.

    The previous mechanical-range normalization made a waist sitting on its
    physical limits too cheap.  Here yaw/roll/pitch squared errors are summed
    directly, so ±0.5 rad bends carry a meaningful cost.
    """
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = robot.data.joint_pos[:, ids]
    unmasked = 1.0 - _selected_command_axes(ballet_mask(env), asset_cfg)
    return (q.square() * unmasked).sum(1)


def linear_velocity_error_penalty(env, asset_cfg=_ROBOT_CFG):
    """Unbounded planar velocity tracking error in heading coordinates."""
    robot = env.scene[asset_cfg.name]
    v = robot.data.root_link_lin_vel_w
    h = robot.data.heading_w
    actual = torch.stack((h.cos()*v[:, 0] + h.sin()*v[:, 1], -h.sin()*v[:, 0] + h.cos()*v[:, 1]), 1)
    cmd = env.command_manager.get_command('ballet')[:, :2]
    return (cmd - actual).square().sum(1)


def upright_feet_clearance(
    env,
    target_height=.08,
    height_sensor_name='foot_height_scan',
    command_name='ballet',
    command_threshold=.08,
    asset_cfg=None,
):
    """Penalize moving feet that stay too low/high during upright locomotion.

    Uses the already-supported ``velocity_mdp.foot_height`` observation helper
    instead of depending on an optional MJLab reward helper.  The error is
    weighted by each foot's planar speed, so the stance foot at ground height
    is not penalized while a swinging/moving foot is encouraged toward the
    requested clearance.
    """
    if asset_cfg is None:
        raise ValueError('asset_cfg with left/right foot site ids is required')
    robot = env.scene[asset_cfg.name]
    height = velocity_mdp.foot_height(env, sensor_name=height_sensor_name)
    # Keep the last dimension aligned to the selected two foot sites.
    height = height.reshape(env.num_envs, -1)[:, :len(asset_cfg.site_ids)]
    planar_speed = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=2)
    error = (height - target_height).square() * planar_speed
    moving = env.command_manager.get_command(command_name)[:, :3].abs().amax(1) >= command_threshold
    active = moving & (~inverted(env))
    return error.sum(1) * active.float()


def upright_support_switch_reward(env, sensor_name='feet_ground_contact', command_threshold=.08):
    """Small event reward for alternating left/right single support.

    No hidden gait phase is introduced.  While commanded to move upright, a
    reward pulse is emitted only when the currently supported foot changes
    from the previous valid single-support foot.  This discourages a static
    one-foot/hopping local optimum without changing the actor observation ABI.
    """
    c = contacts(env, sensor_name, ('left_ankle_roll_link', 'right_ankle_roll_link'))
    single = c[:, 0] ^ c[:, 1]
    side = torch.where(c[:, 0], torch.zeros(env.num_envs, device=env.device, dtype=torch.long), torch.ones(env.num_envs, device=env.device, dtype=torch.long))
    if not hasattr(env, '_flip_last_foot_support'):
        env._flip_last_foot_support = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    reset = env.episode_length_buf <= 1
    env._flip_last_foot_support[reset] = -1
    prev = env._flip_last_foot_support.clone()
    switched = single & (prev >= 0) & (side != prev)
    env._flip_last_foot_support = torch.where(single, side, env._flip_last_foot_support)
    moving = env.command_manager.get_command('ballet')[:, :3].abs().amax(1) >= command_threshold
    return (switched & moving & (~inverted(env))).float()


def track_angular_velocity(env, std, asset_cfg=_ROBOT_CFG):
    command = env.command_manager.get_command('ballet')[:, 2]
    actual = env.scene[asset_cfg.name].data.root_link_ang_vel_w[:, 2]
    return torch.exp(-(command - actual).square() / std**2)


def track_linear_velocity(env, std, asset_cfg=_ROBOT_CFG):
    robot = env.scene[asset_cfg.name]
    v = robot.data.root_link_lin_vel_w
    h = robot.data.heading_w
    actual = torch.stack((h.cos()*v[:, 0] + h.sin()*v[:, 1],
                          -h.sin()*v[:, 0] + h.cos()*v[:, 1]), 1)
    cmd = env.command_manager.get_command('ballet')[:, :2]
    return torch.exp(-(cmd-actual).square().sum(1) / std**2)


def leg_alignment_turn_aware(env, **kwargs):
    score = unmasked_leg_lateral_alignment(env, **kwargs)
    turning = env.command_manager.get_command('ballet')[:, 2].abs() > .15
    return score * (~turning).float()


def feet_alignment(env, **kwargs):
    return leg_alignment_turn_aware(env, **kwargs) * (~inverted(env)).float()


def foot_heading(env, **kwargs):
    return unmasked_foot_heading_alignment(env, **kwargs) * (~inverted(env)).float()


def home_tracking(env, std, upright_cfg, inverted_cfg):
    # Upright: keep unmasked arms near their default pose.
    # Inverted: do NOT pull legs to the standing default; handstand legs are
    # governed by handstand_leg_pose_penalty using HANDSTAND_LEGS instead.
    a = unmasked_home_tracking(env, std, upright_cfg)
    return torch.where(inverted(env), torch.zeros_like(a), a)


def active_masks(env, left_leg_cfg, right_leg_cfg, left_arm_cfg, right_arm_cfg):
    f = inverted(env)
    return tuple(torch.where(f, _group_has_active_mask(env, arm),
                             _group_has_active_mask(env, leg))
                 for leg, arm in [(left_leg_cfg, left_arm_cfg), (right_leg_cfg, right_arm_cfg)])


def support_center(env, feet_cfg, hands_cfg, left_leg_cfg, right_leg_cfg,
                   left_arm_cfg, right_arm_cfg):
    a = support_center_xy_b(env, feet_cfg=feet_cfg,
                           left_leg_cfg=left_leg_cfg, right_leg_cfg=right_leg_cfg)
    b = support_center_xy_b(env, feet_cfg=hands_cfg,
                           left_leg_cfg=left_arm_cfg, right_leg_cfg=right_arm_cfg)
    return torch.where(inverted(env)[:, None], b, a)


def com_support_projection(env, std=.12, velocity_epsilon=.05, **kwargs):
    left, right = active_masks(env, **{k:v for k,v in kwargs.items() if k not in ('feet_cfg','hands_cfg')})
    moving = env.command_manager.get_command('ballet')[:, :3].abs().amax(1) >= velocity_epsilon
    # At zero command, centering COM over the support polygon is useful.
    # During locomotion with no support mask it is deliberately OFF: forcing
    # COM toward the midpoint of both feet/hands suppresses weight transfer and
    # was a strong local optimum for standing instead of walking.
    active = ((~moving) | (left ^ right)) & ~(left & right)
    error = (whole_body_com_xy_b(env) - support_center(env, **kwargs)).square().sum(1)
    return torch.exp(-error / std**2) * active.float()


def contacts(env, sensor_name, body_names):
    sensor = env.scene[sensor_name]
    primary_names = getattr(sensor, "primary_names", None)
    if primary_names is None:
        primary_names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
    ids = [primary_names.index(n) for n in body_names]
    return sensor.data.found[:, ids] > 0


def commanded_contact(env, sensor_name, body_names, left_cfg, right_cfg,
                      mode, velocity_epsilon=.05):
    c = contacts(env, sensor_name, body_names)
    l = _group_has_active_mask(env, left_cfg); r = _group_has_active_mask(env, right_cfg)
    stationary = env.command_manager.get_command('ballet')[:, :3].abs().amax(1) < velocity_epsilon
    no = ~l & ~r; lo = l & ~r; ro = r & ~l
    cost = (stationary & no).float() * (~c).float().sum(1)
    cost += lo.float() * (c[:, 0].float() + (stationary & ~c[:, 1]).float())
    cost += ro.float() * (c[:, 1].float() + (stationary & ~c[:, 0]).float())
    return cost * (inverted(env) == bool(mode)).float()


def slip(env, sensor_name, body_names, asset_cfg, mode):
    c = contacts(env, sensor_name, body_names)
    v = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    return (v.square().sum(2) * c.float()).sum(1) * (inverted(env) == bool(mode)).float()


def forbidden_support(env):
    feet = contacts(env, 'feet_ground_contact', ('left_ankle_roll_link','right_ankle_roll_link'))
    hands = contacts(env, 'hands_ground_contact', ('left_wrist_yaw_link','right_wrist_yaw_link'))
    return torch.where(inverted(env)[:, None], feet, hands).float().sum(1)


def handstand_pose_hold(env, std=.5):
    """Full-arm-chain (shoulder+elbow+wrist) closeness to the fitted
    handstand reference in ``.events.HANDSTAND_LEFT``. Unlike
    ``arm_alignment`` (mask-gated, roll/yaw only, disabled in turns), this
    covers every joint in the reference and is always on while inverted --
    a coarser regularizer to actually hold the fitted shape, not just its
    lateral placement. Weight should stay well below the balance terms so it
    doesn't fight the policy's need to move the arms to recover.
    """
    robot = env.scene['robot']
    names = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
    error = 0.
    count = 0
    for side in ('left', 'right'):
        for joint in names:
            name = f'{side}_{joint}_joint'
            i = robot.joint_names.index(name)
            angle = HANDSTAND_LEFT[joint]
            if side == 'right' and ('roll' in joint or 'yaw' in joint):
                angle = -angle
            error = error + (robot.data.joint_pos[:, i] - angle).square()
            count += 1
    return torch.exp(-error / (count * std**2)) * inverted(env).float()


def head_ground_contact(env):
    """Binary penalty signal for any head contact with terrain."""
    return (env.scene['head_ground_contact'].data.found > 0).reshape(env.num_envs, -1).any(1).float()


def soft_landing(env, mode, **kwargs):
    from mjlab.tasks.velocity.mdp import soft_landing as landing
    return landing(env, **kwargs) * (inverted(env) == bool(mode)).float()


def arm_alignment(env, left_arm_cfg, right_arm_cfg, left_alignment_cfg,
                  right_alignment_cfg, std=.35):
    """Mask-aware lateral arm prior around the fitted handstand, disabled in turns."""
    robot = env.scene['robot']
    mask = []
    scores = []
    for side, group, axes in [('left',left_arm_cfg,left_alignment_cfg),
                              ('right',right_arm_cfg,right_alignment_cfg)]:
        ids = axes.joint_ids
        ref = []
        for i in ids:
            joint = robot.joint_names[i][len(side)+1:-len('_joint')]
            angle = HANDSTAND_LEFT[joint]
            ref.append(-angle if side == 'right' and ('roll' in joint or 'yaw' in joint) else angle)
        target = torch.tensor(ref,device=env.device)
        limits = robot.data.joint_pos_limits[:, ids]
        scale = (.5*(limits[...,1]-limits[...,0])).clamp_min(1e-6)
        error = ((robot.data.joint_pos[:,ids]-target)/scale).square().mean(1)
        scores.append(torch.exp(-error/std**2))
        mask.append(~_group_has_active_mask(env,group))
    eligible = torch.stack(mask,1).float()
    result = (torch.stack(scores,1)*eligible).sum(1)/eligible.sum(1).clamp_min(1.)
    straight = env.command_manager.get_command('ballet')[:,2].abs() <= .15
    return result * (inverted(env) & straight).float()
    
def forbidden_ground_contact_penalty(env) -> torch.Tensor:
    return env.termination_manager.get_term(
        "forbidden_ground_contact"
    ).float()    
    
def forbidden_body_contact_penalty(
    env,
    sensor_name: str = "forbidden_ground_contact",
) -> torch.Tensor:
    found = env.scene[sensor_name].data.found

    return (
        found > 0
    ).reshape(env.num_envs, -1).any(dim=1).float()    
