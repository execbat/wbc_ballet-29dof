"""All reward terms owned by the standalone flip task."""
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply
from .observations import (
    _group_has_active_mask, _world_xy_to_base_heading,
    ballet_mask, ballet_targets, joint_pos_normalized,
    whole_body_com_xy_b, support_center_xy_b,
)
from .events import HANDSTAND_LEFT

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
    # Arms free for inverted locomotion; legs free for upright locomotion.
    a = unmasked_home_tracking(env, std, upright_cfg)
    b = unmasked_home_tracking(env, std, inverted_cfg)
    return torch.where(inverted(env), b, a)


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
    # Hand-support balance is precarious even standing still -- unlike two
    # feet, two hands don't give a trivially stable base for free. Always on
    # while inverted, not just while moving or single-support.
    active = (inverted(env) | moving | (left ^ right)) & ~(left & right)
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
