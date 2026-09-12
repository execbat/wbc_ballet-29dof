"""Command terms for the standalone flip task.

Two term names, matching the command-manager keys used throughout this
task's cfg/reward/observation code:

- ``'ballet'``: the 61D pose/velocity/mask channel, same shape as ballet's
  own command but flip-aware (``FlipBalletCommand`` below).
- ``'flip'``: the 1D upright/inverted mode flag (``FlipCommand`` below).

The mode flag is a separate command class from the composite pose command
-- naming it here needs a little care since the *task* is also called
"flip": ``FlipCommand``/``FlipCommandCfg`` is the boolean mode flag itself
(it was called that before the task was renamed), while the composite
61D command gets the ``FlipBallet*`` name to keep the two distinct.
"""
from dataclasses import dataclass
import numpy as np
import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from wbc_ballet.teleop.flip_protocol import receiver_for, DEFAULT_PORT

_NUM_JOINTS = 29
_TARGET_START = 3
_TARGET_END = _TARGET_START + _NUM_JOINTS
_MASK_START = _TARGET_END
_MASK_END = _MASK_START + _NUM_JOINTS


def blend_joint_targets(current, goal, *, target_scale, target_limit):
    if target_limit <= 0.0:
        raise ValueError("target_limit must be positive")
    blend = min(max(target_scale / target_limit, 0.0), 1.0)
    return torch.lerp(current, goal, blend)


class BalletCommand(CommandTerm):
    """Local 61D velocity/pose/mask generator shared by both flip modes."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, _MASK_END, device=self.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        count = len(env_ids)
        if count == 0:
            return
        robot = self._env.scene["robot"]
        positions = robot.data.joint_pos[env_ids]
        limits = robot.data.joint_pos_limits[env_ids]
        lower, upper = limits[..., 0], limits[..., 1]
        current = (2.0 * (positions - lower) / (upper - lower).clamp_min(1.0e-6) - 1.0).clamp(-1.0, 1.0)
        limit = float(self.cfg.target_limit)
        goal = torch.empty_like(current).uniform_(-limit, limit)
        self._command[env_ids, _TARGET_START:_TARGET_END] = blend_joint_targets(
            current, goal, target_scale=float(self.cfg.target_scale), target_limit=limit
        )
        self._command[env_ids, _MASK_START:_MASK_END] = (
            torch.rand(count, _NUM_JOINTS, device=self.device) < self.cfg.mask_probability
        ).to(self._command.dtype)
        low = torch.tensor(self.cfg.velocity_ranges[0], device=self.device)
        high = torch.tensor(self.cfg.velocity_ranges[1], device=self.device)
        self._command[env_ids, :3] = low + torch.rand(count, 3, device=self.device) * (high - low)
        draw = torch.rand(count, device=self.device)
        self._command[env_ids[draw < 0.20], :3] = 0.0
        self._command[env_ids[(draw >= 0.20) & (draw < 0.45)], :2] = 0.0

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class BalletCommandCfg(CommandTermCfg):
    target_scale: float = 1.0
    target_limit: float = 0.8
    mask_probability: float = 0.15
    velocity_ranges: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    )

    def build(self, env):
        return BalletCommand(self, env)


class FlipCommand(CommandTerm):
    """Upright (0) / inverted-handstand (1) mode flag."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self._blend = torch.zeros_like(self._command)

    @property
    def command(self):
        return self._command

    @property
    def blend(self):
        return self._blend

    def _resample_command(self, env_ids):
        # On reset, match the pose chosen by reset_robot. Later resamples do
        # not move the robot: the policy must perform the transition itself.
        first = self.command_counter[env_ids] == 0
        initial = getattr(self._env, '_flip_mode', None)
        initial = torch.zeros(len(env_ids), device=self.device) if initial is None else initial[env_ids]
        random_mode = (torch.rand(len(env_ids), device=self.device) < self.cfg.probability).float()
        self._command[env_ids, 0] = torch.where(first, initial, random_mode)
        self._blend[env_ids, 0] = torch.where(first, self._command[env_ids, 0], self._blend[env_ids, 0])

    def compute(self, dt):
        super().compute(dt)
        max_delta = 1.0 if self.cfg.transition_duration_s <= 0 else dt / self.cfg.transition_duration_s
        self._blend.add_((self._command - self._blend).clamp(-max_delta, max_delta)).clamp_(0., 1.)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class FlipCommandCfg(CommandTermCfg):
    probability: float = .5
    resampling_time_range: tuple[float, float] = (9.0, 11.0)
    transition_duration_s: float = 2.0

    def build(self, env):
        if not 0 <= self.probability <= 1:
            raise ValueError('flip probability must be in [0, 1]')
        if self.transition_duration_s < 0:
            raise ValueError('transition_duration_s must be non-negative')
        return FlipCommand(self, env)


class FlipBalletCommand(BalletCommand):
    """The 'ballet' pose/velocity/mask command, made flip-mode-aware."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._last_flip = torch.full((self.num_envs,), -1., device=self.device)
        self._limb_groups = []
        for mode, parts in [(0, ('hip','knee','ankle')), (1, ('shoulder','elbow','wrist'))]:
            groups = [torch.tensor([i for i,n in enumerate(env.scene['robot'].joint_names)
                       if n.startswith(side+'_') and any(p in n for p in parts)],
                       device=self.device, dtype=torch.long) for side in ('left','right')]
            self._limb_groups.append((mode, groups))

    def _ensure_support_masks(self, env_ids):
        if not len(env_ids):
            return
        mode = self._env.command_manager.get_command('flip')[env_ids, 0]
        mask = self._command[env_ids, 32:61].clone()
        for value, groups in self._limb_groups:
            both = (mask[:,groups[0]]>=.5).any(1) & (mask[:,groups[1]]>=.5).any(1)
            both &= mode == value
            clear_left = torch.rand(len(env_ids), device=self.device) < .5
            for side, cols in enumerate(groups):
                rows = torch.where(both & (clear_left if side == 0 else ~clear_left))[0]
                mask[rows[:,None], cols[None,:]] = 0
        self._command[env_ids,32:61] = mask
        self._last_flip[env_ids] = mode

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if not len(env_ids):
            return
        low = torch.tensor(self.cfg.velocity_ranges[0], device=self.device)
        high = torch.tensor(self.cfg.velocity_ranges[1], device=self.device)
        self._command[env_ids,:3] = low + torch.rand(len(env_ids),3,device=self.device)*(high-low)
        draw = torch.rand(len(env_ids),device=self.device)
        self._command[env_ids[draw<.20],:3] = 0.
        self._command[env_ids[(draw>=.20)&(draw<.45)],:2] = 0.
        self._ensure_support_masks(env_ids)

    def _update_command(self):
        # flip is computed FIRST by the manager. Repair newly incompatible
        # random masks immediately, even if ballet's timer has not expired.
        mode = self._env.command_manager.get_command('flip')[:,0]
        self._ensure_support_masks(torch.where(mode != self._last_flip)[0])


@dataclass(kw_only=True)
class FlipBalletCommandCfg(BalletCommandCfg):
    def build(self, env):
        return FlipBalletCommand(self, env)


class UdpFlipCommand(FlipCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.receiver = receiver_for(env,cfg.host,cfg.port)

    def _resample_command(self,env_ids):
        self._command[env_ids,0] = self.receiver.packet.flip
        first = self.command_counter[env_ids] == 0
        self._blend[env_ids,0] = torch.where(first, self._command[env_ids,0], self._blend[env_ids,0])

    def compute(self,dt):
        self._command[:,0] = self.receiver.poll().flip
        max_delta = 1.0 if self.cfg.transition_duration_s <= 0 else dt / self.cfg.transition_duration_s
        self._blend.add_((self._command-self._blend).clamp(-max_delta,max_delta)).clamp_(0.,1.)


@dataclass(kw_only=True)
class UdpFlipCommandCfg(FlipCommandCfg):
    host: str = '127.0.0.1'
    port: int = DEFAULT_PORT

    def build(self,env):
        return UdpFlipCommand(self,env)


class UdpFlipBalletCommand(BalletCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.receiver = receiver_for(env,cfg.host,cfg.port)

    def _copy_packet(self,env_ids):
        p = self.receiver.packet
        values = np.concatenate((p.velocity,p.targets,p.mask))
        self._command[env_ids] = torch.as_tensor(values,device=self.device)

    def _resample_command(self,env_ids):
        self._copy_packet(env_ids)

    def compute(self,dt):
        del dt
        self._copy_packet(slice(None))


@dataclass(kw_only=True)
class UdpFlipBalletCommandCfg(BalletCommandCfg):
    host: str = '127.0.0.1'
    port: int = DEFAULT_PORT
    resampling_time_range: tuple[float,float] = (1e12,1e12)

    def build(self,env):
        return UdpFlipBalletCommand(self,env)
