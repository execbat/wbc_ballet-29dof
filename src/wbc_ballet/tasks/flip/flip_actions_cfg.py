"""One shared 29-DoF residual action term for both support modes."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from wbc_ballet.robots.g1 import G1_ACTION_SCALE
from wbc_ballet.utils.configclass import configclass

from .mdp.events import handstand_value


class FlipJointPositionAction(JointPositionAction):
    """Keep one action space while changing only its zero-action reference."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._home = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._handstand = self._home.clone()
        for index, name in enumerate(self._target_names):
            value = handstand_value(name)
            if value is None:
                raise KeyError(f"missing handstand reference for {name!r}")
            self._handstand[:, index] = value

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        term = self._env.command_manager.get_term("flip")
        blend = getattr(term, "blend", term.command).to(dtype=self._home.dtype)
        reference = self._home + blend * (self._handstand - self._home)
        self._processed_actions = self._raw_actions * self._scale + reference
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@dataclass(kw_only=True)
class FlipJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return FlipJointPositionAction(self, env)


@configclass
class FlipActionsCfg:
    joint_pos: FlipJointPositionActionCfg | None = FlipJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_ACTION_SCALE,
        use_default_offset=True,
    )
