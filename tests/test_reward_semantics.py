from types import SimpleNamespace

import torch

from wbc_ballet.mdp import rewards as rewards_module
from wbc_ballet.mdp.events import print_joint_observation_table
from wbc_ballet.mdp.metrics import (
    invalid_physics_value_type,
    nonfinite_physics_component,
    nonfinite_policy_action,
)
from wbc_ballet.mdp.observations import (
    masked_ballet_targets,
    support_center_xy_b,
    whole_body_com_xy_b,
)
from wbc_ballet.mdp.rewards import (
    com_support_projection_tracking,
    commanded_leg_ground_contact,
    masked_pose_tracking,
    unmasked_home_tracking,
)
from wbc_ballet.mdp.terminations import non_finite_state_or_action, pelvis_height_below


class _CommandManager:
    def __init__(self, command: torch.Tensor) -> None:
        self.command = command

    def get_command(self, _name: str) -> torch.Tensor:
        return self.command


class _Scene(dict):
    def __init__(self, robot: SimpleNamespace, origins: torch.Tensor) -> None:
        super().__init__(robot=robot)
        self.env_origins = origins


def _fake_env(
    *, mask: torch.Tensor, targets: torch.Tensor | None = None
) -> tuple[SimpleNamespace, SimpleNamespace]:
    num_envs, num_joints = mask.shape
    if targets is None:
        targets = torch.zeros(num_envs, num_joints)
    velocity = torch.zeros(num_envs, 3)
    command = torch.cat((velocity, targets, mask), dim=1)
    data = SimpleNamespace(
        joint_pos=torch.zeros(num_envs, num_joints),
        default_joint_pos=torch.zeros(num_envs, num_joints),
        joint_pos_limits=torch.tensor([-1.0, 1.0]).repeat(num_envs, num_joints, 1),
        root_link_pos_w=torch.zeros(num_envs, 3),
    )
    robot = SimpleNamespace(
        data=data,
        joint_names=[f"joint_{index}" for index in range(num_joints)],
    )
    scene = _Scene(robot, torch.zeros(num_envs, 3))
    env = SimpleNamespace(
        scene=scene,
        command_manager=_CommandManager(command),
        num_envs=num_envs,
        common_step_counter=123,
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=slice(None))
    return env, asset_cfg


def test_masked_tracking_has_no_free_reward_when_all_masks_are_zero() -> None:
    env, asset_cfg = _fake_env(mask=torch.zeros(2, 29))
    reward = masked_pose_tracking(env, asset_cfg=asset_cfg)
    torch.testing.assert_close(reward, torch.zeros(2))


def test_target_observation_is_zero_where_mask_is_zero() -> None:
    targets = torch.linspace(-1.0, 1.0, 29).unsqueeze(0)
    mask = torch.zeros(1, 29)
    mask[:, (0, 15, 28)] = 1.0
    env, _ = _fake_env(mask=mask, targets=targets)
    torch.testing.assert_close(masked_ballet_targets(env), targets * mask)


def test_joint_table_reports_only_the_policy_visible_target(capsys) -> None:
    targets = torch.linspace(-1.0, 1.0, 29).unsqueeze(0)
    mask = torch.zeros(1, 29)
    mask[:, 0] = 1.0
    env, asset_cfg = _fake_env(mask=mask, targets=targets)

    print_joint_observation_table(env, None, env_index=0, asset_cfg=asset_cfg)

    output = capsys.readouterr().out
    assert "env=0 | step=123" in output
    assert "q_home" in output and "delta" in output
    assert "joint_0" in output and "   -1.0000" in output
    joint_one_line = next(line for line in output.splitlines() if "joint_1 " in line)
    assert joint_one_line.endswith("    0.0000")


def test_unmasked_home_reward_disappears_when_all_selected_axes_are_masked() -> None:
    env, asset_cfg = _fake_env(mask=torch.ones(2, 29))
    reward = unmasked_home_tracking(env, asset_cfg=asset_cfg)
    torch.testing.assert_close(reward, torch.zeros(2))


def test_pelvis_height_is_relative_to_environment_origin() -> None:
    env, asset_cfg = _fake_env(mask=torch.zeros(2, 29))
    env.scene["robot"].data.root_link_pos_w[:, 2] = torch.tensor([0.19, 1.25])
    env.scene.env_origins[:, 2] = torch.tensor([0.0, 1.0])
    terminated = pelvis_height_below(env, minimum_height=0.2, asset_cfg=asset_cfg)
    torch.testing.assert_close(terminated, torch.tensor([True, False]))


def test_support_center_uses_masked_leg_group_semantics() -> None:
    mask = torch.zeros(4, 29)
    mask[1, 0] = 1.0  # one left-leg axis -> right foot only
    mask[2, 6] = 1.0  # one right-leg axis -> left foot only
    mask[3, (0, 6)] = 1.0  # both legs commanded -> both feet
    command = torch.cat((torch.zeros(4, 3), torch.zeros(4, 29), mask), dim=1)

    root_pos = torch.zeros(4, 3)
    feet = torch.tensor([[[0.2, 0.1, 0.0], [0.2, -0.1, 0.0]]]).repeat(4, 1, 1)
    subtree_com = torch.zeros(4, 1, 3)
    subtree_com[:, 0, :2] = torch.tensor([0.2, 0.0])
    data = SimpleNamespace(
        root_link_pos_w=root_pos,
        heading_w=torch.zeros(4),
        site_pos_w=feet,
        indexing=SimpleNamespace(root_body_id=0),
        data=SimpleNamespace(subtree_com=subtree_com),
    )
    env = SimpleNamespace(
        scene=_Scene(SimpleNamespace(data=data), torch.zeros(4, 3)),
        command_manager=_CommandManager(command),
    )
    robot_cfg = SimpleNamespace(name="robot")
    feet_cfg = SimpleNamespace(site_ids=[0, 1])
    left_cfg = SimpleNamespace(joint_ids=list(range(6)))
    right_cfg = SimpleNamespace(joint_ids=list(range(6, 12)))

    support = support_center_xy_b(
        env,
        asset_cfg=robot_cfg,
        feet_cfg=feet_cfg,
        left_leg_cfg=left_cfg,
        right_leg_cfg=right_cfg,
    )
    expected = torch.tensor([[0.2, 0.0], [0.2, -0.1], [0.2, 0.1], [0.2, 0.0]])
    torch.testing.assert_close(support, expected)
    torch.testing.assert_close(
        whole_body_com_xy_b(env, asset_cfg=robot_cfg),
        torch.tensor([[0.2, 0.0]]).repeat(4, 1),
    )

    reward = com_support_projection_tracking(
        env,
        std=0.1,
        asset_cfg=robot_cfg,
        feet_cfg=feet_cfg,
        left_leg_cfg=left_cfg,
        right_leg_cfg=right_cfg,
    )
    torch.testing.assert_close(reward[[0, 3]], torch.ones(2))
    assert torch.all(reward[[1, 2]] < 1.0)


def test_commanded_leg_contact_penalty_uses_side_masks_and_sensor_names(monkeypatch) -> None:
    class _FakeContactSensor:
        def __init__(self, found: torch.Tensor) -> None:
            self.data = SimpleNamespace(found=found)
            # Deliberately reverse the columns to verify name-based mapping.
            self.primary_names = ["right_ankle_roll_link", "left_ankle_roll_link"]

    monkeypatch.setattr(rewards_module, "ContactSensor", _FakeContactSensor)

    mask = torch.zeros(6, 29)
    mask[1, 0] = 1.0
    mask[2, 6] = 1.0
    mask[3, (0, 6)] = 1.0
    mask[4, 0] = 1.0
    mask[5, 6] = 1.0
    env, _ = _fake_env(mask=mask)
    env.scene["feet_ground_contact"] = _FakeContactSensor(
        torch.tensor(
            [
                [1, 1],  # no masks: no penalty
                [1, 0],  # left commanded, only right touches
                [0, 1],  # right commanded, only left touches
                [1, 1],  # both commanded and both touch
                [0, 1],  # left commanded and left touches
                [1, 0],  # right commanded and right touches
            ]
        )
    )
    left_cfg = SimpleNamespace(joint_ids=list(range(6)))
    right_cfg = SimpleNamespace(joint_ids=list(range(6, 12)))

    penalty = commanded_leg_ground_contact(
        env,
        "feet_ground_contact",
        left_leg_cfg=left_cfg,
        right_leg_cfg=right_cfg,
    )
    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 0.0, 2.0, 1.0, 1.0]))


def test_nonfinite_diagnostics_identify_type_component_and_action() -> None:
    data = SimpleNamespace(
        qpos=torch.zeros(3, 2),
        qvel=torch.zeros(3, 2),
        qacc=torch.zeros(3, 2),
        qacc_warmstart=torch.zeros(3, 2),
        sensordata=torch.zeros(3, 2),
    )
    data.qvel[1, 0] = torch.nan
    data.sensordata[2, 1] = torch.inf
    actions = torch.zeros(3, 2)
    actions[0, 0] = torch.nan
    env = SimpleNamespace(
        sim=SimpleNamespace(data=data),
        action_manager=SimpleNamespace(action=actions),
        num_envs=3,
        device="cpu",
    )

    torch.testing.assert_close(
        invalid_physics_value_type(env, "nan"), torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(
        invalid_physics_value_type(env, "inf"), torch.tensor([0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        nonfinite_physics_component(env, "qvel"), torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(nonfinite_policy_action(env), torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(non_finite_state_or_action(env), torch.tensor([True, True, True]))
