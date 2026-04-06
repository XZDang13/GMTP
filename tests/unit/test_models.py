import pytest
import torch
import torch.nn as nn

from gmtp.models import Critic, FiLMResActor, build_actor, get_actor_kwargs, infer_film_res_blocks
from gmtp.models.film import FiLMResStack
from gmtp.runtime.checkpoints import CheckpointV2, build_training_checkpoint
from gmtp.runtime.policy import load_actor_from_checkpoint, resolve_checkpoint_actor_spec


def _joint_params(action_dim: int = 3) -> dict[str, torch.Tensor | list[str]]:
    return {
        "joint_names": [f"joint_{idx}" for idx in range(action_dim)],
        "joint_effort_limits": torch.ones(action_dim),
        "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * action_dim),
        "joint_stiffness": torch.ones(action_dim),
        "joint_damping": torch.full((action_dim,), 0.1),
        "action_offset": torch.zeros(action_dim),
        "action_scale": torch.ones(action_dim),
    }


def _actor_obs_dims(action_dim: int, robot_window_length: int = 1) -> tuple[int, int]:
    motion_obs_dim = 3 + 2 * action_dim
    robot_step_dim = 6 + 3 * action_dim
    return motion_obs_dim, robot_step_dim * robot_window_length


def test_film_res_actor_forward_returns_step():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=3,
        num_blocks=5,
    )

    step = actor(
        {
            "robot_obs": torch.randn(4, robot_obs_dim),
            "motion_obs": torch.randn(4, motion_obs_dim),
        }
    )

    assert step.action.shape == (4, 3)
    assert step.log_prob.shape == (4,)
    assert actor.num_blocks == 5
    assert len(actor.blocks) == 5


def test_build_actor_constructs_film_res_with_requested_depth():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3, robot_window_length=4)
    actor = build_actor(
        {"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
        "film_res",
        action_dim=3,
        actor_kwargs={"num_blocks": 4, "robot_window_length": 4},
    )

    assert isinstance(actor, FiLMResActor)
    assert actor.num_blocks == 4
    assert len(actor.blocks) == 4
    assert get_actor_kwargs(actor, "film_res") == {"num_blocks": 4, "robot_window_length": 4}
    assert infer_film_res_blocks(actor.state_dict()) == 4


def test_film_res_actor_reshapes_windowed_robot_obs_from_ref2act_layout():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        robot_window_length=4,
    )
    robot_obs = torch.arange(robot_obs_dim, dtype=torch.float32).reshape(1, robot_obs_dim)

    reshaped = actor._reshape_robot_obs(robot_obs)

    assert reshaped.shape == (1, 4, 12)
    torch.testing.assert_close(
        reshaped,
        torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 12.0, 13.0, 14.0, 24.0, 25.0, 32.0, 33.0, 40.0, 41.0],
                    [3.0, 4.0, 5.0, 15.0, 16.0, 17.0, 26.0, 27.0, 34.0, 35.0, 42.0, 43.0],
                    [6.0, 7.0, 8.0, 18.0, 19.0, 20.0, 28.0, 29.0, 36.0, 37.0, 44.0, 45.0],
                    [9.0, 10.0, 11.0, 21.0, 22.0, 23.0, 30.0, 31.0, 38.0, 39.0, 46.0, 47.0],
                ]
            ]
        ),
    )


def test_film_res_actor_robot_encoder_uses_temporal_conv1d():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        robot_window_length=4,
    )

    conv_layers = [module for module in actor.robot_encoder if isinstance(module, nn.Conv1d)]

    assert len(conv_layers) == 2
    assert conv_layers[0].in_channels == actor.robot_step_dim
    assert conv_layers[0].out_channels == 256
    assert conv_layers[1].in_channels == 256
    assert conv_layers[1].out_channels == 256


def test_film_res_actor_forward_supports_windowed_robot_obs():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=3,
        robot_window_length=4,
    )

    step = actor(
        {
            "robot_obs": torch.randn(3, robot_obs_dim),
            "motion_obs": torch.randn(3, motion_obs_dim),
        }
    )

    assert step.action.shape == (3, 2)
    assert step.log_prob.shape == (3,)


def test_film_res_stack_accumulates_residuals_layer_by_layer():
    class RecordingBlock(nn.Module):
        def __init__(self, value: torch.Tensor):
            super().__init__()
            self.res_scale = nn.Parameter(torch.ones(4))
            self.value = value
            self.last_input: torch.Tensor | None = None

        def forward(self, x, cond):
            self.last_input = x.clone()
            return self.value

    stack = FiLMResStack(dim=4, cond_dim=4, num_layers=2)
    block_1 = RecordingBlock(torch.full((2, 4), 2.0))
    block_2 = RecordingBlock(torch.full((2, 4), 3.0))
    stack.blocks = nn.ModuleList([block_1, block_2])

    x0 = torch.zeros(2, 4)
    output = stack(x0, torch.randn(2, 4))

    torch.testing.assert_close(block_1.last_input, x0)
    torch.testing.assert_close(block_2.last_input, torch.full((2, 4), 2.0))
    torch.testing.assert_close(output, torch.full((2, 4), 5.0))


def test_film_res_stack_uses_current_state_for_shortcut_and_branch():
    class RecordingBlock(nn.Module):
        def __init__(self, value: torch.Tensor, scale: float):
            super().__init__()
            self.res_scale = nn.Parameter(torch.full((4,), scale))
            self.value = value
            self.last_input: torch.Tensor | None = None

        def forward(self, x, cond):
            self.last_input = x.clone()
            return self.value

    stack = FiLMResStack(dim=4, cond_dim=4, num_layers=1)
    delta = torch.full((2, 4), 8.0)
    block = RecordingBlock(delta, scale=0.25)
    stack.blocks = nn.ModuleList([block])

    x0 = torch.randn(2, 4)
    output = stack(x0, torch.randn(2, 4))

    torch.testing.assert_close(block.last_input, x0)
    torch.testing.assert_close(output, x0 + 0.25 * delta)


def test_checkpoint_spec_preserves_num_blocks():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=4)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 4, "robot_window_length": 1}


def test_checkpoint_override_replaces_num_blocks():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=4)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint, num_blocks=5)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 5, "robot_window_length": 1}


def test_checkpoint_spec_defaults_robot_window_length_when_missing():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 2, "robot_window_length": 1}


def test_load_actor_from_checkpoint_restores_film_res_weights():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 2, "robot_window_length": 1}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    loaded_actor, actor_type, actor_kwargs = load_actor_from_checkpoint(
        checkpoint,
        obs_dims={"robot": robot_obs_dim, "motion": motion_obs_dim, "policy": motion_obs_dim + robot_obs_dim},
        action_dim=3,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_actor, FiLMResActor)
    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 2, "robot_window_length": 1}
    torch.testing.assert_close(
        loaded_actor.state_dict()["stack.blocks.0.res_scale"],
        actor.state_dict()["stack.blocks.0.res_scale"],
    )


def test_checkpoint_spec_rejects_legacy_film_attn_res_actor_type():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=3)
    actor = FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_attn_res", "actor_kwargs": {"num_blocks": 2, "robot_window_length": 1}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    with pytest.raises(ValueError, match="film_res"):
        resolve_checkpoint_actor_spec(checkpoint)
