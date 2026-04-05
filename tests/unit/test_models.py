import torch
import torch.nn as nn

from gmtp.models import Critic, FiLMAttnResActor, build_actor, get_actor_kwargs, infer_film_res_blocks
from gmtp.models.adain import BlockAttnResFiLMStack
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


def test_film_attn_res_actor_forward_returns_step():
    actor = FiLMAttnResActor(
        robot_obs_dim=7,
        motion_obs_dim=5,
        action_dim=3,
        num_blocks=5,
        attn_block_size=2,
    )

    step = actor(
        {
            "robot_obs": torch.randn(4, 7),
            "motion_obs": torch.randn(4, 5),
        }
    )

    assert step.action.shape == (4, 3)
    assert step.log_prob.shape == (4,)
    assert actor.attn_block_size == 2
    for query_proj in actor.query_projs:
        torch.testing.assert_close(query_proj.weight, torch.zeros_like(query_proj.weight))
        torch.testing.assert_close(query_proj.bias, torch.zeros_like(query_proj.bias))


def test_build_actor_constructs_film_attn_res_with_requested_depth():
    actor = build_actor(
        {"robot": 7, "motion": 5, "policy": 12},
        "film_attn_res",
        action_dim=3,
        actor_kwargs={"num_blocks": 4, "attn_block_size": 2},
    )

    assert isinstance(actor, FiLMAttnResActor)
    assert actor.num_blocks == 4
    assert actor.attn_block_size == 2
    assert len(actor.blocks) == 4
    assert get_actor_kwargs(actor, "film_attn_res") == {"num_blocks": 4, "attn_block_size": 2}
    assert infer_film_res_blocks(actor.state_dict()) == 4


def test_block_attn_res_film_stack_flushes_partial_blocks_and_keeps_initial_source():
    class RecordingBlockAttnRes(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[int, bool]] = []
            self.first_completed_block: torch.Tensor | None = None

        def forward(self, completed_blocks, partial_block, query):
            self.calls.append((len(completed_blocks), partial_block is not None))
            if self.first_completed_block is None:
                self.first_completed_block = completed_blocks[0].clone()
            return completed_blocks[0]

    stack = BlockAttnResFiLMStack(dim=8, cond_dim=8, num_layers=5, block_size=2)
    recorder = RecordingBlockAttnRes()
    stack.attn_res = recorder

    x0 = torch.randn(3, 8)
    cond = torch.randn(3, 8)
    _ = stack(x0, cond)

    assert recorder.calls == [(1, False), (1, True), (2, False), (2, True), (3, False)]
    torch.testing.assert_close(recorder.first_completed_block, x0)


def test_block_attn_res_film_stack_uses_same_mixed_state_for_shortcut_and_branch():
    class ConstantBlockAttnRes(nn.Module):
        def __init__(self, value: torch.Tensor):
            super().__init__()
            self.value = value

        def forward(self, completed_blocks, partial_block, query):
            return self.value

    class RecordingBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.res_scale = nn.Parameter(torch.ones(4))
            self.last_input: torch.Tensor | None = None

        def forward(self, x, cond):
            self.last_input = x.clone()
            return torch.zeros_like(x)

    stack = BlockAttnResFiLMStack(dim=4, cond_dim=4, num_layers=1, block_size=1)
    expected_h = torch.randn(2, 4)
    block = RecordingBlock()
    stack.attn_res = ConstantBlockAttnRes(expected_h)
    stack.blocks = nn.ModuleList([block])

    output = stack(torch.randn(2, 4), torch.randn(2, 4))

    torch.testing.assert_close(block.last_input, expected_h)
    torch.testing.assert_close(output, expected_h)


def test_checkpoint_spec_preserves_num_blocks_and_attn_block_size():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=4, attn_block_size=2)
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

    assert actor_type.value == "film_attn_res"
    assert actor_kwargs == {"num_blocks": 4, "attn_block_size": 2}


def test_checkpoint_override_replaces_attn_block_size():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=4, attn_block_size=2)
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

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint, attn_block_size=5)

    assert actor_type.value == "film_attn_res"
    assert actor_kwargs == {"num_blocks": 4, "attn_block_size": 5}


def test_load_actor_from_checkpoint_restores_film_attn_res_weights():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=2, attn_block_size=3)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_attn_res", "actor_kwargs": {"num_blocks": 2, "attn_block_size": 3}},
        model={
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        env={},
        artifacts={},
    )

    loaded_actor, actor_type, actor_kwargs = load_actor_from_checkpoint(
        checkpoint,
        obs_dims={"robot": 7, "motion": 5, "policy": 12},
        action_dim=3,
        device=torch.device("cpu"),
    )

    assert isinstance(loaded_actor, FiLMAttnResActor)
    assert actor_type.value == "film_attn_res"
    assert actor_kwargs == {"num_blocks": 2, "attn_block_size": 3}
    torch.testing.assert_close(
        loaded_actor.state_dict()["stack.blocks.0.res_scale"],
        actor.state_dict()["stack.blocks.0.res_scale"],
    )
