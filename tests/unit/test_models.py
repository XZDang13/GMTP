import re

import torch
import torch.nn as nn
from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer

from gmtp.models import (
    Critic,
    FiLMAttnResActor,
    FiLMResActor,
    build_actor,
    get_actor_kwargs,
    infer_film_res_blocks,
    policy_state_for_storage,
    policy_state_from_storage,
    RecurrentActor,
)
from gmtp.models.adain import BlockAttnResFiLMStack, FiLMResBlock
from gmtp.runtime.checkpoints import CheckpointV2, build_training_checkpoint
from gmtp.runtime.policy import load_actor_from_checkpoint, resolve_checkpoint_actor_spec


def _legacy_adain_res_state_dict(actor: FiLMResActor) -> dict[str, torch.Tensor]:
    legacy_state: dict[str, torch.Tensor] = {}
    for key, value in actor.state_dict().items():
        match = re.match(r"^blocks\.(\d+)\.(.+)$", key)
        if match is None:
            legacy_state[key] = value.clone()
            continue

        block_idx = int(match.group(1)) + 1
        suffix = match.group(2)
        if suffix == "res_scale":
            continue
        if suffix.startswith("modulation.affine."):
            suffix = f"adain.style.{suffix.removeprefix('modulation.affine.')}"
        legacy_state[f"block_{block_idx}.{suffix}"] = value.clone()
    return legacy_state


def test_recurrent_actor_forward_single_step_returns_step_and_state():
    actor = RecurrentActor(obs_dim=7, action_dim=3, hidden_size=8, num_layers=2)
    obs = {"obs": torch.randn(4, 7)}

    step, next_state = actor(obs)

    assert step.action.shape == (4, 3)
    assert step.log_prob.shape == (4,)
    assert next_state.shape == (2, 4, 8)


def test_recurrent_actor_forward_sequence_preserves_time_and_batch_shapes():
    actor = RecurrentActor(obs_dim=5, action_dim=2, hidden_size=8, num_layers=2)
    obs = {"obs": torch.randn(3, 4, 5)}
    actions = torch.randn(3, 4, 2)
    episode_starts = torch.zeros(3, 4, dtype=torch.bool)

    step, next_state = actor(
        obs,
        action=actions,
        initial_state=actor.get_initial_state(batch_size=4),
        episode_starts=episode_starts,
    )

    assert step.action.shape == (3, 4, 2)
    assert step.log_prob.shape == (3, 4)
    assert next_state.shape == (2, 4, 8)


def test_recurrent_actor_episode_starts_reset_selected_hidden_states():
    torch.manual_seed(0)
    actor = RecurrentActor(obs_dim=6, action_dim=2, hidden_size=8, num_layers=2)
    obs = torch.randn(2, 6)
    actions = torch.randn(2, 2)
    initial_state = torch.randn(2, 2, 8)
    episode_starts = torch.tensor([True, False])

    _, next_state = actor(
        {"obs": obs},
        action=actions,
        initial_state=initial_state.clone(),
        episode_starts=episode_starts,
    )
    _, reset_state = actor(
        {"obs": obs[:1]},
        action=actions[:1],
        initial_state=torch.zeros_like(initial_state[:, :1]),
        episode_starts=torch.tensor([True]),
    )
    _, carried_state = actor(
        {"obs": obs[1:]},
        action=actions[1:],
        initial_state=initial_state[:, 1:2],
        episode_starts=torch.tensor([False]),
    )

    torch.testing.assert_close(next_state[:, :1], reset_state)
    torch.testing.assert_close(next_state[:, 1:2], carried_state)


def test_recurrent_actor_sequence_normalizer_flattens_time_and_batch():
    actor = RecurrentActor(obs_dim=5, action_dim=2, hidden_size=8, num_layers=2)
    obs = {"obs": torch.randn(3, 2, 5)}
    episode_starts = torch.zeros(3, 2, dtype=torch.bool)

    step, next_state = actor(
        obs,
        initial_state=actor.get_initial_state(2),
        episode_starts=episode_starts,
        update_normlizer=True,
    )

    assert step.log_prob.shape == (3, 2)
    assert next_state.shape == (2, 2, 8)
    assert actor.normlizer.mean.shape == (5,)
    assert actor.normlizer.count.item() == 7
    assert torch.isfinite(actor.normlizer.mean).all()
    assert torch.isfinite(actor.normlizer.var).all()


def test_policy_rnn_state_roundtrip_through_sequence_batches():
    policy_state = torch.randn(2, 3, 8)
    stored = policy_state_for_storage(policy_state)
    restored = policy_state_from_storage(stored, torch.device("cpu"))
    torch.testing.assert_close(restored, policy_state)


def test_recurrent_policy_loss_accepts_sequence_batches_from_rollout_buffer():
    torch.manual_seed(1)
    actor = RecurrentActor(obs_dim=4, action_dim=2, hidden_size=8, num_layers=2)
    buffer = ReplayBuffer(num_envs=2, steps=3)
    buffer.create_storage_space("policy_observations", (4,), torch.float32)
    buffer.create_storage_space("actions", (2,), torch.float32)
    buffer.create_storage_space("log_probs", (), torch.float32)
    buffer.create_storage_space("advantages", (), torch.float32)
    buffer.create_storage_space("episode_starts", (), torch.bool)
    buffer.create_storage_space("policy_rnn_state", (2, 8), torch.float32)

    policy_state = actor.get_initial_state(2)
    episode_starts = torch.ones(2, dtype=torch.bool)
    for _ in range(3):
        observations = {"obs": torch.randn(2, 4)}
        step, next_state = actor(
            observations,
            initial_state=policy_state,
            episode_starts=episode_starts,
        )
        buffer.add_records(
            {
                "policy_observations": observations["obs"],
                "actions": step.action,
                "log_probs": step.log_prob,
                "advantages": torch.ones(2),
                "episode_starts": episode_starts,
                "policy_rnn_state": policy_state_for_storage(policy_state),
            }
        )
        policy_state = next_state
        episode_starts = torch.zeros(2, dtype=torch.bool)

    batch = next(
        buffer.sample_sequence_batches(
            key_names=["policy_observations", "actions", "log_probs", "advantages", "episode_starts"],
            state_keys=["policy_rnn_state"],
            seq_len=3,
            batch_size=2,
            shuffle=False,
        )
    )

    loss_dict = PPO.compute_policy_loss_recurrent(
        actor,
        batch["log_probs"],
        {"obs": batch["policy_observations"]},
        batch["actions"],
        batch["advantages"],
        0.2,
        episode_starts=batch["episode_starts"],
        initial_state=policy_state_from_storage(batch["policy_rnn_state_init"], torch.device("cpu")),
        valid_mask=batch["valid_mask"],
    )

    assert torch.isfinite(loss_dict["loss"])
    assert torch.isfinite(loss_dict["entropy"])
    assert torch.isfinite(loss_dict["kl_divergence"])


def test_film_res_actor_forward_returns_step():
    actor = FiLMResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=2)

    step = actor(
        {
            "robot_obs": torch.randn(4, 7),
            "motion_obs": torch.randn(4, 5),
        }
    )

    assert step.action.shape == (4, 3)
    assert step.log_prob.shape == (4,)


def test_film_res_block_forward_returns_branch_output():
    block = FiLMResBlock(dim=8, cond_dim=6)
    dx = block(torch.randn(4, 8), torch.randn(4, 6))

    assert dx.shape == (4, 8)


def test_build_actor_constructs_film_res_with_requested_depth():
    actor = build_actor(
        {"robot": 7, "motion": 5, "policy": 12},
        "film_res",
        action_dim=3,
        actor_kwargs={"num_blocks": 4},
    )

    assert isinstance(actor, FiLMResActor)
    assert actor.num_blocks == 4
    assert len(actor.blocks) == 4
    assert get_actor_kwargs(actor, "film_res") == {"num_blocks": 4}
    assert infer_film_res_blocks(actor.state_dict()) == 4


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


def test_film_res_checkpoint_spec_preserves_num_blocks():
    actor = FiLMResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=4)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor_type="film_res",
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params={
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "joint_effort_limits": torch.ones(3),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * 3),
            "joint_stiffness": torch.ones(3),
            "joint_damping": torch.full((3,), 0.1),
            "action_offset": torch.zeros(3),
            "action_scale": torch.ones(3),
        },
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 4}


def test_film_attn_res_checkpoint_spec_preserves_num_blocks_and_attn_block_size():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=4, attn_block_size=2)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor_type="film_attn_res",
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params={
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "joint_effort_limits": torch.ones(3),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * 3),
            "joint_stiffness": torch.ones(3),
            "joint_damping": torch.full((3,), 0.1),
            "action_offset": torch.zeros(3),
            "action_scale": torch.ones(3),
        },
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint)

    assert actor_type.value == "film_attn_res"
    assert actor_kwargs == {"num_blocks": 4, "attn_block_size": 2}


def test_film_attn_res_checkpoint_override_replaces_attn_block_size():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=4, attn_block_size=2)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor_type="film_attn_res",
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params={
            "joint_names": ["joint_0", "joint_1", "joint_2"],
            "joint_effort_limits": torch.ones(3),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * 3),
            "joint_stiffness": torch.ones(3),
            "joint_damping": torch.full((3,), 0.1),
            "action_offset": torch.zeros(3),
            "action_scale": torch.ones(3),
        },
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    actor_type, actor_kwargs = resolve_checkpoint_actor_spec(checkpoint, film_attn_res_block_size=5)

    assert actor_type.value == "film_attn_res"
    assert actor_kwargs == {"num_blocks": 4, "attn_block_size": 5}


def test_load_actor_from_checkpoint_upgrades_legacy_adain_res_weights():
    actor = FiLMResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=3, num_blocks=2)
    critic = Critic(obs_dim=3)
    checkpoint = CheckpointV2(
        meta={"actor_type": "adain_res", "actor_kwargs": {"num_blocks": 2}},
        model={
            "actor": _legacy_adain_res_state_dict(actor),
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

    assert isinstance(loaded_actor, FiLMResActor)
    assert actor_type.value == "film_res"
    assert actor_kwargs == {"num_blocks": 2}
    torch.testing.assert_close(
        loaded_actor.state_dict()["blocks.0.modulation.affine.weight"],
        actor.state_dict()["blocks.0.modulation.affine.weight"],
    )
