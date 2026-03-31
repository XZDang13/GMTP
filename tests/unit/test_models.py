import torch

from RLAlg.alg.ppo import PPO
from RLAlg.buffer.replay_buffer import ReplayBuffer

from gmtp.models import RecurrentActor, policy_state_from_storage, policy_state_for_storage


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
