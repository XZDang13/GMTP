import torch

from gmtp.integrations.ref2act.observation_history import (
    build_motion_policy_window_lengths,
    build_robot_policy_window_lengths,
)
from gmtp.integrations.ref2act.mujoco import normalize_action_mode, resolve_action_mode
from gmtp.models import ActorType, FiLMResActor
from gmtp.runtime.observations import (
    extract_sim2sim_actor_obs_from_mapping,
    extract_sim2sim_metrics,
    infer_actor_observation_dims_from_state_dict,
    infer_env_observation_dims,
    infer_sim2sim_observation_dims,
    parse_sim2sim_obs,
    replace_sim2sim_group_latest_terms,
    split_sim2sim_group_observations,
)


def test_infer_env_observation_dims_requires_expected_keys():
    obs = {
        "motion": torch.randn(2, 5),
        "robot": torch.randn(2, 4, 7),
        "privilege": torch.randn(2, 3),
    }
    assert infer_env_observation_dims(obs) == {"motion": 5, "robot": 28, "critic": 3, "policy": 33}


def test_infer_actor_observation_dims_from_state_dict_for_film_res_actor():
    actor = FiLMResActor(robot_obs_dim=12, motion_obs_dim=5, action_dim=2, num_blocks=3)
    dims = infer_actor_observation_dims_from_state_dict(actor.state_dict(), ActorType.FILM_RES)
    assert dims == {"motion": 5, "robot": 12, "policy": 17}


def test_parse_sim2sim_obs_and_extract_metrics():
    flat_obs = torch.tensor(
        [
            0.1,
            0.2,
            0.3,
            1.0,
            2.0,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            5.0,
            6.0,
            7.0,
            8.0,
            0.05,
            0.06,
        ],
        dtype=torch.float32,
    )

    obs_parts = parse_sim2sim_obs(flat_obs, action_dim=2)
    metrics = extract_sim2sim_metrics(flat_obs, action_dim=2)

    assert infer_sim2sim_observation_dims(2) == {"motion": 5, "robot": 12, "policy": 17}
    assert obs_parts["motion"].shape == (5,)
    assert obs_parts["robot"].shape == (12,)
    assert obs_parts["motion_obs"].shape == (5,)
    assert obs_parts["robot_obs"].shape == (12,)
    assert set(("joint_pos_mae", "gravity_mae")).issubset(metrics)


def test_parse_sim2sim_obs_supports_robot_window_lengths():
    base_flat_obs = torch.tensor(
        [
            0.1,
            0.2,
            0.3,
            1.0,
            2.0,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            5.0,
            6.0,
            7.0,
            8.0,
            0.05,
            0.06,
        ],
        dtype=torch.float32,
    )
    base_parts = parse_sim2sim_obs(base_flat_obs, action_dim=2)
    window_lengths = build_robot_policy_window_lengths(4)

    def _history(latest: torch.Tensor, fill_value: float) -> torch.Tensor:
        return torch.cat(
            [
                torch.full((latest.numel() * 3,), fill_value, dtype=torch.float32),
                latest,
            ]
        )

    windowed_robot = torch.cat(
        [
            _history(base_parts["robot_projected_gravity"], -1.0),
            _history(base_parts["anchor_ang_vel"], -2.0),
            _history(base_parts["robot_joint_pos"], -3.0),
            _history(base_parts["robot_joint_vel"], -4.0),
            _history(base_parts["previous_action"], -5.0),
        ]
    )
    windowed_flat_obs = torch.cat([base_parts["motion"], windowed_robot])

    obs_parts = parse_sim2sim_obs(
        windowed_flat_obs,
        action_dim=2,
        observation_window_lengths=window_lengths,
    )
    metrics = extract_sim2sim_metrics(
        windowed_flat_obs,
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    assert infer_sim2sim_observation_dims(2, window_lengths) == {"motion": 5, "robot": 48, "policy": 53}
    assert obs_parts["motion"].shape == (5,)
    assert obs_parts["robot"].shape == (4, 12)
    assert obs_parts["robot_obs"].shape == (4, 12)
    torch.testing.assert_close(obs_parts["target_joint_pos"], base_parts["target_joint_pos"])
    torch.testing.assert_close(obs_parts["robot_joint_vel"], base_parts["robot_joint_vel"])
    assert set(("joint_pos_mae", "gravity_mae")).issubset(metrics)


def test_extract_sim2sim_actor_obs_from_mapping_structures_windowed_robot_history():
    window_lengths = build_robot_policy_window_lengths(4)
    actor_obs = extract_sim2sim_actor_obs_from_mapping(
        {
            "motion": torch.zeros(5, dtype=torch.float32),
            "robot": torch.arange(48, dtype=torch.float32),
        },
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    assert actor_obs is not None
    assert actor_obs["motion"].shape == (5,)
    assert actor_obs["robot"].shape == (4, 12)
    assert actor_obs["robot_obs"].shape == (4, 12)


def test_parse_sim2sim_obs_supports_motion_window_lengths():
    base_flat_obs = torch.tensor(
        [
            0.1,
            0.2,
            0.3,
            1.0,
            2.0,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            5.0,
            6.0,
            7.0,
            8.0,
            0.05,
            0.06,
        ],
        dtype=torch.float32,
    )
    base_parts = parse_sim2sim_obs(base_flat_obs, action_dim=2)
    window_lengths = build_motion_policy_window_lengths(4)

    def _history(latest: torch.Tensor, fill_value: float) -> torch.Tensor:
        return torch.cat(
            [
                torch.full((latest.numel() * 3,), fill_value, dtype=torch.float32),
                latest,
            ]
        )

    windowed_motion = torch.cat(
        [
            _history(base_parts["target_projected_gravity"], -1.0),
            _history(base_parts["target_joint_pos"], -2.0),
        ]
    )
    windowed_flat_obs = torch.cat([windowed_motion, base_parts["robot"]])

    obs_parts = parse_sim2sim_obs(
        windowed_flat_obs,
        action_dim=2,
        observation_window_lengths=window_lengths,
    )
    metrics = extract_sim2sim_metrics(
        windowed_flat_obs,
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    assert infer_sim2sim_observation_dims(2, window_lengths) == {"motion": 20, "robot": 12, "policy": 32}
    assert obs_parts["motion"].shape == (4, 5)
    assert obs_parts["motion_obs"].shape == (4, 5)
    torch.testing.assert_close(obs_parts["target_joint_pos"], base_parts["target_joint_pos"])
    assert set(("joint_pos_mae", "gravity_mae")).issubset(metrics)


def test_extract_sim2sim_actor_obs_from_mapping_structures_windowed_motion_history():
    window_lengths = build_motion_policy_window_lengths(4)
    actor_obs = extract_sim2sim_actor_obs_from_mapping(
        {
            "motion": torch.arange(20, dtype=torch.float32),
            "robot": torch.zeros(12, dtype=torch.float32),
        },
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    assert actor_obs is not None
    assert actor_obs["motion"].shape == (4, 5)
    assert actor_obs["motion_obs"].shape == (4, 5)
    assert actor_obs["robot"].shape == (12,)


def test_replace_sim2sim_group_latest_terms_updates_latest_window_only():
    window_lengths = build_robot_policy_window_lengths(4)
    robot_obs = parse_sim2sim_obs(
        torch.arange(53, dtype=torch.float32),
        action_dim=2,
        observation_window_lengths=window_lengths,
    )["robot"]

    updated_robot_obs = replace_sim2sim_group_latest_terms(
        robot_obs,
        group_name="robot",
        action_dim=2,
        latest_terms={
            "projected_gravity": torch.tensor([101.0, 102.0, 103.0]),
            "anchor_ang_vel_b": torch.tensor([201.0, 202.0, 203.0]),
        },
        observation_window_lengths=window_lengths,
    )
    obs_parts = split_sim2sim_group_observations(
        torch.zeros(5, dtype=torch.float32),
        updated_robot_obs,
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    torch.testing.assert_close(obs_parts["robot_projected_gravity"], torch.tensor([101.0, 102.0, 103.0]))
    torch.testing.assert_close(obs_parts["anchor_ang_vel"], torch.tensor([201.0, 202.0, 203.0]))
    assert updated_robot_obs.shape == (4, 12)


def test_replace_sim2sim_group_latest_terms_updates_latest_motion_window_only():
    window_lengths = build_motion_policy_window_lengths(4)
    motion_obs = parse_sim2sim_obs(
        torch.arange(32, dtype=torch.float32),
        action_dim=2,
        observation_window_lengths=window_lengths,
    )["motion"]

    updated_motion_obs = replace_sim2sim_group_latest_terms(
        motion_obs,
        group_name="motion",
        action_dim=2,
        latest_terms={
            "target_projected_gravity": torch.tensor([101.0, 102.0, 103.0]),
            "target_joint_pos": torch.tensor([201.0, 202.0]),
        },
        observation_window_lengths=window_lengths,
    )
    obs_parts = split_sim2sim_group_observations(
        updated_motion_obs,
        torch.zeros(12, dtype=torch.float32),
        action_dim=2,
        observation_window_lengths=window_lengths,
    )

    torch.testing.assert_close(obs_parts["target_projected_gravity"], torch.tensor([101.0, 102.0, 103.0]))
    torch.testing.assert_close(obs_parts["target_joint_pos"], torch.tensor([201.0, 202.0]))
    assert updated_motion_obs.shape == (4, 5)


def test_resolve_action_mode_supports_current_residual():
    assert normalize_action_mode("CurrentResidual") == "current_residual"
    resolved, source = resolve_action_mode(
        {"action_mode": "CurrentResidual"},
        None,
        torch.zeros(2),
        torch.ones(2),
        torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
    )
    assert resolved == "current_residual"
    assert source == "checkpoint:action_mode"
