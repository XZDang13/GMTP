import torch

from gmtp.integrations.ref2act.mujoco import normalize_action_mode, resolve_action_mode
from gmtp.models import ActorType, FiLMAttnResActor
from gmtp.runtime.observations import (
    extract_sim2sim_metrics,
    infer_actor_observation_dims_from_state_dict,
    infer_env_observation_dims,
    infer_sim2sim_observation_dims,
    parse_sim2sim_obs,
)


def test_infer_env_observation_dims_requires_expected_keys():
    obs = {
        "motion": torch.randn(2, 5),
        "robot": torch.randn(2, 7),
        "privilege": torch.randn(2, 3),
    }
    assert infer_env_observation_dims(obs) == {"motion": 5, "robot": 7, "critic": 3, "policy": 12}


def test_infer_actor_observation_dims_from_state_dict_for_film_attn_res_actor():
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=2, num_blocks=3, attn_block_size=2)
    dims = infer_actor_observation_dims_from_state_dict(actor.state_dict(), ActorType.FILM_ATTN_RES)
    assert dims == {"motion": 5, "robot": 7, "policy": 12}


def test_parse_sim2sim_obs_and_extract_metrics():
    flat_obs = torch.tensor(
        [
            0.1,
            0.2,
            0.3,
            1.0,
            2.0,
            3.0,
            4.0,
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

    assert infer_sim2sim_observation_dims(2) == {"motion": 7, "robot": 12, "policy": 19}
    assert obs_parts["motion"].shape == (7,)
    assert obs_parts["robot"].shape == (12,)
    assert obs_parts["motion_obs"].shape == (7,)
    assert obs_parts["robot_obs"].shape == (12,)
    assert set(("joint_pos_mae", "joint_vel_mae", "gravity_mae")).issubset(metrics)


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
