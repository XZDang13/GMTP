import json

import numpy as np
import torch

from debug_log import RolloutDebugLogger
from eval import Evaluator
from sim2sim_eval import Sim2SimEvaluator


def test_rollout_debug_logger_stacks_numeric_payloads_and_preserves_bool(tmp_path) -> None:
    logger = RolloutDebugLogger(tmp_path / "isaac_eval_debug")

    logger.log_step(
        0,
        {
            "action": torch.tensor([[1.0, -1.0]], dtype=torch.float32),
            "reward": torch.tensor([0.5], dtype=torch.float32),
            "done": torch.tensor([False]),
        },
    )
    logger.log_step(
        1,
        {
            "action": torch.tensor([[2.0, -2.0]], dtype=torch.float32),
            "reward": torch.tensor([1.5], dtype=torch.float32),
            "done": torch.tensor([True]),
        },
    )

    npz_path, json_path = logger.finish({"summary_key": "value"})

    with np.load(npz_path) as data:
        assert data["step"].shape == (2,)
        assert data["action"].shape == (2, 2)
        assert data["reward"].shape == (2,)
        assert data["done"].dtype == np.bool_
        np.testing.assert_allclose(data["action"], np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32))

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["num_logged_steps"] == 2
    assert "action" in summary["logged_keys"]
    assert summary["summary_key"] == "value"


def test_rollout_debug_logger_excludes_unsupported_values_from_npz_and_summary(tmp_path) -> None:
    logger = RolloutDebugLogger(tmp_path / "sim_eval_debug")

    logger.log_step(0, {"action": torch.tensor([[1.0, 2.0]]), "note": "hello"})
    logger.log_step(1, {"action": torch.tensor([[3.0]]), "note": "world"})

    npz_path, json_path = logger.finish({})

    with np.load(npz_path) as data:
        assert "action" not in data
        assert "note" not in data

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert "action" in summary["excluded_npz_keys"]
    assert "note" in summary["excluded_npz_keys"]


def test_eval_debug_step_payload_contains_reward_done_actor_obs_and_info_fields() -> None:
    obs = {
        "motion": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "robot": torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32),
        "privilege": torch.tensor([[6.0, 7.0]], dtype=torch.float32),
    }
    actor_obs = {"obs": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)}
    action = torch.tensor([[0.1, -0.1]], dtype=torch.float32)
    reward = torch.tensor([1.25], dtype=torch.float32)
    terminate = torch.tensor([False])
    timeout = torch.tensor([True])
    info = {
        "scalar": torch.tensor([2.0], dtype=torch.float32),
        "nested": {"vector": torch.tensor([[3.0, 4.0]], dtype=torch.float32)},
        "text": "skip-me",
    }
    actor_state = torch.ones(1, 1, 4, dtype=torch.float32)

    payload, info_metadata = Evaluator._build_debug_step_payload(
        obs,
        actor_obs,
        action,
        reward,
        terminate,
        timeout,
        info,
        actor_state=actor_state,
    )

    assert set(("obs_motion", "obs_robot", "obs_privilege", "action", "reward", "done")).issubset(payload)
    assert "actor_obs_obs" in payload
    assert "info_scalar" in payload
    assert "info_nested_vector" in payload
    assert payload["done"].dtype == torch.bool
    assert payload["actor_state_l2"] > 0.0
    assert "info_text" in info_metadata


def test_sim2sim_debug_step_payload_contains_metrics_and_sim_state() -> None:
    action_dim = 2
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
    actor_obs = {"obs": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)}
    action = torch.tensor([0.25, -0.25], dtype=torch.float32)
    metrics = {"joint_pos_mae": 0.1, "joint_vel_mae": 0.2, "gravity_mae": 0.3}
    actor_state = torch.ones(1, 1, 4, dtype=torch.float32)

    payload = Sim2SimEvaluator._build_debug_step_payload(
        action_dim=action_dim,
        flat_obs=flat_obs,
        actor_obs=actor_obs,
        action=action,
        metrics=metrics,
        sim_target_pos=np.asarray([1.0, 2.0], dtype=np.float32),
        sim_ctrl=np.asarray([3.0, 4.0], dtype=np.float32),
        sim_qpos=np.asarray([5.0, 6.0, 7.0], dtype=np.float32),
        sim_qvel=np.asarray([8.0, 9.0, 10.0], dtype=np.float32),
        actor_state=actor_state,
        sim_motion_time=1.5,
    )

    assert set(
        (
            "obs_target_joint_pos",
            "obs_robot_joint_pos",
            "obs_previous_action",
            "action",
            "sim_target_pos",
            "sim_ctrl",
            "sim_qpos",
            "sim_qvel",
            "joint_pos_mae",
            "joint_vel_mae",
            "gravity_mae",
            "actor_obs_obs",
            "actor_state_l2",
            "actor_state_max_abs",
            "sim_motion_time",
        )
    ).issubset(payload)
