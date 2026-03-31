import json

import numpy as np
import torch

from gmtp.runtime.debug import RolloutDebugLogger


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
