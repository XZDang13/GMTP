import json
import types

import numpy as np
import pytest
import torch

from gmtp.runtime.train_runner import TrainRunner


def test_build_episode_metrics_payload_separates_return_and_length_namespaces():
    payload = TrainRunner._build_episode_metrics_payload(12.5, 48.0)

    assert payload == {
        "episode/returns": 12.5,
        "episode/lengths": 48.0,
    }


def test_compute_anchor_reset_probabilities_aggregates_per_anchor_and_keeps_zero_entries():
    sampler = types.SimpleNamespace(
        failure_weight_uniform_mix=0.0,
        motion_lib=types.SimpleNamespace(
            clips=[
                types.SimpleNamespace(name="jump_anchor", anchor_times=torch.tensor([0.0, 2.0])),
                types.SimpleNamespace(name="walk_anchor", anchor_times=torch.tensor([1.5])),
            ]
        ),
        bin_fail_counts=[
            torch.tensor([4.0, 0.0, 0.0]),
            torch.tensor([0.0]),
        ],
        bin_sample_counts=[
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0]),
        ],
        bin_reset_eligible=[
            torch.tensor([True, True, False]),
            torch.tensor([True]),
        ],
        bin_reset_times=[
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.5]),
        ],
    )

    probabilities = TrainRunner._compute_anchor_reset_probabilities(sampler, temperature=1.0)

    assert probabilities == [
        {"motion_index": 0, "motion_name": "jump_anchor", "anchor_index": 0, "anchor_time": 0.0, "probability": 1.0},
        {"motion_index": 0, "motion_name": "jump_anchor", "anchor_index": 1, "anchor_time": 2.0, "probability": 0.0},
        {"motion_index": 1, "motion_name": "walk_anchor", "anchor_index": 0, "anchor_time": 1.5, "probability": 0.0},
    ]
    assert sum(entry["probability"] for entry in probabilities) == pytest.approx(1.0)


def test_build_anchor_reset_probability_metrics_uses_low_cardinality_distribution_summaries():
    payload = TrainRunner._build_anchor_reset_probability_metrics(
        [
            {
                "motion_index": 0,
                "motion_name": "jump/anchor 01",
                "anchor_index": 2,
                "anchor_time": 0.0,
                "probability": 0.25,
            },
            {
                "motion_index": 1,
                "motion_name": "walk.anchor",
                "anchor_index": 0,
                "anchor_time": 1.0,
                "probability": 0.75,
            },
        ]
    )

    assert "sampling/anchor_reset_probability/jump_anchor_01/A002" not in payload
    assert "sampling/anchor_reset_probability/walk.anchor/A000" not in payload
    assert payload["sampling/anchor_reset_probability/sum"] == pytest.approx(1.0)
    assert payload["sampling/anchor_reset_probability/max"] == pytest.approx(0.75)
    assert payload["sampling/anchor_reset_probability/active_anchors"] == 2.0
    assert payload["sampling/anchor_reset_probability/top1_mass"] == pytest.approx(0.75)
    assert payload["sampling/anchor_reset_probability/top5_mass"] == pytest.approx(1.0)
    assert payload["sampling/anchor_reset_probability/effective_anchors"] == pytest.approx(
        np.exp(-(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    )
    assert payload["sampling/motion_reset_probability/active_motions"] == 2.0
    assert payload["sampling/motion_reset_probability/top10_mass"] == pytest.approx(1.0)


def test_build_anchor_heatmap_grid_sorts_motions_and_preserves_probability_mass():
    grid = TrainRunner._build_anchor_reset_probability_heatmap_grid(
        [
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.1},
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 1, "anchor_time": 1.0, "probability": 0.2},
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 2, "anchor_time": 2.0, "probability": 0.0},
            {"motion_index": 1, "motion_name": "walk", "anchor_index": 0, "anchor_time": 5.0, "probability": 0.7},
        ],
        num_bins=4,
    )

    assert grid.values.shape == (2, 4)
    assert grid.motion_names == ["walk", "jump"]
    assert grid.motion_probabilities.tolist() == pytest.approx([0.7, 0.3])
    assert float(grid.values.sum()) == pytest.approx(1.0)
    assert grid.values[0].tolist() == pytest.approx([0.7, 0.0, 0.0, 0.0])
    assert grid.values[1].tolist() == pytest.approx([0.1, 0.0, 0.2, 0.0])


def test_write_anchor_reset_probability_artifacts_keeps_latest_and_history(tmp_path):
    runner = TrainRunner.__new__(TrainRunner)
    runner.run_paths = types.SimpleNamespace(debug_dir=tmp_path)
    runner.anchor_heatmap_bins = 4
    runner.update_count = 100
    runner.global_step = 2000
    runner._anchor_heatmap_warning_emitted = False

    first_entries = [
        {"motion_index": 0, "motion_name": "jump", "anchor_index": 0, "anchor_time": 0.0, "probability": 1.0}
    ]
    first_metrics = TrainRunner._build_anchor_reset_probability_metrics(first_entries)
    first_artifacts = runner._write_anchor_reset_probability_artifacts(first_entries, first_metrics)

    first_heatmap = tmp_path / "anchor_reset_probabilities" / "update_000100_heatmap.png"
    latest_heatmap = tmp_path / "anchor_reset_probabilities" / "latest_heatmap.png"
    first_latest_bytes = latest_heatmap.read_bytes()
    assert first_heatmap.exists()
    assert latest_heatmap.exists()
    assert (tmp_path / "anchor_reset_probabilities" / "update_000100.npz").exists()
    assert first_artifacts["latest_heatmap_png"] == str(latest_heatmap)

    runner.update_count = 200
    runner.global_step = 4000
    second_entries = [
        {"motion_index": 0, "motion_name": "jump", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.25},
        {"motion_index": 1, "motion_name": "walk", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.75},
    ]
    second_metrics = TrainRunner._build_anchor_reset_probability_metrics(second_entries)
    runner._write_anchor_reset_probability_artifacts(second_entries, second_metrics)

    assert first_heatmap.exists()
    assert (tmp_path / "anchor_reset_probabilities" / "update_000200_heatmap.png").exists()
    assert latest_heatmap.read_bytes() != first_latest_bytes

    summary_lines = (tmp_path / "anchor_reset_probabilities" / "summary.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["update"] for line in summary_lines] == [100, 200]

    metadata = json.loads((tmp_path / "anchor_reset_probabilities" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["heatmap_bins"] == 4
    assert metadata["num_motions"] == 2

    with np.load(tmp_path / "anchor_reset_probabilities" / "update_000200.npz") as payload:
        assert payload["motion_index"].tolist() == [0, 1]
        assert payload["motion_name"].tolist() == ["jump", "walk"]
        assert payload["probability"].tolist() == pytest.approx([0.25, 0.75])
