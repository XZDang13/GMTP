import json
import types

import numpy as np
import pytest
import torch

from gmtp.runtime.train_runner import ANCHOR_CONSOLE_TOP_K, ANCHOR_DASHBOARD_MAX_RANK_BANDS, TrainRunner


def test_build_episode_metrics_payload_separates_return_and_length_namespaces():
    payload = TrainRunner._build_episode_metrics_payload(12.5, 48.0)

    assert payload == {
        "episode/returns": 12.5,
        "episode/lengths": 48.0,
    }


def test_extract_relative_anchor_pos_sample_uses_gmtp_privilege_term_order():
    privilege = torch.tensor(
        [
            [0.0, 0.1, 0.2, 1.0, 1.1, 2.0, 2.1, 3.0, 4.0, 12.0],
            [0.3, 0.4, 0.5, 1.2, 1.3, 2.2, 2.3, 0.0, 0.0, -2.0],
        ],
        dtype=torch.float32,
    )

    relative_anchor_pos = TrainRunner._extract_relative_anchor_pos_sample(privilege, action_dim=2)

    assert relative_anchor_pos is not None
    torch.testing.assert_close(relative_anchor_pos, torch.tensor([[3.0, 4.0, 12.0], [0.0, 0.0, -2.0]]))


def test_extract_relative_anchor_pos_sample_skips_unavailable_or_nonfinite_privilege_obs():
    assert TrainRunner._extract_relative_anchor_pos_sample(torch.zeros(2, 9), action_dim=2) is None
    assert TrainRunner._extract_relative_anchor_pos_sample(torch.zeros(2, 10), action_dim=0) is None

    nonfinite_privilege = torch.zeros(2, 10)
    nonfinite_privilege[0, 8] = float("nan")

    assert TrainRunner._extract_relative_anchor_pos_sample(nonfinite_privilege, action_dim=2) is None


def test_infer_critic_key_body_count_from_privilege_observation_dim():
    assert TrainRunner._infer_critic_key_body_count(
        critic_obs_dim=67,
        action_dim=2,
        observation_window_lengths={},
    ) == 4


def test_build_location_tracking_metrics_reports_full_xy_z_p95_and_max_errors():
    relative_anchor_pos_samples = [
        torch.tensor([[3.0, 4.0, 12.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
    ]

    payload = TrainRunner._build_location_tracking_metrics(relative_anchor_pos_samples)

    relative_anchor_pos = torch.cat(relative_anchor_pos_samples, dim=0)
    location_error = torch.linalg.vector_norm(relative_anchor_pos, dim=-1)
    xy_error = torch.linalg.vector_norm(relative_anchor_pos[:, :2], dim=-1)
    z_error = torch.abs(relative_anchor_pos[:, 2])
    assert payload == pytest.approx(
        {
            "tracking/location_error_m": float(location_error.mean().item()),
            "tracking/location_error_xy_m": float(xy_error.mean().item()),
            "tracking/location_error_z_m": float(z_error.mean().item()),
            "tracking/location_error_p95_m": float(torch.quantile(location_error, 0.95).item()),
            "tracking/location_error_max_m": float(location_error.max().item()),
        }
    )


def test_build_location_tracking_metrics_skips_invalid_samples():
    assert TrainRunner._build_location_tracking_metrics([]) == {}
    assert TrainRunner._build_location_tracking_metrics([torch.zeros(2, 2)]) == {}

    nonfinite_sample = torch.zeros(2, 3)
    nonfinite_sample[0, 0] = float("inf")

    assert TrainRunner._build_location_tracking_metrics([nonfinite_sample]) == {}


def test_train_prints_failure_reason_and_reraises(capsys):
    class CloseTrackingEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    runner = TrainRunner.__new__(TrainRunner)
    runner.initial_obs = object()
    runner.config = types.SimpleNamespace(num_updates=1)
    runner.update_count = 2
    runner.global_step = 40
    runner.checkpoint_interval = 0
    runner.env = CloseTrackingEnv()
    runner.use_wandb = False

    def failing_rollout(_obs):
        raise RuntimeError("env exploded")

    runner.rollout = failing_rollout

    with pytest.raises(RuntimeError, match="env exploded"):
        runner.train()

    output = capsys.readouterr().out
    assert "training script failed during rollout update 3: RuntimeError: env exploded" in output
    assert runner.env.closed is True


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
                "motion_index": 0,
                "motion_name": "jump/anchor 01",
                "anchor_index": 3,
                "anchor_time": 1.0,
                "probability": 0.0,
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

    assert "sampler/reset_distribution/anchors/jump_anchor_01/A002" not in payload
    assert "sampler/reset_distribution/anchors/walk.anchor/A000" not in payload
    assert payload["sampler/reset_distribution/anchors/probability_sum"] == pytest.approx(1.0)
    assert payload["sampler/reset_distribution/anchors/max_probability"] == pytest.approx(0.75)
    assert payload["sampler/reset_distribution/anchors/active_count"] == 2.0
    assert payload["sampler/reset_distribution/anchors/active_fraction"] == pytest.approx(2.0 / 3.0)
    assert payload["sampler/reset_distribution/anchors/top1_mass"] == pytest.approx(0.75)
    assert payload["sampler/reset_distribution/anchors/top5_mass"] == pytest.approx(1.0)
    assert payload["sampler/reset_distribution/anchors/top20_anchor0_count"] == 1.0
    assert payload["sampler/reset_distribution/anchors/top20_single_anchor_motion_count"] == 1.0
    assert payload["sampler/reset_distribution/anchors/effective_count"] == pytest.approx(
        np.exp(-(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    )
    assert payload["sampler/reset_distribution/motions/active_count"] == 2.0
    assert payload["sampler/reset_distribution/motions/top10_mass"] == pytest.approx(1.0)
    assert payload["sampler/reset_distribution/motions/single_anchor_count"] == 1.0
    assert payload["sampler/reset_distribution/motions/single_anchor_fraction"] == pytest.approx(0.5)


def test_build_sampler_failure_stats_reports_coverage_and_failure_rate():
    sampler = types.SimpleNamespace(
        bin_fail_counts=[
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0]),
        ],
        bin_sample_counts=[
            torch.tensor([4.0, 0.0, 2.0]),
            torch.tensor([0.0, 4.0]),
        ],
        bin_reset_eligible=[
            torch.tensor([True, True, False]),
            torch.tensor([True, True]),
        ],
    )

    payload = TrainRunner._build_sampler_failure_stats(sampler)

    assert payload["sampler/failure_stats/effective_sample_count_sum"] == pytest.approx(8.0)
    assert payload["sampler/failure_stats/effective_failure_count_sum"] == pytest.approx(3.0)
    assert payload["sampler/failure_stats/failure_rate"] == pytest.approx(3.0 / 8.0)
    assert payload["sampler/failure_stats/anchors/total_count"] == 4.0
    assert payload["sampler/failure_stats/anchors/sampled_count"] == 2.0
    assert payload["sampler/failure_stats/anchors/sampled_fraction"] == pytest.approx(0.5)
    assert payload["sampler/failure_stats/motions/sampled_count"] == 2.0


def test_log_anchor_reset_probabilities_prints_only_top_twenty(capsys):
    runner = TrainRunner.__new__(TrainRunner)
    runner.update_count = 100
    runner.global_step = 2000
    runner.sampling_strategy = "failure_weighted"
    runner.segment_source = "anchor"
    entries = [
        {
            "motion_index": index,
            "motion_name": f"motion_{index:02d}",
            "anchor_index": index,
            "anchor_time": float(index),
            "probability": float(25 - index),
        }
        for index in range(25)
    ]
    runner._collect_anchor_reset_probabilities = lambda: entries
    runner._log_metrics = lambda metrics: None
    runner._write_anchor_reset_probability_artifacts = lambda anchor_probabilities, metrics_payload: {}
    runner._sync_anchor_reset_probability_summary_to_wandb = lambda metrics_payload, artifacts: None

    runner._log_anchor_reset_probabilities()
    output = capsys.readouterr().out

    assert ANCHOR_CONSOLE_TOP_K == 20
    motion_lines = [line for line in output.splitlines() if line.startswith("  motion_")]
    assert len(motion_lines) == 40
    assert sum(" active_anchors=" in line for line in motion_lines) == 20
    assert sum(" A" in line for line in motion_lines) == 20
    assert "sampler snapshot after update 100 step=2000:" in output
    assert "reset anchors:" in output
    assert "reset motions:" in output
    assert "top reset motions:" in output
    assert "top reset anchors:" in output
    assert "top multi-anchor phase biases:" not in output
    assert (
        "motion_00 motion_p=25.000000 active_anchors=1/1 "
        "max_anchor=A0@0.000s max_anchor_p=25.000000"
    ) in output
    assert "motion_00 A0 t=0.000s anchor_p=25.000000 motion_p=25.000000 anchor_share=100.0%" in output
    assert (
        "motion_19 motion_p=6.000000 active_anchors=1/1 "
        "max_anchor=A19@19.000s max_anchor_p=6.000000"
    ) in output
    assert "motion_19 A19 t=19.000s anchor_p=6.000000 motion_p=6.000000 anchor_share=100.0%" in output
    assert "motion_20 motion_p=" not in output
    assert "motion_20 A20" not in output
    assert "motion_24 motion_p=" not in output
    assert "motion_24 A24" not in output


def test_build_anchor_rank_band_heatmap_grid_sorts_motions_and_preserves_probability_mass():
    grid = TrainRunner._build_anchor_rank_band_heatmap_grid(
        [
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.1},
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 1, "anchor_time": 1.0, "probability": 0.2},
            {"motion_index": 0, "motion_name": "jump", "anchor_index": 2, "anchor_time": 2.0, "probability": 0.0},
            {"motion_index": 1, "motion_name": "walk", "anchor_index": 0, "anchor_time": 5.0, "probability": 0.7},
        ],
        num_bins=4,
    )

    assert grid.values.shape == (2, 4)
    assert grid.num_motions == 2
    assert grid.num_rank_bands == 2
    assert float(grid.values.sum()) == pytest.approx(1.0)
    assert grid.values[0].tolist() == pytest.approx([0.7, 0.0, 0.0, 0.0])
    assert grid.values[1].tolist() == pytest.approx([0.1, 0.0, 0.2, 0.0])


def test_build_anchor_rank_band_heatmap_grid_caps_bands_and_handles_small_motion_counts():
    many_entries = [
        {
            "motion_index": index,
            "motion_name": f"motion_{index:03d}",
            "anchor_index": 0,
            "anchor_time": 0.0,
            "probability": 1.0 / 85.0,
        }
        for index in range(85)
    ]
    many_grid = TrainRunner._build_anchor_rank_band_heatmap_grid(many_entries, num_bins=2)

    assert many_grid.values.shape == (ANCHOR_DASHBOARD_MAX_RANK_BANDS, 2)
    assert many_grid.num_motions == 85
    assert many_grid.num_rank_bands == ANCHOR_DASHBOARD_MAX_RANK_BANDS
    assert float(many_grid.values.sum()) == pytest.approx(1.0)

    small_entries = many_entries[:3]
    small_grid = TrainRunner._build_anchor_rank_band_heatmap_grid(small_entries, num_bins=2)

    assert small_grid.values.shape == (3, 2)
    assert small_grid.num_motions == 3
    assert small_grid.num_rank_bands == 3


def test_sampler_dashboard_rows_are_sorted_and_truncate_long_motion_names():
    long_motion_name = "very_long_motion_name_that_needs_to_be_truncated"
    entries = [
        {
            "motion_index": 0,
            "motion_name": long_motion_name,
            "anchor_index": 0,
            "anchor_time": 0.0,
            "probability": 0.35,
        },
        {
            "motion_index": 0,
            "motion_name": long_motion_name,
            "anchor_index": 1,
            "anchor_time": 1.0,
            "probability": 0.25,
        },
        {"motion_index": 1, "motion_name": "walk", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.30},
        {"motion_index": 2, "motion_name": "run", "anchor_index": 0, "anchor_time": 0.0, "probability": 0.10},
    ]

    top_motions = TrainRunner._build_top_motion_rows(entries, limit=2, max_name_chars=12)
    top_anchors = TrainRunner._build_top_anchor_rows(entries, limit=3, max_name_chars=12)
    curve = TrainRunner._build_anchor_cumulative_mass_curve(entries, checkpoints=(1, 2, 10))

    assert [row["full_motion_name"] for row in top_motions] == [long_motion_name, "walk"]
    assert top_motions[0]["probability"] == pytest.approx(0.60)
    assert top_motions[0]["motion_name"] == "very_long..."
    assert top_motions[0]["anchor_count"] == 2
    assert top_motions[0]["active_anchor_count"] == 2
    assert top_motions[0]["max_anchor_index"] == 0
    assert top_motions[0]["max_anchor_time"] == pytest.approx(0.0)
    assert top_motions[0]["max_anchor_probability"] == pytest.approx(0.35)
    assert [row["probability"] for row in top_anchors] == pytest.approx([0.35, 0.30, 0.25])
    assert top_anchors[0]["motion_name"] == "very_long..."
    assert top_anchors[0]["motion_probability"] == pytest.approx(0.60)
    assert top_anchors[0]["anchor_share"] == pytest.approx(0.35 / 0.60)
    assert top_anchors[1]["motion_probability"] == pytest.approx(0.30)
    assert top_anchors[1]["anchor_share"] == pytest.approx(1.0)
    assert top_anchors[0]["uniform_ratio"] == pytest.approx(1.4)
    assert curve["anchor_count"].tolist() == [1, 2, 4]
    assert curve["mass"].tolist() == pytest.approx([0.35, 0.65, 1.0])
    assert curve["uniform_mass"].tolist() == pytest.approx([0.25, 0.5, 1.0])


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
