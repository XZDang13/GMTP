import types

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
        {"motion_name": "jump_anchor", "anchor_index": 0, "anchor_time": 0.0, "probability": 1.0},
        {"motion_name": "jump_anchor", "anchor_index": 1, "anchor_time": 2.0, "probability": 0.0},
        {"motion_name": "walk_anchor", "anchor_index": 0, "anchor_time": 1.5, "probability": 0.0},
    ]
    assert sum(entry["probability"] for entry in probabilities) == pytest.approx(1.0)
