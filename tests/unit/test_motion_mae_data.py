from pathlib import Path

import numpy as np
import pytest
import torch

from gmtp.motion_mae import (
    MotionFeatureSequence,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    ReferenceMotionMAEDataset,
    build_motion_feature_bundle,
    build_motion_mae_datasets,
)
from gmtp.motion_mae.adapters import StageIINpzMotionAdapter
from gmtp.motion_mae.data import build_segment_valid_window_centers, build_valid_window_centers
from gmtp.motion_mae.schema import MotionSegment


def test_stageii_adapter_loads_canonical_sequence_and_segments():
    adapter = StageIINpzMotionAdapter()
    sequence = adapter.load_sequence("env/assests/115_02_stageii.npz")

    assert sequence.motion_name == "115_02_stageii"
    assert sequence.fps == 50.0
    assert sequence.joint_pos.shape == (333, 23)
    assert sequence.body_pos_w.shape == (333, 29, 3)
    assert sequence.segments[0].start_frame == 0
    assert sequence.segments[0].end_frame == 15
    assert sequence.segments[0].segment_type is not None


def test_build_valid_window_centers_respects_segment_boundaries():
    segments = (
        MotionSegment(start_frame=0, end_frame=10),
        MotionSegment(start_frame=12, end_frame=20),
    )

    assert build_segment_valid_window_centers(segments[0], past_frames=4, future_frames=2) == [3, 4, 5, 6, 7]
    assert build_valid_window_centers(segments, past_frames=4, future_frames=2) == [3, 4, 5, 6, 7, 15, 16, 17]


def test_reference_motion_mae_dataset_returns_expected_slices():
    reference_features = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    target_features = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    sequence = MotionFeatureSequence(
        motion_file="foo.npz",
        motion_name="foo",
        segments=(MotionSegment(start_frame=0, end_frame=10),),
        full_features=torch.cat((reference_features, target_features[:, :2]), dim=-1),
        reference_features=reference_features,
        target_features=target_features,
    )
    dataset = ReferenceMotionMAEDataset(
        sequences=(sequence,),
        window_indices=((0, 3),),
        past_frames=4,
        future_frames=2,
        reference_mean=torch.zeros(6),
        reference_std=torch.ones(6),
        target_mean=torch.zeros(8),
        target_std=torch.ones(8),
    )

    item = dataset[0]

    torch.testing.assert_close(item["reference"], reference_features[0:4])
    torch.testing.assert_close(item["target"], target_features[4:6])
    assert item["center_t"] == 3


def test_build_motion_feature_bundle_extracts_default_end_effector_features():
    data_config = MotionMAEDataConfig(motion_files=("env/assests/05_05_stageii.npz",))
    bundle = build_motion_feature_bundle(
        ["env/assests/05_05_stageii.npz"],
        data_config=data_config,
        feature_config=MotionMAEFeatureConfig(),
        slice_weights={"root": 1.0, "joint": 1.0, "end_effector": 1.0},
    )

    assert bundle.schema.d_ref == 61
    assert bundle.schema.d_target == 61
    assert bundle.sequences[0].segments


def test_build_motion_mae_datasets_auto_falls_back_to_by_window_for_single_motion():
    config = MotionMAEDataConfig(
        motion_files=("env/assests/115_02_stageii.npz",),
        past_frames=4,
        future_frames=2,
        split_mode="auto",
        val_ratio=0.25,
        seed=7,
        max_train_windows=8,
        max_val_windows=4,
    )

    bundle_a = build_motion_mae_datasets(config, feature_config=MotionMAEFeatureConfig())
    bundle_b = build_motion_mae_datasets(config, feature_config=MotionMAEFeatureConfig())

    assert bundle_a.train_dataset.window_indices == bundle_b.train_dataset.window_indices
    assert bundle_a.val_dataset.window_indices == bundle_b.val_dataset.window_indices


def test_build_motion_feature_bundle_rejects_inconsistent_body_names(tmp_path: Path):
    source = np.load("env/assests/05_05_stageii.npz", allow_pickle=True)
    modified_payload = {key: source[key] for key in source.files}
    modified_body_names = modified_payload["body_names"].copy()
    modified_body_names[0] = "pelvis_changed"
    modified_payload["body_names"] = modified_body_names
    modified_path = tmp_path / "modified_motion.npz"
    np.savez(modified_path, **modified_payload)

    with pytest.raises(ValueError, match="body_names"):
        build_motion_feature_bundle(
            ["env/assests/05_05_stageii.npz", str(modified_path)],
            data_config=MotionMAEDataConfig(
                motion_files=("env/assests/05_05_stageii.npz", str(modified_path)),
            ),
            feature_config=MotionMAEFeatureConfig(),
        )
