from pathlib import Path

import numpy as np
import pytest
import torch

from gmtp.motion_vae import (
    MotionFeatureSequence,
    MotionVAEDataConfig,
    MotionVAEFeatureConfig,
    ReferenceMotionVAEDataset,
    build_motion_feature_bundle,
    build_motion_vae_datasets,
    build_valid_window_centers,
    quat_apply_inverse,
)


def test_build_valid_window_centers_respects_bounds():
    assert build_valid_window_centers(12, past_frames=4, future_frames=3) == [3, 4, 5, 6, 7, 8, 9]
    assert build_valid_window_centers(5, past_frames=4, future_frames=2) == [3]
    assert build_valid_window_centers(4, past_frames=4, future_frames=2) == []


def test_reference_motion_vae_dataset_returns_expected_slices():
    reference_features = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    target_features = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    sequence = MotionFeatureSequence(
        motion_file="foo.npz",
        motion_name="foo",
        full_features=torch.cat((reference_features, target_features[:, :2]), dim=-1),
        reference_features=reference_features,
        target_features=target_features,
    )
    dataset = ReferenceMotionVAEDataset(
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
    torch.testing.assert_close(item["target"], target_features[3:5])
    assert item["center_t"] == 3


def test_build_motion_feature_bundle_extracts_default_end_effector_features():
    bundle = build_motion_feature_bundle(
        ["env/assests/05_05_stageii.npz"],
        feature_config=MotionVAEFeatureConfig(),
        slice_weights={"root": 1.0, "joint": 1.0, "end_effector": 1.0},
    )
    sequence = bundle.sequences[0]
    payload = np.load("env/assests/05_05_stageii.npz", allow_pickle=True)
    body_names = payload["body_names"].tolist()
    anchor_index = body_names.index("pelvis")
    eef_indices = [
        body_names.index("left_ankle_roll_link"),
        body_names.index("right_ankle_roll_link"),
        body_names.index("left_rubber_hand"),
        body_names.index("right_rubber_hand"),
    ]
    anchor_pos = torch.as_tensor(payload["body_pos_w"][:, anchor_index], dtype=torch.float32)
    anchor_quat = torch.as_tensor(payload["body_quat_w"][:, anchor_index], dtype=torch.float32)
    eef_pos = torch.as_tensor(payload["body_pos_w"][:, eef_indices], dtype=torch.float32)
    manual_eef = quat_apply_inverse(
        anchor_quat[:, None, :].expand(-1, len(eef_indices), -1),
        eef_pos - anchor_pos[:, None, :],
    ).reshape(sequence.length, -1)
    eef_slice = next(item for item in bundle.schema.base_slices if item.name == "end_effector")

    assert bundle.schema.d_ref == 61
    assert bundle.schema.d_target == 61
    torch.testing.assert_close(sequence.full_features[:, eef_slice.start : eef_slice.end], manual_eef)


def test_build_motion_vae_datasets_split_modes_are_reproducible():
    motion_files = [
        "env/assests/05_05_stageii.npz",
        "env/assests/115_02_stageii.npz",
        "env/assests/115_06_stageii.npz",
    ]
    by_window = MotionVAEDataConfig(
        motion_files=tuple(motion_files),
        past_frames=4,
        future_frames=2,
        split_mode="by_window",
        val_ratio=0.25,
        seed=7,
        max_train_windows=8,
        max_val_windows=4,
    )
    by_window_a = build_motion_vae_datasets(by_window, feature_config=MotionVAEFeatureConfig())
    by_window_b = build_motion_vae_datasets(by_window, feature_config=MotionVAEFeatureConfig())
    assert by_window_a.train_dataset.window_indices == by_window_b.train_dataset.window_indices
    assert by_window_a.val_dataset.window_indices == by_window_b.val_dataset.window_indices

    by_motion = MotionVAEDataConfig(
        motion_files=tuple(motion_files),
        past_frames=4,
        future_frames=2,
        split_mode="by_motion",
        val_ratio=0.34,
        seed=3,
        max_train_windows=8,
        max_val_windows=4,
    )
    by_motion_a = build_motion_vae_datasets(by_motion, feature_config=MotionVAEFeatureConfig())
    by_motion_b = build_motion_vae_datasets(by_motion, feature_config=MotionVAEFeatureConfig())
    assert by_motion_a.train_motion_names == by_motion_b.train_motion_names
    assert by_motion_a.val_motion_names == by_motion_b.val_motion_names


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
            feature_config=MotionVAEFeatureConfig(),
        )
