from pathlib import Path

import numpy as np
import pytest
import torch

from gmtp.motion_mae import (
    FeatureSliceSpec,
    MotionFeatureSchema,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    MotionMAEModelConfig,
    MotionMAEPretrainConfig,
    MotionFeatureSequence,
    ReferenceMotionMAE,
    ReferenceMotionMAEDataset,
    build_motion_mae_encoder_checkpoint,
    save_motion_mae_encoder_checkpoint,
)
from gmtp.motion_mae.schema import MotionSegment
from gmtp.runtime.motion_mae_visualize import (
    build_motion_mae_model_from_checkpoint,
    build_playback_frame_pair,
    build_whole_motion_visualization_batch,
    compute_visualization_metrics,
    expand_rendered_frames_to_motion_timeline,
    extract_joint_trajectory,
    resolve_renderer_size,
    select_visualization_sample,
    select_visualization_sequence,
    select_future_frame_comparison,
    update_tracking_camera,
    validate_rendered_ground_truth_alignment,
    validate_visualizer_schema_support,
)


def _joint_names(count: int = 23) -> tuple[str, ...]:
    return tuple(f"j{index}" for index in range(count))


def _body_names() -> tuple[str, ...]:
    return (
        "pelvis",
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    )


def _schema(
    *,
    joint_count: int = 23,
    reference_feature_names: tuple[str, ...] = ("root", "joint"),
    target_feature_names: tuple[str, ...] = ("root", "joint"),
    policy_feature_names: tuple[str, ...] = ("root", "joint"),
) -> MotionFeatureSchema:
    base_slices = [
        FeatureSliceSpec("root", 0, 3),
        FeatureSliceSpec("joint", 3, 3 + 2 * joint_count),
        FeatureSliceSpec("end_effector", 3 + 2 * joint_count, 3 + 2 * joint_count + 12),
    ]
    base_slice_map = {item.name: item for item in base_slices}

    def build_named_slices(names: tuple[str, ...]) -> tuple[FeatureSliceSpec, ...]:
        offset = 0
        slices = []
        for name in names:
            base_slice = base_slice_map[name]
            next_offset = offset + base_slice.dim
            slices.append(FeatureSliceSpec(name, offset, next_offset))
            offset = next_offset
        return tuple(slices)

    target_slices = build_named_slices(target_feature_names)
    policy_dim = sum(item.dim for item in target_slices[: len(policy_feature_names)])
    d_ref = sum(base_slice_map[name].dim for name in reference_feature_names)
    d_target = sum(base_slice_map[name].dim for name in target_feature_names)
    return MotionFeatureSchema(
        d_ref=d_ref,
        d_target=d_target,
        full_feature_dim=base_slices[-1].end,
        base_slices=tuple(base_slices),
        reference_slices=build_named_slices(reference_feature_names),
        target_slices=target_slices,
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, policy_dim),
        anchor_body_name="pelvis",
        end_effector_body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ),
        reference_feature_names=reference_feature_names,
        target_feature_names=target_feature_names,
        policy_feature_names=policy_feature_names,
        joint_names=_joint_names(joint_count),
        body_names=_body_names(),
        reference_mean=tuple(0.0 for _ in range(d_ref)),
        reference_std=tuple(1.0 for _ in range(d_ref)),
        target_mean=tuple(0.0 for _ in range(d_target)),
        target_std=tuple(1.0 for _ in range(d_target)),
    )


def _dataset() -> ReferenceMotionMAEDataset:
    reference_features_a = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    target_features_a = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    reference_features_b = torch.arange(120, 180, dtype=torch.float32).reshape(10, 6)
    target_features_b = torch.arange(200, 280, dtype=torch.float32).reshape(10, 8)
    sequence_a = MotionFeatureSequence(
        motion_file="a.npz",
        motion_name="a",
        segments=(MotionSegment(start_frame=0, end_frame=10),),
        full_features=torch.zeros(10, 10),
        reference_features=reference_features_a,
        target_features=target_features_a,
    )
    sequence_b = MotionFeatureSequence(
        motion_file="b.npz",
        motion_name="b",
        segments=(MotionSegment(start_frame=0, end_frame=10),),
        full_features=torch.zeros(10, 10),
        reference_features=reference_features_b,
        target_features=target_features_b,
    )
    return ReferenceMotionMAEDataset(
        sequences=(sequence_a, sequence_b),
        window_indices=((0, 3), (0, 4), (1, 5)),
        past_frames=4,
        future_frames=2,
        reference_mean=torch.zeros(6),
        reference_std=torch.ones(6),
        target_mean=torch.zeros(8),
        target_std=torch.ones(8),
    )


def _write_encoder_checkpoint(tmp_path: Path) -> Path:
    schema = _schema(joint_count=2)
    model = ReferenceMotionMAE(
        input_dim=schema.d_ref,
        target_dim=schema.d_target,
        past_frames=4,
        future_frames=2,
        latent_dim=6,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    checkpoint = build_motion_mae_encoder_checkpoint(
        model=model,
        schema=schema,
        config=MotionMAEPretrainConfig(
            data=MotionMAEDataConfig(
                motion_files=("env/assests/115_02_stageii.npz",),
                past_frames=4,
                future_frames=2,
                split_mode="by_window",
                val_ratio=0.5,
            ),
            feature=MotionMAEFeatureConfig(
                reference_feature_names=("root", "joint"),
                target_feature_names=("root", "joint"),
                policy_feature_names=("root", "joint"),
            ),
            model=MotionMAEModelConfig(
                d_model=16,
                latent_dim=6,
                encoder_layers=2,
                decoder_layers=1,
                nhead=4,
                dim_feedforward=32,
            ),
        ),
        epoch=1,
        best_metric=0.5,
    )
    return save_motion_mae_encoder_checkpoint(checkpoint, tmp_path / "motion_mae_encoder.pth")


def test_select_visualization_sample_filters_by_motion_name_and_index():
    sample = select_visualization_sample(_dataset(), split_name="val", motion_name="a", sample_index=1)

    assert sample.motion_name == "a"
    assert sample.motion_file == "a.npz"
    assert sample.center_t == 4
    assert sample.future_frame_indices == (5, 6)
    torch.testing.assert_close(sample.reference, torch.arange(6, 30, dtype=torch.float32).reshape(4, 6))
    torch.testing.assert_close(sample.target, torch.arange(40, 56, dtype=torch.float32).reshape(2, 8))


def test_select_visualization_sample_raises_for_unknown_motion_or_index():
    dataset = _dataset()

    with pytest.raises(ValueError, match="Motion 'missing'"):
        select_visualization_sample(dataset, split_name="train", motion_name="missing", sample_index=0)

    with pytest.raises(IndexError, match="out of range"):
        select_visualization_sample(dataset, split_name="train", motion_name=None, sample_index=99)


def test_select_visualization_sequence_requires_motion_name_when_multiple_sequences_exist():
    with pytest.raises(ValueError, match="requires motion_name"):
        select_visualization_sequence(_dataset(), motion_name=None)


def test_select_visualization_sequence_returns_unique_named_motion():
    sequence_index, sequence = select_visualization_sequence(_dataset(), motion_name="b")

    assert sequence_index == 1
    assert sequence.motion_name == "b"
    assert sequence.motion_file == "b.npz"


def test_select_future_frame_comparison_extracts_requested_future_step():
    root_state = {
        "future_frame_indices": torch.tensor([8, 9, 10], dtype=torch.long),
        "root_pos": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        "joint_pos": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }

    (
        future_frame_indices,
        target,
        target_normalized,
        prediction,
        prediction_normalized,
        selected_root_state,
    ) = select_future_frame_comparison(
        future_frame_indices=(8, 9, 10),
        future_frame_index=1,
        target=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        target_normalized=torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100.0,
        prediction=torch.arange(12, dtype=torch.float32).reshape(3, 4) + 200.0,
        prediction_normalized=torch.arange(12, dtype=torch.float32).reshape(3, 4) + 300.0,
        root_state=root_state,
    )

    assert future_frame_indices == (9,)
    assert target.shape == (1, 4)
    assert target_normalized.shape == (1, 4)
    assert prediction.shape == (1, 4)
    assert prediction_normalized.shape == (1, 4)
    torch.testing.assert_close(selected_root_state["future_frame_indices"], torch.tensor([9], dtype=torch.long))
    torch.testing.assert_close(selected_root_state["root_pos"], torch.tensor([[3.0, 4.0, 5.0]]))
    torch.testing.assert_close(selected_root_state["joint_pos"], torch.tensor([[2.0, 3.0]]))


def test_select_future_frame_comparison_rejects_invalid_index():
    with pytest.raises(IndexError, match="future_frame_index=4"):
        select_future_frame_comparison(
            future_frame_indices=(8, 9, 10),
            future_frame_index=4,
            target=torch.zeros(3, 4),
            target_normalized=torch.zeros(3, 4),
            prediction=torch.zeros(3, 4),
            prediction_normalized=torch.zeros(3, 4),
            root_state={"future_frame_indices": torch.tensor([8, 9, 10], dtype=torch.long)},
        )


def test_validate_visualizer_schema_support_accepts_root_joint_g1_schema():
    validate_visualizer_schema_support(_schema())


def test_validate_visualizer_schema_support_rejects_non_policy_observation_schema():
    with pytest.raises(ValueError, match="target_feature_names"):
        validate_visualizer_schema_support(_schema(target_feature_names=("root", "joint", "end_effector")))

    with pytest.raises(ValueError, match="23DoF"):
        validate_visualizer_schema_support(_schema(joint_count=2))


def test_extract_joint_trajectory_splits_joint_pos_and_vel():
    joint_count = 23
    target = torch.arange(2 * (3 + 2 * joint_count), dtype=torch.float32).reshape(2, 3 + 2 * joint_count)

    joint_pos, joint_vel = extract_joint_trajectory(target, _schema())

    torch.testing.assert_close(joint_pos, target[:, 3 : 3 + joint_count])
    torch.testing.assert_close(joint_vel, target[:, 3 + joint_count : 3 + 2 * joint_count])


def test_build_playback_frame_pair_reuses_ground_truth_root_state_for_prediction():
    pair = build_playback_frame_pair(
        gt_root_pos=torch.tensor([1.0, 2.0, 3.0]),
        gt_root_quat=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        gt_root_lin_vel=torch.tensor([0.1, 0.2, 0.3]),
        gt_root_ang_vel=torch.tensor([0.4, 0.5, 0.6]),
        gt_joint_pos=torch.tensor([0.1, 0.2]),
        gt_joint_vel=torch.tensor([0.3, 0.4]),
        pred_joint_pos=torch.tensor([0.9, 1.0]),
        pred_joint_vel=torch.tensor([1.1, 1.2]),
    )

    torch.testing.assert_close(pair.gt.root_pos, pair.pred.root_pos)
    torch.testing.assert_close(pair.gt.root_quat, pair.pred.root_quat)
    torch.testing.assert_close(pair.gt.root_lin_vel, pair.pred.root_lin_vel)
    torch.testing.assert_close(pair.gt.root_ang_vel, pair.pred.root_ang_vel)
    torch.testing.assert_close(pair.pred.joint_pos, torch.tensor([0.9, 1.0]))
    torch.testing.assert_close(pair.pred.joint_vel, torch.tensor([1.1, 1.2]))


def test_validate_rendered_ground_truth_alignment_accepts_matching_joint_targets():
    schema = _schema()
    joint_count = len(schema.joint_names)
    target = torch.zeros(2, 3 + 2 * joint_count)
    target[:, 3 : 3 + joint_count] = 1.0
    target[:, 3 + joint_count : 3 + 2 * joint_count] = 2.0

    metrics = validate_rendered_ground_truth_alignment(
        target=target,
        gt_joint_pos=torch.ones(2, joint_count),
        gt_joint_vel=torch.full((2, joint_count), 2.0),
        schema=schema,
    )

    assert metrics["joint_pos_max_abs_error"] == 0.0
    assert metrics["joint_vel_max_abs_error"] == 0.0


def test_validate_rendered_ground_truth_alignment_rejects_mismatched_joint_targets():
    schema = _schema()
    joint_count = len(schema.joint_names)
    target = torch.zeros(2, 3 + 2 * joint_count)

    with pytest.raises(ValueError, match="joint positions used for rendering do not match"):
        validate_rendered_ground_truth_alignment(
            target=target,
            gt_joint_pos=torch.ones(2, joint_count),
            gt_joint_vel=torch.zeros(2, joint_count),
            schema=schema,
        )


def test_compute_visualization_metrics_reports_gravity_and_joint_errors():
    schema = _schema()
    joint_count = len(schema.joint_names)
    target = torch.zeros(2, 3 + 2 * joint_count)
    prediction = torch.zeros_like(target)
    prediction[:, :3] = 1.0
    prediction[:, 3 : 3 + joint_count] = 2.0
    prediction[:, 3 + joint_count : 3 + 2 * joint_count] = 3.0

    metrics = compute_visualization_metrics(
        target=target,
        prediction=prediction,
        schema=schema,
        gt_joint_pos=torch.zeros(2, joint_count),
        gt_joint_vel=torch.zeros(2, joint_count),
        pred_joint_pos=torch.full((2, joint_count), 2.0),
        pred_joint_vel=torch.full((2, joint_count), 3.0),
    )

    assert metrics["gravity_mae"] == pytest.approx(1.0)
    assert metrics["root_mae"] == pytest.approx(1.0)
    assert metrics["joint_pos_mae"] == pytest.approx(2.0)
    assert metrics["joint_vel_mae"] == pytest.approx(3.0)
    assert metrics["gravity_mae_by_frame"] == [1.0, 1.0]
    assert metrics["root_mae_by_frame"] == [1.0, 1.0]


def test_expand_rendered_frames_to_motion_timeline_repeats_gap_frames_for_whole_motion():
    frame_a = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame_b = np.full((2, 2, 3), 20, dtype=np.uint8)
    frame_c = np.full((2, 2, 3), 30, dtype=np.uint8)

    expanded = expand_rendered_frames_to_motion_timeline(
        [frame_a, frame_b, frame_c],
        future_frame_indices=(8, 9, 12),
        whole_motion=True,
    )

    assert len(expanded) == 5
    np.testing.assert_array_equal(expanded[0], frame_a)
    np.testing.assert_array_equal(expanded[1], frame_b)
    np.testing.assert_array_equal(expanded[2], frame_b)
    np.testing.assert_array_equal(expanded[3], frame_b)
    np.testing.assert_array_equal(expanded[4], frame_c)


def test_build_whole_motion_visualization_batch_prefers_smallest_future_step():
    schema = _schema(joint_count=2)
    sequence = MotionFeatureSequence(
        motion_file="demo.npz",
        motion_name="demo",
        segments=(MotionSegment(start_frame=0, end_frame=7),),
        full_features=torch.zeros(7, 7),
        reference_features=torch.arange(49, dtype=torch.float32).reshape(7, 7),
        target_features=torch.arange(49, dtype=torch.float32).reshape(7, 7),
    )
    dataset = ReferenceMotionMAEDataset(
        sequences=(sequence,),
        window_indices=((0, 2), (0, 3), (0, 4)),
        past_frames=3,
        future_frames=2,
        reference_mean=torch.zeros(7),
        reference_std=torch.ones(7),
        target_mean=torch.zeros(7),
        target_std=torch.ones(7),
    )
    canonical_sequence = type(
        "Sequence",
        (),
        {
            "segments": sequence.segments,
            "motion_file": sequence.motion_file,
            "motion_name": sequence.motion_name,
            "body_names": _body_names(),
            "joint_names": _joint_names(2),
            "body_pos_w": torch.zeros(7, len(_body_names()), 3),
            "body_quat_w": torch.zeros(7, len(_body_names()), 4),
            "body_lin_vel_w": torch.zeros(7, len(_body_names()), 3),
            "body_ang_vel_w": torch.zeros(7, len(_body_names()), 3),
            "joint_pos": torch.zeros(7, 2),
            "joint_vel": torch.zeros(7, 2),
        },
    )()

    class DummyModel:
        def __call__(self, reference):
            future_frames = 2
            d_target = 7
            prediction = torch.zeros(reference.shape[0], future_frames, d_target, dtype=torch.float32)
            prediction[:, :, 0] = reference[:, -1, 0].unsqueeze(-1)
            return {"prediction": prediction}

    batch = build_whole_motion_visualization_batch(
        dataset=dataset,
        sequence=sequence,
        canonical_sequence=canonical_sequence,
        schema=schema,
        model=DummyModel(),
        device=torch.device("cpu"),
    )

    assert batch.center_t is None
    assert batch.future_frame_indices == (3, 4, 5, 6)
    assert batch.future_step_ahead == (1, 1, 1, 2)
    assert batch.center_t_by_frame == (2, 3, 4, 4)
    assert batch.reference.shape == (4, 3, 7)
    assert batch.target.shape == (4, 7)


def test_build_motion_mae_model_from_checkpoint_rejects_encoder_checkpoint(tmp_path):
    checkpoint_path = _write_encoder_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="full Motion MAE checkpoint"):
        build_motion_mae_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"))


def test_resolve_renderer_size_returns_requested_size_when_within_framebuffer():
    model = type(
        "Model",
        (),
        {"vis": type("Vis", (), {"global_": type("Global", (), {"offwidth": 800, "offheight": 600})()})()},
    )()

    assert resolve_renderer_size(model, requested_width=640, requested_height=480) == (640, 480)


def test_resolve_renderer_size_scales_down_to_fit_framebuffer():
    model = type(
        "Model",
        (),
        {"vis": type("Vis", (), {"global_": type("Global", (), {"offwidth": 640, "offheight": 480})()})()},
    )()

    assert resolve_renderer_size(model, requested_width=640, requested_height=720) == (426, 480)


def test_update_tracking_camera_centers_and_zooms_to_body_bounds():
    model = type(
        "Model",
        (),
        {
            "vis": type(
                "Vis",
                (),
                {"global_": type("Global", (), {"azimuth": -140.0, "elevation": -20.0, "fovy": 45.0})()},
            )()
        },
    )()
    camera = type(
        "Camera",
        (),
        {
            "type": None,
            "fixedcamid": 0,
            "trackbodyid": 0,
            "lookat": np.zeros(3, dtype=np.float32),
            "distance": 0.0,
            "azimuth": 0.0,
            "elevation": 0.0,
        },
    )()
    mujoco_module = type("Mujoco", (), {"mjtCamera": type("Enum", (), {"mjCAMERA_FREE": "free"})})()
    positions = np.asarray(
        [
            [-0.2, -0.1, 0.0],
            [0.2, 0.1, 1.0],
            [0.1, -0.1, 0.4],
        ],
        dtype=np.float32,
    )

    update_tracking_camera(
        camera,
        mj_model=model,
        body_positions=positions,
        frame_width=640,
        frame_height=480,
        mujoco_module=mujoco_module,
    )

    np.testing.assert_allclose(camera.lookat, np.asarray([0.0, 0.0, 0.5], dtype=np.float32))
    assert camera.type == "free"
    assert camera.fixedcamid == -1
    assert camera.trackbodyid == -1
    assert camera.distance >= 3.0
    assert camera.azimuth == -140.0
    assert camera.elevation == -20.0
