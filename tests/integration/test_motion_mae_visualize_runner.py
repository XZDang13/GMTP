import json
from pathlib import Path

import numpy as np

import gmtp.runtime.motion_mae_visualize as motion_mae_visualize
from gmtp.motion_mae import (
    FeatureSliceSpec,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    MotionMAEModelConfig,
    MotionMAEOptimizerConfig,
    MotionMAEPretrainConfig,
    MotionMAETrainingConfig,
    MotionFeatureSchema,
    ReferenceMotionMAE,
    build_motion_mae_checkpoint,
    save_motion_mae_checkpoint,
)
from gmtp.runtime.config import MotionMAEVisualizationConfig
from gmtp.runtime.motion_mae_visualize import MotionMAEVisualizerRunner

TEST_MOTION_FILE = "env/assests/85_09_stageii.npz"
TEST_MOTION_NAME = "85_09_stageii"


def _load_motion_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    with np.load(TEST_MOTION_FILE, allow_pickle=True) as payload:
        joint_names = tuple(str(item) for item in payload["joint_names"].tolist())
        body_names = tuple(str(item) for item in payload["body_names"].tolist())
    return joint_names, body_names


def _schema() -> MotionFeatureSchema:
    joint_names, body_names = _load_motion_names()
    joint_count = len(joint_names)
    return MotionFeatureSchema(
        d_ref=3 + 2 * joint_count,
        d_target=3 + 2 * joint_count,
        full_feature_dim=3 + 2 * joint_count + 12,
        base_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 3 + 2 * joint_count),
            FeatureSliceSpec("end_effector", 3 + 2 * joint_count, 3 + 2 * joint_count + 12),
        ),
        reference_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 3 + 2 * joint_count),
        ),
        target_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 3 + 2 * joint_count),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, 3 + 2 * joint_count),
        anchor_body_name="pelvis",
        end_effector_body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ),
        reference_feature_names=("root", "joint"),
        target_feature_names=("root", "joint"),
        policy_feature_names=("root", "joint"),
        joint_names=joint_names,
        body_names=body_names,
        reference_mean=tuple(0.0 for _ in range(3 + 2 * joint_count)),
        reference_std=tuple(1.0 for _ in range(3 + 2 * joint_count)),
        target_mean=tuple(0.0 for _ in range(3 + 2 * joint_count)),
        target_std=tuple(1.0 for _ in range(3 + 2 * joint_count)),
    )


def _pretrain_config() -> MotionMAEPretrainConfig:
    return MotionMAEPretrainConfig(
        data=MotionMAEDataConfig(
            motion_files=(TEST_MOTION_FILE,),
            past_frames=4,
            future_frames=2,
            split_mode="by_window",
            val_ratio=0.5,
            max_train_windows=8,
            max_val_windows=4,
        ),
        feature=MotionMAEFeatureConfig(
            anchor_body_name="pelvis",
            reference_feature_names=("root", "joint"),
            target_feature_names=("root", "joint"),
            policy_feature_names=("root", "joint"),
        ),
        model=MotionMAEModelConfig(
            d_model=32,
            latent_dim=8,
            encoder_layers=2,
            decoder_layers=1,
            nhead=4,
            dim_feedforward=64,
        ),
        optimizer=MotionMAEOptimizerConfig(lr=1.0e-3, weight_decay=0.0),
        training=MotionMAETrainingConfig(epochs=1, device="cpu", grad_clip_norm=1.0, log_interval=1),
    )


def _write_config(path: Path) -> Path:
    path.write_text(json.dumps(_pretrain_config().to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_motion_mae_checkpoint(path: Path) -> Path:
    schema = _schema()
    model = ReferenceMotionMAE(
        input_dim=schema.d_ref,
        target_dim=schema.d_target,
        past_frames=4,
        future_frames=2,
        latent_dim=8,
        d_model=32,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=64,
    )
    checkpoint = build_motion_mae_checkpoint(
        model=model,
        optimizer=None,
        schema=schema,
        config=_pretrain_config(),
        epoch=1,
        best_metric=0.25,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )
    return save_motion_mae_checkpoint(checkpoint, path)


class _FakePlayback:
    _counter = 0

    def __init__(self, **kwargs) -> None:
        type(self)._counter += 1
        self._color = 48 if type(self)._counter % 2 else 160

    def render_frame(self, frame_state) -> np.ndarray:
        return np.full((120, 160, 3), self._color, dtype=np.uint8)

    def close(self) -> None:
        return None


class _FakeWriter:
    def __init__(self, path: str | Path, fps: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames: list[np.ndarray] = []
        self.fps = int(fps)

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(np.asarray(frame, dtype=np.uint8))

    def close(self) -> None:
        assert self.frames
        payload = np.stack(self.frames, axis=0)
        self.path.write_bytes(payload.tobytes())


def _fake_imwrite(path: str | Path, frame: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(frame, dtype=np.uint8).tobytes())


def test_motion_mae_visualizer_runner_writes_summary_npz_and_video(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path / "motion_mae_visualize.json")
    checkpoint_path = _write_motion_mae_checkpoint(tmp_path / "motion_mae.pth")
    monkeypatch.setattr(motion_mae_visualize, "PassiveMujocoPlayback", _FakePlayback)
    monkeypatch.setattr(motion_mae_visualize.imageio, "get_writer", _FakeWriter)

    summary = MotionMAEVisualizerRunner(
        MotionMAEVisualizationConfig(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            split="val",
            sample_index=0,
            output_root=str(tmp_path / "runs"),
            device="cpu",
        )
    ).visualize()

    comparison_path = Path(summary["artifacts"]["comparison_npz"])
    video_path = Path(summary["artifacts"]["video"])

    assert comparison_path.exists()
    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert summary["media_type"] == "video"
    assert summary["split"] == "val"
    assert summary["motion_name"] == TEST_MOTION_NAME
    assert "gravity_mae" in summary["metrics"]
    assert "joint_pos_mae" in summary["metrics"]
    assert "root_mae" in summary["metrics"]
    assert summary["pred_view_uses_ground_truth_root_position"] is True
    assert summary["pred_view_uses_ground_truth_root_orientation"] is True
    assert summary["pred_view_uses_ground_truth_root_velocity"] is True
    assert summary["target_joint_alignment_verified"] is True
    assert (
        summary["render_comparison_mode"]
        == "ground_truth_root_position_and_orientation_plus_predicted_joint_trajectory"
    )
    assert summary["future_step_ahead"] == [1, 2]

    payload = np.load(comparison_path)
    assert payload["reference"].shape == (4, 49)
    assert payload["target"].shape == (2, 49)
    assert payload["prediction"].shape == (2, 49)
    assert "pred_root_quat" not in payload
    assert payload["gt_joint_pos"].shape == (2, 23)
    assert payload["pred_joint_vel"].shape == (2, 23)
    assert payload["future_step_ahead"].tolist() == [1, 2]
    assert bool(payload["pred_view_uses_ground_truth_root_position"].item()) is True
    assert bool(payload["pred_view_uses_ground_truth_root_orientation"].item()) is True
    assert bool(payload["pred_view_uses_ground_truth_root_velocity"].item()) is True
    assert bool(payload["target_joint_alignment_verified"].item()) is True


def test_motion_mae_visualizer_runner_can_select_single_future_frame(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path / "motion_mae_visualize_single_frame.json")
    checkpoint_path = _write_motion_mae_checkpoint(tmp_path / "motion_mae_single_frame.pth")
    monkeypatch.setattr(motion_mae_visualize, "PassiveMujocoPlayback", _FakePlayback)
    monkeypatch.setattr(motion_mae_visualize.imageio, "imwrite", _fake_imwrite)

    summary = MotionMAEVisualizerRunner(
        MotionMAEVisualizationConfig(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            split="val",
            sample_index=0,
            future_frame_index=1,
            output_root=str(tmp_path / "runs-single-frame"),
            device="cpu",
        )
    ).visualize()

    payload = np.load(Path(summary["artifacts"]["comparison_npz"]))
    image_path = Path(summary["artifacts"]["image"])
    assert summary["future_frame_index"] == 1
    assert summary["media_type"] == "image"
    assert len(summary["future_frame_indices"]) == 1
    assert summary["future_step_ahead"] == [2]
    assert summary["pred_view_uses_ground_truth_root_position"] is True
    assert summary["pred_view_uses_ground_truth_root_orientation"] is True
    assert image_path.exists()
    assert image_path.suffix == ".png"
    assert payload["future_frame_index"].item() == 1
    assert len(payload["future_frame_indices"].tolist()) == 1
    assert payload["future_step_ahead"].tolist() == [2]
    assert payload["target"].shape == (1, 49)
    assert payload["prediction"].shape == (1, 49)
    assert "pred_root_quat" not in payload
    assert payload["gt_joint_pos"].shape == (1, 23)


def test_motion_mae_visualizer_runner_can_render_whole_motion(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path / "motion_mae_visualize_whole_motion.json")
    checkpoint_path = _write_motion_mae_checkpoint(tmp_path / "motion_mae_whole_motion.pth")
    monkeypatch.setattr(motion_mae_visualize, "PassiveMujocoPlayback", _FakePlayback)
    monkeypatch.setattr(motion_mae_visualize.imageio, "get_writer", _FakeWriter)

    summary = MotionMAEVisualizerRunner(
        MotionMAEVisualizationConfig(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            split="val",
            motion_name=TEST_MOTION_NAME,
            whole_motion=True,
            output_root=str(tmp_path / "runs-whole-motion"),
            device="cpu",
        )
    ).visualize()

    comparison_path = Path(summary["artifacts"]["comparison_npz"])
    video_path = Path(summary["artifacts"]["video"])

    assert comparison_path.exists()
    assert video_path.exists()
    assert summary["whole_motion"] is True
    assert summary["window_source"] == "all_valid_windows_for_motion"
    assert summary["center_t"] is None
    assert len(summary["center_t_by_frame"]) == len(summary["future_frame_indices"])
    assert len(summary["future_frame_indices"]) > 10
    assert summary["future_frame_indices"] == sorted(summary["future_frame_indices"])
    assert summary["future_step_ahead"][0] == 1
    assert summary["rendered_video_frame_count"] >= len(summary["future_frame_indices"])
    assert summary["rendered_video_duration_seconds"] > 0.0

    payload = np.load(comparison_path)
    assert bool(payload["whole_motion"].item()) is True
    assert payload["center_t"].item() == -1
    assert payload["reference"].ndim == 3
    assert payload["target"].ndim == 2
    assert payload["prediction"].ndim == 2
    assert payload["future_frame_indices"].shape[0] == payload["center_t_by_frame"].shape[0]
