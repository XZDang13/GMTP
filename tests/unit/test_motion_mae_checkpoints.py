from pathlib import Path

import pytest
import torch

from gmtp.motion_mae import (
    FeatureSliceSpec,
    MotionFeatureSchema,
    MotionMAEDataConfig,
    MotionMAEFeatureConfig,
    MotionMAEModelConfig,
    MotionMAEPretrainConfig,
    ReferenceMotionMAE,
    build_frozen_motion_mae_encoder,
    build_motion_mae_checkpoint,
    build_motion_mae_encoder_checkpoint,
    load_motion_mae_checkpoint,
    load_motion_mae_encoder_checkpoint,
    save_motion_mae_checkpoint,
    save_motion_mae_encoder_checkpoint,
)
from gmtp.runtime.policy import resolve_motion_mae_checkpoint_path
from gmtp.runtime.checkpoints import CheckpointV2


def _schema() -> MotionFeatureSchema:
    return MotionFeatureSchema(
        d_ref=13,
        d_target=13,
        full_feature_dim=13,
        base_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
            FeatureSliceSpec("end_effector", 7, 13),
        ),
        reference_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
            FeatureSliceSpec("end_effector", 7, 13),
        ),
        target_slices=(
            FeatureSliceSpec("root", 0, 3, weight=1.0),
            FeatureSliceSpec("joint", 3, 7, weight=1.0),
            FeatureSliceSpec("end_effector", 7, 13, weight=0.5),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, 7),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_hand", "right_hand"),
        reference_feature_names=("root", "joint", "end_effector"),
        target_feature_names=("root", "joint", "end_effector"),
        policy_feature_names=("root", "joint"),
        joint_names=("j0", "j1"),
        body_names=("pelvis", "left_hand", "right_hand"),
        reference_mean=tuple(0.0 for _ in range(13)),
        reference_std=tuple(1.0 for _ in range(13)),
        target_mean=tuple(0.0 for _ in range(13)),
        target_std=tuple(1.0 for _ in range(13)),
    )


def _config() -> MotionMAEPretrainConfig:
    return MotionMAEPretrainConfig(
        data=MotionMAEDataConfig(
            motion_files=("env/assests/115_02_stageii.npz",),
            past_frames=4,
            future_frames=2,
            split_mode="by_window",
            val_ratio=0.5,
        ),
        feature=MotionMAEFeatureConfig(
            end_effector_body_names=("left_hand", "right_hand"),
        ),
        model=MotionMAEModelConfig(
            d_model=16,
            latent_dim=6,
            encoder_layers=2,
            decoder_layers=1,
            nhead=4,
            dim_feedforward=32,
        ),
    )
def _write_motion_mae_encoder_checkpoint(tmp_path: Path) -> Path:
    model = ReferenceMotionMAE(
        input_dim=13,
        target_dim=13,
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
        schema=_schema(),
        config=_config(),
        epoch=1,
        best_metric=0.7,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )
    return save_motion_mae_encoder_checkpoint(checkpoint, tmp_path / "motion_mae_encoder.pth")


def test_motion_mae_checkpoint_roundtrip(tmp_path):
    model = ReferenceMotionMAE(
        input_dim=13,
        target_dim=13,
        past_frames=4,
        future_frames=2,
        latent_dim=6,
        d_model=16,
        encoder_layers=2,
        decoder_layers=1,
        nhead=4,
        dim_feedforward=32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    checkpoint = build_motion_mae_checkpoint(
        model=model,
        optimizer=optimizer,
        schema=_schema(),
        config=_config(),
        epoch=2,
        best_metric=1.23,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )

    path = save_motion_mae_checkpoint(checkpoint, tmp_path / "motion_mae.pth")
    loaded = load_motion_mae_checkpoint(path)

    assert loaded.meta["token_dim"] == 16
    assert "pooling_type" not in loaded.meta["model_kwargs"]
    assert loaded.training["epoch"] == 2
    assert loaded.schema.d_ref == 13
    assert "model" in loaded.model
    assert not any(key.startswith("latent_pool.") for key in loaded.model["model"])


def test_motion_mae_encoder_loader_and_frozen_wrapper(tmp_path):
    path = _write_motion_mae_encoder_checkpoint(tmp_path)

    loaded = load_motion_mae_encoder_checkpoint(path)
    frozen_encoder = build_frozen_motion_mae_encoder(path, device="cpu")
    reference = torch.randn(2, 4, 13)

    assert loaded.meta["token_dim"] == 16
    assert "pooling_type" not in loaded.meta["model_kwargs"]
    assert not any(key.startswith("latent_pool.") for key in loaded.model["encoder_state"])
    assert frozen_encoder(reference).shape == (2, 4, 16)
    assert frozen_encoder.token_dim == 16
    assert all(not parameter.requires_grad for parameter in frozen_encoder.parameters())


def test_motion_mae_encoder_loader_ignores_stale_attention_pool_weights(tmp_path):
    path = _write_motion_mae_encoder_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["model"]["encoder_state"]["latent_pool.query"] = torch.randn(1, 1, 16)
    legacy_path = tmp_path / "legacy_motion_mae_encoder.pth"
    torch.save(payload, legacy_path)

    frozen_encoder = build_frozen_motion_mae_encoder(legacy_path, device="cpu")

    assert frozen_encoder(torch.randn(2, 4, 13)).shape == (2, 4, 16)


def test_motion_mae_encoder_loader_rejects_missing_token_encoder_weights(tmp_path):
    path = _write_motion_mae_encoder_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu")
    payload["model"]["encoder_state"].pop("input_proj.weight")
    legacy_path = tmp_path / "missing_token_encoder_motion_mae_encoder.pth"
    torch.save(payload, legacy_path)

    with pytest.raises(ValueError, match="token-encoder weights"):
        build_frozen_motion_mae_encoder(legacy_path, device="cpu")


def test_resolve_motion_mae_checkpoint_path_prefers_override(tmp_path):
    checkpoint_path = _write_motion_mae_encoder_checkpoint(tmp_path)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 4, "robot_window_length": 1}},
        model={"actor": {}, "critic": {}},
        env={},
        artifacts={"motion_mae_encoder_checkpoint": str(tmp_path / "missing_encoder.pth")},
    )

    resolved = resolve_motion_mae_checkpoint_path(checkpoint, override=checkpoint_path)

    assert resolved == checkpoint_path.resolve()
