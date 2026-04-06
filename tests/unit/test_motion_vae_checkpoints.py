import torch

from gmtp.motion_vae import (
    FeatureSliceSpec,
    MotionFeatureSchema,
    MotionVAEDataConfig,
    MotionVAEModelConfig,
    MotionVAEPretrainConfig,
    ReferenceMotionVAE,
    build_frozen_motion_encoder,
    build_motion_encoder_checkpoint,
    build_motion_vae_checkpoint,
    load_motion_encoder_checkpoint_v1,
    load_motion_vae_checkpoint,
    save_motion_encoder_checkpoint,
    save_motion_vae_checkpoint,
)
from gmtp.runtime.policy import load_motion_encoder_checkpoint


def _schema() -> MotionFeatureSchema:
    return MotionFeatureSchema(
        d_ref=7,
        d_target=9,
        full_feature_dim=9,
        base_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
            FeatureSliceSpec("end_effector", 7, 9),
        ),
        reference_slices=(
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
        ),
        target_slices=(
            FeatureSliceSpec("root", 0, 3, weight=1.0),
            FeatureSliceSpec("joint", 3, 7, weight=1.0),
            FeatureSliceSpec("end_effector", 7, 9, weight=0.5),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, 7),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        reference_feature_names=("root", "joint"),
        target_feature_names=("root", "joint", "end_effector"),
        policy_feature_names=("root", "joint"),
        joint_names=("j0", "j1"),
        body_names=("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
        reference_mean=tuple(0.0 for _ in range(7)),
        reference_std=tuple(1.0 for _ in range(7)),
        target_mean=tuple(0.0 for _ in range(9)),
        target_std=tuple(1.0 for _ in range(9)),
    )


def _config() -> MotionVAEPretrainConfig:
    return MotionVAEPretrainConfig(
        data=MotionVAEDataConfig(
            motion_files=("env/assests/05_05_stageii.npz", "env/assests/115_02_stageii.npz"),
            past_frames=4,
            future_frames=2,
            split_mode="by_motion",
            val_ratio=0.5,
        ),
        model=MotionVAEModelConfig(
            latent_dim=6,
            encoder_channels=(8, 8),
            decoder_hidden_dims=(10,),
        ),
    )


def test_motion_vae_checkpoint_roundtrip(tmp_path):
    model = ReferenceMotionVAE(
        input_dim=7,
        target_dim=9,
        past_frames=4,
        future_frames=2,
        latent_dim=6,
        encoder_channels=(8, 8),
        decoder_hidden_dims=(10,),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    checkpoint = build_motion_vae_checkpoint(
        model=model,
        optimizer=optimizer,
        schema=_schema(),
        config=_config(),
        epoch=2,
        best_metric=1.23,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )

    path = save_motion_vae_checkpoint(checkpoint, tmp_path / "motion_vae.pth")
    loaded = load_motion_vae_checkpoint(path)

    assert loaded.meta["latent_dim"] == 6
    assert loaded.training["epoch"] == 2
    assert loaded.schema.d_ref == 7
    assert "encoder" in loaded.model
    assert "decoder" in loaded.model


def test_motion_encoder_loader_and_frozen_wrapper(tmp_path):
    model = ReferenceMotionVAE(
        input_dim=7,
        target_dim=9,
        past_frames=4,
        future_frames=2,
        latent_dim=6,
        encoder_channels=(8, 8),
        decoder_hidden_dims=(10,),
    )
    checkpoint = build_motion_encoder_checkpoint(
        model=model,
        schema=_schema(),
        config=_config(),
        epoch=1,
        best_metric=0.7,
        artifacts={"run_dir": "runs/pretrain/demo"},
    )
    path = save_motion_encoder_checkpoint(checkpoint, tmp_path / "motion_encoder.pth")

    loaded = load_motion_encoder_checkpoint_v1(path)
    encoder, schema, latent_dim = load_motion_encoder_checkpoint(path, device=torch.device("cpu"))
    frozen_encoder = build_frozen_motion_encoder(path, device="cpu")
    reference = torch.randn(2, 4, 7)

    assert loaded.meta["latent_dim"] == 6
    assert latent_dim == 6
    assert schema.d_ref == 7
    assert encoder.encode(reference).shape == (2, 6)
    assert frozen_encoder(reference).shape == (2, 6)
    assert all(not parameter.requires_grad for parameter in frozen_encoder.parameters())
