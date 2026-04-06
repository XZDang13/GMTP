import types
from pathlib import Path

import torch

from gmtp.motion_vae import (
    FeatureSliceSpec,
    MotionEncoderCheckpointV1,
    MotionFeatureSchema,
    save_motion_encoder_checkpoint,
)
from gmtp.runtime.checkpoints import CheckpointV2
from gmtp.runtime.policy import (
    build_motion_latent_adapter,
    resolve_motion_encoder_checkpoint_path,
)


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
            FeatureSliceSpec("root", 0, 3),
            FeatureSliceSpec("joint", 3, 7),
            FeatureSliceSpec("end_effector", 7, 13),
        ),
        policy_motion_slice=FeatureSliceSpec("policy_motion", 0, 7),
        anchor_body_name="pelvis",
        end_effector_body_names=("left_hand", "right_hand"),
        reference_feature_names=("root", "joint", "end_effector"),
        target_feature_names=("root", "joint", "end_effector"),
        policy_feature_names=("root", "joint"),
        gravity_vector=(0.0, 0.0, -1.0),
        joint_names=("j0", "j1"),
        body_names=("pelvis", "left_hand", "right_hand"),
        reference_mean=tuple(0.0 for _ in range(13)),
        reference_std=tuple(1.0 for _ in range(13)),
        target_mean=tuple(0.0 for _ in range(13)),
        target_std=tuple(1.0 for _ in range(13)),
    )


def _write_motion_encoder_checkpoint(tmp_path: Path) -> Path:
    schema = _schema()
    latent_dim = 5
    checkpoint = MotionEncoderCheckpointV1(
        meta={
            "created_at": "2026-04-06T00:00:00",
            "latent_dim": latent_dim,
            "encoder_kwargs": {
                "input_dim": schema.d_ref,
                "window_length": 4,
                "latent_dim": latent_dim,
                "channels": (8,),
                "kernel_size": 3,
                "stride": 1,
                "activation": "silu",
            },
            "frozen": True,
        },
        model={
            "encoder": {
                "conv.0.weight": torch.randn(8, schema.d_ref, 3),
                "conv.0.bias": torch.randn(8),
                "mu_head.weight": torch.randn(latent_dim, 8 * 4),
                "mu_head.bias": torch.randn(latent_dim),
                "logvar_head.weight": torch.randn(latent_dim, 8 * 4),
                "logvar_head.bias": torch.randn(latent_dim),
            }
        },
        schema=schema,
        training={},
        artifacts={},
    )
    return save_motion_encoder_checkpoint(checkpoint, tmp_path / "motion_encoder.pth")


def _reference_motion(batch_size: int = 1):
    return types.SimpleNamespace(
        joint_pos=torch.tensor([[0.1, 0.2]], dtype=torch.float32).repeat(batch_size, 1),
        joint_vel=torch.tensor([[0.3, 0.4]], dtype=torch.float32).repeat(batch_size, 1),
        body_positions=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.2]]],
            dtype=torch.float32,
        ).repeat(batch_size, 1, 1),
        body_quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ).repeat(batch_size, 1, 1),
        anchor_body_index=0,
    )


def test_reference_motion_latent_adapter_supports_isaac_reference_motion(tmp_path):
    checkpoint_path = _write_motion_encoder_checkpoint(tmp_path)
    adapter = build_motion_latent_adapter(checkpoint_path, device="cpu")
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            reference_motion=_reference_motion(batch_size=2),
            motion_lib=types.SimpleNamespace(body_names=("pelvis", "left_hand", "right_hand")),
        )
    )

    adapter.initialize_history(env)
    augmented_dims = adapter.augment_observation_dims({"motion": 7, "robot": 12, "critic": 5, "policy": 19})
    actor_obs = adapter.augment_actor_observation(
        {
            "motion_obs": torch.zeros(2, 7),
            "robot_obs": torch.zeros(2, 12),
        }
    )

    assert adapter.history.shape == (2, 4, 13)
    assert augmented_dims == {"motion": 12, "robot": 12, "critic": 5, "policy": 24}
    assert actor_obs["motion_obs"].shape == (2, 12)
    assert actor_obs["robot_obs"].shape == (2, 12)


def test_reference_motion_latent_adapter_supports_sampled_motion_runtime(tmp_path):
    checkpoint_path = _write_motion_encoder_checkpoint(tmp_path)
    adapter = build_motion_latent_adapter(checkpoint_path, device="cpu")

    sample = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_positions": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.2]]],
            dtype=torch.float32,
        ),
        "body_quaternions": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    }
    env = types.SimpleNamespace(
        motion_id=torch.tensor([0], dtype=torch.long),
        times=torch.tensor([0.0], dtype=torch.float32),
        anchor_body_index=0,
        motion_lib=types.SimpleNamespace(
            body_names=("pelvis", "left_hand", "right_hand"),
            sample_motion=lambda motion_ids, times: sample,
        ),
    )

    adapter.initialize_history(env)
    adapter.update_history(env)

    assert adapter.history.shape == (1, 4, 13)


def test_resolve_motion_encoder_checkpoint_path_prefers_override(tmp_path):
    checkpoint_path = _write_motion_encoder_checkpoint(tmp_path)
    checkpoint = CheckpointV2(
        meta={"actor_type": "film_res", "actor_kwargs": {"num_blocks": 4, "robot_window_length": 1}},
        model={"actor": {}, "critic": {}},
        env={},
        artifacts={"motion_encoder_checkpoint": str(tmp_path / "missing_encoder.pth")},
    )

    resolved = resolve_motion_encoder_checkpoint_path(checkpoint, override=checkpoint_path)

    assert resolved == checkpoint_path.resolve()
