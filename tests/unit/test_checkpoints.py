import torch

from gmtp.models import Critic, FiLMAttnResActor
from gmtp.runtime.checkpoints import (
    CHECKPOINT_VERSION,
    build_training_checkpoint,
    load_checkpoint_v2,
    save_checkpoint_v2,
)


def _joint_params(action_dim: int = 2):
    return {
        "joint_names": [f"joint_{idx}" for idx in range(action_dim)],
        "joint_effort_limits": torch.ones(action_dim),
        "joint_pos_limits": torch.tensor([[-1.0, 1.0]] * action_dim),
        "joint_stiffness": torch.ones(action_dim),
        "joint_damping": torch.full((action_dim,), 0.1),
        "action_offset": torch.zeros(action_dim),
        "action_scale": torch.ones(action_dim),
    }


def test_checkpoint_v2_roundtrip(tmp_path):
    actor = FiLMAttnResActor(robot_obs_dim=7, motion_obs_dim=5, action_dim=2, num_blocks=4, attn_block_size=2)
    critic = Critic(obs_dim=3)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
        artifacts={"run_dir": "runs/train/demo"},
    )

    path = save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")
    loaded = load_checkpoint_v2(path)

    assert loaded.checkpoint_version == CHECKPOINT_VERSION
    assert loaded.meta["actor_type"] == "film_attn_res"
    assert loaded.env["action_mode"] == "offset"
    assert loaded.env["root_name"] == "torso_link"
    assert loaded.env["anchor_body_name"] == "torso_link"
    assert loaded.motion_files[0].endswith("115_06_stageii.npz")


def test_load_checkpoint_v2_rejects_non_v2_payload(tmp_path):
    legacy_path = tmp_path / "legacy.pth"
    torch.save({"actor_type": "film_attn_res"}, legacy_path)

    try:
        load_checkpoint_v2(legacy_path)
    except ValueError as exc:
        assert "CheckpointV2" in str(exc)
    else:
        raise AssertionError("Expected non-v2 payload to be rejected.")
