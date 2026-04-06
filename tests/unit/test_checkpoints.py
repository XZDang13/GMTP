import torch

from gmtp.integrations.ref2act.observation_history import build_robot_policy_window_lengths
from gmtp.models import Critic, FiLMResActor
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


def _actor_obs_dims(action_dim: int, robot_window_length: int = 1) -> tuple[int, int]:
    motion_obs_dim = 3 + 2 * action_dim
    robot_step_dim = 6 + 3 * action_dim
    return motion_obs_dim, robot_step_dim * robot_window_length


def test_checkpoint_v2_roundtrip(tmp_path):
    motion_encoder_checkpoint = tmp_path / "motion_encoder.pth"
    motion_encoder_checkpoint.write_text("stub", encoding="utf-8")
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2, robot_window_length=4)
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=4,
        robot_window_length=4,
    )
    critic = Critic(obs_dim=5)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
        motion_encoder_checkpoint=str(motion_encoder_checkpoint),
        observation_window_lengths=build_robot_policy_window_lengths(4),
        artifacts={"run_dir": "runs/train/demo"},
    )

    path = save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")
    loaded = load_checkpoint_v2(path)

    assert loaded.checkpoint_version == CHECKPOINT_VERSION
    assert loaded.meta["actor_type"] == "film_res"
    assert loaded.meta["actor_kwargs"] == {"num_blocks": 4, "robot_window_length": 4}
    assert loaded.env["action_mode"] == "offset"
    assert loaded.env["root_name"] == "torso_link"
    assert loaded.env["anchor_body_name"] == "torso_link"
    assert loaded.motion_encoder_checkpoint == str(motion_encoder_checkpoint.resolve())
    assert loaded.observation_window_lengths == build_robot_policy_window_lengths(4)
    assert loaded.motion_files[0].endswith("115_06_stageii.npz")


def test_checkpoint_v2_defaults_to_legacy_single_frame_window_lengths_when_missing():
    motion_obs_dim, robot_obs_dim = _actor_obs_dims(action_dim=2)
    checkpoint = build_training_checkpoint(
        actor=FiLMResActor(robot_obs_dim=robot_obs_dim, motion_obs_dim=motion_obs_dim, action_dim=2, num_blocks=4),
        critic=Critic(obs_dim=5),
        motion_files=["env/assests/115_06_stageii.npz"],
        joint_params=_joint_params(),
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
    )

    assert checkpoint.observation_window_lengths == {}


def test_load_checkpoint_v2_rejects_non_v2_payload(tmp_path):
    legacy_path = tmp_path / "legacy.pth"
    torch.save({"actor_type": "film_res"}, legacy_path)

    try:
        load_checkpoint_v2(legacy_path)
    except ValueError as exc:
        assert "CheckpointV2" in str(exc)
    else:
        raise AssertionError("Expected non-v2 payload to be rejected.")
