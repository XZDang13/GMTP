import json
import sys
import types
from pathlib import Path

import pytest
import torch

from gmtp.cli.main import build_parser, main
from gmtp.integrations.ref2act.observation_history import build_robot_policy_window_lengths
from gmtp.motion_vae import (
    FeatureSliceSpec,
    MotionEncoderCheckpointV1,
    MotionFeatureSchema,
    save_motion_encoder_checkpoint,
)
from gmtp.models import Critic, FiLMResActor
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.config import IsaacEvalConfig, RunConfig, Sim2SimEvalConfig
from gmtp.runtime.eval_isaac import IsaacEvalRunner
from gmtp.runtime.train_runner import TrainRunner


class _DummyTrainEnv:
    def __init__(self, batch_size: int):
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.motion_lib = types.SimpleNamespace(body_names=("pelvis", "left_hand", "right_hand"))
        self.step_count = 0
        self.reference_motion = self._build_reference_motion()

    def _build_reference_motion(self):
        return types.SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2]], dtype=torch.float32).repeat(self.batch_size, 1),
            joint_vel=torch.tensor([[0.3, 0.4]], dtype=torch.float32).repeat(self.batch_size, 1),
            body_positions=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.0, 0.3, 0.2]]],
                dtype=torch.float32,
            ).repeat(self.batch_size, 1, 1),
            body_quaternions=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ).repeat(self.batch_size, 1, 1),
            anchor_body_index=0,
        )

    def reset(self):
        self.step_count = 0
        self.reference_motion = self._build_reference_motion()
        obs = {
            "motion": torch.zeros(self.batch_size, 7),
            "robot": torch.zeros(self.batch_size, 48),
            "privilege": torch.zeros(self.batch_size, 5),
        }
        return obs, {}

    def step(self, action):
        self.step_count += 1
        self.reference_motion = self._build_reference_motion()
        obs = {
            "motion": torch.full((self.batch_size, 7), float(self.step_count)),
            "robot": torch.full((self.batch_size, 48), float(self.step_count)),
            "privilege": torch.zeros(self.batch_size, 5),
        }
        reward = torch.ones(self.batch_size)
        terminate = torch.zeros(self.batch_size, dtype=torch.bool)
        timeout = torch.zeros(self.batch_size, dtype=torch.bool)
        return obs, reward, terminate, timeout, {}

    def get_joint_params(self):
        return {
            "joint_names": ["j0", "j1"],
            "joint_effort_limits": torch.ones(2),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "joint_stiffness": torch.ones(2),
            "joint_damping": torch.full((2,), 0.1),
            "action_offset": torch.zeros(2),
            "action_scale": torch.ones(2),
            "action_mode": "offset",
        }

    def close(self):
        return None


def _fake_train_module():
    cfg = types.SimpleNamespace(
        scene=types.SimpleNamespace(num_envs=2),
        action_space=2,
        expert_motion_file=["env/assests/115_06_stageii.npz"],
        action=types.SimpleNamespace(mode="offset"),
        action_mod="Offset",
        root_link_name="torso_link",
        anchor_body_name="torso_link",
    )
    return types.SimpleNamespace(make_training_env=lambda window_lengths=None: (_DummyTrainEnv(batch_size=2), cfg))


def _fake_eval_module():
    cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=1), action_space=2)
    env = _DummyTrainEnv(batch_size=1)
    env.reset = lambda: (
        {
            "motion": torch.zeros(1, 7),
            "robot": torch.zeros(1, 12),
            "privilege": torch.zeros(1, 5),
        },
        {},
    )
    return types.SimpleNamespace(
        make_eval_env=lambda motion_files, show_reference_motion=False, window_lengths=None: (env, cfg)
    )


def _motion_schema() -> MotionFeatureSchema:
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
    schema = _motion_schema()
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


def _write_checkpoint(
    tmp_path,
    *,
    motion_files: list[str] | None = None,
    robot_window_length: int = 1,
    motion_obs_dim: int = 7,
    motion_encoder_checkpoint: str | None = None,
):
    robot_obs_dim = 12 * robot_window_length
    actor = FiLMResActor(
        robot_obs_dim=robot_obs_dim,
        motion_obs_dim=motion_obs_dim,
        action_dim=2,
        num_blocks=4,
        robot_window_length=robot_window_length,
    )
    critic = Critic(obs_dim=5)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=motion_files or ["env/assests/115_06_stageii.npz"],
        joint_params={
            "joint_names": ["j0", "j1"],
            "joint_effort_limits": torch.ones(2),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "joint_stiffness": torch.ones(2),
            "joint_damping": torch.full((2,), 0.1),
            "action_offset": torch.zeros(2),
            "action_scale": torch.ones(2),
        },
        action_mode="offset",
        root_name="torso_link",
        anchor_body_name="torso_link",
        motion_encoder_checkpoint=motion_encoder_checkpoint,
        observation_window_lengths=(
            build_robot_policy_window_lengths(robot_window_length) if robot_window_length > 1 else None
        ),
    )
    return save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")


def _write_isaac_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def _write_sim2sim_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path)


def test_cli_parser_builds_train_and_eval_commands():
    parser = build_parser()
    args = parser.parse_args(["train", "--num-updates", "5", "--motion-encoder-checkpoint", "encoder.pth"])
    assert args.command == "train"
    assert args.num_updates == 5
    assert args.num_blocks == 6
    assert args.robot_window_length == 4
    assert args.motion_encoder_checkpoint == "encoder.pth"

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--attn-block-size", "5"])

    args = parser.parse_args(
        [
            "eval",
            "sim2sim",
            "--checkpoint",
            "foo.pth",
            "--motion-encoder-checkpoint",
            "encoder.pth",
            "--motion-files",
            "foo",
            "bar",
            "--action-mode",
            "residual",
            "--num-steps",
            "12",
        ]
    )
    assert args.command == "eval"
    assert args.eval_target == "sim2sim"
    assert args.motion_files == ["foo", "bar"]
    assert args.motion_encoder_checkpoint == "encoder.pth"
    assert args.action_mode == "residual"
    assert args.num_steps == 12
    assert args.robot_window_length is None
    assert args.save_video is False
    assert not hasattr(args, "attn_block_size")

    args = parser.parse_args(["eval", "sim2sim", "--checkpoint", "foo.pth", "--save-video"])
    assert args.save_video is True

    args = parser.parse_args(["pretrain", "motion-vae", "--config", "motion_vae.json", "--device", "cpu"])
    assert args.command == "pretrain"
    assert args.pretrain_target == "motion-vae"
    assert args.config == "motion_vae.json"
    assert args.device == "cpu"


def test_cli_parser_rejects_removed_migrate_checkpoint_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["migrate-checkpoint", "--checkpoint", "legacy.pth"])


def test_train_runner_dry_construction(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(RunConfig(use_wandb=False))
    assert runner.actor_type.value == "film_res"
    assert runner.observation_window_lengths == build_robot_policy_window_lengths(4)
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    runner.env.close()


def test_train_runner_constructs_film_res_actor(monkeypatch):
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            num_blocks=4,
            use_wandb=False,
        )
    )
    assert runner.actor_type.value == "film_res"
    assert runner.actor_kwargs == {"num_blocks": 4, "robot_window_length": 4}
    assert runner.observation_window_lengths == build_robot_policy_window_lengths(4)
    assert isinstance(runner.actor, FiLMResActor)
    runner.env.close()


def test_train_runner_constructs_with_motion_encoder(tmp_path, monkeypatch):
    motion_encoder_checkpoint = _write_motion_encoder_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            motion_encoder_checkpoint=str(motion_encoder_checkpoint),
        )
    )

    assert runner.motion_encoder_checkpoint == str(motion_encoder_checkpoint.resolve())
    assert runner.obs_dims["motion"] == 12
    runner.env.close()


def test_train_runner_rollout_with_motion_encoder_updates_actor_statistics(tmp_path, monkeypatch):
    motion_encoder_checkpoint = _write_motion_encoder_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_train_module())
    runner = TrainRunner(
        RunConfig(
            use_wandb=False,
            motion_encoder_checkpoint=str(motion_encoder_checkpoint),
            rollout_steps=1,
            num_updates=1,
        )
    )

    returned_obs = runner.rollout(runner.initial_obs)

    assert returned_obs["motion"].shape == (2, 7)
    assert runner.rollout_buffer.data["motion_observations"].shape[-1] == 12
    runner.env.close()


def test_isaac_eval_runner_dry_construction(tmp_path, monkeypatch):
    checkpoint_path = _write_isaac_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_eval_module())
    runner = IsaacEvalRunner(
        IsaacEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            progress_interval=0,
            output_root=str(tmp_path / "runs"),
        )
    )
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    runner.env.close()


def test_isaac_eval_runner_uses_motion_encoder_checkpoint(tmp_path, monkeypatch):
    motion_encoder_checkpoint = _write_motion_encoder_checkpoint(tmp_path)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        motion_obs_dim=12,
        motion_encoder_checkpoint=str(motion_encoder_checkpoint),
    )
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.isaac_env", _fake_eval_module())
    runner = IsaacEvalRunner(
        IsaacEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            progress_interval=0,
            output_root=str(tmp_path / "runs"),
        )
    )

    assert runner.motion_encoder_checkpoint == str(motion_encoder_checkpoint.resolve())
    assert runner.obs_dims["motion"] == 12
    runner.env.close()


@pytest.mark.skipif(pytest.importorskip("mujoco", reason="mujoco required") is None, reason="mujoco required")
def test_sim2sim_runner_accepts_v2_checkpoint(tmp_path):
    from gmtp.runtime.eval_sim2sim import Sim2SimEvalRunner

    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)

    runner = Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            output_root=str(tmp_path / "runs"),
        )
    )

    assert runner.action_mode == "offset"
    assert runner.motion_files[0].endswith("115_06_stageii.npz")


def test_motion_vae_pretrain_cli_dry_run(tmp_path: Path):
    config_path = tmp_path / "motion_vae.json"
    output_root = tmp_path / "runs"
    config_payload = {
        "data": {
            "motion_files": [
                "env/assests/05_05_stageii.npz",
                "env/assests/115_02_stageii.npz",
            ],
            "past_frames": 4,
            "future_frames": 2,
            "split_mode": "by_motion",
            "val_ratio": 0.5,
            "batch_size": 32,
            "num_workers": 0,
            "seed": 0,
            "max_train_windows": 16,
            "max_val_windows": 8,
        },
        "model": {
            "latent_dim": 8,
            "encoder_channels": [16, 16],
            "decoder_hidden_dims": [16],
        },
        "training": {
            "epochs": 1,
            "device": "cpu",
            "log_interval": 100,
        },
        "output_root": str(output_root),
        "run_name": "cli_dry_run",
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    exit_code = main(["pretrain", "motion-vae", "--config", str(config_path)])

    assert exit_code == 0
    summary_paths = list(output_root.glob("pretrain/*/summary.json"))
    assert len(summary_paths) == 1
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    artifacts = summary["artifacts"]
    for artifact_path in artifacts.values():
        assert Path(artifact_path).exists()
