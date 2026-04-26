import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import gmtp.runtime.eval_sim2sim as eval_sim2sim
from gmtp.integrations.ref2act.observation_history import build_robot_policy_window_lengths
from gmtp.models import Critic, FiLMResActor
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.config import Sim2SimEvalConfig


def _write_sim2sim_checkpoint(
    tmp_path: Path,
    *,
    motion_files: list[str] | None = None,
    action_mode: str = "offset",
    root_name: str = "torso_link",
    anchor_body_name: str = "torso_link",
    robot_window_length: int = 1,
) -> Path:
    resolved_motion_files = []
    for raw_motion_file in motion_files or ["env/assests/115_06_stageii.npz"]:
        motion_path = tmp_path / Path(raw_motion_file).name
        motion_path.write_bytes(b"")
        resolved_motion_files.append(str(motion_path))

    actor = FiLMResActor(
        robot_obs_dim=12 * robot_window_length,
        motion_obs_dim=5,
        action_dim=2,
        num_blocks=4,
        robot_window_length=robot_window_length,
    )
    critic = Critic(obs_dim=5)
    checkpoint = build_training_checkpoint(
        actor=actor,
        critic=critic,
        motion_files=resolved_motion_files,
        joint_params={
            "joint_names": ["j0", "j1"],
            "joint_effort_limits": torch.ones(2),
            "joint_pos_limits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "joint_stiffness": torch.ones(2),
            "joint_damping": torch.full((2,), 0.1),
            "action_offset": torch.zeros(2),
            "action_scale": torch.ones(2),
        },
        action_mode=action_mode,
        root_name=root_name,
        anchor_body_name=anchor_body_name,
        observation_window_lengths=(
            build_robot_policy_window_lengths(robot_window_length) if robot_window_length > 1 else None
        ),
    )
    return save_checkpoint_v2(checkpoint, tmp_path / "model_v2.pth")


def _metric_values(bias: float) -> dict[str, float]:
    return {
        "gravity_mae": 0.1 + bias,
        "joint_pos_mae": 0.2 + bias,
        "joint_vel_mae": 0.3 + bias,
    }


def _make_flat_obs(step: int, bias: float) -> torch.Tensor:
    metrics = _metric_values(bias)
    target_projected_gravity = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) + bias + step
    target_joint_pos = torch.tensor([4.0, 5.0], dtype=torch.float32) + bias + step
    target_joint_vel = torch.tensor([6.0, 7.0], dtype=torch.float32) + bias + step
    robot_projected_gravity = target_projected_gravity + metrics["gravity_mae"]
    anchor_ang_vel = torch.tensor([8.0, 9.0, 10.0], dtype=torch.float32) + bias + step
    robot_joint_pos = target_joint_pos + metrics["joint_pos_mae"]
    robot_joint_vel = target_joint_vel + metrics["joint_vel_mae"]
    previous_action = torch.tensor([11.0, 12.0], dtype=torch.float32) + step
    return torch.cat(
        [
            target_projected_gravity,
            target_joint_pos,
            robot_projected_gravity,
            anchor_ang_vel,
            robot_joint_pos,
            robot_joint_vel,
            previous_action,
        ]
    )


def _robot_history_steps(step: int, window_length: int) -> list[int]:
    prefix_count = max(window_length - (step + 1), 0)
    start = max(step - window_length + 1, 0)
    return [0] * prefix_count + list(range(start, step + 1))


def _make_windowed_flat_obs(step: int, bias: float, window_length: int) -> torch.Tensor:
    latest_parts = eval_sim2sim.parse_sim2sim_obs(_make_flat_obs(step=step, bias=bias), action_dim=2)
    history_parts = [
        eval_sim2sim.parse_sim2sim_obs(_make_flat_obs(step=history_step, bias=bias), action_dim=2)
        for history_step in _robot_history_steps(step, window_length)
    ]
    robot_obs = torch.cat(
        [
            torch.cat([parts["robot_projected_gravity"] for parts in history_parts]),
            torch.cat([parts["anchor_ang_vel"] for parts in history_parts]),
            torch.cat([parts["robot_joint_pos"] for parts in history_parts]),
            torch.cat([parts["robot_joint_vel"] for parts in history_parts]),
            torch.cat([parts["previous_action"] for parts in history_parts]),
        ]
    )
    return torch.cat([latest_parts["motion"], robot_obs])


class _FakeMujocoEnv:
    instances: list["_FakeMujocoEnv"] = []

    def __init__(
        self,
        *,
        simulation_dt,
        decimation,
        kp,
        kd,
        effort_limits,
        joint_pos_limits,
        action_offset,
        action_scale,
        expert_motion_file,
        root_link_name,
        anchor_body_name,
        render,
        action_mode,
    ):
        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.policy_dt = simulation_dt * decimation
        self.kp = torch.as_tensor(kp, dtype=torch.float32)
        self.kd = torch.as_tensor(kd, dtype=torch.float32)
        self.effort_limits = torch.as_tensor(effort_limits, dtype=torch.float32)
        self.joint_pos_limits = torch.as_tensor(joint_pos_limits, dtype=torch.float32)
        self.action_offset = torch.as_tensor(action_offset, dtype=torch.float32)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.motion_file = str(expert_motion_file)
        self.root_link_name = root_link_name
        self.anchor_body_name = anchor_body_name
        self.render = render
        self.action_mode = action_mode
        self.action_dim = int(self.action_offset.shape[0])
        self.bias = 0.0 if "115_06" in self.motion_file else 0.25
        self.times = torch.zeros(1, dtype=torch.float32)
        self.target_pos = torch.zeros(self.action_dim, dtype=torch.float32)
        self.mj_model = object()
        self.mj_data = types.SimpleNamespace(
            ctrl=np.zeros(self.action_dim, dtype=np.float32),
            qpos=np.zeros(self.action_dim + 7, dtype=np.float32),
            qvel=np.zeros(self.action_dim + 6, dtype=np.float32),
        )
        self.step_count = 0
        self.closed = False
        type(self).instances.append(self)

    def reset(self):
        self.step_count = 0
        self.times.zero_()
        self.target_pos.zero_()
        self.mj_data.ctrl[:] = 0.0
        self.mj_data.qpos[:] = self.bias
        self.mj_data.qvel[:] = -self.bias
        return _make_flat_obs(step=0, bias=self.bias)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32)
        self.step_count += 1
        self.times = torch.tensor([self.step_count * self.policy_dt], dtype=torch.float32)
        self.target_pos = action + self.bias
        self.mj_data.ctrl[:] = action.numpy()
        self.mj_data.qpos[:] = self.step_count + self.bias
        self.mj_data.qvel[:] = -self.step_count - self.bias
        return _make_flat_obs(step=self.step_count, bias=self.bias)

    def close(self):
        self.closed = True


class _BadObsMujocoEnv(_FakeMujocoEnv):
    def reset(self):
        return torch.zeros(18, dtype=torch.float32)


class _SupportsUnstableInitMujocoEnv(_FakeMujocoEnv):
    instances: list["_SupportsUnstableInitMujocoEnv"] = []

    def __init__(self, *, allow_unstable_init=False, **kwargs):
        super().__init__(**kwargs)
        self.allow_unstable_init = allow_unstable_init


class _StructuredObsMujocoEnv(_FakeMujocoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.get_obs_dict_calls: list[bool] = []

    def _build_obs_parts(self) -> dict[str, torch.Tensor]:
        return eval_sim2sim.parse_sim2sim_obs(_make_flat_obs(step=self.step_count, bias=self.bias), action_dim=self.action_dim)

    def _build_observation_context(self, advance_time: bool = False):
        metrics = _metric_values(self.bias)
        target_projected_gravity = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) + self.bias + self.step_count
        target_joint_pos = torch.tensor([4.0, 5.0], dtype=torch.float32) + self.bias + self.step_count
        target_joint_vel = torch.tensor([6.0, 7.0], dtype=torch.float32) + self.bias + self.step_count
        robot_joint_pos = target_joint_pos + metrics["joint_pos_mae"]
        return types.SimpleNamespace(
            target_projected_gravity=target_projected_gravity,
            target_joint_pos=target_joint_pos,
            target_joint_vel=target_joint_vel,
            projected_gravity=target_projected_gravity + metrics["gravity_mae"],
            anchor_ang_vel_b=torch.tensor([8.0, 9.0, 10.0], dtype=torch.float32) + self.bias + self.step_count,
            joint_pos=robot_joint_pos,
            joint_vel=target_joint_vel + metrics["joint_vel_mae"],
            previous_action=torch.tensor([11.0, 12.0], dtype=torch.float32) + self.step_count,
        )

    def get_obs_dict(self, advance_time: bool = False):
        self.get_obs_dict_calls.append(advance_time)
        parts = self._build_obs_parts()
        return {
            "motion": parts["motion"],
            "robot": parts["robot"],
        }

    def reset(self):
        super().reset()
        return torch.tensor([123.0], dtype=torch.float32)

    def step(self, action):
        super().step(action)
        return torch.tensor([456.0], dtype=torch.float32)


class _WindowedMujocoEnv(_FakeMujocoEnv):
    robot_window_length = 4

    def reset(self):
        super().reset()
        return _make_windowed_flat_obs(step=0, bias=self.bias, window_length=self.robot_window_length)

    def step(self, action):
        super().step(action)
        return _make_windowed_flat_obs(
            step=self.step_count,
            bias=self.bias,
            window_length=self.robot_window_length,
        )


class _WindowedStructuredObsMujocoEnv(_StructuredObsMujocoEnv):
    robot_window_length = 4

    def _build_obs_parts(self) -> dict[str, torch.Tensor]:
        return eval_sim2sim.parse_sim2sim_obs(
            _make_windowed_flat_obs(step=self.step_count, bias=self.bias, window_length=self.robot_window_length),
            action_dim=self.action_dim,
            observation_window_lengths=build_robot_policy_window_lengths(self.robot_window_length),
        )


class _ObservationBuilderMujocoEnv(_FakeMujocoEnv):
    def __init__(self, *, observation_builder=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_builder = observation_builder


class _FakeVideoRecorder:
    instances: list["_FakeVideoRecorder"] = []

    def __init__(self, *, mj_model, mj_data, env=None, output_path, fps, width, height):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.env = env
        self.frames = 0
        self.closed = False
        type(self).instances.append(self)

    def capture_frame(self):
        self.frames += 1

    def close(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(b"fake-video")
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeMujocoEnv.instances.clear()
    _SupportsUnstableInitMujocoEnv.instances.clear()
    _FakeVideoRecorder.instances.clear()


def test_sim2sim_runner_uses_checkpoint_defaults_until_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    override_motion_file = tmp_path / "120_01_stageii.npz"
    override_motion_file.write_bytes(b"")

    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            output_root=str(tmp_path / "runs-default"),
        )
    )
    assert runner.motion_files[0].endswith("115_06_stageii.npz")
    assert runner.action_mode == "offset"
    assert runner.root_name == "torso_link"
    assert runner.anchor_body_name == "torso_link"

    override_runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            motion_files=[str(override_motion_file)],
            num_steps=1,
            action_mode="residual",
            root_name="pelvis",
            anchor_body_name="pelvis",
            output_root=str(tmp_path / "runs-override"),
        )
    )
    summary = override_runner.evaluate()

    assert override_runner.motion_files[0].endswith("120_01_stageii.npz")
    env = _FakeMujocoEnv.instances[-1]
    assert env.motion_file.endswith("120_01_stageii.npz")
    assert env.action_mode == "residual"
    assert env.root_link_name == "pelvis"
    assert env.anchor_body_name == "pelvis"
    assert summary["motions"][0]["steps"] == 1


def test_sim2sim_runner_forwards_allow_unstable_init_when_supported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_SupportsUnstableInitMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            allow_unstable_init=True,
            output_root=str(tmp_path / "runs-unstable-init"),
        )
    )

    summary = runner.evaluate()

    env = _SupportsUnstableInitMujocoEnv.instances[-1]
    assert env.allow_unstable_init is True
    assert summary["allow_unstable_init"] is True

    summary_path = Path(summary["run_dir"]) / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["allow_unstable_init"] is True

    debug_payload = json.loads(Path(summary["motions"][0]["debug_json_path"]).read_text(encoding="utf-8"))
    assert debug_payload["allow_unstable_init"] is True


def test_sim2sim_runner_passes_policy_only_observation_spec_to_mujoco_builder(tmp_path, monkeypatch):
    class _FakeIsaacLabMujocoObservation:
        def __init__(self, spec=None):
            self.spec = spec

    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(
            MujocoEnv=_ObservationBuilderMujocoEnv,
            IsaacLabMujocoObservation=_FakeIsaacLabMujocoObservation,
        ),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=0,
            output_root=str(tmp_path / "runs-builder"),
        )
    )

    runner.evaluate()

    env = _ObservationBuilderMujocoEnv.instances[-1]
    assert env.observation_builder is not None
    assert tuple(group.name for group in env.observation_builder.spec.enabled_groups()) == ("motion", "robot")


def test_sim2sim_runner_restores_windowed_robot_obs_from_checkpoint_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_WindowedMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path, robot_window_length=4)

    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            output_root=str(tmp_path / "runs-windowed"),
        )
    )
    summary = runner.evaluate()

    assert runner.observation_window_lengths == build_robot_policy_window_lengths(4)
    assert runner.obs_dims == {"motion": 5, "robot": 48, "policy": 53}
    assert summary["motions"][0]["steps"] == 1


def test_offscreen_video_recorder_uses_env_tracking_camera(monkeypatch, tmp_path):
    update_calls: list[tuple[int, int]] = []

    class _FakeCamera:
        def __init__(self):
            self.type = -1
            self.fixedcamid = 99
            self.trackbodyid = 88
            self.lookat = np.zeros(3, dtype=np.float64)
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0

    class _FakeRenderer:
        def __init__(self, mj_model, *, height, width):
            self.height = height
            self.width = width
            self.last_camera = None
            self.closed = False

        def update_scene(self, mj_data, camera=None):
            self.last_camera = camera

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    class _FakeWriter:
        def __init__(self):
            self.frames = 0
            self.closed = False

        def append_data(self, frame):
            self.frames += 1

        def close(self):
            self.closed = True

    fake_writer = _FakeWriter()
    fake_mujoco = types.SimpleNamespace(
        Renderer=_FakeRenderer,
        MjvCamera=_FakeCamera,
        mjv_defaultFreeCamera=lambda mj_model, camera: setattr(camera, "distance", 1.5),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setattr(eval_sim2sim.imageio, "get_writer", lambda output_path, fps: fake_writer)

    env = types.SimpleNamespace()

    def _update_tracking_camera(camera, *, frame_width, frame_height, mujoco_module):
        update_calls.append((frame_width, frame_height))
        camera.lookat[:] = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        camera.distance = 4.0

    env._update_tracking_camera = _update_tracking_camera

    recorder = eval_sim2sim.OffscreenMujocoVideoRecorder(
        mj_model=object(),
        mj_data=object(),
        env=env,
        output_path=tmp_path / "tracking.mp4",
        fps=30,
        width=640,
        height=360,
    )

    try:
        recorder.capture_frame()
    finally:
        recorder.close()

    assert update_calls == [(640, 360)]
    assert recorder._renderer.last_camera is recorder._camera
    assert np.allclose(recorder._camera.lookat, np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    assert recorder._camera.distance == pytest.approx(4.0)
    assert fake_writer.frames == 1
    assert fake_writer.closed is True


def test_sim2sim_runner_evaluate_writes_summary_debug_and_video(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv),
    )
    monkeypatch.setattr(eval_sim2sim, "OffscreenMujocoVideoRecorder", _FakeVideoRecorder)
    checkpoint_path = _write_sim2sim_checkpoint(
        tmp_path,
        motion_files=[
            "env/assests/115_06_stageii.npz",
            "env/assests/120_01_stageii.npz",
        ],
    )

    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=2,
            save_video=True,
            output_root=str(tmp_path / "runs"),
        )
    )
    summary = runner.evaluate()

    assert summary["aggregate_steps"] == 4
    assert len(summary["motions"]) == 2
    assert [env.render for env in _FakeMujocoEnv.instances] == [False, False]
    assert len(_FakeVideoRecorder.instances) == 2
    assert all(recorder.frames == 3 for recorder in _FakeVideoRecorder.instances)

    expected_aggregate = {
        "gravity_mae": (_metric_values(0.0)["gravity_mae"] + _metric_values(0.25)["gravity_mae"]) / 2,
        "joint_pos_mae": (_metric_values(0.0)["joint_pos_mae"] + _metric_values(0.25)["joint_pos_mae"]) / 2,
    }
    assert summary["aggregate_metrics"] == pytest.approx(expected_aggregate)

    summary_path = Path(summary["run_dir"]) / "summary.json"
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["aggregate_metrics"] == pytest.approx(expected_aggregate)

    for motion in summary["motions"]:
        assert Path(motion["debug_json_path"]).exists()
        assert Path(motion["debug_npz_path"]).exists()
        assert Path(motion["video_path"]).exists()

        debug_payload = json.loads(Path(motion["debug_json_path"]).read_text(encoding="utf-8"))
        assert {"action", "sim_ctrl", "sim_motion_time", "sim_qpos", "sim_qvel", "sim_target_pos"}.issubset(
            debug_payload["logged_keys"]
        )
        assert debug_payload["steps_executed"] == 2
        assert debug_payload["metrics"] == pytest.approx(motion["metrics"])


def test_sim2sim_runner_rejects_unexpected_obs_dim(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_BadObsMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            output_root=str(tmp_path / "runs"),
        )
    )

    with pytest.raises(ValueError, match="Expected sim2sim observation dim"):
        runner.evaluate()


def test_sim2sim_runner_rejects_allow_unstable_init_for_legacy_bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=1,
            allow_unstable_init=True,
            output_root=str(tmp_path / "runs-legacy-bridge"),
        )
    )

    with pytest.raises(ValueError, match="allow_unstable_init"):
        runner.evaluate()


def test_sim2sim_runner_prefers_structured_obs_dict_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_StructuredObsMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=2,
            output_root=str(tmp_path / "runs"),
        )
    )

    summary = runner.evaluate()

    env = _StructuredObsMujocoEnv.instances[-1]
    assert env.get_obs_dict_calls == [False, False, False]
    assert summary["aggregate_steps"] == 2


def test_sim2sim_runner_prefers_structured_windowed_obs_dict_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        eval_sim2sim,
        "get_mujoco_symbols",
        lambda: types.SimpleNamespace(MujocoEnv=_WindowedStructuredObsMujocoEnv),
    )
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path, robot_window_length=4)
    runner = eval_sim2sim.Sim2SimEvalRunner(
        Sim2SimEvalConfig(
            checkpoint_path=str(checkpoint_path),
            num_steps=2,
            output_root=str(tmp_path / "runs-windowed-structured"),
        )
    )

    summary = runner.evaluate()

    env = _WindowedStructuredObsMujocoEnv.instances[-1]
    assert env.get_obs_dict_calls == [False, False, False]
    assert summary["aggregate_steps"] == 2
