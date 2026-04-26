import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from gmtp.integrations.ref2act.observation_history import build_robot_policy_window_lengths
from gmtp.models import Critic, FiLMResActor
from gmtp.runtime.checkpoints import build_training_checkpoint, save_checkpoint_v2
from gmtp.runtime.observations import parse_sim2sim_obs


def _load_script_module(module_name: str):
    script_path = Path(__file__).resolve().parents[2] / "test_scripts" / "deploy_mujoco.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    latest_parts = parse_sim2sim_obs(_make_flat_obs(step=step, bias=bias), action_dim=2)
    history_parts = [
        parse_sim2sim_obs(_make_flat_obs(step=history_step, bias=bias), action_dim=2)
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
        self.mj_viewer = types.SimpleNamespace(is_alive=True) if render else None
        self.mj_data = types.SimpleNamespace(
            qpos=torch.zeros(self.action_dim + 7, dtype=torch.float32),
            qvel=torch.zeros(self.action_dim + 6, dtype=torch.float32),
        )
        self.step_count = 0
        self.closed = False
        self.actions: list[torch.Tensor] = []
        type(self).instances.append(self)

    def reset(self):
        self.step_count = 0
        self.actions.clear()
        self.mj_data.qpos.zero_()
        self.mj_data.qvel.zero_()
        self.mj_data.qpos[3] = 1.0
        self.mj_data.qvel[3:6] = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        return _make_flat_obs(step=0, bias=self.bias)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32)
        self.actions.append(action.clone())
        self.step_count += 1
        self.mj_data.qpos.zero_()
        self.mj_data.qvel.zero_()
        self.mj_data.qpos[3:7] = torch.tensor([0.9238795, 0.0, 0.38268343, 0.0], dtype=torch.float32)
        self.mj_data.qvel[3:6] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) + self.step_count
        return _make_flat_obs(step=self.step_count, bias=self.bias)

    def close(self):
        self.closed = True


class _StructuredObsMujocoEnv(_FakeMujocoEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.get_obs_dict_calls: list[bool] = []

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        parts = parse_sim2sim_obs(_make_flat_obs(step=self.step_count, bias=self.bias), action_dim=self.action_dim)
        return {
            "motion": parts["motion"],
            "robot": parts["robot"],
        }

    def get_obs_dict(self, advance_time: bool = False):
        self.get_obs_dict_calls.append(advance_time)
        return self._build_obs_dict()

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

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        parts = parse_sim2sim_obs(
            _make_windowed_flat_obs(step=self.step_count, bias=self.bias, window_length=self.robot_window_length),
            action_dim=self.action_dim,
            observation_window_lengths=build_robot_policy_window_lengths(self.robot_window_length),
        )
        return {
            "motion": parts["motion"],
            "robot": parts["robot"],
        }


class _ExplodingEvalModule:
    def __getattr__(self, name):
        raise AssertionError(f"deploy_mujoco.py should not depend on gmtp.runtime.eval_sim2sim ({name}).")


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeMujocoEnv.instances.clear()


def test_script_uses_checkpoint_defaults_and_render_on_by_default(tmp_path, monkeypatch, capsys):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    monkeypatch.setitem(sys.modules, "gmtp.runtime.eval_sim2sim", _ExplodingEvalModule())
    module = _load_script_module("deploy_mujoco_defaults")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv))

    result = module.main(["--checkpoint", str(checkpoint_path), "--num-steps", "2"])

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.motion_file.endswith("115_06_stageii.npz")
    assert env.action_mode == "offset"
    assert env.root_link_name == "torso_link"
    assert env.anchor_body_name == "torso_link"
    assert env.render is True
    assert env.step_count == 2
    assert env.closed is True
    stdout = capsys.readouterr().out
    assert "Starting MuJoCo deploy smoke test" in stdout
    assert "Finished MuJoCo deploy smoke test" in stdout


def test_script_applies_motion_and_name_overrides_in_headless_mode(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    override_motion_file = tmp_path / "120_01_stageii.npz"
    override_motion_file.write_bytes(b"")
    module = _load_script_module("deploy_mujoco_overrides")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv))

    result = module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--motion-file",
            str(override_motion_file),
            "--action-mode",
            "residual",
            "--root-name",
            "pelvis",
            "--anchor-body-name",
            "pelvis",
            "--num-steps",
            "1",
            "--headless",
        ]
    )

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.motion_file.endswith("120_01_stageii.npz")
    assert env.action_mode == "residual"
    assert env.root_link_name == "pelvis"
    assert env.anchor_body_name == "pelvis"
    assert env.render is False
    assert env.mj_viewer is None


def test_script_prefers_structured_obs_dict_when_available(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    module = _load_script_module("deploy_mujoco_structured")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_StructuredObsMujocoEnv))

    result = module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--num-steps",
            "2",
            "--headless",
        ]
    )

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.step_count == 2
    assert env.get_obs_dict_calls == [False, False, False]


def test_script_restores_windowed_robot_obs_from_checkpoint_metadata(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path, robot_window_length=4)
    module = _load_script_module("deploy_mujoco_windowed")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_WindowedMujocoEnv))

    result = module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--num-steps",
            "2",
            "--headless",
        ]
    )

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.step_count == 2


def test_script_prefers_structured_windowed_obs_dict_when_available(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path, robot_window_length=4)
    module = _load_script_module("deploy_mujoco_windowed_structured")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_WindowedStructuredObsMujocoEnv))

    result = module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--num-steps",
            "2",
            "--headless",
        ]
    )

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.step_count == 2
    assert env.get_obs_dict_calls == [False, False, False]


def test_script_stops_after_requested_number_of_steps(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    module = _load_script_module("deploy_mujoco_num_steps")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_FakeMujocoEnv))

    result = module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--num-steps",
            "3",
            "--headless",
        ]
    )

    assert result == 0
    env = _FakeMujocoEnv.instances[-1]
    assert env.step_count == 3
    assert len(env.actions) == 3


def test_extract_obs_parts_uses_mujoco_free_joint_gravity_and_base_ang_vel(tmp_path, monkeypatch):
    checkpoint_path = _write_sim2sim_checkpoint(tmp_path)
    module = _load_script_module("deploy_mujoco_obs_override")
    monkeypatch.setattr(module, "get_mujoco_symbols", lambda: types.SimpleNamespace(MujocoEnv=_StructuredObsMujocoEnv))

    module.main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--num-steps",
            "0",
            "--headless",
        ]
    )
    env = _FakeMujocoEnv.instances[-1]
    initial_obs = env.get_obs_dict(advance_time=False)
    obs_parts = module._extract_obs_parts(env, initial_obs, action_dim=2, observation_window_lengths={})

    assert obs_parts["robot_projected_gravity"] == pytest.approx(torch.tensor([0.0, -0.0, -1.0]))
    assert obs_parts["anchor_ang_vel"] == pytest.approx(torch.tensor([0.1, 0.2, 0.3]))
