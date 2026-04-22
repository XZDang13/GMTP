import copy
import importlib
import sys
import types
from dataclasses import dataclass, field, is_dataclass


def _configclass(cls=None, **kwargs):
    def wrap(c):
        annotations = dict(getattr(c, "__annotations__", {}))
        for name, value in list(vars(c).items()):
            if name.startswith("__"):
                continue
            if callable(value) or isinstance(value, (staticmethod, classmethod, property)):
                continue
            annotations.setdefault(name, type(value))
            if isinstance(value, list):
                setattr(c, name, field(default_factory=lambda value=value: list(value)))
            elif isinstance(value, dict):
                setattr(c, name, field(default_factory=lambda value=value: dict(value)))
            elif is_dataclass(value) and not isinstance(value, type):
                setattr(c, name, field(default_factory=lambda value=value: copy.deepcopy(value)))
        c.__annotations__ = annotations
        return dataclass(c, **kwargs)

    return wrap if cls is None else wrap(cls)


def _install_env_cfg_stubs(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = _configclass
    isaaclab.utils = isaaclab_utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)

    @dataclass
    class ActionCfg:
        mode: str = "default"

    @dataclass
    class RecoveryCfg:
        enabled: bool = False

    @_configclass
    class FakeG1MotionTrackingEnvCfg:
        action = ActionCfg()
        recovery = RecoveryCfg()

    @_configclass
    class FakeG1TrainingEventCfg:
        pass

    class FakeSamplingStrategy:
        FailureWeighted = "FailureWeighted"

    class FakeSegmentSource:
        Anchor = "Anchor"

    @dataclass(frozen=True)
    class FakeEnvCfgSymbols:
        G1MotionTrackingEnvCfg: type
        G1TrainingEventCfg: type
        SamplingStrategy: type
        SegmentSource: type

    compat_mod = types.ModuleType("gmtp.integrations.ref2act.compat")
    compat_mod.load_env_cfg_symbols = lambda: FakeEnvCfgSymbols(
        G1MotionTrackingEnvCfg=FakeG1MotionTrackingEnvCfg,
        G1TrainingEventCfg=FakeG1TrainingEventCfg,
        SamplingStrategy=FakeSamplingStrategy,
        SegmentSource=FakeSegmentSource,
    )
    compat_mod.load_mujoco_symbols = lambda: None

    motion_mod = types.ModuleType("gmtp.integrations.ref2act.motion")
    motion_mod.DEFAULT_EXPERIMENT_MOTION_FILES = ["walk_anchor.npz"]
    motion_mod.infer_motion_files_from_checkpoint = lambda *args, **kwargs: []
    motion_mod.motion_label = lambda *args, **kwargs: "walk"
    motion_mod.motion_names = lambda *args, **kwargs: ["walk"]
    motion_mod.normalize_motion_files = lambda motion_files: motion_files
    motion_mod.resolve_motion_file = lambda motion_file: motion_file
    motion_mod.resolve_motion_files = lambda motion_files: list(motion_files)

    observation_mod = types.ModuleType("gmtp.integrations.ref2act.observation_history")
    observation_mod.build_gmtp_observation_spec = lambda add_noise=True: {"add_noise": add_noise}

    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.compat", compat_mod)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.motion", motion_mod)
    monkeypatch.setitem(sys.modules, "gmtp.integrations.ref2act.observation_history", observation_mod)


def test_training_env_uses_anchor_failure_weighted_sampling(monkeypatch):
    _install_env_cfg_stubs(monkeypatch)
    sys.modules.pop("gmtp.integrations.ref2act.env_cfg", None)
    sys.modules.pop("gmtp.integrations.ref2act", None)

    env_cfg = importlib.import_module("gmtp.integrations.ref2act.env_cfg")

    training_cfg = env_cfg.G1MultiMotionTrainingEnv()
    eval_cfg = env_cfg.G1MultiMotionEnv()

    assert training_cfg.sampling_strategy == env_cfg.SamplingStrategy.FailureWeighted
    assert training_cfg.segment_source == env_cfg.SegmentSource.Anchor
    assert training_cfg.recovery.enabled is False
    assert eval_cfg.recovery.enabled is False
