from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REF2ACT_SRC = (PROJECT_ROOT.parent / "Ref2Act" / "src").resolve()


@dataclass(frozen=True)
class EnvCfgSymbols:
    ActionMod: type
    G1MotionTrackingEnvCfg: type
    G1TrainingEventCfg: type
    SamplingStrategy: type
    SegmentSource: type


@dataclass(frozen=True)
class MujocoSymbols:
    MujocoEnv: type
    IsaacLabMujocoObservation: type | None
    quat_rotate_inverse: object


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_ref2act_on_path() -> None:
    if _module_exists("ref2act") or _module_exists("Ref2Act"):
        return

    candidate_paths: list[Path] = []
    env_override = os.getenv("REF2ACT_SRC")
    if env_override:
        candidate_paths.append(Path(env_override).expanduser().resolve())
    candidate_paths.append(DEFAULT_REF2ACT_SRC)

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
            importlib.invalidate_caches()
        if _module_exists("ref2act") or _module_exists("Ref2Act"):
            return

    raise ModuleNotFoundError(
        "Could not import Ref2Act/ref2act. Install Ref2Act in the active environment, "
        "or point REF2ACT_SRC to the ref2act src directory."
    )


def _import_module(*module_names: str):
    _ensure_ref2act_on_path()
    importlib.invalidate_caches()

    last_error: ModuleNotFoundError | None = None
    for module_name in module_names:
        root_name = module_name.split(".", 1)[0]
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or ""
            if missing_name != root_name and not missing_name.startswith(f"{root_name}."):
                raise
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(f"Could not import any of: {module_names}")


def load_env_cfg_symbols() -> EnvCfgSymbols:
    action_module = _import_module("ref2act.envs.motion_tracking.types", "Ref2Act.config.env_cfg")
    g1_module = _import_module("ref2act.robots.g1", "Ref2Act.config.env_cfg")
    sampling_module = _import_module("ref2act.motion", "Ref2Act.sampler")
    return EnvCfgSymbols(
        ActionMod=action_module.ActionMod,
        G1MotionTrackingEnvCfg=g1_module.G1MotionTrackingEnvCfg,
        G1TrainingEventCfg=g1_module.G1TrainingEventCfg,
        SamplingStrategy=sampling_module.SamplingStrategy,
        SegmentSource=sampling_module.SegmentSource,
    )


def load_mujoco_symbols() -> MujocoSymbols:
    mujoco_module = _import_module("ref2act.bridges.mujoco.env", "Ref2Act.sim2sim")
    observation_module = None
    try:
        observation_module = _import_module("ref2act.bridges.mujoco.observation")
    except ModuleNotFoundError:
        observation_module = None
    return MujocoSymbols(
        MujocoEnv=mujoco_module.MujocoEnv,
        IsaacLabMujocoObservation=(
            observation_module.IsaacLabMujocoObservation if observation_module is not None else None
        ),
        quat_rotate_inverse=mujoco_module.quat_rotate_inverse,
    )
