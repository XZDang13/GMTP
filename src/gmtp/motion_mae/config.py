from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_END_EFFECTOR_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_rubber_hand",
    "right_rubber_hand",
)
VALID_FEATURE_NAMES = {"root", "joint", "end_effector"}
VALID_SPLIT_MODES = {"auto", "by_motion", "by_window"}
VALID_RECONSTRUCTION_LOSSES = {"mse", "l1"}
VALID_ADAPTER_NAMES = {"stageii_npz"}
VALID_ACTIVATIONS = {"relu", "gelu"}


def _to_str_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__}.")
    return tuple(str(item) for item in value)


def _to_int_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value).__name__}.")
    return tuple(int(item) for item in value)


def _normalize_slice_weights(value: Any) -> dict[str, float]:
    default = {"root": 1.0, "joint": 1.0, "end_effector": 1.0}
    if value is None:
        return default
    if not isinstance(value, dict):
        raise TypeError(f"slice_weights must be a mapping, got {type(value).__name__}.")
    normalized = default.copy()
    for key, item in value.items():
        normalized[str(key)] = float(item)
    return normalized


@dataclass(frozen=True)
class MotionMAEFeatureConfig:
    anchor_body_name: str = "pelvis"
    end_effector_body_names: tuple[str, ...] = DEFAULT_END_EFFECTOR_BODY_NAMES
    reference_feature_names: tuple[str, ...] = ("root", "joint", "end_effector")
    target_feature_names: tuple[str, ...] = ("root", "joint", "end_effector")
    policy_feature_names: tuple[str, ...] = ("root", "joint")
    gravity_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def __post_init__(self) -> None:
        for name, values in (
            ("reference_feature_names", self.reference_feature_names),
            ("target_feature_names", self.target_feature_names),
            ("policy_feature_names", self.policy_feature_names),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty.")
            unknown = [item for item in values if item not in VALID_FEATURE_NAMES]
            if unknown:
                raise ValueError(f"{name} contains unsupported features: {unknown}.")
        if tuple(self.target_feature_names[: len(self.policy_feature_names)]) != tuple(self.policy_feature_names):
            raise ValueError("policy_feature_names must form a prefix of target_feature_names.")
        if len(self.gravity_vector) != 3:
            raise ValueError("gravity_vector must have exactly 3 elements.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_body_name": self.anchor_body_name,
            "end_effector_body_names": list(self.end_effector_body_names),
            "reference_feature_names": list(self.reference_feature_names),
            "target_feature_names": list(self.target_feature_names),
            "policy_feature_names": list(self.policy_feature_names),
            "gravity_vector": list(self.gravity_vector),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAEFeatureConfig":
        payload = payload or {}
        gravity_vector = payload.get("gravity_vector", (0.0, 0.0, -1.0))
        return cls(
            anchor_body_name=str(payload.get("anchor_body_name", "pelvis")),
            end_effector_body_names=_to_str_tuple(
                payload.get("end_effector_body_names", DEFAULT_END_EFFECTOR_BODY_NAMES),
                name="end_effector_body_names",
            ),
            reference_feature_names=_to_str_tuple(
                payload.get("reference_feature_names", ("root", "joint", "end_effector")),
                name="reference_feature_names",
            ),
            target_feature_names=_to_str_tuple(
                payload.get("target_feature_names", ("root", "joint", "end_effector")),
                name="target_feature_names",
            ),
            policy_feature_names=_to_str_tuple(
                payload.get("policy_feature_names", ("root", "joint")),
                name="policy_feature_names",
            ),
            gravity_vector=(float(gravity_vector[0]), float(gravity_vector[1]), float(gravity_vector[2])),
        )


@dataclass(frozen=True)
class MotionMAEDataConfig:
    motion_files: tuple[str, ...] | None = None
    adapter_name: str = "stageii_npz"
    past_frames: int = 8
    future_frames: int = 4
    split_mode: str = "auto"
    val_ratio: float = 0.2
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 0
    max_train_windows: int | None = None
    max_val_windows: int | None = None

    def __post_init__(self) -> None:
        if self.adapter_name not in VALID_ADAPTER_NAMES:
            raise ValueError(f"Unsupported adapter_name '{self.adapter_name}'.")
        if self.past_frames < 1 or self.future_frames < 1:
            raise ValueError("past_frames and future_frames must be positive.")
        if self.split_mode not in VALID_SPLIT_MODES:
            raise ValueError(f"Unsupported split_mode '{self.split_mode}'.")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.max_train_windows is not None and self.max_train_windows < 1:
            raise ValueError("max_train_windows must be positive when provided.")
        if self.max_val_windows is not None and self.max_val_windows < 1:
            raise ValueError("max_val_windows must be positive when provided.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_files": list(self.motion_files) if self.motion_files is not None else None,
            "adapter_name": self.adapter_name,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
            "split_mode": self.split_mode,
            "val_ratio": self.val_ratio,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "seed": self.seed,
            "max_train_windows": self.max_train_windows,
            "max_val_windows": self.max_val_windows,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAEDataConfig":
        payload = payload or {}
        motion_files = payload.get("motion_files")
        return cls(
            motion_files=_to_str_tuple(motion_files, name="motion_files") if motion_files is not None else None,
            adapter_name=str(payload.get("adapter_name", "stageii_npz")),
            past_frames=int(payload.get("past_frames", 8)),
            future_frames=int(payload.get("future_frames", 4)),
            split_mode=str(payload.get("split_mode", "auto")),
            val_ratio=float(payload.get("val_ratio", 0.2)),
            batch_size=int(payload.get("batch_size", 128)),
            num_workers=int(payload.get("num_workers", 0)),
            pin_memory=bool(payload.get("pin_memory", False)),
            seed=int(payload.get("seed", 0)),
            max_train_windows=(
                int(payload["max_train_windows"]) if payload.get("max_train_windows") is not None else None
            ),
            max_val_windows=int(payload["max_val_windows"]) if payload.get("max_val_windows") is not None else None,
        )


@dataclass(frozen=True)
class MotionMAEModelConfig:
    d_model: int = 256
    latent_dim: int = 64
    encoder_layers: int = 4
    decoder_layers: int = 2
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.0
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.d_model < 1 or self.latent_dim < 1:
            raise ValueError("d_model and latent_dim must be positive.")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder_layers and decoder_layers must be positive.")
        if self.nhead < 1:
            raise ValueError("nhead must be positive.")
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}.")
        if self.dim_feedforward < self.d_model:
            raise ValueError("dim_feedforward must be at least d_model.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{self.activation}'.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "latent_dim": self.latent_dim,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self.activation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAEModelConfig":
        payload = payload or {}
        return cls(
            d_model=int(payload.get("d_model", 256)),
            latent_dim=int(payload.get("latent_dim", 64)),
            encoder_layers=int(payload.get("encoder_layers", 4)),
            decoder_layers=int(payload.get("decoder_layers", 2)),
            nhead=int(payload.get("nhead", 8)),
            dim_feedforward=int(payload.get("dim_feedforward", 512)),
            dropout=float(payload.get("dropout", 0.0)),
            activation=str(payload.get("activation", "gelu")),
        )


@dataclass(frozen=True)
class MotionMAELossConfig:
    reconstruction_loss: str = "mse"
    slice_weights: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reconstruction_loss not in VALID_RECONSTRUCTION_LOSSES:
            raise ValueError(f"Unsupported reconstruction_loss '{self.reconstruction_loss}'.")
        if self.slice_weights is None:
            object.__setattr__(self, "slice_weights", {"root": 1.0, "joint": 1.0, "end_effector": 1.0})
        if any(value < 0.0 for value in self.slice_weights.values()):
            raise ValueError("slice_weights must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reconstruction_loss": self.reconstruction_loss,
            "slice_weights": dict(self.slice_weights),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAELossConfig":
        payload = payload or {}
        return cls(
            reconstruction_loss=str(payload.get("reconstruction_loss", "mse")),
            slice_weights=_normalize_slice_weights(payload.get("slice_weights")),
        )


@dataclass(frozen=True)
class MotionMAEOptimizerConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAEOptimizerConfig":
        payload = payload or {}
        return cls(
            lr=float(payload.get("lr", 3.0e-4)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
        )


@dataclass(frozen=True)
class MotionMAETrainingConfig:
    epochs: int = 10
    device: str = "auto"
    grad_clip_norm: float | None = 1.0
    log_interval: int = 10

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided.")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "device": self.device,
            "grad_clip_norm": self.grad_clip_norm,
            "log_interval": self.log_interval,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAETrainingConfig":
        payload = payload or {}
        grad_clip_norm = payload.get("grad_clip_norm", 1.0)
        return cls(
            epochs=int(payload.get("epochs", 10)),
            device=str(payload.get("device", "auto")),
            grad_clip_norm=float(grad_clip_norm) if grad_clip_norm is not None else None,
            log_interval=int(payload.get("log_interval", 10)),
        )


@dataclass(frozen=True)
class MotionMAEExportConfig:
    save_optimizer_state: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "save_optimizer_state": self.save_optimizer_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MotionMAEExportConfig":
        payload = payload or {}
        return cls(save_optimizer_state=bool(payload.get("save_optimizer_state", True)))


@dataclass(frozen=True)
class MotionMAEPretrainConfig:
    data: MotionMAEDataConfig = MotionMAEDataConfig()
    feature: MotionMAEFeatureConfig = MotionMAEFeatureConfig()
    model: MotionMAEModelConfig = MotionMAEModelConfig()
    loss: MotionMAELossConfig = MotionMAELossConfig()
    optimizer: MotionMAEOptimizerConfig = MotionMAEOptimizerConfig()
    training: MotionMAETrainingConfig = MotionMAETrainingConfig()
    export: MotionMAEExportConfig = MotionMAEExportConfig()
    output_root: str = "runs"
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.to_dict(),
            "feature": self.feature.to_dict(),
            "model": self.model.to_dict(),
            "loss": self.loss.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "training": self.training.to_dict(),
            "export": self.export.to_dict(),
            "output_root": self.output_root,
            "run_name": self.run_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionMAEPretrainConfig":
        return cls(
            data=MotionMAEDataConfig.from_dict(payload.get("data")),
            feature=MotionMAEFeatureConfig.from_dict(payload.get("feature")),
            model=MotionMAEModelConfig.from_dict(payload.get("model")),
            loss=MotionMAELossConfig.from_dict(payload.get("loss")),
            optimizer=MotionMAEOptimizerConfig.from_dict(payload.get("optimizer")),
            training=MotionMAETrainingConfig.from_dict(payload.get("training")),
            export=MotionMAEExportConfig.from_dict(payload.get("export")),
            output_root=str(payload.get("output_root", "runs")),
            run_name=(str(payload["run_name"]) if payload.get("run_name") is not None else None),
        )


def load_motion_mae_pretrain_config(path: str | Path) -> MotionMAEPretrainConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Motion MAE config must be a JSON object, got {type(payload).__name__}.")
    return MotionMAEPretrainConfig.from_dict(payload)


def apply_motion_mae_cli_overrides(
    config: MotionMAEPretrainConfig,
    *,
    motion_files: list[str] | None = None,
    output_root: str | None = None,
    run_name: str | None = None,
    device: str | None = None,
) -> MotionMAEPretrainConfig:
    data_config = config.data
    training_config = config.training
    if motion_files is not None:
        data_config = replace(data_config, motion_files=tuple(str(item) for item in motion_files))
    if device is not None:
        training_config = replace(training_config, device=str(device))
    return replace(
        config,
        data=data_config,
        training=training_config,
        output_root=str(output_root) if output_root is not None else config.output_root,
        run_name=run_name if run_name is not None else config.run_name,
    )
