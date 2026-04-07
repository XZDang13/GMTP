from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureSliceSpec:
    name: str
    start: int
    end: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Feature slice '{self.name}' must satisfy end > start, got {self.start}:{self.end}.")
        if self.weight < 0.0:
            raise ValueError(f"Feature slice '{self.name}' weight must be non-negative, got {self.weight}.")

    @property
    def dim(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureSliceSpec":
        return cls(
            name=str(payload["name"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class MotionSegment:
    start_frame: int
    end_frame: int
    segment_type: int | None = None

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError(f"MotionSegment.start_frame must be non-negative, got {self.start_frame}.")
        if self.end_frame <= self.start_frame:
            raise ValueError(
                "MotionSegment.end_frame must be greater than start_frame, "
                f"got {self.start_frame}:{self.end_frame}."
            )

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "segment_type": self.segment_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionSegment":
        return cls(
            start_frame=int(payload["start_frame"]),
            end_frame=int(payload["end_frame"]),
            segment_type=int(payload["segment_type"]) if payload.get("segment_type") is not None else None,
        )


@dataclass(frozen=True)
class CanonicalMotionSequence:
    motion_file: str
    motion_name: str
    fps: float
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_pos: Any
    joint_vel: Any
    body_pos_w: Any
    body_quat_w: Any
    body_lin_vel_w: Any
    body_ang_vel_w: Any
    segments: tuple[MotionSegment, ...]

    @property
    def resolved_motion_file(self) -> str:
        return str(Path(self.motion_file).expanduser().resolve())

    @property
    def length(self) -> int:
        return int(self.joint_pos.shape[0])


@dataclass(frozen=True)
class MotionFeatureSchema:
    d_ref: int
    d_target: int
    full_feature_dim: int
    base_slices: tuple[FeatureSliceSpec, ...]
    reference_slices: tuple[FeatureSliceSpec, ...]
    target_slices: tuple[FeatureSliceSpec, ...]
    policy_motion_slice: FeatureSliceSpec
    anchor_body_name: str
    end_effector_body_names: tuple[str, ...]
    reference_feature_names: tuple[str, ...]
    target_feature_names: tuple[str, ...]
    policy_feature_names: tuple[str, ...]
    gravity_vector: tuple[float, float, float] = (0.0, 0.0, -1.0)
    joint_names: tuple[str, ...] = ()
    body_names: tuple[str, ...] = ()
    reference_mean: tuple[float, ...] = ()
    reference_std: tuple[float, ...] = ()
    target_mean: tuple[float, ...] = ()
    target_std: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.d_ref < 1 or self.d_target < 1 or self.full_feature_dim < 1:
            raise ValueError("Motion feature dimensions must be positive.")
        if self.d_ref != sum(item.dim for item in self.reference_slices):
            raise ValueError("Reference slices do not sum to d_ref.")
        if self.d_target != sum(item.dim for item in self.target_slices):
            raise ValueError("Target slices do not sum to d_target.")
        if self.full_feature_dim != sum(item.dim for item in self.base_slices):
            raise ValueError("Base slices do not sum to full_feature_dim.")
        if self.policy_motion_slice.end > self.d_target:
            raise ValueError("policy_motion_slice extends beyond d_target.")
        if len(self.gravity_vector) != 3:
            raise ValueError("gravity_vector must have exactly 3 elements.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_ref": self.d_ref,
            "d_target": self.d_target,
            "full_feature_dim": self.full_feature_dim,
            "base_slices": [item.to_dict() for item in self.base_slices],
            "reference_slices": [item.to_dict() for item in self.reference_slices],
            "target_slices": [item.to_dict() for item in self.target_slices],
            "policy_motion_slice": self.policy_motion_slice.to_dict(),
            "anchor_body_name": self.anchor_body_name,
            "end_effector_body_names": list(self.end_effector_body_names),
            "gravity_vector": list(self.gravity_vector),
            "reference_feature_names": list(self.reference_feature_names),
            "target_feature_names": list(self.target_feature_names),
            "policy_feature_names": list(self.policy_feature_names),
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "reference_mean": list(self.reference_mean),
            "reference_std": list(self.reference_std),
            "target_mean": list(self.target_mean),
            "target_std": list(self.target_std),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionFeatureSchema":
        return cls(
            d_ref=int(payload["d_ref"]),
            d_target=int(payload["d_target"]),
            full_feature_dim=int(payload["full_feature_dim"]),
            base_slices=tuple(FeatureSliceSpec.from_dict(item) for item in payload["base_slices"]),
            reference_slices=tuple(FeatureSliceSpec.from_dict(item) for item in payload["reference_slices"]),
            target_slices=tuple(FeatureSliceSpec.from_dict(item) for item in payload["target_slices"]),
            policy_motion_slice=FeatureSliceSpec.from_dict(payload["policy_motion_slice"]),
            anchor_body_name=str(payload["anchor_body_name"]),
            end_effector_body_names=tuple(str(item) for item in payload.get("end_effector_body_names", ())),
            gravity_vector=tuple(float(item) for item in payload.get("gravity_vector", (0.0, 0.0, -1.0))),
            reference_feature_names=tuple(str(item) for item in payload.get("reference_feature_names", ())),
            target_feature_names=tuple(str(item) for item in payload.get("target_feature_names", ())),
            policy_feature_names=tuple(str(item) for item in payload.get("policy_feature_names", ())),
            joint_names=tuple(str(item) for item in payload.get("joint_names", ())),
            body_names=tuple(str(item) for item in payload.get("body_names", ())),
            reference_mean=tuple(float(item) for item in payload.get("reference_mean", ())),
            reference_std=tuple(float(item) for item in payload.get("reference_std", ())),
            target_mean=tuple(float(item) for item in payload.get("target_mean", ())),
            target_std=tuple(float(item) for item in payload.get("target_std", ())),
        )

    def with_normalization(
        self,
        *,
        reference_mean: list[float] | tuple[float, ...],
        reference_std: list[float] | tuple[float, ...],
        target_mean: list[float] | tuple[float, ...],
        target_std: list[float] | tuple[float, ...],
    ) -> "MotionFeatureSchema":
        if len(reference_mean) != self.d_ref or len(reference_std) != self.d_ref:
            raise ValueError("Reference normalization statistics do not match d_ref.")
        if len(target_mean) != self.d_target or len(target_std) != self.d_target:
            raise ValueError("Target normalization statistics do not match d_target.")
        return replace(
            self,
            reference_mean=tuple(float(item) for item in reference_mean),
            reference_std=tuple(float(item) for item in reference_std),
            target_mean=tuple(float(item) for item in target_mean),
            target_std=tuple(float(item) for item in target_std),
        )

    def target_slice_map(self) -> dict[str, FeatureSliceSpec]:
        return {item.name: item for item in self.target_slices}

    def reference_slice_map(self) -> dict[str, FeatureSliceSpec]:
        return {item.name: item for item in self.reference_slices}
