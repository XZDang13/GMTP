from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from gmtp.integrations.ref2act.motion import resolve_motion_files

from .adapters import MotionSourceAdapter, build_motion_source_adapter
from .config import MotionMAEDataConfig, MotionMAEFeatureConfig
from .schema import CanonicalMotionSequence, FeatureSliceSpec, MotionFeatureSchema, MotionSegment


@dataclass(frozen=True)
class MotionFeatureSequence:
    motion_file: str
    motion_name: str
    segments: tuple[MotionSegment, ...]
    full_features: torch.Tensor
    reference_features: torch.Tensor
    target_features: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.full_features.shape[0])


@dataclass(frozen=True)
class MotionFeatureBundle:
    sequences: tuple[MotionFeatureSequence, ...]
    schema: MotionFeatureSchema


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(q) / torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_inv(q), v)


def _build_named_slices(
    names: tuple[str, ...],
    base_slice_map: dict[str, FeatureSliceSpec],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[FeatureSliceSpec, ...]:
    slices = []
    offset = 0
    for name in names:
        try:
            base_slice = base_slice_map[name]
        except KeyError as exc:
            raise KeyError(f"Unknown feature block '{name}'.") from exc
        next_offset = offset + base_slice.dim
        slices.append(
            FeatureSliceSpec(
                name=name,
                start=offset,
                end=next_offset,
                weight=float((weights or {}).get(name, 1.0)),
            )
        )
        offset = next_offset
    return tuple(slices)


def _select_feature_blocks(
    full_feature: torch.Tensor,
    names: tuple[str, ...],
    base_slice_map: dict[str, FeatureSliceSpec],
) -> torch.Tensor:
    parts = [full_feature[:, base_slice_map[name].start : base_slice_map[name].end] for name in names]
    return torch.cat(parts, dim=-1)


def _resolve_anchor_and_end_effector_indices(
    sequence: CanonicalMotionSequence,
    *,
    feature_config: MotionMAEFeatureConfig,
) -> tuple[int, list[int]]:
    try:
        anchor_body_index = sequence.body_names.index(feature_config.anchor_body_name)
    except ValueError as exc:
        raise ValueError(
            f"Anchor body '{feature_config.anchor_body_name}' was not found in {sequence.motion_file}."
        ) from exc

    try:
        end_effector_body_indices = [sequence.body_names.index(name) for name in feature_config.end_effector_body_names]
    except ValueError as exc:
        raise ValueError(
            "One or more end-effector bodies "
            f"{feature_config.end_effector_body_names} were not found in {sequence.motion_file}."
        ) from exc

    return anchor_body_index, end_effector_body_indices


def build_motion_feature_bundle(
    motion_files: list[str] | tuple[str, ...] | None,
    *,
    data_config: MotionMAEDataConfig,
    feature_config: MotionMAEFeatureConfig,
    slice_weights: dict[str, float] | None = None,
) -> MotionFeatureBundle:
    resolved_motion_files = resolve_motion_files(motion_files or data_config.motion_files)
    adapter: MotionSourceAdapter = build_motion_source_adapter(data_config.adapter_name)

    sequences: list[MotionFeatureSequence] = []
    joint_names_ref: tuple[str, ...] | None = None
    body_names_ref: tuple[str, ...] | None = None
    schema: MotionFeatureSchema | None = None

    for motion_file in resolved_motion_files:
        sequence = adapter.load_sequence(motion_file)
        if joint_names_ref is None:
            joint_names_ref = sequence.joint_names
            body_names_ref = sequence.body_names
        else:
            if sequence.joint_names != joint_names_ref:
                raise ValueError("All motion assets must share the same joint_names ordering.")
            if sequence.body_names != body_names_ref:
                raise ValueError("All motion assets must share the same body_names ordering.")

        anchor_body_index, end_effector_body_indices = _resolve_anchor_and_end_effector_indices(
            sequence,
            feature_config=feature_config,
        )

        anchor_pos_w = sequence.body_pos_w[:, anchor_body_index]
        anchor_quat_w = sequence.body_quat_w[:, anchor_body_index]
        gravity = torch.as_tensor(feature_config.gravity_vector, dtype=torch.float32).reshape(1, 3).expand(
            sequence.length, 3
        )
        root_features = quat_apply_inverse(anchor_quat_w, gravity)
        joint_pos_features = sequence.joint_pos
        joint_vel_features = sequence.joint_vel
        joint_features = torch.cat((joint_pos_features, joint_vel_features), dim=-1)

        end_effector_pos_w = sequence.body_pos_w[:, end_effector_body_indices]
        rel_end_effector_pos = quat_apply_inverse(
            anchor_quat_w[:, None, :].expand(sequence.length, len(end_effector_body_indices), 4),
            end_effector_pos_w - anchor_pos_w[:, None, :],
        ).reshape(sequence.length, -1)

        root_end = root_features.shape[-1]
        joint_pos_end = root_end + joint_pos_features.shape[-1]
        joint_vel_end = joint_pos_end + joint_vel_features.shape[-1]
        end_effector_end = joint_vel_end + rel_end_effector_pos.shape[-1]
        base_slices = (
            FeatureSliceSpec(name="root", start=0, end=root_end),
            FeatureSliceSpec(name="joint_pos", start=root_end, end=joint_pos_end),
            FeatureSliceSpec(name="joint_vel", start=joint_pos_end, end=joint_vel_end),
            FeatureSliceSpec(name="end_effector", start=joint_vel_end, end=end_effector_end),
        )
        base_slice_map = {item.name: item for item in base_slices}
        base_slice_map["joint"] = FeatureSliceSpec(name="joint", start=root_end, end=joint_vel_end)

        full_features = torch.cat((root_features, joint_features, rel_end_effector_pos), dim=-1)
        reference_features = _select_feature_blocks(full_features, feature_config.reference_feature_names, base_slice_map)
        target_features = _select_feature_blocks(full_features, feature_config.target_feature_names, base_slice_map)

        if schema is None:
            target_slices = _build_named_slices(
                feature_config.target_feature_names,
                base_slice_map,
                weights=slice_weights,
            )
            policy_prefix_dim = sum(
                next(item.dim for item in target_slices if item.name == feature_name)
                for feature_name in feature_config.policy_feature_names
            )
            schema = MotionFeatureSchema(
                d_ref=int(reference_features.shape[-1]),
                d_target=int(target_features.shape[-1]),
                full_feature_dim=int(full_features.shape[-1]),
                base_slices=base_slices,
                reference_slices=_build_named_slices(feature_config.reference_feature_names, base_slice_map),
                target_slices=target_slices,
                policy_motion_slice=FeatureSliceSpec(name="policy_motion", start=0, end=policy_prefix_dim),
                anchor_body_name=feature_config.anchor_body_name,
                end_effector_body_names=feature_config.end_effector_body_names,
                reference_feature_names=feature_config.reference_feature_names,
                target_feature_names=feature_config.target_feature_names,
                policy_feature_names=feature_config.policy_feature_names,
                gravity_vector=feature_config.gravity_vector,
                joint_names=sequence.joint_names,
                body_names=sequence.body_names,
            )

        sequences.append(
            MotionFeatureSequence(
                motion_file=str(Path(sequence.motion_file).resolve()),
                motion_name=sequence.motion_name,
                segments=sequence.segments,
                full_features=full_features,
                reference_features=reference_features,
                target_features=target_features,
            )
        )

    if not sequences or schema is None:
        raise ValueError("No motion sequences were loaded for Motion MAE pretraining.")

    return MotionFeatureBundle(sequences=tuple(sequences), schema=schema)
