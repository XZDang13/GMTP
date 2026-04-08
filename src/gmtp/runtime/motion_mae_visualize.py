from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from gmtp.motion_mae import (
    MotionFeatureSchema,
    MotionMAEPretrainConfig,
    ReferenceMotionMAE,
    apply_motion_mae_cli_overrides,
    build_motion_mae_datasets,
    load_motion_mae_checkpoint,
    load_motion_mae_pretrain_config,
)
from gmtp.motion_mae.adapters import MotionSourceAdapter, build_motion_source_adapter
from gmtp.motion_mae.data import ReferenceMotionMAEDataset, build_valid_window_centers
from gmtp.motion_mae.schema import CanonicalMotionSequence
from gmtp.runtime.config import MotionMAEVisualizationConfig
from gmtp.runtime.io import build_run_paths, write_json

DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FRAME_WIDTH = 640
DEFAULT_JOINT_LIMIT = float(np.pi)
VIDEO_MACRO_BLOCK_SIZE = 16
DEFAULT_CAMERA_AZIMUTH = -140.0
DEFAULT_CAMERA_ELEVATION = -20.0
DEFAULT_CAMERA_FOVY = 45.0
CAMERA_PADDING = 1.5
MIN_CAMERA_DISTANCE = 3.0
SUPPORTED_FEATURE_NAMES = ("root", "joint")
SUPPORTED_ANCHOR_BODY_NAME = "pelvis"
SUPPORTED_JOINT_COUNT = 23
PREDICTION_RENDER_MODE = "ground_truth_root_position_and_orientation_plus_predicted_joint_trajectory"
PREDICTION_RENDER_NOTE = (
    "PRED pane uses the ground-truth future root position, orientation, and root velocities, and renders the "
    "predicted joint position/velocity trajectory."
)

_FONT_3X5 = {
    " ": ("   ", "   ", "   ", "   ", "   "),
    "-": ("   ", "   ", "###", "   ", "   "),
    ".": ("   ", "   ", "   ", "   ", " # "),
    "/": ("  #", "  #", " # ", "#  ", "#  "),
    ":": ("   ", " # ", "   ", " # ", "   "),
    "_": ("   ", "   ", "   ", "   ", "###"),
    "?": ("## ", "  #", " # ", "   ", " # "),
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", " ##", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", " # ", " # "),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    "A": (" # ", "# #", "###", "# #", "# #"),
    "B": ("## ", "# #", "## ", "# #", "## "),
    "C": (" ##", "#  ", "#  ", "#  ", " ##"),
    "D": ("## ", "# #", "# #", "# #", "## "),
    "E": ("###", "#  ", "## ", "#  ", "###"),
    "F": ("###", "#  ", "## ", "#  ", "#  "),
    "G": (" ##", "#  ", "# #", "# #", " ##"),
    "H": ("# #", "# #", "###", "# #", "# #"),
    "I": ("###", " # ", " # ", " # ", "###"),
    "J": ("  #", "  #", "  #", "# #", " # "),
    "K": ("# #", "# #", "## ", "# #", "# #"),
    "L": ("#  ", "#  ", "#  ", "#  ", "###"),
    "M": ("# #", "###", "###", "# #", "# #"),
    "N": ("# #", "###", "###", "###", "# #"),
    "O": ("###", "# #", "# #", "# #", "###"),
    "P": ("## ", "# #", "## ", "#  ", "#  "),
    "Q": ("###", "# #", "# #", "###", "  #"),
    "R": ("## ", "# #", "## ", "# #", "# #"),
    "S": (" ##", "#  ", "###", "  #", "## "),
    "T": ("###", " # ", " # ", " # ", " # "),
    "U": ("# #", "# #", "# #", "# #", "###"),
    "V": ("# #", "# #", "# #", "# #", " # "),
    "W": ("# #", "# #", "###", "###", "# #"),
    "X": ("# #", "# #", " # ", "# #", "# #"),
    "Y": ("# #", "# #", "###", " # ", " # "),
    "Z": ("###", "  #", " # ", "#  ", "###"),
}


@dataclass(frozen=True)
class MotionMAEVisualizationSample:
    motion_name: str
    motion_file: str
    sequence_index: int
    center_t: int
    future_frame_indices: tuple[int, ...]
    reference: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class MotionMAEVisualizationBatch:
    motion_name: str
    motion_file: str
    center_t: int | None
    center_t_by_frame: tuple[int, ...]
    future_frame_indices: tuple[int, ...]
    future_step_ahead: tuple[int, ...]
    reference: torch.Tensor
    reference_normalized: torch.Tensor
    target: torch.Tensor
    target_normalized: torch.Tensor
    prediction: torch.Tensor
    prediction_normalized: torch.Tensor
    root_state: dict[str, torch.Tensor]


@dataclass(frozen=True)
class MotionFrameState:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


@dataclass(frozen=True)
class MotionFramePair:
    gt: MotionFrameState
    pred: MotionFrameState


def resolve_visualization_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _as_feature_stat(values: tuple[float, ...], *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(1, 1, -1)


def _normalize_reference(reference: torch.Tensor, schema: MotionFeatureSchema, *, device: torch.device) -> torch.Tensor:
    reference = torch.as_tensor(reference, dtype=torch.float32, device=device)
    return (reference - _as_feature_stat(schema.reference_mean, device=device)) / _as_feature_stat(
        schema.reference_std,
        device=device,
    )


def _normalize_target(target: torch.Tensor, schema: MotionFeatureSchema, *, device: torch.device) -> torch.Tensor:
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    return (target - _as_feature_stat(schema.target_mean, device=device)) / _as_feature_stat(
        schema.target_std,
        device=device,
    )


def denormalize_target(target: torch.Tensor, schema: MotionFeatureSchema) -> torch.Tensor:
    target = torch.as_tensor(target, dtype=torch.float32)
    mean = torch.as_tensor(schema.target_mean, dtype=torch.float32).reshape(1, 1, -1)
    std = torch.as_tensor(schema.target_std, dtype=torch.float32).reshape(1, 1, -1)
    return target * std + mean


def normalize_quaternion(quat: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    quat = torch.as_tensor(quat, dtype=torch.float32)
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(eps)


def build_motion_mae_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[ReferenceMotionMAE, Any]:
    try:
        checkpoint = load_motion_mae_checkpoint(checkpoint_path)
    except ValueError as exc:
        raise ValueError(
            "motion-mae-visualize requires a full Motion MAE checkpoint with checkpoint_type='motion_mae'."
        ) from exc

    model_kwargs = dict(checkpoint.meta["model_kwargs"])
    model = ReferenceMotionMAE(**model_kwargs).to(device)
    model.load_state_dict(checkpoint.model["model"])
    model.eval()
    return model, checkpoint


def validate_visualizer_schema_support(schema: MotionFeatureSchema) -> None:
    if tuple(schema.reference_feature_names) != SUPPORTED_FEATURE_NAMES:
        raise ValueError(
            "motion-mae-visualize supports only policy-observation Motion MAE configs "
            "with reference_feature_names=('root', 'joint')."
        )
    if tuple(schema.target_feature_names) != SUPPORTED_FEATURE_NAMES:
        raise ValueError(
            "motion-mae-visualize supports only policy-observation Motion MAE configs "
            "with target_feature_names=('root', 'joint')."
        )
    if tuple(schema.policy_feature_names) != SUPPORTED_FEATURE_NAMES:
        raise ValueError(
            "motion-mae-visualize supports only policy-observation Motion MAE configs "
            "with policy_feature_names=('root', 'joint')."
        )
    if "end_effector" in schema.target_slice_map():
        raise ValueError("motion-mae-visualize does not support checkpoints with end_effector targets.")
    if schema.anchor_body_name != SUPPORTED_ANCHOR_BODY_NAME:
        raise ValueError(
            "motion-mae-visualize currently supports only G1 pelvis-anchored checkpoints "
            f"(got anchor_body_name='{schema.anchor_body_name}')."
        )
    if len(schema.joint_names) != SUPPORTED_JOINT_COUNT:
        raise ValueError(
            "motion-mae-visualize currently supports only G1 23DoF checkpoints "
            f"(got {len(schema.joint_names)} joints)."
        )
    root_slice = schema.target_slice_map().get("root")
    if root_slice is None:
        raise ValueError("motion-mae-visualize requires a target 'root' slice.")
    if root_slice.dim != 3:
        raise ValueError(
            "motion-mae-visualize expected the target root slice to contain a 3D projected-gravity term "
            f"(got dim={root_slice.dim})."
        )
    joint_slice = schema.target_slice_map().get("joint")
    if joint_slice is None:
        raise ValueError("motion-mae-visualize requires a target 'joint' slice.")
    expected_joint_dim = 2 * len(schema.joint_names)
    if joint_slice.dim != expected_joint_dim:
        raise ValueError(
            "motion-mae-visualize expected the target joint slice to contain joint position and velocity terms "
            f"(expected dim={expected_joint_dim}, got dim={joint_slice.dim})."
        )


def validate_schema_compatibility(dataset_schema: MotionFeatureSchema, checkpoint_schema: MotionFeatureSchema) -> None:
    comparisons = (
        ("d_ref", dataset_schema.d_ref, checkpoint_schema.d_ref),
        ("d_target", dataset_schema.d_target, checkpoint_schema.d_target),
        ("anchor_body_name", dataset_schema.anchor_body_name, checkpoint_schema.anchor_body_name),
        (
            "reference_feature_names",
            tuple(dataset_schema.reference_feature_names),
            tuple(checkpoint_schema.reference_feature_names),
        ),
        ("target_feature_names", tuple(dataset_schema.target_feature_names), tuple(checkpoint_schema.target_feature_names)),
        ("policy_feature_names", tuple(dataset_schema.policy_feature_names), tuple(checkpoint_schema.policy_feature_names)),
        ("joint_names", tuple(dataset_schema.joint_names), tuple(checkpoint_schema.joint_names)),
        ("body_names", tuple(dataset_schema.body_names), tuple(checkpoint_schema.body_names)),
    )
    for field_name, actual, expected in comparisons:
        if actual != expected:
            raise ValueError(
                f"Visualization assets/config are incompatible with the Motion MAE checkpoint schema for '{field_name}': "
                f"expected {expected!r}, got {actual!r}."
            )


def validate_checkpoint_and_config_compatibility(
    checkpoint_schema: MotionFeatureSchema,
    model: ReferenceMotionMAE,
    config: MotionMAEPretrainConfig,
) -> None:
    if model.past_frames != int(config.data.past_frames):
        raise ValueError(
            f"Config past_frames={config.data.past_frames} does not match checkpoint past_frames={model.past_frames}."
        )
    if model.future_frames != int(config.data.future_frames):
        raise ValueError(
            f"Config future_frames={config.data.future_frames} does not match checkpoint future_frames={model.future_frames}."
        )
    if model.input_dim != int(checkpoint_schema.d_ref):
        raise ValueError(f"Checkpoint model input_dim={model.input_dim} does not match schema d_ref={checkpoint_schema.d_ref}.")
    if model.target_dim != int(checkpoint_schema.d_target):
        raise ValueError(
            f"Checkpoint model target_dim={model.target_dim} does not match schema d_target={checkpoint_schema.d_target}."
        )


def select_visualization_sample(
    dataset: ReferenceMotionMAEDataset,
    *,
    split_name: str,
    motion_name: str | None,
    sample_index: int,
) -> MotionMAEVisualizationSample:
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}.")

    available_motion_names = sorted({sequence.motion_name for sequence in dataset.sequences})
    filtered: list[tuple[int, int]] = []
    for sequence_index, center_t in dataset.window_indices:
        sequence = dataset.sequences[sequence_index]
        if motion_name is not None and sequence.motion_name != motion_name:
            continue
        filtered.append((sequence_index, center_t))

    if motion_name is not None and not filtered:
        raise ValueError(
            f"Motion '{motion_name}' was not found in the {split_name} split. "
            f"Available motions: {available_motion_names}."
        )
    if not filtered:
        raise ValueError(f"The {split_name} split does not contain any legal Motion MAE windows.")
    if sample_index >= len(filtered):
        raise IndexError(
            f"sample_index={sample_index} is out of range for the {split_name} split "
            f"(available windows: {len(filtered)})."
        )

    sequence_index, center_t = filtered[sample_index]
    sequence = dataset.sequences[sequence_index]
    reference_start = center_t - dataset.past_frames + 1
    reference_end = center_t + 1
    future_start = center_t + 1
    future_end = center_t + dataset.future_frames + 1
    return MotionMAEVisualizationSample(
        motion_name=sequence.motion_name,
        motion_file=sequence.motion_file,
        sequence_index=sequence_index,
        center_t=center_t,
        future_frame_indices=tuple(range(future_start, future_end)),
        reference=sequence.reference_features[reference_start:reference_end].clone(),
        target=sequence.target_features[future_start:future_end].clone(),
    )


def select_visualization_sequence(
    dataset: ReferenceMotionMAEDataset,
    *,
    motion_name: str | None,
) -> tuple[int, Any]:
    available_motion_names = sorted({sequence.motion_name for sequence in dataset.sequences})
    matching_indices = [
        sequence_index
        for sequence_index, sequence in enumerate(dataset.sequences)
        if motion_name is None or sequence.motion_name == motion_name
    ]
    if motion_name is not None and not matching_indices:
        raise ValueError(f"Motion '{motion_name}' was not found. Available motions: {available_motion_names}.")
    if not matching_indices:
        raise ValueError("No motion sequences are available for Motion MAE visualization.")
    if motion_name is None and len(matching_indices) > 1:
        raise ValueError(
            "whole-motion visualization requires motion_name when multiple motions are available. "
            f"Available motions: {available_motion_names}."
        )
    if len(matching_indices) > 1:
        raise ValueError(
            f"whole-motion visualization requires a unique motion asset for '{motion_name}', "
            f"but found {len(matching_indices)} matches."
        )
    sequence_index = matching_indices[0]
    return sequence_index, dataset.sequences[sequence_index]


def select_future_frame_comparison(
    *,
    future_frame_indices: tuple[int, ...],
    future_frame_index: int | None,
    target: torch.Tensor,
    target_normalized: torch.Tensor,
    prediction: torch.Tensor,
    prediction_normalized: torch.Tensor,
    root_state: dict[str, torch.Tensor],
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    target = torch.as_tensor(target, dtype=torch.float32)
    target_normalized = torch.as_tensor(target_normalized, dtype=torch.float32)
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    prediction_normalized = torch.as_tensor(prediction_normalized, dtype=torch.float32)

    if future_frame_index is None:
        return future_frame_indices, target, target_normalized, prediction, prediction_normalized, root_state

    if future_frame_index < 0:
        raise ValueError(f"future_frame_index must be non-negative, got {future_frame_index}.")
    if future_frame_index >= len(future_frame_indices):
        raise IndexError(
            f"future_frame_index={future_frame_index} is out of range for this sample "
            f"(available zero-based indices: 0..{len(future_frame_indices) - 1})."
        )

    frame_slice = slice(future_frame_index, future_frame_index + 1)
    selected_root_state = {
        key: value[frame_slice].clone()
        if isinstance(value, torch.Tensor) and value.shape[0] == len(future_frame_indices)
        else value
        for key, value in root_state.items()
    }
    return (
        (future_frame_indices[future_frame_index],),
        target[frame_slice].clone(),
        target_normalized[frame_slice].clone(),
        prediction[frame_slice].clone(),
        prediction_normalized[frame_slice].clone(),
        selected_root_state,
    )


def build_sample_visualization_batch(
    *,
    sample: MotionMAEVisualizationSample,
    sequence: CanonicalMotionSequence,
    schema: MotionFeatureSchema,
    model: ReferenceMotionMAE,
    device: torch.device,
    future_frame_index: int | None,
) -> MotionMAEVisualizationBatch:
    reference = torch.as_tensor(sample.reference, dtype=torch.float32)
    target = torch.as_tensor(sample.target, dtype=torch.float32)
    reference_normalized = _normalize_reference(reference.unsqueeze(0), schema, device=device).cpu()
    target_normalized = _normalize_target(target.unsqueeze(0), schema, device=torch.device("cpu")).cpu()
    with torch.no_grad():
        outputs = model(reference_normalized.to(device))
        prediction_normalized = outputs["prediction"].detach().cpu()
    prediction = denormalize_target(prediction_normalized, schema).squeeze(0)
    prediction_normalized = prediction_normalized.squeeze(0)

    root_state = _extract_future_root_and_joint_states(
        sequence,
        schema=schema,
        future_frame_indices=sample.future_frame_indices,
    )
    (
        selected_future_frame_indices,
        target,
        target_normalized,
        prediction,
        prediction_normalized,
        root_state,
    ) = select_future_frame_comparison(
        future_frame_indices=sample.future_frame_indices,
        future_frame_index=future_frame_index,
        target=target,
        target_normalized=target_normalized.squeeze(0),
        prediction=prediction,
        prediction_normalized=prediction_normalized,
        root_state=root_state,
    )
    future_step_lookup = {frame: index + 1 for index, frame in enumerate(sample.future_frame_indices)}
    selected_future_step_ahead = tuple(future_step_lookup[frame] for frame in selected_future_frame_indices)
    return MotionMAEVisualizationBatch(
        motion_name=sample.motion_name,
        motion_file=sample.motion_file,
        center_t=sample.center_t,
        center_t_by_frame=tuple(sample.center_t for _ in selected_future_frame_indices),
        future_frame_indices=selected_future_frame_indices,
        future_step_ahead=selected_future_step_ahead,
        reference=reference,
        reference_normalized=reference_normalized.squeeze(0),
        target=target,
        target_normalized=target_normalized,
        prediction=prediction,
        prediction_normalized=prediction_normalized,
        root_state=root_state,
    )


def build_whole_motion_visualization_batch(
    *,
    dataset: ReferenceMotionMAEDataset,
    sequence: Any,
    canonical_sequence: CanonicalMotionSequence,
    schema: MotionFeatureSchema,
    model: ReferenceMotionMAE,
    device: torch.device,
) -> MotionMAEVisualizationBatch:
    center_ts = build_valid_window_centers(
        sequence.segments,
        past_frames=dataset.past_frames,
        future_frames=dataset.future_frames,
    )
    if not center_ts:
        raise ValueError(f"Motion '{sequence.motion_name}' does not contain any legal Motion MAE windows.")

    selected_frames: dict[int, dict[str, Any]] = {}
    cpu_device = torch.device("cpu")
    for center_t in center_ts:
        reference_start = center_t - dataset.past_frames + 1
        reference_end = center_t + 1
        future_start = center_t + 1
        future_end = center_t + dataset.future_frames + 1
        future_frame_indices = tuple(range(future_start, future_end))

        reference = torch.as_tensor(sequence.reference_features[reference_start:reference_end], dtype=torch.float32)
        target = torch.as_tensor(sequence.target_features[future_start:future_end], dtype=torch.float32)
        reference_normalized = _normalize_reference(reference.unsqueeze(0), schema, device=device)
        target_normalized = _normalize_target(target.unsqueeze(0), schema, device=cpu_device).squeeze(0)
        with torch.no_grad():
            outputs = model(reference_normalized)
            prediction_normalized = outputs["prediction"].detach().cpu().squeeze(0)
        prediction = denormalize_target(prediction_normalized.unsqueeze(0), schema).squeeze(0)
        root_state = _extract_future_root_and_joint_states(
            canonical_sequence,
            schema=schema,
            future_frame_indices=future_frame_indices,
        )

        for local_index, frame_number in enumerate(future_frame_indices):
            future_step_ahead = local_index + 1
            existing = selected_frames.get(frame_number)
            if existing is not None and future_step_ahead >= existing["future_step_ahead"]:
                continue
            selected_frames[frame_number] = {
                "center_t": center_t,
                "future_step_ahead": future_step_ahead,
                "reference": reference.clone(),
                "reference_normalized": reference_normalized.squeeze(0).cpu().clone(),
                "target": target[local_index].clone(),
                "target_normalized": target_normalized[local_index].clone(),
                "prediction": prediction[local_index].clone(),
                "prediction_normalized": prediction_normalized[local_index].clone(),
                "root_pos": root_state["root_pos"][local_index].clone(),
                "root_quat": root_state["root_quat"][local_index].clone(),
                "root_lin_vel": root_state["root_lin_vel"][local_index].clone(),
                "root_ang_vel": root_state["root_ang_vel"][local_index].clone(),
                "joint_pos": root_state["joint_pos"][local_index].clone(),
                "joint_vel": root_state["joint_vel"][local_index].clone(),
            }

    selected_future_frame_indices = tuple(sorted(selected_frames))
    root_state = {
        "future_frame_indices": torch.as_tensor(selected_future_frame_indices, dtype=torch.long),
        "root_pos": torch.stack([selected_frames[frame]["root_pos"] for frame in selected_future_frame_indices], dim=0),
        "root_quat": torch.stack([selected_frames[frame]["root_quat"] for frame in selected_future_frame_indices], dim=0),
        "root_lin_vel": torch.stack([selected_frames[frame]["root_lin_vel"] for frame in selected_future_frame_indices], dim=0),
        "root_ang_vel": torch.stack([selected_frames[frame]["root_ang_vel"] for frame in selected_future_frame_indices], dim=0),
        "joint_pos": torch.stack([selected_frames[frame]["joint_pos"] for frame in selected_future_frame_indices], dim=0),
        "joint_vel": torch.stack([selected_frames[frame]["joint_vel"] for frame in selected_future_frame_indices], dim=0),
    }
    return MotionMAEVisualizationBatch(
        motion_name=sequence.motion_name,
        motion_file=sequence.motion_file,
        center_t=None,
        center_t_by_frame=tuple(selected_frames[frame]["center_t"] for frame in selected_future_frame_indices),
        future_frame_indices=selected_future_frame_indices,
        future_step_ahead=tuple(selected_frames[frame]["future_step_ahead"] for frame in selected_future_frame_indices),
        reference=torch.stack([selected_frames[frame]["reference"] for frame in selected_future_frame_indices], dim=0),
        reference_normalized=torch.stack(
            [selected_frames[frame]["reference_normalized"] for frame in selected_future_frame_indices],
            dim=0,
        ),
        target=torch.stack([selected_frames[frame]["target"] for frame in selected_future_frame_indices], dim=0),
        target_normalized=torch.stack(
            [selected_frames[frame]["target_normalized"] for frame in selected_future_frame_indices],
            dim=0,
        ),
        prediction=torch.stack([selected_frames[frame]["prediction"] for frame in selected_future_frame_indices], dim=0),
        prediction_normalized=torch.stack(
            [selected_frames[frame]["prediction_normalized"] for frame in selected_future_frame_indices],
            dim=0,
        ),
        root_state=root_state,
    )


def extract_joint_trajectory(target: torch.Tensor, schema: MotionFeatureSchema) -> tuple[torch.Tensor, torch.Tensor]:
    joint_slice = schema.target_slice_map().get("joint")
    if joint_slice is None:
        raise ValueError("Motion MAE schema does not expose a target 'joint' slice.")
    joint_count = len(schema.joint_names)
    if joint_slice.dim != 2 * joint_count:
        raise ValueError(
            f"Expected joint slice dim={2 * joint_count} for {joint_count} joints, got {joint_slice.dim}."
        )
    joint_terms = torch.as_tensor(target, dtype=torch.float32)[..., joint_slice.start : joint_slice.end]
    return joint_terms[..., :joint_count], joint_terms[..., joint_count:]


def extract_root_trajectory(target: torch.Tensor, schema: MotionFeatureSchema) -> torch.Tensor:
    root_slice = schema.target_slice_map().get("root")
    if root_slice is None:
        raise ValueError("Motion MAE schema does not expose a target 'root' slice.")
    return torch.as_tensor(target, dtype=torch.float32)[..., root_slice.start : root_slice.end]


def validate_rendered_ground_truth_alignment(
    *,
    target: torch.Tensor,
    gt_joint_pos: torch.Tensor,
    gt_joint_vel: torch.Tensor,
    schema: MotionFeatureSchema,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> dict[str, float]:
    target_joint_pos, target_joint_vel = extract_joint_trajectory(target, schema)
    gt_joint_pos = torch.as_tensor(gt_joint_pos, dtype=torch.float32)
    gt_joint_vel = torch.as_tensor(gt_joint_vel, dtype=torch.float32)

    joint_pos_max_abs_error = float(torch.max(torch.abs(target_joint_pos - gt_joint_pos)).item())
    joint_vel_max_abs_error = float(torch.max(torch.abs(target_joint_vel - gt_joint_vel)).item())
    if not torch.allclose(target_joint_pos, gt_joint_pos, atol=atol, rtol=rtol):
        raise ValueError(
            "Ground-truth joint positions used for rendering do not match the Motion MAE target window "
            f"(max_abs_error={joint_pos_max_abs_error:.6e})."
        )
    if not torch.allclose(target_joint_vel, gt_joint_vel, atol=atol, rtol=rtol):
        raise ValueError(
            "Ground-truth joint velocities used for rendering do not match the Motion MAE target window "
            f"(max_abs_error={joint_vel_max_abs_error:.6e})."
        )

    return {
        "joint_pos_max_abs_error": joint_pos_max_abs_error,
        "joint_vel_max_abs_error": joint_vel_max_abs_error,
    }


def build_playback_frame_pair(
    *,
    gt_root_pos: torch.Tensor,
    gt_root_quat: torch.Tensor,
    gt_root_lin_vel: torch.Tensor,
    gt_root_ang_vel: torch.Tensor,
    gt_joint_pos: torch.Tensor,
    gt_joint_vel: torch.Tensor,
    pred_joint_pos: torch.Tensor,
    pred_joint_vel: torch.Tensor,
) -> MotionFramePair:
    gt_state = MotionFrameState(
        root_pos=torch.as_tensor(gt_root_pos, dtype=torch.float32),
        root_quat=torch.as_tensor(gt_root_quat, dtype=torch.float32),
        root_lin_vel=torch.as_tensor(gt_root_lin_vel, dtype=torch.float32),
        root_ang_vel=torch.as_tensor(gt_root_ang_vel, dtype=torch.float32),
        joint_pos=torch.as_tensor(gt_joint_pos, dtype=torch.float32),
        joint_vel=torch.as_tensor(gt_joint_vel, dtype=torch.float32),
    )
    pred_state = MotionFrameState(
        root_pos=gt_state.root_pos.clone(),
        root_quat=normalize_quaternion(gt_state.root_quat.clone()),
        root_lin_vel=gt_state.root_lin_vel.clone(),
        root_ang_vel=gt_state.root_ang_vel.clone(),
        joint_pos=torch.as_tensor(pred_joint_pos, dtype=torch.float32),
        joint_vel=torch.as_tensor(pred_joint_vel, dtype=torch.float32),
    )
    return MotionFramePair(gt=gt_state, pred=pred_state)


def compute_visualization_metrics(
    *,
    target: torch.Tensor,
    prediction: torch.Tensor,
    schema: MotionFeatureSchema,
    gt_joint_pos: torch.Tensor,
    gt_joint_vel: torch.Tensor,
    pred_joint_pos: torch.Tensor,
    pred_joint_vel: torch.Tensor,
) -> dict[str, Any]:
    target = torch.as_tensor(target, dtype=torch.float32)
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    gt_joint_pos = torch.as_tensor(gt_joint_pos, dtype=torch.float32)
    gt_joint_vel = torch.as_tensor(gt_joint_vel, dtype=torch.float32)
    pred_joint_pos = torch.as_tensor(pred_joint_pos, dtype=torch.float32)
    pred_joint_vel = torch.as_tensor(pred_joint_vel, dtype=torch.float32)
    gt_root = extract_root_trajectory(target, schema=schema)
    pred_root = extract_root_trajectory(prediction, schema=schema)
    # All per-frame errors are computed independently for the aligned +k future frame,
    # using mean absolute error over the feature dimensions/joints for that frame only.
    gravity_mae_by_frame = torch.mean(torch.abs(pred_root - gt_root), dim=-1)
    joint_pos_mae_by_frame = torch.mean(torch.abs(pred_joint_pos - gt_joint_pos), dim=-1)
    joint_vel_mae_by_frame = torch.mean(torch.abs(pred_joint_vel - gt_joint_vel), dim=-1)
    gravity_mae = float(torch.mean(torch.abs(pred_root - gt_root)).item())
    return {
        "full_target_mse": float(torch.mean(torch.square(prediction - target)).item()),
        "full_target_l1": float(torch.mean(torch.abs(prediction - target)).item()),
        "gravity_mae": gravity_mae,
        "root_mae": gravity_mae,
        "joint_pos_mae": float(torch.mean(torch.abs(pred_joint_pos - gt_joint_pos)).item()),
        "joint_vel_mae": float(torch.mean(torch.abs(pred_joint_vel - gt_joint_vel)).item()),
        "gravity_mae_by_frame": gravity_mae_by_frame.tolist(),
        "root_mae_by_frame": gravity_mae_by_frame.tolist(),
        "joint_pos_mae_by_frame": joint_pos_mae_by_frame.tolist(),
        "joint_vel_mae_by_frame": joint_vel_mae_by_frame.tolist(),
    }


def _load_canonical_sequence(
    adapter: MotionSourceAdapter,
    *,
    motion_file: str,
    expected_schema: MotionFeatureSchema,
) -> CanonicalMotionSequence:
    sequence = adapter.load_sequence(motion_file)
    if tuple(sequence.joint_names) != tuple(expected_schema.joint_names):
        raise ValueError("Motion asset joint_names do not match the Motion MAE checkpoint schema.")
    if tuple(sequence.body_names) != tuple(expected_schema.body_names):
        raise ValueError("Motion asset body_names do not match the Motion MAE checkpoint schema.")
    return sequence


def _extract_future_root_and_joint_states(
    sequence: CanonicalMotionSequence,
    *,
    schema: MotionFeatureSchema,
    future_frame_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    frame_indices = torch.as_tensor(future_frame_indices, dtype=torch.long)
    try:
        root_body_index = sequence.body_names.index(schema.anchor_body_name)
    except ValueError as exc:
        raise ValueError(f"Root body '{schema.anchor_body_name}' was not found in motion asset body_names.") from exc
    return {
        "future_frame_indices": frame_indices,
        "root_pos": sequence.body_pos_w[frame_indices, root_body_index].clone(),
        "root_quat": sequence.body_quat_w[frame_indices, root_body_index].clone(),
        "root_lin_vel": sequence.body_lin_vel_w[frame_indices, root_body_index].clone(),
        "root_ang_vel": sequence.body_ang_vel_w[frame_indices, root_body_index].clone(),
        "joint_pos": sequence.joint_pos[frame_indices].clone(),
        "joint_vel": sequence.joint_vel[frame_indices].clone(),
    }


def _build_dummy_joint_params(joint_dim: int) -> dict[str, torch.Tensor]:
    joint_limits = torch.full((joint_dim, 2), DEFAULT_JOINT_LIMIT, dtype=torch.float32)
    joint_limits[:, 0] *= -1.0
    return {
        "kp": torch.zeros(joint_dim, dtype=torch.float32),
        "kd": torch.zeros(joint_dim, dtype=torch.float32),
        "effort_limits": torch.full((joint_dim,), 1.0e3, dtype=torch.float32),
        "joint_pos_limits": joint_limits,
        "action_offset": torch.zeros(joint_dim, dtype=torch.float32),
        "action_scale": torch.ones(joint_dim, dtype=torch.float32),
    }


def resolve_renderer_size(
    mj_model: Any,
    *,
    requested_width: int,
    requested_height: int,
) -> tuple[int, int]:
    requested_width = max(1, int(requested_width))
    requested_height = max(1, int(requested_height))

    vis = getattr(mj_model, "vis", None)
    global_vis = getattr(vis, "global_", None)
    max_width = int(getattr(global_vis, "offwidth", requested_width))
    max_height = int(getattr(global_vis, "offheight", requested_height))
    max_width = max(1, max_width)
    max_height = max(1, max_height)

    if requested_width <= max_width and requested_height <= max_height:
        return requested_width, requested_height

    scale = min(max_width / requested_width, max_height / requested_height)
    resolved_width = max(1, min(max_width, int(np.floor(requested_width * scale))))
    resolved_height = max(1, min(max_height, int(np.floor(requested_height * scale))))
    return resolved_width, resolved_height


def _camera_globals(mj_model: Any) -> Any:
    vis = getattr(mj_model, "vis", None)
    return getattr(vis, "global_", None)


def _resolve_camera_angles(mj_model: Any) -> tuple[float, float]:
    global_vis = _camera_globals(mj_model)
    azimuth = float(getattr(global_vis, "azimuth", DEFAULT_CAMERA_AZIMUTH))
    elevation = float(getattr(global_vis, "elevation", DEFAULT_CAMERA_ELEVATION))
    return azimuth, elevation


def _resolve_camera_fovy(mj_model: Any) -> float:
    global_vis = _camera_globals(mj_model)
    return float(getattr(global_vis, "fovy", DEFAULT_CAMERA_FOVY))


def update_tracking_camera(
    camera: Any,
    *,
    mj_model: Any,
    body_positions: np.ndarray,
    frame_width: int,
    frame_height: int,
    mujoco_module: Any,
) -> None:
    positions = np.asarray(body_positions, dtype=np.float32).reshape(-1, 3)
    if positions.size == 0:
        return

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 1.0e-3)
    bbox_center = 0.5 * (bbox_min + bbox_max)

    vertical_half_extent = 0.5 * float(bbox_size[2])
    horizontal_half_extent = 0.5 * float(max(bbox_size[0], bbox_size[1]))
    fovy = np.deg2rad(_resolve_camera_fovy(mj_model))
    aspect_ratio = max(1.0, float(frame_width) / max(float(frame_height), 1.0))
    half_fov_y = max(fovy / 2.0, np.deg2rad(1.0))
    half_fov_x = np.arctan(np.tan(half_fov_y) * aspect_ratio)
    bbox_radius = 0.5 * float(np.linalg.norm(bbox_size))

    distance_from_height = vertical_half_extent / np.tan(half_fov_y)
    distance_from_width = horizontal_half_extent / np.tan(half_fov_x)
    camera_distance = CAMERA_PADDING * max(distance_from_height, distance_from_width, bbox_radius)
    camera_distance = max(MIN_CAMERA_DISTANCE, float(camera_distance))
    azimuth, elevation = _resolve_camera_angles(mj_model)

    camera.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.lookat[:] = bbox_center
    camera.distance = camera_distance
    camera.azimuth = azimuth
    camera.elevation = elevation


class PassiveMujocoPlayback:
    def __init__(
        self,
        *,
        motion_file: str,
        root_body_name: str,
        anchor_body_name: str,
        width: int = DEFAULT_FRAME_WIDTH,
        height: int = DEFAULT_FRAME_HEIGHT,
    ) -> None:
        from gmtp.integrations.ref2act.compat import load_mujoco_symbols

        import mujoco

        joint_dim = SUPPORTED_JOINT_COUNT
        params = _build_dummy_joint_params(joint_dim)
        self._mujoco = mujoco
        self._symbols = load_mujoco_symbols()
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.env = self._symbols.MujocoEnv(
            simulation_dt=1.0 / 200.0,
            decimation=4,
            kp=params["kp"],
            kd=params["kd"],
            effort_limits=params["effort_limits"],
            joint_pos_limits=params["joint_pos_limits"],
            action_offset=params["action_offset"],
            action_scale=params["action_scale"],
            expert_motion_file=motion_file,
            root_link_name=root_body_name,
            anchor_body_name=anchor_body_name,
            render=False,
            action_mode="absolute",
        )
        self.env.reset()
        renderer_width, renderer_height = resolve_renderer_size(
            self.env.mj_model,
            requested_width=width,
            requested_height=height,
        )
        self.frame_width = renderer_width
        self.frame_height = renderer_height
        mujoco.mjv_defaultFreeCamera(self.env.mj_model, self.camera)
        self.renderer = mujoco.Renderer(self.env.mj_model, height=renderer_height, width=renderer_width)

    def render_frame(self, frame_state: MotionFrameState) -> np.ndarray:
        joint_pos = np.asarray(frame_state.joint_pos.detach().cpu(), dtype=np.float32)
        joint_vel = np.asarray(frame_state.joint_vel.detach().cpu(), dtype=np.float32)
        free_joint_pose, free_joint_velocity = self.env._solve_free_joint_state_from_root_reference(
            root_pos=np.asarray(frame_state.root_pos.detach().cpu(), dtype=np.float32),
            root_quat=np.asarray(frame_state.root_quat.detach().cpu(), dtype=np.float32),
            root_linear_vel=np.asarray(frame_state.root_lin_vel.detach().cpu(), dtype=np.float32),
            root_angular_vel=np.asarray(frame_state.root_ang_vel.detach().cpu(), dtype=np.float32),
            joint_positions=joint_pos[self.env.isaac2mujoco],
            joint_velocities=joint_vel[self.env.isaac2mujoco],
        )
        self.env.mj_data.qpos[:7] = free_joint_pose
        self.env.mj_data.qpos[7:] = joint_pos[self.env.isaac2mujoco]
        self.env.mj_data.qvel[:6] = free_joint_velocity
        self.env.mj_data.qvel[6:] = joint_vel[self.env.isaac2mujoco]
        self._mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
        update_tracking_camera(
            self.camera,
            mj_model=self.env.mj_model,
            body_positions=np.asarray(self.env.mj_data.xpos[1:], dtype=np.float32),
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            mujoco_module=self._mujoco,
        )
        self.renderer.update_scene(self.env.mj_data, camera=self.camera)
        return np.asarray(self.renderer.render(), dtype=np.uint8).copy()

    def close(self) -> None:
        try:
            if self.renderer is not None:
                self.renderer.close()
        finally:
            self.env.close()


def _draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    *,
    color: tuple[int, int, int],
    scale: int = 2,
) -> None:
    cursor_x = int(x)
    for char in str(text).upper():
        glyph = _FONT_3X5.get(char, _FONT_3X5["?"])
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == " ":
                    continue
                x_start = cursor_x + col_index * scale
                y_start = int(y) + row_index * scale
                image[y_start : y_start + scale, x_start : x_start + scale] = color
        cursor_x += 4 * scale


def compose_annotated_frame(
    gt_frame: np.ndarray,
    pred_frame: np.ndarray,
    *,
    motion_name: str,
    frame_index: int,
    total_frames: int,
    future_frame_number: int,
    k_step_ahead: int,
    gravity_mae: float,
    joint_pos_mae: float,
    joint_vel_mae: float,
) -> np.ndarray:
    composite = np.concatenate((gt_frame, pred_frame), axis=1)
    header_height = 72
    canvas = np.zeros((header_height + composite.shape[0], composite.shape[1], 3), dtype=np.uint8)
    canvas[header_height:] = composite
    _draw_text(canvas, 8, 4, f"MOTION {motion_name}", color=(255, 255, 255))
    _draw_text(
        canvas,
        8,
        20,
        f"FRAME +{k_step_ahead} IDX {future_frame_number}",
        color=(255, 255, 255),
    )
    _draw_text(
        canvas,
        8,
        36,
        f"G:{gravity_mae:.2f} Q:{joint_pos_mae:.2f} QD:{joint_vel_mae:.2f}",
        color=(255, 255, 255),
    )
    _draw_text(canvas, 8, 52, f"VIEW {frame_index + 1}/{total_frames} POS GT ORI GT", color=(255, 192, 0))
    _draw_text(canvas, 8, header_height + 8, "GT", color=(0, 192, 255))
    _draw_text(canvas, gt_frame.shape[1] + 8, header_height + 8, "PRED", color=(255, 192, 0))
    padded_height = int(np.ceil(canvas.shape[0] / VIDEO_MACRO_BLOCK_SIZE) * VIDEO_MACRO_BLOCK_SIZE)
    padded_width = int(np.ceil(canvas.shape[1] / VIDEO_MACRO_BLOCK_SIZE) * VIDEO_MACRO_BLOCK_SIZE)
    if padded_height == canvas.shape[0] and padded_width == canvas.shape[1]:
        return canvas
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded[: canvas.shape[0], : canvas.shape[1]] = canvas
    return padded


def expand_rendered_frames_to_motion_timeline(
    rendered_frames: list[np.ndarray],
    *,
    future_frame_indices: tuple[int, ...],
    whole_motion: bool,
) -> list[np.ndarray]:
    if not whole_motion or len(rendered_frames) <= 1:
        return list(rendered_frames)
    if len(rendered_frames) != len(future_frame_indices):
        raise ValueError(
            "Rendered frame count does not match future_frame_indices for whole-motion MAE visualization."
        )

    expanded_frames: list[np.ndarray] = []
    for frame_index, frame in enumerate(rendered_frames):
        expanded_frames.append(np.asarray(frame, dtype=np.uint8).copy())
        if frame_index == len(rendered_frames) - 1:
            continue
        frame_gap = int(future_frame_indices[frame_index + 1]) - int(future_frame_indices[frame_index])
        if frame_gap < 1:
            raise ValueError("whole-motion future_frame_indices must be strictly increasing.")
        expanded_frames.extend(np.asarray(frame, dtype=np.uint8).copy() for _ in range(frame_gap - 1))
    return expanded_frames


class MotionMAEVisualizerRunner:
    def __init__(self, config: MotionMAEVisualizationConfig) -> None:
        self.config = config
        self.checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        self.config_path = Path(config.config_path).expanduser().resolve()

    def _load_pretrain_config(self) -> MotionMAEPretrainConfig:
        config = load_motion_mae_pretrain_config(self.config_path)
        return apply_motion_mae_cli_overrides(
            config,
            motion_files=self.config.motion_files,
            output_root=self.config.output_root,
            run_name=self.config.run_name,
            device=self.config.device,
        )

    def _build_run_name(self, *, motion_name: str) -> str:
        if self.config.run_name is not None:
            return self.config.run_name
        if self.config.whole_motion:
            return f"{self.checkpoint_path.stem}_{motion_name}_whole"
        frame_suffix = ""
        if self.config.future_frame_index is not None:
            frame_suffix = f"_future{self.config.future_frame_index}"
        return (
            f"{self.checkpoint_path.stem}_{self.config.split}_{motion_name}_{self.config.sample_index}"
            f"{frame_suffix}"
        )

    def _write_comparison_payload(
        self,
        *,
        output_path: Path,
        batch: MotionMAEVisualizationBatch,
        future_frame_indices: tuple[int, ...],
        future_step_ahead: tuple[int, ...],
        sequence: CanonicalMotionSequence,
        gt_joint_pos: torch.Tensor,
        gt_joint_vel: torch.Tensor,
        pred_joint_pos: torch.Tensor,
        pred_joint_vel: torch.Tensor,
        ground_truth_alignment: dict[str, float],
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            motion_name=np.asarray(batch.motion_name),
            motion_file=np.asarray(batch.motion_file),
            motion_fps=np.asarray(sequence.fps, dtype=np.float32),
            center_t=np.asarray(-1 if batch.center_t is None else batch.center_t, dtype=np.int64),
            center_t_by_frame=np.asarray(batch.center_t_by_frame, dtype=np.int64),
            whole_motion=np.asarray(self.config.whole_motion),
            future_frame_index=(
                np.asarray(-1, dtype=np.int64)
                if self.config.future_frame_index is None
                else np.asarray(self.config.future_frame_index, dtype=np.int64)
            ),
            future_frame_indices=np.asarray(future_frame_indices, dtype=np.int64),
            future_step_ahead=np.asarray(future_step_ahead, dtype=np.int64),
            reference=batch.reference.numpy(),
            reference_normalized=batch.reference_normalized.numpy(),
            target=batch.target.numpy(),
            target_normalized=batch.target_normalized.numpy(),
            prediction=batch.prediction.numpy(),
            prediction_normalized=batch.prediction_normalized.numpy(),
            gt_root_pos=batch.root_state["root_pos"].numpy(),
            gt_root_quat=batch.root_state["root_quat"].numpy(),
            gt_root_lin_vel=batch.root_state["root_lin_vel"].numpy(),
            gt_root_ang_vel=batch.root_state["root_ang_vel"].numpy(),
            gt_joint_pos=gt_joint_pos.numpy(),
            gt_joint_vel=gt_joint_vel.numpy(),
            pred_joint_pos=pred_joint_pos.numpy(),
            pred_joint_vel=pred_joint_vel.numpy(),
            render_comparison_mode=np.asarray(PREDICTION_RENDER_MODE),
            render_note=np.asarray(PREDICTION_RENDER_NOTE),
            pred_view_uses_ground_truth_root_position=np.asarray(True),
            pred_view_uses_ground_truth_root_orientation=np.asarray(True),
            pred_view_uses_ground_truth_root_velocity=np.asarray(True),
            target_joint_alignment_verified=np.asarray(True),
            target_joint_pos_max_abs_error=np.asarray(ground_truth_alignment["joint_pos_max_abs_error"], dtype=np.float32),
            target_joint_vel_max_abs_error=np.asarray(ground_truth_alignment["joint_vel_max_abs_error"], dtype=np.float32),
        )
        return output_path

    def visualize(self) -> dict[str, Any]:
        pretrain_config = self._load_pretrain_config()
        if self.config.whole_motion and self.config.future_frame_index is not None:
            raise ValueError("whole-motion visualization does not support future_frame_index selection.")
        device = resolve_visualization_device(pretrain_config.training.device)
        model, checkpoint = build_motion_mae_model_from_checkpoint(self.checkpoint_path, device=device)
        checkpoint_schema = checkpoint.schema
        validate_visualizer_schema_support(checkpoint_schema)
        validate_checkpoint_and_config_compatibility(checkpoint_schema, model, pretrain_config)

        data_bundle = build_motion_mae_datasets(
            pretrain_config.data,
            feature_config=pretrain_config.feature,
            slice_weights=pretrain_config.loss.slice_weights,
        )
        validate_schema_compatibility(data_bundle.schema, checkpoint_schema)

        dataset = data_bundle.train_dataset if self.config.split == "train" else data_bundle.val_dataset
        adapter = build_motion_source_adapter(pretrain_config.data.adapter_name)
        if self.config.whole_motion:
            sequence_index, selected_sequence = select_visualization_sequence(
                dataset,
                motion_name=self.config.motion_name,
            )
            sequence = _load_canonical_sequence(
                adapter,
                motion_file=selected_sequence.motion_file,
                expected_schema=checkpoint_schema,
            )
            batch = build_whole_motion_visualization_batch(
                dataset=dataset,
                sequence=selected_sequence,
                canonical_sequence=sequence,
                schema=checkpoint_schema,
                model=model,
                device=device,
            )
        else:
            sample = select_visualization_sample(
                dataset,
                split_name=self.config.split,
                motion_name=self.config.motion_name,
                sample_index=self.config.sample_index,
            )
            sequence = _load_canonical_sequence(adapter, motion_file=sample.motion_file, expected_schema=checkpoint_schema)
            batch = build_sample_visualization_batch(
                sample=sample,
                sequence=sequence,
                schema=checkpoint_schema,
                model=model,
                device=device,
                future_frame_index=self.config.future_frame_index,
            )

        run_paths = build_run_paths(
            pretrain_config.output_root,
            "pretrain-motion-mae-visualize",
            self._build_run_name(motion_name=batch.motion_name),
        )
        write_json(
            run_paths.config_path,
            {
                "command": "pretrain motion-mae-visualize",
                "config": self.config,
                "pretrain_config": pretrain_config.to_dict(),
                "checkpoint": str(self.checkpoint_path),
            },
        )

        selected_future_frame_indices = batch.future_frame_indices
        selected_future_step_ahead = batch.future_step_ahead
        gt_joint_pos = batch.root_state["joint_pos"]
        gt_joint_vel = batch.root_state["joint_vel"]
        ground_truth_alignment = validate_rendered_ground_truth_alignment(
            target=batch.target,
            gt_joint_pos=gt_joint_pos,
            gt_joint_vel=gt_joint_vel,
            schema=checkpoint_schema,
        )
        pred_joint_pos, pred_joint_vel = extract_joint_trajectory(batch.prediction, checkpoint_schema)
        metrics = compute_visualization_metrics(
            target=batch.target,
            prediction=batch.prediction,
            schema=checkpoint_schema,
            gt_joint_pos=gt_joint_pos,
            gt_joint_vel=gt_joint_vel,
            pred_joint_pos=pred_joint_pos,
            pred_joint_vel=pred_joint_vel,
        )

        fps = int(self.config.fps) if self.config.fps is not None else max(1, round(float(sequence.fps)))
        is_single_frame = len(selected_future_frame_indices) == 1
        media_path = run_paths.videos_dir / (
            f"{batch.motion_name}_comparison.png" if is_single_frame else f"{batch.motion_name}_comparison.mp4"
        )
        comparison_path = run_paths.checkpoints_dir / "comparison.npz"

        gt_viewer = PassiveMujocoPlayback(
            motion_file=batch.motion_file,
            root_body_name=checkpoint_schema.anchor_body_name,
            anchor_body_name=checkpoint_schema.anchor_body_name,
        )
        pred_viewer = PassiveMujocoPlayback(
            motion_file=batch.motion_file,
            root_body_name=checkpoint_schema.anchor_body_name,
            anchor_body_name=checkpoint_schema.anchor_body_name,
        )
        try:
            rendered_frames: list[np.ndarray] = []
            for frame_index in range(len(selected_future_frame_indices)):
                frame_pair = build_playback_frame_pair(
                    gt_root_pos=batch.root_state["root_pos"][frame_index],
                    gt_root_quat=batch.root_state["root_quat"][frame_index],
                    gt_root_lin_vel=batch.root_state["root_lin_vel"][frame_index],
                    gt_root_ang_vel=batch.root_state["root_ang_vel"][frame_index],
                    gt_joint_pos=gt_joint_pos[frame_index],
                    gt_joint_vel=gt_joint_vel[frame_index],
                    pred_joint_pos=pred_joint_pos[frame_index],
                    pred_joint_vel=pred_joint_vel[frame_index],
                )
                gt_frame = gt_viewer.render_frame(frame_pair.gt)
                pred_frame = pred_viewer.render_frame(frame_pair.pred)
                video_frame = compose_annotated_frame(
                    gt_frame,
                    pred_frame,
                    motion_name=batch.motion_name,
                    frame_index=frame_index,
                    total_frames=len(selected_future_frame_indices),
                    future_frame_number=selected_future_frame_indices[frame_index],
                    k_step_ahead=selected_future_step_ahead[frame_index],
                    gravity_mae=float(metrics["gravity_mae_by_frame"][frame_index]),
                    joint_pos_mae=float(metrics["joint_pos_mae_by_frame"][frame_index]),
                    joint_vel_mae=float(metrics["joint_vel_mae_by_frame"][frame_index]),
                )
                rendered_frames.append(video_frame)
            video_frames = expand_rendered_frames_to_motion_timeline(
                rendered_frames,
                future_frame_indices=selected_future_frame_indices,
                whole_motion=self.config.whole_motion,
            )
            if is_single_frame:
                imageio.imwrite(media_path, video_frames[0])
            else:
                writer = imageio.get_writer(media_path, fps=fps)
                try:
                    for frame in video_frames:
                        writer.append_data(frame)
                finally:
                    writer.close()
        finally:
            gt_viewer.close()
            pred_viewer.close()

        comparison_artifact = self._write_comparison_payload(
            output_path=comparison_path,
            batch=batch,
            future_frame_indices=selected_future_frame_indices,
            future_step_ahead=selected_future_step_ahead,
            sequence=sequence,
            ground_truth_alignment=ground_truth_alignment,
            gt_joint_pos=gt_joint_pos,
            gt_joint_vel=gt_joint_vel,
            pred_joint_pos=pred_joint_pos,
            pred_joint_vel=pred_joint_vel,
        )
        summary = {
            "checkpoint": str(self.checkpoint_path),
            "config_path": str(self.config_path),
            "split": self.config.split,
            "whole_motion": self.config.whole_motion,
            "window_source": "all_valid_windows_for_motion" if self.config.whole_motion else "selected_sample",
            "motion_name": batch.motion_name,
            "motion_file": batch.motion_file,
            "center_t": batch.center_t,
            "center_t_by_frame": list(batch.center_t_by_frame),
            "future_frame_index": self.config.future_frame_index,
            "future_frame_indices": list(selected_future_frame_indices),
            "future_step_ahead": list(selected_future_step_ahead),
            "fps": fps,
            "rendered_video_frame_count": len(video_frames),
            "rendered_video_duration_seconds": len(video_frames) / float(fps),
            "media_type": "image" if is_single_frame else "video",
            "metrics": metrics,
            "render_comparison_mode": PREDICTION_RENDER_MODE,
            "render_note": PREDICTION_RENDER_NOTE,
            "pred_view_uses_ground_truth_root_position": True,
            "pred_view_uses_ground_truth_root_orientation": True,
            "pred_view_uses_ground_truth_root_velocity": True,
            "target_joint_alignment_verified": True,
            "target_joint_alignment": ground_truth_alignment,
            "artifacts": (
                {
                    "comparison_npz": str(comparison_artifact.resolve()),
                    "image": str(media_path.resolve()),
                }
                if is_single_frame
                else {
                    "comparison_npz": str(comparison_artifact.resolve()),
                    "video": str(media_path.resolve()),
                }
            ),
            "run_dir": str(run_paths.root),
        }
        write_json(run_paths.summary_path, summary)
        return summary
