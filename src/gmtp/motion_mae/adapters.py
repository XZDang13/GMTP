from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from gmtp.integrations.ref2act.motion import resolve_motion_file

from .schema import CanonicalMotionSequence, MotionSegment


@runtime_checkable
class MotionSourceAdapter(Protocol):
    name: str

    def load_sequence(self, motion_file: str) -> CanonicalMotionSequence:
        raise NotImplementedError


def _round_frame_index(time_s: float, fps: float) -> int:
    return int(round(float(time_s) * float(fps)))


def _build_segments(
    *,
    segment_start_times: np.ndarray,
    segment_end_times: np.ndarray,
    segment_types: np.ndarray | None,
    fps: float,
    num_frames: int,
) -> tuple[MotionSegment, ...]:
    if segment_start_times.shape != segment_end_times.shape:
        raise ValueError(
            "segment_start_times and segment_end_times must have the same shape, "
            f"got {segment_start_times.shape} vs {segment_end_times.shape}."
        )
    if segment_types is not None and segment_types.shape != segment_start_times.shape:
        raise ValueError(
            "segment_types must match segment time shape, "
            f"got {segment_types.shape} vs {segment_start_times.shape}."
        )

    segments: list[MotionSegment] = []
    for index in range(int(segment_start_times.shape[0])):
        start_frame = _round_frame_index(float(segment_start_times[index]), fps)
        end_frame = _round_frame_index(float(segment_end_times[index]), fps)
        start_frame = min(max(start_frame, 0), num_frames)
        end_frame = min(max(end_frame, 0), num_frames)
        if end_frame <= start_frame:
            raise ValueError(
                f"Segment {index} collapsed after frame rounding/clipping: {start_frame}:{end_frame}."
            )
        segment_type = None if segment_types is None else int(segment_types[index])
        segments.append(
            MotionSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                segment_type=segment_type,
            )
        )

    if not segments:
        raise ValueError("Motion source did not provide any valid segments.")
    return tuple(segments)


class StageIINpzMotionAdapter:
    name = "stageii_npz"

    REQUIRED_KEYS = (
        "fps",
        "joint_names",
        "body_names",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "segment_start_times",
        "segment_end_times",
    )

    def load_sequence(self, motion_file: str) -> CanonicalMotionSequence:
        resolved_motion_file = resolve_motion_file(motion_file)
        with np.load(resolved_motion_file, allow_pickle=True) as payload:
            missing_keys = [key for key in self.REQUIRED_KEYS if key not in payload]
            if missing_keys:
                raise KeyError(f"StageII motion asset is missing required keys: {missing_keys}.")

            fps = float(payload["fps"])
            joint_names = tuple(str(item) for item in payload["joint_names"].tolist())
            body_names = tuple(str(item) for item in payload["body_names"].tolist())
            joint_pos = torch.as_tensor(payload["joint_pos"], dtype=torch.float32)
            joint_vel = torch.as_tensor(payload["joint_vel"], dtype=torch.float32)
            body_pos_w = torch.as_tensor(payload["body_pos_w"], dtype=torch.float32)
            body_quat_w = torch.as_tensor(payload["body_quat_w"], dtype=torch.float32)
            body_lin_vel_w = torch.as_tensor(payload["body_lin_vel_w"], dtype=torch.float32)
            body_ang_vel_w = torch.as_tensor(payload["body_ang_vel_w"], dtype=torch.float32)

            num_frames = int(joint_pos.shape[0])
            if num_frames < 1:
                raise ValueError(f"Motion asset {resolved_motion_file} does not contain any frames.")
            if joint_vel.shape != joint_pos.shape:
                raise ValueError(
                    f"joint_pos/joint_vel shape mismatch: {tuple(joint_pos.shape)} vs {tuple(joint_vel.shape)}."
                )
            if body_pos_w.shape[:2] != body_quat_w.shape[:2]:
                raise ValueError(
                    "body_pos_w/body_quat_w must share frame/body axes, "
                    f"got {tuple(body_pos_w.shape)} vs {tuple(body_quat_w.shape)}."
                )

            segments = _build_segments(
                segment_start_times=np.asarray(payload["segment_start_times"], dtype=np.float32).reshape(-1),
                segment_end_times=np.asarray(payload["segment_end_times"], dtype=np.float32).reshape(-1),
                segment_types=(
                    np.asarray(payload["segment_types"], dtype=np.int64).reshape(-1)
                    if "segment_types" in payload
                    else None
                ),
                fps=fps,
                num_frames=num_frames,
            )

        return CanonicalMotionSequence(
            motion_file=str(Path(resolved_motion_file).resolve()),
            motion_name=Path(resolved_motion_file).stem,
            fps=fps,
            joint_names=joint_names,
            body_names=body_names,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            segments=segments,
        )


def build_motion_source_adapter(adapter_name: str) -> MotionSourceAdapter:
    normalized = str(adapter_name).strip().lower()
    if normalized == "stageii_npz":
        return StageIINpzMotionAdapter()
    raise ValueError(f"Unsupported motion source adapter '{adapter_name}'.")
