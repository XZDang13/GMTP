from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from .config import MotionVAEDataConfig, MotionVAEFeatureConfig
from .features import MotionFeatureBundle, MotionFeatureSequence, build_motion_feature_bundle
from .schema import MotionFeatureSchema


WindowIndex = tuple[int, int]


@dataclass(frozen=True)
class MotionVAEDataBundle:
    train_dataset: "ReferenceMotionVAEDataset"
    val_dataset: "ReferenceMotionVAEDataset"
    schema: MotionFeatureSchema
    train_motion_names: tuple[str, ...]
    val_motion_names: tuple[str, ...]

    @property
    def train_window_count(self) -> int:
        return len(self.train_dataset)

    @property
    def val_window_count(self) -> int:
        return len(self.val_dataset)


def build_valid_window_centers(sequence_length: int, past_frames: int, future_frames: int) -> list[int]:
    start = int(past_frames) - 1
    end = int(sequence_length) - int(future_frames)
    if end < start:
        return []
    return list(range(start, end + 1))


class ReferenceMotionVAEDataset(Dataset):
    def __init__(
        self,
        sequences: tuple[MotionFeatureSequence, ...],
        window_indices: tuple[WindowIndex, ...],
        *,
        past_frames: int,
        future_frames: int,
        reference_mean: torch.Tensor,
        reference_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        self.sequences = sequences
        self.window_indices = window_indices
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.reference_mean = reference_mean.to(dtype=torch.float32)
        self.reference_std = reference_std.to(dtype=torch.float32)
        self.target_mean = target_mean.to(dtype=torch.float32)
        self.target_std = target_std.to(dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, center_t = self.window_indices[index]
        sequence = self.sequences[sequence_index]
        past = sequence.reference_features[center_t - self.past_frames + 1 : center_t + 1]
        future = sequence.target_features[center_t : center_t + self.future_frames]
        normalized_past = (past - self.reference_mean) / self.reference_std
        normalized_future = (future - self.target_mean) / self.target_std
        return {
            "reference": normalized_past,
            "target": normalized_future,
            "motion_name": sequence.motion_name,
            "motion_file": sequence.motion_file,
            "center_t": center_t,
        }


def _all_window_indices(
    sequences: tuple[MotionFeatureSequence, ...],
    *,
    past_frames: int,
    future_frames: int,
) -> tuple[WindowIndex, ...]:
    window_indices: list[WindowIndex] = []
    for sequence_index, sequence in enumerate(sequences):
        for center_t in build_valid_window_centers(sequence.length, past_frames, future_frames):
            window_indices.append((sequence_index, center_t))
    return tuple(window_indices)


def _split_sequence_indices(sequence_count: int, *, val_ratio: float, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if sequence_count < 2:
        raise ValueError("split_mode='by_motion' requires at least 2 motion files.")
    ordered_indices = list(range(sequence_count))
    random.Random(seed).shuffle(ordered_indices)
    val_count = max(1, int(sequence_count * val_ratio))
    val_count = min(val_count, sequence_count - 1)
    val_indices = tuple(sorted(ordered_indices[:val_count]))
    train_indices = tuple(sorted(ordered_indices[val_count:]))
    if not train_indices or not val_indices:
        raise ValueError("split_mode='by_motion' produced an empty train or val split.")
    return train_indices, val_indices


def _split_window_indices(
    window_indices: tuple[WindowIndex, ...],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[tuple[WindowIndex, ...], tuple[WindowIndex, ...]]:
    if len(window_indices) < 2:
        raise ValueError("split_mode='by_window' requires at least 2 valid windows.")
    shuffled = list(window_indices)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    val_windows = tuple(shuffled[:val_count])
    train_windows = tuple(shuffled[val_count:])
    if not train_windows or not val_windows:
        raise ValueError("split_mode='by_window' produced an empty train or val split.")
    return train_windows, val_windows


def _window_stats(
    sequences: tuple[MotionFeatureSequence, ...],
    window_indices: tuple[WindowIndex, ...],
    *,
    past_frames: int,
    future_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not window_indices:
        raise ValueError("Cannot compute normalization statistics for an empty window split.")

    d_ref = sequences[0].reference_features.shape[-1]
    d_target = sequences[0].target_features.shape[-1]
    ref_sum = torch.zeros(d_ref, dtype=torch.float64)
    ref_sq_sum = torch.zeros(d_ref, dtype=torch.float64)
    ref_count = 0
    target_sum = torch.zeros(d_target, dtype=torch.float64)
    target_sq_sum = torch.zeros(d_target, dtype=torch.float64)
    target_count = 0

    for sequence_index, center_t in window_indices:
        sequence = sequences[sequence_index]
        past = sequence.reference_features[center_t - past_frames + 1 : center_t + 1].to(dtype=torch.float64)
        future = sequence.target_features[center_t : center_t + future_frames].to(dtype=torch.float64)
        ref_sum += past.sum(dim=0)
        ref_sq_sum += torch.square(past).sum(dim=0)
        ref_count += int(past.shape[0])
        target_sum += future.sum(dim=0)
        target_sq_sum += torch.square(future).sum(dim=0)
        target_count += int(future.shape[0])

    reference_mean = ref_sum / ref_count
    reference_var = ref_sq_sum / ref_count - torch.square(reference_mean)
    reference_std = torch.sqrt(reference_var.clamp_min(1.0e-6))
    target_mean = target_sum / target_count
    target_var = target_sq_sum / target_count - torch.square(target_mean)
    target_std = torch.sqrt(target_var.clamp_min(1.0e-6))
    return (
        reference_mean.to(dtype=torch.float32),
        reference_std.to(dtype=torch.float32),
        target_mean.to(dtype=torch.float32),
        target_std.to(dtype=torch.float32),
    )


def _limit_windows(
    window_indices: tuple[WindowIndex, ...],
    *,
    max_windows: int | None,
) -> tuple[WindowIndex, ...]:
    if max_windows is None or len(window_indices) <= max_windows:
        return window_indices
    return tuple(window_indices[:max_windows])


def build_motion_vae_datasets(
    data_config: MotionVAEDataConfig,
    *,
    feature_config: MotionVAEFeatureConfig,
    slice_weights: dict[str, float] | None = None,
) -> MotionVAEDataBundle:
    feature_bundle: MotionFeatureBundle = build_motion_feature_bundle(
        data_config.motion_files,
        feature_config=feature_config,
        slice_weights=slice_weights,
    )
    sequences = feature_bundle.sequences
    all_windows = _all_window_indices(
        sequences,
        past_frames=data_config.past_frames,
        future_frames=data_config.future_frames,
    )
    if not all_windows:
        raise ValueError("No legal motion windows were found for the requested past/future lengths.")

    if data_config.split_mode == "by_motion":
        train_sequence_indices, val_sequence_indices = _split_sequence_indices(
            len(sequences),
            val_ratio=data_config.val_ratio,
            seed=data_config.seed,
        )
        train_windows = tuple(item for item in all_windows if item[0] in train_sequence_indices)
        val_windows = tuple(item for item in all_windows if item[0] in val_sequence_indices)
        train_motion_names = tuple(sequences[index].motion_name for index in train_sequence_indices)
        val_motion_names = tuple(sequences[index].motion_name for index in val_sequence_indices)
    else:
        train_windows, val_windows = _split_window_indices(
            all_windows,
            val_ratio=data_config.val_ratio,
            seed=data_config.seed,
        )
        train_motion_names = tuple(sorted({sequences[index].motion_name for index, _ in train_windows}))
        val_motion_names = tuple(sorted({sequences[index].motion_name for index, _ in val_windows}))

    train_windows = _limit_windows(train_windows, max_windows=data_config.max_train_windows)
    val_windows = _limit_windows(val_windows, max_windows=data_config.max_val_windows)
    if not train_windows or not val_windows:
        raise ValueError("Train/val split is empty after applying window limits.")

    reference_mean, reference_std, target_mean, target_std = _window_stats(
        sequences,
        train_windows,
        past_frames=data_config.past_frames,
        future_frames=data_config.future_frames,
    )
    schema = feature_bundle.schema.with_normalization(
        reference_mean=reference_mean.tolist(),
        reference_std=reference_std.tolist(),
        target_mean=target_mean.tolist(),
        target_std=target_std.tolist(),
    )
    train_dataset = ReferenceMotionVAEDataset(
        sequences,
        train_windows,
        past_frames=data_config.past_frames,
        future_frames=data_config.future_frames,
        reference_mean=reference_mean,
        reference_std=reference_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = ReferenceMotionVAEDataset(
        sequences,
        val_windows,
        past_frames=data_config.past_frames,
        future_frames=data_config.future_frames,
        reference_mean=reference_mean,
        reference_std=reference_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    return MotionVAEDataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        schema=schema,
        train_motion_names=train_motion_names,
        val_motion_names=val_motion_names,
    )
