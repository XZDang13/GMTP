from __future__ import annotations

from collections.abc import Sequence

import torch

from .schema import FeatureSliceSpec


def _pointwise_reconstruction_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "l1":
        return torch.abs(prediction - target)
    if loss_type == "mse":
        return torch.square(prediction - target)
    raise ValueError(f"Unsupported reconstruction loss '{loss_type}'.")


def compute_motion_mae_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_slices: Sequence[FeatureSliceSpec],
    reconstruction_loss: str = "mse",
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}.")

    pointwise_error = _pointwise_reconstruction_error(
        prediction,
        target,
        loss_type=reconstruction_loss,
    )
    loss_dict: dict[str, torch.Tensor] = {}
    reconstruction_total = prediction.new_tensor(0.0)
    for slice_spec in target_slices:
        slice_error = pointwise_error[..., slice_spec.start : slice_spec.end].mean()
        weighted_error = slice_error * float(slice_spec.weight)
        loss_dict[f"{slice_spec.name}_loss"] = slice_error
        loss_dict[f"{slice_spec.name}_weighted_loss"] = weighted_error
        reconstruction_total = reconstruction_total + weighted_error

    loss_dict["reconstruction_loss"] = reconstruction_total
    loss_dict["loss"] = reconstruction_total
    return loss_dict
