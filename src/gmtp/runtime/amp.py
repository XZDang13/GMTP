from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

AMP_DTYPE = torch.float16
AMP_DTYPE_NAME = "float16"


def normalize_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def resolve_amp_enabled(use_amp: bool, device: torch.device | str) -> bool:
    return bool(use_amp) and normalize_device(device).type == "cuda"


def autocast_context(device: torch.device | str, enabled: bool) -> ContextManager[None]:
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=normalize_device(device).type, dtype=AMP_DTYPE)


def build_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device="cuda", enabled=enabled)
