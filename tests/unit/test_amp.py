import pytest
import torch

from gmtp.runtime.amp import AMP_DTYPE, AMP_DTYPE_NAME, normalize_device, resolve_amp_enabled


@pytest.mark.parametrize(
    ("use_amp", "device", "expected"),
    [
        (True, torch.device("cpu"), False),
        (True, "cpu", False),
        (False, torch.device("cpu"), False),
        (False, torch.device("cuda"), False),
        (False, "cuda:0", False),
        (True, torch.device("cuda"), True),
        (True, "cuda:0", True),
    ],
)
def test_resolve_amp_enabled_matches_use_amp_and_device_type(
    use_amp: bool,
    device: torch.device | str,
    expected: bool,
) -> None:
    assert resolve_amp_enabled(use_amp, device) is expected


def test_amp_dtype_is_cuda_float16() -> None:
    assert AMP_DTYPE is torch.float16
    assert AMP_DTYPE_NAME == "float16"


def test_normalize_device_accepts_string() -> None:
    assert normalize_device("cuda:0") == torch.device("cuda:0")
