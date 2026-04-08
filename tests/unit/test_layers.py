import warnings

import torch

from gmtp.models.layers import AmpSafeRMSNorm


def test_amp_safe_rms_norm_avoids_mismatched_dtype_warning():
    norm = AmpSafeRMSNorm(8)
    x = torch.randn(3, 8, dtype=torch.float16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y = norm(x)

    assert y.dtype == x.dtype
    assert not caught
