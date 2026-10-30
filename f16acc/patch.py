"""Route FP16 nn.Linear through the chunked FP16-accumulate GEMM.

    import f16acc; f16acc.enable()

Only F.linear is replaced (every nn.Linear calls it). Inputs that aren't fp16
on CUDA, need gradients, or have fewer than MIN_ROWS rows (token-by-token
decode, which is memory-bound and gains nothing) go to the original. FP16
chunk overflow is handled inside the kernel (the tile is redone with FP32
accumulate), so there is no host sync per call.
"""

import torch
import torch.nn.functional as F

from .kernel import mm

MIN_ROWS = 128
_orig = F.linear
calls = {"chunked": 0, "original": 0}


def linear(x, w, b=None):
    if not (x.is_cuda and x.dtype == w.dtype == torch.float16 and x.numel() // x.shape[-1] >= MIN_ROWS
            and not torch.is_grad_enabled()):
        calls["original"] += 1
        return _orig(x, w, b)
    x2 = x.reshape(-1, x.shape[-1])
    y = mm(x2 if x2.is_contiguous() else x2.contiguous(), w.t())
    if b is not None:
        y += b
    calls["chunked"] += 1
    return y.view(*x.shape[:-1], w.shape[0])


def enable():
    F.linear = linear
    torch.nn.modules.linear.F.linear = linear  # nn.Linear looks it up through this module's F


def disable():
    F.linear = _orig
    torch.nn.modules.linear.F.linear = _orig
