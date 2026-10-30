"""4096 and 8192 GEMM: f16acc vs Triton FP32 accumulate vs cuBLAS in both modes,
speed and max relative error vs FP64."""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")
import torch  # noqa: E402

import _util  # noqa: E402
from f16acc.kernel import mm  # noqa: E402

if __name__ == "__main__":
    m = torch.backends.cuda.matmul
    m.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(0)
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda").half(); b = torch.randn(n, n, device="cuda").half()
        ref = a.double() @ b.double()
        rel = lambda x: ((x.double() - ref).abs().max() / ref.abs().max()).item()
        rows = [("f16acc", lambda: mm(a, b)), ("triton f32-acc", lambda: mm(a, b, chunked=False))]
        for acc in (False, True):
            def f(acc=acc):
                m.allow_fp16_accumulation = acc
                r = torch.mm(a, b)
                m.allow_fp16_accumulation = False
                return r
            rows.append((f"cuBLAS {'f16' if acc else 'f32'}-acc", f))
        for name, op in rows:
            op()  # autotune outside the timing
            med = _util.timed(op, n)
            clk, pw, cap = _util.sample_under_load(op)
            print(f"n={n} {name:15} {med:6.1f} TFLOPS at {clk:.0f} MHz {pw:.0f} W{' capped' if cap else ''}  err {rel(op()):.1e}")
