"""f16acc vs cuBLAS (FP32 accumulate) on Qwen2.5-1.5B's linear-layer shapes, 2048 tokens."""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")
import torch  # noqa: E402

from f16acc import mm  # noqa: E402

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
M = 2048


def tflops(op, K, N, reps=20):
    for _ in range(3):
        op()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        op()
    e.record(); torch.cuda.synchronize()
    return 2 * M * K * N * reps / (s.elapsed_time(e) / 1e3) / 1e12


if __name__ == "__main__":
    for K, N, name in ((1536, 1536, "q/o"), (1536, 256, "k/v"), (1536, 8960, "gate/up"),
                       (8960, 1536, "down"), (1536, 151936, "lm_head")):
        x = torch.randn(M, K, device="cuda").half(); w = torch.randn(N, K, device="cuda").half()
        base = tflops(lambda: torch.nn.functional.linear(x, w), K, N)
        ours = tflops(lambda: mm(x, w.t()), K, N)
        print(f"{name:8} K={K:5} N={N:6}  cuBLAS {base:5.1f}  f16acc {ours:5.1f} TFLOPS  {ours / base:.2f}x")
