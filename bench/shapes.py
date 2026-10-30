import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Administrator\Downloads\gpu-src\ada-unlock-kit")
import torch
from f16acc import kernel as tri
from adakit import bench
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
M = 2048
for K, N, name in ((1536, 1536, "q/o"), (1536, 256, "k/v"), (1536, 8960, "gate/up"), (8960, 1536, "down"), (1536, 151936, "lm_head")):
    x = torch.randn(M, K, device="cuda").half(); w = torch.randn(N, K, device="cuda").half()
    res = {}
    for label, op in (("cuBLAS", lambda: torch.nn.functional.linear(x, w)), ("chunked w.t()", lambda: tri.mm(x, w.t())),
                      ("chunked contig", (lambda wt=w.t().contiguous(): tri.mm(x, wt)))):
        op()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(3): op()
        s.record()
        for _ in range(20): op()
        e.record(); torch.cuda.synchronize()
        res[label] = 2 * M * K * N * 20 / (s.elapsed_time(e) / 1e3) / 1e12
    print(f"{name:8} K={K:5} N={N:6}  " + "  ".join(f"{k} {v:5.1f}" for k, v in res.items()) + "  TFLOPS")
