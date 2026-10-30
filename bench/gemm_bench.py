import sys; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
import torch
from f16acc.kernel import mm
if __name__ == "__main__":
    sys.path.insert(0, r"C:\Users\Administrator\Downloads\gpu-src\ada-unlock-kit")
    from adakit import bench
    m = torch.backends.cuda.matmul
    m.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(0)
    # odd sizes exercise the masks
    a = torch.randn(1000, 777, device="cuda").half(); b = torch.randn(777, 1234, device="cuda").half()
    ref = a.double() @ b.double()
    err = ((mm(a, b, torch.float32).double() - ref).abs().max() / ref.abs().max()).item()
    assert err < 2e-3, f"wrong result: {err}"
    print(f"odd-size check ok (rel err {err:.1e})")
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda").half(); b = torch.randn(n, n, device="cuda").half()
        ref = a.double() @ b.double()
        rel = lambda x: ((x.double() - ref).abs().max() / ref.abs().max()).item()
        rows = [("triton chunked", lambda: mm(a, b)), ("triton f32-acc", lambda: mm(a, b, chunked=False))]
        for acc in (False, True):
            def f(acc=acc):
                m.allow_fp16_accumulation = acc
                r = torch.mm(a, b)
                m.allow_fp16_accumulation = False
                return r
            rows.append((f"cuBLAS {'f16' if acc else 'f32'}-acc", f))
        for name, op in rows:
            op()  # autotune outside the timing
            med, _, _ = bench.timed(torch, op, n, warmup=10, iters=30)
            clk, pw, cap = bench.sample_under_load(torch, op)
            print(f"n={n} {name:15} {med:6.1f} TFLOPS at {clk:.0f} MHz {pw} W{' capped' if cap else ''}  err {rel(op()):.1e}")
