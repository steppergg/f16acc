"""Error of chunked FP16-accumulate GEMM vs FP64, against plain FP16-acc and FP32-acc.
Chunks: K split into pieces of `c`; each piece is an FP16-accumulate GEMM (fp16 out),
pieces summed in FP32."""
import warnings; warnings.filterwarnings("ignore")
import torch
m = torch.backends.cuda.matmul
assert hasattr(m, "allow_fp16_accumulation"), "torch lacks allow_fp16_accumulation"
m.allow_fp16_reduced_precision_reduction = False
torch.manual_seed(0)

def f32acc(a, b):
    m.allow_fp16_accumulation = False
    return torch.mm(a, b).double()

def f16acc(a, b):
    m.allow_fp16_accumulation = True
    r = torch.mm(a, b).double(); m.allow_fp16_accumulation = False
    return r

def chunked(a, b, c):
    m.allow_fp16_accumulation = True
    M, K = a.shape; N = b.shape[1]
    ac = a.view(M, K // c, c).transpose(0, 1)          # [K/c, M, c]
    bc = b.view(K // c, c, N)                          # [K/c, c, N]
    r = torch.bmm(ac.contiguous(), bc).float().sum(0).double()
    m.allow_fp16_accumulation = False
    return r

def rel(x, ref):
    return ((x - ref).abs().max() / ref.abs().max()).item()

M = N = 1024
for dist, gen in (("randn", lambda s: torch.randn(*s)), ("relu-acts x randn/sqrtK", None)):
    for K in (1024, 4096, 16384):
        if gen:
            a = gen((M, K)).half().cuda(); b = gen((K, N)).half().cuda()
        else:  # LLM-like: positive activations, small weights; sums don't cancel
            a = torch.randn(M, K).relu().half().cuda(); b = (torch.randn(K, N) / K ** 0.5).half().cuda()
        ref = a.double() @ b.double()
        row = [f"f32acc {rel(f32acc(a,b),ref):.1e}", f"f16acc {rel(f16acc(a,b),ref):.1e}"]
        row += [f"c{c} {rel(chunked(a,b,c),ref):.1e}" for c in (16, 32, 64, 128)]
        print(f"{dist:24} K={K:5}  " + "  ".join(row))
