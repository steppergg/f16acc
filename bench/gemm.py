"""Chunked FP16-accumulate GEMM: C (fp32) = A (M x K, fp16 row-major) @ B^T
where Bt is N x K fp16 row-major (both operands K-contiguous).

Tensor cores accumulate in FP16 (full rate on stock GeForce); every CHUNK
elements of K the FP16 partial sums are converted and added to FP32 totals,
which bounds the FP16 rounding error to a CHUNK-long sum.

shortcut: M, N multiples of 128 and K of 32 only; pad or add edge tiles before real use.
"""

import ctypes
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Administrator\Downloads\gpu-src\ada-unlock-kit")
import torch  # noqa: E402
from adakit import mma, sysbin  # noqa: E402

BM, BN, BK = 128, 128, 32

SRC = r"""
#define BM 128
#define BN 128
#define BK BKVAL
#define LDS (BK + 8)   // row stride +16 B: the 8 rows an ldmatrix reads land in distinct banks
#define LPT (BM * BK / 8 / 256)   // 16-byte copies per thread per operand per tile

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    unsigned r;
    asm("{.reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t;}" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void h2f(unsigned v, float& x, float& y) {
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(x), "=f"(y) : "r"(v));
}

extern "C" __global__ void __launch_bounds__(256)
gemm(const unsigned short* __restrict__ A, const unsigned short* __restrict__ B, float* __restrict__ C,
     int M, int N, int K) {
    extern __shared__ __align__(16) unsigned short smem[];
    unsigned short* As = smem;                         // [STAGES][BM * LDS]
    unsigned short* Bs = smem + STAGES * BM * LDS;     // [STAGES][BN * LDS]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp >> 2, wn = warp & 3;          // warp tile 64 x 32
    // grouped order: consecutive blocks cover a GROUP-tall patch, so a wave
    // reuses its A and B tiles from L2 instead of re-reading DRAM
    const int npm = M / BM, npn = N / BN, pid = blockIdx.x;
    const int first = (pid / (GROUP * npn)) * GROUP, gsz = min(npm - first, GROUP);
    const int bm = (first + pid % gsz) * BM, bn = ((pid % (GROUP * npn)) / gsz) * BN;
    const unsigned short* Ag = A + (size_t)bm * K;
    const unsigned short* Bg = B + (size_t)bn * K;
    const int KT = K / BK;

    unsigned acc[4][4][2];
    float tot[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) tot[i][j][r] = 0.f;

    // all addresses computed once; per tile only a stage offset and a K step are added
    const unsigned sA = smem_addr(As), sB = smem_addr(Bs);
    const unsigned STAGE_B = BM * LDS * 2;            // bytes per stage (BM == BN)
    unsigned ldA[LPT], ldB[LPT];
    const unsigned short* gA[LPT];
    const unsigned short* gB[LPT];
    #pragma unroll
    for (int i = 0; i < LPT; ++i) {
        int c = tid + i * 256, row = c / (BK / 8), col = (c % (BK / 8)) * 8;
        ldA[i] = sA + (row * LDS + col) * 2; ldB[i] = sB + (row * LDS + col) * 2;
        gA[i] = Ag + (size_t)row * K + col;  gB[i] = Bg + (size_t)row * K + col;
    }
    unsigned frA[4], frB[2];
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) frA[mi] = sA + ((wm * 64 + mi * 16 + (lane & 15)) * LDS + (lane >> 4) * 8) * 2;
    #pragma unroll
    for (int np = 0; np < 2; ++np) frB[np] = sB + ((wn * 32 + np * 16 + (lane & 7) + (lane >> 4) * 8) * LDS + ((lane >> 3) & 1) * 8) * 2;

    auto load = [&](int stage, int k0) {
        #pragma unroll
        for (int i = 0; i < LPT; ++i) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(ldA[i] + stage * STAGE_B), "l"(gA[i] + k0));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(ldB[i] + stage * STAGE_B), "l"(gB[i] + k0));
        }
        asm volatile("cp.async.commit_group;");
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < KT) load(st, st * BK); else asm volatile("cp.async.commit_group;");
    }
    for (int kt = 0; kt < KT; ++kt) {
        asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 2));
        __syncthreads();   // tile kt landed for all threads; slot (kt-1) % STAGES is free
        const int nxt = kt + STAGES - 1;
        if (nxt < KT) load(nxt % STAGES, nxt * BK); else asm volatile("cp.async.commit_group;");
        const int s = kt % STAGES;
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            const int kk = kt * (BK / 16) + ks;   // global k16 step
            if (kk % CK16 == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) { acc[i][j][0] = 0u; acc[i][j][1] = 0u; }
            }
            unsigned a[4][4], b[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "r"(frA[mi] + s * STAGE_B + ks * 32));
            #pragma unroll
            for (int np = 0; np < 2; ++np)
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b[np * 2][0]), "=r"(b[np * 2][1]), "=r"(b[np * 2 + 1][0]), "=r"(b[np * 2 + 1][1])
                             : "r"(frB[np] + s * STAGE_B + ks * 32));
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                                 : "+r"(acc[mi][ni][0]), "+r"(acc[mi][ni][1])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        if ((kk + 1) % CK16 == 0 || kk == KT * (BK / 16) - 1) {
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    float x0, x1, x2, x3;
                    h2f(acc[mi][ni][0], x0, x1); h2f(acc[mi][ni][1], x2, x3);
                    tot[mi][ni][0] += x0; tot[mi][ni][1] += x1; tot[mi][ni][2] += x2; tot[mi][ni][3] += x3;
                }
        }
        }
    }
    const int g = lane >> 2, q = (lane & 3) * 2;
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int r = bm + wm * 64 + mi * 16 + g, c = bn + wn * 32 + ni * 8 + q;
            C[(size_t)r * N + c] = tot[mi][ni][0];       C[(size_t)r * N + c + 1] = tot[mi][ni][1];
            C[(size_t)(r + 8) * N + c] = tot[mi][ni][2]; C[(size_t)(r + 8) * N + c + 1] = tot[mi][ni][3];
        }
}
"""


class Gemm:
    def __init__(self, chunk=32, stages=4, group=8, bk=32):
        assert chunk % 16 == 0 and stages >= 2 and bk in (32, 64)
        props = torch.cuda.get_device_properties(0)
        self.bk = bk
        self.smem = stages * (BM + BN) * (bk + 8) * 2
        mma.SRC = (SRC.replace("CK16", str(chunk // 16)).replace("STAGES", str(stages))
                   .replace("GROUP", str(group)).replace("BKVAL", str(bk)))
        ptx = mma._ptx(torch, (props.major, props.minor))
        torch.zeros(1, device="cuda")
        self.cu = cu = ctypes.WinDLL(str(sysbin.SYSTEM32 / "nvcuda.dll"))
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        mma._check("cuda", cu.cuModuleLoadData(ctypes.byref(self.mod), ptx), "module load")
        mma._check("cuda", cu.cuModuleGetFunction(ctypes.byref(self.fn), self.mod, b"gemm"), "get function")
        mma._check("cuda", cu.cuFuncSetAttribute(self.fn, 8, self.smem), "set dynamic smem")  # MAX_DYNAMIC_SHARED_SIZE_BYTES
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3

    def __call__(self, a, bt, out=None):
        M, K = a.shape
        N = bt.shape[0]
        assert a.dtype == bt.dtype == torch.float16 and a.is_contiguous() and bt.is_contiguous() and bt.shape[1] == K
        assert M % BM == 0 and N % BN == 0 and K % self.bk == 0, "shortcut: tile multiples only"
        out = torch.empty(M, N, device="cuda") if out is None else out
        vals = [ctypes.c_uint64(a.data_ptr()), ctypes.c_uint64(bt.data_ptr()), ctypes.c_uint64(out.data_ptr()),
                ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
        args = (ctypes.c_void_p * 6)(*[ctypes.addressof(v) for v in vals])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        mma._check("cuda", self.cu.cuLaunchKernel(self.fn, (N // BN) * (M // BM), 1, 1, 256, 1, 1, self.smem, stream, args, None), "launch")
        return out


if __name__ == "__main__":
    from adakit import bench
    m = torch.backends.cuda.matmul
    m.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(0)
    # correctness at a small size first: every tile and fragment mapping
    a = torch.randn(256, 512, device="cuda").half(); bt = torch.randn(384, 512, device="cuda").half()
    ref = a.double() @ bt.double().t()
    got = Gemm(32, 2, 8, 64)(a, bt).double()
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 2e-3, "bk64 wrong"
    got = Gemm(32)(a, bt).double()
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    assert err < 2e-3, f"wrong result: rel err {err}"
    print(f"small check ok (rel err {err:.1e})")

    n = 4096
    a = torch.randn(n, n, device="cuda").half(); bt = torch.randn(n, n, device="cuda").half()
    b = bt.t()
    ref = a.double() @ b.double()
    rel = lambda x: ((x.double() - ref).abs().max() / ref.abs().max()).item()
    m.allow_fp16_accumulation = False
    print(f"cuBLAS f32-acc: err {rel(torch.mm(a, b)):.1e}")
    m.allow_fp16_accumulation = True
    print(f"cuBLAS f16-acc: err {rel(torch.mm(a, b)):.1e}")
    m.allow_fp16_accumulation = False
    for chunk, stages, bk in ((32, 4, 32), (128, 4, 32), (32, 2, 64), (32, 3, 64), (64, 3, 64), (128, 3, 64)):
        if stages * (BM + BN) * (bk + 8) * 2 > 99 * 1024:
            continue
        g = Gemm(chunk, stages, 8, bk)
        out = torch.empty(n, n, device="cuda")
        op = lambda: g(a, bt, out)
        med, _, _ = bench.timed(torch, op, n, warmup=10, iters=30)
        clk, pw, cap = bench.sample_under_load(torch, op)
        print(f"chunk {chunk:>3} stages {stages} bk {bk}: {med:6.1f} TFLOPS at {clk:.0f} MHz {pw} W{' capped' if cap else ''}  err {rel(out):.1e}")
    for acc in (False, True):
        m.allow_fp16_accumulation = acc
        op = lambda: torch.mm(a, b)
        med, _, _ = bench.timed(torch, op, n, warmup=10, iters=30)
        print(f"cuBLAS {'f16' if acc else 'f32'}-acc: {med:6.1f} TFLOPS")
