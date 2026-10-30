"""Compute ceiling of chunked FP16 accumulate: every `steps` mma k16 steps the
FP16 accumulators are converted to FP32, added to FP32 totals and zeroed.
steps=2 is a 32-long chunk (what f16acc uses), 4 is 64, 8 is 128."""
import warnings; warnings.filterwarnings("ignore")
import _util

TPL = r"""
__device__ __forceinline__ void h2f(unsigned v, float& x, float& y) {
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(x), "=f"(y) : "r"(v));
}
extern "C" __global__ void loop(float* out, int iters) {
    unsigned a = 0x3c00u * (threadIdx.x & 1u);
    unsigned c[8][2];
    float t[8][4] = {};
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) { c[j][0] = 0u; c[j][1] = 0u; }
        #pragma unroll
        for (int s = 0; s < STEPS; ++s) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                             : "+r"(c[j][0]), "+r"(c[j][1]) : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float x0, x1, x2, x3;
            h2f(c[j][0], x0, x1); h2f(c[j][1], x2, x3);
            t[j][0] += x0; t[j][1] += x1; t[j][2] += x2; t[j][3] += x3;
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += t[j][0] + t[j][1] + t[j][2] + t[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""

if __name__ == "__main__":
    for steps in (1, 2, 4, 8):
        t, op = _util.mma_loop(TPL.replace("STEPS", str(steps)), steps * 8 * 4096)
        clk, pw, _ = _util.sample_under_load(op)
        print(f"chunk {16 * steps:>3}: {t:6.1f} TFLOPS at {clk:.0f} MHz {pw:.0f} W = {_util.efficiency(t, clk):.3f} of full rate")
