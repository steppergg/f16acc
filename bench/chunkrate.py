"""Compute ceiling of chunked FP16 accumulate: every `steps` mma k16 steps, the
FP16 accumulators are converted to FP32, added to FP32 totals and zeroed.
steps=2 -> chunk 32, 4 -> 64, 8 -> 128."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import torch
from adakit import mma, bench

TPL = r"""
__device__ __forceinline__ void h2f(unsigned v, float& x, float& y) {
    asm("{.reg .f16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h;}" : "=f"(x), "=f"(y) : "r"(v));
}
extern "C" __global__ void mma_loop(float* out, int iters) {
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
sms = torch.cuda.get_device_properties(0).multi_processor_count
base = mma.flops
for steps in (1, 2, 4, 8):
    mma.SRC = TPL.replace("STEPS", str(steps))
    mma.flops = lambda blocks, iters, s=steps: base(blocks, iters) * s  # one iter = `steps` mma per chain
    t, op = mma.measure(torch)
    clk, pw, _ = bench.sample_under_load(torch, op)
    print(f"chunk {16*steps:>3} (steps={steps}): {t:6.1f} TFLOPS at {clk:.0f} MHz {pw} W = {t/(sms*1024*clk*1e6/1e12):.3f}")
