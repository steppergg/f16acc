"""Raw tensor-core rate on this card: mma.sync m16n8k16 FP16 x FP16 with FP32
accumulate vs FP16 accumulate, as a fraction of the full dense rate.
Stock GeForce: about 0.5 vs 1.0."""
import warnings; warnings.filterwarnings("ignore")
import _util

LOOP = r"""
extern "C" __global__ void loop(float* out, int iters) {
    unsigned a = 0x3c00u * (threadIdx.x & 1u);  // 1.0 or 0.0 halves; keeps sums finite
    ACC_DECL
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            asm volatile(MMA : OPERANDS : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
    }
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) s += SUM;
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
"""
F32 = {"ACC_DECL": "float c[8][4] = {};",
       "MMA": '"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"',
       "OPERANDS": '"+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])',
       "SUM": "c[j][0] + c[j][1] + c[j][2] + c[j][3]"}
F16 = {"ACC_DECL": "unsigned c[8][2] = {};",
       "MMA": '"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"',
       "OPERANDS": '"+r"(c[j][0]), "+r"(c[j][1])',
       "SUM": "(float)(c[j][0] ^ c[j][1])"}


def source(parts):
    s = LOOP
    for k, v in parts.items():
        s = s.replace(k, v)
    return s


if __name__ == "__main__":
    for name, parts in (("FP32 accumulate", F32), ("FP16 accumulate", F16)):
        t, op = _util.mma_loop(source(parts), 8 * 4096)  # 8 chains x one m16n8k16 (4096 flops) per warp
        clk, pw, _ = _util.sample_under_load(op)
        print(f"{name}: {t:6.1f} TFLOPS at {clk:.0f} MHz {pw:.0f} W = {_util.efficiency(t, clk):.3f} of full rate")
