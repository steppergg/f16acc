"""mma.sync m16n8k16 throughput: FP32 accumulate vs FP16 accumulate."""
import sys, time, warnings, ctypes
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import torch
from adakit import mma, bench

F16ACC = mma.SRC.replace("float c[8][4] = {};", "unsigned c[8][2] = {};").replace(
    '"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "\n                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n                         : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])',
    '"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "\n                         "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"\n                         : "+r"(c[j][0]), "+r"(c[j][1])').replace(
    "s += c[j][0] + c[j][1] + c[j][2] + c[j][3];", "s += (float)(c[j][0] ^ c[j][1]);")
assert F16ACC != mma.SRC and "f16.f16.f16.f16" in F16ACC
sms = torch.cuda.get_device_properties(0).multi_processor_count
for name, src in (("f32-acc", mma.SRC), ("f16-acc", F16ACC)):
    mma.SRC = src
    t, op = mma.measure(torch)
    clk, pw, cap = bench.sample_under_load(torch, op)
    print(f"{name}: {t:6.1f} TFLOPS at {clk:.0f} MHz {pw} W  = {t/(sms*1024*clk*1e6/1e12):.3f} of 1024/SM/clk")
