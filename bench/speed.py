import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import torch
from adakit import bench
m = torch.backends.cuda.matmul
m.allow_fp16_reduced_precision_reduction = False
sms = torch.cuda.get_device_properties(0).multi_processor_count
for n in (4096, 8192):
    for acc in (False, True):
        m.allow_fp16_accumulation = acc
        op = bench.make_op(torch, n, torch.float16)
        med, _, _ = bench.timed(torch, op, n, warmup=10, iters=30)
        clk, pw, cap = bench.sample_under_load(torch, op)
        print(f"n={n} {'f16-acc' if acc else 'f32-acc'}: {med:6.1f} TFLOPS  {clk:.0f} MHz {pw} W {'capped' if cap else ''}  eff {med/(sms*1024*clk*1e6/1e12):.2f}")
