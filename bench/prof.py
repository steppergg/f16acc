import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import warnings; warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM
import f16acc as patch
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float16).cuda().eval()
print("attn:", m.config._attn_implementation)
x = torch.randint(0, 150000, (1, 2048), device="cuda")
for on in (False, True):
    patch.enable() if on else patch.disable()
    with torch.no_grad():
        m(x); torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            m(x); torch.cuda.synchronize()
    ev = p.key_averages()
    tot = sum(e.device_time_total for e in ev if e.device_time_total and not e.key.startswith(("aten::", "cudaLaunch")))
    print("patched" if on else "cuBLAS", f"GPU total {tot/1e3:.1f} ms")
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:6]:
        print(f"   {e.self_device_time_total/1e3:7.2f} ms  {e.key[:90]}")
