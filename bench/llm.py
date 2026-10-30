import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""LLM prompt processing A/B on a stock card: cuBLAS (FP32 accumulate, half
rate) vs the chunked FP16-accumulate kernel via patch.enable().

Speed: prefill tokens/s. Quality: perplexity on a fixed text and top-1
agreement of next-token predictions with the cuBLAS run.
"""

import time
import warnings

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import f16acc as patch  # noqa: E402
import transformers.integrations.sdpa_attention as sdpa  # noqa: E402

# Windows torch has no flash attention, the only SDPA kernel that takes
# enable_gqa; with it transformers lands on the FP32 math fallback. Expanding
# K/V instead lets the memory-efficient kernel run (same for both A/B sides).
sdpa.use_gqa_in_sdpa = lambda *a, **k: False

MODEL = "Qwen/Qwen2.5-1.5B"
TEXT = os.path.join(sys.base_prefix, "LICENSE.txt")  # Python's own license text: same on every install of a version

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
ids = tok(open(TEXT, encoding="utf-8").read(), return_tensors="pt").input_ids[:, :2048].cuda()


@torch.no_grad()
def quality():
    logits = model(ids).logits.float()
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    return torch.exp(nll).item(), logits[0].argmax(-1)


@torch.no_grad()
def prefill_tps(batch, seq, reps=5):
    x = torch.randint(0, tok.vocab_size, (batch, seq), device="cuda")
    model(x)  # warm-up / autotune
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter(); model(x); torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    times.sort()
    return batch * seq / times[len(times) // 2]


rows = {}
for name, on in (("cuBLAS", False), ("chunked", True)):
    patch.enable() if on else patch.disable()
    ppl, top1 = quality()
    rows[name] = {"ppl": ppl, "top1": top1, "tps": [prefill_tps(1, 2048), prefill_tps(4, 2048)]}
patch.disable()

agree = (rows["chunked"]["top1"] == rows["cuBLAS"]["top1"]).float().mean().item()
print(f"{MODEL}, {ids.shape[1]} tokens of {TEXT}, stock GPU: {torch.cuda.get_device_name(0)}")
for name, r in rows.items():
    print(f"  {name:8} perplexity {r['ppl']:.4f}   prefill 1x2048 {r['tps'][0]:8.0f} tok/s   4x2048 {r['tps'][1]:8.0f} tok/s")
c, b = rows["chunked"]["tps"], rows["cuBLAS"]["tps"]
print(f"  speed-up {c[0] / b[0]:.2f}x / {c[1] / b[1]:.2f}x   top-1 agreement {agree:.2%}   linear calls routed: {patch.calls}")
