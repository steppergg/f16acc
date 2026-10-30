# f16acc

Faster FP16 matrix multiplies on **stock** GeForce RTX cards, with near-FP32 accuracy. No firmware unlock, no driver changes: one Triton kernel and a one-line switch for PyTorch.

```python
import f16acc
f16acc.enable()        # every FP16 nn.Linear (inference) now uses the chunked kernel
f16acc.disable()
y = f16acc.mm(a, b)    # or call the GEMM directly (fp16 in, fp16 or fp32 out)
```

Load the model in FP16 (`.half()` or `dtype=torch.float16`). Under `torch.autocast`, `nn.Linear` receives FP32 tensors, so every call stays on cuBLAS. `f16acc.calls` shows how many calls were routed.

## Why it works

GeForce cards run FP16 tensor math at **half rate when accumulating in FP32**, which is what cuBLAS and PyTorch use. **With FP16 accumulation they run at full rate**, but long FP16 sums lose accuracy and overflow above 65504.

f16acc accumulates each 32-long chunk of K in FP16 (full rate), then adds the chunks in FP32. Rounding error stays close to FP32 accumulation, because no FP16 sum is longer than 32 terms.

If a chunk overflows, the tile is redone with FP32 accumulation by a second, nearly free launch. The result never depends on operand magnitude, and there's no host sync.

## Results (stock RTX 4060 Ti 8 GB, Windows 11, torch 2.11 + cu128, triton-windows 3.9)

One run of `bench/gemm_bench.py`. Numbers move about ±5% between runs as the card heats up.

| GEMM 4096³ | TFLOPS | max rel. error vs FP64 |
|---|---|---|
| cuBLAS, FP32 accumulate (PyTorch default) | 43.2 | 3.7e-4 |
| **f16acc** | **65.7** | **5.1e-4** |
| cuBLAS, FP16 accumulate (`allow_fp16_accumulation`) | 77.8 | 8.5e-3 |

- **Raw tensor rate** (`bench/rate.py`): `mma.sync` reaches 0.47–0.50 of full rate with FP32 accumulate and 0.94–0.99 with FP16 accumulate.
  - `bench/chunkrate.py` shows the cost of converting each 32-long chunk to FP32: a ceiling of about 0.77–0.80.
- **Per layer shape** (`bench/shapes.py`, Qwen2.5-1.5B layers, 2048 tokens): 1.18–1.48× faster than cuBLAS. The narrow 256-wide k/v projection gains least.
- **Qwen2.5-1.5B FP16, 2048-token prompt processing** (`bench/llm.py`): 1.22–1.24× faster. Perplexity 2.1179 → 2.1175, and 99.9% of top-1 predictions match.
- **Token-by-token generation:** no change. It's limited by memory bandwidth, and inputs under 128 rows stay on cuBLAS.

On a card whose FP32-accumulate rate has been unlocked, plain cuBLAS is faster; use f16acc on stock cards.

**Windows note:** `transformers` sends grouped-query models to PyTorch's FP32 math attention on Windows wheels before torch 2.14, because there's no flash attention. That costs 2.5× in prompt processing on its own. See huggingface/transformers#49564. `bench/llm.py` works around it the same way for both sides of the A/B.

## Limits

- Inference only: `enable()` skips anything that needs gradients.
- FP16 only: no BF16, and no autocast (see above).
- Measured on one Ada card. Ampere should behave the same (the same half-rate rule) but is untested.
- First use of each layer shape autotunes for a few seconds.
- Call it from one thread. The overflow pass reads Triton's last-chosen config.

## Security notes

- **The package** (`f16acc/`) only launches GPU kernels. No network access, no file writes, no subprocesses.
- **Input checks:** `mm()` checks shapes, dtypes and devices with real errors, not asserts, so `python -O` can't remove them. 64-bit indexing turns on automatically when an offset can pass 2³¹.
- **The benches** start `nvidia-smi` by full path only, never by bare name. They load `nvcuda.dll` from System32 and NVRTC from PyTorch's own folder.
- **Dependencies:** on Windows, Triton comes from [`triton-windows`](https://github.com/woct0rdho/triton-windows), a community build that isn't published by OpenAI. Install it from PyPI and review it as you would any compiler toolchain.

## Prior work

- The same idea in C#/.NET, with per-shape routing against cuBLAS and in-kernel overflow recovery: [ooples/AiDotNet.Tensors#1565](https://github.com/ooples/AiDotNet.Tensors/pull/1565) (merged 2026-10-27), discussed in [#1549](https://github.com/ooples/AiDotNet.Tensors/issues/1549).
- FP16-accumulate Triton attention kernels: [Disty0/sdnq](https://github.com/Disty0/sdnq).
- Promoting low-precision tensor-core accumulators to FP32 at fixed intervals: DeepSeek-V3 (FP8 on Hopper).

f16acc is a PyTorch/Triton take on it: a drop-in for `nn.Linear`, measured on a real LLM.

## Tests and benches

```
python -m unittest discover -s tests -t .
python bench/gemm_bench.py      # also: rate.py, chunkrate.py, shapes.py, llm.py (needs transformers)
```

The tests cover odd sizes, transposed weights, accuracy against FP64 and both cuBLAS modes, routing and fallbacks, zero-width layers, and a forced FP16 overflow that must come out exact.

`rate.py` and `chunkrate.py` compile their loops with the NVRTC that ships inside PyTorch, and load `nvcuda.dll`, so they're Windows-only.
