# f16acc

Faster FP16 matrix multiplies on **stock** GeForce RTX cards, with near-FP32 accuracy. No firmware unlock, no driver changes: one Triton kernel and a one-line switch for PyTorch.

```python
import f16acc
f16acc.enable()        # every FP16 nn.Linear (inference) now uses the chunked kernel
f16acc.disable()
y = f16acc.mm(a, b)    # or call the GEMM directly (fp16 in, fp16 or fp32 out)
```

## Why it works

GeForce cards run FP16 tensor math at **half rate when accumulating in FP32**, which is what cuBLAS and PyTorch use. **With FP16 accumulation they run at full rate**, but long FP16 sums lose accuracy and overflow above 65504.

f16acc accumulates each 32-long chunk of K in FP16 (full rate), then adds the chunks in FP32. Rounding error stays close to FP32 accumulation, because no FP16 sum is longer than 32 terms.

If a chunk overflows, the tile is redone with FP32 accumulation by a second, nearly free launch. The result never depends on operand magnitude, and there's no host sync.

## Results (stock RTX 4060 Ti 8 GB, Windows 11, torch 2.11 + cu128, triton-windows 3.9)

| GEMM 4096³ | TFLOPS | max rel. error vs FP64 |
|---|---|---|
| cuBLAS, FP32 accumulate (PyTorch default) | 46.7 | 3.6e-4 |
| **f16acc** | **65.0** | **4.8e-4** |
| cuBLAS, FP16 accumulate (`allow_fp16_accumulation`) | 77.9 | 8.8e-3 |

- **Raw tensor rate** (`bench/rate.py`): `mma.sync` with FP32 accumulate reaches 0.50 of full rate, with FP16 accumulate 0.99.
- **Qwen2.5-1.5B FP16, 2048-token prompt processing** (`bench/llm.py`): 1.22–1.24× faster. Perplexity 2.1179 → 2.1175, and 99.9% of top-1 predictions match.
- **Per layer shape** (`bench/shapes.py`): 1.2–1.5× faster than cuBLAS.
- **Token-by-token generation:** no change. It's limited by memory bandwidth, and inputs under 128 rows stay on cuBLAS.

On a card whose FP32-accumulate rate has been unlocked, plain cuBLAS is faster; use f16acc on stock cards.

**Windows note:** `transformers` sends grouped-query models to PyTorch's FP32 math attention on Windows wheels before torch 2.14, because there's no flash attention. That costs 2.5× in prompt processing on its own. See huggingface/transformers#49564. `bench/llm.py` works around it the same way for both sides of the A/B.

## Limits

- Inference only: `enable()` skips anything that needs gradients.
- FP16 only, not BF16.
- Measured on one Ada card. Ampere should behave the same (the same half-rate rule) but is untested.
- First use of each layer shape autotunes for a few seconds.

## Prior work

- The same idea in C#/.NET, with per-shape routing against cuBLAS and in-kernel overflow recovery: [ooples/AiDotNet.Tensors#1565](https://github.com/ooples/AiDotNet.Tensors/pull/1565) (merged 2026-10-27), discussed in [#1549](https://github.com/ooples/AiDotNet.Tensors/issues/1549).
- FP16-accumulate Triton attention kernels: [Disty0/sdnq](https://github.com/Disty0/sdnq).
- Promoting low-precision tensor-core accumulators to FP32 at fixed intervals: DeepSeek-V3 (FP8 on Hopper).

f16acc is a PyTorch/Triton take on it: a drop-in for `nn.Linear`, measured on a real LLM.

## Tests

```
python -m unittest discover -s tests -t .
```

The tests cover odd sizes, transposed weights, accuracy against FP64 and both cuBLAS modes, routing and fallbacks, and a forced FP16 overflow that must come out exact.

`bench/gemm.py` is the hand-written NVRTC + `mma.sync` version (52 TFLOPS), kept as a reference.
