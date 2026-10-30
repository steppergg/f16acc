"""Chunked FP16-accumulate GEMM in Triton: C = A @ B, fp16 in, fp32 accumulate
of FP16-accumulated K chunks. Each BLOCK_K tile is one tl.dot with an FP16
accumulator (full tensor rate on stock GeForce), promoted to FP32 right after,
so the FP16 rounding error is bounded by a BLOCK_K-long sum.

Overflow: a chunk sum above 65504 leaves inf/NaN in its tile. The main kernel
only flags such tiles; a second launch redoes flagged tiles with FP32
accumulate and exits at once for the rest. No host sync, and the main loop
stays lean (doing the redo inside the main kernel cost about 8%).
"""

import torch
import triton
import triton.language as tl

CONFIGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk, "G": 8}, num_stages=s, num_warps=w)
           # BK is the FP16 chunk length, so it stays 32: the accuracy depends on it
           for bm, bn, bk, s, w in [(128, 128, 32, 4, 4), (128, 128, 32, 4, 8), (128, 256, 32, 3, 8),
                                    (256, 128, 32, 3, 8), (64, 128, 32, 4, 4), (128, 64, 32, 4, 4)]]


@triton.jit
def _tile(M, N, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    first = (pid // (G * npn)) * G
    gsz = min(npm - first, G)
    pm = first + (pid % (G * npn)) % gsz
    pn = (pid % (G * npn)) // gsz
    return pid, pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)


@triton.jit
def _loop(a_ptr, b_ptr, M, N, K, sam, sak, sbk, sbn, rm, rn,
          CHUNKED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        x = tl.load(a, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        y = tl.load(b, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        if CHUNKED:
            acc += tl.dot(x, y, out_dtype=tl.float16).to(tl.float32)
        else:  # plain FP32 accumulate (half rate on stock GeForce)
            acc = tl.dot(x, y, acc)
        a += BK * sak
        b += BK * sbk
    return acc


@triton.jit
def _store(c_ptr, acc, rm, rn, M, N, scm, scn):
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.autotune(configs=CONFIGS, key=["N", "K", "CHUNKED"])  # per layer shape; M (tokens) varies per call
@triton.jit
def _gemm(a_ptr, b_ptr, c_ptr, flag_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
          CHUNKED: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid, rm, rn = _tile(M, N, BM, BN, G)
    acc = _loop(a_ptr, b_ptr, M, N, K, sam, sak, sbk, sbn, rm, rn, CHUNKED, BM, BN, BK)
    if CHUNKED:  # abs(NaN) < inf is false, so NaN is flagged too
        tl.store(flag_ptr + pid, tl.max(tl.where(tl.abs(acc) < float("inf"), 0, 1)).to(tl.int8))
    _store(c_ptr, acc, rm, rn, M, N, scm, scn)


@triton.jit
def _redo(a_ptr, b_ptr, c_ptr, flag_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid, rm, rn = _tile(M, N, BM, BN, G)
    if tl.load(flag_ptr + pid) != 0:
        acc = _loop(a_ptr, b_ptr, M, N, K, sam, sak, sbk, sbn, rm, rn, False, BM, BN, BK)
        _store(c_ptr, acc, rm, rn, M, N, scm, scn)


def mm(a, b, out_dtype=torch.float16, chunked=True):
    assert a.dtype == b.dtype == torch.float16 and a.shape[1] == b.shape[0]
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=out_dtype)
    # sized for the smallest tile (64 x 64) so any autotuned config fits
    flags = torch.empty(triton.cdiv(M, 64) * triton.cdiv(N, 64), device=a.device, dtype=torch.int8)
    args = (a, b, c, flags, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    _gemm[lambda m: (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)](*args, CHUNKED=chunked)
    if chunked:
        cfg = _gemm.best_config
        k = cfg.kwargs
        _redo[(triton.cdiv(M, k["BM"]) * triton.cdiv(N, k["BN"]),)](
            *args, BM=k["BM"], BN=k["BN"], BK=k["BK"], G=k["G"], num_warps=cfg.num_warps)
    return c
