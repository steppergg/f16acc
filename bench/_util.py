"""Shared bench helpers: GEMM timing, clock/power sampling under load, and a
raw mma.sync loop compiled with the NVRTC inside PyTorch (Windows: nvcuda.dll).
"""

import ctypes
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
SMI = next((str(p) for p in (SYSTEM32 / "nvidia-smi.exe", Path("/usr/bin/nvidia-smi")) if p.is_file()), "nvidia-smi")
LOAD_S = 1.5


def timed(op, n, warmup=10, iters=30):
    """Median TFLOPS of op(), an n x n x n GEMM."""
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); op(); e.record()
        torch.cuda.synchronize()
        vals.append(2.0 * n ** 3 / (s.elapsed_time(e) / 1000.0) / 1e12)
    return statistics.median(vals)


def sample_under_load(op):
    """(SM MHz, watts, power-capped) read by nvidia-smi while op() keeps running.
    The driver refreshes these about once a second, so load for LOAD_S first."""
    end = time.perf_counter() + LOAD_S
    while time.perf_counter() < end:
        op(); torch.cuda.synchronize()
    p = torch.cuda.get_device_properties(0)
    proc = subprocess.Popen([SMI, "-i", f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0",
                             "--query-gpu=clocks.sm,power.draw,clocks_event_reasons.sw_power_cap",
                             "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, text=True)
    while proc.poll() is None:
        op(); torch.cuda.synchronize()
    clock, power, cap = proc.stdout.read().strip().splitlines()[0].split(", ")
    return float(clock), float(power), cap == "Active"


def efficiency(tflops, clock_mhz):
    """Fraction of the full dense FP16 tensor rate: SMs x clock x 1024 flops (Ada and Ampere)."""
    return tflops / (torch.cuda.get_device_properties(0).multi_processor_count * 1024 * clock_mhz * 1e6 / 1e12)


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed with error {rc}")


def _ptx(src):
    lib = Path(torch.__file__).parent / "lib"
    nvrtc = sorted(p for p in lib.glob("nvrtc64_*.dll") if ".alt." not in p.name)
    os.add_dll_directory(str(lib))
    nv = ctypes.WinDLL(str(nvrtc[-1]))
    prog = ctypes.c_void_p()
    _check(nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"loop.cu", 0, None, None), "nvrtc create")
    cc = torch.cuda.get_device_capability(0)
    opts = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{cc[0]}{cc[1]}".encode())
    if nv.nvrtcCompileProgram(prog, 1, opts) != 0:
        size = ctypes.c_size_t(); nv.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
        log = ctypes.create_string_buffer(size.value); nv.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("nvrtc: " + log.value.decode(errors="replace"))
    size = ctypes.c_size_t()
    _check(nv.nvrtcGetPTXSize(prog, ctypes.byref(size)), "ptx size")
    ptx = ctypes.create_string_buffer(size.value)
    _check(nv.nvrtcGetPTX(prog, ptx), "ptx")
    return ptx.raw


def mma_loop(src, flops_per_warp_iter, blocks_per_sm=4, warps=8, target_s=0.2):
    """Compile `src` (a kernel `loop(float* out, int iters)` that touches no
    memory in its loop) and time it. Returns (median TFLOPS, op) with op()
    running one launch."""
    ptx = _ptx(src)
    torch.zeros(1, device="cuda")  # make PyTorch's context current
    cu = ctypes.WinDLL(str(SYSTEM32 / "nvcuda.dll"))
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    _check(cu.cuModuleLoadData(ctypes.byref(mod), ptx), "module load")
    _check(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"loop"), "get function")
    cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    blocks = torch.cuda.get_device_properties(0).multi_processor_count * blocks_per_sm
    out = torch.empty(blocks * warps * 32, device="cuda")
    ptr, stream = ctypes.c_uint64(out.data_ptr()), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def launch(iters):
        n = ctypes.c_int(iters)
        args = (ctypes.c_void_p * 2)(ctypes.addressof(ptr), ctypes.addressof(n))
        _check(cu.cuLaunchKernel(fn, blocks, 1, 1, warps * 32, 1, 1, 0, stream, args, None), "launch")

    iters = 256
    while True:  # grow until one launch is long enough to time
        t = time.perf_counter(); launch(iters); torch.cuda.synchronize(); dt = time.perf_counter() - t
        if dt >= target_s / 4 or iters >= 1 << 24:
            break
        iters *= 4
    iters = max(1, int(iters * target_s / dt))
    for _ in range(3):  # let clocks settle
        launch(iters)
    vals = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); launch(iters); e.record(); torch.cuda.synchronize()
        vals.append(blocks * warps * iters * flops_per_warp_iter / (s.elapsed_time(e) / 1000.0) / 1e12)
    keep = (out, mod)  # buffer and module must outlive every launch

    def op(keep=keep):
        launch(iters)
    return statistics.median(vals), op
