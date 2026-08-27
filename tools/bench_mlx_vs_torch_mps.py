#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MLX vs PyTorch-MPS engineering benchmark for OUR critical-path ops.

Tag: ``[macOS-MLX/MPS engineering benchmark]`` — ADVISORY engineering
speed/correctness measurement. Benchmarking MPS *as a compute substrate* is fine
here; it does NOT make MPS a score authority. The contest score authority is
unchanged (numpy-fp32 CPU / CUDA; MPS is gradient-only-never-score). Do not
conflate a faster kernel with a better score.

What this measures
------------------
For each op on the witness-training critical path (and a generic-primitive
landscape baseline), we run BOTH an MLX implementation and a PyTorch-MPS
implementation on the SAME seeded inputs, assert numerical parity FIRST, then
time each with warmup + N timed trials (median + IQR), and record peak unified
memory. A faster *wrong* kernel is not a win — parity gates the timing.

GPU exclusivity
---------------
A live witness training arm (e.g. ``thetastar_muon`` / ``train_levelset_*``)
owns the single Metal GPU under the one-GPU rule. ``--gpu-sweep`` REFUSES to run
if such an arm is alive (it would contaminate both the numbers and the training).
Use ``--device cpu`` for a non-contending dry run any time. ``--force-gpu-unsafe``
overrides the gate (only when you KNOW the GPU is free).

Usage
-----
    # Non-contending CPU dry run (correctness + relative shape, always safe):
    .venv/bin/python tools/bench_mlx_vs_torch_mps.py --device cpu --quick

    # Full GPU sweep (only when no training arm is alive):
    .venv/bin/python tools/bench_mlx_vs_torch_mps.py --gpu-sweep \
        --out .omx/research/mlx_vs_torch_mps_bench_$(date -u +%Y%m%dT%H%M%SZ).json

    # Filter to one op / one category:
    .venv/bin/python tools/bench_mlx_vs_torch_mps.py --device cpu --op matmul
    .venv/bin/python tools/bench_mlx_vs_torch_mps.py --gpu-sweep --category critical_path

Resumability: if ``--out`` already exists, completed (op, variant, size) cells
are loaded and skipped unless ``--no-resume``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "upstream", REPO_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

BENCH_TAG = "[macOS-MLX/MPS engineering benchmark]"

# --------------------------------------------------------------------------- #
# GPU exclusivity gate
# --------------------------------------------------------------------------- #

_TRAINING_ARM_PATTERNS = (
    "thetastar",
    "train_levelset_witness",
    "train_levelset",
    "train_witness",
)


def training_arm_alive() -> tuple[bool, str]:
    """Return (alive, matched_line) by scanning the process table.

    A live training arm owns the single Metal GPU; a GPU sweep alongside it would
    contaminate both. This is the gate that keeps the benchmark from colliding.
    """

    try:
        out = subprocess.run(  # subprocess-no-check-OK: best-effort ps collision census; failure degrades via the except arm
            ["ps", "-axww", "-o", "pid,command"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception as exc:  # pragma: no cover - ps should always exist on macOS
        return False, f"ps_failed:{exc}"
    self_pid = str(os.getpid())
    for line in out.splitlines():
        low = line.lower()
        if "bench_mlx_vs_torch_mps" in low:
            continue  # don't match ourselves
        if line.strip().split(" ", 1)[0] == self_pid:
            continue
        if any(pat in low for pat in _TRAINING_ARM_PATTERNS):
            # ignore the grep/ps invocation itself
            if "grep" in low or " ps " in low:
                continue
            return True, line.strip()
    return False, ""


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    op: str
    category: str
    variant: str  # e.g. "mlx", "torch", "mlx_custom", "mlx_ref"
    framework: str  # "mlx" | "torch"
    size_id: str
    shape: str
    dtype: str
    device: str
    trials: int
    median_ms: float
    iqr_ms: float
    mean_ms: float
    min_ms: float
    peak_mem_mb: float
    parity_ok: bool | None
    parity_max_abs: float | None
    parity_max_rel: float | None
    parity_ref: str | None
    notes: str = ""


@dataclass
class Candidate:
    """One timed implementation of an op for a given size."""

    label: str
    framework: str  # "mlx" | "torch"
    fn: Callable[[], Any]  # closure that runs the op and returns the device array
    to_numpy: Callable[[Any], np.ndarray]


@dataclass
class OpSize:
    size_id: str
    shape_str: str
    candidates: list[Candidate]
    parity_ref: str | None  # label whose numpy output is the parity reference
    parity_tol: float  # absolute tolerance for allclose


@dataclass
class OpSpec:
    name: str
    category: str
    build: Callable[[str, np.random.Generator, str], list[OpSize]]
    # build(device, rng, dtype) -> list[OpSize]


# --------------------------------------------------------------------------- #
# Timing core
# --------------------------------------------------------------------------- #


def _sync(framework: str) -> None:
    if framework == "mlx":
        import mlx.core as mx

        mx.synchronize()
    else:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def time_candidate(
    cand: Candidate,
    *,
    warmup: int,
    trials: int,
    device: str,
) -> tuple[list[float], float, Any]:
    """Warm up, then time `trials` synchronized runs. Returns (timings_ms, peak_mb, last_out)."""

    import mlx.core as mx  # noqa: F401  (import here so CPU-only dry runs still load)

    # Warmup
    out = None
    for _ in range(warmup):
        out = cand.fn()
        if cand.framework == "mlx":
            mx.eval(out)
        _sync(cand.framework)

    # Peak memory reset
    peak_mb = 0.0
    if cand.framework == "mlx" and device != "cpu":
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

    timings_ms: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = cand.fn()
        if cand.framework == "mlx":
            mx.eval(out)
        _sync(cand.framework)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1e3)

    if device != "cpu":
        try:
            if cand.framework == "mlx":
                peak_mb = float(mx.get_peak_memory()) / 1e6
            else:
                import torch

                peak_mb = float(torch.mps.driver_allocated_memory()) / 1e6
        except Exception:
            peak_mb = 0.0

    return timings_ms, peak_mb, out


def summarize(timings_ms: list[float]) -> dict[str, float]:
    s = sorted(timings_ms)
    n = len(s)
    median = statistics.median(s)
    q1 = s[max(0, int(0.25 * (n - 1)))]
    q3 = s[min(n - 1, int(0.75 * (n - 1)))]
    return {
        "median_ms": median,
        "iqr_ms": q3 - q1,
        "mean_ms": statistics.fmean(s),
        "min_ms": s[0],
    }


def parity_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf"), float("inf")
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < 1e-12, 1.0, denom)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# Helpers to move numpy <-> framework arrays
# --------------------------------------------------------------------------- #


def mlx_from_np(x: np.ndarray) -> Any:
    import mlx.core as mx

    return mx.array(x)


def mlx_to_np(x: Any) -> np.ndarray:
    import mlx.core as mx

    mx.eval(x)
    return np.array(x, copy=False)


def torch_from_np(x: np.ndarray, device: str) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


def torch_to_np(x: Any) -> np.ndarray:
    return x.detach().to("cpu").numpy()


def torch_device(device: str) -> str:
    return "mps" if device == "gpu" else "cpu"


# --------------------------------------------------------------------------- #
# Generic primitive ops
# --------------------------------------------------------------------------- #


def _build_matmul(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    sizes = [(128, 128, 128), (512, 512, 512), (2048, 2048, 2048), (4096, 4096, 1024)]
    out: list[OpSize] = []
    for (m, k, n) in sizes:
        a = rng.standard_normal((m, k)).astype(npdt)
        b = rng.standard_normal((k, n)).astype(npdt)
        a_mx, b_mx = mlx_from_np(a), mlx_from_np(b)
        a_t, b_t = torch_from_np(a, tdev), torch_from_np(b, tdev)

        def mk_mlx(a_mx=a_mx, b_mx=b_mx):
            import mlx.core as mx

            return mx.matmul(a_mx, b_mx)

        def mk_torch(a_t=a_t, b_t=b_t):
            return torch.matmul(a_t, b_t)

        out.append(
            OpSize(
                size_id=f"{m}x{k}x{n}",
                shape_str=f"({m},{k})@({k},{n})",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=2e-2 if dtype == "fp32" else 5e-1,
            )
        )
    return out


def _conv_shapes() -> list[tuple[int, int, int, int, int, int, int]]:
    # (N, Cin, Cout, H, W, kernel, stride)
    return [
        (4, 32, 64, 96, 128, 3, 1),  # mid feature map
        (4, 64, 128, 48, 64, 3, 2),  # strided downsample
        (1, 3, 32, 384, 512, 3, 2),  # scorer stem-ish (full res input)
        (8, 96, 96, 24, 32, 3, 1),   # deeper small map, batch 8
    ]


def _build_conv2d(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for (N, Cin, Cout, H, W, k, s) in _conv_shapes():
        pad = k // 2
        x = rng.standard_normal((N, Cin, H, W)).astype(npdt)  # NCHW for torch
        w = (rng.standard_normal((Cout, Cin, k, k)) * 0.05).astype(npdt)  # OIHW
        # MLX NHWC layout
        x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        w_ohwi = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
        x_mx, w_mx = mlx_from_np(x_nhwc), mlx_from_np(w_ohwi)
        x_t, w_t = torch_from_np(x, tdev), torch_from_np(w, tdev)

        def mk_mlx(x_mx=x_mx, w_mx=w_mx, s=s, pad=pad):
            import mlx.core as mx

            return mx.conv2d(x_mx, w_mx, stride=(s, s), padding=(pad, pad))

        def mk_torch(x_t=x_t, w_t=w_t, s=s, pad=pad):
            return torch.nn.functional.conv2d(x_t, w_t, stride=s, padding=pad)

        # parity: compare on numpy in NHWC (transpose torch back)
        def mlx_np(o):
            return mlx_to_np(o)

        def torch_np(o):
            return o.detach().to("cpu").numpy().transpose(0, 2, 3, 1)

        out.append(
            OpSize(
                size_id=f"N{N}_C{Cin}-{Cout}_{H}x{W}_k{k}s{s}",
                shape_str=f"x(N{N},C{Cin},{H},{W}) w(O{Cout},I{Cin},{k},{k}) s{s}",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_np),
                    Candidate("torch", "torch", mk_torch, torch_np),
                ],
                parity_ref="torch",
                parity_tol=3e-2 if dtype == "fp32" else 1.0,
            )
        )
    return out


def _build_depthwise_fwd(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    # (N, C, H, W, k, s)  groups=C depthwise
    shapes = [(4, 64, 48, 64, 3, 1), (4, 96, 48, 64, 3, 2), (8, 128, 24, 32, 3, 1)]
    for (N, C, H, W, k, s) in shapes:
        pad = k // 2
        x = rng.standard_normal((N, C, H, W)).astype(npdt)
        w = (rng.standard_normal((C, 1, k, k)) * 0.1).astype(npdt)  # depthwise OIHW (I=1)
        x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        w_ohwi = np.ascontiguousarray(w.transpose(0, 2, 3, 1))  # (C,k,k,1)
        x_mx, w_mx = mlx_from_np(x_nhwc), mlx_from_np(w_ohwi)
        x_t, w_t = torch_from_np(x, tdev), torch_from_np(w, tdev)

        def mk_mlx(x_mx=x_mx, w_mx=w_mx, s=s, pad=pad, C=C):
            import mlx.core as mx

            return mx.conv2d(x_mx, w_mx, stride=(s, s), padding=(pad, pad), groups=C)

        def mk_torch(x_t=x_t, w_t=w_t, s=s, pad=pad, C=C):
            return torch.nn.functional.conv2d(x_t, w_t, stride=s, padding=pad, groups=C)

        out.append(
            OpSize(
                size_id=f"N{N}_C{C}_{H}x{W}_k{k}s{s}_dw",
                shape_str=f"depthwise x(N{N},C{C},{H},{W}) k{k} s{s} groups={C}",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, lambda o: o.detach().to("cpu").numpy().transpose(0, 2, 3, 1)),
                ],
                parity_ref="torch",
                parity_tol=3e-2 if dtype == "fp32" else 1.0,
            )
        )
    return out


def _build_groupnorm(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for (N, C, H, W, G) in [(4, 64, 48, 64, 8), (8, 128, 24, 32, 16)]:
        x = rng.standard_normal((N, C, H, W)).astype(npdt)
        x_t = torch_from_np(x, tdev)
        x_mx = mlx_from_np(np.ascontiguousarray(x.transpose(0, 2, 3, 1)))

        def mk_torch(x_t=x_t, G=G, C=C):
            return torch.nn.functional.group_norm(x_t, G)

        def mk_mlx(x_mx=x_mx, G=G, C=C, N=N, H=H, W=W):
            import mlx.core as mx

            # NHWC -> (N,H,W,G,C/G) group over channel
            cg = C // G
            xr = x_mx.reshape(N, H, W, G, cg)
            mean = mx.mean(xr, axis=(1, 2, 4), keepdims=True)
            var = mx.var(xr, axis=(1, 2, 4), keepdims=True)
            xn = (xr - mean) * mx.rsqrt(var + 1e-5)
            return xn.reshape(N, H, W, C)

        def torch_np(o):
            return o.detach().to("cpu").numpy().transpose(0, 2, 3, 1)

        out.append(
            OpSize(
                size_id=f"N{N}_C{C}_{H}x{W}_G{G}",
                shape_str=f"groupnorm x(N{N},C{C},{H},{W}) G{G}",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_np),
                ],
                parity_ref="torch",
                parity_tol=5e-2 if dtype == "fp32" else 1.0,
            )
        )
    return out


def _build_layernorm(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for (B, T, D) in [(8, 256, 512), (4, 1024, 768)]:
        x = rng.standard_normal((B, T, D)).astype(npdt)
        x_t = torch_from_np(x, tdev)
        x_mx = mlx_from_np(x)

        def mk_torch(x_t=x_t, D=D):
            return torch.nn.functional.layer_norm(x_t, (D,))

        def mk_mlx(x_mx=x_mx, D=D):
            import mlx.core as mx

            mean = mx.mean(x_mx, axis=-1, keepdims=True)
            var = mx.var(x_mx, axis=-1, keepdims=True)
            return (x_mx - mean) * mx.rsqrt(var + 1e-5)

        out.append(
            OpSize(
                size_id=f"B{B}_T{T}_D{D}",
                shape_str=f"layernorm ({B},{T},{D})",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=5e-2 if dtype == "fp32" else 1.0,
            )
        )
    return out


def _build_argmax(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    """The d_seg path: argmax over the 5-class channel dim of (N,H,W,C)."""
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for (N, H, W, C) in [(1, 384, 512, 5), (4, 384, 512, 5), (8, 192, 256, 5)]:
        x = rng.standard_normal((N, H, W, C)).astype(npdt)
        x_t = torch_from_np(x, tdev)
        x_mx = mlx_from_np(x)

        def mk_torch(x_t=x_t):
            return torch.argmax(x_t, dim=-1)

        def mk_mlx(x_mx=x_mx):
            import mlx.core as mx

            return mx.argmax(x_mx, axis=-1)

        out.append(
            OpSize(
                size_id=f"N{N}_{H}x{W}_C{C}",
                shape_str=f"argmax over C of ({N},{H},{W},{C})",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, lambda o: mlx_to_np(o).astype(np.int64)),
                    Candidate("torch", "torch", mk_torch, lambda o: o.detach().to("cpu").numpy().astype(np.int64)),
                ],
                parity_ref="torch",
                parity_tol=0.0,  # integer indices must match exactly
            )
        )
    return out


def _build_gather(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for (rows, dim, n_idx) in [(65536, 256, 16384), (262144, 128, 65536)]:
        table = rng.standard_normal((rows, dim)).astype(npdt)
        idx = rng.integers(0, rows, size=(n_idx,)).astype(np.int64)
        t_mx, i_mx = mlx_from_np(table), mlx_from_np(idx.astype(np.int32))
        t_t, i_t = torch_from_np(table, tdev), torch.from_numpy(idx).to(tdev)

        def mk_mlx(t_mx=t_mx, i_mx=i_mx):
            return t_mx[i_mx]

        def mk_torch(t_t=t_t, i_t=i_t):
            return t_t[i_t]

        out.append(
            OpSize(
                size_id=f"rows{rows}_d{dim}_n{n_idx}",
                shape_str=f"gather table({rows},{dim}) idx({n_idx})",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=1e-3,
            )
        )
    return out


def _build_elementwise(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    out: list[OpSize] = []
    for n in [1 << 20, 1 << 24]:
        x = rng.standard_normal((n,)).astype(npdt)
        x_mx, x_t = mlx_from_np(x), torch_from_np(x, tdev)

        def mk_mlx(x_mx=x_mx):
            import mlx.core as mx

            return mx.maximum(x_mx, 0) + mx.sigmoid(x_mx) * x_mx

        def mk_torch(x_t=x_t):
            return torch.relu(x_t) + torch.sigmoid(x_t) * x_t

        out.append(
            OpSize(
                size_id=f"n{n}",
                shape_str=f"elementwise relu+silu ({n})",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=2e-2 if dtype == "fp32" else 1.0,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# CRITICAL PATH: grouped conv2d BACKWARD — custom Metal vs reference vs torch
# --------------------------------------------------------------------------- #


def _build_grouped_conv_backward(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    """The headline lever: strided grouped/depthwise Conv2d backward.

    MLX native strided-grouped VJP is numerically WRONG, so the repo ships:
      - mlx_custom: @mx.custom_function with two correct Metal kernels (fast)
      - mlx_ref:    python-loop reference forward whose autograd is correct (slow)
    We time grad_input+grad_weight for each, plus torch autograd backward.
    Parity reference is the python-loop reference backward (the trusted authority).
    """

    import torch

    if dtype != "fp32":
        return []  # backward path is fp32 (the witness is fp32)

    from tac.local_acceleration.metal_grouped_conv_backward import (
        make_grouped_conv2d_nhwc,
        metal_grouped_conv2d_backend_available,
    )
    from tac.local_acceleration.mlx_scorer_adapters import mlx_reference_conv2d_nhwc

    import mlx.core as mx

    tdev = torch_device(device)
    out: list[OpSize] = []
    # (N, C, H, W, k, s) depthwise (groups=C), strided — the broken-native case
    shapes = [(4, 96, 48, 64, 3, 2), (8, 96, 48, 64, 3, 2), (4, 128, 96, 128, 3, 2)]
    for (N, C, H, W, k, s) in shapes:
        pad = k // 2
        x = rng.standard_normal((N, C, H, W)).astype(np.float32)
        w = (rng.standard_normal((C, 1, k, k)) * 0.1).astype(np.float32)
        x_nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        w_ohwi = np.ascontiguousarray(w.transpose(0, 2, 3, 1))  # (C,k,k,1)

        x_mx = mlx_from_np(x_nhwc)
        w_mx = mlx_from_np(w_ohwi)

        # ----- mlx custom Metal backward -----
        conv_custom = make_grouped_conv2d_nhwc(stride=(s, s), padding=(pad, pad), groups=C)

        def loss_custom(xx, ww, conv_custom=conv_custom):
            return mx.sum(conv_custom(xx, ww))

        grad_custom = mx.grad(loss_custom, argnums=(0, 1))

        def mk_custom(x_mx=x_mx, w_mx=w_mx, grad_custom=grad_custom):
            gx, gw = grad_custom(x_mx, w_mx)
            return (gx, gw)

        def custom_to_np(o):
            gx, gw = o
            return np.concatenate([mlx_to_np(gx).ravel(), mlx_to_np(gw).ravel()])

        # ----- mlx reference (python-loop) backward -----
        def loss_ref(xx, ww, s=s, pad=pad, C=C):
            return mx.sum(
                mlx_reference_conv2d_nhwc(
                    xx, ww, None, stride=(s, s), padding=(pad, pad), groups=C
                )
            )

        grad_ref = mx.grad(loss_ref, argnums=(0, 1))

        def mk_ref(x_mx=x_mx, w_mx=w_mx, grad_ref=grad_ref):
            gx, gw = grad_ref(x_mx, w_mx)
            return (gx, gw)

        # ----- torch autograd backward -----
        x_t = torch_from_np(x, tdev).requires_grad_(True)
        w_t = torch_from_np(w, tdev).requires_grad_(True)

        def mk_torch(x_t=x_t, w_t=w_t, s=s, pad=pad, C=C):
            if x_t.grad is not None:
                x_t.grad = None
            if w_t.grad is not None:
                w_t.grad = None
            y = torch.nn.functional.conv2d(x_t, w_t, stride=s, padding=pad, groups=C)
            y.sum().backward()
            return (x_t.grad, w_t.grad)

        def torch_to_np_grads(o):
            gx, gw = o
            gx_nhwc = gx.detach().to("cpu").numpy().transpose(0, 2, 3, 1)
            gw_ohwi = gw.detach().to("cpu").numpy().transpose(0, 2, 3, 1)
            return np.concatenate([gx_nhwc.ravel(), gw_ohwi.ravel()])

        cands = [
            Candidate("mlx_ref", "mlx", mk_ref, custom_to_np),
            Candidate("torch", "torch", mk_torch, torch_to_np_grads),
        ]
        # custom Metal backward only valid on GPU
        if device != "cpu" and metal_grouped_conv2d_backend_available():
            cands.insert(0, Candidate("mlx_custom", "mlx", mk_custom, custom_to_np))

        out.append(
            OpSize(
                size_id=f"N{N}_C{C}_{H}x{W}_k{k}s{s}_dw_bwd",
                shape_str=f"grouped/depthwise BACKWARD x(N{N},C{C},{H},{W}) k{k} s{s} groups={C}",
                candidates=cands,
                parity_ref="mlx_ref",  # the trusted python-loop reference backward
                parity_tol=2e-3,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# CRITICAL PATH: coord-INR trunk (real witness shapes) — MLX vs torch
# --------------------------------------------------------------------------- #


def _build_inr_trunk(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    """The witness coord-INR trunk at REAL shapes (P_px=render_h*render_w).

    in_proj Linear(in_feat->96) -> 4x [Linear(96->96) * film_scale + film_shift
    + activation] -> out_sdf Linear(96->5) -> softmax(5) -> palette matmul(5->3)
    -> sigmoid*255. Synthetic weights (the trunk is MLX-only in production; this
    gives an apples-to-apples MLX-vs-torch ratio for the Linear/FiLM/softmax mix
    that dominates the forward).
    """

    import torch

    npdt = np.float32 if dtype == "fp32" else np.float16
    tdev = torch_device(device)
    in_feat, hidden, nclass = 88, 96, 5
    out: list[OpSize] = []
    for (rh, rw) in [(384, 512), (192, 256)]:
        P = rh * rw
        feats = (rng.standard_normal((P, in_feat)) * 0.5).astype(npdt)
        win = (rng.standard_normal((hidden, in_feat)) * 0.1).astype(npdt)
        bin_ = np.zeros((hidden,), npdt)
        wh = [(rng.standard_normal((hidden, hidden)) * 0.1).astype(npdt) for _ in range(4)]
        scale = [(1.0 + rng.standard_normal((hidden,)) * 0.1).astype(npdt) for _ in range(4)]
        shift = [(rng.standard_normal((hidden,)) * 0.1).astype(npdt) for _ in range(4)]
        wsdf = (rng.standard_normal((nclass, hidden)) * 0.1).astype(npdt)
        palette = (rng.standard_normal((nclass, 3)) * 0.5).astype(npdt)

        f_mx = mlx_from_np(feats)
        win_mx, bin_mx = mlx_from_np(win), mlx_from_np(bin_)
        wh_mx = [mlx_from_np(w) for w in wh]
        sc_mx = [mlx_from_np(s) for s in scale]
        sh_mx = [mlx_from_np(s) for s in shift]
        wsdf_mx, pal_mx = mlx_from_np(wsdf), mlx_from_np(palette)

        f_t = torch_from_np(feats, tdev)
        win_t, bin_t = torch_from_np(win, tdev), torch_from_np(bin_, tdev)
        wh_t = [torch_from_np(w, tdev) for w in wh]
        sc_t = [torch_from_np(s, tdev) for s in scale]
        sh_t = [torch_from_np(s, tdev) for s in shift]
        wsdf_t, pal_t = torch_from_np(wsdf, tdev), torch_from_np(palette, tdev)

        def mk_mlx(f=f_mx, win=win_mx, bin_=bin_mx, wh=wh_mx, sc=sc_mx, sh=sh_mx, wsdf=wsdf_mx, pal=pal_mx):
            import mlx.core as mx

            h = f @ win.T + bin_
            for i in range(4):
                h = mx.maximum(h @ wh[i].T, 0) * sc[i] + sh[i]
            phi = h @ wsdf.T
            soft = mx.softmax(phi, axis=-1)
            rgb = mx.sigmoid(soft @ pal) * 255.0
            return rgb

        def mk_torch(f=f_t, win=win_t, bin_=bin_t, wh=wh_t, sc=sc_t, sh=sh_t, wsdf=wsdf_t, pal=pal_t):
            h = f @ win.T + bin_
            for i in range(4):
                h = torch.relu(h @ wh[i].T) * sc[i] + sh[i]
            phi = h @ wsdf.T
            soft = torch.softmax(phi, dim=-1)
            rgb = torch.sigmoid(soft @ pal) * 255.0
            return rgb

        out.append(
            OpSize(
                size_id=f"P{P}_{rh}x{rw}_in{in_feat}_h{hidden}",
                shape_str=f"INR trunk P={P} in_feat={in_feat} hidden={hidden}x4 -> rgb",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=5e-2 if dtype == "fp32" else 2.0,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# CRITICAL PATH: the R operator (render -> bicubic up -> uint8 -> bilinear down)
# --------------------------------------------------------------------------- #


def _build_render_R(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    """The contest-faithful R roundtrip: bicubic up 384->874, uint8, bilinear down.

    MLX production path = apply_contest_faithful_roundtrip_nhwc. torch twin =
    F.interpolate bicubic/bilinear. NOTE: bicubic *coefficients differ* between
    MLX and torch (a=-0.5 vs -0.75) so a small algorithmic delta on the up-step
    is EXPECTED — parity here is informational (the MLX path is the witness's
    production R). Reported delta is recorded; gate tol is generous.
    """

    if dtype != "fp32":
        return []
    import torch

    try:
        from tac.local_acceleration.pr95_hnerv_mlx_training import (
            apply_contest_faithful_roundtrip_nhwc,
        )
    except Exception:
        return []

    out: list[OpSize] = []
    # Fused-R Metal kernel candidate (push-plan P2): only on a Metal device (the
    # custom mx.fast.metal_kernel requires GPU). Gated + lazy so the CPU dry run still
    # loads. The fused forward is bit-faithful to the MLX production R (same tap-sum
    # order); its dedicated per-chip parity gate is
    # tac.local_acceleration.metal_fused_r_operator.assert_metal_matches_cpu_oracle.
    fused_fn = None
    if device != "cpu":
        try:
            from tac.local_acceleration.metal_fused_r_operator import (
                fused_r_roundtrip,
                metal_fused_r_available,
            )

            if metal_fused_r_available():
                fused_fn = fused_r_roundtrip
        except Exception:
            fused_fn = None

    for (N, rh, rw) in [(1, 384, 512), (4, 384, 512)]:
        rgb = (rng.random((N, rh, rw, 3)) * 255.0).astype(np.float32)
        rgb_mx = mlx_from_np(rgb)
        # torch NCHW
        rgb_t = torch_from_np(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)), torch_device(device))

        def mk_mlx(rgb_mx=rgb_mx):
            return apply_contest_faithful_roundtrip_nhwc(
                rgb_mx, camera_hw=(874, 1164), output_hw=(384, 512), ste_round=True
            )

        def mk_torch(rgb_t=rgb_t):
            up = torch.nn.functional.interpolate(
                rgb_t, size=(874, 1164), mode="bicubic", align_corners=False
            )
            up = up.clamp(0, 255).round()
            down = torch.nn.functional.interpolate(
                up, size=(384, 512), mode="bilinear", align_corners=False
            )
            return down

        def torch_np(o):
            return o.detach().to("cpu").numpy().transpose(0, 2, 3, 1)

        candidates = [
            Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
            Candidate("torch", "torch", mk_torch, torch_np),
        ]
        if fused_fn is not None:

            def mk_mlx_fused(rgb_mx=rgb_mx, _fn=fused_fn):
                return _fn(
                    rgb_mx, camera_hw=(874, 1164), output_hw=(384, 512), ste_round=True
                )

            candidates.append(Candidate("mlx_fused", "mlx", mk_mlx_fused, mlx_to_np))

        out.append(
            OpSize(
                size_id=f"N{N}_{rh}x{rw}_R",
                shape_str=f"R roundtrip ({N},{rh},{rw},3) up874 uint8 down384",
                candidates=candidates,
                # parity_ref "mlx": the fused kernel MUST match the MLX production R
                # (the oracle, ~0 delta — bit-faithful). torch differs by the known
                # bicubic-coeff algo note, so its parity row stays informational under
                # the generous tol below.
                parity_ref="mlx",
                parity_tol=12.0,  # informational for torch; fused-vs-mlx is ~0
            )
        )
    return out


# --------------------------------------------------------------------------- #
# CRITICAL PATH: frozen SegNet / PoseNet forward — MLX adapter vs torch
# --------------------------------------------------------------------------- #

_SCORER_CACHE: dict[str, Any] = {}


def _load_scorers_for_bench(device: str):
    """Return (mlx_distortion_adapter, torch_posenet, torch_segnet) with identical weights.

    torch built on the bench device (mps/cpu); MLX adapter converted from the
    CPU torch load (the SAME safetensors). Cached per device.
    """

    key = f"scorers::{device}"
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    from tac.scorer import load_default_scorers
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    upstream = REPO_ROOT / "upstream"
    tdev = torch_device(device)
    posenet_t, segnet_t = load_default_scorers(str(upstream), device=tdev)
    mlx_adapter = load_mlx_distortion_scorer_adapter_from_upstream(str(upstream), device="cpu")
    _SCORER_CACHE[key] = (mlx_adapter, posenet_t, segnet_t)
    return _SCORER_CACHE[key]


def _build_segnet_fwd(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    if dtype != "fp32":
        return []
    try:
        import torch

        mlx_adapter, _posenet_t, segnet_t = _load_scorers_for_bench(device)
    except Exception as exc:
        print(f"  (segnet_fwd unavailable: {exc})")
        return []

    tdev = torch_device(device)
    out: list[OpSize] = []
    for N in [1, 4]:
        x = (rng.random((N, 384, 512, 3)) * 255.0).astype(np.float32)  # NHWC
        x_mx = mlx_from_np(x)
        x_t = torch_from_np(np.ascontiguousarray(x.transpose(0, 3, 1, 2)), tdev)  # NCHW

        def mk_mlx(x_mx=x_mx, mlx_adapter=mlx_adapter):
            return mlx_adapter.segnet(x_mx)  # (N,384,512,5)

        def mk_torch(x_t=x_t, segnet_t=segnet_t):
            import torch as _t

            with _t.no_grad():
                return segnet_t(x_t)  # (N,5,384,512)

        def torch_np(o):
            return o.detach().to("cpu").numpy().transpose(0, 2, 3, 1)

        out.append(
            OpSize(
                size_id=f"N{N}_384x512",
                shape_str=f"SegNet fwd NHWC ({N},384,512,3) -> 5cls",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, mlx_to_np),
                    Candidate("torch", "torch", mk_torch, torch_np),
                ],
                parity_ref="torch",
                parity_tol=5e-2,  # same-device MLX vs torch; authority parity is CPU (see memo)
            )
        )
    return out


def _build_posenet_fwd(device: str, rng: np.random.Generator, dtype: str) -> list[OpSize]:
    if dtype != "fp32":
        return []
    try:
        import torch

        mlx_adapter, posenet_t, _segnet_t = _load_scorers_for_bench(device)
    except Exception as exc:
        print(f"  (posenet_fwd unavailable: {exc})")
        return []

    tdev = torch_device(device)
    out: list[OpSize] = []
    for N in [1, 4]:
        x = (rng.random((N, 192, 256, 12)) * 255.0).astype(np.float32)  # NHWC YUV6 pair
        x_mx = mlx_from_np(x)
        x_t = torch_from_np(np.ascontiguousarray(x.transpose(0, 3, 1, 2)), tdev)  # NCHW

        def mk_mlx(x_mx=x_mx, mlx_adapter=mlx_adapter):
            o = mlx_adapter.posenet(x_mx)
            return o["pose"] if isinstance(o, dict) else o

        def mk_torch(x_t=x_t, posenet_t=posenet_t):
            import torch as _t

            with _t.no_grad():
                o = posenet_t(x_t)
            return o["pose"] if isinstance(o, dict) else o

        out.append(
            OpSize(
                size_id=f"N{N}_192x256x12",
                shape_str=f"PoseNet fwd NHWC ({N},192,256,12) -> pose12",
                candidates=[
                    Candidate("mlx", "mlx", mk_mlx, lambda o: mlx_to_np(o)),
                    Candidate("torch", "torch", mk_torch, torch_to_np),
                ],
                parity_ref="torch",
                parity_tol=5e-2,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Op registry
# --------------------------------------------------------------------------- #


def all_ops() -> list[OpSpec]:
    return [
        OpSpec("matmul", "primitive", _build_matmul),
        OpSpec("conv2d", "primitive", _build_conv2d),
        OpSpec("depthwise_fwd", "primitive", _build_depthwise_fwd),
        OpSpec("groupnorm", "primitive", _build_groupnorm),
        OpSpec("layernorm", "primitive", _build_layernorm),
        OpSpec("argmax", "primitive", _build_argmax),
        OpSpec("gather", "primitive", _build_gather),
        OpSpec("elementwise", "primitive", _build_elementwise),
        OpSpec("grouped_conv_backward", "critical_path", _build_grouped_conv_backward),
        OpSpec("inr_trunk", "critical_path", _build_inr_trunk),
        OpSpec("render_R", "critical_path", _build_render_R),
        OpSpec("segnet_fwd", "critical_path", _build_segnet_fwd),
        OpSpec("posenet_fwd", "critical_path", _build_posenet_fwd),
    ]


# scorer / render / training-step ops are registered from the companion module
# (added once the scorer construction is mapped). Import lazily so the generic
# benchmark runs even before they exist.
def _maybe_scorer_ops() -> list[OpSpec]:
    try:
        from tac.local_acceleration.bench_scorer_ops import scorer_ops  # type: ignore

        return scorer_ops()
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def set_device(device: str) -> None:
    import mlx.core as mx

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)


def env_meta(device: str) -> dict[str, Any]:
    import mlx.core as mx
    import torch

    return {
        "tag": BENCH_TAG,
        "advisory": True,
        "score_authority": False,
        "device": device,
        "mlx_default_device": str(mx.default_device()),
        "python": platform.python_version(),
        "mlx_version": getattr(mx, "__version__", "?"),
        "torch_version": torch.__version__,
        "torch_mps_available": bool(torch.backends.mps.is_available()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "TAC_MLX_CUSTOM_GROUPED_BACKWARD": os.environ.get(
            "TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1(default)"
        ),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def cell_key(op: str, variant: str, size_id: str) -> str:
    return f"{op}::{variant}::{size_id}"


def run(args: argparse.Namespace) -> int:
    device = args.device
    # GPU exclusivity gate
    if device == "gpu":
        alive, line = training_arm_alive()
        if alive and not args.force_gpu_unsafe:
            print(
                f"REFUSED gpu sweep: a training arm appears alive on the GPU:\n  {line}\n"
                "Run with --device cpu (non-contending) or wait for the GPU to free.\n"
                "Override with --force-gpu-unsafe only if you KNOW the GPU is free.",
                file=sys.stderr,
            )
            return 3
        if alive:
            print(f"WARNING: training arm alive but --force-gpu-unsafe set:\n  {line}", file=sys.stderr)

    set_device(device)
    rng = np.random.default_rng(args.seed)

    dtypes = ["fp32"]
    if args.dtypes:
        dtypes = args.dtypes.split(",")

    ops = all_ops() + _maybe_scorer_ops()
    if args.op:
        wanted = set(args.op.split(","))
        ops = [o for o in ops if o.name in wanted]
    if args.category:
        ops = [o for o in ops if o.category == args.category]

    out_path = Path(args.out) if args.out else None
    existing: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    if out_path and out_path.exists() and not args.no_resume:
        try:
            prev = json.loads(out_path.read_text())
            results = prev.get("results", [])
            for r in results:
                existing[cell_key(r["op"], r["variant"], r["size_id"])] = r
            print(f"resume: loaded {len(results)} prior cells from {out_path}")
        except Exception:
            pass

    meta = env_meta(device)
    warmup = 2 if args.quick else args.warmup
    trials = 5 if args.quick else args.trials

    for spec in ops:
        try:
            for dtype in dtypes:
                sizes = spec.build(device, np.random.default_rng(args.seed), dtype)
                for sz in sizes:
                    # parity first
                    ref_np = None
                    cand_np: dict[str, np.ndarray] = {}
                    for cand in sz.candidates:
                        # if every variant of this cell already done, skip compute
                        pass
                    # compute reference numpy once (for parity), unless all cells cached
                    all_cached = all(
                        cell_key(spec.name, c.label, sz.size_id) in existing
                        for c in sz.candidates
                    )
                    if all_cached and not args.refresh:
                        print(f"skip cached: {spec.name} {sz.size_id}")
                        continue

                    # parity pass
                    parity: dict[str, tuple[bool, float, float]] = {}
                    try:
                        for cand in sz.candidates:
                            o = cand.fn()
                            cand_np[cand.label] = cand.to_numpy(o)
                        if sz.parity_ref and sz.parity_ref in cand_np:
                            ref_np = cand_np[sz.parity_ref]
                            for label, arr in cand_np.items():
                                ma, mr = parity_metrics(arr, ref_np)
                                ok = ma <= sz.parity_tol or (label == sz.parity_ref)
                                parity[label] = (ok, ma, mr)
                    except Exception as exc:
                        print(f"  PARITY ERROR {spec.name} {sz.size_id}: {exc}")
                        for cand in sz.candidates:
                            parity[cand.label] = (False, float("nan"), float("nan"))

                    for cand in sz.candidates:
                        ck = cell_key(spec.name, cand.label, sz.size_id)
                        if ck in existing and not args.refresh:
                            continue
                        try:
                            timings, peak, _ = time_candidate(
                                cand, warmup=warmup, trials=trials, device=device
                            )
                            stats = summarize(timings)
                            pok, pma, pmr = parity.get(cand.label, (None, None, None))
                            cr = CellResult(
                                op=spec.name,
                                category=spec.category,
                                variant=cand.label,
                                framework=cand.framework,
                                size_id=sz.size_id,
                                shape=sz.shape_str,
                                dtype=dtype,
                                device=device,
                                trials=trials,
                                median_ms=round(stats["median_ms"], 5),
                                iqr_ms=round(stats["iqr_ms"], 5),
                                mean_ms=round(stats["mean_ms"], 5),
                                min_ms=round(stats["min_ms"], 5),
                                peak_mem_mb=round(peak, 2),
                                parity_ok=pok,
                                parity_max_abs=(None if pma is None else float(pma)),
                                parity_max_rel=(None if pmr is None else float(pmr)),
                                parity_ref=sz.parity_ref,
                            )
                            d = asdict(cr)
                            results = [r for r in results if cell_key(r["op"], r["variant"], r["size_id"]) != ck]
                            results.append(d)
                            existing[ck] = d
                            flag = "ok" if pok else ("PARITY_FAIL" if pok is False else "?")
                            print(
                                f"  {spec.name:22s} {cand.label:11s} {sz.size_id:24s} "
                                f"{stats['median_ms']:9.3f} ms  peak {peak:7.1f}MB  parity={flag} (|Δ|={pma if pma is not None else float('nan'):.2e})"
                            )
                        except Exception as exc:
                            print(f"  RUN ERROR {ck}: {exc}")
                        # incremental save
                        if out_path:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        except Exception as exc:
            print(f"OP ERROR {spec.name}: {exc}")

    # Final save + ratio summary
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
        print(f"\nwrote {len(results)} cells -> {out_path}")

    print_gap_table(results)
    return 0


def print_gap_table(results: list[dict[str, Any]]) -> None:
    """Per (op,size): mlx best vs torch — ratio (torch_ms / mlx_ms); >1 => MLX faster."""

    by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    for r in results:
        by_cell.setdefault((r["op"], r["size_id"]), {})[r["variant"]] = r
    print(f"\n=== GAP TABLE {BENCH_TAG} (ratio = torch_ms / mlx_ms; >1 MLX faster) ===")
    print(f"{'op':22s} {'size':24s} {'mlx_ms':>9s} {'torch_ms':>9s} {'ratio':>7s}  verdict")
    for (op, size), variants in sorted(by_cell.items()):
        # choose best mlx variant (prefer mlx_custom, then mlx, then mlx_ref)
        mlx_v = None
        for pref in ("mlx_custom", "mlx", "mlx_ref"):
            if pref in variants:
                mlx_v = variants[pref]
                break
        torch_v = variants.get("torch")
        if not mlx_v or not torch_v:
            continue
        mlx_ms = mlx_v["median_ms"]
        torch_ms = torch_v["median_ms"]
        ratio = (torch_ms / mlx_ms) if mlx_ms > 0 else float("inf")
        verdict = "MLX faster" if ratio >= 1.1 else ("torch faster" if ratio <= 0.9 else "parity")
        pflag = "" if mlx_v.get("parity_ok") else "  [PARITY?]"
        print(f"{op:22s} {size:24s} {mlx_ms:9.3f} {torch_ms:9.3f} {ratio:7.2f}  {verdict}{pflag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="cpu = non-contending dry run; gpu = Metal (gated on no training arm)")
    ap.add_argument("--gpu-sweep", action="store_true", help="alias for --device gpu")
    ap.add_argument("--force-gpu-unsafe", action="store_true", help="override the training-arm gate")
    ap.add_argument("--op", default="", help="comma-separated op names to filter")
    ap.add_argument("--category", default="", help="primitive | critical_path")
    ap.add_argument("--dtypes", default="", help="comma list, e.g. fp32,fp16 (default fp32)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2 warmup / 5 trials for a fast smoke")
    ap.add_argument("--out", default="", help="JSON output path (resumable)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="recompute even cached cells")
    args = ap.parse_args()
    if args.gpu_sweep:
        args.device = "gpu"
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
