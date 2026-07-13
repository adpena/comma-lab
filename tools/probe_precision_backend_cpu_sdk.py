#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record CPU-torch and installed Apple-SDK convolution precision support.

This is an existence/semantics probe, not a contest score and not a training
launch.  It times only the EfficientNet-B2 stem geometry on deterministic real
pixels, verifies the narrow raw-int16 CPU primitive (including overflow),
checks PyTorch dispatcher registration, and quotes the installed SDK headers
that distinguish compressed UInt8 MPSCNN weights from integer convolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
MPS_CNN_HEADER = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
    "MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/"
    "Frameworks/MPSNeuralNetwork.framework/Versions/A/Headers/MPSCNNConvolution.h"
)
MPS_GRAPH_QUANT_HEADER = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
    "MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/Versions/A/"
    "Headers/MPSGraphQuantizationOps.h"
)
MPS_TYPES_HEADER = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
    "MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/"
    "Frameworks/MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h"
)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _lines_matching(path: Path, needles: tuple[str, ...], context: int = 2) -> list[dict[str, Any]]:
    lines = path.read_text(errors="replace").splitlines()
    selected: list[dict[str, Any]] = []
    emitted: set[tuple[int, int]] = set()
    for index, line in enumerate(lines):
        if not any(needle in line for needle in needles):
            continue
        lo = max(0, index - context)
        hi = min(len(lines), index + context + 1)
        if (lo, hi) in emitted:
            continue
        emitted.add((lo, hi))
        selected.append(
            {
                "line_start": lo + 1,
                "line_end": hi,
                "text": "\n".join(lines[lo:hi]),
            }
        )
    return selected


def _median_time(call: Callable[[], torch.Tensor], *, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        call()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "samples_s": samples,
    }


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    x = reference.detach().to(torch.float64).reshape(-1)
    y = candidate.detach().to(torch.float64).reshape(-1)
    return float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))


def _real_input(batch_size: int) -> torch.Tensor:
    with np.load(GT_CACHE, mmap_mode="r", allow_pickle=False) as cache:
        source = np.asarray(cache["gt_f1"][0], dtype=np.uint8).copy()
    value = torch.from_numpy(source).permute(2, 0, 1)[None].to(torch.float32) / 255.0
    value = F.interpolate(value, size=(384, 512), mode="bilinear", align_corners=False)
    return value.repeat(batch_size, 1, 1, 1).contiguous()


def _dtype_primitive(dtype: torch.dtype) -> dict[str, Any]:
    torch.manual_seed(20260713)
    x = torch.randint(-2, 3, (1, 3, 17, 19), dtype=torch.int32).to(dtype)
    w = torch.randint(-2, 3, (5, 3, 3, 3), dtype=torch.int32).to(dtype)
    if dtype.is_floating_point:
        x.requires_grad_(True)
    try:
        output = F.conv2d(x, w, stride=2, padding=1)
        gradient = False
        if dtype.is_floating_point:
            output.to(torch.float32).sum().backward()
            gradient = x.grad is not None
        return {
            "status": "EXECUTED",
            "input_dtype": str(dtype),
            "output_dtype": str(output.dtype),
            "output_shape": list(output.shape),
            "autograd": gradient,
        }
    except Exception as exc:  # pragma: no cover - receipt path
        return {"status": "ERROR", "input_dtype": str(dtype), "error": f"{type(exc).__name__}: {exc}"}


def _raw_int16_overflow() -> dict[str, Any]:
    x = torch.full((1, 1, 3, 3), 300, dtype=torch.int16)
    w = torch.full((1, 1, 3, 3), 300, dtype=torch.int16)
    output = F.conv2d(x, w)
    mathematical = 9 * 300 * 300
    return {
        "mathematical_int64_result": mathematical,
        "torch_int16_result": int(output.item()),
        "same_dtype_output": str(output.dtype),
        "overflow_observed": int(output.item()) != mathematical,
        "interpretation": (
            "raw aten int16 convolution exists, but same-dtype overflow and absent autograd/scale semantics "
            "make it unusable as a quantized frozen-SegNet convolution"
        ),
    }


def _quantized_stem(batch_size: int, *, warmup: int, repeats: int) -> dict[str, Any]:
    torch.manual_seed(20260713)
    x = _real_input(batch_size)
    weight = torch.randn(32, 3, 3, 3, dtype=torch.float32) * 0.05
    bias = torch.randn(32, dtype=torch.float32) * 0.01
    reference = F.conv2d(x, weight, bias, stride=2, padding=1)

    weight_scale = weight.abs().amax(dim=(1, 2, 3)).clamp_min(1.0e-12) / 127.0
    weight_zero = torch.zeros(32, dtype=torch.int64)
    qweight = torch.quantize_per_channel(weight, weight_scale, weight_zero, 0, torch.qint8)
    x_scale = float(x.max().item() / 255.0) or 1.0e-12
    qx = torch.quantize_per_tensor(x, x_scale, 0, torch.quint8)
    qconv = torch.ao.nn.quantized.Conv2d(3, 32, 3, stride=2, padding=1)
    qconv.set_weight_bias(qweight, bias)
    qconv.scale = max(float(reference.abs().max().item() / 127.0), 1.0e-12)
    qconv.zero_point = 128
    quantized_output = qconv(qx).dequantize()
    fp_timing = _median_time(
        lambda: F.conv2d(x, weight, bias, stride=2, padding=1), warmup=warmup, repeats=repeats
    )
    int8_timing = _median_time(lambda: qconv(qx), warmup=warmup, repeats=repeats)
    return {
        "geometry": [batch_size, 3, 384, 512],
        "engine": torch.backends.quantized.engine,
        "fp32": fp_timing,
        "w8a8_quantized_cpu": int8_timing,
        "w8a8_speedup_vs_fp32": fp_timing["median_s"] / int8_timing["median_s"],
        "output_cosine_vs_fp32": _cosine(reference, quantized_output),
        "output_argmax_flips": int(
            torch.count_nonzero(reference.argmax(dim=1) != quantized_output.argmax(dim=1)).item()
        ),
        "output_argmax_elements": int(reference.shape[0] * reference.shape[2] * reference.shape[3]),
        "scope": "deterministic real RGB pixels with synthetic stem weights; operator-existence receipt only",
    }


def _dispatcher() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for operator in ("quantized::conv2d.new", "aten::conv2d"):
        rows[operator] = {}
        for key in ("QuantizedCPU", "MPS", "CPU"):
            try:
                rows[operator][key] = torch._C._dispatch_has_kernel_for_dispatch_key(operator, key)
            except RuntimeError as exc:
                rows[operator][key] = f"ERROR: {exc}"
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.warmup < 1 or args.repeats < 3:
        parser.error("--warmup must be >=1 and --repeats must be >=3")
    if str(args.out).startswith(("/tmp/", "/private/tmp/")):
        parser.error("refusing temporary durable evidence path")

    prior_threads = torch.get_num_threads()
    prior_engine = torch.backends.quantized.engine
    torch.set_num_threads(1)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    try:
        dtype_rows = {
            name: _dtype_primitive(dtype)
            for name, dtype in (
                ("fp32", torch.float32),
                ("fp16", torch.float16),
                ("bf16", torch.bfloat16),
                ("int8_raw", torch.int8),
                ("int16_raw", torch.int16),
                ("int32_raw", torch.int32),
            )
        }
        timing = {
            f"batch_{batch}": _quantized_stem(batch, warmup=args.warmup, repeats=args.repeats)
            for batch in (1, 8)
        }
    finally:
        if prior_engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = prior_engine
        torch.set_num_threads(prior_threads)

    headers = {}
    for name, path, needles in (
        (
            "mpscnn_convolution",
            MPS_CNN_HEADER,
            (
                "Same scheme will be used to dequantize weights to fp16",
                "must return a kernelWeightsDataType of MPSDataTypeFloat16",
            ),
        ),
        (
            "mpsgraph_quantization",
            MPS_GRAPH_QUANT_HEADER,
            ("Convert the float `tensor` to an i8 or u8 tensor", "i8, u8, i4 or u4"),
        ),
        (
            "mps_datatypes",
            MPS_TYPES_HEADER,
            ("MPSDataTypeInt4", "MPSDataTypeInt8", "MPSDataTypeInt16", "MPSDataTypeBFloat16"),
        ),
    ):
        headers[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "matched_excerpts": _lines_matching(path, needles),
        }

    payload = {
        "schema": "precision_backend_cpu_sdk.v1",
        "status": "MEASURED-EXISTENCE-RECEIPT",
        "research_only": True,
        "score_claim": False,
        "training_launched": False,
        "written_at_utc": _utc(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "torch_threads_during_probe": 1,
        },
        "torch_dispatcher": _dispatcher(),
        "cpu_conv2d_dtype_primitives": dtype_rows,
        "raw_int16_overflow": _raw_int16_overflow(),
        "cpu_w8a8_stem": timing,
        "installed_sdk_headers": headers,
        "verdict_scope": (
            "existence and semantics of installed CPU-torch and public Apple SDK surfaces; no claim that "
            "an enum dtype implies a convolution implementation, and no Metal/ANE execution claim"
        ),
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
