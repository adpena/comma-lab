#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Source/API audit for native MLX int8 convolution and local CoreML readiness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


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


def _mlx_audit() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("mlx")
    root = Path(distribution.locate_file(""))
    quantized = root / "mlx/nn/layers/quantized.py"
    convolution = root / "mlx/nn/layers/convolution.py"
    quantized_text = quantized.read_text()
    convolution_text = convolution.read_text()
    quantized_classes = [
        name
        for name in ("QuantizedLinear", "QuantizedEmbedding", "QuantizedConv1d", "QuantizedConv2d", "QuantizedConv3d")
        if f"class {name}" in quantized_text
    ]
    conv_has_to_quantized = "def to_quantized" in convolution_text
    return {
        "status": "SOURCE_VERIFIED",
        "installed_version": distribution.version,
        "requested_family": "native quantized int8 convolution",
        "quantized_layer_source": {
            "path": str(quantized),
            "sha256": _sha256(quantized),
            "classes_found": quantized_classes,
        },
        "convolution_layer_source": {
            "path": str(convolution),
            "sha256": _sha256(convolution),
            "has_to_quantized": conv_has_to_quantized,
        },
        "quantized_matmul_api_present_in_stubs": "quantized_matmul"
        in (root / "mlx/core/__init__.pyi").read_text(errors="ignore"),
        "native_quantized_conv_supported": bool(
            any(name.startswith("QuantizedConv") for name in quantized_classes) or conv_has_to_quantized
        ),
        "verdict_scope": (
            "public Python layer/API surface of the installed MLX build only; raw integer dtype "
            "acceptance or a private Metal kernel is not equivalent to a supported quantized Conv API"
        ),
    }


def _coreml_audit() -> dict[str, Any]:
    coremltools_spec = importlib.util.find_spec("coremltools")
    xcrun = shutil.which("xcrun")
    compiler = None
    if xcrun:
        result = subprocess.run([xcrun, "-f", "coremlcompiler"], text=True, capture_output=True, check=False)
        if result.returncode == 0:
            compiler = result.stdout.strip()
    compilable = coremltools_spec is not None and compiler is not None
    return {
        "status": "READY_TO_COMPILE" if compilable else "BLOCKED_NOT_MEASURED",
        "coremltools_importable": coremltools_spec is not None,
        "coremltools_origin": None if coremltools_spec is None else coremltools_spec.origin,
        "xcrun": xcrun,
        "coremlcompiler": compiler,
        "w8a8_segnet_compiled": False,
        "ane_forward_latency_ms": None,
        "baseline_mlx_fp32_forward_median_ms": 20.1974,
        "overlap_measured": False,
        "ticket": (
            "convert frozen SegNet; calibrate activation quantization on the same real n600 scorer "
            "inputs; linear-quantize activations then weights; compile; run a persistent CPU_AND_NE "
            "MLModel; report warm median/p05/p95, compute-unit trace, argmax flips, and concurrent "
            "GPU-witness overlap against the 20.1974 ms MLX-fp32 baseline"
        ),
        "verdict_scope": (
            "this contained interpreter/toolchain only; absence of coremltools blocks conversion and "
            "is not a verdict against W8A8 CoreML/ANE on the M5 host"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if str(out).startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit("refusing a temporary durable evidence path")
    payload = {
        "schema": "int8_teacher_backend_support_audit.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_int8_training_rungs_20260713",
        "axis": "[local source/toolchain audit; no score authority]",
        "labels": {"mlx_api": "MEASURED_BY_SOURCE_INSPECTION", "ane_latency": "NOT_MEASURED_TICKET"},
        "training_launched": False,
        "provenance": {
            "audit_source": str(Path(__file__).resolve().relative_to(REPO)),
            "audit_source_sha256": _sha256(Path(__file__).resolve()),
        },
        "mlx": _mlx_audit(),
        "coreml_ane": _coreml_audit(),
        "pointer_delta": "ZERO",
    }
    _atomic_json(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
