#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-operation MLX vs PyTorch drift measurement for PR95 HNeRV decoder.

PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING 2026-05-25 operator-facing tool
per CLAUDE.md "Max observability - non-negotiable" + Catalog #305 observability
surface declaration.

## Goal

Measure per-operation drift between MLX and PyTorch on the PR95 HNeRV decoder's
forward pass under shared weights. Emit canonical drift attestations that
classify each operation as ``BYTE_STABLE_BY_DEFAULT``,
``NUMERIC_TOLERANCE_INHERENT``, or ``FRAMEWORK_DIFFERENT`` per
``tac.local_acceleration.deterministic_primitives``.

## Canonical operations measured

1. ``bilinear_resize_2x_align_corners_false_nhwc`` - MLX NHWC manual path vs
   PyTorch ``F.interpolate(..., mode='bilinear', align_corners=False)``
2. ``sin`` - MLX ``mx.sin`` vs PyTorch ``torch.sin``
3. ``sigmoid`` - MLX ``mx.sigmoid`` vs PyTorch ``torch.sigmoid``
4. ``pixel_shuffle_2x_nhwc`` - MLX manual NHWC reshape/transpose vs PyTorch
   ``F.pixel_shuffle(..., 2)``
5. ``linear_stem`` - MLX ``nn.Linear`` vs PyTorch ``nn.Linear``
6. ``conv2d_3x3_pad1`` - MLX ``nn.Conv2d`` vs PyTorch ``nn.Conv2d``
7. ``hnerv_decoder_full`` - full PR95 decoder forward vs PyTorch
   ``HNeRVDecoder``

## Non-promotable semantics

Per CLAUDE.md "MLX portable-local-substrate authority" non-negotiable:
- ``score_claim=false``
- ``promotion_eligible=false``
- ``rank_or_kill_eligible=false``
- ``ready_for_exact_eval_dispatch=false``
- ``evidence_grade=MLX-research-signal``

This tool is the canonical drift-measurement surface - it does NOT promote any
MLX result to contest-axis score authority.

## Usage

```bash
.venv/bin/python tools/measure_pr95_mlx_pytorch_per_op_drift.py \\
  --report-out .omx/tmp/pr95_drift_probe/per_op_drift_report.json \\
  --mlx-device cpu \\
  --seed 42
```

The report is consumed by:
- Slot 1 export bridge VERDICT upgrade
- ``tac.canonical_equations`` empirical anchor for equation
  ``mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1``
  (FORMALIZATION_PENDING per Catalog #344)
- ``tools/cathedral_autopilot_autonomous_loop.py`` substrate-cost-model
  (operator-routable; not auto-wired)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PR95_PUBLIC_MODEL = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/src/model.py"
)
SCHEMA_VERSION = "pr95_mlx_pytorch_per_op_drift_report.v1"


def _load_public_pr95_decoder(source_model: Path) -> Any:
    """Import the public PR95 HNeRVDecoder from the canonical clone."""
    if not source_model.is_file():
        raise FileNotFoundError(f"public PR95 model.py not found: {source_model}")
    spec = importlib.util.spec_from_file_location(
        "public_pr95_hnerv_model_for_drift_probe",
        source_model,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import public PR95 model.py: {source_model}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "HNeRVDecoder"):
        raise RuntimeError(f"{source_model}: missing HNeRVDecoder")
    return module


def _torch_nchw_to_mlx_nhwc(x_nchw_np: np.ndarray, mx: Any) -> Any:
    return mx.array(np.transpose(x_nchw_np, (0, 2, 3, 1)).copy())


def _mlx_nhwc_to_numpy_nchw(x_mlx: Any) -> np.ndarray:
    return np.transpose(np.asarray(x_mlx), (0, 3, 1, 2))


def _measure_op(
    op_name: str,
    pytorch_output: np.ndarray,
    mlx_output: np.ndarray,
) -> dict[str, Any]:
    """Build a per-op drift measurement row via canonical helper."""
    from tac.local_acceleration.deterministic_primitives import (
        validate_mlx_pytorch_parity_within_tolerance,
    )

    attestation = validate_mlx_pytorch_parity_within_tolerance(
        operation_name=op_name,
        mlx_output=mlx_output,
        pytorch_output=pytorch_output,
    )
    return attestation.as_dict()


def measure_per_op_drift(
    *,
    mlx_device: str = "cpu",
    seed: int = 42,
    source_model: Path = DEFAULT_PR95_PUBLIC_MODEL,
) -> dict[str, Any]:
    """Run the full per-operation drift measurement and return the canonical report."""

    import mlx.core as mx
    import mlx.nn as mlx_nn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from tac.local_acceleration.pr95_hnerv_mlx import (
        HNeRVDecoderMLX,
        bilinear_resize2x_align_corners_false_nhwc,
        load_pytorch_state_dict_into_mlx,
        pixel_shuffle_2x_nhwc,
    )

    if mlx_device == "cpu":
        mx.set_default_device(mx.cpu)
    elif mlx_device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError(f"mlx_device must be 'cpu' or 'gpu'; got {mlx_device!r}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    per_op_results: dict[str, dict[str, Any]] = {}

    # 1) bilinear_resize_2x_align_corners_false_nhwc
    x_np = np.random.randn(1, 36, 6, 8).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_mlx = _torch_nchw_to_mlx_nhwc(x_np, mx)
    with torch.no_grad():
        y_torch = F.interpolate(
            x_torch, scale_factor=2, mode="bilinear", align_corners=False
        ).numpy()
    y_mlx = bilinear_resize2x_align_corners_false_nhwc(x_mlx)
    mx.eval(y_mlx)
    y_mlx_np = _mlx_nhwc_to_numpy_nchw(y_mlx)
    per_op_results["bilinear_resize_2x_align_corners_false_nhwc"] = _measure_op(
        "bilinear_resize_2x_align_corners_false_nhwc", y_torch, y_mlx_np
    )

    # 2) sin
    x_np2 = np.random.randn(1, 36, 12, 16).astype(np.float32)
    x_torch2 = torch.from_numpy(x_np2)
    x_mlx2 = _torch_nchw_to_mlx_nhwc(x_np2, mx)
    with torch.no_grad():
        y_torch2 = torch.sin(x_torch2).numpy()
    y_mlx2 = mx.sin(x_mlx2)
    mx.eval(y_mlx2)
    y_mlx2_np = _mlx_nhwc_to_numpy_nchw(y_mlx2)
    per_op_results["sin"] = _measure_op("sin", y_torch2, y_mlx2_np)

    # 3) sigmoid
    with torch.no_grad():
        y_torch3 = torch.sigmoid(x_torch2).numpy()
    y_mlx3 = mx.sigmoid(x_mlx2)
    mx.eval(y_mlx3)
    y_mlx3_np = _mlx_nhwc_to_numpy_nchw(y_mlx3)
    per_op_results["sigmoid"] = _measure_op("sigmoid", y_torch3, y_mlx3_np)

    # 4) pixel_shuffle_2x_nhwc
    x_np4 = np.random.randn(1, 4 * 36, 6, 8).astype(np.float32)
    x_torch4 = torch.from_numpy(x_np4)
    x_mlx4 = _torch_nchw_to_mlx_nhwc(x_np4, mx)
    with torch.no_grad():
        y_torch4 = F.pixel_shuffle(x_torch4, 2).numpy()
    y_mlx4 = pixel_shuffle_2x_nhwc(x_mlx4)
    mx.eval(y_mlx4)
    y_mlx4_np = _mlx_nhwc_to_numpy_nchw(y_mlx4)
    per_op_results["pixel_shuffle_2x_nhwc"] = _measure_op(
        "pixel_shuffle_2x_nhwc", y_torch4, y_mlx4_np
    )

    # 5) linear_stem
    torch_lin = nn.Linear(28, 36 * 6 * 8)
    mlx_lin = mlx_nn.Linear(28, 36 * 6 * 8)
    mlx_lin.weight = mx.array(
        torch_lin.weight.detach().numpy().astype(np.float32).copy()
    )
    mlx_lin.bias = mx.array(torch_lin.bias.detach().numpy().astype(np.float32).copy())
    z = np.random.randn(1, 28).astype(np.float32)
    z_torch = torch.from_numpy(z)
    z_mlx = mx.array(z)
    with torch.no_grad():
        y_torch_lin = torch_lin(z_torch).numpy()
    y_mlx_lin = mlx_lin(z_mlx)
    mx.eval(y_mlx_lin)
    per_op_results["linear_stem"] = _measure_op(
        "linear_stem", y_torch_lin, np.asarray(y_mlx_lin)
    )

    # 6) conv2d_3x3_pad1
    torch_conv = nn.Conv2d(36, 144, 3, padding=1)
    mlx_conv = mlx_nn.Conv2d(36, 144, 3, padding=1)
    W = torch_conv.weight.detach().numpy().astype(np.float32)
    b = torch_conv.bias.detach().numpy().astype(np.float32)
    mlx_conv.weight = mx.array(np.transpose(W, (0, 2, 3, 1)).copy())
    mlx_conv.bias = mx.array(b.copy())
    x_np6 = np.random.randn(1, 36, 6, 8).astype(np.float32)
    x_torch6 = torch.from_numpy(x_np6)
    x_mlx6 = _torch_nchw_to_mlx_nhwc(x_np6, mx)
    with torch.no_grad():
        y_torch6 = torch_conv(x_torch6).numpy()
    y_mlx6 = mlx_conv(x_mlx6)
    mx.eval(y_mlx6)
    y_mlx6_np = _mlx_nhwc_to_numpy_nchw(y_mlx6)
    per_op_results["conv2d_3x3_pad1"] = _measure_op(
        "conv2d_3x3_pad1", y_torch6, y_mlx6_np
    )

    # 7) hnerv_decoder_full
    public_pr95 = _load_public_pr95_decoder(source_model)
    torch_decoder = public_pr95.HNeRVDecoder(latent_dim=28, base_channels=36).eval()
    torch_state = {k: v.detach() for k, v in torch_decoder.state_dict().items()}

    mlx_decoder = HNeRVDecoderMLX(latent_dim=28, base_channels=36)
    load_pytorch_state_dict_into_mlx(
        mlx_decoder, {k: v.numpy() for k, v in torch_state.items()}
    )

    z_full = np.random.randn(2, 28).astype(np.float32) * 0.1
    z_full_torch = torch.from_numpy(z_full)
    z_full_mlx = mx.array(z_full)
    with torch.no_grad():
        y_torch_full = torch_decoder(z_full_torch).cpu().numpy()
    y_mlx_full = mlx_decoder(z_full_mlx)
    mx.eval(y_mlx_full)
    y_mlx_full_np = np.asarray(y_mlx_full)
    per_op_results["hnerv_decoder_full"] = _measure_op(
        "hnerv_decoder_full", y_torch_full, y_mlx_full_np
    )

    # Aggregate verdict
    all_byte_stable_or_within = all(
        row["attested_within_band"] for row in per_op_results.values()
    )
    forbidden_count = sum(
        1
        for row in per_op_results.values()
        if row["actual_class"] == "framework_different"
    )

    return {
        "schema": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/measure_pr95_mlx_pytorch_per_op_drift.py",
        "mlx_device": mlx_device,
        "seed": seed,
        "source_model_path": str(source_model),
        "per_op": per_op_results,
        "aggregate": {
            "all_within_attested_bands": all_byte_stable_or_within,
            "framework_different_op_count": forbidden_count,
            "op_count": len(per_op_results),
        },
        # Canonical Provenance per Catalog #287/#323/#192/#1.
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "local_mlx_pytorch_per_op_drift_probe_is_not_contest_auth_eval",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
        ],
        "operator_routable_summary": (
            "Per-op drift measurement complete. Operator routes through Slot 1 "
            "export bridge VERDICT upgrade if all ops within attested bands."
        ),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-out",
        type=Path,
        required=True,
        help="Output path for canonical drift report JSON.",
    )
    parser.add_argument(
        "--mlx-device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="MLX default device for the measurement (default: cpu for determinism).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input + weight initialization (default: 42).",
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=DEFAULT_PR95_PUBLIC_MODEL,
        help="Path to the public PR95 HNeRVDecoder model.py (canonical PyTorch reference).",
    )
    parser.add_argument(
        "--require-all-within-bands",
        action="store_true",
        help="Exit nonzero if any operation drift exceeds its canonical attested band.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = measure_per_op_drift(
        mlx_device=args.mlx_device,
        seed=args.seed,
        source_model=args.source_model,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        f"[pr95-drift-measure] mlx_device={args.mlx_device} seed={args.seed}"
    )
    for name, row in report["per_op"].items():
        in_band = "PASS" if row["attested_within_band"] else "FAIL"
        print(
            f"  {in_band} {name:55s} "
            f"max_abs={row['measured_max_abs']:.4e} "
            f"mean_abs={row['measured_mean_abs']:.4e} "
            f"actual={row['actual_class']:30s} expected={row['expected_class']}"
        )
    agg = report["aggregate"]
    print(
        f"[pr95-drift-measure] all_within_attested_bands={agg['all_within_attested_bands']} "
        f"framework_different_count={agg['framework_different_op_count']} "
        f"op_count={agg['op_count']}"
    )
    print(f"[pr95-drift-measure] report={args.report_out}")

    if args.require_all_within_bands and not agg["all_within_attested_bands"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
