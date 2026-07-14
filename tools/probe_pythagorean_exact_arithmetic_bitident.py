#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-process MLX probe for integer-lowered resize-adjoint accumulation.

This is the decisive local probe for the 2026-07-13 Pythagorean exact-
arithmetic investigation.  It exercises a real operation from render-R: the
duplicate-index accumulation in the transpose of one 384 -> 874 bicubic resize
axis.  The current float32 formulation and a Q15/int32 lowering use identical
indices and cotangents.

Authority: [macOS-MLX research-signal] for on-device cross-process hash
identity; NumPy-fp32/int32 is the numerical reference.  This tool never emits
a contest score and never launches training, evaluation, or a paid job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / ".omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json"
SCHEMA = "pythagorean_exact_arithmetic_bitident_probe.v1"

# Actual vertical axis of the contest-faithful witness render-R.
IN_SIZE = 384
OUT_SIZE = 874
TAPS = 4
LANES = 128 * 3  # one 128-wide RGB slice; independent resize-adjoint lanes
Q_BITS = 15
Q_SCALE = 1 << Q_BITS
SEED = 157
N_PROCESSES = 10
VARIANTS = ("float_atomic", "fixed_q15_int32_atomic")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
    return _sha256_bytes(header + contiguous.tobytes())


def bicubic_indices_weights_numpy(*, in_size: int = IN_SIZE, out_size: int = OUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the MLX reference-R align_corners=False cubic coordinate map."""

    scale = np.float32(float(in_size) / float(out_size))
    out = np.arange(out_size, dtype=np.float32)
    real = ((out + np.float32(0.5)) * scale - np.float32(0.5)).astype(np.float32)
    base = np.floor(real).astype(np.int32)
    offsets = np.asarray([-1, 0, 1, 2], dtype=np.int32)
    unclipped = base[:, None] + offsets[None, :]
    distance = real[:, None] - unclipped.astype(np.float32)

    a = np.float32(-0.75)
    x = np.abs(distance).astype(np.float32)
    x2 = (x * x).astype(np.float32)
    x3 = (x2 * x).astype(np.float32)
    inner = ((a + np.float32(2.0)) * x3 - (a + np.float32(3.0)) * x2 + np.float32(1.0)).astype(np.float32)
    outer = (a * x3 - np.float32(5.0) * a * x2 + np.float32(8.0) * a * x - np.float32(4.0) * a).astype(np.float32)
    weights = np.where(
        x <= np.float32(1.0),
        inner,
        np.where(x < np.float32(2.0), outer, np.float32(0.0)),
    ).astype(np.float32)
    indices = np.clip(unclipped, 0, in_size - 1).astype(np.int32)
    return indices, weights


def build_resize_adjoint_fixture(*, seed: int = SEED, lanes: int = LANES) -> dict[str, np.ndarray | float | int]:
    """Build common float/fixed-point operands and deterministic NumPy references."""

    indices, weights = bicubic_indices_weights_numpy()
    rng = np.random.default_rng(seed)
    cotangent_i16 = rng.integers(-127, 128, size=(OUT_SIZE, lanes), dtype=np.int16)
    cotangent_f32 = cotangent_i16.astype(np.float32)

    lane_ids = np.arange(lanes, dtype=np.int32)
    destination = (
        indices[:, :, None].astype(np.int64) * np.int64(lanes) + lane_ids[None, None, :].astype(np.int64)
    ).reshape(-1)
    destination_u32 = destination.astype(np.uint32)

    float_contrib = (weights[:, :, None] * cotangent_f32[:, None, :]).astype(np.float32)
    float_reference = np.zeros(IN_SIZE * lanes, dtype=np.float32)
    np.add.at(float_reference, destination, float_contrib.reshape(-1))

    weights_q15 = np.rint(weights.astype(np.float64) * float(Q_SCALE)).astype(np.int32)
    int_contrib_i64 = weights_q15[:, :, None].astype(np.int64) * cotangent_i16[:, None, :].astype(np.int64)
    int_reference_i64 = np.zeros(IN_SIZE * lanes, dtype=np.int64)
    np.add.at(int_reference_i64, destination, int_contrib_i64.reshape(-1))

    abs_accum_i64 = np.zeros(IN_SIZE * lanes, dtype=np.int64)
    np.add.at(abs_accum_i64, destination, np.abs(int_contrib_i64).reshape(-1))
    max_abs_integer_accumulator = int(np.max(abs_accum_i64))
    if max_abs_integer_accumulator > np.iinfo(np.int32).max:
        raise OverflowError(
            f"Q15 resize-adjoint fixture exceeds int32 accumulator range: {max_abs_integer_accumulator}"
        )
    int_reference = int_reference_i64.astype(np.int32)

    # A priori parity envelope for Q15 lowering versus a float32 fixed-order
    # authority.  Quantization error is summed exactly per destination.  The
    # floating reduction term uses the standard gamma_n bound twice (one order
    # for NumPy, one arbitrary atomic order for MLX).
    dequantized_weights = weights_q15.astype(np.float64) / float(Q_SCALE)
    per_term_quant_error = (
        np.abs(dequantized_weights - weights.astype(np.float64))[:, :, None]
        * np.abs(cotangent_i16.astype(np.float64))[:, None, :]
    )
    quant_bound = np.zeros(IN_SIZE * lanes, dtype=np.float64)
    np.add.at(quant_bound, destination, per_term_quant_error.reshape(-1))

    abs_float_terms = np.zeros(IN_SIZE * lanes, dtype=np.float64)
    np.add.at(abs_float_terms, destination, np.abs(float_contrib.astype(np.float64)).reshape(-1))
    counts = np.zeros(IN_SIZE * lanes, dtype=np.int32)
    np.add.at(counts, destination, 1)
    eps = float(np.finfo(np.float32).eps)
    max_count = int(np.max(counts))
    gamma = (max_count * eps) / (1.0 - max_count * eps)
    rounding_bound = 2.0 * gamma * abs_float_terms
    float_reorder_tolerance = float(np.max(rounding_bound) + 8.0 * eps)
    authority_tolerance = float(np.max(quant_bound + rounding_bound) + 8.0 * eps)

    return {
        "destination_u32": destination_u32,
        "float_contrib": float_contrib.reshape(-1),
        "int_contrib": int_contrib_i64.astype(np.int32).reshape(-1),
        "float_reference": float_reference,
        "int_reference": int_reference,
        "authority_tolerance": authority_tolerance,
        "float_reorder_tolerance": float_reorder_tolerance,
        "max_abs_integer_accumulator": max_abs_integer_accumulator,
        "max_contributions_per_destination": max_count,
    }


def run_mlx_variant(variant: str) -> dict[str, Any]:
    """Execute one child cell on MLX GPU and compare it to NumPy authority."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    fixture = build_resize_adjoint_fixture()
    destination = mx.array(fixture["destination_u32"], dtype=mx.uint32)
    output_size = IN_SIZE * LANES

    if variant == "float_atomic":
        contribution = mx.array(fixture["float_contrib"], dtype=mx.float32)
        output_mx = mx.zeros((output_size,), dtype=mx.float32).at[destination].add(contribution)
        mx.eval(output_mx)
        output = np.asarray(output_mx, dtype=np.float32)
        reference = np.asarray(fixture["float_reference"], dtype=np.float32)
        max_abs_error = float(np.max(np.abs(output.astype(np.float64) - reference.astype(np.float64))))
        parity = {
            "reference_dtype": "numpy-fp32",
            "bit_identical": bool(np.array_equal(output.view(np.uint32), reference.view(np.uint32))),
            "max_abs_error": max_abs_error,
            "derived_two_order_fp32_tolerance": float(fixture["float_reorder_tolerance"]),
            "within_derived_tolerance": bool(max_abs_error <= float(fixture["float_reorder_tolerance"])),
            "interpretation": "fp32 atomic reduction-order delta; not authority bytes",
        }
    else:
        contribution = mx.array(fixture["int_contrib"], dtype=mx.int32)
        output_mx = mx.zeros((output_size,), dtype=mx.int32).at[destination].add(contribution)
        mx.eval(output_mx)
        output = np.asarray(output_mx, dtype=np.int32)
        reference = np.asarray(fixture["int_reference"], dtype=np.int32)
        dequantized = output.astype(np.float64) / float(Q_SCALE)
        float_reference = np.asarray(fixture["float_reference"], dtype=np.float32).astype(np.float64)
        max_abs_error = float(np.max(np.abs(dequantized - float_reference)))
        tolerance = float(fixture["authority_tolerance"])
        parity = {
            "reference_dtype": "numpy-int32 plus numpy-fp32 resize-adjoint authority",
            "integer_bit_identical": bool(np.array_equal(output, reference)),
            "dequantized_max_abs_error_vs_numpy_fp32": max_abs_error,
            "derived_quantization_plus_fp32_tolerance": tolerance,
            "within_derived_tolerance": bool(max_abs_error <= tolerance),
        }

    return {
        "variant": variant,
        "output_sha256": _hash_array(output),
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
        "parity": parity,
        "mlx_version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "fixture": {
            "in_size": IN_SIZE,
            "out_size": OUT_SIZE,
            "taps": TAPS,
            "lanes": LANES,
            "q_bits": Q_BITS,
            "seed": SEED,
            "max_abs_integer_accumulator": int(fixture["max_abs_integer_accumulator"]),
            "int32_limit": int(np.iinfo(np.int32).max),
            "max_contributions_per_destination": int(fixture["max_contributions_per_destination"]),
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _base_receipt(n: int) -> dict[str, Any]:
    fixture = build_resize_adjoint_fixture()
    dequantized = np.asarray(fixture["int_reference"], dtype=np.int32).astype(np.float64) / float(Q_SCALE)
    float_reference = np.asarray(fixture["float_reference"], dtype=np.float32).astype(np.float64)
    error = dequantized - float_reference
    return {
        "schema": SCHEMA,
        "lane_id": "pythagorean_exact_arithmetic_bitident",
        "axis": "[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "training": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
        "git_head_at_probe": _git_head(),
        "host": platform.node(),
        "platform": platform.platform(),
        "n_requested_per_variant": n,
        "contract": {
            "real_op": "one-axis transpose accumulation of render-R bicubic resize 384->874",
            "float_formulation": "MLX float32 duplicate-index atomic scatter-add",
            "integer_formulation": "same indices/cotangent; Q15 cubic weights; bounded MLX int32 atomic add",
            "acceptance": (
                "float cross-process divergence; integer 0/N divergence; exact NumPy-int32 parity; "
                "dequantized result within derived Q15+fp32 authority tolerance"
            ),
        },
        "numpy_static_contract": {
            "int_reference_sha256": _hash_array(np.asarray(fixture["int_reference"], dtype=np.int32)),
            "float_reference_sha256": _hash_array(np.asarray(fixture["float_reference"], dtype=np.float32)),
            "max_abs_integer_accumulator": int(fixture["max_abs_integer_accumulator"]),
            "int32_positive_limit": int(np.iinfo(np.int32).max),
            "overflow_headroom_x": float(np.iinfo(np.int32).max) / float(fixture["max_abs_integer_accumulator"]),
            "max_contributions_per_destination": int(fixture["max_contributions_per_destination"]),
            "float_reorder_tolerance": float(fixture["float_reorder_tolerance"]),
            "dequantized_max_abs_error_vs_numpy_fp32": float(np.max(np.abs(error))),
            "dequantized_rmse_vs_numpy_fp32": float(np.sqrt(np.mean(error * error))),
            "derived_quantization_plus_fp32_tolerance": float(fixture["authority_tolerance"]),
            "within_derived_tolerance": bool(float(np.max(np.abs(error))) <= float(fixture["authority_tolerance"])),
        },
        "source_custody": {
            "probe": {
                "path": "tools/probe_pythagorean_exact_arithmetic_bitident.py",
                "sha256": _sha256_file(Path(__file__)),
            },
            "reference_r": {
                "path": "src/tac/local_acceleration/pr95_hnerv_mlx_training.py",
                "sha256": _sha256_file(REPO / "src/tac/local_acceleration/pr95_hnerv_mlx_training.py"),
            },
            "l70_probe": {
                "path": "tools/mlx_gpu_determinism_probe.py",
                "sha256": _sha256_file(REPO / "tools/mlx_gpu_determinism_probe.py"),
            },
        },
        "trials": {variant: [] for variant in VARIANTS},
    }


def _summarize(receipt: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = receipt["trials"][variant]
        hashes = [row["output_sha256"] for row in rows if "output_sha256" in row]
        counts = Counter(hashes)
        representative_count = max(counts.values()) if counts else 0
        entry: dict[str, Any] = {
            "n": len(hashes),
            "unique_hashes": len(counts),
            "cross_process_identical": len(counts) == 1 and len(hashes) > 0,
            "divergent_from_modal_hash": len(hashes) - representative_count,
            "hashes": hashes,
        }
        if rows:
            entry["parity_all"] = all(
                (
                    row["parity"].get("integer_bit_identical", True)
                    and row["parity"].get("within_derived_tolerance", True)
                )
                for row in rows
            )
            entry["max_abs_error_vs_numpy"] = max(
                float(
                    row["parity"].get(
                        "dequantized_max_abs_error_vs_numpy_fp32",
                        row["parity"].get("max_abs_error", 0.0),
                    )
                )
                for row in rows
            )
        summary[variant] = entry

    requested = int(receipt["n_requested_per_variant"])
    complete = all(int(summary[variant]["n"]) == requested for variant in VARIANTS)
    float_diverges = bool(summary["float_atomic"]["n"] > 0 and not summary["float_atomic"]["cross_process_identical"])
    integer_identical = summary["fixed_q15_int32_atomic"]["cross_process_identical"]
    integer_parity = bool(summary["fixed_q15_int32_atomic"].get("parity_all", False))
    decisive_positive = bool(complete and float_diverges and integer_identical and integer_parity)
    if not complete:
        verdict = "INCOMPLETE"
    elif decisive_positive:
        verdict = "REAL-L70-LEVER"
    elif float_diverges and not integer_identical:
        verdict = "L70-DEEPER-THAN-FP-REORDER"
    else:
        verdict = "INERT-CURIO"
    summary["decisive_positive"] = decisive_positive
    summary["complete"] = complete
    summary["overall_verdict"] = verdict
    summary["verdict_scope"] = (
        "INSTANCE x MLX-0.31.2 x M5-Max-Metal x render-R bicubic 384->874 one-axis "
        "transpose accumulation x Q15/int32 bounded fixture"
    )
    return summary


def run_parent(*, output: Path, n: int, resume: bool) -> dict[str, Any]:
    if n < 2:
        raise ValueError("n must be >=2 for a cross-process verdict")
    if resume and output.exists():
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA:
            raise ValueError(f"cannot resume incompatible receipt {output}")
        if int(receipt.get("n_requested_per_variant", -1)) != n:
            raise ValueError("resume n does not match existing receipt")
        recorded_probe_sha = receipt.get("source_custody", {}).get("probe", {}).get("sha256")
        current_probe_sha = _sha256_file(Path(__file__))
        if recorded_probe_sha != current_probe_sha:
            raise ValueError(
                "resume probe bytes differ from receipt custody; start a fresh receipt "
                f"({recorded_probe_sha!r} != {current_probe_sha!r})"
            )
    else:
        receipt = _base_receipt(n)
        _atomic_write_json(output, receipt)

    receipt.pop("failure", None)
    receipt.pop("completed", None)
    _atomic_write_json(output, receipt)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for variant in VARIANTS:
        rows: list[dict[str, Any]] = receipt["trials"][variant]
        while len(rows) < n:
            trial_index = len(rows)
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", variant],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if process.returncode != 0:
                receipt["failure"] = {
                    "variant": variant,
                    "trial_index": trial_index,
                    "returncode": process.returncode,
                    "stderr_tail": process.stderr[-1200:],
                    "stdout_tail": process.stdout[-1200:],
                }
                _atomic_write_json(output, receipt)
                raise RuntimeError(f"child failed for {variant} trial {trial_index}: {process.stderr[-400:]}")
            row = json.loads(process.stdout.strip().splitlines()[-1])
            row["trial_index"] = trial_index
            rows.append(row)
            receipt["summary"] = _summarize(receipt)
            _atomic_write_json(output, receipt)

    receipt["summary"] = _summarize(receipt)
    receipt["completed"] = True
    _atomic_write_json(output, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=VARIANTS, help="internal: execute one MLX process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=N_PROCESSES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--numpy-only", action="store_true", help="emit the pure NumPy fixture contract")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.child:
        print(json.dumps(run_mlx_variant(args.child), sort_keys=True))
        return 0
    if args.numpy_only:
        fixture = build_resize_adjoint_fixture()
        print(
            json.dumps(
                {
                    "in_size": IN_SIZE,
                    "out_size": OUT_SIZE,
                    "lanes": LANES,
                    "q_bits": Q_BITS,
                    "max_abs_integer_accumulator": fixture["max_abs_integer_accumulator"],
                    "max_contributions_per_destination": fixture["max_contributions_per_destination"],
                    "authority_tolerance": fixture["authority_tolerance"],
                    "float_reorder_tolerance": fixture["float_reorder_tolerance"],
                    "int_reference_sha256": _hash_array(np.asarray(fixture["int_reference"], dtype=np.int32)),
                    "float_reference_sha256": _hash_array(np.asarray(fixture["float_reference"], dtype=np.float32)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt = run_parent(output=args.output, n=args.n, resume=args.resume)
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
