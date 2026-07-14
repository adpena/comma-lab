#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Host-only n600 parity/timing receipt for the integer render-R adjoint backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.local_acceleration import metal_integer_r_adjoint as integer_r  # noqa: E402
from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.local_acceleration.metal_fused_r_operator import (  # noqa: E402
    _fused_r_metal_vjp,
)

DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_FULL_R_RECEIPT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600.json"
)
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/integer_r_backend_n600.json"
)


def _load_probe_module() -> Any:
    path = REPO / "tools/probe_pythagorean_exact_arithmetic_bitident.py"
    spec = importlib.util.spec_from_file_location("task494_full_r_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _time_call(fn: Any) -> tuple[np.ndarray, float]:
    import mlx.core as mx

    started = time.perf_counter()
    output = fn()
    mx.eval(output)
    elapsed = time.perf_counter() - started
    return np.asarray(output, dtype=np.float32), float(elapsed)


def _validate_full_r_receipt(path: Path, *, pair_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "pythagorean_exact_arithmetic_full_r_n600.v2":
        raise ValueError("full-R receipt schema mismatch")
    summary = receipt.get("summary", {})
    if pair_count == 600:
        if summary.get("overall_verdict") != "REAL-L70-LEVER-FULL-R-N600":
            raise ValueError("n600 backend timing requires the decisive full-R receipt")
        if not summary.get("complete") or not summary.get("decisive_positive"):
            raise ValueError("full-R receipt is incomplete")
        contract = receipt.get("contract", {})
        authority = summary.get("authority", {})
        integer = summary.get("fixed_q15_int32_atomic", {})
        if not (
            int(contract.get("pair_start", -1)) == 0
            and int(contract.get("pair_count", -1)) == 600
            and int(contract.get("frames", -1)) == 1200
            and authority.get("coverage_exact") is True
            and int(authority.get("frames", -1)) == 1200
            and integer.get("cross_process_identical") is True
            and integer.get("exact_numpy_int_corpus_parity") is True
        ):
            raise ValueError("full-R receipt lacks exact 0..599 x {f0,f1} custody")
    return receipt


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
    except Exception as exc:
        payload = {
            "schema": "integer_r_adjoint_backend_benchmark.v1",
            "status": "BLOCKED_NOT_MEASURED",
            "blocker": f"Metal unavailable: {type(exc).__name__}: {exc}",
            "verdict_scope": "ENVIRONMENT: this host lacks evaluated MLX Metal",
        }
        atomic_json(args.output, payload)
        return payload, 3

    source_receipt = _validate_full_r_receipt(args.full_r_receipt, pair_count=args.pair_count)
    probe = _load_probe_module()
    frames = probe._ordered_real_frames(
        gt_cache=args.gt_cache,
        pair_start=args.pair_start,
        pair_count=args.pair_count,
    )
    expected_integer_hashes = {
        (int(row["pair_index"]), str(row["member"])): str(row["integer_output_sha256"])
        for row in source_receipt.get("numpy_authority", {}).get("rows", [])
    }
    config = integer_r.IntegerRAdjointConfig(cotangent_unit=1.0)
    config.validate()
    integer_r.full_r_int32_proof()

    float_seconds: list[float] = []
    integer_seconds: list[float] = []
    maximum_error = 0.0
    sum_squared_error = 0.0
    elements = 0
    float_digest = hashlib.sha256()
    integer_digest = hashlib.sha256()
    deterministic_hashes: list[str] = []
    exact_numpy_integer_frames = 0
    frame_count = 0
    for frame_number, (pair_index, member, frame) in enumerate(frames):
        frame_count += 1
        fixture = probe.prepare_real_full_r_fixture(frame)
        low_mx = mx.array(fixture["low"][None], dtype=mx.float32)
        cot_mx = mx.array(fixture["cotangent"][None], dtype=mx.float32)
        mask_mx = mx.array(fixture["clip_mask"][None])

        float_output, float_elapsed = _time_call(
            lambda low_mx=low_mx, cot_mx=cot_mx: _fused_r_metal_vjp(
                low_mx,
                cot_mx,
                camera_hw=(874, 1164),
                output_hw=(384, 512),
                ste_round=True,
            )
        )
        integer_started = time.perf_counter()
        integer_state_mx = integer_r.integer_r_vjp_state_metal(
            cot_mx, mask_mx, config=config
        )
        integer_output_mx = integer_state_mx.astype(mx.float32) / float(
            1 << integer_r.STATE_BITS_BY_BOUNDARY[-1]
        )
        mx.eval(integer_state_mx, integer_output_mx)
        integer_elapsed = time.perf_counter() - integer_started
        integer_state = np.asarray(integer_state_mx, dtype=np.int32)
        integer_output = np.asarray(integer_output_mx, dtype=np.float32)
        if frame_number >= args.warmup_frames:
            float_seconds.append(float_elapsed)
            integer_seconds.append(integer_elapsed)
        difference = integer_output.astype(np.float64) - float_output.astype(np.float64)
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
        sum_squared_error += float(np.sum(difference * difference, dtype=np.float64))
        elements += int(difference.size)
        float_hash = _hash_array(float_output)
        integer_hash = _hash_array(integer_output)
        expected_integer_hash = expected_integer_hashes.get((pair_index, member))
        integer_state_authority = (
            integer_state[0] if integer_state.shape == (1, 384, 512, 3) else integer_state
        )
        if _hash_array(integer_state_authority) == expected_integer_hash:
            exact_numpy_integer_frames += 1
        float_digest.update(f"{pair_index}:{member}:{float_hash}\n".encode("ascii"))
        integer_digest.update(f"{pair_index}:{member}:{integer_hash}\n".encode("ascii"))

        if frame_number == 0:
            deterministic_hashes.append(integer_hash)
            for _ in range(args.determinism_repeats - 1):
                repeated, _ = _time_call(
                    lambda cot_mx=cot_mx, mask_mx=mask_mx: integer_r.integer_r_vjp_metal(
                        cot_mx, mask_mx, config=config
                    )
                )
                deterministic_hashes.append(_hash_array(repeated))

    float_median = float(np.median(float_seconds)) if float_seconds else None
    integer_median = float(np.median(integer_seconds)) if integer_seconds else None
    speedup = (
        float(float_median / integer_median)
        if float_median is not None and integer_median not in (None, 0.0)
        else None
    )
    derived_bound = float(
        source_receipt.get("derived_integer_error_bound", {}).get(
            "final_max_abs_error_bound", float("nan")
        )
    )
    full_coverage = (
        args.pair_start == 0 and args.pair_count == 600 and frame_count == 1200
    )
    measured = bool(float_seconds and integer_seconds)
    exact_numpy_integer_parity = bool(
        exact_numpy_integer_frames == frame_count and frame_count > 0
    )
    parity = bool(
        np.isfinite(derived_bound)
        and maximum_error <= derived_bound
        and exact_numpy_integer_parity
    )
    deterministic = len(set(deterministic_hashes)) == 1
    positive_speed = speedup is not None and speedup > 1.0
    admitted = bool(full_coverage and measured and parity and deterministic and positive_speed)
    payload = {
        "schema": "integer_r_adjoint_backend_benchmark.v1",
        "lane_id": "throughput_authority_ladder",
        "task_id": 494,
        "axis": "[macOS-MLX research-signal; non-promotable MEANS]",
        "status": "MEASURED",
        "score_claim": False,
        "pointer_moved": False,
        "promotion_eligible": False,
        "git_head": _git_head(),
        "host": platform.node(),
        "device": str(mx.default_device()),
        "coverage": {
            "pair_start": args.pair_start,
            "pair_count": args.pair_count,
            "frames": frame_count,
            "full_real_n600": full_coverage,
        },
        "parity": {
            "dequantized_max_abs_error_vs_fixed_float_metal": maximum_error,
            "dequantized_rmse_vs_fixed_float_metal": (
                float(np.sqrt(sum_squared_error / elements)) if elements else None
            ),
            "derived_numpy_fp32_bound": derived_bound,
            "within_derived_bound": parity,
            "exact_numpy_integer_frames": exact_numpy_integer_frames,
            "exact_numpy_integer_corpus_parity": exact_numpy_integer_parity,
            "integer_first_frame_repeat_hashes": deterministic_hashes,
            "integer_repeat_bit_identical": deterministic,
            "float_corpus_sha256": float_digest.hexdigest(),
            "integer_corpus_sha256": integer_digest.hexdigest(),
        },
        "timing": {
            "warmup_frames": args.warmup_frames,
            "timed_frames": len(float_seconds),
            "float_fixed_order_median_seconds_per_frame": float_median,
            "integer_order_independent_median_seconds_per_frame": integer_median,
            "speedup_x": speedup,
            "positive_speed": positive_speed,
            "synchronization": "mx.eval after every call",
        },
        "admission": {
            "admitted_for_training": admitted,
            "requirements": {
                "full_real_n600": full_coverage,
                "within_derived_bound": parity,
                "exact_numpy_integer_corpus_parity": exact_numpy_integer_parity,
                "repeat_bit_identical": deterministic,
                "speedup_gt_1": positive_speed,
            },
            "terminal_score_authority": "unchanged exact contest CPU/CUDA archive replay",
        },
        "source_custody": {
            "backend": {
                "path": "src/tac/local_acceleration/metal_integer_r_adjoint.py",
                "sha256": sha256_file(
                    REPO / "src/tac/local_acceleration/metal_integer_r_adjoint.py"
                ),
            },
            "probe": {
                "path": "tools/probe_pythagorean_exact_arithmetic_bitident.py",
                "sha256": sha256_file(
                    REPO / "tools/probe_pythagorean_exact_arithmetic_bitident.py"
                ),
            },
            "full_r_receipt": {
                "path": str(args.full_r_receipt),
                "sha256": sha256_file(args.full_r_receipt),
            },
            "gt_cache": {"path": str(args.gt_cache), "sha256": sha256_file(args.gt_cache)},
        },
        "verdict": "ADMIT_TRAINING_BACKEND" if admitted else "HOLD_TRAINING_BACKEND",
        "verdict_scope": (
            "n600 INSTANCE: integer render-R VJP custom Metal gather backend on this host"
        ),
    }
    atomic_json(args.output, payload)
    return payload, 0 if admitted else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--full-r-receipt", type=Path, default=DEFAULT_FULL_R_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=600)
    parser.add_argument("--warmup-frames", type=int, default=4)
    parser.add_argument("--determinism-repeats", type=int, default=10)
    return parser


def main() -> int:
    payload, code = run(_parser().parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
