#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-process real-n600 host gate for custom Metal fixed-point SegNet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
    stored_npy_memmap,
)
from tac.local_acceleration.metal_fixedpoint_verdict import (  # noqa: E402
    build_metal_fixedpoint_segnet_adapter,
    fixedpoint_verdict_signature,
)

SCHEMA = "metal_fixedpoint_segnet_n600.v1"
DEFAULT_CALIBRATION_RECEIPT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "metal_dynamic_fixedpoint_segnet_n600.json"
)
N600 = 600
TOLERANCE = 3.3e-5


def _hash_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_segnet() -> tuple[Any, Path]:
    import torch
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    model = SegNet().eval().cpu()
    weights = Path(segnet_sd_path)
    model.load_state_dict(load_file(str(weights), device="cpu"))
    return model, weights


def _load_calibration(
    path: Path, *, bits: int | None
) -> tuple[dict[str, float], dict[str, Any], str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    contract = receipt.get("contract", {})
    activation_scale_mode = str(
        contract.get("activation_scale_mode")
        or (
            "fixed_calibration"
            if receipt.get("schema") == "fixedpoint_scorer_forward_n600.v2"
            else ""
        )
    )
    expected_schema = {
        "fixed_calibration": "fixedpoint_scorer_forward_n600.v2",
        "dynamic_exact_absmax": "dynamic_fixedpoint_scorer_forward_n600.v1",
    }.get(activation_scale_mode)
    if receipt.get("schema") != expected_schema:
        raise ValueError("incompatible calibration receipt")
    if receipt.get("summary", {}).get("status") != "MEASURED":
        raise ValueError("calibrated scorer receipt is incomplete")
    if receipt.get("summary", {}).get("full_real_n600") is not True:
        raise ValueError("calibrated scorer receipt lacks full real-n600 custody")
    control_rows = receipt.get("arms", {}).get("fp32_control", {}).get("segnet_rows", [])
    control_indices = [int(row.get("pair_index", -1)) for row in control_rows]
    if (
        len(control_rows) != N600
        or set(control_indices) != set(range(N600))
        or any(not row.get("reference_argmax_sha256") for row in control_rows)
    ):
        raise ValueError("calibrated scorer receipt lacks exact 0..599 control hashes")
    selected_arm = receipt.get("summary", {}).get("minimum_argmax_exact_arm")
    if bits is None:
        if not isinstance(selected_arm, str) or not selected_arm.startswith("w"):
            raise ValueError("calibration receipt has no exact-argmax arm")
        bits = int(selected_arm[1 : selected_arm.index("a")])
    arm = f"w{bits}a{bits}"
    arm_row = receipt.get("summary", {}).get("arms", {}).get(arm, {})
    if arm_row.get("argmax_exact_admitted") is not True:
        raise ValueError(f"calibration receipt has no exact-argmax-admitted {arm} arm")
    absmax = receipt.get("calibration", {}).get("segnet_operator_absmax")
    if not isinstance(absmax, dict) or not absmax:
        raise ValueError("calibration receipt lacks SegNet operator absmax")
    return (
        {str(key): float(value) for key, value in absmax.items()},
        receipt,
        activation_scale_mode,
        int(bits),
    )


def _seg_input(frame_hwc: np.ndarray) -> tuple[Any, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    nchw = torch.from_numpy(np.ascontiguousarray(frame_hwc.transpose(2, 0, 1))).unsqueeze(0)
    resized = functional.interpolate(nchw.to(torch.float32), size=(384, 512), mode="bilinear")
    nhwc = np.ascontiguousarray(resized.numpy().transpose(0, 2, 3, 1))
    return resized, nhwc


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(probe)
        if float(probe.item()) != 28.0:
            raise RuntimeError("evaluated Metal probe returned wrong value")
    except Exception as exc:
        return {
            "status": "BLOCKED_NOT_MEASURED",
            "blocker": f"evaluated Metal unavailable: {type(exc).__name__}: {exc}",
            "verdict_scope": "ENVIRONMENT: no evaluated MLX Metal device",
        }
    import torch

    calibration, calibration_receipt, activation_scale_mode, selected_bits = _load_calibration(
        args.calibration_receipt, bits=args.bits
    )
    model, weights = _load_segnet()
    adapter, adapter_manifest = build_metal_fixedpoint_segnet_adapter(
        model,
        operator_absmax=calibration,
        bits=selected_bits,
        activation_scale_mode=activation_scale_mode,
        require_opt_in=False,
    )
    frame1 = stored_npy_memmap(args.gt_cache, "gt_f1.npy")
    labels = stored_npy_memmap(args.gt_cache, "lstars.npy")
    margins = stored_npy_memmap(args.gt_cache, "margins.npy")
    if tuple(frame1.shape) != (600, 874, 1164, 3):
        raise ValueError("gt_f1 cache geometry mismatch")
    stop = args.pair_start + args.pair_count
    digest = hashlib.sha256()
    per_pair: list[dict[str, Any]] = []
    metal_seconds: list[float] = []
    cpu_seconds: list[float] = []
    maximum_error = 0.0
    sum_squared_error = 0.0
    error_elements = 0
    uncertified = 0
    cache_mismatch_pixels = 0
    cache_mismatch_pairs = 0
    with torch.inference_mode():
        for pair_index in range(args.pair_start, stop):
            cpu_input, nhwc = _seg_input(np.asarray(frame1[pair_index]))
            reference_logits = None
            cached_argmax = np.asarray(labels[pair_index])
            if args.fidelity:
                cpu_started = time.perf_counter()
                reference_logits = model(cpu_input)
                cpu_seconds.append(time.perf_counter() - cpu_started)
                reference_argmax = reference_logits.argmax(dim=1)[0].cpu().numpy()
                cache_delta = int(np.count_nonzero(reference_argmax != cached_argmax))
                cache_mismatch_pixels += cache_delta
                cache_mismatch_pairs += int(cache_delta > 0)
                top2 = reference_logits.topk(2, dim=1).values
                reference_margin = (
                    (top2[:, 0] - top2[:, 1]).clamp_min(0.0)[0].cpu().numpy()
                )
            else:
                reference_argmax = cached_argmax
                reference_margin = np.asarray(margins[pair_index])
            x = mx.array(nhwc, dtype=mx.float32)
            metal_started = time.perf_counter()
            candidate = adapter(x)
            mx.eval(candidate)
            metal_seconds.append(time.perf_counter() - metal_started)
            candidate_np = np.asarray(candidate, dtype=np.float32)
            if not np.all(np.isfinite(candidate_np)):
                raise RuntimeError("custom Metal fixed-point SegNet emitted non-finite logits")
            candidate_argmax = np.argmax(candidate_np, axis=-1)[0]
            flips = candidate_argmax != reference_argmax
            argmax_hash = _hash_array(candidate_argmax)
            digest.update(f"{pair_index}:{argmax_hash}\n".encode("ascii"))
            row = {
                "pair_index": pair_index,
                "flips": int(np.count_nonzero(flips)),
                "pixels": int(flips.size),
                "flip_fraction": float(np.mean(flips)),
                "argmax_sha256": argmax_hash,
                "comparison_reference": (
                    "one-thread CPU-Torch fp32 control" if args.fidelity else "legacy cache audit"
                ),
            }
            if reference_logits is not None:
                reference_nhwc = reference_logits.cpu().numpy().transpose(0, 2, 3, 1)
                error = np.abs(candidate_np - reference_nhwc)
                class_error = np.max(error, axis=-1)[0]
                pair_uncertified = reference_margin <= 2.0 * class_error
                difference = candidate_np.astype(np.float64) - reference_nhwc.astype(np.float64)
                maximum_error = max(maximum_error, float(np.max(error)))
                sum_squared_error += float(np.sum(difference * difference, dtype=np.float64))
                error_elements += int(difference.size)
                uncertified += int(np.count_nonzero(pair_uncertified))
                row.update(
                    {
                        "max_abs_logit_error": float(np.max(error)),
                        "uncertified_pixels": int(np.count_nonzero(pair_uncertified)),
                    }
                )
            per_pair.append(row)
    flips = sum(int(row["flips"]) for row in per_pair)
    pixels = sum(int(row["pixels"]) for row in per_pair)
    worst = max(per_pair, key=lambda row: (row["flip_fraction"], -row["pair_index"]))
    return {
        "status": "MEASURED",
        "fidelity": bool(args.fidelity),
        "bits": selected_bits,
        "activation_scale_mode": activation_scale_mode,
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "argmax_corpus_sha256": digest.hexdigest(),
        "flips": flips,
        "pixels": pixels,
        "aggregate_flip_fraction": float(flips / pixels),
        "worst_pair_index": int(worst["pair_index"]),
        "worst_pair_flip_fraction": float(worst["flip_fraction"]),
        "per_pair": per_pair if args.fidelity else None,
        "certificate": (
            {
                "max_abs_logit_error": maximum_error,
                "rmse_logit_error": float(np.sqrt(sum_squared_error / error_elements)),
                "uncertified_pixels": uncertified,
                "uncertified_fraction": float(uncertified / pixels),
                "strict_interval_certified_fraction": float(1.0 - uncertified / pixels),
                "rule": "cached top1-top2 margin > 2*max_class_abs_error",
            }
            if args.fidelity
            else None
        ),
        "legacy_cache_custody": {
            "mismatch_pairs": cache_mismatch_pairs,
            "argmax_mismatch_pixels": cache_mismatch_pixels,
            "role": "audit only; one-thread computed control owns fidelity",
        },
        "timing": {
            "metal_median_seconds_per_pair": statistics.median(metal_seconds),
            "metal_total_seconds": sum(metal_seconds),
            "cpu_median_seconds_per_pair": (
                statistics.median(cpu_seconds) if cpu_seconds else None
            ),
            "cpu_to_metal_speedup_x": (
                statistics.median(cpu_seconds) / statistics.median(metal_seconds)
                if cpu_seconds
                else None
            ),
            "synchronization": "mx.eval after each candidate forward",
        },
        "adapter_manifest": adapter_manifest,
        "device": str(mx.default_device()),
        "host": platform.node(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_custody": {
            "weights_sha256": sha256_file(weights),
            "calibration_receipt_sha256": sha256_file(args.calibration_receipt),
            "calibration_receipt_fingerprint": calibration_receipt.get("fingerprint"),
            "backend_sha256": sha256_file(
                REPO / "src/tac/local_acceleration/metal_fixedpoint_verdict.py"
            ),
        },
    }


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = list(receipt.get("trials", []))
    blockers = [row for row in rows if row.get("status") == "BLOCKED_NOT_MEASURED"]
    measured = [row for row in rows if row.get("status") == "MEASURED"]
    hashes = [row["argmax_corpus_sha256"] for row in measured]
    fidelity = next((row for row in measured if row.get("fidelity")), None)
    complete = len(measured) == int(receipt["contract"]["n_processes"])
    full_real_n600 = bool(
        int(receipt["contract"]["pair_start"]) == 0
        and int(receipt["contract"]["pair_count"]) == N600
        and all(
            int(row.get("pair_start", -1)) == 0
            and int(row.get("pair_count", -1)) == N600
            for row in measured
        )
    )
    exact = bool(fidelity and int(fidelity["flips"]) == 0)
    tolerance = bool(
        fidelity
        and float(fidelity["aggregate_flip_fraction"]) <= TOLERANCE
        and float(fidelity["worst_pair_flip_fraction"]) <= TOLERANCE
    )
    deterministic = bool(complete and len(set(hashes)) == 1)
    certified = bool(
        fidelity and int(fidelity.get("certificate", {}).get("uncertified_pixels", -1)) == 0
    )
    positive_speed = bool(
        fidelity
        and float(fidelity.get("timing", {}).get("cpu_to_metal_speedup_x") or 0.0) > 1.0
    )
    admitted = bool(
        complete
        and full_real_n600
        and exact
        and deterministic
        and certified
        and positive_speed
    )
    if blockers:
        verdict = "BLOCKED_NOT_MEASURED"
        scope = "ENVIRONMENT: attempted process has no evaluated Metal"
    elif not complete:
        verdict = "INCOMPLETE"
        scope = "INSTANCE: cross-process real-n600 coverage"
    elif admitted:
        verdict = "ADMIT_CANDIDATE_AUTHORITY_FILTER"
        scope = (
            "n600 INSTANCE: custom Metal fixed-point SegNet on this host; terminal contest "
            "CPU/CUDA replay still required"
        )
    else:
        verdict = "HOLD_METAL_FIXEDPOINT_FORMULATION"
        scope = (
            f"FORMULATION: uniform W{receipt['contract']['bits']}A{receipt['contract']['bits']} "
            f"{receipt['contract']['activation_scale_mode']} direct-int64 Metal SegNet; "
            "not the fixed-point family"
        )
    return {
        "complete": complete,
        "full_real_n600": full_real_n600,
        "measured_processes": len(measured),
        "unique_argmax_corpus_hashes": len(set(hashes)),
        "cross_process_argmax_identical": deterministic,
        "argmax_exact": exact,
        "training_tolerance": tolerance,
        "strict_interval_certified": certified,
        "positive_speed": positive_speed,
        "admitted_candidate_authority_filter": admitted,
        "fidelity": fidelity,
        "overall_verdict": verdict,
        "verdict_scope": scope,
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    calibration, calibration_receipt, activation_scale_mode, selected_bits = (
        _load_calibration(args.calibration_receipt, bits=args.bits)
    )
    contract = {
        "bits": selected_bits,
        "activation_scale_mode": activation_scale_mode,
        "pair_start": args.pair_start,
        "pair_count": args.pair_count,
        "n_processes": args.n_processes,
        "calibration_digest": hashlib.sha256(
            json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qdq_receipt_sha256": sha256_file(args.calibration_receipt),
        "qdq_receipt_fingerprint": calibration_receipt.get("fingerprint"),
        "kernel_signature": fixedpoint_verdict_signature(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume Metal fixed-point receipt fingerprint differs")
    else:
        receipt = {
            "schema": SCHEMA,
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[macOS-MLX Metal research-signal; non-promotable MEANS]",
            "score_claim": False,
            "pointer_moved": False,
            "git_head": _git_head(),
            "contract": contract,
            "fingerprint": fingerprint,
            "trials": [],
        }
        atomic_json(args.output, receipt)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    while len(receipt["trials"]) < args.n_processes:
        trial_index = len(receipt["trials"])
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--bits",
            str(selected_bits),
            "--pair-start",
            str(args.pair_start),
            "--pair-count",
            str(args.pair_count),
            "--calibration-receipt",
            str(args.calibration_receipt),
            "--gt-cache",
            str(args.gt_cache),
        ]
        if trial_index == 0:
            argv.append("--fidelity")
        process = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
        )
        if process.returncode not in (0, 3):
            raise RuntimeError(
                f"Metal fixed-point child {trial_index} failed: {process.stderr[-1600:]}"
            )
        row = json.loads(process.stdout.strip().splitlines()[-1])
        row["trial_index"] = trial_index
        receipt["trials"].append(row)
        receipt["summary"] = _summary(receipt)
        atomic_json(args.output, receipt)
        if row.get("status") == "BLOCKED_NOT_MEASURED":
            break
    receipt["summary"] = _summary(receipt)
    receipt["completed"] = receipt["summary"]["complete"]
    atomic_json(args.output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--fidelity", action="store_true")
    parser.add_argument("--bits", type=int)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=600)
    parser.add_argument("--n-processes", type=int, default=10)
    parser.add_argument("--calibration-receipt", type=Path, default=DEFAULT_CALIBRATION_RECEIPT)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.pair_start < 0 or args.pair_count <= 0 or args.pair_start + args.pair_count > N600:
        raise ValueError("invalid pair interval")
    if args.child:
        payload = run_child(args)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "MEASURED" else 3
    payload = run_parent(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
