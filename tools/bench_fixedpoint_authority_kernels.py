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
from tac.local_acceleration.argmax_tie_snap import (  # noqa: E402
    class_pair_tie_snap_argmax_mlx,
    tie_snap_argmax_mlx,
)
from tac.local_acceleration.metal_fixedpoint_verdict import (  # noqa: E402
    build_metal_fixedpoint_segnet_adapter,
    fixedpoint_verdict_signature,
)
from tac.local_acceleration.metal_mixed_int64_fixedpoint_verdict import (  # noqa: E402
    build_metal_mixed_int64_segnet_adapter,
    build_metal_weight_l1_int64_segnet_adapter,
    mixed_fixedpoint_verdict_signature,
    weight_l1_fixedpoint_verdict_signature,
)

SCHEMA = "metal_fixedpoint_segnet_n600.v1"
DEFAULT_CALIBRATION_RECEIPT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_INTEGER_PRECURSOR_RECEIPT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "exact_int64_fixedpoint_scorer_forward_n600.json"
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


def _load_integer_precursor(
    path: Path,
    *,
    bits: int,
    qdq_receipt_path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    schema = receipt.get("schema")
    summary = receipt.get("summary", {})
    manifest = receipt.get("model_manifest", {})
    if schema == "exact_int64_fixedpoint_scorer_n600.v1":
        if (
            summary.get("status") != "MEASURED"
            or summary.get("full_real_n600") is not True
            or summary.get("argmax_exact_admitted") is not True
            or int(summary.get("bits", -1)) != int(bits)
        ):
            raise ValueError("exact-int64 precursor has not admitted this full-n600 arm")
        if (
            int(manifest.get("bits", -1)) != int(bits)
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("accumulation") != "exact_signed_int64"
        ):
            raise ValueError("exact-int64 precursor model manifest differs")
    elif schema == "mixed_int64_fixedpoint_scorer_n600.v1":
        if (
            summary.get("status") != "MEASURED"
            or summary.get("full_real_n600") is not True
            or summary.get("argmax_exact_admitted") is not True
            or int(summary.get("minimum_bits", -1)) != int(bits)
            or int(summary.get("maximum_bits", -1)) != 30
        ):
            raise ValueError("mixed exact-int64 precursor has not admitted full n600")
        if (
            int(manifest.get("minimum_bits", -1)) != int(bits)
            or int(manifest.get("maximum_bits", -1)) != 30
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("accumulation") != "exact_signed_int64"
            or manifest.get("assignment_rule")
            != "largest_geometry_safe_bits_with_signed_int64_static_bound"
        ):
            raise ValueError("mixed exact-int64 precursor model manifest differs")
    elif schema == "weight_l1_int64_fixedpoint_scorer_n600.v1":
        if (
            summary.get("status") != "MEASURED"
            or summary.get("full_real_n600") is not True
            or summary.get("argmax_exact_admitted") is not True
            or int(summary.get("minimum_bits", -1)) != int(bits)
            or int(summary.get("maximum_bits", -1)) != 31
        ):
            raise ValueError("weight-L1 exact-int64 precursor has not admitted full n600")
        if (
            int(manifest.get("minimum_bits", -1)) != int(bits)
            or int(manifest.get("maximum_bits", -1)) != 31
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("accumulation") != "exact_signed_int64"
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
        ):
            raise ValueError("weight-L1 exact-int64 precursor manifest differs")
    elif schema == "weight_l1_tie_snap_scorer_n600.v1":
        selected_epsilon = summary.get("minimum_calibration_exact_epsilon")
        if (
            summary.get("status") != "MEASURED"
            or summary.get("full_real_n600") is not True
            or summary.get("argmax_exact_admitted") is not True
            or not summary.get("minimum_calibration_exact_arm")
            or selected_epsilon is None
            or float(selected_epsilon) <= 0.0
            or summary.get("selected_heldout_exact") is not True
            or summary.get("selected_full_exact") is not True
        ):
            raise ValueError("tie-snap exact-int64 precursor has not admitted full n600")
        if (
            int(manifest.get("minimum_bits", -1)) != int(bits)
            or int(manifest.get("maximum_bits", -1)) != 31
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("accumulation") != "exact_signed_int64"
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
            or receipt.get("contract", {}).get("decision_rule")
            != "lowest class index within epsilon of candidate maximum"
            or receipt.get("contract", {}).get("epsilon_selection")
            != "minimum calibration-exact epsilon; no heldout reselection"
        ):
            raise ValueError("tie-snap exact-int64 precursor manifest differs")
    elif schema == "weight_l1_class_pair_tie_snap_scorer_n600.v1":
        if (
            summary.get("status") != "MEASURED"
            or summary.get("full_real_n600") is not True
            or summary.get("argmax_exact_admitted") is not True
            or summary.get("design_exact") is not True
            or summary.get("second_validation_exact") is not True
        ):
            raise ValueError("class-pair tie-snap precursor has not admitted full n600")
        contract = receipt.get("contract", {})
        if (
            int(manifest.get("minimum_bits", -1)) != int(bits)
            or int(manifest.get("maximum_bits", -1)) != 31
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("accumulation") != "exact_signed_int64"
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
            or contract.get("design_split") != [0, 264]
            or contract.get("second_validation_split") != [264, 600]
            or contract.get("candidate_winner_class") != 4
            or contract.get("candidate_runner_class") != 0
            or contract.get("replacement_class") != 0
            or float(contract.get("epsilon", -1.0)) != float(2.0**-19)
            or contract.get("rule_frozen_before_second_validation_access") is not True
            or contract.get("second_validation_reselection") is not False
            or contract.get("runtime_label_or_frame_dependent") is not False
        ):
            raise ValueError("class-pair tie-snap exact-int64 precursor manifest differs")
    else:
        raise ValueError("incompatible exact-int64 precursor receipt")
    if receipt.get("custody", {}).get("qdq_precursor_sha256") != sha256_file(
        qdq_receipt_path
    ):
        raise ValueError("exact-int64 precursor is not bound to this QDQ receipt")
    return receipt


def _selected_tie_snap_epsilon(receipt: dict[str, Any] | None) -> float | None:
    rule = _selected_tie_snap_rule(receipt)
    return None if rule is None else float(rule["epsilon"])


def _realized_precision_bounds(receipt: dict[str, Any]) -> tuple[int, int]:
    manifest = receipt.get("model_manifest", {})
    histogram = manifest.get("precision_histogram")
    if not isinstance(histogram, dict):
        raise ValueError("exact-int64 precursor lacks a realized precision histogram")
    try:
        realized = sorted(
            int(raw_bits)
            for raw_bits, raw_count in histogram.items()
            if int(raw_count) > 0
        )
        count = sum(int(raw_count) for raw_count in histogram.values())
    except (TypeError, ValueError) as exc:
        raise ValueError("exact-int64 precursor precision histogram is malformed") from exc
    configured_minimum = int(manifest.get("minimum_bits", -1))
    configured_maximum = int(manifest.get("maximum_bits", -1))
    if (
        not realized
        or count != int(manifest.get("converted_conv2d_count", -1))
        or realized[0] < configured_minimum
        or realized[-1] > configured_maximum
    ):
        raise ValueError("exact-int64 precursor precision histogram coverage differs")
    return realized[0], realized[-1]


def _selected_tie_snap_rule(receipt: dict[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    schema = receipt.get("schema")
    if schema == "weight_l1_tie_snap_scorer_n600.v1":
        value = receipt.get("summary", {}).get("minimum_calibration_exact_epsilon")
        if value is None or float(value) <= 0.0:
            raise ValueError("tie-snap precursor lacks a positive selected epsilon")
        return {"kind": "global_lowest_class", "epsilon": float(value)}
    if schema != "weight_l1_class_pair_tie_snap_scorer_n600.v1":
        return None
    contract = receipt.get("contract", {})
    value = contract.get("epsilon")
    if value is None or float(value) <= 0.0:
        raise ValueError("class-pair tie-snap precursor lacks a positive epsilon")
    return {
        "kind": "ordered_class_pair",
        "epsilon": float(value),
        "winner_class": int(contract["candidate_winner_class"]),
        "runner_class": int(contract["candidate_runner_class"]),
        "replacement_class": int(contract["replacement_class"]),
    }


def _tie_snap_assignment_suffix(rule: dict[str, Any] | None) -> str:
    if rule is None:
        return ""
    epsilon = float(rule["epsilon"])
    if rule["kind"] == "global_lowest_class":
        return f"_tie_snap_{epsilon.hex()}"
    return (
        f"_class_pair_tie_snap_w{rule['winner_class']}_r{rule['runner_class']}"
        f"_to{rule['replacement_class']}_eps_{epsilon.hex()}"
    )


def _load_calibration(
    path: Path,
    *,
    bits: int | None,
    integer_precursor_path: Path | None = None,
) -> tuple[dict[str, float], dict[str, Any], str, int, dict[str, Any] | None]:
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
    integer_precursor: dict[str, Any] | None = None
    if bits is None:
        if isinstance(selected_arm, str) and selected_arm.startswith("w"):
            bits = int(selected_arm[1 : selected_arm.index("a")])
        elif integer_precursor_path is not None:
            precursor_payload = json.loads(integer_precursor_path.read_text(encoding="utf-8"))
            bits = int(precursor_payload.get("summary", {}).get("bits", -1))
        else:
            raise ValueError("calibration receipt has no exact-argmax arm")
    arm = f"w{bits}a{bits}"
    arm_row = receipt.get("summary", {}).get("arms", {}).get(arm, {})
    if arm_row.get("argmax_exact_admitted") is not True:
        if integer_precursor_path is None:
            raise ValueError(f"calibration receipt has no exact-argmax-admitted {arm} arm")
        if arm_row.get("training_tolerance_admitted") is not True:
            raise ValueError(f"calibration receipt lacks a tolerance-admitted {arm} precursor")
        integer_precursor = _load_integer_precursor(
            integer_precursor_path,
            bits=int(bits),
            qdq_receipt_path=path,
        )
    absmax = receipt.get("calibration", {}).get("segnet_operator_absmax")
    if not isinstance(absmax, dict) or not absmax:
        raise ValueError("calibration receipt lacks SegNet operator absmax")
    return (
        {str(key): float(value) for key, value in absmax.items()},
        receipt,
        activation_scale_mode,
        int(bits),
        integer_precursor,
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

    (
        calibration,
        calibration_receipt,
        activation_scale_mode,
        selected_bits,
        integer_precursor,
    ) = _load_calibration(
        args.calibration_receipt,
        bits=args.bits,
        integer_precursor_path=args.integer_precursor_receipt,
    )
    tie_snap_rule = _selected_tie_snap_rule(integer_precursor)
    tie_snap_epsilon = _selected_tie_snap_epsilon(integer_precursor)
    model, weights = _load_segnet()
    if args.weight_l1_safe:
        if integer_precursor is None or integer_precursor.get("schema") not in {
            "weight_l1_int64_fixedpoint_scorer_n600.v1",
            "weight_l1_tie_snap_scorer_n600.v1",
            "weight_l1_class_pair_tie_snap_scorer_n600.v1",
        }:
            raise ValueError(
                "weight-L1 Metal child requires its admitted exact-int64 precursor"
            )
        adapter, adapter_manifest = build_metal_weight_l1_int64_segnet_adapter(
            model,
            operator_absmax=calibration,
            require_opt_in=False,
        )
        realized_minimum_bits, realized_maximum_bits = _realized_precision_bounds(
            integer_precursor
        )
        precision_assignment = (
            f"frozen_weight_l1_safe_W{realized_minimum_bits}_to_W{realized_maximum_bits}"
        )
        precision_assignment += _tie_snap_assignment_suffix(tie_snap_rule)
        reported_bits = realized_minimum_bits
    elif args.mixed_geometry_safe:
        adapter, adapter_manifest = build_metal_mixed_int64_segnet_adapter(
            model,
            operator_absmax=calibration,
            require_opt_in=False,
        )
        precision_assignment = "geometry_safe_W26_to_W30"
        reported_bits = selected_bits
    else:
        adapter, adapter_manifest = build_metal_fixedpoint_segnet_adapter(
            model,
            operator_absmax=calibration,
            bits=selected_bits,
            activation_scale_mode=activation_scale_mode,
            require_opt_in=False,
        )
        precision_assignment = f"uniform_W{selected_bits}A{selected_bits}"
        reported_bits = selected_bits
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
            if tie_snap_rule is None:
                decision = None
            elif tie_snap_rule["kind"] == "global_lowest_class":
                decision = tie_snap_argmax_mlx(candidate, epsilon=tie_snap_epsilon)
            else:
                decision = class_pair_tie_snap_argmax_mlx(
                    candidate,
                    epsilon=tie_snap_epsilon,
                    winner_class=int(tie_snap_rule["winner_class"]),
                    runner_class=int(tie_snap_rule["runner_class"]),
                )
            if decision is None:
                mx.eval(candidate)
            else:
                mx.eval(candidate, decision)
            metal_seconds.append(time.perf_counter() - metal_started)
            candidate_np = np.asarray(candidate, dtype=np.float32)
            if not np.all(np.isfinite(candidate_np)):
                raise RuntimeError("custom Metal fixed-point SegNet emitted non-finite logits")
            plain_candidate_argmax = np.argmax(candidate_np, axis=-1)[0]
            candidate_argmax = (
                np.asarray(decision, dtype=np.int64)[0]
                if decision is not None
                else plain_candidate_argmax
            )
            flips = candidate_argmax != reference_argmax
            argmax_hash = _hash_array(candidate_argmax)
            digest.update(f"{pair_index}:{argmax_hash}\n".encode("ascii"))
            row = {
                "pair_index": pair_index,
                "flips": int(np.count_nonzero(flips)),
                "pixels": int(flips.size),
                "flip_fraction": float(np.mean(flips)),
                "argmax_sha256": argmax_hash,
                "tie_snap_epsilon": tie_snap_epsilon,
                "tie_snap_rule": tie_snap_rule,
                "tie_snap_pixels": int(
                    np.count_nonzero(candidate_argmax != plain_candidate_argmax)
                ),
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
        "bits": reported_bits,
        "qdq_precursor_bits": selected_bits,
        "precision_assignment": precision_assignment,
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
            "integer_precursor_receipt_sha256": (
                sha256_file(args.integer_precursor_receipt)
                if integer_precursor is not None
                else None
            ),
            "integer_precursor_fingerprint": (
                integer_precursor.get("fingerprint") if integer_precursor is not None else None
            ),
            "backend_sha256": sha256_file(
                REPO
                / (
                    "src/tac/local_acceleration/metal_mixed_int64_fixedpoint_verdict.py"
                    if args.mixed_geometry_safe or args.weight_l1_safe
                    else "src/tac/local_acceleration/metal_fixedpoint_verdict.py"
                )
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
    # The source is a fixed, exhaustively measured n600 corpus.  Direct exact
    # argmax on every pixel plus cross-process identity is the decision
    # authority; the interval enclosure is a separately reported sufficient
    # certificate and can be conservative around exact/tiny ties.
    admitted = bool(complete and full_real_n600 and exact and deterministic and positive_speed)
    if blockers:
        verdict = "BLOCKED_NOT_MEASURED"
        scope = "ENVIRONMENT: attempted process has no evaluated Metal"
    elif not complete:
        verdict = "INCOMPLETE"
        scope = "INSTANCE: cross-process real-n600 coverage"
    elif admitted:
        verdict = "ADMIT_CANDIDATE_AUTHORITY_FILTER"
        scope = (
            "n600 SOURCE-CORPUS INSTANCE: custom Metal fixed-point SegNet on this host; "
            "actual evolving witness frames and terminal contest CPU/CUDA remain separate gates"
        )
    else:
        verdict = "HOLD_METAL_FIXEDPOINT_FORMULATION"
        scope = (
            f"FORMULATION: {receipt['contract']['precision_assignment']} "
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
        "strict_interval_certificate_required_for_admission": False,
        "certificate_policy": (
            "report the sufficient enclosure separately; exhaustive exact n600 argmax plus "
            "cross-process identity owns local candidate admission"
        ),
        "positive_speed": positive_speed,
        "admitted_candidate_authority_filter": admitted,
        "fidelity": fidelity,
        "overall_verdict": verdict,
        "verdict_scope": scope,
    }


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    (
        calibration,
        calibration_receipt,
        activation_scale_mode,
        selected_bits,
        integer_precursor,
    ) = _load_calibration(
        args.calibration_receipt,
        bits=args.bits,
        integer_precursor_path=args.integer_precursor_receipt,
    )
    tie_snap_rule = _selected_tie_snap_rule(integer_precursor)
    tie_snap_epsilon = _selected_tie_snap_epsilon(integer_precursor)
    if args.weight_l1_safe:
        if integer_precursor is None or integer_precursor.get("schema") not in {
            "weight_l1_int64_fixedpoint_scorer_n600.v1",
            "weight_l1_tie_snap_scorer_n600.v1",
            "weight_l1_class_pair_tie_snap_scorer_n600.v1",
        }:
            raise ValueError(
                "weight-L1 Metal gate requires its admitted exact-int64 or tie-snap precursor"
            )
        kernel_signature = weight_l1_fixedpoint_verdict_signature()
        realized_minimum_bits, realized_maximum_bits = _realized_precision_bounds(
            integer_precursor
        )
        precision_assignment = (
            f"frozen_weight_l1_safe_W{realized_minimum_bits}_to_W{realized_maximum_bits}"
        )
        precision_assignment += _tie_snap_assignment_suffix(tie_snap_rule)
        reported_bits = realized_minimum_bits
        if tie_snap_rule is not None:
            kernel_signature = {
                **kernel_signature,
                "decision_head": tie_snap_rule["kind"],
                "tie_snap_rule": tie_snap_rule,
                "tie_snap_selection": (
                    "frozen design-split rule; disjoint validation without reselection"
                ),
            }
    elif args.mixed_geometry_safe:
        if integer_precursor is None or integer_precursor.get("schema") != (
            "mixed_int64_fixedpoint_scorer_n600.v1"
        ):
            raise ValueError("mixed Metal gate requires an admitted mixed exact-int64 precursor")
        kernel_signature = mixed_fixedpoint_verdict_signature()
        precision_assignment = "geometry_safe_W26_to_W30"
        reported_bits = selected_bits
    else:
        kernel_signature = fixedpoint_verdict_signature()
        precision_assignment = f"uniform_W{selected_bits}A{selected_bits}"
        reported_bits = selected_bits
    contract = {
        "bits": reported_bits,
        "qdq_precursor_bits": selected_bits,
        "maximum_realized_bits": (
            realized_maximum_bits if args.weight_l1_safe else reported_bits
        ),
        "precision_assignment": precision_assignment,
        "mixed_geometry_safe": bool(args.mixed_geometry_safe),
        "weight_l1_safe": bool(args.weight_l1_safe),
        "tie_snap_epsilon": tie_snap_epsilon,
        "tie_snap_rule": tie_snap_rule,
        "activation_scale_mode": activation_scale_mode,
        "pair_start": args.pair_start,
        "pair_count": args.pair_count,
        "input_surface": "real source gt_f1 pairs from the bound n600 cache",
        "n_processes": args.n_processes,
        "calibration_digest": hashlib.sha256(
            json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qdq_receipt_sha256": sha256_file(args.calibration_receipt),
        "qdq_receipt_fingerprint": calibration_receipt.get("fingerprint"),
        "qdq_argmax_exact_precursor": bool(
            calibration_receipt.get("summary", {})
            .get("arms", {})
            .get(f"w{selected_bits}a{selected_bits}", {})
            .get("argmax_exact_admitted")
        ),
        "exact_int64_cpu_precursor_sha256": (
            sha256_file(args.integer_precursor_receipt)
            if integer_precursor is not None
            else None
        ),
        "exact_int64_cpu_precursor_fingerprint": (
            integer_precursor.get("fingerprint") if integer_precursor is not None else None
        ),
        "kernel_signature": kernel_signature,
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
        if args.integer_precursor_receipt is not None:
            argv.extend(
                ["--integer-precursor-receipt", str(args.integer_precursor_receipt)]
            )
        if args.mixed_geometry_safe:
            argv.append("--mixed-geometry-safe")
        if args.weight_l1_safe:
            argv.append("--weight-l1-safe")
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
    parser.add_argument(
        "--mixed-geometry-safe",
        action="store_true",
        help="run the geometry-derived W26..W30 exact-int64 SegNet contract",
    )
    parser.add_argument(
        "--weight-l1-safe",
        action="store_true",
        help="run the configured W26..W31 frozen-weight-L1 exact-int64 SegNet contract",
    )
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=600)
    parser.add_argument("--n-processes", type=int, default=10)
    parser.add_argument("--calibration-receipt", type=Path, default=DEFAULT_CALIBRATION_RECEIPT)
    parser.add_argument(
        "--integer-precursor-receipt",
        type=Path,
        default=None,
        help=(
            "optional exact-int64 CPU receipt permitting an actual-integer Metal arm when the "
            "QDQ/fp32 precursor is tolerance-admitted but not exact"
        ),
    )
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.mixed_geometry_safe and args.weight_l1_safe:
        raise ValueError("choose at most one mixed precision assignment")
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
