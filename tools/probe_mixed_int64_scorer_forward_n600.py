#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable n600 geometry-safe mixed W26..W30 exact-int64 SegNet probe."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_exact_int64_scorer_forward_n600 import (  # noqa: E402
    _fingerprint,
    _storage_preflight,
    _validate_qdq_precursor,
)
from probe_fixedpoint_scorer_forward_n600 import (  # noqa: E402
    N600,
    TRAINING_TOLERANCE,
    _aggregate_cache_custody,
    _aggregate_rows,
    _cache_arrays,
    _git_head,
    _hash_array,
    _load_models,
    _pair_inputs,
    _seg_row,
)

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.local_acceleration.mixed_int64_fixedpoint_scorer import (  # noqa: E402
    MAXIMUM_BITS,
    MINIMUM_BITS,
    build_mixed_int64_model,
)

SCHEMA = "mixed_int64_fixedpoint_scorer_n600.v1"
ARM = "mixed_w26_w30_geometry_safe"
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_QDQ = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json"
)
DEFAULT_UNIFORM = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/exact_int64_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/mixed_int64_fixedpoint_scorer_forward_n600.json"
)


def _validate_uniform_predecessor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if payload.get("schema") != "exact_int64_fixedpoint_scorer_n600.v1":
        raise ValueError("uniform exact-int64 predecessor schema mismatch")
    if summary.get("status") != "MEASURED" or summary.get("full_real_n600") is not True:
        raise ValueError("uniform exact-int64 predecessor lacks full-n600 custody")
    if summary.get("argmax_exact_admitted") is True:
        raise ValueError("uniform exact-int64 predecessor is already exact; mixed rerun refused")
    return payload


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    start = int(receipt["contract"]["pair_start"])
    count = int(receipt["contract"]["pair_count"])
    expected = set(range(start, start + count))
    rows = list(receipt["arms"][ARM]["segnet_rows"])
    controls = list(receipt["arms"]["fp32_control"]["segnet_rows"])
    observed = [int(row["pair_index"]) for row in rows]
    control_observed = [int(row["pair_index"]) for row in controls]
    cache = _aggregate_cache_custody(
        list(receipt.get("cache_custody", {}).get("segnet_rows", [])),
        expected_indices=expected,
    )
    complete = bool(
        len(rows) == count
        and set(observed) == expected
        and len(controls) == count
        and set(control_observed) == expected
    )
    if not complete:
        return {
            "status": "INCOMPLETE",
            "pairs": len(rows),
            "unique_pair_indices": len(set(observed)),
            "full_real_n600": False,
            "cache_custody": cache,
            "rung2_mixed_integer_verdict": "INCOMPLETE",
        }
    candidate = {split: _aggregate_rows(rows, split=split) for split in ("calibration", "heldout", "full")}
    control = {split: _aggregate_rows(controls, split=split) for split in ("calibration", "heldout", "full")}
    full_n600 = bool(start == 0 and count == N600 and cache.get("status") == "MEASURED")
    exact = bool(candidate["heldout"]["argmax_exact_gate"] and candidate["full"]["argmax_exact_gate"])
    reference_seconds = [float(row["reference_seconds"]) for row in rows]
    candidate_seconds = [float(row["candidate_seconds"]) for row in rows]
    return {
        "status": "MEASURED",
        "full_real_n600": full_n600,
        "arm": ARM,
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_BITS,
        "argmax_exact_admitted": bool(full_n600 and exact),
        "training_tolerance_admitted": bool(
            full_n600
            and candidate["heldout"]["training_tolerance_gate"]
            and candidate["full"]["training_tolerance_gate"]
        ),
        "candidate": candidate,
        "fp32_control": control,
        "cache_custody": cache,
        "timing": {
            "axis": "[macOS CPU-Torch one-thread advisory]",
            "reference_total_seconds": float(sum(reference_seconds)),
            "candidate_total_seconds": float(sum(candidate_seconds)),
            "reference_median_seconds_per_pair": statistics.median(reference_seconds),
            "candidate_median_seconds_per_pair": statistics.median(candidate_seconds),
            "candidate_speedup_vs_reference_x": statistics.median(reference_seconds)
            / statistics.median(candidate_seconds),
            "throughput_interpretation": ("CPU exact-int64 is a numerical twin only; custom Metal must measure speed"),
        },
        "rung2_mixed_integer_verdict": (
            "MIXED_INT64_ARGMAX_FEASIBLE"
            if full_n600 and exact
            else "NO_EXACT_ARGMAX_IN_GEOMETRY_SAFE_MIXED_INSTANCE"
            if full_n600
            else "INCOMPLETE"
        ),
        "verdict_scope": (
            "n600 INSTANCE: frozen SegNet, geometry-only maximum signed W26..W30 precision "
            "under per-layer static int64 bounds, dynamic max-absolute codes, exact int64 "
            "Conv2d, one fp32 finalization, unchanged fp32 non-Conv operators"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args.output = args.output.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.qdq_precursor = args.qdq_precursor.resolve()
    args.uniform_predecessor = args.uniform_predecessor.resolve()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    stop = int(args.pair_start) + int(args.pair_count)
    if args.pair_start < 0 or args.pair_count <= 0 or stop > N600:
        raise ValueError(f"invalid pair interval [{args.pair_start},{stop})")
    preflight = _storage_preflight(args.output)
    qdq = _validate_qdq_precursor(args.qdq_precursor)
    uniform = _validate_uniform_predecessor(args.uniform_predecessor)
    arrays = _cache_arrays(args.gt_cache)
    reference, _, weights, _ = _load_models(include_pose=False)
    candidate, manifest = build_mixed_int64_model(reference)
    contract = {
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "arm": ARM,
        "minimum_bits": MINIMUM_BITS,
        "maximum_bits": MAXIMUM_BITS,
        "assignment_rule": manifest.assignment_rule,
        "precision_histogram": manifest.to_dict()["precision_histogram"],
        "activation_scale_mode": "dynamic_exact_absmax",
        "accumulation": "exact_signed_int64_torch_conv2d",
        "finalization": "single_fp32_scale_and_bias_per_output",
        "non_conv_operators": "unchanged_fp32",
        "checkpoint_every_pairs": 1,
        "resumable_from_disk": True,
        "native_integer_speed_claim": True,
        "metal_speed_claim": False,
        "training_tolerance": TRAINING_TOLERANCE,
    }
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "mixed_module_sha256": sha256_file(REPO / "src/tac/local_acceleration/mixed_int64_fixedpoint_scorer.py"),
        "qdq_precursor_sha256": sha256_file(args.qdq_precursor),
        "qdq_precursor_fingerprint": qdq.get("fingerprint"),
        "uniform_predecessor_sha256": sha256_file(args.uniform_predecessor),
        "uniform_predecessor_fingerprint": uniform.get("fingerprint"),
        "gt_cache_sha256": sha256_file(args.gt_cache),
        "segnet_weights_sha256": sha256_file(weights),
    }
    fingerprint = _fingerprint(
        {"schema": SCHEMA, "contract": contract, "custody": custody, "manifest": manifest.to_dict()}
    )
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume mixed-int64 receipt fingerprint differs")
    else:
        receipt = {
            "schema": SCHEMA,
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[macOS CPU-Torch mixed exact-int64 local derivation; MEANS]",
            "score_claim": False,
            "pointer_moved": False,
            "promotion_eligible": False,
            "git_head": _git_head(),
            "host": platform.node(),
            "argv": list(sys.argv),
            "contract": contract,
            "custody": custody,
            "model_manifest": manifest.to_dict(),
            "storage_preflight": preflight,
            "fingerprint": fingerprint,
            "cache_custody": {"segnet_rows": []},
            "arms": {
                "fp32_control": {"segnet_rows": []},
                ARM: {"segnet_rows": []},
            },
        }
        receipt["summary"] = _summary(receipt)
        atomic_json(args.output, receipt)
    control_rows = receipt["arms"]["fp32_control"]["segnet_rows"]
    rows = receipt["arms"][ARM]["segnet_rows"]
    completed = {int(row["pair_index"]) for row in rows}
    if completed != {int(row["pair_index"]) for row in control_rows}:
        raise ValueError("resume control/candidate pair sets differ")
    if completed and completed != set(range(int(args.pair_start), int(args.pair_start) + len(completed))):
        raise ValueError("resume receipt is not a contiguous prefix")
    started = time.perf_counter()
    with torch.inference_mode():
        for pair_index in range(int(args.pair_start), stop):
            if pair_index in completed:
                continue
            btchw, _ = _pair_inputs(arrays, pair_index)
            seg_input = reference.preprocess_input(btchw)
            tick = time.perf_counter()
            reference_logits = reference(seg_input)
            reference_seconds = time.perf_counter() - tick
            reference_argmax = reference_logits.argmax(dim=1)[0].cpu().numpy()
            top2 = reference_logits.topk(2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)[0].cpu().numpy()
            cached_argmax = np.asarray(arrays["lstars.npy"][pair_index])
            cached_margin = np.asarray(arrays["margins.npy"][pair_index])
            receipt["cache_custody"]["segnet_rows"].append(
                {
                    "pair_index": pair_index,
                    "argmax_mismatch_pixels": int(np.count_nonzero(reference_argmax != cached_argmax)),
                    "one_thread_argmax_sha256": _hash_array(reference_argmax),
                    "cached_argmax_sha256": _hash_array(cached_argmax),
                    "margin_max_abs_delta": float(np.max(np.abs(margin - cached_margin))),
                    "one_thread_margin_min": float(np.min(margin)),
                    "cached_margin_min": float(np.min(cached_margin)),
                    "threads": 1,
                }
            )
            control = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=margin,
                candidate_logits=reference_logits,
            )
            control.update(
                reference_seconds=float(reference_seconds),
                candidate_seconds=float(reference_seconds),
            )
            control_rows.append(control)
            tick = time.perf_counter()
            candidate_logits = candidate(seg_input)
            candidate_seconds = time.perf_counter() - tick
            row = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=margin,
                candidate_logits=candidate_logits,
            )
            row.update(
                reference_seconds=float(reference_seconds),
                candidate_seconds=float(candidate_seconds),
            )
            rows.append(row)
            completed.add(pair_index)
            receipt["summary"] = _summary(receipt)
            receipt["last_completed_pair"] = pair_index
            receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
            atomic_json(args.output, receipt)
    receipt["summary"] = _summary(receipt)
    receipt["completed"] = receipt["summary"].get("status") == "MEASURED"
    receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
    atomic_json(args.output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=N600)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--qdq-precursor", type=Path, default=DEFAULT_QDQ)
    parser.add_argument("--uniform-predecessor", type=Path, default=DEFAULT_UNIFORM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"].get("status") == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
