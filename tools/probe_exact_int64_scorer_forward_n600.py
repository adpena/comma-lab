#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-n600 exact-int64 frozen-SegNet forward feasibility probe.

The sister QDQ ladder deliberately uses fp32 Conv2d accumulation.  This probe
tests the distinct arithmetic implemented by the custom-Metal backend: signed
integer activation/weight codes, exact signed-int64 Conv2d accumulation, and a
single fp32 scale/bias finalization.  It is a numerical feasibility and CPU
reference receipt; it does not claim Metal placement or speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
from tac.local_acceleration.exact_int64_fixedpoint_scorer import (  # noqa: E402
    build_exact_int64_model,
)

SCHEMA = "exact_int64_fixedpoint_scorer_n600.v1"
ARM = "w26a26"
BITS = 26
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_QDQ_RECEIPT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json"
)
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/exact_int64_fixedpoint_scorer_forward_n600.json"
)
MINIMUM_FREE_BYTES = 64 * 1024 * 1024


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_qdq_precursor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "dynamic_fixedpoint_scorer_forward_n600.v1":
        raise ValueError("exact-int64 probe requires the dynamic QDQ precursor schema")
    summary = receipt.get("summary", {})
    if summary.get("status") != "MEASURED" or summary.get("full_real_n600") is not True:
        raise ValueError("dynamic QDQ precursor lacks exact full-n600 custody")
    arm = summary.get("arms", {}).get(ARM, {})
    if arm.get("status") != "MEASURED" or arm.get("training_tolerance_admitted") is not True:
        raise ValueError(f"dynamic QDQ precursor lacks a measured tolerance-admitted {ARM}")
    contract = receipt.get("contract", {})
    if contract.get("activation_scale_mode") != "dynamic_exact_absmax":
        raise ValueError("dynamic QDQ precursor scale mode differs")
    if contract.get("quantized_code_clamp") != "round_fp32_then_exact_signed_int64_clamp":
        raise ValueError("dynamic QDQ precursor lacks the exact integer-domain clamp")
    return receipt


def _storage_preflight(output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output.parent)
    payload = {
        "path": str(output.parent.relative_to(REPO)),
        "free_bytes": int(usage.free),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "status": "PASS" if usage.free >= MINIMUM_FREE_BYTES else "BLOCK",
        "expected_artifact_class": "small resumable JSON receipt",
        "cleanup": "no bulky rebuildable artifact is produced",
    }
    if payload["status"] != "PASS":
        raise RuntimeError(f"storage preflight failed: {payload}")
    return payload


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    start = int(receipt["contract"]["pair_start"])
    count = int(receipt["contract"]["pair_count"])
    expected = set(range(start, start + count))
    candidate_rows = list(receipt["arms"][ARM]["segnet_rows"])
    control_rows = list(receipt["arms"]["fp32_control"]["segnet_rows"])
    candidate_indices = [int(row["pair_index"]) for row in candidate_rows]
    control_indices = [int(row["pair_index"]) for row in control_rows]
    complete = bool(
        len(candidate_rows) == count
        and set(candidate_indices) == expected
        and len(control_rows) == count
        and set(control_indices) == expected
    )
    cache = _aggregate_cache_custody(
        list(receipt.get("cache_custody", {}).get("segnet_rows", [])),
        expected_indices=expected,
    )
    if not complete:
        return {
            "status": "INCOMPLETE",
            "pairs": len(candidate_rows),
            "unique_pair_indices": len(set(candidate_indices)),
            "full_real_n600": False,
            "cache_custody": cache,
            "rung2_integer_verdict": "INCOMPLETE",
        }
    candidate = {split: _aggregate_rows(candidate_rows, split=split) for split in ("calibration", "heldout", "full")}
    control = {split: _aggregate_rows(control_rows, split=split) for split in ("calibration", "heldout", "full")}
    full_n600 = bool(start == 0 and count == N600 and cache.get("status") == "MEASURED")
    exact = bool(candidate["heldout"]["argmax_exact_gate"] and candidate["full"]["argmax_exact_gate"])
    control_seconds = [float(row["reference_seconds"]) for row in candidate_rows]
    candidate_seconds = [float(row["candidate_seconds"]) for row in candidate_rows]
    return {
        "status": "MEASURED",
        "full_real_n600": full_n600,
        "bits": BITS,
        "arm": ARM,
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
            "reference_total_seconds": float(sum(control_seconds)),
            "candidate_total_seconds": float(sum(candidate_seconds)),
            "reference_median_seconds_per_pair": float(np.median(control_seconds)),
            "candidate_median_seconds_per_pair": float(np.median(candidate_seconds)),
            "candidate_speedup_vs_reference_x": float(np.median(control_seconds) / np.median(candidate_seconds)),
            "throughput_interpretation": ("CPU exact-int64 is a numerical twin only; custom Metal must measure speed"),
        },
        "rung2_integer_verdict": (
            "EXACT_INT64_ARGMAX_FEASIBLE"
            if full_n600 and exact
            else "NO_EXACT_ARGMAX_IN_W26_DIRECT_INT64_INSTANCE"
            if full_n600
            else "INCOMPLETE"
        ),
        "verdict_scope": (
            "n600 INSTANCE: frozen SegNet, W26A26 label-free dynamic max-absolute "
            "codes, exact signed-int64 Conv2d accumulation, one fp32 finalization and "
            "unchanged fp32 non-Conv operators on one-thread CPU-Torch"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args.output = args.output.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.qdq_precursor = args.qdq_precursor.resolve()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("one-thread CPU Torch contract failed")
    stop = int(args.pair_start) + int(args.pair_count)
    if args.pair_start < 0 or args.pair_count <= 0 or stop > N600:
        raise ValueError(f"invalid pair interval [{args.pair_start},{stop})")
    preflight = _storage_preflight(args.output)
    qdq = _validate_qdq_precursor(args.qdq_precursor)
    arrays = _cache_arrays(args.gt_cache)
    reference, _, seg_weights, _ = _load_models(include_pose=False)
    candidate, manifest = build_exact_int64_model(
        reference,
        bits=BITS,
        activation_scale_mode="dynamic_exact_absmax",
    )
    contract = {
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "arm": ARM,
        "bits": BITS,
        "activation_scale_mode": "dynamic_exact_absmax",
        "activation_codes": "signed_int64_containing_exact_signed_int32_range",
        "weight_codes": "signed_int64_containing_exact_signed_int32_range",
        "accumulation": "exact_signed_int64_torch_conv2d",
        "finalization": "single_fp32_scale_and_bias_per_output",
        "non_conv_operators": "unchanged_fp32",
        "quantized_code_clamp": "round_fp32_then_exact_signed_int64_clamp",
        "threads": {"intraop": 1, "interop": 1},
        "checkpoint_every_pairs": 1,
        "resumable_from_disk": True,
        "native_integer_speed_claim": True,
        "metal_speed_claim": False,
        "training_tolerance": TRAINING_TOLERANCE,
        "segnet_reference_control": (
            "recomputed one-thread CPU-Torch fp32 logits/argmax/top1-top2 margin on each real frame"
        ),
        "legacy_cache_role": "frame custody + explicit thread-geometry audit; not authority",
    }
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "exact_int64_module_sha256": sha256_file(REPO / "src/tac/local_acceleration/exact_int64_fixedpoint_scorer.py"),
        "metal_twin_module_sha256": sha256_file(REPO / "src/tac/local_acceleration/metal_fixedpoint_verdict.py"),
        "qdq_precursor": str(args.qdq_precursor.resolve().relative_to(REPO)),
        "qdq_precursor_sha256": sha256_file(args.qdq_precursor),
        "qdq_precursor_fingerprint": qdq.get("fingerprint"),
        "gt_cache_sha256": sha256_file(args.gt_cache),
        "segnet_weights_sha256": sha256_file(seg_weights),
    }
    fingerprint = _fingerprint(
        {"schema": SCHEMA, "contract": contract, "custody": custody, "manifest": manifest.to_dict()}
    )
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume exact-int64 receipt fingerprint differs")
    else:
        receipt = {
            "schema": SCHEMA,
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[macOS CPU-Torch exact-int64 local derivation; non-promotable MEANS]",
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
    candidate_rows = receipt["arms"][ARM]["segnet_rows"]
    control_indices = {int(row["pair_index"]) for row in control_rows}
    candidate_indices = {int(row["pair_index"]) for row in candidate_rows}
    if control_indices != candidate_indices:
        raise ValueError("resume control/candidate pair sets differ")
    if control_indices and control_indices != set(
        range(int(args.pair_start), int(args.pair_start) + len(control_indices))
    ):
        raise ValueError("resume receipt is not a contiguous completed prefix")

    started = time.perf_counter()
    with torch.inference_mode():
        for pair_index in range(int(args.pair_start), stop):
            if pair_index in candidate_indices:
                continue
            btchw, _ = _pair_inputs(arrays, pair_index)
            seg_input = reference.preprocess_input(btchw)
            reference_started = time.perf_counter()
            reference_logits = reference(seg_input)
            reference_seconds = time.perf_counter() - reference_started
            reference_argmax = reference_logits.argmax(dim=1)[0].cpu().numpy()
            top2 = reference_logits.topk(2, dim=1).values
            baseline_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)[0].cpu().numpy()
            cached_argmax = np.asarray(arrays["lstars.npy"][pair_index])
            cached_margin = np.asarray(arrays["margins.npy"][pair_index])
            receipt["cache_custody"]["segnet_rows"].append(
                {
                    "pair_index": pair_index,
                    "argmax_mismatch_pixels": int(np.count_nonzero(reference_argmax != cached_argmax)),
                    "one_thread_argmax_sha256": _hash_array(reference_argmax),
                    "cached_argmax_sha256": _hash_array(cached_argmax),
                    "margin_max_abs_delta": float(np.max(np.abs(baseline_margin - cached_margin))),
                    "one_thread_margin_min": float(np.min(baseline_margin)),
                    "cached_margin_min": float(np.min(cached_margin)),
                    "threads": 1,
                }
            )
            control = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=baseline_margin,
                candidate_logits=reference_logits,
            )
            control["reference_seconds"] = float(reference_seconds)
            control["candidate_seconds"] = float(reference_seconds)
            control_rows.append(control)

            candidate_started = time.perf_counter()
            candidate_logits = candidate(seg_input)
            candidate_seconds = time.perf_counter() - candidate_started
            row = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=baseline_margin,
                candidate_logits=candidate_logits,
            )
            row["reference_seconds"] = float(reference_seconds)
            row["candidate_seconds"] = float(candidate_seconds)
            candidate_rows.append(row)
            candidate_indices.add(pair_index)
            receipt["summary"] = _summary(receipt)
            receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
            receipt["last_completed_pair"] = pair_index
            atomic_json(args.output, receipt)
    receipt["summary"] = _summary(receipt)
    receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
    receipt["completed"] = receipt["summary"].get("status") == "MEASURED"
    atomic_json(args.output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=N600)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--qdq-precursor", type=Path, default=DEFAULT_QDQ_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"].get("status") == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
