#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-n600 second-validation probe for a frozen class-pair tie rule."""

from __future__ import annotations

import argparse
import hashlib
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
)
from probe_fixedpoint_scorer_forward_n600 import (  # noqa: E402
    N600,
    _aggregate_cache_custody,
    _aggregate_rows,
    _cache_arrays,
    _git_head,
    _hash_array,
    _load_models,
    _pair_inputs,
    _quantiles,
    _seg_row,
)
from probe_weight_l1_tie_snap_scorer_forward_n600 import (  # noqa: E402
    _validate_weight_predecessor,
)

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.local_acceleration.argmax_tie_snap import (  # noqa: E402
    class_pair_tie_snap_argmax_numpy,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (  # noqa: E402
    build_weight_l1_int64_model,
)

SCHEMA = "weight_l1_class_pair_tie_snap_scorer_n600.v1"
DESIGN_STOP = 264
SECOND_VALIDATION_START = DESIGN_STOP
EPSILON = float(2.0**-19)
WINNER_CLASS = 4
RUNNER_CLASS = 0
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_PREDECESSOR = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_int64_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_DESIGN_RECEIPT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_tie_conflict_diagnostic_design_0_263.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_class_pair_tie_snap_scorer_forward_n600.json"
)


def _validate_design_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("contract", {})
    if payload.get("schema") != "weight_l1_tie_conflict_diagnostic.v1":
        raise ValueError("class-pair design receipt schema mismatch")
    if (
        contract.get("design_pairs") != [0, DESIGN_STOP - 1]
        or contract.get("second_validation_pairs") != [SECOND_VALIDATION_START, N600 - 1]
        or contract.get("second_validation_not_used_to_design") is not True
        or float(contract.get("epsilon", -1.0)) != EPSILON
    ):
        raise ValueError("class-pair design/second-validation contract differs")
    rows = {int(row["pair_index"]): row for row in payload.get("rows", [])}
    if set(rows) != {11, 195, 263}:
        raise ValueError("class-pair design receipt conflict set differs")
    target = [
        pixel
        for pixel in rows[11].get("pixels", [])
        if pixel.get("candidate_top2_classes") == [WINNER_CLASS, RUNNER_CLASS]
        and int(pixel.get("reference_class", -1)) == RUNNER_CLASS
        and int(pixel.get("candidate_plain_class", -1)) == WINNER_CLASS
        and int(pixel.get("candidate_tie_snap_class", -1)) == RUNNER_CLASS
        and float(pixel.get("candidate_winner_runner_margin", float("inf"))) <= EPSILON
    ]
    if len(target) != 1:
        raise ValueError("class-pair design receipt lacks the unique 4-over-0 target")
    for pair_index in (195, 263):
        conflicts = [
            pixel
            for pixel in rows[pair_index].get("pixels", [])
            if pixel.get("candidate_top2_classes") == [1, 0]
            and int(pixel.get("reference_class", -1)) == 1
            and int(pixel.get("candidate_plain_class", -1)) == 1
            and int(pixel.get("candidate_tie_snap_class", -1)) == 0
        ]
        if len(conflicts) != 1:
            raise ValueError(
                f"class-pair design receipt lacks global-snap conflict at pair {pair_index}"
            )
    custody = payload.get("custody", {})
    if not custody.get("weight_receipt_sha256") or not custody.get(
        "tie_receipt_sha256_at_freeze"
    ):
        raise ValueError("class-pair design receipt lacks predecessor custody")
    return payload


def _aggregate_decisions(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    selected = rows if split == "full" else [row for row in rows if row["split"] == split]
    if not selected:
        return {"status": "INCOMPLETE", "pairs": 0, "argmax_exact_gate": False}
    flips = sum(int(row["flips"]) for row in selected)
    pixels = sum(int(row["pixels"]) for row in selected)
    worst = max(
        selected,
        key=lambda row: (float(row["flip_fraction"]), -int(row["pair_index"])),
    )
    digest = hashlib.sha256()
    for row in sorted(selected, key=lambda item: int(item["pair_index"])):
        digest.update(
            f"{row['pair_index']}:{row['candidate_argmax_sha256']}\n".encode("ascii")
        )
    return {
        "status": "MEASURED",
        "pairs": len(selected),
        "flips": flips,
        "pixels": pixels,
        "aggregate_flip_fraction": float(flips / pixels),
        "worst_pair_index": int(worst["pair_index"]),
        "worst_pair_flip_fraction": float(worst["flip_fraction"]),
        "argmax_corpus_sha256": digest.hexdigest(),
        "argmax_exact_gate": flips == 0,
        "snapped_pixels": sum(int(row["snapped_pixels"]) for row in selected),
        "pairs_with_snaps": sum(int(row["snapped_pixels"] > 0) for row in selected),
        "candidate_margin_min": min(float(row["candidate_margin_min"]) for row in selected),
        "reference_margin_min": min(float(row["reference_margin_min"]) for row in selected),
    }


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    start = int(receipt["contract"]["pair_start"])
    count = int(receipt["contract"]["pair_count"])
    expected = set(range(start, start + count))
    base_rows = list(receipt["base_rows"])
    decision_rows = list(receipt["decision_rows"])
    base_indices = [int(row["pair_index"]) for row in base_rows]
    decision_indices = [int(row["pair_index"]) for row in decision_rows]
    complete = bool(
        len(base_rows) == count
        and set(base_indices) == expected
        and len(decision_rows) == count
        and set(decision_indices) == expected
    )
    cache = _aggregate_cache_custody(
        list(receipt.get("cache_custody", {}).get("segnet_rows", [])),
        expected_indices=expected,
    )
    if not complete:
        return {
            "status": "INCOMPLETE",
            "pairs": len(decision_rows),
            "unique_pair_indices": len(set(decision_indices)),
            "full_real_n600": False,
            "cache_custody": cache,
            "rung2_class_pair_tie_snap_verdict": "INCOMPLETE",
        }
    decisions = {
        split: _aggregate_decisions(decision_rows, split=split)
        for split in ("design", "second_validation", "full")
    }
    full_n600 = bool(start == 0 and count == N600 and cache.get("status") == "MEASURED")
    admitted = bool(
        full_n600
        and decisions["design"]["argmax_exact_gate"]
        and decisions["second_validation"]["argmax_exact_gate"]
        and decisions["full"]["argmax_exact_gate"]
    )
    return {
        "status": "MEASURED",
        "full_real_n600": full_n600,
        "rule_frozen_from_design_surface": "pairs 0..263",
        "untouched_second_validation_surface": "pairs 264..599",
        "epsilon": EPSILON,
        "ordered_candidate_top2": [WINNER_CLASS, RUNNER_CLASS],
        "replacement_class": RUNNER_CLASS,
        "plain_weight_l1": _aggregate_rows(base_rows, split="full"),
        "class_pair_tie_snap": decisions,
        "design_exact": decisions["design"]["argmax_exact_gate"],
        "second_validation_exact": decisions["second_validation"]["argmax_exact_gate"],
        "argmax_exact_admitted": admitted,
        "cache_custody": cache,
        "rung2_class_pair_tie_snap_verdict": (
            "CLASS_PAIR_TIE_SNAP_ARGMAX_FEASIBLE"
            if admitted
            else "NO_EXACT_ARGMAX_IN_CLASS_PAIR_TIE_SNAP_INSTANCE"
        ),
        "verdict_scope": (
            "n600 INSTANCE: frozen SegNet weight-L1-safe W27..W31 exact-int64 logits; "
            "ordered candidate top2 (4,0) with gap <=2^-19 snaps to class 0; rule derived "
            "only from pairs 0..263 and validated without reselection on pairs 264..599"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args.output = args.output.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.weight_predecessor = args.weight_predecessor.resolve()
    args.design_receipt = args.design_receipt.resolve()
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
    predecessor = _validate_weight_predecessor(args.weight_predecessor)
    design = _validate_design_receipt(args.design_receipt)
    if design["custody"]["weight_receipt_sha256"] != sha256_file(args.weight_predecessor):
        raise ValueError("class-pair design receipt names a different weight predecessor")
    if design["custody"]["gt_cache_sha256"] != sha256_file(args.gt_cache):
        raise ValueError("class-pair design receipt names a different GT cache")
    arrays = _cache_arrays(args.gt_cache)
    reference, _, weights, _ = _load_models(include_pose=False)
    if design["custody"]["segnet_weights_sha256"] != sha256_file(weights):
        raise ValueError("class-pair design receipt names different SegNet weights")
    candidate, manifest = build_weight_l1_int64_model(reference)
    contract = {
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "design_split": [0, DESIGN_STOP],
        "second_validation_split": [SECOND_VALIDATION_START, N600],
        "epsilon": EPSILON,
        "candidate_winner_class": WINNER_CLASS,
        "candidate_runner_class": RUNNER_CLASS,
        "replacement_class": RUNNER_CLASS,
        "decision_rule": (
            "if deterministic candidate top2 is ordered (4,0) and gap <=2^-19, "
            "choose class 0; otherwise ordinary lowest-index argmax"
        ),
        "rule_frozen_before_second_validation_access": True,
        "second_validation_reselection": False,
        "runtime_label_or_frame_dependent": False,
        "design_selection_uses_labels": True,
        "logits_backend": "weight_l1_safe_w27_w31_exact_signed_int64_conv",
        "activation_scale_mode": "dynamic_exact_absmax",
        "checkpoint_every_pairs": 1,
        "resumable_from_disk": True,
        "native_integer_speed_claim": True,
        "cpu_throughput_claim": False,
    }
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "tie_snap_module_sha256": sha256_file(
            REPO / "src/tac/local_acceleration/argmax_tie_snap.py"
        ),
        "weight_l1_module_sha256": sha256_file(
            REPO / "src/tac/local_acceleration/weight_l1_int64_fixedpoint_scorer.py"
        ),
        "weight_predecessor_sha256": sha256_file(args.weight_predecessor),
        "weight_predecessor_fingerprint": predecessor.get("fingerprint"),
        "design_receipt_sha256": sha256_file(args.design_receipt),
        "design_tie_receipt_sha256_at_freeze": design["custody"][
            "tie_receipt_sha256_at_freeze"
        ],
        "qdq_precursor_sha256": predecessor.get("custody", {}).get("qdq_precursor_sha256"),
        "qdq_precursor_fingerprint": predecessor.get("custody", {}).get(
            "qdq_precursor_fingerprint"
        ),
        "gt_cache_sha256": sha256_file(args.gt_cache),
        "segnet_weights_sha256": sha256_file(weights),
    }
    fingerprint = _fingerprint(
        {
            "schema": SCHEMA,
            "contract": contract,
            "custody": custody,
            "manifest": manifest.to_dict(),
        }
    )
    if args.resume and args.output.is_file():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA or receipt.get("fingerprint") != fingerprint:
            raise ValueError("resume class-pair tie-snap receipt fingerprint differs")
    else:
        receipt = {
            "schema": SCHEMA,
            "lane_id": "throughput_authority_ladder",
            "task_id": 494,
            "axis": "[macOS CPU-Torch exact-int64 decision-head derivation; MEANS]",
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
            "base_rows": [],
            "decision_rows": [],
        }
        receipt["summary"] = _summary(receipt)
        atomic_json(args.output, receipt)
    base_rows = receipt["base_rows"]
    decision_rows = receipt["decision_rows"]
    completed = {int(row["pair_index"]) for row in base_rows}
    if completed != {int(row["pair_index"]) for row in decision_rows}:
        raise ValueError("resume base/decision pair sets differ")
    if completed and completed != set(
        range(int(args.pair_start), int(args.pair_start) + len(completed))
    ):
        raise ValueError("resume class-pair receipt is not a contiguous prefix")
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
                    "argmax_mismatch_pixels": int(
                        np.count_nonzero(reference_argmax != cached_argmax)
                    ),
                    "one_thread_argmax_sha256": _hash_array(reference_argmax),
                    "cached_argmax_sha256": _hash_array(cached_argmax),
                    "margin_max_abs_delta": float(np.max(np.abs(margin - cached_margin))),
                    "one_thread_margin_min": float(np.min(margin)),
                    "cached_margin_min": float(np.min(cached_margin)),
                    "threads": 1,
                }
            )
            tick = time.perf_counter()
            candidate_logits = candidate(seg_input)
            candidate_seconds = time.perf_counter() - tick
            split = "design" if pair_index < DESIGN_STOP else "second_validation"
            base = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=margin,
                candidate_logits=candidate_logits,
            )
            base.update(
                split=split,
                reference_seconds=float(reference_seconds),
                candidate_seconds=float(candidate_seconds),
            )
            base_rows.append(base)
            candidate_np = candidate_logits.detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )
            plain = np.argmax(candidate_np, axis=1)[0]
            decision = class_pair_tie_snap_argmax_numpy(
                candidate_np,
                epsilon=EPSILON,
                winner_class=WINNER_CLASS,
                runner_class=RUNNER_CLASS,
                class_axis=1,
            )[0]
            candidate_top2 = np.partition(candidate_np, kth=-2, axis=1)[:, -2:, :, :]
            candidate_margin = np.diff(np.sort(candidate_top2, axis=1), axis=1)[0, 0]
            flips = decision != reference_argmax
            decision_rows.append(
                {
                    "pair_index": pair_index,
                    "split": split,
                    "epsilon": EPSILON,
                    "candidate_winner_class": WINNER_CLASS,
                    "candidate_runner_class": RUNNER_CLASS,
                    "flips": int(np.count_nonzero(flips)),
                    "pixels": int(flips.size),
                    "flip_fraction": float(np.mean(flips)),
                    "snapped_pixels": int(np.count_nonzero(decision != plain)),
                    "candidate_margin_min": float(np.min(candidate_margin)),
                    "reference_margin_min": float(np.min(margin)),
                    "candidate_argmax_sha256": _hash_array(decision),
                    "reference_argmax_sha256": _hash_array(reference_argmax),
                    "flipped_reference_margin_quantiles": _quantiles(margin[flips]),
                }
            )
            completed.add(pair_index)
            receipt["summary"] = _summary(receipt)
            receipt["last_completed_pair"] = pair_index
            receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
            atomic_json(args.output, receipt)
    receipt["summary"] = _summary(receipt)
    receipt["completed"] = receipt["summary"].get("status") == "MEASURED"
    receipt["elapsed_seconds_this_process"] = float(time.perf_counter() - started)
    reference_seconds = [float(row["reference_seconds"]) for row in base_rows]
    candidate_seconds = [float(row["candidate_seconds"]) for row in base_rows]
    receipt["summary"]["timing"] = {
        "axis": "[macOS CPU-Torch one-thread advisory]",
        "reference_median_seconds_per_pair": statistics.median(reference_seconds),
        "candidate_median_seconds_per_pair": statistics.median(candidate_seconds),
        "interpretation": "CPU numerical twin only; custom Metal owns native speed",
    }
    atomic_json(args.output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=N600)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--weight-predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--design-receipt", type=Path, default=DEFAULT_DESIGN_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"].get("status") == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
