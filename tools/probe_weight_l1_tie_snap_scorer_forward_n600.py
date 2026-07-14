#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Resumable real-n600 calibration/heldout tie-snap head for weight-L1 SegNet."""

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
    CALIBRATION_START,
    CALIBRATION_STOP,
    HELDOUT_START,
    N600,
    _aggregate_cache_custody,
    _cache_arrays,
    _git_head,
    _hash_array,
    _load_models,
    _pair_inputs,
    _quantiles,
    _seg_row,
)

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.local_acceleration.argmax_tie_snap import (  # noqa: E402
    DYADIC_TIE_EPSILONS,
    epsilon_arm_name,
    tie_snap_argmax_numpy,
)
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (  # noqa: E402
    build_weight_l1_int64_model,
)

SCHEMA = "weight_l1_tie_snap_scorer_n600.v1"
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_PREDECESSOR = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_int64_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/weight_l1_tie_snap_scorer_forward_n600.json"
)


def _validate_weight_predecessor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    manifest = payload.get("model_manifest", {})
    if payload.get("schema") != "weight_l1_int64_fixedpoint_scorer_n600.v1":
        raise ValueError("weight-L1 predecessor schema mismatch")
    if (
        summary.get("status") != "MEASURED"
        or summary.get("full_real_n600") is not True
        or payload.get("completed") is not True
    ):
        raise ValueError("weight-L1 predecessor lacks full real-n600 custody")
    if summary.get("argmax_exact_admitted") is True:
        raise ValueError("weight-L1 predecessor is already exact; tie-snap rerun refused")
    if int(summary.get("candidate", {}).get("full", {}).get("flips", 0)) <= 0:
        raise ValueError("weight-L1 predecessor negative has no measured flipped pixels")
    if (
        int(manifest.get("converted_conv2d_count", -1)) != 125
        or int(manifest.get("maximum_bits", -1)) != 31
        or manifest.get("assignment_rule") != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
        or manifest.get("bound_kind") != "activation_qmax_times_max_output_quantized_weight_l1"
        or manifest.get("label_or_frame_dependent") is not False
    ):
        raise ValueError("weight-L1 predecessor manifest differs")
    if not payload.get("custody", {}).get("qdq_precursor_sha256") or not payload.get(
        "custody", {}
    ).get("qdq_precursor_fingerprint"):
        raise ValueError("weight-L1 predecessor lacks QDQ custody")
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
        digest.update(f"{row['pair_index']}:{row['candidate_argmax_sha256']}\n".encode("ascii"))
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
    base_indices = [int(row["pair_index"]) for row in base_rows]
    arm_rows = {name: list(payload["decision_rows"]) for name, payload in receipt["arms"].items()}
    complete = bool(
        len(base_rows) == count
        and set(base_indices) == expected
        and all(
            len(rows) == count and {int(row["pair_index"]) for row in rows} == expected for rows in arm_rows.values()
        )
    )
    cache = _aggregate_cache_custody(
        list(receipt.get("cache_custody", {}).get("segnet_rows", [])),
        expected_indices=expected,
    )
    if not complete:
        return {
            "status": "INCOMPLETE",
            "pairs": len(base_rows),
            "unique_pair_indices": len(set(base_indices)),
            "full_real_n600": False,
            "cache_custody": cache,
            "rung2_tie_snap_verdict": "INCOMPLETE",
        }
    aggregates = {
        name: {split: _aggregate_decisions(rows, split=split) for split in ("calibration", "heldout", "full")}
        for name, rows in arm_rows.items()
    }
    ladder = [(float(payload["epsilon"]), name) for name, payload in receipt["arms"].items()]
    calibration_exact = [
        (epsilon, name) for epsilon, name in sorted(ladder) if aggregates[name]["calibration"]["argmax_exact_gate"]
    ]
    selected_epsilon, selected_arm = calibration_exact[0] if calibration_exact else (None, None)
    full_n600 = bool(start == 0 and count == N600 and cache.get("status") == "MEASURED")
    heldout_exact = bool(selected_arm and aggregates[selected_arm]["heldout"]["argmax_exact_gate"])
    full_exact = bool(selected_arm and aggregates[selected_arm]["full"]["argmax_exact_gate"])
    admitted = bool(full_n600 and selected_arm and heldout_exact and full_exact)
    return {
        "status": "MEASURED",
        "full_real_n600": full_n600,
        "selection_surface": "calibration pairs 0..119 only",
        "validation_surface": "heldout pairs 120..599 plus full-corpus custody",
        "arms": aggregates,
        "minimum_calibration_exact_arm": selected_arm,
        "minimum_calibration_exact_epsilon": selected_epsilon,
        "selected_heldout_exact": heldout_exact,
        "selected_full_exact": full_exact,
        "argmax_exact_admitted": admitted,
        "cache_custody": cache,
        "rung2_tie_snap_verdict": (
            "TIE_SNAP_ARGMAX_FEASIBLE" if admitted else "NO_CALIBRATION_HELDOUT_EXACT_TIE_SNAP_IN_LADDER"
        ),
        "verdict_scope": (
            "n600 INSTANCE: frozen SegNet weight-L1-safe W27..W31 exact-int64 logits, "
            "preregistered dyadic lowest-class epsilon tie snap selected on pairs 0..119 "
            "and validated without reselection on pairs 120..599"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args.output = args.output.resolve()
    args.gt_cache = args.gt_cache.resolve()
    args.weight_predecessor = args.weight_predecessor.resolve()
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
    arrays = _cache_arrays(args.gt_cache)
    reference, _, weights, _ = _load_models(include_pose=False)
    candidate, manifest = build_weight_l1_int64_model(reference)
    epsilon_ladder = [float(value) for value in DYADIC_TIE_EPSILONS]
    contract = {
        "pair_start": int(args.pair_start),
        "pair_count": int(args.pair_count),
        "calibration_split": [CALIBRATION_START, CALIBRATION_STOP],
        "heldout_start": HELDOUT_START,
        "epsilon_ladder": epsilon_ladder,
        "epsilon_selection": "minimum calibration-exact epsilon; no heldout reselection",
        "decision_rule": "lowest class index within epsilon of candidate maximum",
        "logits_backend": "weight_l1_safe_w26_w31_exact_signed_int64_conv",
        "activation_scale_mode": "dynamic_exact_absmax",
        "runtime_label_or_frame_dependent": False,
        "epsilon_selection_uses_calibration_labels": True,
        "checkpoint_every_pairs": 1,
        "resumable_from_disk": True,
        "native_integer_speed_claim": True,
        "cpu_throughput_claim": False,
    }
    custody = {
        "probe_sha256": sha256_file(Path(__file__)),
        "tie_snap_module_sha256": sha256_file(REPO / "src/tac/local_acceleration/argmax_tie_snap.py"),
        "weight_l1_module_sha256": sha256_file(
            REPO / "src/tac/local_acceleration/weight_l1_int64_fixedpoint_scorer.py"
        ),
        "weight_predecessor_sha256": sha256_file(args.weight_predecessor),
        "weight_predecessor_fingerprint": predecessor.get("fingerprint"),
        "qdq_precursor_sha256": predecessor.get("custody", {}).get(
            "qdq_precursor_sha256"
        ),
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
            raise ValueError("resume tie-snap receipt fingerprint differs")
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
            "arms": {
                epsilon_arm_name(epsilon): {
                    "epsilon": epsilon,
                    "decision_rows": [],
                }
                for epsilon in epsilon_ladder
            },
        }
        receipt["summary"] = _summary(receipt)
        atomic_json(args.output, receipt)
    base_rows = receipt["base_rows"]
    completed = {int(row["pair_index"]) for row in base_rows}
    for payload in receipt["arms"].values():
        if completed != {int(row["pair_index"]) for row in payload["decision_rows"]}:
            raise ValueError("resume base/decision pair sets differ")
    if completed and completed != set(range(int(args.pair_start), int(args.pair_start) + len(completed))):
        raise ValueError("resume tie-snap receipt is not a contiguous prefix")
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
            tick = time.perf_counter()
            candidate_logits = candidate(seg_input)
            candidate_seconds = time.perf_counter() - tick
            base = _seg_row(
                pair_index=pair_index,
                reference_logits=reference_logits,
                reference_argmax=reference_argmax,
                baseline_margin=margin,
                candidate_logits=candidate_logits,
            )
            base.update(
                reference_seconds=float(reference_seconds),
                candidate_seconds=float(candidate_seconds),
            )
            base_rows.append(base)
            candidate_np = candidate_logits.detach().cpu().numpy().astype(np.float32, copy=False)
            plain = np.argmax(candidate_np, axis=1)[0]
            candidate_top2 = np.partition(candidate_np, kth=-2, axis=1)[:, -2:, :, :]
            candidate_margin = np.diff(np.sort(candidate_top2, axis=1), axis=1)[0, 0]
            for payload in receipt["arms"].values():
                decision = tie_snap_argmax_numpy(
                    candidate_np,
                    epsilon=float(payload["epsilon"]),
                    class_axis=1,
                )[0]
                flips = decision != reference_argmax
                payload["decision_rows"].append(
                    {
                        "pair_index": pair_index,
                        "split": ("calibration" if pair_index < HELDOUT_START else "heldout"),
                        "epsilon": float(payload["epsilon"]),
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
    parser.add_argument(
        "--weight-predecessor",
        type=Path,
        default=DEFAULT_PREDECESSOR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["summary"].get("status") == "MEASURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
