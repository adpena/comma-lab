#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diagnose weight-L1 tie-snap conflicts on the frozen 0..263 design surface."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "upstream", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_fixedpoint_scorer_forward_n600 import (  # noqa: E402
    _cache_arrays,
    _git_head,
    _hash_array,
    _load_models,
    _pair_inputs,
)

from tac.local_acceleration.ane_unlock_followup_20260713 import (  # noqa: E402
    atomic_json,
    sha256_file,
)
from tac.local_acceleration.argmax_tie_snap import tie_snap_argmax_numpy  # noqa: E402
from tac.local_acceleration.weight_l1_int64_fixedpoint_scorer import (  # noqa: E402
    build_weight_l1_int64_model,
)

SCHEMA = "weight_l1_tie_conflict_diagnostic.v1"
DESIGN_STOP = 264
VALIDATION_START = 264
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_WEIGHT_RECEIPT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_int64_fixedpoint_scorer_forward_n600.json"
)
DEFAULT_TIE_RECEIPT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_tie_snap_scorer_forward_n600.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "experiments/results/throughput_authority_ladder_20260714/"
    "weight_l1_tie_conflict_diagnostic_design_0_263.json"
)


def _float_hex(values: np.ndarray) -> list[str]:
    return [float(value).hex() for value in np.asarray(values, dtype=np.float32)]


def _validate_receipts(weight_path: Path, tie_path: Path, pairs: list[int]) -> dict[str, Any]:
    weight = json.loads(weight_path.read_text(encoding="utf-8"))
    tie = json.loads(tie_path.read_text(encoding="utf-8"))
    if not (
        weight.get("schema") == "weight_l1_int64_fixedpoint_scorer_n600.v1"
        and weight.get("summary", {}).get("status") == "MEASURED"
        and weight.get("summary", {}).get("full_real_n600") is True
        and weight.get("completed") is True
    ):
        raise ValueError("weight-L1 predecessor lacks complete n600 custody")
    if tie.get("schema") != "weight_l1_tie_snap_scorer_n600.v1":
        raise ValueError("tie-snap receipt schema mismatch")
    base_rows = tie.get("base_rows", [])
    completed = {int(row["pair_index"]) for row in base_rows}
    if not set(pairs).issubset(completed):
        raise ValueError("tie-snap receipt has not checkpointed every requested design pair")
    selected_rows = tie.get("arms", {}).get("epsilon_2m19", {}).get("decision_rows", [])
    base_by_pair = {int(row["pair_index"]): row for row in base_rows}
    by_pair = {int(row["pair_index"]): row for row in selected_rows}
    if any(
        int(base_by_pair[pair].get("flips", 0)) <= 0
        and int(by_pair[pair].get("flips", 0)) <= 0
        for pair in pairs
    ):
        raise ValueError("requested diagnostic pair has no measured plain/tie-snap conflict")
    return {"weight": weight, "tie": tie}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    pairs = sorted({int(pair) for pair in args.pairs})
    if not pairs or pairs[0] < 0 or pairs[-1] >= DESIGN_STOP:
        raise ValueError(f"diagnostic pairs must lie in frozen design surface [0,{DESIGN_STOP})")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    receipts = _validate_receipts(args.weight_receipt, args.tie_receipt, pairs)
    arrays = _cache_arrays(args.gt_cache)
    reference, _, weights, _ = _load_models(include_pose=False)
    candidate, manifest = build_weight_l1_int64_model(reference)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair_index in pairs:
            btchw, _ = _pair_inputs(arrays, pair_index)
            seg_input = reference.preprocess_input(btchw)
            reference_np = reference(seg_input).detach().cpu().numpy().astype(np.float32, copy=False)
            candidate_np = candidate(seg_input).detach().cpu().numpy().astype(np.float32, copy=False)
            reference_argmax = np.argmax(reference_np, axis=1)[0]
            plain = np.argmax(candidate_np, axis=1)[0]
            snapped = tie_snap_argmax_numpy(
                candidate_np,
                epsilon=2.0**-19,
                class_axis=1,
            )[0]
            coordinates = np.argwhere((plain != reference_argmax) | (snapped != reference_argmax))
            pixels: list[dict[str, Any]] = []
            for y, x in coordinates:
                ref_logits = reference_np[0, :, y, x]
                cand_logits = candidate_np[0, :, y, x]
                candidate_order = np.argsort(-cand_logits, kind="stable")
                pixels.append(
                    {
                        "y": int(y),
                        "x": int(x),
                        "reference_class": int(reference_argmax[y, x]),
                        "candidate_plain_class": int(plain[y, x]),
                        "candidate_tie_snap_class": int(snapped[y, x]),
                        "candidate_top2_classes": [int(value) for value in candidate_order[:2]],
                        "reference_logits_float_hex": _float_hex(ref_logits),
                        "candidate_logits_float_hex": _float_hex(cand_logits),
                        "reference_winner_runner_margin": float(
                            np.partition(ref_logits, -2)[-1] - np.partition(ref_logits, -2)[-2]
                        ),
                        "candidate_winner_runner_margin": float(
                            cand_logits[candidate_order[0]] - cand_logits[candidate_order[1]]
                        ),
                    }
                )
            rows.append(
                {
                    "pair_index": pair_index,
                    "reference_argmax_sha256": _hash_array(reference_argmax),
                    "candidate_plain_argmax_sha256": _hash_array(plain),
                    "candidate_tie_snap_argmax_sha256": _hash_array(snapped),
                    "plain_flips": int(np.count_nonzero(plain != reference_argmax)),
                    "tie_snap_flips": int(np.count_nonzero(snapped != reference_argmax)),
                    "pixels": pixels,
                }
            )
    payload = {
        "schema": SCHEMA,
        "lane_id": "throughput_authority_ladder",
        "task_id": 494,
        "axis": "[macOS CPU-Torch exact-int64 diagnostic; research-only MEANS]",
        "score_claim": False,
        "pointer_moved": False,
        "git_head": _git_head(),
        "host": platform.node(),
        "contract": {
            "design_pairs": [0, DESIGN_STOP - 1],
            "second_validation_pairs": [VALIDATION_START, 599],
            "pairs_diagnosed": pairs,
            "epsilon": 2.0**-19,
            "second_validation_not_used_to_design": True,
        },
        "custody": {
            "probe_sha256": sha256_file(Path(__file__)),
            "gt_cache_sha256": sha256_file(args.gt_cache),
            "segnet_weights_sha256": sha256_file(weights),
            "weight_receipt_sha256": sha256_file(args.weight_receipt),
            "weight_receipt_fingerprint": receipts["weight"].get("fingerprint"),
            "tie_receipt_sha256_at_freeze": sha256_file(args.tie_receipt),
            "tie_receipt_fingerprint": receipts["tie"].get("fingerprint"),
        },
        "model_manifest": manifest.to_dict(),
        "rows": rows,
        "verdict_scope": (
            "DESIGN INSTANCE: only measured conflicts in pairs 0..263; pairs 264..599 are "
            "reserved for a second validation split"
        ),
    }
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, nargs="+", default=[11, 195, 263])
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--weight-receipt", type=Path, default=DEFAULT_WEIGHT_RECEIPT)
    parser.add_argument("--tie-receipt", type=Path, default=DEFAULT_TIE_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.gt_cache = args.gt_cache.resolve()
    args.weight_receipt = args.weight_receipt.resolve()
    args.tie_receipt = args.tie_receipt.resolve()
    args.output = args.output.resolve()
    payload = run(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
