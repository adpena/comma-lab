#!/usr/bin/env python3
"""PR101 HStack-of-VStacks empirical anchor — Path B step 3 of the
PARADIGM-Ω-OPT design (per-component independent optimization, predicted 0.110).

## Config space enumeration

Technique class: HStack-of-VStacks (per-tensor independent codec
parameter selection)
Full config space: { per_tensor_brotli_params,
                     per_tensor_codec_choice,  // brotli vs sparsity vs lossy
                     per_tensor_quantization_levels,
                     per_tensor_grouping_strategies }
This tool tests: { per_tensor_brotli_params } — the simplest lossless
HStack-of-VStacks form.
Falsification scope: this tool can only constrain the per-tensor brotli
parameter HStack. The per-tensor codec-choice form (brotli vs sparsity
vs lossy) and the lossy variants remain UNTESTED.

## What this tool tests

The Ω-OPT design predicts -5bp from "per-component independent
optimization." The simplest concrete instantiation: rather than encoding
all 28 tensors with one fixed brotli (quality, lgwin, lgblock) tuple, let
each tensor independently select its optimal triple from a swept grid.

Joint brotli (single triple for all 28 tensors): predefined grid sweep.
Per-tensor brotli: same grid, per-tensor optimal selection.

The byte savings = sum over tensors of (joint_optimum - per_tensor_optimum).
Both encodings are LOSSLESS — no distortion induced. This isolates the
HStack-of-VStacks optimization gain at the encoder-parameter layer.

## Method

1. Quantize PR101 decoder state_dict to int8 symbols (228,958 elements
   across 28 tensors via FIXED_STATE_SCHEMA).
2. For each tensor, compute brotli bytes over a grid of
   (quality ∈ {9, 10, 11}, lgwin ∈ {14, 16, 18, 20, 22},
    lgblock ∈ {17, 18, 19, 20}).
3. JOINT optimum: pick the (q, lgwin, lgblock) triple that minimizes
   SUM over tensors at that fixed triple.
4. PER-TENSOR optimum: for each tensor, pick the triple that minimizes
   ITS OWN bytes. Sum the per-tensor minima.
5. Headline: bytes saved by per-tensor HStack vs joint.

## CLAUDE.md compliance

family_falsified=False, falsification_scope set to the specific config
tested, ready_for_exact_eval_dispatch=False, cuda_eval_worth_testing=False
(this is a CPU encoder-byte test, not a score test).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import json
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_omega_opt_hstack_of_vstacks_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_hstack_of_vstacks_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Path-B-step-3 per-tensor brotli HStack test]"

# Brotli parameter grid (per-tensor sweep)
QUALITY_GRID = [9, 10, 11]
LGWIN_GRID = [14, 16, 18, 20, 22]
LGBLOCK_GRID = [17, 18, 19, 20]


def encode_tensor(symbols: np.ndarray, q: int, lgwin: int, lgblock: int) -> int:
    return len(brotli.compress(symbols.tobytes(), quality=q, lgwin=lgwin, lgblock=lgblock))


def run_experiment(state_dict_path: Path) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    quantized: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        quantized.append((name, qt.q_i8.flatten().astype(np.int8)))
    n_tensors = len(quantized)

    # Per-tensor sweep: dict[(q, lgwin, lgblock)] -> list[bytes per tensor]
    grid = list(itertools.product(QUALITY_GRID, LGWIN_GRID, LGBLOCK_GRID))
    print(f"  sweeping {len(grid)} param triples × {n_tensors} tensors = {len(grid) * n_tensors} brotli compressions...")
    by_triple: dict[tuple[int, int, int], list[int]] = {}
    for triple in grid:
        q, lgwin, lgblock = triple
        per_tensor = []
        for _, syms in quantized:
            per_tensor.append(encode_tensor(syms, q, lgwin, lgblock))
        by_triple[triple] = per_tensor

    # JOINT optimum: pick triple minimizing sum
    joint_sums = {triple: sum(per) for triple, per in by_triple.items()}
    joint_best_triple = min(joint_sums, key=joint_sums.get)
    joint_best_bytes = joint_sums[joint_best_triple]

    # PER-TENSOR optimum
    per_tensor_best: list[tuple[str, tuple[int, int, int], int]] = []
    per_tensor_total = 0
    for tensor_idx, (name, _) in enumerate(quantized):
        best_triple = None
        best_bytes = float("inf")
        for triple, per in by_triple.items():
            if per[tensor_idx] < best_bytes:
                best_bytes = per[tensor_idx]
                best_triple = triple
        per_tensor_best.append((name, best_triple, best_bytes))
        per_tensor_total += best_bytes

    # Side-channel cost: per-tensor params need to be transmitted.
    # 28 tensors × 3 params × 1 byte each = 84 bytes (worst-case raw encoding;
    # could be smaller with brotli on the param sequence, but 84 is the upper bound)
    per_tensor_param_overhead = n_tensors * 3
    per_tensor_total_with_overhead = per_tensor_total + per_tensor_param_overhead

    savings_raw = joint_best_bytes - per_tensor_total
    savings_net = joint_best_bytes - per_tensor_total_with_overhead

    # How many tensors picked something OTHER than the joint best?
    n_diverged = sum(1 for _, triple, _ in per_tensor_best if triple != joint_best_triple)

    # Distribution of choices
    choice_distribution: dict[str, int] = {}
    for _, triple, _ in per_tensor_best:
        key = f"q{triple[0]}_lgwin{triple[1]}_lgblock{triple[2]}"
        choice_distribution[key] = choice_distribution.get(key, 0) + 1

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": False,  # lossless, byte-identical decode
        "charged_bits_changed": True,  # the encoded bytes change
        "family_falsified": False,
        "falsification_scope": "per_tensor_brotli_param_HStack_only",
        "input_state_dict": str(state_dict_path),
        "n_tensors": n_tensors,
        "grid_size": len(grid),
        "joint_optimum_triple": list(joint_best_triple),
        "joint_optimum_bytes": joint_best_bytes,
        "per_tensor_optimum_bytes": per_tensor_total,
        "per_tensor_param_overhead_bytes": per_tensor_param_overhead,
        "per_tensor_optimum_with_overhead_bytes": per_tensor_total_with_overhead,
        "savings_raw_bytes": savings_raw,
        "savings_net_bytes": savings_net,
        "savings_pct": (savings_net / max(joint_best_bytes, 1)) * 100.0,
        "n_tensors_diverged_from_joint_best": n_diverged,
        "per_tensor_choice_distribution": choice_distribution,
        "per_tensor_choices": [
            {"tensor": name, "triple": list(triple), "bytes": b}
            for name, triple, b in per_tensor_best
        ],
        "headline": (
            f"HStack-of-VStacks per-tensor brotli HStack saves {savings_net:+,} bytes "
            f"({(savings_net / max(joint_best_bytes, 1)) * 100:.2f}%) net of {per_tensor_param_overhead}-byte "
            f"per-tensor param sidechannel; {n_diverged}/{n_tensors} tensors picked a non-joint-best triple"
        ),
        "dispatch_blockers": [
            "byte_savings_only_no_score_test",
            "lossless_HStack_does_not_change_decoder_output",
            "no_runtime_built_to_consume_per_tensor_params",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "per_tensor_codec_choice_HStack_brotli_vs_sparsity_vs_lossy",
            "per_tensor_quantization_level_HStack",
            "joint_param_HStack_with_lossy_components",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    print("Path B step 3: per-tensor brotli HStack-of-VStacks empirical anchor")
    print(f"  state_dict: {args.state_dict}")
    manifest = run_experiment(args.state_dict)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_hstack_of_vstacks_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    triple = manifest["joint_optimum_triple"]
    print(f"  JOINT optimum:        q={triple[0]} lgwin={triple[1]} lgblock={triple[2]}  =>  {manifest['joint_optimum_bytes']:>8,} B")
    print(f"  PER-TENSOR optimum:   28 independently-chosen triples            =>  {manifest['per_tensor_optimum_bytes']:>8,} B (raw)")
    print(f"  Param sidechannel:    {manifest['per_tensor_param_overhead_bytes']:>3} bytes")
    print(f"  PER-TENSOR + sidechan:                                              {manifest['per_tensor_optimum_with_overhead_bytes']:>8,} B")
    print(f"  Net savings:          {manifest['savings_net_bytes']:+,} bytes ({manifest['savings_pct']:+.3f}%)")
    print(f"  Divergence:           {manifest['n_tensors_diverged_from_joint_best']}/{manifest['n_tensors']} tensors picked non-joint-best")
    print("\n  Choice distribution (per-tensor selections):")
    for choice, count in sorted(manifest["per_tensor_choice_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"    {choice:30s}  {count:>3} tensors")

    if args.output_evidence:
        evidence_row = {
            "technique": "hstack_of_vstacks_per_tensor_brotli",
            "empirical_archive_bytes": manifest["per_tensor_optimum_with_overhead_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "per_tensor_brotli_HStack_byte_anchor_lossless_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "per_tensor_brotli_param_HStack_only",
            "score_affecting_payload_changed": False,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(per_tensor={manifest['per_tensor_optimum_with_overhead_bytes']:,}B vs "
                f"joint_best={manifest['joint_optimum_bytes']:,}B; "
                f"savings={manifest['savings_net_bytes']:+,}B; "
                f"diverged={manifest['n_tensors_diverged_from_joint_best']}/{manifest['n_tensors']})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": ["per_tensor_brotli_param_HStack"],
            "reactivation_criteria_remaining": manifest["reactivation_criteria_remaining"],
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
