#!/usr/bin/env python3
"""PR101 per-tensor codec-CHOICE HStack — Path B step 4 empirical anchor.

The Ω-OPT HStack-of-VStacks predicted -5bp savings did NOT materialize at
the encoder-parameter layer (Path B step 3: -40 B net after sidechannel).
The reactivation criterion was: test the higher-dimensional codec-CHOICE
search (brotli vs sparsity vs lossy per tensor).

This tool tests that. For each tensor, evaluate K codecs:
  C0: lossless brotli on int8 symbols
  C1: sparsity alpha=0.3 + CSR + brotli  (lossy: 30% of weights zeroed)
  C2: sparsity alpha=0.5 + CSR + brotli  (lossy: 50%)
  C3: sparsity alpha=0.7 + CSR + brotli  (lossy: 70%)
  C4: sparsity alpha=0.9 + CSR + brotli  (lossy: 90%)

Each codec produces (bytes, induced rel_err) per tensor. Then for a series
of GLOBAL rel_err budgets {0%, 1%, 2%, 5%, 10%}, run a greedy selector
that minimizes total bytes subject to weighted rel_err constraint.

## Config space enumeration

Technique class: per-tensor codec-CHOICE HStack-of-VStacks
Full config space: { brotli_only, sparsity_at_K_alphas,
                     int4_quantize, lossy_coarsening, hyperprior_per_tensor,
                     ... }
This tool tests: { brotli_only, sparsity_at_5_alphas } subset.
Falsification scope: this tool can constrain the brotli+sparsity HStack
only. The int4/lossy_coarsening/hyperprior per-tensor variants remain
UNTESTED.

## What this tests vs prior anchors

| Path B step | Codec axis tested | Net byte savings |
|---|---|---|
| 2 (multi-pass IMP) | Pass count | ~0 (avg |Δ| 64 B) |
| 3 (HStack brotli params) | (q, lgwin, lgblock) per tensor | -40 B net |
| **4 (this)** | **codec choice {brotli, sparsity@α} per tensor** | **TBD** |

The expectation: per-tensor codec choice DOES yield positive savings
because the choice space is larger and the codecs have meaningfully
different (bytes, rel_err) profiles per tensor.

## Important: this is BYTE + REL_ERR test, not score test

ready_for_exact_eval_dispatch=False ALWAYS. Sparsity induces per-tensor
distortion that propagates to score; the score impact CANNOT be inferred
from rel_err alone — it requires CUDA + decoder runtime + auth eval.
This tool produces a PROXY-FRONTIER (bytes vs rel_err); the score-frontier
mapping is RESEARCH, not measured.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.per_tensor_codecs import (  # noqa: E402
    encode_brotli_only,
    encode_sparsity_alpha,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_omega_opt_per_tensor_codec_choice_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_per_tensor_codec_choice_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Path-B-step-4 per-tensor codec-CHOICE HStack test]"

# encode_brotli_only and encode_sparsity_alpha re-exported from
# tac.codec.per_tensor_codecs to remove drift across tools.


def measure_per_tensor_codecs(quantized: list[tuple[str, np.ndarray]], alphas: list[float]) -> list[dict]:
    """For each tensor, measure (bytes, rel_err) for each codec choice."""
    rows: list[dict] = []
    for name, syms in quantized:
        codecs: list[dict] = []
        b0, e0 = encode_brotli_only(syms)
        codecs.append({"codec_id": 0, "label": "brotli_only", "alpha": 0.0, "bytes": b0, "rel_err": e0})
        for i, alpha in enumerate(alphas):
            b, e = encode_sparsity_alpha(syms, alpha)
            codecs.append({"codec_id": i + 1, "label": f"sparsity_a{alpha}", "alpha": alpha, "bytes": b, "rel_err": e})
        rows.append({"tensor": name, "n_elements": int(syms.size), "codecs": codecs})
    return rows


def greedy_select_under_rel_err_budget(
    per_tensor: list[dict],
    global_rel_err_budget: float,
) -> dict:
    """Greedy: for each tensor, sort codecs by (bytes/n_elements + lambda*rel_err),
    iteratively reduce lambda to fit budget. Simpler implementation: enumerate
    all combinations? Too expensive (5^28). Instead: for each tensor, use the
    Pareto-optimal codec at the most-aggressive alpha that keeps per-tensor
    rel_err below a per-tensor share of the global budget.

    A clean greedy: per-tensor share = global_budget. For each tensor, pick the
    most-aggressive codec whose rel_err <= per-tensor share. Adjust per-tensor
    share by re-running until total rel_err matches budget exactly (we use a
    simpler fixed share for empirical anchor purposes).
    """
    # Approach: for each tensor, choose the codec with the largest alpha whose
    # rel_err <= global_rel_err_budget. (Per-tensor budget = global budget;
    # actual achieved rel_err may be lower.)
    selections: list[dict] = []
    total_bytes = 0
    for row in per_tensor:
        valid = [c for c in row["codecs"] if c["rel_err"] <= global_rel_err_budget]
        # pick the one with smallest bytes (most aggressive that's still valid)
        best = min(valid, key=lambda c: c["bytes"])
        selections.append({"tensor": row["tensor"], "chosen_codec": best["label"], "bytes": best["bytes"], "rel_err": best["rel_err"], "alpha": best["alpha"]})
        total_bytes += best["bytes"]

    # Element-weighted L2 aggregate. The earlier unweighted mean over tensor
    # rows over-counted tiny tensors and under-counted large tensors; this
    # proxy is still not score evidence, but at least matches the symbol mass
    # represented by each per-tensor rel_err.
    total_elements = sum(int(row["n_elements"]) for row in per_tensor)
    weighted_sq = 0.0
    for row, selection in zip(per_tensor, selections):
        selection["n_elements"] = int(row["n_elements"])
        weighted_sq += float(row["n_elements"]) * float(selection["rel_err"]) ** 2
    total_rel_err = float(np.sqrt(weighted_sq / max(1, total_elements)))
    return {
        "rel_err_budget": global_rel_err_budget,
        "total_bytes": total_bytes,
        "achieved_total_rel_err_element_weighted_l2": total_rel_err,
        "achieved_total_rel_err_l2_avg": total_rel_err,
        "rel_err_form": "element_weighted_l2_over_per_tensor_rel_err",
        "total_elements": int(total_elements),
        "n_tensors_sparsified": sum(1 for s in selections if s["alpha"] > 0),
        "selections": selections,
    }


def run_experiment(state_dict_path: Path, alphas: list[float], rel_err_budgets: list[float]) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    quantized: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        quantized.append((name, qt.q_i8.flatten().astype(np.int8)))
    n_tensors = len(quantized)

    # Per-tensor codec measurements
    print(f"  measuring {n_tensors} tensors × {len(alphas) + 1} codecs = {n_tensors * (len(alphas) + 1)} encodings...")
    per_tensor = measure_per_tensor_codecs(quantized, alphas)

    # Baseline: all-lossless per-tensor brotli
    baseline_bytes = sum(row["codecs"][0]["bytes"] for row in per_tensor)

    # Per-budget HStack selections
    frontier = []
    for budget in rel_err_budgets:
        sel = greedy_select_under_rel_err_budget(per_tensor, budget)
        sidechannel_bytes = n_tensors * 1  # 1 byte per tensor for codec_id (5 codecs fit)
        sel["per_tensor_codec_id_sidechannel_bytes"] = sidechannel_bytes
        sel["total_bytes_with_sidechannel"] = sel["total_bytes"] + sidechannel_bytes
        sel["savings_vs_baseline_lossless"] = baseline_bytes - sel["total_bytes_with_sidechannel"]
        frontier.append(sel)

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,  # sparsity induces per-tensor reconstruction error
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "per_tensor_brotli_OR_sparsity_HStack_only",
        "input_state_dict": str(state_dict_path),
        "n_tensors": n_tensors,
        "alphas_swept": alphas,
        "rel_err_budgets_swept": rel_err_budgets,
        "rel_err_form": "element_weighted_l2_over_per_tensor_rel_err",
        "baseline_lossless_per_tensor_brotli_bytes": baseline_bytes,
        "per_tensor_codec_measurements": per_tensor,
        "frontier_at_rel_err_budgets": frontier,
        "headline": (
            f"Per-tensor codec-CHOICE HStack frontier vs all-lossless baseline "
            f"({baseline_bytes:,} B). At rel_err budgets {rel_err_budgets}, savings range "
            f"from {min(s['savings_vs_baseline_lossless'] for s in frontier):+,} B to "
            f"{max(s['savings_vs_baseline_lossless'] for s in frontier):+,} B (with sidechannel)."
        ),
        "dispatch_blockers": [
            "byte_rel_err_proxy_only_no_score_test",
            "sparsity_induces_distortion_score_impact_unknown",
            "no_runtime_built_to_consume_per_tensor_codec_ids_with_CSR",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "per_tensor_int4_quantize_HStack",
            "per_tensor_lossy_coarsening_HStack",
            "per_tensor_hyperprior_HStack",
            "joint_codec_choice_with_retrain",
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
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--rel-err-budgets", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.10])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    print(f"Path B step 4: per-tensor codec-CHOICE HStack-of-VStacks empirical anchor")
    manifest = run_experiment(args.state_dict, args.alphas, args.rel_err_budgets)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_per_tensor_codec_choice_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(f"  Baseline (all-lossless per-tensor brotli): {manifest['baseline_lossless_per_tensor_brotli_bytes']:,} B")
    print(f"\n  HStack frontier (per-tensor codec choice):")
    print(f"  {'rel_err_budget':>14s} | {'achieved':>10s} | {'bytes':>10s} | {'+sidechan':>10s} | {'savings':>10s} | {'sparsified':>4s}/{manifest['n_tensors']}")
    for sel in manifest["frontier_at_rel_err_budgets"]:
        print(
            f"  {sel['rel_err_budget']:>14.4f} | {sel['achieved_total_rel_err_l2_avg']:>10.5f} "
            f"| {sel['total_bytes']:>10,} | {sel['total_bytes_with_sidechannel']:>10,} "
            f"| {sel['savings_vs_baseline_lossless']:>+10,} | {sel['n_tensors_sparsified']:>4d}"
        )

    if args.output_evidence:
        # Anchor row: pick the budget=0.05 result (5% rel_err, plausibly score-relevant)
        target = next((s for s in manifest["frontier_at_rel_err_budgets"] if abs(s["rel_err_budget"] - 0.05) < 1e-9), manifest["frontier_at_rel_err_budgets"][0])
        evidence_row = {
            "technique": "hstack_per_tensor_codec_choice_brotli_or_sparsity",
            "empirical_archive_bytes": target["total_bytes_with_sidechannel"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "per_tensor_codec_choice_HStack_byte_anchor_at_rel_err_budget_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "per_tensor_brotli_OR_sparsity_HStack_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(rel_err_budget=0.05; achieved={target['achieved_total_rel_err_l2_avg']:.4f}; "
                f"bytes={target['total_bytes_with_sidechannel']:,}; "
                f"savings_vs_lossless={target['savings_vs_baseline_lossless']:+,}; "
                f"sparsified={target['n_tensors_sparsified']}/{manifest['n_tensors']})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": ["per_tensor_codec_choice_brotli_OR_sparsity"],
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
