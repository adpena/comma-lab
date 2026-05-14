#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 multi-pass IMP composition — Path B step 2 empirical anchor for the
second level of the PARADIGM-Ω-OPT design (multi-pass IMP-cycle, predicted
0.115).

## Config space enumeration

Technique class: multi-pass iterative magnitude pruning (IMP) composition
Full config space: { single_pass_at_final_alpha,
                     multi_pass_with_retrain,
                     multi_pass_post_hoc_no_retrain }
This tool tests: { multi_pass_post_hoc_no_retrain }
Falsification scope: this tool can only constrain the post-hoc layer of
multi-pass IMP. The multi-pass-WITH-retrain config (the classical IMP) is
EXPLICITLY OUT-OF-SCOPE and requires GPU + retraining; this tool does NOT
falsify it.

## What multi-pass post-hoc IMP buys (hypothesis to test)

Classical IMP (Frankle & Carbin 2018): train → prune lowest-magnitude
weights → RETRAIN surviving weights → prune again → repeat. The retrain
step is what redistributes representational capacity into surviving weights;
without it, multi-pass = single-pass-at-equivalent-sparsity by
magnitude-pruning's monotonicity.

The Ω-OPT linear-stack level (already anchored at 41,303 B) assumed each
component composes independently. The multi-pass IMP level (predicted 0.115,
-15bp from linear) assumes multi-pass adds value beyond single-pass.

This tool tests the WEAKEST form of that hypothesis: at the post-hoc layer
(no retrain), does iterative pass-by-pass coalescing + brotli-ing produce a
smaller archive than single-pass sparsify at equivalent final sparsity?

Expected result: NEAR-ZERO byte difference — the brotli payload is
determined by the final zero pattern, which is identical between
multi-pass post-hoc and single-pass at equal final alpha.

If true: this CONSTRAINS the Ω-OPT multi-pass-IMP -15bp prediction to come
from the RETRAIN component, NOT the multi-pass-coalesce component. The
predicted -15bp delta is a GPU-retrain claim, not a post-hoc claim.

## Method

For final_alpha in {0.5, 0.7, 0.9}:
  - SINGLE-PASS: sparsify all tensors to final_alpha → encode → brotli
  - MULTI-PASS-2: sparsify to sqrt(final_alpha)=alpha1 → coalesce → encode →
    re-quantize survivors → sparsify those to alpha1 again → encode → brotli
    (final coverage = alpha1 + (1-alpha1)*alpha1 = 1 - (1-alpha1)^2)
  - MULTI-PASS-3: same with 3 passes, alpha1 = 1 - (1-final_alpha)^(1/3)

Compare archive bytes at EQUAL final sparsity. The delta isolates the
"intermediate coalesce" effect.

## CLAUDE.md compliance

Per `forbidden_premature_class_level_falsification`: the verdict applies
ONLY to the post-hoc-no-retrain config tested. Multi-pass-with-retrain
remains UNTESTED. Per `forbidden_CPU_MPS_derived_dispatch_readiness_flag`:
ready_for_exact_eval_dispatch=False ALWAYS. Use cuda_eval_worth_testing
for any positive recommendation.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
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

TOOL_NAME = "tools/pr101_omega_opt_multipass_imp_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_multipass_imp_empirical.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
EVIDENCE_GRADE = "[CPU-prep faithful Path-B-step-2 multi-pass-post-hoc test]"


def sparsify_int8(symbols: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (nz_indices, nz_values) keeping top (1-alpha) by abs magnitude."""
    n = symbols.size
    if alpha <= 0 or n == 0:
        return np.arange(n, dtype=np.uint32), symbols.flatten().astype(np.int8)
    n_keep = max(1, int(round((1.0 - alpha) * n)))
    if n_keep >= n:
        return np.arange(n, dtype=np.uint32), symbols.flatten().astype(np.int8)
    abs_vals = np.abs(symbols.flatten().astype(np.int32))
    top_idx = np.argpartition(abs_vals, n - n_keep)[n - n_keep:]
    top_idx_sorted = np.sort(top_idx)
    return top_idx_sorted.astype(np.uint32), symbols.flatten()[top_idx_sorted].astype(np.int8)


def encode_csr(nz_indices: np.ndarray, nz_values: np.ndarray, n_total: int) -> bytes:
    if nz_values.size == 0:
        return struct.pack("<II", n_total, 0)
    deltas = np.diff(np.concatenate([np.array([0], dtype=np.uint32), nz_indices])).astype(np.uint32)
    return (
        struct.pack("<II", n_total, nz_values.size)
        + deltas.tobytes()
        + nz_values.tobytes()
    )


def assemble_and_brotli(per_tensor_payloads: list[bytes], scales: list[float]) -> int:
    scales_blob = np.array(scales, dtype=np.float16).tobytes()
    full_blob = scales_blob + b"".join(struct.pack("<I", len(p)) + p for p in per_tensor_payloads)
    return len(brotli.compress(full_blob, quality=11, lgwin=16, lgblock=19))


def measure_single_pass(quantized: list[tuple[str, np.ndarray, float]], alpha: float) -> dict:
    payloads, scales = [], []
    n_zeroed = n_total = 0
    for _, syms, scale in quantized:
        scales.append(scale)
        nz_idx, nz_val = sparsify_int8(syms, alpha)
        payloads.append(encode_csr(nz_idx, nz_val, syms.size))
        n_total += syms.size
        n_zeroed += (syms.size - nz_val.size)
    brotli_b = assemble_and_brotli(payloads, scales)
    return {
        "mode": "single_pass",
        "final_alpha": alpha,
        "passes": 1,
        "fraction_zeroed": n_zeroed / max(n_total, 1),
        "brotli_bytes": brotli_b,
        "archive_bytes": brotli_b + ARCHIVE_OVERHEAD_BYTES,
    }


def measure_multi_pass(
    quantized: list[tuple[str, np.ndarray, float]],
    final_alpha: float,
    n_passes: int,
) -> dict:
    """Apply n_passes of post-hoc magnitude pruning, each removing
    1 - (1-final_alpha)^(1/n_passes) of *currently surviving* weights."""
    per_pass_alpha = 1.0 - (1.0 - final_alpha) ** (1.0 / n_passes)
    # Track surviving symbols + their original indices per tensor.
    state: list[tuple[str, np.ndarray, np.ndarray, float]] = []  # (name, surviving_syms, original_indices, scale)
    for name, syms, scale in quantized:
        state.append((name, syms.copy(), np.arange(syms.size, dtype=np.uint32), scale))

    for _ in range(n_passes):
        new_state = []
        for name, syms, orig_idx, scale in state:
            nz_local_idx, nz_val = sparsify_int8(syms, per_pass_alpha)
            kept_orig = orig_idx[nz_local_idx]
            new_state.append((name, nz_val.astype(np.int8), kept_orig, scale))
        state = new_state

    # Final encode using ORIGINAL indices (so consumer can reconstruct full tensor).
    payloads, scales = [], []
    n_zeroed = n_total = 0
    for (name, syms, scale), (_, surviving_vals, orig_indices, _) in zip(quantized, state, strict=True):
        scales.append(scale)
        payloads.append(encode_csr(orig_indices, surviving_vals, syms.size))
        n_total += syms.size
        n_zeroed += (syms.size - surviving_vals.size)
    brotli_b = assemble_and_brotli(payloads, scales)
    return {
        "mode": "multi_pass_post_hoc",
        "final_alpha": final_alpha,
        "passes": n_passes,
        "per_pass_alpha": per_pass_alpha,
        "fraction_zeroed": n_zeroed / max(n_total, 1),
        "brotli_bytes": brotli_b,
        "archive_bytes": brotli_b + ARCHIVE_OVERHEAD_BYTES,
    }


def run_experiment(state_dict_path: Path, final_alphas: list[float], pass_counts: list[int]) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    quantized: list[tuple[str, np.ndarray, float]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        quantized.append((name, qt.q_i8.flatten(), float(qt.scale)))
    n_total = sum(q.size for _, q, _ in quantized)

    results: list[dict] = []
    for alpha in final_alphas:
        sp = measure_single_pass(quantized, alpha)
        results.append(sp)
        for p in pass_counts:
            mp = measure_multi_pass(quantized, alpha, p)
            mp["delta_vs_single_pass_bytes"] = mp["archive_bytes"] - sp["archive_bytes"]
            results.append(mp)

    # Compute headline finding
    deltas_by_alpha: dict[float, list[int]] = {}
    for r in results:
        if r["mode"].startswith("multi_pass"):
            deltas_by_alpha.setdefault(r["final_alpha"], []).append(r["delta_vs_single_pass_bytes"])
    avg_abs_delta_b = (
        float(np.mean([abs(d) for ds in deltas_by_alpha.values() for d in ds]))
        if deltas_by_alpha else 0.0
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "family_falsified": False,
        "falsification_scope": "post_hoc_no_retrain_configuration_only",
        "input_state_dict": str(state_dict_path),
        "n_total_elements": n_total,
        "final_alphas_swept": final_alphas,
        "pass_counts_swept": pass_counts,
        "results": results,
        "headline": {
            "avg_abs_delta_bytes_multipass_vs_singlepass": avg_abs_delta_b,
            "interpretation": (
                "Post-hoc multi-pass IMP composition byte-impact vs single-pass at equal final sparsity. "
                "Near-zero delta confirms the hypothesis that post-hoc magnitude-pruning is monotonic — "
                "multi-pass coalescing buys nothing without retraining. "
                "If non-zero, brotli's compression is sensitive to the specific zero-pattern path."
            ),
        },
        "dispatch_blockers": [
            "post_hoc_no_retrain_substrate_only",
            "multi_pass_with_retrain_NOT_tested_requires_GPU",
            "score_impact_unknown_without_contest_cuda",
            "no_decoder_runtime_built_for_csr_format",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "multi_pass_with_retrain_GPU_test",
            "score_impact_via_contest_CUDA_after_retrain",
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
    p.add_argument("--final-alphas", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--pass-counts", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_experiment(args.state_dict, args.final_alphas, args.pass_counts)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_multipass_imp_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}\n")
    print("  alpha | passes | fraction_zeroed | archive_bytes | delta_vs_1p")
    for r in manifest["results"]:
        d = r.get("delta_vs_single_pass_bytes", 0)
        print(f"  {r['final_alpha']:>5.2f} | {r['passes']:>6d} | {r['fraction_zeroed']:>14.4f} | {r['archive_bytes']:>13,} | {d:>+12,} B")
    print(f"\nheadline: avg |Δ| = {manifest['headline']['avg_abs_delta_bytes_multipass_vs_singlepass']:.1f} B")
    print(f"          {manifest['headline']['interpretation']}")

    if args.output_evidence:
        evidence_row = {
            "technique": "multipass_imp_post_hoc_composition",
            "empirical_archive_bytes": manifest["results"][0]["archive_bytes"],  # singlepass alpha[0] anchor
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "post_hoc_multipass_vs_singlepass_byte_anchor_no_retrain_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "post_hoc_no_retrain_configuration_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(avg |Δ multipass-vs-singlepass| = {manifest['headline']['avg_abs_delta_bytes_multipass_vs_singlepass']:.1f} B)"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": ["post_hoc_no_retrain_multi_pass"],
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
