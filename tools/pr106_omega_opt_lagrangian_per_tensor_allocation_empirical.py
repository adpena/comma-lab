#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR106 cross-substrate port of Path B step 5/6 + UNIWARD per-tensor Lagrangian.

Background
----------
All Path B + UNIWARD experiments to date were on PR101 substrate (228,958 INT8
weights, 28-tensor PR101 schema, brotli baseline 178,144 B). PR106 is the live
contest frontier (0.20454) and a different substrate:

    PR106 archive layout (parser-proven, monolithic-single-file):
        ff_header                       :     4 B
        decoder_packed_brotli           : 170,278 B   <-- charge
        latents_and_sidecar_brotli      :  15,849 B
        ZIP overhead                    :     108 B
        ----------------------------------------------------
        archive total                   : 186,239 B

    decoder_packed_brotli decompresses to 229,070 raw bytes:
        - 28 × q_zz_u8 streams (228,958 bytes total; same #symbols as PR101)
        - 28 × scale_f32 (112 bytes)

    Tensor schema differs from PR101 (different conv shapes, e.g. stem.weight
    (1728, 28) vs PR101's distinct shapes), but element COUNT is identical at
    228,958. This is the cross-substrate transfer test.

Cross-substrate hypothesis
--------------------------
The Lagrangian-allocation MECHANISM (per-tensor K coarsening with global RMS
rel_err target via λ-bisection) operates on the per-tensor int8 symbol stream.
Whether the underlying distribution differs by substrate is an empirical
question. This tool ports Path B step 6 and the UNIWARD-weighted variant
(Path B step 7) directly to PR106's substrate, without changes to the
Lagrangian primitive.

For each rms_target {0.01, 0.02, 0.05, 0.10} we report:

    baseline_lossless              decoder bytes when K=1 (no coarsening)
    greedy_per_tensor_budget       per-tensor K chosen independently
    lagrangian_uniform_weighting   Path B step 6 mechanism
    lagrangian_uniward_weighting   Path B step 7 (1/var) mechanism

All numbers are CPU-only byte proxies. Tagged
``[CPU-prep faithful PR106-cross-substrate-test]``. NO score claim,
NO promotion eligibility, NO dispatch readiness flag — per CLAUDE.md
"forbidden_CPU_MPS_derived_dispatch_readiness_flag" the cross-substrate result
must be byte-anchor only. Exact CUDA auth eval on a runtime-byte-closed packet
is required before ANY score-effect claim.

Falsification scope: ``pr106_lagrangian_per_tensor_only`` — only the
per-tensor-K Lagrangian over the int8 q_zz_u8 stream is tested. Joint
hyperprior, primal-dual ADMM consensus, wavelet-domain UNIWARD residual,
detector-in-loop weighting, and exact CUDA auth eval are NOT tested here.

CLAUDE.md non-negotiables observed
----------------------------------
- ``forbidden_premature_class_level_falsification``: failure tagged
  ``MEASURED_CONFIG_NOT_DISPATCHABLE``; family is NOT falsified.
- ``forbidden_CPU_MPS_derived_dispatch_readiness_flag``: ``ready_for_exact_eval_dispatch=False``.
- ``forbidden_score_claims``: emits no score; tags every byte
  ``[CPU-prep faithful PR106-cross-substrate-test]``.
- ``forbidden_/tmp_paths_in_persisted_artifacts``: outputs under
  ``experiments/results/`` and ``reports/raw/``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.hnerv_decoder_recode import (  # noqa: E402
    PACKED_STATE_SCHEMA,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
)

TOOL_NAME = "tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py"
SCHEMA_VERSION = "pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful PR106-cross-substrate-test]"

# PR106 archive overhead (everything that is NOT the decoder_packed_brotli):
#   ff_header (4) + latents_and_sidecar_brotli (15,849) + ZIP overhead (108)
# Concretely measured from
# experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip
PR106_ARCHIVE_OVERHEAD_BYTES = 15_961
PR106_DECODER_BROTLI_BASELINE_BYTES = 170_278
PR106_ARCHIVE_BASELINE_BYTES = 186_239

# K-range for per-tensor coarsening sweep
K_RANGE = list(range(1, 65))
EPS_VARIANCE = 1e-6
DEFAULT_PR106_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)


DISPATCH_BLOCKERS = [
    "byte_rel_err_proxy_only_no_score_test",
    "no_runtime_dequantize_path_built_for_modified_decoder",
    "missing_exact_cuda_auth_eval",
    "scale_per_tensor_fixed_at_lossless_during_K_sweep",
    "no_iterative_primal_dual_ADMM_consensus",
    "lossless_latents_and_sidecar_assumed_constant_no_joint_optimization",
]


@dataclass
class PR106TensorBlob:
    """One PR106 decoder tensor as a stream of int8 symbols.

    The PR106 packed format stores zigzag-uint8 symbols. We convert to int8
    (signed range) for K-coarsening, and emit the same wire format on encode.
    """

    name: str
    shape: tuple[int, ...]
    raw_i8: np.ndarray  # int32, range [-127, 127]
    scale_f32: bytes  # 4 bytes — preserved verbatim


def _zz_u8_to_i8(buf: bytes) -> np.ndarray:
    """Decode zigzag-uint8 to signed int8 array.

    PR106's packed_decoder uses zigzag encoding so positive/negative magnitudes
    interleave in unsigned [0, 255]. Standard zigzag: u = (n << 1) ^ (n >> 7).
    Inverse: n = (u >> 1) ^ -(u & 1).
    """
    arr_u8 = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
    return (arr_u8 >> 1) ^ -(arr_u8 & 1)


def _i8_to_zz_u8(arr: np.ndarray) -> bytes:
    """Encode signed int8 (as int32) to zigzag-uint8 bytes."""
    arr_i32 = arr.astype(np.int32)
    # zigzag: ((n << 1) ^ (n >> 31)) for 32-bit; clip to int8 first
    np.clip(arr_i32, -127, 127, out=arr_i32)
    zz = (arr_i32 << 1) ^ (arr_i32 >> 31)
    return zz.astype(np.uint8).tobytes()


def collect_pr106_tensors(archive_path: Path) -> list[PR106TensorBlob]:
    """Parse PR106 archive, return per-tensor int8 symbols + scales."""
    sma = read_strict_single_member_zip(archive_path)
    packed = parse_ff_packed_brotli_hnerv(sma.payload)
    parsed = parse_packed_decoder_brotli(packed.decoder_packed_brotli)
    if len(parsed.records) != len(PACKED_STATE_SCHEMA):
        raise ValueError(
            f"PR106 schema mismatch: {len(parsed.records)} vs {len(PACKED_STATE_SCHEMA)}"
        )
    tensors: list[PR106TensorBlob] = []
    for record in parsed.records:
        i8 = _zz_u8_to_i8(record.q_zz_u8)
        tensors.append(
            PR106TensorBlob(
                name=record.name,
                shape=record.shape,
                raw_i8=i8,
                scale_f32=record.scale_f32,
            )
        )
    return tensors


def _encode_decoder_brotli_with_per_tensor_K(
    tensors: list[PR106TensorBlob], Ks: list[int], brotli_quality: int = 11
) -> dict:
    """Apply per-tensor K coarsening, re-emit the decoder_packed_brotli wire,
    and report archive bytes + roundtrip rel_err.

    Wire format mirrors PR106 exactly:
        concat(zz_u8(round(tensor_i / K_i) * K_i)) || concat(scale_f32_i)
        then brotli(quality=11) — the same params PR106 uses.
    """
    abs_orig_total = 0.0
    abs_err_total = 0.0
    rounded_chunks_zz: list[bytes] = []
    for tb, K in zip(tensors, Ks, strict=True):
        rounded = np.round(tb.raw_i8 / K) * K
        err = float(np.abs(rounded - tb.raw_i8).astype(np.float64).sum())
        abs_err_total += err
        abs_orig_total += float(np.abs(tb.raw_i8).astype(np.float64).sum())
        rounded_clipped = np.clip(rounded, -127, 127).astype(np.int32)
        rounded_chunks_zz.append(_i8_to_zz_u8(rounded_clipped))

    # Reproduce PR106 wire format: concat all q_zz_u8 streams, then concat all
    # scale_f32 (preserved verbatim — K coarsens int8 magnitudes, not scales).
    q_concat = b"".join(rounded_chunks_zz)
    scales_concat = b"".join(tb.scale_f32 for tb in tensors)
    decoder_raw = q_concat + scales_concat
    decoder_brotli = brotli.compress(decoder_raw, quality=brotli_quality)

    archive_bytes = len(decoder_brotli) + PR106_ARCHIVE_OVERHEAD_BYTES
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0  # REL_ERR_NON_CANONICAL_OK: global L1 ratio for PR106 omega-opt joint-encoder bisect target; see allocator hook contract. See .omx/research/rel_err_inconsistency_audit_20260508_claude.md
    return {
        "decoder_brotli_bytes": len(decoder_brotli),
        "archive_overhead_bytes": PR106_ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "rel_err": rel_err,
        "Ks": list(Ks),
    }


def _find_best_K_per_tensor(symbols: np.ndarray, budget: float) -> tuple[int, float]:
    """Largest K such that per-tensor rel_err <= budget."""
    abs_sum = float(np.abs(symbols).astype(np.float64).sum())
    if abs_sum < 1e-9:
        return 1, 0.0
    best_K = 1
    best_re = 0.0
    for K in K_RANGE:
        rounded = np.round(symbols / K) * K
        err = float(np.abs(rounded - symbols).astype(np.float64).sum())
        re = err / abs_sum
        if re <= budget:
            best_K = K
            best_re = re
        else:
            break
    return best_K, best_re


def _precompute_K_curves(
    tensors: list[PR106TensorBlob],
) -> list[list[dict]]:
    """For each tensor, compute (K, rel_err, byte_proxy) over K_RANGE."""
    curves: list[list[dict]] = []
    for tb in tensors:
        rows = []
        abs_sum = float(np.abs(tb.raw_i8).astype(np.float64).sum()) + 1e-12
        for K in K_RANGE:
            rounded = np.round(tb.raw_i8 / K) * K
            err = float(np.abs(rounded - tb.raw_i8).astype(np.float64).sum())
            re = err / abs_sum
            rounded_clipped = np.clip(rounded, -127, 127).astype(np.int32)
            zz_u8 = _i8_to_zz_u8(rounded_clipped)
            byte_proxy = len(brotli.compress(zz_u8, quality=11))
            rows.append({"K": K, "rel_err": re, "byte_proxy": byte_proxy})
        curves.append(rows)
    return curves


def _lagrangian_select_uniform(
    curves: list[list[dict]], lam: float
) -> list[int]:
    Ks: list[int] = []
    for tensor_curve in curves:
        cost = [r["byte_proxy"] + lam * r["rel_err"] ** 2 for r in tensor_curve]
        idx = int(np.argmin(cost))
        Ks.append(tensor_curve[idx]["K"])
    return Ks


def _lagrangian_select_uniward(
    curves: list[list[dict]], weights: list[float], lam: float
) -> list[int]:
    Ks: list[int] = []
    for tensor_curve, w in zip(curves, weights, strict=True):
        cost = [r["byte_proxy"] + lam * w * r["rel_err"] ** 2 for r in tensor_curve]
        idx = int(np.argmin(cost))
        Ks.append(tensor_curve[idx]["K"])
    return Ks


def _bisect_for_target(
    tensors: list[PR106TensorBlob],
    curves: list[list[dict]],
    rms_target: float,
    weights: list[float] | None,
    max_iter: int = 80,
) -> dict:
    """Bisect λ such that joint-encoded rel_err <= rms_target.

    If weights is None: uniform Lagrangian (Path B step 6 mechanism).
    Otherwise: UNIWARD-weighted Lagrangian (Path B step 7 mechanism).
    """
    lo, hi = 0.0, 1e15
    last_result = None
    cache: dict[tuple[int, ...], dict] = {}
    for _ in range(max_iter):
        mid = (lo + hi) / 2 if hi < 1e15 else lo * 10 + 1
        if weights is None:
            Ks = _lagrangian_select_uniform(curves, mid)
        else:
            Ks = _lagrangian_select_uniward(curves, weights, mid)
        key = tuple(Ks)
        if key in cache:
            result = cache[key]
        else:
            result = _encode_decoder_brotli_with_per_tensor_K(tensors, Ks)
            cache[key] = result
        rms = result["rel_err"]
        last_result = {"lambda": mid, **result, "rms_rel_err": rms}
        if rms <= rms_target:
            hi = mid
        else:
            lo = mid
        if hi == lo or abs(hi - lo) < 1e-12:
            break
    assert last_result is not None
    return last_result


def _greedy_per_tensor_budget(
    tensors: list[PR106TensorBlob], budget: float
) -> dict:
    Ks = []
    for tb in tensors:
        K, _ = _find_best_K_per_tensor(tb.raw_i8, budget=budget)
        Ks.append(K)
    return _encode_decoder_brotli_with_per_tensor_K(tensors, Ks)


def _compute_uniward_weights(tensors: list[PR106TensorBlob]) -> list[float]:
    """w(t) = 1 / (var(t) + eps) — inverse-variance per CLAUDE.md UNIWARD rule."""
    weights: list[float] = []
    for tb in tensors:
        var = float(np.var(tb.raw_i8.astype(np.float64))) if tb.raw_i8.size > 0 else 0.0
        weights.append(1.0 / (var + EPS_VARIANCE))
    return weights


def run_experiment(archive_path: Path, rms_targets: list[float]) -> dict:
    print(f"  loading PR106 archive: {archive_path}")
    tensors = collect_pr106_tensors(archive_path)
    n_tensors = len(tensors)
    n_symbols = sum(tb.raw_i8.size for tb in tensors)
    print(f"  {n_tensors} tensors, {n_symbols:,} int8 symbols")

    print(f"  precomputing per-tensor K curves ({n_tensors} × {len(K_RANGE)})...")
    curves = _precompute_K_curves(tensors)
    weights = _compute_uniward_weights(tensors)
    var_min = min(1.0 / (w + 1e-30) for w in weights) if weights else 0.0
    var_max = max(1.0 / (w + 1e-30) for w in weights) if weights else 0.0
    print(
        f"  variance range: [{var_min:.3e}, {var_max:.3e}], "
        f"weight ratio max/min = {max(weights)/min(weights):.2e}"
    )

    baseline = _encode_decoder_brotli_with_per_tensor_K(tensors, [1] * n_tensors)
    print(
        f"  baseline (K=1, lossless): decoder_brotli={baseline['decoder_brotli_bytes']:,} B, "
        f"archive={baseline['archive_bytes']:,} B"
    )

    comparison: list[dict] = []
    for rms_t in rms_targets:
        greedy = _greedy_per_tensor_budget(tensors, rms_t)
        lagr_uniform = _bisect_for_target(tensors, curves, rms_t, weights=None)
        lagr_uniward = _bisect_for_target(tensors, curves, rms_t, weights=weights)

        savings_uniward_vs_greedy = (
            greedy["archive_bytes"] - lagr_uniward["archive_bytes"]
        )
        savings_uniward_vs_uniform = (
            lagr_uniform["archive_bytes"] - lagr_uniward["archive_bytes"]
        )
        savings_uniform_vs_greedy = (
            greedy["archive_bytes"] - lagr_uniform["archive_bytes"]
        )
        comparison.append(
            {
                "rms_target": rms_t,
                "greedy_per_tensor_budget": {
                    "archive_bytes": greedy["archive_bytes"],
                    "decoder_brotli_bytes": greedy["decoder_brotli_bytes"],
                    "rel_err": greedy["rel_err"],
                },
                "lagrangian_uniform": {
                    "archive_bytes": lagr_uniform["archive_bytes"],
                    "decoder_brotli_bytes": lagr_uniform["decoder_brotli_bytes"],
                    "rel_err": lagr_uniform["rel_err"],
                    "lambda": lagr_uniform["lambda"],
                    "Ks": lagr_uniform["Ks"],
                },
                "lagrangian_uniward": {
                    "archive_bytes": lagr_uniward["archive_bytes"],
                    "decoder_brotli_bytes": lagr_uniward["decoder_brotli_bytes"],
                    "rel_err": lagr_uniward["rel_err"],
                    "lambda": lagr_uniward["lambda"],
                    "Ks": lagr_uniward["Ks"],
                },
                "uniward_savings_vs_greedy_bytes": savings_uniward_vs_greedy,
                "uniward_savings_vs_uniform_bytes": savings_uniward_vs_uniform,
                "uniform_savings_vs_greedy_bytes": savings_uniform_vs_greedy,
            }
        )

    best_uniward = max(
        comparison, key=lambda r: r["uniward_savings_vs_uniform_bytes"]
    )
    best_uniform = max(
        comparison, key=lambda r: r["uniform_savings_vs_greedy_bytes"]
    )
    best_overall_archive = min(
        comparison,
        key=lambda r: min(
            r["greedy_per_tensor_budget"]["archive_bytes"],
            r["lagrangian_uniform"]["archive_bytes"],
            r["lagrangian_uniward"]["archive_bytes"],
        ),
    )
    best_overall_min = min(
        best_overall_archive["greedy_per_tensor_budget"]["archive_bytes"],
        best_overall_archive["lagrangian_uniform"]["archive_bytes"],
        best_overall_archive["lagrangian_uniward"]["archive_bytes"],
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": "pr106_cross_substrate_lagrangian_per_tensor_allocation_byte_anchor_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "pr106_lagrangian_per_tensor_only",
        "input_archive": str(archive_path),
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "K_range": [K_RANGE[0], K_RANGE[-1]],
        "rms_targets_swept": rms_targets,
        "baseline_lossless_archive_bytes": baseline["archive_bytes"],
        "baseline_lossless_decoder_brotli_bytes": baseline["decoder_brotli_bytes"],
        "pr106_published_archive_bytes": PR106_ARCHIVE_BASELINE_BYTES,
        "pr106_published_decoder_brotli_bytes": PR106_DECODER_BROTLI_BASELINE_BYTES,
        "pr106_archive_overhead_bytes": PR106_ARCHIVE_OVERHEAD_BYTES,
        "comparison_at_rms_targets": comparison,
        "best_uniward_savings_vs_uniform_bytes": best_uniward[
            "uniward_savings_vs_uniform_bytes"
        ],
        "best_uniward_savings_rms_target": best_uniward["rms_target"],
        "best_uniform_savings_vs_greedy_bytes": best_uniform[
            "uniform_savings_vs_greedy_bytes"
        ],
        "best_uniform_savings_rms_target": best_uniform["rms_target"],
        "best_overall_min_archive_bytes": best_overall_min,
        "best_overall_min_archive_rms_target": best_overall_archive["rms_target"],
        "headline": (
            "PR106 cross-substrate Path B step 6 + step 7 (UNIWARD): "
            f"baseline={baseline['archive_bytes']:,} B; "
            f"best_uniward_vs_uniform={best_uniward['uniward_savings_vs_uniform_bytes']:+,} B "
            f"@ rms={best_uniward['rms_target']}; "
            f"best_overall_min_archive={best_overall_min:,} B "
            f"@ rms={best_overall_archive['rms_target']}"
        ),
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "reactivation_criteria_remaining": [
            "byte_closed_runtime_packet_with_modified_decoder",
            "exact_cuda_auth_eval_on_runtime_packet",
            "joint_optimization_over_latents_and_sidecar_brotli",
            "wavelet_domain_uniward_residual_variance_proxy",
            "score_aware_per_tensor_distortion_weights_detector_in_loop",
            "iterative_primal_dual_ADMM_with_consensus",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_PR106_ARCHIVE,
        help="PR106 contest-frontier archive zip.",
    )
    p.add_argument(
        "--rms-targets",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.10],
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.archive.is_file():
        raise SystemExit(f"PR106 archive not found: {args.archive}")

    print(
        "PR106 cross-substrate Path B step 6 + step 7 (UNIWARD-weighted) Lagrangian"
    )
    manifest = run_experiment(args.archive, args.rms_targets)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = (
            REPO_ROOT
            / f"reports/raw/pr106_lagrangian_per_tensor_allocation_{ts}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(
        f"  Baseline (K=1, lossless): archive={manifest['baseline_lossless_archive_bytes']:,} B "
        f"(decoder_brotli={manifest['baseline_lossless_decoder_brotli_bytes']:,} B)"
    )
    print(
        f"  PR106 published baseline: archive={manifest['pr106_published_archive_bytes']:,} B "
        f"(decoder_brotli={manifest['pr106_published_decoder_brotli_bytes']:,} B)\n"
    )
    print(
        f"  {'rms_target':>10s} | {'greedy':>10s} | {'uniform':>10s} {'uniform_λ':>10s} | "
        f"{'uniward':>10s} {'uniward_λ':>10s} | {'Δ_uniw_vs_unif':>14s}"
    )
    for r in manifest["comparison_at_rms_targets"]:
        g = r["greedy_per_tensor_budget"]
        u = r["lagrangian_uniform"]
        w = r["lagrangian_uniward"]
        print(
            f"  {r['rms_target']:>10.4f} | {g['archive_bytes']:>10,} | "
            f"{u['archive_bytes']:>10,} {u['lambda']:>10.2e} | "
            f"{w['archive_bytes']:>10,} {w['lambda']:>10.2e} | "
            f"{r['uniward_savings_vs_uniform_bytes']:>+14,}"
        )
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        # Anchor row: pick best-overall-min config
        target_rms = manifest["best_overall_min_archive_rms_target"]
        target = next(
            r for r in manifest["comparison_at_rms_targets"] if r["rms_target"] == target_rms
        )
        # Choose the smallest of the three families at that RMS
        candidates = [
            ("greedy_per_tensor_budget", target["greedy_per_tensor_budget"]),
            ("lagrangian_uniform", target["lagrangian_uniform"]),
            ("lagrangian_uniward", target["lagrangian_uniward"]),
        ]
        best_kind, best_data = min(candidates, key=lambda x: x[1]["archive_bytes"])

        evidence_row = {
            "technique": f"pr106_lagrangian_per_tensor_allocation_{best_kind}",
            "empirical_archive_bytes": best_data["archive_bytes"],
            "empirical_rel_err": best_data["rel_err"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": manifest["evidence_semantics"],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "pr106_lagrangian_per_tensor_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(rms_target={target_rms}; "
                f"best_kind={best_kind}; "
                f"archive_bytes={best_data['archive_bytes']:,}; "
                f"vs_pr106_published="
                f"{best_data['archive_bytes'] - PR106_ARCHIVE_BASELINE_BYTES:+,})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": [
                "pr106_cross_substrate_per_tensor_K_lagrangian_uniform",
                "pr106_cross_substrate_per_tensor_K_lagrangian_uniward",
                "pr106_cross_substrate_per_tensor_K_greedy_budget",
            ],
            "reactivation_criteria_remaining": manifest[
                "reactivation_criteria_remaining"
            ],
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
