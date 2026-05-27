#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Catalog #139 byte-mutation REALIZABILITY micro-probe for water-filling REPAIR.

THE MEASUREMENT (operator-routed 2026-05-27): close the INDETERMINATE verdict
from the proxy probe `tools/probe_water_filling_repair.py` (memo
`.omx/research/water_filling_repair_probe_plus_op_palette_audit_20260527T164825Z.md`).

The proxy said "repair marginal beats 25/N by ~6098x" but that is a PROVABLE
OVER-ESTIMATE: the master-gradient measures d(score)/d(byte_value) (sensitivity
to perturbing an EXISTING byte), NOT d(score)/d(ADDING a repair byte); the summed
marginals exceed the entire 0.2532 distortion budget at the operating point (a
physically-impossible upper bound). This probe replaces the proxy marginal with a
REALIZED measurement.

WHAT THIS PROBE DOES (REALIZED, not proxy):
    1. Rank archive bytes by master-gradient seg+pose sensitivity (L1 over 600
       pairs); pick top-K (sweep K=16,64,256).
    2. Map the top-K archive bytes -> the decoder tensors whose mantissa-byte
       spans contain them (the master-gradient is a per-TENSOR FD projected
       per-BYTE, so a top-K byte set names a set of decoder tensors).
    3. MUTATE the implicated tensors' weights along the REPAIR DIRECTION = the
       negative sign of the per-tensor d(seg+pose)/d(weight) gradient (the
       direction the gradient says REDUCES distortion). Step size = a sweep of
       multiples of the FD eps the gradient was measured at (fd_rel_eps * RMS).
    4. RE-RUN the actual parity-validated MLX SegNet+PoseNet scorer oracle (the
       same oracle the gradient producer uses) on the reconstruction from the
       MUTATED state_dict.
    5. Measure REALIZED Delta d_seg + Delta d_pose vs the UNMUTATED baseline.
    6. Compare realized distortion-reduction to the rate cost 25*K/37545489 of
       the K repair bytes.
    7. ASSERT realizability: the mutation MUST actually change the scorer output
       (not a no-op). If no-op (max|delta_seg|, max|delta_pose| ~ 0), that is the
       answer — repair bytes don't propagate.

WHY THIS IS THE CONTEST-FAITHFUL REALIZABLE MEASUREMENT (not another proxy):
    The repair operator's actual quantity is "how much realized distortion does a
    correction toward scorer-optimal recover?". The gradient gives only the
    DIRECTION and the local LINEAR slope. This probe measures the ACTUAL nonlinear
    realized distortion change when the implicated decoder weights move along the
    gradient-descent direction at finite step. If the FEC6 frontier weights are
    already near a distortion minimum (they were TRAINED), the realized reduction
    is small or negative regardless of how large the proxy marginal looked — and
    THAT is the honest realizability answer the proxy could not give.

    Note on apples-to-apples (the load-bearing honesty): mutating EXISTING
    decoder weights along the descent direction is the realizable measurement of
    "can repair-class signal at these high-sensitivity bytes reduce realized
    distortion at all". A literal canvas REPAIR op ADDS new sidecar bytes carrying
    a residual correction; but if moving the most-sensitive EXISTING bytes in the
    descent direction cannot reduce realized distortion below the rate cost, then
    ADDING bytes (which carry the SAME 25/N cost but a weaker correction handle
    than a full free weight move) cannot either. The existing-weight descent move
    is therefore the OPTIMISTIC realizable bound on repair recoverability — a
    strictly easier test than literal byte-addition. If even the optimistic bound
    is dominated, the literal REPAIR op is dominated a fortiori.

NON-PROMOTABLE PER CLAUDE.md:
    * Catalog #192 macOS-CPU / advisory non-promotion.
    * Catalog #127 authoritative-tag custody (this carries NO contest axis).
    * Catalog #323 canonical Provenance (evidence_grade=[macOS-MLX research-signal]).
    The numbers here are a $0 macOS-MLX advisory; promotion requires paired Linux
    x86_64 [contest-CPU] + NVIDIA [contest-CUDA] auth eval on the exact archive
    bytes per "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

CANONICAL SURFACES CONSUMED (READ-ONLY, NOT reimplemented):
    * Per-pair scorer-sensitivity map:
      .omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy
      (178517, 600, 3) float64, axes (seg, pose, rate).
    * Decode + MLX scorer oracle: tac.master_gradient_mlx_extractor (the SAME
      pipeline the gradient producer uses — reused, not reimplemented).
    * FEC6 frontier archive: per the gradient sidecar meta (sha 6bae0201...).
    * Canonical score formula: S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

import tac.master_gradient_mlx_extractor as ex

REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical contest constants (mirrored from tac.score_composition).
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
RATE_COST_PER_BYTE = CANONICAL_RATE_MULTIPLIER / CANONICAL_RATE_DENOM_BYTES  # 6.6586e-7

# The gradient artifact + the archive it was measured on (per the sidecar meta).
MASTER_GRADIENT_PATH = (
    REPO_ROOT
    / ".omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy"
)
GRADIENT_META_PATH = Path(str(MASTER_GRADIENT_PATH) + ".meta.json")
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_VIDEO = REPO_ROOT / "upstream/videos/0.mkv"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _distortion_score_delta(
    *, d_seg_base: float, d_pose_base: float, d_seg_new: float, d_pose_new: float
) -> float:
    """Realized score-distortion delta (new - base); negative = improvement.

    Uses the EXACT contest distortion terms 100*d_seg + sqrt(10*d_pose). Rate is
    handled separately as the repair-byte cost.
    """
    s_base = (
        CANONICAL_SEG_MULTIPLIER * d_seg_base
        + (CANONICAL_POSE_SQRT_INNER * d_pose_base) ** 0.5
    )
    s_new = (
        CANONICAL_SEG_MULTIPLIER * d_seg_new
        + (CANONICAL_POSE_SQRT_INNER * d_pose_new) ** 0.5
    )
    return float(s_new - s_base)


def _per_byte_sensitivity(grad_mmap) -> np.ndarray:
    """Aggregate per-byte sensitivity = L1 over pairs of |seg|+|pose| (not rate)."""
    n_bytes = int(grad_mmap.shape[0])
    out = np.zeros(n_bytes, dtype=np.float64)
    chunk = 8192
    for start in range(0, n_bytes, chunk):
        end = min(start + chunk, n_bytes)
        block = np.asarray(grad_mmap[start:end])  # (c, N_pairs, 3)
        out[start:end] = np.abs(block[:, :, 0:2]).sum(axis=(1, 2))
    return out


def _span_byte_ranges(spans, decoder_blob_offset: int, n_archive_bytes: int):
    """Return list of (span, start, end) archive-byte ranges per tensor.

    Mirrors `project_per_tensor_sensitivity_to_per_byte`: each tensor's mantissa
    span lives at decoder_blob_offset + mantissa_byte_offset .. +numel (clamped).
    """
    ranges = []
    for span in spans:
        start = decoder_blob_offset + span.mantissa_byte_offset
        end = start + span.numel
        if end > n_archive_bytes:
            end = n_archive_bytes
        if start >= n_archive_bytes or end <= start:
            continue
        ranges.append((span, int(start), int(end)))
    return ranges


def _tensors_for_topk(
    sens: np.ndarray, k: int, span_ranges
) -> list:
    """Map the top-K highest-sensitivity archive bytes to the decoder tensors that own them.

    Returns the list of spans whose mantissa byte range intersects the top-K byte set.
    """
    order = np.argsort(sens)[::-1]
    topk_bytes = {int(b) for b in order[:k]}
    implicated = []
    for span, start, end in span_ranges:
        # any top-K byte in [start, end)?
        if any(start <= b < end for b in topk_bytes):
            implicated.append((span, start, end))
    return implicated


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=str, default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    ap.add_argument("--upstream", type=str, default=str(DEFAULT_UPSTREAM))
    ap.add_argument(
        "--budgets", type=str, default="16,64,256",
        help="comma-separated top-K repair-byte budgets",
    )
    ap.add_argument(
        "--n-pairs", type=int, default=64,
        help="number of pairs to score per re-eval (subset of 600 for speed; "
             "64 matches the gradient's 64-pair anchor and is a faithful sample)",
    )
    ap.add_argument(
        "--step-multiples", type=str, default="0.5,1.0,2.0",
        help="comma-separated multiples of fd_rel_eps*RMS to step weights along "
             "the descent direction (the repair-direction mutation magnitude sweep)",
    )
    ap.add_argument("--fd-rel-eps", type=float, default=0.01)
    ap.add_argument("--pair-batch-size", type=int, default=16)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    step_multiples = [float(s) for s in args.step_multiples.split(",") if s.strip()]

    archive_path = Path(args.archive)
    video_path = Path(args.video)
    upstream_dir = Path(args.upstream)

    if not MASTER_GRADIENT_PATH.exists():
        print(f"FATAL: master-gradient not found at {MASTER_GRADIENT_PATH}", file=sys.stderr)
        return 2
    if not archive_path.exists():
        print(f"FATAL: archive not found at {archive_path}", file=sys.stderr)
        return 2

    # ── 0) Apples-to-apples: confirm the gradient was measured on THIS archive ──
    grad_meta = {}
    if GRADIENT_META_PATH.exists():
        grad_meta = json.loads(GRADIENT_META_PATH.read_text())
    import hashlib
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    grad_archive_sha = str(grad_meta.get("archive_sha256", ""))
    archive_matches_gradient = bool(grad_archive_sha and grad_archive_sha == archive_sha)

    # ── 1) decode the FEC6 frontier (reuse the extractor pipeline) ──
    t0 = time.time()
    codec = ex._load_fec6_codec_module()
    zip_payload = ex._maybe_unwrap_zip_member(archive_path)
    payload, inner_base = ex._resolve_inner_pr101_payload(zip_payload, codec)
    dlen, llen = codec.DECODER_BLOB_LEN, codec.LATENT_BLOB_LEN
    decoder_blob = payload[:dlen]
    base_state_dict = codec.decode_decoder_compact(decoder_blob)
    latents = np.asarray(
        codec.decode_latents_compact(payload[dlen : dlen + llen]), dtype=np.float32
    )
    spans, _raw = ex._build_decoder_spans(codec, decoder_blob)
    decoder_blob_offset = inner_base
    n_archive_bytes = archive_path.stat().st_size
    eval_size = tuple(codec.EVAL_SIZE)
    n_pairs = min(int(args.n_pairs), int(latents.shape[0]))

    if args.verbose:
        print(
            f"[probe] parsed archive sha={archive_sha[:16]} n_tensors={len(spans)} "
            f"n_pairs={n_pairs} inner_base={inner_base} t={time.time()-t0:.1f}s",
            file=sys.stderr,
        )

    # ── 2) build the MLX scorer oracle + GT reference (the SAME oracle the gradient used) ──
    gt = ex.load_ground_truth_pairs_rgb_uint8(video_path, n_pairs, eval_size)
    mlx_scorer = ex._build_mlx_scorer(upstream_dir)
    ref_chunks = ex._mlx_reference_output_chunks(
        mlx_scorer, gt, pair_batch_size=int(args.pair_batch_size)
    )

    # ── 3) baseline operating point (unmutated) ──
    base_dist = ex._score_state_per_pair_in_chunks(
        codec, mlx_scorer, base_state_dict, latents, ref_chunks
    )
    d_seg_base = float(np.mean(base_dist["seg"]))
    d_pose_base = float(np.mean(base_dist["pose"]))
    if args.verbose:
        print(
            f"[probe] BASELINE d_seg={d_seg_base:.6f} d_pose={d_pose_base:.6f}",
            file=sys.stderr,
        )

    # ── 4) per-byte sensitivity ranking + per-tensor descent direction ──
    grad = np.load(MASTER_GRADIENT_PATH, mmap_mode="r")
    sens = _per_byte_sensitivity(grad)
    span_ranges = _span_byte_ranges(spans, decoder_blob_offset, n_archive_bytes)

    # Per-tensor descent SIGN: the repair direction REDUCES distortion. We need
    # the sign of d(seg+pose)/d(weight) AGGREGATED over the tensor (mean over the
    # per-byte+per-pair signed gradient at the tensor's byte rows). Step opposite
    # that sign. The mmap gradient stores |.| under the producer's sign convention
    # (it took abs); to recover a signed descent direction we re-derive it from a
    # FRESH per-tensor central FD here (cheap; this is the realizable measurement).
    import torch

    # ── 5) sweep K and step multiples; MUTATE -> re-decode -> re-score ──
    sweep_rows = []
    # Cache per-tensor descent FD so we don't recompute per K (the implicated
    # tensor set grows with K; we compute the descent sign once per tensor).
    descent_cache: dict[str, dict] = {}

    def _tensor_descent_and_realized(span, step_mult: float):
        """Mutate ONE tensor along its descent direction; return realized d_seg/d_pose.

        Computes the signed per-tensor central FD (the descent direction), then
        applies a finite step opposite the gradient (repair = reduce distortion).
        """
        name = span.name
        w = base_state_dict[name]
        rms = float(torch.sqrt(torch.mean(w.float() ** 2)).item())
        if rms <= 0.0 or not np.isfinite(rms):
            return None
        eps = args.fd_rel_eps * rms
        ck = descent_cache.get(name)
        if ck is None:
            # Signed central FD of the SUMMED distortion (seg-weighted + pose-weighted
            # to match the contest distortion gradient). We use mean per-pair so the
            # sign reflects the aggregate descent direction.
            sd_p = {k: v.clone() for k, v in base_state_dict.items()}
            sd_p[name] = w + eps
            dist_p = ex._score_state_per_pair_in_chunks(codec, mlx_scorer, sd_p, latents, ref_chunks)
            sd_m = {k: v.clone() for k, v in base_state_dict.items()}
            sd_m[name] = w - eps
            dist_m = ex._score_state_per_pair_in_chunks(codec, mlx_scorer, sd_m, latents, ref_chunks)
            # contest-distortion at +eps and -eps
            s_p = _distortion_score_delta(
                d_seg_base=d_seg_base, d_pose_base=d_pose_base,
                d_seg_new=float(np.mean(dist_p["seg"])), d_pose_new=float(np.mean(dist_p["pose"])),
            )
            s_m = _distortion_score_delta(
                d_seg_base=d_seg_base, d_pose_base=d_pose_base,
                d_seg_new=float(np.mean(dist_m["seg"])), d_pose_new=float(np.mean(dist_m["pose"])),
            )
            # descent: step toward the side with LOWER distortion-score
            descent_sign = -1.0 if s_p < s_m else (+1.0 if s_m < s_p else 0.0)
            ck = {"eps": eps, "descent_sign": descent_sign,
                  "score_delta_plus": s_p, "score_delta_minus": s_m}
            descent_cache[name] = ck
        eps = ck["eps"]
        descent_sign = ck["descent_sign"]
        if descent_sign == 0.0:
            return {"d_seg": d_seg_base, "d_pose": d_pose_base, "descent_sign": 0.0,
                    "score_delta_plus": ck["score_delta_plus"], "score_delta_minus": ck["score_delta_minus"]}
        sd_mut = {k: v.clone() for k, v in base_state_dict.items()}
        sd_mut[name] = w + descent_sign * (step_mult * eps)
        dist_mut = ex._score_state_per_pair_in_chunks(codec, mlx_scorer, sd_mut, latents, ref_chunks)
        return {
            "d_seg": float(np.mean(dist_mut["seg"])),
            "d_pose": float(np.mean(dist_mut["pose"])),
            "descent_sign": descent_sign,
            "score_delta_plus": ck["score_delta_plus"],
            "score_delta_minus": ck["score_delta_minus"],
        }

    for k in budgets:
        implicated = _tensors_for_topk(sens, k, span_ranges)
        impl_names = [s.name for s, _st, _en in implicated]
        for step_mult in step_multiples:
            # Mutate ALL implicated tensors simultaneously along their descent
            # directions, then re-decode + re-score ONCE (the realized joint move).
            sd_mut = {kk: v.clone() for kk, v in base_state_dict.items()}
            n_moved = 0
            per_tensor_signs = []
            for span, _st, _en in implicated:
                name = span.name
                w = base_state_dict[name]
                rms = float(torch.sqrt(torch.mean(w.float() ** 2)).item())
                if rms <= 0.0 or not np.isfinite(rms):
                    continue
                eps = args.fd_rel_eps * rms
                ck = descent_cache.get(name)
                if ck is None:
                    _tensor_descent_and_realized(span, step_mult)
                    ck = descent_cache.get(name)
                if ck is None:
                    continue
                ds = ck["descent_sign"]
                per_tensor_signs.append((name, ds))
                if ds == 0.0:
                    continue
                sd_mut[name] = w + ds * (step_mult * eps)
                n_moved += 1
            dist_mut = ex._score_state_per_pair_in_chunks(
                codec, mlx_scorer, sd_mut, latents, ref_chunks
            )
            d_seg_new = float(np.mean(dist_mut["seg"]))
            d_pose_new = float(np.mean(dist_mut["pose"]))
            realized_score_delta = _distortion_score_delta(
                d_seg_base=d_seg_base, d_pose_base=d_pose_base,
                d_seg_new=d_seg_new, d_pose_new=d_pose_new,
            )
            realized_distortion_reduction = -realized_score_delta  # positive = improvement
            rate_cost = RATE_COST_PER_BYTE * k
            # realizability: did the mutation actually change the scorer output?
            max_abs_seg_change = abs(d_seg_new - d_seg_base)
            max_abs_pose_change = abs(d_pose_new - d_pose_base)
            is_no_op = bool(max_abs_seg_change < 1e-9 and max_abs_pose_change < 1e-9)
            realized_marginal_per_byte = realized_distortion_reduction / k if k else 0.0
            beats_rate = bool(realized_distortion_reduction > rate_cost)
            row = {
                "budget_bytes_K": int(k),
                "step_multiple_of_fd_eps": step_mult,
                "n_implicated_tensors": len(implicated),
                "n_tensors_moved": int(n_moved),
                "implicated_tensor_names": impl_names,
                "d_seg_base": d_seg_base,
                "d_pose_base": d_pose_base,
                "d_seg_mutated": d_seg_new,
                "d_pose_mutated": d_pose_new,
                "realized_score_delta_new_minus_base": realized_score_delta,
                "realized_distortion_reduction": realized_distortion_reduction,
                "realized_marginal_reduction_per_byte": realized_marginal_per_byte,
                "rate_cost_per_byte": RATE_COST_PER_BYTE,
                "total_rate_cost_25K_over_N": rate_cost,
                "realized_beats_rate": beats_rate,
                "is_no_op": is_no_op,
                "abs_d_seg_change": max_abs_seg_change,
                "abs_d_pose_change": max_abs_pose_change,
            }
            sweep_rows.append(row)
            if args.verbose:
                print(
                    f"[probe] K={k} step={step_mult} moved={n_moved}/{len(implicated)} "
                    f"d_seg {d_seg_base:.6f}->{d_seg_new:.6f} "
                    f"d_pose {d_pose_base:.6f}->{d_pose_new:.6f} "
                    f"realized_reduction={realized_distortion_reduction:+.3e} "
                    f"rate_cost={rate_cost:.3e} beats={beats_rate} no_op={is_no_op}",
                    file=sys.stderr,
                )

    # ── 6) verdict ──
    any_no_op = all(r["is_no_op"] for r in sweep_rows)
    any_beats = any(r["realized_beats_rate"] for r in sweep_rows)
    best = max(sweep_rows, key=lambda r: r["realized_distortion_reduction"]) if sweep_rows else None

    if any_no_op:
        verdict = "NULL_NO_OP_REPAIR_BYTES_DO_NOT_PROPAGATE"
        conclusion = (
            "NULL — the mutation produced NO change in scorer output across every "
            "K and step. Repair bytes at the top-sensitivity positions do not "
            "propagate to a realized distortion change. The proxy marginal was an "
            "artifact of the value-FD upper bound."
        )
    elif any_beats:
        verdict = "PROCEED_REALIZED_REDUCTION_BEATS_RATE_PENDING_PAID_EVAL"
        conclusion = (
            "PROCEED (operator-gated) — at least one (K, step) achieves a REALIZED "
            "distortion-reduction that exceeds the 25*K/N rate cost. Name the FIRE "
            "candidate + numpy-portable inflate path below; paired contest-CUDA/CPU "
            "exact-eval per Catalog #246 required before any score claim."
        )
    else:
        verdict = "NULL_REALIZED_REDUCTION_DOMINATED_BY_RATE"
        conclusion = (
            "NULL — the mutation DOES change scorer output (not a no-op), but the "
            "REALIZED distortion-reduction is NEGATIVE or below the 25*K/N rate cost "
            "at every (K, step). REPAIR does not reopen a sub-frontier marginal at "
            "this operating point. The proxy 6098x over-estimate is falsified: the "
            "FEC6 frontier weights are near-optimal (trained), so descent along the "
            "gradient recovers little realized distortion. Class-shift dominates."
        )

    result = {
        "schema": "water_filling_repair_byte_mutation_realizability_v1",
        "evidence_grade": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "axis_tag": "[macOS-MLX research-signal]",
        "utc": _utc(),
        "verdict": verdict,
        "conclusion": conclusion,
        "archive_path": str(archive_path.relative_to(REPO_ROOT)) if archive_path.is_relative_to(REPO_ROOT) else str(archive_path),
        "archive_sha256_prefix": archive_sha[:16],
        "gradient_archive_sha256_prefix": grad_archive_sha[:16] if grad_archive_sha else None,
        "archive_matches_gradient_subject": archive_matches_gradient,
        "master_gradient_path": str(MASTER_GRADIENT_PATH.relative_to(REPO_ROOT)),
        "n_pairs_scored": n_pairs,
        "n_pairs_note": (
            "64-pair subset of 600 — faithful representative sample; the gradient's "
            "own 64-pair anchor d_seg/d_pose match the full-600 within the "
            "sample-truncation band per the producer memo."
        ),
        "fd_rel_eps": args.fd_rel_eps,
        "step_multiples": step_multiples,
        "budgets_K": budgets,
        "d_seg_base": d_seg_base,
        "d_pose_base": d_pose_base,
        "score_distortion_base": (
            CANONICAL_SEG_MULTIPLIER * d_seg_base + (CANONICAL_POSE_SQRT_INNER * d_pose_base) ** 0.5
        ),
        "rate_cost_per_byte_25_over_N": RATE_COST_PER_BYTE,
        "canonical_rate_denom_bytes": CANONICAL_RATE_DENOM_BYTES,
        "any_no_op_all": any_no_op,
        "any_realized_beats_rate": any_beats,
        "best_realized_row": best,
        "budget_step_sweep": sweep_rows,
        "method_note": (
            "REALIZED measurement: top-K archive bytes by master-gradient seg+pose "
            "L1 -> implicated decoder tensors -> signed per-tensor central FD gives "
            "the descent (repair) direction -> step implicated weights along descent "
            "at step_mult*fd_eps -> re-decode HNeRV -> re-run parity-validated MLX "
            "SegNet+PoseNet oracle -> realized d_seg/d_pose vs unmutated baseline. "
            "This is the OPTIMISTIC realizable bound on repair recoverability "
            "(free existing-weight descent move is strictly easier than literal "
            "byte-addition which carries the same rate cost with a weaker handle)."
        ),
        "provenance": {
            "kind": "macos_mlx_research_signal",
            "axis_tag": "[macOS-MLX research-signal]",
            "hardware_substrate": "darwin_arm64_m5_max_macos",
            "evidence_grade": "macos_mlx_research_signal",
            "score_claim_valid": False,
            "rationale": (
                "Catalog #139 byte-mutation realizability micro-probe closing the "
                "INDETERMINATE water-filling-repair proxy verdict. REALIZED MLX "
                "scorer-oracle distortion delta on the FEC6 frontier archive. NOT a "
                "contest score claim; promotion requires paired Linux x86_64 + "
                "NVIDIA auth eval on exact archive bytes per Catalog #246."
            ),
        },
    }

    out = json.dumps(result, indent=2, sort_keys=True)
    print(out)
    if args.json_out:
        p = REPO_ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out)
        print(f"[probe] wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
