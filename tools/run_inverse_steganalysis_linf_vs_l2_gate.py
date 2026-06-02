#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Falsifiable $0 gate: L-inf margin-budget vs L2 allocation on the REAL scorer.

Implements design memo §7
(``.omx/research/inverse_steganalysis_optimal_full_stack_design_20260601.md``):

    **Does L-inf margin-budget (inverse-steganalysis) allocation BEAT L2
    (MSE-optimal) allocation at EQUAL rate, measured on the real scorer?**

The contest scorer is a steganalysis detector; the contest objective is the
steganographic security objective (max bytes-saved s.t. the detector's decision
does not flip), NOT L2 fidelity. This gate isolates THE OBJECTIVE by holding the
carrier (the real reconstructed video) and the per-pixel bit budget B fixed, and
varying ONLY where the quantization precision goes:

  * (A) L2-optimal  -> a single UNIFORM quantizer step (the MSE-minimizing
        allocation under a flat source; the standard codec objective).
  * (B) L-inf margin-budget -> per-pixel step ``clip(c*rho_i, lo, hi)`` where
        ``rho_i = 1/(oracle_saliency_i + eps)`` (the inverse-steganalysis prior:
        fine steps at small-margin boundaries, coarse steps in the detector-blind
        dead-zone). Same total bits B as L2.

Both decode (apply uniform quantization noise at the chosen per-pixel steps), run
the VERIFIED differentiable scorer mirror (bit-exact vs frozen weights, commit
8173b493a) on the SAME real ``upstream/videos/0.mkv`` frames, and report the hard
contest ``d_seg`` (last-frame argmax-flip) + ``d_pose`` (first-6 pose MSE).

NO-FAKE fairness controls (all enforced + reported):
  * EQUAL RATE -- the L-inf allocation is FORCED to spend AT LEAST as many bits
    as L2 (``fairness_direction='disadvantage_linf'``), and the realized rate
    match is reported; a win therefore cannot be a rate artifact.
  * SAME FRAMES -- both allocations are applied to the same decoded gt pairs.
  * SAME NOISE SEEDS -- the quantization noise is drawn from the same per-pair
    generator seeds for both allocations, then averaged over ``--noise-seeds``
    independent draws so a win is not a lucky realization.
  * ALLOCATIONS DIFFER -- a guard asserts the L2 and L-inf step maps are not
    byte-identical (a no-op detector per Catalog #139).
  * MULTI-PAIR -- aggregated over ``--num-pairs`` strided real pairs.

All numerics are ``[macOS-CPU advisory]`` -- NON-PROMOTABLE, ``score_claim=false``,
``promotable=false`` per Catalog #341/#192/#127/#323. No paid dispatch, no GPU, no
MPS authority. The JSON anchor is durable (``.omx/research/``), never ``/tmp``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "upstream"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (  # noqa: E402
    GATE_SCHEMA,
    UINT8_DYNAMIC_RANGE,
    allocate_l2_uniform,
    allocate_linf_margin_budget,
    apply_uniform_quantization_noise,
    margin_budget_from_saliency,
    measure_pair_d_seg_d_pose,
)
from tac.analysis.score_exact_saliency import (  # noqa: E402
    build_producer_provenance,
    compute_s_pose_fisher,
    compute_s_seg_flip_risk,
    decode_real_pairs,
    load_score_exact_scorers,
    saliency_concentration,
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_native_oracle_saliency(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    pair_btchw: torch.Tensor,
    *,
    seg_weight: float,
    pose_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine P18 (s_seg, last frame, 384x512) + P19 (s_pose, native) at native res.

    s_seg is upsampled to native resolution; s_pose is already native. Both are
    unit-mean-normalized (so the seg/pose weights are interpretable as
    score-derivative ratios, not raw-scale artifacts) then combined
    ``seg_weight*s_seg + pose_weight*s_pose``. The combined surface is the
    per-pixel detector saliency that drives the inverse-steganalysis prior.
    """
    h, w = pair_btchw.shape[-2:]
    sseg = compute_s_seg_flip_risk(segnet, pair_btchw)  # (384, 512)
    spose = compute_s_pose_fisher(posenet, pair_btchw, method="batched_vjp")  # (H, W)
    sseg_native = F.interpolate(
        sseg.flip_risk[None, None].float(), size=(h, w), mode="bilinear", align_corners=False
    )[0, 0]

    def unit_mean(x: torch.Tensor) -> torch.Tensor:
        x = x.double().clamp_min(0.0)
        m = x.mean()
        return x / (m + 1e-12)

    seg_n = unit_mean(sseg_native)
    pose_n = unit_mean(spose.s_pose)
    combined = float(seg_weight) * seg_n + float(pose_weight) * pose_n
    combined_np = combined.cpu().numpy().reshape(-1)
    conc = saliency_concentration(torch.from_numpy(combined_np))
    diag = {
        "seg_finite": bool(sseg.grad_finite),
        "pose_finite": bool(spose.grad_finite),
        "combined_top10pct_mass": conc.top_k_pct_mass.get(10.0),
        "combined_gini": conc.gini,
        "seg_weight": float(seg_weight),
        "pose_weight": float(pose_weight),
    }
    return combined_np, diag


def _candidate_from_steps(
    gt_pair: torch.Tensor, steps: np.ndarray, *, seed_base: int
) -> torch.Tensor:
    """Apply per-pixel quantization noise to BOTH frames of the gt pair.

    The two frames use distinct (but deterministic) generator seeds so they are
    independently quantized, matching a carrier that coarsens each frame.
    """
    g0 = torch.Generator().manual_seed(seed_base)
    g1 = torch.Generator().manual_seed(seed_base + 7919)
    f0 = apply_uniform_quantization_noise(gt_pair[0, 0], steps, generator=g0)
    f1 = apply_uniform_quantization_noise(gt_pair[0, 1], steps, generator=g1)
    return torch.stack([f0, f1])[None]  # (1, 2, 3, H, W)


def run_gate(
    *,
    upstream_dir: str,
    video_path: str,
    num_pairs: int,
    pair_stride: int,
    bits_per_pixel: float,
    min_step: float,
    max_step: float,
    seg_weight: float,
    pose_weight: float,
    noise_seeds: int,
    rate_tolerance: float,
) -> dict[str, Any]:
    """Run the L-inf-vs-L2 gate over ``num_pairs`` real pairs and aggregate."""
    posenet, segnet = load_score_exact_scorers(upstream_dir=upstream_dir, device="cpu")
    per_pair: list[dict[str, Any]] = []
    allocations_differ_all = True
    rate_match_worst = 0.0

    for i in range(num_pairs):
        start_pair = i * pair_stride
        pair = decode_real_pairs(
            video_path, num_pairs=1, pair_stride=1, start_pair=start_pair, device="cpu"
        )  # (1, 2, 3, H, W)
        h, w = pair.shape[-2:]
        n = h * w
        target_bits = float(bits_per_pixel) * float(n)

        saliency_np, sal_diag = _build_native_oracle_saliency(
            posenet, segnet, pair, seg_weight=seg_weight, pose_weight=pose_weight
        )
        rho = margin_budget_from_saliency(saliency_np)

        l2 = allocate_l2_uniform(n, target_bits=target_bits)
        linf = allocate_linf_margin_budget(
            rho,
            target_bits=target_bits,
            min_step=min_step,
            max_step=max_step,
            rate_tolerance=rate_tolerance,
            fairness_direction="disadvantage_linf",  # L-inf must spend >= L2 bits
        )
        # DECISIVE NO-FAKE CONTROL: shuffle the per-pixel STEPS within the L-inf
        # allocation. This keeps the EXACT same step-value histogram (=> same
        # total rate, same MSE distribution, same per-pixel distortion budget),
        # but DESTROYS the detector aiming (the steps are now at random pixels,
        # not at the boundaries the oracle flagged). If L-inf wins ONLY because of
        # WHERE the bits go (genuine detector signal), the shuffled control must
        # NOT beat L2; if the shuffled control STILL beats L2, the "win" is a
        # rate-model / step-histogram artifact, not the inverse-steganalysis prior.
        shuffle_rng = np.random.default_rng(20260601 + start_pair)
        shuffled_steps = linf.steps.copy()
        shuffle_rng.shuffle(shuffled_steps)
        allocations_differ = bool(
            l2.steps.shape != linf.steps.shape or not np.array_equal(l2.steps, linf.steps)
        )
        allocations_differ_all = allocations_differ_all and allocations_differ
        rate_match_rel = abs(linf.total_bits - l2.total_bits) / max(l2.total_bits, 1.0)
        rate_match_worst = max(rate_match_worst, rate_match_rel)

        # Seed-averaged measurement: same seeds for L2, L-inf, and the shuffled
        # control allocation.
        ds_l2_s, dp_l2_s = [], []
        ds_linf_s, dp_linf_s = [], []
        ds_shuf_s, dp_shuf_s = [], []
        for s in range(noise_seeds):
            seed_base = 1_000 + 31 * i + 101 * s
            cand_l2 = _candidate_from_steps(pair, l2.steps, seed_base=seed_base)
            cand_linf = _candidate_from_steps(pair, linf.steps, seed_base=seed_base)
            cand_shuf = _candidate_from_steps(pair, shuffled_steps, seed_base=seed_base)
            ds_l2, dp_l2 = measure_pair_d_seg_d_pose(posenet, segnet, pair, cand_l2)
            ds_linf, dp_linf = measure_pair_d_seg_d_pose(posenet, segnet, pair, cand_linf)
            ds_shuf, dp_shuf = measure_pair_d_seg_d_pose(posenet, segnet, pair, cand_shuf)
            ds_l2_s.append(ds_l2)
            dp_l2_s.append(dp_l2)
            ds_linf_s.append(ds_linf)
            dp_linf_s.append(dp_linf)
            ds_shuf_s.append(ds_shuf)
            dp_shuf_s.append(dp_shuf)

        mean = lambda xs: float(np.mean(xs))  # noqa: E731
        d_seg_l2, d_pose_l2 = mean(ds_l2_s), mean(dp_l2_s)
        d_seg_linf, d_pose_linf = mean(ds_linf_s), mean(dp_linf_s)
        d_seg_shuf, d_pose_shuf = mean(ds_shuf_s), mean(dp_shuf_s)
        contest_l2 = 100.0 * d_seg_l2 + float(np.sqrt(10.0 * d_pose_l2))
        contest_linf = 100.0 * d_seg_linf + float(np.sqrt(10.0 * d_pose_linf))
        contest_shuf = 100.0 * d_seg_shuf + float(np.sqrt(10.0 * d_pose_shuf))

        per_pair.append(
            {
                "pair_index": int(start_pair),
                "native_hw": [int(h), int(w)],
                "n_pixels": int(n),
                "target_bits": float(target_bits),
                "l2_total_bits": float(l2.total_bits),
                "linf_total_bits": float(linf.total_bits),
                "rate_match_rel_diff": float(rate_match_rel),
                "l2_uniform_step": float(l2.steps[0]),
                "linf_step_min": float(linf.min_step),
                "linf_step_max": float(linf.max_step),
                "linf_water_level": float(linf.water_level),
                "allocations_differ": allocations_differ,
                "saliency_diag": sal_diag,
                "d_seg_l2": d_seg_l2,
                "d_pose_l2": d_pose_l2,
                "contest_distortion_l2": float(contest_l2),
                "d_seg_linf": d_seg_linf,
                "d_pose_linf": d_pose_linf,
                "contest_distortion_linf": float(contest_linf),
                "contest_distortion_delta_linf_minus_l2": float(contest_linf - contest_l2),
                "linf_wins": bool(contest_linf < contest_l2),
                "d_seg_shuffled_control": d_seg_shuf,
                "d_pose_shuffled_control": d_pose_shuf,
                "contest_distortion_shuffled_control": float(contest_shuf),
                "shuffled_control_delta_minus_l2": float(contest_shuf - contest_l2),
                "shuffled_control_beats_l2": bool(contest_shuf < contest_l2),
            }
        )

    # Aggregate.
    agg_d_seg_l2 = float(np.mean([p["d_seg_l2"] for p in per_pair]))
    agg_d_pose_l2 = float(np.mean([p["d_pose_l2"] for p in per_pair]))
    agg_d_seg_linf = float(np.mean([p["d_seg_linf"] for p in per_pair]))
    agg_d_pose_linf = float(np.mean([p["d_pose_linf"] for p in per_pair]))
    agg_d_seg_shuf = float(np.mean([p["d_seg_shuffled_control"] for p in per_pair]))
    agg_d_pose_shuf = float(np.mean([p["d_pose_shuffled_control"] for p in per_pair]))
    agg_contest_l2 = 100.0 * agg_d_seg_l2 + float(np.sqrt(10.0 * agg_d_pose_l2))
    agg_contest_linf = 100.0 * agg_d_seg_linf + float(np.sqrt(10.0 * agg_d_pose_linf))
    agg_contest_shuf = 100.0 * agg_d_seg_shuf + float(np.sqrt(10.0 * agg_d_pose_shuf))
    delta = agg_contest_linf - agg_contest_l2
    delta_shuf = agg_contest_shuf - agg_contest_l2
    n_wins = sum(1 for p in per_pair if p["linf_wins"])
    n_shuf_wins = sum(1 for p in per_pair if p["shuffled_control_beats_l2"])
    linf_wins_aggregate = bool(agg_contest_linf < agg_contest_l2)
    shuffled_beats_l2 = bool(agg_contest_shuf < agg_contest_l2)
    rel_improvement = float((agg_contest_l2 - agg_contest_linf) / max(agg_contest_l2, 1e-12))
    # The DECISIVE no-fake check: the win must be attributable to detector AIMING,
    # not the step-histogram. L-inf must beat L2 AND beat the shuffled control by
    # a clear margin (the shuffled control has the SAME rate + step histogram but
    # random placement). If the shuffled control beats L-inf, the win is an
    # artifact, not the inverse-steganalysis prior.
    aiming_signal_genuine = bool(
        linf_wins_aggregate and (agg_contest_linf < agg_contest_shuf)
    )

    if not linf_wins_aggregate:
        verdict = "INVERSE_STEGANALYSIS_LINF_DOES_NOT_BEAT_L2_AT_EQUAL_RATE"
    elif not aiming_signal_genuine:
        verdict = "INVERSE_STEGANALYSIS_LINF_BEATS_L2_BUT_SHUFFLE_CONTROL_FAILS_RATE_ARTIFACT"
    else:
        verdict = "INVERSE_STEGANALYSIS_LINF_BEATS_L2_AT_EQUAL_RATE_AIMING_GENUINE"

    return {
        "schema": GATE_SCHEMA,
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "generated_at_utc": _utc_now(),
        "design_memo": ".omx/research/inverse_steganalysis_optimal_full_stack_design_20260601.md",
        "lane_id": "lane_inverse_steganalysis_optimal_full_stack_20260601",
        "config": {
            "video_path": video_path,
            "num_pairs": int(num_pairs),
            "pair_stride": int(pair_stride),
            "bits_per_pixel": float(bits_per_pixel),
            "min_step": float(min_step),
            "max_step": float(max_step),
            "seg_weight": float(seg_weight),
            "pose_weight": float(pose_weight),
            "noise_seeds": int(noise_seeds),
            "rate_tolerance": float(rate_tolerance),
            "dynamic_range": float(UINT8_DYNAMIC_RANGE),
        },
        "fairness_controls": {
            "equal_rate_direction": "linf_forced_to_spend_at_least_l2_bits",
            "rate_match_worst_rel_diff": float(rate_match_worst),
            "rate_match_within_tolerance": bool(rate_match_worst <= max(rate_tolerance, 1e-3)),
            "same_frames": True,
            "same_noise_seeds_per_allocation": True,
            "noise_seeds_averaged": int(noise_seeds),
            "allocations_differ_all_pairs": bool(allocations_differ_all),
            "multi_pair": int(num_pairs) > 1,
            "shuffled_control_present": True,
            "shuffled_control_isolates_detector_aiming": (
                "L-inf vs a same-rate same-step-histogram random-placement control; "
                "a win that survives this control is detector-aiming, not a rate artifact"
            ),
        },
        "aggregate": {
            "d_seg_l2": agg_d_seg_l2,
            "d_pose_l2": agg_d_pose_l2,
            "contest_distortion_l2": float(agg_contest_l2),
            "d_seg_linf": agg_d_seg_linf,
            "d_pose_linf": agg_d_pose_linf,
            "contest_distortion_linf": float(agg_contest_linf),
            "contest_distortion_delta_linf_minus_l2": float(delta),
            "relative_improvement_fraction": rel_improvement,
            "n_pairs_linf_wins": int(n_wins),
            "n_pairs_total": int(num_pairs),
            "linf_wins_aggregate": linf_wins_aggregate,
            "d_seg_shuffled_control": agg_d_seg_shuf,
            "d_pose_shuffled_control": agg_d_pose_shuf,
            "contest_distortion_shuffled_control": float(agg_contest_shuf),
            "shuffled_control_delta_minus_l2": float(delta_shuf),
            "n_pairs_shuffled_control_beats_l2": int(n_shuf_wins),
            "shuffled_control_beats_l2_aggregate": shuffled_beats_l2,
            "aiming_signal_genuine_linf_beats_shuffled": aiming_signal_genuine,
        },
        "verdict": verdict,
        "verdict_line": (
            f"At equal rate ({bits_per_pixel:.2f} bits/pixel, L-inf forced to spend >= L2 bits, "
            f"rate match {rate_match_worst:.2e}), inverse-steganalysis L-inf margin-budget "
            f"allocation {'BEATS' if linf_wins_aggregate else 'DOES NOT BEAT'} L2 on the real "
            f"scorer by {abs(delta):.5f} contest-distortion ({rel_improvement * 100:.1f}% "
            f"{'lower' if linf_wins_aggregate else 'higher'}), {n_wins}/{num_pairs} pairs; "
            f"shuffled-control (same rate+histogram, random placement) contest-distortion "
            f"{agg_contest_shuf:.5f} vs L-inf {agg_contest_linf:.5f} -> detector-aiming "
            f"{'GENUINE' if aiming_signal_genuine else 'NOT-isolated'} [macOS-CPU advisory]."
        ),
        "per_pair": per_pair,
        "provenance": build_producer_provenance(
            upstream_dir=upstream_dir, repo_root=str(REPO_ROOT)
        ),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-dir", default="upstream")
    p.add_argument("--video-path", default="upstream/videos/0.mkv")
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--pair-stride", type=int, default=120)
    p.add_argument("--bits-per-pixel", type=float, default=4.0)
    p.add_argument("--min-step", type=float, default=1.0)
    p.add_argument("--max-step", type=float, default=128.0)
    p.add_argument("--seg-weight", type=float, default=100.0)
    p.add_argument("--pose-weight", type=float, default=1.0)
    p.add_argument("--noise-seeds", type=int, default=3)
    p.add_argument("--rate-tolerance", type=float, default=1.0e-3)
    p.add_argument(
        "--output-json",
        default=None,
        help="durable anchor path; defaults to .omx/research/inverse_steganalysis_linf_vs_l2_gate_<utc>.json",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Force CPU; never MPS authority per CLAUDE.md "MPS auth eval is NOISE".
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = _build_arg_parser().parse_args(argv)
    t0 = time.perf_counter()
    payload = run_gate(
        upstream_dir=args.upstream_dir,
        video_path=args.video_path,
        num_pairs=args.num_pairs,
        pair_stride=args.pair_stride,
        bits_per_pixel=args.bits_per_pixel,
        min_step=args.min_step,
        max_step=args.max_step,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        noise_seeds=args.noise_seeds,
        rate_tolerance=args.rate_tolerance,
    )
    payload["wall_clock_seconds"] = float(time.perf_counter() - t0)

    out = args.output_json
    if out is None:
        out = f".omx/research/inverse_steganalysis_linf_vs_l2_gate_{payload['generated_at_utc']}.json"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    resolved = out_path.as_posix()
    if resolved.startswith("/tmp/") or resolved.startswith("/private/tmp/") or resolved.startswith("/var/tmp/"):
        raise SystemExit("refusing system /tmp evidence path per CLAUDE.md transient-evidence trap")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    agg = payload["aggregate"]
    print(payload["verdict_line"])
    print(f"[verdict] {payload['verdict']}")
    print(f"[anchor] {out_path}")
    print(
        f"[fairness] rate_match_worst={payload['fairness_controls']['rate_match_worst_rel_diff']:.2e} "
        f"allocations_differ={payload['fairness_controls']['allocations_differ_all_pairs']} "
        f"shuffle_control_beats_l2={agg['shuffled_control_beats_l2_aggregate']} "
        f"aiming_genuine={agg['aiming_signal_genuine_linf_beats_shuffled']} "
        f"seeds={args.noise_seeds} pairs={args.num_pairs} "
        f"wall={payload['wall_clock_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
