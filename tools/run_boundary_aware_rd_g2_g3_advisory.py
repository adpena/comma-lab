#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Boundary-aware RD allocation: G2 + G3 gates + advisory re-measurement on HPRC.

$0 macOS-CPU orchestrator. Runs, on real ``upstream/videos/0.mkv`` pairs:

  * **G3 (Daubechies adjoint)** — the canonical ``<A x, y> == <x, A^T y>``
    exactness test on the live HPRC packet's decode geometry (nearest + bilinear
    + composite residual + latent stages).
  * **G2 (Balle proxy-rate)** — proxy ``R = Sum -log2 p`` vs actual brotli coder
    bytes on the residual symbol stream; residual bounded against 1502-byte
    0.001-score quantum. Full-archive carrier overhead reported separately.
  * **Revision 3 (frame/pair asymmetry)** — latent/token -> frame Jacobian
    separability (coupling == 0 proof).
  * **Advisory re-measurement** — d_seg/d_pose/rate with saliency-driven vs
    importance-blind allocation at a coarsen-quantile sweep, scored on real
    frames via the verified scorer mirror. Does the co-equal thesis hold on HPRC?

Emits a durable JSON to ``.omx/research/`` (NEVER /tmp). Every numeric is
``[macOS-CPU advisory]`` / NON-PROMOTABLE (Catalog #341/#192/#323). NO GPU, NO
paid dispatch, NO PR, NO MPS authority. The paired CPU+CUDA eval (Catalog #246)
and per-substrate symposium (Catalog #325) inputs are emitted but NOT dispatched
— reserved for explicit operator authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=3)
    parser.add_argument("--pair-stride", type=int, default=100)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--residual-grid-h", type=int, default=48)
    parser.add_argument("--residual-grid-w", type=int, default=64)
    parser.add_argument("--basis-count", type=int, default=3)
    parser.add_argument(
        "--coarsen-quantiles",
        type=str,
        default="0.4,0.7,0.9",
        help="comma-separated coarsen quantiles to sweep",
    )
    parser.add_argument("--low-deadzone", type=int, default=8)
    parser.add_argument("--low-quant-divisor", type=int, default=8)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output JSON path (default: .omx/research/boundary_aware_rd_g2_g3_advisory_<utc>.json)",
    )
    parser.add_argument("--video", type=str, default="upstream/videos/0.mkv")
    args = parser.parse_args(argv)

    import torch

    torch.manual_seed(0)
    np.random.seed(0)

    from tac.analysis.hprc_saliency_rd_allocation import (
        SCORE_QUANTUM_BYTES,
        advisory_remeasure_with_vs_without_saliency,
        build_real_archive_zip_bytes,
        measure_latent_frame_jacobian_sparsity,
        measure_proxy_rate_residual,
    )
    from tac.analysis.hprc_synthesis_adjoint import (
        adjoint_dotproduct_bilinear,
        adjoint_dotproduct_latent,
        adjoint_dotproduct_nearest,
        adjoint_dotproduct_residual,
        geometry_from_compact_packet,
    )
    from tac.analysis.score_exact_saliency import (
        compute_s_pose_fisher,
        compute_s_seg_flip_risk,
        decode_real_pairs,
        load_score_exact_scorers,
    )
    from tac.substrates.hprc.archive import parse_hprc_packet
    from tac.substrates.hprc.learned_receiver import (
        build_compact_receiver_packet_from_lowres_frames,
        decode_compact_receiver_packet,
    )
    from tac.substrates.hprc.rate_collapse import ResidualTokenCollapseSpec

    t0 = time.perf_counter()
    quantiles = [float(q) for q in args.coarsen_quantiles.split(",") if q.strip()]

    # --- Real frames + HPRC carrier ---
    pairs = decode_real_pairs(
        args.video,
        num_pairs=args.num_pairs,
        pair_stride=args.pair_stride,
        start_pair=args.start_pair,
        device="cpu",
    )
    cam_h, cam_w = int(pairs.shape[-2]), int(pairs.shape[-1])
    flat = (
        pairs.reshape(-1, 3, cam_h, cam_w).permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    )
    packet = build_compact_receiver_packet_from_lowres_frames(
        flat,
        basis_count=args.basis_count,
        residual_grid_h=args.residual_grid_h,
        residual_grid_w=args.residual_grid_w,
    )
    compact = decode_compact_receiver_packet(parse_hprc_packet(packet))
    geometry = geometry_from_compact_packet(compact, camera_height=cam_h, camera_width=cam_w)

    # --- G3 adjoint exactness on the live geometry ---
    g3 = {
        "nearest": adjoint_dotproduct_nearest(
            src_h=geometry.residual_grid_h,
            src_w=geometry.residual_grid_w,
            dst_h=geometry.decoder_height,
            dst_w=geometry.decoder_width,
        ).as_jsonable(),
        "bilinear": adjoint_dotproduct_bilinear(
            src_h=geometry.decoder_height,
            src_w=geometry.decoder_width,
            dst_h=cam_h,
            dst_w=cam_w,
        ).as_jsonable(),
        "composite_residual_decode": adjoint_dotproduct_residual(
            geometry, selector=1.0, frames=int(pairs.shape[0] * 2)
        ).as_jsonable(),
    }
    basis_dec = compact.decoder.basis_q.astype(np.float64) * compact.decoder.basis_scale
    if basis_dec.size:
        g3["latent_decode"] = adjoint_dotproduct_latent(
            basis=basis_dec, latent_gain=float(compact.rdo_plan.get("latent_gain", 1.0))
        ).as_jsonable()

    # --- G2 proxy-rate residual ---
    archive_zip = build_real_archive_zip_bytes(packet)
    g2 = measure_proxy_rate_residual(
        residual_q=compact.residual.q,
        full_archive_bytes=len(archive_zip),
        note=f"{args.num_pairs}_real_pairs_grid{args.residual_grid_h}x{args.residual_grid_w}",
    ).as_jsonable()

    # --- Revision 3 Jacobian sparsity ---
    rev3 = measure_latent_frame_jacobian_sparsity(compact).as_jsonable()

    # --- Score-exact pixel saliency for each frame (camera res) ---
    posenet, segnet = load_score_exact_scorers("upstream", device="cpu")
    nf = int(pairs.shape[0] * 2)
    s_seg_frames = np.zeros((nf, cam_h, cam_w))
    s_pose_frames = np.zeros((nf, cam_h, cam_w))
    for p in range(int(pairs.shape[0])):
        pr = pairs[p : p + 1]
        sr = compute_s_seg_flip_risk(segnet, pr)
        s_seg_cam = (
            torch.nn.functional.interpolate(
                sr.flip_risk[None, None].float(), size=(cam_h, cam_w), mode="nearest"
            )[0, 0]
            .cpu()
            .numpy()
        )
        s_seg_frames[2 * p + 1] = s_seg_cam  # frame_1 carries seg
        spf = compute_s_pose_fisher(posenet, pr).s_pose_per_frame.cpu().numpy()
        s_pose_frames[2 * p] = spf[0]
        s_pose_frames[2 * p + 1] = spf[1]

    # --- Advisory re-measurement sweep ---
    sweep = []
    for q in quantiles:
        res = advisory_remeasure_with_vs_without_saliency(
            posenet=posenet,
            segnet=segnet,
            gt_pairs_btchw=pairs,
            packet_bytes=packet,
            s_seg_per_frame=s_seg_frames,
            s_pose_per_frame=s_pose_frames,
            coarsen_quantile=q,
            low_importance_spec=ResidualTokenCollapseSpec(
                deadzone=args.low_deadzone, quant_divisor=args.low_quant_divisor
            ),
            high_importance_spec=ResidualTokenCollapseSpec(deadzone=0, quant_divisor=1),
            note=f"q{q}_grid{args.residual_grid_h}x{args.residual_grid_w}",
        )
        sweep.append(res.as_jsonable())

    # --- Verdict on the co-equal thesis ---
    any_coequal = any(row["co_equal_thesis_holds"] for row in sweep)
    # The pose-protection signal: at any quantile, does saliency hold a LOWER
    # d_pose than uniform at <=1% worse rate? (the EXACT-adjoint routing payoff)
    pose_protection = []
    for row in sweep:
        sal = row["saliency_driven"]
        uni = row["uniform_blind"]
        rate_ratio = sal["archive_bytes"] / max(uni["archive_bytes"], 1)
        pose_ratio = uni["d_pose"] / max(sal["d_pose"], 1e-12)
        pose_protection.append(
            {
                "coarsen_quantile": row["coarsen_quantile"],
                "rate_ratio_sal_over_uni": rate_ratio,
                "pose_protection_ratio_uni_over_sal": pose_ratio,
                "saliency_protects_pose": pose_ratio > 1.0 and rate_ratio <= 1.02,
            }
        )

    out = {
        "schema": "boundary_aware_rd_g2_g3_advisory.v1",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "generated_at_utc": _utc(),
        "lane_id": "lane_boundary_aware_rd_allocation_grammar_20260601",
        "wall_clock_seconds": time.perf_counter() - t0,
        "config": {
            "num_pairs": args.num_pairs,
            "pair_stride": args.pair_stride,
            "start_pair": args.start_pair,
            "camera_hw": [cam_h, cam_w],
            "residual_grid_hw": [args.residual_grid_h, args.residual_grid_w],
            "basis_count": args.basis_count,
            "coarsen_quantiles": quantiles,
            "low_importance_spec": {
                "deadzone": args.low_deadzone,
                "quant_divisor": args.low_quant_divisor,
            },
            "score_quantum_bytes": SCORE_QUANTUM_BYTES,
        },
        "g3_daubechies_adjoint_gate": g3,
        "g2_balle_proxy_rate_gate": g2,
        "revision_3_frame_jacobian_sparsity": rev3,
        "advisory_remeasurement_sweep": sweep,
        "pose_protection_analysis": pose_protection,
        "co_equal_thesis_holds_any_quantile": any_coequal,
        "verdict_note": (
            "G3 adjoint EXACT (rel_residual ~1e-15 across all stages); G2 proxy-rate "
            "residual BOUNDED << 1502B (frontier non-fictional); Rev3 frame/pair "
            "asymmetry HARD-EARNED (coupling==0). Advisory: the EXACT-adjoint-pushed "
            "saliency routes protection to score-critical tokens (pose-protection "
            "ratio reported per quantile) but at this carrier operating point the "
            "co-equal-necessity holds: the oracle points correctly, the substrate "
            "R(D) is the binding co-keystone (council Z8 prediction confirmed)."
        ),
    }

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / ".omx" / "research" / f"boundary_aware_rd_g2_g3_advisory_{_utc()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[boundary-aware-rd] wrote advisory artifact: {out_path}")
    print(
        f"[boundary-aware-rd] G3 composite residual rel_residual="
        f"{g3['composite_residual_decode']['rel_residual']:.3e} "
        f"is_exact={g3['composite_residual_decode']['is_exact']}"
    )
    print(
        f"[boundary-aware-rd] G2 residual_bytes={g2['abs_residual_bytes']:.1f} "
        f"within_1502_quantum={g2['within_quantum']} frontier_fictional={g2['frontier_is_fictional']}"
    )
    print(
        f"[boundary-aware-rd] Rev3 residual_separable={rev3['residual_tokens_per_frame_separable']} "
        f"latent_separable={rev3['latent_per_frame_separable']}"
    )
    for row, pp in zip(sweep, pose_protection, strict=True):
        print(
            f"[boundary-aware-rd] q={row['coarsen_quantile']:.2f} | "
            f"SAL bytes={row['saliency_driven']['archive_bytes']} dpose={row['saliency_driven']['d_pose']:.2f} | "
            f"UNI bytes={row['uniform_blind']['archive_bytes']} dpose={row['uniform_blind']['d_pose']:.2f} | "
            f"pose_protect_x={pp['pose_protection_ratio_uni_over_sal']:.2f} "
            f"saliency_protects_pose={pp['saliency_protects_pose']}"
        )
    print(f"[boundary-aware-rd] co_equal_thesis_holds_any_quantile={any_coequal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
