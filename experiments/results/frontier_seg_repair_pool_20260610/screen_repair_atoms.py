#!/usr/bin/env python
"""Targeted Class-3 seg-repair atom screening on the REAL frontier (#51).

Consumes the flip-map (build_flip_map.py output) + the REAL frontier render +
contest-EXACT GT, generates a Class-3 repair atom per pair (margin-normal
correction toward GT appearance at the flip pixels), and screens it on the EXACT
CPU-torch DistortionNet with HONEST carrier-byte accounting (the actual coded
correction: sorted position-delta + per-pixel signed value, brotli q=11), then
applies THE LAW.

This is the make-or-break test: does repairing real frontier flips clear THE LAW
(100*Δd_seg + Δsqrt(10*d_pose) + 25*Δbytes/N < 0) with honest carrier bytes?

Key fidelity over the #50 advisory headline (which used a degraded-GT render proxy):
  - REAL frontier comp render (FrontierRenderer, byte-faithful to inflate.py).
  - REAL contest GT (frame_utils.yuv420_to_rgb).
  - HONEST carrier bytes (the actual coded correction stream, not 1.5 B/px est).
  - Camera-res application (correction upsampled to 874x1164, the render grid).

AXIS: [macOS-CPU advisory]. The on-host exact replay ratifies. NO MPS. $0 local.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

R2_ANALYSIS = (
    Path(__file__).resolve().parent.parent
    / "pr110pp_r2_nonmps_candidate_20260609" / "analysis"
)
sys.path.insert(0, str(R2_ANALYSIS))
import render_and_score_lib as L  # noqa: E402

from tac.optimization.frame1_seg_repair_atoms import (  # noqa: E402
    RepairTargets,
    SegRepairAtomConfig,
    generate_seg_repair_atom,
    measure_segnet_argmax,
)
from tac.optimization.frame1_joint_safe_cone import measure_pair_distortion  # noqa: E402

try:
    import brotli
except ImportError:  # pragma: no cover
    import brotlicffi as brotli  # type: ignore

N_CONTEST = 37_545_489
RATE_COEF = 25.0
SEG_W = 100.0
POSE_TEN = 10.0
SCORER_H, SCORER_W = 384, 512


def code_correction_bytes(support_mask: np.ndarray, correction_sc: np.ndarray) -> int:
    """HONEST carrier bytes for a scorer-grid (384x512) sparse correction.

    Carrier grammar (the byte-closed stream a minimal inflate patch consumes):
      - sorted flat support indices as uint16 deltas (positions on 384x512), brotli q=11
      - per-pixel signed correction quantized to int8 per channel (3 channels), brotli q=11
    Returns total compressed bytes. The correction is coded at SCORER res (384x512);
    the inflate patch upsamples it to camera-res (the apply path) so the carrier
    stores the minimal scorer-grid signal, not the 5.3x-larger camera grid.
    """
    flat = np.where(support_mask.reshape(-1))[0].astype(np.int64)
    if flat.size == 0:
        return 0
    s = np.sort(flat)
    deltas = np.diff(np.concatenate([[0], s]))
    # deltas fit in uint16 for 384*512=196608 positions (max gap < 65536 in practice;
    # clamp-split would be needed otherwise, but flips are dense enough that gaps are small)
    if deltas.max() >= 65536:
        # fall back to uint32 deltas
        pos_bytes = deltas.astype("<u4").tobytes()
    else:
        pos_bytes = deltas.astype("<u2").tobytes()
    pos_coded = brotli.compress(pos_bytes, quality=11)
    # values: int8 per channel on support pixels only
    vals = np.round(correction_sc.reshape(-1, correction_sc.shape[-1])[s]).astype(np.int16)
    vals = np.clip(vals, -127, 127).astype(np.int8)
    val_coded = brotli.compress(vals.tobytes(), quality=11)
    # + small fixed header (section magic 4 + n_support u32 + 2 length prefixes u32)
    header = 4 + 4 + 4 + 4
    return header + len(pos_coded) + len(val_coded)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-map-dir", required=True)
    ap.add_argument("--pairs", type=str, default="",
                    help="comma list of pair indices; empty = top-flip pairs from summary")
    ap.add_argument("--n-pairs", type=int, default=40, help="how many top-flip pairs to screen")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--correction-fraction", type=float, default=1.0)
    ap.add_argument("--support-top-fraction", type=float, default=1.0,
                    help="fraction of flip pixels to repair (1.0 = all flips)")
    args = ap.parse_args()

    flip_dir = Path(args.flip_map_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pair = json.loads((flip_dir / "flip_map_per_pair.json").read_text())
    by_flips = sorted(per_pair, key=lambda r: -r["n_flip"])
    if args.pairs:
        sel = [int(x) for x in args.pairs.split(",")]
    else:
        sel = [r["pair_index"] for r in by_flips[: args.n_pairs] if r["n_flip"] > 0]

    t0 = time.time()
    renderer = L.FrontierRenderer()
    scorer = L.ExactScorer()
    print(f"[screen] screening {len(sel)} pairs (correction_fraction={args.correction_fraction}, "
          f"support_top_fraction={args.support_top_fraction})", flush=True)

    comp = renderer.render_baseline_pairs(sel)
    gt = L.decode_gt_pairs(sel)

    cfg = SegRepairAtomConfig(
        support_top_fraction=args.support_top_fraction,
        correction_fraction=args.correction_fraction,
        include_fragile=False,  # no #35 fragile mask here; repair ALL flips on the boundary
        boundary_margin_percentile=1.0,  # all flips qualify as boundary (they ARE flips)
        thin_class_max_fraction=1.0,
        min_support_pixels=1,
    )

    rows = []
    n_accept = 0
    total_byte = 0
    total_dS_accepted = 0.0
    for pi in sel:
        comp_pair = comp[pi]            # (2,3,Hc,Wc) camera-res float rounded
        gt_pair = gt[pi].float()        # (2,Hc,Wc,3) uint8->float

        # --- build repair targets at SCORER grid (where flips live) ---
        # SegNet downsamples camera->384x512; we measure argmax/margin there.
        # comp frame-1 at scorer grid:
        comp_bthwc = L.comp_pair_to_bthwc(comp_pair)  # (1,2,Hc,Wc,3)
        with torch.inference_mode():
            comp_seg_in = scorer.net.segnet.preprocess_input(
                comp_bthwc.permute(0, 1, 4, 2, 3).contiguous().float())  # (1,3,384,512)
            comp_logits = scorer.net.segnet(comp_seg_in)
            comp_argmax = comp_logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            top2 = torch.topk(comp_logits, k=2, dim=1)
            comp_margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0)[0].cpu().numpy().astype(np.float64)
            # GT frame-1 at scorer grid
            gt_chw = gt_pair.permute(0, 3, 1, 2).contiguous()  # (2,3,Hc,Wc)
            gt_bthwc = L.comp_pair_to_bthwc(gt_chw)
            gt_seg_in = scorer.net.segnet.preprocess_input(
                gt_bthwc.permute(0, 1, 4, 2, 3).contiguous().float())
            gt_argmax = scorer.net.segnet(gt_seg_in).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
            # appearance gap at scorer grid: GT - comp (frame-1), both bilinear-downsampled
            comp_f1_sc = torch.nn.functional.interpolate(
                comp_pair[1:2], size=(SCORER_H, SCORER_W), mode="bilinear")[0].permute(1, 2, 0).cpu().numpy()
            gt_f1_sc = torch.nn.functional.interpolate(
                gt_chw[1:2].float(), size=(SCORER_H, SCORER_W), mode="bilinear")[0].permute(1, 2, 0).cpu().numpy()
        appearance_gap = gt_f1_sc - comp_f1_sc  # (384,512,3)

        targets = RepairTargets(
            rendered_argmax=comp_argmax,
            gt_argmax=gt_argmax,
            rendered_margin=comp_margin,
            fragile_mask=np.zeros((SCORER_H, SCORER_W), dtype=bool),
            appearance_gap=appearance_gap,
        )
        try:
            atom = generate_seg_repair_atom(pair_index=pi, targets=targets, config=cfg,
                                            prefer_mlx=True)
        except Exception as exc:  # no flips on boundary region
            rows.append({"pair_index": pi, "skipped": str(exc)})
            continue

        # --- HONEST carrier bytes (scorer-grid coded correction) ---
        carrier_bytes = code_correction_bytes(atom.support_mask, atom.correction)

        # --- apply at CAMERA res (the atom upsamples the correction) + screen ---
        # baseline d_seg/d_pose (comp vs GT)
        comp_for_score = L.comp_pair_to_bthwc(comp_pair)   # (1,2,Hc,Wc,3)
        gt_for_score = gt_pair.unsqueeze(0)                # (1,2,Hc,Wc,3)
        base_seg, base_pose = measure_pair_distortion(scorer.net, gt_for_score, comp_for_score)
        # repaired: apply atom to camera-res frame-1
        repaired_pair = atom.apply_to_pair(comp_for_score)  # (1,2,Hc,Wc,3) torch
        rep_seg, rep_pose = measure_pair_distortion(scorer.net, gt_for_score, repaired_pair)

        d_seg_delta = rep_seg - base_seg
        d_pose_delta = rep_pose - base_pose
        seg_term = SEG_W * d_seg_delta
        pose_term = float(np.sqrt(POSE_TEN * max(rep_pose, 0)) - np.sqrt(POSE_TEN * max(base_pose, 0)))
        rate_term = RATE_COEF * carrier_bytes / N_CONTEST
        dS = seg_term + pose_term + rate_term

        seg_reduced = d_seg_delta < 0
        law = dS < 0
        accepted = bool(seg_reduced and law)
        reason = "" if accepted else ("seg_not_reduced" if not seg_reduced else "law_net_nonnegative")
        if accepted:
            n_accept += 1
            total_byte += carrier_bytes
            total_dS_accepted += dS

        rows.append({
            "pair_index": pi,
            "n_flip_target": int(atom.n_support_pixels),
            "base_d_seg": base_seg, "rep_d_seg": rep_seg, "d_seg_delta": d_seg_delta,
            "base_d_pose": base_pose, "rep_d_pose": rep_pose, "d_pose_delta": d_pose_delta,
            "carrier_bytes": carrier_bytes,
            "bytes_per_flip": carrier_bytes / max(atom.n_support_pixels, 1),
            "seg_term": seg_term, "pose_term": pose_term, "rate_term": rate_term,
            "score_delta_advisory": dS,
            "value_per_byte": (-dS / carrier_bytes) if carrier_bytes else 0.0,
            "accepted": accepted, "rejected_reason": reason,
        })
        print(f"  pair {pi}: n={atom.n_support_pixels} dSeg={d_seg_delta:.2e} "
              f"dPose={d_pose_delta:.2e} bytes={carrier_bytes} ({carrier_bytes/max(atom.n_support_pixels,1):.2f}/px) "
              f"dS={dS:.3e} {'ACCEPT' if accepted else reason}", flush=True)

    summary = {
        "schema": "frontier_seg_repair_screen.v1",
        "frontier_archive_sha256_16": "b7106c9bdbb8a2df",
        "n_pairs_screened": len(sel),
        "n_accepted": n_accept,
        "correction_fraction": args.correction_fraction,
        "support_top_fraction": args.support_top_fraction,
        "total_carrier_bytes_accepted": total_byte,
        "total_dS_accepted_advisory_SUM": total_dS_accepted,
        "note_composition": "solo-row SUM is NOT the composed ΔS; domain-recompute on-host",
        "axis_tag": "[macOS-CPU advisory]",
        "provenance": {"score_claim": False, "promotion_eligible": False,
                       "hardware_substrate": "local_macos_cpu"},
        "total_seconds": time.time() - t0,
    }
    (out_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "screen_rows.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
