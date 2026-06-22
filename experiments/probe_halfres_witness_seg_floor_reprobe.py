# SPDX-License-Identifier: MIT
"""Half-res 192x256 witness seg-floor re-probe ($0, CPU, N<=24, single pass).

[contest-CPU advisory] — a SEG-FLOOR DIAGNOSTIC, NOT a score claim; NON-PROMOTABLE.

HYPOTHESIS (yousfi_96_nonneural_witness_deepdive + CLAUDE.md SegNet blind-spot):
SegNet = smp.Unet('tu-efficientnet_b2') has a STRIDE-2 STEM -> it loses half
resolution immediately -> "artifacts below (256,192) invisible". So a witness
that RENDERS the frame at ~192x256 (half of the 384x512 SegNet-input grid) may
cost ~0 extra d_seg because the stem blurs sub-192 detail anyway. If true, the
witness seg-side floor (boundary set, flip count, residual bytes) is ~4x smaller
than the 384x512 estimate.

THE MEASUREMENT: the witness renders an RGB FRAME; the resolution that SURVIVES
SegNet is the question. So we COARSEN the rendered frame (decoder output, 384x512
float) by bilinear down -> up to render-resolution R, then run the EXACT vendored
eval round-trip:
    decoder -> [coarsen to R] -> bicubic_up(874,1164) -> clamp/round/uint8
    -> SegNet.preprocess (last-frame bilinear-down 384x512) -> SegNet -> argmax.

This is a 1:1 port of the render path in
``tac.torch_vehicle.driver.kit_aware_exact_eval`` (no vendored edit, no MPS).

Render path source-of-truth: driver.kit_aware_exact_eval lines ~3760-3789.
Render-res coarsen point: the decoder's 384x512 float output (the witness's own
spatial-detail budget), BEFORE the camera bicubic upsample.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_EVAL_H, _EVAL_W = 384, 512
_CAM_H, _CAM_W = 874, 1164


def _ensure_upstream() -> None:
    root = Path("upstream").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _build_decoder(ema_decoder_sd: dict, *, latent_dim: int, base_channels: int,
                   taper: list[int]) -> torch.nn.Module:
    from tac.torch_vehicle.configurable_taper_decoder import (
        ConfigurableTaperHNeRVDecoder,
    )

    dec = ConfigurableTaperHNeRVDecoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        eval_size=(_EVAL_H, _EVAL_W),
        channels=taper,
    )
    dec.load_state_dict(ema_decoder_sd, strict=True)
    dec.eval()
    return dec


def _render_argmax(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    distortion_net: torch.nn.Module,
    *,
    render_res: tuple[int, int] | None,
    batch_pairs: int,
    device: str,
) -> torch.Tensor:
    """Render N pairs through the vendored round-trip and return SegNet argmax of
    the LAST frame of each pair, shape (N, 384, 512) int64.

    ``render_res=None`` -> no coarsening (the 384x512 full-res baseline, M_384).
    ``render_res=(h,w)`` -> bilinear down to (h,w) then up to 384x512 BEFORE the
    camera bicubic upsample (the witness at render resolution h x w).
    """
    dev = torch.device(device)
    decoder = decoder.to(dev)
    n_pairs = int(latents.shape[0])
    out_argmax: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, batch_pairs):
            end = min(start + batch_pairs, n_pairs)
            z = latents[start:end].to(dev)
            b = z.shape[0]
            decoded = decoder(z)  # (b,2,3,384,512) float [0,255]
            flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
            if render_res is not None:
                rh, rw = render_res
                # Coarsen the WITNESS render to (rh,rw) then back to the 384x512
                # decoder grid. bilinear matches the decoder's own upsample mode
                # and the SegNet preprocess mode (faithful witness simulation).
                down = F.interpolate(flat, size=(rh, rw), mode="bilinear", align_corners=False)
                flat = F.interpolate(down, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            up = F.interpolate(flat, size=(_CAM_H, _CAM_W), mode="bicubic", align_corners=False)
            cam = up.reshape(b, 2, 3, _CAM_H, _CAM_W).permute(0, 1, 3, 4, 2)
            cand = cam.clamp(0, 255).round().to(torch.uint8)  # (b,2,874,1164,3) uint8
            # SegNet uses the LAST frame; run the candidate pair through the scorer.
            # We only need SegNet argmax -> use the full DistortionNet forward and
            # take the segnet output (out2 in compute_distortion).
            _pose_out, seg_out = distortion_net(cand)  # seg_out (b,5,384,512) logits
            out_argmax.append(seg_out.argmax(dim=1).cpu())  # (b,384,512)
    return torch.cat(out_argmax, dim=0).to(torch.int64)


def _render_seg_logits_lastframe(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    distortion_net: torch.nn.Module,
    *,
    render_res: tuple[int, int] | None,
    batch_pairs: int,
    device: str,
) -> torch.Tensor:
    """Like _render_argmax but returns the raw SegNet logits (N,5,384,512) so we
    can compute the boundary band |top1-top2|. Done as a SEPARATE small pass only
    at the resolutions we need the band for (keeps memory bounded)."""
    dev = torch.device(device)
    decoder = decoder.to(dev)
    n_pairs = int(latents.shape[0])
    out_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, batch_pairs):
            end = min(start + batch_pairs, n_pairs)
            z = latents[start:end].to(dev)
            b = z.shape[0]
            decoded = decoder(z)
            flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
            if render_res is not None:
                rh, rw = render_res
                down = F.interpolate(flat, size=(rh, rw), mode="bilinear", align_corners=False)
                flat = F.interpolate(down, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            up = F.interpolate(flat, size=(_CAM_H, _CAM_W), mode="bicubic", align_corners=False)
            cam = up.reshape(b, 2, 3, _CAM_H, _CAM_W).permute(0, 1, 3, 4, 2)
            cand = cam.clamp(0, 255).round().to(torch.uint8)
            _pose_out, seg_out = distortion_net(cand)  # (b,5,384,512)
            out_logits.append(seg_out.float().cpu())
    return torch.cat(out_logits, dim=0)


def _boundary_band_frac(logits: torch.Tensor, thresh: float) -> float:
    """Fraction of pixels with top1-top2 logit margin < thresh (the thin boundary
    band the witness must spend bytes on)."""
    top2 = logits.topk(2, dim=1).values  # (N,2,H,W)
    margin = (top2[:, 0] - top2[:, 1])
    return float((margin < thresh).float().mean().item())


def _flip_stats(m_a: torch.Tensor, m_b: torch.Tensor) -> dict:
    """Disagreement (d_seg-style) between two argmax maps + contiguity of flips."""
    flips = (m_a != m_b)  # (N,H,W) bool
    n_flip = int(flips.sum().item())
    n_tot = int(flips.numel())
    rate = n_flip / max(n_tot, 1)
    # contiguous fraction: a flip pixel whose 4-neighborhood contains >=1 other
    # flip pixel is "contiguous" (cheap to RLE/region-code); isolated flips are
    # the expensive sparse residual. Vectorized 4-neighbor OR.
    f = flips
    nb = torch.zeros_like(f)
    nb[:, 1:, :] |= f[:, :-1, :]
    nb[:, :-1, :] |= f[:, 1:, :]
    nb[:, :, 1:] |= f[:, :, :-1]
    nb[:, :, :-1] |= f[:, :, 1:]
    contiguous = (f & nb)
    n_contig = int(contiguous.sum().item())
    contig_frac = (n_contig / n_flip) if n_flip else 0.0
    return {
        "n_flipped": n_flip,
        "n_total": n_tot,
        "flip_rate": rate,
        "contiguous_fraction": contig_frac,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="experiments/results/yousfi_r3_taper_marginhinge_e5_20260620/torch_vehicle_checkpoint_state.pt")
    ap.add_argument("--gt-cache",
                    default="experiments/results/capstone_gt_targets_cache/gt_targets_n24.pt")
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-json",
                    default=".omx/research/halfres_witness_seg_floor_reprobe_n24_20260621.json")
    ap.add_argument("--sanity-target", type=float, default=0.002277950717446705,
                    help="run-reported d_seg for the M_384 sanity check")
    ap.add_argument("--full-res-only", action="store_true",
                    help="run only the 384 baseline + 192 (skip the curve points)")
    args = ap.parse_args(argv)

    if args.device.startswith("mps"):
        raise SystemExit("REFUSED: MPS is forbidden for this probe (CPU only; do not touch the live MPS run).")

    t0 = time.time()
    _ensure_upstream()

    # --- load checkpoint EMA shadow (READ-ONLY) ---
    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ema_decoder_sd = st["ema_decoder"]
    ema_latents = st["ema_latents"]  # (600,28)
    n = min(args.n_pairs, int(ema_latents.shape[0]))
    latents = ema_latents[:n].contiguous()

    # --- manifest params (from best_meta / checkpoint manifest) ---
    base_channels = 20
    latent_dim = 28
    taper = [16, 16, 17, 19, 19, 14, 10]

    decoder = _build_decoder(ema_decoder_sd, latent_dim=latent_dim,
                             base_channels=base_channels, taper=taper)

    # --- frozen scorer (CPU) ---
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    dnet = load_frozen_distortion_net(device=args.device)

    # --- GT argmax targets (the d_seg reference the evaluator charges) ---
    gt = torch.load(args.gt_cache, map_location="cpu", weights_only=False)
    gt_seg = gt["seg"][:n].to(torch.int64)  # (n,384,512)

    # === Deliverable 1: M_384 baseline vs M_192-rendered ===
    m_384 = _render_argmax(decoder, latents, dnet, render_res=None,
                           batch_pairs=args.batch_pairs, device=args.device)

    # sanity: d_seg(M_384 vs GT) should match the run's reported d_seg
    d_seg_384_vs_gt = _flip_stats(m_384, gt_seg)["flip_rate"]
    sanity_ok = abs(d_seg_384_vs_gt - args.sanity_target) < 0.0015  # tolerant: N=24 subset
    print(f"[sanity] d_seg(M_384 vs GT) = {d_seg_384_vs_gt:.6f}  target {args.sanity_target:.6f}  ok={sanity_ok}")
    if not sanity_ok:
        # Per the prompt: if the render path doesn't sanity-match, STOP and report.
        result = {
            "authority": "[contest-CPU advisory] — STOPPED: render-path sanity mismatch; NON-PROMOTABLE",
            "STOP_REASON": "render_path_does_not_match_run_d_seg",
            "d_seg_384_vs_gt": d_seg_384_vs_gt,
            "sanity_target": args.sanity_target,
            "abs_diff": abs(d_seg_384_vs_gt - args.sanity_target),
            "n_pairs": n,
            "wall_clock_s": round(time.time() - t0, 1),
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 2

    # render-resolution curve points. All are integer k*(48,64) multiples so the
    # bilinear down/up keeps a clean aspect ratio (avoids the resize-aliasing
    # collapse seen at non-divisor widths like 340).
    res_points: list[tuple[int, int]] = [(192, 256)]
    if not args.full_res_only:
        res_points += [(240, 320), (288, 384), (336, 448)]

    curve: dict[str, dict] = {}
    for (rh, rw) in res_points:
        m_r = _render_argmax(decoder, latents, dnet, render_res=(rh, rw),
                             batch_pairs=args.batch_pairs, device=args.device)
        # d_seg cost of half-res rendering:
        #   (a) vs the full-res render (M_384) -> the EXTRA cost of coarsening
        #   (b) vs GT -> the operationally-relevant evaluator-charged d_seg
        vs_384 = _flip_stats(m_r, m_384)
        vs_gt = _flip_stats(m_r, gt_seg)
        curve[f"{rh}x{rw}"] = {
            "render_res": [rh, rw],
            "d_seg_vs_M384": vs_384["flip_rate"],
            "d_seg_vs_GT": vs_gt["flip_rate"],
            "extra_d_seg_over_full_res": vs_gt["flip_rate"] - d_seg_384_vs_gt,
            "flip_vs_M384_n": vs_384["n_flipped"],
            "flip_vs_M384_contiguous_fraction": vs_384["contiguous_fraction"],
            "flip_vs_GT_n": vs_gt["n_flipped"],
            "flip_vs_GT_contiguous_fraction": vs_gt["contiguous_fraction"],
        }
        print(f"[curve] render {rh}x{rw}: d_seg vs GT = {vs_gt['flip_rate']:.6f} "
              f"(full-res {d_seg_384_vs_gt:.6f}); extra = {vs_gt['flip_rate'] - d_seg_384_vs_gt:+.6f}; "
              f"contig(vs384) = {vs_384['contiguous_fraction']:.3f}")

    # === Deliverable 2 + 4: boundary set size at 384 vs 192 ===
    # Separate logit pass (memory-bounded) at the two key resolutions.
    band_thresh = 0.5
    logits_384 = _render_seg_logits_lastframe(decoder, latents, dnet, render_res=None,
                                              batch_pairs=args.batch_pairs, device=args.device)
    logits_192 = _render_seg_logits_lastframe(decoder, latents, dnet, render_res=(192, 256),
                                              batch_pairs=args.batch_pairs, device=args.device)
    band_384 = _boundary_band_frac(logits_384, band_thresh)
    band_192 = _boundary_band_frac(logits_192, band_thresh)
    band_384_1 = _boundary_band_frac(logits_384, 1.0)
    band_192_1 = _boundary_band_frac(logits_192, 1.0)
    # NOTE: both bands are measured at the SegNet OUTPUT grid (384x512). The
    # render-res coarsening changes the INPUT detail; the band ratio shows whether
    # the half-res render produces a thinner/fatter decision-boundary set.
    boundary_ratio_05 = (band_384 / band_192) if band_192 > 0 else float("inf")
    del logits_384, logits_192

    # === Verdict ===
    extra_192 = curve["192x256"]["d_seg_vs_GT"] - d_seg_384_vs_gt
    # "HOLDS" if the half-res render costs negligible extra d_seg (within ~20% of
    # the full-res d_seg, i.e. the stem blurs the lost detail). "FAILS" if d_seg
    # rises materially (>~50% relative increase) at 192.
    rel_increase_192 = (extra_192 / d_seg_384_vs_gt) if d_seg_384_vs_gt > 0 else float("inf")
    if rel_increase_192 <= 0.20:
        verdict = "HOLDS_STRONG: half-res render costs <=20% extra d_seg -> witness can render at 192; seg-floor materially smaller"
    elif rel_increase_192 <= 0.50:
        verdict = "HOLDS_WEAK: half-res render costs 20-50% extra d_seg -> partial blind-spot benefit"
    else:
        verdict = "FAILS: d_seg rises >50% at 192 -> SegNet effective resolution exceeds 192; witness must render finer"

    # SegNet effective resolution: first res point (descending) where extra d_seg
    # crosses ~20% relative increase over full-res.
    seg_eff_res = "<=192 (192 is fine)"
    if not args.full_res_only:
        # finest -> coarsest (only keys actually present in the curve)
        ordered = [k for k in ("336x448", "288x384", "240x320", "192x256") if k in curve]
        prev_ok = "384x512(baseline)"
        for key in ordered:
            ri = (curve[key]["d_seg_vs_GT"] - d_seg_384_vs_gt) / d_seg_384_vs_gt if d_seg_384_vs_gt > 0 else float("inf")
            if ri > 0.20:
                seg_eff_res = f"between {key} and {prev_ok} (d_seg rises >20% at {key})"
                break
            prev_ok = key
        else:
            seg_eff_res = "<=192 across all tested points (stem-blind hypothesis holds to 192)"

    result = {
        "authority": "[contest-CPU advisory] — half-res witness seg-floor DIAGNOSTIC, NOT a score claim; NON-PROMOTABLE",
        "n_pairs_measured": n,
        "device": args.device,
        "render_path": "decoder->[coarsen to R: bilinear down->up 384x512]->bicubic_up(874,1164)->clamp/round/uint8->SegNet.preprocess(last-frame bilinear-down 384x512)->SegNet->argmax",
        "coarsen_point": "decoder 384x512 float output (the witness spatial-detail budget), BEFORE camera bicubic upsample",
        "checkpoint": {
            "path": args.checkpoint,
            "ema_source": "torch_vehicle_checkpoint_state.pt::ema_decoder/ema_latents (CURRENT shadow, READ-ONLY)",
            "base_channels": base_channels,
            "latent_dim": latent_dim,
            "taper_channels": taper,
        },
        "deliverable_1_sanity": {
            "d_seg_M384_vs_GT": d_seg_384_vs_gt,
            "run_reported_d_seg": args.sanity_target,
            "sanity_match": sanity_ok,
        },
        "deliverable_1_dseg_cost_of_halfres": {
            "M384_vs_GT": d_seg_384_vs_gt,
            "M192up_vs_GT": curve["192x256"]["d_seg_vs_GT"],
            "M192up_vs_M384": curve["192x256"]["d_seg_vs_M384"],
            "extra_d_seg_at_192_over_full_res": extra_192,
            "relative_increase_at_192": rel_increase_192,
        },
        "deliverable_2_boundary_set": {
            "band_thresh_logit_margin": band_thresh,
            "frac_margin_lt_0.5_at_384render": band_384,
            "frac_margin_lt_0.5_at_192render": band_192,
            "boundary_ratio_384_over_192_at_0.5": boundary_ratio_05,
            "frac_margin_lt_1.0_at_384render": band_384_1,
            "frac_margin_lt_1.0_at_192render": band_192_1,
            "note": "bands measured at SegNet OUTPUT grid (384x512); render-res differs at INPUT. predicted ~4x smaller if stem-blind",
        },
        "deliverable_3_flip_counts_and_contiguity": {
            "at_384_vs_GT": {
                "n_flipped": int(d_seg_384_vs_gt * m_384.numel()),
                "n_total": int(m_384.numel()),
            },
            "at_192_vs_M384": {
                "n_flipped": curve["192x256"]["flip_vs_M384_n"],
                "contiguous_fraction": curve["192x256"]["flip_vs_M384_contiguous_fraction"],
            },
            "at_192_vs_GT": {
                "n_flipped": curve["192x256"]["flip_vs_GT_n"],
                "contiguous_fraction": curve["192x256"]["flip_vs_GT_contiguous_fraction"],
            },
        },
        "deliverable_4_dseg_vs_render_resolution_curve": curve,
        "deliverable_4_segnet_effective_decision_resolution": seg_eff_res,
        "deliverable_5_verdict": verdict,
        "wall_clock_s": round(time.time() - t0, 1),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
