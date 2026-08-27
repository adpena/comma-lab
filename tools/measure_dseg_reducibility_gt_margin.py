#!/usr/bin/env python3
"""dseg_reducibility_gt_margin.v1 — the DECISIVE $0 disambiguation of residual d_seg.

THE QUESTION (gates the entire next campaign): is the residual ``d_seg`` of our
current small-basis decoder REDUCIBLE (our reconstruction fails CONFIDENT GT
targets → a better horizon high-frequency decoder + horizon-weighted margin loss
recovers it) or IRREDUCIBLE (intrinsic frozen-SegNet label-noise at near-zero-margin
boundary pixels → no decoder can cheaply satisfy those flips)?

THE DECISIVE SIGNAL — the cross-tab of GT top-2 logit MARGIN against the per-pixel
FLIP set:
  * d_seg = mean over pixels of (argmax(SegNet(GT_frame1)) != argmax(SegNet(OUR_frame1)))
    (the EXACT upstream/modules.py:112 definition; SegNet reads ONLY the last frame).
  * GT margin at a pixel = top1_logit - top2_logit of SegNet(GT_frame1) — the
    confidence of the GT label there. Low margin = SegNet itself is a near-coin-flip
    at that pixel (label-noise frontier); high margin = SegNet is confident.
  * If FLIP pixels concentrate at LOW GT margin (relative to non-flip pixels) →
    IRREDUCIBLE: the flips live where the GT label is itself unstable; a decoder
    cannot cheaply pin a coin-flip. The residual d_seg is near an architectural floor.
  * If FLIP pixels have HEALTHY GT margin (comparable to / higher than non-flip) →
    REDUCIBLE: OUR reconstruction is flipping pixels the GT is CONFIDENT about — a
    better decoder recovers them. Quantify the reducible-margin headroom as a ΔS.

METHOD (apples-to-apples with the authority eval; ONLY per-pixel maps differ):
  1. GT: decode source pairs via ``frame_utils.yuv420_to_rgb`` (the canonical GT
     path — PyAV rgb24 is FORBIDDEN, ~100x phantom pose), stack (B,2,874,1164,3) uint8.
  2. OUR: load the EMA decoder + latents (READ-ONLY copy of the live ``best/``),
     render each pair, bicubic↑ to camera res, clamp/round to uint8 — the SAME
     render→roundtrip the inflate path uses (driver.kit_aware_exact_eval).
  3. SegNet preprocess BOTH (``x[:,-1]`` last frame, bilinear↓ to (384,512)) and run
     the FROZEN SegNet → (B,5,384,512) logits for GT and OUR.
  4. Per pixel: gt_argmax, gt_margin (top1-top2), our_argmax, flip=(gt!=our).
  5. SANITY CHECK: aggregate mean(flip) must reproduce the live run's reported
     d_seg (~0.00211) within ~15% — else STOP (pipeline mismatch, not a verdict).
  6. Cross-tab: margin distribution at flip vs non-flip; per-image-row flip-rate vs
     median GT margin; reducible-headroom ΔS (flips at healthy GT margin).

[contest-CPU advisory] / [macOS-CPU advisory] — NON-PROMOTABLE, mechanism only
(promotable=false, score_claim=false). NEVER MPS (a live MPS training run owns the
GPU). CPU-only, $0.

Run detached:
  nohup .venv/bin/python tools/measure_dseg_reducibility_gt_margin.py \\
    --run-dir experiments/results/yousfi_r3_MUONJUMP_stage8_lr1e3_20260623T180100Z \\
    --n-pairs 48 \\
    --out .omx/research/dseg_reducibility_gt_margin_20260623.json \\
    </dev/null >.omx/tmp/dseg_reducibility.log 2>&1 & disown
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO / "src"), str(_REPO / "upstream"), str(_REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EVAL_H, _EVAL_W = 384, 512  # decoder output / SegNet model input
_CAM_H, _CAM_W = 874, 1164  # camera resolution (the inflate raw-frame size)
_SEG_H, _SEG_W = 384, 512  # SegNet model input size (frame_utils.segnet_model_input_size = (512, 384))


def _utc() -> str:
    return subprocess.run(  # subprocess-no-check-OK: timestamp capture; date(1) failure yields empty string in a receipt, never a decision
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip()


def _copy_best_readonly(run_dir: Path, scratch: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Copy the live ``best/`` EMA decoder + latents into a scratch dir (READ-ONLY
    on the source) so the live MPS training run's directory is never opened for
    writing. Returns (decoder_copy, latents_copy, best_meta)."""
    best = run_dir / "best"
    dec_src = best / "best_ema_decoder.pt"
    lat_src = best / "best_ema_latents.pt"
    meta_src = best / "best_meta.json"
    for p in (dec_src, lat_src):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} (the canonical best/ checkpoint layout)")
    scratch.mkdir(parents=True, exist_ok=True)
    dec_dst = scratch / "best_ema_decoder.pt"
    lat_dst = scratch / "best_ema_latents.pt"
    shutil.copy2(dec_src, dec_dst)
    shutil.copy2(lat_src, lat_dst)
    best_meta: dict[str, Any] = {}
    if meta_src.exists():
        best_meta = json.loads(meta_src.read_text())
    return dec_dst, lat_dst, best_meta


def _build_decoder(taper_channels: list[int], latent_dim: int, base_channels: int):
    """Build the vehicle's eval decoder (ConfigurableTaperHNeRVDecoder with the
    manifest's taper) on CPU — the same factory ``driver._new_vendored_decoder``
    uses when ``cfg.taper_channels`` is set."""
    import torch  # noqa: F401

    from tac.torch_vehicle.configurable_taper_decoder import ConfigurableTaperHNeRVDecoder

    dec = ConfigurableTaperHNeRVDecoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        eval_size=(_EVAL_H, _EVAL_W),
        channels=[int(c) for c in taper_channels],
    )
    return dec


def _load_ema_into(decoder, dec_path: Path, lat_path: Path):
    """Load the bare EMA state_dict + latents (the driver's
    ``_load_warm_start_into`` layout) STRICT onto CPU. Returns the latents
    tensor (n_pairs, latent_dim)."""
    import torch

    sd = torch.load(dec_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    lat = torch.load(lat_path, map_location="cpu", weights_only=False)
    if isinstance(lat, dict):
        lat = lat.get("latents", next(iter(lat.values())))
    decoder.load_state_dict({k: v.to("cpu") for k, v in sd.items()})
    decoder.eval()
    return lat.detach().to("cpu")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-pairs", type=int, default=48)
    ap.add_argument("--batch-pairs", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--live-d-seg",
        type=float,
        default=0.0021093410983060797,
        help="the live run's last-eval d_seg for the sanity check",
    )
    ap.add_argument("--live-d-pose", type=float, default=0.0003658535717598473)
    ap.add_argument("--live-archive-bytes", type=int, default=82457)
    ap.add_argument(
        "--sanity-tol-frac",
        type=float,
        default=0.15,
        help="STOP if |measured - live| / live exceeds this (pipeline mismatch)",
    )
    args = ap.parse_args(argv)

    if str(args.device) != "cpu":
        raise SystemExit("device must be cpu (NEVER MPS — a live MPS training run owns the GPU)")

    import av
    import numpy as np
    import torch

    from frame_utils import yuv420_to_rgb

    from tac.contest_score import compute_contest_score
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    torch.manual_seed(0)
    np.random.seed(0)

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "torch_vehicle_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    taper_channels = manifest.get("taper_channels")
    latent_dim = int(manifest.get("latent_dim", 28))
    base_channels = int(manifest.get("base_channels", 20))
    if not taper_channels:
        raise SystemExit("manifest missing taper_channels — cannot build the eval decoder faithfully")

    scratch = (_REPO / ".omx" / "tmp" / "dseg_reducibility_work").resolve()
    if "/tmp" in str(scratch) and ".omx/tmp" not in str(scratch):
        raise SystemExit("scratch must be under .omx/tmp/")
    dec_path, lat_path, best_meta = _copy_best_readonly(run_dir, scratch)

    # --- build + load OUR eval decoder (CPU, READ-ONLY copy of the live best/) ---
    decoder = _build_decoder(taper_channels, latent_dim, base_channels)
    latents = _load_ema_into(decoder, dec_path, lat_path)
    n_total = int(latents.shape[0])
    N = min(int(args.n_pairs), n_total)

    # --- frozen scorer (canonical loader; CPU authority) ---
    net = load_frozen_distortion_net(upstream_dir=str(_REPO / "upstream"), device=args.device)
    segnet = net.segnet
    segnet.eval()
    dev = torch.device(args.device)

    # --- iterate pairs: GT decode (yuv420_to_rgb) + OUR render, both -> per-pixel SegNet ---
    container = av.open(str(_REPO / "upstream" / "videos" / "0.mkv"))
    frames_iter = container.decode(container.streams.video[0])

    def _next_pair() -> torch.Tensor | None:
        f0 = None
        for frame in frames_iter:
            rgb = yuv420_to_rgb(frame)  # (H,W,3) uint8 camera res
            if f0 is None:
                f0 = rgb
                continue
            return torch.stack([f0, rgb])  # (2,H,W,3)
        return None

    # accumulators (kept compact: histograms + per-row sums, not the full pixel cloud)
    n_pixels_per_pair = _SEG_H * _SEG_W
    # margin histogram edges (logit units). The SegNet logits are unbounded; choose a
    # fine grid near 0 (where label-noise lives) and coarser out to a large scale.
    hist_edges = np.array(
        [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1e9],
        dtype=np.float64,
    )
    n_bins = len(hist_edges) - 1
    flip_margin_hist = np.zeros(n_bins, dtype=np.int64)
    nonflip_margin_hist = np.zeros(n_bins, dtype=np.int64)

    # absolute-threshold flip fractions (cumulative): fraction of flips with margin < t
    abs_thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    flips_below = {t: 0 for t in abs_thresholds}

    # margin quantile collectors (reservoir-free: we keep per-pair medians too, and a
    # downsampled pool for global quantiles — flips are sparse so we can keep them all)
    flip_margins_pool: list[np.ndarray] = []
    # non-flip is huge; keep a uniform subsample to estimate its distribution
    nonflip_margin_subsample: list[np.ndarray] = []
    nonflip_subsample_rate = 0.002  # ~ keeps ~ (0.002 * 384*512 * N) values

    # per-row (SegNet-input row, 0.._SEG_H-1) flip count + margin sum + count
    row_flip_count = np.zeros(_SEG_H, dtype=np.int64)
    row_pixel_count = np.zeros(_SEG_H, dtype=np.int64)
    row_margin_sum = np.zeros(_SEG_H, dtype=np.float64)
    # per-row median GT margin needs per-row values; accumulate per-row margin lists
    # via a coarse per-row histogram instead (memory-bounded). Use the same hist edges.
    row_margin_hist = np.zeros((_SEG_H, n_bins), dtype=np.int64)
    # per-row flip margin hist (to see if high-flip rows are low-margin flips)
    row_flip_margin_hist = np.zeros((_SEG_H, n_bins), dtype=np.int64)

    total_flips = 0
    total_pixels = 0
    per_pair_d_seg: list[float] = []

    pair_idx = 0
    rng = np.random.default_rng(0)
    while pair_idx < N:
        batch_gt = []
        for _ in range(min(int(args.batch_pairs), N - pair_idx)):
            pair = _next_pair()
            if pair is None:
                break
            batch_gt.append(pair)
        if not batch_gt:
            break
        b = len(batch_gt)
        gt = torch.stack(batch_gt).to(dev)  # (b,2,Hc,Wc,3) uint8

        # OUR render: decoder -> (b,2,3,384,512) [0,255] -> bicubic^ -> uint8
        idx = torch.arange(pair_idx, pair_idx + b, device=dev)
        with torch.inference_mode():
            z = latents[idx].to(dev)
            decoded = decoder(z)  # (b,2,3,384,512) float in [0,255]
            flat = decoded.reshape(b * 2, 3, _EVAL_H, _EVAL_W)
            up = torch.nn.functional.interpolate(
                flat, size=(_CAM_H, _CAM_W), mode="bicubic", align_corners=False
            )
            cam_float = up.reshape(b, 2, 3, _CAM_H, _CAM_W).permute(0, 1, 3, 4, 2)
            cand = cam_float.clamp(0, 255).round().to(torch.uint8)  # (b,2,Hc,Wc,3) uint8

            # SegNet preprocess expects (B,T,C,H,W) float; mirror DistortionNet.preprocess_input
            # which rearranges (b t h w c -> b t c h w). SegNet.preprocess_input then takes
            # x[:,-1] and bilinear-resizes to (segnet_model_input_size[1], [0]) = (384,512).
            def _seg_logits(pair_uint8: torch.Tensor) -> torch.Tensor:
                x = pair_uint8.permute(0, 1, 4, 2, 3).float()  # (b,2,3,Hc,Wc)
                seg_in = segnet.preprocess_input(x)  # x[:,-1] + bilinear -> (b,3,384,512)
                return segnet(seg_in)  # (b,5,384,512) logits

            gt_logits = _seg_logits(gt)
            our_logits = _seg_logits(cand)

        gt_arg = gt_logits.argmax(dim=1)  # (b,384,512)
        our_arg = our_logits.argmax(dim=1)
        # GT top-2 margin (top1 - top2) per pixel
        top2 = torch.topk(gt_logits, k=2, dim=1).values  # (b,2,384,512)
        gt_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (b,384,512)
        flip = (gt_arg != our_arg)  # (b,384,512) bool

        gt_margin_np = gt_margin.cpu().numpy()
        flip_np = flip.cpu().numpy()

        # per-pair d_seg (the exact upstream definition, mean over pixels)
        for k in range(b):
            per_pair_d_seg.append(float(flip_np[k].mean()))

        flat_margin = gt_margin_np.reshape(-1)
        flat_flip = flip_np.reshape(-1)
        total_flips += int(flat_flip.sum())
        total_pixels += int(flat_flip.size)

        # margin histograms (flip / non-flip)
        fm = flat_margin[flat_flip]
        nfm = flat_margin[~flat_flip]
        flip_margin_hist += np.histogram(fm, bins=hist_edges)[0]
        nonflip_margin_hist += np.histogram(nfm, bins=hist_edges)[0]

        # cumulative below-threshold flip counts
        for t in abs_thresholds:
            flips_below[t] += int((fm < t).sum())

        # pools for quantiles
        flip_margins_pool.append(fm.astype(np.float32))
        if nfm.size:
            keep = rng.random(nfm.size) < nonflip_subsample_rate
            if keep.any():
                nonflip_margin_subsample.append(nfm[keep].astype(np.float32))

        # per-row accumulation
        for k in range(b):
            rm = gt_margin_np[k]  # (384,512)
            rf = flip_np[k]
            row_flip_count += rf.sum(axis=1)
            row_pixel_count += rf.shape[1]
            row_margin_sum += rm.sum(axis=1)
            for r in range(_SEG_H):
                row_margin_hist[r] += np.histogram(rm[r], bins=hist_edges)[0]
                if rf[r].any():
                    row_flip_margin_hist[r] += np.histogram(rm[r][rf[r]], bins=hist_edges)[0]

        pair_idx += b

    container.close()

    measured_d_seg = float(np.mean(per_pair_d_seg)) if per_pair_d_seg else float("nan")
    sanity_delta = measured_d_seg - args.live_d_seg
    sanity_frac = abs(sanity_delta) / args.live_d_seg if args.live_d_seg else float("inf")
    sanity_ok = sanity_frac <= args.sanity_tol_frac

    # --- distributional summaries ---
    all_flip_margins = (
        np.concatenate(flip_margins_pool) if flip_margins_pool else np.array([], dtype=np.float32)
    )
    all_nonflip_sub = (
        np.concatenate(nonflip_margin_subsample)
        if nonflip_margin_subsample
        else np.array([], dtype=np.float32)
    )

    def _q(a: np.ndarray, qs: list[float]) -> dict[str, float]:
        if a.size == 0:
            return {f"p{int(q*100)}": None for q in qs}
        vals = np.quantile(a, qs)
        return {f"p{int(q*100)}": float(v) for q, v in zip(qs, vals)}

    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    flip_q = _q(all_flip_margins, qs)
    nonflip_q = _q(all_nonflip_sub, qs)
    flip_median = float(np.median(all_flip_margins)) if all_flip_margins.size else None
    nonflip_median = float(np.median(all_nonflip_sub)) if all_nonflip_sub.size else None

    # global GT-margin distribution (flip + nonflip subsample is biased; use the full
    # histogram for the reference scale)
    global_margin_hist = (flip_margin_hist + nonflip_margin_hist).tolist()

    # --- REDUCIBLE-HEADROOM ΔS ---
    # A flip at a "healthy" GT margin is one the GT is confident about → a perfect
    # horizon decoder could recover it. We report the headroom at SEVERAL margin
    # thresholds so no single arbitrary epsilon is load-bearing. The reducible
    # d_seg at threshold t = (# flips with GT margin >= t) / total_pixels; the ΔS
    # is 100 * that (seg term is linear), holding pose+rate at the live values.
    n_flips = int(all_flip_margins.size)
    headroom = {}
    for t in abs_thresholds:
        n_healthy_flips = int((all_flip_margins >= t).sum())
        reducible_d_seg = n_healthy_flips / total_pixels if total_pixels else 0.0
        # ΔS if we removed exactly these flips (d_seg -> d_seg - reducible_d_seg)
        new_d_seg = max(measured_d_seg - reducible_d_seg, 0.0)
        s_now = compute_contest_score(measured_d_seg, args.live_d_pose, args.live_archive_bytes)
        s_after = compute_contest_score(new_d_seg, args.live_d_pose, args.live_archive_bytes)
        headroom[f"margin_ge_{t}"] = {
            "n_healthy_flips": n_healthy_flips,
            "frac_of_flips": (n_healthy_flips / n_flips) if n_flips else None,
            "reducible_d_seg": reducible_d_seg,
            "delta_S_if_recovered": float(s_now - s_after),
        }

    # --- per-row table (downsample to a readable set of rows + the horizon band) ---
    row_median_margin = []
    for r in range(_SEG_H):
        h = row_margin_hist[r]
        tot = int(h.sum())
        if tot == 0:
            row_median_margin.append(None)
            continue
        # median bin via cumulative; report bin-left edge (coarse but bounded-memory)
        cum = np.cumsum(h)
        half = tot / 2.0
        bi = int(np.searchsorted(cum, half))
        bi = min(bi, n_bins - 1)
        row_median_margin.append(float(hist_edges[bi]))
    row_flip_rate = [
        (int(row_flip_count[r]) / int(row_pixel_count[r])) if row_pixel_count[r] else 0.0
        for r in range(_SEG_H)
    ]
    row_mean_margin = [
        (float(row_margin_sum[r]) / int(row_pixel_count[r])) if row_pixel_count[r] else None
        for r in range(_SEG_H)
    ]
    # compact per-row table (every row, but only the loaded fields)
    per_row = [
        {
            "row": r,
            "flip_rate": row_flip_rate[r],
            "median_margin_binleft": row_median_margin[r],
            "mean_margin": row_mean_margin[r],
        }
        for r in range(_SEG_H)
    ]
    # the top-flip rows (where the d_seg lives) for the quick read
    top_flip_rows = sorted(range(_SEG_H), key=lambda r: row_flip_rate[r], reverse=True)[:20]
    top_flip_rows_table = [per_row[r] for r in top_flip_rows]

    # --- VERDICT ---
    # Decisive comparison: do flips concentrate at LOW GT margin relative to non-flip?
    # Use the median ratio AND the fraction of flips below a small absolute logit scale.
    frac_flips_below_0_1 = (flips_below[0.1] / n_flips) if n_flips else None
    frac_flips_below_0_5 = (flips_below[0.5] / n_flips) if n_flips else None
    # reference: fraction of ALL pixels below those margins (the global rate)
    global_hist = np.array(global_margin_hist, dtype=np.float64)
    global_total = float(global_hist.sum())
    # cumulative fraction of all pixels below each edge
    cum_global = np.cumsum(global_hist) / global_total if global_total else None

    def _global_frac_below(t: float) -> float | None:
        if cum_global is None:
            return None
        bi = int(np.searchsorted(hist_edges, t)) - 1
        bi = max(min(bi, n_bins - 1), 0)
        return float(cum_global[bi])

    global_frac_below_0_1 = _global_frac_below(0.1)
    global_frac_below_0_5 = _global_frac_below(0.5)

    verdict = "INCONCLUSIVE"
    verdict_detail = ""
    if not sanity_ok:
        verdict = "BLOCKER_PIPELINE_MISMATCH"
        verdict_detail = (
            f"measured d_seg={measured_d_seg:.6f} vs live {args.live_d_seg:.6f} "
            f"(|delta|/live={sanity_frac:.1%} > tol {args.sanity_tol_frac:.0%}). "
            "The render/preprocess pipeline does NOT match the authority eval; the "
            "cross-tab cannot be trusted. Do NOT issue a reducibility verdict."
        )
    else:
        # The reducible headroom at a moderate-confidence threshold is the score-units
        # signal. Use margin_ge_0.5 as the headline (a SegNet logit gap of 0.5 is
        # clearly above the near-coin-flip noise floor) but report the full curve.
        head_05 = headroom["margin_ge_0.5"]
        frac_reducible_05 = head_05["frac_of_flips"]
        if frac_reducible_05 is None:
            verdict = "INCONCLUSIVE"
            verdict_detail = "no flips measured"
        elif frac_reducible_05 >= 0.5:
            verdict = "REDUCIBLE"
            verdict_detail = (
                f"{frac_reducible_05:.1%} of flips are at GT margin >= 0.5 (the GT is "
                f"CONFIDENT there); our reconstruction is the cause. Reducible headroom "
                f"deltaS (margin>=0.5) = {head_05['delta_S_if_recovered']:.5f}. Build the "
                f"horizon high-frequency decoder + horizon-weighted margin loss."
            )
        elif frac_reducible_05 <= 0.15:
            verdict = "IRREDUCIBLE"
            verdict_detail = (
                f"only {frac_reducible_05:.1%} of flips are at GT margin >= 0.5; the flips "
                f"concentrate at LOW GT margin (near-coin-flip SegNet label-noise frontier). "
                f"No decoder cheaply pins a coin-flip. Residual d_seg is near an architectural "
                f"floor — bank rate+pose, report the ceiling."
            )
        else:
            verdict = "MIXED"
            verdict_detail = (
                f"{frac_reducible_05:.1%} of flips are at GT margin >= 0.5 (reducible), the "
                f"rest are low-margin label-noise. Reducible headroom deltaS (margin>=0.5) = "
                f"{head_05['delta_S_if_recovered']:.5f}; chase only that fraction."
            )

    artifact: dict[str, Any] = {
        "schema": "dseg_reducibility_gt_margin.v1",
        "utc": _utc(),
        "authority_tier": "exact_cpu_advisory",
        "metric_family": "exact_segnet_per_pixel",
        "promotable": False,
        "score_claim": False,
        "mechanism_update_eligible": True,
        "score_roadmap_update_eligible": False,
        "device": args.device,
        "run_dir": str(run_dir),
        "manifest": {
            "taper_channels": taper_channels,
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "best_ep": manifest.get("best_ep"),
            "stage_name": manifest.get("stage_name"),
        },
        "best_meta": best_meta,
        "n_pairs_scored": len(per_pair_d_seg),
        "n_total_pairs_available": n_total,
        "sanity_check": {
            "measured_d_seg": measured_d_seg,
            "live_d_seg": args.live_d_seg,
            "delta": sanity_delta,
            "frac_off": sanity_frac,
            "tol_frac": args.sanity_tol_frac,
            "passed": sanity_ok,
        },
        "live_operating_point": {
            "d_pose": args.live_d_pose,
            "archive_bytes": args.live_archive_bytes,
        },
        "totals": {
            "total_pixels": total_pixels,
            "total_flips": total_flips,
            "n_flip_margin_values": n_flips,
            "n_nonflip_subsample": int(all_nonflip_sub.size),
            "nonflip_subsample_rate": nonflip_subsample_rate,
        },
        "margin_histogram": {
            "edges": hist_edges.tolist(),
            "flip_counts": flip_margin_hist.tolist(),
            "nonflip_counts": nonflip_margin_hist.tolist(),
            "global_counts": global_margin_hist,
        },
        "margin_quantiles": {
            "flip": flip_q,
            "flip_median": flip_median,
            "nonflip_subsample": nonflip_q,
            "nonflip_median": nonflip_median,
        },
        "flips_below_threshold": {
            str(t): {
                "count": flips_below[t],
                "frac_of_flips": (flips_below[t] / n_flips) if n_flips else None,
                "global_frac_of_all_pixels_below": _global_frac_below(t),
            }
            for t in abs_thresholds
        },
        "decisive_separation": {
            "flip_median_margin": flip_median,
            "nonflip_median_margin": nonflip_median,
            "frac_flips_below_0.1": frac_flips_below_0_1,
            "frac_flips_below_0.5": frac_flips_below_0_5,
            "global_frac_all_pixels_below_0.1": global_frac_below_0_1,
            "global_frac_all_pixels_below_0.5": global_frac_below_0_5,
            "interpretation": (
                "If frac_flips_below_t >> global_frac_all_pixels_below_t, flips "
                "concentrate at low GT margin => IRREDUCIBLE label-noise. If flip "
                "margin median ~ nonflip margin median, flips hit confident GT => REDUCIBLE."
            ),
        },
        "reducible_headroom": headroom,
        "per_row_top_flip": top_flip_rows_table,
        "per_row_full": per_row,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print(f"=== d_seg reducibility GT-margin cross-tab (N={len(per_pair_d_seg)} pairs) ===")
    print(f"  measured d_seg={measured_d_seg:.6f}  live d_seg={args.live_d_seg:.6f}  "
          f"frac_off={sanity_frac:.1%}  sanity_ok={sanity_ok}")
    print(f"  flips={total_flips}/{total_pixels}  flip_median_margin={flip_median}  "
          f"nonflip_median_margin={nonflip_median}")
    print(f"  frac_flips_below_0.5={frac_flips_below_0_5}  "
          f"global_frac_all_below_0.5={global_frac_below_0_5}")
    print(f"  reducible headroom (margin>=0.5): frac_of_flips="
          f"{headroom['margin_ge_0.5']['frac_of_flips']}  "
          f"deltaS={headroom['margin_ge_0.5']['delta_S_if_recovered']:.5f}")
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print(f"  wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
