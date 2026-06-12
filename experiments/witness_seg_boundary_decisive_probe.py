# SPDX-License-Identifier: MIT
"""WITNESS SEG-BOUNDARY decisive $0 probe (Track-B GO/HYBRID/NO-GO, seg half).

Resolves the SEG half + the scorer-conditional-MDL measurement of the
"scorer-conditional witness carrier" reframe (memos
``layer1_carrier_first_principles_20260612T171912Z.md`` rank-4 first step +
``frozen_contest_space_council_lenses_synthesis_20260612T173627Z.md`` D.2 step 2).

The carrier analysis reduces the seg-axis class-shift question to four measurements
on the FROZEN basin checkpoint (READ-ONLY), all $0 CPU (reuse ``seg_out``; no GPU;
no basin daemon contention):

  A. BOUNDARY GEOMETRY. Render the basin frames, run the frozen SegNet, compute the
     per-pixel margin (top1-top2 logit). The boundary set ``∂`` = pixels where the
     basin argmax != GT argmax (the actual d_seg flips) ∪ the near-flip band. Measure
     |∂|/|frame| (sparsity) and |flips| (the d_seg debt count).

  B. CONDITIONAL-POSITION BYTE COST vs the d_seg win. Code ``∂`` CONDITIONAL on the
     decoder-regenerated margin field (FREE: the decoder reproduces the margin), via
     the canonical ``margin_conditional_residual`` coder. Get B/flip. Compare to the
     1.27 B/flip break-even (``WATERLINE_BYTES_PER_FLIP``). Below it -> net-negative-S.

  C. ROUND-TRIP SURVIVAL. Do the boundary corrections survive bicubic↑874 ->
     bilinear↓384 -> uint8? Flip K boundary pixels in the RENDERED 384x512 frame,
     push through the EXACT eval round-trip, re-measure the argmax. Fraction that
     still flips the right way = survival.

  D. SCORER-CONDITIONAL MDL. Estimate the AMORTIZED witness bytes = mask-grammar
     (the 600-frame argmax partition, AMORTIZED via a small decoder / contour-coded,
     NOT direct-stored which LOSES 524 KB) + pose-trajectory (~1.5 KB measured) +
     boundary-residual (from B). Does the sum come BELOW 177 KB and toward 24.6-64.6 KB?

NO FAKE: the frozen contest SegNet (``load_frozen_distortion_net``), GT via
``frame_utils.yuv420_to_rgb`` (the canonical path inside ``RealScorerContext`` /
``precompute_targets``), the EXACT eval round-trip (``_decoded_to_camera`` bicubic↑874
+ ``segnet.preprocess_input`` bilinear↓384 + uint8). Margins, flips, bytes, survival
are all MEASURED on the real rendered frames. No score is claimed; every advisory
number is ``[contest-CPU advisory]`` NON-PROMOTABLE until a byte-closed archive is run
through ``upstream/evaluate.py``. The frontier is UNMOVED.

The "direct partition storage LOSES 524 KB -> must AMORTIZE" caveat is honored: D
reports BOTH the direct-store cost AND the amortized estimate and flags which wins.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# ── canonical reused helpers (SEARCH-FIRST: do not reinvent) ─────────────────
from tac.boundary_math.margin_conditional_residual import (
    WATERLINE_BYTES_PER_FLIP,
    measure_code_cost,
)
from tac.boundary_math.bitmask_dseg import d_seg_reference, flip_count

_EVAL_H, _EVAL_W = 384, 512  # SegNet working resolution (= scored grid)
_N_SCORED_PER_FRAME = _EVAL_H * _EVAL_W  # 196,608
_RATE_DENOM = 37_545_489
_N_FRAMES = 600
_FRONTIER_BYTES = 177_169  # pointer; the witness must beat this on bytes for the class shift
_CONDITIONAL_MDL_LO = 24_600  # DERIVED ~24.6 KB (smaller_learned_basis_deep_math §3)
_CONDITIONAL_MDL_HI = 64_600  # DERIVED ~64.6 KB
_POSE_TRAJ_BYTES = 1_557  # MEASURED pose-output entropy (information_theoretic_floor_T_floor P6)


@dataclass
class PerPairBoundary:
    pair_index: int
    n_flips: int
    boundary_size_tau05: int  # |∂| at margin tau=0.5 (the canonical low-margin band)
    flips_in_boundary_tau05: int
    cond_bytes_per_flip: float
    uncond_bytes_per_flip: float
    cond_total_bytes: float
    gt_n_regions: int  # connected-ish proxy: number of distinct (argmax) labels present
    rendered_frame1_argmax_bytes_lzma_proxy: int  # direct-store proxy for amortization caveat


def _load_basin_decoder(ckpt_path: Path, which: str):
    """Load the basin decoder + latents (READ-ONLY). ``which`` in {ema, live}."""
    from tac.torch_vehicle.vendored_imports import import_vendored

    common = import_vendored("common")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec = common.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dkey = "ema_decoder" if which == "ema" else "decoder"
    lkey = "ema_latents" if which == "ema" else "latents"
    dec.load_state_dict(sd[dkey])
    dec.eval()
    latents = sd[lkey].detach().float()
    return dec, latents


def _render_and_segforward(dec, net, score_mod, latents, idx):
    """Render the basin frames for ``idx`` and run the frozen SegNet on the EXACT
    eval round-trip. Returns (seg_out_rendered (B,5,384,512), the rendered last-frame
    384x512 uint8 for round-trip survival). NO FAKE: identical path to evaluate_decoder."""
    with torch.inference_mode():
        z = latents[idx]
        decoded = dec(z)  # (B,2,3,384,512) float[0,255]
        B = decoded.shape[0]
        flat = decoded.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
        up = score_mod._decoded_to_camera(flat)  # bicubic ↑874x1164
        bhwc = (
            up.reshape(B, 2, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
            .permute(0, 1, 3, 4, 2)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
        )
        _posenet_in, segnet_in = net.preprocess_input(bhwc)  # last-frame, bilinear↓384, uint8
        seg_out = net.segnet(segnet_in)  # (B,5,384,512)
    return seg_out, segnet_in  # segnet_in is the 384x512 round-tripped input (float)


def _margin_map(seg_out: torch.Tensor) -> torch.Tensor:
    """Per-pixel top1-top2 SegNet logit margin (>=0). (B,5,H,W) -> (B,H,W)."""
    top2, _ = torch.topk(seg_out, k=2, dim=1, largest=True, sorted=True)
    return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)


def _direct_store_proxy_bytes(argmax_hw: np.ndarray) -> int:
    """Direct-store cost proxy for the per-frame argmax partition (the AMORTIZATION
    caveat): zlib over the raw uint8 label map. This is a LOOSE upper bound; the memo's
    measured LZMA-over-labels baseline is ~896 B/frame. We report zlib as a cheap proxy
    and flag that direct-store LOSES vs the amortized decoder."""
    return len(zlib.compress(argmax_hw.astype(np.uint8).tobytes(), 9))


def run_probe(
    *,
    ckpt_path: Path,
    video_path: Path,
    which_decoder: str,
    n_pairs: int,
    tau: float,
    survival_k: int,
    batch: int,
    targets_cache: Path,
) -> dict:
    t_start = time.time()
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    from tac.torch_vehicle.vendored_imports import import_vendored

    score_mod = import_vendored("score")
    net = load_frozen_distortion_net(device="cpu")
    dec, latents = _load_basin_decoder(ckpt_path, which_decoder)

    # GT seg targets (argmax label maps at 384x512) via the canonical RealScorerContext
    # (GT decoded ONLY via yuv420_to_rgb inside precompute_targets / build_gt_targets).
    from tac.torch_vehicle.scorer_context import RealScorerContext

    ctx = RealScorerContext(
        str(video_path), device="cpu", max_pairs=n_pairs, targets_cache=str(targets_cache)
    )
    gt_argmax = ctx.seg_targets_hard.cpu().numpy()  # (n, 384, 512) int64 GT argmax
    n_pairs = int(min(n_pairs, gt_argmax.shape[0]))

    per_pair: list[PerPairBoundary] = []
    survival_records: list[dict] = []

    for start in range(0, n_pairs, batch):
        idx = torch.arange(start, min(start + batch, n_pairs))
        seg_out, segnet_in = _render_and_segforward(dec, net, score_mod, latents, idx)
        margin = _margin_map(seg_out).cpu().numpy()  # (B,384,512)
        rendered_argmax = seg_out.argmax(dim=1).cpu().numpy()  # (B,384,512)

        for j, pidx in enumerate(idx.tolist()):
            g = gt_argmax[pidx]  # (384,512)
            r = rendered_argmax[j]
            m = margin[j]
            flips_mask = r != g
            n_flips = int(flips_mask.sum())
            # boundary set ∂ at tau (the decoder-known low-margin band)
            boundary_mask = m < tau
            boundary_size = int(boundary_mask.sum())
            flips_in_b = int((flips_mask & boundary_mask).sum())
            # conditional-position byte cost of the ACTUAL flip set (cond on free margin)
            flip_idx = np.flatnonzero(flips_mask.reshape(-1))
            target_cls = g.reshape(-1)[flip_idx]  # the GT class to correct to
            if n_flips > 0:
                cost = measure_code_cost(
                    m.reshape(-1).astype(np.float64), flip_idx, target_cls, tau=tau
                )
                cbpf = cost.bytes_per_flip
                ubpf = cost.unconditional_bytes_per_flip
                ctot = cost.total_bits / 8.0
            else:
                cbpf = ubpf = ctot = 0.0
            per_pair.append(
                PerPairBoundary(
                    pair_index=pidx,
                    n_flips=n_flips,
                    boundary_size_tau05=boundary_size,
                    flips_in_boundary_tau05=flips_in_b,
                    cond_bytes_per_flip=cbpf,
                    uncond_bytes_per_flip=ubpf,
                    cond_total_bytes=ctot,
                    gt_n_regions=int(len(np.unique(g))),
                    rendered_frame1_argmax_bytes_lzma_proxy=_direct_store_proxy_bytes(g),
                )
            )

        # ── C. ROUND-TRIP SURVIVAL on the FIRST batch only (decisive, $0) ──
        if start == 0:
            survival_records = _measure_roundtrip_survival(
                dec, net, score_mod, latents, idx, gt_argmax, margin, rendered_argmax,
                tau=tau, survival_k=survival_k,
            )

    # ── aggregate ────────────────────────────────────────────────────────────
    agg = _aggregate(per_pair, survival_records, n_pairs=n_pairs, tau=tau)
    agg["wall_seconds"] = round(time.time() - t_start, 1)
    agg["which_decoder"] = which_decoder
    agg["n_pairs_measured"] = n_pairs
    agg["per_pair_sample"] = [asdict(p) for p in per_pair[:8]]  # a peek, not the full dump
    return agg


def _measure_roundtrip_survival(
    dec, net, score_mod, latents, idx, gt_argmax, margin, rendered_argmax,
    *, tau: float, survival_k: int,
) -> list[dict]:
    """C. Flip K boundary pixels per pair toward GT in the RENDERED 384x512 frame,
    push through the EXACT eval round-trip (bicubic↑874 -> bilinear↓384 -> uint8),
    re-run SegNet, and check whether the argmax now matches GT at those pixels.

    The correction is applied in the 384x512 RENDERED frame BEFORE the round-trip
    (this is where a witness sidecar would inject it). We measure how many survive.
    NO FAKE: we re-render, perturb, re-roundtrip, re-forward through the real SegNet.
    """
    records: list[dict] = []
    with torch.inference_mode():
        z = latents[idx]
        decoded = dec(z)  # (B,2,3,384,512) float[0,255]
        B = decoded.shape[0]
        for j, pidx in enumerate(idx.tolist()):
            g = gt_argmax[pidx]
            r = rendered_argmax[j]
            m = margin[j]
            flips_mask = (r != g) & (m < tau)  # boundary flips (the cheap, fixable set)
            flip_idx = np.flatnonzero(flips_mask.reshape(-1))
            if len(flip_idx) == 0:
                continue
            k = int(min(survival_k, len(flip_idx)))
            # pick the K LOWEST-margin flips (the cheapest, most-fixable)
            order = np.argsort(m.reshape(-1)[flip_idx])
            chosen = flip_idx[order[:k]]
            ys, xs = np.unravel_index(chosen, (_EVAL_H, _EVAL_W))

            # The witness correction is at the SegNet's 384x512 INPUT grid. To inject a
            # correction that must survive the round-trip, we perturb the RENDERED native
            # 384x512 frame1 toward a value that pushes the SegNet argmax to GT. The most
            # faithful, non-cheating perturbation: set the corrected pixels to the MEAN
            # RGB of GT-class pixels in the rendered frame (a class-prototype nudge — the
            # strongest a per-pixel sidecar could legally do). Then round-trip + re-score.
            frame1 = decoded[j, 1].clone()  # (3,384,512) float[0,255]
            # build a per-class prototype color from the rendered frame's own pixels
            r_flat = r.reshape(-1)
            f_flat = frame1.reshape(3, -1)
            corrected = frame1.clone()
            for cls in np.unique(g.reshape(-1)[chosen]):
                cls_pixels = r_flat == cls
                if cls_pixels.sum() == 0:
                    # GT class absent in render -> use global-darkest as a fallback nudge
                    proto = f_flat.min(dim=1).values
                else:
                    proto = f_flat[:, torch.from_numpy(cls_pixels)].mean(dim=1)
                # apply prototype to the chosen pixels whose GT class == cls
                sel = g.reshape(-1)[chosen] == cls
                sel_y = ys[sel]
                sel_x = xs[sel]
                for c in range(3):
                    corrected[c, sel_y, sel_x] = proto[c]

            # round-trip the CORRECTED frame1 through the exact eval channel
            up = score_mod._decoded_to_camera(corrected.unsqueeze(0))  # (1,3,874,1164)
            bhwc = (
                up.unsqueeze(1)  # treat as a single-frame "last frame"
                .reshape(1, 1, 3, score_mod.CAMERA_H, score_mod.CAMERA_W)
                .permute(0, 1, 3, 4, 2)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
            )
            # preprocess_input expects (B,2,H,W,3); duplicate to a 2-frame stack (seg
            # uses only the last frame, so a duplicate is exact for the seg path)
            bhwc2 = bhwc.repeat(1, 2, 1, 1, 1)
            _pin, segnet_in_c = net.preprocess_input(bhwc2)
            seg_out_c = net.segnet(segnet_in_c)
            new_argmax = seg_out_c.argmax(dim=1)[0].cpu().numpy()  # (384,512)

            g_target = g[ys, xs]
            survived = int((new_argmax[ys, xs] == g_target).sum())
            records.append(
                {
                    "pair_index": pidx,
                    "k_attempted": k,
                    "k_survived": survived,
                    "survival_fraction": survived / k if k else 0.0,
                }
            )
    return records


def _aggregate(per_pair, survival_records, *, n_pairs, tau) -> dict:
    flips = np.array([p.n_flips for p in per_pair], dtype=np.int64)
    bsize = np.array([p.boundary_size_tau05 for p in per_pair], dtype=np.int64)
    cbpf = np.array([p.cond_bytes_per_flip for p in per_pair if p.n_flips > 0], dtype=np.float64)
    ubpf = np.array([p.uncond_bytes_per_flip for p in per_pair if p.n_flips > 0], dtype=np.float64)
    ctot = np.array([p.cond_total_bytes for p in per_pair], dtype=np.float64)
    direct_store = np.array(
        [p.rendered_frame1_argmax_bytes_lzma_proxy for p in per_pair], dtype=np.int64
    )

    total_flips = int(flips.sum())
    mean_dseg = float(flips.sum() / (n_pairs * _N_SCORED_PER_FRAME))
    seg_term = 100.0 * mean_dseg  # the 100*d_seg contribution at this base

    # A. boundary geometry
    boundary_frac = float(bsize.mean() / _N_SCORED_PER_FRAME)
    flips_frac = float(flips.mean() / _N_SCORED_PER_FRAME)

    # B. conditional-position cost vs 1.27 B/flip break-even
    mean_cbpf = float(cbpf.mean()) if len(cbpf) else 0.0
    median_cbpf = float(np.median(cbpf)) if len(cbpf) else 0.0
    frac_pairs_under_waterline = (
        float((cbpf < WATERLINE_BYTES_PER_FLIP).mean()) if len(cbpf) else 0.0
    )
    # aggregate flip-set cost: total conditional residual bytes for ALL flips
    total_residual_bytes = float(ctot.sum())
    # if we coded EVERY flip at the measured conditional cost, the seg term drops to ~0
    seg_win_if_all_fixed = seg_term  # full d_seg -> 0
    rate_cost_all_fixed = total_residual_bytes * (25.0 / _RATE_DENOM)
    # but scale total_residual_bytes from n_pairs to the full 600 (amortize)
    scale = _N_FRAMES / max(n_pairs, 1)
    total_residual_bytes_600 = total_residual_bytes * scale
    net_delta_S_boundary_sidecar = (
        -seg_win_if_all_fixed + total_residual_bytes_600 * (25.0 / _RATE_DENOM)
    )

    # C. round-trip survival
    if survival_records:
        surv_frac = float(np.mean([r["survival_fraction"] for r in survival_records]))
        surv_k_total = int(sum(r["k_attempted"] for r in survival_records))
        surv_survived = int(sum(r["k_survived"] for r in survival_records))
    else:
        surv_frac = 0.0
        surv_k_total = surv_survived = 0

    # D. scorer-conditional MDL — AMORTIZED estimate vs direct-store
    direct_store_600 = float(direct_store.mean() * _N_FRAMES)  # direct-store LOSES
    # amortized seg-core: the conditional MDL band's seg-core is ~20-55 KB (DERIVED).
    # the witness sums: amortized mask-grammar (use the band) + pose (1.5 KB) + boundary residual
    amortized_seg_core_lo = 20_000
    amortized_seg_core_hi = 55_000
    witness_bytes_lo = amortized_seg_core_lo + _POSE_TRAJ_BYTES + total_residual_bytes_600
    witness_bytes_hi = amortized_seg_core_hi + _POSE_TRAJ_BYTES + total_residual_bytes_600

    # ── verdict logic (seg half) ──
    sparse_boundary = boundary_frac < 0.10  # thin ∂
    survives = surv_frac >= 0.50  # high round-trip survival
    cost_under_break_even = mean_cbpf < WATERLINE_BYTES_PER_FLIP and mean_cbpf > 0
    mdl_below_frontier = witness_bytes_hi < _FRONTIER_BYTES
    boundary_sidecar_net_negative_S = net_delta_S_boundary_sidecar < 0

    if (
        sparse_boundary
        and survives
        and cost_under_break_even
        and mdl_below_frontier
        and boundary_sidecar_net_negative_S
    ):
        verdict = "WITNESS_SEG_HALF_GO"
    elif (sparse_boundary or cost_under_break_even) and not (
        survives and boundary_sidecar_net_negative_S
    ):
        verdict = "HYBRID_FOLD_INTO_TRAINING"
    elif not sparse_boundary or not survives:
        verdict = "NO_GO_SEG_AXIS"
    else:
        verdict = "HYBRID_FOLD_INTO_TRAINING"

    return {
        "evidence_grade": "[contest-CPU advisory] NON-PROMOTABLE",
        "tau": tau,
        "waterline_bytes_per_flip": WATERLINE_BYTES_PER_FLIP,
        # base d_seg state
        "total_flips_measured": total_flips,
        "mean_d_seg": mean_dseg,
        "seg_term_100_dseg": seg_term,
        # A. geometry
        "A_boundary_frac_of_frame": boundary_frac,
        "A_flips_frac_of_frame": flips_frac,
        "A_mean_flips_per_pair": float(flips.mean()),
        "A_mean_boundary_size_per_pair": float(bsize.mean()),
        "A_sparse_boundary": sparse_boundary,
        # B. conditional cost vs break-even
        "B_mean_cond_bytes_per_flip": mean_cbpf,
        "B_median_cond_bytes_per_flip": median_cbpf,
        "B_mean_uncond_bytes_per_flip": float(ubpf.mean()) if len(ubpf) else 0.0,
        "B_frac_pairs_cond_under_waterline": frac_pairs_under_waterline,
        "B_cost_under_break_even": cost_under_break_even,
        "B_total_residual_bytes_600_scaled": total_residual_bytes_600,
        "B_net_delta_S_boundary_sidecar_if_all_fixed": net_delta_S_boundary_sidecar,
        "B_boundary_sidecar_net_negative_S": boundary_sidecar_net_negative_S,
        # C. round-trip survival
        "C_survival_fraction": surv_frac,
        "C_survived_over_attempted": f"{surv_survived}/{surv_k_total}",
        "C_survives_threshold_0p50": survives,
        # D. scorer-conditional MDL
        "D_direct_store_partition_bytes_600": direct_store_600,
        "D_direct_store_LOSES_vs_frontier": direct_store_600 > _FRONTIER_BYTES,
        "D_witness_bytes_amortized_lo": witness_bytes_lo,
        "D_witness_bytes_amortized_hi": witness_bytes_hi,
        "D_frontier_bytes": _FRONTIER_BYTES,
        "D_conditional_mdl_band": [_CONDITIONAL_MDL_LO, _CONDITIONAL_MDL_HI],
        "D_pose_traj_bytes_measured": _POSE_TRAJ_BYTES,
        "D_witness_mdl_below_frontier": mdl_below_frontier,
        "D_witness_mdl_in_conditional_band": witness_bytes_hi <= _CONDITIONAL_MDL_HI * 1.5,
        # VERDICT
        "VERDICT_SEG_HALF": verdict,
        "n_survival_records": len(survival_records),
        "survival_records": survival_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        default="experiments/results/forkpoints/basin_bc20_20260612T121523Z/torch_vehicle_checkpoint_state.pt",
    )
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--which-decoder", choices=["ema", "live"], default="live",
                    help="ema = the EMA shadow (lags per the shadow-lag memo); live = the "
                         "trained decoder (descends further). Default live: the real basin state.")
    ap.add_argument("--n-pairs", type=int, default=120,
                    help="pairs to measure (subsample spans the video; 120 is decisive + $0).")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="margin threshold for the boundary band ∂ (canonical 0.5).")
    ap.add_argument("--survival-k", type=int, default=64,
                    help="boundary pixels per pair to test round-trip survival on.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--targets-cache", default=".omx/tmp/witness_probe_targets")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    result = run_probe(
        ckpt_path=Path(args.ckpt),
        video_path=Path(args.video),
        which_decoder=args.which_decoder,
        n_pairs=args.n_pairs,
        tau=args.tau,
        survival_k=args.survival_k,
        batch=args.batch,
        targets_cache=Path(args.targets_cache),
    )
    print(json.dumps(result, indent=2, default=float))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=float))
        print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
