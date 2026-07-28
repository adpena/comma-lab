#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_da1 D4 -- VALUES 10 MB repricing: minimal-amplitude distribution to flip argmax.

r2s priced per-pixel residual VALUES at 10.3 B/err (int8x3 over ~5.29M camera flip sites). The
value's ONLY job is to flip SegNet argmax at that site to the labeled class through R+uint8. It does
NOT need to reconstruct GT. So: what is the MINIMAL amplitude that flips the argmax?

The GT residual (f1 - f0) is a KNOWN delta that flips argmax to correct (SegNet(f1)=lstars by def).
We line-search the range(A)-projected residual: input(alpha) = f0 + round( P_rangeA( alpha*(f1-f0) ) ),
uint8-clamped, alpha in a grid. For each (384,512) flip site, record the SMALLEST alpha that corrects
its argmax (argmax == lstars). The per-site amplitude = alpha* * |resid| gives the minimal uint8-step
delta distribution. If the median correcting alpha (or uint8 amplitude) is small, the VALUES stream
compresses to near amplitude-free (sign+context) and r2s's 10 MB is a loose upper bound.

Range(A) projection is legitimate: both scorers see only range(A); the ker(A) part of the residual
is scorer-invisible, so restricting the delta to range(A) cannot hurt the flip and is the honest
"scorer-sufficient" delta. Sampled over a subset of pairs (chunked). `[macOS-CPU advisory]`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "ddm_da1_d4_minimal_amplitude.v1"


def _load_segnet(repo_root: Path, weights: Path):
    sys.path.insert(0, str(repo_root / "upstream"))
    import torch
    from modules import SegNet
    from safetensors.torch import load_file
    net = SegNet().eval()
    net.load_state_dict(load_file(str(weights)))
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    return net


def _segnet_argmax(net, frames_bhwc_uint8):
    import einops
    import torch
    x = torch.from_numpy(frames_bhwc_uint8).float()[:, None]
    x = einops.rearrange(x, "b t h w c -> b t c h w")
    with torch.inference_mode():
        inp = net.preprocess_input(x)
        out = net(inp)
        return out.argmax(dim=1).cpu().numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-frames", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("/Users/adpena/Projects/pact"))
    ap.add_argument("--segnet-weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-pairs", type=int, default=60, help="evenly-spaced pairs to sample")
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args(argv)
    t0 = time.time()

    from tac.boundary_math.range_a_projection import apply_projection

    gt = np.load(str(args.gt_frames), mmap_mode="r")
    lstars = gt["lstars"]
    P = min(args.max_pairs, gt["gt_f0"].shape[0])
    sample_idx = np.linspace(0, P - 1, args.sample_pairs).astype(int)
    sample_idx = np.unique(sample_idx)
    seg = _load_segnet(args.repo_root, args.segnet_weights)

    alphas = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0])
    # per correcting-alpha histogram of flip sites; also record uint8 amplitude at correction
    n_alpha = len(alphas)
    corrected_at = np.zeros(n_alpha, dtype=np.int64)  # sites first corrected at alphas[j]
    never_corrected = 0
    total_flip_sites = 0
    amp_samples = []  # uint8 amplitude (max-abs over 3 chan, at scorer-res proj) at correcting alpha

    # to map a (384,512) correction back to a camera uint8 amplitude, we measure the amplitude in
    # camera space of the applied delta at the correcting alpha, sampled at the flip sites' camera
    # footprint is ~2.3px; we report the mean-abs camera delta magnitude at correction as the amplitude.
    for cstart in range(0, len(sample_idx), args.chunk):
        idxs = sample_idx[cstart:cstart + args.chunk]
        f0 = np.asarray(gt["gt_f0"][idxs]).astype(np.float64)  # (b,874,1164,3)
        f1 = np.asarray(gt["gt_f1"][idxs]).astype(np.float64)
        ls = np.asarray(lstars[idxs])  # (b,384,512)
        resid = f1 - f0  # camera residual
        presid = apply_projection(resid, out_dtype=np.float64, compute_dtype=np.float64)  # range(A) resid
        # baseline copy argmax (alpha=0)
        base_arg = _segnet_argmax(seg, np.clip(np.round(f0), 0, 255).astype(np.uint8))
        flip0 = base_arg != ls  # (b,384,512) the flip sites to correct
        nflip = int(flip0.sum())
        total_flip_sites += nflip
        # track first-correcting alpha per site
        first_corr = np.full(ls.shape, -1, dtype=np.int64)  # index into alphas; -1 = never
        camera_amp_at_corr = np.zeros(ls.shape, dtype=np.float64)
        for j, al in enumerate(alphas):
            if al == 0.0:
                continue
            delta = presid * al
            inp = np.clip(np.round(f0 + delta), 0, 255).astype(np.uint8)
            arg = _segnet_argmax(seg, inp)
            now_correct = (arg == ls) & flip0 & (first_corr < 0)
            # camera amplitude at this alpha: mean-abs delta over channels, downsampled to (384,512)
            # via block-mean is expensive; approximate amplitude by scorer-res proxy: use al * (proj
            # residual magnitude) is camera-res; instead record al as the fraction + measure realized
            # camera uint8 amplitude = mean|round(delta)| over channels, then we take at flip sites we
            # need (384,512) alignment -> use the applied camera delta max-abs, resized by mean pooling.
            if now_correct.any():
                # camera delta magnitude (mean over channel) at camera res -> pool to (384,512)
                dmag = np.abs(np.round(delta)).mean(axis=3)  # (b,874,1164)
                # mean-pool to (384,512): reshape not exact (874/384); use simple stride sampling
                import torch
                dm = torch.from_numpy(dmag)[:, None]
                dm_s = torch.nn.functional.interpolate(dm, size=(384, 512), mode="area")[:, 0].numpy()
                camera_amp_at_corr[now_correct] = dm_s[now_correct]
                first_corr[now_correct] = j
        newly = first_corr >= 0
        for j in range(n_alpha):
            corrected_at[j] += int(((first_corr == j) & flip0).sum())
        never_corrected += int((flip0 & (first_corr < 0)).sum())
        amp_samples.append(camera_amp_at_corr[flip0 & newly])
        print(f"[d4] sample pairs {cstart}-{cstart+len(idxs)} flips={nflip} corrected={int(newly.sum())} ({time.time()-t0:.0f}s)", flush=True)

    amp_all = np.concatenate(amp_samples) if amp_samples else np.array([0.0])
    # cumulative correction fraction vs alpha
    cum = np.cumsum(corrected_at) / max(1, total_flip_sites)
    curve = [{"alpha": float(alphas[j]), "cum_frac_corrected": float(cum[j])} for j in range(n_alpha)]

    result = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] real frozen SegNet line-search over range(A)-projected residual; NOT a byte-closed evaluate.py row",
        "sample_pairs": int(len(sample_idx)),
        "total_flip_sites_sampled": total_flip_sites,
        "never_corrected_by_full_rangeA_resid": never_corrected,
        "never_corrected_frac": never_corrected / max(1, total_flip_sites),
        "correction_alpha_curve": curve,
        "camera_uint8_amplitude_at_correction": {
            "median": float(np.median(amp_all)),
            "p25": float(np.percentile(amp_all, 25)),
            "p75": float(np.percentile(amp_all, 75)),
            "p90": float(np.percentile(amp_all, 90)),
            "mean": float(amp_all.mean()),
            "frac_le_2_uint8": float((amp_all <= 2.0).mean()),
            "frac_le_4_uint8": float((amp_all <= 4.0).mean()),
        },
        "interpretation": "cum_frac_corrected at small alpha + median uint8 amplitude decide whether VALUES compress to near amplitude-free (sign+context) vs need full residual magnitude",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[d4] done ({time.time()-t0:.0f}s) -> {args.out}", flush=True)
    print(json.dumps(result["correction_alpha_curve"], indent=2), flush=True)
    print(json.dumps(result["camera_uint8_amplitude_at_correction"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
