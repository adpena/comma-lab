# SPDX-License-Identifier: MIT
"""EGO-HOOD STATIC-REGION d_seg probe (comma-sweep finding #2).

comma10k class 4 = "my car" (ego hood + dashboard + mounts) — a FIXED, frame-bottom
region whose GT-SegNet argmax is known with CERTAINTY across all frames (no
supercombo<->SegNet IoU gamble, unlike the lane prior). Question: of the basin's
ACTUAL d_seg flips, what fraction live inside the always-class-4 STATIC CORE? Those
flips are 0-byte-correctable (the region is static + its class is known), so they are
a SAFE d_seg reduction.

NO FAKE: real frozen SegNet (``load_frozen_distortion_net(device='cpu')``), real basin
archive parse-back (``best/best_archive.bin`` — the contest-visible bytes), real GT via
``frame_utils.yuv420_to_rgb`` (CLAUDE.md non-negotiable). Reuses the PROVEN detector-cost-B
infra (no duplication). CPU-only (MPS owns the live train; MPS would 2x-corrupt SegNet).
[contest-CPU advisory] NON-PROMOTABLE — a $0 disambiguator, not a score row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the proven, real-input detector-cost-B plumbing (no duplication per CLAUDE.md).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_yousfi_detector_cost_blindspot_b import (  # noqa: E402
    BASIN_DIR,
    _decode_gt_pairs,
    _load_basin_decoder,
    _measure_pair,
)

_HOOD_CLASS = 4  # comma10k "my car"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basin-dir", default=BASIN_DIR)
    ap.add_argument("--n-pairs", type=int, default=96)
    ap.add_argument("--out-json", default=".omx/research/ego_hood_static_dseg_20260617.json")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[probe] loading basin decoder parse-back from {args.basin_dir}/best/best_archive.bin")
    dec, latents, meta, arch_bytes = _load_basin_decoder(args.basin_dir)
    n = min(args.n_pairs, latents.shape[0])
    print(f"[probe] basin: base_ch={meta['base_channels']} latent_dim={meta['latent_dim']} "
          f"n_pairs_ckpt={latents.shape[0]} arch_bytes={arch_bytes}")

    from tac.score_aware_loop.targets import load_frozen_distortion_net

    print("[probe] loading REAL frozen SegNet (CPU AUTHORITY — NO MPS)")
    segnet = load_frozen_distortion_net(device="cpu").segnet

    print(f"[probe] decoding {n} GT pairs (yuv420_to_rgb) + measuring real argmax-flips ...")
    gt_pairs = _decode_gt_pairs(n)
    flips = []   # (n, H, W) bool
    gt_am = []   # (n, H, W) int64  (argmax SegNet on GT frame = the d_seg reference)
    for i in range(n):
        f = _measure_pair(dec, latents, segnet, gt_pairs[i], i)
        flips.append(f.flip)
        gt_am.append(f.gt_argmax)
        if (i + 1) % 16 == 0:
            print(f"  ... {i + 1}/{n} pairs")
    flips = np.stack(flips)          # (n, H, W) bool
    gt_am = np.stack(gt_am)          # (n, H, W) int64
    n_, H, W = flips.shape
    tot_px = n_ * H * W

    d_seg = float(flips.mean())      # the basin's d_seg on these n pairs

    # --- the STATIC ego-hood core: pixels that are class-4 in EVERY measured pair ----
    is_hood = gt_am == _HOOD_CLASS          # (n, H, W)
    p_hood = is_hood.mean(0)                # (H, W) fraction-of-frames hood
    static_core = p_hood == 1.0             # always-hood (the certain, fixed region)
    ever_hood = p_hood > 0.0

    core_px = int(static_core.sum())
    flips_in_core = int(flips[:, static_core].sum()) if core_px else 0
    flips_total = int(flips.sum())
    flips_in_ever = int(flips[:, ever_hood].sum()) if ever_hood.any() else 0

    # flip RATE inside the static core vs outside (is the hood a flip hotspot or quiet?)
    rate_core = float(flips[:, static_core].mean()) if core_px else 0.0
    rate_outside = float(flips[:, ~static_core].mean()) if (~static_core).any() else 0.0

    # the SAFE d_seg reduction = remove every static-core flip (0-byte clamp to the
    # known fixed class). New d_seg = (flips_total - flips_in_core) / tot_px.
    d_seg_after_core_clamp = (flips_total - flips_in_core) / tot_px
    d_seg_after_ever_clamp = (flips_total - flips_in_ever) / tot_px
    dseg_drop_core = d_seg - d_seg_after_core_clamp
    # score units: 100 * d_seg
    score_drop_core = 100.0 * dseg_drop_core
    score_drop_ever = 100.0 * (d_seg - d_seg_after_ever_clamp)

    # where are the core flips? row band (hood is bottom; flips likely at its TOP edge)
    core_flip_rows = None
    if flips_in_core:
        per_row = (flips & static_core[None]).sum(axis=(0, 2))  # (H,)
        nz = np.where(per_row > 0)[0]
        core_flip_rows = [int(nz.min()), int(nz.max())]

    out = {
        "probe": "ego_hood_static_dseg",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "basin_dir": args.basin_dir,
        "arch_bytes": arch_bytes,
        "n_pairs_measured": n_,
        "frame_hw": [H, W],
        "d_seg_basin_on_n_pairs": d_seg,
        "static_core_px": core_px,
        "static_core_frac_of_frame": core_px / (H * W),
        "static_core_row_span": (
            [int(np.where(static_core.any(1))[0].min()),
             int(np.where(static_core.any(1))[0].max())] if core_px else None
        ),
        "ever_hood_frac_of_frame": float(ever_hood.mean()),
        "flips_total": flips_total,
        "flips_in_static_core": flips_in_core,
        "flips_in_ever_hood": flips_in_ever,
        "frac_of_dseg_flips_in_static_core": (flips_in_core / flips_total) if flips_total else 0.0,
        "frac_of_dseg_flips_in_ever_hood": (flips_in_ever / flips_total) if flips_total else 0.0,
        "flip_rate_inside_static_core": rate_core,
        "flip_rate_outside_static_core": rate_outside,
        "core_flip_row_span": core_flip_rows,
        "d_seg_after_static_core_clamp": d_seg_after_core_clamp,
        "d_seg_after_ever_hood_clamp": d_seg_after_ever_clamp,
        "SCORE_DROP_from_static_core_clamp": score_drop_core,
        "SCORE_DROP_from_ever_hood_clamp": score_drop_ever,
        "elapsed_sec": time.time() - t0,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("\n=== EGO-HOOD STATIC-REGION d_seg VERDICT ===")
    print(json.dumps(out, indent=2))
    print(f"\n[written] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
