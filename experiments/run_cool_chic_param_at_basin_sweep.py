# SPDX-License-Identifier: MIT
"""Cool-Chic param-at-d_seg-basin sweep — the decisive deliverable.

Lane ``lane_cool_chic_score_aware_basis_20260611``. Sweeps the multiresolution
latent-grid resolution (and synthesis width) and records the CHARGED BYTES at
which the score-aware-fit exact ``d_seg`` (live frozen SegNet) descends. The
deliverable curve is ``charged_bytes -> exact d_seg`` — the single number that
answers basis-specific vs fundamental (does Cool-Chic reach the 5.6e-4 basin at
FAR fewer bytes than conv-HNeRV's ~162-256K params?).

Honesty discipline (CLAUDE.md "Forbidden score claims" + MLX/MPS rules):
  - GPU is busy; runs torch-CPU only. NO MPS (MPS corrupts SegNet ~2x).
  - Every row is ``[macOS-CPU advisory]`` / ``promotable=False``. The exact
    ``d_seg`` here is the LIVE SegNet argmax-disagreement (NOT a proxy) BUT the
    contest score still requires Linux-x86_64 evaluate.py on a byte-closed
    archive — this measures the d_seg axis only, not S.
  - ``--scorer-hw`` defaults to a REDUCED 192x256 for sweep tractability on CPU
    (faithful gradient path, ~4x cheaper than the 384x512 camera-eval size).
    The full-res anchor is the separate deep_fit_fullres run. Each row records
    its scorer_hw so measured-vs-derived is explicit.

Reuses: ScoreAwareTrainer (live-SegNet loss/EMA/eval-roundtrip/exact d_seg),
the n48 gt-targets cache, CoolChicPairCarrier.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tac.residual_basis.cool_chic_carrier import CoolChicGridSpec, CoolChicPairCarrier
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE = "experiments/results/capstone_gt_targets_cache/gt_targets_n48.pt"


def run_one(
    net,
    seg_targets,
    *,
    base_h: int,
    base_w: int,
    n_grids: int,
    channels_per_grid: int,
    synth_hidden: int,
    out_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    epochs: int,
    decoder_lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    n_pairs = int(seg_targets.shape[0])
    spec = CoolChicGridSpec(
        base_h=base_h, base_w=base_w, n_grids=n_grids,
        channels_per_grid=channels_per_grid,
    )
    carrier = CoolChicPairCarrier(
        n_pairs=n_pairs, spec=spec, synth_hidden=synth_hidden, out_hw=out_hw
    )
    charged = carrier.charged_bytes()
    cfg = ScoreAwareLoopConfig(
        epochs=epochs, batch_size=n_pairs, scorer_hw=scorer_hw,
        pose_enabled=False, eval_every=max(epochs // 4, 1),
        seg_loss_form="ce_seg_loss", decoder_lr=decoder_lr,
        latent_lr_mult=10.0, ema_decay=0.99, seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg_targets, None, cfg)
    t0 = time.time()
    res = tr.train()
    # CRITICAL (recursive-greenup, the EMA-shadow-lag question): on SHORT fits the
    # warmup-EMA shadow still lags the LIVE weights, so the shadow d_seg UNDER-reports
    # descent. Record the LIVE-render d_seg too — it is the un-confounded headline.
    d_seg_live = tr.exact_d_seg(use_ema=False)
    return {
        "base_hw": [base_h, base_w],
        "n_grids": n_grids,
        "channels_per_grid": channels_per_grid,
        "synth_hidden": synth_hidden,
        "out_hw": list(out_hw),
        "scorer_hw": list(scorer_hw),
        "epochs": epochs,
        "decoder_lr": decoder_lr,
        "charged_bytes": charged,
        "d_seg_initial": res["d_seg_initial"],
        "d_seg_final_ema": res["d_seg_final_ema"],
        "d_seg_best_ema": res["d_seg_best_ema"],
        "d_seg_live": d_seg_live,
        "descended": res["descended"],
        "descended_live": bool(d_seg_live < res["d_seg_initial"] - 1e-4),
        "wall_seconds": round(time.time() - t0, 1),
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--decoder-lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument(
        "--out",
        default="experiments/results/lane_cool_chic_score_aware_basis_20260611/param_at_basin_sweep.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    seg = d["seg"][: args.n_pairs]
    net = load_frozen_distortion_net(device="cpu")

    # Sweep grid resolution coarse->fine; the latent count (and charged bytes)
    # grows with base resolution. out_hw is the render res (fixed cheap); the
    # scorer roundtrip resizes to scorer_hw.
    grid_configs = [
        # (base_h, base_w, n_grids, ch_per_grid, synth_hidden, out_h, out_w)
        (12, 16, 3, 2, 12, 48, 64),
        (24, 32, 3, 2, 16, 48, 64),
        (24, 32, 4, 2, 16, 64, 96),
        (48, 64, 4, 2, 16, 96, 128),
        (48, 64, 4, 3, 24, 96, 128),
        (96, 128, 4, 3, 24, 96, 128),
    ]
    rows = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for (bh, bw, ng, cpg, sh, oh, ow) in grid_configs:
        row = run_one(
            net, seg,
            base_h=bh, base_w=bw, n_grids=ng, channels_per_grid=cpg,
            synth_hidden=sh, out_hw=(oh, ow), scorer_hw=(args.scorer_h, args.scorer_w),
            epochs=args.epochs, decoder_lr=args.decoder_lr, seed=0,
        )
        rows.append(row)
        cb = row["charged_bytes"]
        print(
            f"base={bh}x{bw} g{ng}c{cpg} h{sh}: "
            f"charged={cb['total_bytes']:.0f}B (lat {cb['latent_bytes']:.0f} "
            f"+ wt {cb['weight_bytes']:.0f}) latents={cb['latent_count']:.0f} "
            f"| d_seg {row['d_seg_initial']:.4f} -> live {row['d_seg_live']:.4f} "
            f"(ema {row['d_seg_best_ema']:.4f}) "
            f"({'DESC' if row['descended_live'] else 'flat'}) {row['wall_seconds']}s",
            flush=True,
        )
        # incremental write so a crash/kill is resumable-by-inspection.
        with open(args.out, "w") as f:
            json.dump(
                {
                    "lane": "lane_cool_chic_score_aware_basis_20260611",
                    "axis_tag": "[macOS-CPU advisory]",
                    "promotable": False,
                    "n_pairs": args.n_pairs,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    print("SWEEP DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
