# SPDX-License-Identifier: MIT
"""DECISIVE $0 de-risk: does COMPUTE-HEAVY weight-TIED decode break the
Cool-Chic 0.014 exact-d_seg capacity wall at MATCHED-or-fewer stored bytes?

THE OPERATOR INSIGHT (2026-06-11): "the compress time could TRAIN the inflate
time." A FIXED cheap decoder walls at d_seg ~0.014 (latent_heavy_isolation, the
REFUTED smaller-basis verdict). But compress-time can TRAIN a RICHER inflate-time
decoder whose effective capacity comes from inflate-time COMPUTE (shared weights
applied many times) rather than STORED params. The 30-min inflate budget is 95.5%
unspent, so heavy deterministic decode is affordable.

THE HYPOTHESIS UNDER TEST: a weight-TIED RECURRENT refinement synth (ONE small
conv block applied K times to iteratively refine the rendered frame) gets depth
(= effective capacity) from K, not from K-independent stored params. If d_seg
FALLS as K rises toward the 5.6e-4 basin at stored bytes <= the 0.014-wall arms,
the wall is COMPUTE-breakable -> de-risks a paid compute-heavy retrain. If d_seg
PLATEAUS at ~0.014 regardless of K, the wall is COMPUTE-INVARIANT -> compute does
NOT substitute for the missing representational structure (hardens the verdict).

REUSE (NO rebuild): CoolChicPairCarrier (latent grids + ARM rate + delta + byte
accounting + the reconstruct_pair trainer contract), ScoreAwareTrainer (PR95
live-SegNet loss + EMA-warmup B4-fix + exact d_seg + eval roundtrip), the n8/n48
GT cache, the exact frozen modules.py SegNet. ONLY the SYNTHESIS is overridden:
single-pass 1x1 conv -> weight-TIED recurrent refinement.

NO-FAKE: exact d_seg through the REAL SegNet (live AND EMA, agreement confirmed);
stored bytes counted honestly (shared block stored ONCE = the whole point);
decode is inflate-LEGAL (deterministic, GT-free, scorer-free, self-contained).
[macOS-CPU advisory] research-signal, NON-PROMOTABLE. The bar is the LOCAL d_seg
wall, NOT a contest score.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.residual_basis.cool_chic_carrier import (
    CoolChicGridSpec,
    CoolChicPairCarrier,
)
from tac.score_aware_loop.targets import load_frozen_distortion_net
from tac.score_aware_loop.trainer import ScoreAwareLoopConfig, ScoreAwareTrainer

CACHE_TMPL = "experiments/results/capstone_gt_targets_cache/gt_targets_n{n}.pt"


class WeightTiedRecurrentCarrier(CoolChicPairCarrier):
    """Cool-Chic carrier whose synthesis is a WEIGHT-TIED RECURRENT refinement.

    Capacity-from-compute test. The synthesis is replaced by:

        h = in_proj(feat)                      # (hidden, H, W)   -- 1x1, stored once
        for _ in range(K):                     # K shared passes  -- COMPUTE, not bytes
            h = h + block(h)                    # residual refine; block stored ONCE
        rgb = sigmoid(out_proj(h))             # (3, H, W)        -- 1x1, stored once

    ``block`` is a single small conv stack (3x3 depthwise-ish -> GELU -> 1x1)
    applied K times with the SAME weights (weight-tied / recurrent / unrolled
    fixed-point refinement). Stored params are INDEPENDENT of K -- the whole
    point: effective depth (capacity) scales with inflate-time compute K while
    stored bytes stay fixed and SMALL.

    Inflate-legality: every op is a deterministic forward pass of STORED weights
    on STORED latents. No GT, no scorer, self-contained. Projected inflate cost
    scales ~linearly in K (K conv passes per frame); reported below.

    The base CoolChicPairCarrier provides latent_grids / frame1_delta / ARM rate
    / charged_bytes / reconstruct_pair. We override _synth ONLY.
    """

    def __init__(
        self,
        n_pairs: int,
        spec: CoolChicGridSpec,
        *,
        synth_hidden: int = 16,
        n_passes: int = 4,
        block_kernel: int = 3,
        out_hw: tuple[int, int] = (96, 128),
        bytes_per_param: float = 2.0,
        quant_step: float = 1.0,
    ) -> None:
        super().__init__(
            n_pairs,
            spec,
            synth_hidden=synth_hidden,
            out_hw=out_hw,
            bytes_per_param=bytes_per_param,
            quant_step=quant_step,
            pose_film_enabled=False,
        )
        # REMOVE the base single-pass synth params from the optimizer/byte path:
        # we replace _synth entirely, so w1/w2/b1/b2 are unused. Re-purpose w1/b1
        # as the in_proj (c_in -> hidden) and w2/b2 as out_proj (hidden -> 3) so
        # the base byte accounting (weight_param_count) stays HONEST and counts
        # them. (They ARE used by the new _synth.)
        self.n_passes = int(n_passes)
        self.block_kernel = int(block_kernel)
        c_in = spec.c_in
        H = synth_hidden
        # in_proj = base w1/b1 (c_in -> hidden), out_proj = base w2/b2 (hidden -> 3)
        # are reused AS-IS (already (hidden,c_in) and (3,hidden)). The new tied
        # refinement block lives in dedicated params counted by weight_param_count
        # override below.
        k = self.block_kernel
        pad = k // 2
        # Shared (weight-tied) refinement block: 3x3 conv (H->H) -> GELU -> 1x1
        # conv (H->H). Stored ONCE; applied n_passes times.
        self.tied_conv1_w = nn.Parameter(
            torch.randn(H, H, k, k) * (1.0 / math.sqrt(H * k * k))
        )
        self.tied_conv1_b = nn.Parameter(torch.zeros(H))
        self.tied_conv2_w = nn.Parameter(torch.randn(H, H, 1, 1) * (1.0 / math.sqrt(H)))
        self.tied_conv2_b = nn.Parameter(torch.zeros(H))
        self._tied_pad = pad

    def set_n_passes(self, k: int) -> None:
        self.n_passes = int(k)

    def _tied_block(self, h: torch.Tensor) -> torch.Tensor:
        """One shared refinement pass: residual 3x3 conv -> GELU -> 1x1 conv.

        ``h`` is ``(1, H, hh, ww)``. Returns the refined ``h`` (same shape).
        """
        r = F.conv2d(h, self.tied_conv1_w, self.tied_conv1_b, padding=self._tied_pad)
        r = F.gelu(r)
        r = F.conv2d(r, self.tied_conv2_w, self.tied_conv2_b)
        return h + r  # residual fixed-point refinement

    def _synth(
        self,
        feat: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """in_proj -> [K tied residual refinement passes] -> out_proj -> sigmoid.

        ``feat`` is ``(c_in, H, W)``; returns ``(3, H, W)`` in ``[0, 1]``.
        Weight-tied: the SAME block runs ``self.n_passes`` times (compute, not
        bytes). ``film`` is ignored (pose-fold disabled for this test).
        """
        c_in, hh, ww = feat.shape
        flat = feat.reshape(c_in, hh * ww)
        # in_proj (reuse base w1/b1): (hidden, c_in) @ (c_in, N) -> (hidden, N)
        hidden = self.w1 @ flat + self.b1[:, None]
        h = hidden.reshape(1, self.synth_hidden, hh, ww)
        for _ in range(self.n_passes):
            h = self._tied_block(h)
        hf = h.reshape(self.synth_hidden, hh * ww)
        out = self.w2 @ hf + self.b2[:, None]  # out_proj (hidden -> 3)
        return torch.sigmoid(out).reshape(3, hh, ww)

    def weight_param_count(self) -> int:
        """Honest stored-param count: base synth/ARM/delta_proj + the TIED block.

        The tied block is stored ONCE regardless of n_passes -- so stored bytes
        are K-INDEPENDENT. This is the capacity-from-compute claim made HONEST.
        """
        base = super().weight_param_count()
        tied = (
            self.tied_conv1_w.numel()
            + self.tied_conv1_b.numel()
            + self.tied_conv2_w.numel()
            + self.tied_conv2_b.numel()
        )
        return int(base + tied)


def run_arm(
    net,
    seg,
    pose,
    *,
    label,
    seg_loss_form,
    base_h,
    base_w,
    n_grids,
    cpg,
    synth_hidden,
    n_passes,
    out_hw,
    scorer_hw,
    epochs,
    decoder_lr,
    seed,
):
    torch.manual_seed(seed)
    n = int(seg.shape[0])
    spec = CoolChicGridSpec(
        base_h=base_h, base_w=base_w, n_grids=n_grids, channels_per_grid=cpg
    )
    carrier = WeightTiedRecurrentCarrier(
        n_pairs=n,
        spec=spec,
        synth_hidden=synth_hidden,
        n_passes=n_passes,
        out_hw=out_hw,
    )
    cfg = ScoreAwareLoopConfig(
        epochs=epochs,
        batch_size=n,
        scorer_hw=scorer_hw,
        pose_enabled=pose is not None,
        eval_every=max(epochs // 8, 1),
        seg_loss_form=seg_loss_form,
        decoder_lr=decoder_lr,
        latent_lr_mult=10.0,
        ema_decay=0.99,
        seed=seed,
    )
    tr = ScoreAwareTrainer(carrier, net, seg, pose, cfg)

    # NO-FAKE gradient-flow guard: the TIED block AND a latent grid must receive
    # gradient (proves the K-pass refinement is actually in the graph).
    idx = torch.arange(n)
    loss_parts = tr.compute_loss(idx)
    loss_parts["total"].backward()
    grad_tied = (
        float(carrier.tied_conv1_w.grad.abs().sum())
        if carrier.tied_conv1_w.grad is not None
        else 0.0
    )
    grad_grid = (
        float(carrier.latent_grids[0].grad.abs().sum())
        if carrier.latent_grids[0].grad is not None
        else 0.0
    )
    carrier.zero_grad(set_to_none=True)

    bytes_init = carrier.charged_bytes()
    t0 = time.time()
    res = tr.train()
    wall = time.time() - t0
    d_seg_ema = tr.exact_d_seg(use_ema=True)
    d_seg_live = tr.exact_d_seg(use_ema=False)  # EMA-lag guard
    bytes_post = carrier.charged_bytes()

    # Inflate-cost projection: K conv passes per rendered frame. Time one render
    # of the full batch (both frames) as the per-K cost proxy, then project to
    # 600 contest pairs (the inflate budget is 30 min = 1800 s for 600 pairs).
    with torch.no_grad():
        carrier.eval()
        tr0 = time.time()
        for _ in range(3):
            carrier.reconstruct_pair(torch.arange(min(n, 8)))
        render_wall = (time.time() - tr0) / 3.0
    per_pair_render_s = render_wall / max(min(n, 8), 1)
    projected_600_inflate_s = per_pair_render_s * 600.0

    return {
        "label": label,
        "n_passes": n_passes,
        "seg_loss_form": seg_loss_form,
        "config": {
            "base_hw": [base_h, base_w],
            "n_grids": n_grids,
            "channels_per_grid": cpg,
            "synth_hidden": synth_hidden,
            "n_passes": n_passes,
            "out_hw": list(out_hw),
            "scorer_hw": list(scorer_hw),
            "epochs": epochs,
            "decoder_lr": decoder_lr,
        },
        "charged_bytes_init": bytes_init,
        "charged_bytes_post": bytes_post,
        "stored_weight_param_count": carrier.weight_param_count(),
        "stored_weight_bytes": carrier.weight_bytes(),
        "grad_tied_block_abs_sum_first_step": grad_tied,
        "grad_grid0_abs_sum_first_step": grad_grid,
        "d_seg_initial": res["d_seg_initial"],
        "d_seg_best_ema": res["d_seg_best_ema"],
        "d_seg_final_ema_exact": d_seg_ema,
        "d_seg_final_live_exact": d_seg_live,
        "ema_live_agreement_abs": abs(d_seg_ema - d_seg_live),
        "descended": res["descended"],
        "trajectory": res["trajectory"],
        "train_wall_seconds": round(wall, 1),
        "per_pair_render_seconds": round(per_pair_render_s, 5),
        "projected_600pair_inflate_seconds": round(projected_600_inflate_s, 1),
        "inflate_legal_under_30min": projected_600_inflate_s <= 1800.0,
        "axis_tag": "[macOS-CPU advisory]",
        "promotable": False,
        "score_claim": False,
    }


def _p(r):
    cbp = r["charged_bytes_post"]
    print(
        f"{r['label']} K={r['n_passes']:>2}: d_seg init={r['d_seg_initial']:.4f} "
        f"best_ema={r['d_seg_best_ema']:.4f} final_ema={r['d_seg_final_ema_exact']:.4f} "
        f"live={r['d_seg_final_live_exact']:.4f} (agree {r['ema_live_agreement_abs']:.1e}) | "
        f"grad_tied={r['grad_tied_block_abs_sum_first_step']:.2e} | "
        f"STORED w={r['stored_weight_bytes']:.0f}B params={r['stored_weight_param_count']} "
        f"latent={cbp['latent_bytes']:.0f}B total={cbp['total_bytes']:.0f}B | "
        f"inflate~{r['projected_600pair_inflate_seconds']:.0f}s "
        f"(legal={r['inflate_legal_under_30min']}) | {r['train_wall_seconds']}s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--scorer-h", type=int, default=192)
    ap.add_argument("--scorer-w", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--synth-hidden", type=int, default=16)
    ap.add_argument(
        "--k-sweep",
        type=str,
        default="1,2,4,8,16",
        help="comma-separated n_passes values to sweep",
    )
    ap.add_argument(
        "--out",
        default="experiments/results/lane_compute_heavy_weight_tied_decoder_20260611/k_sweep.json",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    n = args.n_pairs
    cache = CACHE_TMPL.format(n=n)
    d = torch.load(cache, map_location="cpu", weights_only=False)
    seg = d["seg"][:n]
    pose = d["pose"][:n].float()
    net = load_frozen_distortion_net(device="cpu")
    scorer_hw = (args.scorer_h, args.scorer_w)
    HINGE = "hard_pixel_curriculum_seg_loss"  # the default hinge curriculum
    OHW = (96, 128)

    ks = [int(x) for x in args.k_sweep.split(",") if x.strip()]
    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush():
        with open(out_path, "w") as f:
            json.dump(
                {
                    "lane": "lane_compute_heavy_weight_tied_decoder_20260611",
                    "test": "compute_heavy_weight_tied_recurrent_decode_vs_cool_chic_0p014_wall",
                    "status": "partial" if len(rows) < len(ks) else "complete",
                    "axis_tag": "[macOS-CPU advisory]",
                    "promotable": False,
                    "score_claim": False,
                    "n_pairs": n,
                    "epochs": args.epochs,
                    "cool_chic_0p014_wall_reference": 0.0140,
                    "cool_chic_L2_stored_weight_bytes_reference": 1290,
                    "basin_d_seg_reference": 5.6e-4,
                    "corrected_bar_d_seg": [0.0011, 0.0017],
                    "k_sweep": ks,
                    "rows": rows,
                },
                f,
                indent=2,
            )
    # Capacity-from-compute sweep: HOLD the carrier fixed (modest grids + modest
    # synth_hidden so stored bytes <= the Cool-Chic 0.014-wall arms), sweep ONLY
    # the number of inflate-time refinement passes K. Does d_seg FALL with K
    # (capacity-from-compute) or PLATEAU at 0.014 (compute-invariant wall)?
    # Grids 6x4 @ base 40x56 matches the L2 arm (the 0.0140 wall point) so the
    # stored-byte comparison is apples-to-apples.
    for k in ks:
        rows.append(
            run_arm(
                net,
                seg,
                pose,
                label="WTR_recurrent",
                seg_loss_form=HINGE,
                base_h=40,
                base_w=56,
                n_grids=6,
                cpg=4,
                synth_hidden=args.synth_hidden,
                n_passes=k,
                out_hw=OHW,
                scorer_hw=scorer_hw,
                epochs=args.epochs,
                decoder_lr=3e-3,
                seed=0,
            )
        )
        _p(rows[-1])
        _flush()  # incremental durable write -> partial runs are usable

    _flush()
    print("COMPUTE-HEAVY WEIGHT-TIED K-SWEEP DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
