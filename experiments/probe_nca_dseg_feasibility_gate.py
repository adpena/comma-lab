#!/usr/bin/env python3
"""GENERATIVE-AXIS GATE — can a BYTE-CHEAP Neural Cellular Automata d_seg-core reach sub-0.15 d_seg?

THE one representational family the sub-0.15 campaign has NOT tested: store the GENERATOR (a tiny
shared local update rule + a tiny seed), REPLAY (iterate N steps) to reconstruct the SegNet-decision
partition. Kolmogorov/MDL-optimal axis (Schmidhuber): the shortest description of a complex pattern is
the shortest PROGRAM that outputs it. Every prior family stored a STATIC description (decoder weights /
polygon control points / partition stores) and hit the capacity<->rate wall; ITERATION generates detail
for free (temporal weight-sharing = effective depth N with the SAME rule params).

THE TWO WALLS THIS GATE MUST BEAT (measured by the sister gates this campaign):
  - factored RANK-1 LF (learned pixel decoder): RED, d_seg ~ 29.3*params^-0.71 CAPACITY wall;
  - curve-core (static-stored geometry): the SURVIVAL wall — geometry matches L* (geo_recon->0.002)
    but realized d_seg PLATEAUS ~0.007 because the camera-res->384 bilinear downsample mixes the 1px
    boundary band -> SegNet argmax flips 17-26% of boundary px regardless of geometry.

THE BET (narrow, falsifiable): the NCA's temporal weight-sharing reaches comparable realized d_seg at
FAR fewer bytes (a few KB rule vs the 270-650 control points the curve core needed) — the genuine
CAPACITY escape. It still INHERITS the survival wall (it renders a frame through the SAME roundtrip +
SAME SegNet), with the same survival lever the curve core has (fit colours/boundary-band THROUGH the
roundtrip so gradients pre-compensate mixing). The gate measures BOTH: is the capacity escape real, AND
does the combination cross S < 0.15 or does it hit the survival floor?

THE GATE:
  Is there an NCA rule-size that is BYTE-CHEAP (rate << 0.05) AND grows a partition that, THROUGH the
  real SegNet + exact uint8 roundtrip, gives realized d_seg low enough that
  100*d_seg + sqrt(10*d_pose) + 25*B/B0 < 0.15?
    GREEN -> the generative axis escapes BOTH walls -> THE sub-0.15 path -> spec the build.
    RED   -> the generative axis ALSO caps. If geometric matches but realized doesn't -> SURVIVAL is
             the TERMINAL wall across ALL families (airtight finding). If geometric itself is high ->
             the NCA's capacity escape is not real for this boundary.

WHY THIS IS THE RIGHT, FAITHFUL TEST (NO-FAKE, measurement-first):
  - REUSES the curve gate's EXACT eval roundtrip, realized-d_seg measurement, GT L* targets, byte/rate
    model (NCA param counts), and GREEN/RED/AMBER verdict structure -> apples-to-apples with the curve
    gate's survival number. Only the REPRESENTATION (NCA generator) is swapped.
  - Real frozen contest SegNet (CPU AUTHORITY; NEVER MPS for the score). Real GT argmax L* (the `seg`
    field of the capstone GT cache = SegNet argmax on GT frame1, 384x512).
  - MEASUREMENT-FIRST verdict: the NCA-FIT surrogate loss (CE) dropping is NOT the verdict; the realized
    d_seg of the HARD argmax-decoded frame THROUGH the real SegNet + roundtrip IS. The factored-LF
    gate's degenerate-fit false-GREEN is the cautionary anchor: guard against false-GREEN AND false-RED.

CARMACK MVP-FIRST: dominant class-pair option, small n_frames (3), modest N (48-64 iters) -> a
feasibility ESTIMATE, not a final train. $0, MPS fp32 gradient + CPU authority, no paid GPU, no PR,
no self-promote. Resumable: per-rule-size JSON checkpoint.

ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(UPSTREAM))

GT_TARGETS = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n16.pt"

# --- the campaign's measured anchors (advisory, for the GREEN/RED thresholds) ----
FRONTIER_DSEG = 0.00257  # the frontier's own d_seg (the bar to beat on the d_seg axis)
SUB015_DSEG = 0.0006  # ~sub-0.15-grade d_seg (the target the generator must approach)
PARTITION_STORE_REALIZED_DSEG = 0.00641  # the static-store survival wall (mu-optimized)
CURVE_CORE_SURVIVAL_FLOOR = 0.0071  # the curve gate's plateau (mp32 realized d_seg) — the survival wall
GREEN_DSEG_THRESHOLD = 0.0012  # decisively past the frontier d_seg, heading sub-0.15
B0 = 37_545_489  # contest archive normalizer
HELD_POSE = 0.00034  # frontier trunk d_pose (held; the generator targets seg only)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# REUSE the curve gate's exact eval chain (import to guarantee apples-to-apples)
# ===========================================================================
# The curve gate is a sibling module in experiments/; add this dir to the path so the
# reuse works whether run as `python experiments/probe_nca_*.py` or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_curve_core_dseg_feasibility_gate import (
    _eval_roundtrip_t,
    _segnet_argmax_of_frame,
    _segnet_logits_of_frame,
    class_canonical_colors,
    rate_from_total_bytes,
)


# ===========================================================================
# Byte cost of the NCA rule (the GENERATOR = the stored bytes)
# ===========================================================================
def nca_param_bytes(rule_param_count, seed_grid_cells, n_classes=5, weight_bits=8):
    """Advisory byte cost of the quantized NCA GENERATOR.

    The stored representation is the RULE (the only learned weights) + a tiny seed grid. This is the
    Kolmogorov-cheap part: a few-KB program, not a many-KB static description.

    - rule weights: rule_param_count * weight_bits, entropy-coded (0.55 factor, the same advisory
      dense-raster LZMA-family factor the curve gate used) -> apples-to-apples rate.
    - seed: seed_grid_cells * ceil(log2(n_classes)) bits (a coarse class-init grid), entropy-coded.
    The rule is QUASI-STATIC across 600 frames (one rule for the whole clip); only the seed gets a tiny
    per-frame ego-motion delta (the contour barely moves: geometric-solve identity residual 0.33px).
    We report BOTH the per-frame full cost AND the quasi-static amortized cost (rule once + small
    per-frame seed delta), mirroring the curve gate's amortization model exactly.
    """
    packed_factor = 0.55  # advisory entropy-coding factor (same family as the curve gate)
    rule_bytes = rule_param_count * weight_bits * packed_factor / 8.0
    seed_bits_per_cell = max(1, math.ceil(math.log2(max(2, n_classes))))
    seed_bytes_full = seed_grid_cells * seed_bits_per_cell * packed_factor / 8.0
    # the rule is stored ONCE for all 600 frames; the seed gets a tiny per-frame delta (~10% of seed).
    delta_frac = 0.10
    total_600_full = (rule_bytes + seed_bytes_full) * 600.0  # naive per-frame (NOT the real model)
    total_600_amortized = rule_bytes + seed_bytes_full + (seed_bytes_full * delta_frac) * 600.0
    return {
        "rule_param_count": int(rule_param_count),
        "rule_bytes": rule_bytes,
        "seed_grid_cells": int(seed_grid_cells),
        "seed_bytes_full": seed_bytes_full,
        "total_600_full_bytes": total_600_full,
        "total_600_amortized_bytes": total_600_amortized,
    }


# ===========================================================================
# The Neural Cellular Automata generator (the REPRESENTATION under test)
# ===========================================================================
class NCAGenerator:
    """A tiny shared local update rule iterated N steps to GROW per-class logits.

    State: (1, C, H, W) grid. Perception: fixed depthwise (identity + Sobel-x + Sobel-y) per channel
    -> 3C perception channels (NO learned params — the classic Growing-NCA perception). Update rule:
    1x1 conv 3C->hidden (ReLU) -> 1x1 conv hidden->C (zero-init, the only stochastic-update gate). The
    LEARNED params are the two 1x1 convs (the rule). After N steps, a fixed linear readout (C->n_classes,
    learned) maps the final state to per-class logits -> softmax -> differentiable per-class-colour render.

    Stored bytes = the two 1x1 conv weights + the readout (the RULE) + the seed. n_steps does NOT add
    bytes (temporal weight-sharing) — that IS the capacity escape under test.
    """

    def __init__(self, n_channels, hidden, n_classes, shape, seed_label, device):
        import torch

        self.C = n_channels
        self.hidden = hidden
        self.n_classes = n_classes
        self.shape = shape
        self.device = device
        H, W = shape

        # --- fixed perception kernels (identity + Sobel x/y), depthwise, NO learned params ----
        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        sob_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sob_y = sob_x.t().contiguous()
        # depthwise weight (3C, 1, 3, 3): for each channel, 3 perception filters
        per = torch.stack([ident, sob_x, sob_y], dim=0)  # (3,3,3)
        w = per.repeat(self.C, 1, 1).unsqueeze(1)  # (3C,1,3,3)
        self.perception_w = w.to(device)  # fixed

        # --- learned rule: 1x1 conv 3C->hidden (ReLU), 1x1 conv hidden->C (zero-init) ----
        g = torch.Generator().manual_seed(1234)
        self.w1 = torch.nn.Parameter(
            (torch.randn(hidden, 3 * self.C, 1, 1, generator=g) * (1.0 / math.sqrt(3 * self.C))).to(
                device
            )
        )
        self.b1 = torch.nn.Parameter(torch.zeros(hidden, device=device))
        # zero-init the update conv (classic NCA: start as a no-op, grow gradually)
        self.w2 = torch.nn.Parameter(torch.zeros(self.C, hidden, 1, 1, device=device))
        # --- learned readout: final state C -> n_classes logits (1x1 conv) ----
        self.readout = torch.nn.Parameter(
            (torch.randn(n_classes, self.C, 1, 1, generator=g) * (1.0 / math.sqrt(self.C))).to(device)
        )
        self.readout_b = torch.nn.Parameter(torch.zeros(n_classes, device=device))

        # --- seed: a coarse class-init upsampled to (H,W), written into the first n_classes channels.
        # This is the tiny stored seed grid (a downsampled L* class map). It gives the NCA a coarse
        # anchor to grow from (the openpilot-style coarse partition); the rule refines the boundary.
        self.seed = self._build_seed(seed_label, device)  # (1,C,H,W) initial state

    def _build_seed(self, seed_label, device):
        import torch
        import torch.nn.functional as F

        H, W = self.shape
        # seed_label: (h_s, w_s) int coarse class map (already downsampled by the caller)
        sl = torch.tensor(seed_label, dtype=torch.long, device=device)
        hs, ws = sl.shape
        onehot = F.one_hot(sl, num_classes=self.n_classes).float().permute(2, 0, 1).unsqueeze(0)
        up = F.interpolate(onehot, size=(H, W), mode="nearest")  # (1,n_classes,H,W)
        state = torch.zeros(1, self.C, H, W, device=device)
        kc = min(self.n_classes, self.C)
        state[:, :kc] = up[:, :kc]
        return state

    def params(self):
        return [self.w1, self.b1, self.w2, self.readout, self.readout_b]

    def rule_param_count(self):
        # the STORED rule + readout (perception + seed are not learned rule weights;
        # seed bytes are counted separately in nca_param_bytes).
        n = self.w1.numel() + self.b1.numel() + self.w2.numel()
        n += self.readout.numel() + self.readout_b.numel()
        return int(n)

    def grow_logits(self, n_steps):
        """Iterate the rule n_steps from the seed, return per-class logits (1,n_classes,H,W)."""
        import torch.nn.functional as F

        x = self.seed
        for _ in range(n_steps):
            per = F.conv2d(x, self.perception_w, padding=1, groups=self.C)  # (1,3C,H,W) fixed
            h = F.relu(F.conv2d(per, self.w1, self.b1))  # (1,hidden,H,W)
            dx = F.conv2d(h, self.w2)  # (1,C,H,W) zero-init update
            x = x + dx
        logits = F.conv2d(x, self.readout, self.readout_b)  # (1,n_classes,H,W)
        return logits


def downsample_label(L, factor):
    """Downsample a label map (H,W) to (H//factor, W//factor) by majority vote (the coarse seed)."""
    import numpy as np

    H, W = L.shape
    hs, ws = H // factor, W // factor
    out = np.zeros((hs, ws), dtype=np.int64)
    for i in range(hs):
        for j in range(ws):
            blk = L[i * factor : (i + 1) * factor, j * factor : (j + 1) * factor]
            vals, cnts = np.unique(blk, return_counts=True)
            out[i, j] = int(vals[int(np.argmax(cnts))])
    return out


# ===========================================================================
# Optimize ONE NCA rule-size THROUGH the real SegNet + roundtrip
# ===========================================================================
def optimize_nca_core(L, frame_rgb, segnet_cpu, segnet_train, hidden, n_channels, n_steps,
                      seed_factor, args, dominant_only):
    """Two-stage NCA-core feasibility measurement at a given rule size (hidden width).

    STAGE 1 (capacity check): grow the partition from the rule (no roundtrip), measure geometric d_seg =
    SegNet argmax of the rendered grown frame vs L*. The rule param count is the byte driver.

    STAGE 2 (survival check): make the per-class fill colours + boundary-band offset + the RULE itself
    differentiable THROUGH the real SegNet + exact uint8 roundtrip; optimize CE-vs-L*. Then measure
    REALIZED d_seg = argmax-flip-rate of the HARD (argmax-decoded, then no-STE roundtrip) frame.

    Returns geometric d_seg, realized d_seg, boundary-band flip, rule bytes, S projection.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    H, W = L.shape
    tdev = torch.device(args.train_device)
    n_classes = 5

    # ---- coarse seed (the tiny stored init grid) from a downsampled L* ----
    seed_label = downsample_label(L, seed_factor)  # (H//f, W//f)
    seed_grid_cells = seed_label.size

    gen = NCAGenerator(n_channels, hidden, n_classes, (H, W), seed_label, tdev)

    # ---- learnable per-class colours (+ boundary-band offset), the survival lever ----
    init_colors = class_canonical_colors(L, frame_rgb)  # (n_classes,3)
    colors = torch.nn.Parameter(torch.tensor(init_colors, dtype=torch.float32, device=tdev))
    # boundary band of L* (where survival flips concentrate) -> per-class band offset
    bmask = np.zeros((H, W), dtype=bool)
    bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
    bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
    bmask[:-1, :] |= L[:-1, :] != L[1:, :]
    bmask[1:, :] |= L[:-1, :] != L[1:, :]
    from scipy import ndimage

    band_np = ndimage.binary_dilation(bmask, iterations=2)
    band_t = torch.tensor(band_np, dtype=torch.float32, device=tdev)  # (H,W)
    band_offset = torch.nn.Parameter(torch.zeros((n_classes, 3), dtype=torch.float32, device=tdev))

    Lt = torch.tensor(L, dtype=torch.long, device=tdev)
    params = [*gen.params(), colors, band_offset]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    def render_frame_from_logits(logits, hard=False, tau=None):
        """Per-class soft (or hard) probs -> differentiable RGB frame (3,H,W).

        The soft render uses an ANNEALED temperature so it converges to the hard argmax render used
        at measurement — this kills the train/eval render mismatch (a constant tau=1.0 trains on a
        BLENDED frame the SegNet never sees at measurement, producing false interior flips).
        """
        if hard:
            cls = logits.argmax(dim=1)  # (1,H,W)
            probs = F.one_hot(cls, num_classes=n_classes).float().permute(0, 3, 1, 2)  # (1,C,H,W)
        else:
            t = args.render_tau if tau is None else tau
            probs = F.softmax(logits / max(t, 1e-3), dim=1)  # (1,n_classes,H,W)
        # base colour: sum_c probs_c * colour_c -> (1,3,H,W)
        base = torch.einsum("bchw,cd->bdhw", probs, colors)
        off = torch.einsum("bchw,cd->bdhw", probs, band_offset)
        rgb = base + band_t.unsqueeze(0).unsqueeze(0) * off
        return rgb[0].clamp(0, 255)  # (3,H,W)

    t0 = time.time()
    losses = []
    base_lr = args.lr
    warmup = max(1, int(0.05 * args.iters))  # short LR warmup (stabilizes the deep unroll)
    for _it in range(args.iters):
        # LR warmup then hold (NCA over N unrolled steps is sensitive to early large steps).
        lr = base_lr * min(1.0, (_it + 1) / warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        # anneal render temperature tau_start -> tau_end (soft -> near-hard) across training so the
        # differentiable soft render converges to the hard argmax render used at measurement.
        frac = _it / max(1, args.iters - 1)
        cur_tau = args.render_tau * (args.render_tau_end / args.render_tau) ** frac
        logits = gen.grow_logits(n_steps)  # (1,n_classes,H,W) — the GENERATED partition
        # primary objective: the realized partition through the roundtrip + real SegNet matches L*.
        frame = render_frame_from_logits(logits, hard=False, tau=cur_tau)  # soft render (diff)
        rt = _eval_roundtrip_t(frame, ste=True)[0]  # exact roundtrip (STE round)
        seg_logits = _segnet_logits_of_frame(segnet_train, rt)  # (1,C,H,W) real SegNet
        ce_seg = F.cross_entropy(seg_logits, Lt.unsqueeze(0))
        # auxiliary: the NCA's OWN logits should also match L* (anchors the generator to grow the
        # right partition; weight small so the SegNet+roundtrip objective dominates the verdict).
        ce_gen = F.cross_entropy(logits, Lt.unsqueeze(0))
        loss = ce_seg + args.gen_anchor_w * ce_gen
        loss.backward()
        # CANONICAL Growing-NCA stability trick (Mordvintsev): per-parameter gradient NORMALIZATION
        # (grad / (||grad|| + eps)) — the deep N-step unroll otherwise explodes/vanishes. This is the
        # decisive fix for the false-RED divergence (lr=4e-2 collapsed to a constant without it).
        if args.grad_norm:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(p.grad.norm() + 1e-8)
        else:
            torch.nn.utils.clip_grad_norm_(params, 50.0)
        opt.step()
        losses.append((float(ce_seg.item()), float(ce_gen.item())))

    # ---- MEASUREMENT-FIRST: realized d_seg of the HARD frame through the EXACT chain ----
    with torch.no_grad():
        logits = gen.grow_logits(n_steps)
        # geometric_dseg_recon = the NCA's OWN argmax partition vs L* (pure capacity check)
        gen_argmax = logits.argmax(dim=1)[0].cpu().numpy()
        geo_dseg_recon = float((gen_argmax != L).mean())
        # hard render (argmax-decoded class colours) -> the frame the inflate path would emit
        hard = render_frame_from_logits(logits, hard=True).cpu()  # (3,H,W)
        # (a) geometric (no roundtrip): the painted frame's SegNet argmax vs L*
        geo_argmax = _segnet_argmax_of_frame(segnet_cpu, hard).cpu().numpy()
        geo_dseg = float((geo_argmax != L).mean())
        # (b) REALIZED d_seg: through the EXACT uint8 roundtrip (the survival check)
        rt_hard = _eval_roundtrip_t(hard, ste=False)[0]
        real_argmax = _segnet_argmax_of_frame(segnet_cpu, rt_hard).cpu().numpy()
        real_dseg = float((real_argmax != L).mean())
        # boundary-band realized flip (the wall the static store hit at ~24%, curve core ~17-26%)
        band1 = ndimage.binary_dilation(bmask, iterations=1)
        boundary_flip = (
            float((real_argmax[band1] != L[band1]).mean()) if band1.any() else float("nan")
        )
        interior = ~band1
        interior_flip = (
            float((real_argmax[interior] != L[interior]).mean())
            if interior.any()
            else float("nan")
        )

    rule_pc = gen.rule_param_count()
    bytes_info = nca_param_bytes(rule_pc, seed_grid_cells)
    rate_amort = rate_from_total_bytes(bytes_info["total_600_amortized_bytes"])
    rate_full = rate_from_total_bytes(bytes_info["total_600_full_bytes"])
    s_amort = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_amort
    s_full = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_full

    return {
        "hidden": int(hidden),
        "n_channels": int(n_channels),
        "n_steps": int(n_steps),
        "seed_factor": int(seed_factor),
        "dominant_only": bool(dominant_only),
        "rule_param_count": rule_pc,
        "seed_grid_cells": int(seed_grid_cells),
        # geometric_dseg_recon = the NCA's OWN argmax partition vs L* (capacity check: can the rule
        # GROW a partition matching L* at all, ignoring SegNet + roundtrip?).
        "geometric_dseg_recon": geo_dseg_recon,
        # geometric_dseg = SegNet argmax of the rendered frame WITHOUT roundtrip vs L*.
        "geometric_dseg": geo_dseg,
        # realized_dseg = THROUGH the exact uint8 roundtrip (the survival-checked AUTHORITY number).
        "realized_dseg": real_dseg,
        "boundary_band_flip": boundary_flip,
        "interior_flip": interior_flip,
        "final_ce_seg": losses[-1][0] if losses else None,
        "final_ce_gen": losses[-1][1] if losses else None,
        "ce_seg_first": losses[0][0] if losses else None,
        "bytes": bytes_info,
        "rate_amortized": rate_amort,
        "rate_full": rate_full,
        "S_projected_amortized": s_amort,
        "S_projected_full": s_full,
        "realized_dseg_x_frontier": real_dseg / FRONTIER_DSEG,
        "realized_dseg_x_curve_survival_floor": real_dseg / CURVE_CORE_SURVIVAL_FLOOR,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default=str(REPO / "experiments/results/nca_dseg_feasibility_gate")
    )
    ap.add_argument(
        "--hiddens",
        default="32,64,128,256",
        help="comma-sep NCA hidden widths (the rule-size = byte sweep)",
    )
    ap.add_argument("--n-channels", type=int, default=16, help="NCA state channels")
    ap.add_argument("--n-steps", type=int, default=64, help="NCA iteration steps (free depth)")
    ap.add_argument("--seed-factor", type=int, default=16, help="seed downsample factor (coarse init)")
    ap.add_argument("--n-frames", type=int, default=3, help="GT frames to measure over")
    ap.add_argument("--iters", type=int, default=300, help="diff-fit iters per rule-size")
    ap.add_argument("--lr", type=float, default=2e-2, help="AdamW lr")
    ap.add_argument("--render-tau", type=float, default=1.0, help="START softmax temp for soft render")
    ap.add_argument(
        "--render-tau-end", type=float, default=0.15,
        help="END softmax temp (annealed soft->near-hard to match the measurement hard render)",
    )
    ap.add_argument("--gen-anchor-w", type=float, default=0.3, help="weight on the gen-own-logits CE")
    ap.add_argument(
        "--grad-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="canonical Growing-NCA per-param gradient normalization (stabilizes the deep N-step unroll)",
    )
    ap.add_argument(
        "--dominant-only",
        action="store_true",
        help="frame the byte-attribution to the dominant class-pair (metric still vs full L*)",
    )
    ap.add_argument("--train-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument(
        "--timing-smoke",
        action="store_true",
        help="single hidden (64), 1 frame, 30 iters to calibrate s/iter then exit",
    )
    args = ap.parse_args()

    import torch

    if args.train_device == "mps":
        try:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patch_scorer_for_mps()
        except Exception as e:  # pragma: no cover
            print(f"[warn] patch_scorer_for_mps failed ({e}); continuing", flush=True)

    from tac.boundary_math.seg_core import decode_gt_frame1_pairs
    from tac.scorer import load_default_segnet

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "gate_state.json"
    result_json = REPO / ".omx/research" / f"nca_dseg_feasibility_gate_{_now()}.json"

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior rows: {list(state.get('rows', {}).keys())}", flush=True)
    state.setdefault("rows", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")  # AUTHORITY
    segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)

    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"].numpy()  # (16,384,512) int64 = L*
    n_avail = seg_targets.shape[0]
    n_frames = min(args.n_frames, n_avail)

    # decode matching GT RGB frames (for per-class mu colours) via the EXACT path
    import torch.nn.functional as F

    gt_frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        t = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0)
        rs = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_frames[pidx] = rs[0].permute(1, 2, 0).numpy()
        if len(gt_frames) >= n_frames:
            break

    if args.timing_smoke:
        print("[timing-smoke] hidden=64, 1 frame, 30 iters ...", flush=True)
        sa = argparse.Namespace(**vars(args))
        sa.iters = 30
        L = seg_targets[0]
        fr = gt_frames.get(0)
        r = optimize_nca_core(
            L, fr, segnet_cpu, segnet_train, 64, args.n_channels, args.n_steps,
            args.seed_factor, sa, args.dominant_only,
        )
        if r:
            spi = r["elapsed_s"] / sa.iters
            n_h = len(args.hiddens.split(","))
            print(
                f"[timing-smoke] {r['elapsed_s']:.0f}s/30it -> ~{spi:.2f}s/it ; "
                f"full {args.iters}it x {n_h}h x {n_frames}fr "
                f"~ {spi*args.iters*n_h*n_frames/60:.1f} min",
                flush=True,
            )
            print(
                f"   geo_recon={r['geometric_dseg_recon']:.5f} geo_seg={r['geometric_dseg']:.5f} "
                f"realized={r['realized_dseg']:.5f} bnd_flip={r['boundary_band_flip']:.3f} "
                f"rule_pc={r['rule_param_count']} rate={r['rate_amortized']:.5f} "
                f"S~{r['S_projected_amortized']:.3f}",
                flush=True,
            )
        return 0

    hiddens = [int(x) for x in args.hiddens.split(",") if x.strip()]

    for hid in hiddens:
        key = f"h{hid}"
        if key in state["rows"]:
            print(f"[resume] {key} already done; skip", flush=True)
            continue
        print(f"\n=== NCA RULE-SIZE {key} (hidden={hid}, C={args.n_channels}, "
              f"N={args.n_steps}) ===", flush=True)
        per_frame = []
        for fidx in range(n_frames):
            L = seg_targets[fidx]
            fr = gt_frames.get(fidx)
            r = optimize_nca_core(
                L, fr, segnet_cpu, segnet_train, hid, args.n_channels, args.n_steps,
                args.seed_factor, args, args.dominant_only,
            )
            if r is None:
                continue
            per_frame.append(r)
            print(
                f"   frame{fidx}: rule_pc={r['rule_param_count']:6d} "
                f"geo_recon={r['geometric_dseg_recon']:.5f} geo_seg={r['geometric_dseg']:.5f} "
                f"realized={r['realized_dseg']:.5f} bnd_flip={r['boundary_band_flip']:.3f} "
                f"rate={r['rate_amortized']:.5f} S~{r['S_projected_amortized']:.3f} "
                f"{r['elapsed_s']:.0f}s",
                flush=True,
            )
        if not per_frame:
            continue

        def avg(k, _pf=per_frame):
            vals = [
                p[k]
                for p in _pf
                if p[k] is not None and not (isinstance(p[k], float) and math.isnan(p[k]))
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "hidden": hid,
            "n_channels": args.n_channels,
            "n_steps": args.n_steps,
            "seed_factor": args.seed_factor,
            "dominant_only": args.dominant_only,
            "n_frames": len(per_frame),
            "avg_rule_param_count": avg("rule_param_count"),
            "avg_geometric_dseg_recon": avg("geometric_dseg_recon"),
            "avg_geometric_dseg": avg("geometric_dseg"),
            "avg_realized_dseg": avg("realized_dseg"),
            "avg_boundary_band_flip": avg("boundary_band_flip"),
            "avg_interior_flip": avg("interior_flip"),
            "avg_rate_amortized": avg("rate_amortized"),
            "avg_rate_full": avg("rate_full"),
            "avg_S_projected_amortized": avg("S_projected_amortized"),
            "avg_S_projected_full": avg("S_projected_full"),
            "per_frame": per_frame,
        }
        state["rows"][key] = row
        save_state()
        print(
            f"   -> {key}: avg_realized={row['avg_realized_dseg']:.5f} "
            f"({row['avg_realized_dseg']/FRONTIER_DSEG:.1f}x frontier, "
            f"{row['avg_realized_dseg']/CURVE_CORE_SURVIVAL_FLOOR:.2f}x curve-survival-floor) "
            f"avg_rate={row['avg_rate_amortized']:.5f} avg_S~{row['avg_S_projected_amortized']:.3f}",
            flush=True,
        )

    # =====================================================================
    # VERDICT (MEASUREMENT-FIRST: realized d_seg through the chain is authority)
    # =====================================================================
    rows = state["rows"]
    if not rows:
        print("[error] no rows produced", flush=True)
        return 1
    best_realized = min(r["avg_realized_dseg"] for r in rows.values())
    best_geometric = min(r["avg_geometric_dseg"] for r in rows.values())
    best_geo_recon = min(r["avg_geometric_dseg_recon"] for r in rows.values())
    best_S = min(r["avg_S_projected_amortized"] for r in rows.values())
    best_row = min(rows.values(), key=lambda r: r["avg_S_projected_amortized"])

    cheap_rows = [r for r in rows.values() if r["avg_rate_amortized"] < 0.05]
    cheap_and_low = [r for r in cheap_rows if r["avg_realized_dseg"] < GREEN_DSEG_THRESHOLD]
    sub015_rows = [
        r
        for r in rows.values()
        if r["avg_rate_amortized"] < 0.05 and r["avg_S_projected_amortized"] < 0.15
    ]

    if sub015_rows and cheap_and_low:
        verdict = "GREEN_NCA_CORE_REACHES_SUB015_BYTE_CHEAP"
    elif best_realized < GREEN_DSEG_THRESHOLD:
        verdict = "AMBER_LOW_DSEG_BUT_NOT_SUB015_OR_BYTE_CHEAP"
    elif best_realized >= 0.9 * CURVE_CORE_SURVIVAL_FLOOR:
        verdict = "RED_NCA_CORE_HITS_SURVIVAL_WALL_LIKE_CURVE_CORE"
    else:
        verdict = "RED_NCA_CORE_CAPS_ABOVE_SUB015"

    # diagnose the wall: capacity (the rule can't grow L*) vs survival (grows L* but roundtrip flips it).
    survival_gap = best_realized - best_geo_recon
    # is the BYTE escape real? compare rule param count to the curve core's control-point bytes.
    byte_cheap_achieved = bool(cheap_rows)
    if best_geo_recon > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = (
            "CAPACITY (the NCA rule cannot GROW a partition matching L* even ignoring SegNet + "
            "roundtrip; the iteration/weight-sharing does NOT buy enough effective capacity for this "
            "boundary at byte-cheap rule sizes — the capacity escape is NOT real here)"
        )
    elif survival_gap > 0.5 * max(best_realized, 1e-9) and best_realized > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = (
            "SURVIVAL (the NCA GROWS a partition matching L* — the capacity escape IS real — but the "
            "bilinear roundtrip flips the boundary band -> realized d_seg >> geometric, the SAME wall "
            "the curve core hit. SURVIVAL is the terminal wall across ALL families)"
        )
    elif best_realized > GREEN_DSEG_THRESHOLD:
        wall_diagnosis = "BOTH_CONTRIBUTE (capacity + survival both above the threshold)"
    else:
        wall_diagnosis = "BOTH_OK_BUT_RATE_OR_S_HIGH"

    payload = {
        "schema": "nca_dseg_feasibility_gate.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_nca_dseg_feasibility_gate.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "the_question": (
            "Is there an NCA rule-size that is BYTE-CHEAP (rate << 0.05) AND grows a partition that, "
            "THROUGH the real SegNet + uint8 roundtrip, gives realized d_seg low enough that S < 0.15? "
            "GREEN -> the generative axis escapes BOTH the capacity + survival walls -> THE sub-0.15 "
            "path. RED -> the generative axis ALSO caps; if survival is the wall, it is the TERMINAL "
            "wall across all 3 families (learned-decoder + static-geometry + generative)."
        ),
        "thresholds": {
            "frontier_dseg": FRONTIER_DSEG,
            "sub015_dseg": SUB015_DSEG,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "curve_core_survival_floor_dseg": CURVE_CORE_SURVIVAL_FLOOR,
            "partition_store_survival_wall_dseg": PARTITION_STORE_REALIZED_DSEG,
            "byte_cheap_rate_threshold": 0.05,
            "S_target": 0.15,
        },
        "method": {
            "representation": (
                "Neural Cellular Automata: fixed depthwise perception (identity+Sobel, NO params) -> "
                "1x1 conv 3C->hidden (ReLU) -> 1x1 conv hidden->C (zero-init) = the LEARNED RULE, "
                "iterated N steps from a coarse downsampled-L* seed -> 1x1 readout C->n_classes logits "
                "-> softmax -> differentiable per-class-colour render. STORED BYTES = the rule + seed "
                "(N steps add NO bytes = the capacity escape under test)."
            ),
            "fit_objective": (
                "CE(real-SegNet(roundtrip(render(grown-logits))), L*) + gen_anchor_w * CE(grown-logits, "
                "L*); backprop into the rule + colours + boundary-band offset THROUGH the exact uint8 "
                "roundtrip (the survival lever)."
            ),
            "dseg_metric": (
                "realized argmax-flip-rate of the HARD (argmax-decoded class colours) frame through the "
                "EXACT roundtrip vs L* (the survival-checked authority). geometric_dseg_recon (the "
                "NCA's OWN argmax partition vs L*) = the capacity check; geometric_dseg (rendered "
                "frame's SegNet argmax, no roundtrip) = the pre-survival SegNet match."
            ),
            "byte_cost": (
                "quantized rule weights (8b, 0.55 entropy factor) + coarse seed grid; rule stored ONCE "
                "for 600 frames + tiny per-frame seed delta (amortized) — the same model as the curve "
                "gate for apples-to-apples rate."
            ),
            "train_device": args.train_device,
            "authority_device": "cpu",
            "n_channels": args.n_channels,
            "n_steps": args.n_steps,
            "seed_factor": args.seed_factor,
            "dominant_only": args.dominant_only,
            "n_frames": n_frames,
        },
        "rows": rows,
        "best_realized_dseg": best_realized,
        "best_geometric_dseg": best_geometric,
        "best_geometric_dseg_recon": best_geo_recon,
        "best_realized_dseg_x_frontier": best_realized / FRONTIER_DSEG,
        "best_realized_dseg_x_curve_survival_floor": best_realized / CURVE_CORE_SURVIVAL_FLOOR,
        "best_S_projected_amortized": best_S,
        "best_row_key": f"h{best_row['hidden']}",
        "n_byte_cheap_rows": len(cheap_rows),
        "n_byte_cheap_AND_low_dseg_rows": len(cheap_and_low),
        "n_sub015_rows": len(sub015_rows),
        "byte_escape_real": byte_cheap_achieved,
        "survival_gap_realized_minus_geo_recon": survival_gap,
        "wall_diagnosis": wall_diagnosis,
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: driven by the realized d_seg of the HARD-rendered frame THROUGH the "
            "real SegNet + exact uint8 roundtrip (the survival check), NOT the CE fit loss (a "
            "surrogate). A low CE / low geometric d_seg with high realized d_seg = the survival wall "
            "(the curve-core failure mode). Guard against false-GREEN (the factored-LF degenerate-fit "
            "anchor) AND false-RED (under-trained rule)."
        ),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    state["final_verdict"] = verdict
    state["result_json"] = str(result_json.relative_to(REPO))
    save_state()

    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}", flush=True)
    print(
        f"[best realized d_seg] {best_realized:.5f} ({best_realized/FRONTIER_DSEG:.1f}x frontier, "
        f"{best_realized/CURVE_CORE_SURVIVAL_FLOOR:.2f}x curve-survival-floor)",
        flush=True,
    )
    print(
        f"[best geometric d_seg] {best_geometric:.5f}  geo_recon={best_geo_recon:.5f}  "
        f"survival_gap={survival_gap:.5f}",
        flush=True,
    )
    print(f"[best projected S] {best_S:.4f}  wall: {wall_diagnosis}", flush=True)
    print(f"[byte escape real?] {byte_cheap_achieved}", flush=True)
    print(f"[VERDICT] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
