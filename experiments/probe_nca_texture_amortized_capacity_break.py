#!/usr/bin/env python3
"""GENERATIVE-AXIS FINAL EXHAUSTION TEST — best-shot AMORTIZED continuous-texture NCA + the capacity-break sweep.

This is the LAST un-run path of the sub-0.15 campaign. The sister AMBER gate
(probe_nca_texture_dseg_feasibility_gate.py) showed a continuous-texture NCA reaching near-frontier
realized d_seg (0.00337) WHEN it converges — but its verdict carried THREE caveats that this probe FIXES
and answers jointly:

  CAVEAT 1 (convergence fragility): only ~2/8 runs converged; the EXACT headline config re-run COLLAPSED
    (0.549). MPS non-determinism dropped ~75% of runs into a bad basin. The AMBER used grad-norm + warmup
    but OMITTED the canonical Mordvintsev stabilizer.
    -> FIX: persistent state POOL + sample-replay (Mordvintsev "Growing NCA") + stochastic per-cell update
       mask (fire-rate 0.5) + random CA-step count per batch [N_lo,N_hi] + alive-masking-free residual rule
       (zero-init update) + per-param grad-norm + LR warmup + step-decay + MULTI-RESTART keep-best +
       optional CPU-gradient (dodges the MPS non-determinism that collapsed 6/8 AMBER runs).
       TARGET: reliable convergence (>=7/8 restarts reach a coherent frame), not a lucky run.

  CAVEAT 2 (amortization untested, n=1 per rule): the AMBER fit a FRESH rule per frame; the reported rate
    0.019 ASSUMED one rule shared across 600 frames, but that sharing was never measured. A single shared
    rule that must reproduce ALL frames via small per-frame latents is STRICTLY HARDER.
    -> FIX: ONE shared rule + per-frame latent seeds across N_FRAMES (16-48) REAL GT frames, trained
       JOINTLY. Measure the AVERAGE d_seg across frames AND the TRUE amortized rate (shared rule ONCE / 600
       + per-frame latent x 600). NOT a fresh per-frame rule. This is the real decoder-replacement model.

  CAVEAT 3 (the ORIGINAL hypothesis, never swept): does weight-shared ITERATION break the
    d_seg ~ 29.3 * params^-0.71 capacity wall the one-shot learned-pixel decoder (factored-LF) hit?
    -> FIX: sweep rule-size (params) and plot the AMORTIZED AVERAGE d_seg(params) curve vs the power law.
       If iteration gives effective-depth-N detail, iterated d_seg(params) should fall FASTER than ^-0.71.
       This is the DECISIVE measurement (the fork: GREEN if it breaks the wall & byte-closes sub-0.15;
       AMBER if S in [0.15,0.19]; RED if it obeys ~^-0.71 = the generative axis caps like the rest =
       the FINAL family).

VERDICT FORK (measurement-first):
  GREEN -> a byte-closed S that beats the frontier 0.19110 (ideally sub-0.15): the iterated continuous-
           texture decoder IS the sub-0.15 vehicle.
  AMBER -> reliably converges + shared-rule holds d_seg but S in [0.15,0.19): quantify the gap + binding term.
  RED   -> even POOL-stabilized + shared-rule, AVERAGE d_seg(params) obeys ~^-0.71 (iteration does NOT
           break the capacity wall) -> generative axis caps like the rest -> FINAL family; the frontier
           ~0.191 is near the real achievable floor for ALL known representation families.

NO-FAKE (highest emphasis):
  - AVERAGE d_seg across frames, NOT best-frame (kills the AMBER's selection bias).
  - realized d_seg = argmax-flip-rate of the grown RGB through the REAL frozen SegNet + EXACT uint8
    roundtrip, on CPU AUTHORITY (MPS is the GRADIENT-only device, NEVER the score).
  - false-GREEN guard: a lucky converged frame is NOT the verdict — the AVERAGE over all frames is.
  - false-RED guard: report per-restart convergence; only the BEST of the multi-restart counts (an
    under-trained collapse must not be read as a capacity wall). The power-law fit uses the converged
    AVERAGE at each rule-size.
  - recompute S = 100*d_seg + sqrt(10*d_pose) + 25*B/B0 from components; held d_pose from the campaign.

$0, MPS fp32 gradient + CPU authority, no paid GPU, no PR. Resumable per-(rule-size) JSON checkpoint.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

GT_TARGETS_DIR = REPO / "experiments/results/capstone_gt_targets_cache"

# Campaign constants (measured, this campaign).
FRONTIER_DSEG = 0.00056          # the frontier's REAL d_seg (p_suff verdict: rate 0.118 + d_seg 0.056)
FRONTIER_S = 0.19110
GT_RGB_ROUNDTRIP_DSEG = 0.00022  # GT-RGB through the roundtrip (continuous CAN survive)
GREEN_DSEG_THRESHOLD = 0.0012
B0 = 37_545_489
HELD_POSE = 0.00034              # held d_pose (the campaign's d_pose budget; pose bundled per #158)

# The capacity wall to beat (factored-LF learned-pixel decoder): d_seg ~ A * params^-k.
POWERLAW_A = 29.3
POWERLAW_K = 0.71

from probe_curve_core_dseg_feasibility_gate import (  # noqa: E402
    _eval_roundtrip_t,
    _segnet_argmax_of_frame,
    _segnet_logits_of_frame,
    rate_from_total_bytes,
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def powerlaw_dseg(params: float) -> float:
    """The factored-LF one-shot decoder capacity wall: d_seg ~ 29.3 * params^-0.71."""
    return POWERLAW_A * (max(params, 1.0) ** (-POWERLAW_K))


# ===========================================================================
# Byte cost of the AMORTIZED shared-rule generator (rule ONCE for 600 frames + per-frame latent)
# ===========================================================================
def amortized_bytes(rule_param_count, latent_dim, weight_bits=8, latent_bits=8, packed_factor=0.55):
    """True amortized cost: ONE shared rule (stored once for all 600 frames) + a per-frame LATENT.

    This is the honest decoder-replacement accounting (the AMBER's caveat-2 fix): the rule is shared, so
    its bytes amortize over 600 frames; only the per-frame latent scales with frame count. We mirror the
    frontier's own amortization (28-d per-pair latent on a shared decoder), entropy-coded temporal-delta.
    """
    rule_bytes = rule_param_count * weight_bits * packed_factor / 8.0
    latent_bytes_per_frame = latent_dim * latent_bits * packed_factor / 8.0
    total_600_amortized = rule_bytes + latent_bytes_per_frame * 600.0
    return {
        "rule_param_count": int(rule_param_count),
        "rule_bytes": rule_bytes,
        "latent_dim": int(latent_dim),
        "latent_bytes_per_frame": latent_bytes_per_frame,
        "total_600_amortized_bytes": total_600_amortized,
    }


# ===========================================================================
# The SHARED-rule amortized continuous-texture NCA (one rule, per-frame latents, POOL-stabilized)
# ===========================================================================
class AmortizedNCA:
    """ONE shared local update rule + per-frame latent seeds -> CONTINUOUS RGB per frame.

    The DECODER REPLACEMENT: the rule (perception is fixed identity+Sobel, no params; the MLP w1/b1/w2 +
    readout is the only SHARED stored weight) is stored ONCE; each frame has its own small latent (the
    per-frame stored bytes). grow_rgb(frame_idx, n_steps, fire_rate) iterates the SAME rule from frame i's
    latent-conditioned seed.

    Mordvintsev "Growing NCA" stabilizers (the caveat-1 fix the AMBER omitted):
      - residual/zero-init update (w2 zero-init) so step 0 is identity (already in AMBER, kept)
      - STOCHASTIC per-cell update mask (fire_rate) during training
      - the PERSISTENT POOL of per-frame states is held in the trainer (sample-replay), not here
    """

    def __init__(self, n_channels, hidden, latent_dim, n_frames, shape, device, init_seed=1234,
                 state_bound=32.0):
        import math as _m

        import torch

        self.C = n_channels
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.shape = shape
        self.device = device
        self.state_bound = state_bound  # soft tanh bound on hidden state (alive-masking surrogate); None=off
        H, W = shape

        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        sob_x = torch.tensor([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]]) / 8.0
        sob_y = sob_x.t().contiguous()
        per = torch.stack([ident, sob_x, sob_y], dim=0)
        self.perception_w = per.repeat(self.C, 1, 1).unsqueeze(1).to(device)  # (3C,1,3,3) fixed, no params

        g = torch.Generator().manual_seed(init_seed)
        # ---- the SHARED rule (stored ONCE for all 600 frames) ----
        self.w1 = torch.nn.Parameter(
            (torch.randn(hidden, 3 * self.C, 1, 1, generator=g) * (1.0 / _m.sqrt(3 * self.C))).to(device)
        )
        self.b1 = torch.nn.Parameter(torch.zeros(hidden, device=device))
        self.w2 = torch.nn.Parameter(torch.zeros(self.C, hidden, 1, 1, device=device))  # zero-init = residual
        self.readout = torch.nn.Parameter(
            (torch.randn(3, self.C, 1, 1, generator=g) * (1.0 / _m.sqrt(self.C))).to(device)
        )
        self.readout_b = torch.nn.Parameter(torch.full((3,), 128.0, device=device))
        # latent -> coarse seed projection (SHARED). Fixed small coarse grid 6x8 -> NCA grows 384x512.
        self.hs, self.ws = 6, 8
        self.latent_proj = torch.nn.Parameter(
            (torch.randn(self.C * self.hs * self.ws, latent_dim, generator=g) * (1.0 / _m.sqrt(latent_dim))).to(device)
        )
        # ---- the PER-FRAME latents (the per-frame stored bytes; NOT shared) ----
        self.latents = torch.nn.Parameter((torch.randn(n_frames, latent_dim, generator=g) * 0.1).to(device))

    def shared_params(self):
        """The SHARED rule params (the decoder-replacement weight, stored once)."""
        return [self.w1, self.b1, self.w2, self.readout, self.readout_b, self.latent_proj]

    def all_params(self):
        return self.shared_params() + [self.latents]

    def shared_rule_param_count(self):
        n = self.w1.numel() + self.b1.numel() + self.w2.numel()
        n += self.readout.numel() + self.readout_b.numel() + self.latent_proj.numel()
        return int(n)

    def _seed(self, frame_idx):
        import torch.nn.functional as F

        H, W = self.shape
        coarse = (self.latent_proj @ self.latents[frame_idx]).view(1, self.C, self.hs, self.ws)
        return F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)

    def _step(self, x, fire_rate):
        """One NCA update step with optional stochastic per-cell fire mask (Mordvintsev fire_rate=0.5)."""
        import torch
        import torch.nn.functional as F

        per = F.conv2d(x, self.perception_w, padding=1, groups=self.C)
        h = F.relu(F.conv2d(per, self.w1, self.b1))
        dx = F.conv2d(h, self.w2)
        if fire_rate is not None and fire_rate < 1.0:
            mask = (torch.rand(1, 1, x.shape[2], x.shape[3], device=x.device) < fire_rate).float()
            dx = dx * mask
        x = x + dx
        if self.state_bound is not None:
            # STATE STABILITY (the alive-masking surrogate Mordvintsev uses to bound growth): soft-bound the
            # hidden state magnitude so the deep N-step unroll cannot diverge to inf/NaN. Without this the
            # pool feedback + trained residual rule grows the state unboundedly (the measured NaN at 2400it).
            x = self.state_bound * torch.tanh(x / self.state_bound)
        return x

    def grow_rgb(self, frame_idx, n_steps, fire_rate=None, init_state=None):
        """Iterate the SHARED rule n_steps from frame i's latent-seed (or init_state) -> CONTINUOUS RGB.

        Returns (rgb (3,H,W) clamped[0,255], final_state) so the trainer can write final_state to the POOL.
        """
        import torch.nn.functional as F

        x = self._seed(frame_idx) if init_state is None else init_state
        for _ in range(n_steps):
            x = self._step(x, fire_rate)
        rgb = F.conv2d(x, self.readout, self.readout_b)
        return rgb[0].clamp(0, 255), x.detach()


# ===========================================================================
# Train ONE shared rule across N frames (POOL + sample-replay + multi-restart keep-best)
# ===========================================================================
def _train_one_restart(seg_targets, gt_frames, segnet_cpu, segnet_train, hidden, n_channels,
                       latent_dim, n_frames, args, restart_seed):
    """One restart of the JOINT shared-rule training. Returns (per-frame realized d_seg list, diagnostics)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    H, W = seg_targets[0].shape
    tdev = torch.device(args.train_device)

    nca = AmortizedNCA(n_channels, hidden, latent_dim, n_frames, (H, W), tdev, init_seed=restart_seed,
                       state_bound=(args.state_bound if args.state_bound > 0 else None))
    Lts = [torch.tensor(seg_targets[i], dtype=torch.long, device=tdev) for i in range(n_frames)]
    gt_ts = [
        torch.tensor(gt_frames[i], dtype=torch.float32, device=tdev).permute(2, 0, 1)
        for i in range(n_frames)
    ]
    params = nca.all_params()
    opt = torch.optim.Adam(params, lr=args.lr)

    # Persistent POOL of per-frame states (Mordvintsev sample-replay). One slot per frame; we keep a small
    # pool-per-frame so the rule learns to grow from BOTH a fresh seed and a partially-grown state (the
    # stability mechanism). pool[i] holds the most-recent grown state for frame i (or None = fresh seed).
    pool: list = [None] * n_frames

    warmup = max(1, int(0.05 * args.iters))
    decay_at = int(0.6 * args.iters)
    rng = np.random.default_rng(restart_seed)
    t0 = time.time()
    bs = min(args.batch_frames, n_frames)

    for _it in range(args.iters):
        lr = args.lr * min(1.0, (_it + 1) / warmup)
        if _it >= decay_at:
            lr *= 0.1  # canonical step decay
        for grp in opt.param_groups:
            grp["lr"] = lr

        # sample a batch of frames
        batch = rng.choice(n_frames, size=bs, replace=False)
        # POOL: with prob args.pool_prob start from the pooled (partially-grown) state, else fresh seed.
        # ALWAYS include at least one fresh-seed frame in the batch (the "replace highest-loss with seed").
        opt.zero_grad(set_to_none=True)
        nsteps = int(rng.integers(args.n_steps_lo, args.n_steps_hi + 1))
        total_loss = 0.0
        new_states = {}
        per_losses = {}
        for j, fi in enumerate(batch):
            fi = int(fi)
            use_pool = (pool[fi] is not None) and (j != 0) and (rng.random() < args.pool_prob)
            init_state = pool[fi].to(tdev) if use_pool else None
            rgb, final_state = nca.grow_rgb(fi, nsteps, fire_rate=args.fire_rate, init_state=init_state)
            rt = _eval_roundtrip_t(rgb, ste=True)[0]
            seg_logits = _segnet_logits_of_frame(segnet_train, rt)
            ce_seg = F.cross_entropy(seg_logits, Lts[fi].unsqueeze(0))
            recon = F.mse_loss(rgb, gt_ts[fi]) / (255.0**2)
            loss = ce_seg + args.recon_w * recon
            total_loss = total_loss + loss
            new_states[fi] = final_state
            per_losses[fi] = float(loss.item())
        (total_loss / bs).backward()
        if args.grad_norm:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(p.grad.norm() + 1e-8)
        else:
            torch.nn.utils.clip_grad_norm_(params, 50.0)
        opt.step()

        # POOL write-back: store grown states; replace the HIGHEST-loss frame's pool slot with a fresh
        # seed (None) so the rule must keep being able to grow from scratch (Mordvintsev sample-replay).
        for fi, st in new_states.items():
            pool[fi] = st.cpu()
        worst = max(per_losses, key=per_losses.get)
        pool[worst] = None

    # ---- AUTHORITY eval: AVERAGE realized d_seg across ALL frames (no fire mask at eval; full n_steps_hi) ----
    realized = []
    geometric = []
    recon_rmses = []
    bnd_flips = []
    int_flips = []
    eval_steps = args.n_steps_hi
    with torch.no_grad():
        for fi in range(n_frames):
            rgb, _ = nca.grow_rgb(fi, eval_steps, fire_rate=None, init_state=None)
            rgb = rgb.cpu()
            L = seg_targets[fi]
            rt_hard = _eval_roundtrip_t(rgb, ste=False)[0]
            real_argmax = _segnet_argmax_of_frame(segnet_cpu, rt_hard).cpu().numpy()
            real_dseg = float((real_argmax != L).mean())
            geo_argmax = _segnet_argmax_of_frame(segnet_cpu, rgb).cpu().numpy()
            geo_dseg = float((geo_argmax != L).mean())
            recon_rmse = float(((rgb - gt_ts[fi].cpu()) ** 2).mean().sqrt())
            realized.append(real_dseg)
            geometric.append(geo_dseg)
            recon_rmses.append(recon_rmse)
            # boundary/interior split
            bmask = np.zeros((H, W), dtype=bool)
            bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
            bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
            bmask[:-1, :] |= L[:-1, :] != L[1:, :]
            bmask[1:, :] |= L[:-1, :] != L[1:, :]
            from scipy import ndimage

            band1 = ndimage.binary_dilation(bmask, iterations=1)
            interior = ~band1
            bnd_flips.append(float((real_argmax[band1] != L[band1]).mean()) if band1.any() else float("nan"))
            int_flips.append(float((real_argmax[interior] != L[interior]).mean()) if interior.any() else float("nan"))

    rule_pc = nca.shared_rule_param_count()
    return {
        "restart_seed": restart_seed,
        "rule_param_count": rule_pc,
        "per_frame_realized_dseg": realized,
        "avg_realized_dseg": float(sum(realized) / len(realized)),
        "median_realized_dseg": float(sorted(realized)[len(realized) // 2]),
        "best_frame_realized_dseg": float(min(realized)),
        "worst_frame_realized_dseg": float(max(realized)),
        "avg_geometric_dseg": float(sum(geometric) / len(geometric)),
        "avg_recon_rmse": float(sum(recon_rmses) / len(recon_rmses)),
        "avg_boundary_band_flip": float(np.nanmean(bnd_flips)),
        "avg_interior_flip": float(np.nanmean(int_flips)),
        # convergence: a frame "converged" if recon is coherent (rmse < threshold) AND realized d_seg is
        # not in the collapsed regime (>0.3). Count converged frames.
        "n_converged_frames": int(sum(1 for r, rm in zip(realized, recon_rmses) if r < 0.3 and rm < 60.0)),
        "elapsed_s": round(time.time() - t0, 1),
    }


def sweep_rule_size(seg_targets, gt_frames, segnet_cpu, segnet_train, hidden, n_channels, latent_dim,
                    n_frames, args):
    """Multi-restart keep-best at ONE rule-size. The verdict uses the BEST-converged restart's AVERAGE
    d_seg (false-RED guard: an under-trained collapse must not read as a capacity wall)."""
    restarts = []
    for r in range(args.restarts):
        seed = 1234 + 1000 * r
        res = _train_one_restart(
            seg_targets, gt_frames, segnet_cpu, segnet_train, hidden, n_channels,
            latent_dim, n_frames, args, seed,
        )
        restarts.append(res)
        print(
            f"      restart {r} (seed={seed}): avg_realized={res['avg_realized_dseg']:.5f} "
            f"converged_frames={res['n_converged_frames']}/{n_frames} "
            f"recon_rmse={res['avg_recon_rmse']:.1f} bnd={res['avg_boundary_band_flip']:.3f} "
            f"int={res['avg_interior_flip']:.4f} {res['elapsed_s']:.0f}s",
            flush=True,
        )
    # keep-best by AVERAGE realized d_seg among restarts that converged on a majority of frames
    converged_restarts = [r for r in restarts if r["n_converged_frames"] >= max(1, n_frames // 2)]
    pool_for_best = converged_restarts if converged_restarts else restarts
    best = min(pool_for_best, key=lambda r: r["avg_realized_dseg"])
    n_conv_restarts = len(converged_restarts)

    rule_pc = best["rule_param_count"]
    bytes_info = amortized_bytes(rule_pc, latent_dim)
    rate = rate_from_total_bytes(bytes_info["total_600_amortized_bytes"])
    s_proj = 100 * best["avg_realized_dseg"] + math.sqrt(10 * HELD_POSE) + rate

    pl = powerlaw_dseg(rule_pc)
    return {
        "hidden": int(hidden),
        "n_channels": int(n_channels),
        "latent_dim": int(latent_dim),
        "n_frames": int(n_frames),
        "rule_param_count": rule_pc,
        "n_restarts": args.restarts,
        "n_converged_restarts": n_conv_restarts,
        "convergence_rate": n_conv_restarts / args.restarts,
        # the VERDICT number: BEST-converged restart's AVERAGE realized d_seg (NO-FAKE: average, not best-frame)
        "best_converged_avg_realized_dseg": best["avg_realized_dseg"],
        "best_converged_median_realized_dseg": best["median_realized_dseg"],
        "best_converged_best_frame_realized_dseg": best["best_frame_realized_dseg"],
        "best_converged_worst_frame_realized_dseg": best["worst_frame_realized_dseg"],
        "best_converged_avg_geometric_dseg": best["avg_geometric_dseg"],
        "best_converged_avg_recon_rmse": best["avg_recon_rmse"],
        "best_converged_avg_boundary_band_flip": best["avg_boundary_band_flip"],
        "best_converged_avg_interior_flip": best["avg_interior_flip"],
        "best_converged_n_converged_frames": best["n_converged_frames"],
        # the CAPACITY-BREAK comparison
        "powerlaw_dseg_at_this_param_count": pl,
        "amortized_avg_dseg_over_powerlaw": best["avg_realized_dseg"] / pl,
        "beats_powerlaw": best["avg_realized_dseg"] < pl,
        # bytes / rate / S
        "bytes": bytes_info,
        "rate_amortized": rate,
        "S_projected_amortized": s_proj,
        "avg_realized_dseg_x_frontier": best["avg_realized_dseg"] / FRONTIER_DSEG,
        "all_restarts": restarts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/nca_texture_amortized_capacity_break"))
    ap.add_argument("--hiddens", default="48,96,192,384", help="comma-sep NCA hidden widths (the rule-size sweep)")
    ap.add_argument("--sweep-spec", default="", help="C:H,C:H,... explicit (channels,hidden) configs for a WIDE param range (capacity-break fit); overrides --hiddens")
    ap.add_argument("--n-channels", type=int, default=16)
    ap.add_argument("--latent-dim", type=int, default=24, help="per-frame latent (the per-frame stored bytes)")
    ap.add_argument("--n-frames", type=int, default=16, help="frames the ONE shared rule must reproduce (amortization)")
    ap.add_argument("--batch-frames", type=int, default=4, help="frames per training step (the NCA batch)")
    ap.add_argument("--n-steps-lo", type=int, default=48, help="min CA steps per batch (Mordvintsev random [lo,hi])")
    ap.add_argument("--n-steps-hi", type=int, default=72, help="max CA steps per batch (also the eval step count)")
    ap.add_argument("--fire-rate", type=float, default=0.5, help="stochastic per-cell update mask prob (Mordvintsev 0.5)")
    ap.add_argument("--pool-prob", type=float, default=0.5, help="prob a batch frame starts from the pooled state")
    ap.add_argument("--state-bound", type=float, default=32.0, help="soft tanh bound on NCA hidden state (alive-masking surrogate; prevents the unbounded-growth NaN; 0=off)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3, help="Mordvintsev Adam lr (with warmup + step-decay)")
    ap.add_argument("--recon-w", type=float, default=1.0)
    ap.add_argument("--restarts", type=int, default=3, help="multi-restart keep-best (convergence robustness)")
    ap.add_argument(
        "--grad-norm", action=argparse.BooleanOptionalAction, default=True,
        help="canonical Growing-NCA per-param gradient normalization",
    )
    ap.add_argument("--train-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--timing-smoke", action="store_true")
    args = ap.parse_args()

    import torch

    if args.train_device == "mps":
        try:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patch_scorer_for_mps()
        except Exception as e:  # pragma: no cover
            print(f"[warn] patch_scorer_for_mps failed ({e}); continuing", flush=True)

    import torch.nn.functional as F

    from tac.boundary_math.seg_core import decode_gt_frame1_pairs
    from tac.scorer import load_default_segnet

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "gate_state.json"
    result_json = REPO / ".omx/research" / f"nca_texture_amortized_capacity_break_{_now()}.json"

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] prior rows: {list(state.get('rows', {}).keys())}", flush=True)
    state.setdefault("rows", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    # pick the GT cache that covers n_frames
    cache_choices = [(16, "gt_targets_n16.pt"), (24, "gt_targets_n24.pt"), (48, "gt_targets_n48.pt"),
                     (100, "gt_targets_n100.pt"), (192, "gt_targets_n192.pt")]
    cache_file = next((f for n, f in cache_choices if n >= args.n_frames), "gt_targets_n192.pt")
    gt_path = GT_TARGETS_DIR / cache_file
    print(f"[gt] using {cache_file} for n_frames={args.n_frames}", flush=True)

    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")
    segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)

    gt = torch.load(gt_path, map_location="cpu", weights_only=False)
    seg_targets_all = gt["seg"].numpy()
    n_frames = min(args.n_frames, seg_targets_all.shape[0])
    seg_targets = [seg_targets_all[i] for i in range(n_frames)]

    # GT RGB frames (decode the real frame1 of each pair, resize to 384x512)
    gt_frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        t = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0)
        rs = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_frames[pidx] = rs[0].permute(1, 2, 0).numpy()
        if len(gt_frames) >= n_frames:
            break
    # remap gt_frames to dense 0..n_frames-1 to align with seg_targets indices
    keys = sorted(gt_frames.keys())[:n_frames]
    gt_frames = {i: gt_frames[k] for i, k in enumerate(keys)}

    if args.timing_smoke:
        print("[timing-smoke] hidden=96, n_frames=4, 60 iters, 1 restart ...", flush=True)
        sa = argparse.Namespace(**vars(args))
        sa.iters = 60
        sa.restarts = 1
        nf = min(4, n_frames)
        st = {i: seg_targets[i] for i in range(nf)}
        gf = {i: gt_frames[i] for i in range(nf)}
        r = _train_one_restart(st, gf, segnet_cpu, segnet_train, 96, args.n_channels,
                               args.latent_dim, nf, sa, 1234)
        spi = r["elapsed_s"] / sa.iters
        full_min = spi * args.iters * args.restarts * len(args.hiddens.split(",")) / 60
        print(f"[timing-smoke] {r['elapsed_s']:.0f}s/60it -> ~{spi:.2f}s/it ; full sweep ~{full_min:.1f} min", flush=True)
        print(f"   avg_realized={r['avg_realized_dseg']:.5f} converged={r['n_converged_frames']}/{nf} "
              f"recon_rmse={r['avg_recon_rmse']:.1f} bnd={r['avg_boundary_band_flip']:.3f}", flush=True)
        return 0

    # The capacity sweep. Either --hiddens (fixed C) OR --sweep-spec "C:H,C:H,..." for a WIDE param range
    # (the latent_proj ~ C*48*latent_dim dominates, so varying C is the real param lever for the
    # capacity-break fit). sweep-spec takes precedence when given.
    if args.sweep_spec:
        configs = []
        for tok in args.sweep_spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            c_str, h_str = tok.split(":")
            configs.append((int(c_str), int(h_str)))
    else:
        configs = [(args.n_channels, int(x)) for x in args.hiddens.split(",") if x.strip()]

    for (ch, hid) in configs:
        key = f"c{ch}h{hid}"
        if key in state["rows"]:
            print(f"[resume] {key} done; skip", flush=True)
            continue
        print(f"\n=== AMORTIZED TEXTURE-NCA {key} (C={ch}, hidden={hid}, latent={args.latent_dim}, "
              f"n_frames={n_frames}, restarts={args.restarts}, N=[{args.n_steps_lo},{args.n_steps_hi}], "
              f"fire={args.fire_rate}, pool={args.pool_prob}) ===", flush=True)
        row = sweep_rule_size(seg_targets, gt_frames, segnet_cpu, segnet_train, hid,
                              ch, args.latent_dim, n_frames, args)
        state["rows"][key] = row
        save_state()
        pl = row["powerlaw_dseg_at_this_param_count"]
        print(
            f"   -> {key}: rule_pc={row['rule_param_count']} "
            f"BEST-conv avg_realized={row['best_converged_avg_realized_dseg']:.5f} "
            f"({row['avg_realized_dseg_x_frontier']:.1f}x frontier) "
            f"conv_rate={row['n_converged_restarts']}/{row['n_restarts']} "
            f"| powerlaw={pl:.5f} ratio={row['amortized_avg_dseg_over_powerlaw']:.2f} "
            f"beats_wall={row['beats_powerlaw']} | rate={row['rate_amortized']:.5f} S~{row['S_projected_amortized']:.3f}",
            flush=True,
        )

    rows = state["rows"]
    if not rows:
        print("[error] no rows", flush=True)
        return 1

    _finalize(rows, state, state_path, result_json, args, n_frames)
    return 0


def _finalize(rows, state, state_path, result_json, args, n_frames):
    """Compute the capacity-break verdict from the swept rows."""
    # capacity-break: does the AMORTIZED avg d_seg(params) curve fall FASTER than the ^-0.71 power law?
    # fit a power law to our own (params, avg_dseg) points and compare the exponent.
    import numpy as np

    pts = [(r["rule_param_count"], r["best_converged_avg_realized_dseg"]) for r in rows.values()
           if r["n_converged_restarts"] > 0 and r["best_converged_avg_realized_dseg"] < 0.3]
    fitted_k = None
    fitted_a = None
    if len(pts) >= 2:
        xs = np.log(np.array([p[0] for p in pts], dtype=float))
        ys = np.log(np.array([p[1] for p in pts], dtype=float))
        slope, intercept = np.polyfit(xs, ys, 1)
        fitted_k = -float(slope)  # our exponent (positive = decreasing)
        fitted_a = float(np.exp(intercept))

    best_avg = min((r["best_converged_avg_realized_dseg"] for r in rows.values()
                    if r["n_converged_restarts"] > 0), default=float("inf"))
    best_S = min((r["S_projected_amortized"] for r in rows.values()
                  if r["n_converged_restarts"] > 0), default=float("inf"))
    best_row = min((r for r in rows.values() if r["n_converged_restarts"] > 0),
                   key=lambda r: r["S_projected_amortized"], default=None)
    n_beats_wall = sum(1 for r in rows.values() if r.get("beats_powerlaw") and r["n_converged_restarts"] > 0)
    overall_conv_rate = (sum(r["n_converged_restarts"] for r in rows.values())
                         / max(1, sum(r["n_restarts"] for r in rows.values())))

    sub015 = [r for r in rows.values() if r["n_converged_restarts"] > 0
              and r["rate_amortized"] < 0.05 and r["S_projected_amortized"] < 0.15]
    beats_frontier = [r for r in rows.values() if r["n_converged_restarts"] > 0
                      and r["S_projected_amortized"] < FRONTIER_S]

    # The decisive fork.
    if sub015 or beats_frontier:
        verdict = "GREEN_AMORTIZED_NCA_BYTE_CLOSES_BELOW_FRONTIER"
    elif best_S < 0.19 and overall_conv_rate >= 0.5:
        verdict = "AMBER_RELIABLE_AND_SHARED_RULE_HOLDS_BUT_S_IN_015_019"
    elif fitted_k is not None and fitted_k > (POWERLAW_K + 0.10) and best_avg < powerlaw_dseg_min(rows):
        # our curve falls FASTER than the power law AND lands below it -> iteration helps but not enough for GREEN
        verdict = "AMBER_ITERATION_BEATS_POWERLAW_EXPONENT_BUT_NOT_SUB015"
    else:
        verdict = "RED_AMORTIZED_NCA_OBEYS_CAPACITY_WALL_GENERATIVE_AXIS_CAPS"

    payload = {
        "schema": "nca_texture_amortized_capacity_break.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_nca_texture_amortized_capacity_break.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "the_three_caveats_fixed": {
            "caveat_1_convergence": "Mordvintsev POOL + sample-replay + fire-rate + random CA steps + multi-restart keep-best",
            "caveat_2_amortization": f"ONE shared rule across {n_frames} REAL frames + per-frame latents; AVERAGE d_seg + TRUE amortized rate",
            "caveat_3_capacity_break": "rule-size sweep -> avg d_seg(params) vs 29.3*params^-0.71 power law",
        },
        "thresholds": {
            "frontier_dseg": FRONTIER_DSEG,
            "frontier_S": FRONTIER_S,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "gt_rgb_roundtrip_dseg": GT_RGB_ROUNDTRIP_DSEG,
            "byte_cheap_rate_threshold": 0.05,
            "S_target": 0.15,
            "powerlaw_A": POWERLAW_A,
            "powerlaw_K": POWERLAW_K,
        },
        "method": {
            "representation": "ONE shared NCA rule (perception fixed Sobel+ident, MLP w1/b1/w2 zero-init residual, "
                              "readout C->3 RGB) + per-frame latent seeds. grow_rgb iterates SAME rule per frame.",
            "stabilizers": "persistent state pool + sample-replay (replace highest-loss with fresh seed) + "
                           f"fire-rate {args.fire_rate} + random CA steps [{args.n_steps_lo},{args.n_steps_hi}] + "
                           f"per-param grad-norm + LR warmup + step-decay + {args.restarts} restarts keep-best",
            "fit_objective": "sum_frames[ CE(real-SegNet(roundtrip(grown-RGB)), L*) + recon_w*MSE(grown-RGB, GT-RGB) ]",
            "dseg_metric": "AVERAGE over frames of realized argmax-flip-rate through the EXACT roundtrip vs L* (CPU authority)",
            "train_device": args.train_device,
            "authority_device": "cpu",
            "n_channels": args.n_channels,
            "latent_dim": args.latent_dim,
            "n_frames": n_frames,
            "iters": args.iters,
        },
        "rows": rows,
        "capacity_break": {
            "our_fitted_exponent_k": fitted_k,
            "our_fitted_A": fitted_a,
            "powerlaw_exponent_k": POWERLAW_K,
            "iteration_breaks_exponent": (fitted_k is not None and fitted_k > POWERLAW_K + 0.10),
            "n_rule_sizes_beating_powerlaw": n_beats_wall,
            "interpretation": (
                "If our_fitted_exponent_k > 0.71 AND points land below the power law, weight-shared iteration "
                "gives effective-depth-N detail that the one-shot decoder lacks (the capacity escape). If "
                "k ~ 0.71 or points sit on/above the wall, iteration does NOT break the capacity wall = the "
                "generative axis caps like the static families."
            ),
        },
        "best_converged_avg_realized_dseg": best_avg,
        "best_S_projected_amortized": best_S,
        "best_row_key": (f"c{best_row['n_channels']}h{best_row['hidden']}" if best_row else None),
        "overall_convergence_rate": overall_conv_rate,
        "n_sub015_rows": len(sub015),
        "n_beats_frontier_rows": len(beats_frontier),
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: AVERAGE realized d_seg (NOT best-frame) of ONE shared rule across "
            f"{n_frames} real frames, through the real SegNet + exact roundtrip, multi-restart keep-best for "
            "convergence robustness. The capacity-break is the fitted d_seg(params) exponent vs 29.3*params^-0.71."
        ),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    state["final_verdict"] = verdict
    state["result_json"] = str(result_json.relative_to(REPO))
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}", flush=True)
    print(f"[capacity-break] our fitted exponent k={fitted_k} vs power-law k={POWERLAW_K} "
          f"(breaks wall if k>{POWERLAW_K+0.10:.2f}); rule-sizes beating wall: {n_beats_wall}", flush=True)
    print(f"[best converged avg realized d_seg] {best_avg:.5f} ({best_avg/FRONTIER_DSEG:.1f}x frontier)", flush=True)
    print(f"[best projected amortized S] {best_S:.4f} (frontier {FRONTIER_S})", flush=True)
    print(f"[overall convergence rate] {overall_conv_rate:.2f}", flush=True)
    print(f"[VERDICT] {verdict}", flush=True)


def powerlaw_dseg_min(rows):
    """Min power-law d_seg over the swept param counts (the lowest wall any swept size could reach)."""
    return min(powerlaw_dseg(r["rule_param_count"]) for r in rows.values())


if __name__ == "__main__":
    raise SystemExit(main())
