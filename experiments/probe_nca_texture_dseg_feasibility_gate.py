#!/usr/bin/env python3
"""GENERATIVE-AXIS GATE v2 — CONTINUOUS-TEXTURE NCA d_seg-core (the operator's TRUE reframe).

The sister flat-partition NCA gate (probe_nca_dseg_feasibility_gate.py) tested the WRONG representation:
the NCA grew a 5-class flat-colour PARTITION (class -> fixed colour lookup), a generative version of the
curve core, and capped RED (realized ~0.02) because a flat-fill boundary can't beat the survival wall.

THIS gate tests the OPERATOR'S ACTUAL REFRAME: the NCA grows CONTINUOUS RGB TEXTURE directly (each pixel a
full learned colour, like Mordvintsev's Growing-NCA) — the generator IS the renderer, REPLACING the 161KB
feedforward decoder with a few-KB iterated rule + a small per-frame latent seed.

THE DECISIVE EMPIRICAL ANCHOR (measured, this gate's premise): the GT RGB frame (continuous texture) through
the EXACT roundtrip realizes d_seg = 0.00022 — FRONTIER-GRADE, 30x better than the flat-partition NCA's
0.0067 and BELOW the GREEN threshold. So continuous texture CAN survive the roundtrip. The bet: can a
few-KB iterated NCA rule reproduce that texture well enough that its realized d_seg stays near 0.00022?

THE GATE:
  Is there an NCA rule-size (few KB) + small per-frame latent that grows a CONTINUOUS RGB frame whose
  realized d_seg (THROUGH the real SegNet + exact uint8 roundtrip) is low enough that
  100*d_seg + sqrt(10*d_pose) + 25*B/B0 < 0.15?
    GREEN -> the generative axis (continuous-texture) REPLACES the decoder at few-KB rate -> THE sub-0.15
             path (rate ~0.013 -> S ~0.086 IF d_seg matches frontier).
    RED   -> even continuous-texture iteration cannot reproduce the d_seg-critical RGB at few-KB rule size
             -> the generative axis is fully closed (both sub-variants).

WHY THIS IS THE RIGHT, FAITHFUL TEST (NO-FAKE, measurement-first):
  - The NCA readout is C->3 RGB (CONTINUOUS), NOT C->n_classes logits + colour lookup. The generator
    outputs texture, not a partition. This is the structural fix the sister flagged.
  - Target = the GT RGB frame (whose SegNet argmax IS L*, realized d_seg 0.00022 through the roundtrip) —
    the prize is reproducing the d_seg-critical RGB at few-KB rule cost.
  - Objective: CE(real-SegNet(roundtrip(grown-RGB)), L*)  [the d_seg objective DIRECTLY]  +  recon MSE to
    GT-RGB (anchors the texture). Backprop into the rule + latent THROUGH the exact roundtrip on MPS fp32;
    realized d_seg measured on CPU AUTHORITY.
  - REUSES the curve gate's exact roundtrip + realized-d_seg metric + GT load for apples-to-apples.
  - Stabilized like the sister: canonical Growing-NCA per-param grad-norm + LR warmup (the flat-partition
    gate proved lr=4e-2 diverges without it).

CARMACK MVP-FIRST: small n_frames (3), modest N (64), rule-size sweep -> feasibility ESTIMATE.
$0, MPS fp32 gradient + CPU authority, no paid GPU, no PR. Resumable per-rule-size JSON checkpoint.

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

GT_TARGETS = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n16.pt"

FRONTIER_DSEG = 0.00257
SUB015_DSEG = 0.0006
GT_RGB_ROUNDTRIP_DSEG = 0.00022  # the measured premise: continuous texture survives the roundtrip
FLAT_PARTITION_NCA_FLOOR = 0.0162  # the sister flat-partition NCA best (the bar to beat)
GREEN_DSEG_THRESHOLD = 0.0012
B0 = 37_545_489
HELD_POSE = 0.00034

from probe_curve_core_dseg_feasibility_gate import (
    _eval_roundtrip_t,
    _segnet_argmax_of_frame,
    _segnet_logits_of_frame,
    rate_from_total_bytes,
)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# Byte cost of the continuous-texture NCA GENERATOR (rule + per-frame latent)
# ===========================================================================
def nca_texture_bytes(rule_param_count, latent_dim, n_classes=5, weight_bits=8, latent_bits=8):
    """Advisory byte cost: the RULE (stored once for 600 frames) + a small per-frame LATENT seed.

    This is the decoder-replacement bet: replace the 161KB feedforward decoder with a few-KB rule + a
    per-frame latent (like the frontier's 28-d per-pair latents). The rule is quasi-static; the latent is
    per-frame (entropy-coded temporal-delta like the frontier latents). We mirror the curve/flat-NCA gate's
    amortization for apples-to-apples rate.
    """
    packed_factor = 0.55
    rule_bytes = rule_param_count * weight_bits * packed_factor / 8.0
    latent_bytes_per_frame = latent_dim * latent_bits * packed_factor / 8.0
    # rule once + latent per-frame (temporal-delta entropy-coded ~ the per-frame cost) over 600 frames
    total_600_amortized = rule_bytes + latent_bytes_per_frame * 600.0
    total_600_full = (rule_bytes + latent_bytes_per_frame) * 600.0
    return {
        "rule_param_count": int(rule_param_count),
        "rule_bytes": rule_bytes,
        "latent_dim": int(latent_dim),
        "latent_bytes_per_frame": latent_bytes_per_frame,
        "total_600_amortized_bytes": total_600_amortized,
        "total_600_full_bytes": total_600_full,
    }


# ===========================================================================
# The continuous-texture NCA generator (readout C->3 RGB, the generator IS the renderer)
# ===========================================================================
class NCATextureGenerator:
    """A tiny shared local update rule iterated N steps from a LATENT-conditioned seed -> CONTINUOUS RGB.

    State: (1,C,H,W). Perception: fixed depthwise identity+Sobel (NO params). Rule: 1x1 conv 3C->hidden
    (ReLU) -> 1x1 conv hidden->C (zero-init). Readout: 1x1 conv C->3 RGB (the generator IS the renderer —
    continuous RGB, NOT a class partition). Seed: a small per-frame LATENT (latent_dim) broadcast +
    projected to the C-channel grid at a coarse resolution then upsampled (the per-frame stored bytes).

    Stored bytes = the rule + readout (quasi-static) + the per-frame latent. n_steps adds NO bytes.
    """

    def __init__(self, n_channels, hidden, latent_dim, shape, device, seed_factor=16):
        import math as _m

        import torch

        self.C = n_channels
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.shape = shape
        self.device = device
        self.seed_factor = seed_factor
        H, W = shape

        ident = torch.zeros(3, 3)
        ident[1, 1] = 1.0
        sob_x = torch.tensor([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]]) / 8.0
        sob_y = sob_x.t().contiguous()
        per = torch.stack([ident, sob_x, sob_y], dim=0)
        self.perception_w = per.repeat(self.C, 1, 1).unsqueeze(1).to(device)  # (3C,1,3,3) fixed

        g = torch.Generator().manual_seed(1234)
        self.w1 = torch.nn.Parameter(
            (torch.randn(hidden, 3 * self.C, 1, 1, generator=g) * (1.0 / _m.sqrt(3 * self.C))).to(device)
        )
        self.b1 = torch.nn.Parameter(torch.zeros(hidden, device=device))
        self.w2 = torch.nn.Parameter(torch.zeros(self.C, hidden, 1, 1, device=device))  # zero-init update
        # readout C->3 RGB (continuous), init small
        self.readout = torch.nn.Parameter(
            (torch.randn(3, self.C, 1, 1, generator=g) * (1.0 / _m.sqrt(self.C))).to(device)
        )
        self.readout_b = torch.nn.Parameter(torch.full((3,), 128.0, device=device))  # mid-grey bias
        # latent -> coarse seed projection at a SMALL FIXED grid (6x8). The NCA's JOB is to GROW detail
        # from a coarse anchor via iteration (the capacity escape); the seed must be TINY so the stored
        # bytes stay few-KB. (seed_factor is kept as a CLI knob but the seed grid is fixed-small here so
        # latent_proj does not blow up to ~400K params — the byte-cheap premise of the whole gate.)
        self.hs, self.ws = 6, 8  # fixed small coarse grid (the NCA grows 384x512 from this)
        self.latent_proj = torch.nn.Parameter(
            (torch.randn(self.C * self.hs * self.ws, latent_dim, generator=g) * (1.0 / _m.sqrt(latent_dim))).to(device)
        )
        # the per-frame latent (the stored per-frame bytes), learnable
        self.latent = torch.nn.Parameter((torch.randn(latent_dim, generator=g) * 0.1).to(device))

    def params(self):
        return [self.w1, self.b1, self.w2, self.readout, self.readout_b, self.latent_proj, self.latent]

    def rule_param_count(self):
        # the RULE + readout + latent_proj are stored ONCE for all 600 frames (the decoder replacement).
        # the per-frame latent is counted separately in nca_texture_bytes.
        n = self.w1.numel() + self.b1.numel() + self.w2.numel()
        n += self.readout.numel() + self.readout_b.numel() + self.latent_proj.numel()
        return int(n)

    def _seed(self):
        import torch.nn.functional as F

        H, W = self.shape
        coarse = (self.latent_proj @ self.latent).view(1, self.C, self.hs, self.ws)
        return F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)

    def grow_rgb(self, n_steps):
        """Iterate the rule n_steps from the latent-seed, return CONTINUOUS RGB (3,H,W)."""
        import torch.nn.functional as F

        x = self._seed()
        for _ in range(n_steps):
            per = F.conv2d(x, self.perception_w, padding=1, groups=self.C)
            h = F.relu(F.conv2d(per, self.w1, self.b1))
            dx = F.conv2d(h, self.w2)
            x = x + dx
        rgb = F.conv2d(x, self.readout, self.readout_b)  # (1,3,H,W)
        return rgb[0].clamp(0, 255)


# ===========================================================================
# Optimize ONE rule-size THROUGH the real SegNet + roundtrip
# ===========================================================================
def optimize_nca_texture(L, gt_rgb, segnet_cpu, segnet_train, hidden, n_channels, latent_dim,
                         n_steps, seed_factor, args):
    """Train the continuous-texture NCA to match L* (via SegNet+roundtrip) + GT-RGB (recon anchor).

    Measures realized d_seg = argmax-flip-rate of the grown RGB through the EXACT roundtrip vs L*.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    H, W = L.shape
    tdev = torch.device(args.train_device)

    gen = NCATextureGenerator(n_channels, hidden, latent_dim, (H, W), tdev, seed_factor)
    Lt = torch.tensor(L, dtype=torch.long, device=tdev)
    gt_t = torch.tensor(gt_rgb, dtype=torch.float32, device=tdev).permute(2, 0, 1)  # (3,H,W)
    params = gen.params()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    t0 = time.time()
    losses = []
    warmup = max(1, int(0.05 * args.iters))
    for _it in range(args.iters):
        lr = args.lr * min(1.0, (_it + 1) / warmup)
        for grp in opt.param_groups:
            grp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        rgb = gen.grow_rgb(n_steps)  # (3,H,W) CONTINUOUS texture
        rt = _eval_roundtrip_t(rgb, ste=True)[0]  # exact roundtrip (STE)
        seg_logits = _segnet_logits_of_frame(segnet_train, rt)  # real SegNet
        ce_seg = F.cross_entropy(seg_logits, Lt.unsqueeze(0))  # the d_seg objective DIRECTLY
        recon = F.mse_loss(rgb, gt_t) / (255.0**2)  # texture anchor (normalized)
        loss = ce_seg + args.recon_w * recon
        loss.backward()
        if args.grad_norm:
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(p.grad.norm() + 1e-8)
        else:
            torch.nn.utils.clip_grad_norm_(params, 50.0)
        opt.step()
        losses.append((float(ce_seg.item()), float(recon.item())))

    with torch.no_grad():
        rgb = gen.grow_rgb(n_steps).cpu()
        # geometric (no roundtrip): grown-RGB SegNet argmax vs L*
        geo_argmax = _segnet_argmax_of_frame(segnet_cpu, rgb).cpu().numpy()
        geo_dseg = float((geo_argmax != L).mean())
        # REALIZED: through the exact roundtrip
        rt_hard = _eval_roundtrip_t(rgb, ste=False)[0]
        real_argmax = _segnet_argmax_of_frame(segnet_cpu, rt_hard).cpu().numpy()
        real_dseg = float((real_argmax != L).mean())
        # recon quality (how close the grown texture is to GT-RGB)
        recon_rmse = float(((rgb - gt_t.cpu()) ** 2).mean().sqrt())
        # boundary/interior split
        bmask = np.zeros((H, W), dtype=bool)
        bmask[:, :-1] |= L[:, :-1] != L[:, 1:]
        bmask[:, 1:] |= L[:, :-1] != L[:, 1:]
        bmask[:-1, :] |= L[:-1, :] != L[1:, :]
        bmask[1:, :] |= L[:-1, :] != L[1:, :]
        from scipy import ndimage

        band1 = ndimage.binary_dilation(bmask, iterations=1)
        boundary_flip = float((real_argmax[band1] != L[band1]).mean()) if band1.any() else float("nan")
        interior = ~band1
        interior_flip = (
            float((real_argmax[interior] != L[interior]).mean()) if interior.any() else float("nan")
        )

    rule_pc = gen.rule_param_count()
    bytes_info = nca_texture_bytes(rule_pc, latent_dim)
    rate_amort = rate_from_total_bytes(bytes_info["total_600_amortized_bytes"])
    s_amort = 100 * real_dseg + math.sqrt(10 * HELD_POSE) + rate_amort

    return {
        "hidden": int(hidden),
        "n_channels": int(n_channels),
        "latent_dim": int(latent_dim),
        "n_steps": int(n_steps),
        "rule_param_count": rule_pc,
        "geometric_dseg": geo_dseg,
        "realized_dseg": real_dseg,
        "recon_rmse": recon_rmse,
        "boundary_band_flip": boundary_flip,
        "interior_flip": interior_flip,
        "final_ce_seg": losses[-1][0] if losses else None,
        "final_recon": losses[-1][1] if losses else None,
        "ce_seg_first": losses[0][0] if losses else None,
        "bytes": bytes_info,
        "rate_amortized": rate_amort,
        "S_projected_amortized": s_amort,
        "realized_dseg_x_frontier": real_dseg / FRONTIER_DSEG,
        "realized_dseg_x_gt_rgb_floor": real_dseg / GT_RGB_ROUNDTRIP_DSEG,
        "realized_dseg_x_flat_partition_nca": real_dseg / FLAT_PARTITION_NCA_FLOOR,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/nca_texture_dseg_feasibility_gate"))
    ap.add_argument("--hiddens", default="64,128,256", help="comma-sep NCA hidden widths (rule-size sweep)")
    ap.add_argument("--n-channels", type=int, default=16)
    ap.add_argument("--latent-dim", type=int, default=32, help="per-frame latent (the per-frame stored bytes)")
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--seed-factor", type=int, default=16)
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--recon-w", type=float, default=1.0, help="weight on the GT-RGB recon anchor")
    ap.add_argument(
        "--grad-norm", action=argparse.BooleanOptionalAction, default=True,
        help="canonical Growing-NCA per-param gradient normalization (stabilizes the deep unroll)",
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
    result_json = REPO / ".omx/research" / f"nca_texture_dseg_feasibility_gate_{_now()}.json"

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] prior rows: {list(state.get('rows', {}).keys())}", flush=True)
    state.setdefault("rows", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")
    segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)

    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"].numpy()
    n_frames = min(args.n_frames, seg_targets.shape[0])
    gt_frames = {}
    for pidx, _f0, f1 in decode_gt_frame1_pairs(n_pairs=n_frames):
        t = torch.from_numpy(f1).float().permute(2, 0, 1).unsqueeze(0)
        rs = F.interpolate(t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_frames[pidx] = rs[0].permute(1, 2, 0).numpy()
        if len(gt_frames) >= n_frames:
            break

    if args.timing_smoke:
        print("[timing-smoke] hidden=128, 1 frame, 30 iters ...", flush=True)
        sa = argparse.Namespace(**vars(args))
        sa.iters = 30
        r = optimize_nca_texture(
            seg_targets[0], gt_frames.get(0), segnet_cpu, segnet_train, 128,
            args.n_channels, args.latent_dim, args.n_steps, args.seed_factor, sa,
        )
        spi = r["elapsed_s"] / sa.iters
        print(
            f"[timing-smoke] {r['elapsed_s']:.0f}s/30it -> ~{spi:.2f}s/it ; full ~"
            f"{spi*args.iters*len(args.hiddens.split(','))*n_frames/60:.1f} min",
            flush=True,
        )
        print(
            f"   geo_seg={r['geometric_dseg']:.5f} realized={r['realized_dseg']:.5f} "
            f"recon_rmse={r['recon_rmse']:.1f} bnd={r['boundary_band_flip']:.3f} "
            f"rule_pc={r['rule_param_count']} rate={r['rate_amortized']:.5f} S~{r['S_projected_amortized']:.3f}",
            flush=True,
        )
        return 0

    hiddens = [int(x) for x in args.hiddens.split(",") if x.strip()]
    for hid in hiddens:
        key = f"h{hid}"
        if key in state["rows"]:
            print(f"[resume] {key} done; skip", flush=True)
            continue
        print(f"\n=== TEXTURE-NCA {key} (hidden={hid}, C={args.n_channels}, "
              f"latent={args.latent_dim}, N={args.n_steps}) ===", flush=True)
        per_frame = []
        for fidx in range(n_frames):
            r = optimize_nca_texture(
                seg_targets[fidx], gt_frames.get(fidx), segnet_cpu, segnet_train, hid,
                args.n_channels, args.latent_dim, args.n_steps, args.seed_factor, args,
            )
            per_frame.append(r)
            print(
                f"   frame{fidx}: rule_pc={r['rule_param_count']:6d} geo_seg={r['geometric_dseg']:.5f} "
                f"realized={r['realized_dseg']:.5f} recon_rmse={r['recon_rmse']:.1f} "
                f"bnd={r['boundary_band_flip']:.3f} rate={r['rate_amortized']:.5f} "
                f"S~{r['S_projected_amortized']:.3f} {r['elapsed_s']:.0f}s",
                flush=True,
            )

        def avg(k, _pf=per_frame):
            vals = [p[k] for p in _pf if p[k] is not None and not (isinstance(p[k], float) and math.isnan(p[k]))]
            return sum(vals) / len(vals) if vals else float("nan")

        row = {
            "hidden": hid,
            "n_channels": args.n_channels,
            "latent_dim": args.latent_dim,
            "n_steps": args.n_steps,
            "n_frames": len(per_frame),
            "avg_rule_param_count": avg("rule_param_count"),
            "avg_geometric_dseg": avg("geometric_dseg"),
            "avg_realized_dseg": avg("realized_dseg"),
            "avg_recon_rmse": avg("recon_rmse"),
            "avg_boundary_band_flip": avg("boundary_band_flip"),
            "avg_interior_flip": avg("interior_flip"),
            "avg_rate_amortized": avg("rate_amortized"),
            "avg_S_projected_amortized": avg("S_projected_amortized"),
            "best_realized_dseg": min(p["realized_dseg"] for p in per_frame),
            "per_frame": per_frame,
        }
        state["rows"][key] = row
        save_state()
        print(
            f"   -> {key}: avg_realized={row['avg_realized_dseg']:.5f} best={row['best_realized_dseg']:.5f} "
            f"({row['avg_realized_dseg']/FRONTIER_DSEG:.1f}x frontier, "
            f"{row['avg_realized_dseg']/GT_RGB_ROUNDTRIP_DSEG:.1f}x GT-RGB floor) "
            f"avg_rate={row['avg_rate_amortized']:.5f} avg_S~{row['avg_S_projected_amortized']:.3f}",
            flush=True,
        )

    rows = state["rows"]
    if not rows:
        print("[error] no rows", flush=True)
        return 1
    best_realized = min(r["best_realized_dseg"] for r in rows.values())
    best_avg_realized = min(r["avg_realized_dseg"] for r in rows.values())
    best_S = min(r["avg_S_projected_amortized"] for r in rows.values())
    best_row = min(rows.values(), key=lambda r: r["avg_S_projected_amortized"])
    cheap_rows = [r for r in rows.values() if r["avg_rate_amortized"] < 0.05]
    cheap_low = [r for r in cheap_rows if r["best_realized_dseg"] < GREEN_DSEG_THRESHOLD]
    sub015 = [r for r in rows.values() if r["avg_rate_amortized"] < 0.05 and r["avg_S_projected_amortized"] < 0.15]

    if sub015 and cheap_low:
        verdict = "GREEN_TEXTURE_NCA_REACHES_SUB015_BYTE_CHEAP"
    elif best_realized < GREEN_DSEG_THRESHOLD:
        verdict = "AMBER_LOW_DSEG_BUT_NOT_SUB015_OR_BYTE_CHEAP"
    elif best_realized < 0.5 * FLAT_PARTITION_NCA_FLOOR:
        verdict = "AMBER_TEXTURE_BEATS_FLAT_PARTITION_BUT_NOT_GREEN"
    else:
        verdict = "RED_TEXTURE_NCA_CAPS_ABOVE_SUB015"

    improvement_vs_flat = FLAT_PARTITION_NCA_FLOOR / max(best_realized, 1e-9)
    payload = {
        "schema": "nca_texture_dseg_feasibility_gate.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_nca_texture_dseg_feasibility_gate.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "the_question": (
            "Can a few-KB iterated NCA rule grow CONTINUOUS RGB TEXTURE (replacing the 161KB decoder) whose "
            "realized d_seg (through the real SegNet + roundtrip) gives S < 0.15? The premise: GT-RGB through "
            "the roundtrip realizes d_seg 0.00022 (frontier-grade) — continuous texture CAN survive; the bet "
            "is whether a few-KB rule reproduces it."
        ),
        "premise_gt_rgb_roundtrip_dseg": GT_RGB_ROUNDTRIP_DSEG,
        "thresholds": {
            "frontier_dseg": FRONTIER_DSEG,
            "sub015_dseg": SUB015_DSEG,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
            "flat_partition_nca_floor": FLAT_PARTITION_NCA_FLOOR,
            "byte_cheap_rate_threshold": 0.05,
            "S_target": 0.15,
        },
        "method": {
            "representation": (
                "Continuous-texture NCA: fixed depthwise perception -> 1x1 conv 3C->hidden (ReLU) -> 1x1 "
                "conv hidden->C (zero-init) = the RULE, iterated N steps from a per-frame LATENT-conditioned "
                "seed -> 1x1 readout C->3 RGB (the generator IS the renderer; CONTINUOUS RGB, not a class "
                "partition). Stored bytes = rule + readout + latent_proj (once) + per-frame latent."
            ),
            "fit_objective": "CE(real-SegNet(roundtrip(grown-RGB)), L*) [d_seg DIRECT] + recon_w*MSE(grown-RGB, GT-RGB)",
            "dseg_metric": "realized argmax-flip-rate of the grown RGB through the EXACT roundtrip vs L* (authority)",
            "train_device": args.train_device,
            "authority_device": "cpu",
            "n_channels": args.n_channels,
            "latent_dim": args.latent_dim,
            "n_steps": args.n_steps,
            "n_frames": n_frames,
        },
        "rows": rows,
        "best_realized_dseg": best_realized,
        "best_avg_realized_dseg": best_avg_realized,
        "best_realized_dseg_x_frontier": best_realized / FRONTIER_DSEG,
        "best_realized_dseg_x_gt_rgb_floor": best_realized / GT_RGB_ROUNDTRIP_DSEG,
        "improvement_vs_flat_partition_nca": improvement_vs_flat,
        "best_S_projected_amortized": best_S,
        "best_row_key": f"h{best_row['hidden']}",
        "n_byte_cheap_rows": len(cheap_rows),
        "n_sub015_rows": len(sub015),
        "byte_escape_real": bool(cheap_rows),
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: realized d_seg of the grown CONTINUOUS RGB through the real SegNet + exact "
            "roundtrip. The continuous-texture variant (the operator's true reframe) is tested here, distinct "
            "from the sister flat-partition NCA. Reports best-per-frame to guard false-GREEN/false-RED."
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
        f"{best_realized/GT_RGB_ROUNDTRIP_DSEG:.1f}x GT-RGB floor, "
        f"{improvement_vs_flat:.1f}x better than flat-partition NCA)",
        flush=True,
    )
    print(f"[best projected S] {best_S:.4f}", flush=True)
    print(f"[VERDICT] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
