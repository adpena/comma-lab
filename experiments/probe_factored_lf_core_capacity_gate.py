#!/usr/bin/env python3
"""EARLY CAPACITY FALSIFICATION GATE — can a SMALL LF-only d_seg-core floor d_seg << 0.0022?

The RANK-1 factored concentrated-saliency vehicle (capstone #78, surviving path) bets
that a SMALL high-precision LF/structure CORE that renders ONLY the SegNet-decision-band
low-frequency class geometry — spending ALL its capacity on d_seg — can reach
frontier-grade d_seg (~0.0003-0.0006) at FEW bytes, while a separate cheap HF periphery
carries the d_seg-blind detail + pose.

THE GATE (the MVP-first decisive measurement BEFORE the multi-day build):
  Can a small LF-only core reach d_seg << bc20's 0.0022 floor (toward ~0.0003-0.0006)
  at small bytes?
    GREEN -> the factored vehicle escapes the capacity<->rate tension; spec the full build.
    RED   -> the small LF-only core ALSO walls at ~0.002 (the same capacity wall) -> the
             factorization does not escape it -> route to RANK-4 geometry-anchored core
             (or report sub-0.15 needs a non-learned-decoder representation).

WHY THIS IS THE RIGHT TEST (NO-FAKE, faithful-to-the-question):
  The bc20 basin (83,356 params) is a DENSE decoder that floors d_seg at ~0.0022 while
  ALSO carrying recon+pose. The factored bet is that if a core renders ONLY the LF class
  structure (not the HF texture), a SMALL such core can floor d_seg LOWER per byte. So the
  gate trains LF-ONLY cores (narrow-channel HNeRV decoders) DIRECTLY against the d_seg
  objective (CE on GT argmax through the exact eval roundtrip) and measures the achievable
  d_seg-vs-byte trajectory. If even a from-scratch d_seg-dedicated small core cannot beat
  the dense basin's 0.0022 (let alone reach ~0.0004), the LF factorization does NOT buy the
  capacity escape -> RED. If it descends well below 0.0022 -> GREEN.

  This is STRICTER than the basin: the basin spent capacity on recon+pose too; here the core
  spends 100% on d_seg. If a 100%-d_seg small core STILL walls, the factorization is dead.

CARMACK MVP-FIRST: this is a BOUNDED train, NOT the full multi-day build. Early
falsification: a stretched-exponential fit to the d_seg trajectory projects the asymptote;
if the projected floor cannot reach << 0.0022 within the budget, STOP and report RED (the
gate working).

ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
MPS = fp32 GRADIENT only (NEVER authority). d_seg measured on the CPU-authority frozen
SegNet over the EXACT eval path (argmax-flip-rate vs GT), on the available GT pairs (a
faithful-DEFINITION proxy: the exact metric on a subset of pairs). EMA shadow w/ warmup
decay (the capstone EMA-shadow-lag lesson). $0, no paid GPU, no PR, no self-promote.

Resumable: writes a JSON state checkpoint after each LF-core size; re-run resumes from the
last completed size. Checkpoint-on-full-train-completion per size.
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

BASIN_CKPT = (
    REPO
    / "experiments/results/torch_vehicle_full_mps_basin_bc20_n600"
    / "torch_vehicle_checkpoint_state.pt"
)
GT_TARGETS = REPO / "experiments/results/capstone_gt_targets_cache/gt_targets_n100.pt"

# --- the campaign's measured anchors (advisory, for the GREEN/RED thresholds) ----
BC20_DSEG_FLOOR = 0.0022  # dense bc20 basin d_seg floor (capacity-limited, the wall)
FRONTIER_DSEG = 0.0003  # frontier-grade d_seg (the target the LF core must approach)
# GREEN if the LF core reaches well below the bc20 wall (toward frontier grade).
GREEN_DSEG_THRESHOLD = 0.0012  # << 0.0022, decisively past the wall, heading to frontier
B0 = 37_545_489  # contest archive normalizer


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _eval_roundtrip(decoded_pair_bthwc):
    """The contest uint8 eval roundtrip (driver.py): camera-res bicubic-up to
    874x1164 -> bilinear-down to 384x512 -> clamp -> round. Input/return (B,2,3,384,512)."""
    import torch.nn.functional as F

    b = decoded_pair_bthwc.shape[0]
    flat = decoded_pair_bthwc.reshape(b * 2, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    clamped = down.reshape(b, 2, 3, 384, 512).clamp(0, 255)
    return (clamped + (clamped.round() - clamped).detach()).detach()


def _measure_dseg(decoder, latents, segnet, seg_targets, frame_indices, device):
    """REAL d_seg = mean argmax-flip-rate vs GT hard labels over a frame subset.

    EXACT eval path: decoder PAIR [rgb_0, rgb_1] -> uint8 eval roundtrip -> SegNet scores
    the LAST frame (x[:, -1] = rgb_1) -> argmax vs GT target (SegNet argmax on GT, last
    frame). Subset of pairs => faithful-definition proxy for the full-600 exact d_seg.
    """
    import einops
    import torch

    flips = []
    with torch.no_grad():
        for i in frame_indices:
            out = decoder(latents[i : i + 1])  # (1,2,3,384,512) float [0,255]
            rt = _eval_roundtrip(out)  # uint8 roundtrip
            x = einops.rearrange(rt, "b t c h w -> b t h w c").to(device)
            x = einops.rearrange(x, "b t h w c -> b t c h w")
            seg_in = segnet.preprocess_input(x)  # uses x[:, -1] = rgb_1
            logits = segnet(seg_in)  # (1, C, H, W)
            pred = logits.argmax(dim=1)[0]  # (H,W)
            tgt = seg_targets[i].to(pred.device)
            flips.append((pred != tgt).float().mean().item())
    return sum(flips) / len(flips)


def _stretched_exp_fit(eps, vals):
    """Fit d_seg(ep) ~ a*exp(-(ep/tau)^beta) + c, return (a, tau, beta, c, asymptote).

    A coarse grid-search fit (the campaign's measured shape; 16x better than power law).
    `c` is a floor offset (>=0). The asymptote = c (the projected floor as ep->inf).
    Robust to short series: falls back to a 2-point exp decay if <4 points.
    """
    import numpy as np

    eps = np.asarray(eps, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 3:
        return None
    v0 = float(vals[0])
    vlast = float(vals[-1])
    # grid over (tau, beta, c); a = v0 - c at ep=0 (exp term = 1 at ep=0 only if first ep~0)
    best = None
    c_grid = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.0010, 0.0015, 0.0020]
    tau_grid = [200, 500, 1000, 2000, 4000, 8000, 16000]
    beta_grid = [0.5, 0.7, 0.86, 1.0, 1.3]
    ep0 = float(eps[0])
    for c in c_grid:
        if c >= vlast:  # floor must be below the last observed value
            continue
        for tau in tau_grid:
            for beta in beta_grid:
                # a from the first point: v0 = a*exp(-(ep0/tau)^beta) + c
                denom = math.exp(-((ep0 / tau) ** beta))
                if denom <= 1e-9:
                    continue
                a = (v0 - c) / denom
                if a <= 0:
                    continue
                pred = a * np.exp(-((eps / tau) ** beta)) + c
                sse = float(np.sum((pred - vals) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, a, tau, beta, c)
    if best is None:
        return None
    _sse, a, tau, beta, c = best
    return {"a": a, "tau": tau, "beta": beta, "c": c, "asymptote": c, "sse": _sse}


def build_lf_core(base_channels, latent_dim=28):
    """LF/structure core = a narrow-channel HNeRV decoder.

    Renders the coarse class-region geometry through the same 7-stage upsample path
    (6x8 -> 384x512) but with FEW channels, so its capacity is small and byte-cheap.
    This IS the design memo's RANK-1 LF core ("narrow channels, renders the coarse
    class-region geometry"). We reuse the vendored HNeRVDecoder architecture (the
    canonical, exact-eval-compatible renderer) at a narrow width -- UNIQUE choice of
    width, ADOPT_CANONICAL architecture (the eval-roundtrip + preprocess contract must
    match the contest exactly; forking the renderer would break that faithfulness).
    """
    from tac.torch_vehicle.vendored_imports import import_vendored

    model = import_vendored("model")
    dec = model.HNeRVDecoder(
        latent_dim=latent_dim, base_channels=base_channels, eval_size=(384, 512)
    )
    return dec


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


def _lf_core_byte_estimate(n_params):
    """Advisory byte estimate for the LF core stored at high precision.

    The basin's 83,356 params -> ~89KB archive (fp ~ 10.7 bits/param effective after
    brotli on the basin). We use the basin's measured params->bytes ratio as the LF
    core estimate at the SAME (high) precision -- the LF core stays high-precision by
    design (it carries d_seg). 89000 bytes / 83356 params = ~1.068 bytes/param.
    """
    basin_bytes = 89_000.0
    basin_params = 83_356.0
    return n_params * (basin_bytes / basin_params)


class _EMA:
    """Weight EMA with warmup decay (the capstone EMA-shadow-lag fix).

    warmup-decay: decay_t = min(decay, (1+t)/(10+t)) so the shadow tracks the live
    weights early (short runs) instead of freezing near init. Shadow is the inference
    checkpoint (we measure d_seg on the shadow).
    """

    def __init__(self, module, decay=0.999):
        import torch

        self.decay = decay
        self.t = 0
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}
        self._torch = torch

    def update(self, module):
        self.t += 1
        d = min(self.decay, (1.0 + self.t) / (10.0 + self.t))
        for k, v in module.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, module):
        module.load_state_dict({k: v.clone() for k, v in self.shadow.items()})


def train_one_lf_core(
    base_channels,
    latents,
    seg_targets,
    n_gt,
    gt_dseg_frames,
    args,
):
    """Train ONE LF-only core from scratch against d_seg (CE on GT argmax through the
    exact eval roundtrip) + EMA; measure the d_seg trajectory; early-falsify.

    Returns a dict: params, byte_estimate, d_seg curve [(ep, dseg)], best d_seg, fit,
    early_falsified flag.
    """
    import einops
    import torch
    import torch.nn.functional as F

    from tac.scorer import load_default_segnet

    tdev = torch.device(args.train_device)
    # MPS scorer-compat patch (BN-contiguous) so the SegNet gradient path runs on MPS.
    if args.train_device == "mps":
        try:
            from tac.torch_mps_compat import patch_scorer_for_mps

            patch_scorer_for_mps()
        except Exception as e:  # pragma: no cover - best-effort
            print(f"      [warn] patch_scorer_for_mps failed ({e}); continuing")

    dec = build_lf_core(base_channels).to(tdev).train()
    n_params = _n_params(dec)
    byte_est = _lf_core_byte_estimate(n_params)
    print(
        f"   [LF core bc={base_channels}] params={n_params} "
        f"byte_est~{byte_est/1000:.1f}KB rate_est~{25*byte_est/B0:.4f}"
    )

    # Learnable per-pair latents for the LF core (small: n_gt x latent_dim). The core
    # renders from a latent; we co-train the latents (like the basin). Init from the
    # basin's EMA latents (warm-start the latent geometry, NOT the decoder).
    lat = torch.nn.Parameter(latents[:n_gt].clone().to(tdev))

    seg_t = seg_targets.to(tdev)
    segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)
    segnet_cpu = load_default_segnet(str(UPSTREAM), device="cpu")  # authority eval

    opt = torch.optim.AdamW(
        [*dec.parameters(), lat], lr=args.lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    ema = _EMA(dec, decay=args.ema_decay)

    def _scored_logits(idx):
        """LF core PAIR -> differentiable uint8 roundtrip (STE round) -> SegNet on the
        scored LAST frame (rgb_1). Matches the exact eval path. Returns (1,C,H,W)."""
        out = dec(lat[idx : idx + 1])  # (1,2,3,384,512)
        flat = out.reshape(2, 3, 384, 512)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
        rt = down.reshape(1, 2, 3, 384, 512)
        clamped = rt.clamp(0, 255)
        ste = clamped + (clamped.round() - clamped).detach()  # STE round
        x = einops.rearrange(ste, "b t c h w -> b t h w c").contiguous()
        x = einops.rearrange(x, "b t h w c -> b t c h w").contiguous()
        seg_in = segnet_train.preprocess_input(x)  # uses x[:, -1] = rgb_1
        return segnet_train(seg_in)

    curve = []
    best_dseg = float("inf")
    best_shadow = None
    t0 = time.time()
    early_falsified = False
    bs = args.batch_size

    for ep in range(args.epochs):
        dec.train()
        opt.zero_grad(set_to_none=True)
        bi = [(ep * bs + j) % n_gt for j in range(bs)]
        ce_loss = torch.zeros((), device=tdev)
        for i in bi:
            logits = _scored_logits(i)  # (1,C,H,W)
            ce_loss = ce_loss + F.cross_entropy(logits, seg_t[i : i + 1])
        ce_loss = ce_loss / len(bi)
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_([*dec.parameters(), lat], 1.0)
        opt.step()
        sched.step()
        ema.update(dec)

        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            # measure d_seg on the EMA SHADOW (the inference checkpoint), CPU authority
            shadow_dec = build_lf_core(base_channels)
            ema.copy_to(shadow_dec)
            shadow_dec.eval()
            lat_cpu = lat.detach().cpu()
            dseg = _measure_dseg(
                shadow_dec, lat_cpu, segnet_cpu, seg_targets, gt_dseg_frames, "cpu"
            )
            curve.append({"ep": ep, "dseg": dseg, "ce": float(ce_loss.item())})
            if dseg < best_dseg:
                best_dseg = dseg
                best_shadow = {k: v.detach().cpu().clone() for k, v in shadow_dec.state_dict().items()}
            el = time.time() - t0
            print(
                f"      ep{ep:4d} ce={ce_loss.item():.4f} d_seg={dseg:.5f} "
                f"(best {best_dseg:.5f}) {el:.0f}s"
            )

            # --- EARLY FALSIFICATION (the gate working) ---------------------
            # Once we have >=4 eval points past the warmup, fit the stretched-exp
            # and project the asymptote. If the projected floor cannot reach the
            # GREEN threshold AND d_seg is no longer descending meaningfully, STOP.
            if len(curve) >= args.min_fit_points and ep >= args.warmup_epochs:
                eps_arr = [p["ep"] for p in curve]
                vals_arr = [p["dseg"] for p in curve]
                fit = _stretched_exp_fit(eps_arr, vals_arr)
                # recent slope (last 3 eval points): is it still descending?
                recent = vals_arr[-3:]
                rel_drop = (recent[0] - recent[-1]) / max(recent[0], 1e-9)
                proj_floor = fit["asymptote"] if fit else best_dseg
                # FALSIFY if: projected floor is above the wall (>= ~0.0018, i.e. not
                # decisively past bc20's 0.0022) AND we've stopped descending (rel drop
                # < 3% over the last 3 evals). That's the "walls at the same place" case.
                walled = (proj_floor >= 0.0018) and (rel_drop < 0.03)
                if walled and ep >= args.min_falsify_epoch:
                    early_falsified = True
                    print(
                        f"      [EARLY FALSIFY] projected floor {proj_floor:.5f} >= 0.0018 "
                        f"(near bc20 wall {BC20_DSEG_FLOOR}) AND recent rel-drop {rel_drop:.1%} < 3% "
                        f"-> the small LF core WALLS at the same place. STOP (gate working)."
                    )
                    break

    eps_arr = [p["ep"] for p in curve]
    vals_arr = [p["dseg"] for p in curve]
    fit = _stretched_exp_fit(eps_arr, vals_arr)
    return {
        "base_channels": base_channels,
        "n_params": n_params,
        "byte_estimate": byte_est,
        "rate_estimate": 25 * byte_est / B0,
        "curve": curve,
        "best_dseg_subset": best_dseg,
        "stretched_exp_fit": fit,
        "projected_floor": (fit["asymptote"] if fit else best_dseg),
        "early_falsified": early_falsified,
        "elapsed_s": round(time.time() - t0, 1),
        "best_shadow_available": best_shadow is not None,
    }, best_shadow


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "experiments/results/factored_lf_core_capacity_gate"),
    )
    ap.add_argument(
        "--core-sizes",
        default="8,12",
        help="comma-sep base_channels for the LF cores to test (small first)",
    )
    ap.add_argument("--epochs", type=int, default=1200, help="bounded budget per core")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-dseg-frames", type=int, default=24)
    ap.add_argument("--warmup-epochs", type=int, default=200)
    ap.add_argument("--min-fit-points", type=int, default=5)
    ap.add_argument("--min-falsify-epoch", type=int, default=400)
    ap.add_argument("--train-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument(
        "--timing-smoke",
        action="store_true",
        help="run a tiny 20-epoch single-core pass to calibrate seconds/epoch then exit",
    )
    args = ap.parse_args()

    import torch

    from tac.torch_vehicle.vendored_imports import import_vendored

    import_vendored("model")  # ensures sys.path + challenge root resolved

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "gate_state.json"
    result_json = REPO / ".omx/research" / f"factored_lf_core_capacity_gate_{_now()}.json"

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior cores: {list(state.get('cores', {}).keys())}")
    state.setdefault("cores", {})

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    # --- load REAL latents + GT targets (CPU authority) ----------------------
    ck = torch.load(BASIN_CKPT, map_location="cpu", weights_only=False)
    latents = ck["ema_latents"]  # (600, 28)
    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"]  # (100, 384, 512) int64
    n_gt = int(seg_targets.shape[0])
    dseg_frames = [
        round(i * (n_gt - 1) / max(1, args.n_dseg_frames - 1))
        for i in range(args.n_dseg_frames)
    ]

    # --- TIMING SMOKE (calibrate the bounded budget) -------------------------
    if args.timing_smoke:
        print("[timing-smoke] 20-epoch single-core pass on bc=8 ...")
        smoke_args = argparse.Namespace(**vars(args))
        smoke_args.epochs = 20
        smoke_args.eval_every = 10
        smoke_args.warmup_epochs = 1000  # disable falsify in smoke
        smoke_args.min_falsify_epoch = 1000
        res, _ = train_one_lf_core(8, latents, seg_targets, n_gt, dseg_frames, smoke_args)
        spe = res["elapsed_s"] / max(1, res["curve"][-1]["ep"] + 1)
        print(
            f"[timing-smoke] {res['elapsed_s']:.0f}s for {res['curve'][-1]['ep']+1} ep "
            f"-> ~{spe:.2f} s/ep ; projected {args.epochs} ep ~ {spe*args.epochs/60:.1f} min/core"
        )
        return 0

    core_sizes = [int(x) for x in args.core_sizes.split(",") if x.strip()]

    for bc in core_sizes:
        key = f"bc{bc}"
        if key in state["cores"]:
            print(f"[resume] core {key} already done; skipping")
            continue
        print(f"\n=== TRAIN LF CORE {key} ===")
        res, best_shadow = train_one_lf_core(
            bc, latents, seg_targets, n_gt, dseg_frames, args
        )
        if best_shadow is not None:
            torch.save(best_shadow, out_dir / f"lf_core_{key}_best_shadow.pt")
            res["best_shadow_path"] = str((out_dir / f"lf_core_{key}_best_shadow.pt").relative_to(REPO))
        state["cores"][key] = res
        save_state()
        print(
            f"   -> {key}: best d_seg(subset)={res['best_dseg_subset']:.5f} "
            f"projected_floor={res['projected_floor']:.5f} "
            f"rate_est={res['rate_estimate']:.4f} "
            f"{'EARLY-FALSIFIED' if res['early_falsified'] else 'completed'}"
        )

    # =====================================================================
    # VERDICT
    # =====================================================================
    cores = state["cores"]
    best_overall = min(
        (c["best_dseg_subset"] for c in cores.values()), default=float("inf")
    )
    best_proj = min((c["projected_floor"] for c in cores.values()), default=float("inf"))
    # the small-core S projection (LF core d_seg + held trunk pose + LF core bytes +
    # cheap int4 HF periphery estimate). pose held at the basin trunk value 0.00034.
    held_pose = 0.00034
    # best-S core: pick the core minimizing projected S
    s_rows = []
    for key, c in cores.items():
        dseg = c["best_dseg_subset"]
        # full vehicle S model: LF core (high-prec) + HF periphery (int4, ~0.70 of a
        # dense decoder's params at q=0.36). We model the HF periphery byte cost as the
        # design memo's estimate: B ~ (B_LF + 0.70*B_dense*0.36). Use the basin dense
        # bytes (89KB) for B_dense.
        b_lf = c["byte_estimate"]
        b_hf = 0.70 * 89_000.0 * 0.36  # int4 HF periphery (design-memo model)
        b_total = b_lf + b_hf
        s_proj = 100 * dseg + math.sqrt(10 * held_pose) + 25 * b_total / B0
        s_rows.append(
            {
                "core": key,
                "lf_core_dseg_subset": dseg,
                "projected_floor": c["projected_floor"],
                "b_lf": b_lf,
                "b_hf_int4_model": b_hf,
                "b_total_model": b_total,
                "rate_total_model": 25 * b_total / B0,
                "S_projected": s_proj,
                "early_falsified": c["early_falsified"],
            }
        )
    s_rows.sort(key=lambda r: r["S_projected"])
    best_s_row = s_rows[0] if s_rows else None

    # --- capacity extrapolation (power-law over the measured cores) ---------
    # d_seg ~ A * params^-k from >=2 measured (params, d_seg) points. Tells us how
    # much capacity a small LF core would need to reach the wall / frontier (the
    # CRUX: if it needs a HUGE core, the factorization forfeits the rate win).
    capacity_extrapolation = None
    pts = sorted(
        ((c["n_params"], c["best_dseg_subset"]) for c in cores.values()),
        key=lambda t: t[0],
    )
    if len(pts) >= 2:
        (p0, d0), (p1, d1) = pts[0], pts[-1]
        if p1 > p0 and d0 > 0 and d1 > 0 and d0 != d1:
            k = (math.log(d0) - math.log(d1)) / (math.log(p1) - math.log(p0))
            a = d0 * (p0**k)
            basin_params = 83356
            targets = {}
            if k > 0:
                for tgt, lbl in [
                    (BC20_DSEG_FLOOR, "bc20_wall"),
                    (0.0006, "sub_015_ish"),
                    (FRONTIER_DSEG, "frontier"),
                ]:
                    need = (a / tgt) ** (1.0 / k)
                    targets[lbl] = {
                        "dseg_target": tgt,
                        "params_needed": need,
                        "x_basin": need / basin_params,
                    }
            capacity_extrapolation = {
                "power_law": {"A": a, "k": k},
                "note": "d_seg ~ A * params^-k from the measured LF cores",
                "params_needed_for_targets": targets,
            }

    # --- VERDICT (MEASUREMENT-FIRST; the fit is a surrogate, never authority) ---
    # The probe's d_seg metric on the EMA shadow over the GT subset is the
    # authority here (advisory CPU; not the exact 600-eval). The stretched-exp
    # `projected_floor` is a SURROGATE that can degenerate (a c=0 / long-tau grid
    # solution yields projected_floor=0.0 even when the MEASURED floor is high).
    # Per NO-FAKE "surrogate != authority", the verdict is driven by the MEASURED
    # floor (best_overall), NOT the projection. A projection can only TIGHTEN a
    # tentative-GREEN when it is non-degenerate AND below the wall AND the measured
    # floor is itself already near the wall.
    green_measured = best_overall < GREEN_DSEG_THRESHOLD
    # a credible projection: non-degenerate asymptote (>0), below threshold, AND
    # the measured floor is within 2x of the wall (so the projection is plausibly
    # the next-few-hundred-epochs floor, not a far extrapolation).
    credible_proj = (
        (0.0 < best_proj < GREEN_DSEG_THRESHOLD)
        and (best_overall <= 2.0 * BC20_DSEG_FLOOR)
    )
    if green_measured:
        verdict = "GREEN_LF_CORE_ESCAPES_CAPACITY_WALL"
    elif credible_proj:
        verdict = "GREEN_TENTATIVE_CREDIBLE_PROJECTION_BELOW_WALL"
    elif best_overall <= 1.5 * BC20_DSEG_FLOOR:
        verdict = "AMBER_LF_CORE_NEAR_WALL_NEEDS_LONGER_TRAIN"
    else:
        verdict = "RED_LF_CORE_WALLS_ABOVE_CAPACITY_LIMIT"

    payload = {
        "schema": "factored_lf_core_capacity_gate.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_factored_lf_core_capacity_gate.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "the_question": (
            "Can a SMALL LF-only d_seg-core floor d_seg << bc20's 0.0022 (toward "
            "~0.0003-0.0006) at small bytes? GREEN -> spec the factored build; RED -> "
            "route to RANK-4 geometry core."
        ),
        "thresholds": {
            "bc20_dseg_wall": BC20_DSEG_FLOOR,
            "frontier_dseg_target": FRONTIER_DSEG,
            "green_dseg_threshold": GREEN_DSEG_THRESHOLD,
        },
        "vehicle_basis": {
            "basin_ckpt": str(BASIN_CKPT.relative_to(REPO)),
            "basin_base_channels": 20,
            "basin_n_params": 83356,
            "basin_dseg_floor": BC20_DSEG_FLOOR,
            "n_gt_pairs": n_gt,
            "n_dseg_frames": len(dseg_frames),
        },
        "method": {
            "lf_core": "narrow-channel HNeRV decoder (canonical exact-eval renderer), trained 100% on d_seg",
            "train_objective": "CE on GT argmax through the EXACT eval roundtrip (scored last frame rgb_1)",
            "dseg_metric": "argmax-flip-rate vs GT over subset of GT pairs (exact metric def, faithful proxy)",
            "ema": "warmup-decay shadow (capstone EMA-shadow-lag fix); d_seg measured on shadow",
            "train_device": args.train_device,
            "authority_device": "cpu",
            "epochs_budget": args.epochs,
            "early_falsification": "stretched-exp asymptote projection + recent-slope (gate working)",
        },
        "cores": cores,
        "best_lf_core_dseg_subset": best_overall,
        "best_projected_floor_SURROGATE_not_authority": best_proj,
        "best_lf_core_dseg_x_bc20_wall": best_overall / BC20_DSEG_FLOOR,
        "best_lf_core_dseg_x_frontier": best_overall / FRONTIER_DSEG,
        "capacity_extrapolation": capacity_extrapolation,
        "S_projection_rows": s_rows,
        "best_S_projected": best_s_row,
        "verdict": verdict,
        "verdict_basis": (
            "MEASUREMENT-FIRST: driven by the MEASURED EMA-shadow d_seg floor "
            "(best_lf_core_dseg_subset), NOT the stretched-exp projection (a "
            "surrogate that degenerates to 0.0 on a c=0/long-tau grid solution)."
        ),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    state["final_verdict"] = verdict
    state["result_json"] = str(result_json.relative_to(REPO))
    save_state()
    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}")
    print(f"[best LF core d_seg(subset)] {best_overall:.5f} (wall {BC20_DSEG_FLOOR})")
    if best_s_row:
        print(
            f"[best projected S] {best_s_row['S_projected']:.4f} "
            f"(core {best_s_row['core']}, rate {best_s_row['rate_total_model']:.4f})"
        )
    print(f"[VERDICT] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
