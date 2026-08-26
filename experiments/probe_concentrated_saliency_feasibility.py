#!/usr/bin/env python3
"""$0 FEASIBILITY PROBE — can d_seg-criticality be CONCENTRATED into a small core?

The capstone (#78) "concentrated-saliency vehicle" design rests on ONE load-bearing
assumption: that a DENSE decoder's d_seg-criticality (the per-weight margin-saliency)
can be PUSHED into a small, designated CORE set of weight tensors WITHOUT raising
d_seg — so the d_seg-blind PERIPHERY can then be shed/coarsened for the rate win.

The whole campaign's flat-saliency finding (frontier 5.5x spread, bc20 6.8x spread,
weight-saliency Gini ~0.27) says the criticality is currently SPREAD. The question
this probe answers is whether that spread is INTRINSIC (the direction is dead) or
merely UNREGULARIZED (a concentration penalty can fix it -> the direction is GO).

THE PROBE (all REAL: real frozen SegNet, real bc20 basin decoder, real GT argmax
targets, real autograd through the detector margin):
  1. BASELINE: measure the bc20 basin's per-weight-tensor margin-saliency
     distribution (Gini, top-k mass, max/min spread) + its d_seg on a frame subset.
  2. CORE/PERIPHERY split: designate a small CORE (the top-k most-d_seg-critical
     weight tensors by baseline saliency) and the PERIPHERY (the rest).
  3. CONCENTRATION SMOKE: short MPS train of the decoder with a regularizer that
     penalizes PERIPHERY margin-saliency (drives criticality OUT of the periphery
     INTO the core) while keeping a CE seg-fidelity term (so d_seg does not blow up).
  4. RE-MEASURE: the weight-saliency Gini/top-k-mass (did concentration go UP?) and
     d_seg (did it hold?). Compute the implied byte-shed-at-fixed-d_seg if the
     now-blind periphery were coarsened to int4.

VERDICT:
  GO  -> concentration achieved (Gini up, periphery saliency mass DOWN) at ~flat d_seg
         => the concentrated-saliency own-vehicle is feasible; spec the build.
  NO-GO -> d_seg-criticality cannot be moved off the periphery without raising d_seg
         => the spread is intrinsic; sub-0.15 needs a NON-dense-decoder representation.

ALL numbers `[contest-CPU advisory]` NON-PROMOTABLE. Exact pointer UNMOVED 0.19110.
MPS = fp32 GRADIENT only (never authority). d_seg measured on the CPU-authority
SegNet over a frame subset (a FAITHFUL-DEFINITION proxy: the exact argmax-flip-rate
metric, on a subset of pairs; NOT the full-600 exact eval). $0, no paid GPU, no PR.

Resumable: writes a JSON checkpoint after each phase; re-run resumes from the last
completed phase.
"""

from __future__ import annotations

import argparse
import json
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


def _gini(xs: list[float]) -> float:
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    return (2.0 * sum((i + 1) * x for i, x in enumerate(xs)) / (n * s)) - (n + 1) / n


def _topk_mass(d: dict[str, float], k: int) -> float:
    vals = sorted(d.values(), reverse=True)
    tot = sum(vals)
    return (sum(vals[:k]) / tot) if tot > 0 else 0.0


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _render_head(decoder, latent_row, head_idx):
    out = decoder(latent_row)  # (1, 2, 3, 384, 512), float [0,255], graph intact
    return out[0, head_idx]  # (3, H, W)


def _weight_saliency(decoder, latents, segnet, frame_indices, topk_margin_fn):
    """Per-weight-tensor ||d(sum margin)/dw|| summed over rgb_0+rgb_1, real autograd.

    Returns dict[name -> float] for `.weight` tensors only (the byte carriers).
    """
    import einops
    import torch

    params = [(n, p) for n, p in decoder.named_parameters() if n.endswith(".weight")]
    for _n, p in params:
        p.requires_grad_(True)
    acc = dict.fromkeys((n for n, _ in params), 0.0)
    for head_idx in (0, 1):
        for i in frame_indices:
            decoder.zero_grad(set_to_none=True)
            frame = _render_head(decoder, latents[i : i + 1], head_idx)  # (3,H,W)
            hwc = frame.permute(1, 2, 0)
            pair = torch.stack([hwc, hwc]).unsqueeze(0)  # (1,2,H,W,3)
            x = einops.rearrange(pair, "b t h w c -> b t c h w")
            seg_in = segnet.preprocess_input(x)
            logits = segnet(seg_in)
            margin = topk_margin_fn(logits)[0]  # (H,W)
            scalar = margin.sum()
            grads = torch.autograd.grad(
                scalar, [p for _n, p in params], retain_graph=False, allow_unused=True
            )
            for (name, _p), g in zip(params, grads, strict=True):
                if g is not None:
                    acc[name] += float(g.detach().norm().item())
    return acc


def _eval_roundtrip(decoded_pair_bthwc):
    """The contest uint8 eval roundtrip (driver.py:1722-1726): camera-res
    bicubic-up to 874x1164 -> bilinear-down to 384x512 -> clamp -> round.
    Input/return (B,2,3,384,512)."""
    import torch.nn.functional as F

    b = decoded_pair_bthwc.shape[0]
    flat = decoded_pair_bthwc.reshape(b * 2, 3, 384, 512)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    clamped = down.reshape(b, 2, 3, 384, 512).clamp(0, 255)
    return (clamped + (clamped.round() - clamped).detach()).detach()


def _measure_dseg(decoder, latents, segnet, seg_targets, frame_indices, device):
    """REAL d_seg = mean argmax-flip-rate vs GT hard labels over a frame subset.

    EXACT eval path (matches driver.py + upstream): build the decoder PAIR
    [rgb_0, rgb_1], apply the uint8 eval roundtrip, SegNet scores the LAST frame
    (x[:, -1] = rgb_1), compare argmax to the GT target (which is the SegNet argmax
    on the GT pair, also last-frame). Subset of pairs => faithful-definition proxy
    for the full-600 exact d_seg.
    """
    import einops
    import torch

    flips = []
    with torch.no_grad():
        for i in frame_indices:
            out = decoder(latents[i : i + 1])  # (1,2,3,384,512) float [0,255]
            rt = _eval_roundtrip(out)  # uint8 roundtrip
            # (1,2,3,H,W) -> (1,2,H,W,3) for preprocess_input contract
            x = einops.rearrange(rt, "b t c h w -> b t h w c").to(device)
            x = einops.rearrange(x, "b t h w c -> b t c h w")
            seg_in = segnet.preprocess_input(x)  # uses x[:, -1] = rgb_1
            logits = segnet(seg_in)  # (1, C, H, W)
            pred = logits.argmax(dim=1)[0]  # (H,W)
            tgt = seg_targets[i].to(pred.device)
            flips.append((pred != tgt).float().mean().item())
    return sum(flips) / len(flips)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO / "experiments/results/concentrated_saliency_feasibility"))
    ap.add_argument("--n-saliency-frames", type=int, default=6, help="frames for saliency accumulation")
    ap.add_argument("--n-dseg-frames", type=int, default=24, help="frames for d_seg measurement")
    ap.add_argument("--core-size", type=int, default=3, help="number of weight tensors in the CORE")
    ap.add_argument("--reg-epochs", type=int, default=150, help="short concentration-train budget")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reg-weight", type=float, default=1.0, help="periphery-saliency penalty weight")
    ap.add_argument("--ce-weight", type=float, default=3.0, help="CE seg-fidelity weight (hold d_seg)")
    ap.add_argument("--train-device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--reg-saliency-frames", type=int, default=4, help="frames per reg step (saliency penalty)")
    args = ap.parse_args()

    import torch

    from tac.margin_saliency_map import _topk_margin
    from tac.scorer import load_default_segnet
    from tac.torch_vehicle.vendored_imports import import_vendored

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "feasibility_state.json"
    result_json = REPO / ".omx/research" / f"concentrated_saliency_feasibility_{_now()}.json"

    state: dict = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded prior phases: {list(state.keys())}")

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    # --- load REAL vehicle + targets + scorer (CPU authority) ----------------
    model = import_vendored("model")
    HNeRVDecoder = model.HNeRVDecoder
    ck = torch.load(BASIN_CKPT, map_location="cpu", weights_only=False)
    latents = ck["ema_latents"]
    gt = torch.load(GT_TARGETS, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"]  # (100, 384, 512) int64 — first 100 pairs
    n_gt = int(seg_targets.shape[0])
    segnet = load_default_segnet(str(UPSTREAM), device="cpu")

    sal_frames = [round(i * (n_gt - 1) / max(1, args.n_saliency_frames - 1)) for i in range(args.n_saliency_frames)]
    dseg_frames = [round(i * (n_gt - 1) / max(1, args.n_dseg_frames - 1)) for i in range(args.n_dseg_frames)]

    def fresh_decoder(sd):
        d = HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
        d.load_state_dict(sd)
        return d

    # =====================================================================
    # PHASE 1 — BASELINE characterization (intrinsic concentration shape)
    # =====================================================================
    if "phase1_baseline" not in state:
        print("[1/3] BASELINE: per-weight-tensor saliency + d_seg on bc20 basin ...")
        t0 = time.time()
        dec = fresh_decoder(ck["ema_decoder"]).eval()
        base_sal = _weight_saliency(dec, latents, segnet, sal_frames, _topk_margin)
        numel = {n_: int(p.numel()) for n_, p in dec.named_parameters() if n_.endswith(".weight")}
        base_dseg = _measure_dseg(dec, latents, segnet, seg_targets, dseg_frames, "cpu")
        ranked = sorted(base_sal.items(), key=lambda kv: kv[1], reverse=True)
        core = [name for name, _ in ranked[: args.core_size]]
        periphery = [name for name, _ in ranked[args.core_size :]]
        core_param_frac = sum(numel[c] for c in core) / sum(numel.values())
        periph_param_frac = sum(numel[p] for p in periphery) / sum(numel.values())
        vals = list(base_sal.values())
        nz = [v for v in vals if v > 0]
        state["phase1_baseline"] = {
            "per_weight_saliency": base_sal,
            "weight_numel": numel,
            "weight_saliency_gini": _gini(vals),
            "weight_saliency_spread_max_over_min": (max(nz) / min(nz)) if nz else None,
            "core_tensors": core,
            "periphery_tensors": periphery,
            "core_param_frac": core_param_frac,
            "periphery_param_frac": periph_param_frac,
            "periphery_saliency_mass_frac": sum(base_sal[p] for p in periphery) / sum(vals),
            "baseline_dseg_subset": base_dseg,
            "n_dseg_frames": len(dseg_frames),
            "elapsed_s": round(time.time() - t0, 1),
        }
        save_state()
        print(
            f"      Gini={state['phase1_baseline']['weight_saliency_gini']:.3f} "
            f"spread={state['phase1_baseline']['weight_saliency_spread_max_over_min']:.2f}x "
            f"d_seg(subset)={base_dseg:.5f}  core={core} ({core_param_frac:.1%} params) "
            f"periphery saliency-mass={state['phase1_baseline']['periphery_saliency_mass_frac']:.1%}"
        )

    # =====================================================================
    # PHASE 2 — CONCENTRATION SMOKE (short MPS train w/ periphery-saliency penalty)
    # =====================================================================
    if "phase2_concentration_train" not in state:
        print(f"[2/3] CONCENTRATION TRAIN: {args.reg_epochs} ep, penalize periphery saliency ...")
        t0 = time.time()
        core = state["phase1_baseline"]["core_tensors"]
        periphery = state["phase1_baseline"]["periphery_tensors"]
        tdev = torch.device(args.train_device)
        try:
            dec = fresh_decoder(ck["ema_decoder"]).to(tdev).train()
        except Exception as e:  # MPS unavailable -> CPU fallback
            print(f"      [warn] {args.train_device} unavailable ({e}); CPU fallback")
            tdev = torch.device("cpu")
            args.train_device = "cpu"
            dec = fresh_decoder(ck["ema_decoder"]).to(tdev).train()
        lat_t = latents.to(tdev)
        seg_t_train = seg_targets.to(tdev)
        # SegNet on train device for the gradient path (fp32; never authority).
        segnet_train = load_default_segnet(str(UPSTREAM), device=args.train_device)
        opt = torch.optim.AdamW(dec.parameters(), lr=args.lr)
        import einops

        from tac.margin_saliency_map import _topk_margin as topk

        periph_params = [p for n_, p in dec.named_parameters() if n_ in periphery]
        reg_frames = [round(i * (n_gt - 1) / max(1, args.reg_saliency_frames - 1)) for i in range(args.reg_saliency_frames)]

        def _scored_logits(latent_idx):
            """Decoder PAIR -> differentiable uint8 roundtrip (STE round) -> SegNet
            on the scored LAST frame (rgb_1). Matches the exact eval path. Returns
            (1,C,H,W) logits WITH graph."""
            import torch.nn.functional as F

            out = dec(lat_t[latent_idx : latent_idx + 1])  # (1,2,3,384,512)
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
        for ep in range(args.reg_epochs):
            opt.zero_grad(set_to_none=True)
            # --- CE seg-fidelity term (hold d_seg) over a small rotating batch ---
            bi = [(ep * 3 + j) % n_gt for j in range(3)]
            ce_loss = torch.zeros((), device=tdev)
            for i in bi:
                logits = _scored_logits(i)  # (1,C,H,W)
                ce_loss = ce_loss + torch.nn.functional.cross_entropy(
                    logits, seg_t_train[i : i + 1]
                )
            ce_loss = ce_loss / len(bi)
            # --- periphery-saliency penalty: ||d(margin)/d(periphery weights)||^2 ---
            # create-graph so the saliency NORM is itself differentiable w.r.t. weights.
            # Use the SAME scored frame as the saliency map (rgb_1, the scored head),
            # so the penalty matches the d_seg-relevant margin gradient exactly.
            ri = reg_frames[ep % len(reg_frames)]
            logits = _scored_logits(ri)
            margin = topk(logits)[0].sum()
            periph_grads = torch.autograd.grad(
                margin, periph_params, create_graph=True, retain_graph=True, allow_unused=True
            )
            periph_sal = torch.zeros((), device=tdev)
            for g in periph_grads:
                if g is not None:
                    periph_sal = periph_sal + (g.contiguous().pow(2).sum())
            # normalize the penalty scale so reg_weight is interpretable
            periph_sal = periph_sal / 1e12
            loss = args.ce_weight * ce_loss + args.reg_weight * periph_sal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            if ep % 25 == 0 or ep == args.reg_epochs - 1:
                curve.append(
                    {
                        "ep": ep,
                        "ce": float(ce_loss.item()),
                        "periph_sal_penalty": float(periph_sal.item()),
                        "loss": float(loss.item()),
                    }
                )
                print(
                    f"      ep{ep:4d} ce={ce_loss.item():.4f} "
                    f"periph_sal={periph_sal.item():.4e} loss={loss.item():.4f}"
                )
        # export trained decoder back to CPU for authority re-measure
        trained_sd = {k: v.detach().cpu() for k, v in dec.state_dict().items()}
        torch.save(trained_sd, out_dir / "concentrated_decoder.pt")
        state["phase2_concentration_train"] = {
            "reg_epochs": args.reg_epochs,
            "lr": args.lr,
            "reg_weight": args.reg_weight,
            "ce_weight": args.ce_weight,
            "train_device": args.train_device,
            "curve": curve,
            "elapsed_s": round(time.time() - t0, 1),
        }
        save_state()

    # =====================================================================
    # PHASE 3 — RE-MEASURE (concentration achieved? d_seg held?)
    # =====================================================================
    if "phase3_remeasure" not in state:
        print("[3/3] RE-MEASURE: concentrated decoder saliency Gini + d_seg ...")
        t0 = time.time()
        trained_sd = torch.load(out_dir / "concentrated_decoder.pt", map_location="cpu", weights_only=False)
        dec = HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(384, 512))
        dec.load_state_dict(trained_sd)
        dec.eval()
        new_sal = _weight_saliency(dec, latents, segnet, sal_frames, _topk_margin)
        new_dseg = _measure_dseg(dec, latents, segnet, seg_targets, dseg_frames, "cpu")
        b = state["phase1_baseline"]
        core = b["core_tensors"]
        periphery = b["periphery_tensors"]
        numel = {k: int(v) for k, v in b["weight_numel"].items()}
        vals = list(new_sal.values())
        nz = [v for v in vals if v > 0]
        new_gini = _gini(vals)
        new_periph_mass = sum(new_sal[p] for p in periphery) / sum(vals)
        # implied byte-shed-at-fixed-d_seg: periphery param fraction is the int4-able
        # mass; shed = periphery_param_frac * (1 - 4/11) for FP11->int4 (a model, not a
        # measurement — flagged advisory). The DECISIVE quantity is whether new_dseg
        # holds vs baseline AND periphery saliency mass dropped.
        fp11_to_int4_frac = 1.0 - 4.0 / 11.0
        implied_byte_shed_frac = b["periphery_param_frac"] * fp11_to_int4_frac
        dseg_delta = new_dseg - b["baseline_dseg_subset"]
        periph_mass_drop = b["periphery_saliency_mass_frac"] - new_periph_mass
        gini_up = new_gini - b["weight_saliency_gini"]
        # VERDICT logic
        concentration_achieved = (gini_up > 0.02) and (periph_mass_drop > 0.02)
        dseg_held = dseg_delta < 0.0008  # within ~+0.0008 (the campaign's break-even band)
        if concentration_achieved and dseg_held:
            verdict = "GO_CONCENTRATION_FEASIBLE"
        elif (not concentration_achieved) and dseg_held:
            verdict = "NOGO_SALIENCY_SPREAD_IS_INTRINSIC"
        elif concentration_achieved and (not dseg_held):
            verdict = "PARTIAL_CONCENTRATION_AT_DSEG_COST"
        else:
            verdict = "NOGO_NO_CONCENTRATION_AND_DSEG_REGRESSED"
        state["phase3_remeasure"] = {
            "new_per_weight_saliency": new_sal,
            "new_weight_saliency_gini": new_gini,
            "gini_delta_vs_baseline": gini_up,
            "new_periphery_saliency_mass_frac": new_periph_mass,
            "periphery_saliency_mass_drop": periph_mass_drop,
            "new_dseg_subset": new_dseg,
            "dseg_delta_vs_baseline": dseg_delta,
            "implied_byte_shed_frac_if_periphery_int4": implied_byte_shed_frac,
            "concentration_achieved": concentration_achieved,
            "dseg_held": dseg_held,
            "verdict": verdict,
            "elapsed_s": round(time.time() - t0, 1),
        }
        save_state()
        print(
            f"      Gini {b['weight_saliency_gini']:.3f} -> {new_gini:.3f} (Δ{gini_up:+.3f}) | "
            f"periphery saliency-mass {b['periphery_saliency_mass_frac']:.1%} -> {new_periph_mass:.1%} "
            f"(drop {periph_mass_drop:+.1%}) | d_seg {b['baseline_dseg_subset']:.5f} -> {new_dseg:.5f} "
            f"(Δ{dseg_delta:+.5f})"
        )
        print(f"      VERDICT: {verdict}")

    # --- emit the canonical advisory JSON -----------------------------------
    payload = {
        "schema": "concentrated_saliency_feasibility.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_concentrated_saliency_feasibility.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "vehicle": {
            "basin_ckpt": str(BASIN_CKPT.relative_to(REPO)),
            "base_channels": 20,
            "n_params": 83356,
            "reported_basin_dseg_full600": 0.002600919948890805,
        },
        "method": {
            "saliency": "||d(sum SegNet top1-top2 margin)/dw_t|| autograd through bc20 decoder -> frozen SegNet (CPU)",
            "dseg_metric": "mean argmax-flip-rate vs GT hard labels over a frame subset (exact metric def, subset of pairs = faithful-definition proxy)",
            "concentration_reg": "penalize ||d(margin)/d(periphery weights)||^2 + CE seg-fidelity, short MPS fp32 gradient train",
            "n_saliency_frames": args.n_saliency_frames,
            "n_dseg_frames": args.n_dseg_frames,
            "core_size": args.core_size,
        },
        "phases": state,
        "verdict": state.get("phase3_remeasure", {}).get("verdict", "INCOMPLETE"),
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}")
    print(f"[verdict] {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
