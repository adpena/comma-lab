# SPDX-License-Identifier: MIT
"""LENS-A d_seg-SLOPE A/B — CE (vendored) vs margin_hinge vs soft_cosine_T0.3 (±L5).

THE MANDATE (RECURSIVE MATH-OPT LENS A, 2026-06-19): is the vendored CE seg loss
the OPTIMAL training loss for descending the contest d_seg (argmax-flip rate), or
does a boundary-weighted MARGIN/HINGE loss descend d_seg FASTER/LOWER?

Prior evidence:
  * ``lever2_softcosine_vs_ce_flipfix`` (real frozen SegNet, CPU, slice-0): on the
    CONVERGED forkpoint, soft_cosine_T0.3 modestly BEAT CE (Δd_seg 0.000503 vs CE
    0.000200) — but on RANDOM init CE crushed soft_cosine (LEVER-2-DOWNGRADE). That
    probe did NOT include the margin_hinge arm.
  * ``accel1_margin_hinge_exponent`` (random init, power-law fit): margin_hinge had
    d50_ratio_vs_ce = 0.424 (best — ~2.4x lower d_seg at the 50k extrapolation) and a
    +0.118 steeper power-law exponent than CE.

THE GAP this probe closes: NO probe has run CE vs **margin_hinge** vs soft_cosine_T0.3
on the REAL frozen scorer, FROM THE CONVERGED BASIN (the actual decisive operating
point, d_seg~0.0026), measuring the actual d_seg descent SLOPE over a meaningful
window. This is the head-to-head the mandate asks for.

DISCIPLINE (CLAUDE.md):
  * Apples-to-apples: the seg-loss FUNCTION is the ONLY variable. Same forkpoint EMA
    decoder+latent init (deepcopy per arm), same pairs, same LR, same seg_weight, NO
    pose term (seg isolated), same optimizer, same step count.
  * MPS is the TRAIN device (gradient backend, 104x faster) ONLY. The d_seg is
    measured on the SAME MPS-forward SegNet for EVERY arm (consistent ⟹ a valid
    SLOPE/ranking comparison) — it is NOT a contest score (which requires CPU
    authority + byte-closed archive). Every number ``[macOS-MPS research-signal]``
    NON-PROMOTABLE.
  * Read-only on driver.py lever code (routes through ``_seg_loss_for_spec``).
  * SHORT (n96 + few hundred steps): MPS is shared with a sister optimizer-probe.

USAGE:
  .venv/bin/python experiments/probe_lensA_ce_vs_margin_dseg_slope.py \
      --n-pairs 96 --n-steps 300 --eval-every 30 \
      --out-json .omx/research/lensA_ce_vs_margin_dseg_slope_<UTC>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import _seg_loss_for_spec, import_vendored_bundle
from tac.torch_vehicle.scorer_context import RealScorerContext
from tac.torch_vehicle.vendored_imports import import_vendored

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5

_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = Path("upstream/videos/0.mkv")
_TARGETS_CACHE = Path(".omx/tmp/lensA_targets_cache")


def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """1:1 port of the driver per-step frame math: bicubic↑→bilinear↓→uint8-STE."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    decoded_rounded = decoded_clamped.round()
    return decoded_clamped + (decoded_rounded - decoded_clamped).detach()


@torch.no_grad()
def _measure_dseg(decoder, latents, idx, scorer) -> tuple[float, int, float]:
    """Exact d_seg (argmax-flip rate vs GT SegNet argmax) on the slice, on the train
    device's SegNet (consistent for every arm). Also returns total flips + the MEDIAN
    flip margin Δ = pred_argmax_logit − pred_gt_logit over the flipped pixels (the
    live resonance statistic the anneal calculus tracks)."""
    decoder.eval()
    decoded_pair = decoder(latents[idx])
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # (B, 5, Hs, Ws) on train device
    decoded_argmax = seg_out.argmax(dim=1)  # (B, Hs, Ws)
    gt_argmax = scorer.seg_targets_hard[idx]  # (B, Hs, Ws)
    flips = decoded_argmax != gt_argmax
    d_seg = float(flips.float().mean().item())
    n_flips = int(flips.sum().item())
    # Flip-margin Δ for the resonance diagnostic: on flipped pixels, the gap between
    # the winning (wrong) logit and the GT-class logit.
    median_margin = float("nan")
    if n_flips > 0:
        gt = gt_argmax.unsqueeze(1)  # (B,1,Hs,Ws)
        gt_logit = torch.gather(seg_out, 1, gt).squeeze(1)  # (B,Hs,Ws)
        top1 = seg_out.max(dim=1).values  # (B,Hs,Ws) = winning logit
        margin = (top1 - gt_logit)[flips]  # (n_flips,) >= 0 on flips
        # .median() on MPS has a broken index kernel on some shapes (AcceleratorError
        # "index out of bounds"); the median is a $0 diagnostic, so move to CPU.
        median_margin = float(margin.detach().to("cpu").median().item())
    return d_seg, n_flips, median_margin


@dataclass
class _ArmTrace:
    arm: str
    seg_surrogate: str | None
    temperature: float | None
    margin_weight_tau: float | None
    margin_hinge_target: float | None
    d_seg_curve: list[tuple[int, float]] = field(default_factory=list)  # (step, d_seg)
    flips_curve: list[tuple[int, int]] = field(default_factory=list)
    median_margin_curve: list[tuple[int, float]] = field(default_factory=list)
    d_seg_before: float = 0.0
    d_seg_after: float = 0.0
    grad_norm_first: float = 0.0
    loss_first: float = 0.0
    loss_last: float = 0.0
    seconds: float = 0.0


def _make_spec(arm, *, seg_weight, adamw_lr, grad_clip, ce_seg_loss) -> StageSpec:
    """A seg-only StageSpec per arm; pose_weight=0 isolates seg."""
    surrogate, temp, mtau, hinge_t = arm["surrogate"], arm["temp"], arm["mtau"], arm["hinge_t"]
    return StageSpec(
        name=f"lensA_{arm['label']}",
        epochs=1,
        seg_loss_fn=ce_seg_loss,
        eval_every=1,
        batch_size=1,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=adamw_lr,
        muon_lr=0.0,
        muon_weight_decay=0.0,
        latent_lr_mult=1.0,
        grad_clip=grad_clip,
        grad_clip_muon=None,
        lr_floor_ratio=1.0,
        seg_weight=seg_weight,
        pose_weight=0.0,
        cat_lambda=0.0,
        cat_sigma=0.0,
        use_qat=False,
        init_latents_random=False,
        seg_surrogate=surrogate,
        seg_temperature=temp if temp is not None else 1.0,
        margin_weight_tau=mtau,
        seg_margin_hinge_target=hinge_t if hinge_t is not None else 1.0,
    )


def _run_arm(
    *, arm, base_decoder_state, base_latents, scorer, decoder_factory, idx, n_steps,
    eval_every, seg_weight, adamw_lr, grad_clip, ce_seg_loss, train_device,
) -> _ArmTrace:
    t0 = time.time()
    decoder = decoder_factory().to(train_device)
    decoder.load_state_dict(copy.deepcopy(base_decoder_state))
    decoder.train()
    latents = base_latents.clone().detach().to(train_device).requires_grad_(True)

    spec = _make_spec(arm, seg_weight=seg_weight, adamw_lr=adamw_lr,
                      grad_clip=grad_clip, ce_seg_loss=ce_seg_loss)
    temp = arm["temp"]

    tr = _ArmTrace(
        arm=arm["label"], seg_surrogate=arm["surrogate"], temperature=temp,
        margin_weight_tau=arm["mtau"], margin_hinge_target=arm["hinge_t"],
    )
    d0, f0, m0 = _measure_dseg(decoder, latents, idx, scorer)
    tr.d_seg_before = d0
    tr.d_seg_curve.append((0, d0))
    tr.flips_curve.append((0, f0))
    tr.median_margin_curve.append((0, m0))

    opt = torch.optim.AdamW(
        [{"params": decoder.parameters()}, {"params": [latents]}], lr=adamw_lr
    )
    for step in range(1, n_steps + 1):
        decoder.train()
        opt.zero_grad()
        decoded_pair = decoder(latents[idx])
        decoded_bhwc = _roundtrip(decoded_pair)
        seg_out = scorer.seg_forward_train(decoded_bhwc)
        seg_l = _seg_loss_for_spec(
            spec, seg_out, scorer.seg_targets_hard[idx],
            temperature=(temp if temp is not None else None),
        )
        loss = spec.seg_weight * seg_l
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [*decoder.parameters(), latents], spec.grad_clip
        )
        if step == 1:
            tr.grad_norm_first = float(gn)
            tr.loss_first = float(loss.item())
        tr.loss_last = float(loss.item())
        opt.step()
        if step % eval_every == 0 or step == n_steps:
            d, f, m = _measure_dseg(decoder, latents, idx, scorer)
            tr.d_seg_curve.append((step, d))
            tr.flips_curve.append((step, f))
            tr.median_margin_curve.append((step, m))

    tr.d_seg_after = tr.d_seg_curve[-1][1]
    tr.seconds = time.time() - t0
    return tr


def _slope_log(curve: list[tuple[int, float]]) -> float:
    """Fit log10(d_seg) ~ a + b*log10(step) over step>=eval_every (skip step 0). Returns
    b (the descent power-law exponent; more negative = faster d_seg descent)."""
    xs = [(math.log10(s), math.log10(max(d, 1e-9))) for s, d in curve if s > 0 and d > 0]
    if len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx = sum(x for x, _ in xs) / n
    my = sum(y for _, y in xs) / n
    num = sum((x - mx) * (y - my) for x, y in xs)
    den = sum((x - mx) ** 2 for x, _ in xs)
    return num / den if den != 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=96)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--slice-start", type=int, default=0)
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--adamw-lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--train-device", type=str, default="mps")
    ap.add_argument("--init-latents", type=str, default="forkpoint",
                    choices=["forkpoint", "random"],
                    help="forkpoint = EMA-shadow latents (floor regime, d_seg~0.003 — which "
                         "loss holds/damages least); random = high-flip start (repair regime, "
                         "the from-0 dynamic — which loss DRIVES d_seg down fastest).")
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)

    if not _FORKPOINT.exists():
        raise SystemExit(f"forkpoint not found: {_FORKPOINT}")
    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")

    print(f"[lensA] loading forkpoint {_FORKPOINT}", flush=True)
    ckpt = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    base_channels, latent_dim = 20, 28
    base_decoder_state = ckpt["ema_decoder"]
    ema_latents = ckpt["ema_latents"].clone().detach()
    n_pairs_ckpt = ema_latents.shape[0]
    print(f"[lensA] base_ch={base_channels} latent_dim={latent_dim} n_pairs_ckpt={n_pairs_ckpt}", flush=True)

    max_pairs_needed = min(n_pairs_ckpt, args.slice_start + args.n_pairs)
    print(f"[lensA] building RealScorerContext (real frozen SegNet, train_device={args.train_device}, CPU authority, max_pairs={max_pairs_needed})", flush=True)
    scorer = RealScorerContext(
        video_path=_VIDEO,
        device="cpu",  # AUTHORITY (unused for d_seg measure; seg_forward_train uses train net)
        train_device=args.train_device,  # MPS gradient backend
        max_pairs=max_pairs_needed,
        targets_cache=_TARGETS_CACHE,
    )
    print(f"[lensA] scorer ready: n_pairs={scorer.n_pairs}", flush=True)

    vbundle = import_vendored_bundle()

    def decoder_factory():
        return vbundle.HNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels, eval_size=(_EVAL_H, _EVAL_W)
        )

    ce_seg_loss = import_vendored("stages.stage1_v328_ce").ce_seg_loss

    # The arm matrix: CE (vendored baseline) + margin_hinge (the flip-targeting hinge,
    # the mandate's hypothesis) + soft_cosine_T0.3 (the prior best surrogate) ±L5.
    arms = [
        {"label": "CE",                  "surrogate": None,          "temp": None, "mtau": None, "hinge_t": None},
        {"label": "margin_hinge_t1.0",   "surrogate": "margin_hinge","temp": None, "mtau": None, "hinge_t": 1.0},
        {"label": "margin_hinge_t0.5",   "surrogate": "margin_hinge","temp": None, "mtau": None, "hinge_t": 0.5},
        {"label": "margin_hinge_t1.0_L5","surrogate": "margin_hinge","temp": None, "mtau": 0.3,  "hinge_t": 1.0},
        {"label": "soft_cosine_T0.3",    "surrogate": "soft_cosine", "temp": 0.3,  "mtau": None, "hinge_t": None},
        {"label": "soft_cosine_T0.3_L5", "surrogate": "soft_cosine", "temp": 0.3,  "mtau": 0.3,  "hinge_t": None},
    ]

    idx = torch.arange(args.slice_start, args.slice_start + args.n_pairs).to(scorer.train_device)

    if args.init_latents == "random":
        # Seeded-random latents on the EMA decoder => high-flip repair regime (the from-0
        # dynamic). Deterministic per --seed so the A/B init is bit-identical across arms.
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        base_latents = torch.randn(n_pairs_ckpt, latent_dim, generator=g)
        print("[lensA] init-latents=random (repair regime, high-flip start)", flush=True)
    else:
        base_latents = ema_latents
        print("[lensA] init-latents=forkpoint (floor regime, converged basin)", flush=True)

    traces: list[_ArmTrace] = []
    for arm in arms:
        tr = _run_arm(
            arm=arm, base_decoder_state=base_decoder_state, base_latents=base_latents,
            scorer=scorer, decoder_factory=decoder_factory, idx=idx, n_steps=args.n_steps,
            eval_every=args.eval_every, seg_weight=args.seg_weight, adamw_lr=args.adamw_lr,
            grad_clip=args.grad_clip, ce_seg_loss=ce_seg_loss, train_device=scorer.train_device,
        )
        traces.append(tr)
        slope = _slope_log(tr.d_seg_curve)
        red = tr.d_seg_before - tr.d_seg_after
        print(
            f"  {tr.arm:24s} d_seg {tr.d_seg_before:.6f}->{tr.d_seg_after:.6f} "
            f"(Δ={red:+.6f}, {100*red/max(tr.d_seg_before,1e-9):+.1f}%) "
            f"slope_b={slope:+.3f} gn0={tr.grad_norm_first:.3e} "
            f"loss {tr.loss_first:.3f}->{tr.loss_last:.3f} [{tr.seconds:.0f}s]",
            flush=True,
        )
        for st, d in tr.d_seg_curve:
            mm = dict(tr.median_margin_curve).get(st, float("nan"))
            print(f"       step {st:4d}  d_seg={d:.6f}  median_flip_margin={mm:.3f}", flush=True)

    # Verdict: rank by final d_seg (lower = better d_seg-finisher) + by descent slope.
    ce = next(t for t in traces if t.arm == "CE")
    ranked = sorted(traces, key=lambda t: t.d_seg_after)
    print("\n[lensA] === RANKING by FINAL d_seg (lower = better) ===", flush=True)
    for t in ranked:
        ratio = t.d_seg_after / max(ce.d_seg_after, 1e-9)
        tag = "  <-- CE baseline" if t.arm == "CE" else (
            f"  ({ratio:.3f}x CE; {'BEATS' if t.d_seg_after < ce.d_seg_after else 'loses to'} CE)"
        )
        print(f"  {t.arm:24s} final_d_seg={t.d_seg_after:.6f}{tag}", flush=True)

    best = ranked[0]
    verdict = (
        "MARGIN_BEATS_CE" if best.arm != "CE" and "margin" in best.arm else
        "SOFTCOSINE_BEATS_CE" if best.arm != "CE" else
        "CE_OPTIMAL"
    )
    print(f"\n[lensA] VERDICT: {verdict} (best arm = {best.arm}, "
          f"{best.d_seg_after/max(ce.d_seg_after,1e-9):.3f}x CE final d_seg)", flush=True)

    out = {
        "probe": "lensA_ce_vs_margin_dseg_slope",
        "authority": "[macOS-MPS research-signal] NON-PROMOTABLE",
        "forkpoint": str(_FORKPOINT),
        "config": vars(args),
        "verdict": verdict,
        "best_arm": best.arm,
        "ce_final_d_seg": ce.d_seg_after,
        "arms": [
            {
                "arm": t.arm, "seg_surrogate": t.seg_surrogate, "temperature": t.temperature,
                "margin_weight_tau": t.margin_weight_tau, "margin_hinge_target": t.margin_hinge_target,
                "d_seg_before": t.d_seg_before, "d_seg_after": t.d_seg_after,
                "d_seg_reduction": t.d_seg_before - t.d_seg_after,
                "final_ratio_vs_ce": t.d_seg_after / max(ce.d_seg_after, 1e-9),
                "slope_b": _slope_log(t.d_seg_curve),
                "grad_norm_first": t.grad_norm_first,
                "loss_first": t.loss_first, "loss_last": t.loss_last,
                "d_seg_curve": t.d_seg_curve, "flips_curve": t.flips_curve,
                "median_margin_curve": t.median_margin_curve, "seconds": t.seconds,
            }
            for t in traces
        ],
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[lensA] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
