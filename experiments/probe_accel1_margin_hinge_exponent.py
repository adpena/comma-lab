# SPDX-License-Identifier: MIT
"""ACCELERATOR PROBE 1 — does a FLIP-TARGETING margin-hinge seg loss BEND the d_seg
power-law exponent vs CE / soft-cosine, enough to overturn Probe C's sub-0.15 miss?

THE QUESTION (the top lever). Probe C proved the d_seg "wall" the 5-day run fights is
REAL (not a shadow artifact) and that an epochs-only descent lands ~2.5-3.7× ABOVE the
sub-0.15 d_seg target 0.000322, because the CE/soft-cosine seg-loss gradient VANISHES
exactly at the residual argmax-flips (soft-cosine grad ∝ p_gt(1-p_gt) → 0 as p_gt → 0;
CE never zeroes but spends magnitude on confident-INTERIOR pixels the argmax already
gets right). This probe tests the fix: a margin-hinge

    L(pixel) = max(0, margin_target - (logit[GT] - max_{c≠GT} logit[c]))

which is ~0 on correct-with-margin pixels (NO wasted gradient) and a CONSTANT-magnitude
pull on every flip / near-flip regardless of how confidently wrong (verified vs
soft-cosine in tests/test_segnet_margin_hinge_loss.py: on a confident flip the hinge
keeps grad ≈ -1.0 while soft-cosine's is ~1e-22). If concentrating ALL gradient on the
flip set BENDS the d_seg-vs-step power-law exponent (faster descent), it overturns
Probe C's projection — the prize. If it hits the SAME floor, the wall is capacity, not
the loss.

DECISIVE $0 SMOKE (small-n overfit, real frozen SegNet, same compute/seed). Three arms
from the SAME forkpoint init (deepcopy per arm): (A) CE baseline, (B) soft_cosine (the
current oomph lever), (C) margin_hinge (+ optional road↔lane class emphasis — Probe E
found 64% of flips are road↔lane). Each arm trains the SAME many steps; we record the
FULL d_seg(step) trajectory (exact contest argmax-flip rate vs GT), FIT
``d_seg = A·step^(-p)``, and project d_seg(50k) per arm. Higher exponent p = faster
descent = the d_seg wall BENDS.

DISCIPLINE (CLAUDE.md):
  * $0 local torch-CPU advisory; REAL frozen SegNet only (NO MPS — never a score
    authority). Every number is ``[contest-CPU advisory] NON-PROMOTABLE``.
  * Apples-to-apples: the seg-loss FUNCTION (+ its config) is the ONLY variable. Same
    forkpoint decoder/latent init (deepcopy per arm), same pairs, same LR, same
    seg_weight, NO pose term (seg isolated), same optimizer, same step count, same seed.
  * Routes the SAME driver loss router ``_seg_loss_for_spec`` the real trainer runs
    (read-only on the live trajectory; deepcopy init so nothing is perturbed).
  * Contention-aware: small pair slice + bounded steps (a live MPS train owns the GPU;
    this is CPU-only and shares CPU with siblings).

The d_seg "flip" is EXACTLY the contest metric: a pixel where the decoded frame's SegNet
argmax != the GT frame's SegNet argmax (= ``scorer.seg_targets_hard`` = argmax(SegNet(GT))).
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
_TARGETS_CACHE = Path(".omx/tmp/accel1_margin_hinge_targets_cache")

# The sub-0.15 d_seg target the projection is judged against. Per the symposium /
# small-basis reframe: base_ch20 small-basis rate+pose floor ≈ 0.1178, so sub-0.15
# needs d_seg < 0.000322 (the corrected arithmetic in the SUB-0.15 REFRAME memory).
_SUB015_DSEG_TARGET = 0.000322


def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """``decoded_pair`` (B,2,3,384,512) [0,255] -> (B,2,384,512,3) BHWC, eval-roundtripped
    (bicubic↑ to camera, bilinear↓ to eval) + uint8-STE rounded — a 1:1 port of the
    driver's per-step frame math (read-only)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    decoded_rounded = decoded_clamped.round()
    return decoded_clamped + (decoded_rounded - decoded_clamped).detach()


@torch.no_grad()
def _measure_d_seg(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    idx: torch.Tensor,
    scorer: RealScorerContext,
) -> float:
    """Exact d_seg (argmax-flip rate vs GT) on slice ``idx`` — the contest metric EXACTLY
    (same roundtrip + SegNet forward + ``scorer.seg_targets_hard`` reference)."""
    decoder.eval()
    decoded_pair = decoder(latents[idx])
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # (B, 5, Hs, Ws)
    decoded_argmax = seg_out.argmax(dim=1)
    gt_argmax = scorer.seg_targets_hard[idx]
    return (decoded_argmax != gt_argmax).float().mean().item()


def _fit_power_law(steps: list[int], d_seg: list[float], *, skip: int = 0) -> dict:
    """Fit ``d_seg = A·step^(-p)`` via OLS on ``log d_seg = log A - p·log step``.

    Returns the exponent ``p`` (higher = faster descent), ``log_A``, ``r2``, and the
    projected ``d_seg`` at 50k steps. ``skip`` drops the first ``skip`` warmup points.
    Only finite, positive d_seg with step >= 1 are used."""
    xs, ys = [], []
    for s, d in zip(steps, d_seg, strict=False):
        if s >= 1 and d > 0.0 and math.isfinite(d):
            xs.append(math.log(float(s)))
            ys.append(math.log(float(d)))
    if skip > 0:
        xs, ys = xs[skip:], ys[skip:]
    n = len(xs)
    if n < 3:
        return {"p": float("nan"), "log_A": float("nan"), "r2": float("nan"),
                "d_seg_50k": float("nan"), "n_points": n}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    if sxx <= 0.0:
        return {"p": float("nan"), "log_A": float("nan"), "r2": float("nan"),
                "d_seg_50k": float("nan"), "n_points": n}
    slope = sxy / sxx  # = -p
    intercept = my - slope * mx  # = log A
    p = -slope
    log_A = intercept
    # r2
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=False))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    d_seg_50k = math.exp(log_A) * (50_000.0 ** (-p))
    return {"p": p, "log_A": log_A, "r2": r2, "d_seg_50k": d_seg_50k, "n_points": n}


@dataclass
class _ArmResult:
    arm: str
    seg_surrogate: str | None
    config: dict
    steps: list[int] = field(default_factory=list)
    d_seg_traj: list[float] = field(default_factory=list)
    d_seg_before: float = 0.0
    d_seg_after: float = 0.0
    fit_full: dict = field(default_factory=dict)
    fit_late: dict = field(default_factory=dict)  # late half (steady-state exponent)
    grad_norm_first: float = 0.0
    loss_first: float = 0.0
    loss_last: float = 0.0
    seconds: float = 0.0


def _make_spec(
    arm: dict,
    *,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    ce_seg_loss,
) -> StageSpec:
    """Seg-only StageSpec (pose_weight=0 isolates seg). ``arm`` carries surrogate +
    its config (temperature / margin_target / road_lane_emphasis / margin_weight_tau)."""
    return StageSpec(
        name=f"probe_{arm['label']}",
        epochs=1,
        seg_loss_fn=ce_seg_loss,  # used iff seg_surrogate is None
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
        seg_surrogate=arm["surrogate"],
        seg_temperature=arm.get("temperature", 1.0),
        seg_temperature_end=None,
        margin_weight_tau=arm.get("margin_weight_tau"),
        seg_margin_hinge_target=arm.get("margin_target", 1.0),
        road_lane_emphasis=arm.get("road_lane_emphasis", 1.0),
    )


def _run_arm(
    *,
    arm: dict,
    base_decoder_state: dict,
    base_latents: torch.Tensor,
    scorer: RealScorerContext,
    decoder_factory,
    idx: torch.Tensor,
    n_steps: int,
    measure_every: int,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    ce_seg_loss,
) -> _ArmResult:
    """Train ONE arm from the SHARED forkpoint init (deepcopy) for ``n_steps`` seg-only
    steps, recording the d_seg(step) trajectory every ``measure_every`` steps. EVERY arm
    gets a fresh deepcopy of the identical init + a fresh AdamW with the same LR — the
    seg-loss function/config is the ONLY variable."""
    t0 = time.time()
    decoder = decoder_factory()
    decoder.load_state_dict(copy.deepcopy(base_decoder_state))
    decoder.train()
    latents = base_latents.clone().detach().requires_grad_(True)

    spec = _make_spec(
        arm, seg_weight=seg_weight, adamw_lr=adamw_lr, grad_clip=grad_clip, ce_seg_loss=ce_seg_loss
    )
    opt = torch.optim.AdamW(
        [{"params": decoder.parameters()}, {"params": [latents]}], lr=adamw_lr
    )

    steps: list[int] = []
    d_seg_traj: list[float] = []
    d0 = _measure_d_seg(decoder, latents, idx, scorer)
    steps.append(0)
    d_seg_traj.append(d0)

    grad_norm_first = 0.0
    loss_first = 0.0
    loss_last = 0.0
    temp = arm.get("temperature", 1.0)
    for step in range(1, n_steps + 1):
        decoder.train()
        opt.zero_grad()
        decoded_pair = decoder(latents[idx])
        decoded_bhwc = _roundtrip(decoded_pair)
        seg_out = scorer.seg_forward_train(decoded_bhwc)
        seg_l = _seg_loss_for_spec(
            spec, seg_out, scorer.seg_targets_hard[idx], temperature=temp
        )
        loss = spec.seg_weight * seg_l
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [*decoder.parameters(), latents], spec.grad_clip
        )
        if step == 1:
            grad_norm_first = float(gn)
            loss_first = float(loss.item())
        loss_last = float(loss.item())
        opt.step()
        if step % measure_every == 0 or step == n_steps:
            steps.append(step)
            d_seg_traj.append(_measure_d_seg(decoder, latents, idx, scorer))

    half = len(steps) // 2
    return _ArmResult(
        arm=arm["label"],
        seg_surrogate=arm["surrogate"],
        config={k: v for k, v in arm.items() if k not in ("label", "surrogate")},
        steps=steps,
        d_seg_traj=d_seg_traj,
        d_seg_before=d_seg_traj[0],
        d_seg_after=d_seg_traj[-1],
        fit_full=_fit_power_law(steps, d_seg_traj, skip=1),  # skip step-0 warmup point
        fit_late=_fit_power_law(steps[half:], d_seg_traj[half:]),
        grad_norm_first=grad_norm_first,
        loss_first=loss_first,
        loss_last=loss_last,
        seconds=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=12, help="pair-slice size (small for contention)")
    ap.add_argument("--n-steps", type=int, default=120, help="seg-only steps per arm (the trajectory length the exponent is fit on)")
    ap.add_argument("--measure-every", type=int, default=5, help="record exact d_seg every N steps")
    ap.add_argument("--seg-weight", type=float, default=100.0)
    ap.add_argument("--adamw-lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--slices", type=str, default="0,1",
        help="comma list of slice-START indices (robustness across >=2 slices)",
    )
    ap.add_argument(
        "--init-latents", type=str, default="random,forkpoint",
        help=(
            "comma list of latent-init regimes. 'random' = high-flip start (d_seg~0.04, "
            "ABUNDANT flips → the genuine-REPAIR regime, closest to the from-0 dynamic — "
            "the PRIMARY discriminator for 'does the loss bend the descent'). 'forkpoint' "
            "= the converged EMA basin (d_seg~0.003, near the floor — 'which loss holds / "
            "damages least')."
        ),
    )
    ap.add_argument("--soft-cosine-temp", type=float, default=0.3, help="soft-cosine T (the gradient-alive sweet spot per the lever probes)")
    ap.add_argument("--margin-target", type=float, default=1.0, help="margin-hinge target margin (logit units)")
    ap.add_argument("--road-lane-emphasis", type=float, default=2.0, help="margin-hinge road↔lane class emphasis (Probe E: 64% of flips)")
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)

    if not _FORKPOINT.exists():
        raise SystemExit(f"forkpoint not found: {_FORKPOINT}")
    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")

    print(f"[probe] loading forkpoint {_FORKPOINT}", flush=True)
    ckpt = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    base_channels = 20
    latent_dim = 28
    base_decoder_state = ckpt["ema_decoder"]  # EMA shadow = the inference/score state
    ema_latents = ckpt["ema_latents"].clone().detach()
    n_pairs_ckpt = ema_latents.shape[0]
    print(f"[probe] base_ch={base_channels} latent_dim={latent_dim} n_pairs_ckpt={n_pairs_ckpt}", flush=True)

    max_slice_end = max(int(s) for s in args.slices.split(",")) + args.n_pairs
    max_pairs_needed = min(n_pairs_ckpt, max_slice_end)

    print(f"[probe] building RealScorerContext (REAL frozen SegNet, CPU AUTHORITY — NO MPS, max_pairs={max_pairs_needed})", flush=True)
    scorer = RealScorerContext(
        video_path=_VIDEO,
        device="cpu",  # CPU AUTHORITY — NO MPS
        max_pairs=max_pairs_needed,
        targets_cache=_TARGETS_CACHE,
    )
    print(f"[probe] scorer ready: n_pairs={scorer.n_pairs}", flush=True)

    vbundle = import_vendored_bundle()

    def decoder_factory():
        return vbundle.HNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels, eval_size=(_EVAL_H, _EVAL_W)
        ).to("cpu")

    ce_seg_loss = import_vendored("stages.stage1_v328_ce").ce_seg_loss

    # The 3 arms (the task's A/B/C). C also gets a +road↔lane variant (4th arm) so the
    # class-emphasis contribution is isolated from the bare hinge.
    arms: list[dict] = [
        {"label": "A_CE", "surrogate": None},
        {"label": "B_soft_cosine", "surrogate": "soft_cosine", "temperature": args.soft_cosine_temp},
        {"label": "C_margin_hinge", "surrogate": "margin_hinge", "margin_target": args.margin_target},
        {
            "label": "C_margin_hinge_roadlane",
            "surrogate": "margin_hinge",
            "margin_target": args.margin_target,
            "road_lane_emphasis": args.road_lane_emphasis,
        },
    ]

    regimes = [r.strip() for r in args.init_latents.split(",") if r.strip()]
    slice_starts = [int(s) for s in args.slices.split(",")]

    def _base_latents_for(regime: str) -> torch.Tensor:
        if regime == "forkpoint":
            return ema_latents.clone().detach()
        if regime == "random":
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            return torch.randn(n_pairs_ckpt, latent_dim, generator=g)
        raise SystemExit(f"unknown --init-latents regime: {regime!r}")

    all_results: dict[str, list[dict]] = {}
    for regime in regimes:
        base_latents = _base_latents_for(regime)
        print(f"\n[probe] ##### REGIME init-latents={regime} #####", flush=True)
        for s_start in slice_starts:
            if s_start + args.n_pairs > scorer.n_pairs:
                print(f"[probe] SKIP slice {s_start} (exceeds n_pairs={scorer.n_pairs})", flush=True)
                continue
            idx = torch.arange(s_start, s_start + args.n_pairs, device="cpu")
            print(f"\n[probe] === REGIME {regime} SLICE start={s_start} n={args.n_pairs} ===", flush=True)
            slice_results: list[dict] = []
            for arm in arms:
                res = _run_arm(
                    arm=arm,
                    base_decoder_state=base_decoder_state,
                    base_latents=base_latents,
                    scorer=scorer,
                    decoder_factory=decoder_factory,
                    idx=idx,
                    n_steps=args.n_steps,
                    measure_every=args.measure_every,
                    seg_weight=args.seg_weight,
                    adamw_lr=args.adamw_lr,
                    grad_clip=args.grad_clip,
                    ce_seg_loss=ce_seg_loss,
                )
                row = dict(res.__dict__)
                row["regime"] = regime
                slice_results.append(row)
                print(
                    f"  {res.arm:24s} d_seg {res.d_seg_before:.5f}->{res.d_seg_after:.5f}  "
                    f"p(full)={res.fit_full['p']:+.3f} r2={res.fit_full['r2']:.3f}  "
                    f"p(late)={res.fit_late['p']:+.3f}  "
                    f"d_seg(50k)={res.fit_full['d_seg_50k']:.2e}  "
                    f"gn0={res.grad_norm_first:.2e}  [{res.seconds:.1f}s]",
                    flush=True,
                )
            all_results[f"{regime}__slice_{s_start}"] = slice_results

    # --- Verdict synthesis (PER REGIME, averaged across slices) -------------
    def _avg_fit(regime: str, arm_label: str, which: str, key: str) -> float:
        vals = [
            r[which][key]
            for slr in all_results.values()
            for r in slr
            if r["arm"] == arm_label and r.get("regime") == regime
            and math.isfinite(r[which].get(key, float("nan")))
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def _avg_field(regime: str, arm_label: str, key: str) -> float:
        vals = [
            r[key]
            for slr in all_results.values()
            for r in slr
            if r["arm"] == arm_label and r.get("regime") == regime
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    arm_labels = [a["label"] for a in arms]
    per_regime: dict[str, dict] = {}
    for regime in regimes:
        # is this regime a REPAIR regime? (CE actually drives d_seg down)
        ce_before = _avg_field(regime, "A_CE", "d_seg_before")
        ce_after = _avg_field(regime, "A_CE", "d_seg_after")
        is_repair = ce_after < ce_before
        print(
            f"\n[probe] ===== VERDICT (regime={regime}; "
            f"{'REPAIR — d_seg descends' if is_repair else 'BASIN/floor'}) =====",
            flush=True,
        )
        rows = []
        for lab in arm_labels:
            p_full = _avg_fit(regime, lab, "fit_full", "p")
            p_late = _avg_fit(regime, lab, "fit_late", "p")
            d50 = _avg_fit(regime, lab, "fit_full", "d_seg_50k")
            d_after = _avg_field(regime, lab, "d_seg_after")
            reaches = d50 < _SUB015_DSEG_TARGET if math.isfinite(d50) else False
            rows.append({
                "arm": lab, "p_full": p_full, "p_late": p_late,
                "d_seg_50k": d50, "d_seg_after": d_after,
                "reaches_sub015_dseg": reaches,
            })
            print(
                f"  {lab:24s} p(full)={p_full:+.3f}  p(late)={p_late:+.3f}  "
                f"d_seg(50k)={d50:.2e}  ({'REACHES' if reaches else 'misses'} target {_SUB015_DSEG_TARGET:.2e})",
                flush=True,
            )
        # exponent BEND vs CE: positive = the loss descends faster than CE
        ce_p = next((r["p_full"] for r in rows if r["arm"] == "A_CE"), float("nan"))
        ce_d50 = next((r["d_seg_50k"] for r in rows if r["arm"] == "A_CE"), float("nan"))
        for r in rows:
            r["p_bend_vs_ce"] = r["p_full"] - ce_p if math.isfinite(ce_p) else float("nan")
            r["d50_ratio_vs_ce"] = (r["d_seg_50k"] / ce_d50) if (math.isfinite(ce_d50) and ce_d50 > 0) else float("nan")
        hinge_rows = [r for r in rows if "margin_hinge" in r["arm"]]
        best_hinge = max(hinge_rows, key=lambda r: (r["p_full"] if math.isfinite(r["p_full"]) else -1e9)) if hinge_rows else None
        any_reaches = any(r["reaches_sub015_dseg"] for r in rows)
        hinge_bends = best_hinge is not None and math.isfinite(best_hinge["p_bend_vs_ce"]) and best_hinge["p_bend_vs_ce"] > 0.02
        if any_reaches:
            verdict = "OVERTURNS_PROBE_C_SOME_ARM_REACHES_SUB015_DSEG"
        elif hinge_bends:
            verdict = "HINGE_BENDS_EXPONENT_VS_CE_BUT_PROJECTION_STILL_MISSES"
        else:
            verdict = "SAME_FLOOR_NO_EXPONENT_BEND"
        per_regime[regime] = {
            "is_repair": is_repair, "ce_p_full": ce_p, "ce_d_seg_50k": ce_d50,
            "arms": rows, "best_hinge_arm": best_hinge["arm"] if best_hinge else None,
            "best_hinge_p_bend_vs_ce": best_hinge["p_bend_vs_ce"] if best_hinge else None,
            "verdict": verdict,
        }
        print(f"  [regime {regime}] VERDICT: {verdict}", flush=True)

    # Primary discriminator = the repair regime (genuine flips to fix; closest to from-0).
    repair = [r for r in regimes if per_regime[r]["is_repair"]]
    primary = repair[0] if repair else regimes[0]
    overall = per_regime[primary]["verdict"]
    print(f"\n[probe] PRIMARY discriminator regime = {primary}", flush=True)
    print(f"[probe] ACCEL-1 VERDICT: {overall}", flush=True)

    out = {
        "probe": "accel1_margin_hinge_exponent",
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "forkpoint": str(_FORKPOINT),
        "sub015_dseg_target": _SUB015_DSEG_TARGET,
        "config": vars(args),
        "primary_regime": primary,
        "accel1_verdict": overall,
        "per_regime_verdict": per_regime,
        "results": all_results,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
