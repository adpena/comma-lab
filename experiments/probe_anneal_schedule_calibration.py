# SPDX-License-Identifier: MIT
"""ANNEAL-SCHEDULE CALIBRATION probe (#119) — which seg-temperature SCHEDULE reaches
the LOWEST d_seg over a real training window on the Lever-3 v2 clean-``rgb_1`` decoder?

THE QUESTION (the SEG-side gate for the from-0 run #116; the POSE side #118 is SEALED).
Lever-2's soft-cosine surrogate ``1 − softmax(pred/T)[gt]`` is MARGIN-RESONANT: the
optimal-anneal calculus
(``.omx/research/anneal_optimal_math_geometry_calculus_20260613.md``) derives the
per-flip GT-logit gradient magnitude on a confidently-wrong flip of logit margin
``Δ = pred_argmax − pred_gt > 0``::

    |grad_g(Δ, T)|  ≈  (1/T)·e^{−Δ/T}     (maximized at  T* = Δ)

So the surrogate has a TEMPERATURE RESONANCE at the flip's own margin: below T=Δ the
gradient dies super-exponentially (the R12 dead zone, T≤0.1 ≈ 10 orders down), above it
fades ~1/T (untargeted — the gate's T=1.0 row LOST to CE). The flips' margins shrink as
training fixes the deep errors, so the MAX-SIGNAL schedule TRACKS the live flip-margin
median down to the ~0.3 gradient-alive floor.

This probe runs a SHORT real-scorer seg-only training window under FOUR schedules (T
varying PER STEP) on the Lever-3 v2 ``PoseFiLMHNeRVWrapperV2`` decoder and measures which
reaches the lowest exact d_seg (argmax-flip rate of ``f1``'s SegNet mask). The v2 decoder
renders the SegNet frame ``f1`` from the CLEAN trunk feature (FiLM-free) — so the seg
signal is pose-DECOUPLED (verified: ``f1`` is bit-invariant to the FiLM/pose).

THE FOUR ARMS (apples-to-apples — ONLY the schedule + τ-slaving differs):
  (a) static-T0.3            : T(t) = 0.3 for every step (the resonance, held).
  (b) fast-cool→0.3-hold     : T cosine 1.0 → 0.3 over the first hold_frac of the window,
                               then HOLDS at 0.3 (a brief warm phase for from-0 stability).
  (c) cosine-1.0→0.05        : the MISCALIBRATED baseline — plain cosine over the whole
                               window (~52% at T≥0.5 untargeted + ~15% in the dead zone).
  (d) margin-adaptive        : T(t) = clamp(running median of the LIVE flip margins Δ,
                               0.3, 0.6) measured each step; Lever-5 τ(t) = T(t) (slaved).
                               The theoretically-optimal / max-signal schedule.

The schedule math for (a)/(b)/(c) is the CANONICAL ``curriculum.seg_temperature_for_epoch``
(reused with a one-stage spec where ``epochs == n_steps`` so each step is an "epoch"); (d)
uses the new ``curriculum.seg_temperature_margin_adaptive`` (pure clamp of the measured
median). The PER-PIXEL seg loss is the EXACT driver router ``_seg_loss_for_spec``
(read-only). Lever-5 is ON for every arm (margin_weight_tau set); for (d) τ is slaved per
step, for (a)/(b)/(c) τ is held at the arm's static τ.

DISCIPLINE (CLAUDE.md):
  * $0 local torch-CPU advisory; REAL frozen SegNet only (NO MPS — never a score
    authority). Every number is ``[macOS-CPU advisory] NON-PROMOTABLE``.
  * Apples-to-apples: same v2 decoder init (deepcopy per arm), same pairs, same LR, same
    seg_weight, same step count, NO pose term (seg isolated on the clean ``f1``). ONLY the
    per-step temperature schedule (+ τ-slaving for (d)) differs.
  * NO edit to driver.py / pose_film_v2.py lever code (read-only). The only library change
    is a DEFAULT-PRESERVING ``seg_temperature_margin_adaptive`` schedule fn in curriculum.
  * Contention-aware + in-call: small pair slice + few steps, run SYNCHRONOUSLY.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import (
    SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR,
    SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX,
    StageSpec,
    seg_temperature_for_epoch,
    seg_temperature_margin_adaptive,
    seg_temperature_weighted_mean_resonant,
)
from tac.torch_vehicle.driver import _seg_loss_for_spec, import_vendored_bundle
from tac.torch_vehicle.pose_film_v2 import PoseFiLMHNeRVWrapperV2, _POSE_DIM
from tac.torch_vehicle.scorer_context import RealScorerContext
from tac.torch_vehicle.vendored_imports import import_vendored

_EVAL_H, _EVAL_W = 384, 512
_SEG_NUM_CLASSES = 5

_FORKPOINT = Path(
    "experiments/results/forkpoints/basin_bc20_20260612T121523Z/"
    "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = Path("upstream/videos/0.mkv")
_TARGETS_CACHE = Path(".omx/tmp/anneal_calib_v2_targets_cache")


# ---------------------------------------------------------------------------
# Roundtrip — a 1:1 port of the driver's per-step frame math (read-only; driver.py).
# Bicubic↑ to camera, bilinear↓ to eval, uint8-STE round.
# ---------------------------------------------------------------------------
def _roundtrip(decoded_pair: torch.Tensor) -> torch.Tensor:
    """``decoded_pair`` (B, 2, 3, 384, 512) float in [0,255] -> (B, 2, 384, 512, 3)
    BHWC, eval-roundtripped + uint8-STE rounded (gradient preserved)."""
    B = decoded_pair.shape[0]
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    decoded_clamped = decoded_bhwc.clamp(0, 255)
    decoded_rounded = decoded_clamped.round()
    return decoded_clamped + (decoded_rounded - decoded_clamped).detach()


def _flip_margins(seg_out: torch.Tensor, gt_argmax: torch.Tensor) -> torch.Tensor:
    """The RAW per-flip-pixel logit margin vector ``Δ = pred[argmax] − pred[gt] > 0`` over
    the CURRENT flip set (pixels where ``argmax(seg_out) != gt_argmax``). This IS the live
    flip-margin distribution the adaptive schedules consume (§6/§6′). ``seg_out`` is
    ``(B, 5, Hs, Ws)`` SegNet logits; ``gt_argmax`` is ``(B, Hs, Ws)`` (= argmax(SegNet(GT))).
    Returns an empty 1-D tensor when there are no flips (the caller holds the prior T)."""
    with torch.no_grad():
        pred_argmax = seg_out.argmax(dim=1)  # (B, Hs, Ws)
        flips = pred_argmax != gt_argmax  # (B, Hs, Ws) bool
        if int(flips.sum().item()) == 0:
            return seg_out.new_empty((0,))
        top1 = seg_out.max(dim=1).values  # (B, Hs, Ws) = pred[argmax]
        gt_logit = seg_out.gather(1, gt_argmax.unsqueeze(1)).squeeze(1)  # pred[gt]
        return (top1 - gt_logit)[flips]  # (n_flips,) >= 0 on a flip


def _flip_margin_median(seg_out: torch.Tensor, gt_argmax: torch.Tensor) -> float:
    """The MEDIAN of the live flip-margin distribution (the §6 margin-adaptive statistic).
    Returns ``nan`` when there are no flips (the caller holds the prior T)."""
    margins = _flip_margins(seg_out, gt_argmax)
    if margins.numel() == 0:
        return float("nan")
    return float(margins.median().item())


@dataclass
class _SchedStats:
    """Exact d_seg for one decoder state on the slice (the contest metric)."""

    d_seg: float
    n_flips: int


@torch.no_grad()
def _measure_dseg(
    wrapper: torch.nn.Module,
    latents: torch.Tensor,
    idx: torch.Tensor,
    scorer: RealScorerContext,
) -> _SchedStats:
    """Exact d_seg (argmax-flip rate vs GT) on the slice ``idx`` of the v2 decoder. Uses
    the SAME roundtrip + SegNet forward the driver/eval run, the CLEAN ``f1`` frame (the
    scorer uses the LAST frame = index 1 = ``f1``), and ``scorer.seg_targets_hard`` (= GT
    SegNet argmax) as the flip reference — the contest d_seg numerator EXACTLY (NO proxy)."""
    wrapper.eval()
    decoded_pair = wrapper(latents[idx], idx)  # (B, 2, 3, 384, 512); [:, 1] = clean f1
    decoded_bhwc = _roundtrip(decoded_pair)
    seg_out = scorer.seg_forward_train(decoded_bhwc)  # SegNet on the LAST frame (f1)
    decoded_argmax = seg_out.argmax(dim=1)
    gt_argmax = scorer.seg_targets_hard[idx]
    flips = decoded_argmax != gt_argmax
    return _SchedStats(d_seg=flips.float().mean().item(), n_flips=int(flips.sum().item()))


@dataclass
class _ArmResult:
    arm: str
    schedule: str
    margin_weight_tau: float | None
    d_seg_before: float
    d_seg_after: float
    d_seg_reduction: float  # before - after (positive = improvement)
    d_seg_min: float  # the LOWEST d_seg reached over the window (per-step tracked)
    flips_before: int
    flips_after: int
    flips_fixed: int
    # per-step trajectories (the apples-to-apples + Δ_median-collapse evidence)
    temps_per_step: list[float] = field(default_factory=list)
    taus_per_step: list[float] = field(default_factory=list)
    margin_median_per_step: list[float] = field(default_factory=list)
    dseg_per_step: list[float] = field(default_factory=list)
    loss_first: float = 0.0
    loss_last: float = 0.0
    grad_norm_first: float = 0.0
    seconds: float = 0.0


def _make_spec(
    *,
    margin_weight_tau: float | None,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    n_steps: int,
    seg_temperature: float,
    seg_temperature_end: float | None,
    seg_temperature_hold_frac: float | None,
    ce_seg_loss,
) -> StageSpec:
    """A minimal seg-only StageSpec on the soft_cosine surrogate. pose_weight=0 isolates
    seg. ``epochs == n_steps`` so the canonical ``seg_temperature_for_epoch`` cosine /
    fast-cool-hold math indexes each STEP as an epoch (apples-to-apples with the driver's
    per-epoch anneal applied at the step granularity of this short window)."""
    return StageSpec(
        name="probe_anneal",
        epochs=int(n_steps),  # each STEP is an "epoch" for the schedule fn
        seg_loss_fn=ce_seg_loss,  # unused on the soft_cosine path; required field
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
        pose_weight=0.0,  # ISOLATE seg — no pose term
        cat_lambda=0.0,
        cat_sigma=0.0,
        use_qat=False,
        init_latents_random=False,
        seg_surrogate="soft_cosine",
        seg_temperature=seg_temperature,
        seg_temperature_end=seg_temperature_end,
        seg_temperature_hold_frac=seg_temperature_hold_frac,
        margin_weight_tau=margin_weight_tau,
    )


# The four calibration arms: (label, schedule-id, schedule kwargs). The schedule-id drives
# how T(t) (and τ(t) for margin-adaptive) is computed each step.
_ARM_SCHEDULES: list[tuple[str, str, dict]] = [
    ("static_T0.3", "static", {"seg_temperature": 0.3}),
    (
        "fastcool_1.0to0.3_hold",
        "fast_cool_hold",
        {"seg_temperature": 1.0, "seg_temperature_end": 0.3, "hold_frac": 0.3},
    ),
    (
        "cosine_1.0to0.05",
        "cosine",
        {"seg_temperature": 1.0, "seg_temperature_end": 0.05},
    ),
    ("margin_adaptive", "margin_adaptive", {}),
    # The #119 CRUX FIX: the e^{−Δ/T}-WEIGHTED-MEAN fixed point of the live flip margins —
    # the true argmax of the aggregate surrogate gradient G(T)=(1/T)Σe^{−Δ_i/T}, which the
    # MEDIAN (margin_adaptive) over-estimates on a right-skewed flip distribution. τ slaved.
    ("weighted_mean_resonant", "weighted_mean", {}),
]


def _temp_for_step_static_cosine(spec: StageSpec, step: int) -> float:
    """T(t) for the static / cosine / fast-cool-hold arms — the CANONICAL
    ``seg_temperature_for_epoch`` indexed by step (epoch == step)."""
    return seg_temperature_for_epoch(spec, step)


def _run_arm(
    *,
    arm_label: str,
    schedule_id: str,
    schedule_kwargs: dict,
    base_wrapper_state: dict,
    base_latents: torch.Tensor,
    base_stored_pose: torch.Tensor,
    scorer: RealScorerContext,
    wrapper_factory,
    idx: torch.Tensor,
    n_steps: int,
    seg_weight: float,
    adamw_lr: float,
    grad_clip: float,
    margin_weight_tau: float,
    ce_seg_loss,
    t_floor: float,
    t_max: float,
) -> _ArmResult:
    """Run ONE arm from the SHARED v2 forkpoint init (deepcopy) for ``n_steps`` seg-only
    optimization steps on the SAME slice ``idx``. The seg-temperature SCHEDULE (and τ for
    margin-adaptive) is the ONLY variable across arms — identical decoder/latent init,
    fresh AdamW with the same LR, same seg_weight, same slice, same step count."""
    t0 = time.time()
    wrapper = wrapper_factory()
    wrapper.load_state_dict(copy.deepcopy(base_wrapper_state))
    with torch.no_grad():
        wrapper.set_stored_pose(base_stored_pose)
    wrapper.train()
    latents = base_latents.clone().detach().requires_grad_(True)

    # Adaptive arms compute T(t) per STEP from the LIVE flip margins (margin_adaptive =
    # MEDIAN; weighted_mean = the e^{−Δ/T}-weighted-mean fixed point — the #119 crux fix).
    is_adaptive = schedule_id in ("margin_adaptive", "weighted_mean")
    # Static/cosine/fast-cool arms: a one-stage spec drives the canonical schedule fn;
    # τ (margin_weight_tau) is held at the arm's static value.
    static_spec: StageSpec | None = None
    if not is_adaptive:
        static_spec = _make_spec(
            margin_weight_tau=margin_weight_tau,
            seg_weight=seg_weight,
            adamw_lr=adamw_lr,
            grad_clip=grad_clip,
            n_steps=n_steps,
            seg_temperature=float(schedule_kwargs.get("seg_temperature", 1.0)),
            seg_temperature_end=schedule_kwargs.get("seg_temperature_end"),
            seg_temperature_hold_frac=schedule_kwargs.get("hold_frac"),
            ce_seg_loss=ce_seg_loss,
        )

    before = _measure_dseg(wrapper, latents, idx, scorer)
    opt = torch.optim.AdamW(
        [{"params": wrapper.parameters()}, {"params": [latents]}], lr=adamw_lr
    )

    temps: list[float] = []
    taus: list[float] = []
    margin_medians: list[float] = []
    dseg_traj: list[float] = []
    loss_first = 0.0
    loss_last = 0.0
    grad_norm_first = 0.0
    d_seg_min = before.d_seg
    gt_argmax_slice = scorer.seg_targets_hard[idx]
    prev_temp = t_floor  # margin-adaptive: hold prior T when a step has no flips

    for step in range(n_steps):
        wrapper.train()
        opt.zero_grad()
        decoded_pair = wrapper(latents[idx], idx)
        decoded_bhwc = _roundtrip(decoded_pair)
        seg_out = scorer.seg_forward_train(decoded_bhwc)  # SegNet on f1 (clean)

        if is_adaptive:
            # Measure the LIVE flip-margin distribution, set T(t), slave τ(t)=T(t).
            margins_vec = _flip_margins(seg_out, gt_argmax_slice)
            # Record the median for the Δ-collapse evidence (schedule-independent property).
            med = float(margins_vec.median().item()) if margins_vec.numel() else float("nan")
            margin_medians.append(med)
            if margins_vec.numel() == 0:  # no flips this step -> hold the prior T
                temp = prev_temp
            elif schedule_id == "weighted_mean":
                temp = seg_temperature_weighted_mean_resonant(
                    margins_vec, t_floor=t_floor, t_max=t_max
                )
            else:  # margin_adaptive (MEDIAN)
                temp = seg_temperature_margin_adaptive(med, t_floor=t_floor, t_max=t_max)
            prev_temp = temp
            tau = temp  # τ SLAVED to T (the §7 co-tuning)
            step_spec = _make_spec(
                margin_weight_tau=tau,
                seg_weight=seg_weight,
                adamw_lr=adamw_lr,
                grad_clip=grad_clip,
                n_steps=n_steps,
                seg_temperature=temp,
                seg_temperature_end=None,
                seg_temperature_hold_frac=None,
                ce_seg_loss=ce_seg_loss,
            )
        else:
            assert static_spec is not None
            # The flip-margin median is RECORDED for every arm (the Δ_median-collapse
            # evidence is schedule-independent — it is a property of the run, not the
            # schedule) but does NOT drive T for the non-adaptive arms.
            margin_medians.append(_flip_margin_median(seg_out, gt_argmax_slice))
            temp = _temp_for_step_static_cosine(static_spec, step)
            tau = margin_weight_tau
            step_spec = static_spec

        temps.append(float(temp))
        taus.append(float(tau))

        seg_l = _seg_loss_for_spec(
            step_spec, seg_out, gt_argmax_slice, temperature=temp
        )
        loss = step_spec.seg_weight * seg_l  # NO pose term — seg isolated
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [*wrapper.parameters(), latents], grad_clip
        )
        if step == 0:
            grad_norm_first = float(gn)
            loss_first = float(loss.item())
        loss_last = float(loss.item())
        opt.step()

        # Per-step d_seg (the trajectory + the min the schedule reaches).
        step_stat = _measure_dseg(wrapper, latents, idx, scorer)
        dseg_traj.append(step_stat.d_seg)
        d_seg_min = min(d_seg_min, step_stat.d_seg)

    after = _measure_dseg(wrapper, latents, idx, scorer)
    return _ArmResult(
        arm=arm_label,
        schedule=schedule_id,
        margin_weight_tau=margin_weight_tau,
        d_seg_before=before.d_seg,
        d_seg_after=after.d_seg,
        d_seg_reduction=before.d_seg - after.d_seg,
        d_seg_min=d_seg_min,
        flips_before=before.n_flips,
        flips_after=after.n_flips,
        flips_fixed=before.n_flips - after.n_flips,
        temps_per_step=temps,
        taus_per_step=taus,
        margin_median_per_step=margin_medians,
        dseg_per_step=dseg_traj,
        loss_first=loss_first,
        loss_last=loss_last,
        grad_norm_first=grad_norm_first,
        seconds=time.time() - t0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=8, help="pair-slice size (small for contention/in-call)")
    ap.add_argument("--n-steps", type=int, default=24, help="seg-only optimization steps per arm")
    ap.add_argument("--seg-weight", type=float, default=100.0, help="matched seg Lagrangian weight")
    ap.add_argument("--adamw-lr", type=float, default=1e-3, help="matched LR for every arm")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--margin-weight-tau", type=float, default=0.3,
                    help="STATIC Lever-5 τ for arms a/b/c (held); arm d slaves τ(t)=T(t)")
    ap.add_argument("--t-floor", type=float, default=SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR)
    ap.add_argument("--t-max", type=float, default=SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX)
    ap.add_argument(
        "--slices", type=str, default="0,1",
        help="comma list of slice-START indices (each takes n-pairs from there); robustness across >=2 slices",
    )
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)  # SegNet has non-deterministic interp; eval is no_grad

    if not _FORKPOINT.exists():
        raise SystemExit(f"forkpoint not found: {_FORKPOINT}")
    if not _VIDEO.exists():
        raise SystemExit(f"video not found: {_VIDEO}")

    print(f"[probe] loading forkpoint {_FORKPOINT}", flush=True)
    ckpt = torch.load(_FORKPOINT, map_location="cpu", weights_only=False)
    base_channels = 20
    latent_dim = 28
    base_decoder_state = ckpt["ema_decoder"]  # bare vendored HNeRVDecoder state (EMA shadow)
    ema_latents = ckpt["ema_latents"].clone().detach()
    n_pairs_ckpt = ema_latents.shape[0]
    print(f"[probe] base_ch={base_channels} latent_dim={latent_dim} n_pairs_ckpt={n_pairs_ckpt}", flush=True)

    max_slice_end = max(int(s) for s in args.slices.split(",")) + args.n_pairs
    max_pairs_needed = min(n_pairs_ckpt, max_slice_end)
    print(f"[probe] building RealScorerContext (real frozen SegNet/PoseNet, CPU authority, max_pairs={max_pairs_needed})", flush=True)
    scorer = RealScorerContext(
        video_path=_VIDEO,
        device="cpu",  # CPU AUTHORITY — NO MPS
        max_pairs=max_pairs_needed,
        targets_cache=_TARGETS_CACHE,
    )
    print(f"[probe] scorer ready: n_pairs={scorer.n_pairs}", flush=True)

    vbundle = import_vendored_bundle()
    ce_seg_loss = import_vendored("stages.stage1_v328_ce").ce_seg_loss

    # Build the SHARED v2 wrapper init ONCE: vendored decoder loaded from the forkpoint,
    # wrapped with the v2 residual pose-FiLM (identity at init -> f1 bit-clean). The shared
    # state dict + stored pose are deepcopied per arm so every arm starts bit-identical.
    def wrapper_factory():
        dec = vbundle.HNeRVDecoder(
            latent_dim=latent_dim, base_channels=base_channels, eval_size=(_EVAL_H, _EVAL_W)
        ).to("cpu")
        return PoseFiLMHNeRVWrapperV2(dec, n_pairs=n_pairs_ckpt, pose_dim=_POSE_DIM).to("cpu")

    shared = wrapper_factory()
    shared.decoder.load_state_dict(base_decoder_state)
    # Stored pose = the GT pose targets the scorer precomputed (the Quantizr STORE payload).
    # f1 is FiLM-clean so seg is pose-invariant regardless; we set it for path fidelity.
    stored_pose = torch.zeros(n_pairs_ckpt, _POSE_DIM)
    pose_targets = getattr(scorer, "pose_targets", None)
    if pose_targets is not None:
        n_have = min(int(pose_targets.shape[0]), n_pairs_ckpt)
        stored_pose[:n_have] = pose_targets[:n_have].detach().to(stored_pose.dtype).cpu()
    base_wrapper_state = copy.deepcopy(shared.state_dict())

    slice_starts = [int(s) for s in args.slices.split(",")]
    all_results: dict[str, list[dict]] = {}

    for s_start in slice_starts:
        if s_start + args.n_pairs > scorer.n_pairs:
            print(f"[probe] SKIP slice {s_start} (would exceed n_pairs={scorer.n_pairs})", flush=True)
            continue
        idx = torch.arange(s_start, s_start + args.n_pairs, device="cpu")
        print(f"\n[probe] === SLICE start={s_start} n={args.n_pairs} steps={args.n_steps} ===", flush=True)
        slice_results: list[dict] = []
        for arm_label, schedule_id, schedule_kwargs in _ARM_SCHEDULES:
            res = _run_arm(
                arm_label=arm_label,
                schedule_id=schedule_id,
                schedule_kwargs=schedule_kwargs,
                base_wrapper_state=base_wrapper_state,
                base_latents=ema_latents,
                base_stored_pose=stored_pose,
                scorer=scorer,
                wrapper_factory=wrapper_factory,
                idx=idx,
                n_steps=args.n_steps,
                seg_weight=args.seg_weight,
                adamw_lr=args.adamw_lr,
                grad_clip=args.grad_clip,
                margin_weight_tau=args.margin_weight_tau,
                ce_seg_loss=ce_seg_loss,
                t_floor=args.t_floor,
                t_max=args.t_max,
            )
            row = dict(res.__dict__)
            row["slice_start"] = s_start
            slice_results.append(row)
            t_lo = min(res.temps_per_step) if res.temps_per_step else float("nan")
            t_hi = max(res.temps_per_step) if res.temps_per_step else float("nan")
            print(
                f"  {arm_label:24s} d_seg {res.d_seg_before:.5f}->{res.d_seg_after:.5f} "
                f"(Δ={res.d_seg_reduction:+.5f}) min={res.d_seg_min:.5f} "
                f"flips_fixed={res.flips_fixed:+d} T[{t_lo:.2f},{t_hi:.2f}] "
                f"gn0={res.grad_norm_first:.3e} loss {res.loss_first:.3f}->{res.loss_last:.3f} "
                f"[{res.seconds:.1f}s]",
                flush=True,
            )
        all_results[f"slice_{s_start}"] = slice_results

    # --- Verdict synthesis -------------------------------------------------
    # The schedule that reaches the LOWEST d_seg over the window WINS. We rank by the
    # AVERAGE (across slices) of d_seg_min (the lowest d_seg the schedule reached — the
    # max-signal metric: a schedule that dips low then drifts is still credited the dip)
    # AND report d_seg_after (the end-of-window state). Both are reported per arm.
    def _avg(arm_label: str, fieldname: str) -> float:
        vals = [
            r[fieldname]
            for slr in all_results.values()
            for r in slr
            if r["arm"] == arm_label
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    arm_labels = [a[0] for a in _ARM_SCHEDULES]
    ranking = []
    for arm_label in arm_labels:
        ranking.append(
            {
                "arm": arm_label,
                "avg_d_seg_min": _avg(arm_label, "d_seg_min"),
                "avg_d_seg_after": _avg(arm_label, "d_seg_after"),
                "avg_d_seg_reduction": _avg(arm_label, "d_seg_reduction"),
            }
        )
    ranking.sort(key=lambda r: r["avg_d_seg_min"])  # lowest d_seg_min first = best
    best = ranking[0]
    margin_adaptive_row = next((r for r in ranking if r["arm"] == "margin_adaptive"), None)
    margin_adaptive_wins = best["arm"] == "margin_adaptive"

    # Δ_median collapse evidence: the per-step flip-margin median should DECREASE toward
    # the ~0.3 floor over the window (validates the resonance theory). Report from the
    # margin_adaptive arm of the FIRST slice (every arm records it; adaptive is canonical).
    def _first_adaptive_margins() -> list[float]:
        for slr in all_results.values():
            for r in slr:
                if r["arm"] == "margin_adaptive":
                    return r["margin_median_per_step"]
        return []

    margins = [m for m in _first_adaptive_margins() if m == m]  # drop NaN
    collapse = None
    if len(margins) >= 4:
        head = sum(margins[:2]) / 2.0
        tail = sum(margins[-2:]) / 2.0
        collapse = {
            "head_median_margin": head,
            "tail_median_margin": tail,
            "decreased": tail < head,
            "tail_near_floor": tail <= (args.t_floor + 0.15),
        }

    print("\n[probe] ===== CALIBRATION VERDICT (lowest avg d_seg_min WINS) =====", flush=True)
    for r in ranking:
        marker = "  <-- BEST" if r["arm"] == best["arm"] else ""
        print(
            f"  {r['arm']:24s} avg_d_seg_min={r['avg_d_seg_min']:.6f} "
            f"avg_d_seg_after={r['avg_d_seg_after']:.6f} "
            f"avg_Δd_seg={r['avg_d_seg_reduction']:+.6f}{marker}",
            flush=True,
        )
    print(f"[probe] BEST schedule = {best['arm']} "
          f"(margin_adaptive {'WINS' if margin_adaptive_wins else 'does NOT win'})", flush=True)
    if collapse is not None:
        print(
            f"[probe] Δ_median collapse: head={collapse['head_median_margin']:.3f} -> "
            f"tail={collapse['tail_median_margin']:.3f} "
            f"(decreased={collapse['decreased']}, near_floor={collapse['tail_near_floor']})",
            flush=True,
        )

    out = {
        "probe": "anneal_schedule_calibration_v2",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "decoder": "PoseFiLMHNeRVWrapperV2 (Lever-3 v2 clean-rgb_1)",
        "forkpoint": str(_FORKPOINT),
        "config": vars(args),
        "ranking": ranking,
        "best_schedule": best["arm"],
        "margin_adaptive_wins": margin_adaptive_wins,
        "margin_adaptive_row": margin_adaptive_row,
        "delta_median_collapse": collapse,
        "results": all_results,
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"[probe] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
