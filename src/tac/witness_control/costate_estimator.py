"""Costate estimation from MEASURED witness trajectories (task #303 Phase A).

The costate λ_i = ∂S/∂x_i is the shadow price of state component x_i against the
contest score law

    S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37_545_489     (upstream/evaluate.py)

Three honesty tiers, each carried on every estimate (NO-FAKE discipline):

* ``ANALYTIC``       — exact partials of the score law itself (the "equations" leg of
                       the triality). Zero estimation noise; state-dependent where the
                       law is nonlinear (λ_pose = 5/sqrt(10·d_pose)).
* ``MEASURED``       — finite differences over real n600 verdict rows (least-squares
                       slopes with standard errors from fit residuals; event jumps
                       with noise floors from neighboring-fit residuals). The verdict
                       rows are the trainer's advisory n600 evals — all numbers stay
                       ``[macOS advisory] NON-PROMOTABLE``.
* ``PARTIAL``        — a measured sensitivity of an INTERNAL state (e.g. focal-γ →
                       island-grad-share from a calibration sweep) whose chain to S is
                       NOT measured. Usable as direction/ranking evidence only.
* ``UNIDENTIFIABLE`` — the honest refusal: the data cannot identify this costate
                       (n too small, confounded A/B, stage never ran). Carries the
                       probe that WOULD identify it instead of a guess.

Pure math + dataclasses only. No IO, no trainer imports, no actuation capability
(structural CONTAINMENT — see the package docstring).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# ── score-law constants (upstream/evaluate.py; the "equations" leg) ──
SEG_WEIGHT = 100.0
POSE_WEIGHT_INNER = 10.0          # S_pose = sqrt(10 · d_pose)
RATE_NUMERATOR = 25.0
RATE_DENOMINATOR = 37_545_489.0   # exact archive-byte normalizer

# status sentinels (falling honesty ladder)
ANALYTIC = "ANALYTIC"
MEASURED = "MEASURED"
PARTIAL = "PARTIAL"
UNIDENTIFIABLE = "UNIDENTIFIABLE"

#: minimum same-stage verdict rows before a slope's stderr is defined (n-2 dof)
MIN_ROWS_FOR_SLOPE = 3
#: z-multiplier for the reported ± band (≈95%)
BAND_Z = 2.0


@dataclass(frozen=True)
class CostateEstimate:
    """One costate (or control-sensitivity) estimate with its full evidence chain."""

    name: str                      # e.g. "lambda_d_seg", "dS_depoch[stage=ce]"
    value: float | None            # central estimate (None iff UNIDENTIFIABLE)
    stderr: float | None           # standard error (None = not defined / exact 0 noise)
    status: str                    # ANALYTIC | MEASURED | PARTIAL | UNIDENTIFIABLE
    units: str                     # e.g. "S per unit d_seg", "S per epoch"
    method: str                    # how it was computed (Rudin readback)
    evidence: tuple[str, ...]      # row/probe citations that produced it
    n: int = 0                     # number of measured rows behind it

    def band(self) -> tuple[float, float] | None:
        """±BAND_Z·stderr band around the central value (None when undefined)."""
        if self.value is None or self.stderr is None:
            return None
        h = BAND_Z * self.stderr
        return (self.value - h, self.value + h)

    def to_dict(self) -> dict:
        d = asdict(self)
        b = self.band()
        d["band"] = list(b) if b is not None else None
        d["evidence"] = list(self.evidence)
        return d


@dataclass(frozen=True)
class SlopeFit:
    """Least-squares slope with honestly-propagated uncertainty."""

    slope: float
    stderr: float | None           # None when n < MIN_ROWS_FOR_SLOPE (no dof)
    n: int
    x_lo: float = 0.0
    x_hi: float = 0.0
    resid_std: float = 0.0         # residual std around the fit (level-noise floor)


def slope_with_stderr(xs: list[float], ys: list[float]) -> SlopeFit:
    """Least-squares slope dy/dx with the standard error from fit residuals.

    stderr = sqrt( (RSS/(n-2)) / Sxx ) — the classical OLS slope SE. n < 2 → slope 0,
    n < MIN_ROWS_FOR_SLOPE → stderr None (no residual degrees of freedom: the honest
    answer is "slope exists, uncertainty UNIDENTIFIABLE from these rows").
    """
    n = min(len(xs), len(ys))
    xs, ys = list(xs[:n]), list(ys[:n])
    if n < 2:
        return SlopeFit(slope=0.0, stderr=None, n=n)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return SlopeFit(slope=0.0, stderr=None, n=n,
                        x_lo=min(xs), x_hi=max(xs))
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    resid_std = math.sqrt(rss / n) if n else 0.0
    if n < MIN_ROWS_FOR_SLOPE:
        return SlopeFit(slope=slope, stderr=None, n=n, x_lo=min(xs), x_hi=max(xs),
                        resid_std=resid_std)
    stderr = math.sqrt((rss / (n - 2)) / sxx)
    return SlopeFit(slope=slope, stderr=stderr, n=n, x_lo=min(xs), x_hi=max(xs),
                    resid_std=resid_std)


# ─────────────────────────── analytic costates (exact) ───────────────────────────
def analytic_costates(d_pose_latest: float | None) -> list[CostateEstimate]:
    """The exact partials of the contest score law at the current operating point.

    λ_seg = 100 (constant) · λ_rate = 25/37_545_489 per byte (constant) ·
    λ_pose = 5/sqrt(10·d_pose) (state-dependent; UNIDENTIFIABLE at d_pose ≤ 0 —
    the derivative diverges as the pose term hits its floor, which is itself the
    operating-point crossover finding in CLAUDE.md "SegNet vs PoseNet importance").
    """
    law = "S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37545489 (upstream/evaluate.py:score law)"
    out = [
        CostateEstimate(
            name="lambda_d_seg", value=SEG_WEIGHT, stderr=0.0, status=ANALYTIC,
            units="S per unit d_seg", method="exact partial ∂S/∂d_seg of the score law",
            evidence=(law,), n=0),
        CostateEstimate(
            name="lambda_bytes", value=RATE_NUMERATOR / RATE_DENOMINATOR, stderr=0.0,
            status=ANALYTIC, units="S per archive byte",
            method="exact partial ∂S/∂bytes of the score law", evidence=(law,), n=0),
    ]
    if d_pose_latest is None or d_pose_latest <= 0.0:
        out.append(CostateEstimate(
            name="lambda_d_pose", value=None, stderr=None, status=UNIDENTIFIABLE,
            units="S per unit d_pose",
            method="∂S/∂d_pose = 5/sqrt(10·d_pose) diverges at d_pose ≤ 0",
            evidence=(law, f"d_pose_latest={d_pose_latest!r} (no positive operating point)"),
            n=0))
    else:
        lam = 5.0 / math.sqrt(POSE_WEIGHT_INNER * d_pose_latest)
        out.append(CostateEstimate(
            name="lambda_d_pose", value=lam, stderr=0.0, status=ANALYTIC,
            units="S per unit d_pose",
            method=f"exact partial 5/sqrt(10·d_pose) at measured d_pose={d_pose_latest:.6g}",
            evidence=(law, f"operating point d_pose={d_pose_latest:.6g}"), n=0))
    return out


# ─────────────────────── trajectory costates (measured) ───────────────────────
def _stage_rows(verdicts: list[dict], stage: str) -> list[dict]:
    return [v for v in verdicts
            if str(v.get("seg_form", "")) == stage
            and isinstance(v.get("d_seg"), (int, float))
            and isinstance(v.get("epoch"), (int, float))]


#: local-fit window (same-stage verdict rows) — matches the canonical monitor's
#: window=5 semantics. The costate λ(t) is TIME-VARYING (the tau-onset transient
#: decays ~10x from onset to steady creep on the #205 trace); a full-stage fit
#: over-weights the decayed transient, so the LOCAL window is the honest estimator
#: for a forward horizon projection. window=0 → full-stage (the persistence test).
DEFAULT_LOCAL_WINDOW = 5


def stage_epoch_costates(verdicts: list[dict], stage: str,
                         window: int = DEFAULT_LOCAL_WINDOW) -> dict[str, SlopeFit]:
    """Per-epoch finite-difference sensitivities of each state channel within a stage.

    Returns ``{"d_seg": SlopeFit, "d_pose": ..., "blob_bytes": ..., "ep_loss": ...}``
    over the LAST ``window`` same-stage verdict rows (0 = all; see
    DEFAULT_LOCAL_WINDOW for why local). These are the raw dx/dep pieces the chain
    rule combines into dS/dep. Channels missing from the rows fit over what exists.
    """
    rows = _stage_rows(verdicts, stage)
    if window and window > 0:
        rows = rows[-int(window):]
    eps = [float(v["epoch"]) for v in rows]
    out: dict[str, SlopeFit] = {}
    for ch in ("d_seg", "d_pose", "blob_bytes", "ep_loss"):
        pts = [(e, float(v[ch])) for e, v in zip(eps, rows, strict=True)
               if isinstance(v.get(ch), (int, float))]
        out[ch] = slope_with_stderr([p[0] for p in pts], [p[1] for p in pts])
    return out


def chain_ds_depoch(fits: dict[str, SlopeFit], d_pose_latest: float | None,
                    stage: str, evidence: tuple[str, ...]) -> CostateEstimate:
    """dS/depoch within a stage = λ_seg·(dd_seg/dep) + λ_pose·(dd_pose/dep) + λ_bytes·(dbytes/dep).

    Uncertainty: sqrt(Σ (λ_i·se_i)²) — independence approximation across channels,
    stated openly (the channels share epochs, so this is a floor, not gospel).
    MEASURED only when the d_seg slope has a defined stderr; else UNIDENTIFIABLE.
    """
    seg = fits.get("d_seg")
    if seg is None or seg.n < MIN_ROWS_FOR_SLOPE or seg.stderr is None:
        return CostateEstimate(
            name=f"dS_depoch[stage={stage}]", value=None, stderr=None,
            status=UNIDENTIFIABLE, units="S per epoch",
            method="chain rule over per-stage slopes",
            evidence=(*evidence, f"only {0 if seg is None else seg.n} same-stage verdict rows " f"(need ≥ {MIN_ROWS_FOR_SLOPE}) — wait for more evals"),
            n=(0 if seg is None else seg.n))
    lam_pose = (5.0 / math.sqrt(POSE_WEIGHT_INNER * d_pose_latest)
                if (d_pose_latest is not None and d_pose_latest > 0.0) else None)
    lam_bytes = RATE_NUMERATOR / RATE_DENOMINATOR
    val = SEG_WEIGHT * seg.slope
    var = (SEG_WEIGHT * seg.stderr) ** 2
    terms = [f"100·({seg.slope:+.3e})"]
    pose = fits.get("d_pose")
    if lam_pose is not None and pose is not None and pose.stderr is not None:
        val += lam_pose * pose.slope
        var += (lam_pose * pose.stderr) ** 2
        terms.append(f"{lam_pose:.3g}·({pose.slope:+.3e})")
    byt = fits.get("blob_bytes")
    if byt is not None and byt.stderr is not None:
        val += lam_bytes * byt.slope
        var += (lam_bytes * byt.stderr) ** 2
        terms.append(f"{lam_bytes:.3g}·({byt.slope:+.3e})")
    return CostateEstimate(
        name=f"dS_depoch[stage={stage}]", value=val, stderr=math.sqrt(var),
        status=MEASURED, units="S per epoch",
        method="chain rule: dS/dep = " + " + ".join(terms)
               + " (independence approx across channels)",
        evidence=evidence, n=seg.n)


def transition_jump_costate(verdicts: list[dict], from_stage: str,
                            to_stage: str) -> CostateEstimate:
    """The measured ΔS of a fired stage-advance control: the boundary jump.

    Jump = (first ``to_stage`` verdict d_seg) − (last ``from_stage`` verdict d_seg),
    converted to S units via λ_seg (+ pose channel where present). Noise floor: the
    combined residual std of the two neighboring within-stage fits (a one-point event
    diff has no dof of its own — the honest uncertainty is the level noise around it).
    """
    pre = _stage_rows(verdicts, from_stage)
    post = _stage_rows(verdicts, to_stage)
    name = f"dS_jump[{from_stage}->{to_stage}]"
    if not pre or not post:
        return CostateEstimate(
            name=name, value=None, stderr=None, status=UNIDENTIFIABLE,
            units="S per transition",
            method="boundary jump (first-post − last-pre)",
            evidence=(f"pre rows n={len(pre)}, post rows n={len(post)} — the "
                      f"{to_stage} stage has not produced a verdict yet",),
            n=len(pre) + len(post))
    last_pre = pre[-1]
    first_post = post[0]
    d_jump = float(first_post["d_seg"]) - float(last_pre["d_seg"])
    val = SEG_WEIGHT * d_jump
    pre_fit = slope_with_stderr([float(v["epoch"]) for v in pre],
                                [float(v["d_seg"]) for v in pre])
    post_fit = slope_with_stderr([float(v["epoch"]) for v in post],
                                 [float(v["d_seg"]) for v in post])
    noise = SEG_WEIGHT * math.sqrt(pre_fit.resid_std ** 2 + post_fit.resid_std ** 2)
    return CostateEstimate(
        name=name, value=val, stderr=(noise if noise > 0 else None), status=MEASURED,
        units="S per transition",
        method=(f"100·(d_seg first-{to_stage} {float(first_post['d_seg']):.6f}@ep"
                f"{int(first_post['epoch'])} − last-{from_stage} "
                f"{float(last_pre['d_seg']):.6f}@ep{int(last_pre['epoch'])}); noise floor "
                f"= combined neighboring-fit residual std"),
        evidence=(f"verdict@ep{int(last_pre['epoch'])} [{from_stage}]",
                  f"verdict@ep{int(first_post['epoch'])} [{to_stage}]"),
        n=len(pre) + len(post))


def _implied_s(v: dict) -> float | None:
    """implied_S from the row when present; else recomputed from components."""
    s = v.get("implied_S")
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return (SEG_WEIGHT * float(v["d_seg"])
                + math.sqrt(POSE_WEIGHT_INNER * float(v["d_pose"]))
                + RATE_NUMERATOR * float(v["blob_bytes"]) / RATE_DENOMINATOR)
    except (KeyError, TypeError, ValueError):
        return None


def rollback_gain(verdicts: list[dict]) -> CostateEstimate:
    """ΔS of rolling back to the best checkpoint: S_best − S_latest (≤ 0 = recoverable gain).

    Pure arithmetic on measured rows (best-by-d_seg mirrors the trainer's BEST-checkpoint
    rule). Zero estimation noise; the residual caveat is row-level measurement noise,
    which the trainer's advisory verdicts do not resample — stated in the method string.
    """
    rows = [v for v in verdicts if isinstance(v.get("d_seg"), (int, float))
            and isinstance(v.get("epoch"), (int, float))]
    if len(rows) < 2:
        return CostateEstimate(
            name="dS_rollback_to_best", value=None, stderr=None, status=UNIDENTIFIABLE,
            units="S per rollback",
            method="S_best − S_latest over measured verdict rows",
            evidence=(f"only {len(rows)} verdict rows — nothing to roll back across",),
            n=len(rows))
    best = min(rows, key=lambda v: float(v["d_seg"]))
    latest = rows[-1]
    s_best, s_latest = _implied_s(best), _implied_s(latest)
    if s_best is None or s_latest is None:
        return CostateEstimate(
            name="dS_rollback_to_best", value=None, stderr=None, status=UNIDENTIFIABLE,
            units="S per rollback", method="S_best − S_latest",
            evidence=("rows lack implied_S and full components",), n=len(rows))
    return CostateEstimate(
        name="dS_rollback_to_best", value=s_latest - s_best, stderr=None, status=MEASURED,
        units="S recoverable by rollback (positive = rollback improves S by this much)",
        method=(f"S_latest({s_latest:.4f}@ep{int(latest['epoch'])}) − "
                f"S_best({s_best:.4f}@ep{int(best['epoch'])}); exact arithmetic on measured "
                f"rows (single-eval rows — no resample noise estimate available)"),
        evidence=(f"verdict@ep{int(best['epoch'])} (best d_seg={float(best['d_seg']):.6f})",
                  f"verdict@ep{int(latest['epoch'])} (latest d_seg={float(latest['d_seg']):.6f})"),
        n=len(rows))


# ─────────────────────── cross-run / probe costates ───────────────────────
def cross_run_lever_costate(lever_diffs: dict[str, tuple], run_a_label: str,
                            run_b_label: str, d_seg_a: float, d_seg_b: float,
                            epoch: int) -> CostateEstimate:
    """A/B lever costate from two runs at a matched epoch — HONEST about confounding.

    Identifiable ONLY when the two configs differ in exactly one lever. With ≥ 2
    diffs the per-lever attribution is CONFOUNDED → UNIDENTIFIABLE, and the estimate
    carries the full diff list as the reason (the probe queue entry is "run the
    single-lever A/B"). This is the NO-FAKE boundary: a 13-change config pair yields
    a JOINT effect, never 13 per-lever numbers.
    """
    joint = SEG_WEIGHT * (d_seg_b - d_seg_a)
    levers = sorted(lever_diffs)
    if len(lever_diffs) == 1:
        lever = levers[0]
        a_val, b_val = lever_diffs[lever]
        return CostateEstimate(
            name=f"dS_dlever[{lever}]", value=joint, stderr=None, status=MEASURED,
            units=f"S per ({a_val!r} -> {b_val!r})",
            method=(f"matched-epoch(ep{epoch}) single-lever A/B: "
                    f"100·(d_seg[{run_b_label}] − d_seg[{run_a_label}]) "
                    f"(single-eval rows; no resample noise estimate)"),
            evidence=(f"{run_a_label} verdict@ep{epoch} d_seg={d_seg_a:.6f}",
                      f"{run_b_label} verdict@ep{epoch} d_seg={d_seg_b:.6f}"),
            n=2)
    return CostateEstimate(
        name=f"dS_dlever[{'+'.join(levers)}]", value=None, stderr=None,
        status=UNIDENTIFIABLE, units="S per lever",
        method=(f"matched-epoch(ep{epoch}) A/B CONFOUNDED: configs differ in "
                f"{len(lever_diffs)} levers — joint effect {joint:+.4f} S is measured "
                f"but per-lever attribution is not"),
        evidence=(*tuple(f"diff {k}: {a!r} -> {b!r}" for k, (a, b) in sorted(lever_diffs.items())), f"{run_a_label} d_seg={d_seg_a:.6f} vs {run_b_label} d_seg={d_seg_b:.6f} @ep{epoch}", "probe: single-lever A/B pair required for attribution"),
        n=2)


def sweep_finite_difference(name: str, points: list[tuple[float, float]],
                            units: str, evidence: tuple[str, ...],
                            chained_to_S: bool = False) -> CostateEstimate:
    """Generic probe-sweep sensitivity (e.g. focal γ → island-grad-share).

    Fits d(observable)/d(knob) over the sweep points. When the observable is an
    INTERNAL state (``chained_to_S=False``) the status is PARTIAL — measured
    direction, unmeasured chain to S (the focal-calibration exemplar: the γ-sweep
    measures ∂grad-share/∂γ; ∂S/∂grad-share is NOT measured).
    """
    fit = slope_with_stderr([p[0] for p in points], [p[1] for p in points])
    if fit.n < 2:
        return CostateEstimate(
            name=name, value=None, stderr=None, status=UNIDENTIFIABLE, units=units,
            method="finite difference over a probe sweep",
            evidence=(*evidence, f"only {fit.n} sweep points"), n=fit.n)
    status = MEASURED if chained_to_S else PARTIAL
    note = "" if chained_to_S else " (internal observable — chain to S NOT measured)"
    return CostateEstimate(
        name=name, value=fit.slope, stderr=fit.stderr, status=status, units=units,
        method=f"least-squares over {fit.n} sweep points{note}",
        evidence=evidence, n=fit.n)
