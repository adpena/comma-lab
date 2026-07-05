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


# ═══════════════════════════════════════════════════════════════════════════
# BINDING-TERM-STALL DETECTOR (task #315) — kill the scalar-S deadlock blind spot
# ═══════════════════════════════════════════════════════════════════════════
# The score has three terms; their STRUCTURAL weights are 100·d_seg, sqrt(10·d_pose),
# 25·bytes/N. The MISSION-BINDING term is d_seg (λ_seg = 100, the largest CONSTANT
# weight; per CLAUDE.md "the witness's sole binding controllable job is d_seg" —
# pose rides the stored-target sidecar, bytes are ~fixed by the blob). The v5
# deadlock (ep110-172, frozen-descending-S) is the failure the scalar classifier
# MISSES: d_seg (the binding term) FLAT while implied_S still DESCENDS via a
# non-binding term (pose noise / bytes drift) OR the ep_loss surrogate still falls.
# The scalar classifier reads d_seg alone → calls a FLAT binding term "PLATEAU"
# (recommends advance/early-stop = "converged, fine") when the run is actually
# DEADLOCKED on the term that matters. This detector reads d_seg AND (implied_S,
# ep_loss) jointly and fires BINDING_TERM_STALL when the binding term is flat but
# a non-binding signal is still materially moving. Pure math; NO actuation.

NO_STALL = "NO_STALL"
BINDING_TERM_STALL = "BINDING_TERM_STALL"
BINDING_STALL_UNIDENTIFIABLE = "BINDING_STALL_UNIDENTIFIABLE"

#: |slope|/level per epoch AT or BELOW which the binding term is "flat" (stall gate).
#: CALIBRATED on the real logs (2026-07-05): the l7 flat-stall window (v2_attrclean
#: ep600-725) sits at |d_seg_rel_slope| ~1e-4/ep; a healthy CE descent is ~-1e-2/ep
#: and a slow tau descent ~-2e-3/ep — BOTH ≥ an order above 3e-4. 3e-4 keeps the
#: genuine flat-binding stalls and rejects the still-descending windows (a 2e-3 gate
#: false-fired on -1.5e-3/ep CE descent — MEASURED + corrected).
DEFAULT_STALL_REL_EPS = 3e-4
#: |slope|/level per epoch AT or ABOVE which a NON-binding channel counts as
#: "materially moving" (so a flat binding term + a moving other-signal = stall).
DEFAULT_MOVE_REL_EPS = 5e-4
#: default within-stage window (matches the canonical monitor's window=5)
DEFAULT_STALL_WINDOW = 5


def _rel_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """(slope, level, rel_slope=slope/|level|) over (xs, ys). level = mean(ys).
    rel_slope 0.0 when level ≈ 0 (degenerate — reported flat, never a div-by-zero)."""
    fit = slope_with_stderr(xs, ys)
    level = (sum(ys) / len(ys)) if ys else 0.0
    rel = (fit.slope / abs(level)) if abs(level) > 0.0 else 0.0
    return fit.slope, level, rel


@dataclass(frozen=True)
class BindingStallVerdict:
    """Whether the mission-binding term (d_seg) is DEADLOCKED while a non-binding
    signal still moves — the failure the scalar d_seg classifier misses."""

    classification: str            # NO_STALL | BINDING_TERM_STALL | BINDING_STALL_UNIDENTIFIABLE
    binding_term: str              # "d_seg" (structural mission-binding term)
    level_dominant_term: str       # which term dominates S LEVEL (transparency; may != binding)
    stage: str
    n_window: int
    d_seg_rel_slope: float         # per-epoch slope/level of the binding term (flat when |·|≤stall_eps)
    s_rel_slope: float             # per-epoch slope/level of implied_S
    loss_rel_slope: float          # per-epoch slope/level of ep_loss (surrogate)
    reason: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        return d

    def fired(self) -> bool:
        return self.classification == BINDING_TERM_STALL


def _level_dominant_term(row: dict) -> str:
    """Which of the three score terms dominates the CURRENT S LEVEL (transparency).
    100·d_seg vs sqrt(10·d_pose) vs 25·bytes/N. Honest 'unknown' when fields missing."""
    parts: dict[str, float] = {}
    if isinstance(row.get("d_seg"), (int, float)):
        parts["d_seg"] = SEG_WEIGHT * float(row["d_seg"])
    if isinstance(row.get("d_pose"), (int, float)) and float(row["d_pose"]) >= 0.0:
        parts["d_pose"] = math.sqrt(POSE_WEIGHT_INNER * float(row["d_pose"]))
    if isinstance(row.get("blob_bytes"), (int, float)):
        parts["bytes"] = RATE_NUMERATOR * float(row["blob_bytes"]) / RATE_DENOMINATOR
    if not parts:
        return "unknown"
    return max(parts, key=lambda k: parts[k])


def binding_term_stall(
    verdicts: list[dict], *, window: int = DEFAULT_STALL_WINDOW,
    stall_rel_eps: float = DEFAULT_STALL_REL_EPS,
    move_rel_eps: float = DEFAULT_MOVE_REL_EPS,
) -> BindingStallVerdict:
    """Detect a BINDING-TERM STALL over the last ``window`` SAME-STAGE verdict rows.

    Fires BINDING_TERM_STALL iff the binding term (d_seg) is FLAT
    (|d_seg_rel_slope| ≤ stall_rel_eps) AND a NON-binding signal is still
    materially moving DOWNWARD — either implied_S descending (s_rel_slope ≤
    -move_rel_eps: the frozen-descending-S signature) OR the ep_loss surrogate
    descending (loss_rel_slope ≤ -move_rel_eps: the surrogate↔verdict decoupling).
    A flat binding term with EVERYTHING flat is a genuine plateau → NO_STALL (the
    scalar classifier's PLATEAU is then correct). Rising d_seg is EROSION, handled
    by the canonical DIVERGING_ERASING rule → NO_STALL here (not our lane).

    Pure: same rows → same verdict. UNIDENTIFIABLE when < 2 same-stage rows carry
    d_seg (no slope) OR no non-binding signal (implied_S/ep_loss) is present."""
    same_stage = [v for v in verdicts
                  if isinstance(v.get("d_seg"), (int, float))
                  and isinstance(v.get("epoch"), (int, float))]
    if same_stage:
        latest_stage = str(same_stage[-1].get("seg_form", ""))
        same_stage = [v for v in same_stage if str(v.get("seg_form", "")) == latest_stage]
    else:
        latest_stage = ""
    win = same_stage[-int(window):] if (window and window > 0) else same_stage
    if len(win) < 2:
        return BindingStallVerdict(
            classification=BINDING_STALL_UNIDENTIFIABLE, binding_term="d_seg",
            level_dominant_term="unknown", stage=latest_stage, n_window=len(win),
            d_seg_rel_slope=0.0, s_rel_slope=0.0, loss_rel_slope=0.0,
            reason=f"only {len(win)} same-stage verdict row(s) — need ≥ 2 for a slope",
            evidence=(f"{len(win)} usable rows in the window (stage={latest_stage!r})",))
    eps = [float(v["epoch"]) for v in win]
    dsegs = [float(v["d_seg"]) for v in win]
    _, _, ds_rel = _rel_slope(eps, dsegs)

    # implied_S per row (recomputed from components when the field is absent).
    s_vals = [_implied_s(v) for v in win]
    have_s = all(s is not None for s in s_vals)
    s_rel = 0.0
    if have_s:
        _, _, s_rel = _rel_slope(eps, [float(s) for s in s_vals])  # type: ignore[arg-type]
    loss_vals = [v.get("ep_loss") for v in win]
    have_loss = all(isinstance(x, (int, float)) for x in loss_vals)
    loss_rel = 0.0
    if have_loss:
        _, _, loss_rel = _rel_slope(eps, [float(x) for x in loss_vals])  # type: ignore[arg-type]

    dom = _level_dominant_term(win[-1])
    if not have_s and not have_loss:
        return BindingStallVerdict(
            classification=BINDING_STALL_UNIDENTIFIABLE, binding_term="d_seg",
            level_dominant_term=dom, stage=latest_stage, n_window=len(win),
            d_seg_rel_slope=ds_rel, s_rel_slope=0.0, loss_rel_slope=0.0,
            reason="no non-binding signal (neither implied_S nor ep_loss present) — "
                   "cannot tell a stall from a genuine plateau",
            evidence=(f"d_seg_rel_slope={ds_rel:+.2e}/ep over ep{int(eps[0])}-{int(eps[-1])}",
                      "rows carry d_seg only — add implied_S or ep_loss to disambiguate"))

    binding_flat = abs(ds_rel) <= stall_rel_eps
    s_descending = have_s and (s_rel <= -move_rel_eps)
    loss_descending = have_loss and (loss_rel <= -move_rel_eps)
    ev = (f"d_seg_rel_slope={ds_rel:+.2e}/ep (flat gate ≤{stall_rel_eps:.0e}) over "
          f"ep{int(eps[0])}-{int(eps[-1])} [stage={latest_stage}]",
          (f"implied_S_rel_slope={s_rel:+.2e}/ep" if have_s else "implied_S absent"),
          (f"ep_loss_rel_slope={loss_rel:+.2e}/ep" if have_loss else "ep_loss absent"),
          f"S-level dominated by {dom} (transparency; binding term is d_seg by "
          f"structural λ per CLAUDE.md)")
    if binding_flat and (s_descending or loss_descending):
        drivers = []
        if s_descending:
            drivers.append(f"implied_S still descending ({s_rel:+.2e}/ep — via the "
                           f"non-binding {dom} term)")
        if loss_descending:
            drivers.append(f"ep_loss surrogate still descending ({loss_rel:+.2e}/ep — "
                           "surrogate↔verdict decoupling)")
        return BindingStallVerdict(
            classification=BINDING_TERM_STALL, binding_term="d_seg",
            level_dominant_term=dom, stage=latest_stage, n_window=len(win),
            d_seg_rel_slope=ds_rel, s_rel_slope=s_rel, loss_rel_slope=loss_rel,
            reason="binding term d_seg is FLAT while " + " AND ".join(drivers)
                   + " — NOT a converged plateau; the scalar classifier's "
                   "PLATEAU/CONVERGING 'advance-or-stop' verdict is a FALSE GREEN. "
                   "INVESTIGATE the deadlock (stale-weight async verdicts, spike-guard "
                   "freeze, LR collapse) before advancing/stopping.",
            evidence=ev)
    return BindingStallVerdict(
        classification=NO_STALL, binding_term="d_seg", level_dominant_term=dom,
        stage=latest_stage, n_window=len(win), d_seg_rel_slope=ds_rel,
        s_rel_slope=s_rel, loss_rel_slope=loss_rel,
        reason=("binding term moving (not flat)" if not binding_flat else
                "binding term flat AND non-binding signals also flat = genuine plateau"),
        evidence=ev)


# ─────────────────────── per-class within-flip costates (task #315) ───────────────────────
# The nucleus/handoff readiness rows (trainer `handoff_readiness` telemetry) carry a
# per-class breakdown: within_flip[c] (fraction of GT-class-c pixels mislabeled) +
# part_frac[c] (predicted partition area of class c). A per-class dS/dep is the
# surgical costate: which CLASS's flip mass is stalling. Read HONESTLY — absent
# per-class data → UNIDENTIFIABLE with the field-gap evidence, NEVER fabricated
# from the scalar d_seg.
def _row_per_class(row: dict) -> dict | None:
    """Extract {class_id: {"within_flip": float, "part_frac": float}} from a row.

    Tolerant of two shapes: (a) ``row["per_class"]`` = {c: {"within_flip"/"disagree",
    "part_frac"}}; (b) parallel ``row["within_flip"]`` + ``row["part_frac"]`` dicts.
    Returns None when neither is present (the honest 'no per-class data' signal)."""
    pc = row.get("per_class")
    if isinstance(pc, dict) and pc:
        out: dict = {}
        for k, v in pc.items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict):
                wf = v.get("within_flip", v.get("disagree"))
                out[ci] = {"within_flip": (float(wf) if isinstance(wf, (int, float)) else None),
                           "part_frac": (float(v["part_frac"])
                                         if isinstance(v.get("part_frac"), (int, float)) else None)}
        return out or None
    wf = row.get("within_flip")
    pf = row.get("part_frac")
    if isinstance(wf, dict) and wf:
        out = {}
        for k, v in wf.items():
            try:
                ci = int(k)
            except (TypeError, ValueError):
                continue
            out[ci] = {"within_flip": (float(v) if isinstance(v, (int, float)) else None),
                       "part_frac": (float(pf[k]) if isinstance(pf, dict)
                                     and isinstance(pf.get(k), (int, float)) else None)}
        return out or None
    return None


def per_class_within_flip_costates(verdicts: list[dict], stage: str,
                                   window: int = DEFAULT_LOCAL_WINDOW) -> CostateEstimate:
    """Per-class dS/dep from within_flip[c] slopes (ΔS = λ_seg · Σ_c part_frac[c]·Δwithin_flip[c]).

    Approximation: S_seg = 100·d_seg and d_seg = Σ_c part_frac_gt[c]·within_flip[c] over GT
    classes (weights = GT-class area). We fit within_flip[c] per class and report the
    class whose flip mass is stalling worst. Returns UNIDENTIFIABLE (with the exact
    field gap) when NO row carries per-class data — never fabricated from scalar d_seg."""
    rows = [v for v in verdicts
            if str(v.get("seg_form", "")) == stage
            and isinstance(v.get("epoch"), (int, float))]
    if window and window > 0:
        rows = rows[-int(window):]
    pcs = [(float(v["epoch"]), _row_per_class(v)) for v in rows]
    pcs = [(e, d) for e, d in pcs if d is not None]
    if len(pcs) < MIN_ROWS_FOR_SLOPE:
        return CostateEstimate(
            name=f"per_class_dS_depoch[stage={stage}]", value=None, stderr=None,
            status=UNIDENTIFIABLE, units="S per epoch (per-class)",
            method="per-class within_flip slopes → dS/dep",
            evidence=(f"only {len(pcs)} rows carry per-class data (need ≥ {MIN_ROWS_FOR_SLOPE}); "
                      "per-class breakdown comes from the trainer `handoff_readiness` telemetry "
                      "(NOT scalar d_seg — refusing to fabricate)",),
            n=len(pcs))
    classes = sorted({c for _, d in pcs for c in d})
    worst_c, worst_slope, per_c_slopes = None, None, {}
    for c in classes:
        pts = [(e, d[c]["within_flip"]) for e, d in pcs
               if c in d and isinstance(d[c].get("within_flip"), (int, float))]
        if len(pts) < MIN_ROWS_FOR_SLOPE:
            continue
        fit = slope_with_stderr([p[0] for p in pts], [p[1] for p in pts])
        per_c_slopes[c] = fit.slope
        if worst_slope is None or fit.slope > worst_slope:   # most POSITIVE = worst (rising flip)
            worst_slope, worst_c = fit.slope, c
    if worst_c is None:
        return CostateEstimate(
            name=f"per_class_dS_depoch[stage={stage}]", value=None, stderr=None,
            status=UNIDENTIFIABLE, units="S per epoch (per-class)",
            method="per-class within_flip slopes → dS/dep",
            evidence=(f"per-class rows present (n={len(pcs)}) but no single class has "
                      f"≥ {MIN_ROWS_FOR_SLOPE} within_flip points",),
            n=len(pcs))
    return CostateEstimate(
        name=f"per_class_dS_depoch[stage={stage}]", value=SEG_WEIGHT * worst_slope,
        stderr=None, status=MEASURED, units="S per epoch (worst-class within_flip)",
        method=(f"worst class = {worst_c} (within_flip slope {worst_slope:+.3e}/ep, "
                f"×λ_seg=100); per-class slopes " +
                ", ".join(f"{c}:{s:+.2e}" for c, s in sorted(per_c_slopes.items()))),
        evidence=(f"per-class within_flip over {len(pcs)} `handoff_readiness` rows [stage={stage}]",
                  f"worst-stalling class = {worst_c} (canonical order 0=Road 1=Lane "
                  "2=Undrivable 3=Movable 4=MyCar)"),
        n=len(pcs))
