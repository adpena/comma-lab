# SPDX-License-Identifier: MIT
"""The speedup acceptance gate — the canonical BOTH-TERMS descent-equivalence gate.

Every future gradient-throughput speedup (a faster scorer backend: the custom
Metal grouped-conv backward, bf16/fp16 scorer fwd+bwd, ``mx.compile`` fusion,
channels-last, a NAX/fast-matmul kernel, ...) MUST pass THIS gate before it is
used to drive a real (n600) training run. The gate exists because of a concrete,
expensive NO-FAKE near-miss:

  The custom MLX grouped/depthwise-conv backward was validated at **n8 / 40
  epochs on d_seg ONLY** (gradient cosine ~1.0, descent-equivalent on d_seg,
  5.5x faster) and wired into an n600 basin run. It then **DIVERGED at n600 on
  the POSE axis** (d_pose 0.835 -> 6.94 -> 36.46) because its PoseNet gradient
  was wrong — a divergence the d_seg-only validation could never see. A 5.5x
  kernel that diverges is WORSE than useless: unwatched it manufactures a fake
  "the architecture cannot reach the basin" capacity verdict that is really a
  broken gradient (the exact "surrogate-optimized-but-not-exact-authority-
  verified" fake-implementation class). See
  ``.omx/research/mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md``.

The structural lessons this gate encodes (each is a hard, testable rule):

1. **BOTH terms, always.** A candidate passes ONLY IF its exact-d_seg AND its
   exact-d_pose trajectories both track the baseline within tolerance. A
   d_seg-only pass is structurally REFUSED — ``evaluate_descent_equivalence``
   raises :class:`DSegOnlyGateMisuse` if the caller supplies d_seg but no
   d_pose, and a candidate whose d_pose diverges is REJECTED even when its
   d_seg is perfect. The pose term blew up first and worst in the real incident,
   so it is a FIRST-CLASS gated quantity, not an afterthought.

2. **The REAL n, not n8.** n8 descent-equivalence does not generalize: a
   ~0.07%/step gradient error stays bounded over 40 n8 epochs but compounds over
   n600's ~75x more steps/epoch into divergence. The gate records the n it ran
   at and FLAGS (``generalization_warning``) any PASS obtained at n below
   ``min_trustworthy_n`` (default 600) — a small-n PASS is provisional, never a
   green light for the real run.

3. **Divergence, not just gap.** The incident was DIVERGENCE (a monotone blow-up
   toward random/exploding), not bounded oscillation. The gate detects an
   explicit divergence signature (a term climbing back toward — or past — its
   init while the baseline descends) separately from a small tracking gap, so a
   late-epoch blow-up cannot be averaged away by good early epochs.

This module is PURE and backend-agnostic: it consumes two already-measured
trajectories (the authority torch-CPU exact d_seg/d_pose of a BASELINE arm and a
CANDIDATE arm, the SAME metric on both — only the gradient that drove the steps
differs) and returns a typed verdict. The measurement is done by the caller
(``experiments/measure_descent_equivalence.py`` / the speedup-gate CLI); the
gate is the reusable ACCEPTANCE LOGIC so every speedup is judged by one
contract. A fast gradient-cosine PRE-CHECK on BOTH the SegNet-path and
PoseNet-path adapters (:func:`gradient_cosine_precheck_verdict`) is provided as a
cheap first filter — but it is explicitly NOT sufficient on its own (the custom
backward had per-layer cosine 1.0 and still diverged at n600), so a cosine PASS
only LICENSES the bounded-n600 A/B; it never replaces it.

Authority: the d_seg/d_pose fed in MUST be the torch-CPU exact authority for
BOTH arms (the candidate's GRADIENT is research-signal; its REPORTED metric is
recomputed on torch-CPU). This gate never reads a score off the fast backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MIN_TRUSTWORTHY_N",
    "DSegOnlyGateMisuse",
    "EpochMetric",
    "GateConfig",
    "GateVerdict",
    "GradientCosinePrecheck",
    "TermTrajectoryVerdict",
    "evaluate_descent_equivalence",
    "gradient_cosine_precheck_verdict",
]

# The n at/above which a PASS is trustworthy for a real basin run. The n600
# incident proved an n8 PASS is NOT trustworthy; the gate flags any PASS below
# this as provisional.
DEFAULT_MIN_TRUSTWORTHY_N = 600


class DSegOnlyGateMisuse(ValueError):
    """Raised when a caller tries to gate on d_seg with no d_pose trajectory.

    The n600 incident is exactly this misuse made structural: a speedup
    validated on d_seg alone passed and then diverged on the unmeasured pose
    axis. The gate refuses to render a verdict from a seg-only trajectory so the
    blind spot cannot be re-opened. Supply BOTH terms (recomputed on the
    torch-CPU authority) or do not call the gate.
    """


@dataclass(frozen=True)
class EpochMetric:
    """One epoch's EXACT (torch-CPU authority) score terms for one arm.

    ``d_pose`` is REQUIRED (may not be ``None``) — the whole point of the gate is
    that both terms are always present. Use a sentinel-free contract so a missing
    pose cannot silently pass as "0".
    """

    epoch: int
    d_seg: float
    d_pose: float

    def __post_init__(self) -> None:
        if self.d_pose is None:  # type: ignore[unreachable]
            raise DSegOnlyGateMisuse(
                f"EpochMetric at epoch {self.epoch} has d_pose=None; the "
                "acceptance gate requires BOTH d_seg AND d_pose on every epoch "
                "(the n600 pose-divergence lesson). Recompute d_pose on the "
                "torch-CPU authority and supply it."
            )


@dataclass(frozen=True)
class GateConfig:
    """Tolerances for the BOTH-terms descent-equivalence gate.

    The gate compares the candidate arm's per-epoch exact d_seg/d_pose against
    the baseline arm's at aligned epochs. A term PASSES when its final-epoch
    absolute gap is within an absolute floor OR a relative fraction of the
    BASELINE's own descent on that term (whichever is more permissive), AND no
    divergence signature fired on that term.

    Separate seg/pose tolerances because the two live at very different scales
    (d_seg ~ 1e-2 -> ~5e-4; d_pose ~ 1e-1 -> ~3e-5 near the frontier). The pose
    tolerances are the load-bearing ones — pose is where the incident blew up.
    """

    # d_seg: pass if |gap| <= max(seg_abs_tol, seg_rel_tol * |baseline_descent|).
    seg_abs_tol: float = 5.0e-3
    seg_rel_tol: float = 0.25
    # d_pose: tighter — pose drift is the documented failure axis.
    pose_abs_tol: float = 1.0e-3
    pose_rel_tol: float = 0.25
    # Divergence signature: a term is DIVERGING if, after this many aligned eval
    # points, the candidate's term has risen to >= divergence_init_fraction of
    # its OWN init value while the baseline's same-term has descended below
    # baseline_descend_fraction of ITS init. (Climbing back toward random while
    # the authority descends = the n600 signature.)
    divergence_min_points: int = 2
    divergence_init_fraction: float = 0.75
    baseline_descend_fraction: float = 0.75
    # A PASS at n below this is provisional (generalization_warning set True).
    min_trustworthy_n: int = DEFAULT_MIN_TRUSTWORTHY_N


@dataclass(frozen=True)
class TermTrajectoryVerdict:
    """Per-term (seg or pose) sub-verdict."""

    term: str  # "d_seg" or "d_pose"
    baseline_init: float
    baseline_final: float
    candidate_init: float
    candidate_final: float
    baseline_descent: float
    final_abs_gap: float
    final_rel_gap: float
    tracks_within_tol: bool
    diverged: bool
    diverged_at_epoch: int | None
    reason: str


@dataclass(frozen=True)
class GateVerdict:
    """The acceptance-gate verdict for one candidate vs baseline A/B."""

    passed: bool
    n_pairs: int
    epochs_compared: int
    seg: TermTrajectoryVerdict
    pose: TermTrajectoryVerdict
    generalization_warning: bool
    reasons: tuple[str, ...]
    config: GateConfig
    axis: str = "[macOS-CPU advisory]"  # the exact terms are torch-CPU authority


@dataclass(frozen=True)
class GradientCosinePrecheck:
    """A cheap first-filter cosine pre-check on BOTH adapter paths.

    The candidate gradient is compared (cosine) to the baseline gradient on a
    single batch, SEPARATELY for the SegNet-path pixel cotangent and the
    PoseNet-path pixel cotangent. This is the cheap filter that catches a grossly
    wrong gradient (e.g. the native strided-grouped VJP whose pose cosine was
    ~0.025). It is NECESSARY-NOT-SUFFICIENT: the custom backward had per-layer
    cosine 1.0 and STILL diverged at n600, so a cosine PASS only licenses the
    bounded-n600 A/B; it never replaces it.
    """

    seg_path_cosine: float
    pose_path_cosine: float
    min_cosine: float = 0.999

    @property
    def seg_ok(self) -> bool:
        return self.seg_path_cosine >= self.min_cosine

    @property
    def pose_ok(self) -> bool:
        return self.pose_path_cosine >= self.min_cosine

    @property
    def passed(self) -> bool:
        """Both paths must clear the bar. A seg-only cosine pass is NOT a pass."""
        return self.seg_ok and self.pose_ok


def gradient_cosine_precheck_verdict(
    seg_path_cosine: float,
    pose_path_cosine: float,
    *,
    min_cosine: float = 0.999,
) -> GradientCosinePrecheck:
    """Build the BOTH-paths cosine pre-check verdict.

    Both the SegNet-path AND PoseNet-path pixel-cotangent cosines must clear
    ``min_cosine``. A high seg cosine with a low pose cosine is REJECTED (the
    n600 signature: seg-correct, pose-wrong gradient). This is a FAST filter run
    before the (slow) bounded-n600 trajectory A/B; passing it does NOT admit the
    speedup — it only licenses running the full gate.
    """
    return GradientCosinePrecheck(
        seg_path_cosine=float(seg_path_cosine),
        pose_path_cosine=float(pose_path_cosine),
        min_cosine=float(min_cosine),
    )


def _coerce(records: Sequence[EpochMetric | dict]) -> list[EpochMetric]:
    out: list[EpochMetric] = []
    for r in records:
        if isinstance(r, EpochMetric):
            out.append(r)
            continue
        if not isinstance(r, dict):
            raise TypeError(f"trajectory entry must be EpochMetric or dict, got {type(r)}")
        if "epoch" not in r:
            raise ValueError(f"trajectory entry missing 'epoch': {r}")
        # Accept the measure_descent_equivalence.py field names (exact_d_seg /
        # mean_d_pose) AND the canonical (d_seg / d_pose).
        d_seg = r.get("d_seg", r.get("exact_d_seg"))
        d_pose = r.get("d_pose", r.get("mean_d_pose"))
        if d_seg is None:
            raise ValueError(f"trajectory entry missing d_seg/exact_d_seg: {r}")
        if d_pose is None:
            # Structural refusal of the d_seg-only gate.
            raise DSegOnlyGateMisuse(
                f"trajectory entry at epoch {r['epoch']} has no d_pose/mean_d_pose. "
                "The acceptance gate REFUSES a d_seg-only trajectory — that is the "
                "exact n600 blind spot (a speedup validated on d_seg alone diverged "
                "on the unmeasured pose axis). Recompute d_pose on the torch-CPU "
                "authority for every eval epoch and supply it."
            )
        out.append(EpochMetric(epoch=int(r["epoch"]), d_seg=float(d_seg), d_pose=float(d_pose)))
    if not out:
        raise ValueError("empty trajectory")
    return out


def _aligned(
    baseline: list[EpochMetric], candidate: list[EpochMetric]
) -> list[tuple[EpochMetric, EpochMetric]]:
    by_b = {m.epoch: m for m in baseline}
    by_c = {m.epoch: m for m in candidate}
    common = sorted(set(by_b) & set(by_c))
    if not common:
        raise ValueError(
            "baseline and candidate trajectories share no common epoch; the A/B "
            "must eval BOTH arms at the SAME epochs (same eval_every)."
        )
    return [(by_b[e], by_c[e]) for e in common]


def _term_verdict(
    term: str,
    pairs: list[tuple[EpochMetric, EpochMetric]],
    *,
    abs_tol: float,
    rel_tol: float,
    cfg: GateConfig,
) -> TermTrajectoryVerdict:
    getter = (lambda m: m.d_seg) if term == "d_seg" else (lambda m: m.d_pose)
    base_init = getter(pairs[0][0])
    cand_init = getter(pairs[0][1])
    base_final = getter(pairs[-1][0])
    cand_final = getter(pairs[-1][1])
    baseline_descent = base_init - base_final  # positive when the authority descended
    final_abs_gap = abs(cand_final - base_final)
    final_rel_gap = final_abs_gap / max(abs(baseline_descent), 1e-12)
    tol = max(abs_tol, rel_tol * abs(baseline_descent))
    tracks = final_abs_gap <= tol

    # Divergence signature: the candidate term DESCENDED and then CLIMBED BACK UP
    # (a blow-up), while the baseline keeps descending. The real n600 signature is
    # a RISE from a prior lower value — NOT merely "still near init early" (a
    # candidate that is just slower to descend must NOT be flagged). So we track
    # the candidate's running minimum and require BOTH (a) the current value has
    # risen well above that running minimum (it climbed back up), AND (b) the
    # current value is back near/above its own init, AND (c) the baseline has
    # descended. This distinguishes "diverging" from "slow-to-descend".
    diverged = False
    diverged_at: int | None = None
    cand_running_min = getter(pairs[0][1])
    for i, (bm, cm) in enumerate(pairs):
        b_val = getter(bm)
        c_val = getter(cm)
        if i + 1 >= cfg.divergence_min_points:
            climbed_back_up = c_val >= cfg.divergence_init_fraction * max(cand_init, 1e-12)
            # rose meaningfully above the candidate's own best-so-far (a blow-up,
            # not a monotone slow descent that never dipped).
            rose_from_min = c_val > cand_running_min * (1.0 + cfg.divergence_init_fraction)
            baseline_is_descending = (
                b_val <= cfg.baseline_descend_fraction * max(base_init, 1e-12)
            )
            if baseline_is_descending and climbed_back_up and rose_from_min:
                diverged = True
                diverged_at = bm.epoch
                break
        cand_running_min = min(cand_running_min, c_val)

    if diverged:
        reason = (
            f"{term} DIVERGED at epoch {diverged_at}: candidate climbed back to "
            f">= {cfg.divergence_init_fraction:.0%} of its init while the baseline "
            f"descended below {cfg.baseline_descend_fraction:.0%} of its init "
            "(the n600 divergence signature)."
        )
    elif tracks:
        reason = (
            f"{term} tracks: final |gap|={final_abs_gap:.3e} <= tol={tol:.3e} "
            f"(baseline descent {baseline_descent:.3e})."
        )
    else:
        reason = (
            f"{term} GAP too large: final |gap|={final_abs_gap:.3e} > tol={tol:.3e} "
            f"(baseline descent {baseline_descent:.3e})."
        )

    return TermTrajectoryVerdict(
        term=term,
        baseline_init=base_init,
        baseline_final=base_final,
        candidate_init=cand_init,
        candidate_final=cand_final,
        baseline_descent=baseline_descent,
        final_abs_gap=final_abs_gap,
        final_rel_gap=final_rel_gap,
        tracks_within_tol=tracks and not diverged,
        diverged=diverged,
        diverged_at_epoch=diverged_at,
        reason=reason,
    )


def evaluate_descent_equivalence(
    baseline_trajectory: Sequence[EpochMetric | dict],
    candidate_trajectory: Sequence[EpochMetric | dict],
    *,
    n_pairs: int,
    config: GateConfig | None = None,
) -> GateVerdict:
    """The canonical BOTH-TERMS acceptance gate.

    Args:
        baseline_trajectory: per-epoch EXACT (torch-CPU authority) ``d_seg`` AND
            ``d_pose`` of the BASELINE (trusted-gradient) arm. Either
            :class:`EpochMetric` records or dicts with keys
            ``{epoch, d_seg|exact_d_seg, d_pose|mean_d_pose}``.
        candidate_trajectory: same shape, for the CANDIDATE (faster-gradient)
            arm. The exact terms MUST be recomputed on the torch-CPU authority
            (never read off the fast backend).
        n_pairs: the n the A/B ran at. A PASS below
            ``config.min_trustworthy_n`` sets ``generalization_warning=True``.
        config: tolerances (default :class:`GateConfig`).

    Returns:
        :class:`GateVerdict` — ``passed`` is True ONLY IF BOTH the d_seg AND the
        d_pose sub-verdicts track within tolerance AND neither diverged.

    Raises:
        DSegOnlyGateMisuse: if either trajectory lacks a d_pose on any epoch (the
            structural refusal of the n600 blind spot).
        ValueError: empty / non-overlapping trajectories.
    """
    cfg = config or GateConfig()
    base = _coerce(baseline_trajectory)
    cand = _coerce(candidate_trajectory)
    pairs = _aligned(base, cand)

    seg = _term_verdict(
        "d_seg", pairs, abs_tol=cfg.seg_abs_tol, rel_tol=cfg.seg_rel_tol, cfg=cfg
    )
    pose = _term_verdict(
        "d_pose", pairs, abs_tol=cfg.pose_abs_tol, rel_tol=cfg.pose_rel_tol, cfg=cfg
    )

    seg_pass = seg.tracks_within_tol and not seg.diverged
    pose_pass = pose.tracks_within_tol and not pose.diverged
    passed = bool(seg_pass and pose_pass)

    reasons: list[str] = []
    if passed:
        reasons.append(
            "PASS: BOTH d_seg AND d_pose track the authority within tolerance "
            "with no divergence."
        )
    else:
        if not seg_pass:
            reasons.append(f"REJECT (seg): {seg.reason}")
        if not pose_pass:
            reasons.append(f"REJECT (pose): {pose.reason}")
        # Make the BOTH-terms lesson explicit when seg passes but pose fails.
        if seg_pass and not pose_pass:
            reasons.append(
                "NOTE: d_seg passed but d_pose did NOT — this is the EXACT n600 "
                "failure class (a speedup that is seg-correct but pose-wrong). A "
                "d_seg-only gate would have wrongly admitted this candidate."
            )

    generalization_warning = bool(passed and n_pairs < cfg.min_trustworthy_n)
    if generalization_warning:
        reasons.append(
            f"PROVISIONAL: passed at n={n_pairs} < min_trustworthy_n="
            f"{cfg.min_trustworthy_n}. An n8/small-n PASS does NOT generalize "
            "to n600 (the per-step error compounds over ~75x more steps/epoch). "
            "Re-run the gate at the REAL n before using this speedup for a basin "
            "run."
        )

    return GateVerdict(
        passed=passed,
        n_pairs=int(n_pairs),
        epochs_compared=len(pairs),
        seg=seg,
        pose=pose,
        generalization_warning=generalization_warning,
        reasons=tuple(reasons),
        config=cfg,
    )
