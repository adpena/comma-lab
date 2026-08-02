"""ddm_lg1 (#808) — the CONSTRAIN-AND-PROTECT layer for burn-4.

Scorer-coordinate Lane guard. Three protection mechanisms, all expressed as
ADDITIVE per-pixel loss-weight addends folded into the tr1 trainer's EXISTING
``seg_pixel_w`` hook (train_tr1_partition_renderer_mlx.py L1517-1520):

  1. lambda_lane primal-dual constraint  — hold realized Lane error <= the ep641
     endpoint budget. Dual ascent at GATE cadence; the multiplier lambda_lane
     enters the loss as extra per-GT-Lane-pixel weight.
  2. born_lane protection mask            — up-weight the currently-WON Lane
     support (gt==Lane & realized==Lane) so bulk-class descent cannot erode it;
     weighted by the measured Lane head sensitivity.
  3. lane margin-floor emphasis           — extra weight on low-QA80-margin Lane
     pixels (the flip-prone tail), floor DERIVED from the Lane margin field p10.

DEFAULT-OFF and byte-identical when disabled: the trainer calls this ONLY under
``if cfg.lane_guard``; when off, nothing here runs, no state is touched, no RNG.
Consumes the a1 gate's EXISTING realized argmax (``realized_gate._realized_argmax``)
=> ZERO new scorer passes.

Authority: ``[macOS-CPU advisory]``; ``research_only=True``; ``score_claim=False``;
pointer ``0.1910828242 [contest-CPU]`` UNMOVED. Everything here is MEANS.

Scorer-coordinate provenance (constants-are-poison: every default carries its
value-provenance LADDER CLASS and its source.  Re-derived at source 2026-08-01 by
ddm_hl1; one entry below is class 4 / UNVERIFIED and says so — the previous blanket
claim that every default was artifact-DERIVED was itself the disguise this file's
own copied-table bug wore):

  * Budget ``LANE_BUDGET_S_UNITS = 0.12589`` (Lane per-class S at the ep641
    endpoint) — /Volumes/VertigoDataTier/pact/ddm_xp1_20260731/xp1_verdict.json
    ``base_per_class_S_units[1]``; ckpt stage_seg_trunk_tau_final.npz sha256
    40553db8be98215a67205d3670aa15d9b9edbe2322380ce169d8448af670f2db (ep641).
    d_seg units = S/100 = 0.0012589.
    LADDER CLASS 3 (measured_anchor), re-verified 2026-08-01: value and ckpt sha
    both match the artifact EXACTLY.  Caveat: the artifact is OUT OF REPO on an
    external volume, so no gate can check it and it is unverifiable on any other
    host — the citation is only as durable as that volume.
  * Lane head sensitivity ratio (derive_lane_head_sensitivity_ratio) — the four
    Lane-pair rank-4 head normals are the four LARGEST of all ten class pairs:
    Lane-Movable 4.007, Road-Lane 3.953, Lane-MyCar 3.862, Lane-Undrivable 3.748
    (mean 3.8925) vs all-ten-pair mean 3.2544 => 1.1961.  RESOLVED THROUGH the
    canonical producer ``tac.canonical_equations.segnet_head_rank4_flipdist_20260715``
    (equation ``segnet_head_rank4_linear_flipdist_v1``; head is EXACTLY rank-4
    linear, flip distance d = |margin_cc'| / ||w_c - w_c'||) — NOT copied as
    literals.  Memo: .omx/research/segnet_recursive_fractal_factorization_20260715.md §2.
  * Measured unprotected erosion ``EROSION_S_MEASURED = 0.00151`` (rung-1's
    unprotected continuation eroded the Lane pool by +0.00151 S while bulk classes
    descended) — xp1 (task #808 brief) — the scale that DERIVES eta_lambda.
    LADDER CLASS 4 (hardcoded), UNVERIFIED as of 2026-08-01: the value appears
    NOWHERE in xp1_verdict.json (all 23 keys checked); the citation is a prose
    brief, not a machine-readable artifact, and it carries no typed
    HardcodedWaiverCustody.  RE-DERIVATION TRIGGER: emit the rung-1 unprotected
    Lane erosion into a machine-readable verdict artifact and cite that, OR
    record typed waiver custody naming an owner.  Owner: lg1 successor.
    Consequence if wrong: it scales ``derive_eta_lambda`` only (dual step size),
    so an error changes how FAST the constraint engages, not the budget it holds.
  * Per-class Lane error definition matches the budget EXACTLY: ddm_qa92
    ``_per_class_flip_counts`` + P formula ``100 * flips / (n*384*512)``
    (experiments/ddm_qa92_carrier_discriminator.py L188, L353).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS

# ---- MEASURED scorer-coordinate constants (comma10k canonical order) -----------
LANE_CLASS = 1  # [Road, Lane, Undrivable, Movable, MyCar] — MEASURED (CLAUDE.md); NEVER luma-sort
N_CLASSES = 5
SEG_H, SEG_W = 384, 512  # frozen SegNet argmax plane (modules.py preprocess)

# ddm_xp1_20260731/xp1_verdict.json base_per_class_S_units[1] @ ep641 endpoint.
LANE_BUDGET_S_UNITS = 0.12589
LANE_BUDGET_DSEG = LANE_BUDGET_S_UNITS / 100.0  # 0.0012589

# Rank-4 head class-pair normals ||w_c - w_c'||, RESOLVED from the canonical producer
# (equation ``segnet_head_rank4_linear_flipdist_v1``) rather than copied as literals: a
# literal copied out of a measured artifact LOOKS wired, passes review, and cannot track
# its source — if the head is re-measured the copy goes stale in silence.  Descending
# order is fixed so the derived mean is bit-reproducible.
# The FOUR Lane pairs are the four LARGEST of all ten (the frozen net amplifies Lane).
_LANE_PAIR_NORMS = tuple(
    sorted((v for k, v in HEAD_PAIR_NORMS.items() if "Lane" in k.split("-")), reverse=True)
)
_ALL_PAIR_NORMS = tuple(sorted(HEAD_PAIR_NORMS.values(), reverse=True))

# Fail-closed shape canary: the frozen 5-class head has exactly C(5,2)=10 pairs, 4 of
# which touch Lane.  If the canonical producer ever changes shape (class renamed, class
# added), REFUSE at import rather than silently averaging the wrong set.
if len(_ALL_PAIR_NORMS) != 10 or len(_LANE_PAIR_NORMS) != 4:  # pragma: no cover - guard
    raise ImportError(
        "lane_guard: canonical HEAD_PAIR_NORMS shape changed "
        f"({len(_ALL_PAIR_NORMS)} pairs, {len(_LANE_PAIR_NORMS)} Lane pairs; expected 10/4) "
        "— re-derive the Lane head-sensitivity ratio before using the guard"
    )

# xp1-measured unprotected Lane erosion over the rung-1 continuation (S-units).
EROSION_S_MEASURED = 0.00151


def derive_lane_head_sensitivity_ratio() -> float:
    """Ratio of mean Lane-pair head-normal magnitude to the all-pair mean (MEASURED,
    fractal memo §2).  ~1.196: per unit penultimate perturbation, Lane boundaries move
    ~1.2x more than the average class pair — so Lane content is proportionally more
    scorer-sensitive and protection on it is proportionally more valuable."""
    return float(np.mean(_LANE_PAIR_NORMS) / np.mean(_ALL_PAIR_NORMS))


LANE_HEAD_SENSITIVITY_RATIO = derive_lane_head_sensitivity_ratio()  # 1.19607...


# ---- DERIVED dual hyperparameters (no bare constants) --------------------------
def derive_eta_lambda(
    lambda_target: float = 1.0,
    n_gates_to_engage: int = 10,
    erosion_s: float = EROSION_S_MEASURED,
) -> tuple[float, dict[str, Any]]:
    """Dual step size eta_lambda, DERIVED so a SUSTAINED violation of the measured
    erosion magnitude (``erosion_s`` S-units) drives lambda from 0 to ``lambda_target``
    over ``n_gates_to_engage`` gates:

        eta_lambda = lambda_target / (n_gates_to_engage * erosion_s)

    Rationale for each input (all MEASURED / principled, none tuned-to-fit):
      * lambda_target = 1.0 — one extra unit of per-Lane-pixel loss weight, the SAME
        natural scale as the sn1 ``class_weight_lane`` lever (1.0 = off; +1.0 doubles
        the marginal Lane pressure).
      * n_gates_to_engage = 10 — a DELIBERATELY slow engage so the dual reacts to a
        persistent drift, not single-gate descent noise (the a1 gate's own smooth
        channel varies gate-to-gate).
      * erosion_s = 0.00151 — the xp1-MEASURED unprotected Lane erosion; the typical
        sustained-violation magnitude the constraint must answer.

    At steady erosion (g ~ erosion_s) one gate's update is eta*g ~ lambda_step_cap
    (see derive_lambda_step_cap) => the caps-law-bounded step is self-consistent."""
    eta = float(lambda_target) / (float(n_gates_to_engage) * float(erosion_s))
    prov = {
        "formula": "lambda_target / (n_gates_to_engage * erosion_s)",
        "lambda_target": float(lambda_target),
        "n_gates_to_engage": int(n_gates_to_engage),
        "erosion_s": float(erosion_s),
        "erosion_s_source": "ddm_xp1_20260731 unprotected rung-1 Lane erosion",
        "value": eta,
    }
    return eta, prov


def derive_lambda_step_cap(
    lambda_target: float = 1.0, n_gates_to_engage: int = 10,
) -> float:
    """Per-gate dual-step ceiling (caps-law): |d lambda| <= lambda_target/n_gates so
    NO single gate can move the multiplier more than one engage-increment — one gate
    cannot thrash the primal."""
    return float(lambda_target) / float(n_gates_to_engage)


# lambda_max: bounded safety ceiling at 5x the natural unit (a sustained SEVERE
# violation can lift Lane weight to ~5, comparable to a strong class_weight_lane;
# above that the primal would be dominated by one class => refuse further ascent).
LAMBDA_MAX_DEFAULT = 5.0


def derive_margin_floor(
    lane_margins: np.ndarray, pct: float = 10.0,
) -> tuple[float, dict[str, Any]]:
    """Margin floor DERIVED as the ``pct``-th percentile of the QA80 margin field
    RESTRICTED to GT-Lane pixels (the flip-prone low-margin tail — 100% of flips are
    in the bottom GT-margin decile; sg1 §1.3).  Data-derived per run, never a bare
    constant.  Cross-check: the fractal memo §3 measured Road-Lane FEATURE flip-dist
    p10 = 0.024 (a different metric; this floor lives in the trainer's QA80 field)."""
    vals = np.asarray(lane_margins, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, {"note": "no lane-margin samples", "pct": float(pct), "value": 0.0}
    floor = float(np.percentile(vals, pct))
    return floor, {
        "formula": f"percentile_{pct}(QA80 margin | gt==Lane)",
        "pct": float(pct),
        "n_samples": int(vals.size),
        "value": floor,
        "cross_check_fractal_memo_feature_p10": 0.024,
    }


# ---- budget-matched realized Lane error (matches xp1/qa92 exactly) --------------
def per_class_lane_flip_S(
    realized: np.ndarray, gts: np.ndarray, lane_class: int = LANE_CLASS,
) -> float:
    """Realized Lane per-class error in S-units, using the EXACT ddm_qa92 definition
    so it is directly comparable to the ``LANE_BUDGET_S_UNITS`` budget:

        S_lane = 100 * sum_pairs #(gt==Lane & realized != gt) / (n_pairs * H * W)

    ``realized`` / ``gts`` are (n, H, W) int arrays (or stacked lists thereof)."""
    realized = np.asarray(realized)
    gts = np.asarray(gts)
    if realized.shape != gts.shape:
        raise ValueError(f"realized {realized.shape} != gts {gts.shape}")
    n = realized.shape[0]
    total_px = float(n * realized.shape[1] * realized.shape[2])
    flip = realized != gts
    lane_flips = int(((gts == lane_class) & flip).sum())
    return 100.0 * lane_flips / total_px


def born_lane_support_mask(
    realized: np.ndarray, gt: np.ndarray, lane_class: int = LANE_CLASS,
) -> np.ndarray:
    """(H,W) float32 mask of the currently-WON Lane support: pixels where the GT is
    Lane AND the realized argmax already agrees (gt==Lane & realized==Lane).  These
    are exactly the pixels holding the born (surviving) Lane components alive — the
    content the unprotected continuation ERODES first.  Protecting them (extra loss
    weight) preserves won structure without blanket-freezing all Lane.

    Scipy-free (a correctly-classified Lane pixel is by construction inside a
    surviving component); the per-component / per-cell Fisher-anchor refinement is
    the DEFERRED heavier variant (see the memo deferred table)."""
    realized = np.asarray(realized)
    gt = np.asarray(gt)
    return ((gt == lane_class) & (realized == lane_class)).astype(np.float32)


def per_component_min_flip_distance(
    margin: np.ndarray,
    born_mask: np.ndarray,
    dw_norm: float = max(_LANE_PAIR_NORMS),
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form per-born-component MIN flip distance in the head-hyperplane metric
    (piece-3 helper; the rank-4-head exact form d = |m| / ||dw_cc'|| — fractal memo §3,
    canonical equation ``segnet_head_rank4_linear_flipdist_v1``).

    ``margin``: (H,W) scorer margin field (winner-vs-runner |m|, the QA80/gt-cache field).
    ``born_mask``: (H,W) born-lane support (born_lane_support_mask output).
    ``dw_norm``: the head-normal magnitude; default = max of the four MEASURED Lane-pair
    normals (4.007, Lane-Movable) => the CONSERVATIVE (smallest-d) bound, since the
    binding pair for a component is whichever gives min d.

    Returns ``(labels, min_d)``: 8-conn component labels (0 = background) over the born
    mask + the per-component min flip distance array indexed by label-1.  A per-component
    hinge on min_d is the DEFERRED loss term (see the memo deferred table for the exact
    insertion point in pair_loss); this helper is the landed closed-form surface."""
    from scipy import ndimage

    born = np.asarray(born_mask) > 0.5
    m = np.abs(np.asarray(margin, dtype=np.float64))
    labels, n = ndimage.label(born, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labels, np.zeros((0,), dtype=np.float64)
    min_m = ndimage.minimum(m, labels=labels, index=np.arange(1, n + 1))
    return labels, np.asarray(min_m, dtype=np.float64) / float(dw_norm)


# ---- config + state ------------------------------------------------------------
@dataclass(frozen=True)
class LaneGuardConfig:
    """DERIVED defaults (constants-are-poison).  ``enabled=False`` => the trainer
    never invokes the guard => byte-identical."""

    enabled: bool = False
    budget_s: float = LANE_BUDGET_S_UNITS          # ep641 Lane S (xp1)
    eta_lambda: float = 0.0                          # 0.0 => derive at build (derive_eta_lambda)
    lambda_step_cap: float = 0.0                     # 0.0 => derive (derive_lambda_step_cap)
    lambda_max: float = LAMBDA_MAX_DEFAULT
    born_protect_weight: float = 0.0                 # 0.0 => born mask OFF; scaled by sensitivity
    margin_floor_weight: float = 0.0                 # 0.0 => margin-floor emphasis OFF
    margin_floor_pct: float = 10.0                   # percentile for derive_margin_floor
    lane_sensitivity_ratio: float = LANE_HEAD_SENSITIVITY_RATIO

    def resolved(self) -> LaneGuardConfig:
        """Fill any 0.0-sentinel derived field from its derivation (idempotent).
        Fail-closed on sign-inverting values (a negative eta would make the dual
        DESCEND on violation — the constraint silently becomes a reward)."""
        if (self.eta_lambda < 0.0 or self.lambda_step_cap < 0.0
                or self.lambda_max <= 0.0):
            raise ValueError(
                f"lane_guard config invalid: eta_lambda={self.eta_lambda}, "
                f"lambda_step_cap={self.lambda_step_cap}, lambda_max={self.lambda_max} "
                "(eta/cap must be >= 0, lambda_max > 0)")
        eta = self.eta_lambda or derive_eta_lambda()[0]
        cap = self.lambda_step_cap or derive_lambda_step_cap()
        return LaneGuardConfig(
            enabled=self.enabled, budget_s=self.budget_s, eta_lambda=eta,
            lambda_step_cap=cap, lambda_max=self.lambda_max,
            born_protect_weight=self.born_protect_weight,
            margin_floor_weight=self.margin_floor_weight,
            margin_floor_pct=self.margin_floor_pct,
            lane_sensitivity_ratio=self.lane_sensitivity_ratio,
        )


@dataclass
class LaneGuardState:
    """Mutable dual + protection state.  Updated ONLY at gate cadence."""

    lambda_lane: float = 0.0
    margin_floor: float | None = None
    born_masks: dict[int, np.ndarray] = field(default_factory=dict)
    n_gates: int = 0
    last_g_s: float = 0.0
    last_realized_lane_s: float = 0.0


def dual_ascent(state: LaneGuardState, cfg: LaneGuardConfig, realized_lane_s: float) -> float:
    """Bounded projected dual ascent at gate cadence:

        g       = realized_lane_s - budget_s              (constraint violation, S-units)
        step    = clip(eta * g, -cap, +cap)               (caps-law: one gate can't thrash)
        lambda <- clip(lambda + step, 0, lambda_max)       (KKT: multiplier >= 0)

    CAPS-LAW RECONCILIATION: the v9 rule "loss weights change at STAGE boundaries
    only" governs the PRIMAL loss weights.  lambda_lane is NOT a loss weight — it is a
    KKT dual multiplier whose update cadence is, by primal-dual optimization theory,
    the constraint-EVALUATION cadence (here the a1 GATE, the only point the realized
    constraint g is measured).  Within a gate interval lambda is CONSTANT, so the
    primal sees a slowly-varying, per-interval-fixed weight — no per-step thrash,
    consistent with the caps-law spirit.  ``lambda_step_cap`` + ``lambda_max`` bound
    it so a single noisy gate cannot dominate the primal."""
    g = float(realized_lane_s) - float(cfg.budget_s)
    step = max(-cfg.lambda_step_cap, min(cfg.lambda_step_cap, cfg.eta_lambda * g))
    new_lambda = max(0.0, min(cfg.lambda_max, state.lambda_lane + step))
    state.lambda_lane = float(new_lambda)
    state.last_g_s = float(g)
    state.last_realized_lane_s = float(realized_lane_s)
    return state.lambda_lane


def gate_update(
    state: LaneGuardState,
    cfg: LaneGuardConfig,
    realized_argmax: np.ndarray,
    gts: np.ndarray,
    gate_ids: tuple[int, ...] | list[int],
    lane_margins_by_id: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run once per a1 gate.  Reads the EXISTING realized argmax (zero new scorer
    passes), advances the dual, refreshes the born-lane masks for the gate pairs, and
    derives the margin floor on the first gate.  Returns a telemetry row.

    NOTE on geometry (honest): at n600 the gate set is the fd2 SUBSET (~36 pairs), so
    ``realized_lane_s`` here is an UNBIASED-if-representative ESTIMATE of the full-n600
    Lane level the budget was measured on; below n600 the gate set is ALL pairs and it
    is exact.  The dual reacts to a persistent drift regardless of the subset's
    absolute-level noise."""
    state.n_gates += 1
    realized_arr = np.asarray(realized_argmax)
    gts_arr = np.asarray(gts)
    realized_lane_s = per_class_lane_flip_S(realized_arr, gts_arr, LANE_CLASS)
    lam = dual_ascent(state, cfg, realized_lane_s)

    # born-lane support masks for THIS gate's pairs (used next epoch's pair_loss).
    if cfg.born_protect_weight > 0.0:
        for k, gid in enumerate(gate_ids):
            state.born_masks[int(gid)] = born_lane_support_mask(
                realized_arr[k], gts_arr[k], LANE_CLASS)

    # derive the margin floor once, from the first gate's Lane-restricted margins.
    if cfg.margin_floor_weight > 0.0 and state.margin_floor is None and lane_margins_by_id:
        lane_vals = []
        for k, gid in enumerate(gate_ids):
            m = lane_margins_by_id.get(int(gid))
            if m is not None:
                lane_vals.append(np.asarray(m)[gts_arr[k] == LANE_CLASS])
        if lane_vals:
            floor, _ = derive_margin_floor(np.concatenate(lane_vals), cfg.margin_floor_pct)
            state.margin_floor = floor

    return {
        "event": "lane_guard",
        "n_gates": state.n_gates,
        "realized_lane_s_units": float(realized_lane_s),
        "budget_s_units": float(cfg.budget_s),
        "g_s_units": float(state.last_g_s),
        "lambda_lane": float(lam),
        # KKT complementarity residual lambda*g (the #549 KKTDiagnostics-aligned surface:
        # ~0 at equilibrium — either the constraint is slack or the dual has settled).
        "complementarity": float(lam * state.last_g_s),
        "lambda_max": float(cfg.lambda_max),
        "eta_lambda": float(cfg.eta_lambda),
        "lambda_step_cap": float(cfg.lambda_step_cap),
        "margin_floor": (None if state.margin_floor is None else float(state.margin_floor)),
        "born_mask_pairs": (len(state.born_masks) if cfg.born_protect_weight > 0.0 else 0),
        "lane_sensitivity_ratio": float(cfg.lane_sensitivity_ratio),
        "score_neutral": False,
        "score_claim": False,
        "evidence_axis": "[macOS-CPU advisory]",
    }


def pixel_weight_addend(
    lstar: np.ndarray,
    lane_margin: np.ndarray | None,
    state: LaneGuardState,
    cfg: LaneGuardConfig,
    idx: int,
) -> np.ndarray | None:
    """Additive per-pixel loss-weight for ONE pair, folded into the trainer's
    ``seg_pixel_w`` (final weight = 1 + class_weight_lane_term + THIS).  Returns None
    if the total addend is identically zero (so ``seg_pixel_w`` stays None and the
    byte path is preserved).

        w += lambda_lane                              on GT-Lane pixels        (constraint)
        w += born_protect_weight * sensitivity        on born-lane support     (protection)
        w += margin_floor_weight  * relu(1 - m/floor) on low-margin GT-Lane    (margin floor)
    """
    lstar = np.asarray(lstar)
    is_lane = (lstar == LANE_CLASS).astype(np.float32)
    addend = np.zeros(lstar.shape, dtype=np.float32)
    active = False

    if state.lambda_lane > 0.0:
        addend = addend + state.lambda_lane * is_lane
        active = True

    if cfg.born_protect_weight > 0.0:
        bm = state.born_masks.get(int(idx))
        if bm is not None:
            addend = addend + (cfg.born_protect_weight * cfg.lane_sensitivity_ratio) * bm
            active = True

    if (cfg.margin_floor_weight > 0.0 and state.margin_floor is not None
            and state.margin_floor > 0.0 and lane_margin is not None):
        m = np.asarray(lane_margin, dtype=np.float32)
        deficit = np.maximum(0.0, 1.0 - m / float(state.margin_floor))
        addend = addend + cfg.margin_floor_weight * deficit * is_lane
        active = True

    if not active or not np.any(addend):
        return None
    return addend


__all__ = [
    "EROSION_S_MEASURED",
    "LAMBDA_MAX_DEFAULT",
    "LANE_BUDGET_DSEG",
    "LANE_BUDGET_S_UNITS",
    "LANE_CLASS",
    "LANE_HEAD_SENSITIVITY_RATIO",
    "N_CLASSES",
    "SEG_H",
    "SEG_W",
    "LaneGuardConfig",
    "LaneGuardState",
    "born_lane_support_mask",
    "derive_eta_lambda",
    "derive_lambda_step_cap",
    "derive_lane_head_sensitivity_ratio",
    "derive_margin_floor",
    "dual_ascent",
    "gate_update",
    "per_class_lane_flip_S",
    "per_component_min_flip_distance",
    "pixel_weight_addend",
]
