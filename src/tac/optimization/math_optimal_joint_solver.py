# SPDX-License-Identifier: MIT
"""tac.optimization.math_optimal_joint_solver — the math-optimal joint decoder solver.

The operator asked for "solving and math optimal everywhere": ingest ALL measured
response surfaces and emit the math-optimal joint decoder config

    (C, T, Q, E) = (capacity base_ch, taper, weight-bits, training-epochs)

that minimises the exact contest score

    S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/N

subject to the joint constraints, AND the MEASURED achievable-S lower bound (the real
T_floor over the measured surface, replacing the loose analytic 0.118).

This is a DESK MODEL over MEASURED surfaces, NOT a measurement. Every number it emits
is ``[advisory]`` / ``[macOS-CPU advisory]`` and NON-PROMOTABLE (``score_claim=false``,
``promotion_eligible=false``). The exact pointer stays pointer-only (0.19110). The
OUTPUT is a config recommendation that feeds the next training run — it tells us what
the clean anchor + its prune-path should become.

REUSE, not rebuild (CLAUDE.md "search-and-familiarize"):
  * :mod:`tac.contest_score` — ALL score arithmetic (the compliance bedrock; never
    hand-rolled). ``compute_contest_score`` / ``break_even_d_seg`` / ``rate_term`` /
    ``pose_term`` / ``seg_term``.
  * :mod:`tac.capacity_rd_qat` — the C×Q (capacity × weight-bits) 1D solve already in
    the repo: measured anchors (bc20 small-basis basin, the pr110 frontier), the EXACT
    ``decoder_param_count`` byte model, the MEASURED int8→int-N byte-shrink ratios. This
    module EXTENDS it to the full (C, T, Q, E) joint with the convergence (E) and taper
    (T) axes + the new d_seg-384 achievability-floor surface + the KKT/water-fill solve.

The four axes and their MEASURED grounding:
  * C (capacity / base_ch): bytes(C) exact from ``decoder_param_count``; d_seg(C) a
    power law fit on the bc20+frontier anchors, BOUNDED BELOW by the 384 floor.
  * T (taper / boundary-band reallocation): a byte-NEUTRAL d_seg multiplier (the RD win
    that does not touch the rate term). Conservative default 1.0 (no measured A/B yet);
    a measured taper-d_seg ratio plugs straight in.
  * Q (weight-bits): the MEASURED int8→int-N byte fraction (capacity_rd_qat) + a
    d_seg-hold spill (the QAT distortion-hold contract — an ASSUMPTION until measured).
  * E (training epochs): a d_seg convergence curve d_seg(E) toward the capacity/floor
    asymptote, fit to the deepmath memo's measured (epochs, d_seg) anchors. Gives the
    training-time→score Pareto + the min-budget config per S threshold.

THE EXISTENCE-PROOF CROSS-CHECK (CLAUDE.md feedback_terminal_conclusion 2026-06-23):
every "floor / achievable lower bound / unreachable" the solver emits is cross-checked
against the known measured artifacts (PR95 d_seg 5.6e-4; the 384 d_seg floor 0.0187 S;
the 0.191 frontier; the bc20 basin). A floor claim is downgraded to
"our-current-config-limited" the instant a known artifact beats it. See
:func:`existence_proof_crosscheck`.

Cross-refs (provenance for every ingested surface):
  * ``.omx/research/dseg_384_achievability_floor_verdict_20260623.md`` (+ the n600 JSON):
    FLOOR-384 d_seg = 1.875e-4 (S 0.0187) — the absolute d_seg floor for a 384-output
    decoder. CAPACITY-LIMITED verdict (the pipeline does NOT floor d_seg).
  * ``.omx/research/dseg_reducibility_gt_margin_verdict_20260623.md``: our decoder's
    residual flips are at low GT margin (IRREDUCIBLE for OUR flip set; ΔS ceiling 0.012).
  * ``.omx/research/pr95_vs_ours_convergence_gap_and_capacity_rd_deepmath_20260623.md``:
    the capacity power law α∈[0.9,1.5]; the capacity-RD optimum S(p); the DECISIVE
    entropy measurement (rate axis needs a retrain, not a recode); the (epochs, d_seg)
    convergence anchors.
  * ``reports/fp_shrink_ptq_bc20_n600.json``: the measured int8→int-N byte ratios.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from tac import capacity_rd_qat as crq
from tac.contest_score import (
    break_even_d_seg,
    compute_contest_score,
    rate_term,
    seg_term,
)
from tac.torch_vehicle.configurable_taper_decoder import (
    decoder_param_count,
    vendored_taper,
)

__all__ = [
    "DSEG_384_FLOOR",
    "DSEG_384_FLOOR_S_UNITS",
    "FRONTIER_S",
    "ExistenceProof",
    "IngestedSurfaces",
    "JointConfig",
    "JointSolveResult",
    "PruneCapacityStep",
    "PruneRdPathPlan",
    "TrainingParetoPoint",
    "dseg_capacity_floor_bounded",
    "dseg_convergence",
    "existence_proof_crosscheck",
    "load_ingested_surfaces",
    "min_training_budget_for_threshold",
    "physical_achievable_floor",
    "plan_capacity_rd_prune_path",
    "solve_math_optimal_joint",
    "training_time_pareto",
]

# ---------------------------------------------------------------------------
# CANONICAL CONSTANTS — every one cites a measured artifact.
# ---------------------------------------------------------------------------

#: FLOOR-384 d_seg — the absolute achievability floor for ANY 384-output decoder
#: (a PERFECT bilinear-downsampled GT reconstruction through the exact eval round-trip).
#: MEASURED N=600. Cite: dseg_384_achievability_floor_n600_20260623.json
#: (floors.floor_384.d_seg). This is the d_seg ASYMPTOTE that bounds the capacity power
#: law from below — no 384-native decoder can beat it; the 384 bottleneck (0.016 S) +
#: uint8 (0.003 S) are the only pipeline contributions, both far below sub-0.15.
DSEG_384_FLOOR: float = 1.875e-4
DSEG_384_FLOOR_S_UNITS: float = 100.0 * DSEG_384_FLOOR  # 0.01875

#: The resolution-bottleneck-only floor (uint8 round skipped). A higher-resolution
#: (>=camera-res) decoder could in principle dip below DSEG_384_FLOOR toward this, but
#: only by 0.003 S — the 384 vs higher-res lever is tiny. Cite: floor_384_float.
DSEG_384_FLOAT_FLOOR: float = 1.596e-4

#: The exact contest frontier pointer (the existence-proof anchor #1). Cite:
#: .omx/state/canonical_frontier_pointer.json (S=0.19110, [contest-CPU]).
FRONTIER_S: float = 0.19109982419209975

#: PR95 author's reported own-trained d_seg (existence-proof anchor #2). Cite:
#: pr95_vs_ours_convergence_gap_and_capacity_rd_deepmath_20260623.md PART A
#: (SegNet 0.00061212; our recode 0.00055978). This is BELOW any "our decoder is
#: d_seg-walled" claim — a known artifact reaches 5.6e-4.
PR95_DSEG: float = 0.00055978

#: bc20 small-basis basin d_seg (existence-proof anchor #3; our own clean measured).
#: Cite: capacity_rd_qat.ANCHOR_BC20 (reports/fp_shrink_ptq_bc20_n600.json fp32 row).
BC20_BASIN_DSEG: float = crq.ANCHOR_BC20.d_seg  # 0.0026

# --- POSE axis (d_pose) — convergence-dependent, DECOUPLED from capacity. ---
# Pose is NOT a function of decoder capacity (deepmath: "pose decoupled from capacity");
# it is a function of TRAINING CONVERGENCE. The bc20 BASIN value (3.42e-4) is the
# UNDER-TRAINED, stage-1-only pose; a fully-trained decoder reaches the frontier's pose
# (2.93e-5, the converged basin). The pose term sqrt(10·d_pose) is 0.0585 at the basin
# vs 0.0171 at the frontier — a 0.041 S swing that DOMINATES the sub-0.15 question, so
# modelling pose as convergence-dependent (not a fixed basin constant) is essential.
#: Under-trained (stage-1) pose. Cite: capacity_rd_qat.ANCHOR_BC20.d_pose.
DPOSE_BASIN: float = crq.ANCHOR_BC20.d_pose  # 3.42e-4
#: Converged pose asymptote = the frontier's measured d_pose. Cite:
#: capacity_rd_qat.ANCHOR_FRONTIER.d_pose (backed out from the 0.191 pointer).
DPOSE_CONVERGED: float = crq.ANCHOR_FRONTIER.d_pose  # 2.93e-5
#: Pose convergence time-constant (epochs). The two pose anchors (basin@ep2325,
#: frontier@converged) are too sparse for a clean tau; we model pose as converging on
#: roughly the SAME timescale as d_seg's deep stages (the curriculum's late stages move
#: both). Conservative tau so pose is at the basin at small E and at the frontier by
#: full convergence. MODELLED — flagged. A measured (epochs, d_pose) curve plugs in.
DPOSE_TAU_E: float = 6000.0


# ---------------------------------------------------------------------------
# E-AXIS — d_seg convergence curve d_seg(E) toward the capacity/floor asymptote.
# ---------------------------------------------------------------------------

# MEASURED (epochs, d_seg) anchors at the FIXED-recipe bc20, from the deepmath memo
# PART B.1 clean-anchor table (matched recipe muon_lr 0.03 / clip 50). All are STAGE-1
# CE-only mid-descent reads (the curriculum's d_seg FINISHER stages 2/3/5/8 are the
# un-measured share — so this convergence curve is a CONSERVATIVE upper bound on d_seg:
# the real converged d_seg is at or below the asymptote, never above the curve).
#
#   bc20_p48  120 CE epochs (48 pairs)      -> 0.0037602   [MEASURED, deepmath B.1]
#   bc20_n600 ~2325 CE epochs (600 pairs, paused stage1/8) -> 0.0025607  [MEASURED]
#
# The bc24 matched-recipe anchor (113K params, 120 CE) -> 0.0028546 grounds the
# CAPACITY axis (see dseg_capacity_power_law). The E-axis convergence model below is a
# per-capacity exponential approach to the capacity asymptote d_seg_inf(C):
#
#   d_seg(E; C) = d_seg_inf(C) + (d_seg_0 - d_seg_inf(C)) * exp(-E / tau_E)
#
# where d_seg_0 is the init d_seg (~0.507, the measured curriculum init), tau_E is the
# convergence time-constant fit to the two bc20 epoch anchors, and d_seg_inf(C) is the
# capacity asymptote (the power law, BOUNDED BELOW by the 384 floor). This is a MODELLED
# curve flagged as such; its single job is the training-time Pareto + min-budget query.

#: Measured curriculum init d_seg (the BUGGY-vs-FIXED A/B both start here). Cite:
#: deepmath B.1 (init 0.50727).
DSEG_INIT: float = 0.50727

# Two measured bc20 (epochs, d_seg) anchors to fit tau_E. These are STAGE-1 CE-only.
_E_ANCHOR_LOW = (120.0, 0.0037602)  # bc20_p48 120 CE epochs
_E_ANCHOR_HIGH = (2325.0, 0.0025607)  # bc20_n600 ~ep2325 stage-1


def _fit_tau_E(d_seg_inf: float) -> float:
    """Fit the convergence time-constant tau_E to the two measured bc20 epoch anchors,
    given the capacity asymptote ``d_seg_inf``. Solves the exponential-approach model at
    the two anchors and returns the geometric-mean tau (robust to the 2-point fit).

    d_seg(E) - d_seg_inf = (DSEG_INIT - d_seg_inf) * exp(-E / tau)
    => tau = -E / ln((d_seg(E) - d_seg_inf) / (DSEG_INIT - d_seg_inf))
    """
    taus = []
    for E, d in (_E_ANCHOR_LOW, _E_ANCHOR_HIGH):
        num = d - d_seg_inf
        den = DSEG_INIT - d_seg_inf
        if num <= 0 or den <= 0:
            # asymptote already at/below the measured point at that epoch — the anchor is
            # below d_seg_inf (can happen if d_seg_inf is set too high); skip it.
            continue
        ratio = num / den
        if ratio <= 0 or ratio >= 1:
            continue
        taus.append(-E / math.log(ratio))
    if not taus:
        # Degenerate: fall back to the single high anchor's implied tau at a tiny epsilon.
        return _E_ANCHOR_HIGH[0]
    # geometric mean of the per-anchor taus.
    return math.exp(sum(math.log(t) for t in taus) / len(taus))


def dseg_convergence(
    epochs: float,
    d_seg_inf: float,
    *,
    d_seg_init: float = DSEG_INIT,
    tau_E: float | None = None,
) -> float:
    """d_seg after ``epochs`` of training, approaching the asymptote ``d_seg_inf``.

    Exponential approach fit to the two MEASURED bc20 epoch anchors. ``epochs`` is the
    EFFECTIVE training budget (curriculum-equivalent epochs). Returns a d_seg that is
    >= ``d_seg_inf`` for all finite epochs and -> ``d_seg_inf`` as epochs -> inf.

    MODELLED (flagged): a 2-point exponential fit on stage-1 CE-only anchors. The real
    curriculum's d_seg FINISHER stages drive d_seg toward the asymptote FASTER than this
    CE-only curve, so this is a CONSERVATIVE (high) d_seg estimate at any finite E.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be >= 0, got {epochs}")
    if tau_E is None:
        tau_E = _fit_tau_E(d_seg_inf)
    gap = max(0.0, d_seg_init - d_seg_inf)
    return d_seg_inf + gap * math.exp(-epochs / tau_E)


def dpose_convergence(
    epochs: float,
    *,
    d_pose_init: float = DPOSE_BASIN,
    d_pose_inf: float = DPOSE_CONVERGED,
    tau_E: float = DPOSE_TAU_E,
) -> float:
    """d_pose after ``epochs`` of training, approaching the converged frontier pose.

    Pose is DECOUPLED from capacity (deepmath) — it is a convergence quantity. Exponential
    approach from the under-trained basin pose to the frontier's converged pose. MODELLED
    (sparse 2-anchor fit); a measured (epochs, d_pose) curve from the E-axis sister agent
    plugs straight in. This term is the DOMINANT sub-0.15 swing (0.041 S basin->frontier),
    so the convergence model matters more here than anywhere.
    """
    if epochs < 0:
        raise ValueError(f"epochs must be >= 0, got {epochs}")
    gap = max(0.0, d_pose_init - d_pose_inf)
    return d_pose_inf + gap * math.exp(-epochs / tau_E)


# ---------------------------------------------------------------------------
# C-AXIS — d_seg(C) power law BOUNDED BELOW by the 384 floor.
# ---------------------------------------------------------------------------


def dseg_capacity_power_law(base_ch: int) -> tuple[float, str]:
    """The CAPACITY asymptote d_seg_inf(C) — the d_seg a FULLY-converged decoder of
    capacity ``base_ch`` reaches, as a power law in decoder params, BOUNDED BELOW by the
    384 achievability floor.

    Reuses ``capacity_rd_qat.dseg_at_capacity`` (the two-point power law on the
    bc20+frontier endpoints) but CLAMPS the result at ``DSEG_384_FLOOR`` — no 384-output
    decoder can beat the perfect-reconstruction floor (the existence-proof lower bound).
    Returns (d_seg_inf, evidence_tag).
    """
    d_raw, ev = crq.dseg_at_capacity(base_ch)
    d_clamped = max(d_raw, DSEG_384_FLOOR)
    if d_clamped > d_raw:
        ev = ev + f" [CLAMPED at 384 floor {DSEG_384_FLOOR:.4g}]"
    return d_clamped, ev


def dseg_capacity_floor_bounded(base_ch: int) -> float:
    """Convenience: the 384-floor-bounded capacity asymptote d_seg_inf(C)."""
    return dseg_capacity_power_law(base_ch)[0]


# ---------------------------------------------------------------------------
# T-AXIS — taper d_seg multiplier (byte-NEUTRAL boundary-band reallocation).
# ---------------------------------------------------------------------------

# The vendored taper puts ~69% of params in low-res stages SegNet's stride-2 stem
# discards; only ~7.76K params sit at the high-res boundary band where flips live
# (deepmath E.3 / RECURSIVE_REVIEW #3). Reallocating to the boundary band is
# byte-NEUTRAL (+0.05%) and could bend d_seg. NO measured A/B yet -> default multiplier
# 1.0 (no effect). A measured taper-d_seg ratio (arm_b/control) plugs straight in as
# ``taper_dseg_multiplier`` < 1.0.


# ---------------------------------------------------------------------------
# Joint config + the solve.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointConfig:
    """A point in the (C, T, Q, E) joint config space + its modelled score components."""

    base_ch: int  # C
    taper_dseg_multiplier: float  # T (1.0 = vendored taper, <1 = boundary-band realloc)
    qat_nbits: int  # Q
    qat_frac_low_precision: float  # Q (fraction of weights at low precision)
    epochs: float  # E (effective curriculum-equivalent training epochs)

    decoder_params: int
    d_seg: float  # modelled d_seg(C,T,Q,E)
    d_seg_inf: float  # the C-axis asymptote (fully converged, before T/Q)
    d_seg_evidence: str
    d_pose: float
    archive_bytes: int
    S: float

    def with_score(self) -> JointConfig:  # pragma: no cover - trivial
        return self


@dataclass(frozen=True)
class ExistenceProof:
    """The result of the existence-proof cross-check on a floor/achievable-S claim."""

    claimed_floor_S: float
    best_known_artifact: str
    best_known_S: float
    is_valid_floor: bool  # True if NO known artifact beats the claim
    verdict: str  # human-readable


@dataclass(frozen=True)
class TrainingParetoPoint:
    epochs: float
    d_seg: float
    S: float


@dataclass(frozen=True)
class PruneCapacityStep:
    """One capacity rung on the prune-path: prune the converged big decoder DOWN to this
    capacity, KD-finetune, byte-close, and exact-score. The predicted (d_seg, bytes, S)
    are the solver's MODEL prediction; the measured columns are filled by the runner."""

    target_base_ch: int
    target_decoder_params: int
    target_taper: tuple[int, ...]
    predicted_d_seg_inf: float  # solver's power-law prediction (CLAMPED at 384 floor)
    predicted_native_bytes: int
    predicted_native_S: float
    # filled by the runner when the checkpoint converges + the prune-path is executed:
    measured_d_seg: float | None = None
    measured_bytes: int | None = None
    measured_S: float | None = None


@dataclass(frozen=True)
class PruneRdPathPlan:
    """The ready-to-run capacity-RD prune-path: train ONE big decoder to convergence, then
    structured-prune/KD-distill DOWN to each capacity rung and byte-close + exact-score.
    This pins the capacity power-law exponent α (the dominant gating uncertainty) with
    APPLES-TO-APPLES points (same training, same recipe — only capacity varies), replacing
    the contaminated 2-point fit. READY when the clean big-decoder anchor converges."""

    source_checkpoint_glob: str  # where the converged big decoder is expected
    source_base_ch: int  # the big decoder's capacity (prune FROM here)
    steps: tuple[PruneCapacityStep, ...]
    method: str  # structured-prune + KD-finetune contract
    runner_contract: tuple[str, ...]  # the per-step measurement steps the runner executes
    gated_on: str  # what must land before this can run
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestedSurfaces:
    """The measured response surfaces ingested by the solver, with provenance. Sister
    Q-axis / E-axis agent surfaces are read-when-available; until then the defaults are
    the deepmath-memo measured anchors (flagged)."""

    dseg_384_floor: float
    dseg_384_floor_provenance: str
    qaxis_byte_shrink: dict[int, float]
    qaxis_provenance: str
    eaxis_anchors: tuple[tuple[float, float], ...]
    eaxis_provenance: str
    capacity_anchors: tuple[tuple[int, float], ...]
    capacity_provenance: str
    frontier_S: float
    pose_hold: float
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JointSolveResult:
    optimum: JointConfig
    achievable_S_lower_bound: float  # the surface-MODEL lower bound (power-law converged)
    achievable_S_lower_bound_config: JointConfig
    achievable_S_existence_proof: ExistenceProof
    # The PHYSICAL achievable floor = the existence-proof floor: a perfect decoder
    # (d_seg at the 384 achievability floor + converged pose) at the cheapest viable byte
    # budget. This is the TRUE T_floor (replacing the loose analytic 0.118). The gap
    # between the surface-model lower bound and this physical floor is the
    # capacity-realization question (can training reach the 384 floor at small bytes?).
    physical_floor_S: float
    physical_floor_config: JointConfig
    frontier_S: float
    training_pareto: list[TrainingParetoPoint]
    grid: list[JointConfig]
    gating: dict[str, str]  # what's gated + the value that flips the optimum
    surfaces: IngestedSurfaces
    notes: list[str] = field(default_factory=list)


def load_ingested_surfaces(
    *,
    research_dir: str | Path = ".omx/research",
    pose_hold: float = crq.ANCHOR_BC20.d_pose,
    qaxis_surface_path: str | Path | None = None,
    eaxis_surface_path: str | Path | None = None,
) -> IngestedSurfaces:
    """Ingest the measured response surfaces. Sister Q-axis / E-axis agent JSONs are read
    when present (their canonical paths are
    ``.omx/research/qaxis_bitdepth_response_surface_20260623.json`` and
    ``.omx/research/eaxis_training_time_optimization_surface_20260623.json``); until they
    land, the defaults are the deepmath-memo measured anchors.

    The d_seg-384 floor is read from the n600 JSON if present (authoritative), else the
    pinned constant. This is the only surface that is fully landed today.
    """
    research = Path(research_dir)
    notes: list[str] = []

    # --- d_seg 384 floor (LANDED today) ---
    floor = DSEG_384_FLOOR
    floor_prov = "pinned DSEG_384_FLOOR constant (1.875e-4)"
    floor_json = research / "dseg_384_achievability_floor_n600_20260623.json"
    if floor_json.is_file():
        try:
            d = json.loads(floor_json.read_text())
            floor = float(d["floors"]["floor_384"]["d_seg"])
            floor_prov = f"{floor_json} floors.floor_384.d_seg (N={d.get('n_pairs_scored', '?')})"
        except (KeyError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
            notes.append(f"floor JSON parse failed ({exc}); using pinned constant")

    # --- Q-axis byte shrink (sister agent or deepmath measured) ---
    qaxis = {n: crq.qat_byte_fraction(n) for n in (8, 7, 6, 5, 4)}
    qaxis_prov = "capacity_rd_qat.MEASURED_BYTE_SHRINK_BC20 (reports/fp_shrink_ptq_bc20_n600.json)"
    q_path = (
        Path(qaxis_surface_path)
        if qaxis_surface_path is not None
        else research / "qaxis_bitdepth_response_surface_20260623.json"
    )
    if q_path.is_file():
        try:
            d = json.loads(q_path.read_text())
            # interface contract: {"byte_fraction": {"8": 1.0, "4": 0.52, ...}}
            bf = d.get("byte_fraction") or d.get("qaxis_byte_fraction")
            if bf:
                qaxis = {int(k): float(v) for k, v in bf.items()}
                qaxis_prov = f"{q_path} byte_fraction (sister Q-axis agent)"
                notes.append("Q-axis surface ingested from sister agent.")
        except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
            notes.append(f"Q-axis JSON parse failed ({exc}); using deepmath measured")
    else:
        notes.append(
            "Q-axis sister surface not yet published; using capacity_rd_qat measured "
            "int8->int-N byte ratios (bc20 post-int8-brotli)."
        )

    # --- E-axis convergence anchors (sister agent or deepmath measured) ---
    eaxis = (_E_ANCHOR_LOW, _E_ANCHOR_HIGH)
    eaxis_prov = "deepmath B.1 measured bc20 (epochs,d_seg): (120,0.00376),(2325,0.00256)"
    e_path = (
        Path(eaxis_surface_path)
        if eaxis_surface_path is not None
        else research / "eaxis_training_time_optimization_surface_20260623.json"
    )
    if e_path.is_file():
        try:
            d = json.loads(e_path.read_text())
            # interface contract: {"epoch_dseg_anchors": [[epochs, d_seg], ...]}
            anchors = d.get("epoch_dseg_anchors")
            if anchors:
                eaxis = tuple((float(e), float(s)) for e, s in anchors)
                eaxis_prov = f"{e_path} epoch_dseg_anchors (sister E-axis agent)"
                notes.append("E-axis surface ingested from sister agent.")
        except (ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
            notes.append(f"E-axis JSON parse failed ({exc}); using deepmath measured")
    else:
        notes.append(
            "E-axis sister surface not yet published; using deepmath measured bc20 "
            "(epochs,d_seg) anchors. Convergence curve is a conservative (high) d_seg."
        )

    # --- Capacity anchors (clean matched-recipe pair) ---
    cap_anchors = ((20, BC20_BASIN_DSEG), (36, crq.ANCHOR_FRONTIER.d_seg))
    cap_prov = (
        "capacity_rd_qat anchors: bc20 basin 0.0026 + frontier d_seg 0.00056 (bc36-class). "
        "Clean matched-recipe pair bc20_p48 0.00376 / bc24_p48 0.00285 (deepmath B.1) "
        "confirms the power-law direction."
    )

    return IngestedSurfaces(
        dseg_384_floor=floor,
        dseg_384_floor_provenance=floor_prov,
        qaxis_byte_shrink=qaxis,
        qaxis_provenance=qaxis_prov,
        eaxis_anchors=eaxis,
        eaxis_provenance=eaxis_prov,
        capacity_anchors=cap_anchors,
        capacity_provenance=cap_prov,
        frontier_S=FRONTIER_S,
        pose_hold=pose_hold,
        notes=notes,
    )


def existence_proof_crosscheck(
    claimed_floor_S: float,
    *,
    d_pose: float,
    surfaces: IngestedSurfaces | None = None,
) -> ExistenceProof:
    """The MANDATORY cross-check (CLAUDE.md feedback_terminal_conclusion 2026-06-23):
    does a known measured artifact ALREADY beat the claimed floor S?

    Cross-checks ``claimed_floor_S`` against the known artifacts:
      * the 0.191 contest frontier (S directly),
      * a PR95-class own-trained decoder at the bc20 byte budget (d_seg 5.6e-4),
      * a PERFECT 384 decoder at the bc20 byte budget (the 384 d_seg floor),
      * the bc20 basin.

    Returns an ExistenceProof. If ANY known artifact's S < claimed_floor_S, the floor is
    INVALID (it is a capacity/recipe/apparatus artifact, not a physics floor) — downgrade.
    """
    if surfaces is None:
        surfaces = load_ingested_surfaces(pose_hold=d_pose)

    candidates: list[tuple[str, float]] = [("contest frontier (0.191)", surfaces.frontier_S)]
    # PR95-class decoder at the bc20 BYTE budget (the cheapest existence proof of low d_seg
    # at a small archive). Uses the bc20 basin's archive bytes — the rate-headroom budget.
    bc20_bytes = crq.ANCHOR_BC20.archive_bytes
    candidates.append(
        (
            "PR95-class d_seg @ bc20 bytes",
            compute_contest_score(PR95_DSEG, d_pose, bc20_bytes),
        )
    )
    # A PERFECT 384 decoder at the bc20 byte budget (the 384 d_seg floor).
    candidates.append(
        (
            "perfect-384 d_seg @ bc20 bytes",
            compute_contest_score(surfaces.dseg_384_floor, d_pose, bc20_bytes),
        )
    )

    best_label, best_S = min(candidates, key=lambda kv: kv[1])
    is_valid = best_S >= claimed_floor_S
    if is_valid:
        verdict = (
            f"VALID floor: no known artifact beats S={claimed_floor_S:.5f} "
            f"(best known {best_label} S={best_S:.5f})."
        )
    else:
        verdict = (
            f"INVALID floor — {best_label} ALREADY achieves S={best_S:.5f} < claimed "
            f"{claimed_floor_S:.5f}. The claim is a capacity/recipe artifact (MP2), NOT a "
            f"physics floor. Downgrade to 'our-current-config-limited' and find the gap."
        )
    return ExistenceProof(
        claimed_floor_S=claimed_floor_S,
        best_known_artifact=best_label,
        best_known_S=best_S,
        is_valid_floor=is_valid,
        verdict=verdict,
    )


def physical_achievable_floor(
    *,
    surfaces: IngestedSurfaces | None = None,
    floor_base_ch: int = 16,
    floor_qat_nbits: int = 4,
    floor_qat_frac_low: float = 1.0,
) -> JointConfig:
    """The PHYSICAL achievable-S floor — the true T_floor over the measured surface,
    replacing the loose analytic 0.118.

    Construction (every term an existence-proof anchor, NO power-law extrapolation):
      * d_seg = the 384 ACHIEVABILITY floor (1.875e-4, MEASURED N=600) — the provable
        lower bound a PERFECT 384-output decoder hits. NOT the power-law value (which is
        a pessimistic capacity extrapolation); this is the physics floor.
      * d_pose = the CONVERGED frontier pose (2.93e-5, MEASURED).
      * bytes = the cheapest viable byte budget: a small (floor_base_ch) decoder after
        int4 QAT. The smallest decoder still has to render the frame, so we do NOT go
        below a real small-basis byte count.

    This floor assumes a decoder that BOTH reaches the 384 d_seg floor AND fits the small
    byte budget — the open capacity-realization question. It is the honest answer to "what
    is the lowest S the measured surface permits?" — and it is comfortably sub-0.15.
    """
    if surfaces is None:
        surfaces = load_ingested_surfaces()
    dec = decoder_param_count(vendored_taper(floor_base_ch))
    native = crq.native_archive_bytes(floor_base_ch)
    low_frac = surfaces.qaxis_byte_shrink.get(floor_qat_nbits, crq.qat_byte_fraction(floor_qat_nbits))
    mixed_frac = (1.0 - floor_qat_frac_low) * 1.0 + floor_qat_frac_low * low_frac
    arch_bytes = round(native * mixed_frac)
    d_seg = surfaces.dseg_384_floor
    d_pose = DPOSE_CONVERGED
    S = compute_contest_score(d_seg, d_pose, arch_bytes)
    return JointConfig(
        base_ch=floor_base_ch,
        taper_dseg_multiplier=1.0,
        qat_nbits=floor_qat_nbits,
        qat_frac_low_precision=floor_qat_frac_low,
        epochs=float("inf"),
        decoder_params=dec,
        d_seg=d_seg,
        d_seg_inf=d_seg,
        d_seg_evidence="PHYSICAL FLOOR: 384 achievability floor (MEASURED N=600), not power-law",
        d_pose=d_pose,
        archive_bytes=arch_bytes,
        S=S,
    )


def plan_capacity_rd_prune_path(
    *,
    source_base_ch: int = 36,
    source_checkpoint_glob: str = "experiments/results/*bc36*n600*/best/",
    target_base_chs: tuple[int, ...] = (16, 20, 24, 28, 32),
    surfaces: IngestedSurfaces | None = None,
) -> PruneRdPathPlan:
    """Build the READY-to-run capacity-RD prune-path measurement plan.

    The capacity power-law exponent α is the dominant gating uncertainty (the 2-point
    bc20↔frontier fit is recipe+convergence-confounded). The clean way to pin it: train
    ONE big decoder (``source_base_ch``) to convergence ONCE, then STRUCTURED-PRUNE +
    KD-finetune DOWN to each ``target_base_chs`` rung and byte-close + exact-score each.
    Every rung shares the SAME teacher + recipe, so the (params, d_seg) points are
    apples-to-apples — the contaminated-fit problem dissolves.

    This emits the PLAN (per-rung predicted (d_seg, bytes, S) + the runner contract). It
    does NOT run (the converged big checkpoint does not exist yet; CPU-only/$0 here). When
    the bc36 n600 run converges, the runner executes ``runner_contract`` per rung and fills
    the ``measured_*`` columns — then re-running ``solve_math_optimal_joint`` with the
    measured (params, d_seg) anchors sharpens the optimal C automatically.
    """
    if surfaces is None:
        surfaces = load_ingested_surfaces()
    steps: list[PruneCapacityStep] = []
    for bc in target_base_chs:
        taper = vendored_taper(bc)
        dec = decoder_param_count(taper)
        d_inf, _ = dseg_capacity_power_law(bc)
        native = crq.native_archive_bytes(bc)
        native_S = compute_contest_score(d_inf, DPOSE_CONVERGED, native)
        steps.append(
            PruneCapacityStep(
                target_base_ch=bc,
                target_decoder_params=dec,
                target_taper=tuple(taper),
                predicted_d_seg_inf=d_inf,
                predicted_native_bytes=native,
                predicted_native_S=native_S,
            )
        )
    runner_contract = (
        "1. load the converged big decoder EMA checkpoint (source_checkpoint_glob).",
        "2. structured-prune channels to the target taper (L2-norm channel importance; "
        "keep the d_seg-critical mid-late stages per the gate-2 sensitivity map).",
        "3. KD-finetune the pruned decoder from the big teacher (NOT from scratch) at the "
        "matched recipe (muon_lr 0.03, clip 50) for a fixed short budget per rung.",
        "4. build the int8-brotli archive (vendored build_archive) -> byte-close.",
        "5. exact-score via upstream/evaluate.py on the byte-closed archive (CPU; CUDA for "
        "any frontier/submission claim) -> fill measured_d_seg / measured_bytes / measured_S.",
        "6. append the (target_decoder_params, measured_d_seg) anchor to the E/C surface "
        "JSON; re-run solve_math_optimal_joint to sharpen alpha + the optimal C.",
    )
    notes = (
        "GATED: the converged big (bc36) n600 decoder must exist first (the never-fired "
        "run, deepmath E.1). This plan is the actuator that runs the instant it lands.",
        "The prune-path gives APPLES-TO-APPLES (params, d_seg) — same teacher, same recipe "
        "— so it pins alpha cleanly (replacing the contaminated 2-point fit, deepmath C.1).",
        "Predicted columns are the solver's power-law model (CLAMPED at the 384 floor); the "
        "measured columns are the ground truth the runner fills. A measured d_seg ABOVE the "
        "prediction at low capacity means the small decoder cannot realize the floor "
        "(capacity wall EARNED); BELOW means more sub-0.15 headroom than the model shows.",
    )
    return PruneRdPathPlan(
        source_checkpoint_glob=source_checkpoint_glob,
        source_base_ch=source_base_ch,
        steps=tuple(steps),
        method="structured channel prune (L2 importance) + KD-finetune from the big teacher",
        runner_contract=runner_contract,
        gated_on="converged bc36 n600 decoder checkpoint (deepmath E.1, the never-fired run)",
        notes=notes,
    )


def _eval_joint_config(
    *,
    base_ch: int,
    taper_dseg_multiplier: float,
    qat_nbits: int,
    qat_frac_low_precision: float,
    epochs: float,
    d_pose: float | None,
    qat_d_seg_hold_delta: float,
    qaxis_byte_shrink: dict[int, float],
) -> JointConfig:
    """Compute the modelled (d_seg, d_pose, bytes, S) for one (C,T,Q,E) point.

    d_seg(C,T,Q,E) = [ dseg_convergence(E -> d_seg_inf(C)) * T_mult ] + Q_hold_spill,
      clamped at the 384 floor (no decoder beats the perfect-reconstruction floor).
    d_pose(E) = dpose_convergence(E -> frontier pose) — DECOUPLED from capacity; a fixed
      ``d_pose`` override pins it (e.g. for a sensitivity sweep). Pose is the dominant
      sub-0.15 swing, so it is convergence-dependent by default, NOT a basin constant.
    bytes(C,Q) = native_bytes(C) * mixed_byte_fraction(Q).
    """
    dec = decoder_param_count(vendored_taper(base_ch))
    d_seg_inf, ev = dseg_capacity_power_law(base_ch)
    # E-axis: converge toward the asymptote.
    d_seg_E = dseg_convergence(epochs, d_seg_inf)
    # T-axis: byte-neutral boundary-band reallocation multiplier.
    d_seg_T = d_seg_E * taper_dseg_multiplier
    # Q-axis: the QAT distortion-hold spill (added d_seg from low-bit quantization).
    # frac_low=0 -> int8, no spill; scale the spill by the low-precision fraction.
    q_spill = qat_d_seg_hold_delta * qat_frac_low_precision
    d_seg = d_seg_T + q_spill
    # Clamp at the 384 floor (existence-proof lower bound; cannot go below).
    d_seg = max(d_seg, DSEG_384_FLOOR)

    # Pose: convergence-dependent (decoupled from capacity), unless pinned.
    d_pose_eff = dpose_convergence(epochs) if d_pose is None else d_pose

    # bytes: native byte model * mixed byte fraction.
    native = crq.native_archive_bytes(base_ch)
    low_frac = qaxis_byte_shrink.get(qat_nbits, crq.qat_byte_fraction(qat_nbits))
    mixed_frac = (1.0 - qat_frac_low_precision) * 1.0 + qat_frac_low_precision * low_frac
    arch_bytes = round(native * mixed_frac)

    S = compute_contest_score(d_seg, d_pose_eff, arch_bytes)
    return JointConfig(
        base_ch=base_ch,
        taper_dseg_multiplier=taper_dseg_multiplier,
        qat_nbits=qat_nbits,
        qat_frac_low_precision=qat_frac_low_precision,
        epochs=epochs,
        decoder_params=dec,
        d_seg=d_seg,
        d_seg_inf=d_seg_inf,
        d_seg_evidence=ev,
        d_pose=d_pose_eff,
        archive_bytes=arch_bytes,
        S=S,
    )


def solve_math_optimal_joint(
    *,
    base_chs: tuple[int, ...] = (16, 20, 24, 28, 32, 36, 40),
    taper_dseg_multipliers: tuple[float, ...] = (1.0,),
    qat_nbits_options: tuple[int, ...] = (4, 5, 6, 8),
    qat_frac_low_options: tuple[float, ...] = (0.0, 0.70, 1.0),
    epochs_options: tuple[float, ...] = (2325.0, 10000.0, 30000.0, 1e6),
    qat_d_seg_hold_delta: float = 0.0003,
    surfaces: IngestedSurfaces | None = None,
    d_pose: float | None = None,
) -> JointSolveResult:
    """Solve min S(C,T,Q,E) over the measured surfaces and emit the math-optimal config,
    the measured achievable-S lower bound (real T_floor), the training-time Pareto, and
    the gating sensitivities.

    The solve is a STRUCTURED grid over the four axes (the surfaces are
    measured-at-anchors + power-law/exponential MODELS between them, so a closed-form KKT
    is solved on the C-axis inside ``capacity_rd_qat``; the joint over T/Q/E is small and
    enumerated). For the C-axis the convex S(p) optimum is already KKT-solved by
    ``capacity_rd_qat.run_desk_calc``; this joint adds the T/Q/E axes around it.

    The achievable-S LOWER BOUND is the fully-converged (E->inf), best-T, best-Q config —
    the real T_floor over the measured surface. It is cross-checked by the existence proof.
    """
    if surfaces is None:
        surfaces = load_ingested_surfaces()
    # d_pose=None means convergence-driven (pose -> frontier as E grows); an explicit
    # value PINS pose for a sensitivity sweep.

    grid: list[JointConfig] = []
    for bc in base_chs:
        for t in taper_dseg_multipliers:
            for q in qat_nbits_options:
                for fl in qat_frac_low_options:
                    # frac_low=0 means int8 regardless of nbits — dedupe to nbits=8.
                    eff_nbits = 8 if fl == 0.0 else q
                    for E in epochs_options:
                        cfg = _eval_joint_config(
                            base_ch=bc,
                            taper_dseg_multiplier=t,
                            qat_nbits=eff_nbits,
                            qat_frac_low_precision=fl,
                            epochs=E,
                            d_pose=d_pose,  # None -> convergence-driven per E
                            qat_d_seg_hold_delta=qat_d_seg_hold_delta,
                            qaxis_byte_shrink=surfaces.qaxis_byte_shrink,
                        )
                        grid.append(cfg)

    # dedupe (int8 at frac_low=0 appears once per nbits).
    seen: dict[tuple, JointConfig] = {}
    for c in grid:
        key = (c.base_ch, c.taper_dseg_multiplier, c.qat_nbits, c.qat_frac_low_precision, c.epochs)
        seen[key] = c
    grid = list(seen.values())

    optimum = min(grid, key=lambda c: c.S)

    # Achievable-S lower bound = fully-converged (E -> inf) best config = the real T_floor.
    converged = [c for c in grid if c.epochs >= 1e6]
    lb_config = min(converged, key=lambda c: c.S) if converged else optimum
    achievable_S = lb_config.S
    # Existence-proof uses the CONVERGED pose (the lower bound config's pose), since the
    # known artifacts (frontier, PR95) are themselves fully-converged decoders.
    proof = existence_proof_crosscheck(
        achievable_S, d_pose=lb_config.d_pose, surfaces=surfaces
    )

    # Training-time Pareto at the OPTIMUM's (C,T,Q): S vs epochs.
    pareto = training_time_pareto(
        base_ch=optimum.base_ch,
        taper_dseg_multiplier=optimum.taper_dseg_multiplier,
        qat_nbits=optimum.qat_nbits,
        qat_frac_low_precision=optimum.qat_frac_low_precision,
        d_pose=d_pose,
        qat_d_seg_hold_delta=qat_d_seg_hold_delta,
        surfaces=surfaces,
    )

    # The PHYSICAL achievable floor (the true T_floor) — the existence-proof construction.
    phys_floor = physical_achievable_floor(surfaces=surfaces)

    gating = _gating_sensitivities(optimum, lb_config, surfaces, d_pose, qat_d_seg_hold_delta)

    notes = [
        "ALL numbers [advisory] / [macOS-CPU advisory] NON-PROMOTABLE; exact pointer "
        "stays pointer-only (0.19110). The output is a CONFIG RECOMMENDATION for the next "
        "training run, not a score claim.",
        "C-axis d_seg(C) is the bc20+frontier power law, CLAMPED at the 384 floor "
        "(1.875e-4). T-axis multiplier default 1.0 (no measured taper A/B yet). Q-axis "
        "byte fractions MEASURED (int8->int-N bc20); the d_seg-hold spill is the QAT "
        "distortion-hold ASSUMPTION (must be measured). E-axis convergence is a "
        "conservative 2-point exponential fit on stage-1 CE-only anchors.",
        "The achievable-S lower bound is the FULLY-CONVERGED best config; its validity is "
        "gated by the existence-proof cross-check.",
    ]
    notes.extend(surfaces.notes)
    # The model-LB existence-proof "INVALID" is EXPECTED and informative here: it is the
    # discipline confirming that the surface-MODEL lower bound (the pessimistic 2-point
    # d_seg power law + the int4 spill ASSUMPTION) is NOT the physics floor — the
    # physical_floor_S (perfect-384 d_seg + converged pose + small bytes) is far lower.
    # The two-layer answer: the model says "no config reaches sub-0.15 under the current
    # power-law d_seg", but the physical floor says "sub-0.15 is permitted IF a small
    # decoder can be trained to the 384 d_seg floor". The GAP between them is the single
    # binding open question (capacity-realization), NOT a contradiction.
    notes.append(
        f"TWO-LAYER FLOOR: surface-model lower bound S={achievable_S:.4f} (power-law d_seg, "
        f"pessimistic) vs PHYSICAL floor S={phys_floor.S:.4f} (perfect-384 d_seg + converged "
        f"pose + bc{phys_floor.base_ch} int4 bytes). The gap is the capacity-realization "
        "question: the power law is a recipe+convergence-confounded extrapolation (deepmath "
        "B), NOT a physics floor — the 384 measurement proves d_seg 1.875e-4 is achievable."
    )
    if not proof.is_valid_floor:
        notes.append("EXISTENCE-PROOF (on the MODEL LB, EXPECTED): " + proof.verdict)

    return JointSolveResult(
        optimum=optimum,
        achievable_S_lower_bound=achievable_S,
        achievable_S_lower_bound_config=lb_config,
        achievable_S_existence_proof=proof,
        physical_floor_S=phys_floor.S,
        physical_floor_config=phys_floor,
        frontier_S=surfaces.frontier_S,
        training_pareto=pareto,
        grid=grid,
        gating=gating,
        surfaces=surfaces,
        notes=notes,
    )


def training_time_pareto(
    *,
    base_ch: int,
    taper_dseg_multiplier: float = 1.0,
    qat_nbits: int = 8,
    qat_frac_low_precision: float = 0.0,
    d_pose: float | None = None,
    qat_d_seg_hold_delta: float = 0.0003,
    surfaces: IngestedSurfaces | None = None,
    epochs_grid: tuple[float, ...] = (
        500.0, 1000.0, 2325.0, 5000.0, 10000.0, 20000.0, 30000.0, 50000.0, 1e6,
    ),
) -> list[TrainingParetoPoint]:
    """The S-vs-training-budget Pareto surface at a fixed (C,T,Q). The training-time axis
    the operator asked for: each point is (epochs, d_seg(epochs), S)."""
    if surfaces is None:
        surfaces = load_ingested_surfaces()
    # d_pose=None -> convergence-driven (pose improves with epochs alongside d_seg); the
    # training-time Pareto therefore moves BOTH d_seg and d_pose down the curve.
    out: list[TrainingParetoPoint] = []
    for E in epochs_grid:
        cfg = _eval_joint_config(
            base_ch=base_ch,
            taper_dseg_multiplier=taper_dseg_multiplier,
            qat_nbits=qat_nbits,
            qat_frac_low_precision=qat_frac_low_precision,
            epochs=E,
            d_pose=d_pose,
            qat_d_seg_hold_delta=qat_d_seg_hold_delta,
            qaxis_byte_shrink=surfaces.qaxis_byte_shrink,
        )
        out.append(TrainingParetoPoint(epochs=E, d_seg=cfg.d_seg, S=cfg.S))
    return out


def min_training_budget_for_threshold(
    target_S: float,
    *,
    base_ch: int,
    taper_dseg_multiplier: float = 1.0,
    qat_nbits: int = 8,
    qat_frac_low_precision: float = 0.0,
    d_pose: float | None = None,
    qat_d_seg_hold_delta: float = 0.0003,
    surfaces: IngestedSurfaces | None = None,
    max_epochs: float = 1e7,
) -> float | None:
    """The minimum training budget (epochs) at which a (C,T,Q) config first reaches
    ``target_S``. Returns None if the config's fully-converged S never reaches target.

    Solves the d_seg(E) convergence curve for the E at which S(E) == target_S, via the
    closed-form inverse of the exponential-approach model (then validates).
    """
    if surfaces is None:
        surfaces = load_ingested_surfaces()

    # The fully-converged config — if even E->inf can't reach target, return None.
    conv = _eval_joint_config(
        base_ch=base_ch,
        taper_dseg_multiplier=taper_dseg_multiplier,
        qat_nbits=qat_nbits,
        qat_frac_low_precision=qat_frac_low_precision,
        epochs=max_epochs,
        d_pose=d_pose,  # None -> convergence-driven (converged pose at max_epochs)
        qat_d_seg_hold_delta=qat_d_seg_hold_delta,
        qaxis_byte_shrink=surfaces.qaxis_byte_shrink,
    )
    if target_S + 1e-12 < conv.S:
        return None

    # Invert: we need d_seg(E) such that S == target_S. Pose is taken at the CONVERGED
    # value (conv.d_pose) — an APPROXIMATION: at the budgets where d_seg crosses threshold
    # pose is near-converged, so this slightly UNDER-states the required budget (pose is
    # still improving). The exact joint inversion (both d_seg AND d_pose moving) would
    # need a 1D root-find; the converged-pose approximation is the honest closed form.
    pose_for_inversion = conv.d_pose
    required_d_seg_total = break_even_d_seg(target_S, pose_for_inversion, conv.archive_bytes)
    if required_d_seg_total <= DSEG_384_FLOOR:
        # Even the 384 floor is above the requirement at these bytes -> never reachable
        # by d_seg alone at this (C,T,Q); but conv.S<=target means floor satisfies it.
        return 0.0 if target_S >= conv.S else None

    # Undo the T multiplier + Q spill to get the required PRE-T/Q d_seg(E).
    q_spill = qat_d_seg_hold_delta * qat_frac_low_precision
    required_d_seg_E = (required_d_seg_total - q_spill) / max(taper_dseg_multiplier, 1e-9)
    d_seg_inf = dseg_capacity_floor_bounded(base_ch)
    if required_d_seg_E <= d_seg_inf:
        return float("inf")  # need the asymptote — effectively unbounded budget
    # invert d_seg(E) = d_seg_inf + gap*exp(-E/tau)  for E.
    tau = _fit_tau_E(d_seg_inf)
    gap = max(0.0, DSEG_INIT - d_seg_inf)
    if gap <= 0:
        return 0.0
    ratio = (required_d_seg_E - d_seg_inf) / gap
    if ratio >= 1.0:
        return 0.0  # already there at E=0
    if ratio <= 0.0:
        return float("inf")
    E = -tau * math.log(ratio)
    return max(0.0, E)


def _gating_sensitivities(
    optimum: JointConfig,
    lb_config: JointConfig,
    surfaces: IngestedSurfaces,
    d_pose: float | None,
    qat_d_seg_hold_delta: float,
) -> dict[str, str]:
    """What's gated + the value that flips the optimum (the operator's explicit ask)."""
    g: dict[str, str] = {}

    # 1) The QAT d_seg-hold delta is the dominant ASSUMPTION. Find the spill at which the
    #    int4 path stops beating int8 at the optimum capacity.
    g["qat_d_seg_hold_delta"] = (
        f"ASSUMPTION (current {qat_d_seg_hold_delta:.4g}). This is the QAT distortion-hold "
        "spill — UNMEASURED. The int4/int5 byte win is real; whether d_seg holds within "
        "this spill is the gating measurement. Flips the optimum away from low-bit Q if "
        "the measured spill exceeds the byte saving in S-units "
        f"(byte saving int4 vs int8 at optimum ~"
        f"{rate_term(crq.native_archive_bytes(optimum.base_ch)) - rate_term(round(crq.native_archive_bytes(optimum.base_ch) * crq.qat_byte_fraction(4))):.4f} S)."
    )

    # 2) The capacity power-law exponent (alpha in [0.9, 1.5]) gates the optimal C.
    g["capacity_alpha"] = (
        "MODELLED alpha in [0.9, 1.5] (deepmath C.1, 2-point clean fit, under-converged "
        "-> lower bound). The capacity-RD optimum p* shifts with alpha (deepmath C.2: "
        "alpha 0.91 -> bc27, 1.12 -> bc29, 1.50 -> bc32). The clean bc36 anchor + its "
        "prune-path will pin alpha and SHARPEN the optimal C. Until then C* is "
        f"{optimum.base_ch} (current surfaces)."
    )

    # 3) The taper d_seg multiplier (T) is unmeasured -> default 1.0.
    g["taper_dseg_multiplier"] = (
        "DEFAULT 1.0 (no measured taper A/B). A boundary-band-reallocated taper is "
        "byte-NEUTRAL; if a measured arm_b/control d_seg ratio < 1.0 lands, it lowers "
        "d_seg at zero byte cost -> directly lowers S and can flip the optimum toward "
        "LOWER capacity (less capacity needed for the same d_seg)."
    )

    # 4) The 384 floor is the hard d_seg lower bound -> a higher-res decoder is the only
    #    way below it (and only by 0.003 S).
    g["dseg_384_floor"] = (
        f"HARD lower bound {surfaces.dseg_384_floor:.4g} (S {seg_term(surfaces.dseg_384_floor):.4f}). "
        "A >=camera-res decoder could dip toward the float floor (1.596e-4) but only buys "
        "~0.003 S on d_seg — not worth a resolution rebuild. Capacity-within-384 is the "
        "larger d_seg lever."
    )

    # 5) The convergence (E) gate: how many epochs to the asymptote at the optimum C.
    e90 = min_training_budget_for_threshold(
        optimum.S + 0.005,
        base_ch=optimum.base_ch,
        taper_dseg_multiplier=optimum.taper_dseg_multiplier,
        qat_nbits=optimum.qat_nbits,
        qat_frac_low_precision=optimum.qat_frac_low_precision,
        d_pose=d_pose,
        qat_d_seg_hold_delta=qat_d_seg_hold_delta,
        surfaces=surfaces,
    )
    g["convergence_epochs"] = (
        f"The optimum needs ~{e90 if e90 is None else round(e90)} effective epochs to come "
        "within 0.005 S of its asymptote (conservative CE-only fit; the curriculum's "
        "d_seg finisher stages converge faster). This is the never-fired run's budget."
    )
    return g
