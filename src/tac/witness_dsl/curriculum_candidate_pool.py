"""Curriculum-candidate POOL — the P0 tracked costate class (task #403).

Per the CLAUDE.md NON-NEGOTIABLE "'Off' is a tracked queue, never a forgotten default" +
the OPERATOR-PRIORITY P0 binding 2026-07-10 (memory
``curriculum_candidate_pool_p0_orphan_class_20260710`` + design memo
``.omx/research/curriculum_candidate_pool_p0_20260710.md``): a CURRICULUM candidate in ANY form
(stages / losses / init / preconditioning / data-order / averaging / solve-interleave /
state-evolution) that is designed-or-built-but-never-fired is the SAME orphaned-signal class the
lever activation ledger tracks — but a curriculum candidate is often a TOOL, a stage, a data-order,
or a whole vehicle DOF that is NOT a single ``--flag`` DSL ``Lever``, so the lever-scoped
``activation_ledger`` event vocabulary cannot hold it without overloading its schema (the #400
agent's finding). This module is the SIBLING store that holds those candidates WITHOUT that overload.

A chat-only inventory of built-never-fired curriculum candidates IS the orphan bug — so the inventory
lives here as a durable, ranked, controller-held queue that ``tools/costate_digest.py`` §curriculum-pool
reads. The controller remembers and surfaces the next-fireable rows; the operator never has to.

Store: canonical, APPEND-ONLY, fcntl-locked JSONL at ``.omx/state/curriculum_candidate_pool.jsonl``
(mirrors the ``.omx/state/*.jsonl`` canonical stores; latest-row-wins per ``candidate`` key on read).

NO-FAKE discipline (per CLAUDE.md supreme rule): every row carries a ``source_anchor`` (the commit /
memo / DAG-FEED / task-# that BUILT or DESIGNED it) and its HONEST ``status`` — a designed-but-unbuilt
candidate is ``needs-build`` (NOT ``measured``); a built-but-never-launched candidate is
``built-never-fired`` (NOT ``measured``). ``measured`` requires a byte-closed verdict anchor. This store
MEASURES nothing and mints no law (equations leg N/A-with-rationale) — it CITES existing registered
equations. It moves the pointer by ZERO: it is apparatus (means), not a score (end).

State (``status``) machine, latest-row-wins per candidate:
  needs-build / built-never-fired / reformulation-queue / armed  --measured-->  measured
  (and --retired--> retired-with-reason from any state; terminal, dormant-with-reactivation).

The #247 costate SENSE layer consumes :func:`duty_to_measure_pool` to rank never-fired high-value
curriculum candidates into its DECIDE queue beside the lever duty-to-measure line.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PATH = _REPO_ROOT / ".omx" / "state" / "curriculum_candidate_pool.jsonl"

# ── canonical status vocabulary (latest-row-wins per candidate) ──────────────────────────────────
STATUS_ARMED = "armed"                            # already flying in the sealed launch config
STATUS_BUILT_NEVER_FIRED = "built-never-fired"    # code exists (default-off byte-identical), never launched
STATUS_NEEDS_BUILD = "needs-build"                # design/task exists, no code on THIS vehicle
STATUS_REFORMULATION_QUEUE = "reformulation-queue"  # prior formulation measured NO-GO; named reformulation owed
STATUS_MEASURED = "measured"                       # a byte-closed verdict landed (requires verdict anchor)
STATUS_RETIRED = "retired-with-reason"             # retired with a recorded MEASURED law (dormant-w-reactivation)
VALID_STATUSES = frozenset({
    STATUS_ARMED, STATUS_BUILT_NEVER_FIRED, STATUS_NEEDS_BUILD,
    STATUS_REFORMULATION_QUEUE, STATUS_MEASURED, STATUS_RETIRED,
})

# ── canonical form-class vocabulary (the "ANY form" the operator P0 binding enumerated) ──────────
VALID_FORM_CLASSES = frozenset({
    "loss-geometry", "optimizer-stage", "regularizer-schedule", "data-curriculum",
    "architecture-growth", "averaging", "preconditioning", "discrete-solve-interleave",
    "init-warm-start", "state-evolution", "pose-carrier",
})

VALID_AXES = frozenset({"d_seg", "d_pose", "rate"})

# ranking order: the duty queue the controller drains (built-never-fired highest readiness) first.
_STATUS_ORDER = {
    STATUS_BUILT_NEVER_FIRED: 0,
    STATUS_NEEDS_BUILD: 1,
    STATUS_REFORMULATION_QUEUE: 2,
    STATUS_ARMED: 3,
    STATUS_MEASURED: 4,
    STATUS_RETIRED: 5,
}
# the statuses that are OWED a measurement (the controller's duty-to-measure queue for curricula).
_DUTY_STATUSES = frozenset({STATUS_BUILT_NEVER_FIRED, STATUS_NEEDS_BUILD, STATUS_REFORMULATION_QUEUE})


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_locked_jsonl(p: Path, row: dict) -> None:
    """fcntl-locked APPEND of ONE json row (canonical .omx/state pattern; best-effort off-POSIX)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    try:
        import fcntl
        with open(p, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover - non-POSIX fallback
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)


def record_candidate(
    candidate: str,
    status: str,
    *,
    form_class: str,
    source_anchor: str,
    gate: str,
    justification: str = "",
    dsl_lever: str | None = None,
    dsl_na_reason: str | None = None,
    slot: str = "",
    owner: str = "",
    est_delta_s: float | None = None,
    axis: str | None = None,
    verdict_ref: str | None = None,
    agent: str | None = None,
    path: Path | None = None,
) -> dict:
    """Append ONE curriculum-candidate row (APPEND-ONLY, fcntl-locked). Returns the written row.

    NO-FAKE contract, enforced here:
      * ``status`` ∈ :data:`VALID_STATUSES`; ``form_class`` ∈ :data:`VALID_FORM_CLASSES`.
      * ``source_anchor`` REQUIRED — every row cites the commit / memo / DAG-FEED / task-# that
        BUILT or DESIGNED it (never a candidate asserted from thin air).
      * EXACTLY-ONE-OF ``dsl_lever`` (the held ``curriculum_dsl`` factory name) OR ``dsl_na_reason``
        (why this candidate is NOT a single-flag DSL lever — tool-side / vehicle-level / not-yet-built).
        This is the per-row DSL-leg discipline (the triality DSL leg for the pool).
      * ``status == 'measured'`` REQUIRES a ``verdict_ref`` (a byte-closed verdict artifact) — a
        designed/built candidate may NEVER be recorded ``measured`` without one (NO-FAKE #8: no
        surrogate-as-authority).
      * ``est_delta_s`` (optional) is the POSITIVE ΔS magnitude the candidate buys, joining the same
        relative-significance math the lever queue uses; requires ``axis`` ∈ :data:`VALID_AXES`.
    """
    if not candidate or not isinstance(candidate, str):
        raise ValueError(f"candidate must be a non-empty str, got {candidate!r}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; must be one of {sorted(VALID_STATUSES)}")
    if form_class not in VALID_FORM_CLASSES:
        raise ValueError(f"invalid form_class {form_class!r}; must be one of {sorted(VALID_FORM_CLASSES)}")
    if not source_anchor:
        raise ValueError("source_anchor is required (NO-FAKE: every candidate cites its build/design anchor)")
    # exactly-one-of {dsl_lever, dsl_na_reason}
    if bool(dsl_lever) == bool(dsl_na_reason):
        raise ValueError(
            "exactly one of dsl_lever / dsl_na_reason is required (the per-row DSL-leg discipline): "
            f"got dsl_lever={dsl_lever!r} dsl_na_reason={dsl_na_reason!r}")
    if status == STATUS_MEASURED and not verdict_ref:
        raise ValueError(
            "status 'measured' requires a verdict_ref (byte-closed verdict artifact); NO-FAKE: a "
            "designed/built candidate is never 'measured' without a byte-closed row")
    if est_delta_s is not None:
        est_delta_s = float(est_delta_s)
        if est_delta_s < 0:
            raise ValueError(f"est_delta_s must be a positive ΔS magnitude or None, got {est_delta_s!r}")
        if axis not in VALID_AXES:
            raise ValueError(f"est_delta_s requires axis ∈ {sorted(VALID_AXES)}, got {axis!r}")
    row = {
        "candidate": candidate,
        "status": status,
        "form_class": form_class,
        "source_anchor": source_anchor,
        "gate": gate,
        "justification": justification,
        "dsl_lever": dsl_lever,
        "dsl_na_reason": dsl_na_reason,
        "slot": slot,
        "owner": owner,
        "est_delta_s": est_delta_s,
        "axis": axis,
        "verdict_ref": verdict_ref,
        "agent": agent,
        "ts": _utc(),
    }
    _append_locked_jsonl(Path(path) if path is not None else POOL_PATH, row)
    return row


def _read_pool(path: Path | None = None, *, include_seed: bool | None = None) -> dict[str, dict]:
    """Latest-row-wins map ``candidate -> row`` (lenient; skips corrupt lines).

    On the DEFAULT store path (``path is None``) the committed :data:`_SEED` inventory is the READ
    BASELINE (``include_seed`` defaults True) — so a fresh checkout is never empty WITHOUT tracking a
    runtime ``.jsonl`` (the store is gitignored, matching the sibling ledgers; the durable inventory
    lives in code). Real recorded rows (fires / measures / retires appended via
    :func:`record_candidate`) OVERLAY the seed baseline (latest-wins). An EXPLICIT ``path`` (tests /
    tooling) reads that file PURELY (``include_seed`` defaults False) for isolation.
    """
    if include_seed is None:
        include_seed = path is None
    out: dict[str, dict] = {}
    if include_seed:
        out.update(_seed_map())  # committed inventory baseline (overlaid by real store rows below)
    p = Path(path) if path is not None else POOL_PATH
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(row, dict) and row.get("candidate") and row.get("status") in VALID_STATUSES:
            out[row["candidate"]] = row  # later row overwrites earlier -> latest-wins (over seed too)
    return out


@dataclass(frozen=True)
class CandidateStatus:
    candidate: str
    status: str
    form_class: str
    dsl_lever: str | None
    dsl_na_reason: str | None
    gate: str
    source_anchor: str
    slot: str
    owner: str
    est_delta_s: float | None
    axis: str | None
    in_duty_queue: bool


def candidate_status(candidate: str, path: Path | None = None) -> CandidateStatus | None:
    """The latest recorded status of ONE candidate, or ``None`` if it is not in the pool."""
    row = _read_pool(path).get(candidate)
    if row is None:
        return None
    return _row_to_status(row)


def _row_to_status(row: dict) -> CandidateStatus:
    return CandidateStatus(
        candidate=row["candidate"],
        status=row["status"],
        form_class=row.get("form_class", ""),
        dsl_lever=row.get("dsl_lever"),
        dsl_na_reason=row.get("dsl_na_reason"),
        gate=row.get("gate", ""),
        source_anchor=row.get("source_anchor", ""),
        slot=row.get("slot", ""),
        owner=row.get("owner", ""),
        est_delta_s=row.get("est_delta_s"),
        axis=row.get("axis"),
        in_duty_queue=row["status"] in _DUTY_STATUSES,
    )


def _sort_key(row: dict) -> tuple:
    """Rank: status order (built-never-fired first) -> est_delta_s desc (None last) -> candidate."""
    est = row.get("est_delta_s")
    return (
        _STATUS_ORDER.get(row.get("status"), 9),
        0 if est is not None else 1,
        -(est or 0.0),
        row.get("candidate", ""),
    )


def pool_report(path: Path | None = None) -> list[dict]:
    """The operator-facing pool: EVERY tracked candidate, ranked built-never-fired -> needs-build ->
    reformulation-queue -> armed -> measured -> retired. Each row carries its DSL-leg disposition +
    gate + source anchor, so a reviewer scans one table and sees what is owed a fire vs already flying."""
    rows = sorted(_read_pool(path).values(), key=_sort_key)
    return rows


def duty_to_measure_pool(path: Path | None = None) -> list[dict]:
    """Curriculum candidates OWED a measurement (built-never-fired / needs-build / reformulation-queue),
    ranked. The costate SENSE reads these into its DECIDE queue beside the lever duty-to-measure line —
    the CONTROLLER holds the curriculum queue, the operator never has to remember it."""
    return [r for r in pool_report(path) if r.get("status") in _DUTY_STATUSES]


def pool_summary(path: Path | None = None) -> dict:
    """Machine-readable counts per status + the top next-fireable rows (for the digest / --json)."""
    pool = _read_pool(path)
    counts: dict[str, int] = {s: 0 for s in VALID_STATUSES}
    for r in pool.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    duty = duty_to_measure_pool(path)
    return {
        "total": len(pool),
        "counts": counts,
        "owed": len(duty),
        "top_fireable": duty[:6],
    }


# ── CANONICAL SEED INVENTORY (from the #403 design memo, re-derived from primary artifacts) ──────
# Each tuple is a candidate row. This is the durable inventory the memo re-derived (labels MEASURED /
# DERIVED / ESTIMATED per row); it is the SEED for the store so a fresh checkout is never empty. Rows
# with a held DSL factory carry ``dsl_lever``; tool-side / vehicle-level / not-yet-built rows carry
# ``dsl_na_reason`` (the per-row DSL-leg discipline — exactly one). NO-FAKE: unbuilt rows are
# ``needs-build``, built-never-launched are ``built-never-fired`` — none is ``measured``.
_SEED: tuple[dict, ...] = (
    # ── §1 ARMED (already flying in the sealed v7.5.2 launch config) ──────────────────────────────
    dict(candidate="seg_form_unify_tau", status=STATUS_ARMED, form_class="loss-geometry",
         dsl_lever="SegFormUnifyTau", slot="in-run-stage", owner="#302",
         gate="composed in v752 launch-1 (DERIVED blinded, #302 §1: CE stage IS the τ≈1 arc)",
         justification="witness-native schedule derivation (dissolves the PR95 CE→tau switch)",
         source_anchor=".omx/research/witness_native_schedule_derivation_20260709.md §1.2"),
    dict(candidate="tail_k_warm_restart", status=STATUS_ARMED, form_class="optimizer-stage",
         dsl_lever="TailCycles", slot="in-run-stage", owner="#302",
         gate="composed in v752 (DERIVED #302 §2 finite-τ turnpike)",
         justification="TAIL turnpike cycles", source_anchor="witness_native_schedule_derivation_20260709.md §2"),
    dict(candidate="n323_ladder_island_homotopy", status=STATUS_ARMED, form_class="loss-geometry",
         dsl_lever="LadderIslandHomotopy", slot="in-run-stage", owner="#302",
         gate="composed in v752 (MEASURED-vindicated Phase-2 element-4 STRONG MATCH)",
         justification="LADDER island-birth per-class-λ homotopy",
         source_anchor="witness_native_schedule_derivation_20260709.md Phase-2 element-4"),
    dict(candidate="r7_polyak_finisher", status=STATUS_ARMED, form_class="averaging",
         dsl_lever="PolyakFinisher", slot="terminal-band", owner="#302",
         gate="composed in v752 (extra ckpt candidate; EMA shadow NEVER replaced)",
         justification="turnpike orbit → uniform tail mean O(1/√n) beats phase-carrying EMA",
         source_anchor="witness_native_schedule_derivation_20260709.md (turnpike averaging)"),
    dict(candidate="dseg_aware_taper", status=STATUS_ARMED, form_class="regularizer-schedule",
         dsl_lever="DsegAwareTaper", slot="in-run-stage", owner="ladder",
         gate="v752 delta (INSTANCE-grade; owed-15 n600 fresh-arm isolation converts INSTANCE→MEASURED-or-rollback)",
         justification="byte-neutral spectral reallocation by GT margin saliency (#121)",
         source_anchor="canonical_equations.dseg_aware_fourier_taper_20260709",
         est_delta_s=None, axis=None),

    # ── §2 BUILT-NEVER-FIRED (the duty-to-measure core; highest readiness) ────────────────────────
    dict(candidate="hardness_oversample_lever5", status=STATUS_BUILT_NEVER_FIRED,
         form_class="data-curriculum", dsl_lever="HardnessOversample", slot="in-run-stage", owner="#403",
         gate="fair A/B = --hardness-weighted on/off at fixed oversample (same total steps, different "
              "allocation); source=realized (trainer-recommended)",
         justification="MEASURED anchors: 44%-of-CE-residual-spikes-are-LANE (#205 CE-floor L67) + "
                       "margin-saliency #141; trainer L11303-11324 default-off byte-identical; DSL leg folded this landing",
         source_anchor="experiments/train_levelset_witness_realized_through_R_mlx.py L11303-11324 (LEVER-5)"),
    dict(candidate="step_native_finer", status=STATUS_BUILT_NEVER_FIRED, form_class="architecture-growth",
         dsl_lever="StepNativeActivation", slot="in-run-stage", owner="#310",
         gate="pinned v7.5.2 ladder rung; byte-close A/B annealed-β+FINER vs OFF; surviving flips must move to no-ring survival",
         justification="ESTIMATED 31.6% of remaining descent (FEED-stepnative); negcure RANK-4 second-exemplar",
         source_anchor="DAG FEED-stepnative (factory+flags+equation 07-07; guard-hardened 07-09)",
         est_delta_s=None, axis="d_seg"),
    dict(candidate="focal_boundary_distance_301", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="SegFocalGamma", slot="in-run-stage", owner="#301",
         gate="PRE-REGISTERED (#301 item 3): ep50→100 witness-alone slope flattens (|Δd_seg|<0.02/25ep, "
              "islands>50% of residual) → deploy focal at γ*; steep → HOLD. CAVEAT: re-check γ* on live ckpt (C17 γ*=0)",
         justification="MEASURED build (task #301 completed: default-OFF byte-identical, tested, READY-not-deployed)",
         source_anchor="task #301 (completed); BoundaryDistance sister lever"),
    dict(candidate="head_geometry_218_etf_am", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="HeadGeometry", slot="in-run-stage", owner="#218",
         gate="fire ETF first (byte-free + rate-win, neural-collapse minority-norm fix); AM-hinge needs "
              "--margin-field-head-weight>0",
         justification="DERIVED (#218: rare-class lane margin fix; trainer L10814-10824 built default-off); "
                       "DSL leg folded this landing",
         source_anchor="experiments/train_levelset_witness_realized_through_R_mlx.py L10814-10824 (#218 facet-1)"),
    dict(candidate="persistence_topology_218_224", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="PersistenceTopology", slot="in-run-stage", owner="#218",
         gate="warm-up epochs param; fire with the #218 rung; --persistence-classes auto self-detects erasure-tail classes",
         justification="MEASURED build (#224 wired; dash erasure law dash_erasure_homogenization_v1 is the target, L65)",
         source_anchor="canonical_equations.dash_erasure_homogenization_v1"),
    dict(candidate="msal_uni_sr_reachability_268", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="MarginSaliencyReachability", slot="in-run-stage", owner="#268",
         gate="ZERO build: gt_n600_sR.npz READY; A/B on the fragile annulus band",
         justification="MEASURED: texture proxy AT CHANCE vs through-R reachability (L76); negcure RANK-2 second-exemplar-grade",
         source_anchor="L76 (LEVER-4 msal_uni through-R reachability)"),
    dict(candidate="weight_entropy_penalty_balle", status=STATUS_BUILT_NEVER_FIRED, form_class="regularizer-schedule",
         dsl_lever="WeightEntropyPenaltyMLX", slot="in-run-stage", owner="#397",
         gate="fire on a rate-attack arm; MEASURED −19.6% bytes (torch ancestor — ancestor-rule: number does NOT transfer, mechanism does)",
         justification="FEED-reactivation-397 T-397-1 (built-never-fired factory)",
         source_anchor="DAG FEED-reactivation-397 T-397-1", est_delta_s=None, axis="rate"),
    dict(candidate="laguerre_ot_head_offset", status=STATUS_BUILT_NEVER_FIRED, form_class="discrete-solve-interleave",
         dsl_lever="HeadOffsetSolver", slot="terminal-band", owner="#397",
         gate="fire as a terminal-band solve rung after gradient descent bottoms",
         justification="deep-math #284 (argmax = Laguerre power diagram, L-v8)",
         source_anchor="DAG FEED-reactivation-397 T-397-1 + #284"),
    dict(candidate="steik_normalized_316", status=STATUS_BUILT_NEVER_FIRED, form_class="regularizer-schedule",
         dsl_na_reason="eikonal sub-knobs compiled via EikonalViscosity-family config, not a bare-name lever (fold owed if promoted)",
         slot="in-run-stage", owner="#316",
         gate="#316 FAIR eikonal-viscosity test first (operator-GO flagged; D1 known-tainted era)",
         justification="MEASURED: RAW StEik NO-GO n24 (self-amplifying 575×-1431×, verdict_scope: formulation); normalized = named follow-up (FEED-05v)",
         source_anchor="DAG FEED-05v"),
    dict(candidate="viscoreg_eps_continuation_316", status=STATUS_BUILT_NEVER_FIRED, form_class="regularizer-schedule",
         dsl_lever="EikonalViscosity", slot="in-run-stage", owner="#316",
         gate="#316 fair test (first non-confounded measurement; re-grades D1/D2/D7)",
         justification="MEASURED n24: ε=0.3 STABLE + d_seg 2.3× better than control; ε=1.0 explodes (two-sided window; FEED-05v)",
         source_anchor="DAG FEED-05v", est_delta_s=None, axis="d_seg"),
    dict(candidate="per_param_grad_normalize", status=STATUS_BUILT_NEVER_FIRED, form_class="optimizer-stage",
         dsl_na_reason="stability-preset surface (autoconfig resolve_stability_config), not a swept DSL lever yet",
         slot="in-run-stage", owner="—",
         gate="OWED trajectory A/B (alters seg-vs-pose gradient scale ratio — documented caveat)",
         justification="MEASURED-built FEED-collapsefix (byte-identical default; candidate better PRIMARY than global clip for batch=1)",
         source_anchor="DAG FEED-collapsefix"),
    dict(candidate="muon_event_derived_switch", status=STATUS_BUILT_NEVER_FIRED, form_class="optimizer-stage",
         dsl_na_reason="--muon-start-event flag exists (unmapped sub-knob of the event scaffold); fold owed if promoted",
         slot="in-run-stage", owner="#302",
         gate="fire when conditioning event (σ_min/curvature) trips AND nucleation done (#302 §3c)",
         justification="DERIVED (#302: fixed-726 = un-derived knee-transfer residual; event scaffold present)",
         source_anchor="witness_native_schedule_derivation_20260709.md §3c"),
    dict(candidate="length_sigma_matrix_rung1b", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="LengthSigma", slot="in-run-stage", owner="ladder",
         gate="pinned rung 1b: fires at the tau boundary BEFORE any other length-touching lever",
         justification="MEASURED build (fitted-20260707 inherited then deliberately reverted to σ≡1 to keep Class-A attribution clean)",
         source_anchor="SPEC_v752 §A.8/§1b"),
    dict(candidate="chroma_rung_752", status=STATUS_BUILT_NEVER_FIRED, form_class="loss-geometry",
         dsl_lever="SegChromaBoundary", slot="in-run-stage", owner="ladder",
         gate="rung fires per FEED-chroma-rung registration (its own increment A/B); SegNet reads RGB → chroma is a d_seg actuator",
         justification="registered-off rung (CLAUDE.md §Chroma: any verdict ignoring chroma is provisional)",
         source_anchor="DAG FEED-chroma-rung", est_delta_s=None, axis="d_seg"),

    # ── §3 NEEDS-BUILD (design/task exists, no code on THIS vehicle) ──────────────────────────────
    dict(candidate="swa_tail_soup", status=STATUS_NEEDS_BUILD, form_class="averaging",
         dsl_na_reason="checkpoint-space op (byte-close-side tool + trainer export hook), not a flag; unbuilt",
         slot="terminal-band", owner="NEW (follow-up task)",
         gate="design-memo-first; A/B vs PolyakFinisher (built sibling) + EMA; EMA never replaced",
         justification="DERIVED: turnpike cycles produce K basin-endpoint iterates; cross-cycle soup is the unbuilt complement (grep: zero swa/soup code in tree)",
         source_anchor="pool row §3.1 (grep MEASURED: no swa/soup code)"),
    dict(candidate="fisher_gn_head_full_p_solve", status=STATUS_NEEDS_BUILD, form_class="preconditioning",
         dsl_na_reason="in-trainer CG solve block, not a flag; unbuilt",
         slot="in-run-stage", owner="#341-adjacent (follow-up task)",
         gate="JOINS #341: quadratic head chart CONFIRMED (LM ρ 0.847/0.868) but K=8 subset-solve overfits (+5.1% net) → ONLY full-P in-trainer GPU solve admissible (~11min/CG-iter@17×, L77)",
         justification="MEASURED foundation: margin=Fisher ρ0.978 (L1) + eq quadratic_head_chart_subset_solve_gap_v1",
         source_anchor="canonical_equations.quadratic_head_chart_subset_solve_gap_v1"),
    dict(candidate="in_run_click_interleave", status=STATUS_NEEDS_BUILD, form_class="discrete-solve-interleave",
         dsl_na_reason="mc_finisher is a byte-close-side tool (#400 diagonal); in-run interleave needs a trainer-side design; unbuilt",
         slot="terminal-band → in-run-stage", owner="#400 (+ follow-up task)",
         gate="promotion gate: #400 diagonal MEASUREMENT first (terminal-band); only then design the in-run interleave",
         justification="MEASURED: n8 click row MOVED the pointer −1.7e-5 (FEED-pointer-move-n8click); n600 sweep in-flight",
         source_anchor="DAG FEED-pointer-move-n8click"),
    dict(candidate="iga_boundary_tangent_ntk_309", status=STATUS_NEEDS_BUILD, form_class="preconditioning",
         dsl_na_reason="NTK-level preconditioning, not a flag; unbuilt",
         slot="in-run-stage", owner="#309",
         gate="HELD-WEAKENED: owed-16 v1 basis ≈0 zero-shot AND owed16v2 REBALANCE no-benefit (FEED-legdisposition-owed16v2) → re-derive whether the NTK mechanism survives before building",
         justification="MEASURED 3.2× along-tangent deficit (L65) stands, but two basis-level cures measured ≈0 — negative↔cure join points AWAY",
         source_anchor="DAG FEED-legdisposition-owed16v2"),
    dict(candidate="sam_flat_minima_mdl_242", status=STATUS_NEEDS_BUILD, form_class="regularizer-schedule",
         dsl_na_reason="SAM pre-quant stage, not a flag; unbuilt",
         slot="in-run-stage", owner="#242",
         gate="design-memo-first; adjacency: WeightEntropyPenaltyMLX (§2.7) covers the rate-in-loss half — fire that FIRST, build SAM only if the entropy penalty measures insufficient",
         justification="DERIVED (compression-as-intelligence; task #242 pending)",
         source_anchor="task #242 (pending)"),
    dict(candidate="grids_bulk_inr_annulus_308", status=STATUS_NEEDS_BUILD, form_class="architecture-growth",
         dsl_na_reason="vehicle-level (grids+INR hybrid), not a flag; unbuilt",
         slot="next-vehicle", owner="#308",
         gate="negcure HELD (no matched MEASURED violated fact); v8 per-class carriers partially embody it — re-scope AGAINST v8 increment-1 before building",
         justification="DERIVED (NeurIPS'25 grids-beat-INRs-except-boundary; matched-bytes protocol pre-registered)",
         source_anchor="task #308 (pending)"),
    dict(candidate="post_muon_sgld_217", status=STATUS_NEEDS_BUILD, form_class="optimizer-stage",
         dsl_na_reason="SGLD sub-stage, not a flag; unbuilt (grep MEASURED: zero SGLD code in trainer)",
         slot="terminal-band", owner="#217/#216",
         gate="GATED on #216 saddle-to-saddle signature test ($0, run first); components (i) Damian reweight ≈ hardness/margin-saliency (§2.1/§2.6 partially cover)",
         justification="DERIVED (MFLD multi-index leap theory; Muon≈Stiefel so it applies exactly)",
         source_anchor="pool row §3.7 (grep: zero SGLD code)"),
    dict(candidate="kd_warm_start_129", status=STATUS_NEEDS_BUILD, form_class="init-warm-start",
         dsl_na_reason="production actuator spanning export, not a flag; PARTIALLY built (FiLM-v2 trunk-decoupling DONE 867ff3af5; actuator pending)",
         slot="next-vehicle", owner="#129",
         gate="bind-all spec production_readiness_bind_all_ingredients_20260616.md; #301 banked it as the 3rd rung (KD-from-#205-teacher on island band)",
         justification="DERIVED (Hinton KD; production linchpin)",
         source_anchor="commit 867ff3af5 (FiLM-v2 trunk-decoupling; ∂d_seg/∂pose=0 proven)"),
    dict(candidate="simpletes_k_gt1_319", status=STATUS_NEEDS_BUILD, form_class="discrete-solve-interleave",
         dsl_na_reason="shadow-controller shape (campaign layer), not a trainer flag; unbuilt (grep: not in witness_control)",
         slot="in-run-stage (advisory layer)", owner="#319/#315",
         gate="GATED behind #315 + BINDING backtest against v1-v5+#205 logs before adoption; fire when through-R band spans 0 at n=3",
         justification="DERIVED (SimpleTES DF-1 split-verdict; costate core NOT-RELEVANT refused)",
         source_anchor="pool row §3.9 (grep: not in witness_control)"),
    dict(candidate="md_decoupling_195", status=STATUS_NEEDS_BUILD, form_class="optimizer-stage",
         dsl_na_reason="no --optimizer/--md-base in the levelset trainer = TRAINER-GAP; unbuilt on this vehicle",
         slot="in-run-stage", owner="#195",
         gate="build the optimizer flag, then LR-transfer A/B per #195",
         justification="DERIVED (MD stable-by-construction claim is UNVERIFIED on this vehicle — that is the point of the A/B); 5-LENS CONTRADICT row",
         source_anchor="pool row §3.10 (5-LENS trainer-gap)"),
    dict(candidate="pose_inverse_carrier_distill", status=STATUS_NEEDS_BUILD, form_class="pose-carrier",
         dsl_na_reason="decoder-native generator + offline distillation, not a flag; unbuilt (research_only)",
         slot="terminal-band / next-vehicle", owner="#248/#366",
         gate="ADVISORY only (research_only=true): offline PoseNet discovery/distill of frame-0 corrections; archive must decode WITHOUT PoseNet/scorer/GT tables; gate = #248 P-B FiLM read-back decisive first",
         justification="DERIVED (ADVISORY_sdf_pose_inverse_carrier_20260710: unconstrained frame-0 inverse proves evaluator admits accurate witnesses)",
         source_anchor=".omx/research/ADVISORY_sdf_pose_inverse_carrier_20260710.md"),
    dict(candidate="som_organized_codebooks", status=STATUS_NEEDS_BUILD, form_class="state-evolution",
         dsl_na_reason="design-banked representation-side codebook, not a flag; unbuilt",
         slot="terminal-band / next-vehicle", owner="NEW (design-banked)",
         gate="EXACT-GATED A/B ONLY (representation-side chart levers repeatedly MEASURED ≈0 realized through R); pays twice IF measured: click-polish ±1/±2 locality + temporal-delta rate",
         justification="DERIVED: SOM magnification law under-allocates rare regions = the measured lane starvation; the conscience CURE already embodied by LADDER per-class λ + #218 — codebook is the only NEW piece",
         source_anchor=".omx/research/papers_checked_kohonen_som_20260710.md item 4"),

    # ── §4 REFORMULATION-QUEUE (prior formulation measured NO-GO; named reformulation owed) ───────
    dict(candidate="nca_lppn_state_146", status=STATUS_REFORMULATION_QUEUE, form_class="state-evolution",
         dsl_na_reason="vehicle-level NCA/LPPN stage, not a flag; unbuilt",
         slot="next-vehicle", owner="#146/P10",
         gate="fires ONLY on AMBER/P10 reactivation; reformulation = coarse-NCA-grid + our trunk as the LPPN (the split #146's arm lacked)",
         justification="MEASURED wall stands (#146 33K-rule generalization gate); reformulation named in papers_checked",
         source_anchor=".omx/research/papers_checked_cells2pixels_nca_lppn_20260710.md"),
    dict(candidate="curvelet_from_scratch_trajectory", status=STATUS_REFORMULATION_QUEUE, form_class="init-warm-start",
         dsl_na_reason="trajectory-shaping curvelet init, not a flag; re-derivation owed before build",
         slot="in-run-stage", owner="#397",
         gate="owed16v2 measured no-benefit ⇒ confound resolved AGAINST the axis; gate = a fresh derivation must name a mechanism that survives BOTH measured negatives before any build",
         justification="MEASURED: naive-palette realized ceiling F=0.0337 ≫ directional-direct 0.0037; trained trunk redundant with basis |Δ|≤1.4% (FEED-reactivation-397 FIRE-1)",
         source_anchor="DAG FEED-reactivation-397 FIRE-1"),

    # ── §5 EXPLICITLY EXCLUDED (retired-with-reason — do not re-propose) ──────────────────────────
    dict(candidate="gradnorm_per_step_balancing", status=STATUS_RETIRED, form_class="optimizer-stage",
         dsl_na_reason="per-step loss balancing FORBIDDEN by law (not a lever)",
         slot="—", owner="#312",
         gate="RETIRED: #312 law — naive GradNorm would DOWN-weight the eikonal CANARY mid-runaway; loss weights move at STAGE BOUNDARIES ONLY (assert_loss_weights_stage_boundary_only)",
         justification="MEASURED law (FEED-05r §4); sanctioned form = the #312 stage-boundary gradient-share checkpoint probe (task completed)",
         source_anchor="DAG FEED-05r §4 + SPEC_v75 §8-C"),
    dict(candidate="mod_dim_as_capacity_reopen", status=STATUS_RETIRED, form_class="architecture-growth",
         dsl_na_reason="mod-dim-as-capacity re-open (not a lever); CLOSED",
         slot="—", owner="#299",
         gate="RETIRED: #299 CLOSED (verdict_scope: formulation) — refuted by #300's measured island-gradient-starvation mechanism",
         justification="MEASURED (FEED-reactivation-397)",
         source_anchor="DAG FEED-reactivation-397 (#299/#300)"),
    dict(candidate="raw_gradient_steik", status=STATUS_RETIRED, form_class="regularizer-schedule",
         dsl_na_reason="raw-gradient StEik (not a lever); fire the NORMALIZED variant (§2.9) instead",
         slot="—", owner="#316",
         gate="RETIRED: MEASURED NO-GO n24 (self-amplifying 575×–1431×); the normalized variant is steik_normalized_316 — fire THAT, never the raw form",
         justification="MEASURED (FEED-05v)",
         source_anchor="DAG FEED-05v"),
)


def _seed_map() -> dict[str, dict]:
    """``{candidate: full-row}`` from the committed :data:`_SEED` inventory — the READ BASELINE for the
    default store path. Each seed dict is normalized to the full row shape (missing optional fields
    defaulted) so consumers read it identically to a recorded row. Pure; deterministic."""
    out: dict[str, dict] = {}
    for row in _SEED:
        out[row["candidate"]] = {
            "candidate": row["candidate"],
            "status": row["status"],
            "form_class": row.get("form_class", ""),
            "source_anchor": row.get("source_anchor", ""),
            "gate": row.get("gate", ""),
            "justification": row.get("justification", ""),
            "dsl_lever": row.get("dsl_lever"),
            "dsl_na_reason": row.get("dsl_na_reason"),
            "slot": row.get("slot", ""),
            "owner": row.get("owner", ""),
            "est_delta_s": row.get("est_delta_s"),
            "axis": row.get("axis"),
            "verdict_ref": row.get("verdict_ref"),
            "agent": "seed",
            "ts": "seed",
        }
    return out


def seed_default_pool(path: Path | None = None, *, agent: str = "task_403_seed") -> int:
    """Seed the store from :data:`_SEED` — writes ONLY candidates not already present (idempotent by
    ``candidate`` key, so re-running never duplicates a row). Returns the number of rows written.

    This is what makes a fresh checkout's pool non-empty (the durable inventory the memo re-derived);
    a later real event (a fire/measure/retire) is recorded via :func:`record_candidate` and wins on
    read (latest-wins). NO-FAKE: seeds carry the memo's HONEST status per row (needs-build for unbuilt,
    built-never-fired for built-never-launched — never measured)."""
    existing = set(_read_pool(path).keys())
    written = 0
    for row in _SEED:
        if row["candidate"] in existing:
            continue
        record_candidate(**{**row, "agent": agent}, path=path)
        written += 1
    return written


__all__ = [
    "POOL_PATH",
    "STATUS_ARMED",
    "STATUS_BUILT_NEVER_FIRED",
    "STATUS_MEASURED",
    "STATUS_NEEDS_BUILD",
    "STATUS_REFORMULATION_QUEUE",
    "STATUS_RETIRED",
    "VALID_AXES",
    "VALID_FORM_CLASSES",
    "VALID_STATUSES",
    "CandidateStatus",
    "candidate_status",
    "duty_to_measure_pool",
    "pool_report",
    "pool_summary",
    "record_candidate",
    "seed_default_pool",
]
