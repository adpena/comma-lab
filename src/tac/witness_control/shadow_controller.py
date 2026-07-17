"""The θ* costate SHADOW controller (task #303 Phase A): observe → estimate → recommend → STOP.

Reads a witness run directory's ``run.log`` (the trainer's real JSONL telemetry),
estimates the estimable costates with honestly-propagated uncertainty
(``tac.witness_control.costate_estimator``), classifies the trajectory through the
CANONICAL monitor classifier (``tools/witness_control_monitor.classify_trajectory`` —
imported, never forked; same math the trainer's in-run closed loop replicates), and
emits ranked recommendation rows to ``<run_dir>/costate_shadow.jsonl``.

Every recommendation carries its evidence chain (which measured rows/probes produced
the costate behind it — Rudin readback, no silent scores) and a predicted-ΔS band
(ΔS < 0 = improvement). The NEVER-REGRESS guard (POWERPLAY frontier-selection) is
enforced by construction at the recommendation layer: any candidate whose central
predicted ΔS is > 0 (would RAISE S) is REFUSED into the ``refused`` list with the
reason — it can never rank. (A neutral ΔS == 0 stays ranked but scores 0/cost, so it
sinks below every real improvement — an advisory "no-op / watch" floor, not a fire.)

CONTAINMENT (structural): this module has NO actuation capability — no subprocess,
no os.system/exec surfaces, no trainer imports. ``actuation`` is the literal string
``"NONE"`` on every row. Phase-B actuation is design-only and operator-GO-gated.

Reuse, not duplication: verdict/stage-row parsing is delegated to the existing
``tools/witness_control_monitor`` + ``tools/dashboard_control_telemetry`` modules
(the same parsers the dashboards run). The launch.sh flag parser replicates the
verbatim contract of ``tools/render_levelset_dashboard._parse_launch_sh_flags``
(that module imports matplotlib at import time, so it cannot be imported from here;
parity is regression-guarded in ``src/tac/tests/test_witness_control_costate.py``).

Craft discipline (``docs/operating_manual_craft_handoff.md``): the controller's costate
tiers (MEASURED / DERIVED-analytic / UNIDENTIFIABLE) implement the manual's §5
known-vs-guessed labeling — every recommendation states HOW its costate was obtained,
and a caveat travels with the number. Its review cadence is bound by §8.8
(round-finished ≠ clean pass: a fix round resets the counter; SEAL only on consecutive
clean rounds).
"""
from __future__ import annotations

import importlib
import itertools
import math
import sys
import warnings
from dataclasses import dataclass, field as _dc_field
from datetime import UTC, datetime
from pathlib import Path

from tac.jsonl_store import append_locked_jsonl
from tac.causal_manifest import (
    MANIFEST_FILENAME as CAUSAL_MANIFEST_FILENAME,
    ApparatusState as CausalApparatusState,
    ArmPropensity,
    CausalManifestWriter,
    ExplorationDecisionRow,
    RealizedOutcome as CausalRealizedOutcome,
    StateSummary as CausalStateSummary,
    boundary_sequence_index as causal_boundary_sequence_index,
    canonical_sha256 as causal_sha256,
)
from tac.witness_control.costate_estimator import (
    BINDING_TERM_STALL,
    MEASURED,
    UNIDENTIFIABLE,
    CostateEstimate,
    analytic_costates,
    binding_term_stall,
    chain_ds_depoch,
    per_class_within_flip_costates,
    rollback_gain,
    stage_epoch_costates,
    transition_jump_costate,
)
from tac.witness_control.verdict_trend_alarm import (
    TRAIN_VERDICT_DECOUPLING,
    verdict_trend_alarm,
)
from tac.witness_control.label_floor_detector import (
    LABEL_FLOOR_REACHED,
    _is_phase_tail_lever,
    label_floor_reached,
)

#: overlay classification when the verdict-trend alarm fires on a scalar false-green
#: (the operator-catch 2026-07-09 class): CONVERGING is FORBIDDEN when the advisory
#: verdict d_seg is materially RISING — it is DRIFTING/decoupling, not converging.
VERDICT_RISING_DECOUPLING = "verdict_rising_decoupling"
#: scalar classes that are FALSE GREENS while the verdict is rising (must be downgraded)
_FALSE_GREEN_CLASSES = frozenset({"converging", "plateau"})

_REPO = Path(__file__).resolve().parents[3]

AXIS_TAG = "[macOS advisory] NON-PROMOTABLE"
POINTER_NOTE = "pointer 0.19110 UNMOVED"
ACTUATION = "NONE"   # Phase A invariant — never anything else in this package

#: default horizon (epochs) over which per-epoch costates are projected into a ΔS band
DEFAULT_HORIZON_EPOCHS = 25

# ─── ΔS-PER-COST ranking (task #247) — the missing per-cost divisor ───
# The costate controller ranks recommendations by predicted ΔS. But a big raw ΔS
# that costs a whole horizon of GPU is NOT the same bang-for-buck as a smaller ΔS
# from a light $0 control move. #247's costate control law ranks by ΔS / COST, so
# the cheapest-biggest-drop wins. Costs are ENACTMENT effort (compute/epochs), pure
# and testable; the never-regress refusal still gates on the RAW ΔS (cost is
# strictly positive → it can never flip a sign).
#: default cost (compute/effort units) when a candidate carries no known cost.
DEFAULT_ACTION_COST = 1.0
#: unit baseline for a light $0 control move (rollback / stop / watch / investigate).
_LIGHT_ACTION_COST = 1.0
#: actions that spend real multi-epoch training to realize their predicted ΔS — their
#: cost scales with the horizon (many epochs of GPU), so a big raw drop that costs a
#: whole horizon can be out-ranked per-cost by a cheap light move.
_HEAVY_MULTI_EPOCH_ACTIONS = frozenset({"CONTINUE_STAGE"})
#: epsilon guard so the per-cost divisor never divides by ~0.
COST_EPSILON = 1e-9


def per_cost_score(predicted_dS: float, cost: float) -> float:
    """ΔS-per-cost = ``predicted_dS / max(cost, COST_EPSILON)`` (task #247).

    Most-negative = best bang-for-buck. Cost is strictly positive by construction
    (see ``candidate_cost``); the epsilon guard is belt-and-suspenders against a
    caller-supplied ~0 cost (never a divide-by-zero). Sign-preserving: a positive
    ΔS divided by a positive cost stays positive, so the never-regress refusal is
    unchanged whether it gates on the raw ΔS or the per-cost score."""
    return predicted_dS / max(cost, COST_EPSILON)


def candidate_cost(candidate: dict, horizon_epochs: int = DEFAULT_HORIZON_EPOCHS) -> float:
    """Pure, testable ENACTMENT-cost estimate (compute/effort units) for a candidate.

    Falling-priority source (Rudin-style readback, no hidden model): (1) an explicit
    positive ``candidate["cost"]`` field; (2) the per-action-KIND cost model — a HEAVY
    multi-epoch action (``CONTINUE_STAGE`` spends ``horizon_epochs`` of real training)
    costs its horizon, while a light $0 control move (rollback / stop / watch /
    investigate / widen) costs the unit baseline; (3) ``DEFAULT_ACTION_COST`` when the
    action is unknown/absent. Always returns a strictly positive value."""
    c = candidate.get("cost")
    if isinstance(c, (int, float)) and c > 0.0:
        return float(c)
    action = candidate.get("action")
    if action in _HEAVY_MULTI_EPOCH_ACTIONS:
        h = candidate.get("horizon_epochs")
        h = float(h) if isinstance(h, (int, float)) and h > 0.0 else float(horizon_epochs)
        return max(h, _LIGHT_ACTION_COST)
    if action:
        return _LIGHT_ACTION_COST
    return DEFAULT_ACTION_COST


def _load_tools_module(name: str):
    """Import a module from ``tools/`` (the dashboards' own pattern — they sys.path
    the tools dir; we mirror it so the canonical classifier/parsers are never forked)."""
    tools_dir = str(_REPO / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return importlib.import_module(name)


def parse_launch_sh_flags(text: str) -> dict:
    """Verbatim-contract replica of ``tools/render_levelset_dashboard._parse_launch_sh_flags``
    (parity regression-guarded; see module docstring for why it is not imported).

    ``--flag value`` pairs from a launch.sh trainer invocation; ``--flag`` followed by
    another flag/EOL is boolean True; ``--flag=value`` splits; env assignments and
    non-flag tokens are ignored; never raises."""
    toks: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.endswith("\\"):
            s = s[:-1].strip()
        if not s or s.startswith("#"):
            continue
        toks.extend(t for t in s.split() if t)
    flags: dict = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                flags[k] = v
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[key] = toks[i + 1]
                i += 1
            else:
                flags[key] = True
        i += 1
    return flags


# ─────────────────────────── run inputs ───────────────────────────
@dataclass
class RunInputs:
    """Everything the shadow controller reads from a run dir (read-only, CPU-light)."""

    run_dir: Path
    verdicts: list[dict]
    stage_rows: dict          # dashboard_control_telemetry.parse_stage_rows output
    flags: dict
    as_of_epoch: int | None = None


def _truncate_as_of(rows: list[dict], as_of: int | None) -> list[dict]:
    if as_of is None:
        return rows
    return [r for r in rows if isinstance(r.get("epoch"), (int, float))
            and float(r["epoch"]) <= as_of]


def load_run_inputs(run_dir: str | Path, as_of_epoch: int | None = None) -> RunInputs:
    """Read run.log + launch.sh (read-only). ``as_of_epoch`` truncates every
    epoch-carrying row for backtests ("what would the controller have said at ep N")."""
    run_dir = Path(run_dir)
    wcm = _load_tools_module("witness_control_monitor")
    dct = _load_tools_module("dashboard_control_telemetry")
    log = run_dir / "run.log"
    verdicts: list[dict] = []
    lines: list[str] = []
    if log.is_file():
        verdicts = wcm._read_verdicts(log)
        lines = log.read_text(errors="replace").splitlines()
    stage_rows = dct.parse_stage_rows(lines)
    flags: dict = {}
    launch = run_dir / "launch.sh"
    if launch.is_file():
        flags = parse_launch_sh_flags(launch.read_text(errors="replace"))
    verdicts = _truncate_as_of(verdicts, as_of_epoch)
    stage_rows = dict(stage_rows)
    stage_rows["transitions"] = _truncate_as_of(stage_rows.get("transitions") or [],
                                                as_of_epoch)
    stage_rows["closed_loop"] = _truncate_as_of(stage_rows.get("closed_loop") or [],
                                                as_of_epoch)
    return RunInputs(run_dir=run_dir, verdicts=verdicts, stage_rows=stage_rows,
                     flags=flags, as_of_epoch=as_of_epoch)


# ─────────────────────────── the shadow report ───────────────────────────
@dataclass
class ShadowReport:
    run_dir: str
    as_of_epoch: int | None
    epoch_latest: int | None
    state: dict
    costates: list[CostateEstimate]
    classification: dict | None
    recommendations: list[dict]
    refused: list[dict]
    probe_queue: list[dict]
    # #247 SENSE→DECIDE: the never-fired DSL levers owed a measurement (the activation-ledger queue
    # the controller SURFACES so "off" is never a forgotten default). These carry NO predicted ΔS — a
    # never-fired lever has no MEASURED costate (NO-FAKE); they enter the ΔS-per-cost ranking ONLY after
    # they fire + are measured. Ordered never-fired-first. Default empty (backward-compatible).
    duty_to_measure: list[dict] = _dc_field(default_factory=list)
    # #247 de-orphaning: every previously-orphaned producer (sensitivity-map per-axis EV weights,
    # master-gradient per-byte anchor, cathedral-autopilot ranker) read into the ONE controller's
    # SENSE. Each row is available-with-real-signal OR available=False with an honest reason — NEVER
    # a fabricated value (NO-FAKE). Default empty (backward-compatible).
    producer_signals: list[dict] = _dc_field(default_factory=list)
    # #247 continual-learning: the cross-run costate POSTERIOR (inverse-variance combination of every
    # PAST run's MEASURED costates). This is what makes session N+1 smarter than N — the live controller
    # SEES what earlier runs measured for each lever. READ-only + advisory; default empty.
    costate_prior: list[dict] = _dc_field(default_factory=list)
    # #247 EIG-bridge: the owed (never-fired) levers RANKED by measurement cost ascending — the honest
    # EIG-per-cost ordering under an uninformative prior (cheapest owed lever first). This is the DECIDE
    # ordering of the duty-to-measure queue (invoked, not just listed). NO fabricated ΔS.
    duty_ranked: list[dict] = _dc_field(default_factory=list)
    # Task #516: the exact-factorized adjoint is part of THIS consumed shadow organ,
    # not a parallel digest-only controller.  It carries the full backtest gate and
    # exact/derived/learned provenance even when admission fails.
    factorized_adjoint: dict | None = None
    # Morse-Smale + #344 NCDE warnings are stage-boundary advisories only.  They never
    # mutate a schedule and never manufacture a ΔS.
    event_advisories: list[dict] = _dc_field(default_factory=list)
    # Task #522: the on-device FM (fmtools) ADVISORY sense layer — {regime supplement (fm_regime
    # + agreement-with-numeric), event intelligence}. Populated ONLY when the fmtools venv is
    # present AND build_shadow_report(with_fm_advisory=True); default None ⇒ omitted from to_row
    # ⇒ byte-identical schema when absent.  ADVISORY ONLY: never feeds a verdict/actuation/score.
    fm_advisory: dict | None = None

    def to_row(self) -> dict:
        return {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_dir": self.run_dir,
            "as_of_epoch": self.as_of_epoch,
            "epoch": self.epoch_latest,
            "state": self.state,
            "costates": [c.to_dict() for c in self.costates],
            "classification": self.classification,
            "recommendations": self.recommendations,
            "refused": self.refused,
            "probe_queue": self.probe_queue,
            "duty_to_measure": self.duty_to_measure,
            "producer_signals": self.producer_signals,
            "costate_prior": self.costate_prior,
            "duty_ranked": self.duty_ranked,
            "factorized_adjoint": self.factorized_adjoint,
            "event_advisories": self.event_advisories,
            "actuation": ACTUATION,
            "axis": AXIS_TAG,
            "pointer": POINTER_NOTE,
            # additive #522 FM advisory: included ONLY when populated (byte-identical when absent).
            **({"fm_advisory": self.fm_advisory} if self.fm_advisory is not None else {}),
        }


def _state_snapshot(inputs: RunInputs) -> dict:
    rows = [v for v in inputs.verdicts if isinstance(v.get("d_seg"), (int, float))]
    if not rows:
        return {"n_verdicts": 0}
    latest = rows[-1]
    best = min(rows, key=lambda v: float(v["d_seg"]))
    stages = []
    for v in rows:
        sf = v.get("seg_form")
        if sf and sf not in stages:
            stages.append(sf)
    lever_keys = ("lane-prior-phi1-mode", "seed-islands", "eikonal-weight",
                  "eikonal-weight-end", "mod-dim", "closed-loop-control",
                  "curriculum-event-triggered", "muon-start-epoch", "l7-start-epoch",
                  "tau-softplus-start-epoch")
    return {
        "n_verdicts": len(rows),
        "epoch": latest.get("epoch"),
        "stage": latest.get("seg_form"),
        "d_seg": latest.get("d_seg"),
        "d_pose": latest.get("d_pose"),
        "blob_bytes": latest.get("blob_bytes"),
        "implied_S": latest.get("implied_S"),
        "best_d_seg": best.get("d_seg"),
        "best_epoch": best.get("epoch"),
        "stages_seen": stages,
        "transitions": inputs.stage_rows.get("transitions") or [],
        "levers": {k: inputs.flags.get(k) for k in lever_keys if k in inputs.flags},
    }


def _classify(inputs: RunInputs) -> dict | None:
    wcm = _load_tools_module("witness_control_monitor")
    usable = [v for v in inputs.verdicts
              if v.get("seg_form") and isinstance(v.get("d_seg"), (int, float))]
    if len(usable) < 2:
        return None
    try:
        cv = wcm.classify_trajectory(usable)
    except (ValueError, KeyError):
        return None
    out = {
        "classification": cv.classification, "stage": cv.stage,
        "epoch_latest": cv.epoch_latest, "d_seg_latest": cv.d_seg_latest,
        "d_seg_slope_per_ep": cv.d_seg_slope_per_ep,
        "ep_loss_slope_per_ep": cv.ep_loss_slope_per_ep,
        "recommendation": cv.recommendation,
        "config_diffs": list(cv.config_diffs),
        "source": "tools/witness_control_monitor.classify_trajectory (canonical, unforked)",
    }
    # (task #315) BINDING-TERM-STALL OVERLAY — the scalar classifier reads d_seg
    # ALONE, so a FLAT binding term reads as PLATEAU/CONVERGING (advance/early-stop
    # = "fine"). But if implied_S / ep_loss are still MOVING while d_seg is frozen,
    # the run is DEADLOCKED on the term that matters (the v5 frozen-descending-S the
    # scalar monitor missed live). This overlay OVERRIDES the false green.
    bs = binding_term_stall(inputs.verdicts)
    out["binding_stall"] = bs.to_dict()
    if bs.fired():
        out["scalar_classification"] = cv.classification          # preserve what the scalar said
        out["classification"] = BINDING_TERM_STALL                # the overlay wins
        out["recommendation"] = ("BINDING-TERM STALL (task #315): " + bs.reason
                                 + " [scalar monitor said "
                                 + f"'{cv.classification}': {cv.recommendation}]")
    # (operator-catch 2026-07-09) VERDICT-TREND / TRAIN-VERDICT-DECOUPLING OVERLAY —
    # the scalar classifier fits a least-squares d_seg slope over a window that still
    # spans the early CE drop, so a NET-DOWN-but-recently-RISING binding term reads as
    # CONVERGING ("healthy descent") — the false green the operator caught. This overlay
    # reads the recent within-stage RISE (materially above the calibrated flat gate) and
    # FORBIDS a 'converging'/'plateau' green when the advisory verdict d_seg is rising.
    vt = verdict_trend_alarm(inputs.verdicts)
    out["verdict_trend_alarm"] = vt.to_dict()
    if vt.fired() and out["classification"] in _FALSE_GREEN_CLASSES:
        out.setdefault("scalar_classification", cv.classification)
        out["classification"] = VERDICT_RISING_DECOUPLING          # rising is NOT converging
        out["recommendation"] = (
            ("TRAIN<->VERDICT DECOUPLING" if vt.classification == TRAIN_VERDICT_DECOUPLING
             else "RISING VERDICT") + " (operator-catch 2026-07-09): " + vt.reason
            + f" [scalar monitor said '{cv.classification}': {cv.recommendation}]")
    # (task #247/#303 phase-reframe fold, 2026-07-10) LABEL-FLOOR / PHASE-REGIME SENSE.
    # ORTHOGONAL to the false-green overlays above (a healthy transition, not a confound):
    # when a label-smooth witness has settled at the temporal-majority persistence floor
    # (d_seg ~0.0053, |slope|~0), label descent is EXHAUSTED — the remaining 4.5-7x descent
    # is APPEARANCE-PHASE (flicker memo §2.5). This is a REGIME TAG the DECIDE layer keys on
    # to recommend the phase-tail hand-off (T1 + #360 + T2); it does NOT override the
    # false-green classes (which win when a real decoupling/stall is also present). It DOES
    # suppress a naive plateau early-stop at the floor (that would abandon the sub-floor path).
    lf = label_floor_reached(inputs.verdicts)
    out["label_floor"] = lf.to_dict()
    out["phase_regime"] = lf.phase_regime   # LABEL_FLOOR_REACHED or None
    return out


def _band_from(cost: CostateEstimate, scale: float) -> tuple[float, list[float] | None]:
    """Scale a per-epoch costate into a horizon ΔS with its band."""
    central = (cost.value or 0.0) * scale
    b = cost.band()
    if b is None:
        return central, None
    lo, hi = sorted((b[0] * scale, b[1] * scale))
    return central, [lo, hi]


def _recommendations(inputs: RunInputs, costates: list[CostateEstimate],
                     classification: dict | None,
                     horizon_epochs: int = DEFAULT_HORIZON_EPOCHS
                     ) -> tuple[list[dict], list[dict]]:
    """Build ranked recommendations from IDENTIFIABLE costates only, then apply the
    NEVER-REGRESS guard: a candidate whose central predicted ΔS > 0 is refused by
    construction (POWERPLAY: only frontier-improving moves may rank). Survivors are
    ranked by ΔS-PER-COST (task #247): ``predicted_dS / max(cost, COST_EPSILON)``,
    most-negative first, so the cheapest-biggest-drop wins (a light $0 control move
    can out-rank a heavy multi-epoch move whose raw drop is larger). Every row exposes
    its ``cost`` + ``predicted_dS_per_cost`` (Rudin-interpretable, no hidden weighting)."""
    by_name = {c.name: c for c in costates}
    stage = (classification or {}).get("stage") or ""
    ds_dep = by_name.get(f"dS_depoch[stage={stage}]")
    rollback = by_name.get("dS_rollback_to_best")
    cls = (classification or {}).get("classification")
    phase_regime = (classification or {}).get("phase_regime")   # LABEL_FLOOR_REACHED or None
    candidates: list[dict] = []

    def _cand(action: str, delta: float, band: list[float] | None, rationale: str,
              evidence: tuple[str, ...], costate_name: str | None) -> None:
        candidates.append({
            "action": action, "predicted_dS": delta, "predicted_dS_band": band,
            "horizon_epochs": horizon_epochs, "rationale": rationale,
            "evidence": list(evidence), "costate": costate_name,
        })

    # (task #247/#303 phase-reframe fold, 2026-07-10) THE TWO-MOVE PHASE HAND-OFF.
    # When SENSE reports the label-floor regime, the DECIDE layer recommends the
    # curriculum's phase-tail (the flow's FINEST persistence level, where the
    # deterministic sub-pixel phase lives): DESCEND the ξ-coherent deterministic phase
    # (T1 + #360's four within-pair forces) WHILE DOWNWEIGHTING the aleatoric residue
    # (T2 = #274 spike-downweight — don't chase AA-irremovable GT-side noise). The two
    # moves have OPPOSITE signs on the same annulus tie-field (flicker memo §4). This is
    # ADVISORY only (CONTAINMENT): predicted ΔS is 0.0 (a never-fired lever has NO measured
    # costate — NO-FAKE), so the rows carry the DERIVED rationale + the coupled per-dimension
    # costate reading, NOT a fabricated drop. The two moves are ranked so T2 (BUILT, one
    # flag flip) precedes T1 (next crucible increment), per the memo sequencing.
    if phase_regime == LABEL_FLOOR_REACHED:
        lf = (classification or {}).get("label_floor") or {}
        # COUPLED per-dimension costate reading (coordinator 2026-07-10: λ = the flow's
        # Pontryagin costate; read per DIMENSION, not per-lever-independent). The phase
        # levers act on the d_seg dimension (λ_seg = 100, the largest term); λ_pose is the
        # NON-RISING guard (the moves must not raise d_pose); λ_rate = 25/37545489 is the
        # carrier dimension (#336 sets the phase-carrier's rate by sensitivity waterfill).
        lam_pose_c = by_name.get("lambda_d_pose")
        lam_pose_txt = (f"; coupled λ_pose={lam_pose_c.value:.3g} (non-rising guard)"
                        if lam_pose_c is not None and lam_pose_c.value is not None else
                        "; λ_pose non-rising guard (unidentified until d_pose series lands)")
        floor_txt = (f"d_seg {lf.get('d_seg_latest')} at the oracle floor "
                     f"{lf.get('oracle_floor')} (stage {lf.get('stage')})")
        _cand("FIRE_T2_SPIKE_DOWNWEIGHT_ALEATORIC", 0.0, None,
              "PHASE HAND-OFF move 1 of 2 (aleatoric branch, BUILT lever #274 --seg-spike-"
              "downweight, one flag flip): STOP chasing the sensor-driven single-frame spikes "
              "(~30% of annulus jitter is the 2.66-LSB sensor floor) — hard-CE to a spiked label "
              "injects a wrong-target gradient at the highest-loss pixels (44% of CE-residual = "
              "Lane spikes, L67). Derived value --seg-spike-downweight in {0.0, 0.25} (CE mass "
              "removed <= 0.53% of px, domination-safe). COUPLING: do NOT --seg-coherent-upweight "
              "(the coherent branch is owned by T1/#360 — no double-weight, #360 §5.2). Acts on "
              "the d_seg dimension (λ_seg=100)" + lam_pose_txt + ". Caps the witness at the "
              "0.0053 floor if run alone — mid-game noise-hygiene, NOT the sub-floor descent.",
              ("flicker memo §4 T2 + §2.5; " + floor_txt,
               "eq gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1 (owed-registration)",
               "duty: seg_spike_downweight (activation ledger; BUILT never-fired, L31)"),
              "phase_regime[aleatoric]")
        _cand("ENGAGE_T1_PHASE_ADVECTION_PLUS_360_FORCES", 0.0, None,
              "PHASE HAND-OFF move 2 of 2 (predictable-phase branch, the sub-floor descent path): "
              "DESCEND the deterministic ξ-coherent appearance phase — engage T1 "
              "(phase_advection_consistency: cross-pair ξ-advected sub-pixel tie-phase on the "
              "stride-2 scored sequence) + #360's four within-pair forces at the l7/sharpening "
              "boundary (the flow's FINEST curvelet/persistence scale, where the sub-pixel phase "
              "lives — that is WHY the forces engage at l7, not a hand-set boundary). This is the "
              "ONLY path below the 0.0053 label floor (existence proof: real-frame content through "
              "R d_seg 0.00086, FEED-ma; need band 0.00077-0.00118). Derived weight w_p = "
              "0.4*w_subpix (flicker fraction 0.42-0.44, blink-back 41.8%); ramp at stage "
              "boundaries only, cap <=10% total loss (under the 40% domination alarm). COUPLING: "
              "the schedule anneals TOWARD T1's ξ-coherent phase and must NOT re-open #360's "
              "satisficing hinge (margin-chasing cap); the terminal Morse-Smale skeleton IS what "
              "the phase-carrier stores. Acts on the d_seg dimension (λ_seg=100)" + lam_pose_txt +
              "; carrier rate set by #336 sensitivity waterfill (λ_rate=25/37545489).",
              ("flicker memo §4 T1 + §3 (Morse-Smale/eikonal) + §2.5",
               "coordinator 2026-07-10: curriculum = co-equal dimension of the one level-set flow",
               "duty: phase_advection_consistency + p0_force* (activation ledger, ranked HIGH)"),
              "phase_regime[predictable_phase]")

    if cls == VERDICT_RISING_DECOUPLING:
        vt = (classification or {}).get("verdict_trend_alarm") or {}
        pcs = vt.get("per_class_alarms") or []
        pc_txt = (" worst-rising class(es): "
                  + ", ".join(f"{r.get('class_name')}(+{r.get('rise_rel_value', 0) * 100:.0f}% of value)"
                              for r in pcs)) if pcs else ""
        _cand("INVESTIGATE_VERDICT_RISING_DECOUPLING", 0.0, None,
              "the advisory verdict d_seg is materially RISING over >= k consecutive "
              "verdicts while the train seg-loss descends (the operator-catch false-green): "
              "more of the same gradient is NOT lowering the argmax verdict. Do NOT "
              "advance/early-stop on the scalar 'converging/plateau' green; investigate the "
              "curriculum/stage (a decoupling, not a converged plateau)." + pc_txt,
              tuple(vt.get("evidence") or ("verdict-trend alarm (operator-catch 2026-07-09)",)),
              "verdict_trend_alarm")
    elif cls == BINDING_TERM_STALL:
        bs = (classification or {}).get("binding_stall") or {}
        _cand("INVESTIGATE_BINDING_TERM_DEADLOCK", 0.0, None,
              "the binding term d_seg is FLAT while a non-binding signal (implied_S / "
              "ep_loss) still descends — a DEADLOCK, not a converged plateau. Do NOT "
              "advance the stage or early-stop on the scalar green; check stale-weight "
              "async verdicts, spike-guard freeze, or LR collapse first",
              tuple(bs.get("evidence") or ("binding-term-stall detector (task #315)",)),
              "binding_term_stall")
    elif cls == "diverging_erasing":
        if rollback is not None and rollback.status == MEASURED and rollback.value:
            _cand("ROLLBACK_TO_BEST_CHECKPOINT", -float(rollback.value), None,
                  "sustained within-stage erosion (d_seg rising while ep_loss falls); "
                  "the BEST checkpoint is preserved — rolling back recovers the measured gap",
                  rollback.evidence, rollback.name)
        if ds_dep is not None and ds_dep.status == MEASURED and (ds_dep.value or 0) > 0:
            central, band = _band_from(ds_dep, float(horizon_epochs))
            _cand("STOP_OR_RETREAT_STAGE", -central,
                  ([-band[1], -band[0]] if band else None),
                  f"continuing this stage costs the measured creep for the next "
                  f"{horizon_epochs} epochs; stopping avoids it "
                  f"(monitor config-diffs: {(classification or {}).get('config_diffs')})",
                  ds_dep.evidence, ds_dep.name)
    elif cls == "transition_transient":
        _cand("WATCH_NO_ACTION", 0.0, None,
              "d_seg rise is RECENT after a boundary — recoverable transient vs erosion "
              "is only distinguishable by persistence (min_sustained_windows); acting now "
              "risks fighting a transient (the #205 transition-analysis rule)",
              tuple((classification or {}).get("recommendation", "").splitlines()) or
              ("canonical monitor persistence rule",), None)
    elif cls == "plateau" and not phase_regime:
        _cand("ADVANCE_STAGE_OR_EARLY_STOP", 0.0,
              (_band_from(ds_dep, float(horizon_epochs))[1] if ds_dep and ds_dep.stderr
               else None),
              "|dS/dep| ≈ 0 within stage — the Lyapunov certificate has flattened; the "
              "marginal value of more epochs here is ~0, so advance or stop",
              (ds_dep.evidence if ds_dep else ()), (ds_dep.name if ds_dep else None))
    elif cls == "plateau" and phase_regime == LABEL_FLOOR_REACHED:
        # a plateau AT the label floor is NOT an early-stop signal — the flat d_seg is the
        # persistence floor, and the sub-floor descent is the phase tail (emitted above).
        _cand("PHASE_HANDOFF_NOT_EARLY_STOP", 0.0, None,
              "|dS/dep| ≈ 0 here is the LABEL-SMOOTH PERSISTENCE FLOOR, not a converged stop: "
              "advancing/early-stopping abandons the 4.5-7x sub-floor descent that only the "
              "phase tail (T1 + #360, above) reaches. Continue into the phase regime, do NOT "
              "early-stop.",
              ("flicker memo §2.5 (floor != convergence)",
               "the phase-tail hand-off supersedes the naive plateau early-stop"),
              None)
    elif cls == "converging" and ds_dep is not None and ds_dep.status == MEASURED:
        central, band = _band_from(ds_dep, float(horizon_epochs))
        _cand("CONTINUE_STAGE", central, band,
              f"healthy descent: measured dS/dep {ds_dep.value:+.3e} projects this ΔS "
              f"over the next {horizon_epochs} epochs",
              ds_dep.evidence, ds_dep.name)
    elif cls == "volatile":
        _cand("WIDEN_WINDOW_OR_CHECK_COLLISION", 0.0, None,
              "high within-window variance — no clean slope; widen the classification "
              "window or check for a stage-transition collision before acting",
              (ds_dep.evidence if ds_dep else ()), None)

    # pose-vs-seg marginal advisory (operating-point crossover; CLAUDE.md §SegNet vs PoseNet)
    lam_pose = by_name.get("lambda_d_pose")
    fits_ev = None
    if (lam_pose is not None and lam_pose.value is not None and ds_dep is not None
            and ds_dep.status == MEASURED):
        fits_ev = (f"lambda_d_pose={lam_pose.value:.3g} at the current operating point "
                   f"(vs lambda_d_seg=100)")

    ranked, refused = [], []
    for c in candidates:
        # (task #247) attach the enactment cost + the ΔS-per-cost score BEFORE the
        # refusal branch so BOTH ranked and refused rows carry them (Rudin readback).
        cost = candidate_cost(c, horizon_epochs)
        c["cost"] = cost
        c["predicted_dS_per_cost"] = per_cost_score(c["predicted_dS"], cost)
        pds = c["predicted_dS"]
        # (review-fix) a NON-FINITE predicted ΔS (NaN/inf from a corrupted verdict-row input
        # propagating through the slope fit) passes `pds > 0.0` as False in Python and would
        # silently enter `ranked`, then destabilize the sort (NaN keys → undefined order).
        # Treat non-finite as UNIDENTIFIABLE → refused, never ranked (NO-FAKE: an unknowable ΔS
        # is not a negative-ΔS recommendation).
        if not (isinstance(pds, (int, float)) and math.isfinite(pds)):
            refused.append({**c, "refusal_reason":
                            "NON_FINITE predicted ΔS (NaN/inf from a corrupted verdict input) — "
                            "treated as UNIDENTIFIABLE and refused, never ranked"})
        elif pds > 0.0:
            refused.append({**c, "refusal_reason":
                            "NEVER_REGRESS (POWERPLAY): central predicted ΔS > 0 — a "
                            "recommendation that would raise measured S is refused by "
                            "construction"})
        else:
            ranked.append(c)
    # (task #247) rank by ΔS-PER-COST, cheapest-biggest-drop first: a light $0 control
    # move with a modest drop can now out-rank a heavy multi-epoch move with a larger
    # raw drop. The never-regress refusal above still gates on the RAW ΔS (cost is
    # strictly positive, so the divisor never flips a candidate's sign).
    ranked.sort(key=lambda c: c["predicted_dS_per_cost"])   # most-negative (best bang/buck) first
    if fits_ev:
        for c in ranked:
            c.setdefault("notes", []).append(fits_ev)
    return ranked, refused


def _probe_queue(costates: list[CostateEstimate]) -> list[dict]:
    """The honest complement of the recommendation list: every UNIDENTIFIABLE costate
    with the evidence gap that a probe would close."""
    return [{"costate": c.name, "why_unidentifiable": c.method,
             "evidence_gap": list(c.evidence)}
            for c in costates if c.status == UNIDENTIFIABLE]


def _duty_to_measure(phase_active: bool = False) -> list[dict]:
    """The #247 SENSE→DECIDE queue: DSL levers the activation ledger records as OWED a measurement
    (never-fired OR fired-but-unmeasured, not retired). Fail-safe: the ledger must NEVER break a
    shadow report, so any import/read error yields an empty queue (the report degrades to legacy).

    NO-FAKE: each row carries the ledger's activation state ONLY — no predicted ΔS (a never-fired
    lever has no measured costate). The controller surfaces these so "off" is a tracked queue the
    operator never has to remember; they enter the ΔS-per-cost DECIDE ranking after they fire+measure.

    (#247/#303 phase-reframe fold) when ``phase_active`` (SENSE reports the label-floor regime),
    the phase-tail levers (T1 + #360's forces + T2 + phase-carrier) are the sub-floor descent path,
    so they are marked ``phase_regime_high`` and floated to the FRONT of the queue (duty-to-measure
    ranking); a synthetic #336 sensitivity-bit-alloc duty row is appended (the COMPRESS half of
    train-big-compress-small: the phase-carrier's rate is set by the sensitivity waterfill, tied to
    the first witness checkpoint). All advisory — CONTAINMENT unchanged.
    """
    try:
        from tac.witness_dsl import activation_ledger as _al
        owed = set(_al.duty_to_measure())
        rows = [r for r in _al.activation_report() if r["lever"] in owed]
        for r in rows:
            r["why"] = ("registered DSL lever OWED a measurement (default-off is a tracked queue, "
                        "not a forgotten default); NO predicted ΔS until fired+measured")
            r["phase_regime_high"] = bool(phase_active and _is_phase_tail_lever(r.get("lever", "")))
        if phase_active:
            # #336 fold: the sensitivity bit-allocation is a TRAINING-TIME lever (the compress half
            # of the phase channel); tie it to the first witness checkpoint. Surfaced as a duty row
            # (NOT a fabricated ledger lever — it is a producer, not a Lever factory) with no ΔS.
            rows.append({
                "lever": "sensitivity_bit_alloc_phase_carrier (#336/#157)",
                "status": "duty_to_measure_producer", "ever_fired": False,
                "phase_regime_high": True,
                "why": ("#336 sensitivity waterfill sets the phase-carrier's rate (train-big-"
                        "compress-small); MEASURE at the first witness checkpoint — the carrier "
                        "stores the terminal Morse-Smale skeleton residual, rate not a fixed budget"),
            })
            # float phase-high rows to the front (stable: preserve intra-group order)
            rows.sort(key=lambda r: 0 if r.get("phase_regime_high") else 1)
        return rows
    except Exception:  # noqa: BLE001 — advisory sidecar must never break the report
        return []


def _producer_signals(inputs: RunInputs) -> list[dict]:
    """The #247 de-orphaning: read EVERY orphaned producer (sensitivity-map axis weights,
    master-gradient anchor, cathedral-autopilot ranker) into the ONE controller's SENSE. Fail-safe:
    the producer bridge never breaks a report; each producer contributes a real signal or an honest
    available=False reason (NO-FAKE). ``archive_sha256`` comes from the run flags if a byte-close has
    landed one (None during live training → master-gradient honestly reports no-archive)."""
    try:
        from tac.witness_control.producer_bridge import read_producer_signals
        flags = inputs.flags if isinstance(inputs.flags, dict) else {}
        sha = flags.get("archive_sha256") or flags.get("archive_sha") or None
        op = flags.get("operating_point") or "pr106_r2"
        return read_producer_signals(archive_sha256=sha, operating_point=op)
    except Exception:  # noqa: BLE001 — advisory bridge, never breaks the report
        return []


def _costate_prior() -> list[dict]:
    """The #247 continual-learning READ: the cross-run costate posterior (what past runs MEASURED per
    lever). Fail-safe: never breaks a report. Empty until at least one run's byte-close records its
    measured costates into the posterior."""
    try:
        from tac.witness_control.costate_posterior import all_posteriors
        return all_posteriors()
    except Exception:  # noqa: BLE001 — advisory memory, never breaks the report
        return []


def _duty_ranked() -> list[dict]:
    """The #247 EIG-bridge: the owed levers RANKED by measurement cost ascending (invoked, not just
    listed). Fail-safe."""
    try:
        from tac.witness_control.producer_bridge import rank_duty_to_measure
        return rank_duty_to_measure()
    except Exception:  # noqa: BLE001
        return []


def _factorized_overlay(inputs: RunInputs, horizon_epochs: int) -> dict:
    """Backtest + DECIDE payload for the exact-factorized arm (read-only, numpy CPU).

    The arm enters the existing recommendation list only after the existing λ-net
    tri-gate says BACKTESTED-PASS.  A failed/undersampled arm remains visible here
    with its precise reason, so consumption is auditable without granting authority.
    """
    from tac.witness_control.factorized_adjoint import (
        ARCHITECTURE,
        AXIS_TAG as FACTOR_AXIS,
        factorization_provenance,
        morse_smale_event_prior,
    )

    out = {
        "architecture": ARCHITECTURE,
        "available": False,
        "admission": "UNAVAILABLE",
        "factorization": factorization_provenance(),
        "event_prior": morse_smale_event_prior(),
        "axis": FACTOR_AXIS,
        "score_claim": False,
        "actuation": ACTUATION,
        "validation_scope": (
            "DEVELOPMENT_SET_PASS; residual ridge selected on #205; "
            "independent compatible trajectory owed"),
    }
    try:
        import numpy as np

        from tac.witness_control.control_alphabet import hamiltonian_decide
        from tac.witness_control.lambda_net import (
            backtest,
            build_intervals,
            fit_score_composition,
            lever_features,
            make_model,
            read_trajectory,
        )

        traj = read_trajectory(inputs.run_dir, log_name="run.log")
        intervals = build_intervals(traj)
        if len(intervals) < 3:
            out["reason"] = (
                "need >=3 measured intervals with d_seg_by_class + dense loss_terms; "
                f"have {len(intervals)}")
            return out
        report, field = backtest(traj, architecture=ARCHITECTURE, seed=0)
        comp = fit_score_composition(traj.verdicts)
        phis = np.stack([lever_features(n) for n in traj.lever_names])
        model = make_model(ARCHITECTURE)
        model.set_score_composition(comp)
        model.fit(intervals, phis, seed=0)
        diagnostics = (model.diagnostics.to_dict() if model.diagnostics is not None else None)

        last = intervals[-1]
        current = {n: float(last.u_mean[j]) for j, n in enumerate(traj.lever_names)}
        # Never let a feature-structured, never-varied lever authorize a share move.
        # The full field remains visible as duty-to-measure; DECIDE uses only shares
        # whose variation was observed in this trajectory.
        admitted_field = {k: v for k, v in field.per_lever.items()
                          if field.identified.get(k, False)}
        decision = hamiltonian_decide(
            admitted_field, current, budget=0.10, tier=field.status)
        h_current = float(sum(admitted_field.get(k, 0.0) * v for k, v in current.items()))
        delta = float((decision.hamiltonian_value - h_current) * horizon_epochs)
        # Empirical noise floor in the SAME ΔS units.  The backtest MAE is a held-out
        # Δd_seg interval error; the exact score multiplier is 100.
        wf_mae = float(report.walkforward_mae_model)
        noise = 100.0 * wf_mae if math.isfinite(wf_mae) else None
        band = ([delta - noise, delta + noise] if noise is not None else None)
        changed = any(abs(decision.proposed_shares.get(k, 0.0) - v) > 1e-12
                      for k, v in current.items())
        out.update({
            "available": True,
            "admission": ("BACKTESTED-PASS" if report.passed else "BACKTESTED-FAIL"),
            "backtest": report.to_dict(),
            "lambda_field": {
                "epoch": field.epoch,
                "status": field.status,
                "ranked": [[k, v] for k, v in field.ranked()],
                "identified": field.identified,
            },
            "learned_residual": diagnostics,
            "decision": {
                "proposed_shares": decision.proposed_shares,
                "hamiltonian_current": h_current,
                "hamiltonian_proposed": decision.hamiltonian_value,
                "predicted_dS": delta,
                "predicted_dS_band": band,
                "horizon_epochs": horizon_epochs,
                "changed": changed,
                "identified_levers_only": sorted(admitted_field),
                "unidentified_levers_are_duty_to_measure_not_authority": sorted(
                    k for k in field.per_lever if not field.identified.get(k, False)),
                "why": (
                    "exact rank-4 head x ker(A)-projected visible support x inverse-gain "
                    "pair prior; differentiated-GP temporal residual; five-scalar event "
                    "amplitude admitted only by a past-only inner gate"),
            },
        })
        if report.passed and changed and math.isfinite(delta) and delta <= 0.0:
            out["recommendation_candidate"] = {
                "action": "REALLOCATE_LOSS_SHARE_FACTOR_ADVISORY",
                "predicted_dS": delta,
                "predicted_dS_band": band,
                "horizon_epochs": horizon_epochs,
                "why": out["decision"]["why"],
                "evidence": [
                    "lambda_net BACKTESTED-PASS (LOO + past-only walk-forward + binding AUROC)",
                    "LawRef segnet_head_rank4_linear_flipdist_v1",
                    "LawRef realization_necessity_preimage_per_stratum_v1",
                    "LawRef lane_gain_chain_composed_v1",
                ],
                "source_costate": ARCHITECTURE,
                "proposed_shares": decision.proposed_shares,
                "confidence": (
                    "BACKTESTED-PASS on #205 development trajectory; post-hoc residual "
                    "ridge; independent compatible trajectory owed; advisory only"),
                "cost": float(horizon_epochs),
            }
        return out
    except Exception as exc:  # fail-open observability, never break the core controller
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out


def _event_advisories(inputs: RunInputs, classification: dict | None,
                      factorized: dict) -> list[dict]:
    """Fold the measured Morse-Smale prior and #344 NCDE into DECIDE visibility."""
    prior = factorized.get("event_prior") or {}
    if not prior:
        return []
    ncde = None
    try:
        probe = _load_tools_module("ncde_trajectory_probe")
        ncde_report = probe.run_probe(inputs.run_dir, window=12, emit=False,
                                      do_backtest=False)
        ncde = ncde_report.get("verdict_latest_advisory")
    except Exception as exc:  # noqa: BLE001 - advisory sensor, recorded unavailable
        ncde = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    latest = factorized.get("lambda_field", {}).get("epoch")
    transitions = inputs.stage_rows.get("transitions") or []
    at_transition = any(isinstance(r, dict) and r.get("epoch") == latest for r in transitions)
    ncde_fire = bool(isinstance(ncde, dict) and ncde.get("fire"))
    phase_boundary = ((classification or {}).get("phase_regime") == LABEL_FLOOR_REACHED)
    warning_active = bool(at_transition or ncde_fire or phase_boundary)
    return [{
        "kind": "STAGE_BOUNDARY_MORSE_SMALE_NCDE_ADVISORY",
        "warning_active": warning_active,
        "eligible_at": "stage boundary only",
        "morse_smale": prior,
        "ncde_344": ncde,
        "recommendation": (
            "At the next governed stage boundary, prioritize phase/event-aware duty "
            "rows when the Road-Lane critical-lambda prior and NCDE basin warning agree; "
            "do not mutate the live config from this advisory."),
        "why": (
            "MEASURED low Lane persistence + high birth/death turnover; DERIVED inverse-"
            "gain critical pair pressure; #344 supplies trajectory basin timing"),
        "predicted_dS": None,
        "score_claim": False,
        "actuation": ACTUATION,
        "axis": AXIS_TAG,
    }]


def _merge_factorized_candidate(recs: list[dict], refused: list[dict],
                                factorized: dict, horizon_epochs: int) -> None:
    cand = factorized.get("recommendation_candidate")
    if not isinstance(cand, dict):
        return
    pds = cand.get("predicted_dS")
    if not (isinstance(pds, (int, float)) and math.isfinite(pds)):
        refused.append({**cand, "refusal_reason": "NON_FINITE factorized predicted ΔS"})
        return
    cand["cost"] = candidate_cost(cand, horizon_epochs)
    cand["predicted_dS_per_cost"] = per_cost_score(float(pds), cand["cost"])
    if pds > 0.0:
        refused.append({**cand, "refusal_reason": "NEVER_REGRESS factorized predicted ΔS > 0"})
    else:
        recs.append(cand)
        recs.sort(key=lambda c: c["predicted_dS_per_cost"])


def _gather_fm_texts(inputs: RunInputs, classification: dict | None) -> tuple[list, list]:
    """Extract (telemetry_texts, event_texts) for the #522 FM sense layer from RunInputs.
    Telemetry = the last few verdict rows + the numeric classification (regime input); events
    = stage transitions + the classification's phase/reason strings (event-intelligence input).
    Pure gathering; the FM call itself is the advisory (never a decision)."""
    verdicts = [v for v in inputs.verdicts if isinstance(v, dict)]
    telemetry_texts: list = []
    for v in verdicts[-3:]:
        telemetry_texts.append({k: v.get(k) for k in ("epoch", "d_seg", "d_pose", "seg_form", "ep_loss")})
    if classification:
        telemetry_texts.append({k: classification.get(k) for k in ("classification", "phase_regime", "reason")})
    event_texts: list = []
    for tr in (inputs.stage_rows.get("transitions") or [])[-4:]:
        if isinstance(tr, dict):
            event_texts.append({k: tr.get(k) for k in ("stage", "epoch", "kind")})
    if classification and classification.get("reason"):
        event_texts.append({"reason": classification.get("reason"), "epoch": classification.get("epoch")})
    return telemetry_texts, event_texts


def _attach_fm_advisory(report: ShadowReport, inputs: RunInputs) -> None:
    """Populate report.fm_advisory from the on-device FM sense layer. Fail-open (any error /
    fmtools absent ⇒ leaves fm_advisory=None ⇒ byte-identical row). ADVISORY ONLY."""
    try:
        from tac import fm_advisory as _fm

        if not _fm.available():
            return
        telemetry_texts, event_texts = _gather_fm_texts(inputs, report.classification)
        report.fm_advisory = _fm.shadow_advisory(
            telemetry_texts=telemetry_texts,
            event_texts=event_texts,
            classification=report.classification,
        )
    except Exception:  # advisory sense layer never breaks the shadow pass
        report.fm_advisory = None


def build_shadow_report(inputs: RunInputs,
                        horizon_epochs: int = DEFAULT_HORIZON_EPOCHS,
                        *, with_fm_advisory: bool = False) -> ShadowReport:
    """The full shadow pass: state → costates → classification → ranked recommendations.

    ``with_fm_advisory`` (default OFF — a compute-cost subprocess, so a tracked-queue default-off
    per CLAUDE.md "'Off' is a tracked queue"): when True AND the fmtools venv is present, attach
    the #522 on-device FM ADVISORY sense layer (regime + event intelligence) to ``fm_advisory``.
    Byte-identical output when off/absent."""
    state = _state_snapshot(inputs)
    rows = [v for v in inputs.verdicts if isinstance(v.get("d_seg"), (int, float))]
    d_pose_latest = None
    if rows and isinstance(rows[-1].get("d_pose"), (int, float)):
        d_pose_latest = float(rows[-1]["d_pose"])
    costates: list[CostateEstimate] = analytic_costates(d_pose_latest)

    stages_seen: list[str] = state.get("stages_seen") or []
    for st in stages_seen:
        fits = stage_epoch_costates(inputs.verdicts, st)
        seg = fits["d_seg"]
        ev = (f"verdict rows [stage={st}] ep{int(seg.x_lo)}–{int(seg.x_hi)} "
              f"(n={seg.n}) in {inputs.run_dir}/run.log",)
        costates.append(chain_ds_depoch(fits, d_pose_latest, st, ev))
        # (task #315) per-class within_flip costate — UNIDENTIFIABLE (honest) until the
        # trainer `handoff_readiness` telemetry lands per-class rows; identifies the
        # worst-stalling CLASS surgically once present. Never fabricated from scalar d_seg.
        costates.append(per_class_within_flip_costates(inputs.verdicts, st))
    for a, b in itertools.pairwise(stages_seen):
        costates.append(transition_jump_costate(inputs.verdicts, a, b))
    costates.append(rollback_gain(inputs.verdicts))

    classification = _classify(inputs)
    factorized = _factorized_overlay(inputs, horizon_epochs)
    event_advisories = _event_advisories(inputs, classification, factorized)
    if classification is None and not rows:
        # no data at all: the honest empty report
        report = ShadowReport(
            run_dir=str(inputs.run_dir), as_of_epoch=inputs.as_of_epoch,
            epoch_latest=None, state=state, costates=costates, classification=None,
            recommendations=[], refused=[],
            probe_queue=[*_probe_queue(costates), {"costate": "ALL_TRAJECTORY_COSTATES", "why_unidentifiable": "no verdict rows in run.log yet", "evidence_gap": ["wait for the first n600 advisory verdict"]}],
            duty_to_measure=_duty_to_measure(), producer_signals=_producer_signals(inputs),
            costate_prior=_costate_prior(), duty_ranked=_duty_ranked(),
            factorized_adjoint=factorized, event_advisories=event_advisories)
        if with_fm_advisory:
            _attach_fm_advisory(report, inputs)
        return report

    recs, refused = _recommendations(inputs, costates, classification, horizon_epochs)
    _merge_factorized_candidate(recs, refused, factorized, horizon_epochs)
    phase_active = (classification or {}).get("phase_regime") == LABEL_FLOOR_REACHED
    report = ShadowReport(
        run_dir=str(inputs.run_dir), as_of_epoch=inputs.as_of_epoch,
        epoch_latest=(int(rows[-1]["epoch"]) if rows and
                      isinstance(rows[-1].get("epoch"), (int, float)) else None),
        state=state, costates=costates, classification=classification,
        recommendations=recs, refused=refused, probe_queue=_probe_queue(costates),
        duty_to_measure=_duty_to_measure(phase_active), producer_signals=_producer_signals(inputs),
        costate_prior=_costate_prior(), duty_ranked=_duty_ranked(),
        factorized_adjoint=factorized, event_advisories=event_advisories)
    if with_fm_advisory:
        _attach_fm_advisory(report, inputs)
    return report


def write_shadow_row(run_dir: str | Path, report: ShadowReport) -> Path:
    """Append the report plus its typed deterministic arm-decision observation.

    Both writes are advisory sidecars in the run dir, never mutations of checkpoint/run bytes.
    The causal row exposes the D40 exploration hook but keeps it disabled pending operator GO;
    this function still has no actuation or randomization capability.

    fcntl-locked append via the canonical .omx/state helper (tac.jsonl_store); aligns with
    the sibling stores (costate_posterior.py). See
    .omx/research/fcntl_lock_canonicalization_plan_20260710.md Batch 1.
    """
    from tac import witness_run_artifacts as _wra
    run_path = Path(run_dir)
    out = run_path / _wra.COSTATE_JSONL
    report_row = report.to_row()
    append_locked_jsonl(out, report_row)
    try:
        _write_causal_arm_decision(run_path, report, report_row)
    except Exception as exc:  # score-neutral observability is loud, never an actuator/blocker
        warnings.warn(
            f"causal-manifest arm decision was not appended: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def _write_causal_arm_decision(run_dir: Path, report: ShadowReport, report_row: dict) -> None:
    """Record the shadow policy's chosen arm and all considered alternatives.

    This is the D40 build-side hook only.  Propensity is exactly 1/0 because the current policy is
    deterministic, ``executed=False``, and ``actuation=NONE``.  A future randomized policy must pass
    the schema's externally-authorized-hook + actual seed/draw validation before it can append.
    """

    run_id = run_dir.name
    emitted_at = str(report_row["ts"])
    report_digest = causal_sha256(report_row)
    epoch = int(report.epoch_latest) if report.epoch_latest is not None else 0
    actions = [
        str(item["action"])
        for item in [*report.recommendations, *report.refused]
        if isinstance(item, dict) and item.get("action")
    ]
    alternatives = tuple(dict.fromkeys(actions)) or ("NO_RANKED_ARM",)
    chosen = alternatives[0]
    policy_id = "costate_shadow_rank_v1"
    policy_sha = causal_sha256({
        "classification": report.classification,
        "ordered_alternatives": alternatives,
        "policy_id": policy_id,
    })
    state_values = report.state if isinstance(report.state, dict) else {}
    if any(isinstance(state_values.get(name), (int, float)) for name in ("d_seg", "d_pose", "implied_S")):
        outcome = CausalRealizedOutcome(
            observed=True,
            through_r=True,
            d_seg=(float(state_values["d_seg"])
                   if isinstance(state_values.get("d_seg"), (int, float)) else None),
            d_pose=(float(state_values["d_pose"])
                    if isinstance(state_values.get("d_pose"), (int, float)) else None),
            archive_bytes=(int(state_values["blob_bytes"])
                           if isinstance(state_values.get("blob_bytes"), (int, float)) else None),
            implied_score=(float(state_values["implied_S"])
                           if isinstance(state_values.get("implied_S"), (int, float)) else None),
            axis=AXIS_TAG,
        )
    else:
        outcome = CausalRealizedOutcome(
            observed=False,
            through_r=False,
            axis=AXIS_TAG,
            missing_reason="costate shadow ran before the first realized-through-R verdict",
        )
    state = CausalStateSummary(
        boundary_id=f"costate_shadow:{epoch}:{emitted_at}:{report_digest[:16]}",
        sequence_index=causal_boundary_sequence_index(epoch, "costate_decision"),
        boundary_kind="costate_decision",
        epoch=epoch,
        stage=str(state_values.get("stage") or "unknown"),
        policy_sha256=policy_sha,
        data_order_cursor=None,
        telemetry_history_sha256=causal_sha256({
            "classification": report.classification,
            "state": state_values,
        }),
        telemetry_history_rows=int(state_values.get("n_verdicts", 0) or 0),
        checkpoint=None,
        resume_state_sha256=None,
        rng_state_sha256=None,
        controller_state_sha256=None,
        apparatus=CausalApparatusState(
            guard_path="costate_shadow.never_regress_then_delta_s_per_cost",
            measurement_mode="costate_shadow_observational",
        ),
        outcome=outcome,
        observed_at_utc=emitted_at,
    )
    decision_id = f"costate_shadow:{epoch}:{emitted_at}:{report_digest[:16]}"
    decision = ExplorationDecisionRow(
        row_id=f"exploration_decision:{run_id}:{decision_id}",
        decision_id=decision_id,
        run_id=run_id,
        state=state,
        chosen_arm=chosen,
        arm_propensities=tuple(
            ArmPropensity(arm_id=arm, propensity=1.0 if arm == chosen else 0.0)
            for arm in alternatives
        ),
        policy_id=policy_id,
        policy_sha256=policy_sha,
        policy_mode="deterministic",
        exploration_hook="disabled_pending_operator_go",
        executed=False,
        actuation=ACTUATION,
        random_seed=None,
        random_draw=None,
        emitted_at_utc=emitted_at,
    )
    CausalManifestWriter(run_dir / CAUSAL_MANIFEST_FILENAME, run_id).record_decision(decision)
