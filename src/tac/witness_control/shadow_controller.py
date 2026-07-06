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
"""
from __future__ import annotations

import importlib
import itertools
import json
import sys
from dataclasses import dataclass, field as _dc_field
from datetime import UTC, datetime
from pathlib import Path

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
            "actuation": ACTUATION,
            "axis": AXIS_TAG,
            "pointer": POINTER_NOTE,
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
    candidates: list[dict] = []

    def _cand(action: str, delta: float, band: list[float] | None, rationale: str,
              evidence: tuple[str, ...], costate_name: str | None) -> None:
        candidates.append({
            "action": action, "predicted_dS": delta, "predicted_dS_band": band,
            "horizon_epochs": horizon_epochs, "rationale": rationale,
            "evidence": list(evidence), "costate": costate_name,
        })

    if cls == BINDING_TERM_STALL:
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
    elif cls == "plateau":
        _cand("ADVANCE_STAGE_OR_EARLY_STOP", 0.0,
              (_band_from(ds_dep, float(horizon_epochs))[1] if ds_dep and ds_dep.stderr
               else None),
              "|dS/dep| ≈ 0 within stage — the Lyapunov certificate has flattened; the "
              "marginal value of more epochs here is ~0, so advance or stop",
              (ds_dep.evidence if ds_dep else ()), (ds_dep.name if ds_dep else None))
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
        if c["predicted_dS"] > 0.0:
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


def _duty_to_measure() -> list[dict]:
    """The #247 SENSE→DECIDE queue: DSL levers the activation ledger records as OWED a measurement
    (never-fired OR fired-but-unmeasured, not retired). Fail-safe: the ledger must NEVER break a
    shadow report, so any import/read error yields an empty queue (the report degrades to legacy).

    NO-FAKE: each row carries the ledger's activation state ONLY — no predicted ΔS (a never-fired
    lever has no measured costate). The controller surfaces these so "off" is a tracked queue the
    operator never has to remember; they enter the ΔS-per-cost DECIDE ranking after they fire+measure.
    """
    try:
        from tac.witness_dsl import activation_ledger as _al
        owed = set(_al.duty_to_measure())
        rows = [r for r in _al.activation_report() if r["lever"] in owed]
        for r in rows:
            r["why"] = ("registered DSL lever OWED a measurement (default-off is a tracked queue, "
                        "not a forgotten default); NO predicted ΔS until fired+measured")
        return rows
    except Exception:  # noqa: BLE001 — advisory sidecar must never break the report
        return []


def build_shadow_report(inputs: RunInputs,
                        horizon_epochs: int = DEFAULT_HORIZON_EPOCHS) -> ShadowReport:
    """The full shadow pass: state → costates → classification → ranked recommendations."""
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
    if classification is None and not rows:
        # no data at all: the honest empty report
        return ShadowReport(
            run_dir=str(inputs.run_dir), as_of_epoch=inputs.as_of_epoch,
            epoch_latest=None, state=state, costates=costates, classification=None,
            recommendations=[], refused=[],
            probe_queue=[*_probe_queue(costates), {"costate": "ALL_TRAJECTORY_COSTATES", "why_unidentifiable": "no verdict rows in run.log yet", "evidence_gap": ["wait for the first n600 advisory verdict"]}],
            duty_to_measure=_duty_to_measure())

    recs, refused = _recommendations(inputs, costates, classification, horizon_epochs)
    return ShadowReport(
        run_dir=str(inputs.run_dir), as_of_epoch=inputs.as_of_epoch,
        epoch_latest=(int(rows[-1]["epoch"]) if rows and
                      isinstance(rows[-1].get("epoch"), (int, float)) else None),
        state=state, costates=costates, classification=classification,
        recommendations=recs, refused=refused, probe_queue=_probe_queue(costates),
        duty_to_measure=_duty_to_measure())


def write_shadow_row(run_dir: str | Path, report: ShadowReport) -> Path:
    """Append the report row to ``<run_dir>/costate_shadow.jsonl`` (the ONLY write this
    package performs — an advisory sidecar in the run dir, never a mutation of the run)."""
    out = Path(run_dir) / "costate_shadow.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report.to_row(), sort_keys=True) + "\n")
    return out
