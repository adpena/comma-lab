"""Campaign-REPL inspection surface for the θ* costate controller (task #324).

The Duck-Harness orchestrator pattern (tufalabs.ai/research/duck-harness, re-implemented
natively — the repo ships NO license), mapped onto our campaign: bind the live run +
history + lever space as a compact, inspectable WORLD MODEL; expose one GOVERNED action
that emits launcher argv but NEVER fires paid GPU (CONTAINMENT); and score progress by
ACTION-EFFICIENCY = |ΔS| per paid dispatch (Duck's "fewest actions" = our means/ends
firewall). This DE-ORPHANS onto the existing shadow controller (#247) — it does NOT
rebuild observe→estimate→recommend→STOP; it adds the three genuinely-new pieces:

  1. ``campaign_world_model`` — a compact summary-carry struct (crux · ranked open levers ·
     last measured deltas · what's ruled-out) regenerated from the run + shadow report
     each cycle. Summary-CARRY, never eviction (durable DAG/memory stays the source of
     truth per the anti-forgetting non-negotiable).
  2. ``propose_argv`` — the single governed action. Emits the launcher argv a lever choice
     implies, tagged ``go_required=True``; it NEVER dispatches. Actuation stays gated on
     operator GO + the governed launcher (Phase-B costate rules).
  3. ``action_efficiency`` — |ΔS| realized per paid dispatch, from the frontier history.

All read-only, CPU-light. Actuation is structurally impossible from this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tac.witness_control.shadow_controller import (
    build_shadow_report,
    load_run_inputs,
)


@dataclass(frozen=True)
class CampaignWorldModel:
    """The compact re-injectable campaign state (Duck's 'working world model')."""
    run_dir: str
    epoch_latest: int | None
    crux: str                                   # the one-line current bottleneck
    best_d_seg: float | None
    last_delta_d_seg: float | None              # most recent verdict-to-verdict move
    ruled_out: list[str] = field(default_factory=list)
    ranked_open_levers: list[dict] = field(default_factory=list)   # from shadow recs
    frontier_pointer: float | None = None
    generated_at: str = ""

    def to_row(self) -> dict:
        return asdict(self)


def _load_frontier_pointer() -> float | None:
    p = Path(__file__).resolve().parents[3] / ".omx/state/canonical_frontier_pointer.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    # tolerate a few shapes; return the local-frontier exact score if present
    for k in ("local_frontier_score", "frontier_score", "score"):
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def campaign_world_model(run_dir: str | Path, as_of_epoch: int | None = None,
                         ruled_out: list[str] | None = None) -> CampaignWorldModel:
    """Regenerate the compact campaign world model from the live run + shadow report.

    Reuses the existing shadow controller (does not re-derive costates). ``ruled_out`` is
    the durable list of dead levers (kept in the DAG/memory, passed in — this module does
    not invent it, per anti-forgetting: durable memory is the source of truth)."""
    inputs = load_run_inputs(run_dir, as_of_epoch=as_of_epoch)
    report = build_shadow_report(inputs)
    verdicts = inputs.verdicts
    best = None
    for v in verdicts:
        d = v.get("d_seg")
        if isinstance(d, (int, float)):
            best = d if best is None else min(best, d)
    last_delta = None
    ds = [v.get("d_seg") for v in verdicts if isinstance(v.get("d_seg"), (int, float))]
    if len(ds) >= 2:
        last_delta = float(ds[-1] - ds[-2])
    # (review-fix) the classification dict's key is "classification" (see shadow_controller
    # _classify), NOT "label" — the old .get("label") ALWAYS returned None, so crux silently
    # fell through to "no-classification" even on a well-identified trajectory. Same shape as the
    # CLOSE-gate wrong-key bug: a wrong dict key fabricating a false "nothing happening" narrative.
    crux = (report.classification or {}).get("classification") if report.classification else None
    return CampaignWorldModel(
        run_dir=str(Path(run_dir)),
        epoch_latest=report.epoch_latest,
        crux=crux or "no-classification",
        best_d_seg=best,
        last_delta_d_seg=last_delta,
        ruled_out=list(ruled_out or []),
        ranked_open_levers=list(report.recommendations or []),
        frontier_pointer=_load_frontier_pointer(),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@dataclass(frozen=True)
class ProposedAction:
    """A governed lever proposal — emits argv but NEVER dispatches (CONTAINMENT)."""
    lever: str
    argv: list[str]
    rationale: str
    go_required: bool = True          # ALWAYS True — this module cannot fire paid GPU
    dispatched: bool = False          # ALWAYS False — actuation is structurally absent

    def to_row(self) -> dict:
        return asdict(self)


def propose_argv(lever: str, base_argv: list[str], overrides: dict,
                 rationale: str) -> ProposedAction:
    """Emit the launcher argv a lever choice implies. Governed: NEVER fires.

    ``overrides`` maps ``--flag`` → value (value None ⇒ a bare boolean flag). The result
    is an argv list the operator (or the governed launcher) may run AFTER explicit GO —
    this function only constructs and returns it.
    """
    argv = list(base_argv)
    for flag, val in overrides.items():
        f = flag if flag.startswith("--") else f"--{flag}"
        if val is None:
            argv.append(f)
        else:
            argv += [f, str(val)]
    return ProposedAction(lever=lever, argv=argv, rationale=rationale)


def action_efficiency(frontier_history: list[float]) -> dict:
    """|ΔS| realized per paid dispatch (Duck 'fewest actions' = means/ends firewall).

    ``frontier_history`` = the exact-eval score after each paid dispatch, oldest→newest.
    Returns total improvement, dispatch count, and mean |ΔS| per dispatch (the metric to
    MAXIMIZE: bias toward the lever with the largest predicted |ΔS| per expensive action).
    """
    xs = [float(s) for s in frontier_history if isinstance(s, (int, float))]
    if len(xs) < 2:
        return {"dispatches": max(len(xs) - 1, 0), "total_improvement": 0.0,
                "mean_abs_dS_per_dispatch": None}
    improvements = [xs[i] - xs[i + 1] for i in range(len(xs) - 1)]   # positive = score fell
    n = len(improvements)
    return {
        "dispatches": n,
        "total_improvement": float(sum(improvements)),
        "mean_abs_dS_per_dispatch": float(sum(abs(d) for d in improvements) / n),
    }


def write_world_model_row(run_dir: str | Path, wm: CampaignWorldModel) -> Path:
    """Append the world-model as a trace row (Duck ``traces.py`` machine-readable episode)
    to the run dir, closing the DAG→DSL→run→rows→equations cycle. Never mutates state."""
    out = Path(run_dir) / "campaign_world_model.jsonl"
    with out.open("a") as fh:
        fh.write(json.dumps(wm.to_row()) + "\n")
    return out
