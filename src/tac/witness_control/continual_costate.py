# SPDX-License-Identifier: MIT
"""Continual-learning COMPOUNDING for the #426 costate organ — TRIALITY-NATIVE memory.

Operator bindings 2026-07-11: (a) "full authority to ... make more sophisticated and
autonomous and continual learning compounding"; (b) "It should be triality native as
its language and memory surface." So the organ's cross-session memory is NOT a bespoke
binary store: it is an append-only MARKDOWN LEDGER of FEED-style blocks
(``.omx/research/costate_organ_trajectory_ledger.md``) — a durable DAG-leg artifact the
#411 graph-memory INDEXES into typed ``regime`` nodes (reconstruct-not-retrieve). The
next session reconstructs a smarter organ by PARSING the ledger (the source of truth),
or by graph recall; there is no parallel registry to drift.

What compounds, per recorded trajectory:
  * the architecture tournament's LOO + WALK-FORWARD results (the accruing backtest
    sample — the measured graduation curve toward the VAPO learned-critic crossover);
  * the PRISM/BSF prototype set (named regimes + anchors) — the prototype LIBRARY is
    reconstructed by union-with-dedup over all records (new regimes accrue; recurring
    regimes gain observation counts);
  * the duty-to-measure queue snapshot (what the organ wanted measured, and why).

Architecture GRADUATION is MEASURED, never hard-coded: a candidate graduates to
recommended only when its cumulative walk-forward record beats the incumbent solve
family on ≥ ``GRADUATION_MIN_RECORDS`` trajectories (the Gödel discipline applied to
the organ's own growth). At n=1 trajectory nothing learned can graduate — stated.

Containment: this module READS telemetry and APPENDS one advisory markdown block;
no process surface, no actuation, nothing here is a score ([macOS advisory]
NON-PROMOTABLE). Deterministic: sorted, rounded, stamped payloads.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
ORGAN_LEDGER = _REPO / ".omx" / "research" / "costate_organ_trajectory_ledger.md"

#: measured-graduation floor: a learned arch needs at least this many trajectory
#: records (each an independent campaign) with a better cumulative walk-forward MAE
#: than the solve family before it may be recommended. NOT a VAPO-crossover guess —
#: the crossover is detected by the record itself.
GRADUATION_MIN_RECORDS = 3
#: the incumbent (closed-form / interpretable) family a learned arch must beat
INCUMBENT_ARCHS = ("A_ridge_solve", "E_prototype", "E_prototype_bregman", "F_bsf",
                   "G_ridge_scorerprior")
LEARNED_ARCHS = ("B_mlp", "C_gru_path", "D_deeponet")

_BLOCK_RE = re.compile(
    r"^## FEED-426-organ-(?P<stamp>\S+)[^\n]*\n(?P<body>.*?)(?=^## FEED-426-organ-|\Z)",
    re.M | re.S)
_JSON_FENCE_RE = re.compile(r"```json\n(.*?)\n```", re.S)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _locked_append(path: Path, text: str) -> None:
    """fcntl-locked append (markdown ledger; mirrors the .omx/state JSONL discipline)."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(text)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def compose_trajectory_record(traj, arch_reports: dict, prototypes: list,
                              duty_top: list[dict] | None = None) -> dict:
    """Compose the machine-readable payload for one trajectory's organ fit.

    ``arch_reports``: architecture → BacktestReport (from ``benchmark_all``).
    ``prototypes``: fitted Prototype objects (their ``to_jsonable`` rides the block)."""
    reports = {}
    for arch, r in sorted(arch_reports.items()):
        reports[arch] = {
            "loo_mae": round(float(r.forecast_mae_model), 8),
            "loo_heur": round(float(r.forecast_mae_heuristic), 8),
            "wf_mae": (round(float(r.walkforward_mae_model), 8)
                       if math.isfinite(r.walkforward_mae_model) else None),
            "wf_heur": (round(float(r.walkforward_mae_heuristic), 8)
                        if math.isfinite(r.walkforward_mae_heuristic) else None),
            "perclass_loo": round(float(r.forecast_perclass_mae_model), 8),
            "passed": bool(r.passed),
            "passed_walkforward": bool(r.passed_walkforward),
        }
    finite_wf = {a: v["wf_mae"] for a, v in reports.items() if v["wf_mae"] is not None}
    winner_wf = min(finite_wf, key=lambda a: finite_wf[a]) if finite_wf else None
    ep0 = float(traj.verdicts[0]["epoch"]) if traj.verdicts else None
    ep1 = float(traj.verdicts[-1]["epoch"]) if traj.verdicts else None
    return {
        "run_ref": Path(traj.run_dir).name,
        "generated_at": _utc(),
        "n_verdicts": traj.n_verdicts,
        "n_intervals": max(traj.n_verdicts - 1, 0),
        "epoch_span": [ep0, ep1],
        "arch_reports": reports,
        "winner_walkforward": winner_wf,
        "prototypes": [p.to_jsonable() for p in prototypes],
        "duty_queue_top": (duty_top or [])[:5],
        "axis_tag": "[macOS advisory] NON-PROMOTABLE",
        "score_claim": False,
    }


def append_trajectory_record(payload: dict, ledger_path: Path | None = None) -> str:
    """Append ONE FEED-426-organ block to the triality ledger (locked; append-only).
    Returns the block text (the durable artifact)."""
    stamp = payload.get("generated_at") or _utc()
    winner = payload.get("winner_walkforward")
    n_iv = payload.get("n_intervals")
    protos = payload.get("prototypes") or []
    names = sorted({p.get("name", "?") for p in protos})
    block = (
        f"## FEED-426-organ-{stamp} run={payload.get('run_ref', '?')}\n"
        f"organ trajectory record: {n_iv} intervals; walk-forward winner "
        f"{winner or 'n/a'}; regimes {', '.join(names) or 'none'}. "
        "Every number [macOS advisory] NON-PROMOTABLE; the organ is MEANS "
        "(#426; graph-memory indexes this block into regime nodes).\n\n"
        "```json\n" + json.dumps(payload, indent=1, sort_keys=True) + "\n```\n\n"
    )
    p = ledger_path or ORGAN_LEDGER
    if not p.exists():
        _locked_append(p, "# costate organ trajectory ledger (#426) — append-only, "
                          "triality-native (graph-memory source)\n\n")
    _locked_append(p, block)
    return block


@dataclass(frozen=True)
class OrganMemory:
    """The reconstructed cross-session organ memory (parse of the ledger blocks)."""

    records: tuple[dict, ...]
    prototype_library: tuple[dict, ...]     # deduped named regimes with observation counts
    total_intervals: int
    n_runs: int

    @property
    def n_records(self) -> int:
        return len(self.records)


def load_organ_memory(ledger_path: Path | None = None) -> OrganMemory:
    """Reconstruct-not-retrieve: parse every FEED-426-organ block; union+dedup regimes."""
    p = ledger_path or ORGAN_LEDGER
    records: list[dict] = []
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in _BLOCK_RE.finditer(text):
            jm = _JSON_FENCE_RE.search(m.group("body"))
            if not jm:
                continue
            try:
                payload = json.loads(jm.group(1))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
    records.sort(key=lambda r: str(r.get("generated_at", "")))
    # latest-per-run semantics (the ledger is append-only; re-recording the same run
    # supersedes, never double-counts — mirrors costate_posterior's per-run dedup)
    latest: dict[str, dict] = {}
    for rec in records:
        latest[str(rec.get("run_ref", "?"))] = rec
    effective = sorted(latest.values(), key=lambda r: str(r.get("generated_at", "")))
    lib: dict[str, dict] = {}
    for rec in effective:
        for proto in rec.get("prototypes") or []:
            name = str(proto.get("name", "?"))
            center = np.asarray(proto.get("center", []), dtype=float)
            key = name
            prev = lib.get(key)
            if prev is not None:
                pc = np.asarray(prev["center"], dtype=float)
                if pc.shape == center.shape and center.size:
                    prev["n_observations"] += 1
                    prev["center"] = ((pc * (prev["n_observations"] - 1) + center)
                                      / prev["n_observations"]).round(8).tolist()
                prev["last_run"] = rec.get("run_ref")
            else:
                lib[key] = {"name": name, "scale": proto.get("scale"),
                            "center": proto.get("center"),
                            "block_dim": proto.get("block_dim", 0),
                            "n_observations": 1,
                            "first_run": rec.get("run_ref"),
                            "last_run": rec.get("run_ref")}
    total_iv = sum(int(r.get("n_intervals") or 0) for r in effective)
    return OrganMemory(records=tuple(effective),
                       prototype_library=tuple(lib[k] for k in sorted(lib)),
                       total_intervals=total_iv, n_runs=len(effective))


@dataclass(frozen=True)
class ArchitectureArbitration:
    """The measured graduation state: which arm the accumulated record recommends."""

    per_arch: dict[str, dict]
    recommended: str
    recommendation_basis: str
    graduation: dict[str, str]         # learned arch → status line (measured)
    n_records: int
    total_intervals: int
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"


def arbitrate_architecture(memory: OrganMemory) -> ArchitectureArbitration:
    """Cumulative walk-forward-first arbitration over the accrued record.

    Recommended = the arch with the best cumulative (record-mean) walk-forward MAE
    among those that ever passed their own backtest gate; learned archs additionally
    need ≥GRADUATION_MIN_RECORDS records beating the best incumbent (measured, never
    asserted). With zero records the incumbent default is the ridge solve."""
    per: dict[str, dict] = {}
    for rec in memory.records:
        for arch, r in (rec.get("arch_reports") or {}).items():
            row = per.setdefault(arch, {"wf_maes": [], "loo_maes": [], "passes": 0,
                                        "wf_passes": 0, "n": 0})
            row["n"] += 1
            row["passes"] += int(bool(r.get("passed")))
            row["wf_passes"] += int(bool(r.get("passed_walkforward")))
            if r.get("wf_mae") is not None:
                row["wf_maes"].append(float(r["wf_mae"]))
            row["loo_maes"].append(float(r.get("loo_mae", float("nan"))))
    for arch, row in per.items():
        row["wf_mean"] = float(np.mean(row["wf_maes"])) if row["wf_maes"] else None
        row["loo_mean"] = (float(np.nanmean(row["loo_maes"]))
                           if row["loo_maes"] else None)
        del row["wf_maes"], row["loo_maes"]

    incumbents = {a: v for a, v in per.items()
                  if a in INCUMBENT_ARCHS and v["wf_mean"] is not None and v["passes"] > 0}
    if not incumbents:
        return ArchitectureArbitration(
            per_arch=per, recommended="A_ridge_solve",
            recommendation_basis="no passing records yet — incumbent default (solve)",
            graduation={a: "no records" for a in LEARNED_ARCHS},
            n_records=memory.n_records, total_intervals=memory.total_intervals)
    best_inc = min(incumbents, key=lambda a: incumbents[a]["wf_mean"])
    best_inc_wf = incumbents[best_inc]["wf_mean"]

    graduation: dict[str, str] = {}
    recommended, basis = best_inc, (
        f"best cumulative walk-forward among passing incumbents "
        f"({best_inc_wf:.6f} over {incumbents[best_inc]['n']} record(s))")
    for arch in LEARNED_ARCHS:
        row = per.get(arch)
        if not row or row["wf_mean"] is None:
            graduation[arch] = "no walk-forward records"
            continue
        wins = row["wf_mean"] < best_inc_wf
        if wins and row["n"] >= GRADUATION_MIN_RECORDS and row["wf_passes"] > 0:
            graduation[arch] = (f"GRADUATED: wf {row['wf_mean']:.6f} < incumbent "
                                f"{best_inc_wf:.6f} over {row['n']} records")
            if row["wf_mean"] < per[recommended]["wf_mean"]:
                recommended = arch
                basis = graduation[arch]
        else:
            graduation[arch] = (
                f"not graduated: wf {row['wf_mean']:.6f} vs incumbent {best_inc_wf:.6f} "
                f"({row['n']}/{GRADUATION_MIN_RECORDS} records"
                f"{'' if wins else '; not winning'}) — the VAPO-crossover is detected "
                "by this record, not scheduled")
    return ArchitectureArbitration(
        per_arch=per, recommended=recommended, recommendation_basis=basis,
        graduation=graduation, n_records=memory.n_records,
        total_intervals=memory.total_intervals)


def organ_summary(ledger_path: Path | None = None) -> dict:
    """The digest-facing one-call summary (fail-open at the caller)."""
    mem = load_organ_memory(ledger_path)
    arb = arbitrate_architecture(mem)
    return {
        "n_records": mem.n_records,
        "n_runs": mem.n_runs,
        "total_intervals": mem.total_intervals,
        "n_regimes": len(mem.prototype_library),
        "regimes": [p["name"] for p in mem.prototype_library],
        "recommended_architecture": arb.recommended,
        "recommendation_basis": arb.recommendation_basis,
        "graduation": arb.graduation,
        "ledger": str(ledger_path or ORGAN_LEDGER),
    }
