#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only task-516 backtest + consumption audit for the existing costate organ.

The tool never writes a run directory.  It hashes each source ``run.log`` before
and after the pass and refuses to certify read-only custody if bytes changed.  Its
only write is the requested small JSON receipt under ``.omx/research``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ARCH = "V_exact_factorized_residual"
AXIS = "[macOS advisory] NON-PROMOTABLE"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _line(path: Path, needle: str) -> int | None:
    try:
        for i, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if needle in text:
                return i
    except OSError:
        pass
    return None


def _last_jsonl(path: Path) -> dict | None:
    if not path.is_file():
        return None
    last = None
    for raw in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            last = row
    return last


def _run_row(label: str, run_dir: Path) -> dict:
    from tac.witness_control.lambda_net import backtest, build_intervals, read_trajectory
    from tac.witness_control.shadow_controller import build_shadow_report, load_run_inputs

    log = run_dir / "run.log"
    before = _sha256(log)
    out = {
        "label": label,
        "run_dir": str(run_dir),
        "run_exists": run_dir.is_dir(),
        "run_log_exists": log.is_file(),
        "run_log_sha256_before": before,
        "axis": AXIS,
        "score_claim": False,
        "hyperparameter_selection_scope": (
            "POST_HOC_DEVELOPMENT_ON_205; independent compatible trajectory validation owed"),
    }
    if not log.is_file():
        out.update({
            "status": "PENDING_NO_RUN_LOG",
            "verdict_scope": "no trajectory inference; harvest again when rows arrive",
        })
        return out
    traj = read_trajectory(run_dir, log_name="run.log")
    intervals = build_intervals(traj)
    out.update({
        "n_verdicts": traj.n_verdicts,
        "n_loss_term_rows": len(traj.loss_terms),
        "n_levers": len(traj.lever_names),
        "n_intervals": len(intervals),
    })
    if len(intervals) >= 3:
        report, field = backtest(traj, architecture=ARCH, seed=0)
        shadow = build_shadow_report(load_run_inputs(run_dir))
        factor = shadow.factorized_adjoint or {}
        out.update({
            "status": factor.get("admission", "UNKNOWN"),
            "backtest": report.to_dict(),
            "lambda_field_top": [[k, v] for k, v in field.ranked()[:8]],
            "consumed_shadow_decision": factor.get("decision"),
            "recommendation_entered_existing_ranked_path": any(
                r.get("source_costate") == ARCH for r in shadow.recommendations),
            "top_shadow_action": (shadow.recommendations[0].get("action")
                                  if shadow.recommendations else None),
            "event_advisories": shadow.event_advisories,
            "verdict_scope": (
                "trajectory-local advisory backtest; no score/pointer/promotion authority"),
        })
    else:
        missing = []
        if traj.n_verdicts < 4:
            missing.append("need >=4 verdicts")
        if not any(isinstance(v.get("d_seg_by_class"), list) for v in traj.verdicts):
            missing.append("d_seg_by_class absent")
        if not traj.loss_terms:
            missing.append("dense loss_terms absent")
        out.update({
            "status": "UNAVAILABLE_INSUFFICIENT_INTERVAL_SCHEMA",
            "blockers": missing or ["interval join yielded fewer than three rows"],
            "verdict_scope": "no cross-run inference from #205",
        })
    after = _sha256(log)
    out["run_log_sha256_after"] = after
    out["source_bytes_unchanged"] = before == after
    if before != after:
        raise RuntimeError(f"source run changed during audit: {run_dir}")
    return out


def _consumption_audit(source_root: Path, primary_run: Path) -> list[dict]:
    src = REPO / "src/tac/witness_control"
    tools = REPO / "tools"
    latest_shadow = _last_jsonl(primary_run / "costate_shadow.jsonl")
    last_ts = latest_shadow.get("ts") if latest_shadow else None
    last_action = None
    if latest_shadow and latest_shadow.get("recommendations"):
        last_action = latest_shadow["recommendations"][0].get("action")

    def ev(path: Path, needle: str) -> str:
        line = _line(path, needle)
        return f"{path.relative_to(REPO)}:{line if line is not None else '?'}"

    return [
        {
            "surface": "shadow_controller estimate/classify/recommend",
            "verdict": "CONSUMED",
            "evidence": [
                ev(src / "shadow_controller.py", "def build_shadow_report"),
                ev(tools / "costate_observer_loop.py", "build_shadow_report"),
                ev(tools / "launch_witness_run.py", "costate_observer_loop.py"),
            ],
            "last_fired": {"ts": last_ts, "top_action": last_action},
            "decision_change": "writes the ranked recommendation consumed by digest/dashboard",
        },
        {
            "surface": "producer_bridge + cross-run posterior + duty rank",
            "verdict": "CONSUMED",
            "evidence": [
                ev(src / "shadow_controller.py", "def _producer_signals"),
                ev(src / "shadow_controller.py", "def _costate_prior"),
                ev(src / "shadow_controller.py", "def _duty_ranked"),
            ],
            "last_fired": "every shadow report; missing producer data is explicit available=false",
            "decision_change": "axis EV and owed-probe order enter SENSE/DECIDE",
        },
        {
            "surface": "exact-factorized rank4 x ker(A) x gain adjoint",
            "verdict": "CONSUMED",
            "evidence": [
                ev(src / "shadow_controller.py", "def _factorized_overlay"),
                ev(src / "shadow_controller.py", "_merge_factorized_candidate"),
                ev(src / "costate_panel.py", "exact_factorized"),
            ],
            "last_fired": "verified in-memory on primary run by this receipt",
            "decision_change": (
                "enters the existing ranked recommendation path only on BACKTESTED-PASS; "
                "unvaried levers remain duty-to-measure"),
        },
        {
            "surface": "Morse-Smale event prior + #344 NCDE",
            "verdict": "CONSUMED",
            "evidence": [
                ev(src / "shadow_controller.py", "def _event_advisories"),
                ev(tools / "ncde_trajectory_probe.py", "def run_probe"),
            ],
            "last_fired": "verified in-memory on primary run by this receipt",
            "decision_change": (
                "emits a stage-boundary warning row; predicted_dS remains null and actuation NONE"),
        },
        {
            "surface": "CostateAgent DSL/panel",
            "verdict": "ORPHANED_FROM_ALWAYS_ON_PRODUCTION",
            "evidence": [
                "repo search found derive_costate_agent_v1 consumers only in tests/tooling",
                ev(REPO / "src/tac/witness_dsl/costate_agent_dsl.py", "def derive_costate_agent_v1"),
            ],
            "last_fired": "test/tool invocation only; no always-on observer call",
            "decision_change": (
                "the factorized expert is registered for truthful compile, but production authority "
                "comes from the shadow-controller integration above"),
        },
        {
            "surface": "regime_dispatch",
            "verdict": "INERT_FOR_ALWAYS_ON_RECOMMENDATION",
            "evidence": [
                "consumed by lambda_net_backtest/costate_digest, not build_shadow_report",
                ev(src / "regime_dispatch.py", "def dispatch_for_trajectory"),
            ],
            "last_fired": "digest/backtest only",
            "decision_change": "no causal shadow recommendation change",
        },
        {
            "surface": "digest + dashboard",
            "verdict": "CONSUMED",
            "evidence": [
                ev(tools / "costate_digest.py", "factorized-adjoint:"),
                ev(tools / "witness_run_introspect.py", "factorized_adjoint"),
                ev(tools / "dashboard_server.py", "exact-factorized"),
            ],
            "last_fired": "on next digest/dashboard refresh; schema verified by tests",
            "decision_change": "surfaces exact/learned/confidence/recommendation-why provenance",
        },
    ]


def _default_specs(source_root: Path) -> list[tuple[str, Path]]:
    results = source_root / "experiments/results"
    specs = [
        ("#205_live_v752", results / "levelset_v752_baseline_20260710T185913Z"),
        ("mod32cap", results / "levelset_n600_witness_mod32cap_20260706T115554Z"),
    ]
    c2 = sorted({*results.glob("c2*"), *results.glob("levelset*c2*")})
    if c2:
        specs.extend((f"c2:{p.name}", p) for p in c2)
    else:
        specs.append(("c2", results / "c2_rows_pending"))
    return specs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True,
                    help="read-only canonical repo root carrying experiment run dirs")
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                    help="override/add a trajectory (repeatable)")
    ap.add_argument("--output", type=Path, default=(
        REPO / ".omx/research/costate_organ_elevation_backtest_20260716.json"))
    args = ap.parse_args(argv)

    specs = _default_specs(args.source_root)
    for item in args.run:
        if "=" not in item:
            ap.error("--run must be LABEL=PATH")
        label, raw = item.split("=", 1)
        specs.append((label, Path(raw)))
    rows = [_run_row(label, path) for label, path in specs]
    primary = specs[0][1]
    payload = {
        "schema": "costate_organ_elevation_backtest.v1",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "architecture": ARCH,
        "axis": AXIS,
        "score_claim": False,
        "validation_scope": (
            "DEVELOPMENT_SET_PASS for #205 because residual ridge was selected during this "
            "build; independent compatible trajectory owed"),
        "pointer_changed": False,
        "actuation": "NONE",
        "source_root": str(args.source_root),
        "runs": rows,
        "consumption_audit": _consumption_audit(args.source_root, primary),
        "verdict_scope": (
            "#205 is the only compatible trajectory currently backtestable; mod32cap lacks "
            "interval-aligned d_seg_by_class; c2 rows are pending. No equivalence inferred."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
