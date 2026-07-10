#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Witness launch-READINESS gate — refuse a long-running witness launch whose
config has NOT reconciled against the current high-EV owed-updates queue.

The failure this extincts (operator, 2026-07-10): a multi-day witness pointer
run was fired on a *stale* validated ``launch.sh`` that SKIPPED the fire-now
levers the just-built default-off decision table (task #405) identified as the
top of the remaining descent (HorizonWeightedMargin 43.8%, StepNativeActivation
31.6%). The existing guards (``launch_guard_hook`` raw-launch admission, the
daemon memory/governor admission gate) all PASSED — none of them knows whether
the config is the *right* config. This gate is the missing "optimal-form /
config-freshness" check: it binds the #405 decision table to the launch
decision, so a naive "run the old validated config right away" is refused
(or loudly flagged) with the SPECIFIC missing rungs named.

Design invariants (mirrors ``launch_guard_hook``):
  * PURE decision surface — :func:`assess_launch_readiness` does the file reads
    at the edge; :func:`decide_readiness` is pure (flags-set + table-rows in →
    verdict out) and unit-tested.
  * FAIL-OPEN on its OWN errors (missing table ⇒ cannot assess ⇒ PROCEED with a
    NOTE, never brick a launch on gate breakage) — but REFUSE (or WARN) when it
    CAN assess and finds un-reconciled high-EV rungs. The asymmetry is the
    point: the gate only blocks when it has positive evidence of a naive launch.
  * Every REFUSE names the missing rung + its rel-sig + the escape hatch
    (include the flag, OR record a per-rung deferral in the launch manifest).

Escape hatch (deliberate, audited): a launch that INTENTIONALLY defers a
fire-now rung records it in a sidecar ``<run_dir>/launch_readiness_manifest.json``
(or an inline ``# LAUNCH_READINESS_DEFER:<rung>=<reason>`` comment in launch.sh)
with a non-placeholder reason. The gate consumes those and downgrades the row
from REFUSE to ACKNOWLEDGED-DEFERRED.

Not placed in ``tac`` — Claude/campaign launch-workflow apparatus, not
contest/codec logic. Sister of the #405 consume-gate
(``check_default_off_decision_table_consumed``): that gate protects
config-finalization *artifacts*; THIS gate protects the *launch* itself.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DECISION_TABLE = _REPO / ".omx/research/default_off_decision_table_20260710.jsonl"
_FIRE_NOW_DISPOSITION = "fire-now-rung(v7.5.3)"
# Only rungs whose share of the remaining descent clears this bar are
# launch-BLOCKING; lower-EV fire-now rows are advisory (WARN, never REFUSE) so
# the gate does not nag on marginal levers. Derived, not magic: 10% of the
# 0.19108->0.15 descent is ~4e-3 S, well above the exact-eval noise floor.
_BLOCKING_REL_SIG_PCT = 10.0
_DEFER_INLINE_RE = re.compile(
    r"#\s*LAUNCH_READINESS_DEFER:\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$", re.MULTILINE)
_DEFER_PLACEHOLDERS = frozenset({"", "tbd", "todo", "<reason>", "placeholder", "pending"})


@dataclass(frozen=True)
class RungExpectation:
    """One fire-now decision-table rung the launch is expected to honor."""

    name: str
    rel_sig_pct: float | None
    flag_tokens: tuple[str, ...]  # argv tokens whose presence == "included"


@dataclass
class ReadinessVerdict:
    proceed: bool
    blocking_missing: list[str] = field(default_factory=list)
    advisory_missing: list[str] = field(default_factory=list)
    acknowledged_deferred: list[str] = field(default_factory=list)
    included: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines: list[str] = []
        head = "PROCEED" if self.proceed else "REFUSE"
        lines.append(f"[witness-launch-readiness] {head}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        if self.blocking_missing:
            lines.append("  BLOCKING — fire-now rungs neither included nor deferred:")
            for m in self.blocking_missing:
                lines.append(f"    - {m}")
            lines.append(
                "  fix: add the rung's flag to launch.sh, OR record "
                "`# LAUNCH_READINESS_DEFER:<rung>=<reason>` in launch.sh "
                "(non-placeholder reason).")
        if self.advisory_missing:
            lines.append("  advisory (low-EV, not blocking): " + ", ".join(self.advisory_missing))
        if self.acknowledged_deferred:
            lines.append("  deferred-with-reason (OK): " + ", ".join(self.acknowledged_deferred))
        if self.included:
            lines.append(f"  included: {len(self.included)} fire-now rung(s) present")
        return "\n".join(lines)


def load_fire_now_expectations(table_path: Path) -> list[RungExpectation]:
    """Parse the fire-now rungs + their flag tokens from the #405 decision table.

    Each JSONL row may declare ``fire_or_gate_criterion`` and (optionally) a
    ``launch_flag_tokens`` list; when the token list is absent we fall back to a
    conservative name-derived probe so a table that predates the field still
    gates on presence-of-any-related-flag rather than silently passing."""
    rows = [
        json.loads(ln)
        for ln in table_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and '"_meta"' not in ln
    ]
    out: list[RungExpectation] = []
    for r in rows:
        if r.get("disposition") != _FIRE_NOW_DISPOSITION:
            continue
        ev = r.get("ev") or {}
        toks = tuple(r.get("launch_flag_tokens") or ())
        out.append(RungExpectation(
            name=r.get("name", "?"),
            rel_sig_pct=ev.get("rel_sig_pct"),
            flag_tokens=toks,
        ))
    return out


def _rung_is_included(exp: RungExpectation, launch_flags: set[str], launch_text: str) -> bool:
    if exp.flag_tokens:
        return any(t in launch_flags or t in launch_text for t in exp.flag_tokens)
    # Fallback: no declared tokens → look for a kebab-cased flag matching the
    # rung name (e.g. HorizonWeightedMargin → --horizon-weighted-margin*).
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", exp.name).lower()
    probe = f"--{kebab}"
    return any(f.startswith(probe) for f in launch_flags) or probe in launch_text


def decide_readiness(
    expectations: list[RungExpectation],
    launch_flags: set[str],
    launch_text: str,
    deferrals: dict[str, str],
) -> ReadinessVerdict:
    """PURE: given the fire-now expectations, the launch's flag set + raw text,
    and the {rung: reason} deferral map, return the readiness verdict."""
    v = ReadinessVerdict(proceed=True)
    if not expectations:
        v.notes.append(
            "no fire-now rungs in the decision table (or table absent) — "
            "cannot assess; proceeding (fail-open).")
        return v
    for exp in expectations:
        if _rung_is_included(exp, launch_flags, launch_text):
            v.included.append(exp.name)
            continue
        reason = deferrals.get(exp.name, "")
        if reason and reason.strip().lower() not in _DEFER_PLACEHOLDERS:
            v.acknowledged_deferred.append(f"{exp.name} ({reason.strip()[:48]})")
            continue
        rel = exp.rel_sig_pct
        label = f"{exp.name} (rel_sig={rel}%)" if rel is not None else exp.name
        if rel is not None and rel >= _BLOCKING_REL_SIG_PCT:
            v.blocking_missing.append(label)
        else:
            v.advisory_missing.append(label)
    v.proceed = not v.blocking_missing
    return v


def _read_deferrals(launch_text: str, run_dir: Path | None) -> dict[str, str]:
    d: dict[str, str] = {m.group(1): m.group(2) for m in _DEFER_INLINE_RE.finditer(launch_text)}
    if run_dir is not None:
        man = run_dir / "launch_readiness_manifest.json"
        if man.is_file():
            try:
                obj = json.loads(man.read_text(encoding="utf-8"))
                for k, val in (obj.get("deferrals") or {}).items():
                    d.setdefault(str(k), str(val))
            except (OSError, json.JSONDecodeError):
                pass
    return d


def assess_launch_readiness(
    launch_sh: Path, table_path: Path = _DECISION_TABLE
) -> ReadinessVerdict:
    """Edge: read the launch.sh + decision table, return the readiness verdict.
    Fail-open on gate-own errors (missing/unreadable table or launch.sh)."""
    try:
        launch_text = launch_sh.read_text(encoding="utf-8")
    except OSError as exc:
        v = ReadinessVerdict(proceed=True)
        v.notes.append(f"launch.sh unreadable ({exc}); proceeding (fail-open).")
        return v
    if not table_path.is_file():
        v = ReadinessVerdict(proceed=True)
        v.notes.append(f"decision table {table_path.name} absent; proceeding (fail-open).")
        return v
    try:
        expectations = load_fire_now_expectations(table_path)
    except (OSError, json.JSONDecodeError) as exc:
        v = ReadinessVerdict(proceed=True)
        v.notes.append(f"decision table unparseable ({exc}); proceeding (fail-open).")
        return v
    launch_flags = set(re.findall(r"(--[a-z0-9-]+)", launch_text))
    run_dir = launch_sh.resolve().parent
    deferrals = _read_deferrals(launch_text, run_dir)
    return decide_readiness(expectations, launch_flags, launch_text, deferrals)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launch-sh", required=True, help="path to the launch.sh to assess")
    ap.add_argument("--table", default=str(_DECISION_TABLE), help="decision-table JSONL path")
    ap.add_argument("--strict", action="store_true",
                    help="exit 3 on REFUSE (default: exit 0, print verdict — advisory)")
    a = ap.parse_args(argv)
    v = assess_launch_readiness(Path(a.launch_sh), Path(a.table))
    print(v.render())
    if a.strict and not v.proceed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
