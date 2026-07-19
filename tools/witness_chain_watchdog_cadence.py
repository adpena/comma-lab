#!/usr/bin/env python3
"""Repository-owned watchdog CADENCE around ``tools/witness_chain_watchdog.py`` (A2 of the
harness-engineering crosswalk, ``.omx/research/harness_engineering_crosswalk_20260719_codex.md``).

CLOSES ledger class ``phantom_death_buffered_log_plus_misfired_grep_liveness`` (row 59,
``owed_fix``: "cron the watchdog (~15 min cadence) so chain-dead-without-receipt is flagged
automatically"). The existing watchdog computes a COMPOSITE liveness verdict (pid-tree ×
file-mtimes × receipt) — this wrapper gives it a durable, idempotent, cron-able CADENCE:

  * runs ``witness_chain_watchdog.scan()`` once per invocation (cron supplies the ~15-min
    cadence: ``*/15 * * * * .venv/bin/python tools/witness_chain_watchdog_cadence.py``);
  * writes a durable receipt row per run to ``.omx/state/watchdog_cadence_receipts.jsonl``;
  * ALERTS (rc=2) ONLY on a genuine ``CHAIN_DEAD_NO_RECEIPT`` verdict — buffered-log silence
    (``RUNNING_QUIET``) must NEVER trigger death (that is the exact phantom-death class);
  * IDEMPOTENT deduplication: the same observation (stable per (label,pid,verdict) hash)
    is actioned ONCE — replaying an identical scan emits a receipt but takes no duplicate
    alert action, so a cron loop does not re-fire on a still-dead chain every 15 min.

Exit codes: 0 = all healthy / no NEW actionable death; 2 = a NEW chain-dead-no-receipt was
detected and actioned this run; 3 = the underlying registry was unreadable.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import witness_chain_watchdog as _wcw  # noqa: E402

DEFAULT_RECEIPTS_PATH = _REPO / ".omx" / "state" / "watchdog_cadence_receipts.jsonl"

#: The only verdict that is a genuine silent death worth alerting on. Everything else —
#: including RUNNING_QUIET (buffered log frozen but process ALIVE) — is explicitly NOT death.
_ALERT_VERDICTS = frozenset({"CHAIN_DEAD_NO_RECEIPT"})
#: Verdicts that mean "measured alive / benign" — never an alert (kills the phantom-death class).
_BENIGN_VERDICTS = frozenset({
    "RUNNING_HEALTHY", "RUNNING_QUIET", "CHAIN_DEAD_RECEIPTED", "NO_RUN_DIR",
})


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _observation_key(verdict_row: dict) -> str:
    """Stable identity of an actionable observation for idempotent dedup."""
    label = str(verdict_row.get("label") or "")
    pid = str(verdict_row.get("pid") or "")
    verdict = str(verdict_row.get("verdict") or "")
    run_dir = str(verdict_row.get("run_dir") or "")
    raw = f"{label}\x00{pid}\x00{verdict}\x00{run_dir}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _already_actioned(receipts_path: Path) -> set[str]:
    """Observation keys already ALERTED in a prior receipt (idempotency memory)."""
    seen: set[str] = set()
    if not receipts_path.exists():
        return seen
    for ln in receipts_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            continue
        for key in row.get("actioned_keys") or []:
            seen.add(str(key))
    return seen


def _append_receipt(receipts_path: Path, receipt: dict) -> None:
    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    lock = receipts_path.with_name("." + receipts_path.name + ".lock")
    line = json.dumps(receipt, sort_keys=True, allow_nan=False)
    with lock.open("a") as lockfh:
        fcntl.flock(lockfh.fileno(), fcntl.LOCK_EX)
        try:
            with receipts_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        finally:
            fcntl.flock(lockfh.fileno(), fcntl.LOCK_UN)


def run_cadence(
    *,
    stale_s: float = 900.0,
    registry_path: Path | None = None,
    manifest_path: Path | None = None,
    receipts_path: Path | None = None,
    write_receipt: bool = True,
) -> dict:
    """One watchdog cadence tick. Returns a receipt dict; appends it when ``write_receipt``.

    ``new_actions`` lists deaths actioned FOR THE FIRST TIME this run (idempotent — a death
    already actioned in a prior receipt is reported under ``suppressed_duplicates``)."""
    receipts = receipts_path or DEFAULT_RECEIPTS_PATH
    verdicts = _wcw.scan(stale_s=stale_s, registry_path=registry_path,
                         manifest_path=manifest_path)
    unreadable = any(v.get("verdict") == "REGISTRY_UNREADABLE" for v in verdicts)

    already = _already_actioned(receipts)
    alerts = [v for v in verdicts if v.get("verdict") in _ALERT_VERDICTS]
    new_actions, suppressed = [], []
    for v in alerts:
        key = _observation_key(v)
        (suppressed if key in already else new_actions).append(
            {"key": key, "label": v.get("label"), "pid": v.get("pid"),
             "verdict": v.get("verdict"), "run_dir": v.get("run_dir")}
        )

    receipt = {
        "ts": _utc(),
        "cadence": "watchdog",
        "n_verdicts": len(verdicts),
        "registry_unreadable": unreadable,
        "alert_count": len(alerts),
        "new_action_count": len(new_actions),
        "suppressed_duplicate_count": len(suppressed),
        "new_actions": new_actions,
        "suppressed_duplicates": suppressed,
        "actioned_keys": [a["key"] for a in new_actions],
        "benign_verdicts": sorted({
            str(v.get("verdict")) for v in verdicts
            if v.get("verdict") in _BENIGN_VERDICTS
        }),
    }
    if write_receipt:
        _append_receipt(receipts, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stale-s", type=float, default=900.0,
                    help="mtime freshness threshold passed to the watchdog scan")
    ap.add_argument("--json", action="store_true", help="print the receipt as JSON")
    ap.add_argument("--no-receipt", action="store_true",
                    help="do not append a durable receipt (dry inspection)")
    args = ap.parse_args(argv)

    receipt = run_cadence(stale_s=args.stale_s, write_receipt=not args.no_receipt)
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"watchdog cadence @ {receipt['ts']}: {receipt['n_verdicts']} verdict(s), "
              f"{receipt['new_action_count']} NEW chain-dead action(s), "
              f"{receipt['suppressed_duplicate_count']} duplicate(s) suppressed"
              + (" [REGISTRY UNREADABLE]" if receipt["registry_unreadable"] else ""))
        for a in receipt["new_actions"]:
            print(f"  SILENT DEATH actioned: {a['label']} pid={a['pid']} ({a['run_dir']})")

    if receipt["registry_unreadable"]:
        return 3
    return 2 if receipt["new_action_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
