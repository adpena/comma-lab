#!/usr/bin/env python3
"""codex_status.py — show the status of codex delegations launched via codex_delegate.py.

Reads the ledger (.omx/state/codex_delegations.jsonl), checks each run's .done marker
and whether its codex process is still alive, and prints a table. Marks rows done when
their marker appears. --json for machine-readable.

USAGE:  .venv/bin/python tools/codex_status.py [--json]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / ".omx" / "state" / "codex_delegations.jsonl"


def _alive(label: str, stamp: str) -> bool:
    # a codex whose -o path contains "<label>_<stamp>" is this run's process
    r = subprocess.run(["pgrep", "-f", f"{label}_{stamp}"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _read_done(marker: str) -> dict | None:
    p = Path(marker)
    if not p.is_file():
        return None
    out: dict = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not LEDGER.is_file():
        print("(no codex delegations yet)")
        return 0

    rows: list[dict] = []
    # latest ledger row per label+stamp wins
    seen: dict[str, dict] = {}
    for line in LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen[f"{d.get('label')}_{d.get('stamp')}"] = d

    for key, d in seen.items():
        done = _read_done(d.get("done_marker", ""))
        alive = _alive(d.get("label", ""), d.get("stamp", ""))
        status = "DONE" if done else ("RUNNING" if alive else "UNKNOWN(no-marker,no-proc)")
        rows.append({
            "label": d.get("label"), "stamp": d.get("stamp"),
            "model": d.get("model"), "effort": d.get("effort"),
            "status": status, "rc": (done or {}).get("rc"),
            "finished_utc": (done or {}).get("finished_utc"),
            "launched_utc": d.get("launched_utc"), "log": d.get("log"),
        })

    rows.sort(key=lambda r: r.get("launched_utc") or "")
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("(no codex delegations yet)")
        return 0
    print(f"{'LABEL':<24} {'STATUS':<26} {'RC':<4} {'MODEL/EFFORT':<22} LAUNCHED")
    for r in rows:
        me = f"{r.get('model') or '?'}/{r.get('effort') or '?'}"
        print(f"{(r['label'] or '?'):<24} {r['status']:<26} {str(r.get('rc') or '-'):<4} {me:<22} {r.get('launched_utc') or '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
