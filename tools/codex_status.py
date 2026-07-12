#!/usr/bin/env python3
"""codex_status.py — show the status of codex delegations launched via codex_delegate.py.

Reads the ledger (.omx/state/codex_delegations.jsonl), checks each run's .done marker
and whether its codex process is still alive, and prints a table. Marks rows done when
their marker appears. --json for machine-readable.

With --classify, each DONE run's final message (`.last.txt`) is classified SEMANTICALLY
via fmtools.local_extract (our on-device Apple FM, structured generation against a closed
schema) into {landed_result, landed_with_blocker, stalled_no_result, errored} + a one-line
reason + did-it-commit — instead of brittle regex over the log. fmtools lives in its own
venv (~/Projects/fmtools/.venv/bin/python); run --classify from there, or the base venv
falls back to an honest "fm-unavailable" outcome (never a faked classification).

USAGE:  .venv/bin/python tools/codex_status.py [--json]
        ~/Projects/fmtools/.venv/bin/python tools/codex_status.py --classify [--json]
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


def _classify_outcome(last_text: str) -> dict:
    """Classify one codex agent's FINAL MESSAGE into a structured outcome via the
    on-device Apple FM (fmtools.local_extract, structured generation against a closed
    schema). Returns {"ok": True, "outcome", "reason", "committed"} on success, or an
    honest {"ok": False, "why": ...} when fmtools/apple_fm_sdk is unavailable (base
    venv) or the model errors — NEVER a faked/regex label. Mirrors the
    tools/dashboard_fm_events.py pattern (lazy import, guardrail-tolerant prose input)."""
    try:
        import asyncio

        import apple_fm_sdk as fm
        from fmtools import local_extract
    except Exception as exc:  # base venv has no fmtools — honest skip, not a fake label
        return {"ok": False, "why": f"fm-unavailable ({type(exc).__name__})"}

    @fm.generable()
    class CodexOutcome:
        outcome: str = fm.guide(anyOf=[
            "landed_result", "landed_with_blocker", "stalled_no_result", "errored"])
        reason: str = fm.guide(
            description="One short plain sentence: what the agent accomplished, or "
                        "why it stopped. Use only facts present in the message; never "
                        "invent numbers or outcomes.")
        committed: str = fm.guide(anyOf=["yes", "no", "unknown"])

    _instructions = (
        "You label the FINAL MESSAGE of a coding agent that was delegated a task. "
        "outcome: 'landed_result' if it reports finishing and landing/committing real "
        "work; 'landed_with_blocker' if it did partial work but names an unresolved "
        "blocker; 'stalled_no_result' if it ran but produced no usable landing; "
        "'errored' if the message itself reports a crash/error/refusal. reason: one "
        "short plain sentence using ONLY facts in the message. committed: 'yes' if the "
        "message clearly states it committed/landed code, 'no' if it clearly did not, "
        "else 'unknown'.")

    @local_extract(CodexOutcome, retries=2, instructions=_instructions)
    async def _classify(msg: str) -> CodexOutcome:
        """(instructions provided explicitly above)"""

    # prose-frame + cap: the tail carries the verdict; a dense head can trip the guardrail
    body = last_text.strip()[-1800:] or "(empty final message)"
    try:
        r = asyncio.run(_classify(f"The delegated coding agent's final message was: {body}"))
    except Exception as exc:
        return {"ok": False, "why": f"fm-error ({type(exc).__name__})"}
    return {
        "ok": True,
        "outcome": str(getattr(r, "outcome", "stalled_no_result")),
        "reason": str(getattr(r, "reason", ""))[:200],
        "committed": str(getattr(r, "committed", "unknown")),
        "classifier": "apple-fm-on-device",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--classify", action="store_true",
                    help="classify each DONE run's final message via fmtools on-device FM "
                         "(run from the fmtools venv; base venv reports fm-unavailable)")
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

    for d in seen.values():
        done = _read_done(d.get("done_marker", ""))
        alive = _alive(d.get("label", ""), d.get("stamp", ""))
        status = "DONE" if done else ("RUNNING" if alive else "UNKNOWN(no-marker,no-proc)")
        row = {
            "label": d.get("label"), "stamp": d.get("stamp"),
            "model": d.get("model"), "effort": d.get("effort"),
            "status": status, "rc": (done or {}).get("rc"),
            "finished_utc": (done or {}).get("finished_utc"),
            "launched_utc": d.get("launched_utc"), "log": d.get("log"),
        }
        if args.classify and done:  # semantic outcome only makes sense once finished
            last_path = Path(d.get("last", "") or "")
            if last_path.is_file():
                row["outcome"] = _classify_outcome(
                    last_path.read_text(encoding="utf-8", errors="ignore"))
            else:
                row["outcome"] = {"ok": False, "why": "no-last-txt"}
        rows.append(row)

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
        print(f"{(r['label'] or '?'):<24} {r['status']:<26} {r.get('rc') or '-'!s:<4} {me:<22} {r.get('launched_utc') or '?'}")
        oc = r.get("outcome")  # present only under --classify on DONE runs
        if oc:
            if oc.get("ok"):
                print(f"    ↳ {oc['outcome']} (committed={oc.get('committed', '?')}) — {oc.get('reason', '')}")
            else:
                print(f"    ↳ [not classified: {oc.get('why', '?')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
