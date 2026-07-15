#!/usr/bin/env python3
"""Wait for the canonical codex fleet to drain without false-stuck alarms.

A wall-clock timeout is not evidence that an arm is wedged.  At the deadline
this helper classifies every remaining RUNNING arm using two independent
liveness signals: a recently advancing log mtime and an advancing progress
cursor.  Healthy slow work exits non-alarmingly with an explicit status;
only arms with neither signal produce the WEDGED alarm (rc=3).

The fleet snapshot comes from :func:`codex_status.status_rows`; this module does
not hand-roll process discovery.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:  # direct script execution puts tools/ on sys.path
    import codex_status
except ModuleNotFoundError:  # package/dynamic preflight import from repo root
    from tools import codex_status

DEFAULT_TIMEOUT_SECONDS = 14_400.0
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_LIVENESS_WINDOW_SECONDS = 15.0 * 60.0

DRAINED = "DRAINED"
HEALTHY_BUT_SLOW = "HEALTHY_BUT_SLOW"
WEDGED = "WEDGED"


def _cursor_from_json(path: Path) -> int | float | None:
    """Read a monotone cursor from common progress JSON shapes."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    candidates: list[dict] = [payload]
    for key in ("progress", "state", "cursor"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for obj in candidates:
        for key in ("index", "cursor", "step", "completed", "epoch", "pass"):
            value = obj.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
    return None


def _progress_path(row: dict, overrides: dict[str, Path]) -> Path | None:
    label = str(row.get("label") or "")
    if label in overrides:
        return overrides[label]
    explicit = row.get("progress_path")
    if explicit:
        return Path(str(explicit))
    log = Path(str(row.get("log") or ""))
    stamp = str(row.get("stamp") or "")
    worktree = row.get("worktree")
    candidates = [
        log.with_suffix(".progress.json") if str(log) else None,
        log.parent / f"{label}_{stamp}.progress.json" if str(log) else None,
        log.parent / label / "progress.json" if str(log) else None,
        Path(str(worktree)) / "progress.json" if worktree else None,
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


def observe_running(
    rows: list[dict], *, progress_overrides: dict[str, Path] | None = None
) -> dict[str, dict[str, Any]]:
    """Capture activity evidence for each canonical RUNNING label+stamp."""
    overrides = progress_overrides or {}
    observations: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "RUNNING":
            continue
        label, stamp = str(row.get("label") or ""), str(row.get("stamp") or "")
        key = f"{label}_{stamp}"
        log = Path(str(row.get("log") or ""))
        progress = _progress_path(row, overrides)
        try:
            log_mtime = log.stat().st_mtime
        except OSError:
            log_mtime = None
        observations[key] = {
            "label": label,
            "stamp": stamp,
            "log": str(log),
            "log_mtime": log_mtime,
            "progress_path": str(progress) if progress else None,
            "progress_cursor": _cursor_from_json(progress) if progress else None,
        }
    return observations


def _cursor_advanced(before: object, after: object) -> bool:
    if after is None:
        return False
    if before is None and isinstance(after, (int, float)):
        return True
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after > before
    return False


def classify_timeout(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    *,
    now: float,
    liveness_window_seconds: float = DEFAULT_LIVENESS_WINDOW_SECONDS,
) -> tuple[str, list[dict[str, Any]]]:
    """Classify timeout state from concrete activity evidence.

    A remaining arm is healthy when its log was written within the liveness
    window OR its progress cursor advanced since the baseline.  Every remaining
    arm must be healthy for ``HEALTHY_BUT_SLOW``; any arm lacking both signals
    makes the result ``WEDGED``.
    """
    if not current:
        return DRAINED, []
    details: list[dict[str, Any]] = []
    any_wedged = False
    for key, after in sorted(current.items()):
        before = baseline.get(key, {})
        mtime = after.get("log_mtime")
        log_recent = isinstance(mtime, (int, float)) and 0 <= now - mtime <= liveness_window_seconds
        progress_advanced = _cursor_advanced(
            before.get("progress_cursor"), after.get("progress_cursor")
        )
        healthy = bool(log_recent or progress_advanced)
        any_wedged = any_wedged or not healthy
        details.append(
            {
                **after,
                "log_recent": bool(log_recent),
                "progress_advanced": bool(progress_advanced),
                "health": "healthy" if healthy else "wedged",
            }
        )
    return (WEDGED if any_wedged else HEALTHY_BUT_SLOW), details


def exit_code_for_status(status: str) -> int:
    """Only a genuine liveness-negative WEDGED classification is alarming."""
    return 3 if status == WEDGED else 0


def _parse_progress_overrides(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        label, sep, raw_path = value.partition("=")
        if not sep or not label or not raw_path:
            raise ValueError(f"invalid --progress {value!r}; expected LABEL=PATH")
        out[label] = Path(raw_path).expanduser().resolve()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument(
        "--liveness-window-seconds", type=float, default=DEFAULT_LIVENESS_WINDOW_SECONDS
    )
    parser.add_argument(
        "--progress", action="append", default=[], metavar="LABEL=PATH",
        help="explicit per-arm progress JSON when it is not recorded in the delegation ledger",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout_seconds < 0 or args.poll_seconds <= 0 or args.liveness_window_seconds <= 0:
        parser.error("timeout must be >=0; poll and liveness windows must be >0")
    try:
        overrides = _parse_progress_overrides(args.progress)
    except ValueError as exc:
        parser.error(str(exc))

    baseline = observe_running(codex_status.status_rows(), progress_overrides=overrides)
    deadline = time.monotonic() + args.timeout_seconds
    current = baseline
    while current and time.monotonic() < deadline:
        time.sleep(min(args.poll_seconds, max(0.0, deadline - time.monotonic())))
        current = observe_running(codex_status.status_rows(), progress_overrides=overrides)

    status, details = classify_timeout(
        baseline,
        current,
        now=time.time(),
        liveness_window_seconds=args.liveness_window_seconds,
    )
    payload = {"status": status, "remaining": len(details), "arms": details}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif status == DRAINED:
        print("DRAINED: 0 RUNNING arms remain")
    elif status == HEALTHY_BUT_SLOW:
        labels = ", ".join(item["label"] for item in details)
        print(f"HEALTHY_BUT_SLOW: {len(details)} RUNNING arm(s) exceeded the wall-clock window "
              f"but have advancing liveness evidence: {labels}")
    else:
        labels = ", ".join(item["label"] for item in details if item["health"] == "wedged")
        print(f"WEDGED: {len(details)} RUNNING arm(s) remain; no recent log or advancing progress "
              f"for: {labels}")
    return exit_code_for_status(status)


if __name__ == "__main__":
    raise SystemExit(main())
