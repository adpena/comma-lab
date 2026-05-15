# SPDX-License-Identifier: MIT
"""Catalog #284 - canonical nightly catalog-gate regression CI helper.

Per OMNIBUS-BUG-CLASS-AUDIT GAP-3 (`feedback_omnibus_bug_class_audit_landed_20260515.md`):
runs every catalog gate's tests + every empirical bug fixture nightly with
<24h drift detection. Posts a daily green/red signal so an agent + operator
sees regression within one day, not multi-day "ad-hoc subagent discovery"
windows.

Operator standing directive 2026-05-15: *"fix all bug classes and meta bugs
and all versions permanently and self protect against them"*.

This module is the canonical entry point invoked by
``.github/workflows/nightly_catalog_gate_regression.yml``. The accompanying
STRICT preflight gate ``check_nightly_catalog_gate_regression_workflow_canonical_use``
(Catalog #284) refuses any modification to the workflow file that does not
route through this helper.

Per CLAUDE.md "Subagent coherence-by-default" + Catalog #110/#113:
- the per-day status JSON written under ``.omx/state/`` is HISTORICAL_PROVENANCE
  (one file per day; APPEND-ONLY in the sense of "never mutate prior day"); a
  rerun of the same UTC day MAY rewrite the same-day file with a new ``run_id``
  but historical days remain immutable.
- per-day status JSONs land in
  ``.omx/state/nightly_catalog_gate_regression_<YYYYMMDD>.json`` and are written
  via fcntl-locked atomic ``.tmp.<uuid12>`` + ``os.replace``.

Composition:

1. ``run_preflight()`` invokes ``.venv/bin/python -m tac.preflight`` and parses
   the rc + stdout to a structured summary.
2. ``run_pytest()`` invokes ``.venv/bin/python -m pytest src/tac/tests/`` with
   ``--maxfail=20 --tb=short`` and parses the trailing pytest summary line into
   counts (passed / failed / skipped).
3. ``run_meta_meta_drift_checks()`` invokes Catalogs #118 / #159 / #176 / #185
   directly via Python (no subprocess) and reports any drift between
   CLAUDE.md catalog text and live preflight strict-mode behavior.
4. ``write_status_json()`` aggregates the three results into the canonical
   v1 status schema and persists to ``.omx/state/`` with fcntl locking.
5. ``main()`` orchestrates all 3 + writes JSON + exits with rc=0 on full
   green / rc=1 on any drift.

CLI flags:
- ``--repo-root <path>`` override (default: cwd / git toplevel)
- ``--run-id <text>`` override the run id stamped into status JSON
  (default: GitHub Actions ``$GITHUB_RUN_ID`` or ``local-<utc>``)
- ``--skip-pytest`` skip the pytest stage (preflight + meta-meta only;
  for fast preflight-only rerun)
- ``--strict`` exit non-zero on any drift (default; mirrors CI semantics)
- ``--no-strict`` always exit rc=0 (advisory mode for local dev runs)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_STATUS_SCHEMA_VERSION = "nightly_catalog_gate_regression_status_v1"

_NIGHTLY_STATE_DIR_RELATIVE = Path(".omx/state")

_CANONICAL_LOCK_PATH_NAME = ".nightly_catalog_gate_regression.lock"

# Pytest summary line captures e.g. "5 passed, 2 failed, 3 skipped in 1.23s".
_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s*"
    r"(?:(\d+)\s+failed,?\s*)?"
    r"(?:(\d+)\s+passed,?\s*)?"
    r"(?:(\d+)\s+skipped,?\s*)?"
    r"(?:(\d+)\s+errors?,?\s*)?"
    r"(?:(\d+)\s+xfailed,?\s*)?"
    r"(?:(\d+)\s+xpassed,?\s*)?"
    r"(?:(\d+)\s+warnings?,?\s*)?"
    r"in\s+[\d.]+\s*s?"
    r"\s*=+",
    re.IGNORECASE,
)

# Conservative META-meta gate roster; deliberate overlap with CLAUDE.md
# Catalog #118/#159/#176/#185 — the "no-drift-in-the-strictness-ledger"
# gates the operator depends on for the daily green/red signal.
_META_META_GATE_NAMES: tuple[str, ...] = (
    "check_claude_md_catalog_no_duplicate_numbers",  # #118
    "check_claude_md_catalog_text_matches_preflight_strict_value",  # #159
    "check_strict_preflight_callsites_have_claude_md_catalog_row",  # #176
    "check_strict_flipped_catalog_entries_have_live_count_zero",  # #185
)


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _utc_iso(now: _dt.datetime | None = None) -> str:
    return (now or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_yyyymmdd(now: _dt.datetime | None = None) -> str:
    return (now or _utc_now()).strftime("%Y%m%d")


def _resolve_repo_root(repo_root: Path | str | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    # Try git rev-parse --show-toplevel; fall back to cwd.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip()).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path.cwd().resolve()


def _resolve_python_bin(repo_root: Path) -> str:
    """Return the venv python binary if available, else the running interpreter."""
    venv_py = repo_root / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


def _resolve_run_id(override: str | None) -> str:
    if override:
        return override
    gha_run = os.environ.get("GITHUB_RUN_ID", "")
    if gha_run:
        return f"gha-{gha_run}"
    return f"local-{_utc_iso().replace(':', '').replace('-', '')}"


def run_preflight(
    repo_root: Path,
    *,
    timeout_seconds: int = 600,
) -> tuple[bool, dict[str, Any]]:
    """Invoke ``tac.preflight`` and return (passed, summary)."""
    py = _resolve_python_bin(repo_root)
    env = os.environ.copy()
    pythonpath_parts = [
        str(repo_root / "src"),
        str(repo_root / "upstream"),
        str(repo_root),
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*pythonpath_parts, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    started = _utc_now()
    try:
        proc = subprocess.run(
            [py, "-m", "tac.preflight"],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timed_out = True
    finished = _utc_now()
    elapsed = (finished - started).total_seconds()
    passed = rc == 0 and not timed_out
    summary = {
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "started_at_utc": _utc_iso(started),
        "finished_at_utc": _utc_iso(finished),
    }
    return passed, summary


def _parse_pytest_summary(stdout: str) -> dict[str, int]:
    """Parse the trailing pytest summary line into integer counts.

    Returns a dict with at least the keys passed/failed/skipped/errors. Missing
    sections default to 0.
    """
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "warnings": 0,
    }
    if not stdout:
        return counts
    # Find LAST match — pytest prints multiple "==" lines.
    last = None
    for m in _PYTEST_SUMMARY_RE.finditer(stdout):
        last = m
    if last is None:
        return counts
    g = last.groups()
    counts["failed"] = int(g[0] or 0)
    counts["passed"] = int(g[1] or 0)
    counts["skipped"] = int(g[2] or 0)
    counts["errors"] = int(g[3] or 0)
    counts["xfailed"] = int(g[4] or 0)
    counts["xpassed"] = int(g[5] or 0)
    counts["warnings"] = int(g[6] or 0)
    return counts


def run_pytest(
    repo_root: Path,
    *,
    timeout_seconds: int = 1800,
    maxfail: int = 20,
) -> tuple[bool, dict[str, Any]]:
    """Invoke pytest over canonical test paths, return (passed, summary)."""
    py = _resolve_python_bin(repo_root)
    env = os.environ.copy()
    pythonpath_parts = [
        str(repo_root / "src"),
        str(repo_root / "upstream"),
        str(repo_root),
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [*pythonpath_parts, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    cmd = [
        py,
        "-m",
        "pytest",
        "src/tac/tests/",
        f"--maxfail={maxfail}",
        "--tb=short",
        "-q",
        "--no-header",
        "-m",
        "not slow",
    ]
    started = _utc_now()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        rc = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = -1
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timed_out = True
    finished = _utc_now()
    elapsed = (finished - started).total_seconds()
    counts = _parse_pytest_summary(stdout)
    # Pytest rc 0 = all pass; rc 5 = no tests collected; anything else = fail.
    passed = rc == 0 and not timed_out
    summary = {
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "counts": counts,
        "total_tests": (
            counts["passed"]
            + counts["failed"]
            + counts["skipped"]
            + counts["errors"]
        ),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "started_at_utc": _utc_iso(started),
        "finished_at_utc": _utc_iso(finished),
    }
    return passed, summary


def run_meta_meta_drift_checks(
    repo_root: Path,
) -> tuple[bool, dict[str, Any]]:
    """Invoke META-meta drift gates directly (in-process, no subprocess).

    Per Catalog #118/#159/#176/#185 these are the canonical strictness-ledger
    drift detectors. Returns (no_drift, per-gate detail).
    """
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from tac import preflight as _preflight  # type: ignore[import-not-found]
    except Exception as exc:
        return False, {
            "import_error": repr(exc),
            "no_drift": False,
            "per_gate": {},
        }
    per_gate: dict[str, Any] = {}
    no_drift = True
    for gate_name in _META_META_GATE_NAMES:
        fn = getattr(_preflight, gate_name, None)
        if fn is None:
            per_gate[gate_name] = {
                "status": "missing",
                "violations": [],
                "violation_count": -1,
            }
            no_drift = False
            continue
        try:
            violations = fn(strict=False, verbose=False)
        except TypeError:
            try:
                violations = fn(strict=False)
            except Exception as exc:
                per_gate[gate_name] = {
                    "status": "error",
                    "exception": repr(exc),
                    "violations": [],
                    "violation_count": -1,
                }
                no_drift = False
                continue
        except Exception as exc:
            per_gate[gate_name] = {
                "status": "error",
                "exception": repr(exc),
                "violations": [],
                "violation_count": -1,
            }
            no_drift = False
            continue
        if not isinstance(violations, list):
            violations = [str(violations)]
        per_gate[gate_name] = {
            "status": "ok" if not violations else "drift",
            "violations": violations[:20],
            "violation_count": len(violations),
        }
        if violations:
            no_drift = False
    return no_drift, {"no_drift": no_drift, "per_gate": per_gate}


@contextlib.contextmanager
def _fcntl_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def write_status_json(
    repo_root: Path,
    *,
    run_id: str,
    timestamp_utc: str,
    preflight_passed: bool,
    preflight_summary: dict[str, Any],
    pytest_skipped: bool,
    pytest_passed: bool | None,
    pytest_summary: dict[str, Any] | None,
    meta_meta_no_drift: bool,
    meta_meta_summary: dict[str, Any],
) -> Path:
    """Write the canonical per-day status JSON under .omx/state/.

    Returns the absolute path written.
    """
    now = _utc_now()
    yyyymmdd = _utc_yyyymmdd(now)
    state_dir = repo_root / _NIGHTLY_STATE_DIR_RELATIVE
    state_dir.mkdir(parents=True, exist_ok=True)
    out_path = state_dir / f"nightly_catalog_gate_regression_{yyyymmdd}.json"
    lock_path = state_dir / _CANONICAL_LOCK_PATH_NAME
    pytest_counts = (pytest_summary or {}).get("counts", {})
    payload = {
        "schema_version": _STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "host": socket.gethostname(),
        "preflight_pass": bool(preflight_passed),
        "pytest_skipped": bool(pytest_skipped),
        "pytest_pass": (
            None if pytest_skipped else (pytest_passed is True and not pytest_skipped)
        ),
        "meta_meta_drift": not bool(meta_meta_no_drift),
        "total_tests": pytest_counts.get("passed", 0)
        + pytest_counts.get("failed", 0)
        + pytest_counts.get("skipped", 0)
        + pytest_counts.get("errors", 0),
        "passed": pytest_counts.get("passed", 0),
        "failed": pytest_counts.get("failed", 0),
        "skipped": pytest_counts.get("skipped", 0),
        "errors": pytest_counts.get("errors", 0),
        "preflight_summary": {
            "rc": preflight_summary.get("rc"),
            "timed_out": preflight_summary.get("timed_out"),
            "elapsed_seconds": preflight_summary.get("elapsed_seconds"),
            "stdout_tail_len": len(preflight_summary.get("stdout_tail") or ""),
            "stderr_tail_len": len(preflight_summary.get("stderr_tail") or ""),
            "stdout_tail": preflight_summary.get("stdout_tail"),
            "stderr_tail": preflight_summary.get("stderr_tail"),
        },
        "pytest_summary": (
            None
            if pytest_skipped
            else {
                "rc": (pytest_summary or {}).get("rc"),
                "timed_out": (pytest_summary or {}).get("timed_out"),
                "elapsed_seconds": (pytest_summary or {}).get("elapsed_seconds"),
                "counts": pytest_counts,
                "stdout_tail": (pytest_summary or {}).get("stdout_tail"),
                "stderr_tail": (pytest_summary or {}).get("stderr_tail"),
            }
        ),
        "meta_meta_summary": meta_meta_summary,
        "overall_pass": bool(
            preflight_passed
            and (pytest_skipped or (pytest_passed is True))
            and meta_meta_no_drift
        ),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    with _fcntl_lock(lock_path):
        tmp.write_text(serialized + "\n", encoding="utf-8")
        os.replace(tmp, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_nightly_catalog_gate_regression",
        description=(
            "Catalog #284 — canonical nightly catalog-gate regression CI helper. "
            "Per OMNIBUS GAP-3: runs preflight + pytest + META-meta drift "
            "checks; emits a per-day status JSON under .omx/state/."
        ),
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit non-zero on any drift (default).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always exit rc=0 (advisory mode for local dev runs).",
    )
    parser.add_argument(
        "--preflight-timeout-seconds", type=int, default=600,
    )
    parser.add_argument(
        "--pytest-timeout-seconds", type=int, default=1800,
    )
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repo_root)
    run_id = _resolve_run_id(args.run_id)
    started_iso = _utc_iso()

    print(f"[nightly] run_id={run_id} repo_root={repo_root} started_at_utc={started_iso}")

    preflight_passed, preflight_summary = run_preflight(
        repo_root, timeout_seconds=args.preflight_timeout_seconds
    )
    print(f"[nightly] preflight rc={preflight_summary['rc']} pass={preflight_passed}")

    if args.skip_pytest:
        pytest_skipped = True
        pytest_passed: bool | None = None
        pytest_summary: dict[str, Any] | None = None
        print("[nightly] pytest SKIPPED (--skip-pytest)")
    else:
        pytest_skipped = False
        pytest_passed, pytest_summary = run_pytest(
            repo_root, timeout_seconds=args.pytest_timeout_seconds
        )
        counts = pytest_summary.get("counts", {})
        print(
            f"[nightly] pytest rc={pytest_summary['rc']} pass={pytest_passed} "
            f"counts={counts}"
        )

    no_drift, meta_meta_summary = run_meta_meta_drift_checks(repo_root)
    print(f"[nightly] meta_meta no_drift={no_drift}")

    out_path = write_status_json(
        repo_root,
        run_id=run_id,
        timestamp_utc=started_iso,
        preflight_passed=preflight_passed,
        preflight_summary=preflight_summary,
        pytest_skipped=pytest_skipped,
        pytest_passed=pytest_passed,
        pytest_summary=pytest_summary,
        meta_meta_no_drift=no_drift,
        meta_meta_summary=meta_meta_summary,
    )
    print(f"[nightly] status written: {out_path.relative_to(repo_root)}")

    overall_pass = preflight_passed and (
        pytest_skipped or (pytest_passed is True)
    ) and no_drift
    print(f"[nightly] overall_pass={overall_pass}")

    if not overall_pass and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
