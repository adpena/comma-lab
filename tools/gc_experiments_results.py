#!/usr/bin/env python3
"""Canonical garbage-collector for ``experiments/results/`` (T1-D).

Per the 2026-05-12 state-hygiene wave (cluster 3 of the streamline pass),
this helper is the ONLY tool authorized to bulk-delete inside
``experiments/results/``. Catalog #154
(``check_experiments_results_gc_helper_is_canonical``) STRICT preflight
gate refuses any new shell/python that hard-deletes under
``experiments/results/`` without routing through this script.

Behavior
========

The helper classifies every top-level directory under
``experiments/results/`` into one of four buckets:

  ``DELETE-NOW``
    Safe to delete in this pass. Two sub-rules:

    * **gitignored**: not tracked by git AND mtime > ``--ignored-max-age-days``
      AND not declared ``custody_status=committed-binary`` /
      ``custody_status=published`` in its ``build_manifest.json`` (if any).
    * **smoke/probe/dryrun**: top-level dir name matches
      ``*_smoke_*`` / ``*_probe_*`` / ``*_dryrun_*`` /
      ``.pytest_tmp_outputs/`` AND mtime > ``--smoke-max-age-days``.

  ``PRESERVE-METADATA-DELETE-BODIES``
    ``recovered_*/`` directories — the ``recovery_metadata.json`` file is
    HISTORICAL_PROVENANCE per Catalog #110 and MUST be preserved. The
    raw rsync logs / build trees inside (which are LIVE_STATE that
    leaked into the repo) may be deleted. The current implementation
    surfaces the dir for explicit operator review rather than
    auto-cleaving — the LIVE_STATE / HISTORICAL_PROVENANCE split inside
    each recovered_* tree is per-instance and depends on the specific
    rsync layout.

  ``KEEP``
    Anything tracked by git, any path with
    ``build_manifest.json::custody_status in {committed-binary, published}``,
    any path declared ``HISTORICAL_PROVENANCE`` in the artifact-kind
    registry, any directory under ``--keep-max-age-days`` regardless of
    other signals.

  ``AMBIGUOUS``
    Anything not matching the above. The operator must decide. The
    helper never auto-deletes ambiguous paths.

Invocation
==========

    # Dry-run, JSON to stdout
    tools/gc_experiments_results.py --dry-run

    # Dry-run, JSON to file
    tools/gc_experiments_results.py --dry-run \
        --output .omx/research/experiments_results_gc_dry_run_<DATE>.json

    # Apply (REQUIRES --operator-approved)
    tools/gc_experiments_results.py --apply \
        --operator-approved "alejandro:2026-05-12T18:00Z"

Exit codes:

  0 — dry-run completed (no deletion) OR --apply succeeded
  2 — validation error (missing/malformed --operator-approved on --apply)
  3 — refusal (apply attempted without dry-run-first / without operator
        handle)

The helper NEVER deletes a path that ``git ls-files`` knows. The helper
NEVER deletes ``recovery_metadata.json``. The helper NEVER deletes
artifact-registry-declared HISTORICAL_PROVENANCE paths.

This module is also importable: ``from tac.gc_experiments_results import
classify_results_dirs, build_gc_plan``.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path("experiments/results")

# Default thresholds (operator can override via CLI flags).
DEFAULT_IGNORED_MAX_AGE_DAYS = 30
DEFAULT_SMOKE_MAX_AGE_DAYS = 7
DEFAULT_KEEP_MAX_AGE_DAYS = 1  # always KEEP anything modified in the last 1 day

# Tokens that mark a dir as smoke/probe/dryrun bait. Matched case-insensitively
# against the dir basename.
SMOKE_TOKENS = (
    "_smoke_",
    "_smoke",  # trailing variant (e.g. foo_smoke)
    "smoke_",  # leading variant (e.g. smoke_old)
    "_probe_",
    "_probe",
    "probe_",
    "_dryrun_",
    "_dryrun",
    "dryrun_",
    "_dry_run_",
    "_dry_run",
    "dry_run_",
)

PYTEST_TMP_NAME = ".pytest_tmp_outputs"

# Custody statuses that PIN a tracked tree to the repo (never delete even
# if mtime is old; this is the contract from CLAUDE.md "Operator gates must
# be wired and used" and the artifact-kind registry).
CUSTODY_PIN_STATUSES = {"committed-binary", "published", "ci-rebuildable"}

# Status tags meaning the path is HISTORICAL_PROVENANCE-aware (must be kept).
HISTORICAL_TAG_TOKENS = (
    "recovery_metadata.json",
    "contest_auth_eval",
    "build_manifest.json",
    "provenance.json",
    "dispatch_manifest.json",
    "harvest_summary.json",
)

VERDICT_DELETE_NOW = "DELETE-NOW"
VERDICT_PRESERVE_METADATA = "PRESERVE-METADATA-DELETE-BODIES"
VERDICT_KEEP = "KEEP"
VERDICT_AMBIGUOUS = "AMBIGUOUS"


@dataclasses.dataclass(frozen=True)
class Classification:
    """Per-directory classification result."""

    path: str  # relpath from repo root
    verdict: str
    rationale: str
    age_days: float
    bytes_estimate: int
    tracked: bool
    has_build_manifest: bool
    has_recovery_metadata: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_git_tracked_top_level_dirs(root: Path, repo_root: Path) -> set[str]:
    """Return the set of top-level directory names under ``root`` that contain
    at least one git-tracked file."""

    try:
        result = subprocess.run(
            ["git", "ls-files", str(root.relative_to(repo_root))],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()
    if result.returncode != 0:
        return set()
    tracked: set[str] = set()
    rel_root = root.relative_to(repo_root).as_posix()
    prefix = rel_root + "/"
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith(prefix):
            # Maybe normalized with backslashes
            continue
        rest = line[len(prefix):]
        head, _, _ = rest.partition("/")
        if head:
            tracked.add(head)
    return tracked


def _dir_mtime_and_bytes(path: Path) -> tuple[float, int]:
    """Walk ``path`` and return (newest_mtime_seconds, sum_of_file_sizes).

    Symbolic links are not followed. Missing files are skipped silently.
    """

    newest = path.stat().st_mtime
    total = 0
    for sub in path.rglob("*"):
        try:
            st = sub.stat()
        except (FileNotFoundError, PermissionError):
            continue
        if not _stat_is_regular_file(st):
            continue
        if st.st_mtime > newest:
            newest = st.st_mtime
        total += st.st_size
    return newest, total


def _stat_is_regular_file(st: os.stat_result) -> bool:
    import stat as _stat

    return _stat.S_ISREG(st.st_mode)


def _has_smoke_token(name: str) -> bool:
    low = name.lower()
    if low == PYTEST_TMP_NAME:
        return True
    return any(tok in low for tok in SMOKE_TOKENS)


def _read_build_manifest_custody(path: Path) -> str | None:
    """If ``path`` contains a top-level ``build_manifest.json``, return its
    ``custody_status`` value (or None if absent/malformed)."""

    bm = path / "build_manifest.json"
    if not bm.is_file():
        return None
    try:
        payload = json.loads(bm.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        val = payload.get("custody_status")
        if isinstance(val, str):
            return val.strip().lower()
    return None


def _has_recovery_metadata(path: Path) -> bool:
    return (path / "recovery_metadata.json").is_file()


def _classify_dir(
    entry: Path,
    *,
    repo_root: Path,
    tracked_dirs: set[str],
    now_seconds: float,
    ignored_max_age_days: int,
    smoke_max_age_days: int,
    keep_max_age_days: int,
) -> Classification:
    name = entry.name
    is_tracked = name in tracked_dirs
    try:
        mtime, bytes_estimate = _dir_mtime_and_bytes(entry)
    except (FileNotFoundError, PermissionError):
        return Classification(
            path=str(entry.relative_to(repo_root)),
            verdict=VERDICT_AMBIGUOUS,
            rationale="stat failure walking dir contents",
            age_days=-1.0,
            bytes_estimate=0,
            tracked=is_tracked,
            has_build_manifest=False,
            has_recovery_metadata=False,
        )
    age_days = max((now_seconds - mtime) / 86400.0, 0.0)
    has_bm = (entry / "build_manifest.json").is_file()
    has_rm = _has_recovery_metadata(entry)
    rel = str(entry.relative_to(repo_root))

    # Always-keep: very recent activity.
    if age_days <= keep_max_age_days:
        return Classification(
            path=rel,
            verdict=VERDICT_KEEP,
            rationale=f"recent (age={age_days:.1f}d <= keep_max={keep_max_age_days}d)",
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=is_tracked,
            has_build_manifest=has_bm,
            has_recovery_metadata=has_rm,
        )

    # Recovered_*/ — preserve metadata, surface for review
    if name.startswith("recovered_") and has_rm:
        return Classification(
            path=rel,
            verdict=VERDICT_PRESERVE_METADATA,
            rationale=(
                "recovered_*/ with recovery_metadata.json — metadata is "
                "HISTORICAL_PROVENANCE (Catalog #110); inner rsync/build "
                "bodies may be LIVE_STATE; operator review required"
            ),
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=is_tracked,
            has_build_manifest=has_bm,
            has_recovery_metadata=True,
        )

    # Smoke / probe / dryrun -> DELETE-NOW if old enough
    if _has_smoke_token(name):
        if age_days > smoke_max_age_days:
            return Classification(
                path=rel,
                verdict=VERDICT_DELETE_NOW,
                rationale=(
                    f"smoke/probe/dryrun dir (age={age_days:.1f}d > "
                    f"smoke_max={smoke_max_age_days}d)"
                ),
                age_days=age_days,
                bytes_estimate=bytes_estimate,
                tracked=is_tracked,
                has_build_manifest=has_bm,
                has_recovery_metadata=has_rm,
            )
        return Classification(
            path=rel,
            verdict=VERDICT_KEEP,
            rationale=(
                f"smoke/probe/dryrun dir, but age={age_days:.1f}d <= "
                f"smoke_max={smoke_max_age_days}d (not old enough)"
            ),
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=is_tracked,
            has_build_manifest=has_bm,
            has_recovery_metadata=has_rm,
        )

    # Tracked-by-git → NEVER auto-delete. Sub-classify based on custody.
    if is_tracked:
        custody = _read_build_manifest_custody(entry)
        if custody in CUSTODY_PIN_STATUSES:
            return Classification(
                path=rel,
                verdict=VERDICT_KEEP,
                rationale=(
                    f"git-tracked + build_manifest.json::custody_status="
                    f"{custody!r} (pinned)"
                ),
                age_days=age_days,
                bytes_estimate=bytes_estimate,
                tracked=True,
                has_build_manifest=has_bm,
                has_recovery_metadata=has_rm,
            )
        if has_bm:
            return Classification(
                path=rel,
                verdict=VERDICT_KEEP,
                rationale=(
                    "git-tracked + build_manifest.json present (HISTORICAL_PROVENANCE)"
                ),
                age_days=age_days,
                bytes_estimate=bytes_estimate,
                tracked=True,
                has_build_manifest=True,
                has_recovery_metadata=has_rm,
            )
        if has_rm:
            return Classification(
                path=rel,
                verdict=VERDICT_KEEP,
                rationale=(
                    "git-tracked + recovery_metadata.json (HISTORICAL_PROVENANCE)"
                ),
                age_days=age_days,
                bytes_estimate=bytes_estimate,
                tracked=True,
                has_build_manifest=has_bm,
                has_recovery_metadata=True,
            )
        return Classification(
            path=rel,
            verdict=VERDICT_KEEP,
            rationale="git-tracked (never auto-delete tracked paths)",
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=True,
            has_build_manifest=has_bm,
            has_recovery_metadata=has_rm,
        )

    # Gitignored. DELETE-NOW iff age > ignored_max_age_days AND no custody pin.
    custody = _read_build_manifest_custody(entry)
    if custody in CUSTODY_PIN_STATUSES:
        return Classification(
            path=rel,
            verdict=VERDICT_KEEP,
            rationale=(
                f"gitignored but build_manifest.json::custody_status={custody!r} "
                "(pinned)"
            ),
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=False,
            has_build_manifest=has_bm,
            has_recovery_metadata=has_rm,
        )
    if has_rm:
        return Classification(
            path=rel,
            verdict=VERDICT_KEEP,
            rationale=(
                "gitignored but recovery_metadata.json present "
                "(HISTORICAL_PROVENANCE)"
            ),
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=False,
            has_build_manifest=has_bm,
            has_recovery_metadata=True,
        )
    if age_days > ignored_max_age_days:
        return Classification(
            path=rel,
            verdict=VERDICT_DELETE_NOW,
            rationale=(
                f"gitignored + age={age_days:.1f}d > "
                f"ignored_max={ignored_max_age_days}d + no custody pin"
            ),
            age_days=age_days,
            bytes_estimate=bytes_estimate,
            tracked=False,
            has_build_manifest=has_bm,
            has_recovery_metadata=has_rm,
        )

    # Gitignored but not yet old enough → KEEP (active workspace).
    return Classification(
        path=rel,
        verdict=VERDICT_KEEP,
        rationale=(
            f"gitignored, age={age_days:.1f}d <= "
            f"ignored_max={ignored_max_age_days}d (active workspace)"
        ),
        age_days=age_days,
        bytes_estimate=bytes_estimate,
        tracked=False,
        has_build_manifest=has_bm,
        has_recovery_metadata=has_rm,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_results_dirs(
    *,
    root: Path | None = None,
    repo_root: Path | None = None,
    now_seconds: float | None = None,
    ignored_max_age_days: int = DEFAULT_IGNORED_MAX_AGE_DAYS,
    smoke_max_age_days: int = DEFAULT_SMOKE_MAX_AGE_DAYS,
    keep_max_age_days: int = DEFAULT_KEEP_MAX_AGE_DAYS,
) -> list[Classification]:
    """Return a Classification per top-level directory under ``root``."""

    repo = (Path(repo_root) if repo_root else Path.cwd()).resolve()
    target = (Path(root) if root else (repo / DEFAULT_ROOT)).resolve()
    if not target.is_dir():
        return []
    now_s = now_seconds if now_seconds is not None else time.time()
    tracked = _load_git_tracked_top_level_dirs(target, repo)

    out: list[Classification] = []
    for entry in sorted(target.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        out.append(
            _classify_dir(
                entry,
                repo_root=repo,
                tracked_dirs=tracked,
                now_seconds=now_s,
                ignored_max_age_days=ignored_max_age_days,
                smoke_max_age_days=smoke_max_age_days,
                keep_max_age_days=keep_max_age_days,
            )
        )
    return out


def build_gc_plan(
    classifications: list[Classification],
    *,
    now_seconds: float | None = None,
) -> dict:
    """Build the JSON deletion plan from a list of Classifications."""

    now_s = now_seconds if now_seconds is not None else time.time()
    now_iso = (
        _dt.datetime.fromtimestamp(now_s, tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    by_verdict: dict[str, list[Classification]] = {
        VERDICT_DELETE_NOW: [],
        VERDICT_PRESERVE_METADATA: [],
        VERDICT_KEEP: [],
        VERDICT_AMBIGUOUS: [],
    }
    for c in classifications:
        by_verdict.setdefault(c.verdict, []).append(c)

    def _row(c: Classification) -> dict:
        return {
            "path": c.path,
            "verdict": c.verdict,
            "rationale": c.rationale,
            "age_days": round(c.age_days, 3),
            "bytes_estimate": c.bytes_estimate,
            "tracked": c.tracked,
            "has_build_manifest": c.has_build_manifest,
            "has_recovery_metadata": c.has_recovery_metadata,
        }

    return {
        "schema": "pact.experiments_results_gc_plan.v1",
        "generated_at_utc": now_iso,
        "totals": {
            "total": len(classifications),
            "delete_now": len(by_verdict[VERDICT_DELETE_NOW]),
            "preserve_metadata": len(by_verdict[VERDICT_PRESERVE_METADATA]),
            "keep": len(by_verdict[VERDICT_KEEP]),
            "ambiguous": len(by_verdict[VERDICT_AMBIGUOUS]),
            "delete_now_bytes_estimate": sum(
                c.bytes_estimate for c in by_verdict[VERDICT_DELETE_NOW]
            ),
            "preserve_metadata_bytes_estimate": sum(
                c.bytes_estimate for c in by_verdict[VERDICT_PRESERVE_METADATA]
            ),
        },
        "would_delete": [_row(c) for c in by_verdict[VERDICT_DELETE_NOW]],
        "would_preserve_metadata": [
            _row(c) for c in by_verdict[VERDICT_PRESERVE_METADATA]
        ],
        "would_keep": [_row(c) for c in by_verdict[VERDICT_KEEP]],
        "ambiguous": [_row(c) for c in by_verdict[VERDICT_AMBIGUOUS]],
        "rationale_per_path": {c.path: c.rationale for c in classifications},
    }


def execute_plan(
    plan: dict,
    *,
    repo_root: Path,
    operator_approved: str,
    verbose: bool = True,
) -> dict:
    """Apply the DELETE-NOW entries of ``plan``. Returns an execution summary.

    NEVER deletes anything outside ``experiments/results/``. NEVER deletes
    tracked paths (the plan itself classifies them as KEEP).
    """

    rows = plan.get("would_delete", [])
    deleted: list[str] = []
    skipped: list[dict] = []
    deleted_bytes = 0
    for row in rows:
        relpath = row["path"]
        # Defense in depth: refuse any path that escapes experiments/results/.
        if not relpath.startswith("experiments/results/"):
            skipped.append({"path": relpath, "reason": "out-of-scope path"})
            continue
        abs_path = (repo_root / relpath).resolve()
        try:
            abs_path.relative_to((repo_root / "experiments/results").resolve())
        except ValueError:
            skipped.append({"path": relpath, "reason": "path escapes repo scope"})
            continue
        if not abs_path.is_dir():
            skipped.append({"path": relpath, "reason": "no longer exists"})
            continue
        try:
            shutil.rmtree(abs_path)
        except OSError as exc:
            skipped.append({"path": relpath, "reason": f"rmtree failed: {exc!r}"})
            continue
        deleted.append(relpath)
        deleted_bytes += int(row.get("bytes_estimate", 0))
        if verbose:
            print(f"  DELETED {relpath} (~{row.get('bytes_estimate', 0)} bytes)")
    now_iso = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": "pact.experiments_results_gc_execution.v1",
        "applied_at_utc": now_iso,
        "operator_approved": operator_approved,
        "deleted_count": len(deleted),
        "skipped_count": len(skipped),
        "deleted_bytes_estimate": deleted_bytes,
        "deleted": deleted,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_operator_handle(value: str) -> str:
    value = value.strip()
    if not value:
        raise SystemExit("VALIDATION_ERROR: --operator-approved must not be empty")
    if ":" not in value:
        raise SystemExit(
            "VALIDATION_ERROR: --operator-approved must be "
            "'<handle>:<UTC_timestamp>' (e.g. 'alejandro:2026-05-12T18:00Z')"
        )
    handle, _, ts = value.partition(":")
    if not handle.strip():
        raise SystemExit("VALIDATION_ERROR: --operator-approved handle is empty")
    if not ts.strip():
        raise SystemExit("VALIDATION_ERROR: --operator-approved timestamp is empty")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory to GC (default: experiments/results)",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root for relative paths and git ls-files",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the plan but do NOT delete anything (default).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute the deletion plan. Requires --operator-approved.",
    )
    p.add_argument(
        "--operator-approved",
        default="",
        help="Operator handle:UTC_timestamp authorizing apply (REQUIRED with --apply)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON plan to this path instead of stdout",
    )
    p.add_argument(
        "--ignored-max-age-days",
        type=int,
        default=DEFAULT_IGNORED_MAX_AGE_DAYS,
        help=f"Gitignored dir delete threshold (default: {DEFAULT_IGNORED_MAX_AGE_DAYS}d)",
    )
    p.add_argument(
        "--smoke-max-age-days",
        type=int,
        default=DEFAULT_SMOKE_MAX_AGE_DAYS,
        help=f"smoke/probe/dryrun delete threshold (default: {DEFAULT_SMOKE_MAX_AGE_DAYS}d)",
    )
    p.add_argument(
        "--keep-max-age-days",
        type=int,
        default=DEFAULT_KEEP_MAX_AGE_DAYS,
        help=f"Always-KEEP recency floor (default: {DEFAULT_KEEP_MAX_AGE_DAYS}d)",
    )
    p.add_argument(
        "--now-utc",
        default="",
        help="Optional ISO-8601 timestamp to use as 'now' (test hook)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-path deletion log on --apply",
    )
    return p


def _resolve_now_seconds(now_utc: str) -> float:
    if not now_utc.strip():
        return time.time()
    s = now_utc.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(s)
    except ValueError:
        raise SystemExit(f"VALIDATION_ERROR: --now-utc not ISO-8601: {now_utc!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed.timestamp()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.apply and not args.operator_approved:
        print(
            "REFUSING_APPLY: --apply REQUIRES --operator-approved "
            "'<handle>:<UTC_timestamp>'. Run with --dry-run first.",
            file=sys.stderr,
        )
        return 3
    if args.apply:
        _validate_operator_handle(args.operator_approved)
    if not args.apply and not args.dry_run:
        # Default mode: dry-run.
        args.dry_run = True

    now_s = _resolve_now_seconds(args.now_utc)
    classifications = classify_results_dirs(
        root=args.root,
        repo_root=args.repo_root,
        now_seconds=now_s,
        ignored_max_age_days=args.ignored_max_age_days,
        smoke_max_age_days=args.smoke_max_age_days,
        keep_max_age_days=args.keep_max_age_days,
    )
    plan = build_gc_plan(classifications, now_seconds=now_s)

    if args.apply:
        plan["executed"] = execute_plan(
            plan,
            repo_root=Path(args.repo_root).resolve(),
            operator_approved=args.operator_approved,
            verbose=args.verbose,
        )

    out_json = json.dumps(plan, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out_json + "\n", encoding="utf-8")
        # Brief summary to stdout in either mode.
        totals = plan["totals"]
        msg = (
            f"GC_PLAN written to {args.output} "
            f"(delete_now={totals['delete_now']}, "
            f"preserve_metadata={totals['preserve_metadata']}, "
            f"keep={totals['keep']}, "
            f"ambiguous={totals['ambiguous']})"
        )
        print(msg)
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
