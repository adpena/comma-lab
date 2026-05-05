#!/usr/bin/env python3
"""Score-tracking dashboard: sorted view of every contest_auth_eval.json on disk.

Scans `experiments/results/**/contest_auth_eval*.json` (and adjudicated variants),
extracts final_score + components + custody, prints a sorted table to the
operator. Defaults to ascending score (best first).

Useful once any of the launch-ready dispatches (Lane Ω-W-V3, Lane #04 ×5
variants, Lane SJ-KL) lands a contest-CUDA score — the dashboard automatically
includes it without the operator having to find the JSON path.

Per CLAUDE.md `[contest-CUDA]` discipline: the dashboard tags every row with
the device + sample count + gpu_match boolean from the JSON, so non-CUDA or
short-sample rows are visible (and not silently aggregated with real T4 A++
scores).

Usage:
    .venv/bin/python tools/score_dashboard.py                       # all scores, best first
    .venv/bin/python tools/score_dashboard.py --top 20              # top 20 only
    .venv/bin/python tools/score_dashboard.py --filter pr106         # substring filter on path
    .venv/bin/python tools/score_dashboard.py --since 2026-05-04    # date filter
    .venv/bin/python tools/score_dashboard.py --json                # machine-readable output
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOT = REPO_ROOT / "experiments" / "results"


@dataclass
class ScoreRow:
    score: float | None
    archive_bytes: int | None
    archive_sha256: str | None
    seg_dist_avg: float | None
    pose_dist_avg: float | None
    rate: float | None
    samples: int | None
    device: str
    gpu_match: bool
    path: str
    mtime_utc: str
    extra: dict[str, Any] = field(default_factory=dict)


def _safe_get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _parse_one(path: Path, repo_root: Path) -> ScoreRow | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    score = (
        _safe_get(data, "final_score")
        or _safe_get(data, "score")
        or _safe_get(data, "total_score")
        or _safe_get(data, "result", "final_score")
    )
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return ScoreRow(
        score=score,
        archive_bytes=_safe_get(data, "archive_bytes") or _safe_get(data, "bytes"),
        archive_sha256=_safe_get(data, "archive_sha256") or _safe_get(data, "sha256"),
        seg_dist_avg=_safe_get(data, "seg_dist_avg") or _safe_get(data, "components", "segnet_avg"),
        pose_dist_avg=_safe_get(data, "pose_dist_avg") or _safe_get(data, "components", "posenet_avg"),
        rate=_safe_get(data, "rate") or _safe_get(data, "components", "rate"),
        samples=_safe_get(data, "samples") or _safe_get(data, "n_samples"),
        device=str(_safe_get(data, "device", default="?")),
        gpu_match=bool(_safe_get(data, "gpu_t4_match", default=False)),
        path=str(rel),
        mtime_utc=mtime.strftime("%Y-%m-%dT%H:%MZ"),
        extra=data,
    )


def _parse_log(path: Path, repo_root: Path) -> ScoreRow | None:
    """Extract embedded RESULT_JSON from an auth_eval.log file.

    The contest_auth_eval.py runner emits a single line `RESULT_JSON: {...}`
    in its stdout log. When the canonical JSON wasn't synced back from a
    remote (Lightning, Vast.ai), the auth_eval.log is the only on-disk source.
    """
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    marker = "RESULT_JSON: "
    idx = text.rfind(marker)
    if idx < 0:
        return None
    json_start = idx + len(marker)
    json_end = text.find("\n", json_start)
    if json_end < 0:
        json_end = len(text)
    try:
        data = json.loads(text[json_start:json_end])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    score = (
        _safe_get(data, "canonical_score")
        or _safe_get(data, "score_recomputed_from_components")
        or _safe_get(data, "final_score")
    )
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    prov = _safe_get(data, "provenance", default={})
    return ScoreRow(
        score=score,
        archive_bytes=_safe_get(data, "archive_size_bytes"),
        archive_sha256=_safe_get(prov, "archive_sha256") if isinstance(prov, dict) else None,
        seg_dist_avg=_safe_get(data, "avg_segnet_dist"),
        pose_dist_avg=_safe_get(data, "avg_posenet_dist"),
        rate=_safe_get(data, "rate_unscaled"),
        samples=_safe_get(data, "n_samples"),
        device=str(_safe_get(prov, "device", default="?")) if isinstance(prov, dict) else "?",
        gpu_match=bool(_safe_get(prov, "gpu_t4_match", default=False)) if isinstance(prov, dict) else False,
        path=str(rel),
        mtime_utc=mtime.strftime("%Y-%m-%dT%H:%MZ"),
        extra=data,
    )


def scan(
    repo_root: Path,
    scan_root: Path | None = None,
    *,
    path_filter: str | None = None,
    since: dt.date | None = None,
) -> list[ScoreRow]:
    root = scan_root or (repo_root / "experiments" / "results")
    if not root.is_dir():
        return []
    rows: list[ScoreRow] = []
    seen_dirs: set[Path] = set()
    # Pass 1: canonical contest_auth_eval*.json files (preferred when present)
    for p in root.rglob("contest_auth_eval*.json"):
        if path_filter and path_filter not in str(p):
            continue
        if since:
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).date()
            if mtime < since:
                continue
        row = _parse_one(p, repo_root)
        if row is not None:
            rows.append(row)
            seen_dirs.add(p.parent)
    # Pass 2: auth_eval.log fallback for dispatches whose JSON didn't sync back
    for p in root.rglob("auth_eval.log"):
        if p.parent in seen_dirs:
            continue  # canonical JSON already covers this dispatch
        if path_filter and path_filter not in str(p):
            continue
        if since:
            mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).date()
            if mtime < since:
                continue
        row = _parse_log(p, repo_root)
        if row is not None:
            rows.append(row)
    return rows


def _format_table(rows: list[ScoreRow], *, top: int | None = None) -> str:
    if not rows:
        return "(no scores found)"
    rows_sorted = sorted(rows, key=lambda r: (r.score is None, r.score))
    if top:
        rows_sorted = rows_sorted[:top]
    out_lines: list[str] = []
    header = f"{'rank':>4}  {'score':>10}  {'bytes':>9}  {'seg_avg':>9}  {'pose_avg':>9}  {'samples':>7}  {'device':<8}  {'mtime':<17}  path"
    out_lines.append(header)
    out_lines.append("-" * len(header))
    for i, r in enumerate(rows_sorted, start=1):
        score_str = f"{r.score:.6f}" if r.score is not None else "?"
        bytes_str = f"{r.archive_bytes:,}" if r.archive_bytes else "?"
        seg_str = f"{r.seg_dist_avg:.6f}" if r.seg_dist_avg is not None else "?"
        pose_str = f"{r.pose_dist_avg:.6f}" if r.pose_dist_avg is not None else "?"
        samples_str = str(r.samples) if r.samples else "?"
        # Mark non-CUDA rows
        device_str = r.device
        if r.device != "cuda" and r.device != "?":
            device_str = f"{r.device}*"  # asterisk marks non-CUDA
        out_lines.append(
            f"{i:>4}  {score_str:>10}  {bytes_str:>9}  {seg_str:>9}  {pose_str:>9}  "
            f"{samples_str:>7}  {device_str:<8}  {r.mtime_utc:<17}  {r.path}"
        )
    out_lines.append("")
    out_lines.append("(* = non-CUDA device; treat as advisory only per CLAUDE.md MPS-auth-eval-is-NOISE)")
    out_lines.append(f"({len(rows)} total scores; {len(rows_sorted)} displayed; sorted ascending — best first)")
    return "\n".join(out_lines)


def _format_json(rows: list[ScoreRow], *, top: int | None = None) -> str:
    rows_sorted = sorted(rows, key=lambda r: (r.score is None, r.score))
    if top:
        rows_sorted = rows_sorted[:top]
    out = []
    for r in rows_sorted:
        out.append({
            "score": r.score,
            "archive_bytes": r.archive_bytes,
            "archive_sha256": r.archive_sha256,
            "seg_dist_avg": r.seg_dist_avg,
            "pose_dist_avg": r.pose_dist_avg,
            "rate": r.rate,
            "samples": r.samples,
            "device": r.device,
            "gpu_t4_match": r.gpu_match,
            "path": r.path,
            "mtime_utc": r.mtime_utc,
        })
    return json.dumps({"n_total": len(rows), "n_displayed": len(out), "rows": out}, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=None,
                        help="Show only top N scores (default: all).")
    parser.add_argument("--filter", type=str, default=None,
                        help="Substring filter on file path (e.g. 'pr106' or 'apogee_int5').")
    parser.add_argument("--since", type=str, default=None,
                        help="Only include scores from JSONs modified on/after this date (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output instead of human table.")
    parser.add_argument("--scan-root", type=Path, default=None,
                        help="Override scan root (default: experiments/results).")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                        help="Repo root for relative-path display (auto-detected).")
    args = parser.parse_args(argv)

    since = None
    if args.since:
        try:
            since = dt.datetime.strptime(args.since, "%Y-%m-%d").date()
        except ValueError:
            print(f"FATAL: --since must be YYYY-MM-DD (got '{args.since}')", file=sys.stderr)
            return 2

    rows = scan(
        args.repo_root,
        scan_root=args.scan_root,
        path_filter=args.filter,
        since=since,
    )

    if args.json:
        print(_format_json(rows, top=args.top))
    else:
        print(_format_table(rows, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
