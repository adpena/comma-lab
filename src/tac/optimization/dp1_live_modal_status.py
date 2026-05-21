# SPDX-License-Identifier: MIT
"""Parse DP1 live Modal training logs into a queryable status packet."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "dp1_live_modal_status_v1"
TOOL_PATH = "tools/inspect_dp1_live_modal_status.py"

_STAGE_RE = re.compile(
    r"^\[archive-only-eval\]\s+(?P<ts>\S+)\s+Stage\s+(?P<stage>\d+[^:]*):\s+(?P<text>.*)$"
)
_CLAIM_RE = re.compile(
    r"^\[lane-dpp\]\s+(?P<ts>\S+)\s+stage_0b_dispatch_claim_verified\s+lane=(?P<lane>\S+)\s+job=(?P<job>\S+)"
)
_ROUNDTRIP_RE = re.compile(
    r"archive pack/parse roundtrip:\s+(?P<bytes>\d+)\s+bytes;\s+pairs=(?P<pairs>\d+);\s+header=(?P<header>\d+)"
)
_PROCEDURAL_RE = re.compile(
    r"procedural codebook replacement:\s+(?P<before>\d+)\s+B\s+->\s+(?P<after>\d+)\s+B\s+"
    r"\(saved\s+(?P<saved>\d+)\s+B;\s+predicted\s+[^=]+=?(?P<delta>[-+0-9.eE]+)\)"
)
_WROTE_RE = re.compile(r"wrote\s+(?P<kind>[A-Za-z0-9_.-]+):\s+(?P<path>\S+)")
_FINISHED_RE = re.compile(r"finished in\s+(?P<seconds>[0-9.]+)s\s+rc=(?P<rc>-?\d+)")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_dp1_live_modal_log(
    path: str | Path,
    *,
    variant: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return structured DP1 progress from one synced Modal lane log."""

    root = Path(repo_root).resolve()
    log_path = Path(path).resolve()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stage_events: list[dict[str, Any]] = []
    written_artifacts: dict[str, str] = {}
    procedural_replacement: dict[str, Any] | None = None
    smoke_roundtrip: dict[str, Any] | None = None
    dispatch_claim: dict[str, Any] | None = None
    finished: dict[str, Any] | None = None
    deterministic_warnings = 0

    for line in text.splitlines():
        if "does not have a deterministic implementation" in line:
            deterministic_warnings += 1
        if match := _CLAIM_RE.search(line):
            dispatch_claim = {
                "timestamp_utc": match.group("ts"),
                "lane_id": match.group("lane"),
                "job_id": match.group("job"),
            }
            continue
        if match := _STAGE_RE.search(line):
            stage_events.append(
                {
                    "timestamp_utc": match.group("ts"),
                    "stage": match.group("stage"),
                    "text": match.group("text"),
                }
            )
            continue
        if match := _ROUNDTRIP_RE.search(line):
            smoke_roundtrip = {
                "archive_bytes": int(match.group("bytes")),
                "pairs": int(match.group("pairs")),
                "header_bytes": int(match.group("header")),
            }
            continue
        if match := _PROCEDURAL_RE.search(line):
            procedural_replacement = {
                "before_bytes": int(match.group("before")),
                "after_bytes": int(match.group("after")),
                "saved_bytes": int(match.group("saved")),
                "predicted_delta_s": float(match.group("delta")),
            }
            continue
        if match := _WROTE_RE.search(line):
            written_artifacts[match.group("kind")] = match.group("path")
            continue
        if match := _FINISHED_RE.search(line):
            finished = {
                "elapsed_seconds": float(match.group("seconds")),
                "returncode": int(match.group("rc")),
            }

    last_stage = stage_events[-1] if stage_events else None
    return {
        "schema": "dp1_live_modal_variant_status_v1",
        "variant": variant,
        "log_path": _repo_rel(log_path, root),
        "log_bytes": log_path.stat().st_size,
        "log_sha256": _sha256_file(log_path),
        "dispatch_claim": dispatch_claim,
        "stage_events": stage_events,
        "last_stage": last_stage,
        "stage4_full_training_started": any(
            str(event.get("stage", "")).startswith("4") for event in stage_events
        ),
        "finished": finished is not None,
        "finish": finished,
        "smoke_roundtrip": smoke_roundtrip,
        "procedural_codebook_replacement": procedural_replacement,
        "written_artifacts": written_artifacts,
        "deterministic_warning_count": deterministic_warnings,
    }


def build_dp1_live_modal_status(
    *,
    baseline_log: str | Path,
    procedural_log: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return paired baseline/procedural live status from synced logs."""

    baseline = parse_dp1_live_modal_log(
        baseline_log,
        variant="baseline",
        repo_root=repo_root,
    )
    procedural = parse_dp1_live_modal_log(
        procedural_log,
        variant="procedural",
        repo_root=repo_root,
    )
    blockers: list[str] = []
    if not baseline["stage4_full_training_started"]:
        blockers.append("baseline_stage4_full_training_not_started")
    if not procedural["stage4_full_training_started"]:
        blockers.append("procedural_stage4_full_training_not_started")
    if baseline["finished"] and baseline.get("finish", {}).get("returncode") != 0:
        blockers.append("baseline_finished_nonzero")
    if procedural["finished"] and procedural.get("finish", {}).get("returncode") != 0:
        blockers.append("procedural_finished_nonzero")

    baseline_bytes = None
    procedural_bytes = None
    if isinstance(baseline.get("smoke_roundtrip"), Mapping):
        baseline_bytes = baseline["smoke_roundtrip"].get("archive_bytes")
    if isinstance(procedural.get("smoke_roundtrip"), Mapping):
        procedural_bytes = procedural["smoke_roundtrip"].get("archive_bytes")
    smoke_delta_bytes = (
        int(procedural_bytes) - int(baseline_bytes)
        if isinstance(baseline_bytes, int) and isinstance(procedural_bytes, int)
        else None
    )
    both_finished = bool(baseline["finished"] and procedural["finished"])
    aggregate_status = "finished" if both_finished and not blockers else "running"

    return {
        "schema": SCHEMA,
        "producer": TOOL_PATH,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "status": aggregate_status if not blockers else "needs_attention",
        "blockers": blockers,
        "baseline": baseline,
        "procedural": procedural,
        "smoke_delta_bytes_procedural_minus_baseline": smoke_delta_bytes,
        "smoke_delta_interpretation": (
            "Stage-3 smoke parser/packer signal only; Stage-4 training and paired CPU/CUDA auth eval remain required."
        ),
    }


def render_markdown(status: Mapping[str, Any]) -> str:
    baseline = status.get("baseline") if isinstance(status.get("baseline"), Mapping) else {}
    procedural = (
        status.get("procedural") if isinstance(status.get("procedural"), Mapping) else {}
    )
    rows = [
        "# DP1 Live Modal Status",
        "",
        f"- Schema: `{status.get('schema')}`",
        f"- Status: `{status.get('status')}`",
        f"- Blockers: `{', '.join(status.get('blockers') or []) or 'none'}`",
        f"- Smoke byte delta procedural-baseline: `{status.get('smoke_delta_bytes_procedural_minus_baseline')}`",
        "",
        "| variant | stage 4 started | finished | smoke bytes | deterministic warnings |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in (baseline, procedural):
        smoke = row.get("smoke_roundtrip") if isinstance(row.get("smoke_roundtrip"), Mapping) else {}
        rows.append(
            "| {variant} | {stage4} | {finished} | {bytes} | {warnings} |".format(
                variant=row.get("variant"),
                stage4=row.get("stage4_full_training_started"),
                finished=row.get("finished"),
                bytes=smoke.get("archive_bytes", ""),
                warnings=row.get("deterministic_warning_count"),
            )
        )
    replacement = procedural.get("procedural_codebook_replacement")
    if isinstance(replacement, Mapping):
        rows.extend(
            [
                "",
                "## Procedural Replacement",
                "",
                f"- Before bytes: `{replacement.get('before_bytes')}`",
                f"- After bytes: `{replacement.get('after_bytes')}`",
                f"- Saved bytes: `{replacement.get('saved_bytes')}`",
                f"- Predicted rate-only delta S: `{replacement.get('predicted_delta_s')}`",
            ]
        )
    return "\n".join(rows) + "\n"


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SCHEMA",
    "TOOL_PATH",
    "build_dp1_live_modal_status",
    "parse_dp1_live_modal_log",
    "render_markdown",
    "write_json",
]
