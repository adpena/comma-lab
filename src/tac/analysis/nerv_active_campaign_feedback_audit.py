# SPDX-License-Identifier: MIT
"""Audit active NeRV campaign telemetry ingestion.

Long-running SNeRV/HiNeRV jobs usually write bulky telemetry on SSD. This
module checks the dispatch-claim ledger for active NeRV rows, follows their
claimed output paths, and verifies that visible telemetry/progress artifacts
have also been converted into planner-visible false-authority feedback.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.deploy.claims import is_terminal_status

SCHEMA = "nerv_active_campaign_feedback_audit.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

_CLAIM_TABLE_KEYS = (
    "timestamp_utc",
    "agent",
    "lane_id",
    "platform",
    "instance_job_id",
    "predicted_eta_utc",
    "status",
    "notes",
)
_NERV_MARKERS = ("hi_nerv", "hinerv", "snerv", "nerv")
_ACTIVE_STATUS_MARKERS = ("active", "running", "training")
_PATH_RE = re.compile(
    r"(?P<path>(?:/Volumes/(?:VertigoDataTier|APDataStore)/pact[^\s|,;)]+"
    r"|/Users/adpena/Projects/pact[^\s|,;)]+"
    r"|/private/[^\s|,;)]+"
    r"|/tmp/[^\s|,;)]+"
    r"|\.omx/research[^\s|,;)]+))"
)
_ARTIFACT_NAMES = {
    "telemetry.jsonl",
    "local_mlx_prefilter_progress.jsonl",
    "nerv_candidate_byte_feedback.jsonl",
    "compact_renderer_mlx_spine_runner_report.json",
    "snerv_scorer_loop_qat_progress.jsonl",
    "snerv_scorer_loop_decoder_qat_progress.jsonl",
    "decoder_weight_gradient_saliency.json",
    "training_artifact.json",
}
_INGESTION_NAME_MARKERS = (
    "nerv_training_telemetry_feedback",
    "nerv_candidate_training_telemetry_feedback",
    "nerv_queue_training_feedback_refresh",
    "nerv_candidate_byte_feedback",
    "nerv_long_training_campaign_plan",
)
_CLAIM_MATCH_STOP_TOKENS = {
    "active",
    "advisory",
    "apdatastore",
    "archive",
    "authority",
    "campaign",
    "candidate",
    "codex",
    "controls",
    "exact",
    "false",
    "hinerv",
    "local",
    "mlx",
    "nerv",
    "official",
    "output",
    "pact",
    "projects",
    "proof",
    "research",
    "score",
    "snerv",
    "training",
    "users",
    "vertigodatatier",
    "volumes",
}


def build_nerv_active_campaign_feedback_audit(
    *,
    claims_path: str | Path,
    repo_root: str | Path,
    research_dir: str | Path | None = None,
    stale_epoch_tolerance: int = 512,
    max_artifacts_per_claim: int = 64,
) -> dict[str, Any]:
    """Build a false-authority audit of active NeRV campaign feedback custody."""

    repo = Path(repo_root).expanduser().resolve(strict=False)
    claims = Path(claims_path).expanduser().resolve(strict=False)
    research = (
        Path(research_dir).expanduser().resolve(strict=False)
        if research_dir is not None
        else repo / ".omx" / "research"
    )
    claim_rows = _active_nerv_claim_rows(_claim_rows_from_markdown(claims))
    ingestion_files = tuple(_iter_ingestion_files(research))
    audited_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    artifact_count = 0
    for claim in claim_rows:
        row = _audit_claim_row(
            claim,
            repo_root=repo,
            ingestion_files=ingestion_files,
            stale_epoch_tolerance=int(stale_epoch_tolerance),
            max_artifacts_per_claim=max(1, int(max_artifacts_per_claim)),
        )
        audited_rows.append(row)
        artifact_count += len(row["artifacts"])
        for blocker in row["blockers"]:
            blockers.append(
                f"{row['lane_id']}:{row['instance_job_id']}:{blocker}"
            )
    blockers = _dedupe(blockers)
    return {
        "schema": SCHEMA,
        "claims_path": claims.as_posix(),
        "research_dir": research.as_posix(),
        "active_claim_count": len(claim_rows),
        "audited_claim_count": len(audited_rows),
        "artifact_count": artifact_count,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "active_claim_rows": audited_rows,
        "stale_epoch_tolerance": int(stale_epoch_tolerance),
        **FALSE_AUTHORITY,
    }


def render_nerv_active_campaign_feedback_audit_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing audit summary."""

    lines = [
        "# NeRV Active Campaign Feedback Audit",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Active claims: `{report.get('active_claim_count')}`",
        f"Audited claims: `{report.get('audited_claim_count')}`",
        f"Artifacts: `{report.get('artifact_count')}`",
        f"Blockers: `{report.get('blocker_count')}`",
        f"Score claim: `{report.get('score_claim')}`",
        f"Ready for exact dispatch: `{report.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = list(report.get("blockers") or [])
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Active Rows", ""])
    for row in report.get("active_claim_rows") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"- `{row.get('lane_id')}` / `{row.get('instance_job_id')}` "
            f"status=`{row.get('status')}` artifacts=`{len(row.get('artifacts') or [])}`"
        )
        for artifact in row.get("artifacts") or []:
            if not isinstance(artifact, Mapping):
                continue
            ingestion = dict(artifact.get("ingestion") or {})
            lines.append(
                f"  - `{artifact.get('kind')}` `{artifact.get('path')}` "
                f"latest_epoch=`{artifact.get('latest_epoch')}` "
                f"max_ingested_epoch=`{ingestion.get('max_ingested_epoch')}` "
                f"ingested=`{ingestion.get('ingested')}`"
            )
    return "\n".join(lines) + "\n"


def _claim_rows_from_markdown(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if (
            not stripped.startswith("|")
            or "---" in stripped
            or "timestamp_utc" in stripped
        ):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < len(_CLAIM_TABLE_KEYS):
            continue
        rows.append(dict(zip(_CLAIM_TABLE_KEYS, cells[: len(_CLAIM_TABLE_KEYS)], strict=True)))
    return rows


def _active_nerv_claim_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in rows:
        row = {key: str(source.get(key) or "") for key in _CLAIM_TABLE_KEYS}
        key = (row["lane_id"], row["instance_job_id"])
        if key in seen:
            continue
        seen.add(key)
        status = row["status"]
        if _is_terminal_for_active_feedback_audit(status):
            continue
        text = " ".join(
            (row["lane_id"], row["instance_job_id"], row["platform"], status, row["notes"])
        ).lower()
        if not any(marker in text for marker in _NERV_MARKERS):
            continue
        if not any(marker in status.lower() for marker in _ACTIVE_STATUS_MARKERS):
            continue
        out.append(row)
    return out


def _audit_claim_row(
    claim: Mapping[str, str],
    *,
    repo_root: Path,
    ingestion_files: tuple[Path, ...],
    stale_epoch_tolerance: int,
    max_artifacts_per_claim: int,
) -> dict[str, Any]:
    artifacts = _discover_claim_artifacts(
        claim,
        repo_root=repo_root,
        max_artifacts=max_artifacts_per_claim,
    )
    blockers: list[str] = []
    if not artifacts:
        blockers.append("active_campaign_feedback_artifacts_missing")
    artifact_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        row = _artifact_row(
            artifact,
            ingestion_files=ingestion_files,
            stale_epoch_tolerance=stale_epoch_tolerance,
        )
        artifact_rows.append(row)
        blockers.extend(row["blockers"])
    return {
        "timestamp_utc": claim.get("timestamp_utc"),
        "lane_id": claim.get("lane_id"),
        "instance_job_id": claim.get("instance_job_id"),
        "platform": claim.get("platform"),
        "status": claim.get("status"),
        "artifact_roots": [path.as_posix() for path in _paths_from_claim(claim, repo_root=repo_root)],
        "artifacts": artifact_rows,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }


def _discover_claim_artifacts(
    claim: Mapping[str, str],
    *,
    repo_root: Path,
    max_artifacts: int,
) -> list[Path]:
    artifacts: list[Path] = []
    for root in _paths_from_claim(claim, repo_root=repo_root):
        if root.is_file():
            if root.name in _ARTIFACT_NAMES or _looks_feedback_related(root):
                artifacts.append(root)
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in _ARTIFACT_NAMES:
                artifacts.append(path)
            if len(artifacts) >= max_artifacts:
                break
        if len(artifacts) >= max_artifacts:
            break
    return _dedupe_paths(artifacts)


def _paths_from_claim(claim: Mapping[str, str], *, repo_root: Path) -> list[Path]:
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("notes", "instance_job_id", "lane_id")
    )
    out: list[Path] = []
    for match in _PATH_RE.finditer(text):
        raw = match.group("path").rstrip(".:;]")
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        resolved = path.expanduser().resolve(strict=False)
        out.append(resolved)
        out.extend(
            _paths_from_queue_or_plan_file(
                resolved,
                claim=claim,
                repo_root=repo_root,
            )
        )
    return _dedupe_paths(out)


def _is_terminal_for_active_feedback_audit(status: str) -> bool:
    lowered = status.lower()
    return is_terminal_status(status) or lowered.startswith(("refused_", "completed_"))


def _paths_from_queue_or_plan_file(
    path: Path,
    *,
    claim: Mapping[str, str],
    repo_root: Path,
) -> list[Path]:
    if not path.is_file() or path.suffix != ".json":
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    claim_tokens = _claim_match_tokens(claim)
    out: list[Path] = []
    for command in _iter_commands(payload):
        for output in _output_dirs_from_command(command):
            if claim_tokens and not _path_matches_any_token(output, claim_tokens):
                continue
            out.append(_resolve_claim_path(output, repo_root=repo_root))
    return _dedupe_paths(out)


def _iter_commands(payload: Any) -> Iterable[list[str]]:
    if isinstance(payload, Mapping):
        command = payload.get("command") or payload.get("command_argv")
        if isinstance(command, list):
            yield [str(item) for item in command]
        for value in payload.values():
            yield from _iter_commands(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_commands(item)


def _output_dirs_from_command(command: list[str]) -> Iterable[str]:
    for index, item in enumerate(command[:-1]):
        if item == "--output-dir":
            yield command[index + 1]


def _resolve_claim_path(raw: str, *, repo_root: Path) -> Path:
    path = Path(str(raw).strip())
    if not path.is_absolute():
        path = repo_root / path
    return path.expanduser().resolve(strict=False)


def _claim_match_tokens(claim: Mapping[str, str]) -> set[str]:
    text = " ".join(
        str(claim.get(key) or "")
        for key in ("lane_id", "instance_job_id", "notes")
    ).lower()
    raw_tokens = re.split(r"[^a-z0-9]+", text)
    tokens = {
        token
        for token in raw_tokens
        if (
            len(token) >= 8
            and not token.isdigit()
            and token not in _CLAIM_MATCH_STOP_TOKENS
            and not re.fullmatch(r"\d{8}t\d+z?", token)
        )
    }
    for latent_dim, embed_dim in re.findall(r"ld(\d+)[^a-z0-9]+ed(\d+)", text):
        tokens.add(f"ld{latent_dim}ed{embed_dim}")
    return tokens


def _path_matches_any_token(path: str, tokens: set[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", Path(path).name.lower())
    return any(token in normalized for token in tokens)


def _artifact_row(
    path: Path,
    *,
    ingestion_files: tuple[Path, ...],
    stale_epoch_tolerance: int,
) -> dict[str, Any]:
    latest_epoch = _latest_jsonl_epoch(path) if path.name == "telemetry.jsonl" else None
    ingestion = _ingestion_for_artifact(
        path,
        ingestion_files=ingestion_files,
        latest_epoch=latest_epoch,
        stale_epoch_tolerance=stale_epoch_tolerance,
    )
    blockers: list[str] = []
    if not ingestion["ingested"]:
        blockers.append(f"active_campaign_feedback_not_ingested:{path.as_posix()}")
    if ingestion["stale"]:
        blockers.append(f"active_campaign_feedback_ingestion_stale:{path.as_posix()}")
    return {
        "path": path.as_posix(),
        "kind": _artifact_kind(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "latest_epoch": latest_epoch,
        "ingestion": ingestion,
        "blockers": blockers,
    }


def _ingestion_for_artifact(
    path: Path,
    *,
    ingestion_files: tuple[Path, ...],
    latest_epoch: int | None,
    stale_epoch_tolerance: int,
) -> dict[str, Any]:
    needle = path.as_posix()
    matches: list[dict[str, Any]] = []
    max_ingested_epoch: int | None = None
    for candidate in ingestion_files:
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if needle not in text:
            continue
        epochs = _last_epochs_from_text(text)
        if epochs:
            best = max(epochs)
            max_ingested_epoch = best if max_ingested_epoch is None else max(max_ingested_epoch, best)
        matches.append(
            {
                "path": candidate.as_posix(),
                "last_epoch_max": max(epochs) if epochs else None,
            }
        )
    stale = (
        latest_epoch is not None
        and max_ingested_epoch is not None
        and latest_epoch - max_ingested_epoch > int(stale_epoch_tolerance)
    )
    return {
        "ingested": bool(matches),
        "match_count": len(matches),
        "matches": matches[:12],
        "max_ingested_epoch": max_ingested_epoch,
        "stale": bool(stale),
        "stale_epoch_tolerance": int(stale_epoch_tolerance),
    }


def _iter_ingestion_files(research_dir: Path) -> Iterable[Path]:
    if not research_dir.is_dir():
        return ()
    files: list[Path] = []
    for path in research_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        name = path.name
        parent = path.parent.name
        if any(marker in name or marker in parent for marker in _INGESTION_NAME_MARKERS):
            files.append(path)
    return tuple(sorted(files))


def _latest_jsonl_epoch(path: Path) -> int | None:
    latest: int | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, Mapping):
                    continue
                value = _int_or_none(payload.get("epoch"))
                if value is not None:
                    latest = value if latest is None else max(latest, value)
    except OSError:
        return None
    return latest


def _last_epochs_from_text(text: str) -> list[int]:
    epochs: list[int] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            epochs.extend(_collect_last_epochs(payload))
        return epochs
    return _collect_last_epochs(payload)


def _collect_last_epochs(payload: Any) -> list[int]:
    out: list[int] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "last_epoch":
                parsed = _int_or_none(value)
                if parsed is not None:
                    out.append(parsed)
            else:
                out.extend(_collect_last_epochs(value))
    elif isinstance(payload, list):
        for item in payload:
            out.extend(_collect_last_epochs(item))
    return out


def _artifact_kind(path: Path) -> str:
    if path.name == "telemetry.jsonl":
        return "training_telemetry"
    if "progress" in path.name:
        return "progress_jsonl"
    if path.name == "nerv_candidate_byte_feedback.jsonl":
        return "candidate_byte_feedback"
    if path.name == "compact_renderer_mlx_spine_runner_report.json":
        return "compact_runner_report"
    if path.name == "decoder_weight_gradient_saliency.json":
        return "decoder_weight_saliency"
    return "artifact"


def _looks_feedback_related(path: Path) -> bool:
    return any(marker in path.name for marker in _INGESTION_NAME_MARKERS)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


__all__ = [
    "SCHEMA",
    "build_nerv_active_campaign_feedback_audit",
    "render_nerv_active_campaign_feedback_audit_markdown",
]
