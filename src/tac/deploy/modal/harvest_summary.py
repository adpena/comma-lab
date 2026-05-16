# SPDX-License-Identifier: MIT
"""Modal training harvest summary helpers.

Bulk harvest is repeatedly re-run to close old provider state. Already
harvested calls must keep their original return code, elapsed time, artifact
count, and crash classification so the aggregate summary is a durable custody
surface instead of a lossy status list.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PRESERVED_RESULT_KEYS = (
    "rc",
    "elapsed_seconds",
    "timed_out",
    "n_artifacts",
    "crash_kind",
    "archive_sha256",
    "archive_bytes",
    "artifact_signal_warning",
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
)
RC_STATUS_RE = re.compile(r"(?:^|_)rc_(\d+)$")
ELAPSED_NOTES_RE = re.compile(
    r"(?:^|[;\s])elapsed_seconds=([0-9]+(?:\.[0-9]+)?)"
)
PAYLOAD_ARTIFACT_NAMES = frozenset(
    {
        "0.bin",
        "archive.zip",
        "contest_auth_eval.json",
        "contest_auth_eval_cpu.json",
        "contest_auth_eval_cuda.json",
        "provenance.json",
        "run.log",
    }
)


def _artifact_file_count(artifacts_dir: Path) -> int:
    """Return local harvested artifact count, treating missing dirs as empty."""

    root = Path(artifacts_dir)
    if not root.is_dir():
        return 0
    return len([p for p in root.iterdir() if p.is_file()])


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _rc_from_terminal_claim(terminal_claim: Mapping[str, Any] | None) -> int | None:
    if not isinstance(terminal_claim, Mapping):
        return None
    status = str(terminal_claim.get("status") or "")
    match = RC_STATUS_RE.search(status)
    if match:
        return int(match.group(1))
    notes = str(terminal_claim.get("notes") or "")
    match = re.search(r"(?:^|[;\s])rc=([0-9]+)", notes)
    if match:
        return int(match.group(1))
    return None


def _elapsed_from_terminal_claim(
    terminal_claim: Mapping[str, Any] | None,
) -> float | None:
    if not isinstance(terminal_claim, Mapping):
        return None
    notes = str(terminal_claim.get("notes") or "")
    match = ELAPSED_NOTES_RE.search(notes)
    if not match:
        return None
    return float(match.group(1))


def _payload_files(out_dir: Path | None) -> list[Path]:
    if out_dir is None:
        return []
    root = Path(out_dir)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name in PAYLOAD_ARTIFACT_NAMES
    )


def _archive_facts(out_dir: Path | None) -> tuple[str | None, int | None]:
    archives = [p for p in _payload_files(out_dir) if p.name == "archive.zip"]
    if not archives:
        return None, None
    archive = archives[0]
    raw = archive.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def modal_training_summary_entry(
    *,
    label: str,
    call_id: str,
    status: str | None = None,
    harvested: Mapping[str, Any] | None = None,
    cost_anchor: Mapping[str, Any] | None = None,
    terminal_claim: Mapping[str, Any] | None = None,
    terminal_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one aggregate Modal training harvest row without signal loss."""

    row: dict[str, Any] = {"label": label, "call_id": call_id}
    if status is not None:
        row["status"] = status
    if harvested is not None:
        for key in PRESERVED_RESULT_KEYS:
            if key in harvested:
                row[key] = harvested[key]
    if cost_anchor is not None:
        row["cost_band_anchor"] = dict(cost_anchor)
    if terminal_claim is not None:
        row["terminal_claim"] = dict(terminal_claim)
    if terminal_evidence is not None:
        row["terminal_evidence"] = dict(terminal_evidence)
    return row


def enrich_modal_training_result_summary(
    loaded: Mapping[str, Any],
    *,
    artifacts_dir: Path,
    out_dir: Path | None = None,
    cost_anchor: Mapping[str, Any] | None = None,
    terminal_claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover structured signal from already-harvested Modal sidecars.

    Some historical harvest rows were terminal but shallow: ``rc`` and elapsed
    time were only present in the terminal claim/cost marker, and artifacts
    such as ``archive.zip`` lived in the provider output tree rather than
    ``harvested_artifacts``. This helper enriches the summary without turning
    it into score authority.
    """

    row = dict(loaded)
    if _as_int(row.get("rc")) is None:
        rc = _rc_from_terminal_claim(terminal_claim)
        if rc is None and isinstance(cost_anchor, Mapping):
            rc = _as_int(cost_anchor.get("returncode"))
        if rc is not None:
            row["rc"] = rc
            row.setdefault("crash_kind", "OK" if rc == 0 else f"RC_{rc}")
    if _as_float(row.get("elapsed_seconds")) is None:
        elapsed = None
        if isinstance(cost_anchor, Mapping):
            elapsed = _as_float(cost_anchor.get("elapsed_seconds"))
        if elapsed is None:
            elapsed = _elapsed_from_terminal_claim(terminal_claim)
        if elapsed is not None:
            row["elapsed_seconds"] = elapsed

    local_artifact_count = _artifact_file_count(artifacts_dir)
    payload_files = _payload_files(out_dir)
    n_artifacts = _as_int(row.get("n_artifacts"))
    if n_artifacts is None or n_artifacts == 0:
        recovered_count = max(local_artifact_count, len(payload_files))
        if recovered_count:
            row["n_artifacts"] = recovered_count
            if local_artifact_count == 0 and payload_files:
                row["artifact_signal_warning"] = (
                    "harvested_artifacts_empty_but_provider_output_payloads_exist"
                )

    archive_sha, archive_bytes = _archive_facts(out_dir)
    if archive_sha is not None:
        row.setdefault("archive_sha256", archive_sha)
    if archive_bytes is not None:
        row.setdefault("archive_bytes", archive_bytes)
    row.setdefault("score_claim", False)
    row.setdefault("promotion_eligible", False)
    row.setdefault("rank_or_kill_eligible", False)
    return row


def normalise_modal_training_result_summary(
    loaded: Mapping[str, Any],
    *,
    artifacts_dir: Path,
    source_summary: Path,
) -> dict[str, Any]:
    """Normalize legacy Modal training harvest summaries to bulk-harvest shape."""

    rc = loaded.get("returncode", loaded.get("rc"))
    timed_out = bool(loaded.get("timed_out", False))
    if timed_out:
        crash_kind = "TIMEOUT"
    elif isinstance(rc, int) and not isinstance(rc, bool):
        crash_kind = "OK" if rc == 0 else f"RC_{rc}"
    else:
        crash_kind = "HARVESTED_PARTIAL"
    return {
        **dict(loaded),
        "rc": rc,
        "timed_out": timed_out,
        "n_artifacts": _artifact_file_count(artifacts_dir),
        "crash_kind": crash_kind,
        "source_summary": str(source_summary),
    }


def partial_modal_training_result_summary(*, artifacts_dir: Path) -> dict[str, Any]:
    """Represent harvested artifacts that predate structured result summaries."""

    return {
        "timed_out": False,
        "n_artifacts": _artifact_file_count(artifacts_dir),
        "crash_kind": "HARVESTED_PARTIAL",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
    }


__all__ = [
    "PRESERVED_RESULT_KEYS",
    "enrich_modal_training_result_summary",
    "modal_training_summary_entry",
    "normalise_modal_training_result_summary",
    "partial_modal_training_result_summary",
]
