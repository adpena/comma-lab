"""Modal training harvest summary helpers.

Bulk harvest is repeatedly re-run to close old provider state. Already
harvested calls must keep their original return code, elapsed time, artifact
count, and crash classification so the aggregate summary is a durable custody
surface instead of a lossy status list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PRESERVED_RESULT_KEYS = (
    "rc",
    "elapsed_seconds",
    "timed_out",
    "n_artifacts",
    "crash_kind",
)


def _artifact_file_count(artifacts_dir: Path) -> int:
    """Return local harvested artifact count, treating missing dirs as empty."""

    root = Path(artifacts_dir)
    if not root.is_dir():
        return 0
    return len([p for p in root.iterdir() if p.is_file()])


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
    "modal_training_summary_entry",
    "normalise_modal_training_result_summary",
    "partial_modal_training_result_summary",
]
