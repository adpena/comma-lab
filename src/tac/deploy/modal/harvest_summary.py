"""Modal training harvest summary helpers.

Bulk harvest is repeatedly re-run to close old provider state. Already
harvested calls must keep their original return code, elapsed time, artifact
count, and crash classification so the aggregate summary is a durable custody
surface instead of a lossy status list.
"""

from __future__ import annotations

from typing import Any, Mapping


PRESERVED_RESULT_KEYS = (
    "rc",
    "elapsed_seconds",
    "timed_out",
    "n_artifacts",
    "crash_kind",
)


def modal_training_summary_entry(
    *,
    label: str,
    call_id: str,
    status: str | None = None,
    harvested: Mapping[str, Any] | None = None,
    cost_anchor: Mapping[str, Any] | None = None,
    terminal_claim: Mapping[str, Any] | None = None,
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
    return row


__all__ = ["PRESERVED_RESULT_KEYS", "modal_training_summary_entry"]
