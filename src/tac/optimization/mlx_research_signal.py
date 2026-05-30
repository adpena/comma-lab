# SPDX-License-Identifier: MIT
"""MLX research-signal manifest rows for local Apple Silicon training.

MLX-local measurements are useful acquisition and substrate-training signal,
but they are not contest score authority. This module is the MLX sibling of
``tac.optimization.mps_research_signal``: it appends typed JSONL rows while
auto-stamping fail-closed authority fields.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "mlx_research_signal_manifest.v1"
EVIDENCE_GRADE = "macOS-MLX-research-signal"
EVIDENCE_TAG = "[macOS-MLX research-signal]"
EVIDENCE_SEMANTICS = "mlx_local_research_signal_only"
HARDWARE_SUBSTRATE = "macos_arm64_mlx"
DISPATCH_BLOCKERS = (
    "macos_mlx_research_signal_not_contest_authority",
    "requires_paired_contest_cpu_plus_cuda_for_score_claim",
    "mlx_local_signal_requires_archive_custody_and_exact_axis_replay",
    "not_promotion_eligible",
)


class MLXResearchSignalError(ValueError):
    """Raised when an MLX research-signal row violates fail-closed authority."""


def append_manifest_row_to_jsonl(
    row: Mapping[str, Any],
    *,
    output_path: Path,
) -> None:
    """Append one fail-closed MLX research-signal row to a JSONL manifest."""

    output_str = str(output_path)
    if output_str.startswith("/tmp/") or "/private/tmp/" in output_str or "/var/tmp/" in output_str:
        raise ValueError(f"refusing to write MLX research-signal manifest to forbidden /tmp path: {output_str!r}")

    serializable = dict(row)
    serializable.setdefault("schema", SCHEMA_VERSION)
    serializable.setdefault("score_claim", False)
    serializable.setdefault("promotion_eligible", False)
    serializable.setdefault("rank_or_kill_eligible", False)
    serializable.setdefault("ready_for_exact_eval_dispatch", False)
    serializable.setdefault("dispatch_attempted", False)
    serializable.setdefault("evidence_grade", EVIDENCE_GRADE)
    serializable.setdefault("evidence_tag", EVIDENCE_TAG)
    serializable.setdefault("axis_tag", EVIDENCE_TAG)
    serializable.setdefault("hardware_substrate", HARDWARE_SUBSTRATE)
    serializable.setdefault("device", "mlx")
    serializable.setdefault("evidence_semantics", EVIDENCE_SEMANTICS)
    serializable.setdefault("dispatch_blockers", list(DISPATCH_BLOCKERS))

    if (
        serializable["score_claim"]
        or serializable["promotion_eligible"]
        or serializable["ready_for_exact_eval_dispatch"]
        or serializable["rank_or_kill_eligible"]
    ):
        raise MLXResearchSignalError(
            "MLX research-signal manifest rows cannot carry score authority. "
            "Use contest-CPU or contest-CUDA exact-eval custody for score claims."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(serializable, sort_keys=True, allow_nan=False)
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


__all__ = [
    "DISPATCH_BLOCKERS",
    "EVIDENCE_GRADE",
    "EVIDENCE_SEMANTICS",
    "EVIDENCE_TAG",
    "HARDWARE_SUBSTRATE",
    "SCHEMA_VERSION",
    "MLXResearchSignalError",
    "append_manifest_row_to_jsonl",
]
