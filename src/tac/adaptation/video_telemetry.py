# SPDX-License-Identifier: MIT
"""Schema-first telemetry for dynamic per-video adaptation.

DVAR1 is a compress-time planning surface. It may use scorers, xray
artifacts, and GTScorerCache while building telemetry, but emitted artifacts
must not claim a score or imply inflate-time scorer access. The contest packet
remains deterministic, self-contained, byte-charged, and scorer-free.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "dynamic_video_telemetry_v1"
FAMILY = "dynamic_video_adaptive_receiver"
AXIS_LABELS = frozenset(
    {
        "[contest-CUDA]",
        "[contest-CPU]",
        "[macOS-CPU advisory]",
        "[MPS screening]",
        "[proxy]",
        "[synthetic-smoke]",
    }
)
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


class DynamicVideoTelemetryError(ValueError):
    """Raised when DVAR1 telemetry would be ambiguous or non-compliant."""


@dataclass(frozen=True)
class TelemetryPairRow:
    """One per-pair telemetry row used by allocator and MDL probes."""

    pair_idx: int
    frame_indices: tuple[int, int]
    seg_dist: float
    pose_dist: float
    score_contribution: float
    hard_pair_rank: int
    hard_frame_flags: tuple[str, ...] = ()
    xray_artifact_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pair_idx < 0:
            raise DynamicVideoTelemetryError("pair_idx must be non-negative")
        if len(self.frame_indices) != 2:
            raise DynamicVideoTelemetryError("frame_indices must have exactly two entries")
        if self.frame_indices[0] < 0 or self.frame_indices[1] < 0:
            raise DynamicVideoTelemetryError("frame indices must be non-negative")
        if self.seg_dist < 0 or self.pose_dist < 0:
            raise DynamicVideoTelemetryError("seg_dist and pose_dist must be non-negative")
        if self.hard_pair_rank < 1:
            raise DynamicVideoTelemetryError("hard_pair_rank must be 1-based")


def _axis_label(axis_label: str) -> str:
    label = axis_label.strip()
    if label not in AXIS_LABELS:
        raise DynamicVideoTelemetryError(
            f"axis_label {axis_label!r} is not recognized; expected one of "
            f"{sorted(AXIS_LABELS)}"
        )
    return label


def _sha256_hex(value: str, *, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in _SHA256_HEX_CHARS for ch in digest):
        raise DynamicVideoTelemetryError(f"{field_name} must be a 64-character hex sha256")
    return digest


def _optional_sha256_hex(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _sha256_hex(value, field_name=field_name)


def _eval_custody(axis_label: str, eval_custody: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate exact-eval custody for authoritative contest-axis telemetry."""

    if axis_label not in {"[contest-CUDA]", "[contest-CPU]"}:
        return dict(eval_custody) if eval_custody is not None else None
    if not isinstance(eval_custody, Mapping):
        raise DynamicVideoTelemetryError(
            f"{axis_label} telemetry requires eval_custody with command, "
            "hardware, sample_count, component_recomputed, and auth_eval_json_sha256"
        )
    required = ("command", "hardware", "sample_count", "component_recomputed")
    missing = [name for name in required if name not in eval_custody]
    if missing:
        raise DynamicVideoTelemetryError(
            f"eval_custody missing required fields: {', '.join(missing)}"
        )
    sample_count = int(eval_custody["sample_count"])
    if sample_count <= 0:
        raise DynamicVideoTelemetryError("eval_custody.sample_count must be positive")
    if bool(eval_custody["component_recomputed"]) is not True:
        raise DynamicVideoTelemetryError("eval_custody.component_recomputed must be true")
    auth_eval_sha = eval_custody.get("auth_eval_json_sha256")
    if auth_eval_sha is None:
        raise DynamicVideoTelemetryError("eval_custody.auth_eval_json_sha256 is required")
    out = dict(eval_custody)
    out["sample_count"] = sample_count
    out["auth_eval_json_sha256"] = _sha256_hex(
        str(auth_eval_sha),
        field_name="eval_custody.auth_eval_json_sha256",
    )
    return out


def _row_from_mapping(row: Mapping[str, Any]) -> TelemetryPairRow:
    frame_indices = row.get("frame_indices", ())
    return TelemetryPairRow(
        pair_idx=int(row["pair_idx"]),
        frame_indices=(int(frame_indices[0]), int(frame_indices[1])),
        seg_dist=float(row["seg_dist"]),
        pose_dist=float(row["pose_dist"]),
        score_contribution=float(row["score_contribution"]),
        hard_pair_rank=int(row["hard_pair_rank"]),
        hard_frame_flags=tuple(str(x) for x in row.get("hard_frame_flags", ())),
        xray_artifact_refs=tuple(str(x) for x in row.get("xray_artifact_refs", ())),
    )


def _normalize_rows(rows: Iterable[TelemetryPairRow | Mapping[str, Any]]) -> list[TelemetryPairRow]:
    out: list[TelemetryPairRow] = []
    for row in rows:
        out.append(row if isinstance(row, TelemetryPairRow) else _row_from_mapping(row))
    if not out:
        raise DynamicVideoTelemetryError("at least one telemetry row is required")
    pair_indices = [row.pair_idx for row in out]
    if len(set(pair_indices)) != len(pair_indices):
        raise DynamicVideoTelemetryError("pair_idx values must be unique")
    ranks = sorted(row.hard_pair_rank for row in out)
    if ranks != list(range(1, len(out) + 1)):
        raise DynamicVideoTelemetryError("hard_pair_rank values must form 1..N")
    return sorted(out, key=lambda row: row.hard_pair_rank)


def build_dynamic_video_telemetry(
    *,
    video_sha256: str,
    runtime_tree_sha256: str,
    axis_label: str,
    rows: Iterable[TelemetryPairRow | Mapping[str, Any]],
    scorer_cache_ref: str | None = None,
    source_archive_sha256: str | None = None,
    eval_custody: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a machine-checkable DVAR1 telemetry artifact."""

    normalized_rows = _normalize_rows(rows)
    telemetry_rows = [asdict(row) for row in normalized_rows]
    axis = _axis_label(axis_label)
    eval_custody_out = _eval_custody(axis, eval_custody)
    return {
        "schema": SCHEMA,
        "family": FAMILY,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "inflate_allowed": False,
        "scorer_allowed_at_inflate": False,
        "compress_time_scorer_use_allowed": True,
        "axis_label": axis,
        "eval_custody": eval_custody_out,
        "video_sha256": _sha256_hex(video_sha256, field_name="video_sha256"),
        "source_archive_sha256": _optional_sha256_hex(
            source_archive_sha256,
            field_name="source_archive_sha256",
        ),
        "runtime_tree_sha256": _sha256_hex(
            runtime_tree_sha256,
            field_name="runtime_tree_sha256",
        ),
        "n_pairs": len(normalized_rows),
        "scorer_cache_ref": scorer_cache_ref,
        "pair_rows": telemetry_rows,
        "hard_pair_indices": [row.pair_idx for row in normalized_rows],
        "wire_in_hooks_engaged": [
            "sensitivity_map",
            "bit_allocator",
            "cathedral_autopilot",
            "continual_learning",
            "probe_disambiguator",
        ],
        "dispatch_blockers": [
            "telemetry_only_no_candidate_archive",
            "requires_byte_closed_packet",
            "requires_exact_eval_before_score_claim",
        ],
        "notes": list(notes),
    }


def telemetry_to_hard_pair_indices(
    telemetry: Mapping[str, Any],
    *,
    top_k: int,
) -> list[int]:
    """Return the top-K hard pair indices from a telemetry artifact."""

    if str(telemetry.get("schema")) != SCHEMA:
        raise DynamicVideoTelemetryError(f"expected schema {SCHEMA}")
    if top_k <= 0:
        raise DynamicVideoTelemetryError("top_k must be positive")
    rows = telemetry.get("pair_rows")
    if not isinstance(rows, list) or not rows:
        raise DynamicVideoTelemetryError("telemetry missing pair_rows")
    normalized = _normalize_rows(row for row in rows if isinstance(row, Mapping))
    return [row.pair_idx for row in normalized[: min(top_k, len(normalized))]]


def write_hard_pair_indices_file(
    telemetry: Mapping[str, Any],
    output_path: str | Path,
    *,
    top_k: int,
) -> Path:
    """Write a deterministic pair-index file for the MDL ablation CLI."""

    indices = telemetry_to_hard_pair_indices(telemetry, top_k=top_k)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(indices, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


__all__ = [
    "AXIS_LABELS",
    "DynamicVideoTelemetryError",
    "FAMILY",
    "SCHEMA",
    "TelemetryPairRow",
    "build_dynamic_video_telemetry",
    "telemetry_to_hard_pair_indices",
    "write_hard_pair_indices_file",
]
