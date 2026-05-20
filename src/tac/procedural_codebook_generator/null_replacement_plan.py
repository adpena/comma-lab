# SPDX-License-Identifier: MIT
"""Plan archive-charged seed replacements for master-gradient null bytes.

This module consumes the observability output from
``tools/probe_null_byte_master_gradient.py`` and turns it into fail-closed
packet-compiler targets. It does not rewrite archives, run scorers, dispatch
jobs, or claim score movement. The only score-like values are rate-term
upper bounds from bytes that would have to be removed by a future
byte-closed packet compiler and then validated by exact eval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.authority_contract import apply_false_authority_contract
from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "null_space_seed_replacement_plan_v1"
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES


@dataclass(frozen=True)
class NullSpanCandidate:
    """One null-byte span that a future packet compiler may replace with a seed."""

    candidate_id: str
    source: str
    section: str
    start: int
    end: int
    original_bytes: int
    seed_bytes: int
    runtime_header_bytes: int = 0
    original_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("candidate range must be non-empty and non-negative")
        if self.original_bytes != self.end - self.start:
            raise ValueError("original_bytes must equal end - start")
        if self.seed_bytes <= 0:
            raise ValueError("seed_bytes must be positive")
        if self.runtime_header_bytes < 0:
            raise ValueError("runtime_header_bytes must be non-negative")

    @property
    def net_saved_inner_bytes(self) -> int:
        return self.original_bytes - self.seed_bytes - self.runtime_header_bytes

    @property
    def predicted_rate_delta_upper_bound(self) -> float:
        return -RATE_SCORE_PER_BYTE * max(0, self.net_saved_inner_bytes)

    def as_dict(self) -> dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "section": self.section,
            "range": [self.start, self.end],
            "original_bytes": self.original_bytes,
            "seed_bytes": self.seed_bytes,
            "runtime_header_bytes": self.runtime_header_bytes,
            "net_saved_inner_bytes": self.net_saved_inner_bytes,
            "predicted_rate_delta_upper_bound": self.predicted_rate_delta_upper_bound,
            "original_sha256": self.original_sha256,
            "required_next_proofs": [
                "stock_inflate_runtime_consumes_seed_inside_archive_zip",
                "seed_mutation_changes_inflated_frames_or_expected_null_region",
                "packet_compiler_candidate_not_raw_byte_splice",
                "exact_cuda_eval_on_shrunk_archive",
            ],
            "research_only": True,
        }
        return apply_false_authority_contract(
            row,
            preserve_dispatch_ready=False,
            reason="null_seed_replacement_candidate_requires_packet_compiler_and_exact_eval",
        )


def contiguous_runs(indices: np.ndarray | Sequence[int]) -> list[tuple[int, int]]:
    """Return sorted half-open contiguous runs from byte indices."""

    arr = np.asarray(indices, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("indices must be a one-dimensional array")
    if arr.size == 0:
        return []
    unique = np.unique(arr)
    if np.any(unique < 0):
        raise ValueError("indices must be non-negative")
    breaks = np.nonzero(np.diff(unique) != 1)[0] + 1
    parts = np.split(unique, breaks)
    return [(int(part[0]), int(part[-1]) + 1) for part in parts if part.size]


def build_null_seed_replacement_plan(
    *,
    null_summary: Mapping[str, Any],
    null_indices: np.ndarray | Sequence[int],
    inner_bytes: bytes | None = None,
    seed_bytes: int = 8,
    runtime_header_bytes: int = 0,
    min_run_length: int = 16,
    max_candidates: int = 50,
) -> dict[str, Any]:
    """Build a fail-closed replacement plan from null-byte probe outputs."""

    n_total = _require_int(null_summary, "n_total_bytes")
    n_null = _require_int(null_summary, "n_null_bytes")
    epsilon = null_summary.get("epsilon")
    _validate_positive_int(seed_bytes, "seed_bytes")
    _validate_nonnegative_int(runtime_header_bytes, "runtime_header_bytes")
    _validate_positive_int(min_run_length, "min_run_length")
    _validate_positive_int(max_candidates, "max_candidates")
    if inner_bytes is not None and len(inner_bytes) != n_total:
        raise ValueError(
            f"inner_bytes length {len(inner_bytes)} does not match n_total_bytes {n_total}"
        )

    indices = _normalise_null_indices(null_indices, n_total=n_total)
    if int(indices.size) != n_null:
        raise ValueError(
            f"null_indices count {indices.size} does not match n_null_bytes {n_null}"
        )

    raw_candidates = [
        *_section_candidates(
            null_summary=null_summary,
            inner_bytes=inner_bytes,
            seed_bytes=seed_bytes,
            runtime_header_bytes=runtime_header_bytes,
            min_run_length=min_run_length,
        ),
        *_run_candidates(
            null_summary=null_summary,
            null_indices=indices,
            inner_bytes=inner_bytes,
            seed_bytes=seed_bytes,
            runtime_header_bytes=runtime_header_bytes,
            min_run_length=min_run_length,
        ),
    ]
    candidates = _rank_candidates(_dedupe_candidates(raw_candidates))
    positive = [candidate for candidate in candidates if candidate.net_saved_inner_bytes > 0]
    disjoint = _greedy_disjoint(positive)
    selected = candidates[:max_candidates]

    plan = {
        "schema": SCHEMA,
        "inputs": {
            "n_total_bytes": n_total,
            "n_null_bytes": n_null,
            "null_fraction": float(null_summary.get("null_fraction", 0.0)),
            "epsilon": epsilon,
            "seed_bytes": int(seed_bytes),
            "runtime_header_bytes": int(runtime_header_bytes),
            "min_run_length": int(min_run_length),
            "max_candidates": int(max_candidates),
            "inner_bytes_sha256": (
                hashlib.sha256(inner_bytes).hexdigest() if inner_bytes is not None else None
            ),
            "source_null_probe_schema": null_summary.get("schema"),
        },
        "summary": {
            "candidate_count": len(candidates),
            "emitted_candidate_count": len(selected),
            "positive_candidate_count": len(positive),
            "best_net_saved_inner_bytes": (
                max((candidate.net_saved_inner_bytes for candidate in candidates), default=0)
            ),
            "greedy_disjoint_positive_candidate_count": len(disjoint),
            "greedy_disjoint_net_saved_inner_bytes_upper_bound": sum(
                candidate.net_saved_inner_bytes for candidate in disjoint
            ),
            "greedy_disjoint_rate_delta_upper_bound": -RATE_SCORE_PER_BYTE
            * sum(candidate.net_saved_inner_bytes for candidate in disjoint),
        },
        "candidates": [candidate.as_dict() for candidate in selected],
        "research_only": True,
        "planner_note": (
            "Planning-only null-span ranking. A future packet compiler must replace "
            "the span with archive-charged seed bytes, prove runtime consumption, "
            "and pass exact CUDA eval before any score claim."
        ),
    }
    return apply_false_authority_contract(
        plan,
        preserve_dispatch_ready=False,
        reason="null_seed_replacement_plan_requires_packet_compiler_and_exact_eval",
    )


def render_null_seed_replacement_markdown(plan: Mapping[str, Any]) -> str:
    """Render a compact operator-readable summary for a replacement plan."""

    summary = _require_mapping(plan.get("summary"), "summary")
    inputs = _require_mapping(plan.get("inputs"), "inputs")
    lines = [
        "# Null-Space Seed Replacement Plan",
        "",
        f"- Schema: `{plan.get('schema')}`",
        f"- Score claim: `{str(plan.get('score_claim')).lower()}`",
        f"- Promotion eligible: `{str(plan.get('promotion_eligible')).lower()}`",
        f"- Input bytes: `{inputs.get('n_total_bytes')}`",
        f"- Null bytes: `{inputs.get('n_null_bytes')}`",
        f"- Seed bytes per candidate: `{inputs.get('seed_bytes')}`",
        f"- Runtime header bytes per candidate: `{inputs.get('runtime_header_bytes')}`",
        f"- Candidate count: `{summary.get('candidate_count')}`",
        f"- Positive candidates: `{summary.get('positive_candidate_count')}`",
        f"- Best net saved inner bytes: `{summary.get('best_net_saved_inner_bytes')}`",
        "- Greedy disjoint rate-delta upper bound: "
        f"`{float(summary.get('greedy_disjoint_rate_delta_upper_bound', 0.0)):.12f}`",
        "",
        "| rank | source | section | range | original bytes | net saved | rate delta upper bound |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, candidate in enumerate(plan.get("candidates", []), start=1):
        row = _require_mapping(candidate, f"candidate:{rank}")
        start, end = row.get("range", [None, None])
        lines.append(
            "| {rank} | `{source}` | `{section}` | `{start}-{end}` | {orig} | {net} | {delta:.12f} |".format(
                rank=rank,
                source=row.get("source"),
                section=row.get("section"),
                start=start,
                end=end,
                orig=int(row.get("original_bytes", 0)),
                net=int(row.get("net_saved_inner_bytes", 0)),
                delta=float(row.get("predicted_rate_delta_upper_bound", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "This is not a score artifact. It is a packet-compiler target list for",
            "archive-charged seed replacement candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def _section_candidates(
    *,
    null_summary: Mapping[str, Any],
    inner_bytes: bytes | None,
    seed_bytes: int,
    runtime_header_bytes: int,
    min_run_length: int,
) -> list[NullSpanCandidate]:
    candidates: list[NullSpanCandidate] = []
    for section, (start, end) in _section_ranges(null_summary).items():
        info = _require_mapping(
            _require_mapping(null_summary.get("section_breakdown"), "section_breakdown").get(
                section
            ),
            f"section_breakdown.{section}",
        )
        if float(info.get("null_fraction_within_section", 0.0)) != 1.0:
            continue
        if end - start < min_run_length:
            continue
        candidates.append(
            _candidate(
                source="whole_null_section",
                section=section,
                start=start,
                end=end,
                inner_bytes=inner_bytes,
                seed_bytes=seed_bytes,
                runtime_header_bytes=runtime_header_bytes,
            )
        )
    return candidates


def _run_candidates(
    *,
    null_summary: Mapping[str, Any],
    null_indices: np.ndarray,
    inner_bytes: bytes | None,
    seed_bytes: int,
    runtime_header_bytes: int,
    min_run_length: int,
) -> list[NullSpanCandidate]:
    candidates: list[NullSpanCandidate] = []
    section_ranges = _section_ranges(null_summary)
    for start, end in contiguous_runs(null_indices):
        if end - start < min_run_length:
            continue
        candidates.append(
            _candidate(
                source="contiguous_null_run",
                section=_section_for_range(section_ranges, start, end),
                start=start,
                end=end,
                inner_bytes=inner_bytes,
                seed_bytes=seed_bytes,
                runtime_header_bytes=runtime_header_bytes,
            )
        )
    return candidates


def _candidate(
    *,
    source: str,
    section: str,
    start: int,
    end: int,
    inner_bytes: bytes | None,
    seed_bytes: int,
    runtime_header_bytes: int,
) -> NullSpanCandidate:
    section_id = section.replace(":", "_").replace("+", "_")
    candidate_id = f"null-seed-{source}-{section_id}-{start}-{end}"
    return NullSpanCandidate(
        candidate_id=candidate_id,
        source=source,
        section=section,
        start=start,
        end=end,
        original_bytes=end - start,
        seed_bytes=seed_bytes,
        runtime_header_bytes=runtime_header_bytes,
        original_sha256=_slice_sha(inner_bytes, start, end),
    )


def _section_ranges(null_summary: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    section_breakdown = _require_mapping(
        null_summary.get("section_breakdown"), "section_breakdown"
    )
    out: dict[str, tuple[int, int]] = {}
    for section, info_raw in section_breakdown.items():
        if str(section).startswith("_"):
            continue
        info = _require_mapping(info_raw, f"section_breakdown.{section}")
        value = info.get("range")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"section_breakdown.{section}.range must be [start, end]")
        start = int(value[0])
        end = int(value[1])
        if start < 0 or end <= start:
            raise ValueError(f"section {section} has invalid range {value!r}")
        out[str(section)] = (start, end)
    return dict(sorted(out.items(), key=lambda item: (item[1][0], item[1][1], item[0])))


def _section_for_range(
    section_ranges: Mapping[str, tuple[int, int]], start: int, end: int
) -> str:
    covered = [
        name
        for name, (section_start, section_end) in section_ranges.items()
        if start >= section_start and end <= section_end
    ]
    if covered:
        return covered[0]
    touched = [
        name
        for name, (section_start, section_end) in section_ranges.items()
        if start < section_end and end > section_start
    ]
    return "+".join(touched) if touched else "unbucketed"


def _dedupe_candidates(candidates: Sequence[NullSpanCandidate]) -> list[NullSpanCandidate]:
    seen: set[tuple[str, int, int]] = set()
    out: list[NullSpanCandidate] = []
    for candidate in candidates:
        key = (candidate.source, candidate.start, candidate.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _rank_candidates(candidates: Sequence[NullSpanCandidate]) -> list[NullSpanCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.net_saved_inner_bytes,
            candidate.original_bytes,
            -candidate.start,
            candidate.source,
        ),
        reverse=True,
    )


def _greedy_disjoint(candidates: Sequence[NullSpanCandidate]) -> list[NullSpanCandidate]:
    selected: list[NullSpanCandidate] = []
    occupied: list[tuple[int, int]] = []
    for candidate in _rank_candidates(candidates):
        if any(candidate.start < end and candidate.end > start for start, end in occupied):
            continue
        selected.append(candidate)
        occupied.append((candidate.start, candidate.end))
    return selected


def _slice_sha(inner_bytes: bytes | None, start: int, end: int) -> str | None:
    if inner_bytes is None:
        return None
    return hashlib.sha256(inner_bytes[start:end]).hexdigest()


def _normalise_null_indices(
    null_indices: np.ndarray | Sequence[int], *, n_total: int
) -> np.ndarray:
    arr = np.asarray(null_indices, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("null_indices must be a one-dimensional array")
    arr = np.unique(arr)
    if np.any(arr < 0) or np.any(arr >= n_total):
        raise ValueError("null_indices contains offsets outside n_total_bytes")
    return arr


def _require_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


__all__ = [
    "RATE_SCORE_PER_BYTE",
    "SCHEMA",
    "NullSpanCandidate",
    "build_null_seed_replacement_plan",
    "contiguous_runs",
    "render_null_seed_replacement_markdown",
]
