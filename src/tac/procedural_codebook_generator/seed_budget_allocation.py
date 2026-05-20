# SPDX-License-Identifier: MIT
"""Per-frame seed-budget planning for procedural codebook candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES

SCHEMA = "procedural_codebook_seed_budget_allocation_v1"
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES
FRAME_SCOPE_KEYS = ("affected_frame_indices", "target_frame_indices", "frame_indices")
DEFAULT_SEED_BUDGET_CANDIDATES = (16, 32, 64, 128, 256)


class SeedBudgetAllocationError(ValueError):
    """Raised when seed-budget allocation inputs are malformed."""


def allocate_seed_budget_from_frame_sensitivity(
    *,
    procedural_candidate: Mapping[str, Any],
    per_frame_decomposition: Mapping[str, Any],
    seed_budget_candidates: Sequence[int] = DEFAULT_SEED_BUDGET_CANDIDATES,
    default_seed_bytes: int = 32,
) -> dict[str, Any]:
    """Allocate seed-budget hints over explicit frame scope.

    The caller must provide the frames affected by the procedural candidate.
    Top-frame ordering alone is only a sensitivity ranking; it is not a frame
    scope declaration.
    """

    n_codebook_bytes = _require_positive_int(
        procedural_candidate.get("n_codebook_bytes"),
        "n_codebook_bytes",
    )
    seed_candidates = _normalise_seed_candidates(seed_budget_candidates)
    if default_seed_bytes <= 0:
        raise SeedBudgetAllocationError("default_seed_bytes must be positive")

    affected_frames = _extract_frame_scope(procedural_candidate)
    if not affected_frames:
        return _fail_closed(
            status="missing_frame_scope",
            reason=(
                "procedural candidate must declare one of "
                f"{', '.join(FRAME_SCOPE_KEYS)}"
            ),
            n_codebook_bytes=n_codebook_bytes,
            seed_budget_candidates=seed_candidates,
        )

    top_by_frame = _top_frame_index(per_frame_decomposition)
    if not top_by_frame:
        return _fail_closed(
            status="missing_per_frame_top_frames",
            reason="per-frame decomposition missing usable top_frames",
            n_codebook_bytes=n_codebook_bytes,
            seed_budget_candidates=seed_candidates,
            affected_frame_indices=affected_frames,
        )

    matched = []
    for frame_index in affected_frames:
        top = top_by_frame.get(frame_index)
        total_l1 = 0.0 if top is None else float(top["total_l1"])
        matched.append(
            {
                "frame_index": frame_index,
                "present_in_top_frames": top is not None,
                "rank": None if top is None else top.get("rank"),
                "total_l1": total_l1,
                "seg_l1": 0.0 if top is None else float(top.get("seg_l1", 0.0)),
                "pose_l1": 0.0 if top is None else float(top.get("pose_l1", 0.0)),
                "rate_l1": 0.0 if top is None else float(top.get("rate_l1", 0.0)),
            }
        )

    recommended_k = (
        default_seed_bytes if default_seed_bytes in seed_candidates else seed_candidates[0]
    )
    allocation = _allocate_integer_budget(
        matched,
        total_budget=recommended_k,
    )
    return {
        "schema": SCHEMA,
        "allocation_status": "allocated",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[predicted]",
        "n_codebook_bytes": n_codebook_bytes,
        "seed_budget_candidates": [
            _candidate_delta(n_codebook_bytes=n_codebook_bytes, k_seed_bytes=k)
            for k in seed_candidates
        ],
        "recommended_k_seed_bytes": recommended_k,
        "affected_frame_indices": affected_frames,
        "matched_frame_sensitivity": matched,
        "allocation": allocation,
        "topology": per_frame_decomposition.get("topology"),
        "n_pairs": per_frame_decomposition.get("n_pairs"),
        "n_frames": per_frame_decomposition.get("n_frames"),
    }


def _fail_closed(
    *,
    status: str,
    reason: str,
    n_codebook_bytes: int,
    seed_budget_candidates: tuple[int, ...],
    affected_frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "allocation_status": status,
        "reason": reason,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[predicted]",
        "n_codebook_bytes": n_codebook_bytes,
        "seed_budget_candidates": [
            _candidate_delta(n_codebook_bytes=n_codebook_bytes, k_seed_bytes=k)
            for k in seed_budget_candidates
        ],
        "recommended_k_seed_bytes": None,
        "affected_frame_indices": [] if affected_frame_indices is None else affected_frame_indices,
        "matched_frame_sensitivity": [],
        "allocation": [],
    }


def _candidate_delta(*, n_codebook_bytes: int, k_seed_bytes: int) -> dict[str, Any]:
    bytes_saved = max(0, n_codebook_bytes - k_seed_bytes)
    return {
        "k_seed_bytes": k_seed_bytes,
        "bytes_saved": bytes_saved,
        "predicted_delta_s": -RATE_SCORE_PER_BYTE * bytes_saved,
    }


def _extract_frame_scope(procedural_candidate: Mapping[str, Any]) -> list[int]:
    for key in FRAME_SCOPE_KEYS:
        value = procedural_candidate.get(key)
        if value is None:
            continue
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SeedBudgetAllocationError(f"{key} must be a sequence of frame indices")
        out: list[int] = []
        for item in value:
            if not isinstance(item, int) or item < 0:
                raise SeedBudgetAllocationError(f"{key} contains invalid frame index {item!r}")
            if item not in out:
                out.append(item)
        return out
    return []


def _top_frame_index(per_frame_decomposition: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = per_frame_decomposition.get("top_frames")
    if not isinstance(rows, list):
        return {}
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("frame_index"), int):
            continue
        total_l1 = row.get("total_l1")
        try:
            total_l1_float = float(total_l1)
        except (TypeError, ValueError):
            continue
        if total_l1_float < 0.0:
            continue
        out[int(row["frame_index"])] = row
    return out


def _allocate_integer_budget(
    frame_rows: list[dict[str, Any]],
    *,
    total_budget: int,
) -> list[dict[str, Any]]:
    if total_budget <= 0 or not frame_rows:
        return []
    total_l1 = sum(float(row["total_l1"]) for row in frame_rows)
    if total_l1 <= 0.0:
        weights = [1.0 / len(frame_rows)] * len(frame_rows)
    else:
        weights = [float(row["total_l1"]) / total_l1 for row in frame_rows]
    raw = [weight * total_budget for weight in weights]
    floors = [int(value) for value in raw]
    remaining = total_budget - sum(floors)
    order = sorted(
        range(len(frame_rows)),
        key=lambda i: (raw[i] - floors[i], weights[i]),
        reverse=True,
    )
    for index in order[:remaining]:
        floors[index] += 1
    return [
        {
            "frame_index": int(row["frame_index"]),
            "relative_sensitivity_weight": weights[index],
            "seed_budget_hint_bytes": floors[index],
        }
        for index, row in enumerate(frame_rows)
    ]


def _normalise_seed_candidates(values: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise SeedBudgetAllocationError("seed budget candidates must be positive ints")
        if value not in out:
            out.append(value)
    if not out:
        raise SeedBudgetAllocationError("seed budget candidates must be non-empty")
    return tuple(sorted(out))


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise SeedBudgetAllocationError(f"{name} must be a positive int")
    return value
