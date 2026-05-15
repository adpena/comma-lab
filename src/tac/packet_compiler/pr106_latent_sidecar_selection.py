# SPDX-License-Identifier: MIT
"""PR106 latent-sidecar score-table selection helpers.

This module is scorer-free and archive-emission-free. It lowers an already
measured PR106 latent score table into deterministic per-pair ``(dim, delta_q)``
corrections, then profiles the byte cost of those corrections through the
canonical PR106 sidecar PacketIR grammars.

The score table is a compress-time selector signal only. Nothing in this module
creates an exact-eval packet or a score claim.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_DEFAULT_MEMBER_NAME,
    PR106_LATENT_N_DIMS,
    PR106_NO_OP_DIM,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED,
    lossless_pr106_sidecar_recode_candidates,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
    sha256_hex,
)

ARCHIVE_BYTES_DENOMINATOR = 37_545_489


def build_latent_candidate_grid(
    *,
    latent_dim: int = PR106_LATENT_N_DIMS,
    delta_radius: int = 1,
) -> np.ndarray:
    """Return canonical ``[dim, delta_q]`` candidates for score-table search.

    Row 0 is the no-op sentinel ``[255, 0]``. Remaining rows enumerate every
    latent dimension with nonzero integer deltas in
    ``[-delta_radius, +delta_radius]``.
    """

    if latent_dim <= 0 or latent_dim >= PR106_NO_OP_DIM:
        raise ValueError(
            f"latent_dim must be in 1..{PR106_NO_OP_DIM - 1}, got {latent_dim}"
        )
    if delta_radius < 1 or delta_radius > 127:
        raise ValueError(f"delta_radius must be in 1..127, got {delta_radius}")
    rows: list[tuple[int, int]] = [(PR106_NO_OP_DIM, 0)]
    for dim in range(latent_dim):
        for delta_q in range(-delta_radius, delta_radius + 1):
            if delta_q == 0:
                continue
            rows.append((dim, delta_q))
    return np.asarray(rows, dtype=np.int16)


def latent_candidate_grid_npy_sha256(candidates: np.ndarray) -> str:
    """Return deterministic ``.npy`` SHA-256 for candidate-grid custody."""

    raw = io.BytesIO()
    np.save(raw, np.asarray(candidates, dtype=np.int16), allow_pickle=False)
    return hashlib.sha256(raw.getvalue()).hexdigest()


def _validated_score_table_and_candidates(
    score_table: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    scores = np.asarray(score_table, dtype=np.float64)
    cands = np.asarray(candidates)
    if cands.dtype.kind not in {"i", "u"}:
        raise TypeError(f"candidates must be integer typed, got {cands.dtype}")
    if cands.ndim != 2 or cands.shape[1] != 2:
        raise ValueError(f"candidates must have shape (n_candidates, 2), got {cands.shape}")
    if scores.ndim != 2 or scores.shape[1] != cands.shape[0]:
        raise ValueError(
            "score_table must have shape (n_pairs, n_candidates), got "
            f"{scores.shape} for {cands.shape[0]} candidates"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_table contains NaN/Inf")
    noop_matches = np.flatnonzero(
        (cands[:, 0] == PR106_NO_OP_DIM) & (cands[:, 1] == 0)
    )
    if len(noop_matches) != 1:
        raise ValueError("candidates must contain exactly one [255, 0] no-op row")
    return scores, cands, int(noop_matches[0])


def _select_latent_corrections_from_scores(
    score_table: np.ndarray,
    candidates: np.ndarray,
    *,
    top_k: int | None = None,
    require_improvement: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    scores, cands, noop_idx = _validated_score_table_and_candidates(score_table, candidates)
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    best_idx = scores.argmin(axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    noop_scores = scores[:, noop_idx]
    raw_improvements = noop_scores - best_scores
    selected_improvements = raw_improvements.copy()
    if require_improvement:
        best_idx = np.where(raw_improvements > 0.0, best_idx, noop_idx)
        selected_improvements = np.where(raw_improvements > 0.0, raw_improvements, 0.0)

    selected = cands[best_idx].astype(np.int16, copy=True)
    selected_nonzero = (selected[:, 0] != PR106_NO_OP_DIM) & (selected[:, 1] != 0)
    if top_k is not None and top_k < int(selected_nonzero.sum()):
        keep = np.zeros(scores.shape[0], dtype=bool)
        if top_k > 0:
            # Stable descending improvement, ascending pair index. This makes
            # repeated tool runs byte-identical even when improvements tie.
            ranked_pairs = np.lexsort((np.arange(scores.shape[0]), -selected_improvements))
            kept = 0
            for pair_idx in ranked_pairs.tolist():
                if selected_nonzero[pair_idx]:
                    keep[pair_idx] = True
                    kept += 1
                    if kept >= top_k:
                        break
        selected[~keep] = (PR106_NO_OP_DIM, 0)
        selected_improvements = np.where(keep, selected_improvements, 0.0)
        selected_nonzero = (selected[:, 0] != PR106_NO_OP_DIM) & (selected[:, 1] != 0)

    dim_arr = selected[:, 0].astype(np.uint8)
    delta_q_arr = selected[:, 1].astype(np.int8)
    raw_positive = raw_improvements > 0.0
    diagnostics: dict[str, Any] = {
        "search_mode": "score_table",
        "n_pairs": int(scores.shape[0]),
        "n_corrections": int(selected_nonzero.sum()),
        "n_no_op": int((~selected_nonzero).sum()),
        "nonzero_delta_count": int(np.count_nonzero(selected[:, 1])),
        "candidate_count": int(cands.shape[0]),
        "noop_candidate_index": int(noop_idx),
        "selected_nonzero_pair_count": int(selected_nonzero.sum()),
        "selected_noop_pair_count": int((~selected_nonzero).sum()),
        "strict_improvement_pair_count": int(raw_positive.sum()),
        "best_improvement_min": float(raw_improvements.min()),
        "best_improvement_mean": float(raw_improvements.mean()),
        "best_improvement_max": float(raw_improvements.max()),
        "selected_improvement_sum": float(selected_improvements.sum()),
        "selected_improvement_mean_per_pair": float(selected_improvements.mean()),
        "selected_improvement_max": float(selected_improvements.max())
        if selected_improvements.size
        else 0.0,
        "require_strict_improvement_over_noop": bool(require_improvement),
        "top_k_cap": None if top_k is None else int(top_k),
        "scorer_available": False,
    }
    return dim_arr, delta_q_arr, diagnostics, selected_improvements


def choose_latent_corrections_from_scores(
    score_table: np.ndarray,
    candidates: np.ndarray,
    *,
    top_k: int | None = None,
    require_improvement: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reduce a scorer table into one latent correction per pair.

    ``score_table[p, c]`` must be the compress-time measured pair objective for
    pair ``p`` under candidate ``c``. This reducer does not load scorers or
    create packets; it only turns measured table entries into correction arrays.
    """

    dim_arr, delta_q_arr, diagnostics, _selected_improvements = (
        _select_latent_corrections_from_scores(
            score_table,
            candidates,
            top_k=top_k,
            require_improvement=require_improvement,
        )
    )
    return dim_arr, delta_q_arr, diagnostics


def choose_latent_corrections_from_score_table_file(
    score_table_npy: Path,
    *,
    n_pairs: int,
    latent_dim: int,
    delta_radius: int,
    top_k: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a scorer table and reduce it into deterministic corrections."""

    candidates = build_latent_candidate_grid(
        latent_dim=latent_dim,
        delta_radius=delta_radius,
    )
    loaded = np.load(score_table_npy, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        raise TypeError(f"score table must be a .npy ndarray, got {type(loaded).__name__}")
    if loaded.shape != (n_pairs, len(candidates)):
        raise ValueError(
            "score table shape mismatch: expected "
            f"({n_pairs}, {len(candidates)}), got {loaded.shape}"
        )
    dim_arr, delta_q_arr, diagnostics = choose_latent_corrections_from_scores(
        loaded,
        candidates,
        top_k=top_k,
        require_improvement=True,
    )
    diagnostics.update(
        {
            "latent_dim": int(latent_dim),
            "delta_radius": int(delta_radius),
            "candidate_grid_sha256": latent_candidate_grid_npy_sha256(candidates),
            "score_table_shape": [int(loaded.shape[0]), int(loaded.shape[1])],
        }
    )
    return dim_arr, delta_q_arr, diagnostics


def validate_score_table_manifest(
    manifest_path: Path,
    *,
    score_table_npy: Path,
    source_archive: Path,
    n_pairs: int,
    latent_dim: int,
    delta_radius: int,
    candidate_count: int,
) -> dict[str, object]:
    """Validate PR106 latent score-table provenance before reducing to bytes."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"score table manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("score table manifest must be a JSON object")
    if manifest.get("manifest_schema") != "pr106_latent_score_table_manifest_v1":
        raise ValueError("score table manifest_schema mismatch")
    if manifest.get("producer") != "experiments/build_pr106_latent_score_table.py":
        raise ValueError("score table manifest producer mismatch")
    if manifest.get("score_claim") is not False:
        raise ValueError("score table manifest must keep score_claim=false")
    if manifest.get("ready_for_builder") is not True:
        raise ValueError("score table manifest must have ready_for_builder=true")
    archive_sha256 = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    archive_sha256_matches = manifest.get("source_archive_sha256") == archive_sha256
    member = read_single_stored_member_archive(source_archive.read_bytes())
    member_name_matches = manifest.get("source_archive_member_name") in (None, member.name)
    member_sha256 = sha256_hex(member.payload)
    member_sha256_matches = manifest.get("source_archive_member_sha256") == member_sha256
    member_format_id: int | None = None
    try:
        member_format_id = parse_pr106_sidecar_packet(member.payload).format_id
    except ValueError:
        member_format_id = None
    format0c_requires_member_sha = (
        member_format_id
        == PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_EXACT_RADIX_DIM_FIXED_META_NOOP_RANK_ELIDED
    )
    if format0c_requires_member_sha and manifest.get("source_archive_member_name") != member.name:
        raise ValueError(
            "score table manifest source_archive_member_name mismatch for "
            "format0C source archive"
        )
    if format0c_requires_member_sha and not member_sha256_matches:
        raise ValueError(
            "score table manifest source_archive_member_sha256 mismatch for "
            "format0C source archive"
        )
    zero_bin_sha256_matches = False
    if not archive_sha256_matches:
        source_zero_bin_sha256 = manifest.get("source_zero_bin_sha256")
        legacy_zero_bin_fallback_allowed = (
            member.name == PR106_DEFAULT_MEMBER_NAME
            and not format0c_requires_member_sha
        )
        if isinstance(source_zero_bin_sha256, str) and legacy_zero_bin_fallback_allowed:
            zero_bin_sha256_matches = member_sha256 == source_zero_bin_sha256
        if not member_sha256_matches and not zero_bin_sha256_matches:
            raise ValueError("score table manifest source archive payload mismatch")
    if not member_name_matches:
        raise ValueError("score table manifest source archive member name mismatch")
    if manifest.get("score_table_npy_sha256") != hashlib.sha256(
        score_table_npy.read_bytes()
    ).hexdigest():
        raise ValueError("score table manifest score_table_npy_sha256 mismatch")
    expected_grid_sha256 = latent_candidate_grid_npy_sha256(
        build_latent_candidate_grid(latent_dim=latent_dim, delta_radius=delta_radius)
    )
    if manifest.get("candidate_grid_sha256") != expected_grid_sha256:
        raise ValueError("score table manifest candidate_grid_sha256 mismatch")
    if manifest.get("n_pairs") != int(n_pairs):
        raise ValueError("score table manifest n_pairs mismatch")
    if manifest.get("latent_dim") != int(latent_dim):
        raise ValueError("score table manifest latent_dim mismatch")
    if manifest.get("delta_radius") != int(delta_radius):
        raise ValueError("score table manifest delta_radius mismatch")
    if manifest.get("candidate_count") != int(candidate_count):
        raise ValueError("score table manifest candidate_count mismatch")
    if manifest.get("score_table_shape") != [int(n_pairs), int(candidate_count)]:
        raise ValueError("score table manifest score_table_shape mismatch")
    if manifest.get("ready_for_exact_eval_dispatch") is True:
        raise ValueError("score table manifest must not claim exact-eval dispatch readiness")
    if manifest.get("dispatch_attempted") is True or manifest.get("remote_jobs_dispatched") is True:
        raise ValueError("score table manifest must not mark dispatch attempted")
    manifest = dict(manifest)
    manifest["validated_source_archive_sha256_match"] = archive_sha256_matches
    manifest["validated_source_archive_member_name_match"] = member_name_matches
    manifest["validated_source_archive_member_sha256_match"] = member_sha256_matches
    manifest["validated_source_zero_bin_sha256_match"] = zero_bin_sha256_matches
    manifest["validated_source_sidecar_format_id"] = (
        None if member_format_id is None else f"0x{member_format_id:02X}"
    )
    return manifest


def _normalized_top_k_values(
    values: list[int] | tuple[int, ...],
    *,
    max_positive: int,
) -> list[int]:
    all_values = {0, int(max_positive)}
    for value in values:
        if value < 0:
            raise ValueError(f"top_k values must be non-negative, got {value}")
        all_values.add(int(value))
    return sorted(all_values)


def _best_runtime_candidate(row_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    runtime_rows = [
        row
        for row in row_candidates
        if row["applicable"] and row["runtime_decoder_implemented"]
    ]
    if not runtime_rows:
        return None
    return min(runtime_rows, key=lambda row: (row["charged_sidecar_bytes"], row["name"]))


def profile_latent_sidecar_topk_pareto(
    score_table: np.ndarray,
    candidates: np.ndarray,
    *,
    top_k_values: list[int] | tuple[int, ...],
    require_improvement: bool = True,
) -> dict[str, Any]:
    """Build a planning-only top-k Pareto profile for PR106 sidecar selection."""

    scores, cands, noop_idx = _validated_score_table_and_candidates(score_table, candidates)
    noop_scores = scores[:, noop_idx]
    best_scores = scores[np.arange(scores.shape[0]), scores.argmin(axis=1)]
    raw_improvements = noop_scores - best_scores
    positive_count = int((raw_improvements > 0.0).sum()) if require_improvement else scores.shape[0]
    evaluated_top_k = _normalized_top_k_values(top_k_values, max_positive=positive_count)

    rows: list[dict[str, Any]] = []
    baseline_runtime_bytes: int | None = None
    full_improvement_sum: float | None = None
    for top_k in evaluated_top_k:
        dim_arr, delta_q_arr, diagnostics, selected_improvements = (
            _select_latent_corrections_from_scores(
                scores,
                cands,
                top_k=top_k,
                require_improvement=require_improvement,
            )
        )
        recode_rows: list[dict[str, Any]] = []
        for recode in lossless_pr106_sidecar_recode_candidates(dim_arr, delta_q_arr):
            applicable = bool(recode.encoded_bytes)
            recode_rows.append(
                {
                    "name": recode.name,
                    "applicable": applicable,
                    "charged_sidecar_bytes": recode.charged_bytes if applicable else None,
                    "sidecar_format_id": None
                    if recode.sidecar_format_id is None
                    else f"0x{recode.sidecar_format_id:02X}",
                    "encoded_payload_bytes": len(recode.encoded_bytes),
                    "encoded_payload_sha256": sha256_hex(recode.encoded_bytes)
                    if applicable
                    else None,
                    "framing_meta_bytes": len(recode.framing_meta_bytes),
                    "framing_meta_sha256": sha256_hex(recode.framing_meta_bytes)
                    if recode.framing_meta_bytes
                    else None,
                    "runtime_decoder_implemented": recode.runtime_decoder_implemented,
                    "lossless_semantic_equivalence_proven": applicable,
                    "notes": list(recode.notes),
                }
            )
        best_runtime = _best_runtime_candidate(recode_rows)
        runtime_bytes = None if best_runtime is None else best_runtime["charged_sidecar_bytes"]
        improvement_sum = float(selected_improvements.sum())
        if top_k == 0:
            baseline_runtime_bytes = runtime_bytes
        if top_k == positive_count:
            full_improvement_sum = improvement_sum
        rows.append(
            {
                "top_k_cap": int(top_k),
                "n_corrections": int(diagnostics["n_corrections"]),
                "selector_improvement_sum": improvement_sum,
                "selector_improvement_mean_per_pair": float(selected_improvements.mean()),
                "selector_improvement_sha256": sha256_hex(
                    np.asarray(selected_improvements, dtype=np.float64).tobytes()
                ),
                "dim_sha256": sha256_hex(dim_arr.astype(np.uint8).tobytes()),
                "delta_q_sha256": sha256_hex(delta_q_arr.astype(np.int8).tobytes()),
                "diagnostics": diagnostics,
                "sidecar_candidates": recode_rows,
                "best_runtime_consumed_sidecar": best_runtime,
                "best_runtime_consumed_charged_bytes": runtime_bytes,
            }
        )

    if baseline_runtime_bytes is None:
        baseline_runtime_bytes = rows[0]["best_runtime_consumed_charged_bytes"]
    if full_improvement_sum is None:
        full_improvement_sum = max(float(row["selector_improvement_sum"]) for row in rows)
    for row in rows:
        runtime_bytes = row["best_runtime_consumed_charged_bytes"]
        if runtime_bytes is None or baseline_runtime_bytes is None:
            row["rate_score_delta_vs_topk0_sidecar_if_runtime_consumed"] = None
        else:
            row["rate_score_delta_vs_topk0_sidecar_if_runtime_consumed"] = (
                25.0 * (runtime_bytes - baseline_runtime_bytes) / ARCHIVE_BYTES_DENOMINATOR
            )
        row["selector_improvement_retained_fraction_vs_full"] = (
            None
            if not full_improvement_sum
            else float(row["selector_improvement_sum"]) / float(full_improvement_sum)
        )

    frontier: list[dict[str, Any]] = []
    for row in rows:
        row_bytes = row["best_runtime_consumed_charged_bytes"]
        if row_bytes is None:
            continue
        row_gain = float(row["selector_improvement_sum"])
        dominated = False
        for other in rows:
            other_bytes = other["best_runtime_consumed_charged_bytes"]
            if other_bytes is None or other is row:
                continue
            other_gain = float(other["selector_improvement_sum"])
            if other_bytes <= row_bytes and other_gain >= row_gain and (
                other_bytes < row_bytes or other_gain > row_gain
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "top_k_cap": row["top_k_cap"],
                    "n_corrections": row["n_corrections"],
                    "selector_improvement_sum": row["selector_improvement_sum"],
                    "best_runtime_consumed_charged_bytes": row_bytes,
                    "rate_score_delta_vs_topk0_sidecar_if_runtime_consumed": row[
                        "rate_score_delta_vs_topk0_sidecar_if_runtime_consumed"
                    ],
                }
            )

    return {
        "schema": "pr106_latent_sidecar_topk_pareto_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "selector_objective_is_exact_score": False,
        "adversarial_claim_check": {
            "verdict": "planning_only_selector_profile",
            "interpretation": (
                "selector_improvement_* values are compress-time score-table deltas "
                "only. They are not contest scores and cannot promote or retire a "
                "packet without byte-closed runtime consumption and exact contest eval."
            ),
        },
        "score_table_shape": [int(scores.shape[0]), int(scores.shape[1])],
        "candidate_grid_sha256": latent_candidate_grid_npy_sha256(cands),
        "noop_candidate_index": int(noop_idx),
        "strict_improvement_pair_count": int((raw_improvements > 0.0).sum()),
        "best_raw_improvement_sum": float(np.where(raw_improvements > 0, raw_improvements, 0).sum()),
        "evaluated_top_k_values": evaluated_top_k,
        "rows": rows,
        "pareto_frontier": frontier,
        "dispatch_blockers": [
            "no_candidate_archive_emitted",
            "selector_objective_not_exact_contest_score",
            "missing_no_op_runtime_consumption_proof_for_new_semantic_selection",
            "missing_exact_contest_eval_for_any_candidate",
        ],
    }


__all__ = [
    "ARCHIVE_BYTES_DENOMINATOR",
    "build_latent_candidate_grid",
    "choose_latent_corrections_from_score_table_file",
    "choose_latent_corrections_from_scores",
    "latent_candidate_grid_npy_sha256",
    "profile_latent_sidecar_topk_pareto",
    "validate_score_table_manifest",
]
