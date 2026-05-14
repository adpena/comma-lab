#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR85 QMA9 token/context entropy for local planning.

This is a planning-only profiler over the decoded PR85 mask token tensor. It
does not build a replacement stream, run a scorer, or unlock dispatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKEN_PATH = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_PROFILE_JSON = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_token_source_profile.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_qma9_context_entropy_20260504_worker"
)

SCHEMA = "pr85_qma9_context_entropy_profile_v1"
TOOL = "experiments/profile_pr85_qma9_context_entropy.py"
CONTEST_RATE_LAMBDA = 25.0 / 37_545_489.0


class ProfileError(ValueError):
    """Raised when the PR85 token/profile inputs are inconsistent."""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ProfileError(f"profile JSON must be an object: {path}")
    return payload


def entropy_from_counts(counts: np.ndarray) -> float:
    """Return Shannon entropy in bits for a count vector."""

    counts64 = np.asarray(counts, dtype=np.float64)
    total = float(counts64.sum())
    if total <= 0.0:
        return 0.0
    probs = counts64[counts64 > 0.0] / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return 0.0 if abs(entropy) < 1e-15 else entropy


def counts_for_values(values: np.ndarray, *, alphabet_size: int) -> np.ndarray:
    return np.bincount(values.reshape(-1), minlength=alphabet_size)[:alphabet_size].astype(
        np.int64, copy=False
    )


def _counts_record(counts: np.ndarray) -> dict[str, Any]:
    counts64 = np.asarray(counts, dtype=np.int64)
    total = int(counts64.sum())
    entropy = entropy_from_counts(counts64)
    dominant_symbol = int(counts64.argmax()) if total else None
    dominant_count = int(counts64.max()) if total else 0
    return {
        "counts": {str(i): int(v) for i, v in enumerate(counts64.tolist())},
        "dominant_count": dominant_count,
        "dominant_fraction": float(dominant_count / total) if total else 0.0,
        "dominant_symbol": dominant_symbol,
        "entropy_bits_per_token": entropy,
        "ideal_entropy_bytes": float(entropy * total / 8.0),
        "token_count": total,
    }


def _axis_entropy_records(
    tokens: np.ndarray,
    *,
    axis: int,
    label: str,
    alphabet_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(tokens.shape[axis]):
        values = np.take(tokens, index, axis=axis)
        record = _counts_record(counts_for_values(values, alphabet_size=alphabet_size))
        record[f"{label}_index"] = int(index)
        records.append(record)
    return records


def _summarize_axis(records: Sequence[Mapping[str, Any]], *, index_key: str) -> dict[str, Any]:
    if not records:
        return {"max_entropy": None, "mean_entropy": 0.0, "min_entropy": None}
    entropies = np.array([float(row["entropy_bits_per_token"]) for row in records], dtype=np.float64)
    ranked_high = sorted(
        records,
        key=lambda row: (-float(row["entropy_bits_per_token"]), int(row[index_key])),
    )[:8]
    ranked_low = sorted(
        records,
        key=lambda row: (float(row["entropy_bits_per_token"]), int(row[index_key])),
    )[:8]
    return {
        "max_entropy": {
            index_key: int(ranked_high[0][index_key]),
            "entropy_bits_per_token": float(ranked_high[0]["entropy_bits_per_token"]),
        },
        "mean_entropy_bits_per_token": float(entropies.mean()),
        "min_entropy": {
            index_key: int(ranked_low[0][index_key]),
            "entropy_bits_per_token": float(ranked_low[0]["entropy_bits_per_token"]),
        },
        "top_high_entropy": [
            {
                index_key: int(row[index_key]),
                "dominant_fraction": float(row["dominant_fraction"]),
                "entropy_bits_per_token": float(row["entropy_bits_per_token"]),
            }
            for row in ranked_high
        ],
        "top_low_entropy": [
            {
                index_key: int(row[index_key]),
                "dominant_fraction": float(row["dominant_fraction"]),
                "entropy_bits_per_token": float(row["entropy_bits_per_token"]),
            }
            for row in ranked_low
        ],
    }


def _valid_slices_for_offsets(
    shape: Sequence[int],
    offsets: Sequence[tuple[int, int, int]],
) -> tuple[tuple[slice, slice, slice], list[tuple[slice, slice, slice]]]:
    starts = [0, 0, 0]
    for offset in offsets:
        for axis, delta in enumerate(offset):
            starts[axis] = max(starts[axis], delta)
    current = tuple(slice(starts[axis], shape[axis]) for axis in range(3))
    context_slices: list[tuple[slice, slice, slice]] = []
    for offset in offsets:
        context_slices.append(
            tuple(
                slice(starts[axis] - offset[axis], shape[axis] - offset[axis])
                for axis in range(3)
            )
        )
    return current, context_slices


def conditional_entropy_profile(
    tokens: np.ndarray,
    *,
    name: str,
    offsets: Sequence[tuple[int, int, int]],
    alphabet_size: int,
    charged_mask_bytes: int,
    total_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Profile H(symbol | context offsets).

    Offsets are positive distances from the current token in
    ``(frame, column, row)`` storage coordinates.
    """

    if not offsets:
        raise ProfileError("conditional context requires at least one offset")
    if total_counts is None:
        total_counts = counts_for_values(tokens, alphabet_size=alphabet_size)
    current_slice, context_slices = _valid_slices_for_offsets(tokens.shape, offsets)
    current = tokens[current_slice].reshape(-1)
    context_code = np.zeros(current.size, dtype=np.int16)
    for context_slice in context_slices:
        context_values = tokens[context_slice].reshape(-1).astype(np.int16, copy=False)
        context_code = context_code * alphabet_size + context_values

    context_state_count = alphabet_size ** len(offsets)
    pair_code = context_code * alphabet_size + current.astype(np.int16, copy=False)
    joint_counts = np.bincount(
        pair_code,
        minlength=context_state_count * alphabet_size,
    ).astype(np.int64, copy=False)
    joint = joint_counts.reshape(context_state_count, alphabet_size)
    context_counts = joint.sum(axis=1)
    valid_counts = joint.sum(axis=0)
    valid_count = int(current.size)
    total_count = int(np.asarray(total_counts, dtype=np.int64).sum())
    excluded_counts = np.asarray(total_counts, dtype=np.int64) - valid_counts
    excluded_count = int(excluded_counts.sum())

    joint_entropy = entropy_from_counts(joint_counts)
    context_entropy = entropy_from_counts(context_counts)
    conditional_entropy = max(0.0, joint_entropy - context_entropy)
    excluded_entropy = entropy_from_counts(excluded_counts)
    lower_bound_bits = conditional_entropy * valid_count + excluded_entropy * excluded_count
    lower_bound_bytes = lower_bound_bits / 8.0
    charged_prorated_bytes = charged_mask_bytes * (valid_count / total_count)
    valid_lower_bound_bytes = conditional_entropy * valid_count / 8.0
    full_stream_bytes_saved_lower_bound = charged_mask_bytes - lower_bound_bytes
    valid_region_bytes_saved_lower_bound = charged_prorated_bytes - valid_lower_bound_bytes

    top_contexts: list[dict[str, Any]] = []
    ranked_context_indices = sorted(
        [int(i) for i in np.flatnonzero(context_counts)],
        key=lambda i: (-int(context_counts[i]), i),
    )[:10]
    for context_index in ranked_context_indices:
        counts = joint[context_index]
        decoded_symbols: list[int] = []
        code = context_index
        for _ in range(len(offsets)):
            decoded_symbols.append(int(code % alphabet_size))
            code //= alphabet_size
        decoded_symbols.reverse()
        top_contexts.append(
            {
                "context_count": int(context_counts[context_index]),
                "context_index": context_index,
                "context_symbols": decoded_symbols,
                **_counts_record(counts),
            }
        )

    return {
        "break_even_overhead_bytes_full_stream": float(full_stream_bytes_saved_lower_bound),
        "charged_mask_bytes": int(charged_mask_bytes),
        "charged_prorated_valid_region_bytes": float(charged_prorated_bytes),
        "conditional_entropy_bits_per_token": float(conditional_entropy),
        "context_offsets_frame_col_row": [list(offset) for offset in offsets],
        "context_state_count": int(context_state_count),
        "excluded_entropy_bits_per_token": float(excluded_entropy),
        "excluded_token_count": excluded_count,
        "full_stream_bytes_saved_lower_bound": float(full_stream_bytes_saved_lower_bound),
        "full_stream_ideal_entropy_bytes_with_unconditional_border": float(lower_bound_bytes),
        "name": name,
        "nonzero_context_state_count": int(np.count_nonzero(context_counts)),
        "planning_only": True,
        "rate_score_delta_full_stream_lower_bound": float(
            -full_stream_bytes_saved_lower_bound * CONTEST_RATE_LAMBDA
        ),
        "top_contexts_by_count": top_contexts,
        "valid_region_bytes_saved_lower_bound": float(valid_region_bytes_saved_lower_bound),
        "valid_region_ideal_entropy_bytes": float(valid_lower_bound_bytes),
        "valid_token_count": valid_count,
    }


def _run_lengths_from_flat(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.array([], dtype=np.int64)
    change_positions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            change_positions.astype(np.int64, copy=False),
            np.array([flat.size], dtype=np.int64),
        ]
    )
    return np.diff(boundaries)


def _run_summary_from_sequences(sequences: np.ndarray, *, name: str) -> dict[str, Any]:
    seq = np.asarray(sequences)
    if seq.ndim != 2:
        raise ProfileError(f"{name}: expected 2D sequences")
    total = int(seq.size)
    if total == 0:
        return {"name": name, "token_count": 0}
    if seq.shape[1] <= 1:
        same_adjacent_count = 0
    else:
        same_adjacent_count = int((seq[:, 1:] == seq[:, :-1]).sum())
    run_count = total - same_adjacent_count
    return {
        "adjacent_pairs": int(seq.shape[0] * max(seq.shape[1] - 1, 0)),
        "adjacent_same_fraction": float(
            same_adjacent_count / (seq.shape[0] * max(seq.shape[1] - 1, 0))
        )
        if seq.shape[1] > 1
        else 0.0,
        "average_run_length": float(total / run_count) if run_count else 0.0,
        "name": name,
        "run_count": int(run_count),
        "same_adjacent_count": int(same_adjacent_count),
        "sequence_count": int(seq.shape[0]),
        "sequence_length": int(seq.shape[1]),
        "token_count": total,
    }


def _run_length_profile(tokens: np.ndarray) -> dict[str, Any]:
    flat_lengths = _run_lengths_from_flat(tokens)
    quantiles: dict[str, float] = {}
    if flat_lengths.size:
        for q in (0.5, 0.75, 0.9, 0.99, 0.999):
            quantiles[f"p{int(q * 1000):03d}"] = float(np.quantile(flat_lengths, q))
    row_major = np.transpose(tokens, (0, 2, 1)).reshape(-1, tokens.shape[1])
    col_major = tokens.reshape(-1, tokens.shape[2])
    time_major = np.transpose(tokens, (1, 2, 0)).reshape(-1, tokens.shape[0])
    return {
        "column_sequences_along_rows_storage_axis2": _run_summary_from_sequences(
            col_major,
            name="storage_contiguous_height_axis_runs",
        ),
        "frame_time_sequences_per_pixel": _run_summary_from_sequences(
            time_major,
            name="time_axis_runs_per_pixel",
        ),
        "row_sequences_along_width_axis": _run_summary_from_sequences(
            row_major,
            name="visual_row_width_axis_runs",
        ),
        "storage_order_flat": {
            "average_run_length": float(tokens.size / flat_lengths.size) if flat_lengths.size else 0.0,
            "max_run_length": int(flat_lengths.max()) if flat_lengths.size else 0,
            "quantiles": quantiles,
            "run_count": int(flat_lengths.size),
            "token_count": int(tokens.size),
        },
    }


def _rank_opportunities(
    *,
    global_record: Mapping[str, Any],
    contexts: Sequence[Mapping[str, Any]],
    charged_mask_bytes: int,
    token_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_bytes = float(global_record["ideal_entropy_bytes"])
    global_saved = charged_mask_bytes - global_bytes
    rows.append(
        {
            "break_even_overhead_bytes": float(global_saved),
            "estimated_bytes_saved_lower_bound": float(global_saved),
            "estimated_ideal_bytes": global_bytes,
            "model": "global_symbol_model",
            "non_arbitrary_basis": "exact global symbol counts from decoded PR85 QMA9 token tensor",
            "rate_score_delta_lower_bound": float(-global_saved * CONTEST_RATE_LAMBDA),
            "token_count": int(token_count),
        }
    )
    for context in contexts:
        saved = float(context["full_stream_bytes_saved_lower_bound"])
        rows.append(
            {
                "break_even_overhead_bytes": float(context["break_even_overhead_bytes_full_stream"]),
                "conditional_entropy_bits_per_token": float(
                    context["conditional_entropy_bits_per_token"]
                ),
                "estimated_bytes_saved_lower_bound": saved,
                "estimated_ideal_bytes": float(
                    context["full_stream_ideal_entropy_bytes_with_unconditional_border"]
                ),
                "model": str(context["name"]),
                "non_arbitrary_basis": "exact conditional counts over aligned PR85 QMA9 token contexts",
                "rate_score_delta_lower_bound": float(
                    context["rate_score_delta_full_stream_lower_bound"]
                ),
                "token_count": int(context["valid_token_count"]),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["estimated_bytes_saved_lower_bound"]),
            str(row["model"]),
        ),
    )


def _build_recommendations(
    *,
    ranked: Sequence[Mapping[str, Any]],
    run_length_profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    positive = [row for row in ranked if float(row["estimated_bytes_saved_lower_bound"]) > 0.0]
    best = ranked[0]
    row_runs = run_length_profile["row_sequences_along_width_axis"]
    time_runs = run_length_profile["frame_time_sequences_per_pixel"]
    recommendations: list[dict[str, Any]] = []
    if positive:
        recommendations.append(
            {
                "action": "prototype_best_positive_context_model_locally",
                "basis": "at least one exact entropy lower bound beats charged QMA9 mask bytes before overhead",
                "model": str(positive[0]["model"]),
                "planning_only": True,
                "score_claim": False,
            }
        )
    else:
        recommendations.append(
            {
                "action": "do_not_dispatch_simple_symbol_context_entropy_replacement",
                "basis": (
                    "all measured global/single-context/multi-context entropy lower bounds exceed "
                    "the charged PR85 QMA9 mask segment before model overhead"
                ),
                "best_model": str(best["model"]),
                "best_model_byte_gap": float(-float(best["estimated_bytes_saved_lower_bound"])),
                "planning_only": True,
                "score_claim": False,
            }
        )
    recommendations.append(
        {
            "action": "if_pursuing_mask_bytes_next_target_qma9_native_run_grammar_not_generic_entropy",
            "basis": (
                "row/time runs are extremely long, but charged QMA9 is already far below the "
                "simple conditional entropy bounds; any win needs native grammar/table overhead "
                "reduction or a structurally different run representation"
            ),
            "row_same_adjacent_fraction": float(row_runs["adjacent_same_fraction"]),
            "row_average_run_length": float(row_runs["average_run_length"]),
            "time_same_adjacent_fraction": float(time_runs["adjacent_same_fraction"]),
            "time_average_run_length": float(time_runs["average_run_length"]),
            "planning_only": True,
            "score_claim": False,
        }
    )
    recommendations.append(
        {
            "action": "use_left_plus_up_as_the_first_lossless_parity_control_if_a_coder_is_built",
            "basis": "left_plus_up is the best measured simple context even though it is not byte-positive versus PR85 QMA9",
            "break_even_overhead_bytes": float(best["break_even_overhead_bytes"]),
            "estimated_bytes_saved_lower_bound": float(best["estimated_bytes_saved_lower_bound"]),
            "rate_score_delta_lower_bound": float(best["rate_score_delta_lower_bound"]),
            "planning_only": True,
            "score_claim": False,
        }
    )
    return recommendations


def load_tokens_from_profile(token_path: Path, profile_json: Path) -> tuple[np.ndarray, dict[str, Any]]:
    profile = _load_json_object(profile_json)
    token_source = profile.get("token_source")
    if not isinstance(token_source, Mapping):
        raise ProfileError("profile missing token_source object")
    dtype = token_source.get("dtype")
    if dtype != "uint8":
        raise ProfileError(f"expected uint8 token source, got {dtype!r}")
    shape_raw = token_source.get("shape")
    if (
        not isinstance(shape_raw, list)
        or len(shape_raw) != 3
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape_raw)
    ):
        raise ProfileError(f"invalid token shape in profile: {shape_raw!r}")
    shape = tuple(int(v) for v in shape_raw)
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    actual_bytes = token_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ProfileError(
            f"token byte count mismatch: profile shape requires {expected_bytes}, file has {actual_bytes}"
        )
    expected_sha = token_source.get("sha256")
    actual_sha = _sha256_file(token_path)
    if isinstance(expected_sha, str) and expected_sha and actual_sha != expected_sha:
        raise ProfileError(f"token SHA mismatch: expected {expected_sha}, got {actual_sha}")
    tokens = np.memmap(token_path, mode="r", dtype=np.uint8, shape=shape)
    return tokens, profile


def build_profile(
    *,
    token_path: Path,
    profile_json: Path,
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    tokens, source_profile = load_tokens_from_profile(token_path, profile_json)
    token_source = source_profile["token_source"]
    range_contract = token_source.get("range_contract", {})
    alphabet_size = int(range_contract.get("max", int(tokens.max()))) + 1
    if alphabet_size <= 0 or alphabet_size > 256:
        raise ProfileError(f"invalid alphabet size: {alphabet_size}")
    observed_min = int(tokens.min())
    observed_max = int(tokens.max())
    if observed_min < 0 or observed_max >= alphabet_size:
        raise ProfileError(
            f"observed token range [{observed_min}, {observed_max}] outside alphabet {alphabet_size}"
        )

    mask_segment_identity = source_profile.get("mask_segment_identity", {})
    if not isinstance(mask_segment_identity, Mapping):
        raise ProfileError("profile missing mask_segment_identity object")
    charged_mask_bytes = int(mask_segment_identity.get("bytes", 0))
    if charged_mask_bytes <= 0:
        raise ProfileError("profile mask_segment_identity.bytes must be positive")

    total_counts = counts_for_values(tokens, alphabet_size=alphabet_size)
    global_record = _counts_record(total_counts)

    frame_records = _axis_entropy_records(tokens, axis=0, label="frame", alphabet_size=alphabet_size)
    col_records = _axis_entropy_records(tokens, axis=1, label="col", alphabet_size=alphabet_size)
    row_records = _axis_entropy_records(tokens, axis=2, label="row", alphabet_size=alphabet_size)

    context_specs = [
        ("left_col_prev", [(0, 1, 0)]),
        ("up_row_prev", [(0, 0, 1)]),
        ("up_left_col_row_prev", [(0, 1, 1)]),
        ("time_prev_frame", [(1, 0, 0)]),
        ("left_plus_up", [(0, 1, 0), (0, 0, 1)]),
        ("left_plus_time_prev", [(0, 1, 0), (1, 0, 0)]),
        ("up_plus_time_prev", [(0, 0, 1), (1, 0, 0)]),
        ("left_up_time_prev", [(0, 1, 0), (0, 0, 1), (1, 0, 0)]),
    ]
    contexts = [
        conditional_entropy_profile(
            tokens,
            name=name,
            offsets=offsets,
            alphabet_size=alphabet_size,
            charged_mask_bytes=charged_mask_bytes,
            total_counts=total_counts,
        )
        for name, offsets in context_specs
    ]

    run_lengths = _run_length_profile(tokens)
    ranked = _rank_opportunities(
        global_record=global_record,
        contexts=contexts,
        charged_mask_bytes=charged_mask_bytes,
        token_count=int(tokens.size),
    )
    recommendations = _build_recommendations(ranked=ranked, run_length_profile=run_lengths)

    return {
        "axis_convention": {
            "axis0": "frame",
            "axis1": "storage_width_column",
            "axis2": "storage_height_row",
            "source_storage_order": source_profile.get("decode", {}).get(
                "storage_order",
                "frame_major_header_width_by_header_height",
            ),
        },
        "charged_baseline": {
            "contest_rate_lambda_points_per_byte": CONTEST_RATE_LAMBDA,
            "mask_segment_bytes": charged_mask_bytes,
            "mask_segment_sha256": mask_segment_identity.get("sha256"),
            "qma9_bits_per_token_charged": float(charged_mask_bytes * 8.0 / tokens.size),
        },
        "conditional_context_entropy": contexts,
        "dispatch_performed": False,
        "evidence_grade": "empirical/local_context_entropy_planning",
        "gpu_required": False,
        "input_profile": {
            "path": str(profile_json),
            "schema": source_profile.get("schema"),
            "sha256": _sha256_file(profile_json),
        },
        "input_token_source": {
            "bytes": int(token_path.stat().st_size),
            "dtype": "uint8",
            "path": str(token_path),
            "sha256": _sha256_file(token_path),
            "shape": [int(v) for v in tokens.shape],
        },
        "opportunity_ranking": ranked,
        "per_axis_entropy": {
            "cols": {
                "records": col_records,
                "summary": _summarize_axis(col_records, index_key="col_index"),
            },
            "frames": {
                "records": frame_records,
                "summary": _summarize_axis(frame_records, index_key="frame_index"),
            },
            "rows": {
                "records": row_records,
                "summary": _summarize_axis(row_records, index_key="row_index"),
            },
        },
        "per_symbol_entropy": global_record,
        "planning_only": True,
        "recommendations": recommendations,
        "recorded_at_utc": recorded_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "run_length_opportunities": run_lengths,
        "schema": SCHEMA,
        "score_claim": False,
        "tool": TOOL,
    }


def render_markdown(profile: Mapping[str, Any], *, top_k: int = 12) -> str:
    baseline = profile["charged_baseline"]
    lines = [
        "# PR85 QMA9 Context Entropy Planning Profile",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        f"- token_source: `{profile['input_token_source']['path']}`",
        f"- token_sha256: `{profile['input_token_source']['sha256']}`",
        f"- tensor_shape: `{profile['input_token_source']['shape']}`",
        f"- charged_qma9_mask_bytes: {baseline['mask_segment_bytes']}",
        f"- charged_qma9_bits_per_token: {baseline['qma9_bits_per_token_charged']:.9f}",
        f"- positive_byte_saving_models: {sum(1 for row in profile['opportunity_ranking'] if float(row['estimated_bytes_saved_lower_bound']) > 0.0)}",
        "",
        "## Top Planning Opportunities",
        "",
        "| rank | model | est ideal bytes | est bytes saved | rate-score delta | break-even overhead bytes |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(profile["opportunity_ranking"][:top_k], start=1):
        lines.append(
            "| {rank} | `{model}` | {ideal:.3f} | {saved:.3f} | {delta:.9f} | {overhead:.3f} |".format(
                rank=rank,
                model=row["model"],
                ideal=float(row["estimated_ideal_bytes"]),
                saved=float(row["estimated_bytes_saved_lower_bound"]),
                delta=float(row["rate_score_delta_lower_bound"]),
                overhead=float(row["break_even_overhead_bytes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Axis Entropy Highlights",
            "",
        ]
    )
    for axis_name, index_key in [("frames", "frame_index"), ("cols", "col_index"), ("rows", "row_index")]:
        summary = profile["per_axis_entropy"][axis_name]["summary"]
        max_entropy = summary["max_entropy"]
        min_entropy = summary["min_entropy"]
        lines.append(
            "- {axis}: mean={mean:.6f} bits/token, max {key}={max_i} ({max_h:.6f}), "
            "min {key}={min_i} ({min_h:.6f})".format(
                axis=axis_name,
                mean=float(summary["mean_entropy_bits_per_token"]),
                key=index_key,
                max_i=max_entropy[index_key],
                max_h=float(max_entropy["entropy_bits_per_token"]),
                min_i=min_entropy[index_key],
                min_h=float(min_entropy["entropy_bits_per_token"]),
            )
        )
    rle = profile["run_length_opportunities"]
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
        ]
    )
    for row in profile["recommendations"]:
        lines.append(f"- `{row['action']}`: {row['basis']}")
    lines.extend(
        [
            "",
            "## Run-Length Signals",
            "",
            "- storage_order_flat: runs={runs}, avg_run={avg:.6f}, max_run={max_run}".format(
                runs=rle["storage_order_flat"]["run_count"],
                avg=float(rle["storage_order_flat"]["average_run_length"]),
                max_run=rle["storage_order_flat"]["max_run_length"],
            ),
            "- visual_row_width_axis_runs: same_adjacent_fraction={:.6f}, avg_run={:.6f}".format(
                float(rle["row_sequences_along_width_axis"]["adjacent_same_fraction"]),
                float(rle["row_sequences_along_width_axis"]["average_run_length"]),
            ),
            "- time_axis_runs_per_pixel: same_adjacent_fraction={:.6f}, avg_run={:.6f}".format(
                float(rle["frame_time_sequences_per_pixel"]["adjacent_same_fraction"]),
                float(rle["frame_time_sequences_per_pixel"]["average_run_length"]),
            ),
            "",
            "These are entropy/rate planning estimates only. A replacement coder would still need byte-closed archive parity, runtime output parity, and exact CUDA auth eval before any score claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json-name", default="pr85_qma9_context_entropy_profile.json")
    parser.add_argument("--output-md-name", default="pr85_qma9_context_entropy_profile.md")
    parser.add_argument(
        "--recorded-at-utc",
        default=None,
        help="Override timestamp for deterministic tests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = build_profile(
        token_path=args.token_path,
        profile_json=args.profile_json,
        recorded_at_utc=args.recorded_at_utc,
    )
    output_json = args.output_dir / args.output_json_name
    output_md = args.output_dir / args.output_md_name
    _write_json(output_json, profile)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(profile), encoding="utf-8")
    print(
        json.dumps(
            {
                "dispatch_performed": False,
                "output_json": str(output_json),
                "output_md": str(output_md),
                "planning_only": True,
                "score_claim": False,
                "top_model": profile["opportunity_ranking"][0]["model"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
