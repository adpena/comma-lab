# SPDX-License-Identifier: MIT
"""Local PR91/HPM1 range uint32 re-emission diagnostics.

This module is forensic tooling only. It loads the public PR91 HPM1 payload and
reference-token prior, builds a small deterministic probability-row prefix, and
tests range word-order, finalization, precision, and context-row hypotheses.
It never dispatches GPU work, runs contest eval, or claims a score.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac import pr86_hpac_codec as _pr86_hpac_codec
from tac.pr86_hpac_codec import (
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    PROB_EPS,
    _categorical_from_probs,
    collect_dependency_report,
    resolve_hpac_probability_variant,
    sha256_bytes,
    supported_hpac_probability_variant_names,
)
from tac.pr91_hpm1_codec import (
    DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    DEFAULT_PR91_ARCHIVE,
    PR91_HPM1_TOKEN_WORD_ORDER_CANDIDATES,
    F,
    Pr91Hpm1Error,
    _archive_report,
    _common_prefix_uint32_words,
    _group_masks,
    _hpm1_group_coords_for_spatial_order,
    _hpm1_token_words_for_candidate,
    _jsonable,
    _load_reference_tokens,
    _range_decoder_state_summary,
    _range_encoder_state_summary,
    _spatial_order_description,
    _uint32_words_digest,
    compare_hpm1_to_pr86_hpac_contract,
    extract_pr91_hpm1_payload,
    load_hpm1_hpac_model,
    torch,
    validate_hpm1_static_contract,
    write_json_report,
)

DEFAULT_PR91_HPM1_RANGE_CONTRACT_SYMBOL_LIMIT = 8
DEFAULT_PR91_HPM1_RANGE_CONTRACT_SPATIAL_ORDER = "phase_major_row_major"
RANGE_FINALIZATION_HYPOTHESES = (
    "clone_get_compressed",
    "direct_get_compressed",
)


def _require_range_coder() -> None:
    if _pr86_hpac_codec.constriction is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "constriction_missing")


def _validate_symbol_limit(symbol_limit: int) -> int:
    value = int(symbol_limit)
    if value <= 0:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "symbol_limit_must_be_positive",
            symbol_limit=symbol_limit,
        )
    if value > 4096:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "symbol_limit_too_large_for_focused_local_diagnostic",
            symbol_limit=value,
            maximum=4096,
        )
    return value


def _resolve_probability_variants(
    variants: Sequence[str] | None,
) -> tuple[str, ...]:
    requested = tuple(
        dict.fromkeys(
            str(name)
            for name in (variants or supported_hpac_probability_variant_names())
            if str(name)
        )
    )
    if not requested:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "at_least_one_probability_variant_required",
        )
    for name in requested:
        resolve_hpac_probability_variant(name)
    return requested


def _first_uint32_mismatch(
    local_words: np.ndarray,
    submitted_words: np.ndarray,
    *,
    local_label: str,
    submitted_label: str,
) -> dict[str, Any] | None:
    local = np.ascontiguousarray(local_words, dtype=np.uint32)
    submitted = np.ascontiguousarray(submitted_words, dtype=np.uint32)
    common = _common_prefix_uint32_words(local, submitted)
    limit = min(int(local.size), int(submitted.size))
    if common == limit and int(local.size) == int(submitted.size):
        return None
    row: dict[str, Any] = {
        "word_index": int(common),
        "byte_offset": int(common) * 4,
        "common_prefix_word_count": int(common),
        "local_label": local_label,
        "submitted_label": submitted_label,
    }
    if common < int(local.size):
        row["local_word_hex"] = f"0x{int(local[common]):08x}"
    else:
        row["local_stream_ended"] = True
    if common < int(submitted.size):
        row["submitted_word_hex"] = f"0x{int(submitted[common]):08x}"
    else:
        row["submitted_stream_ended"] = True
    if common == limit:
        row["reason"] = "length_mismatch_after_common_prefix"
    else:
        row["reason"] = "word_value_mismatch"
    return row


def _word_prefix_comparison(
    local_words: np.ndarray,
    submitted_words: np.ndarray,
    *,
    local_label: str,
    submitted_label: str,
) -> dict[str, Any]:
    local = np.ascontiguousarray(local_words, dtype=np.uint32)
    submitted = np.ascontiguousarray(submitted_words, dtype=np.uint32)
    common = _common_prefix_uint32_words(local, submitted)
    return {
        "local_label": local_label,
        "submitted_label": submitted_label,
        "exact_match": bool(
            int(local.size) == int(submitted.size)
            and np.array_equal(local, submitted)
        ),
        "common_prefix_word_count": int(common),
        "local_word_count": int(local.size),
        "submitted_word_count": int(submitted.size),
        "first_mismatch": _first_uint32_mismatch(
            local,
            submitted,
            local_label=local_label,
            submitted_label=submitted_label,
        ),
    }


def _symbol_prefix_digest(symbols: Sequence[int]) -> dict[str, Any]:
    arr = np.asarray([int(value) for value in symbols], dtype=np.uint8)
    return {
        "count": int(arr.size),
        "sha256": sha256_bytes(arr.tobytes()),
        "first_symbols": [int(value) for value in arr[:16]],
    }


def _probability_row_digest(raw_rows: Sequence[np.ndarray]) -> dict[str, Any]:
    rows = np.ascontiguousarray(np.vstack(raw_rows), dtype=np.float32)
    return {
        "count": int(rows.shape[0]),
        "dtype": str(rows.dtype),
        "shape": [int(value) for value in rows.shape],
        "sha256": sha256_bytes(rows.tobytes()),
        "first_row": [round(float(value), 10) for value in rows[0].tolist()],
    }


def _decode_prefix_against_words(
    words: np.ndarray,
    raw_rows: Sequence[np.ndarray],
    reference_symbols: Sequence[int],
    *,
    probability_variant: str,
    prob_eps: float,
) -> dict[str, Any]:
    _require_range_coder()
    resolved = resolve_hpac_probability_variant(probability_variant)
    arr = np.ascontiguousarray(words, dtype=np.uint32)
    if int(arr.size) == 0:
        return {
            "status": "empty_word_stream_for_nonempty_prefix",
            "passed": False,
            "decoded_symbol_count": 0,
            "matched_prefix_symbol_count": 0,
        }

    decoder = _pr86_hpac_codec.constriction.stream.queue.RangeDecoder(arr)
    decoded: list[int] = []
    for index, (row, expected_symbol) in enumerate(
        zip(raw_rows, reference_symbols, strict=True)
    ):
        cat = _categorical_from_probs(
            np.asarray(row),
            prob_eps=prob_eps,
            variant=resolved,
        )
        try:
            decoded_symbol = int(decoder.decode(cat))
        except Exception as exc:
            return {
                "status": "range_decode_exception",
                "passed": False,
                "decoded_symbol_count": int(index),
                "matched_prefix_symbol_count": int(index),
                "exception_type": type(exc).__name__,
                "exception_text": str(exc),
                "decoder_state_after_exception": _range_decoder_state_summary(
                    decoder,
                    label="after_range_contract_decode_exception",
                ),
                "decoded_prefix": _symbol_prefix_digest(decoded),
            }
        decoded.append(decoded_symbol)
        expected = int(expected_symbol)
        if decoded_symbol != expected:
            return {
                "status": "decoded_symbol_mismatch",
                "passed": False,
                "decoded_symbol_count": int(index + 1),
                "matched_prefix_symbol_count": int(index),
                "first_mismatch": {
                    "symbol_index": int(index),
                    "decoded_symbol": int(decoded_symbol),
                    "reference_symbol": expected,
                },
                "decoded_prefix": _symbol_prefix_digest(decoded),
                "decoder_state_after_mismatch": _range_decoder_state_summary(
                    decoder,
                    label="after_range_contract_symbol_mismatch",
                ),
            }

    return {
        "status": "decoded_reference_prefix",
        "passed": True,
        "decoded_symbol_count": len(reference_symbols),
        "matched_prefix_symbol_count": len(reference_symbols),
        "decoded_prefix": _symbol_prefix_digest(decoded),
        "decoder_state_after_replay": _range_decoder_state_summary(
            decoder,
            label="after_range_contract_prefix_replay",
        ),
    }


def _encode_reference_prefix_words(
    raw_rows: Sequence[np.ndarray],
    reference_symbols: Sequence[int],
    *,
    probability_variant: str,
    prob_eps: float,
    finalization: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_range_coder()
    if finalization not in RANGE_FINALIZATION_HYPOTHESES:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "unsupported_range_finalization_hypothesis",
            requested=finalization,
            supported=list(RANGE_FINALIZATION_HYPOTHESES),
        )
    resolved = resolve_hpac_probability_variant(probability_variant)
    encoder = _pr86_hpac_codec.constriction.stream.queue.RangeEncoder()
    for row, symbol in zip(raw_rows, reference_symbols, strict=True):
        cat = _categorical_from_probs(
            np.asarray(row),
            prob_eps=prob_eps,
            variant=resolved,
        )
        encoder.encode(int(symbol), cat)
    state_before_finalization = _range_encoder_state_summary(
        encoder,
        label=f"after_encoding_{len(reference_symbols)}_symbols",
    )
    words = (
        encoder.clone().get_compressed()
        if finalization == "clone_get_compressed"
        else encoder.get_compressed()
    )
    arr = np.ascontiguousarray(np.asarray(words, dtype=np.uint32), dtype=np.uint32)
    return arr, {
        "finalization": finalization,
        "encoder_state_before_finalization": state_before_finalization,
        "emitted_words": _uint32_words_digest(arr),
    }


def _summarize_finalization_sensitivity(
    local_words: np.ndarray,
    submitted_words: np.ndarray,
    raw_rows: Sequence[np.ndarray],
    reference_symbols: Sequence[int],
    *,
    probability_variant: str,
    prob_eps: float,
) -> dict[str, Any]:
    local_count = int(np.asarray(local_words, dtype=np.uint32).size)
    submitted_same_count = np.ascontiguousarray(
        np.asarray(submitted_words, dtype=np.uint32)[:local_count],
        dtype=np.uint32,
    )
    return {
        "submitted_same_word_count_scope": (
            "submitted stream truncated to the local standalone finalized "
            "prefix word count"
        ),
        "submitted_same_word_count_words": _uint32_words_digest(
            submitted_same_count
        ),
        "submitted_same_word_count_replay": _decode_prefix_against_words(
            submitted_same_count,
            raw_rows,
            reference_symbols,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
        ),
        "submitted_full_stream_replay": _decode_prefix_against_words(
            submitted_words,
            raw_rows,
            reference_symbols,
            probability_variant=probability_variant,
            prob_eps=prob_eps,
        ),
    }


def _best_replay(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    best = max(
        rows,
        key=lambda row: (
            int(row.get("replay", {}).get("matched_prefix_symbol_count", 0)),
            bool(row.get("replay", {}).get("passed") is True),
        ),
    )
    return {
        "word_order_candidate": best.get("word_order_candidate"),
        "probability_variant": best.get("probability_variant"),
        "passed": bool(best.get("replay", {}).get("passed") is True),
        "matched_prefix_symbol_count": int(
            best.get("replay", {}).get("matched_prefix_symbol_count", 0)
        ),
        "status": best.get("replay", {}).get("status"),
        "first_mismatch": best.get("replay", {}).get("first_mismatch"),
    }


def _replay_mismatch_for_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    word_order_candidate: str,
    probability_variant: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("word_order_candidate") != word_order_candidate
            or row.get("probability_variant") != probability_variant
        ):
            continue
        replay = row.get("replay", {})
        if not isinstance(replay, Mapping):
            continue
        first = replay.get("first_mismatch")
        if isinstance(first, Mapping):
            return {
                "word_order_candidate": row.get("word_order_candidate"),
                "probability_variant": row.get("probability_variant"),
                "symbol_index": int(first["symbol_index"]),
                "decoded_symbol": int(first["decoded_symbol"]),
                "reference_symbol": int(first["reference_symbol"]),
                "matched_prefix_symbol_count": int(
                    replay.get("matched_prefix_symbol_count", 0)
                ),
            }
    return None


def _first_replay_mismatch(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        replay = row.get("replay", {})
        if not isinstance(replay, Mapping):
            continue
        first = replay.get("first_mismatch")
        if isinstance(first, Mapping):
            mismatches.append(
                {
                    "word_order_candidate": row.get("word_order_candidate"),
                    "probability_variant": row.get("probability_variant"),
                    "symbol_index": int(first["symbol_index"]),
                    "decoded_symbol": int(first["decoded_symbol"]),
                    "reference_symbol": int(first["reference_symbol"]),
                    "matched_prefix_symbol_count": int(
                        replay.get("matched_prefix_symbol_count", 0)
                    ),
                }
            )
    if not mismatches:
        return None
    return sorted(
        mismatches,
        key=lambda row: (
            int(row["symbol_index"]),
            str(row["word_order_candidate"]),
            str(row["probability_variant"]),
        ),
    )[0]


def _first_local_word_mismatch(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_variant: str,
) -> dict[str, Any] | None:
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        if row.get("finalization") != "clone_get_compressed":
            continue
        for comparison in row.get("submitted_word_comparisons", []):
            if not isinstance(comparison, Mapping):
                continue
            first = comparison.get("first_mismatch")
            if not isinstance(first, Mapping):
                continue
            mismatches.append(
                {
                    "probability_variant": row.get("probability_variant"),
                    "submitted_word_order_candidate": comparison.get(
                        "submitted_label"
                    ),
                    "word_index": int(first["word_index"]),
                    "byte_offset": int(first["byte_offset"]),
                    "reason": first.get("reason"),
                    "local_word_hex": first.get("local_word_hex"),
                    "submitted_word_hex": first.get("submitted_word_hex"),
                    "common_prefix_word_count": int(
                        first.get("common_prefix_word_count", 0)
                    ),
                    "preferred_variant": bool(
                        row.get("probability_variant") == preferred_variant
                    ),
                }
            )
    if not mismatches:
        return None
    return sorted(
        mismatches,
        key=lambda row: (
            not bool(row["preferred_variant"]),
            str(row["submitted_word_order_candidate"]) != "source_little_uint32",
            int(row["word_index"]),
            str(row["submitted_word_order_candidate"]),
            str(row["probability_variant"]),
        ),
    )[0]


def _classify_range_contract(
    submitted_replays: Sequence[Mapping[str, Any]],
    local_reemits: Sequence[Mapping[str, Any]],
    *,
    default_variant: str,
    symbol_count: int,
) -> dict[str, Any]:
    best = _best_replay(submitted_replays)
    any_submitted_passed = any(
        row.get("replay", {}).get("passed") is True for row in submitted_replays
    )
    any_exact_word_match = any(
        comparison.get("exact_match") is True
        for row in local_reemits
        for comparison in row.get("submitted_word_comparisons", [])
        if isinstance(comparison, Mapping)
    )
    clone_direct_disagree = any(
        row.get("clone_direct_words_equal") is False for row in local_reemits
    )

    source_default = [
        row
        for row in submitted_replays
        if row.get("word_order_candidate") == "source_little_uint32"
        and row.get("probability_variant") == default_variant
    ]
    source_default_match = (
        int(source_default[0]["replay"].get("matched_prefix_symbol_count", 0))
        if source_default
        else 0
    )
    best_match = int(best.get("matched_prefix_symbol_count", 0)) if best else 0
    non_source_improves = bool(
        best
        and best.get("word_order_candidate") != "source_little_uint32"
        and best_match > source_default_match
    )

    default_common = 0
    best_common = 0
    best_common_variant: str | None = None
    for row in local_reemits:
        if row.get("finalization") != "clone_get_compressed":
            continue
        comparisons = [
            comparison
            for comparison in row.get("submitted_word_comparisons", [])
            if isinstance(comparison, Mapping)
            and comparison.get("submitted_label") == "source_little_uint32"
        ]
        if not comparisons:
            continue
        common = int(comparisons[0].get("common_prefix_word_count", 0))
        if row.get("probability_variant") == default_variant:
            default_common = max(default_common, common)
        if common > best_common:
            best_common = common
            best_common_variant = str(row.get("probability_variant"))

    finalization_sensitive = any(
        row.get("finalization_sensitivity", {})
        .get("submitted_full_stream_replay", {})
        .get("passed")
        is True
        and row.get("finalization_sensitivity", {})
        .get("submitted_same_word_count_replay", {})
        .get("passed")
        is not True
        for row in local_reemits
    )

    if any_exact_word_match:
        status = "byte_exact_prefix_reemit_hypothesis_found"
        likely = "prefix_reemit_candidate_requires_deeper_validation"
    elif clone_direct_disagree:
        status = "range_encoder_finalization_api_disagrees"
        likely = "range_finalization_or_flush"
    elif non_source_improves:
        status = "uint32_word_order_hypothesis_improves_prefix_replay"
        likely = "uint32_word_order"
    elif best_common > default_common and best_common_variant != default_variant:
        status = "probability_variant_improves_local_word_prefix"
        likely = "precision_or_rounding"
    elif finalization_sensitive:
        status = "submitted_full_stream_passes_but_truncated_prefix_fails"
        likely = "range_finalization_or_flush"
    elif not any_submitted_passed:
        status = "submitted_stream_does_not_decode_reference_prefix"
        likely = "context_row_probability_or_reference_symbol_grammar"
    else:
        status = "range_contract_unresolved_after_prefix_matrix"
        likely = "mixed_or_deeper_range_contract"

    return {
        "schema": "pr91_hpm1_range_contract_classification_v1",
        "status": status,
        "likely_blocker_class": likely,
        "score_claim": False,
        "dispatch_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        "symbol_count": int(symbol_count),
        "any_submitted_word_order_replay_passed": bool(any_submitted_passed),
        "any_local_reemit_exact_word_match": bool(any_exact_word_match),
        "clone_direct_finalization_disagrees": bool(clone_direct_disagree),
        "finalization_sensitive_full_vs_truncated_replay": bool(
            finalization_sensitive
        ),
        "best_submitted_replay": best,
        "default_source_little_matched_prefix_symbol_count": int(
            source_default_match
        ),
        "best_source_little_local_word_common_prefix": int(best_common),
        "default_source_little_local_word_common_prefix": int(default_common),
        "scope_note": (
            "This classification is a bounded local prefix diagnostic. It is "
            "not a full HPM1 decode, byte-exact reencode proof, score claim, "
            "or dispatch unlock."
        ),
    }


def audit_pr91_hpm1_range_contract_from_rows(
    raw_rows: Sequence[np.ndarray],
    reference_symbols: Sequence[int],
    submitted_words_by_candidate: Mapping[str, np.ndarray],
    *,
    probability_variants: Sequence[str] | None = None,
    prob_eps: float = PROB_EPS,
    default_probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
) -> dict[str, Any]:
    """Build a fail-closed range-contract manifest from deterministic rows."""

    _require_range_coder()
    variants = _resolve_probability_variants(probability_variants)
    default_variant = resolve_hpac_probability_variant(default_probability_variant).name
    if len(raw_rows) != len(reference_symbols):
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "row_symbol_count_mismatch",
            row_count=len(raw_rows),
            symbol_count=len(reference_symbols),
        )
    if not raw_rows:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "at_least_one_probability_row_required",
        )
    submitted_words = {
        str(candidate): np.ascontiguousarray(words, dtype=np.uint32)
        for candidate, words in submitted_words_by_candidate.items()
    }
    if not submitted_words:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "at_least_one_submitted_word_candidate_required",
        )

    submitted_replays: list[dict[str, Any]] = []
    for candidate, words in submitted_words.items():
        for variant in variants:
            submitted_replays.append(
                {
                    "word_order_candidate": candidate,
                    "probability_variant": variant,
                    "submitted_words": _uint32_words_digest(words),
                    "replay": _decode_prefix_against_words(
                        words,
                        raw_rows,
                        reference_symbols,
                        probability_variant=variant,
                        prob_eps=prob_eps,
                    ),
                }
            )

    local_reemits: list[dict[str, Any]] = []
    clone_words_by_variant: dict[str, np.ndarray] = {}
    direct_words_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        for finalization in RANGE_FINALIZATION_HYPOTHESES:
            words, finalization_report = _encode_reference_prefix_words(
                raw_rows,
                reference_symbols,
                probability_variant=variant,
                prob_eps=prob_eps,
                finalization=finalization,
            )
            if finalization == "clone_get_compressed":
                clone_words_by_variant[variant] = words
            else:
                direct_words_by_variant[variant] = words
            comparisons = [
                _word_prefix_comparison(
                    words,
                    candidate_words,
                    local_label=f"{variant}:{finalization}",
                    submitted_label=candidate,
                )
                for candidate, candidate_words in submitted_words.items()
            ]
            source_words = submitted_words.get("source_little_uint32")
            finalization_sensitivity = (
                _summarize_finalization_sensitivity(
                    words,
                    source_words,
                    raw_rows,
                    reference_symbols,
                    probability_variant=variant,
                    prob_eps=prob_eps,
                )
                if source_words is not None
                else None
            )
            local_reemits.append(
                {
                    "probability_variant": variant,
                    "finalization": finalization,
                    **finalization_report,
                    "local_replay": _decode_prefix_against_words(
                        words,
                        raw_rows,
                        reference_symbols,
                        probability_variant=variant,
                        prob_eps=prob_eps,
                    ),
                    "submitted_word_comparisons": comparisons,
                    "finalization_sensitivity": finalization_sensitivity,
                }
            )

    for row in local_reemits:
        variant = str(row["probability_variant"])
        clone_words = clone_words_by_variant.get(variant)
        direct_words = direct_words_by_variant.get(variant)
        if clone_words is not None and direct_words is not None:
            row["clone_direct_words_equal"] = bool(
                np.array_equal(clone_words, direct_words)
            )
            row["clone_direct_word_comparison"] = _word_prefix_comparison(
                clone_words,
                direct_words,
                local_label=f"{variant}:clone_get_compressed",
                submitted_label=f"{variant}:direct_get_compressed",
            )

    classification = _classify_range_contract(
        submitted_replays,
        local_reemits,
        default_variant=default_variant,
        symbol_count=len(reference_symbols),
    )
    first_submitted_mismatch = _first_replay_mismatch(submitted_replays)
    source_little_submitted_mismatch = _replay_mismatch_for_candidate(
        submitted_replays,
        word_order_candidate="source_little_uint32",
        probability_variant=default_variant,
    )
    first_local_mismatch = _first_local_word_mismatch(
        local_reemits,
        preferred_variant=default_variant,
    )

    return _jsonable(
        {
            "schema": "pr91_hpm1_range_contract_rows_manifest_v1",
            "status": classification["status"],
            "score_claim": False,
            "dispatch_allowed": False,
            "dispatch_attempted": False,
            "dispatch_performed": False,
            "gpu_or_remote_work": False,
            "local_only": True,
            "ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical",
            "prob_eps": float(prob_eps),
            "probability_variants": list(variants),
            "default_probability_variant": default_variant,
            "prefix": {
                "symbol_count": len(reference_symbols),
                "probability_rows": _probability_row_digest(raw_rows),
                "reference_symbols": _symbol_prefix_digest(reference_symbols),
            },
            "submitted_word_order_replays": submitted_replays,
            "local_reemit_hypotheses": local_reemits,
            "classification": classification,
            "first_mismatch_evidence": {
                "source_little_submitted_reference_decode_mismatch": (
                    source_little_submitted_mismatch
                ),
                "earliest_submitted_reference_decode_mismatch": (
                    first_submitted_mismatch
                ),
                "local_reemit_word_mismatch": first_local_mismatch,
            },
            "next_required_artifacts": [
                "recover true PR91 encoder semantic symbols or context rows",
                "prove full submitted-token decode before re-encode work",
                "prove byte-exact range uint32 re-emission before any dispatch",
            ],
        }
    )


def _collect_reference_prefix_rows(
    payload: Any,
    model: Any,
    reference_tokens: np.ndarray,
    *,
    symbol_limit: int,
    spatial_order_candidate: str,
    device: str,
) -> dict[str, Any]:
    if torch is None or F is None:  # pragma: no cover
        raise Pr91Hpm1Error("dependency_contract", "torch_missing")
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_range_contract_is_cpu_only",
            requested_device=device,
        )

    limit = _validate_symbol_limit(symbol_limit)
    _spatial_order_description(spatial_order_candidate)
    dev = torch.device(device)
    model = model.to(dev).eval()
    masks = _group_masks(
        payload.height,
        payload.width,
        P=payload.predictor_count,
        delta=payload.delta,
        device=dev,
    )
    raw_rows: list[np.ndarray] = []
    reference_symbols: list[int] = []
    positions: list[dict[str, Any]] = []

    decoded_prev = torch.zeros(
        (1, payload.height, payload.width),
        dtype=torch.long,
        device=dev,
    )
    collected = 0
    with torch.no_grad():
        for frame in range(payload.n_frames):
            idx = torch.tensor([frame], dtype=torch.long, device=dev)
            cur = torch.zeros(
                (1, payload.height, payload.width),
                dtype=torch.long,
                device=dev,
            )
            for group, mask in enumerate(masks):
                if mask is None:
                    continue
                coords = _hpm1_group_coords_for_spatial_order(
                    payload,
                    group=group,
                    mask=mask,
                    candidate=spatial_order_candidate,
                    device=dev,
                )
                logits = model(cur, idx, decoded_prev)
                probs = F.softmax(logits.float(), dim=1)
                probs_at_group = (
                    probs[0][:, coords[:, 0], coords[:, 1]]
                    .permute(1, 0)
                    .contiguous()
                )
                probs_np = probs_at_group.cpu().numpy()
                coord_np = coords.detach().cpu().numpy()
                ref_at_group = reference_tokens[
                    frame,
                    coord_np[:, 0],
                    coord_np[:, 1],
                ].astype(np.int64, copy=False)
                for symbol_in_group, row in enumerate(probs_np):
                    if collected >= limit:
                        break
                    raw_rows.append(np.ascontiguousarray(row, dtype=np.float32))
                    reference_symbols.append(int(ref_at_group[symbol_in_group]))
                    y, x = coord_np[symbol_in_group]
                    positions.append(
                        {
                            "global_symbol": int(collected),
                            "frame": int(frame),
                            "group": int(group),
                            "symbol_in_group": int(symbol_in_group),
                            "pixel_yx": {"y": int(y), "x": int(x)},
                            "reference_symbol": int(
                                ref_at_group[symbol_in_group]
                            ),
                        }
                    )
                    collected += 1
                cur[0, coords[:, 0], coords[:, 1]] = torch.from_numpy(
                    ref_at_group
                ).to(dev)
                if collected >= limit:
                    break
            decoded_prev = torch.from_numpy(
                reference_tokens[frame : frame + 1].astype(np.int64, copy=False)
            ).to(dev)
            if collected >= limit:
                break

    if len(raw_rows) != limit:
        raise Pr91Hpm1Error(
            "pr91_hpm1_range_contract",
            "unable_to_collect_requested_prefix_rows",
            requested=limit,
            collected=len(raw_rows),
        )
    return {
        "spatial_order_candidate": spatial_order_candidate,
        "symbol_limit": int(limit),
        "raw_rows": raw_rows,
        "reference_symbols": reference_symbols,
        "positions": positions,
    }


def run_pr91_hpm1_range_contract_diagnostic(
    archive: Path = DEFAULT_PR91_ARCHIVE,
    *,
    reference_tokens_path: Path = DEFAULT_PR85_QMA9_DECODED_REFERENCE_TOKEN_SOURCE,
    reference_layout: str = "legacy_assume_nhw",
    device: str = "cpu",
    spatial_order_candidate: str = DEFAULT_PR91_HPM1_RANGE_CONTRACT_SPATIAL_ORDER,
    symbol_limit: int = DEFAULT_PR91_HPM1_RANGE_CONTRACT_SYMBOL_LIMIT,
    probability_variants: Sequence[str] | None = None,
    prob_eps: float = PROB_EPS,
    require_expected_reference_sha: bool = True,
    output_dir: Path | None = None,
    write_json: bool = True,
) -> dict[str, Any]:
    """Run the PR91-specific local range uint32 contract diagnostic."""

    started_at = time.time()
    if str(device) != "cpu":
        raise Pr91Hpm1Error(
            "device_contract",
            "pr91_hpm1_range_contract_is_cpu_only",
            requested_device=device,
        )
    variants = _resolve_probability_variants(probability_variants)
    archive_path = Path(archive)
    payload = extract_pr91_hpm1_payload(archive_path)
    reference_tokens, reference_report = _load_reference_tokens(
        Path(reference_tokens_path),
        payload.n_frames,
        payload.height,
        payload.width,
        reference_layout,
    )
    reference_sha_matches_expected = bool(
        reference_report["matches_expected_pr85_qma9_token_source"]
    )
    if require_expected_reference_sha and not reference_sha_matches_expected:
        raise Pr91Hpm1Error(
            "reference_token_contract",
            "unexpected_pr85_qma9_reference_token_sha256",
            path=reference_report["path"],
            expected_sha256=reference_report["expected_sha256"],
            actual_sha256=reference_report["sha256"],
            layout=reference_layout,
        )

    model = load_hpm1_hpac_model(payload, device=device)
    prefix = _collect_reference_prefix_rows(
        payload,
        model,
        reference_tokens,
        symbol_limit=symbol_limit,
        spatial_order_candidate=spatial_order_candidate,
        device=device,
    )
    submitted_words_by_candidate = {
        candidate: _hpm1_token_words_for_candidate(payload, candidate)
        for candidate in PR91_HPM1_TOKEN_WORD_ORDER_CANDIDATES
    }
    row_manifest = audit_pr91_hpm1_range_contract_from_rows(
        prefix["raw_rows"],
        prefix["reference_symbols"],
        submitted_words_by_candidate,
        probability_variants=variants,
        prob_eps=prob_eps,
        default_probability_variant=DEFAULT_HPAC_PROBABILITY_VARIANT,
    )
    report = _jsonable(
        {
            "schema": "pr91_hpm1_range_contract_diagnostic_v1",
            "tool": "tac.pr91_hpm1_range_contract.run_pr91_hpm1_range_contract_diagnostic",
            "recorded_at_utc": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat(),
            "status": row_manifest["status"],
            "score_claim": False,
            "dispatch_allowed": False,
            "dispatch_attempted": False,
            "dispatch_performed": False,
            "gpu_or_remote_work": False,
            "local_only": True,
            "ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical",
            "evidence_scope": "local_cpu_pr91_hpm1_range_uint32_contract_prefix",
            "device": device,
            "archive": _archive_report(archive_path),
            "reference_tokens": reference_report,
            "reference_token_sha256_contract": {
                "required": bool(require_expected_reference_sha),
                "matches_expected": reference_sha_matches_expected,
                "expected_sha256": reference_report["expected_sha256"],
                "actual_sha256": reference_report["sha256"],
            },
            "hpm1_static_contract": validate_hpm1_static_contract(payload),
            "pr86_hpac_relationship": compare_hpm1_to_pr86_hpac_contract(
                payload
            ),
            "dependency_report": collect_dependency_report(strict=False),
            "payload": {
                "config": payload.config(),
                "tokens_bytes": len(payload.tokens),
                "tokens_sha256": sha256_bytes(payload.tokens),
                "hpac_bytes": len(payload.hpac),
                "hpac_sha256": sha256_bytes(payload.hpac),
            },
            "prefix_collection": {
                "spatial_order_candidate": prefix["spatial_order_candidate"],
                "symbol_limit": prefix["symbol_limit"],
                "positions": prefix["positions"],
            },
            "range_contract": row_manifest,
            "exact_fail_closed_findings": {
                "full_decode_proven": False,
                "byte_exact_reencode_proven": False,
                "ready_for_exact_eval_dispatch": False,
                "first_mismatch_evidence": row_manifest[
                    "first_mismatch_evidence"
                ],
            },
            "next_required_artifact": (
                "recover a PR91-specific encoder-side symbol/probability/range "
                "trace that makes the submitted uint32 stream replay and "
                "re-emit byte-exactly before any dispatch"
            ),
            "elapsed_sec": round(time.time() - started_at, 3),
        }
    )
    if write_json and output_dir is not None:
        write_json_report(
            report,
            Path(output_dir) / "range_contract_diagnostic.json",
        )
    return report


def write_range_contract_json(report: Mapping[str, Any], path: Path) -> None:
    """Write a range-contract manifest using the repo's JSON conventions."""

    write_json_report(report, Path(path))
