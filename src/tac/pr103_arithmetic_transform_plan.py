"""Fail-closed PR103 arithmetic-transform planning helpers.

The PR103 ``hnerv_lc_ac`` schema profiler proves that the public merged range
stream can round-trip byte-identically and ranks streams by model gap. This
module is the next planning layer: it names one target stream and records the
byte-accounting and runtime-adapter blockers needed before anyone can turn the
idea into a scored archive.

It intentionally emits no candidate archive and never authorizes dispatch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.hnerv_lowlevel_packer import (
    read_strict_single_member_zip,
    write_stored_single_member_zip,
)
from tac.hnerv_pr103_lc_ac_schema import (
    AC_STREAM_SPECS,
    HI_SYMBOL_COUNT,
    PUBLIC_PR103_LAYOUT,
    RATE_SCORE_PER_BYTE,
    Pr103LcAcLayout,
    decode_pr103_auxiliary_models,
    decode_pr103_merged_ac_stream,
    encode_pr103_merged_ac_stream,
    parse_pr103_lc_ac_payload,
)
from tac.repo_io import read_json, repo_relative, sha256_bytes, sha256_file

try:  # pragma: no cover - optional outside the contest venv
    import constriction
except ImportError:  # pragma: no cover
    constriction = None  # type: ignore[assignment]


PLAN_SCHEMA = "pr103_arithmetic_transform_plan_v1"
RETARGET_SCHEMA = "pr103_arithmetic_retarget_probe_v1"
COORDINATE_SCHEMA = "pr103_arithmetic_histogram_coordinate_probe_v1"
BEAM_SCHEMA = "pr103_arithmetic_histogram_beam_probe_v1"
CANDIDATE_SCHEMA = "pr103_arithmetic_histogram_candidate_v1"
DEFAULT_STRATEGY = "retarget_categorical_model_to_decoded_symbols"


class Pr103ArithmeticTransformPlanError(ValueError):
    """Raised when a PR103 arithmetic transform plan input is invalid."""


@dataclasses.dataclass(frozen=True)
class TransformTargetSelection:
    """A deterministic target selector for PR103 arithmetic planning."""

    label: str | None = None
    rank: int | None = None

    def describe(self) -> dict[str, Any]:
        return {"label": self.label or "", "rank": self.rank}


def build_pr103_arithmetic_transform_plan(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    target_label: str | None = None,
    target_rank: int | None = None,
    strategy: str = DEFAULT_STRATEGY,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-score transform plan from a PR103 arithmetic schema manifest.

    ``target_rank`` is one-based and indexes ``next_arithmetic_schema_targets``.
    If neither selector is provided, the top ranked target is used.
    """

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest, manifest_record = _load_manifest(schema_manifest, repo)
    selector = TransformTargetSelection(label=target_label, rank=target_rank)
    target = _select_target(manifest, selector)
    source = _mapping(manifest.get("source_archive"))
    merged = _mapping(manifest.get("merged_arithmetic_stream"))
    manifest_blockers = _schema_manifest_blockers(manifest, merged)
    expected_savings_bytes = max(0, int(target.get("model_gap_bytes_estimate") or 0))
    expected_rate_delta = -float(expected_savings_bytes) * RATE_SCORE_PER_BYTE
    proposal_id = _proposal_id(source, target, strategy)
    blockers = _unique_ordered(
        [
            *manifest_blockers,
            "candidate_archive_missing",
            "candidate_runtime_adapter_missing",
            "candidate_symbol_roundtrip_proof_missing",
            "candidate_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": PLAN_SCHEMA,
        "proposal_id": proposal_id,
        "strategy": strategy,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "target_selection": selector.describe(),
        "source_schema_manifest": manifest_record,
        "source_archive": {
            "path": str(source.get("path") or ""),
            "bytes": source.get("bytes"),
            "sha256": str(source.get("sha256") or ""),
            "member_name": str(source.get("member_name") or ""),
            "member_bytes": source.get("member_bytes"),
            "member_sha256": str(source.get("member_sha256") or ""),
        },
        "merged_arithmetic_stream": {
            "bytes": merged.get("source_bytes"),
            "sha256": str(merged.get("source_sha256") or ""),
            "decoded_symbol_count": merged.get("decoded_symbol_count"),
            "reencoded_byte_identical": merged.get("reencoded_byte_identical") is True,
        },
        "target_stream": _target_record(target),
        "byte_accounting": {
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "expected_savings_bytes_upper_bound": expected_savings_bytes,
            "expected_rate_score_delta_upper_bound": expected_rate_delta,
            "estimate_is_score_claim": False,
            "estimate_caveats": [
                "model_gap_is_a_coding_bound_not_an_archive_delta",
                "histogram_update_overhead_not_accounted_as_candidate_bytes",
                "component_distortion_unknown_until_runtime_adapter_and_exact_cuda",
            ],
        },
        "mutation_surface": {
            "archive_member": source.get("member_name") or "",
            "must_change_sections": [
                "ac_histograms_brotli",
                "merged_range_coded_weights_and_hi_latents",
            ],
            "must_preserve_sections": [
                "scales_fp16",
                "non_ac_weights_brotli",
                "latent_min_scale_fp16",
                "latent_low_bytes_brotli",
                "latent_hi_histogram_brotli",
                "sidecar_corrections_brotli",
            ],
            "requires_fixed_layout_or_runtime_adapter": True,
        },
        "required_proofs_before_archive_preflight": [
            "byte_different_candidate_archive_with_old_new_sha256_pair",
            "candidate_runtime_adapter_consumes_changed_ac_histogram_and_merged_stream",
            "symbol_roundtrip_decodes_expected_target_stream",
            "inflate_output_parity_or_component_delta_report",
            "packet_section_transform_certification",
        ],
        "readiness_blockers": blockers,
        "dispatch_blockers": ["pr103_arithmetic_transform_plan_is_not_dispatch_authorization", *blockers],
    }


def build_pr103_arithmetic_retarget_probe(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    source_archive: str | Path | None = None,
    target_label: str | None = None,
    target_rank: int | None = None,
    strategy: str = DEFAULT_STRATEGY,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Probe one PR103 AC histogram retarget without building an archive."""

    context = _load_pr103_retarget_context(
        schema_manifest=schema_manifest,
        source_archive=source_archive,
        target_label=target_label,
        target_rank=target_rank,
        repo_root=repo_root,
        layout=layout,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    repo = context["repo"]
    manifest = context["manifest"]
    manifest_record = context["manifest_record"]
    target = context["target"]
    archive_path = context["archive_path"]
    source = context["source"]
    histograms = context["histograms"]
    source_merged = context["source_merged"]
    source_histogram_blob = context["source_histogram_blob"]
    source_symbol_streams = context["source_symbol_streams"]
    source_model_weights = context["source_model_weights"]
    source_roundtrip = context["source_roundtrip"]
    target_index = context["target_index"]
    target_record = dict(context["target_record"])
    expected_symbol_sha = str(context["expected_symbol_sha"])
    observed_symbol_sha = str(context["observed_symbol_sha"])
    source_roundtrip_identical = bool(context["source_roundtrip_identical"])
    manifest_blockers = list(context["manifest_blockers"])
    new_model_weights = [np.asarray(item).copy() for item in source_model_weights]
    target_symbols = np.asarray(source_symbol_streams[target_index])
    target_new_weights = _retarget_weights_for_symbols(target_symbols, new_model_weights[target_index])
    new_model_weights[target_index] = target_new_weights
    retargeted_merged = encode_pr103_merged_ac_stream(source_symbol_streams, new_model_weights)

    if target_index >= len(histograms):
        raise Pr103ArithmeticTransformPlanError(
            "latent_hi_bytes retarget is not supported by the q8 AC histogram probe"
        )
    new_histograms = histograms.copy()
    new_histograms[target_index] = target_new_weights.astype(np.uint8)
    retargeted_histogram_raw = new_histograms.astype(np.uint8).tobytes()
    source_histogram_raw = histograms.astype(np.uint8).tobytes()
    retargeted_histogram_blob = brotli.compress(retargeted_histogram_raw, quality=11)

    merged_delta = len(retargeted_merged) - len(source_merged)
    histogram_delta = len(retargeted_histogram_blob) - len(source_histogram_blob)
    estimated_member_delta = merged_delta + histogram_delta
    retarget_noop = (
        retargeted_merged == source_merged
        and retargeted_histogram_raw == source_histogram_raw
    )
    proof_blockers = []
    if context["decoder_maybe_exhausted"] is not True:
        proof_blockers.append("source_merged_range_decoder_not_exhausted")
    if not source_roundtrip_identical:
        proof_blockers.append("source_merged_stream_roundtrip_not_byte_identical")
    if expected_symbol_sha and expected_symbol_sha != observed_symbol_sha:
        proof_blockers.append("target_decoded_symbols_sha_mismatch")
    if retarget_noop:
        proof_blockers.append("retarget_probe_noop")
    blockers = _unique_ordered(
        [
            *manifest_blockers,
            *proof_blockers,
            "candidate_archive_missing",
            "candidate_runtime_adapter_missing",
            "candidate_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": RETARGET_SCHEMA,
        "proposal_id": _proposal_id(
            _mapping(manifest.get("source_archive")),
            target,
            strategy + ":probe",
        ),
        "strategy": strategy,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_schema_manifest": manifest_record,
        "source_archive": {
            "path": repo_relative(archive_path, repo),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
        },
        "target_stream": {
            **_target_record(target),
            "decoded_stream_index": target_index,
            "observed_decoded_symbols_sha256": observed_symbol_sha,
        },
        "source_roundtrip": {
            "merged_ac_bytes": len(source_merged),
            "merged_ac_sha256": sha256_bytes(source_merged),
            "reencoded_bytes": len(source_roundtrip),
            "reencoded_sha256": sha256_bytes(source_roundtrip),
            "byte_identical": source_roundtrip_identical,
            "decoder_maybe_exhausted": context["decoder_maybe_exhausted"] is True,
        },
        "retargeted_histogram": {
            "source_raw_sha256": sha256_bytes(source_histogram_raw),
            "retargeted_raw_sha256": sha256_bytes(retargeted_histogram_raw),
            "source_brotli_bytes": len(source_histogram_blob),
            "retargeted_brotli_bytes": len(retargeted_histogram_blob),
            "brotli_byte_delta": histogram_delta,
            "retargeted_weights_sha256": sha256_bytes(target_new_weights.tobytes()),
        },
        "retargeted_merged_stream": {
            "source_bytes": len(source_merged),
            "source_sha256": sha256_bytes(source_merged),
            "retargeted_bytes": len(retargeted_merged),
            "retargeted_sha256": sha256_bytes(retargeted_merged),
            "byte_delta": merged_delta,
            "no_op": retarget_noop,
        },
        "byte_accounting": {
            "estimated_member_delta_if_runtime_adapter_supported": estimated_member_delta,
            "estimated_rate_score_delta_if_components_unchanged": (
                float(estimated_member_delta) * RATE_SCORE_PER_BYTE
            ),
            "estimate_is_score_claim": False,
            "fixed_layout_requires_runtime_adapter": True,
            "caveats": [
                "archive_not_emitted",
                "runtime_adapter_not_integrated",
                "fixed_pr103_section_lengths_do_not_accept_changed_merged_ac_bytes",
                "component_distortion_unknown_until_exact_cuda",
            ],
        },
        "readiness_blockers": blockers,
        "dispatch_blockers": ["pr103_arithmetic_retarget_probe_is_not_dispatch_authorization", *blockers],
    }


def build_pr103_arithmetic_histogram_coordinate_probe(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    source_archive: str | Path | None = None,
    target_label: str | None = None,
    target_rank: int | None = None,
    top_symbols: int = 32,
    deltas: Sequence[int] = (-2, -1, 1, 2),
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Search small q8 histogram perturbations against PR103 wire bytes."""

    if top_symbols <= 0:
        raise Pr103ArithmeticTransformPlanError("top_symbols must be positive")
    if not deltas:
        raise Pr103ArithmeticTransformPlanError("at least one coordinate delta is required")
    context = _load_pr103_retarget_context(
        schema_manifest=schema_manifest,
        source_archive=source_archive,
        target_label=target_label,
        target_rank=target_rank,
        repo_root=repo_root,
        layout=layout,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    repo = context["repo"]
    manifest = context["manifest"]
    manifest_record = context["manifest_record"]
    target = context["target"]
    archive_path = context["archive_path"]
    source = context["source"]
    histograms = context["histograms"]
    source_merged = context["source_merged"]
    source_histogram_blob = context["source_histogram_blob"]
    source_symbol_streams = context["source_symbol_streams"]
    source_model_weights = context["source_model_weights"]
    source_roundtrip = context["source_roundtrip"]
    target_index = context["target_index"]
    target_record = dict(context["target_record"])
    expected_symbol_sha = str(context["expected_symbol_sha"])
    observed_symbol_sha = str(context["observed_symbol_sha"])
    source_roundtrip_identical = bool(context["source_roundtrip_identical"])
    manifest_blockers = list(context["manifest_blockers"])
    source_histogram_raw = histograms.astype(np.uint8).tobytes()

    if target_index >= len(histograms):
        raise Pr103ArithmeticTransformPlanError(
            "latent_hi_bytes coordinate probe is not supported by the q8 AC histogram probe"
        )
    base_target_weights = np.asarray(source_model_weights[target_index], dtype=np.uint8)
    target_symbols = np.asarray(source_symbol_streams[target_index], dtype=np.int64)
    counts = np.bincount(target_symbols.reshape(-1), minlength=base_target_weights.size)
    candidate_symbols = [
        int(symbol)
        for symbol in sorted(
            np.nonzero(counts)[0].tolist(),
            key=lambda symbol: (-int(counts[int(symbol)]), int(symbol)),
        )[:top_symbols]
    ]
    candidate_rows: list[dict[str, Any]] = []
    for symbol in candidate_symbols:
        old_weight = int(base_target_weights[symbol])
        for delta in deltas:
            new_weight = old_weight + int(delta)
            if new_weight < 0 or new_weight > 255 or new_weight == old_weight:
                continue
            weights = base_target_weights.copy()
            weights[symbol] = np.uint8(new_weight)
            candidate_rows.append(
                _public_candidate_row(
                    _score_histogram_candidate(
                        weights=weights,
                        histograms=histograms,
                        target_index=target_index,
                        source_symbol_streams=source_symbol_streams,
                        source_model_weights=source_model_weights,
                        source_merged=source_merged,
                        source_histogram_blob=source_histogram_blob,
                        moves=[
                            {
                                "round": 1,
                                "symbol": symbol,
                                "symbol_count": int(counts[symbol]),
                                "delta": int(delta),
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                            }
                        ],
                    )
                )
            )
    candidate_rows.sort(key=_candidate_sort_key)
    best = candidate_rows[0] if candidate_rows else {}
    proof_blockers = []
    if context["decoder_maybe_exhausted"] is not True:
        proof_blockers.append("source_merged_range_decoder_not_exhausted")
    if not source_roundtrip_identical:
        proof_blockers.append("source_merged_stream_roundtrip_not_byte_identical")
    if expected_symbol_sha and expected_symbol_sha != observed_symbol_sha:
        proof_blockers.append("target_decoded_symbols_sha_mismatch")
    if not candidate_rows:
        proof_blockers.append("coordinate_probe_no_candidates")
    elif int(best.get("estimated_member_delta_if_runtime_adapter_supported", 0)) >= 0:
        proof_blockers.append("no_byte_positive_histogram_coordinate_candidate")
    blockers = _unique_ordered(
        [
            *manifest_blockers,
            *proof_blockers,
            "candidate_archive_missing",
            "candidate_runtime_adapter_missing",
            "candidate_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": COORDINATE_SCHEMA,
        "proposal_id": _proposal_id(
            _mapping(manifest.get("source_archive")),
            target,
            f"coordinate_probe:top{top_symbols}",
        ),
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_schema_manifest": manifest_record,
        "source_archive": {
            "path": repo_relative(archive_path, repo),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
        },
        "target_stream": {
            **_target_record(target),
            "decoded_stream_index": target_index,
            "observed_decoded_symbols_sha256": observed_symbol_sha,
        },
        "search_config": {
            "top_symbols": int(top_symbols),
            "deltas": [int(delta) for delta in deltas],
            "candidate_symbol_count": len(candidate_symbols),
            "candidate_count": len(candidate_rows),
            "objective": "minimize_merged_ac_delta_plus_ac_histograms_brotli_delta",
        },
        "source_roundtrip": {
            "merged_ac_bytes": len(source_merged),
            "merged_ac_sha256": sha256_bytes(source_merged),
            "reencoded_bytes": len(source_roundtrip),
            "reencoded_sha256": sha256_bytes(source_roundtrip),
            "byte_identical": source_roundtrip_identical,
            "decoder_maybe_exhausted": context["decoder_maybe_exhausted"] is True,
        },
        "best_candidate": best,
        "top_candidates": candidate_rows[:20],
        "byte_accounting": {
            "estimate_is_score_claim": False,
            "fixed_layout_requires_runtime_adapter": True,
            "caveats": [
                "archive_not_emitted",
                "runtime_adapter_not_integrated",
                "fixed_pr103_section_lengths_do_not_accept_changed_merged_ac_bytes",
                "component_distortion_unknown_until_exact_cuda",
            ],
        },
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_arithmetic_histogram_coordinate_probe_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def build_pr103_arithmetic_histogram_beam_probe(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    source_archive: str | Path | None = None,
    target_label: str | None = None,
    target_rank: int | None = None,
    top_symbols: int = 16,
    deltas: Sequence[int] = (-2, -1, 1, 2),
    rounds: int = 3,
    beam_width: int = 8,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Run a deterministic local beam search over q8 histogram coordinates."""

    if rounds <= 0:
        raise Pr103ArithmeticTransformPlanError("rounds must be positive")
    if beam_width <= 0:
        raise Pr103ArithmeticTransformPlanError("beam_width must be positive")
    if top_symbols <= 0:
        raise Pr103ArithmeticTransformPlanError("top_symbols must be positive")
    if not deltas:
        raise Pr103ArithmeticTransformPlanError("at least one coordinate delta is required")
    context = _load_pr103_retarget_context(
        schema_manifest=schema_manifest,
        source_archive=source_archive,
        target_label=target_label,
        target_rank=target_rank,
        repo_root=repo_root,
        layout=layout,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    repo = context["repo"]
    manifest = context["manifest"]
    manifest_record = context["manifest_record"]
    target = context["target"]
    archive_path = context["archive_path"]
    source = context["source"]
    histograms = context["histograms"]
    source_merged = context["source_merged"]
    source_histogram_blob = context["source_histogram_blob"]
    source_symbol_streams = context["source_symbol_streams"]
    source_model_weights = context["source_model_weights"]
    source_roundtrip = context["source_roundtrip"]
    target_index = context["target_index"]
    target_record = dict(context["target_record"])
    expected_symbol_sha = str(context["expected_symbol_sha"])
    observed_symbol_sha = str(context["observed_symbol_sha"])
    source_roundtrip_identical = bool(context["source_roundtrip_identical"])
    manifest_blockers = list(context["manifest_blockers"])
    if target_index >= len(histograms):
        raise Pr103ArithmeticTransformPlanError(
            "latent_hi_bytes beam probe is not supported by the q8 AC histogram probe"
        )

    base_target_weights = np.asarray(source_model_weights[target_index], dtype=np.uint8)
    target_symbols = np.asarray(source_symbol_streams[target_index], dtype=np.int64)
    counts = np.bincount(target_symbols.reshape(-1), minlength=base_target_weights.size)
    candidate_symbols = [
        int(symbol)
        for symbol in sorted(
            np.nonzero(counts)[0].tolist(),
            key=lambda symbol: (-int(counts[int(symbol)]), int(symbol)),
        )[:top_symbols]
    ]

    base_row = _score_histogram_candidate(
        weights=base_target_weights,
        histograms=histograms,
        target_index=target_index,
        source_symbol_streams=source_symbol_streams,
        source_model_weights=source_model_weights,
        source_merged=source_merged,
        source_histogram_blob=source_histogram_blob,
        moves=[],
    )
    beam = [base_row]
    seen = {str(base_row["histogram_raw_sha256"])}
    all_rows: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        expanded: list[dict[str, Any]] = []
        for state in beam:
            state_weights = np.asarray(state["_weights"], dtype=np.uint8)
            for symbol in candidate_symbols:
                old_weight = int(state_weights[symbol])
                for delta in deltas:
                    new_weight = old_weight + int(delta)
                    if new_weight < 0 or new_weight > 255 or new_weight == old_weight:
                        continue
                    weights = state_weights.copy()
                    weights[symbol] = np.uint8(new_weight)
                    row = _score_histogram_candidate(
                        weights=weights,
                        histograms=histograms,
                        target_index=target_index,
                        source_symbol_streams=source_symbol_streams,
                        source_model_weights=source_model_weights,
                        source_merged=source_merged,
                        source_histogram_blob=source_histogram_blob,
                        moves=[
                            *list(state.get("moves") or []),
                            {
                                "round": round_index,
                                "symbol": symbol,
                                "symbol_count": int(counts[symbol]),
                                "delta": int(delta),
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                            },
                        ],
                    )
                    key = str(row["histogram_raw_sha256"])
                    if key in seen:
                        continue
                    seen.add(key)
                    expanded.append(row)
        expanded.sort(key=_candidate_sort_key)
        all_rows.extend(expanded)
        beam = expanded[:beam_width]
        if not beam:
            break
    all_rows.sort(key=_candidate_sort_key)
    best = all_rows[0] if all_rows else {}
    proof_blockers = []
    if context["decoder_maybe_exhausted"] is not True:
        proof_blockers.append("source_merged_range_decoder_not_exhausted")
    if not source_roundtrip_identical:
        proof_blockers.append("source_merged_stream_roundtrip_not_byte_identical")
    if expected_symbol_sha and expected_symbol_sha != observed_symbol_sha:
        proof_blockers.append("target_decoded_symbols_sha_mismatch")
    if not all_rows:
        proof_blockers.append("beam_probe_no_candidates")
    elif int(best.get("estimated_member_delta_if_runtime_adapter_supported", 0)) >= 0:
        proof_blockers.append("no_byte_positive_histogram_beam_candidate")
    blockers = _unique_ordered(
        [
            *manifest_blockers,
            *proof_blockers,
            "candidate_archive_missing",
            "candidate_runtime_adapter_missing",
            "candidate_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": BEAM_SCHEMA,
        "proposal_id": _proposal_id(
            _mapping(manifest.get("source_archive")),
            target,
            f"beam_probe:top{top_symbols}:rounds{rounds}:beam{beam_width}",
        ),
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_schema_manifest": manifest_record,
        "source_archive": {
            "path": repo_relative(archive_path, repo),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
        },
        "target_stream": {
            **_target_record(target),
            "decoded_stream_index": target_index,
            "observed_decoded_symbols_sha256": observed_symbol_sha,
        },
        "search_config": {
            "mode": "beam_search",
            "top_symbols": int(top_symbols),
            "deltas": [int(delta) for delta in deltas],
            "rounds": int(rounds),
            "beam_width": int(beam_width),
            "candidate_symbol_count": len(candidate_symbols),
            "evaluated_candidate_count": len(all_rows),
            "objective": "minimize_merged_ac_delta_plus_ac_histograms_brotli_delta",
        },
        "source_roundtrip": {
            "merged_ac_bytes": len(source_merged),
            "merged_ac_sha256": sha256_bytes(source_merged),
            "reencoded_bytes": len(source_roundtrip),
            "reencoded_sha256": sha256_bytes(source_roundtrip),
            "byte_identical": source_roundtrip_identical,
            "decoder_maybe_exhausted": context["decoder_maybe_exhausted"] is True,
        },
        "best_candidate": _public_candidate_row(best),
        "top_candidates": [_public_candidate_row(row) for row in all_rows[:20]],
        "byte_accounting": {
            "estimate_is_score_claim": False,
            "fixed_layout_requires_runtime_adapter": True,
            "caveats": [
                "archive_not_emitted",
                "runtime_adapter_not_integrated",
                "fixed_pr103_section_lengths_do_not_accept_changed_merged_ac_bytes",
                "component_distortion_unknown_until_exact_cuda",
            ],
        },
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_arithmetic_histogram_beam_probe_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def materialize_pr103_arithmetic_histogram_candidate(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    beam_probe_reports: Sequence[str | Path | Mapping[str, Any]],
    output_archive: str | Path | None = None,
    source_archive: str | Path | None = None,
    member_name: str | None = None,
    repo_root: str | Path | None = None,
    layout: Pr103LcAcLayout = PUBLIC_PR103_LAYOUT,
    stream_specs: Sequence[tuple[str, int, int | None]] = AC_STREAM_SPECS,
    hi_symbol_count: int = HI_SYMBOL_COUNT,
) -> dict[str, Any]:
    """Build a byte-different PR103 histogram candidate and runtime plan.

    The emitted archive is not dispatch-ready by itself: PR103's public runtime
    slices the payload with hard-coded section lengths, so any shorter histogram
    or merged-AC section also requires a reviewed runtime-constant update.
    """

    reports = _load_beam_probe_reports(beam_probe_reports)
    if not reports:
        raise Pr103ArithmeticTransformPlanError("at least one beam probe report is required")
    first_target = str(_mapping(reports[0].get("target_stream")).get("label") or "")
    if not first_target:
        raise Pr103ArithmeticTransformPlanError("beam report missing target_stream.label")
    context = _load_pr103_retarget_context(
        schema_manifest=schema_manifest,
        source_archive=source_archive,
        target_label=first_target,
        target_rank=None,
        repo_root=repo_root,
        layout=layout,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    repo = context["repo"]
    source = context["source"]
    archive_path = context["archive_path"]
    parsed = parse_pr103_lc_ac_payload(source.payload, layout=layout)
    histograms = np.asarray(context["histograms"], dtype=np.uint8).copy()
    source_model_weights = [np.asarray(item).copy() for item in context["source_model_weights"]]
    source_symbol_streams = [np.asarray(item).copy() for item in context["source_symbol_streams"]]
    label_to_index = {
        str(label): index
        for index, (label, _count, _schema_index) in enumerate(stream_specs)
    }
    applied_reports: list[dict[str, Any]] = []
    for report in reports:
        _validate_candidate_report_source(report, source)
        target = _mapping(report.get("target_stream"))
        label = str(target.get("label") or "")
        if label not in label_to_index:
            raise Pr103ArithmeticTransformPlanError(
                f"beam report target is not a q8 AC stream: {label!r}"
            )
        target_index = label_to_index[label]
        best = _mapping(report.get("best_candidate"))
        moves = best.get("moves")
        if not isinstance(moves, Sequence) or isinstance(moves, (str, bytes)) or not moves:
            raise Pr103ArithmeticTransformPlanError(
                f"beam report for {label} has no best_candidate.moves"
            )
        for move in moves:
            item = _mapping(move)
            symbol = int(item.get("symbol"))
            old_weight = int(item.get("old_weight"))
            new_weight = int(item.get("new_weight"))
            if symbol < 0 or symbol >= histograms.shape[1]:
                raise Pr103ArithmeticTransformPlanError(
                    f"beam move symbol out of range for {label}: {symbol}"
                )
            current = int(histograms[target_index, symbol])
            if current != old_weight:
                raise Pr103ArithmeticTransformPlanError(
                    f"beam move old_weight mismatch for {label}[{symbol}]: "
                    f"report={old_weight} current={current}"
                )
            histograms[target_index, symbol] = np.uint8(new_weight)
        source_model_weights[target_index] = histograms[target_index].copy()
        applied_reports.append(
            {
                "target_label": label,
                "move_count": len(moves),
                "source_probe_total_delta": best.get(
                    "estimated_member_delta_if_runtime_adapter_supported"
                ),
                "source_probe_sha256": report.get("tool_run_manifest", {}).get("output_sha256")
                if isinstance(report.get("tool_run_manifest"), Mapping)
                else "",
            }
        )

    candidate_histogram_raw = histograms.astype(np.uint8).tobytes()
    candidate_histogram_blob = brotli.compress(candidate_histogram_raw, quality=11)
    candidate_merged = encode_pr103_merged_ac_stream(source_symbol_streams, source_model_weights)
    candidate_layout = dataclasses.replace(
        layout,
        ac_histograms_brotli=len(candidate_histogram_blob),
        merged_range_coded_weights_and_hi_latents=len(candidate_merged),
    )
    candidate_payload = b"".join(
        [
            parsed.section_bytes("scales_fp16"),
            parsed.section_bytes("non_ac_weights_brotli"),
            candidate_histogram_blob,
            candidate_merged,
            parsed.section_bytes("latent_min_scale_fp16"),
            parsed.section_bytes("latent_low_bytes_brotli"),
            parsed.section_bytes("latent_hi_histogram_brotli"),
            parsed.section_bytes("sidecar_corrections_brotli"),
        ]
    )
    candidate_parsed = parse_pr103_lc_ac_payload(candidate_payload, layout=candidate_layout)
    candidate_aux = decode_pr103_auxiliary_models(
        candidate_parsed,
        ac_stream_count=len(stream_specs),
    )
    candidate_decoded = decode_pr103_merged_ac_stream(
        candidate_parsed.section_bytes("merged_range_coded_weights_and_hi_latents"),
        np.asarray(candidate_aux.pop("_histograms_array"), dtype=np.uint8),
        np.asarray(candidate_aux.pop("_hi_histogram_array")),
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    output_path = Path(output_archive) if output_archive is not None else None
    archive_record = {
        "provided": output_path is not None,
        "path": repo_relative(output_path, repo) if output_path is not None else "",
        "bytes": None,
        "sha256": "",
        "member_name": member_name or source.member_name,
        "member_bytes": len(candidate_payload),
        "member_sha256": sha256_bytes(candidate_payload),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if output_path is not None:
        write_stored_single_member_zip(
            output_path,
            member_name=member_name or source.member_name,
            payload=candidate_payload,
        )
        archive_record.update(
            {
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
        )

    section_diffs = _candidate_section_diffs(
        parsed=parse_pr103_lc_ac_payload(source.payload, layout=layout),
        candidate_parsed=candidate_parsed,
    )
    source_archive_bytes = int(source.archive_bytes)
    candidate_archive_bytes = archive_record["bytes"]
    archive_delta = (
        int(candidate_archive_bytes) - source_archive_bytes
        if candidate_archive_bytes is not None
        else None
    )
    payload_delta = len(candidate_payload) - len(source.payload)
    blockers = _unique_ordered(
        [
            *([] if output_path is not None else ["candidate_archive_missing"]),
            *([] if candidate_payload != source.payload else ["candidate_payload_noop"]),
            *(
                []
                if archive_delta is not None and archive_delta < 0
                else ["candidate_archive_not_smaller_than_source"]
            ),
            *(
                []
                if candidate_decoded.get("reencoded_byte_identical") is True
                else ["candidate_merged_stream_roundtrip_not_byte_identical"]
            ),
            "candidate_runtime_adapter_missing",
            "candidate_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ]
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "planning_only": False,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "source_schema_manifest": context["manifest_record"],
        "source_archive": {
            "path": repo_relative(archive_path, repo),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
            "member_name": source.member_name,
            "member_bytes": source.member_bytes,
            "member_sha256": sha256_bytes(source.payload),
        },
        "candidate_archive": archive_record,
        "byte_accounting": {
            "source_payload_bytes": len(source.payload),
            "candidate_payload_bytes": len(candidate_payload),
            "payload_byte_delta": payload_delta,
            "source_archive_bytes": source_archive_bytes,
            "candidate_archive_bytes": candidate_archive_bytes,
            "archive_byte_delta": archive_delta,
            "estimated_rate_score_delta_if_components_unchanged": (
                float(payload_delta) * RATE_SCORE_PER_BYTE
            ),
            "estimate_is_score_claim": False,
        },
        "runtime_adapter_contract": {
            "public_runtime_constants": _runtime_constants_from_layout(candidate_layout),
            "source_runtime_constants": _runtime_constants_from_layout(layout),
            "required_code_changes": [
                "update HIST_LEN to candidate hists_b length",
                "update MERGED_AC_LEN to candidate merged_ac length",
                "preserve section order in parse_archive",
                "prove inflate consumes candidate archive bytes",
            ],
            "runtime_tree_missing": True,
        },
        "applied_beam_reports": applied_reports,
        "section_diffs": section_diffs,
        "candidate_roundtrip": {
            "merged_ac_bytes": candidate_decoded.get("source_bytes"),
            "merged_ac_sha256": candidate_decoded.get("source_sha256"),
            "reencoded_byte_identical": candidate_decoded.get("reencoded_byte_identical") is True,
            "decoder_maybe_exhausted": candidate_decoded.get("decoder_maybe_exhausted") is True,
            "decoded_symbol_count": candidate_decoded.get("decoded_symbol_count"),
        },
        "readiness_blockers": blockers,
        "dispatch_blockers": [
            "pr103_arithmetic_histogram_candidate_is_not_dispatch_authorization",
            *blockers,
        ],
    }


def render_markdown(plan: Mapping[str, Any]) -> str:
    """Render a compact human review note for a PR103 transform plan."""

    target = _mapping(plan.get("target_stream"))
    byte_accounting = _mapping(plan.get("byte_accounting"))
    lines = [
        "# PR103 Arithmetic Transform Plan",
        "",
        f"- proposal_id: `{plan.get('proposal_id')}`",
        f"- strategy: `{plan.get('strategy')}`",
        f"- score_claim: `{_bool(plan.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(plan.get('dispatch_attempted') is True)}`",
        f"- ready_for_archive_preflight: `{_bool(plan.get('ready_for_archive_preflight') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(plan.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Target",
        "",
        f"- label: `{target.get('label')}`",
        f"- role: `{target.get('role')}`",
        f"- symbols: `{target.get('symbol_count')}`",
        f"- model_gap_bytes_estimate: `{target.get('model_gap_bytes_estimate')}`",
        f"- decoded_symbols_sha256: `{target.get('decoded_symbols_sha256')}`",
        "",
        "## Byte Accounting",
        "",
        f"- expected_savings_bytes_upper_bound: `{byte_accounting.get('expected_savings_bytes_upper_bound')}`",
        f"- expected_rate_score_delta_upper_bound: `{byte_accounting.get('expected_rate_score_delta_upper_bound')}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in plan.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def render_retarget_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human review note for a retarget probe."""

    target = _mapping(report.get("target_stream"))
    merged = _mapping(report.get("retargeted_merged_stream"))
    hist = _mapping(report.get("retargeted_histogram"))
    byte_accounting = _mapping(report.get("byte_accounting"))
    lines = [
        "# PR103 Arithmetic Retarget Probe",
        "",
        f"- proposal_id: `{report.get('proposal_id')}`",
        f"- strategy: `{report.get('strategy')}`",
        f"- score_claim: `{_bool(report.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(report.get('dispatch_attempted') is True)}`",
        f"- ready_for_archive_preflight: `{_bool(report.get('ready_for_archive_preflight') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(report.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Target",
        "",
        f"- label: `{target.get('label')}`",
        f"- stream_index: `{target.get('decoded_stream_index')}`",
        f"- symbols: `{target.get('symbol_count')}`",
        f"- observed_decoded_symbols_sha256: `{target.get('observed_decoded_symbols_sha256')}`",
        "",
        "## Retargeted Bytes",
        "",
        f"- merged_ac_delta: `{merged.get('byte_delta')}`",
        f"- histogram_brotli_delta: `{hist.get('brotli_byte_delta')}`",
        f"- estimated_member_delta_if_runtime_adapter_supported: `{byte_accounting.get('estimated_member_delta_if_runtime_adapter_supported')}`",
        f"- estimated_rate_score_delta_if_components_unchanged: `{byte_accounting.get('estimated_rate_score_delta_if_components_unchanged')}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def render_coordinate_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact review note for a histogram-coordinate probe."""

    target = _mapping(report.get("target_stream"))
    search = _mapping(report.get("search_config"))
    best = _mapping(report.get("best_candidate"))
    lines = [
        "# PR103 Arithmetic Histogram Coordinate Probe",
        "",
        f"- proposal_id: `{report.get('proposal_id')}`",
        f"- score_claim: `{_bool(report.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(report.get('dispatch_attempted') is True)}`",
        f"- ready_for_archive_preflight: `{_bool(report.get('ready_for_archive_preflight') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(report.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Target",
        "",
        f"- label: `{target.get('label')}`",
        f"- stream_index: `{target.get('decoded_stream_index')}`",
        f"- symbols: `{target.get('symbol_count')}`",
        "",
        "## Search",
        "",
        f"- candidate_count: `{search.get('candidate_count')}`",
        f"- top_symbols: `{search.get('top_symbols')}`",
        f"- deltas: `{search.get('deltas')}`",
        "",
        "## Best Candidate",
        "",
        f"- symbol: `{best.get('symbol')}`",
        f"- delta: `{best.get('delta')}`",
        f"- old_weight: `{best.get('old_weight')}`",
        f"- new_weight: `{best.get('new_weight')}`",
        f"- merged_ac_delta: `{best.get('merged_ac_delta')}`",
        f"- histogram_brotli_delta: `{best.get('histogram_brotli_delta')}`",
        f"- estimated_member_delta_if_runtime_adapter_supported: `{best.get('estimated_member_delta_if_runtime_adapter_supported')}`",
        "",
        "## Blockers",
        "",
    ]
    for blocker in report.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def render_beam_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact review note for a histogram beam-search probe."""

    target = _mapping(report.get("target_stream"))
    search = _mapping(report.get("search_config"))
    best = _mapping(report.get("best_candidate"))
    moves = best.get("moves") if isinstance(best.get("moves"), Sequence) else []
    lines = [
        "# PR103 Arithmetic Histogram Beam Probe",
        "",
        f"- proposal_id: `{report.get('proposal_id')}`",
        f"- score_claim: `{_bool(report.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(report.get('dispatch_attempted') is True)}`",
        f"- ready_for_archive_preflight: `{_bool(report.get('ready_for_archive_preflight') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(report.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Target",
        "",
        f"- label: `{target.get('label')}`",
        f"- stream_index: `{target.get('decoded_stream_index')}`",
        f"- symbols: `{target.get('symbol_count')}`",
        "",
        "## Search",
        "",
        f"- evaluated_candidate_count: `{search.get('evaluated_candidate_count')}`",
        f"- candidate_symbol_count: `{search.get('candidate_symbol_count')}`",
        f"- top_symbols: `{search.get('top_symbols')}`",
        f"- rounds: `{search.get('rounds')}`",
        f"- beam_width: `{search.get('beam_width')}`",
        f"- deltas: `{search.get('deltas')}`",
        "",
        "## Best Candidate",
        "",
        f"- change_count: `{best.get('change_count')}`",
        f"- merged_ac_delta: `{best.get('merged_ac_delta')}`",
        f"- histogram_brotli_delta: `{best.get('histogram_brotli_delta')}`",
        f"- estimated_member_delta_if_runtime_adapter_supported: `{best.get('estimated_member_delta_if_runtime_adapter_supported')}`",
        "",
        "## Moves",
        "",
    ]
    for move in moves:
        row = _mapping(move)
        lines.append(
            "- "
            f"round `{row.get('round')}` symbol `{row.get('symbol')}` "
            f"delta `{row.get('delta')}` old `{row.get('old_weight')}` "
            f"new `{row.get('new_weight')}` count `{row.get('symbol_count')}`"
        )
    if not moves:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def render_candidate_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact review note for a materialized histogram candidate."""

    source = _mapping(report.get("source_archive"))
    candidate = _mapping(report.get("candidate_archive"))
    byte_accounting = _mapping(report.get("byte_accounting"))
    runtime_contract = _mapping(report.get("runtime_adapter_contract"))
    constants = _mapping(runtime_contract.get("public_runtime_constants"))
    lines = [
        "# PR103 Arithmetic Histogram Candidate",
        "",
        f"- score_claim: `{_bool(report.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool(report.get('dispatch_attempted') is True)}`",
        f"- ready_for_archive_preflight: `{_bool(report.get('ready_for_archive_preflight') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool(report.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Archive Pair",
        "",
        f"- source: `{source.get('path')}` / `{source.get('bytes')}` bytes / `{source.get('sha256')}`",
        f"- candidate: `{candidate.get('path')}` / `{candidate.get('bytes')}` bytes / `{candidate.get('sha256')}`",
        f"- payload_byte_delta: `{byte_accounting.get('payload_byte_delta')}`",
        f"- archive_byte_delta: `{byte_accounting.get('archive_byte_delta')}`",
        "",
        "## Runtime Constants To Derive",
        "",
    ]
    for key, value in constants.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Section Diffs", ""])
    changed = False
    for row in report.get("section_diffs") or []:
        item = _mapping(row)
        if item.get("changed") is True:
            changed = True
            lines.append(
                "- `{name}`: `{source}` -> `{candidate}` (`{delta}`)".format(
                    name=item.get("name"),
                    source=item.get("source_bytes"),
                    candidate=item.get("candidate_bytes"),
                    delta=item.get("byte_delta"),
                )
            )
    if not changed:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("readiness_blockers") or []:
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _load_manifest(
    source: str | Path | Mapping[str, Any],
    repo: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, Mapping):
        return dict(source), {"provided_inline": True, "score_claim": False}
    path = Path(source)
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise Pr103ArithmeticTransformPlanError("schema manifest JSON must be an object")
    return dict(payload), {
        "provided_inline": False,
        "path": repo_relative(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _select_target(
    manifest: Mapping[str, Any],
    selector: TransformTargetSelection,
) -> dict[str, Any]:
    rows = list(_target_rows(manifest))
    if not rows:
        raise Pr103ArithmeticTransformPlanError(
            "schema manifest has no next_arithmetic_schema_targets"
        )
    if selector.label:
        matches = [row for row in rows if str(row.get("label") or "") == selector.label]
        if not matches:
            raise Pr103ArithmeticTransformPlanError(
                f"target label not found in schema manifest: {selector.label}"
            )
        return dict(matches[0])
    if selector.rank is not None:
        if selector.rank < 1 or selector.rank > len(rows):
            raise Pr103ArithmeticTransformPlanError(
                f"target rank out of range: {selector.rank} not in [1,{len(rows)}]"
            )
        return dict(rows[selector.rank - 1])
    return dict(rows[0])


def _target_rows(manifest: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = manifest.get("next_arithmetic_schema_targets")
    if rows is None:
        rows = _mapping(manifest.get("merged_arithmetic_stream")).get("stream_gap_ranking")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _resolve_source_archive(
    source_archive: str | Path | None,
    manifest: Mapping[str, Any],
    repo: Path,
) -> Path:
    if source_archive is not None:
        return Path(source_archive)
    source = _mapping(manifest.get("source_archive"))
    raw_path = str(source.get("path") or "")
    if not raw_path:
        raise Pr103ArithmeticTransformPlanError(
            "source archive path missing; pass --source-archive"
        )
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return repo / candidate


def _load_pr103_retarget_context(
    *,
    schema_manifest: str | Path | Mapping[str, Any],
    source_archive: str | Path | None,
    target_label: str | None,
    target_rank: int | None,
    repo_root: str | Path | None,
    layout: Pr103LcAcLayout,
    stream_specs: Sequence[tuple[str, int, int | None]],
    hi_symbol_count: int,
) -> dict[str, Any]:
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest, manifest_record = _load_manifest(schema_manifest, repo)
    selector = TransformTargetSelection(label=target_label, rank=target_rank)
    target = _select_target(manifest, selector)
    archive_path = _resolve_source_archive(source_archive, manifest, repo)
    source = read_strict_single_member_zip(archive_path)
    parsed = parse_pr103_lc_ac_payload(source.payload, layout=layout)
    auxiliary = decode_pr103_auxiliary_models(parsed, ac_stream_count=len(stream_specs))
    histograms = np.asarray(auxiliary.pop("_histograms_array"), dtype=np.uint8)
    hi_histogram = np.asarray(auxiliary.pop("_hi_histogram_array"))
    source_merged = parsed.section_bytes("merged_range_coded_weights_and_hi_latents")
    source_histogram_blob = parsed.section_bytes("ac_histograms_brotli")
    decoded = _decode_pr103_merged_ac_symbol_streams(
        source_merged,
        histograms,
        hi_histogram,
        stream_specs=stream_specs,
        hi_symbol_count=hi_symbol_count,
    )
    stream_records = list(decoded["stream_records"])
    target_index = _find_decoded_stream_index(stream_records, str(target.get("label") or ""))
    source_symbol_streams = [np.asarray(item).copy() for item in decoded["symbol_streams"]]
    source_model_weights = [np.asarray(item).copy() for item in decoded["model_weights"]]
    source_roundtrip = encode_pr103_merged_ac_stream(source_symbol_streams, source_model_weights)
    target_record = dict(stream_records[target_index])
    expected_symbol_sha = str(target.get("decoded_symbols_sha256") or "")
    observed_symbol_sha = str(target_record.get("decoded_symbols_sha256") or "")
    return {
        "repo": repo,
        "manifest": manifest,
        "manifest_record": manifest_record,
        "target": target,
        "archive_path": archive_path,
        "source": source,
        "histograms": histograms,
        "source_merged": source_merged,
        "source_histogram_blob": source_histogram_blob,
        "source_symbol_streams": source_symbol_streams,
        "source_model_weights": source_model_weights,
        "source_roundtrip": source_roundtrip,
        "source_roundtrip_identical": source_roundtrip == source_merged,
        "decoder_maybe_exhausted": decoded.get("decoder_maybe_exhausted") is True,
        "target_index": target_index,
        "target_record": target_record,
        "expected_symbol_sha": expected_symbol_sha,
        "observed_symbol_sha": observed_symbol_sha,
        "manifest_blockers": _schema_manifest_blockers(
            manifest,
            _mapping(manifest.get("merged_arithmetic_stream")),
        ),
    }


def _decode_pr103_merged_ac_symbol_streams(
    merged_ac: bytes,
    histograms: np.ndarray,
    hi_histogram: np.ndarray,
    *,
    stream_specs: Sequence[tuple[str, int, int | None]],
    hi_symbol_count: int,
) -> dict[str, Any]:
    """Decode merged AC symbols for local probes without altering schema API."""

    _require_constriction()
    if len(merged_ac) == 0 or len(merged_ac) % 4:
        raise Pr103ArithmeticTransformPlanError(
            "merged_range_coded_weights_and_hi_latents must be non-empty uint32 words"
        )
    hist = np.asarray(histograms)
    if hist.ndim != 2 or hist.shape[0] != len(stream_specs):
        raise Pr103ArithmeticTransformPlanError(
            "histogram rows must match AC stream specs: "
            f"rows={hist.shape[0] if hist.ndim == 2 else 'bad'} specs={len(stream_specs)}"
        )
    hi_hist = np.asarray(hi_histogram)
    if hi_hist.ndim != 1 or hi_hist.size == 0:
        raise Pr103ArithmeticTransformPlanError(
            "latent hi histogram must be a non-empty 1D array"
        )

    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(merged_ac, dtype="<u4"))
    symbol_streams: list[np.ndarray] = []
    model_weights: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for row_index, (label, count, schema_index) in enumerate(stream_specs):
        symbols = _decode_symbols(decoder, hist[row_index], int(count))
        symbol_streams.append(symbols)
        model_weights.append(np.asarray(hist[row_index]).copy())
        records.append(_decoded_record(label, "ac_weight_tensor", schema_index, symbols))
    hi_symbols = _decode_symbols(decoder, hi_hist, int(hi_symbol_count))
    symbol_streams.append(hi_symbols)
    model_weights.append(np.asarray(hi_hist).copy())
    records.append(_decoded_record("latent_hi_bytes", "latent_hi_stream", None, hi_symbols))
    return {
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "decoder_maybe_exhausted": bool(decoder.maybe_exhausted()),
        "stream_records": tuple(records),
        "symbol_streams": tuple(symbol_streams),
        "model_weights": tuple(model_weights),
    }


def _decode_symbols(decoder: Any, weights: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise Pr103ArithmeticTransformPlanError(f"negative decode count: {count}")
    cat = _categorical_from_weights(weights)
    out = np.empty(int(count), dtype=np.int32)
    for index in range(int(count)):
        out[index] = decoder.decode(cat)
    return out


def _categorical_from_weights(weights: np.ndarray) -> Any:
    _require_constriction()
    probs = np.asarray(weights, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise Pr103ArithmeticTransformPlanError("categorical weights must be non-empty")
    if np.any(probs < 0) or not np.all(np.isfinite(probs)):
        raise Pr103ArithmeticTransformPlanError(
            "categorical weights must be finite and non-negative"
        )
    probs = np.maximum(probs, 1e-10)
    probs = probs / probs.sum()
    return constriction.stream.model.Categorical(probs, perfect=False)


def _decoded_record(
    label: str,
    role: str,
    schema_index: int | None,
    symbols: np.ndarray,
) -> dict[str, Any]:
    return {
        "label": label,
        "role": role,
        "schema_index": schema_index,
        "symbol_count": int(symbols.size),
        "decoded_symbols_sha256": sha256_bytes(symbols.astype(np.uint16).tobytes()),
        "score_claim": False,
        "dispatch_attempted": False,
    }


def _require_constriction() -> None:
    if constriction is None:  # pragma: no cover
        raise Pr103ArithmeticTransformPlanError("constriction_missing_for_pr103_ac_probe")


def _find_decoded_stream_index(
    records: Sequence[Mapping[str, Any]],
    label: str,
) -> int:
    for index, row in enumerate(records):
        if row.get("label") == label:
            return index
    raise Pr103ArithmeticTransformPlanError(f"decoded stream label not found: {label}")


def _retarget_weights_for_symbols(
    symbols: np.ndarray,
    old_weights: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        raise Pr103ArithmeticTransformPlanError("cannot retarget empty symbol stream")
    alphabet_size = int(np.asarray(old_weights).reshape(-1).size)
    if int(flat.min()) < 0 or int(flat.max()) >= alphabet_size:
        raise Pr103ArithmeticTransformPlanError(
            "target stream symbols fall outside source model alphabet"
        )
    if alphabet_size != 256:
        raise Pr103ArithmeticTransformPlanError(
            f"only q8 AC tensor histograms are supported, got alphabet={alphabet_size}"
        )
    counts = np.bincount(flat, minlength=alphabet_size).astype(np.float64)
    if counts.max() <= 0:
        raise Pr103ArithmeticTransformPlanError("cannot retarget zero-count histogram")
    if counts.max() <= 255:
        weights = np.maximum(counts, 1.0)
    else:
        scale = 255.0 / float(counts.max())
        scaled = np.round(counts * scale)
        weights = np.where(counts > 0, np.maximum(scaled, 1.0), scaled)
        weights = np.where((counts > 0) & (weights == 0), 1.0, weights)
    return np.clip(weights, 0, 255).astype(np.uint8)


def _load_beam_probe_reports(
    sources: Sequence[str | Path | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for source in sources:
        if isinstance(source, Mapping):
            report = dict(source)
        else:
            payload = read_json(Path(source))
            if not isinstance(payload, Mapping):
                raise Pr103ArithmeticTransformPlanError(
                    f"beam probe report must be a JSON object: {source}"
                )
            report = dict(payload)
        if report.get("schema") != BEAM_SCHEMA:
            raise Pr103ArithmeticTransformPlanError(
                f"expected beam report schema {BEAM_SCHEMA}, got {report.get('schema')!r}"
            )
        if report.get("score_claim") is True or report.get("dispatch_attempted") is True:
            raise Pr103ArithmeticTransformPlanError(
                "beam reports must be local no-score probes"
            )
        reports.append(report)
    return reports


def _validate_candidate_report_source(
    report: Mapping[str, Any],
    source: Any,
) -> None:
    source_record = _mapping(report.get("source_archive"))
    checks = {
        "archive_sha256": (
            str(source_record.get("sha256") or ""),
            str(source.archive_sha256),
        ),
        "archive_bytes": (
            int(source_record.get("bytes") or -1),
            int(source.archive_bytes),
        ),
        "member_name": (
            str(source_record.get("member_name") or ""),
            str(source.member_name),
        ),
        "member_bytes": (
            int(source_record.get("member_bytes") or -1),
            int(source.member_bytes),
        ),
        "member_sha256": (
            str(source_record.get("member_sha256") or ""),
            sha256_bytes(source.payload),
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise Pr103ArithmeticTransformPlanError(
                f"beam report source {label} mismatch: report={observed!r} source={expected!r}"
            )


def _candidate_section_diffs(
    *,
    parsed: Any,
    candidate_parsed: Any,
) -> list[dict[str, Any]]:
    source_rows = {str(row["name"]): row for row in parsed.section_records()}
    candidate_rows = {str(row["name"]): row for row in candidate_parsed.section_records()}
    names = sorted(set(source_rows) | set(candidate_rows))
    diffs: list[dict[str, Any]] = []
    for name in names:
        source = _mapping(source_rows.get(name))
        candidate = _mapping(candidate_rows.get(name))
        source_bytes = int(source.get("bytes") or 0)
        candidate_bytes = int(candidate.get("bytes") or 0)
        source_sha = str(source.get("sha256") or "")
        candidate_sha = str(candidate.get("sha256") or "")
        diffs.append(
            {
                "name": name,
                "source_bytes": source_bytes,
                "candidate_bytes": candidate_bytes,
                "byte_delta": candidate_bytes - source_bytes,
                "source_sha256": source_sha,
                "candidate_sha256": candidate_sha,
                "changed": source_sha != candidate_sha or source_bytes != candidate_bytes,
            }
        )
    return diffs


def _runtime_constants_from_layout(layout: Pr103LcAcLayout) -> dict[str, Any]:
    return {
        "SCA_LEN": int(layout.scales_fp16),
        "BR_LEN": int(layout.non_ac_weights_brotli),
        "HIST_LEN": int(layout.ac_histograms_brotli),
        "MERGED_AC_LEN": int(layout.merged_range_coded_weights_and_hi_latents),
        "LATENT_META_LEN": int(layout.latent_min_scale_fp16),
        "LO_LEN": int(layout.latent_low_bytes_brotli),
        "HI_HIST_LEN": int(layout.latent_hi_histogram_brotli),
        "sidecar_corrections_brotli": "payload_tail",
        "fixed_bytes_before_tail": int(layout.fixed_bytes),
    }


def _score_histogram_candidate(
    *,
    weights: np.ndarray,
    histograms: np.ndarray,
    target_index: int,
    source_symbol_streams: Sequence[np.ndarray],
    source_model_weights: Sequence[np.ndarray],
    source_merged: bytes,
    source_histogram_blob: bytes,
    moves: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    changed_weights = np.asarray(weights, dtype=np.uint8).copy()
    model_weights = [np.asarray(item).copy() for item in source_model_weights]
    model_weights[target_index] = changed_weights
    merged = encode_pr103_merged_ac_stream(source_symbol_streams, model_weights)
    changed_histograms = np.asarray(histograms, dtype=np.uint8).copy()
    changed_histograms[target_index] = changed_weights
    changed_histogram_raw = changed_histograms.astype(np.uint8).tobytes()
    changed_histogram_blob = brotli.compress(changed_histogram_raw, quality=11)
    merged_delta = len(merged) - len(source_merged)
    histogram_delta = len(changed_histogram_blob) - len(source_histogram_blob)
    total_delta = merged_delta + histogram_delta
    move_rows = [dict(move) for move in moves]
    row: dict[str, Any] = {
        "moves": move_rows,
        "change_count": len(move_rows),
        "merged_ac_bytes": len(merged),
        "merged_ac_sha256": sha256_bytes(merged),
        "merged_ac_delta": merged_delta,
        "histogram_brotli_bytes": len(changed_histogram_blob),
        "histogram_brotli_sha256": sha256_bytes(changed_histogram_blob),
        "histogram_brotli_delta": histogram_delta,
        "histogram_raw_sha256": sha256_bytes(changed_histogram_raw),
        "estimated_member_delta_if_runtime_adapter_supported": total_delta,
        "estimated_rate_score_delta_if_components_unchanged": (
            float(total_delta) * RATE_SCORE_PER_BYTE
        ),
        "no_op": merged == source_merged
        and changed_histogram_raw == np.asarray(histograms, dtype=np.uint8).tobytes(),
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "_weights": changed_weights,
    }
    if move_rows:
        last = move_rows[-1]
        for key in ("symbol", "symbol_count", "delta", "old_weight", "new_weight"):
            row[key] = last.get(key)
    return row


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    moves = row.get("moves")
    return (
        int(row.get("estimated_member_delta_if_runtime_adapter_supported") or 0),
        int(row.get("merged_ac_delta") or 0),
        int(row.get("histogram_brotli_delta") or 0),
        len(moves) if isinstance(moves, Sequence) else 0,
        str(row.get("histogram_raw_sha256") or ""),
    )


def _public_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if not str(key).startswith("_")}


def _schema_manifest_blockers(
    manifest: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if manifest.get("ready_for_schema_review") is not True:
        blockers.append("source_schema_manifest_not_ready_for_schema_review")
    if merged.get("reencoded_byte_identical") is not True:
        blockers.append("source_merged_stream_reencode_not_byte_identical")
    if merged.get("decoder_maybe_exhausted") is not True:
        blockers.append("source_merged_range_decoder_not_exhausted")
    if manifest.get("planning_only") is not True:
        blockers.append("source_schema_manifest_planning_only_flag_missing")
    if manifest.get("score_claim") is True:
        blockers.append("source_schema_manifest_must_not_claim_score")
    return blockers


def _target_record(target: Mapping[str, Any]) -> dict[str, Any]:
    keep = [
        "label",
        "role",
        "schema_index",
        "symbol_count",
        "alphabet_size",
        "decoded_symbols_sha256",
        "observed_entropy_bits_per_symbol",
        "model_cross_entropy_bits_per_symbol",
        "observed_entropy_bytes_floor",
        "model_cross_entropy_bytes_floor",
        "model_gap_bytes_estimate",
        "required_next_artifact",
    ]
    return {key: target.get(key) for key in keep if key in target}


def _proposal_id(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    strategy: str,
) -> str:
    seed = "|".join(
        [
            str(source.get("sha256") or ""),
            str(source.get("member_sha256") or ""),
            str(target.get("label") or ""),
            str(target.get("decoded_symbols_sha256") or ""),
            strategy,
        ]
    ).encode("utf-8")
    return "pr103_ac_plan_" + sha256_bytes(seed)[:16]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unique_ordered(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))


def _bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "DEFAULT_STRATEGY",
    "COORDINATE_SCHEMA",
    "BEAM_SCHEMA",
    "CANDIDATE_SCHEMA",
    "PLAN_SCHEMA",
    "RETARGET_SCHEMA",
    "Pr103ArithmeticTransformPlanError",
    "TransformTargetSelection",
    "build_pr103_arithmetic_histogram_beam_probe",
    "build_pr103_arithmetic_histogram_coordinate_probe",
    "build_pr103_arithmetic_retarget_probe",
    "build_pr103_arithmetic_transform_plan",
    "materialize_pr103_arithmetic_histogram_candidate",
    "render_beam_markdown",
    "render_candidate_markdown",
    "render_coordinate_markdown",
    "render_markdown",
    "render_retarget_markdown",
]
