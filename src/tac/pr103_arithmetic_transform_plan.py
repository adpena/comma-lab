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

from tac.hnerv_pr103_lc_ac_schema import RATE_SCORE_PER_BYTE
from tac.repo_io import read_json, repo_relative, sha256_bytes, sha256_file


PLAN_SCHEMA = "pr103_arithmetic_transform_plan_v1"
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
    "PLAN_SCHEMA",
    "Pr103ArithmeticTransformPlanError",
    "TransformTargetSelection",
    "build_pr103_arithmetic_transform_plan",
    "render_markdown",
]
