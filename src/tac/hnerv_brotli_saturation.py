"""Bounded Brotli saturation audits for packed HNeRV decoder sections.

This module is a local byte-custody profiler. It recodes the exact
``decoder_packed_brotli`` section from a strict single-member HNeRV archive
across a deterministic Brotli parameter grid and reports whether any recode
beats the charged current section bytes. It does not build score claims,
dispatch GPU work, or authorize exact eval.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
)

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_brotli_saturation.build_hnerv_decoder_brotli_saturation_audit"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
SECTION_NAME = "decoder_packed_brotli"

DEFAULT_QUALITIES = tuple(range(12))
DEFAULT_LGWINS: tuple[int | None, ...] = (None, *range(10, 25))
DEFAULT_LGBLOCKS: tuple[int | None, ...] = (None,)
DEFAULT_MODES = ("generic", "text", "font")

MODE_VALUES = {
    "generic": brotli.MODE_GENERIC,
    "text": brotli.MODE_TEXT,
    "font": brotli.MODE_FONT,
}
MODE_SORT_ORDER = {"generic": 0, "text": 1, "font": 2}

DISPATCH_BLOCKERS = [
    "brotli_saturation_audit_is_local_planning_only",
    "requires_rate_positive_byte_different_archive_before_exact_eval",
    "requires_archive_manifest_preflight",
    "requires_lane_dispatch_claim_before_gpu",
    "requires_exact_cuda_auth_eval",
]


class HnervBrotliSaturationError(ValueError):
    """Raised when a Brotli saturation audit input is malformed."""


@dataclasses.dataclass(frozen=True)
class BrotliGridParams:
    """A normalized Brotli recode parameter tuple."""

    quality: int
    lgwin: int | None
    lgblock: int | None
    mode: str


def build_hnerv_decoder_brotli_saturation_audit(
    *,
    source_archive: str | Path,
    source_label: str,
    scorecard: Mapping[str, Any] | None = None,
    entropy_ranking: Mapping[str, Any] | None = None,
    qualities: Iterable[int] = DEFAULT_QUALITIES,
    lgwins: Iterable[int | None] = DEFAULT_LGWINS,
    lgblocks: Iterable[int | None] = DEFAULT_LGBLOCKS,
    modes: Iterable[str] = DEFAULT_MODES,
    jobs: int = 1,
) -> dict[str, Any]:
    """Audit whether a deterministic Brotli grid beats the current decoder bytes."""

    archive = read_strict_single_member_zip(source_archive)
    packed = parse_ff_packed_brotli_hnerv(archive.payload)
    source_section = packed.decoder_packed_brotli
    try:
        raw_decoder = brotli.decompress(source_section)
    except brotli.error as exc:
        raise HnervBrotliSaturationError("decoder_packed_brotli is not Brotli-decompressible") from exc

    params = _parameter_grid(
        qualities=qualities,
        lgwins=lgwins,
        lgblocks=lgblocks,
        modes=modes,
    )
    attempts = _run_grid(source_section, raw_decoder, params, jobs=jobs)
    best = min(attempts, key=_attempt_sort_key)
    scorecard_anchor = _scorecard_anchor(
        scorecard=scorecard,
        source_label=source_label,
        archive=archive,
        payload_sha256=sha256_bytes(archive.payload),
        source_section=source_section,
    )
    entropy_anchor = _entropy_ranking_anchor(
        entropy_ranking=entropy_ranking,
        source_label=source_label,
        source_section=source_section,
    )
    blockers = [
        *scorecard_anchor.get("blockers", []),
        *entropy_anchor.get("blockers", []),
    ]
    rate_positive_attempts = [
        row for row in attempts if int(row["byte_delta_vs_source_section"]) < 0
    ]
    same_size_changed_attempts = [
        row
        for row in attempts
        if int(row["candidate_bytes"]) == len(source_section)
        and row["candidate_section_sha256"] != sha256_bytes(source_section)
    ]
    minimum_section_bytes_to_beat = len(source_section) - 1
    if rate_positive_attempts:
        verdict = "rate_positive_brotli_grid_candidate_found"
    elif same_size_changed_attempts:
        verdict = "no_rate_positive_brotli_grid_candidate_same_size_variants_only"
    else:
        verdict = "bounded_brotli_grid_saturated"

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "source_label": source_label,
        "source_archive_path": str(Path(source_archive)),
        "source_archive_sha256": archive.archive_sha256,
        "source_archive_bytes": archive.archive_bytes,
        "source_member_name": archive.member_name,
        "source_member_bytes": archive.member_bytes,
        "source_payload_sha256": sha256_bytes(archive.payload),
        "source_payload_bytes": len(archive.payload),
        "source_decoder_section_name": SECTION_NAME,
        "source_decoder_section_bytes": len(source_section),
        "source_decoder_section_sha256": sha256_bytes(source_section),
        "source_decoder_section_entropy_bits_per_byte": _byte_entropy(source_section),
        "source_decoder_raw_bytes": len(raw_decoder),
        "source_decoder_raw_sha256": sha256_bytes(raw_decoder),
        "source_decoder_raw_entropy_bits_per_byte": _byte_entropy(raw_decoder),
        "scorecard_anchor": scorecard_anchor,
        "entropy_ranking_anchor": entropy_anchor,
        "grid": {
            "qualities": [param.quality for param in _unique_quality_params(params)],
            "lgwins": _sorted_optional_ints(param.lgwin for param in params),
            "lgblocks": _sorted_optional_ints(param.lgblock for param in params),
            "modes": sorted({param.mode for param in params}),
            "attempt_count": len(attempts),
            "jobs": _bounded_jobs(jobs, len(params)),
            "lgblock_scope": (
                "default_only"
                if {param.lgblock for param in params} == {None}
                else "explicit_grid"
            ),
        },
        "best_attempt": best,
        "rate_positive_attempt_count": len(rate_positive_attempts),
        "same_size_byte_different_attempt_count": len(same_size_changed_attempts),
        "proof_summary": {
            "verdict": verdict,
            "charged_source_archive_bytes": archive.archive_bytes,
            "source_decoder_section_bytes": len(source_section),
            "minimum_section_bytes_to_beat": minimum_section_bytes_to_beat,
            "best_grid_section_bytes": best["candidate_bytes"],
            "best_grid_section_sha256": best["candidate_section_sha256"],
            "best_grid_archive_byte_delta_if_swapped": best["byte_delta_vs_source_section"],
            "rate_score_delta_if_best_swapped_and_components_equal": best[
                "rate_score_delta_if_components_equal"
            ],
            "bytes_short_of_rate_positive": max(
                0, int(best["candidate_bytes"]) - minimum_section_bytes_to_beat
            ),
            "best_attempt_is_source_section": (
                best["candidate_section_sha256"] == sha256_bytes(source_section)
            ),
            "no_candidate_reason": (
                "no_grid_attempt_beats_current_charged_decoder_section"
                if not rate_positive_attempts
                else ""
            ),
        },
        "audit_blockers": blockers,
        "attempts": attempts,
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact markdown report for a saturation audit."""

    proof = _as_mapping(manifest.get("proof_summary"), "proof_summary")
    best = _as_mapping(manifest.get("best_attempt"), "best_attempt")
    grid = _as_mapping(manifest.get("grid"), "grid")
    entropy_anchor = _as_mapping(manifest.get("entropy_ranking_anchor"), "entropy_ranking_anchor")
    lines = [
        "# HNeRV Decoder Brotli Saturation Audit",
        "",
        f"- planning_only: `{_bool_text(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool_text(manifest.get('dispatch_attempted') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Source",
        "",
        f"- label: `{manifest.get('source_label')}`",
        f"- archive_bytes: `{manifest.get('source_archive_bytes')}`",
        f"- archive_sha256: `{manifest.get('source_archive_sha256')}`",
        f"- decoder_section_bytes: `{manifest.get('source_decoder_section_bytes')}`",
        f"- decoder_section_sha256: `{manifest.get('source_decoder_section_sha256')}`",
        f"- decoder_section_entropy_bpb: `{manifest.get('source_decoder_section_entropy_bits_per_byte')}`",
        "",
        "## Grid",
        "",
        f"- attempts: `{grid.get('attempt_count')}`",
        f"- qualities: `{grid.get('qualities')}`",
        f"- lgwins: `{grid.get('lgwins')}`",
        f"- lgblocks: `{grid.get('lgblocks')}`",
        f"- modes: `{grid.get('modes')}`",
        "",
        "## Verdict",
        "",
        f"- verdict: `{proof.get('verdict')}`",
        f"- minimum_section_bytes_to_beat: `{proof.get('minimum_section_bytes_to_beat')}`",
        f"- best_grid_section_bytes: `{proof.get('best_grid_section_bytes')}`",
        f"- bytes_short_of_rate_positive: `{proof.get('bytes_short_of_rate_positive')}`",
        f"- best_grid_archive_byte_delta_if_swapped: `{proof.get('best_grid_archive_byte_delta_if_swapped')}`",
        f"- rate_positive_attempt_count: `{manifest.get('rate_positive_attempt_count')}`",
        f"- same_size_byte_different_attempt_count: `{manifest.get('same_size_byte_different_attempt_count')}`",
        "",
        "## Best Attempt",
        "",
        "| mode | quality | lgwin | lgblock | bytes | delta | sha256 |",
        "|---|---:|---:|---:|---:|---:|---|",
        "| {mode} | {quality} | {lgwin} | {lgblock} | {bytes_} | {delta} | `{sha}` |".format(
            mode=best.get("mode"),
            quality=best.get("quality"),
            lgwin=_display_optional(best.get("lgwin")),
            lgblock=_display_optional(best.get("lgblock")),
            bytes_=best.get("candidate_bytes"),
            delta=best.get("byte_delta_vs_source_section"),
            sha=best.get("candidate_section_sha256"),
        ),
    ]
    if entropy_anchor:
        lines.extend(
            [
                "",
                "## Entropy Ranking Anchor",
                "",
                f"- current_frontier_label: `{entropy_anchor.get('current_frontier_label')}`",
                f"- next_target_section: `{entropy_anchor.get('next_target_section')}`",
                f"- ranking_minimum_section_bytes_to_beat: `{entropy_anchor.get('minimum_section_bytes_to_beat')}`",
                f"- top_byte_mass_section_bytes: `{entropy_anchor.get('top_byte_mass_section_bytes')}`",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a bounded local Brotli parameter-grid proof. It is",
            "not an archive preflight result, not a score claim, and not dispatch",
            "authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_grid(
    source_section: bytes,
    raw_decoder: bytes,
    params: Sequence[BrotliGridParams],
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    max_workers = _bounded_jobs(jobs, len(params))
    if max_workers == 1:
        rows = [_brotli_attempt(source_section, raw_decoder, param) for param in params]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_brotli_attempt, source_section, raw_decoder, param)
                for param in params
            ]
            rows = [future.result() for future in futures]
    return sorted(rows, key=_attempt_sort_key)


def _brotli_attempt(
    source_section: bytes,
    raw_decoder: bytes,
    param: BrotliGridParams,
) -> dict[str, Any]:
    kwargs: dict[str, int] = {
        "mode": MODE_VALUES[param.mode],
        "quality": param.quality,
    }
    if param.lgwin is not None:
        kwargs["lgwin"] = param.lgwin
    if param.lgblock is not None:
        kwargs["lgblock"] = param.lgblock
    candidate = brotli.compress(raw_decoder, **kwargs)
    raw_equal = brotli.decompress(candidate) == raw_decoder
    delta = len(candidate) - len(source_section)
    return {
        "mode": param.mode,
        "quality": param.quality,
        "lgwin": param.lgwin,
        "lgblock": param.lgblock,
        "candidate_bytes": len(candidate),
        "source_bytes": len(source_section),
        "byte_delta_vs_source_section": delta,
        "rate_score_delta_if_components_equal": round(delta * RATE_SCORE_PER_BYTE, 12),
        "candidate_section_sha256": sha256_bytes(candidate),
        "changed": candidate != source_section,
        "raw_equal": raw_equal,
        "rate_positive": delta < 0,
    }


def _parameter_grid(
    *,
    qualities: Iterable[int],
    lgwins: Iterable[int | None],
    lgblocks: Iterable[int | None],
    modes: Iterable[str],
) -> tuple[BrotliGridParams, ...]:
    params: list[BrotliGridParams] = []
    for mode in modes:
        normalized_mode = str(mode).lower()
        if normalized_mode not in MODE_VALUES:
            raise HnervBrotliSaturationError(f"unsupported Brotli mode: {mode}")
        for quality in qualities:
            q = int(quality)
            if not 0 <= q <= 11:
                raise HnervBrotliSaturationError(f"Brotli quality out of range: {q}")
            for lgwin in lgwins:
                normalized_lgwin = _normalize_lgwin(lgwin)
                for lgblock in lgblocks:
                    normalized_lgblock = _normalize_lgblock(lgblock)
                    params.append(
                        BrotliGridParams(
                            quality=q,
                            lgwin=normalized_lgwin,
                            lgblock=normalized_lgblock,
                            mode=normalized_mode,
                        )
                    )
    deduped = sorted(set(params), key=_param_sort_key)
    if not deduped:
        raise HnervBrotliSaturationError("Brotli parameter grid is empty")
    return tuple(deduped)


def _normalize_lgwin(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if not 10 <= normalized <= 24:
        raise HnervBrotliSaturationError(f"Brotli lgwin out of range: {normalized}")
    return normalized


def _normalize_lgblock(value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized == 0:
        return None
    if not 16 <= normalized <= 24:
        raise HnervBrotliSaturationError(f"Brotli lgblock out of range: {normalized}")
    return normalized


def _scorecard_anchor(
    *,
    scorecard: Mapping[str, Any] | None,
    source_label: str,
    archive: Any,
    payload_sha256: str,
    source_section: bytes,
) -> dict[str, Any]:
    blockers: list[str] = []
    if scorecard is None:
        return {"matched": False, "blockers": ["missing_scorecard_anchor"]}
    row = _row_by_label(scorecard, source_label)
    if row is None:
        return {"matched": False, "blockers": [f"missing_scorecard_row:{source_label}"]}
    section = _section_by_name(row, SECTION_NAME)
    if section is None:
        blockers.append(f"missing_scorecard_section:{SECTION_NAME}")
    section_sha = sha256_bytes(source_section)
    section_bytes = len(source_section)
    checks = {
        "archive_sha256_match": row.get("archive_sha256") == archive.archive_sha256,
        "archive_bytes_match": row.get("archive_bytes") == archive.archive_bytes,
        "payload_sha256_match": row.get("payload_sha256") == payload_sha256,
        "section_sha256_match": section.get("sha256") == section_sha if section else False,
        "section_bytes_match": section.get("bytes") == section_bytes if section else False,
    }
    for name, ok in checks.items():
        if not ok:
            blockers.append(f"scorecard_{name}_failed")
    return {
        "matched": not blockers,
        "label": source_label,
        "score": row.get("score"),
        "archive_sha256": row.get("archive_sha256"),
        "archive_bytes": row.get("archive_bytes"),
        "payload_sha256": row.get("payload_sha256"),
        "frontier_scope": row.get("frontier_scope"),
        "evidence_grade": row.get("evidence_grade"),
        "canonical_frontier_eligible": row.get("canonical_frontier_eligible"),
        "section_name": SECTION_NAME,
        "section_bytes": section.get("bytes") if section else None,
        "section_sha256": section.get("sha256") if section else None,
        "section_entropy_bits_per_byte": (
            section.get("entropy_bits_per_byte") if section else None
        ),
        "checks": checks,
        "blockers": blockers,
    }


def _entropy_ranking_anchor(
    *,
    entropy_ranking: Mapping[str, Any] | None,
    source_label: str,
    source_section: bytes,
) -> dict[str, Any]:
    blockers: list[str] = []
    if entropy_ranking is None:
        return {"matched": False, "blockers": ["missing_entropy_ranking_anchor"]}
    frontier = entropy_ranking.get("current_frontier")
    if not isinstance(frontier, Mapping):
        return {"matched": False, "blockers": ["entropy_ranking_missing_current_frontier"]}
    if frontier.get("label") != source_label:
        blockers.append("entropy_ranking_current_frontier_label_mismatch")
    top = None
    byte_mass = entropy_ranking.get("frontier_byte_mass_ranking")
    if isinstance(byte_mass, Sequence) and not isinstance(byte_mass, (str, bytes, bytearray)):
        for row in byte_mass:
            if isinstance(row, Mapping) and row.get("section") == SECTION_NAME:
                top = row
                break
    if top is None:
        blockers.append(f"entropy_ranking_missing_byte_mass_section:{SECTION_NAME}")
    section_sha = sha256_bytes(source_section)
    if top is not None and top.get("section_sha256") != section_sha:
        blockers.append("entropy_ranking_decoder_section_sha256_mismatch")
    if top is not None and top.get("section_bytes") != len(source_section):
        blockers.append("entropy_ranking_decoder_section_bytes_mismatch")
    next_action = entropy_ranking.get("next_entropy_research_action")
    if not isinstance(next_action, Mapping):
        next_action = {}
    return {
        "matched": not blockers,
        "current_frontier_label": frontier.get("label"),
        "current_frontier_archive_sha256": frontier.get("archive_sha256"),
        "next_action_id": next_action.get("action_id"),
        "next_target_label": next_action.get("target_label"),
        "next_target_section": next_action.get("target_section"),
        "minimum_section_bytes_to_beat": next_action.get(
            "minimum_section_bytes_to_beat", len(source_section) - 1
        ),
        "top_byte_mass_section_bytes": top.get("section_bytes") if top else None,
        "top_byte_mass_section_sha256": top.get("section_sha256") if top else None,
        "top_byte_mass_entropy_bits_per_byte": (
            top.get("entropy_bits_per_byte") if top else None
        ),
        "blockers": blockers,
    }


def _row_by_label(scorecard: Mapping[str, Any], label: str) -> Mapping[str, Any] | None:
    rows = scorecard.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise HnervBrotliSaturationError("scorecard rows must be a list")
    for row in rows:
        if isinstance(row, Mapping) and row.get("label") == label:
            return row
    return None


def _section_by_name(row: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    sections = row.get("payload_sections") or row.get("top_payload_sections")
    if not isinstance(sections, Sequence) or isinstance(sections, (str, bytes, bytearray)):
        return None
    for section in sections:
        if isinstance(section, Mapping) and section.get("name") == name:
            return section
    return None


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(float(entropy), 12)


def _bounded_jobs(jobs: int, attempt_count: int) -> int:
    if jobs < 1:
        raise HnervBrotliSaturationError(f"jobs must be >= 1, got {jobs}")
    return max(1, min(jobs, attempt_count, os.cpu_count() or 1))


def _attempt_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, str, str]:
    return (
        int(row["candidate_bytes"]),
        0 if row["changed"] is False else 1,
        int(row["quality"]),
        _optional_sort_int(row.get("lgwin")),
        _optional_sort_int(row.get("lgblock")),
        f"{MODE_SORT_ORDER.get(str(row.get('mode')), 99):02d}:{row.get('mode')}",
        str(row.get("candidate_section_sha256")),
    )


def _param_sort_key(param: BrotliGridParams) -> tuple[int, int, int, str]:
    return (
        param.quality,
        _optional_sort_int(param.lgwin),
        _optional_sort_int(param.lgblock),
        f"{MODE_SORT_ORDER.get(param.mode, 99):02d}:{param.mode}",
    )


def _optional_sort_int(value: Any) -> int:
    return -1 if value is None else int(value)


def _unique_quality_params(params: Sequence[BrotliGridParams]) -> list[BrotliGridParams]:
    return sorted(
        {BrotliGridParams(param.quality, None, None, "generic") for param in params},
        key=lambda param: param.quality,
    )


def _sorted_optional_ints(values: Iterable[int | None]) -> list[int | None]:
    return sorted(set(values), key=_optional_sort_int)


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    raise HnervLowlevelPackError(f"{context} must be an object")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _display_optional(value: Any) -> str:
    return "default" if value is None else str(value)


__all__ = [
    "HnervBrotliSaturationError",
    "build_hnerv_decoder_brotli_saturation_audit",
    "render_markdown",
]
