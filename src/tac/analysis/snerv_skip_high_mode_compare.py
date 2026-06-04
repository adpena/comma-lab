# SPDX-License-Identifier: MIT
"""Compare SNeRV skip-high storage modes against local admission gates."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, write_json

SCHEMA = "snerv_skip_high_mode_comparison.v1"
DEFAULT_HARD_BYTE_CEILING = 178_000

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def build_skip_high_mode_comparison(
    *,
    binary_profiles: Mapping[str, str | Path],
    prefilter_profiles: Mapping[str, str | Path] | None = None,
    hard_byte_ceiling: int = DEFAULT_HARD_BYTE_CEILING,
) -> dict[str, Any]:
    """Build a false-authority comparison across SNeRV skip-high profiles."""

    rows = [
        _binary_profile_row(label, path, hard_byte_ceiling=hard_byte_ceiling)
        for label, path in sorted(binary_profiles.items())
    ]
    prefilter_rows = [
        _prefilter_profile_row(label, path, hard_byte_ceiling=hard_byte_ceiling)
        for label, path in sorted((prefilter_profiles or {}).items())
    ]
    blockers = ["snerv_skip_high_mode_comparison_false_authority"]
    if not rows:
        blockers.append("snerv_skip_high_binary_profiles_missing")
    if not any(row["under_hard_byte_ceiling"] for row in rows):
        blockers.append("no_skip_high_binary_profile_under_hard_byte_ceiling")
    if not any(row["under_hard_byte_ceiling"] and not row["scalar_collapse_risk"] for row in rows):
        blockers.append("no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage")
    if any(row["scorer_input_out_of_distribution"] for row in prefilter_rows):
        blockers.append("skip_high_prefilter_scorer_input_out_of_distribution")
    if not any(
        row["under_hard_byte_ceiling"] and row["local_replay_admissible"]
        for row in prefilter_rows
    ):
        blockers.append("no_skip_high_prefilter_profile_admissible_for_local_replay")

    best_rate = min(rows, key=lambda row: row["archive_bytes"]) if rows else None
    best_non_scalar = min(
        (row for row in rows if not row["scalar_collapse_risk"]),
        key=lambda row: row["archive_bytes"],
        default=None,
    )
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "hard_byte_ceiling": int(hard_byte_ceiling),
        "binary_profile_rows": rows,
        "prefilter_profile_rows": prefilter_rows,
        "best_rate_row": _row_ref(best_rate),
        "best_non_scalar_skip_high_row": _row_ref(best_non_scalar),
        "verdict": (
            "NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL"
            if blockers
            else "LOCAL_SKIP_HIGH_PREFILTER_READY_FOR_CPU_REPLAY"
        ),
        "crux": _crux(rows=rows, prefilter_rows=prefilter_rows),
        "next_actions": _next_actions(rows=rows, prefilter_rows=prefilter_rows),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def write_skip_high_mode_comparison(
    *,
    output_json: str | Path,
    output_md: str | Path | None = None,
    binary_profiles: Mapping[str, str | Path],
    prefilter_profiles: Mapping[str, str | Path] | None = None,
    hard_byte_ceiling: int = DEFAULT_HARD_BYTE_CEILING,
) -> dict[str, Any]:
    payload = build_skip_high_mode_comparison(
        binary_profiles=binary_profiles,
        prefilter_profiles=prefilter_profiles,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    write_json(output_json, payload)
    if output_md is not None:
        Path(output_md).write_text(render_markdown_report(payload), encoding="utf-8")
    return payload


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    rows = payload.get("binary_profile_rows") or []
    prefilter_rows = payload.get("prefilter_profile_rows") or []
    lines = [
        "# SNeRV Skip-High Mode Comparison",
        "",
        f"Schema: `{payload.get('schema')}`",
        f"Verdict: `{payload.get('verdict')}`",
        "Axis: `[macOS-CPU/MLX planning:false-authority]`",
        "",
        "## Binary Profiles",
        "",
        "| label | codec | archive bytes | stored shape | stored raw bytes | under cap | scalar collapse |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {codec} | {archive_bytes} | `{shape}` | {stored_raw_bytes} | {under} | {collapse} |".format(
                label=row["label"],
                codec=row["skip_high_codec"],
                archive_bytes=row["archive_bytes"],
                shape=row["skip_high_stored_shape"],
                stored_raw_bytes=row["skip_high_stored_raw_bytes"],
                under=row["under_hard_byte_ceiling"],
                collapse=row["scalar_collapse_risk"],
            )
        )
    lines.extend(["", "## Prefilter Profiles", ""])
    if prefilter_rows:
        lines.extend(
            [
                "| label | score | Seg term | Pose term | local replay | OOD |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in prefilter_rows:
            lines.append(
                "| {label} | {score} | {seg} | {pose} | {replay} | {ood} |".format(
                    label=row["label"],
                    score=_fmt(row["canonical_score"]),
                    seg=_fmt(row["seg_term"]),
                    pose=_fmt(row["pose_term"]),
                    replay=row["local_replay_admissible"],
                    ood=row["scorer_input_out_of_distribution"],
                )
            )
    else:
        lines.append("- No scorer prefilter profiles attached.")
    lines.extend(["", "## Crux", ""])
    for item in payload.get("crux") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in payload.get("next_actions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Blockers", ""])
    for item in payload.get("blockers") or []:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def _binary_profile_row(
    label: str,
    path: str | Path,
    *,
    hard_byte_ceiling: int,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    payload = read_json(resolved)
    decoder = _mapping(payload.get("decoder_payload_header"))
    skip = _mapping(decoder.get("skip_high_storage"))
    archive_bytes = _int_or_none(payload.get("charged_archive_bytes"))
    if archive_bytes is None:
        package = _mapping(payload.get("package_profile"))
        archive_bytes = _int_or_none(package.get("archive_bytes")) or 0
    stored_shape = [int(v) for v in skip.get("stored_shape") or []]
    stored_raw = _int_or_none(skip.get("stored_raw_bytes")) or 0
    source_raw = _int_or_none(skip.get("source_raw_bytes")) or 0
    return {
        "label": str(label),
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "schema": payload.get("schema"),
        "archive_bytes": int(archive_bytes),
        "packet_bytes": _int_or_none(payload.get("snar1_packet_bytes")),
        "under_hard_byte_ceiling": int(archive_bytes) <= int(hard_byte_ceiling),
        "bytes_over_hard_ceiling": int(archive_bytes) - int(hard_byte_ceiling),
        "skip_high_codec": skip.get("codec"),
        "skip_high_stored_shape": stored_shape,
        "skip_high_source_shape": [int(v) for v in skip.get("source_shape") or []],
        "skip_high_stored_raw_bytes": stored_raw,
        "skip_high_source_raw_bytes": source_raw,
        "skip_high_raw_byte_savings": _int_or_none(skip.get("raw_byte_savings")),
        "receiver_expands_skip_high": bool(skip.get("receiver_expands_skip_high")),
        "lossless_relative_to_source_skip_high": bool(
            skip.get("lossless_relative_to_source_skip_high")
        ),
        "scalar_collapse_risk": bool(
            skip.get("receiver_expands_skip_high")
            and (stored_raw <= 8 or _shape_numel(stored_shape) <= 1)
        ),
        "decoder_payload_bytes": _int_or_none(
            _mapping(payload.get("section_summary")).get("largest_section_bytes")
        ),
        "largest_section": _mapping(payload.get("section_summary")).get("largest_section"),
        "verdict": payload.get("verdict"),
        "blockers": [str(v) for v in payload.get("blockers") or []],
        **FALSE_AUTHORITY,
    }


def _prefilter_profile_row(
    label: str,
    path: str | Path,
    *,
    hard_byte_ceiling: int,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=False)
    payload = read_json(resolved)
    score = _mapping(payload.get("score_components"))
    diagnosis = _mapping(payload.get("scorer_input_diagnosis"))
    archive_bytes = _int_or_none(payload.get("archive_bytes")) or 0
    scope = _mapping(payload.get("scope_status"))
    batch_pairs = _int_or_none(payload.get("scorer_batch_pairs"))
    blockers = [str(v) for v in payload.get("blockers") or []]
    out_of_distribution = (
        diagnosis.get("candidate_output_out_of_distribution") is True
        or "mlx_renderer_prefilter_scorer_input_out_of_distribution" in blockers
        or any(blocker.startswith("scorer_input_") for blocker in blockers)
    )
    local_replay_admissible = bool(
        scope.get("full_video") == "executed"
        and batch_pairs == 1
        and int(archive_bytes) <= int(hard_byte_ceiling)
        and not out_of_distribution
    )
    return {
        "label": str(label),
        "path": resolved.as_posix(),
        "sha256": _sha256_file(resolved),
        "schema": payload.get("schema"),
        "archive_bytes": int(archive_bytes),
        "under_hard_byte_ceiling": int(archive_bytes) <= int(hard_byte_ceiling),
        "scope_full_video": scope.get("full_video"),
        "scorer_batch_pairs": batch_pairs,
        "canonical_score": _float_or_none(score.get("canonical_score")),
        "seg_term": _float_or_none(score.get("seg_term")),
        "pose_term": _float_or_none(score.get("pose_term")),
        "rate_term": _float_or_none(score.get("rate_term")),
        "scorer_input_out_of_distribution": out_of_distribution,
        "scorer_input_verdict": diagnosis.get("verdict"),
        "local_replay_admissible": local_replay_admissible,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _crux(
    *,
    rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    scalar_rows = [row for row in rows if row["scalar_collapse_risk"]]
    non_scalar_rows = [row for row in rows if not row["scalar_collapse_risk"]]
    if scalar_rows:
        best = min(scalar_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "rate-admissible scalar skip-high is cheap "
            f"({best['archive_bytes']} bytes) but collapses stored skip-high to "
            f"{best['skip_high_stored_raw_bytes']} raw bytes."
        )
    if non_scalar_rows:
        best = min(non_scalar_rows, key=lambda row: row["archive_bytes"])
        out.append(
            "non-scalar skip-high preserves more value-domain structure but the "
            f"best attached profile is {best['archive_bytes']} bytes "
            f"({best['bytes_over_hard_ceiling']} vs hard ceiling)."
        )
    if any(row["scorer_input_out_of_distribution"] for row in prefilter_rows):
        out.append(
            "attached scorer prefilter evidence is out of distribution; do not "
            "promote or exact-dispatch from these local scores."
        )
    if not out:
        out.append("no attached profiles were sufficient to localize the skip-high crux")
    return out


def _next_actions(
    *,
    rows: list[dict[str, Any]],
    prefilter_rows: list[dict[str, Any]],
) -> list[str]:
    actions = [
        "block Modal/exact auth eval until a byte-closed candidate also passes local scorer-input and cache-quality gates",
        "run the next SNeRV local skip-high smoke on a non-scalar storage mode only after current MLX claims clear",
        "record frame-1 SegNet, two-frame PoseNet, archive bytes, and skip-high storage shape for every mode",
    ]
    if any(row["scalar_collapse_risk"] and row["under_hard_byte_ceiling"] for row in rows):
        actions.append(
            "do not use scalar_mean as the promotion path unless a receiver value-domain xray disproves the collapse mechanism"
        )
    if prefilter_rows and not any(row["local_replay_admissible"] for row in prefilter_rows):
        actions.append(
            "treat current local prefilter rows as acquisition/falsification evidence only"
        )
    return _ordered_unique(actions)


def _row_ref(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "label": row.get("label"),
        "archive_bytes": row.get("archive_bytes"),
        "path": row.get("path"),
        "skip_high_codec": row.get("skip_high_codec"),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _shape_numel(shape: list[int]) -> int:
    out = 1
    if not shape:
        return 0
    for dim in shape:
        out *= int(dim)
    return out


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _fmt(value: Any) -> str:
    parsed = _float_or_none(value)
    return "n/a" if parsed is None else f"{parsed:.6g}"


__all__ = [
    "DEFAULT_HARD_BYTE_CEILING",
    "SCHEMA",
    "build_skip_high_mode_comparison",
    "render_markdown_report",
    "write_skip_high_mode_comparison",
]
