# SPDX-License-Identifier: MIT
"""Bind SNeRV trained ladder rows to decoder-weight waterfill plans.

This adapter consumes receiver-visible SNeRV SNAR1 bytes, including the
``0.bin`` member inside a packaged contest archive ZIP, decodes the archived HF
decoder, and then delegates to the shared NeRV decoder-weight waterfill planner.
It is deliberately false-authority: local replay, missing saliency, non-full600
coverage, or missing exact contest CPU/CUDA replay remain blockers.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.nerv_decoder_weight_waterfill import (
    DEFAULT_ACTION_BITS,
    NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
    build_nerv_decoder_weight_waterfill_plan,
)
from tac.repo_io import sha256_file
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DECODER_SUBBANDS,
    DecodedSnervArchive,
    SnervArchiveError,
    unpack_snerv_archive,
)

SNERV_TRAINED_LADDER_WATERFILL_SCHEMA = "snerv_trained_ladder_waterfill.v1"
SNERV_TRAINED_ROW_SCHEMA = "nerv_trained_ladder_row_payload.v1"


class SnervTrainedLadderWaterfillError(ValueError):
    """Raised when a SNeRV trained-row waterfill input is malformed."""


def build_snerv_trained_ladder_waterfill(
    trained_ladder_row_payload: Mapping[str, Any],
    *,
    saliency_by_name: Mapping[str, float] | None = None,
    action_bits: Sequence[int] = DEFAULT_ACTION_BITS,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Build decoder-weight waterfill plans from SNeRV trained-row artifacts."""

    if trained_ladder_row_payload.get("schema") != SNERV_TRAINED_ROW_SCHEMA:
        raise SnervTrainedLadderWaterfillError(
            f"expected {SNERV_TRAINED_ROW_SCHEMA} payload"
        )
    if str(trained_ladder_row_payload.get("family") or "").lower() != "snerv":
        raise SnervTrainedLadderWaterfillError("trained row family must be snerv")

    rows = []
    section_value_rows = []
    blockers = [
        "contest_cpu_cuda_exact_eval_not_executed",
        "decoder_weight_saliency_replay_required_for_authority",
    ]
    for row_index, trained_row in enumerate(trained_ladder_row_payload.get("rows") or ()):
        if not isinstance(trained_row, Mapping):
            continue
        row_id = str(trained_row.get("row_id") or f"snerv_row_{row_index:04d}")
        row_result = _waterfill_for_trained_row(
            trained_row,
            row_id=row_id,
            source_payload_blockers=_string_list(
                trained_ladder_row_payload.get("blockers")
            ),
            saliency_by_name=saliency_by_name or {},
            action_bits=action_bits,
            candidate_id=candidate_id
            or str(trained_ladder_row_payload.get("candidate_id") or ""),
        )
        rows.append(row_result)
        blockers.extend(row_result["blockers"])
        plan = row_result.get("waterfill_plan")
        if isinstance(plan, Mapping):
            for section in plan.get("section_value_rows") or ():
                if isinstance(section, Mapping):
                    section_value_rows.append(
                        {
                            **dict(section),
                            "row_id": f"{row_id}:{section.get('row_id')}",
                            "section_id": f"{row_id}:{section.get('section_id')}",
                            "trained_ladder_row_id": row_id,
                        }
                    )

    report = {
        "schema": SNERV_TRAINED_LADDER_WATERFILL_SCHEMA,
        "source_schema": trained_ladder_row_payload.get("schema"),
        "family": "snerv",
        "axis_tag": "[planning/control]",
        "authority": "false_authority_snerv_trained_ladder_decoder_waterfill_no_score_claim",
        "candidate_id": candidate_id or trained_ladder_row_payload.get("candidate_id"),
        "source_status": trained_ladder_row_payload.get("status"),
        "source_verdict": trained_ladder_row_payload.get("verdict"),
        "row_count": len(rows),
        "rows": rows,
        "section_value_rows": section_value_rows,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    return report


def render_snerv_trained_ladder_waterfill_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary."""

    lines = [
        "# SNeRV trained ladder decoder waterfill",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        "",
        "| row | groups | byte delta | source codec | blocker count |",
        "|---|---:|---:|---|---:|",
    ]
    for row in report.get("rows") or ():
        summary = row.get("waterfill_summary") or {}
        lines.append(
            "| {row_id} | {groups} | {delta} | {codec} | {blockers} |".format(
                row_id=row.get("row_id"),
                groups=summary.get("group_count", 0),
                delta=summary.get("total_selected_byte_delta", 0),
                codec=row.get("decoder_payload_schema"),
                blockers=len(row.get("blockers") or ()),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _waterfill_for_trained_row(
    trained_row: Mapping[str, Any],
    *,
    row_id: str,
    source_payload_blockers: Sequence[str],
    saliency_by_name: Mapping[str, float],
    action_bits: Sequence[int],
    candidate_id: str,
) -> dict[str, Any]:
    row_blockers = _string_list(trained_row.get("emission_blockers"))
    archive_path = Path(str(trained_row.get("archive_path") or ""))
    archive_sha256 = str(trained_row.get("archive_sha256") or "")
    archive_sha_actual = None
    if not archive_path.is_file():
        return _blocked_row(
            row_id=row_id,
            trained_row=trained_row,
            blockers=[*row_blockers, f"archive_path_not_file:{archive_path}"],
        )
    archive_sha_actual = sha256_file(archive_path)
    blockers = []
    if archive_sha256 and archive_sha_actual != archive_sha256:
        blockers.append("archive_sha256_mismatch")

    try:
        decoded = _decode_snerv_from_archive_path(archive_path)
        state = _decoder_state_dict(decoded)
    except (SnervArchiveError, zipfile.BadZipFile) as exc:
        return _blocked_row(
            row_id=row_id,
            trained_row=trained_row,
            blockers=[
                *row_blockers,
                *blockers,
                _decode_error_blocker(str(exc)),
            ],
            archive_sha_actual=archive_sha_actual,
        )

    full_video_coverage = int(trained_row.get("n_pairs") or 0) >= 600
    receiver_proof_status = (
        "receiver_proof_valid"
        if trained_row.get("receiver_proof_passed") is True
        else "local_receiver_replay_only"
        if trained_row.get("receiver_archive_replay_verified") is True
        else "missing"
    )
    plan = build_nerv_decoder_weight_waterfill_plan(
        state,
        saliency_by_name=saliency_by_name,
        family="snerv",
        candidate_id=f"{candidate_id}:{row_id}" if candidate_id else row_id,
        action_bits=action_bits,
        full_video_coverage=full_video_coverage,
        receiver_proof_status=receiver_proof_status,
        archive_sha256=archive_sha_actual or archive_sha256,
    )
    decoder_header = decoded.metadata.get("decoder_payload_header")
    return {
        "row_id": row_id,
        "archive_path": archive_path.as_posix(),
        "archive_bytes": trained_row.get("archive_bytes"),
        "archive_sha256": archive_sha256 or None,
        "archive_sha256_actual": archive_sha_actual,
        "n_pairs": trained_row.get("n_pairs"),
        "source_axis_tag": trained_row.get("source_axis_tag"),
        "receiver_codec_mode": trained_row.get("receiver_codec_mode"),
        "decoder_precision_mode": trained_row.get("decoder_precision_mode"),
        "decoder_payload_schema": (
            decoder_header.get("schema") if isinstance(decoder_header, Mapping) else None
        ),
        "decoder_state_group_count": len(state),
        "decoder_state_group_names": sorted(state),
        "waterfill_plan_schema": NERV_DECODER_WEIGHT_WATERFILL_SCHEMA,
        "waterfill_summary": {
            "group_count": plan["group_count"],
            "total_baseline_fp32_bytes": plan["total_baseline_fp32_bytes"],
            "total_selected_estimated_bytes": plan["total_selected_estimated_bytes"],
            "total_selected_byte_delta": plan["total_selected_byte_delta"],
        },
        "waterfill_plan": plan,
        "blockers": _ordered_unique(
            [
                *row_blockers,
                *source_payload_blockers,
                *blockers,
                *plan["blockers"],
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _blocked_row(
    *,
    row_id: str,
    trained_row: Mapping[str, Any],
    blockers: Sequence[str],
    archive_sha_actual: str | None = None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "archive_path": trained_row.get("archive_path"),
        "archive_bytes": trained_row.get("archive_bytes"),
        "archive_sha256": trained_row.get("archive_sha256"),
        "archive_sha256_actual": archive_sha_actual,
        "n_pairs": trained_row.get("n_pairs"),
        "waterfill_plan": None,
        "waterfill_summary": None,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _decode_snerv_from_archive_path(path: Path) -> DecodedSnervArchive:
    packet = _snar_bytes_from_archive_path(path)
    decoded = unpack_snerv_archive(packet)
    decoder_header = json.loads(
        json.dumps(_inspect_decoder_payload_header(decoded.sections["decoder_payload"]))
    )
    decoded.metadata["decoder_payload_header"] = decoder_header
    return decoded


def _snar_bytes_from_archive_path(path: Path) -> bytes:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = [info.filename for info in zf.infolist() if not info.is_dir()]
            if "0.bin" not in names:
                raise SnervArchiveError("contest archive zip missing 0.bin SNAR member")
            return zf.read("0.bin")
    return path.read_bytes()


def _decoder_state_dict(decoded: DecodedSnervArchive) -> dict[str, np.ndarray]:
    decoder = decoded.decode_decoder()
    state = {}
    for level in range(int(decoder.levels)):
        level_kernels = decoder.kernels.get(level)
        if not isinstance(level_kernels, Mapping):
            raise SnervArchiveError(f"decoder missing level {level}")
        for subband in DECODER_SUBBANDS:
            values = level_kernels.get(subband)
            if values is None:
                raise SnervArchiveError(f"decoder missing level {level} subband {subband}")
            state[f"decoder.level{level}.{subband}.kernel"] = np.asarray(
                values,
                dtype=np.float64,
            )
    return state


def _inspect_decoder_payload_header(payload: bytes) -> dict[str, Any]:
    from tac.substrates.snerv_inverse_steg_carrier.archive import (  # local to avoid API churn
        inspect_decoder_payload_header,
    )

    return inspect_decoder_payload_header(payload)


def _decode_error_blocker(message: str) -> str:
    if "missing 0.bin" in message:
        return "contest_archive_zip_missing_0_bin"
    if "sha256 mismatch" in message:
        return "snerv_archive_sha256_mismatch"
    if "bad SNeRV archive magic" in message:
        return "snerv_archive_magic_invalid"
    return "snerv_archive_decode_failed"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


__all__ = [
    "SNERV_TRAINED_LADDER_WATERFILL_SCHEMA",
    "SnervTrainedLadderWaterfillError",
    "build_snerv_trained_ladder_waterfill",
    "render_snerv_trained_ladder_waterfill_markdown",
]
