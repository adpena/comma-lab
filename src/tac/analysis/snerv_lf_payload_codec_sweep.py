# SPDX-License-Identifier: MIT
"""False-authority SNeRV LF payload codec sweep.

This measures receiver-visible LF packet bytes for exact lossless integer
grammars. It is a rate-only surface: no visual metric, no scorer replay, and no
promotion authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from tac.analysis.nerv_scorer_objective import SCORER_ONLY_OBJECTIVE_AUTHORITY
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    encode_lf_quant_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_payload_codec import (
    SNERV_LF_PAYLOAD_INTN_CODEC_PROOF,
    inspect_lf_quant_payload_v2,
)

SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA = "snerv_lf_payload_codec_sweep.v1"
DEFAULT_LF_CODEC_MODES: tuple[str, ...] = (
    "int64_lzma",
    "portfolio_auto",
    "zero_run",
    "delta_varint",
    "uint8_escape",
    "uint4_escape",
    "uint2_escape",
    "uint8",
    "uint4",
    "uint2",
    "int8_escape",
    "int4_escape",
    "int2_escape",
    "int8",
    "int4",
    "int2",
)


def build_snerv_lf_payload_codec_sweep(
    lf_quant_planes: Sequence[np.ndarray],
    *,
    modes: Iterable[str] = DEFAULT_LF_CODEC_MODES,
    baseline_mode: str = "int64_lzma",
) -> dict[str, Any]:
    """Measure exact receiver-visible LF payload bytes for several codecs."""

    planes = [np.asarray(plane, dtype=np.int64) for plane in lf_quant_planes]
    if not planes:
        raise ValueError("lf_quant_planes must be non-empty")
    raw_i64_bytes = int(sum(plane.size * np.dtype("<i8").itemsize for plane in planes))
    rows = []
    blockers = [
        "snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    for mode in dict.fromkeys(str(mode).strip() for mode in modes if str(mode).strip()):
        row = _sweep_row(planes, mode=mode, raw_i64_bytes=raw_i64_bytes)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            0 if int(row.get("payload_bytes") or 0) > 0 and not row.get("error") else 1,
            int(row.get("payload_bytes") or 0)
            if int(row.get("payload_bytes") or 0) > 0
            else 2**63 - 1,
            row["mode"],
        )
    )
    baseline_payload_bytes = _baseline_payload_bytes(
        rows,
        baseline_mode=str(baseline_mode),
        fallback_bytes=raw_i64_bytes,
    )
    section_value_rows = _section_value_rows(
        rows,
        baseline_mode=str(baseline_mode),
        baseline_payload_bytes=baseline_payload_bytes,
    )
    selected = next(
        (
            row
            for row in rows
            if int(row.get("payload_bytes") or 0) > 0 and not row.get("error")
        ),
        rows[0],
    )
    report = {
        "schema": SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA,
        "authority": "false_authority_lf_payload_codec_rate_only",
        "objective_authority": SCORER_ONLY_OBJECTIVE_AUTHORITY,
        "codec_proof": SNERV_LF_PAYLOAD_INTN_CODEC_PROOF,
        "family": "snerv",
        "axis_tag": "[planning/control]",
        "plane_count": len(planes),
        "plane_shapes": [[int(v) for v in plane.shape] for plane in planes],
        "raw_i64_bytes": raw_i64_bytes,
        "baseline_mode": str(baseline_mode),
        "baseline_payload_bytes": int(baseline_payload_bytes),
        "rows": rows,
        "section_value_rows": section_value_rows,
        "selected_rate_only_row": selected,
        "marginal_rate_savings": _rate_savings(rows, baseline_bytes=raw_i64_bytes),
        "failed_modes": [
            {
                "mode": str(row.get("mode")),
                "error": str(row.get("error")),
            }
            for row in rows
            if row.get("error")
        ],
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report["byte_price_plan"] = build_nerv_byte_price_plan(report)
    return report


def _sweep_row(
    planes: Sequence[np.ndarray],
    *,
    mode: str,
    raw_i64_bytes: int,
) -> dict[str, Any]:
    blockers = [
        "snerv_lf_payload_codec_row_has_no_scorer_replay",
        "snerv_lf_payload_codec_row_not_exact_eval_authority",
    ]
    try:
        payload = encode_lf_quant_payload(list(planes), codec=mode)
        error = None
    except Exception as exc:
        payload = b""
        error = str(exc)
        blockers.append("snerv_lf_payload_codec_mode_failed")
    packet_bytes = len(payload)
    inspect = {}
    if payload:
        try:
            if mode != "int64_lzma":
                inspect = inspect_lf_quant_payload_v2(payload).as_jsonable()
        except Exception as exc:
            error = str(exc)
            blockers.append("snerv_lf_payload_codec_inspection_failed")
    savings = raw_i64_bytes - packet_bytes if packet_bytes else 0
    return {
        "mode": str(mode),
        "payload_bytes": int(packet_bytes),
        "raw_i64_bytes": int(raw_i64_bytes),
        "bytes_saved_vs_raw_i64": int(savings),
        "rate_score_saved_vs_raw_i64": float(savings * CONTEST_BYTE_PRICE_SCORE),
        "packet_schema": inspect.get("schema", "snerv_lf_quant_payload.v1"),
        "mode_histogram": inspect.get("mode_histogram", {}),
        "wrapper_histogram": inspect.get("wrapper_histogram", {}),
        "error": error,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _rate_savings(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_bytes: int,
) -> list[dict[str, Any]]:
    out = []
    valid = [row for row in rows if int(row.get("payload_bytes") or 0) > 0]
    for row in valid:
        saved = int(baseline_bytes) - int(row["payload_bytes"])
        out.append(
            {
                "mode": row["mode"],
                "payload_bytes": int(row["payload_bytes"]),
                "bytes_saved_vs_raw_i64": int(saved),
                "score_saved_vs_raw_i64": float(saved * CONTEST_BYTE_PRICE_SCORE),
            }
        )
    return out


def _baseline_payload_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str,
    fallback_bytes: int,
) -> int:
    for row in rows:
        if str(row.get("mode")) == baseline_mode:
            value = int(row.get("payload_bytes") or 0)
            if value > 0:
                return value
    return int(fallback_bytes)


def _section_value_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_mode: str,
    baseline_payload_bytes: int,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        payload_bytes = int(row.get("payload_bytes") or 0)
        mode = str(row.get("mode"))
        valid_lossless_packet = payload_bytes > 0 and not row.get("error")
        row_blockers = [
            *[str(blocker) for blocker in row.get("blockers") or ()],
            "snerv_lf_payload_codec_has_packet_exactness_only_no_full_archive_replay",
        ]
        out.append(
            {
                "row_id": f"snerv_lf_payload_codec_{mode}",
                "section_id": "snerv_lf_payload",
                "family": "snerv",
                "scope": "lf_payload_codec_replacement",
                "row_kind": "existing_section_cut",
                "candidate_mode": mode,
                "baseline_mode": str(baseline_mode),
                "baseline_payload_bytes": int(baseline_payload_bytes),
                "payload_bytes": payload_bytes,
                "byte_delta": (
                    int(payload_bytes) - int(baseline_payload_bytes)
                    if valid_lossless_packet
                    else None
                ),
                "delta_nonrate_score": 0.0 if valid_lossless_packet else None,
                "axis_tag": "[planning/control]",
                "receiver_proof_status": (
                    "packet_exact_only_full_archive_replay_missing"
                    if valid_lossless_packet
                    else "missing"
                ),
                "full_video_coverage": False,
                "exact_lossless_lf_packet_codec": bool(valid_lossless_packet),
                "blockers": _ordered_unique(row_blockers),
                **FALSE_AUTHORITY,
            }
        )
    return out


def _ordered_unique(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def render_snerv_lf_payload_codec_sweep_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing LF codec sweep report."""

    lines = [
        "# SNeRV LF payload codec sweep",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Baseline mode: `{report.get('baseline_mode')}`",
        f"Baseline payload bytes: `{report.get('baseline_payload_bytes')}`",
        "",
        "| mode | payload bytes | byte delta vs baseline | economic decision | final decision |",
        "|---|---:|---:|---|---|",
    ]
    decisions = {
        str(row.get("source", {}).get("candidate_mode")): row
        for row in report.get("byte_price_plan", {}).get("decision_rows", ())
        if isinstance(row, Mapping)
    }
    for row in report.get("rows", ()):
        mode = str(row.get("mode"))
        decision = decisions.get(mode, {})
        byte_delta = decision.get("byte_delta")
        lines.append(
            "| {mode} | {payload} | {delta} | {economic} | {decision} |".format(
                mode=mode,
                payload=int(row.get("payload_bytes") or 0),
                delta="n/a" if byte_delta is None else int(byte_delta),
                economic=decision.get("economic_decision", "n/a"),
                decision=decision.get("decision", "n/a"),
            )
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in report.get("blockers", ()):
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_LF_CODEC_MODES",
    "SNERV_LF_PAYLOAD_CODEC_SWEEP_SCHEMA",
    "build_snerv_lf_payload_codec_sweep",
    "render_snerv_lf_payload_codec_sweep_markdown",
]
