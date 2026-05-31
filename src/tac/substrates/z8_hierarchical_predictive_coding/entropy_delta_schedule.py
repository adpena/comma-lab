# SPDX-License-Identifier: MIT
"""Per-subband Z8 detail-quantization schedule selection.

The Z8 detail codec is already near the achievable entropy floor for a fixed
quantization step. The remaining rate lever is the RD operating point: choose a
different detail-band quantization step for each wavelet level/orientation under
a declared distortion budget, then let the archive materializer prove the
byte-closed result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "evidence_grade": "macOS-CPU-advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def parse_aggregate_subband_key(key: str) -> tuple[int, str]:
    """Parse ``L{level}_{lh|hl|hh}`` keys from the headroom report."""

    text = str(key)
    if "_" not in text or not text.startswith("L"):
        raise ValueError(f"expected aggregate subband key like L0_hh, got {key!r}")
    level_text, orient = text[1:].split("_", 1)
    if orient not in {"lh", "hl", "hh"}:
        raise ValueError(f"unsupported Z8 detail orientation: {orient!r}")
    try:
        level_idx = int(level_text)
    except ValueError as exc:
        raise ValueError(f"invalid Z8 level in subband key: {key!r}") from exc
    if level_idx < 0:
        raise ValueError(f"invalid negative Z8 level in subband key: {key!r}")
    return level_idx, orient


def _choose_quant_row(
    rows: list[dict[str, Any]],
    *,
    max_subband_mse: float,
) -> tuple[dict[str, Any], str]:
    admissible = [
        row
        for row in rows
        if float(row.get("distortion_mse", math.inf)) <= float(max_subband_mse)
    ]
    if admissible:
        return (
            min(
                admissible,
                key=lambda row: (
                    float(row.get("live_codec_brotli_bytes_per_coeff", math.inf)),
                    float(row.get("distortion_mse", math.inf)),
                    float(row.get("quant_step", math.inf)),
                ),
            ),
            "within_max_subband_mse_min_bytes",
        )
    return (
        min(
            rows,
            key=lambda row: (
                float(row.get("distortion_mse", math.inf)),
                float(row.get("live_codec_brotli_bytes_per_coeff", math.inf)),
                float(row.get("quant_step", math.inf)),
            ),
        ),
        "no_admissible_step_min_distortion_fallback",
    )


def build_entropy_delta_schedule_from_headroom_report(
    report: dict[str, Any],
    *,
    max_subband_mse: float,
    require_full_archive_coverage: bool = True,
) -> dict[str, Any]:
    """Build a materializer-ready ``entropy_detail_quantization_steps`` map.

    The input is the read-only output of
    ``tools/z8_detail_coeff_entropy_headroom_report.py``. Each aggregate
    subband is measured across both frames, so the selected step is applied to
    ``frame_0_details`` and ``frame_1_details`` for the same level/orientation.
    """

    if not (float(max_subband_mse) >= 0.0 and math.isfinite(float(max_subband_mse))):
        raise ValueError("max_subband_mse must be finite and non-negative")

    steps: dict[str, float] = {}
    chosen: list[dict[str, Any]] = []
    blockers: list[str] = []
    pairs_measured = report.get("pairs_measured")
    total_pairs = report.get("total_pairs_in_archive")
    if (
        require_full_archive_coverage
        and pairs_measured is not None
        and total_pairs is not None
        and int(pairs_measured) < int(total_pairs)
    ):
        blockers.append(f"partial_headroom_coverage:{int(pairs_measured)}/{int(total_pairs)}")
    for subband in report.get("per_subband", []):
        key = str(subband.get("subband"))
        rows = list(subband.get("quant_sweep") or [])
        if not rows:
            blockers.append(f"missing_quant_sweep:{key}")
            continue
        level_idx, orient = parse_aggregate_subband_key(key)
        row, reason = _choose_quant_row(rows, max_subband_mse=float(max_subband_mse))
        step = float(row["quant_step"])
        for frame_key in ("frame_0_details", "frame_1_details"):
            steps[f"{frame_key}:{level_idx}:{orient}"] = step
        chosen.append(
            {
                "aggregate_subband": key,
                "level_idx": int(level_idx),
                "orientation": orient,
                "quant_step": step,
                "selection_reason": reason,
                "distortion_mse": float(row.get("distortion_mse", math.nan)),
                "live_codec_method": row.get("live_codec_method"),
                "live_codec_brotli_bytes_per_coeff": float(
                    row.get("live_codec_brotli_bytes_per_coeff", math.nan)
                ),
            }
        )

    digest = hashlib.sha256(
        repr(sorted(steps.items())).encode("utf-8")
    ).hexdigest()
    source_report_sha256 = hashlib.sha256(
        json.dumps(report, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "z8_entropy_delta_schedule.v1",
        "purpose": (
            "Per-subband detail quantization schedule selected from measured "
            "Z8 entropy headroom RD rows; consumed by the joint P18/P19 "
            "dead-zone materializers as entropy_detail_quantization_steps."
        ),
        **NON_PROMOTABLE_MARKERS,
        "source_report_tool": report.get("tool"),
        "source_archive_path": report.get("archive_path"),
        "source_archive_sha256": report.get("archive_sha256"),
        "source_report_sha256": source_report_sha256,
        "source_archive_total_bytes": report.get("archive_total_bytes"),
        "source_wavelet_blob_bytes": report.get("wavelet_blob_bytes"),
        "pairs_measured": pairs_measured,
        "total_pairs_in_archive": total_pairs,
        "require_full_archive_coverage": bool(require_full_archive_coverage),
        "max_subband_mse": float(max_subband_mse),
        "schedule_sha256": digest,
        "entropy_detail_quantization_steps": steps,
        "chosen_subbands": chosen,
        "blockers": blockers,
        "ready_for_materializer": bool(steps and not blockers),
    }


def coerce_entropy_detail_quantization_steps(
    payload: dict[str, Any],
    *,
    require_ready: bool = True,
) -> dict[str, float]:
    """Extract a materializer step map from a schedule report or raw mapping."""

    raw_steps = payload.get("entropy_detail_quantization_steps", payload)
    if not isinstance(raw_steps, dict):
        raise ValueError(
            "entropy detail step JSON must be an object or contain entropy_detail_quantization_steps"
        )
    if "entropy_detail_quantization_steps" in payload and require_ready:
        blockers = list(payload.get("blockers") or [])
        if payload.get("ready_for_materializer") is not True:
            raise ValueError(
                "entropy detail schedule is not ready_for_materializer"
                + (f": {blockers}" if blockers else "")
            )
    out: dict[str, float] = {}
    for key, value in raw_steps.items():
        step = float(value)
        if step <= 0.0 or not math.isfinite(step):
            raise ValueError(f"entropy detail step for {key!r} must be finite and positive")
        out[str(key)] = step
    return out


def load_entropy_detail_quantization_steps_json(
    path: str | Path | None,
    *,
    require_ready: bool = True,
) -> dict[str, float] | None:
    """Load a raw step map or fail-closed schedule JSON for materializers."""

    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("entropy detail step JSON must be an object")
    return coerce_entropy_detail_quantization_steps(payload, require_ready=require_ready)


__all__ = [
    "NON_PROMOTABLE_MARKERS",
    "build_entropy_delta_schedule_from_headroom_report",
    "coerce_entropy_detail_quantization_steps",
    "load_entropy_detail_quantization_steps_json",
    "parse_aggregate_subband_key",
]
