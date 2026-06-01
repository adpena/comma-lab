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
from collections.abc import Mapping, Sequence
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
Z8_ENTROPY_DELTA_MATERIALIZER_WORK_ORDER_SCHEMA = (
    "z8_entropy_delta_materializer_work_order.v1"
)


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


def _path_text(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = Path(path).as_posix()
    return text or None


def _path_exists(path_text: str, *, repo_root: str | Path | None = None) -> bool:
    path = Path(path_text)
    if path.is_absolute():
        return path.is_file()
    if repo_root is not None:
        return (Path(repo_root) / path).is_file()
    return path.is_file()


def build_entropy_delta_materializer_work_order(
    schedule: Mapping[str, Any],
    *,
    schedule_json_path: str | Path,
    output_dir: str | Path,
    archive_bin: str | Path | None = None,
    repo_root: str | Path | None = None,
    require_existing_archive_bin: bool = True,
    emit_receiver_proof: bool = False,
    run_inflate_runtime_benchmark: bool = False,
    extra_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the smallest executable Z8 materializer row for a ready schedule.

    The schedule itself is advisory. This work order is the bridge into the
    byte-closed materializer: it keeps score authority false, consumes the same
    schedule JSON the solver wrote, and runs the Z8 archive/runtime exporter
    through ``tools/materialize_z8_joint_p18_p19_deadzone_candidate.py``.
    """

    blockers: list[str] = []
    schedule_dict = dict(schedule)
    try:
        steps = coerce_entropy_detail_quantization_steps(schedule_dict)
    except ValueError as exc:
        steps = {}
        blockers.append(f"schedule_not_materializer_ready:{exc}")
    raw_archive_bin = archive_bin if archive_bin is not None else schedule_dict.get("source_archive_path")
    archive_bin_text = _path_text(raw_archive_bin)
    if not archive_bin_text:
        blockers.append("source_archive_bin_missing")
    elif require_existing_archive_bin and not _path_exists(archive_bin_text, repo_root=repo_root):
        blockers.append(f"source_archive_bin_missing_on_disk:{archive_bin_text}")
    output_dir_text = _path_text(output_dir)
    schedule_json_text = _path_text(schedule_json_path)
    if not output_dir_text:
        blockers.append("materializer_output_dir_missing")
    if not schedule_json_text:
        blockers.append("schedule_json_path_missing")
    command: list[str] | None = None
    if not blockers:
        command = [
            ".venv/bin/python",
            "tools/materialize_z8_joint_p18_p19_deadzone_candidate.py",
            "--archive-bin",
            str(archive_bin_text),
            "--output-dir",
            str(output_dir_text),
            "--no-mutate-coefficients",
            "--entropy-code-quantized-details",
            "--entropy-detail-quantization-steps-json",
            str(schedule_json_text),
        ]
        if repo_root is not None:
            command.extend(["--repo-root", Path(repo_root).as_posix()])
        if emit_receiver_proof:
            command.append("--emit-receiver-proof")
        if run_inflate_runtime_benchmark:
            command.append("--run-inflate-runtime-benchmark")
        command.extend(str(item) for item in extra_args)
    work_order = {
        "schema": Z8_ENTROPY_DELTA_MATERIALIZER_WORK_ORDER_SCHEMA,
        "purpose": (
            "Execute a ready Z8 entropy-detail quantization schedule through "
            "the byte-closed Z8HPC1 materializer, then let receiver proof and "
            "contest CPU/CUDA gates decide promotion."
        ),
        **NON_PROMOTABLE_MARKERS,
        "schedule_schema": schedule_dict.get("schema"),
        "schedule_sha256": schedule_dict.get("schedule_sha256"),
        "source_report_sha256": schedule_dict.get("source_report_sha256"),
        "source_archive_sha256": schedule_dict.get("source_archive_sha256"),
        "source_archive_bin": archive_bin_text,
        "require_existing_archive_bin": bool(require_existing_archive_bin),
        "schedule_json_path": schedule_json_text,
        "materializer_output_dir": output_dir_text,
        "step_count": len(steps),
        "emit_receiver_proof": bool(emit_receiver_proof),
        "run_inflate_runtime_benchmark": bool(run_inflate_runtime_benchmark),
        "materializer_command": command,
        "blockers": blockers,
        "ready_for_materializer_execution": bool(command and not blockers),
        "exact_axis_blocker": (
            "contest_cpu_cuda_eval_not_executed"
            if emit_receiver_proof
            else "receiver_proof_and_contest_cpu_cuda_eval_not_executed"
        ),
    }
    return work_order


def build_entropy_delta_campaign_plan(
    report: Mapping[str, Any],
    *,
    max_subband_mse: float,
    schedule_json_path: str | Path,
    materializer_output_dir: str | Path,
    archive_bin: str | Path | None = None,
    repo_root: str | Path | None = None,
    require_full_archive_coverage: bool = True,
    require_existing_archive_bin: bool = True,
    emit_receiver_proof: bool = False,
    run_inflate_runtime_benchmark: bool = False,
    extra_materializer_args: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the queue-consumable headroom -> schedule -> materializer plan.

    This is the package-level bridge that keeps Z8 entropy-delta work out of
    manual JSON handoffs: a headroom report becomes a fail-closed schedule, then
    the exact materializer command that consumes that schedule.
    """

    report_dict = dict(report)
    schedule = build_entropy_delta_schedule_from_headroom_report(
        report_dict,
        max_subband_mse=float(max_subband_mse),
        require_full_archive_coverage=bool(require_full_archive_coverage),
    )
    work_order = build_entropy_delta_materializer_work_order(
        schedule,
        schedule_json_path=schedule_json_path,
        output_dir=materializer_output_dir,
        archive_bin=archive_bin,
        repo_root=repo_root,
        require_existing_archive_bin=bool(require_existing_archive_bin),
        emit_receiver_proof=bool(emit_receiver_proof),
        run_inflate_runtime_benchmark=bool(run_inflate_runtime_benchmark),
        extra_args=tuple(extra_materializer_args),
    )
    blockers = list(schedule.get("blockers") or []) + list(work_order.get("blockers") or [])
    return {
        "schema": "z8_entropy_delta_campaign_plan.v1",
        "purpose": (
            "Queue-owned Z8 detail entropy-delta bridge: headroom report to "
            "per-subband schedule to byte-closed materializer work order."
        ),
        **NON_PROMOTABLE_MARKERS,
        "source_report_schema": report_dict.get("schema"),
        "source_archive_path": report_dict.get("archive_path"),
        "source_archive_sha256": report_dict.get("archive_sha256"),
        "max_subband_mse": float(max_subband_mse),
        "require_full_archive_coverage": bool(require_full_archive_coverage),
        "schedule": schedule,
        "materializer_work_order": work_order,
        "blockers": blockers,
        "ready_for_queue_execution": bool(
            schedule.get("ready_for_materializer")
            and work_order.get("ready_for_materializer_execution")
            and not blockers
        ),
    }


__all__ = [
    "NON_PROMOTABLE_MARKERS",
    "Z8_ENTROPY_DELTA_MATERIALIZER_WORK_ORDER_SCHEMA",
    "build_entropy_delta_campaign_plan",
    "build_entropy_delta_materializer_work_order",
    "build_entropy_delta_schedule_from_headroom_report",
    "coerce_entropy_detail_quantization_steps",
    "load_entropy_detail_quantization_steps_json",
    "parse_aggregate_subband_key",
]
