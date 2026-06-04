#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the false-authority NeRV-family control inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_control_inventory import (  # noqa: E402
    build_nerv_control_inventory,
    render_nerv_control_inventory_markdown,
)
from tac.analysis.snerv_checkpoint_export_lf_payload_codec_report import (  # noqa: E402
    build_snerv_lf_payload_codec_report_from_checkpoint_export,
)
from tac.repo_io import write_json  # noqa: E402

RESEARCH_DIR = REPO_ROOT / ".omx" / "research"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--focus-family",
        action="append",
        default=None,
        help="Carrier family to include. Defaults to hi_nerv and snerv.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Durable JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional durable Markdown report path.",
    )
    parser.add_argument(
        "--repo-root",
        default=REPO_ROOT,
        type=Path,
        help="Repository root used for the implementation sweep.",
    )
    parser.add_argument(
        "--hinerv-archive-size-ladder-json",
        default=None,
        type=Path,
        help="Optional measured false-authority HiNeRV archive-size ladder JSON.",
    )
    parser.add_argument(
        "--hinerv-archive-ladder-waterfill-json",
        default=None,
        type=Path,
        help="Optional false-authority HiNeRV archive-ladder decoder-waterfill JSON.",
    )
    parser.add_argument(
        "--hinerv-archive-ladder-replay-actuator-json",
        default=None,
        type=Path,
        help=(
            "Optional false-authority HiNeRV archive-ladder replay actuator "
            "JSON."
        ),
    )
    parser.add_argument(
        "--hinerv-archive-backend-drift-json",
        default=None,
        type=Path,
        help=(
            "Optional false-authority HiNeRV archive backend byte-drift "
            "comparison JSON."
        ),
    )
    parser.add_argument(
        "--snerv-trained-ladder-waterfill-json",
        default=None,
        type=Path,
        help="Optional false-authority SNeRV trained-ladder decoder-waterfill JSON.",
    )
    parser.add_argument(
        "--hinerv-decoder-weight-saliency-json",
        default=None,
        type=Path,
        help="Optional false-authority HiNeRV decoder-weight saliency replay JSON.",
    )
    parser.add_argument(
        "--snerv-waterfill-mode-assignment-json",
        default=None,
        type=Path,
        help="Optional false-authority SNeRV waterfill decoder-mode assignment JSON.",
    )
    parser.add_argument(
        "--snerv-decoder-mode-probe-json",
        default=None,
        type=Path,
        help="Optional false-authority SNeRV decoder-mode advisory probe JSON.",
    )
    parser.add_argument(
        "--snerv-scorer-loop-qat-json",
        default=None,
        type=Path,
        action="append",
        help="Optional false-authority SNeRV scorer-loop QAT local trainer JSON.",
    )
    parser.add_argument(
        "--snerv-lf-payload-codec-sweep-json",
        default=None,
        type=Path,
        action="append",
        help=(
            "Optional false-authority SNeRV LF payload codec sweep JSON. "
            "Defaults to all latest matching research reports."
        ),
    )
    parser.add_argument(
        "--snerv-checkpoint-export-json",
        default=None,
        type=Path,
        action="append",
        help=(
            "Optional SNeRV checkpoint archive export JSON. Its receiver-visible "
            "LF payload report is converted into the LF codec inventory route."
        ),
    )
    args = parser.parse_args(argv)

    focus = tuple(args.focus_family or ("hi_nerv", "snerv"))
    hinerv_archive_size_ladder_report = _load_optional_report(
        args.hinerv_archive_size_ladder_json,
        pattern="hinerv_archive_size_ladder*.json",
        schema="hinerv_archive_size_ladder.v1",
    )
    hinerv_archive_ladder_waterfill_report = _load_optional_report(
        args.hinerv_archive_ladder_waterfill_json,
        pattern="hinerv_archive_ladder_waterfill*.json",
        schema="hinerv_archive_ladder_waterfill.v1",
    )
    hinerv_archive_ladder_replay_actuator_report = _load_optional_report(
        args.hinerv_archive_ladder_replay_actuator_json,
        pattern="hinerv_archive_ladder_replay_actuator*.json",
        schema="hinerv_archive_ladder_replay_actuator.v1",
    )
    hinerv_archive_backend_drift_report = _load_optional_report(
        args.hinerv_archive_backend_drift_json,
        pattern="hinerv_archive_backend_drift*.json",
        schema="hinerv_archive_backend_drift.v1",
    )
    snerv_trained_ladder_waterfill_report = _load_optional_report(
        args.snerv_trained_ladder_waterfill_json,
        pattern="snerv_trained_ladder_waterfill*.json",
        schema="snerv_trained_ladder_waterfill.v1",
    )
    hinerv_decoder_weight_saliency_report = _load_optional_report(
        args.hinerv_decoder_weight_saliency_json,
        pattern="hinerv_decoder_weight_saliency_replay*.json",
        schema="hinerv_decoder_weight_saliency_replay.v1",
    )
    snerv_waterfill_mode_assignment_report = _load_optional_report(
        args.snerv_waterfill_mode_assignment_json,
        pattern="snerv_waterfill_mode_assignment*.json",
        schema="snerv_waterfill_mode_assignment.v1",
    )
    snerv_decoder_mode_probe_report = _load_optional_report(
        args.snerv_decoder_mode_probe_json,
        pattern="snerv_decoder_mode_assignment_probe*.json",
        schema="snerv_decoder_mode_assignment_probe.v1",
    )
    snerv_scorer_loop_qat_reports = _load_optional_reports(
        args.snerv_scorer_loop_qat_json,
        pattern="snerv_scorer_loop_qat_local_trainer*.json",
        schema="snerv_scorer_loop_qat_local_trainer.v1",
    )
    snerv_lf_payload_codec_sweep_reports = _load_optional_reports(
        args.snerv_lf_payload_codec_sweep_json,
        pattern="snerv_lf_payload_codec_sweep*.json",
        schema="snerv_lf_payload_codec_sweep.v1",
    )
    snerv_lf_payload_codec_sweep_reports.extend(
        _checkpoint_export_lf_payload_reports(args.snerv_checkpoint_export_json)
    )
    report = build_nerv_control_inventory(
        focus_families=focus,
        repo_root=args.repo_root,
        hinerv_archive_size_ladder_report=hinerv_archive_size_ladder_report,
        hinerv_archive_ladder_waterfill_report=(
            hinerv_archive_ladder_waterfill_report
        ),
        hinerv_archive_ladder_replay_actuator_report=(
            hinerv_archive_ladder_replay_actuator_report
        ),
        hinerv_archive_backend_drift_report=hinerv_archive_backend_drift_report,
        snerv_trained_ladder_waterfill_report=(
            snerv_trained_ladder_waterfill_report
        ),
        hinerv_decoder_weight_saliency_report=(
            hinerv_decoder_weight_saliency_report
        ),
        snerv_waterfill_mode_assignment_report=(
            snerv_waterfill_mode_assignment_report
        ),
        snerv_decoder_mode_probe_report=snerv_decoder_mode_probe_report,
        snerv_scorer_loop_qat_reports=snerv_scorer_loop_qat_reports,
        snerv_lf_payload_codec_sweep_reports=snerv_lf_payload_codec_sweep_reports,
    )
    output = Path(args.output_json).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = output.as_posix()
    if args.output_md:
        md_output = Path(args.output_md).expanduser().resolve(strict=False)
        md_output.parent.mkdir(parents=True, exist_ok=True)
        report["markdown_report_path"] = md_output.as_posix()
    write_json(output, report)
    if args.output_md:
        md_output.write_text(
            render_nerv_control_inventory_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(_summary(report), sort_keys=True))
    return 0


def _load_optional_report(
    path: Path | None,
    *,
    pattern: str,
    schema: str,
) -> dict[str, Any] | None:
    if path is not None:
        return _load_json(path)
    for candidate in sorted(RESEARCH_DIR.glob(pattern), reverse=True):
        payload = _load_json(candidate)
        if payload.get("schema") == schema:
            payload.setdefault("source_artifact_path", candidate.as_posix())
            return payload
    return None


def _load_optional_reports(
    paths: list[Path] | None,
    *,
    pattern: str,
    schema: str,
) -> list[dict[str, Any]]:
    if paths:
        return [_load_json(path) for path in paths]
    reports: list[dict[str, Any]] = []
    for candidate in sorted(RESEARCH_DIR.glob(pattern), reverse=True):
        payload = _load_json(candidate)
        if payload.get("schema") == schema:
            payload.setdefault("source_artifact_path", candidate.as_posix())
            reports.append(payload)
    return reports


def _checkpoint_export_lf_payload_reports(
    paths: list[Path] | None,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths or ():
        payload = _load_json(path)
        reports.append(
            build_snerv_lf_payload_codec_report_from_checkpoint_export(
                payload,
                source_artifact_path=path,
            )
        )
    return reports


def _load_json(path: Path) -> dict[str, Any]:
    source = path.expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{source}: expected JSON object")
    payload.setdefault("source_artifact_path", source.as_posix())
    payload.setdefault("source_artifact_bytes", len(raw))
    payload.setdefault("source_artifact_sha256", hashlib.sha256(raw).hexdigest())
    return payload


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "report_path": report.get("report_path"),
        "focus_families": report["focus_families"],
        "control_count": len(report["control_rows"]),
        "binding_gap_count": len(report["binding_gap_rows"]),
        "work_order_count": len(report["recommended_next_work_orders"]),
        "snerv_lf_payload_codec_sweep_report_count": report.get(
            "snerv_lf_payload_codec_sweep_reports", {}
        )
        .get("snerv", {})
        .get("history_count", 0),
        "implementation_sweep_status": report["implementation_sweep"]["status"],
        "score_claim": report["score_claim"],
        "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
