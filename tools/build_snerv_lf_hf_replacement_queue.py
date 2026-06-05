#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the false-authority SNeRV LF/HF replacement queue."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_lf_hf_replacement_queue import (  # noqa: E402
    DEFAULT_LANE_ID,
    DEFAULT_MIN_FREE_BYTES,
    build_snerv_lf_hf_replacement_queue,
    load_json_with_source_identity,
    render_snerv_lf_hf_replacement_queue_markdown,
)
from tac.repo_io import write_json_artifact, write_text_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lf-payload-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "Measured SNeRV LF payload report JSON. Accepts codec sweep, "
            "receiver recode, or recode-admission reports. Repeatable."
        ),
    )
    parser.add_argument(
        "--reroute-queue",
        action="append",
        default=[],
        type=Path,
        help="Existing snerv_lf_over_ceiling_reroute_queue.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--campaign-plan",
        action="append",
        default=[],
        type=Path,
        help="Current nerv_long_training_campaign_plan.v1 JSON. Repeatable.",
    )
    parser.add_argument(
        "--source-forward-artifact",
        action="append",
        default=[],
        type=Path,
        help=(
            "Current snerv_official_mfu_hfr_tub_forward_parity.v1 JSON. "
            "Repeatable; used to clear only proven receiver-frame replay blockers."
        ),
    )
    parser.add_argument(
        "--official-replacement-authority-gate",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1 "
            "JSON. Repeatable; used to admit or fail-close the official "
            "TUB LF/HF decoder replacement row."
        ),
    )
    parser.add_argument(
        "--candidate-feedback-row",
        action="append",
        default=[],
        type=Path,
        help=(
            "Current nerv_candidate_feedback_row.v1 JSON. Repeatable; used to "
            "clear only proven scorer-domain tether guard blockers."
        ),
    )
    parser.add_argument(
        "--value-domain-xray",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_receiver_value_domain_xray.v1 JSON. Repeatable; used to "
            "clear only proven LF-conditioned HF noncollapse blockers."
        ),
    )
    parser.add_argument(
        "--hf-residual-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_lf_conditioned_hf_residual_receiver_proof.v1 JSON. "
            "Repeatable; used to clear only the HF residual receiver payload "
            "implementation blocker."
        ),
    )
    parser.add_argument(
        "--joint-codebook-receiver-payload-proof",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1 JSON. "
            "Repeatable; used to clear only joint codebook implementation, "
            "NumPy receiver, and section-byte telemetry blockers."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--queue-id", default="snerv_lf_hf_replacement_queue.v1")
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument(
        "--allow-local-output",
        action="store_true",
        help="Allow non-SSD output root. Intended only for tests.",
    )
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = (
        args.output_root
        or Path("/Volumes/VertigoDataTier/pact")
        / f"snerv_lf_hf_replacement_queue_{stamp}"
    )
    output_json = args.output_json or output_root / "snerv_lf_hf_replacement_queue.json"
    output_md = args.output_md or output_root / "snerv_lf_hf_replacement_queue.md"
    lf_reports = [load_json_with_source_identity(path) for path in args.lf_payload_report]
    reroute_queues = [load_json_with_source_identity(path) for path in args.reroute_queue]
    campaign_plans = [load_json_with_source_identity(path) for path in args.campaign_plan]
    source_forward_artifacts = [
        load_json_with_source_identity(path) for path in args.source_forward_artifact
    ]
    official_replacement_authority_gates = [
        load_json_with_source_identity(path)
        for path in args.official_replacement_authority_gate
    ]
    candidate_feedback_rows = [
        load_json_with_source_identity(path) for path in args.candidate_feedback_row
    ]
    value_domain_xray_reports = [
        load_json_with_source_identity(path) for path in args.value_domain_xray
    ]
    hf_residual_receiver_payload_proofs = [
        load_json_with_source_identity(path)
        for path in args.hf_residual_receiver_payload_proof
    ]
    joint_codebook_receiver_payload_proofs = [
        load_json_with_source_identity(path)
        for path in args.joint_codebook_receiver_payload_proof
    ]

    report = build_snerv_lf_hf_replacement_queue(
        lf_payload_reports=lf_reports,
        reroute_queues=reroute_queues,
        campaign_plans=campaign_plans,
        source_forward_artifacts=source_forward_artifacts,
        official_replacement_authority_gates=official_replacement_authority_gates,
        candidate_feedback_rows=candidate_feedback_rows,
        value_domain_xray_reports=value_domain_xray_reports,
        hf_residual_receiver_payload_proofs=hf_residual_receiver_payload_proofs,
        joint_codebook_receiver_payload_proofs=joint_codebook_receiver_payload_proofs,
        output_root=output_root,
        lane_id=str(args.lane_id),
        queue_id=str(args.queue_id),
        min_free_bytes=int(args.min_free_bytes),
        allow_local_output=bool(args.allow_local_output),
    )
    json_result = write_json_artifact(output_json, report)
    md_result = write_text_artifact(
        output_md,
        render_snerv_lf_hf_replacement_queue_markdown(report),
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "lane_id": report["lane_id"],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_md": md_result.path,
                "output_md_sha256": md_result.sha256,
                "queue_row_count": report["queue_row_count"],
                "blocked_queue_row_count": report["blocked_queue_row_count"],
                "local_executable_command_row_count": report[
                    "local_executable_command_row_count"
                ],
                "selected_lf_payload_bytes": (
                    None
                    if report.get("selected_lf_payload_evidence") is None
                    else report["selected_lf_payload_evidence"].get("lf_payload_bytes")
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
