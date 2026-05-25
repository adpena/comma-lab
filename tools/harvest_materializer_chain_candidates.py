#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest completed materializer chain manifests into an optimizer source queue.

This is a custody bridge, not a dispatcher. It scans explicit chain manifests,
materializer work-queue postconditions, or bounded chain roots; validates live
archive/artifact bytes; and writes a planning-only optimizer candidate queue.
Exact-ready promotion remains a separate explicit gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comma_lab.scheduler.materializer_chain_harvest import (  # noqa: E402
    harvest_materializer_chain_manifests,
    run_exact_readiness_bridge_for_harvested_queue,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-queue", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--chain-manifest", type=Path, action="append", default=[])
    parser.add_argument("--chain-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--renderer-payload-dfl1-inflate-parity-proof",
        type=Path,
        action="append",
        default=[],
        help=(
            "Attach and re-verify a shell_inflate_parity_proof_v2 sidecar for "
            "renderer_payload_dfl1 rows before exact-readiness bridging."
        ),
    )
    parser.add_argument(
        "--allowed-artifact-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional non-repo root allowed for explicit sidecar artifacts, "
            "for example an external SSD workload root used by the scheduler."
        ),
    )
    parser.add_argument("--source-queue-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--exact-readiness-out-dir",
        type=Path,
        default=None,
        help=(
            "Explicitly run the exact-readiness promoter for harvested source "
            "queue rows and write per-candidate reports/ready queues here. "
            "This does not dispatch or claim score authority."
        ),
    )
    parser.add_argument(
        "--exact-readiness-bridge-report-out",
        type=Path,
        default=None,
        help="Write the aggregate exact-readiness bridge report when enabled.",
    )
    parser.add_argument(
        "--exact-readiness-candidate-id",
        action="append",
        default=[],
        help="Optional candidate id filter for the exact-readiness bridge.",
    )
    parser.add_argument(
        "--exact-readiness-allow-source-blocker",
        action="append",
        default=[],
        help=(
            "Additional source dispatch_blocker string the bridge may clear. "
            "Fails closed unless the scheduler allowlist permits it and an "
            "operator override reason is supplied."
        ),
    )
    parser.add_argument(
        "--exact-readiness-dispatch-claims-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--exact-readiness-claim-ttl-hours", type=float, default=24.0)
    parser.add_argument(
        "--exact-readiness-allow-above-active-floor-dispatch",
        action="store_true",
    )
    parser.add_argument("--exact-readiness-operator-override-reason", default=None)
    parser.add_argument(
        "--exact-readiness-require-ready",
        action="store_true",
        help="Exit nonzero if the explicit exact-readiness bridge yields no ready rows.",
    )
    parser.add_argument(
        "--allow-unfinished-state",
        action="store_true",
        help=(
            "Do not require succeeded experiment_queue state for work-queue "
            "postcondition rows. Manifest custody is still validated."
        ),
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Exit nonzero when no chain manifests validate into source queue rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated JSON outputs. Defaults fail closed.",
    )
    args = parser.parse_args(argv)

    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be >= 1 when provided")
    if (
        args.work_queue is None
        and not args.chain_manifest
        and not args.chain_root
    ):
        raise SystemExit(
            "provide at least one of --work-queue, --chain-manifest, or --chain-root"
        )
    bridge_option_without_bridge = (
        args.exact_readiness_out_dir is None
        and (
            args.exact_readiness_bridge_report_out is not None
            or bool(args.exact_readiness_candidate_id)
            or bool(args.exact_readiness_allow_source_blocker)
            or args.exact_readiness_dispatch_claims_path is not None
            or args.exact_readiness_allow_above_active_floor_dispatch
            or args.exact_readiness_operator_override_reason is not None
            or args.exact_readiness_require_ready
        )
    )
    if bridge_option_without_bridge:
        raise SystemExit(
            "--exact-readiness-out-dir is required when using exact-readiness bridge options"
        )

    result = harvest_materializer_chain_manifests(
        repo_root=args.repo_root,
        work_queue_path=args.work_queue,
        experiment_queue_state_path=args.state,
        experiment_queue_id=args.queue_id,
        chain_manifest_paths=args.chain_manifest,
        chain_roots=args.chain_root,
        renderer_payload_dfl1_inflate_parity_proofs=(
            args.renderer_payload_dfl1_inflate_parity_proof
        ),
        allowed_artifact_roots=args.allowed_artifact_root,
        require_succeeded_state=not args.allow_unfinished_state,
        top_k=args.top_k,
    )
    write_json(args.source_queue_out, result["source_queue"], overwrite=args.overwrite)
    write_json(args.report_out, result["report"], overwrite=args.overwrite)
    report = result["report"]
    bridge_report = None
    if args.exact_readiness_out_dir is not None:
        bridge_report = run_exact_readiness_bridge_for_harvested_queue(
            repo_root=args.repo_root,
            source_queue_path=args.source_queue_out,
            exact_readiness_out_dir=args.exact_readiness_out_dir,
            candidate_ids=args.exact_readiness_candidate_id,
            allow_source_blockers=args.exact_readiness_allow_source_blocker,
            dispatch_claims_path=args.exact_readiness_dispatch_claims_path,
            claim_ttl_hours=args.exact_readiness_claim_ttl_hours,
            allow_above_active_floor_dispatch=(
                args.exact_readiness_allow_above_active_floor_dispatch
            ),
            operator_override_reason=args.exact_readiness_operator_override_reason,
        )
        if args.exact_readiness_bridge_report_out is not None:
            write_json(
                args.exact_readiness_bridge_report_out,
                bridge_report,
                overwrite=args.overwrite,
            )
    print(
        f"harvested {report['accepted_manifest_count']}/"
        f"{report['unique_manifest_count']} materializer chain manifest(s); "
        f"source_queue={args.source_queue_out}; report={args.report_out}; "
        f"dispatch_ready={report['source_queue_dispatch_ready_count']}"
    )
    if bridge_report is not None:
        print(
            "exact-readiness bridge: "
            f"ready={bridge_report['ready_candidate_count']}/"
            f"{bridge_report['candidate_count']}; "
            f"out_dir={args.exact_readiness_out_dir}"
        )
    if args.require_accepted and int(report["accepted_manifest_count"]) < 1:
        return 2
    if (
        args.exact_readiness_require_ready
        and bridge_report is not None
        and int(bridge_report["ready_candidate_count"]) < 1
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
