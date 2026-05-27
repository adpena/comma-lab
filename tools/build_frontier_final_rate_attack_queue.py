#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned final-rate attack against frontier archives.

The default path resolves the current canonical contest-CPU frontier archive,
accepts any explicit comparison archives, and emits an ``experiment_queue.v1``
that runs family-agnostic materializer sweeps locally. It is local/advisory
only; exact auth eval is still required before any score or promotion claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import default_state_path  # noqa: E402
from comma_lab.scheduler.frontier_final_rate_attack_autoloop import (  # noqa: E402
    execute_post_feedback_child_queues,
)
from comma_lab.scheduler.frontier_rate_attack_bootstrap import (  # noqa: E402
    DEFAULT_EXECUTABLE_TARGET_KINDS,
    DEFAULT_FRONTIER_POINTER,
    DEFAULT_OPTIONAL_TARGET_KINDS,
    FrontierRateAttackBootstrapError,
    build_frontier_rate_attack_payloads,
    derive_archive_section_recode_manifests,
    derive_packet_member_merge_contract,
    parse_archive_spec,
    resolve_current_frontier_archive,
)
from comma_lab.scheduler.frontier_rate_attack_target_profile import (  # noqa: E402
    FrontierRateAttackTargetProfileError,
    build_frontier_target_optimization_profile,
)
from tac.optimization.inverse_steganalysis_acquisition import (  # noqa: E402
    FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_KIND,
    MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
    observations_from_queue_observation,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_line,
    json_text,
    sha256_bytes,
    write_json_artifact,
    write_text_artifact,
)

SIGNAL_HARVEST_SCHEMA = "frontier_final_rate_attack_signal_harvest.v1"
QUEUE_OBSERVER_SIGNAL_AXIS = "[local-final-rate-attack-queue advisory]"


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _default_results_root() -> Path:
    for root in (
        Path("/Volumes/VertigoDataTier/experiments/results"),
        Path("/Volumes/APDataStore/experiments/results"),
    ):
        if _can_create_child(root):
            return root
    return REPO_ROOT / "experiments" / "results"


def _can_create_child(root: Path) -> bool:
    probe = root / f".pact_write_read_probe_{os.getpid()}"
    payload = '{"probe":"frontier_final_rate_attack_results_root"}\n'
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_text(payload, encoding="utf-8")
        if probe.read_text(encoding="utf-8") != payload:
            return False
    except OSError:
        return False
    finally:
        try:
            if probe.exists():
                probe.unlink()
        except OSError:
            pass
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--archive", action="append", default=[], help="Explicit label=path archive.")
    parser.add_argument("--frontier-axis", default="contest_cpu")
    parser.add_argument("--frontier-pointer", default=DEFAULT_FRONTIER_POINTER)
    parser.add_argument("--request-search-root", action="append", default=[])
    parser.add_argument("--archive-search-root", action="append", default=[])
    parser.add_argument(
        "--no-current-frontier",
        dest="include_current_frontier",
        action="store_false",
        help="Do not resolve and include the canonical current frontier archive.",
    )
    parser.set_defaults(include_current_frontier=True)
    parser.add_argument("--target-kind", action="append", default=None)
    parser.add_argument(
        "--no-optional-target-blockers",
        dest="include_optional_target_blockers",
        action="store_false",
        help="Do not emit typed blockers for section/tensor targets lacking manifests.",
    )
    parser.set_defaults(include_optional_target_blockers=True)
    parser.add_argument("--member-name", default=None)
    parser.add_argument("--section-manifest", default=None)
    parser.add_argument(
        "--derive-section-manifests",
        dest="derive_section_manifests",
        action="store_true",
        help=(
            "Derive per-archive parser-section manifests for archive_section_entropy_recode_v1 "
            "instead of requiring a hand-supplied single manifest."
        ),
    )
    parser.add_argument(
        "--no-derive-section-manifests",
        dest="derive_section_manifests",
        action="store_false",
        help="Keep archive_section_entropy_recode_v1 as a typed blocker unless --section-manifest is supplied.",
    )
    parser.set_defaults(derive_section_manifests=True)
    parser.add_argument(
        "--section-manifest-parser",
        default="auto",
        help="Parser name passed to derived HNeRV section-manifest generation.",
    )
    parser.add_argument("--section-name", action="append", default=[])
    parser.add_argument("--merge-contract", default=None)
    parser.add_argument(
        "--derive-merge-contracts",
        dest="derive_merge_contracts",
        action="store_true",
        help=(
            "Derive an all-members packet_member_merge_v1 contract for every "
            "archive with at least two ZIP members."
        ),
    )
    parser.add_argument(
        "--no-derive-merge-contracts",
        dest="derive_merge_contracts",
        action="store_false",
        help="Do not derive a packet_member_merge_v1 contract automatically.",
    )
    parser.set_defaults(derive_merge_contracts=True)
    parser.add_argument("--merged-member-name", default=None)
    parser.add_argument("--payload-member-name", default="p")
    parser.add_argument("--full-frame-inflate-parity-proof", default=None)
    parser.add_argument("--tensor-manifest", default=None)
    parser.add_argument("--factorization-contract", default=None)
    parser.add_argument("--tensor-factorize-rank", type=int, default=None)
    parser.add_argument("--zip-compression-method", action="append", default=[])
    parser.add_argument("--zip-compresslevel", action="append", type=int, default=[])
    parser.add_argument("--target-profile-id", default="contest_video_0")
    parser.add_argument(
        "--target-mode",
        default="contest_video_overfit",
        help=(
            "Optimization target mode: contest_video_overfit, "
            "corpus_generalization, or hybrid_contest_plus_corpus."
        ),
    )
    parser.add_argument(
        "--target-video",
        action="append",
        default=[],
        help=(
            "Declared optimization target video path. Defaults to the canonical "
            "contest video when omitted."
        ),
    )
    parser.add_argument("--target-corpus-manifest", default=None)
    parser.add_argument(
        "--allow-unready-target-profile",
        action="store_true",
        help=(
            "Emit blocker-bearing planning artifacts even if the declared target "
            "profile is missing its video/corpus inputs."
        ),
    )
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument(
        "--allow-materializer-overwrite",
        action="store_true",
        help="Pass --allow-overwrite to materializer sweep outputs.",
    )
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument(
        "--include-exact-readiness-followup",
        dest="include_exact_readiness_followup",
        action="store_true",
        help=(
            "Append harvest/exact-readiness/dispatch-plan follow-up steps after "
            "harvestable materializer rows. This is the default for the final-rate "
            "attack entry point."
        ),
    )
    parser.add_argument(
        "--no-exact-readiness-followup",
        dest="include_exact_readiness_followup",
        action="store_false",
        help=(
            "Emit only the materializer execution queue, without chained harvest/"
            "exact-readiness/dispatch-plan handoff."
        ),
    )
    parser.set_defaults(include_exact_readiness_followup=True)
    parser.add_argument(
        "--exact-readiness-followup-require-ready",
        action="store_true",
        help="Make exact-readiness follow-up harvest fail unless a ready candidate is emitted.",
    )
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--post-execute-feedback-refresh",
        dest="post_execute_feedback_refresh",
        action="store_true",
        help=(
            "After --execute observer harvest, immediately fold materializer signal "
            "back into the canonical frontier feedback/planner artifacts."
        ),
    )
    parser.add_argument(
        "--skip-post-execute-feedback-refresh",
        dest="post_execute_feedback_refresh",
        action="store_false",
        help="Do not run the post-execute frontier feedback refresh.",
    )
    parser.set_defaults(post_execute_feedback_refresh=True)
    parser.add_argument(
        "--post-execute-feedback-action-summary",
        default="none",
        help=(
            "Action summary passed to the post-execute feedback refresh. Defaults "
            "to none so final-rate signal ingestion is bounded to this queue run."
        ),
    )
    parser.add_argument("--post-execute-feedback-candidate-limit", type=int, default=4)
    parser.add_argument("--post-execute-feedback-max-files-per-root", type=int, default=64)
    parser.add_argument(
        "--post-execute-feedback-retention-plan",
        action="store_true",
        help=(
            "Include raw/MLX retention planning in the post-execute feedback refresh. "
            "The default keeps this ingestion pass focused on queue signal."
        ),
    )
    parser.add_argument(
        "--execute-post-feedback-queues",
        dest="execute_post_feedback_queues",
        action="store_true",
        help=(
            "After feedback refresh, run a bounded set of generated local child "
            "queues and preserve their observer artifacts."
        ),
    )
    parser.add_argument(
        "--skip-execute-post-feedback-queues",
        dest="execute_post_feedback_queues",
        action="store_false",
        help="Do not automatically run post-feedback child queues.",
    )
    parser.set_defaults(execute_post_feedback_queues=True)
    parser.add_argument("--post-feedback-child-queue-limit", type=int, default=4)
    parser.add_argument("--post-feedback-child-queue-max-steps", type=int, default=8)
    parser.add_argument("--post-feedback-child-queue-max-parallel", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.05)
    parser.add_argument("--idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-cycles", type=int, default=1)
    return parser.parse_args(argv)


def _run_command(command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _write_outputs(output_dir: Path, payloads: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bootstrap": output_dir / "frontier_rate_attack_bootstrap.json",
        "materializer_contexts": output_dir / "materializer_contexts.json",
        "materializer_backlog": output_dir / "materializer_backlog.json",
        "materializer_work_queue": output_dir / "materializer_work_queue.json",
        "experiment_queue": output_dir / "experiment_queue.json",
    }
    if "derived_section_manifests" in payloads:
        paths["derived_section_manifests"] = output_dir / "derived_section_manifests.json"
    if "derived_packet_member_merge_contract" in payloads:
        paths["derived_packet_member_merge_contract"] = (
            output_dir / "derived_packet_member_merge_contract.json"
        )
    write_json_artifact(paths["materializer_contexts"], payloads["contexts"])
    write_json_artifact(paths["materializer_backlog"], payloads["backlog"])
    write_json_artifact(paths["materializer_work_queue"], payloads["work_queue"])
    write_json_artifact(paths["experiment_queue"], payloads["queue"])
    if "derived_section_manifests" in payloads:
        write_json_artifact(paths["derived_section_manifests"], payloads["derived_section_manifests"])
    if "derived_packet_member_merge_contract" in payloads:
        write_json_artifact(
            paths["derived_packet_member_merge_contract"],
            payloads["derived_packet_member_merge_contract"],
        )
    bootstrap = dict(payloads["bootstrap"])
    bootstrap["artifacts"] = {key: _display_path(path) for key, path in paths.items() if key != "bootstrap"}
    bootstrap["operator_commands"] = {
        "validate": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "validate",
        ],
        "init": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "init",
        ],
        "run_worker": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "run-worker",
            "--execute",
            "--max-steps",
            "<N>",
            "--max-parallel",
            "<P>",
        ],
        "refresh_feedback_after_execute": _post_execute_feedback_refresh_command(
            queue_id=str(bootstrap.get("queue_id") or "frontier_final_rate_attack"),
            output_dir=output_dir,
            results_root=Path(str(bootstrap.get("results_root") or "experiments/results")),
            signal_harvest_path=output_dir / "final_rate_attack_signal_harvest.json",
            action_summary="none",
            candidate_limit=4,
            local_cpu_concurrency=1,
            max_files_per_root=64,
            include_retention_plan=False,
        ),
    }
    write_json_artifact(paths["bootstrap"], bootstrap)
    return {key: _display_path(path) for key, path in paths.items()}


def _sha256_or_fallback(value: Any, fallback: str) -> str:
    if isinstance(value, str) and len(value) == 64:
        lowered = value.lower()
        if all(item in "0123456789abcdef" for item in lowered):
            return lowered
    return fallback


def _json_payload_from_stdout(result: dict[str, Any], *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(result.get("stdout") or ""))
    except json.JSONDecodeError as exc:
        raise FrontierRateAttackBootstrapError(
            f"{label} command returned invalid JSON stdout: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FrontierRateAttackBootstrapError(f"{label} command stdout must be a JSON object")
    return payload


def _signal_counts_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            value = "none"
        text = str(value)
        counts[text] = counts.get(text, 0) + 1
    return dict(sorted(counts.items()))


def _build_signal_harvest(
    *,
    observation: dict[str, Any],
    observations: list[dict[str, Any]],
    observer_path: Path,
    signal_path: Path,
) -> dict[str, Any]:
    materializer_kinds = {
        FAMILY_AGNOSTIC_MATERIALIZER_EMPIRICAL_OBSERVATION_KIND,
        MATERIALIZER_ARCHIVE_DELTA_OBSERVATION_KIND,
    }
    materializer_rows = [
        row for row in observations if str(row.get("observation_kind") or "") in materializer_kinds
    ]
    saved_values = [
        int(row.get("saved_bytes") or 0)
        for row in materializer_rows
        if row.get("saved_bytes") is not None
    ]
    rate_positive_rows = [row for row in materializer_rows if row.get("rate_positive") is True]
    rate_nonpositive_rows = [
        row for row in materializer_rows if row.get("rate_positive") is not True
    ]
    materializer_blocking_rows = [
        row
        for row in materializer_rows
        if row.get("receiver_contract_satisfied") is not True
        or (row.get("rate_positive") is not True and row.get("quality_spend_allowed") is not True)
    ]
    return {
        "schema": SIGNAL_HARVEST_SCHEMA,
        "queue_id": observation.get("queue_id"),
        "observer_revalidation_path": _display_path(observer_path),
        "signal_observations_path": _display_path(signal_path),
        "queue_healthy": observation.get("healthy") is True,
        "queue_status_counts": dict(observation.get("status_counts") or {}),
        "queue_blockers": list(observation.get("blockers") or []),
        "observation_count": len(observations),
        "observation_kind_counts": _signal_counts_by_key(observations, "observation_kind"),
        "materializer_observation_count": len(materializer_rows),
        "materializer_target_kind_counts": _signal_counts_by_key(
            materializer_rows,
            "target_kind",
        ),
        "materializer_rate_positive_count": len(rate_positive_rows),
        "materializer_rate_nonpositive_count": len(rate_nonpositive_rows),
        "materializer_blocking_feedback_count": len(materializer_blocking_rows),
        "materializer_saved_bytes_sum": sum(saved_values),
        "materializer_saved_bytes_max": max(saved_values) if saved_values else None,
        "materializer_archive_delta_status_counts": _signal_counts_by_key(
            materializer_rows,
            "archive_delta_status",
        ),
        "allowed_use": "local_final_rate_attack_planning_feedback_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_execution_observer_signal_artifacts(
    *,
    output_dir: Path,
    observe_result: dict[str, Any],
    min_free_bytes: int = 0,
) -> dict[str, Any]:
    observation = _json_payload_from_stdout(observe_result, label="experiment queue observe")
    observation_sha256 = sha256_bytes(json_text(observation).encode("utf-8"))
    queue_sha256 = _sha256_or_fallback(observation.get("queue_sha256"), observation_sha256)
    observer_path = output_dir / "observer_revalidation.json"
    signal_path = output_dir / "materializer_signal_observations.jsonl"
    harvest_path = output_dir / "final_rate_attack_signal_harvest.json"
    write_json_artifact(
        observer_path,
        observation,
        min_free_bytes=min_free_bytes,
    )
    observations = observations_from_queue_observation(
        observation,
        runtime_identity={
            "runtime_contract_sha256": queue_sha256,
            "runtime_contract_kind": "experiment_queue_definition",
        },
        cache_identity={
            "cache_sha256": observation_sha256,
            "cache_key": f"{observation.get('queue_id')}:{observation_sha256}",
        },
        axis=QUEUE_OBSERVER_SIGNAL_AXIS,
        source_path=_display_path(observer_path),
    )
    write_text_artifact(
        signal_path,
        "".join(json_line(row) for row in observations),
        min_free_bytes=min_free_bytes,
    )
    harvest = _build_signal_harvest(
        observation=observation,
        observations=observations,
        observer_path=observer_path,
        signal_path=signal_path,
    )
    write_json_artifact(harvest_path, harvest, min_free_bytes=min_free_bytes)
    return {
        "harvest": harvest,
        "artifacts": {
            "observer_revalidation": _display_path(observer_path),
            "materializer_signal_observations": _display_path(signal_path),
            "final_rate_attack_signal_harvest": _display_path(harvest_path),
        },
    }


def _post_execute_feedback_refresh_command(
    *,
    queue_id: str,
    output_dir: Path,
    results_root: Path,
    signal_harvest_path: Path,
    action_summary: str,
    candidate_limit: int,
    local_cpu_concurrency: int,
    max_files_per_root: int,
    include_retention_plan: bool,
) -> list[str]:
    command = [
        ".venv/bin/python",
        "tools/build_frontier_rate_attack_feedback_refresh.py",
        "--queue-id",
        f"{queue_id}_post_execute_feedback",
        "--output-dir",
        _display_path(output_dir / "post_execute_feedback_refresh"),
        "--results-root",
        _display_path(results_root),
        "--action-summary",
        action_summary,
        "--materializer-feedback",
        _display_path(signal_harvest_path),
        "--frontier-artifact-root",
        _display_path(signal_harvest_path),
        "--local-cpu-eureka-root",
        _display_path(signal_harvest_path),
        "--candidate-limit",
        str(candidate_limit),
        "--local-cpu-concurrency",
        str(local_cpu_concurrency),
        "--local-io-concurrency",
        "1",
        "--max-files-per-root",
        str(max_files_per_root),
    ]
    if not include_retention_plan:
        command.extend(["--skip-raw-retention-plan", "--skip-mlx-retention-plan"])
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = _utc_stamp()
    queue_id = args.queue_id or f"frontier_final_rate_attack_{stamp}"
    output_dir = args.output_dir or (
        REPO_ROOT / ".omx" / "research" / f"frontier_final_rate_attack_{stamp}"
    )
    results_root = args.results_root or (_default_results_root() / "frontier_final_rate_attack")
    try:
        archive_records: list[dict[str, Any]] = []
        frontier_resolution = None
        if args.include_current_frontier:
            frontier_resolution = resolve_current_frontier_archive(
                repo_root=REPO_ROOT,
                frontier_axis=args.frontier_axis,
                pointer_path=args.frontier_pointer,
                request_search_roots=tuple(args.request_search_root),
                archive_search_roots=tuple(args.archive_search_root),
            )
            archive_records.append(dict(frontier_resolution["archive_record"]))
        for spec in args.archive:
            archive_records.append(parse_archive_spec(spec, repo_root=REPO_ROOT))
        target_optimization_profile = build_frontier_target_optimization_profile(
            repo_root=REPO_ROOT,
            target_profile_id=args.target_profile_id,
            target_mode=args.target_mode,
            target_video_paths=tuple(args.target_video),
            target_corpus_manifest_path=args.target_corpus_manifest,
        )
        target_kinds = tuple(args.target_kind or DEFAULT_EXECUTABLE_TARGET_KINDS)
        derived_section_manifests = None
        derived_merge_contract = None
        section_manifest_by_archive_label: dict[str, str] = {}
        section_names_by_archive_label: dict[str, tuple[str, ...]] = {}
        wants_section_targets = (
            "archive_section_entropy_recode_v1" in set(target_kinds)
            or ("archive_section_entropy_recode_v1" in set(DEFAULT_OPTIONAL_TARGET_KINDS)
            and args.include_optional_target_blockers)
        )
        if args.derive_section_manifests and args.section_manifest is None and wants_section_targets:
            derived_section_manifests = derive_archive_section_recode_manifests(
                archive_records=archive_records,
                output_dir=output_dir / "derived_section_manifests",
                repo_root=REPO_ROOT,
                parser=args.section_manifest_parser,
                allow_overwrite=args.allow_materializer_overwrite,
                min_free_bytes=args.min_free_bytes,
            )
            for row in derived_section_manifests["rows"]:
                label = str(row.get("archive_label") or "")
                manifest_path = row.get("section_manifest_path")
                if label and isinstance(manifest_path, str) and manifest_path:
                    section_manifest_by_archive_label[label] = manifest_path
                    section_names_by_archive_label[label] = tuple(
                        str(name)
                        for name in row.get("selected_section_names") or []
                        if str(name)
                    )
        merge_contract = args.merge_contract
        wants_merge_target = "packet_member_merge_v1" in set(target_kinds)
        if (
            args.derive_merge_contracts
            and merge_contract is None
            and wants_merge_target
        ):
            derived_merge_contract = derive_packet_member_merge_contract(
                archive_records=archive_records,
                output_path=output_dir
                / "derived_packet_member_merge_contracts"
                / "merge_contract.json",
                repo_root=REPO_ROOT,
                merged_member_name=args.merged_member_name,
                zip_compression_methods=tuple(
                    args.zip_compression_method or ("stored", "deflated")
                ),
                zip_compresslevels=tuple(args.zip_compresslevel or (1, 6, 9)),
                allow_overwrite=args.allow_materializer_overwrite,
                min_free_bytes=args.min_free_bytes,
            )
            merge_contract = str(derived_merge_contract["merge_contract_path"])
        payloads = build_frontier_rate_attack_payloads(
            repo_root=REPO_ROOT,
            queue_id=queue_id,
            archive_records=archive_records,
            results_root=results_root,
            target_kinds=target_kinds,
            include_optional_target_blockers=args.include_optional_target_blockers,
            member_name=args.member_name,
            section_manifest=args.section_manifest,
            section_manifest_by_archive_label=section_manifest_by_archive_label,
            section_names=tuple(args.section_name),
            section_names_by_archive_label=section_names_by_archive_label,
            merge_contract=merge_contract,
            merged_member_name=args.merged_member_name,
            payload_member_name=args.payload_member_name,
            full_frame_inflate_parity_proof=args.full_frame_inflate_parity_proof,
            tensor_manifest=args.tensor_manifest,
            factorization_contract=args.factorization_contract,
            tensor_factorize_rank=args.tensor_factorize_rank,
            zip_compression_methods=tuple(args.zip_compression_method or ("stored", "deflated")),
            zip_compresslevels=tuple(args.zip_compresslevel or (1, 6, 9)),
            min_free_bytes=args.min_free_bytes,
            allow_overwrite=args.allow_materializer_overwrite,
            local_cpu_concurrency=args.local_cpu_concurrency,
            lane_id=f"lane_{queue_id}",
            source_work_queue_path=output_dir / "materializer_work_queue.json",
            include_exact_readiness_followup=args.include_exact_readiness_followup,
            exact_readiness_followup_require_ready=(
                args.exact_readiness_followup_require_ready
            ),
            target_optimization_profile=target_optimization_profile,
            require_target_profile_ready=not args.allow_unready_target_profile,
        )
        if frontier_resolution is not None:
            payloads["bootstrap"]["frontier_resolution"] = frontier_resolution
        if derived_section_manifests is not None:
            payloads["derived_section_manifests"] = derived_section_manifests
            payloads["bootstrap"]["derived_section_manifests"] = {
                "schema": derived_section_manifests["schema"],
                "manifest_count": derived_section_manifests["manifest_count"],
                "ready_manifest_count": derived_section_manifests["ready_manifest_count"],
                "artifact": _display_path(output_dir / "derived_section_manifests.json"),
            }
        if derived_merge_contract is not None:
            payloads["derived_packet_member_merge_contract"] = derived_merge_contract
            payloads["bootstrap"]["derived_packet_member_merge_contract"] = {
                "schema": derived_merge_contract["schema"],
                "ready_archive_count": derived_merge_contract["ready_archive_count"],
                "archive_count": derived_merge_contract["archive_count"],
                "artifact": _display_path(
                    output_dir / "derived_packet_member_merge_contract.json"
                ),
            }
        artifact_paths = _write_outputs(output_dir, payloads)
        queue_path = Path(artifact_paths["experiment_queue"])
        if not queue_path.is_absolute():
            queue_path = REPO_ROOT / queue_path
        execution_report: dict[str, Any] | None = None
        if args.execute:
            state_path = default_state_path(REPO_ROOT, queue_id)
            commands = [
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "init",
                ],
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "run-worker",
                    "--execute",
                    "--max-steps",
                    str(args.max_steps),
                    "--max-parallel",
                    str(args.max_parallel),
                    "--poll-interval-seconds",
                    str(args.poll_interval_seconds),
                    "--idle-sleep-seconds",
                    str(args.idle_sleep_seconds),
                    "--max-idle-cycles",
                    str(args.max_idle_cycles),
                ],
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "observe",
                    "--format",
                    "json",
                ],
            ]
            results = [_run_command(command) for command in commands]
            observe_result = results[-1]
            execution_report = {
                "schema": "frontier_final_rate_attack_execution_report.v1",
                "queue_id": queue_id,
                "state_path": _display_path(state_path),
                "commands": results,
                "failed_command_count": 0,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            if observe_result["returncode"] == 0:
                observer_signal = _write_execution_observer_signal_artifacts(
                    output_dir=output_dir,
                    observe_result=observe_result,
                    min_free_bytes=args.min_free_bytes,
                )
                artifact_paths.update(observer_signal["artifacts"])
                execution_report["observer_signal_harvest"] = observer_signal["harvest"]
                if args.post_execute_feedback_refresh:
                    feedback_command = _post_execute_feedback_refresh_command(
                        queue_id=queue_id,
                        output_dir=output_dir,
                        results_root=Path(results_root),
                        signal_harvest_path=(
                            output_dir / "final_rate_attack_signal_harvest.json"
                        ),
                        action_summary=args.post_execute_feedback_action_summary,
                        candidate_limit=args.post_execute_feedback_candidate_limit,
                        local_cpu_concurrency=args.local_cpu_concurrency,
                        max_files_per_root=args.post_execute_feedback_max_files_per_root,
                        include_retention_plan=args.post_execute_feedback_retention_plan,
                    )
                    feedback_result = _run_command(feedback_command)
                    results.append(feedback_result)
                    execution_report["post_execute_feedback_refresh"] = {
                        "schema": "frontier_final_rate_attack_post_execute_feedback_refresh.v1",
                        "enabled": True,
                        "command": feedback_command,
                        "output_dir": _display_path(
                            output_dir / "post_execute_feedback_refresh"
                        ),
                        "returncode": feedback_result["returncode"],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                    if feedback_result["returncode"] == 0:
                        feedback_payload = _json_payload_from_stdout(
                            feedback_result,
                            label="post-execute feedback refresh",
                        )
                        execution_report["post_execute_feedback_refresh"][
                            "feedback_refresh_artifacts"
                        ] = feedback_payload.get("artifacts")
                        if isinstance(feedback_payload.get("artifacts"), dict):
                            for key, value in feedback_payload["artifacts"].items():
                                artifact_paths[f"post_execute_feedback_{key}"] = value
                            if args.execute_post_feedback_queues:
                                child_queue_runs = execute_post_feedback_child_queues(
                                    repo_root=REPO_ROOT,
                                    feedback_artifacts=feedback_payload["artifacts"],
                                    output_dir=output_dir,
                                    max_steps=args.post_feedback_child_queue_max_steps,
                                    max_parallel=args.post_feedback_child_queue_max_parallel,
                                    limit=args.post_feedback_child_queue_limit,
                                    poll_interval_seconds=args.poll_interval_seconds,
                                    idle_sleep_seconds=args.idle_sleep_seconds,
                                    max_idle_cycles=args.max_idle_cycles,
                                )
                                execution_report["post_execute_feedback_child_queues"] = (
                                    child_queue_runs
                                )
                                artifact_path = child_queue_runs.get("artifact_path")
                                if isinstance(artifact_path, str):
                                    artifact_paths[
                                        "post_execute_feedback_child_queue_runs"
                                    ] = artifact_path
            execution_report["failed_command_count"] = sum(
                1 for result in results if result["returncode"] != 0
            )
            child_runs = execution_report.get("post_execute_feedback_child_queues")
            if isinstance(child_runs, dict):
                execution_report["failed_command_count"] += int(
                    child_runs.get("failed_command_count") or 0
                )
            report_path = output_dir / "execution_report.json"
            write_json_artifact(report_path, execution_report)
            artifact_paths["execution_report"] = _display_path(report_path)
        result = {
            "schema": "frontier_final_rate_attack_queue_builder_result.v1",
            "queue_id": queue_id,
            "output_dir": _display_path(output_dir),
            "artifacts": artifact_paths,
            "execute": bool(args.execute),
            "execution_report": execution_report,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    except (
        OSError,
        ArtifactWriteError,
        FrontierRateAttackBootstrapError,
        FrontierRateAttackTargetProfileError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"FATAL: frontier final-rate attack bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(result), end="")
    if execution_report is not None and execution_report["failed_command_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
