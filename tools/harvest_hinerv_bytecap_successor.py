#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Self-harvest a local/tmux HiNeRV byte-cap successor run.

The harvester waits for the local MLX/tmux run to stop, selects the best
available checkpoint metadata, exports a byte-closed archive through the
canonical HiNeRV checkpoint exporter, requires receiver proof plus full-video
MLX replay handoff evidence, writes a compact verdict artifact, and appends a
terminal dispatch-claim row. It never claims contest score authority.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    DEFAULT_WORKLOAD_SUBDIR,
    StorageTierSpec,
    parse_storage_tier_specs,
    plan_experiment_storage,
)
from tac.repo_io import sha256_file, write_json_artifact  # noqa: E402
from tools.export_hinerv_checkpoint_archive import export_checkpoint_archive  # noqa: E402

SCHEMA = "hinerv_bytecap_successor_self_harvest_verdict.v1"
DEFAULT_LANE_ID = "lane_hinerv_full600_bytecap_feedback_successor_20260604"
DEFAULT_INSTANCE_JOB_ID = "pact_hinerv_bytecap_successor_20260604T0831Z"
DEFAULT_PLATFORM = "local_mlx_tmux"
DEFAULT_AGENT = "codex:gpt-5"
RUNNER_REPORT_NAME = "compact_renderer_mlx_spine_runner_report.json"
STARTUP_MARKER_NAME = "compact_renderer_mlx_spine_runner_startup.json"
DEFAULT_REQUESTED_BYTES = 12 * 1024**3
DEFAULT_MIN_FREE_BYTES = 2 * 1024**3
DEFAULT_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class HarvestError(RuntimeError):
    """Fail-closed self-harvest error with machine-readable blockers."""

    def __init__(self, message: str, *, blockers: Sequence[str]) -> None:
        super().__init__(message)
        self.blockers = tuple(str(blocker) for blocker in blockers if str(blocker))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--instance-job-id", default=DEFAULT_INSTANCE_JOB_ID)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--tmux-session", default="")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--startup-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verdict-json", type=Path, default=None)
    parser.add_argument(
        "--reuse-existing-export-json",
        type=Path,
        default=None,
        help="Validate an existing checkpoint-export report instead of exporting again.",
    )
    parser.add_argument("--state-kind", choices=("ema", "live"), default="ema")
    parser.add_argument("--decoder-codec", default=None)
    parser.add_argument("--latent-codec", default=None)
    parser.add_argument(
        "--enforce-hard-byte-ceiling-at-export",
        action="store_false",
        dest="allow_over_hard_byte_ceiling_for_measurement",
        help="Disable the normal measurement bypass; over-cap export will fail early.",
    )
    parser.set_defaults(allow_over_hard_byte_ceiling_for_measurement=True)
    parser.add_argument(
        "--skip-mlx-prefilter-profile",
        action="store_false",
        dest="write_mlx_prefilter_profile",
        help="For debugging only. The final verdict will fail closed without this proof.",
    )
    parser.set_defaults(write_mlx_prefilter_profile=True)
    parser.add_argument("--mlx-prefilter-scorer-device", default="cpu")
    parser.add_argument("--mlx-prefilter-scorer-batch-pairs", type=int, default=1)
    parser.add_argument("--mlx-prefilter-progress-every", type=int, default=50)
    parser.add_argument("--source-video-path", type=Path, default=REPO_ROOT / "upstream/videos/0.mkv")
    parser.add_argument("--scorer-upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--claim-ttl-hours", type=float, default=168.0)
    parser.add_argument("--no-close-claim", action="store_true")
    parser.add_argument("--reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--requested-bytes", type=int, default=DEFAULT_REQUESTED_BYTES)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument(
        "--storage-tier",
        action="append",
        default=[],
        help="Optional name=/path storage tier override; default uses the SSD waterfall.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _resolve_path(args.run_dir)
    verdict_path: Path | None = None
    terminal_status = "failed_self_harvest_unhandled"
    verdict: dict[str, Any] | None = None

    try:
        detection = wait_for_terminal_run(
            run_dir=run_dir,
            tmux_session=args.tmux_session,
            wait=bool(args.wait),
            poll_interval_seconds=float(args.poll_interval_seconds),
            timeout_seconds=float(args.timeout_seconds),
        )
        storage_preflight = build_storage_preflight(
            run_dir=run_dir,
            output_dir=args.output_dir,
            requested_bytes=int(args.requested_bytes),
            min_free_bytes=int(args.min_free_bytes),
            reserve_free_gb=float(args.reserve_free_gb),
            storage_tier_values=list(args.storage_tier or []),
        )
        if storage_preflight["blockers"]:
            raise HarvestError(
                "SSD storage preflight failed",
                blockers=storage_preflight["blockers"],
            )

        runner_report_path = run_dir / RUNNER_REPORT_NAME
        runner_report = _read_json_object_if_file(runner_report_path)
        startup_path = _resolve_startup_path(run_dir, args.startup_json)
        checkpoint = select_checkpoint_meta(
            run_dir=run_dir,
            runner_report=runner_report,
            state_kind=args.state_kind,
        )
        if args.reuse_existing_export_json is None:
            export_output_dir = _resolve_export_output_dir(
                run_dir=run_dir,
                output_dir=args.output_dir,
                checkpoint_epoch=checkpoint.get("global_epoch"),
                state_kind=args.state_kind,
            )
            _require_path_under_ssd_roots(export_output_dir, storage_preflight["allowed_ssd_roots"])
            export_report = export_checkpoint_archive(
                startup_json=startup_path,
                checkpoint_meta=Path(str(checkpoint["meta_path"])),
                output_dir=export_output_dir,
                output_json=export_output_dir / "hinerv_checkpoint_archive_export.json",
                state_kind=args.state_kind,
                decoder_codec=args.decoder_codec,
                latent_codec=args.latent_codec,
                emit_receiver_proof=True,
                retain_receiver_proof_output=True,
                allow_over_hard_byte_ceiling_for_measurement=bool(
                    args.allow_over_hard_byte_ceiling_for_measurement
                ),
                write_mlx_prefilter_profile=bool(args.write_mlx_prefilter_profile),
                mlx_prefilter_scorer_device=args.mlx_prefilter_scorer_device,
                mlx_prefilter_scorer_batch_pairs=int(args.mlx_prefilter_scorer_batch_pairs),
                mlx_prefilter_progress_every=int(args.mlx_prefilter_progress_every),
                source_video_path=args.source_video_path,
                scorer_upstream_dir=args.scorer_upstream_dir,
                repo_root=REPO_ROOT,
            )
        else:
            export_report_path = _resolve_path(args.reuse_existing_export_json)
            _require_path_under_ssd_roots(export_report_path, storage_preflight["allowed_ssd_roots"])
            export_report = _read_json_object(export_report_path)
            export_report.setdefault("report_path", export_report_path.as_posix())

        proof = validate_export_proof(export_report)
        terminal_status = terminal_status_for_proof(proof)
        verdict_path = _resolve_verdict_path(
            explicit=args.verdict_json,
            export_report=export_report,
            run_dir=run_dir,
        )
        verdict = build_verdict(
            args=args,
            run_dir=run_dir,
            detection=detection,
            storage_preflight=storage_preflight,
            runner_report_path=runner_report_path if runner_report_path.is_file() else None,
            startup_path=startup_path,
            checkpoint=checkpoint,
            export_report=export_report,
            proof=proof,
            terminal_status=terminal_status,
            terminal_claim=None,
        )
        claim_result = maybe_close_dispatch_claim(args=args, status=terminal_status, verdict=verdict)
        verdict["terminal_claim"] = claim_result
        write_json_artifact(verdict_path, verdict)
        print(json.dumps(_summary(verdict, verdict_path), indent=2, sort_keys=True))
        return 0 if proof["proof_ready"] else 2
    except HarvestError as exc:
        terminal_status = terminal_status_for_blockers(exc.blockers)
        fallback_dir = run_dir if run_dir.exists() else REPO_ROOT
        verdict_path = args.verdict_json or fallback_dir / (
            f"hinerv_bytecap_self_harvest_failed_{_utc_slug()}.json"
        )
        verdict = build_failure_verdict(args=args, run_dir=run_dir, error=exc, status=terminal_status)
        if not args.no_close_claim:
            verdict["terminal_claim"] = maybe_close_dispatch_claim(
                args=args,
                status=terminal_status,
                verdict=verdict,
            )
        write_json_artifact(verdict_path, verdict)
        print(json.dumps(_summary(verdict, verdict_path), indent=2, sort_keys=True))
        return 2


def wait_for_terminal_run(
    *,
    run_dir: Path,
    tmux_session: str,
    wait: bool,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        active = bool(tmux_session) and tmux_session_exists(tmux_session)
        evidence = terminal_evidence(run_dir)
        if not active and evidence["terminal_evidence_found"]:
            return {
                "schema": "hinerv_bytecap_terminal_detection.v1",
                "status": "terminal",
                "tmux_session": tmux_session or None,
                "tmux_session_active": False,
                **evidence,
            }
        if active and not wait:
            raise HarvestError(
                f"tmux session is still active: {tmux_session}",
                blockers=["tmux_session_still_active"],
            )
        if not active and not evidence["terminal_evidence_found"]:
            raise HarvestError(
                "no stopped-run evidence found",
                blockers=["terminal_run_evidence_missing"],
            )
        if timeout_seconds > 0 and time.monotonic() - started >= timeout_seconds:
            raise HarvestError(
                "timed out waiting for tmux session to stop",
                blockers=["tmux_terminal_wait_timeout"],
            )
        time.sleep(max(float(poll_interval_seconds), 1.0))


def tmux_session_exists(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        raise HarvestError("tmux is not installed", blockers=["tmux_binary_missing"]) from None
    return result.returncode == 0


def terminal_evidence(run_dir: Path) -> dict[str, Any]:
    report = run_dir / RUNNER_REPORT_NAME
    startup = run_dir / STARTUP_MARKER_NAME
    checkpoint_metas = checkpoint_meta_candidates(run_dir)
    return {
        "runner_report_path": report.as_posix() if report.is_file() else None,
        "startup_marker_path": startup.as_posix() if startup.is_file() else None,
        "checkpoint_meta_count": len(checkpoint_metas),
        "latest_checkpoint_meta_path": (
            checkpoint_metas[-1].as_posix() if checkpoint_metas else None
        ),
        "terminal_evidence_found": bool(report.is_file() or checkpoint_metas),
    }


def build_storage_preflight(
    *,
    run_dir: Path,
    output_dir: Path | None,
    requested_bytes: int,
    min_free_bytes: int,
    reserve_free_gb: float,
    storage_tier_values: Sequence[str],
) -> dict[str, Any]:
    tiers = parse_storage_tier_specs(
        list(storage_tier_values),
        repo_root=REPO_ROOT,
        reserve_free_gb=reserve_free_gb,
        allow_local_disk=False,
    )
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=DEFAULT_WORKLOAD_SUBDIR,
        requested_bytes=requested_bytes,
        min_free_bytes=min_free_bytes,
        create=True,
        probe_writable=True,
    ).to_dict()
    allowed_roots = allowed_ssd_roots(tiers)
    blockers: list[str] = []
    if not plan.get("selected_workload_root"):
        blockers.append("storage_waterfall_no_eligible_ssd_tier")
    for label, path in (
        ("run_dir", run_dir),
        ("output_dir", output_dir if output_dir is not None else run_dir),
    ):
        try:
            _require_path_under_ssd_roots(path, allowed_roots)
        except HarvestError as exc:
            blockers.extend(f"{label}:{blocker}" for blocker in exc.blockers)
    free_probe_path = _nearest_existing_parent(output_dir or run_dir)
    free_bytes = None
    if free_probe_path is None:
        blockers.append("output_parent_missing_for_disk_usage")
    else:
        usage = shutil.disk_usage(free_probe_path)
        free_bytes = int(usage.free)
        required = int(requested_bytes) + int(min_free_bytes)
        if free_bytes < required:
            blockers.append(f"output_disk_insufficient_free_bytes:{free_bytes}<{required}")
    return {
        "schema": "hinerv_bytecap_self_harvest_storage_preflight.v1",
        "storage_waterfall_plan": plan,
        "allowed_ssd_roots": [root.as_posix() for root in allowed_roots],
        "requested_bytes": int(requested_bytes),
        "min_free_bytes": int(min_free_bytes),
        "output_disk_free_bytes": free_bytes,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def allowed_ssd_roots(tiers: Sequence[StorageTierSpec]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for tier in tiers:
        root = tier.root.expanduser().resolve(strict=False)
        if root.as_posix().startswith("/Volumes/") and not tier.allow_local_disk:
            roots.append(root)
    return tuple(roots)


def select_checkpoint_meta(
    *,
    run_dir: Path,
    runner_report: Mapping[str, Any],
    state_kind: str,
) -> dict[str, Any]:
    candidates = checkpoint_meta_candidates(run_dir)
    by_path = {path.resolve(strict=False): path for path in candidates}
    for path, source in checkpoint_paths_from_runner_report(runner_report):
        resolved = path.expanduser().resolve(strict=False)
        if resolved in by_path or resolved.is_file():
            return checkpoint_summary(resolved, source=source, state_kind=state_kind)
    ranked = sorted(candidates, key=checkpoint_sort_key)
    if not ranked:
        raise HarvestError("no checkpoint metadata found", blockers=["checkpoint_meta_missing"])
    return checkpoint_summary(ranked[-1], source="filesystem_latest", state_kind=state_kind)


def checkpoint_paths_from_runner_report(report: Mapping[str, Any]) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    training = report.get("training_artifact")
    if isinstance(training, Mapping):
        selection = training.get("checkpoint_selection")
        if isinstance(selection, Mapping):
            for key in ("selected_meta_path", "best_meta_path", "final_meta_path"):
                value = selection.get(key)
                if value:
                    paths.append((Path(str(value)), f"training_artifact.checkpoint_selection.{key}"))
        for key in ("selected_meta_path", "best_meta_path", "final_meta_path"):
            value = training.get(key)
            if value:
                paths.append((Path(str(value)), f"training_artifact.{key}"))
    for key in ("selected_meta_path", "best_meta_path", "final_meta_path"):
        value = report.get(key)
        if value:
            paths.append((Path(str(value)), key))
    return paths


def checkpoint_meta_candidates(run_dir: Path) -> list[Path]:
    checkpoint_dirs = (
        run_dir / "hi_nerv_mlx_training" / "checkpoints",
        run_dir / "checkpoints",
    )
    candidates: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.is_dir():
            candidates.extend(checkpoint_dir.glob("*.meta.json"))
    return sorted({path.resolve(strict=False) for path in candidates})


def checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    try:
        meta = _read_json_object(path)
    except Exception:
        meta = {}
    epoch = _optional_int(meta.get("global_epoch"))
    role = str(meta.get("checkpoint_role") or "")
    role_rank = 2 if role == "best" else 1 if path.name.startswith("final_") else 0
    return (int(epoch if epoch is not None else -1), role_rank, path.name)


def checkpoint_summary(path: Path, *, source: str, state_kind: str) -> dict[str, Any]:
    meta = _read_json_object(path)
    state_key = "ema_shadow_state_path" if state_kind == "ema" else "live_state_path"
    state_path_raw = meta.get(state_key)
    if not state_path_raw:
        raise HarvestError(
            f"checkpoint metadata missing {state_key}",
            blockers=[f"checkpoint_{state_key}_missing"],
        )
    state_path = Path(str(state_path_raw)).expanduser().resolve(strict=False)
    if not state_path.is_file():
        raise HarvestError(
            f"checkpoint state missing: {state_path}",
            blockers=["checkpoint_state_missing"],
        )
    return {
        "schema": "hinerv_bytecap_self_harvest_checkpoint_selection.v1",
        "source": source,
        "meta_path": path.as_posix(),
        "meta_sha256": sha256_file(path),
        "state_kind": state_kind,
        "state_path": state_path.as_posix(),
        "state_sha256": sha256_file(state_path),
        "global_epoch": meta.get("global_epoch"),
        "checkpoint_role": meta.get("checkpoint_role"),
        "loss": meta.get("loss"),
    }


def validate_export_proof(report: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if report.get("schema") != "hinerv_checkpoint_archive_export.v1":
        blockers.append("checkpoint_export_report_schema_mismatch")
    archive_path = _path_from_report(report, "archive_path")
    archive_sha = str(report.get("archive_sha256") or "")
    archive_bytes = _optional_int(report.get("archive_bytes"))
    if archive_path is None or not archive_path.is_file():
        blockers.append("archive_zip_missing")
    elif archive_sha and sha256_file(archive_path) != archive_sha:
        blockers.append("archive_sha256_mismatch")
    if archive_bytes is None or archive_bytes <= 0:
        blockers.append("archive_bytes_missing")
    elif archive_path is not None and archive_path.is_file() and archive_path.stat().st_size != archive_bytes:
        blockers.append("archive_bytes_mismatch")
    receiver_path = _path_from_report(report, "receiver_proof_path")
    receiver_sha = str(report.get("receiver_proof_sha256") or "")
    if report.get("receiver_proof_ready") is not True:
        blockers.append("receiver_proof_not_ready")
    if receiver_path is None or not receiver_path.is_file():
        blockers.append("receiver_proof_path_missing")
    elif receiver_sha and sha256_file(receiver_path) != receiver_sha:
        blockers.append("receiver_proof_sha256_mismatch")
    prefilter_path = _path_from_report(report, "local_mlx_prefilter_profile_path")
    if report.get("local_mlx_prefilter_written") is not True:
        blockers.append("full_video_mlx_prefilter_profile_not_written")
    if prefilter_path is None or not prefilter_path.is_file():
        blockers.append("full_video_mlx_prefilter_profile_missing")
    checkpoint_meta = _path_from_report(report, "checkpoint_meta_path")
    checkpoint_state = _path_from_report(report, "checkpoint_state_path")
    if checkpoint_meta is None or not checkpoint_meta.is_file():
        blockers.append("checkpoint_meta_path_missing_in_export_report")
    if checkpoint_state is None or not checkpoint_state.is_file():
        blockers.append("checkpoint_state_path_missing_in_export_report")
    ceiling = _optional_int(report.get("hard_byte_ceiling_requested_by_candidate_or_startup"))
    overrun = None
    if archive_bytes is not None and ceiling is not None:
        overrun = max(0, int(archive_bytes) - int(ceiling))
    proof_ready = not blockers
    verdict = (
        "proof_missing"
        if not proof_ready
        else "overcap_receiver_proof_profiled"
        if overrun and overrun > 0
        else "under_cap_receiver_proof_profiled"
    )
    return {
        "schema": "hinerv_bytecap_self_harvest_proof_gate.v1",
        "proof_ready": proof_ready,
        "verdict": verdict,
        "archive_bytes": archive_bytes,
        "hard_byte_ceiling": ceiling,
        "archive_overrun_bytes": overrun,
        "archive_path": archive_path.as_posix() if archive_path else None,
        "receiver_proof_path": receiver_path.as_posix() if receiver_path else None,
        "local_mlx_prefilter_profile_path": prefilter_path.as_posix() if prefilter_path else None,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def terminal_status_for_proof(proof: Mapping[str, Any]) -> str:
    if proof.get("proof_ready") is not True:
        return "failed_self_harvest_proof_missing"
    if _optional_int(proof.get("archive_overrun_bytes")):
        return "completed_overcap_measurement_receiver_proof_profiled"
    return "completed_receiver_proof_profiled_under_bytecap"


def terminal_status_for_blockers(blockers: Sequence[str]) -> str:
    blocker_set = set(blockers)
    if "tmux_session_still_active" in blocker_set:
        return "failed_self_harvest_tmux_still_running"
    if "tmux_terminal_wait_timeout" in blocker_set:
        return "failed_self_harvest_tmux_wait_timeout"
    if any(blocker.startswith("storage_") or "ssd" in blocker for blocker in blocker_set):
        return "failed_self_harvest_storage_preflight"
    if any("checkpoint" in blocker for blocker in blocker_set):
        return "failed_self_harvest_checkpoint_missing"
    return "failed_self_harvest_proof_missing"


def build_verdict(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    detection: Mapping[str, Any],
    storage_preflight: Mapping[str, Any],
    runner_report_path: Path | None,
    startup_path: Path,
    checkpoint: Mapping[str, Any],
    export_report: Mapping[str, Any],
    proof: Mapping[str, Any],
    terminal_status: str,
    terminal_claim: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers = list(dict.fromkeys([*proof.get("blockers", ()), *export_report.get("blockers", ())]))
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "lane_id": args.lane_id,
        "platform": args.platform,
        "instance_job_id": args.instance_job_id,
        "agent": args.agent,
        "run_dir": run_dir.as_posix(),
        "terminal_detection": dict(detection),
        "storage_preflight": dict(storage_preflight),
        "disk_hygiene": {
            "schema": "hinerv_bytecap_self_harvest_disk_hygiene.v1",
            "large_artifacts_root": str(export_report.get("output_dir") or run_dir.as_posix()),
            "local_disk_allowed": False,
            "scratch_policy": "no_local_scratch_created_by_harvester",
            "cleanup_policy": "lossless_ssd_custody_keep_bytes_until_rebuild_proof_consumed",
            "proof_artifact_is_durable": True,
        },
        "runner_report_path": runner_report_path.as_posix() if runner_report_path else None,
        "runner_report_sha256": sha256_file(runner_report_path) if runner_report_path else None,
        "startup_json_path": startup_path.as_posix(),
        "startup_json_sha256": sha256_file(startup_path),
        "checkpoint_selection": dict(checkpoint),
        "export_report_path": str(export_report.get("report_path") or ""),
        "export_report_sha256": _sha256_optional_path(export_report.get("report_path")),
        "archive_path": export_report.get("archive_path"),
        "archive_sha256": export_report.get("archive_sha256"),
        "archive_bytes": export_report.get("archive_bytes"),
        "hard_byte_ceiling": export_report.get("hard_byte_ceiling_requested_by_candidate_or_startup"),
        "receiver_proof_ready": export_report.get("receiver_proof_ready"),
        "receiver_proof_path": export_report.get("receiver_proof_path"),
        "receiver_proof_sha256": export_report.get("receiver_proof_sha256"),
        "local_mlx_prefilter_written": export_report.get("local_mlx_prefilter_written"),
        "local_mlx_prefilter_profile_path": export_report.get("local_mlx_prefilter_profile_path"),
        "proof_gate": dict(proof),
        "verdict": proof.get("verdict"),
        "terminal_status": terminal_status,
        "terminal_claim": terminal_claim,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def build_failure_verdict(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    error: HarvestError,
    status: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "lane_id": args.lane_id,
        "platform": args.platform,
        "instance_job_id": args.instance_job_id,
        "agent": args.agent,
        "run_dir": run_dir.as_posix(),
        "verdict": "self_harvest_failed_closed",
        "terminal_status": status,
        "error": str(error),
        "blockers": list(error.blockers),
        "terminal_claim": None,
        **FALSE_AUTHORITY,
    }


def maybe_close_dispatch_claim(
    *,
    args: argparse.Namespace,
    status: str,
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    if args.no_close_claim:
        return {
            "schema": "hinerv_bytecap_self_harvest_dispatch_claim.v1",
            "closed": False,
            "reason": "no_close_claim_requested",
        }
    latest = latest_claim_for_job(
        claims_path=args.claims_path,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
    )
    if latest and is_terminal_status(str(latest.get("status") or "")) and latest.get("status") == status:
        return {
            "schema": "hinerv_bytecap_self_harvest_dispatch_claim.v1",
            "closed": False,
            "already_terminal": True,
            "latest_claim": latest,
        }
    notes = claim_notes(verdict)
    cmd = [
        sys.executable,
        (REPO_ROOT / "tools/claim_lane_dispatch.py").as_posix(),
        "claim",
        "--claims-path",
        _resolve_relative_to_repo(args.claims_path).as_posix(),
        "--lane-id",
        args.lane_id,
        "--platform",
        args.platform,
        "--instance-job-id",
        args.instance_job_id,
        "--agent",
        args.agent,
        "--status",
        status,
        "--notes",
        notes,
        "--ttl-hours",
        str(float(args.claim_ttl_hours)),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise HarvestError(
            "terminal dispatch-claim row failed",
            blockers=["dispatch_claim_terminal_row_failed"],
        )
    return {
        "schema": "hinerv_bytecap_self_harvest_dispatch_claim.v1",
        "closed": True,
        "status": status,
        "notes": notes,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def claim_notes(verdict: Mapping[str, Any]) -> str:
    archive_bytes = verdict.get("archive_bytes")
    ceiling = verdict.get("hard_byte_ceiling")
    overrun = None
    if archive_bytes is not None and ceiling is not None:
        overrun = int(archive_bytes) - int(ceiling)
    proof = "receiver_proof+full_video_mlx_prefilter"
    path = str(verdict.get("export_report_path") or "")
    bits = [f"self-harvest {proof}"]
    if archive_bytes is not None:
        bits.append(f"archive={archive_bytes}B")
    if overrun is not None:
        bits.append(f"overrun={overrun}B")
    if path:
        bits.append(f"report={path}")
    bits.append("false-authority macOS-MLX; exact CPU/CUDA not run")
    return "; ".join(bits)[:900]


def latest_claim_for_job(
    *,
    claims_path: Path,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, str] | None:
    path = _resolve_relative_to_repo(claims_path)
    if not path.is_file():
        return None
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "timestamp_utc" in line or "---" in line:
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        row = {
            "timestamp_utc": cells[0],
            "agent": cells[1],
            "lane_id": cells[2],
            "platform": cells[3],
            "instance_job_id": cells[4],
            "predicted_eta_utc": cells[5],
            "status": cells[6],
            "notes": cells[7],
        }
        if row["lane_id"] == lane_id and row["instance_job_id"] == instance_job_id:
            rows.append(row)
    return rows[0] if rows else None


def is_terminal_status(status: str) -> bool:
    return status.startswith(
        (
            "completed_",
            "failed_",
            "timed_out",
            "preempted",
            "cancelled",
            "refused_",
            "stale_",
            "stopped_",
            "falsified_",
            "retired_",
            "config_retired_",
            "measured_implementation_retired_",
        )
    )


def _resolve_startup_path(run_dir: Path, explicit: Path | None) -> Path:
    path = _resolve_path(explicit) if explicit is not None else run_dir / STARTUP_MARKER_NAME
    if not path.is_file():
        raise HarvestError("startup marker missing", blockers=["startup_json_missing"])
    return path


def _resolve_export_output_dir(
    *,
    run_dir: Path,
    output_dir: Path | None,
    checkpoint_epoch: Any,
    state_kind: str,
) -> Path:
    if output_dir is not None:
        return _resolve_path(output_dir)
    epoch = _optional_int(checkpoint_epoch)
    epoch_text = "unknown" if epoch is None else f"{epoch:06d}"
    return run_dir / f"self_harvest_epoch{epoch_text}_{state_kind}_{_utc_slug()}"


def _resolve_verdict_path(
    *,
    explicit: Path | None,
    export_report: Mapping[str, Any],
    run_dir: Path,
) -> Path:
    if explicit is not None:
        return _resolve_path(explicit)
    report_path = _path_from_report(export_report, "report_path")
    if report_path is not None:
        return report_path.parent / "hinerv_bytecap_self_harvest_verdict.json"
    return run_dir / f"hinerv_bytecap_self_harvest_verdict_{_utc_slug()}.json"


def _path_from_report(report: Mapping[str, Any], key: str) -> Path | None:
    value = report.get(key)
    if not value:
        return None
    return Path(str(value)).expanduser().resolve(strict=False)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HarvestError(f"JSON artifact is not an object: {path}", blockers=["json_object_expected"])
    return payload


def _read_json_object_if_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json_object(path)


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _resolve_relative_to_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve(strict=False)


def _require_path_under_ssd_roots(path: Path, roots: Sequence[str | Path]) -> None:
    resolved = _resolve_path(path)
    root_paths = tuple(_resolve_path(root) for root in roots)
    if not root_paths:
        raise HarvestError("no allowed SSD roots configured", blockers=["allowed_ssd_roots_missing"])
    for root in root_paths:
        if resolved == root or root in resolved.parents:
            return
    raise HarvestError(
        f"path is not under an allowed SSD root: {resolved}",
        blockers=["path_not_ssd_backed"],
    )


def _nearest_existing_parent(path: Path) -> Path | None:
    current = _resolve_path(path)
    while True:
        if current.exists():
            return current if current.is_dir() else current.parent
        if current.parent == current:
            return None
        current = current.parent


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_optional_path(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve(strict=False)
    return sha256_file(path) if path.is_file() else None


def _utc_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _summary(verdict: Mapping[str, Any], path: Path) -> dict[str, Any]:
    return {
        "schema": "hinerv_bytecap_self_harvest_summary.v1",
        "verdict_path": path.as_posix(),
        "verdict": verdict.get("verdict"),
        "terminal_status": verdict.get("terminal_status"),
        "archive_bytes": verdict.get("archive_bytes"),
        "hard_byte_ceiling": verdict.get("hard_byte_ceiling"),
        "receiver_proof_ready": verdict.get("receiver_proof_ready"),
        "local_mlx_prefilter_written": verdict.get("local_mlx_prefilter_written"),
        "blockers": verdict.get("blockers"),
        "terminal_claim": verdict.get("terminal_claim"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
