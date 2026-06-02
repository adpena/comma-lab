# SPDX-License-Identifier: MIT
"""Fail-closed orchestration contract for the top-priority NeRV stacks.

SNeRV and HiNeRV are the two active carrier stacks. PR95/HNeRV is the upstream
baseline/control to beat. This module turns that operating rule into a typed
artifact so queue builders, future agents, and operator briefings inherit the
same fail-closed gates instead of re-deriving priority from chat.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "nerv_top_priority_stack_seam.v1"
AXIS_TAG = "[planning/control]"
DEFAULT_LANE_ID = "lane_nerv_top_priority_stack_seam_20260602"
PR95_PR_NUMBER = 95
PR95_PR_URL = "https://github.com/commaai/comma_video_compression_challenge/pull/95"
PR95_SUBMISSION = "hnerv_muon"
PR101_LANE_ID = "lane_pr101_storage_order_len24_exact_cpu_20260601"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "exact_or_full_video_launched": False,
}

TOP_PRIORITY_CARRIERS = ("snerv", "hinerv")
FULL_STACK_COMPONENTS = (
    "architecture",
    "optimizer_qat",
    "allocator",
    "archive_grammar",
    "receiver_proof",
    "eval_control",
)
TERMINAL_STATUS_PREFIXES = (
    "completed_",
    "failed_",
    "timed_out",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)
ACTIVE_STATUS_TOKENS = (
    "active",
    "pending",
    "spawned",
    "spawning",
    "running",
    "eval",
    "training",
    "dispatching",
)
EXACT_OR_FULL_VIDEO_TOKENS = (
    "auth_eval",
    "contest_cpu",
    "contest_cuda",
    "exact",
    "full_video",
    "full-video",
    "paired_cuda",
    "paired-cuda",
    "modal_auth",
    "cuda_ratification",
)
REMOTE_EVAL_PLATFORMS = ("lightning", "modal", "vast", "vastai", "azure", "aws", "gcp")


class NervTopPriorityStackSeamError(ValueError):
    """Raised when the orchestration contract inputs are malformed."""


def build_nerv_top_priority_stack_seam(
    *,
    repo_root: str | Path,
    upstream_repo_dir: str | Path | None = None,
    pr95_intake_root: str | Path | None = None,
    active_claims_path: str | Path | None = None,
    pr95_pr_metadata: Mapping[str, Any] | None = None,
    generated_utc: str | None = None,
    lane_id: str = DEFAULT_LANE_ID,
) -> dict[str, Any]:
    """Build the shared top-priority stack contract.

    The returned payload is intentionally not a score artifact. It grants local
    implementation authority only and blocks exact/full-video dispatch while
    active exact-eval claims remain pending.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NervTopPriorityStackSeamError(f"repo_root is not a directory: {root}")

    generated = generated_utc or datetime.now(UTC).isoformat()
    baseline = discover_pr95_baseline(
        repo_root=root,
        upstream_repo_dir=upstream_repo_dir,
        pr95_intake_root=pr95_intake_root,
        pr95_pr_metadata=pr95_pr_metadata,
    )
    dispatch_blockers = discover_dispatch_blockers(
        active_claims_path,
        now_utc=generated,
    )
    blockers = _unique(
        list(baseline["blockers"])
        + dispatch_blockers
        + [
            "full_600_byte_closed_receiver_proof_missing_for_snerv_and_hinerv",
            "paired_contest_cpu_cuda_pass_missing_for_winner",
            "pr95_same_axis_control_replay_required_before_beat_claim",
        ]
    )
    exact_blocked = bool(dispatch_blockers)

    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "lane_id": lane_id,
        "axis_tag": AXIS_TAG,
        "go_no_go_verdict": (
            "GO_LOCAL_STACK_OPTIMIZATION__NO_GO_EXACT_PROMOTION_OR_SCORE_CLAIM"
        ),
        "baseline_to_beat": "pr95_hnerv_muon",
        "top_priority_carriers": list(TOP_PRIORITY_CARRIERS),
        "priority_policy": {
            "carrier_stacks": list(TOP_PRIORITY_CARRIERS),
            "baseline_control": "pr95_hnerv_muon",
            "individually_fractally_optimized_full_stacks": True,
            "shared_synergy_surfaces_do_not_collapse_carrier_specific_work": True,
            "enhancers_are_not_standalone_carrier_stacks": True,
            "compare_under_same_archive_runtime_eval_axis": True,
            "do_not_launch_new_full_video_or_exact_while_dispatch_blockers_active": True,
        },
        "full_stack_priority": _full_stack_priority(),
        "baseline": baseline,
        "carrier_stacks": [_snerv_stack(), _hinerv_stack()],
        "fractal_work_orders": _fractal_work_orders(),
        "synergy_enhancers": _synergy_enhancers(),
        "shared_promotion_gates": _shared_promotion_gates(),
        "blocked_dispatch": exact_blocked,
        "dispatch_blockers": dispatch_blockers,
        "next_local_actions": _next_local_actions(
            exact_blocked=exact_blocked,
            upstream_repo_dir=baseline["upstream_repo_dir"].get("path"),
        ),
        "forbidden_actions": [
            "claim_pr95_beat_without_same_axis_pr95_control",
            "promote_local_mlx_or_macos_cpu_advisory_score",
            "dispatch_exact_or_full_video_while_pr101_cpu_pending",
            "rerun_closed_form_snerv_scalar_hf_sweeps_as_promotion_evidence",
            "treat_sr_nerv_zero_parameter_interpolation_as_promotable",
        ],
        "blockers": blockers,
        "operator_truth": {
            "main_is_source_of_truth": True,
            "dirty_shared_worktree_is_not_absorbed": True,
            "upstream_pr95_is_forensic_baseline_source": True,
            "large_artifacts_policy": (
                "this artifact is metadata-only; future training/eval outputs must "
                "spill to SSD and preserve deterministic custody before cleanup"
            ),
        },
        **FALSE_AUTHORITY,
    }


def discover_pr95_baseline(
    *,
    repo_root: Path,
    upstream_repo_dir: str | Path | None,
    pr95_intake_root: str | Path | None,
    pr95_pr_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Discover the PR95/HNeRV baseline without mutating upstream or caches."""

    metadata = dict(pr95_pr_metadata or {})
    blockers: list[str] = []
    upstream = _path_record(upstream_repo_dir)
    if upstream["present"]:
        upstream_git = _git_info(Path(str(upstream["path"])))
        upstream["git"] = upstream_git
        if not upstream_git.get("head"):
            blockers.append("upstream_repo_git_head_missing")
        if not upstream_git.get("remote_origin"):
            blockers.append("upstream_repo_git_remote_origin_missing")

    intake = _path_record(pr95_intake_root)
    archive = None
    runtime_files: list[dict[str, Any]] = []
    if intake["present"]:
        intake_root = Path(str(intake["path"]))
        archive_path = intake_root / "archive.zip"
        if archive_path.is_file():
            archive = _file_manifest(archive_path)
        else:
            blockers.append("pr95_public_archive_zip_missing")
        submission_root = intake_root / "source" / "submissions" / PR95_SUBMISSION
        for rel in (
            "inflate.sh",
            "inflate.py",
            "src/model.py",
            "src/codec.py",
            "src/optim.py",
            "src/stages/stage8_muon_finetune.py",
        ):
            path = submission_root / rel
            if path.is_file():
                runtime_files.append(_file_manifest(path))
            else:
                blockers.append(f"pr95_runtime_file_missing:{rel}")
    else:
        blockers.append("pr95_public_intake_root_missing")

    if not upstream["present"]:
        blockers.append("upstream_repo_dir_missing")

    proof_tool = repo_root / "tools" / "prove_pr95_public_archive_runtime_consumption.py"
    if not proof_tool.is_file():
        blockers.append("pr95_runtime_consumption_proof_tool_missing")

    return {
        "schema": "pr95_hnerv_baseline_control_pointer.v1",
        "source_pr": PR95_PR_NUMBER,
        "source_url": metadata.get("url", PR95_PR_URL),
        "title": metadata.get("title", "hnerv_muon submission (0.20)"),
        "state": metadata.get("state"),
        "head_sha": metadata.get("headRefOid"),
        "head_ref": metadata.get("headRefName"),
        "submission": PR95_SUBMISSION,
        "role": "baseline_control_to_beat",
        "upstream_repo_dir": upstream,
        "intake_root": intake,
        "archive": archive,
        "runtime_files": runtime_files,
        "existing_proof_tools": [
            "tools/prove_pr95_public_archive_runtime_consumption.py",
            "tools/prove_pr95_public_archive_full_frame_parity.py",
            "tools/run_pr95_stage8_from_public_archive.py",
            "tools/run_pr95_hnerv_linf_carrier.py",
        ],
        "same_axis_requirement": (
            "SNeRV/HiNeRV winner must be compared against PR95 on the same "
            "archive, runtime, inflate, and eval hardware axis before any beat claim"
        ),
        "blockers": _unique(blockers),
    }


def discover_dispatch_blockers(
    active_claims_path: str | Path | None,
    *,
    now_utc: str | datetime | None = None,
    ttl_hours: float = 24.0,
) -> list[str]:
    """Return active exact/full-video blockers from the lane claim table."""

    if active_claims_path is None:
        return ["active_claims_table_not_supplied"]
    path = Path(active_claims_path)
    if not path.is_file():
        return ["active_claims_table_missing"]
    now = _parse_utc(now_utc) or datetime.now(UTC)

    blockers: list[str] = []
    rows = _latest_claim_rows_by_job(_parse_claim_rows(path.read_text(encoding="utf-8")))
    for claim in rows:
        status = str(claim.get("status", ""))
        if _status_is_terminal(status):
            continue
        if not _claim_is_active(claim):
            continue
        lane_id = str(claim.get("lane_id", ""))
        if lane_id == PR101_LANE_ID:
            blockers.append("pr101_cpu_recovery_pending_blocks_new_exact_or_full_video")
            continue
        if lane_id.startswith("lane_z5_rao_ballard_paired_cuda_ratification_wave2a_20260531"):
            blockers.append("z5_rao_ballard_modal_claims_still_need_terminal_adjudication")
            continue
        if _claim_blocks_exact_or_full_video(claim) and _claim_within_active_window(
            claim,
            now=now,
            ttl_hours=ttl_hours,
        ):
            blockers.append(f"active_exact_or_full_video_claim:{_blocker_slug(lane_id)}")
    return _unique(blockers)


def _snerv_stack() -> dict[str, Any]:
    return {
        "stack_id": "snerv",
        "role": "top_priority_carrier",
        "status": "local_stack_optimization_active",
        "current_read": (
            "linear/closed-form coordinate and scalar HF sweeps are not promotion "
            "evidence; explicit mixed decoder modes and scorer-loop QAT are the "
            "next useful local surfaces"
        ),
        "required_components": [
            "architecture",
            "learned_or_nonlinear_decoder_qat",
            "pose_guarded_scorer_loop_fit",
            "linf_oracle_allocator_inside_decoder_weight_training",
            "mixed_precision_decoder_grammar",
            "byte_closed_receiver_proof",
            "paired_pr95_same_axis_control",
        ],
        "existing_surfaces": [
            "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
            "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
            "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
            "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
            "tools/probe_snerv_decoder_mode_assignments.py",
            "tools/prove_snerv_receiver_archive.py",
        ],
        "next_gate": (
            "pair-robust scorer-loop or NES decoder-QAT continuation with PoseNet "
            "as hard guard and receiver-decoded mixed precision bytes preserved"
        ),
        "optimization_scope": "carrier-specific full stack; not a shared HNeRV clone",
    }


def _hinerv_stack() -> dict[str, Any]:
    return {
        "stack_id": "hinerv",
        "role": "top_priority_carrier",
        "status": "local_stack_optimization_active",
        "current_read": (
            "recent full-600 MLX prefilter was byte-closed but distortion-bad; "
            "rate plumbing is real enough, fit must move upstream into longer "
            "score-aware/coder-aware decoder-weight training"
        ),
        "required_components": [
            "hierarchical_architecture",
            "coder_aware_qat",
            "joint_p18_p19_oracle_weighting",
            "dense_decoder_vjp_linf_allocator",
            "pr95_faithful_curriculum_control",
            "byte_closed_receiver_proof",
            "full_video_mlx_prefilter_then_paired_cpu_cuda",
        ],
        "existing_surfaces": [
            "src/tac/analysis/hinerv_latent_linf_allocation.py",
            "src/tac/substrates/hi_nerv/score_aware_loss.py",
            "tools/run_compact_renderer_mlx_spine_runner.py",
            ".omx/research/codex_findings_hinerv_coder_aware_qat_wired_20260601T224653Z_codex.md",
            ".omx/research/codex_findings_hinerv_600pair_joint_p18_p19_full_prefilter_20260602T021050Z_codex.md",
        ],
        "next_gate": (
            "longer staged real-teacher SegNet/PoseNet training with coder-aware "
            "QAT and local MLX prefilter; no exact spend until prefilter enters a "
            "plausible replay band"
        ),
        "optimization_scope": "carrier-specific full stack; not a SNeRV codec wrapper",
    }


def _full_stack_priority() -> dict[str, Any]:
    return {
        "schema": "nerv_individual_fractal_full_stack_priority.v1",
        "top_priority_carriers": list(TOP_PRIORITY_CARRIERS),
        "components": list(FULL_STACK_COMPONENTS),
        "policy": (
            "Optimize SNeRV and HiNeRV as separate end-to-end stacks at every "
            "component boundary, then compose shared enhancers only through "
            "explicit byte/eval gates."
        ),
        "do_not_average_stacks": True,
        "do_not_promote_shared_enhancer_as_carrier": True,
        "pr95_control_required_for_any_beat_claim": True,
        "fractality_rule": (
            "Each stack component needs its own hypothesis, local command, byte "
            "accounting surface, guard, and promotion blocker; a global stack "
            "memo alone is not implementation authority."
        ),
    }


def _fractal_work_orders() -> list[dict[str, Any]]:
    return [
        {
            "stack_id": "snerv",
            "priority": "top_carrier",
            "work_order": _component_work_orders(
                {
                    "architecture": (
                        "learn nonlinear decoder/HF restoration against real "
                        "SegNet/PoseNet response, not scalar coordinate sweeps"
                    ),
                    "optimizer_qat": (
                        "run pair-robust scorer-loop or NES decoder-QAT with "
                        "PoseNet hard guard and per-pair deltas"
                    ),
                    "allocator": (
                        "move L-infinity allocation into decoder-weight fit; "
                        "latents are diagnostic only until leverage reappears"
                    ),
                    "archive_grammar": (
                        "replace fp32/fake-quant receiver payload with explicit "
                        "mixed decoder modes, int planes, or decoder-delta packing"
                    ),
                    "receiver_proof": (
                        "prove receiver-decoded byte accounting before any "
                        "full-600 authority"
                    ),
                    "eval_control": (
                        "compare against PR95 only after same-axis archive/runtime "
                        "control replay"
                    ),
                },
                local_command_ids=(
                    "snerv_pair_robust_decoder_qat_continuation",
                    "snerv_explicit_decoder_mode_triage",
                ),
            ),
        },
        {
            "stack_id": "hinerv",
            "priority": "top_carrier",
            "work_order": _component_work_orders(
                {
                    "architecture": (
                        "continue hierarchical NeRV carrier training; fit is the "
                        "active blocker, not proof that the rate knob is fake"
                    ),
                    "optimizer_qat": (
                        "stage real-teacher SegNet/PoseNet loss, coder-aware QAT, "
                        "and PR95/Muon curriculum controls"
                    ),
                    "allocator": (
                        "use dense decoder VJP L-infinity allocator with joint "
                        "P18/P19 weighting inside decoder-weight optimization"
                    ),
                    "archive_grammar": (
                        "keep byte budget parameterized and close the quantized "
                        "receiver grammar before exact replay"
                    ),
                    "receiver_proof": (
                        "prove full-600 archive/runtime consumption with no MLX "
                        "or advisory shortcut"
                    ),
                    "eval_control": (
                        "prefilter locally on MLX, then replay paired CPU/CUDA "
                        "only after blocker claims terminalize"
                    ),
                },
                local_command_ids=("hinerv_real_teacher_qat_continuation",),
            ),
        },
    ]


def _component_work_orders(
    descriptions: Mapping[str, str],
    *,
    local_command_ids: Sequence[str],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for component in FULL_STACK_COMPONENTS:
        orders.append(
            {
                "component": component,
                "next_action": descriptions[component],
                "local_command_ids": list(local_command_ids),
                "requires_receiver_byte_accounting": component
                in {"archive_grammar", "receiver_proof", "eval_control"},
                "promotion_authority": False,
            }
        )
    return orders


def _synergy_enhancers() -> list[dict[str, Any]]:
    return [
        {
            "enhancer_id": "sr_nerv_trained_scorer_aware",
            "priority": "highest_enhancer",
            "not_a_standalone_carrier_stack": True,
            "policy": (
                "low-internal-resolution carrier plus trained/scorer-aware SR; "
                "zero-parameter interpolation remains no-go"
            ),
        },
        {
            "enhancer_id": "rnerv_per_video_config_optimizer",
            "priority": "winner_optimizer",
            "not_a_standalone_carrier_stack": True,
            "policy": "optimize the winning SNeRV/HiNeRV carrier configuration per video",
        },
        {
            "enhancer_id": "ffnerv_flow_pose_channel",
            "priority": "pose_channel_enhancer",
            "not_a_standalone_carrier_stack": True,
            "policy": "flow-conditioning is a pose-channel bolt-on after carrier winner emerges",
        },
        {
            "enhancer_id": "boostnerv_decoder_temporal_affine",
            "priority": "cheap_synergy_multiplier",
            "not_a_standalone_carrier_stack": True,
            "policy": "conditional decoder/temporal-affine bolt-on for the winner",
        },
    ]


def _shared_promotion_gates() -> list[dict[str, Any]]:
    return [
        {
            "gate": "pr101_cpu_recovery_terminal",
            "required": True,
            "authority_after_pass": "may consider exact/full-video queue again",
        },
        {
            "gate": "byte_closed_full_600_receiver_proof",
            "required": True,
            "authority_after_pass": "candidate may enter paired auth-eval planning only",
        },
        {
            "gate": "pr95_same_axis_control",
            "required": True,
            "authority_after_pass": "beat/no-beat statement may be made only on that axis",
        },
        {
            "gate": "paired_contest_cpu_cuda_auth_eval",
            "required": True,
            "authority_after_pass": "promotion discussion may begin",
        },
    ]


def _next_local_actions(
    *,
    exact_blocked: bool,
    upstream_repo_dir: str | None,
) -> list[dict[str, Any]]:
    dispatch_note = (
        "blocked while active exact/full-video claims remain nonterminal"
        if exact_blocked
        else "allowed only after lane claim and byte-closed proof"
    )
    upstream_arg = upstream_repo_dir or "<UPSTREAM_REPO_DIR>"
    return [
        {
            "id": "poll_pr101_cpu_recovery",
            "axis": "[contest-CPU recovery]",
            "command": (
                ".venv/bin/python tools/recover_modal_auth_eval.py "
                "--call-id fc-01KT2BZT54G6CXPMD94SY43MMH "
                "--output-dir experiments/results/modal_auth_eval_cpu/"
                "pr101_storage_order_len24_cpu_20260601T1955Z"
            ),
            "allowed_now": True,
            "authority_after_pass": "terminalize claim only; no score shortcut",
        },
        {
            "id": "snerv_pair_robust_decoder_qat_continuation",
            "axis": "[macOS-CPU advisory]",
            "command": (
                ".venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py "
                "--n-pairs 4 --pair-stride 8 --pose-hard-guard "
                "--out .omx/research/snerv_scorer_loop_decoder_qat_next_<UTC>.json"
            ),
            "allowed_now": True,
            "authority_after_pass": "local triage only",
        },
        {
            "id": "snerv_explicit_decoder_mode_triage",
            "axis": "[macOS-CPU advisory]",
            "command": (
                ".venv/bin/python tools/probe_snerv_decoder_mode_assignments.py "
                "--n-pairs 2 --levels 1 --mode-plan magnitude_heuristic "
                f"--mode-plan fp16,int4,int4 --upstream-dir {upstream_arg}"
            ),
            "allowed_now": True,
            "authority_after_pass": "local receiver-decoded rate/fit triage only",
        },
        {
            "id": "hinerv_real_teacher_qat_continuation",
            "axis": "[macOS-MLX research-signal]",
            "command": (
                ".venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py "
                "--execute-family hi_nerv --coder-aware-qat "
                "--real-teacher-segnet-posenet --num-pairs 32 --epochs 8"
            ),
            "allowed_now": True,
            "authority_after_pass": "local MLX prefilter only",
        },
        {
            "id": "paired_exact_eval_or_full_video",
            "axis": "[contest-CPU]/[contest-CUDA]",
            "command": "deferred",
            "allowed_now": False,
            "blocked_reason": dispatch_note,
        },
    ]


def _path_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False}
    p = Path(path).resolve()
    return {"path": p.as_posix(), "present": p.exists(), "is_dir": p.is_dir()}


def _file_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_info(path: Path) -> dict[str, Any]:
    return {
        "head": _git_output(path, "rev-parse", "HEAD"),
        "branch": _git_output(path, "branch", "--show-current"),
        "remote_origin": _git_output(path, "remote", "get-url", "origin"),
    }


def _git_output(path: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", path.as_posix(), *args],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _parse_utc(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_claim_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        if cells[0] == "timestamp_utc" or set(cells[0]) <= {"-"}:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _latest_claim_rows_by_job(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        claim = dict(row)
        key = (
            str(claim.get("lane_id", "")),
            str(claim.get("instance_job_id", "")),
        )
        previous = latest.get(key)
        if previous is None or _claim_row_is_newer(claim, previous):
            latest[key] = claim
    return list(latest.values())


def _claim_row_is_newer(
    candidate: Mapping[str, str],
    previous: Mapping[str, str],
) -> bool:
    candidate_ts = _parse_utc(candidate.get("timestamp_utc", ""))
    previous_ts = _parse_utc(previous.get("timestamp_utc", ""))
    if previous_ts is None:
        return candidate_ts is not None
    if candidate_ts is None:
        return False
    return candidate_ts > previous_ts


def _status_is_terminal(status: str) -> bool:
    lowered = status.lower()
    return any(lowered.startswith(prefix) for prefix in TERMINAL_STATUS_PREFIXES)


def _claim_is_active(claim: Mapping[str, str]) -> bool:
    status = claim.get("status", "").lower()
    lane_id = claim.get("lane_id", "").lower()
    notes = claim.get("notes", "").lower()
    haystack = " ".join((status, lane_id, notes))
    return any(token in haystack for token in ACTIVE_STATUS_TOKENS)


def _claim_blocks_exact_or_full_video(claim: Mapping[str, str]) -> bool:
    status = claim.get("status", "").lower()
    lane_id = claim.get("lane_id", "").lower()
    platform = claim.get("platform", "").lower()
    notes = claim.get("notes", "").lower()
    instance = claim.get("instance_job_id", "").lower()
    haystack = " ".join((lane_id, status, platform, notes, instance))
    if any(token in haystack for token in EXACT_OR_FULL_VIDEO_TOKENS):
        return True
    return platform in REMOTE_EVAL_PLATFORMS and any(
        token in status for token in ("eval", "dispatch", "spawn")
    )


def _claim_within_active_window(
    claim: Mapping[str, str],
    *,
    now: datetime,
    ttl_hours: float,
) -> bool:
    ttl_seconds = max(float(ttl_hours), 0.0) * 3600.0
    timestamp = _parse_utc(claim.get("timestamp_utc", ""))
    if timestamp is not None and (now - timestamp).total_seconds() <= ttl_seconds:
        return True
    predicted_eta = _parse_utc(claim.get("predicted_eta_utc", ""))
    return predicted_eta is not None and predicted_eta >= now


def _blocker_slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum() or char in ("_", "-"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown_lane"


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "AXIS_TAG",
    "DEFAULT_LANE_ID",
    "FALSE_AUTHORITY",
    "FULL_STACK_COMPONENTS",
    "PR95_PR_NUMBER",
    "PR95_PR_URL",
    "PR101_LANE_ID",
    "SCHEMA",
    "TERMINAL_STATUS_PREFIXES",
    "TOP_PRIORITY_CARRIERS",
    "NervTopPriorityStackSeamError",
    "build_nerv_top_priority_stack_seam",
    "discover_dispatch_blockers",
    "discover_pr95_baseline",
]
