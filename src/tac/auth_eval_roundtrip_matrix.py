# SPDX-License-Identifier: MIT
"""Auth-eval roundtrip target matrix contracts.

The matrix separates contest-compliant score axes from diagnostic roundtrip
knobs.  It is a command planner only: rows are not score claims, and diagnostic
rows are explicitly blocked from promotion.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.deploy.modal.auth_eval import modal_uploaded_submission_dir_runtime_manifest


SCHEMA_VERSION = "auth_eval_roundtrip_matrix_v1"
RESULTS_SCHEMA_VERSION = "auth_eval_roundtrip_results_v1"


@dataclass(frozen=True)
class AuthEvalRoundtripInput:
    """Byte-closed archive/runtime packet to evaluate."""

    archive: str
    submission_dir: str
    inflate_sh: str = "inflate.sh"
    label: str = "candidate"
    output_root: str = "experiments/results/auth_eval_roundtrip_matrix"
    lane_id: str = "auth_eval_roundtrip_matrix"


def _repo_rel(path: str | Path, repo_root: Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return value.as_posix()
    try:
        return value.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _host_is_contest_cpu_like(system: str, machine: str) -> bool:
    return system.lower() == "linux" and machine.lower() in {"x86_64", "amd64"}


def _command_value_after(command: list[str], flag: str) -> str | None:
    for index, value in enumerate(command):
        if value == flag and index + 1 < len(command):
            return command[index + 1]
    return None


def _local_command(
    *,
    candidate: AuthEvalRoundtripInput,
    repo_root: Path,
    output_dir: str,
    device: str,
    expected_runtime_tree_sha256: str,
) -> list[str]:
    return [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        _repo_rel(candidate.archive, repo_root),
        "--inflate-sh",
        f"{_repo_rel(candidate.submission_dir, repo_root)}/{candidate.inflate_sh}",
        "--upstream-dir",
        "upstream",
        "--video-names-file",
        "upstream/public_test_video_names.txt",
        "--device",
        device,
        "--work-dir",
        f"{output_dir}/work",
        "--json-out",
        f"{output_dir}/contest_auth_eval.json",
        "--keep-work-dir",
        "--expected-runtime-tree-sha256",
        expected_runtime_tree_sha256,
    ]


def _modal_cuda_command(
    *,
    candidate: AuthEvalRoundtripInput,
    repo_root: Path,
    output_dir: str,
    expected_runtime_tree_sha256: str,
    target_id: str,
    gpu: str = "T4",
    scorer_device: str = "cuda",
    inflate_device: str = "auto",
) -> list[str]:
    lane_id = f"{candidate.lane_id}_{target_id}"
    cmd = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_auth_eval.py",
        "--archive",
        _repo_rel(candidate.archive, repo_root),
        "--output-dir",
        output_dir,
        "--submission-dir",
        _repo_rel(candidate.submission_dir, repo_root),
        "--inflate-sh",
        candidate.inflate_sh,
        "--gpu",
        gpu,
        "--scorer-device",
        scorer_device,
        "--inflate-device",
        inflate_device,
        "--expected-runtime-tree-sha256",
        expected_runtime_tree_sha256,
        "--detach",
        "--provider-detach-ack",
        "--lane-id",
        lane_id,
        "--instance-job-id",
        "<job-id>",
        "--claim-agent",
        "codex:gpt-5.5",
    ]
    return cmd


def _modal_cpu_command(
    *,
    candidate: AuthEvalRoundtripInput,
    repo_root: Path,
    output_dir: str,
    expected_runtime_tree_sha256: str,
    target_id: str,
) -> list[str]:
    lane_id = f"{candidate.lane_id}_{target_id}"
    return [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_auth_eval_cpu.py",
        "--archive",
        _repo_rel(candidate.archive, repo_root),
        "--output-dir",
        output_dir,
        "--submission-dir",
        _repo_rel(candidate.submission_dir, repo_root),
        "--inflate-sh",
        candidate.inflate_sh,
        "--expected-runtime-tree-sha256",
        expected_runtime_tree_sha256,
        "--detach",
        "--provider-detach-ack",
        "--lane-id",
        lane_id,
        "--instance-job-id",
        "<job-id>",
        "--claim-agent",
        "codex:gpt-5.5",
    ]


def _row(
    *,
    target_id: str,
    runner: str,
    command: list[str],
    score_axis: str,
    evidence_grade: str,
    contest_compliant: bool,
    score_claim_possible: bool,
    hardware_substrate: str,
    scorer_device: str,
    inflate_device: str,
    expected_runtime_tree_sha256: str,
    diagnostic_blockers: list[str],
    dispatch_claim_required: bool,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "runner": runner,
        "command": command,
        "score_axis": score_axis,
        "evidence_grade": evidence_grade,
        "contest_compliant": contest_compliant,
        "score_claim_possible_after_recovery": score_claim_possible,
        "promotion_eligible_from_runner": False,
        "hardware_substrate": hardware_substrate,
        "scorer_device": scorer_device,
        "inflate_device": inflate_device,
        "expected_runtime_tree_sha256": expected_runtime_tree_sha256,
        "dispatch_claim_required": dispatch_claim_required,
        "diagnostic_blockers": diagnostic_blockers,
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_auth_eval_roundtrip_matrix(
    *,
    candidate: AuthEvalRoundtripInput,
    runtime_manifest: dict[str, Any],
    repo_root: Path,
    include_diagnostics: bool = True,
    host_system: str | None = None,
    host_machine: str | None = None,
) -> dict[str, Any]:
    """Return JSON-safe auth-eval target rows for a byte-closed packet."""

    host_system = host_system or platform.system()
    host_machine = host_machine or platform.machine()
    local_runtime_tree = str(runtime_manifest["runtime_tree_sha256"])
    modal_cuda_runtime = modal_uploaded_submission_dir_runtime_manifest(
        runtime_manifest,
        remote_submission_dir="/tmp/modal_auth_eval/submission_dir",
    )
    modal_cpu_runtime = modal_uploaded_submission_dir_runtime_manifest(
        runtime_manifest,
        remote_submission_dir="/tmp/modal_auth_eval_cpu/submission_dir",
    )
    base_output = f"{candidate.output_root}/{candidate.label}"
    contest_cpu_like = _host_is_contest_cpu_like(host_system, host_machine)

    rows = [
        _row(
            target_id="modal_contest_cuda_t4_auto",
            runner="modal_auth_eval_cuda",
            command=_modal_cuda_command(
                candidate=candidate,
                repo_root=repo_root,
                output_dir=f"{base_output}/modal_contest_cuda_t4_auto",
                expected_runtime_tree_sha256=str(modal_cuda_runtime["runtime_tree_sha256"]),
                target_id="modal_contest_cuda_t4_auto",
                gpu="T4",
                scorer_device="cuda",
                inflate_device="auto",
            ),
            score_axis="contest_cuda",
            evidence_grade="contest-CUDA",
            contest_compliant=True,
            score_claim_possible=True,
            hardware_substrate="linux_x86_64_t4",
            scorer_device="cuda",
            inflate_device="auto",
            expected_runtime_tree_sha256=str(modal_cuda_runtime["runtime_tree_sha256"]),
            diagnostic_blockers=[],
            dispatch_claim_required=True,
        ),
        _row(
            target_id="modal_contest_cpu_linux_x86_auto",
            runner="modal_auth_eval_cpu",
            command=_modal_cpu_command(
                candidate=candidate,
                repo_root=repo_root,
                output_dir=f"{base_output}/modal_contest_cpu_linux_x86_auto",
                expected_runtime_tree_sha256=str(modal_cpu_runtime["runtime_tree_sha256"]),
                target_id="modal_contest_cpu_linux_x86_auto",
            ),
            score_axis="contest_cpu",
            evidence_grade="contest-CPU",
            contest_compliant=True,
            score_claim_possible=True,
            hardware_substrate="linux_x86_64_cpu",
            scorer_device="cpu",
            inflate_device="auto",
            expected_runtime_tree_sha256=str(modal_cpu_runtime["runtime_tree_sha256"]),
            diagnostic_blockers=[],
            dispatch_claim_required=True,
        ),
        _row(
            target_id="local_cpu_current_host_auto",
            runner="local_contest_auth_eval",
            command=_local_command(
                candidate=candidate,
                repo_root=repo_root,
                output_dir=f"{base_output}/local_cpu_current_host_auto",
                device="cpu",
                expected_runtime_tree_sha256=local_runtime_tree,
            ),
            score_axis="contest_cpu" if contest_cpu_like else "macos_cpu_advisory",
            evidence_grade="contest-CPU" if contest_cpu_like else "macOS-CPU advisory",
            contest_compliant=contest_cpu_like,
            score_claim_possible=contest_cpu_like,
            hardware_substrate=f"{host_system}_{host_machine}",
            scorer_device="cpu",
            inflate_device="auto",
            expected_runtime_tree_sha256=local_runtime_tree,
            diagnostic_blockers=[] if contest_cpu_like else ["local_host_not_linux_x86_64"],
            dispatch_claim_required=False,
        ),
    ]

    if include_diagnostics:
        rows.extend(
            [
                _row(
                    target_id="local_mps_current_host_auto",
                    runner="local_contest_auth_eval",
                    command=_local_command(
                        candidate=candidate,
                        repo_root=repo_root,
                        output_dir=f"{base_output}/local_mps_current_host_auto",
                        device="mps",
                        expected_runtime_tree_sha256=local_runtime_tree,
                    ),
                    score_axis="mps_diagnostic",
                    evidence_grade="MPS diagnostic",
                    contest_compliant=False,
                    score_claim_possible=False,
                    hardware_substrate=f"{host_system}_{host_machine}",
                    scorer_device="mps",
                    inflate_device="auto",
                    expected_runtime_tree_sha256=local_runtime_tree,
                    diagnostic_blockers=["mps_not_contest_axis"],
                    dispatch_claim_required=False,
                ),
                _row(
                    target_id="modal_cuda_scorer_force_inflate_cpu_diagnostic",
                    runner="modal_auth_eval_cuda",
                    command=_modal_cuda_command(
                        candidate=candidate,
                        repo_root=repo_root,
                        output_dir=f"{base_output}/modal_cuda_scorer_force_inflate_cpu_diagnostic",
                        expected_runtime_tree_sha256=str(
                            modal_cuda_runtime["runtime_tree_sha256"]
                        ),
                        target_id="modal_cuda_scorer_force_inflate_cpu_diagnostic",
                        scorer_device="cuda",
                        inflate_device="cpu",
                    ),
                    score_axis="diagnostic_cuda",
                    evidence_grade="diagnostic-CUDA",
                    contest_compliant=False,
                    score_claim_possible=False,
                    hardware_substrate="linux_x86_64_t4",
                    scorer_device="cuda",
                    inflate_device="cpu",
                    expected_runtime_tree_sha256=str(modal_cuda_runtime["runtime_tree_sha256"]),
                    diagnostic_blockers=["inflate_device_override_present"],
                    dispatch_claim_required=True,
                ),
                _row(
                    target_id="modal_cuda_scorer_force_inflate_cuda_diagnostic",
                    runner="modal_auth_eval_cuda",
                    command=_modal_cuda_command(
                        candidate=candidate,
                        repo_root=repo_root,
                        output_dir=f"{base_output}/modal_cuda_scorer_force_inflate_cuda_diagnostic",
                        expected_runtime_tree_sha256=str(
                            modal_cuda_runtime["runtime_tree_sha256"]
                        ),
                        target_id="modal_cuda_scorer_force_inflate_cuda_diagnostic",
                        scorer_device="cuda",
                        inflate_device="cuda",
                    ),
                    score_axis="diagnostic_cuda",
                    evidence_grade="diagnostic-CUDA",
                    contest_compliant=False,
                    score_claim_possible=False,
                    hardware_substrate="linux_x86_64_t4",
                    scorer_device="cuda",
                    inflate_device="cuda",
                    expected_runtime_tree_sha256=str(modal_cuda_runtime["runtime_tree_sha256"]),
                    diagnostic_blockers=["inflate_device_override_present"],
                    dispatch_claim_required=True,
                ),
                _row(
                    target_id="modal_gpu_host_cpu_scorer_force_inflate_cpu_diagnostic",
                    runner="modal_auth_eval_cuda",
                    command=_modal_cuda_command(
                        candidate=candidate,
                        repo_root=repo_root,
                        output_dir=f"{base_output}/modal_gpu_host_cpu_scorer_force_inflate_cpu_diagnostic",
                        expected_runtime_tree_sha256=str(
                            modal_cuda_runtime["runtime_tree_sha256"]
                        ),
                        target_id="modal_gpu_host_cpu_scorer_force_inflate_cpu_diagnostic",
                        scorer_device="cpu",
                        inflate_device="cpu",
                    ),
                    score_axis="diagnostic_cpu",
                    evidence_grade="diagnostic-CPU",
                    contest_compliant=False,
                    score_claim_possible=False,
                    hardware_substrate="linux_x86_64_t4_host",
                    scorer_device="cpu",
                    inflate_device="cpu",
                    expected_runtime_tree_sha256=str(modal_cuda_runtime["runtime_tree_sha256"]),
                    diagnostic_blockers=[
                        "gpu_host_cpu_scorer_not_contest_cpu_runner",
                        "inflate_device_override_present",
                    ],
                    dispatch_claim_required=True,
                ),
            ]
        )

    return {
        "schema": SCHEMA_VERSION,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate": {
            "archive": candidate.archive,
            "submission_dir": candidate.submission_dir,
            "inflate_sh": candidate.inflate_sh,
            "label": candidate.label,
            "lane_id": candidate.lane_id,
        },
        "host": {"system": host_system, "machine": host_machine},
        "runtime_hashes": {
            "local_runtime_tree_sha256": local_runtime_tree,
            "modal_cuda_runtime_tree_sha256": str(modal_cuda_runtime["runtime_tree_sha256"]),
            "modal_cpu_runtime_tree_sha256": str(modal_cpu_runtime["runtime_tree_sha256"]),
            "runtime_content_tree_sha256": str(runtime_manifest["runtime_content_tree_sha256"]),
        },
        "contest_compliant_target_ids": [
            row["target_id"] for row in rows if row["contest_compliant"]
        ],
        "diagnostic_target_ids": [
            row["target_id"] for row in rows if not row["contest_compliant"]
        ],
        "targets": rows,
    }


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _result_dir_from_target(target: MappingLike) -> str | None:
    command = target.get("command")
    if not isinstance(command, list):
        return None
    output_dir = _command_value_after([str(value) for value in command], "--output-dir")
    if output_dir:
        return output_dir
    work_dir = _command_value_after([str(value) for value in command], "--work-dir")
    if work_dir and work_dir.endswith("/work"):
        return work_dir[: -len("/work")]
    return None


def _score_payload_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": payload.get("canonical_score", payload.get("score_recomputed_from_components")),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "n_samples": payload.get("n_samples"),
        "payload_evidence_grade": payload.get("evidence_grade"),
        "payload_score_axis": payload.get("score_axis"),
        "archive_sha256": (
            payload.get("provenance", {}).get("archive_sha256")
            if isinstance(payload.get("provenance"), dict)
            else None
        ),
    }


def _axis_result_class(target: MappingLike, score_payload: dict[str, Any] | None) -> str:
    if score_payload is None:
        return "not_recovered"
    if target.get("contest_compliant") is not True:
        return "diagnostic_only"
    axis = str(target.get("score_axis", ""))
    if axis == "contest_cuda":
        return "contest_cuda_anchor"
    if axis == "contest_cpu":
        return "contest_cpu_anchor"
    return "contest_axis_unknown"


MappingLike = dict[str, Any]


def collect_auth_eval_roundtrip_results(
    matrix: dict[str, Any],
    *,
    target_result_dirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Collect recovered auth-eval results for a previously built target matrix.

    ``target_result_dirs`` lets callers attach early or hand-dispatched runs
    whose output directories differ from the planned matrix command.  Rows
    without recovered JSON are kept as ``pending`` or ``missing`` so partial
    matrices do not silently lose signal.
    """

    target_result_dirs = target_result_dirs or {}
    targets = matrix.get("targets")
    if not isinstance(targets, list):
        raise ValueError("matrix must contain a targets list")

    rows: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id", ""))
        planned_dir = _result_dir_from_target(target)
        result_dir_text = target_result_dirs.get(target_id) or planned_dir
        result_dir = Path(result_dir_text) if result_dir_text else None

        score_payload = None
        recover_summary = None
        status = "missing"
        if result_dir is not None:
            score_payload = _read_json_if_present(result_dir / "contest_auth_eval.json")
            recover_summary = _read_json_if_present(result_dir / "modal_auth_eval_recover_summary.json")
            if score_payload is not None:
                status = "recovered"
            elif recover_summary is not None:
                status = str(recover_summary.get("status", "recover_summary_present"))

        fields = _score_payload_fields(score_payload) if score_payload is not None else {}
        rows.append(
            {
                "target_id": target_id,
                "status": status,
                "result_dir": result_dir.as_posix() if result_dir is not None else None,
                "runner": target.get("runner"),
                "score_axis": target.get("score_axis"),
                "evidence_grade": target.get("evidence_grade"),
                "contest_compliant": bool(target.get("contest_compliant")),
                "diagnostic_blockers": list(target.get("diagnostic_blockers") or []),
                "contest_axis_anchor": bool(
                    status == "recovered"
                    and target.get("score_claim_possible_after_recovery") is True
                    and target.get("contest_compliant") is True
                ),
                "score_claim_possible_after_result_review": bool(
                    target.get("score_claim_possible_after_recovery") is True
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "result_review_blockers": [
                    "roundtrip_matrix_is_command_planner_not_claim_surface",
                    "requires_separate_auth_eval_result_review_before_score_claim",
                ],
                "axis_result_class": _axis_result_class(target, score_payload),
                "recover_summary_status": (
                    recover_summary.get("status") if recover_summary is not None else None
                ),
                **fields,
            }
        )

    recovered_scores = [
        row
        for row in rows
        if row["status"] == "recovered" and isinstance(row.get("score"), int | float)
    ]
    return {
        "schema": RESULTS_SCHEMA_VERSION,
        "matrix_schema": matrix.get("schema"),
        "score_claim": False,
        "promotion_eligible": False,
        "candidate": matrix.get("candidate"),
        "target_count": len(rows),
        "recovered_count": len(recovered_scores),
        "pending_count": sum(1 for row in rows if row["status"] == "pending"),
        "rows": rows,
    }


__all__ = [
    "AuthEvalRoundtripInput",
    "RESULTS_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_auth_eval_roundtrip_matrix",
    "collect_auth_eval_roundtrip_results",
]
