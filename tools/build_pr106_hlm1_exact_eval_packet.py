#!/usr/bin/env python3
"""Build a fail-closed exact-eval operator packet for PR106 HLM1 recodes.

The packet is dispatch authority only for the static handoff shape: archive
custody, runtime-tree custody, static compliance, and the canonical Modal CUDA
auth-eval command. It never runs remote work and never claims a score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.deploy.modal.auth_eval import (  # noqa: E402
    modal_uploaded_submission_dir_runtime_manifest,
    sha256_bytes,
    submission_dir_zip_bytes,
)
from tac.packet_compiler.pr106_hlm1_runtime_consumption import (  # noqa: E402
    prove_pr106_hlm1_runtime_consumption,
)
from tac.repo_io import read_json, repo_relative, sha256_file, write_json  # noqa: E402

DEFAULT_RESULT_DIR = Path(
    "experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex"
)
DEFAULT_STATIC_RELEASE_SURFACE = DEFAULT_RESULT_DIR / "static_release_surface"
DEFAULT_MANIFEST = DEFAULT_RESULT_DIR / "manifest.json"
DEFAULT_PACKETIR_IDENTITY = DEFAULT_RESULT_DIR / "packetir_identity.json"
DEFAULT_RUNTIME_CONSUMPTION = DEFAULT_RESULT_DIR / "runtime_decode_consumption.json"
DEFAULT_HLM1_RUNTIME_CONSUMPTION = DEFAULT_RESULT_DIR / "hlm1_runtime_consumption.json"
DEFAULT_PREFIX_PARITY = DEFAULT_RESULT_DIR / "same_runtime_prefix_parity.json"
DEFAULT_PUBLIC_STATIC_COMPLIANCE = (
    DEFAULT_RESULT_DIR / "pre_submission_compliance.static_clean.public.json"
)
DEFAULT_STATIC_COMPLIANCE = DEFAULT_PUBLIC_STATIC_COMPLIANCE
DEFAULT_RUNTIME_MANIFEST = DEFAULT_RESULT_DIR / "runtime_tree_manifest.json"
DEFAULT_MODAL_TRANSPORT = DEFAULT_RESULT_DIR / "modal_submission_dir_transport.json"
DEFAULT_PACKET = DEFAULT_RESULT_DIR / "hlm1_exact_eval_packet.json"
DEFAULT_JOB_NAME = "exact_eval_hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513"
DEFAULT_LANE_ID = "hnerv_hlm1_fixed_latent_recode_exact_eval"
DEFAULT_OUTPUT_DIR = (
    "experiments/results/modal_auth_eval/"
    "hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513"
)
REQUIRED_ARTIFACTS = (
    "manifest",
    "packetir_identity",
    "runtime_consumption",
    "prefix_parity",
    "static_compliance",
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: Path) -> str:
    return repo_relative(_repo_path(path), REPO_ROOT)


def _q(value: object) -> str:
    text = str(value)
    if text.startswith("$") or text.startswith("${"):
        return text
    return shlex.quote(text)


def _one_liner(parts: list[object]) -> str:
    return " ".join(_q(part) for part in parts)


def _now_utc() -> str:
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _git_bytes(args: list[str]) -> bytes:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return b""


def _source_tree_custody() -> dict[str, Any]:
    """Record build-time source custody, including untracked content hashes."""

    status = _git_bytes(["git", "status", "--short", "--untracked-files=all"]).decode(
        "utf-8",
        errors="replace",
    )
    diff = _git_bytes(["git", "diff", "--binary", "HEAD"])
    untracked_raw = _git_bytes(["git", "ls-files", "--others", "--exclude-standard", "-z"])
    untracked_files: list[dict[str, Any]] = []
    for raw_path in [item for item in untracked_raw.split(b"\0") if item]:
        rel = raw_path.decode("utf-8", errors="surrogateescape")
        path = REPO_ROOT / rel
        if not path.is_file() or path.is_symlink():
            continue
        blob = path.read_bytes()
        untracked_files.append(
            {
                "path": rel,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    untracked_files.sort(key=lambda row: row["path"])
    untracked_hasher = hashlib.sha256()
    for row in untracked_files:
        untracked_hasher.update(f"{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    status_lines = status.splitlines()
    custody_hasher = hashlib.sha256()
    custody_hasher.update(b"provider_dispatch_source_tree_custody_v2\n")
    custody_hasher.update(_git_commit().encode())
    custody_hasher.update(b"\n")
    custody_hasher.update(hashlib.sha256(diff).hexdigest().encode())
    custody_hasher.update(b"\n")
    custody_hasher.update(untracked_hasher.hexdigest().encode())
    custody_hasher.update(b"\n")
    for line in status_lines:
        custody_hasher.update(line.encode("utf-8", errors="surrogateescape"))
        custody_hasher.update(b"\n")
    return {
        "schema": "provider_dispatch_source_tree_custody_v2",
        "head_commit": _git_commit(),
        "dirty": bool(status.strip()),
        "status_short": status_lines,
        "diff_against_head_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_against_head_bytes": len(diff),
        "untracked_path_count": len(untracked_files),
        "untracked_files": untracked_files,
        "untracked_content_tree_sha256": untracked_hasher.hexdigest(),
        "source_tree_custody_sha256": custody_hasher.hexdigest(),
        "custody_scope": "build_time_local_tree_before_remote_dispatch",
        "provider_wrapper_records_actual_source_repo_commit_at_dispatch": True,
        "score_claim": False,
    }


def _contest_auth_eval_module() -> Any:
    module_path = REPO_ROOT / "experiments" / "contest_auth_eval.py"
    spec = importlib.util.spec_from_file_location("pact_hlm1_contest_auth_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load contest_auth_eval module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_manifest(*, static_release_surface: Path, upstream_dir: Path) -> dict[str, Any]:
    module = _contest_auth_eval_module()
    manifest_fn = getattr(module, "_runtime_dependency_manifest")
    return manifest_fn(
        _repo_path(static_release_surface) / "inflate.sh",
        _repo_path(upstream_dir),
        repo_root=REPO_ROOT,
    )


def _sanitize_for_public(payload: Any) -> Any:
    repo_text = str(REPO_ROOT.resolve())
    if isinstance(payload, str):
        return payload.replace(repo_text, "<repo>")
    if isinstance(payload, list):
        return [_sanitize_for_public(item) for item in payload]
    if isinstance(payload, dict):
        return {str(key): _sanitize_for_public(value) for key, value in payload.items()}
    return payload


def _runtime_custody_manifest(
    *,
    local_manifest: dict[str, Any],
    modal_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "pr106_hlm1_runtime_custody_manifest_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_runtime": "modal_uploaded_submission_dir",
        "local_static_release_surface_runtime": _sanitize_for_public(local_manifest),
        "modal_uploaded_submission_dir_runtime": modal_manifest,
        "runtime_tree_sha256": modal_manifest.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": modal_manifest.get("runtime_content_tree_sha256"),
        "local_runtime_tree_sha256": local_manifest.get("runtime_tree_sha256"),
        "local_runtime_content_tree_sha256": local_manifest.get("runtime_content_tree_sha256"),
    }


def _archive_identity(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": _repo_rel(path),
        "exists": full.is_file(),
        "bytes": full.stat().st_size if full.is_file() else None,
        "sha256": sha256_file(full) if full.is_file() else None,
    }


def _artifact_hash(path: Path) -> dict[str, Any]:
    full = _repo_path(path)
    return {
        "path": _repo_rel(path),
        "exists": full.is_file(),
        "bytes": full.stat().st_size if full.is_file() else None,
        "sha256": sha256_file(full) if full.is_file() else None,
    }


def _transport_manifest(static_release_surface: Path) -> dict[str, Any]:
    runtime_dir = _repo_path(static_release_surface)
    payload = submission_dir_zip_bytes(runtime_dir)
    return {
        "schema": "pr106_hlm1_modal_submission_dir_transport_v1",
        "submission_dir": _repo_rel(static_release_surface),
        "transport_zip_bytes": len(payload),
        "transport_zip_sha256": sha256_bytes(payload),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest,
        "packetir_identity": args.packetir_identity,
        "runtime_consumption": args.runtime_consumption,
        "prefix_parity": args.prefix_parity,
        "static_compliance": args.static_compliance,
    }
    payloads: dict[str, Any] = {}
    for name in REQUIRED_ARTIFACTS:
        payloads[name] = read_json(_repo_path(paths[name]))
    return payloads


def _local_blockers(
    *,
    args: argparse.Namespace,
    payloads: dict[str, Any],
    archive: dict[str, Any],
    runtime_manifest: dict[str, Any],
) -> list[str]:
    manifest = payloads["manifest"]
    packetir_identity = payloads["packetir_identity"]
    runtime_consumption = payloads["runtime_consumption"]
    hlm1_runtime_consumption = payloads["hlm1_runtime_consumption"]
    prefix_parity = payloads["prefix_parity"]
    static_compliance = payloads["static_compliance"]
    blockers: list[str] = []
    if archive["exists"] is not True:
        blockers.append("candidate_archive_missing")
    if archive["sha256"] != manifest.get("candidate_archive_sha256"):
        blockers.append("candidate_archive_sha256_mismatch")
    if archive["bytes"] != manifest.get("candidate_archive_bytes"):
        blockers.append("candidate_archive_bytes_mismatch")
    if manifest.get("score_claim") is not False:
        blockers.append("manifest_score_claim_not_false")
    if manifest.get("ready_for_archive_preflight") is not True:
        blockers.append("manifest_archive_preflight_not_ready")
    if packetir_identity.get("packet_ir_identity_passed") is not True:
        blockers.append("packetir_identity_not_passed")
    if packetir_identity.get("runtime_consumption_claim") is not False:
        blockers.append("packetir_identity_overclaims_runtime_consumption")
    if runtime_consumption.get("runtime_sidecar_decode_consumption_claim") is not True:
        blockers.append("runtime_sidecar_decode_consumption_missing")
    if runtime_consumption.get("runtime_sidecar_apply_consumption_claim") is not True:
        blockers.append("runtime_sidecar_apply_consumption_missing")
    if runtime_consumption.get("score_claim") is not False:
        blockers.append("runtime_consumption_score_claim_not_false")
    if hlm1_runtime_consumption.get("runtime_hlm1_decode_consumption_claim") is not True:
        blockers.append("runtime_hlm1_decode_consumption_missing")
    if hlm1_runtime_consumption.get("runtime_hlm1_valid_mutation_changes_raw") is not True:
        blockers.append("runtime_hlm1_valid_mutation_did_not_change_raw")
    if hlm1_runtime_consumption.get("score_claim") is not False:
        blockers.append("hlm1_runtime_consumption_score_claim_not_false")
    if prefix_parity.get("prefix_parity_claim") is not True:
        blockers.append("same_runtime_prefix_parity_missing")
    if prefix_parity.get("full_frame_inflate_output_parity_claim") is not False:
        blockers.append("prefix_parity_overclaims_full_frame")
    if prefix_parity.get("score_claim") is not False:
        blockers.append("prefix_parity_score_claim_not_false")
    if static_compliance.get("passed") is not True:
        blockers.append("static_pre_submission_compliance_not_passed")
    if not runtime_manifest.get("runtime_tree_sha256"):
        blockers.append("runtime_tree_sha256_missing")
    if not (_repo_path(args.static_release_surface) / "inflate.sh").is_file():
        blockers.append("static_release_inflate_sh_missing")
    return blockers


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    payloads = _load_inputs(args)
    manifest = payloads["manifest"]
    archive_path = Path(manifest["candidate_archive_path"])
    archive = _archive_identity(archive_path)
    local_runtime_manifest = _runtime_manifest(
        static_release_surface=args.static_release_surface,
        upstream_dir=args.upstream_dir,
    )
    modal_runtime_manifest = modal_uploaded_submission_dir_runtime_manifest(
        local_runtime_manifest
    )
    runtime_manifest = _runtime_custody_manifest(
        local_manifest=local_runtime_manifest,
        modal_manifest=modal_runtime_manifest,
    )
    hlm1_runtime_consumption = prove_pr106_hlm1_runtime_consumption(
        archive_path=archive_path,
        runtime_dir=args.static_release_surface,
        repo_root=REPO_ROOT,
    )
    transport = _transport_manifest(args.static_release_surface)
    payloads["hlm1_runtime_consumption"] = hlm1_runtime_consumption
    static_compliance_public = _sanitize_for_public(payloads["static_compliance"])
    static_compliance_public["public_hygiene_note"] = (
        "Absolute local repo paths redacted from raw static compliance output; "
        "pass/fail/check payload preserved."
    )
    static_compliance_public["static_compliance_scope"] = "static_pre_exact_eval_only"
    static_compliance_public["exact_eval_required"] = True
    write_json(_repo_path(args.runtime_manifest_out), runtime_manifest)
    write_json(_repo_path(args.modal_transport_out), transport)
    write_json(_repo_path(args.hlm1_runtime_consumption), hlm1_runtime_consumption)
    write_json(_repo_path(args.static_compliance_public_out), static_compliance_public)

    local_blockers = _local_blockers(
        args=args,
        payloads=payloads,
        archive=archive,
        runtime_manifest=runtime_manifest,
    )
    static_ready = not local_blockers
    source_repo_commit = _git_commit()
    modal_submit = "PYTHONPATH=src:upstream:$PWD " + _one_liner(
        [
            ".venv/bin/modal",
            "run",
            "--detach",
            "experiments/modal_auth_eval.py",
            "--archive",
            archive_path,
            "--output-dir",
            args.modal_output_dir,
            "--inflate-sh",
            "inflate.sh",
            "--submission-dir",
            args.static_release_surface,
            "--gpu",
            args.modal_gpu,
            "--scorer-device",
            "cuda",
            "--inflate-device",
            "auto",
            "--detach",
            "--provider-detach-ack",
            "--lane-id",
            args.lane_id,
            "--instance-job-id",
            args.job_name,
            "--claim-agent",
            args.claim_agent,
            "--expected-runtime-tree-sha256",
            runtime_manifest.get("runtime_tree_sha256", ""),
            "--claim-notes",
            (
                f"PR106 HDM4+HLM1 exact CUDA auth eval; archive_sha256={archive['sha256']}; "
                f"bytes={archive['bytes']}; runtime_tree_sha256="
                f"{runtime_manifest.get('runtime_tree_sha256')}; "
                f"submission_dir_zip_sha256={transport['transport_zip_sha256']}"
            ),
        ]
    )
    local_cuda_smoke = _one_liner(
        [
            ".venv/bin/python",
            "experiments/contest_auth_eval.py",
            "--archive",
            archive_path,
            "--inflate-sh",
            args.static_release_surface / "inflate.sh",
            "--upstream-dir",
            args.upstream_dir,
            "--device",
            "cuda",
            "--expected-runtime-tree-sha256",
            runtime_manifest.get("local_runtime_tree_sha256", ""),
            "--json-out",
            args.result_dir / "contest_auth_eval.cuda.json",
        ]
    )
    recover = _one_liner(
        [
            ".venv/bin/python",
            "tools/recover_modal_auth_eval.py",
            "--output-dir",
            args.modal_output_dir,
        ]
    )
    source_tree = _source_tree_custody()

    remaining_required = [
        "modal_cuda_auth_eval_dispatch",
        "modal_auth_eval_harvest",
        "contest_auth_eval_adjudication",
        "operator_score_claim_review",
    ]
    if not args.operator_approved_exact_cuda:
        remaining_required.insert(0, "operator_exact_cuda_approval")
    if local_blockers:
        remaining_required.insert(0, "local_static_blocker_resolution")

    operator_next_steps = {
        "schema": "hnerv_hlm1_operator_next_steps_v1",
        "copy_safe": True,
        "must_run_in_order": True,
        "packet_path": _repo_rel(args.json_out),
        "current_submit_blockers": local_blockers
        + ([] if args.operator_approved_exact_cuda else ["operator_exact_cuda_approval_required"]),
        "first_remote_gpu_step": "submit_modal_exact_cuda",
        "steps": [
            {
                "id": "refresh_static_packet_no_dispatch",
                "order": 1,
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "regenerate HLM1 runtime custody and fail-closed exact-eval packet",
                "copy_safe_command": _one_liner(
                    [
                        ".venv/bin/python",
                        "tools/build_pr106_hlm1_exact_eval_packet.py",
                        "--json-out",
                        args.json_out,
                    ]
                ),
            },
            {
                "id": "optional_local_cuda_exact_eval",
                "order": 2,
                "dispatches_remote_gpu": True,
                "writes_repo_state": True,
                "purpose": "run only on a real CUDA host; MPS and macOS CPU are not promotion axes",
                "copy_safe_command": local_cuda_smoke,
            },
            {
                "id": "submit_modal_exact_cuda",
                "order": 3,
                "dispatches_remote_gpu": True,
                "writes_repo_state": True,
                "purpose": "canonical Modal T4 exact CUDA auth eval; wrapper records the Level-2 claim before spawn",
                "copy_safe_command": modal_submit,
            },
            {
                "id": "harvest_modal_exact_cuda",
                "order": 4,
                "dispatches_remote_gpu": False,
                "writes_repo_state": True,
                "purpose": "recover detached Modal artifacts before any score or promotion claim",
                "copy_safe_command": recover,
            },
        ],
    }

    return {
        "packet_kind": "hnerv_hlm1_exact_eval_operator_packet",
        "schema_version": 1,
        "recorded_at_utc": _now_utc(),
        "source_repo_commit": source_repo_commit,
        "source_tree_custody": source_tree,
        "source_tree_custody_note": (
            "Build-time local source custody only; the Modal wrapper records "
            "the actual source_repo_commit at dispatch and contest_auth_eval "
            "must enforce the expected runtime-tree hash."
        ),
        "candidate_id": manifest.get("candidate_id"),
        "lane_id": args.lane_id,
        "job_name": args.job_name,
        "name": "PR106 HDM4+HLM1 fixed-latent recode",
        "family": "hnerv_fixed_latent_recode",
        "pareto_scope": "hnerv_rate_only_exact_archive",
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch_claim": bool(static_ready),
        "ready_for_submit": bool(static_ready and args.operator_approved_exact_cuda),
        "operator_approved_exact_cuda": bool(args.operator_approved_exact_cuda),
        "approved_exact_eval_target": bool(static_ready and args.operator_approved_exact_cuda),
        "approval_scope": (
            "operator approval for exact CUDA work only; Modal wrapper must claim lane, "
            "dispatch, harvest, and adjudication before any score claim"
        ),
        "blockers": local_blockers
        + ([] if args.operator_approved_exact_cuda else ["operator_exact_cuda_approval_required"]),
        "remaining_required_for_score_or_promotion": remaining_required,
        "missing_env": [],
        "archive_sha256": archive["sha256"],
        "archive_bytes": archive["bytes"],
        "archive_identity": archive,
        "source_archive_sha256": manifest.get("source_archive_sha256"),
        "source_archive_bytes": manifest.get("source_archive_bytes"),
        "byte_delta": manifest.get("candidate_archive_byte_delta"),
        "expected_total_score_delta_rate_only": manifest.get(
            "candidate_rate_score_delta_if_components_equal"
        ),
        "expected_seg_dist_delta": 0.0,
        "expected_pose_dist_delta": 0.0,
        "score_affecting_runtime_changed": True,
        "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
        "runtime_content_tree_sha256": runtime_manifest.get("runtime_content_tree_sha256"),
        "local_runtime_tree_sha256": runtime_manifest.get("local_runtime_tree_sha256"),
        "local_runtime_content_tree_sha256": runtime_manifest.get(
            "local_runtime_content_tree_sha256"
        ),
        "runtime_manifest": {
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_content_tree_sha256": runtime_manifest.get("runtime_content_tree_sha256"),
            "local_runtime_tree_sha256": runtime_manifest.get("local_runtime_tree_sha256"),
            "runtime_file_count": modal_runtime_manifest.get("runtime_file_count"),
            "path": _repo_rel(args.runtime_manifest_out),
        },
        "modal_submission_dir_transport": {
            "path": _repo_rel(args.modal_transport_out),
            "transport_zip_sha256": transport["transport_zip_sha256"],
            "transport_zip_bytes": transport["transport_zip_bytes"],
        },
        "preflight_ready": bool(static_ready),
        "static_packet_ready": bool(static_ready),
        "static_compliance_ok": payloads["static_compliance"].get("passed") is True,
        "static_compliance_scope": "static_pre_exact_eval_only",
        "static_compliance_auth_eval_custody_present": bool(
            (payloads["static_compliance"].get("auth_eval") or {}).get("exists")
        ),
        "exact_eval_required": True,
        "payload_diff_ready": bool(
            manifest.get("candidate_archive_byte_delta", 0) < 0
            and manifest.get("source_archive_sha256") != archive["sha256"]
        ),
        "dry_run_ready": bool(static_ready),
        "full_frame_inflate_output_parity_claim": False,
        "prefix_parity_claim": payloads["prefix_parity"].get("prefix_parity_claim") is True,
        "runtime_sidecar_decode_consumption_claim": payloads["runtime_consumption"].get(
            "runtime_sidecar_decode_consumption_claim"
        )
        is True,
        "runtime_sidecar_apply_consumption_claim": payloads["runtime_consumption"].get(
            "runtime_sidecar_apply_consumption_claim"
        )
        is True,
        "runtime_hlm1_decode_consumption_claim": hlm1_runtime_consumption.get(
            "runtime_hlm1_decode_consumption_claim"
        )
        is True,
        "runtime_hlm1_valid_mutation_changes_raw": hlm1_runtime_consumption.get(
            "runtime_hlm1_valid_mutation_changes_raw"
        )
        is True,
        "artifacts": {
            "manifest": _repo_rel(args.manifest),
            "packetir_identity": _repo_rel(args.packetir_identity),
            "sidecar_runtime_consumption": _repo_rel(args.runtime_consumption),
            "hlm1_runtime_consumption": _repo_rel(args.hlm1_runtime_consumption),
            "prefix_parity": _repo_rel(args.prefix_parity),
            "pre_submission_compliance": _repo_rel(args.static_compliance_public_out),
            "runtime_manifest": _repo_rel(args.runtime_manifest_out),
            "modal_submission_dir_transport": _repo_rel(args.modal_transport_out),
            "static_release_surface": _repo_rel(args.static_release_surface),
            "packet": _repo_rel(args.json_out),
        },
        "artifact_identities": {
            key: _artifact_hash(path)
            for key, path in {
                "manifest": args.manifest,
                "packetir_identity": args.packetir_identity,
                "runtime_consumption": args.runtime_consumption,
                "hlm1_runtime_consumption": args.hlm1_runtime_consumption,
                "prefix_parity": args.prefix_parity,
                "pre_submission_compliance": args.static_compliance_public_out,
            }.items()
        },
        "commands": {
            "claim": (
                "Modal wrapper records the Level-2 active lane claim before provider spawn; "
                "do not pre-claim separately unless using a different provider actuator."
            ),
            "submit": modal_submit,
            "harvest": recover,
            "local_cuda_exact_eval": local_cuda_smoke,
        },
        "operator_next_steps": operator_next_steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--static-release-surface", type=Path, default=DEFAULT_STATIC_RELEASE_SURFACE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--packetir-identity", type=Path, default=DEFAULT_PACKETIR_IDENTITY)
    parser.add_argument("--runtime-consumption", type=Path, default=DEFAULT_RUNTIME_CONSUMPTION)
    parser.add_argument(
        "--hlm1-runtime-consumption",
        type=Path,
        default=DEFAULT_HLM1_RUNTIME_CONSUMPTION,
    )
    parser.add_argument("--prefix-parity", type=Path, default=DEFAULT_PREFIX_PARITY)
    parser.add_argument("--static-compliance", type=Path, default=DEFAULT_STATIC_COMPLIANCE)
    parser.add_argument(
        "--static-compliance-public-out",
        type=Path,
        default=DEFAULT_PUBLIC_STATIC_COMPLIANCE,
    )
    parser.add_argument("--runtime-manifest-out", type=Path, default=DEFAULT_RUNTIME_MANIFEST)
    parser.add_argument("--modal-transport-out", type=Path, default=DEFAULT_MODAL_TRANSPORT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--modal-output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modal-gpu", default="T4", choices=("T4", "A100", "A100-40GB", "A100-80GB", "H100", "H100-80GB"))
    parser.add_argument("--claim-agent", default="codex:gpt-5.5")
    parser.add_argument(
        "--operator-approved-exact-cuda",
        action="store_true",
        help="Mark static packet ready for the canonical exact-CUDA submit command.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_packet(args)
    write_json(_repo_path(args.json_out), packet)
    print(f"wrote {_repo_rel(args.json_out)}")
    print(f"ready_for_submit={packet['ready_for_submit']} blockers={packet['blockers']}")
    return 0 if packet["preflight_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
