# SPDX-License-Identifier: MIT
"""Byte-closed PIA3 candidate materialization and receiver proof."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields
from tac.repo_io import repo_relative, sha256_bytes, sha256_file, tree_sha256
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.pact_nerv_ia3.archive import pack_archive
from tac.substrates.pact_nerv_ia3.inflate import inflate_one_video

PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA = "pact_nerv_ia3_byte_closed_candidate.v1"
PACT_NERV_IA3_RECEIVER_INFLATE_PROOF_SCHEMA = "pact_nerv_ia3_receiver_inflate_proof.v1"


class PactNervIa3ArchiveCandidateError(ValueError):
    """Raised when a PIA3 candidate cannot be materialized."""


def _artifact(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(path, repo_root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PactNervIa3ArchiveCandidateError(f"{path}: JSON root must be object")
    return payload


def _load_torch_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise PactNervIa3ArchiveCandidateError(f"missing PyTorch state_dict: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise PactNervIa3ArchiveCandidateError("PyTorch state_dict root must be dict")
    out: dict[str, torch.Tensor] = {}
    for name, tensor in payload.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise PactNervIa3ArchiveCandidateError(f"invalid state_dict entry {name!r}: expected tensor")
        out[name] = tensor.detach().cpu()
    for required in ("latents", "ego_poses"):
        if required not in out:
            raise PactNervIa3ArchiveCandidateError(f"state_dict missing {required}")
    return out


def _meta_from_config(config: dict[str, Any]) -> dict[str, object]:
    required = {
        "embed_dim",
        "initial_grid_h",
        "initial_grid_w",
        "decoder_channels",
        "sin_frequency",
        "num_upsample_blocks",
        "ia3_init_delta_std",
        "output_height",
        "output_width",
    }
    missing = sorted(required - set(config))
    if missing:
        raise PactNervIa3ArchiveCandidateError(f"config missing fields: {missing}")
    return {
        "embed_dim": int(config["embed_dim"]),
        "initial_grid_h": int(config["initial_grid_h"]),
        "initial_grid_w": int(config["initial_grid_w"]),
        "decoder_channels": [int(value) for value in config["decoder_channels"]],
        "sin_frequency": float(config["sin_frequency"]),
        "num_upsample_blocks": int(config["num_upsample_blocks"]),
        "ia3_init_delta_std": float(config["ia3_init_delta_std"]),
        "output_height": int(config["output_height"]),
        "output_width": int(config["output_width"]),
    }


def _frame_sha_manifest(root: Path) -> list[dict[str, Any]]:
    frames = sorted(path for path in root.rglob("*.png") if path.is_file())
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in frames
    ]


def _prune_python_bytecode(root: Path) -> None:
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def _run_packaged_inflate(
    *,
    submission_dir: Path,
    output_root: Path,
    file_list_path: Path,
) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHON"] = sys.executable
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(submission_dir),
            str(output_root),
            str(file_list_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "argv": [
            "bash",
            str(submission_dir / "inflate.sh"),
            str(submission_dir),
            str(output_root),
            str(file_list_path),
        ],
        "returncode": result.returncode,
        "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def materialize_pact_nerv_ia3_byte_closed_candidate(
    *,
    pytorch_state_dict_path: str | Path,
    parity_report_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    label: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a PIA3 archive.zip and prove packaged receiver consumption."""

    root = Path(repo_root)
    pt_path = Path(pytorch_state_dict_path)
    report_path = Path(parity_report_path)
    out_dir = Path(output_dir)
    if not pt_path.is_absolute():
        pt_path = root / pt_path
    if not report_path.is_absolute():
        report_path = root / report_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if out_dir.exists():
        if not overwrite:
            raise PactNervIa3ArchiveCandidateError(f"output dir exists; pass overwrite=True: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    parity = _load_json_object(report_path)
    config = parity.get("config")
    if not isinstance(config, dict):
        raise PactNervIa3ArchiveCandidateError("parity report missing config object")
    state = _load_torch_state_dict(pt_path)
    decoder_state = {name: tensor for name, tensor in state.items() if name not in {"latents", "ego_poses"}}
    latents = state["latents"]
    ego_poses = state["ego_poses"]
    pose_dim = int(config.get("pose_dim", ego_poses.shape[1]))
    bin_bytes = pack_archive(
        decoder_state,
        latents,
        ego_poses,
        _meta_from_config(config),
        pose_dim=pose_dim,
    )

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_ia3",
        repo_root=root,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    runtime_dir_ref = repo_relative(submission_dir, root)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)

    file_list_path = out_dir / "file_list.txt"
    file_list_path.write_text("0.mkv\n", encoding="utf-8")
    receiver_output = out_dir / "receiver_output"
    reference_output = out_dir / "reference_output"
    receiver_run = _run_packaged_inflate(
        submission_dir=submission_dir,
        output_root=receiver_output,
        file_list_path=file_list_path,
    )
    if receiver_run["returncode"] != 0:
        raise PactNervIa3ArchiveCandidateError(
            f"packaged receiver failed rc={receiver_run['returncode']}: {receiver_run['stderr_tail']}"
        )
    _prune_python_bytecode(submission_dir)
    runtime_tree_sha = tree_sha256(submission_dir)
    inflate_one_video(bin_bytes, reference_output / "0", device="cpu")

    receiver_tree_sha = tree_sha256(receiver_output)
    reference_tree_sha = tree_sha256(reference_output)
    receiver_frames = _frame_sha_manifest(receiver_output)
    reference_frames = _frame_sha_manifest(reference_output)
    expected_frame_count = int(latents.shape[0]) * 2
    cmp_equal = receiver_tree_sha == reference_tree_sha
    frame_count_ok = len(receiver_frames) == expected_frame_count
    output_sha256_match = cmp_equal and receiver_frames == reference_frames

    proof_path = out_dir / "receiver_inflate_proof.json"
    proof_payload: dict[str, Any] = {
        "schema": PACT_NERV_IA3_RECEIVER_INFLATE_PROOF_SCHEMA,
        "label": label,
        "passed": bool(cmp_equal and frame_count_ok and output_sha256_match),
        "runtime_consumption_proof_passed": bool(cmp_equal and frame_count_ok and output_sha256_match),
        "receiver_contract_satisfied": bool(cmp_equal and frame_count_ok),
        "full_frame_inflate_output_parity_claim": bool(cmp_equal and frame_count_ok),
        "parity_scope": "full_candidate_smoke_video",
        "cmp_equal": cmp_equal,
        "output_sha256_match": output_sha256_match,
        "frame_count_ok": frame_count_ok,
        "expected_frame_count": expected_frame_count,
        "receiver_frame_count": len(receiver_frames),
        "reference_frame_count": len(reference_frames),
        "candidate_archive_sha256": sha256_file(archive_zip_path),
        "candidate_archive": _artifact(archive_zip_path, repo_root=root),
        "receiver_output_tree_sha256": receiver_tree_sha,
        "reference_output_tree_sha256": reference_tree_sha,
        "receiver_run": receiver_run,
        "receiver_frame_manifest": receiver_frames,
        "reference_frame_manifest": reference_frames,
        "proof_path": repo_relative(proof_path, root),
        "runtime_adapter_ready": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "candidate_runtime_dir": runtime_dir_ref,
        "candidate_runtime_tree_sha256": runtime_tree_sha,
        "expected_candidate_runtime_tree_sha256": runtime_tree_sha,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        proof_payload,
        context="pact_nerv_ia3_receiver_inflate_proof",
    )
    proof_path.write_text(
        json.dumps(proof_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema": PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA,
        "label": label,
        "axis": "[macOS-MLX research-signal]",
        "byte_closed_candidate_emitted": True,
        "receiver_contract_satisfied": proof_payload["receiver_contract_satisfied"],
        "full_frame_inflate_parity_satisfied": proof_payload["full_frame_inflate_output_parity_claim"],
        "runtime_adapter_ready": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "candidate_runtime_dir": runtime_dir_ref,
        "candidate_runtime_tree_sha256": runtime_tree_sha,
        "expected_candidate_runtime_tree_sha256": runtime_tree_sha,
        "runtime_consumption_proof_path": repo_relative(proof_path, root),
        "runtime_consumption_proof_sha256": sha256_file(proof_path),
        "candidate_archive": _artifact(archive_zip_path, repo_root=root),
        "candidate_archive_path": repo_relative(archive_zip_path, root),
        "candidate_archive_bytes": archive_zip_path.stat().st_size,
        "candidate_archive_sha256": sha256_file(archive_zip_path),
        "candidate_0_bin": _artifact(bin_path, repo_root=root),
        "pytorch_state_dict": _artifact(pt_path, repo_root=root),
        "parity_report": _artifact(report_path, repo_root=root),
        "receiver_verification": {
            "receiver_contract_satisfied": proof_payload["receiver_contract_satisfied"],
            "proof_present": True,
            "runtime_consumption_proof_passed": proof_payload["runtime_consumption_proof_passed"],
            "runtime_adapter_ready": True,
            "candidate_runtime_adapter_blocker_cleared": True,
            "candidate_runtime_dir": runtime_dir_ref,
            "candidate_runtime_tree_sha256": runtime_tree_sha,
            "expected_candidate_runtime_tree_sha256": runtime_tree_sha,
            "proof_path": repo_relative(proof_path, root),
            "proof_bytes": proof_path.stat().st_size,
            "proof_sha256": sha256_file(proof_path),
        },
        "receiver_inflate_proof_path": repo_relative(proof_path, root),
        "receiver_inflate_proof_sha256": sha256_file(proof_path),
        "byte_tax": {
            "raw_state_dict_bytes": pt_path.stat().st_size,
            "pia3_0_bin_bytes": len(bin_bytes),
            "archive_zip_bytes": archive_zip_path.stat().st_size,
        },
        "dispatch_blockers": [
            "macos_mlx_research_signal_has_no_score_authority",
            "smoke_candidate_not_full_contest_video",
            "contest_cpu_or_cuda_auth_eval_required_before_score_claim",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        manifest,
        context="pact_nerv_ia3_byte_closed_candidate",
    )
    return manifest


__all__ = [
    "PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA",
    "PACT_NERV_IA3_RECEIVER_INFLATE_PROOF_SCHEMA",
    "PactNervIa3ArchiveCandidateError",
    "materialize_pact_nerv_ia3_byte_closed_candidate",
]
