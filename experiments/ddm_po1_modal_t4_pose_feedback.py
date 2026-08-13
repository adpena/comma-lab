#!/usr/bin/env python3
"""Governed Modal dispatcher for retained PO1 T4 PoseNet feedback rounds.

MAIN supplies one exact archive/runtime pair plus explicit byte pins.  The
remote worker performs one decode, two SegNet passes, and three PoseNet passes
(GT, decoded, decoded repeat) inside the locked cu128 T4 image.  This wrapper
retains upload payloads before claiming the lane and supports detached recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import modal

try:
    from experiments.modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image
except ModuleNotFoundError:
    from modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image  # type: ignore[no-redef]

from tac.deploy.modal.auth_eval import (
    ClaimSpec,
    claim_modal_auth_eval_dispatch,
    function_call_id,
    terminal_modal_auth_eval_claim,
    write_spawn_metadata,
)
from tac.deploy.modal.call_id_ledger import (
    register_dispatched_call_id_fail_closed,
    update_call_id_outcome,
)
from tac.deploy.modal.single_flight import assert_modal_single_flight

REPO: Final = Path(__file__).resolve().parents[1]
REMOTE_REPO: Final = Path("/workspace/pact")
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_po1_t4_pose_feedback_worker.py"
REMOTE_JS1B_WORKER: Final = (
    REMOTE_REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
)
APP_NAME: Final = "comma-ddm-po1-t4-pose-feedback"
LANE_LABEL: Final = "ddm_po1_t4_pose_feedback"
AXIS: Final = (
    "[contest-CUDA T4 frozen-PoseNet first6 vectors plus SegNet fields, n600, batch=16] COMPONENT-ONLY"
)
VOLUME_NAME: Final = "comma-ddm-po1-pose-feedback-retained"
VOLUME_ROOT: Final = Path("/ddm_po1_retained")
COMMIT_PERIOD_SECONDS: Final = 20.0
CONTEST_LIMIT_SECONDS: Final = 1_800.0
MEASURED_DECODE_SECONDS: Final = 466.0
MEASURED_FULL_SCORER_SECONDS: Final = 39.405
ARCHIVE_COUNT: Final = 1
SCORER_PASSES: Final = 5
RESERVE_SECONDS: Final = 300.0
EXPECTED_RETAINED_PAYLOAD_BYTES: Final = 32_097_566_400

DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
DEFAULT_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_po1_20260813/dispatch/round1_cp135"
)
CP135_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_BYTES: Final = 186_252


class PO1DispatchError(RuntimeError):
    """A PO1 custody, dispatch, or recovery invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload_record(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def _safe_runtime_member(relative: Path) -> bool:
    return not (
        relative.is_absolute()
        or ".." in relative.parts
        or "__pycache__" in relative.parts
        or ".git" in relative.parts
        or relative.name.startswith("._")
        or relative.name.endswith((".pyc", ".pyo"))
        or relative.name in {".DS_Store", "archive.zip"}
    )


def _zip_bytes(files: list[tuple[str, bytes, int]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, payload, mode in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (mode & 0o777) << 16
            archive.writestr(info, payload)
    return stream.getvalue()


def build_runtime_bundle(runtime_root: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    root = runtime_root.resolve()
    if not root.is_dir():
        raise PO1DispatchError(f"missing {label} adapted runtime: {root}")
    files: list[tuple[str, bytes, int]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise PO1DispatchError(f"{label} runtime contains a symlink: {path}")
        relative = path.relative_to(root)
        if not _safe_runtime_member(relative):
            continue
        payload = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        files.append((relative.as_posix(), payload, mode))
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "mode": mode,
            }
        )
    required = {"inflate.sh", "inflate.py", "runtime/f26_inflate.py", "cpr1/inflate.py"}
    present = {row["relative_path"] for row in rows}
    if not required.issubset(present):
        raise PO1DispatchError(f"{label} runtime misses files: {sorted(required - present)}")
    manifest = {
        "schema": "ddm_po1_runtime_bundle.v1",
        "label": label,
        "source_root": str(root),
        "file_count": len(rows),
        "files": rows,
        "excluded_archive_and_generated_residue": True,
    }
    files.append(("PO1_RUNTIME_MANIFEST.json", canonical_json_bytes(manifest), 0o644))
    bundle = _zip_bytes(files)
    manifest["bundle"] = payload_record(bundle)
    return bundle, manifest


def k_arithmetic() -> dict[str, Any]:
    decode_seconds = ARCHIVE_COUNT * MEASURED_DECODE_SECONDS
    scorer_seconds = SCORER_PASSES * MEASURED_FULL_SCORER_SECONDS
    projected_without_reserve = decode_seconds + scorer_seconds
    projected_with_reserve = projected_without_reserve + RESERVE_SECONDS
    return {
        "schema": "ddm_po1_k_arithmetic.v1",
        "k_archives": ARCHIVE_COUNT,
        "measured_prior_seconds_per_decode": MEASURED_DECODE_SECONDS,
        "decode_seconds": decode_seconds,
        "scorer_passes": SCORER_PASSES,
        "scorer_pass_breakdown": {"segnet": 2, "posenet": 3},
        "measured_prior_seconds_per_full_scorer_pass": MEASURED_FULL_SCORER_SECONDS,
        "scorer_seconds": scorer_seconds,
        "reserve_seconds": RESERVE_SECONDS,
        "projected_seconds_without_reserve": projected_without_reserve,
        "projected_seconds_with_reserve": projected_with_reserve,
        "contest_limit_seconds": CONTEST_LIMIT_SECONDS,
        "headroom_seconds": CONTEST_LIMIT_SECONDS - projected_with_reserve,
        "fits_30_minutes": projected_with_reserve <= CONTEST_LIMIT_SECONDS,
        "expected_retained_payload_bytes_before_metadata": EXPECTED_RETAINED_PAYLOAD_BYTES,
        "epistemic_status": "PROJECTED_FROM_MEASURED_PRIOR_T4_COMPONENT_TIMES_NOT_YET_PO1_MEASURED",
    }


def require_exact(path: Path, *, size: int, digest: str) -> None:
    if not path.is_file():
        raise PO1DispatchError(f"missing candidate archive: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise PO1DispatchError("candidate archive differs from explicit bytes/SHA pin")


def prepare_request(
    *,
    archive: Path,
    runtime: Path,
    expected_archive_bytes: int,
    expected_archive_sha256: str,
    run_id: str,
    resume_from: str,
    round_ordinal: int,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise PO1DispatchError("--run-id must be one safe path component")
    if resume_from != run_id:
        raise PO1DispatchError("--resume-from is mandatory and must equal --run-id")
    if round_ordinal not in (1, 2, 3):
        raise PO1DispatchError("--round-ordinal must be 1, 2, or 3")
    if expected_archive_bytes <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha256):
        raise PO1DispatchError("archive bytes/SHA pin is malformed")
    require_exact(
        archive,
        size=expected_archive_bytes,
        digest=expected_archive_sha256,
    )
    runtime_bundle, runtime_manifest = build_runtime_bundle(runtime, label="candidate")
    payloads = {
        "candidate_archive.zip": archive.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
    }
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_po1_modal_request.v1",
        "axis": AXIS,
        "round_ordinal": round_ordinal,
        "run_id": run_id,
        "resume_from": resume_from,
        "seed": 1234,
        "batch_size": 16,
        "candidate_archive": {
            "bytes": expected_archive_bytes,
            "sha256": expected_archive_sha256,
        },
        "source_git_head": git_head,
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": sha256_bytes(git_status),
        "dispatcher_source_sha256": sha256_file(Path(__file__)),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_po1_t4_pose_feedback_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "inputs": {name: payload_record(payload) for name, payload in payloads.items()},
        "runtime_manifest": runtime_manifest,
        "k_arithmetic": k_arithmetic(),
        "retention_volume": VOLUME_NAME,
        "retention_volume_run_path": str(VOLUME_ROOT / run_id),
        "resume_required": True,
        "per_stage_checkpoints": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    return payloads, request


def persist_local_fire_inputs(
    output_dir: Path,
    payloads: dict[str, bytes],
    request: dict[str, Any],
) -> None:
    root = output_dir / "fire_inputs"
    for name, payload in payloads.items():
        path = root / name
        if path.is_file():
            if sha256_file(path) != sha256_bytes(payload):
                raise PO1DispatchError(f"local retained fire input differs: {path}")
        else:
            atomic_bytes(path, payload)
    atomic_json(output_dir / "PO1_LOCAL_REQUEST.json", request)


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
feedback_image = (
    eval_image.add_local_file(
        "experiments/ddm_po1_t4_pose_feedback_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py",
        remote_path=str(REMOTE_JS1B_WORKER),
        copy=False,
    )
    .add_local_python_source("ddm_po1_modal_t4_pose_feedback")
)


@app.function(
    image=feedback_image,
    gpu="T4",
    timeout=int(CONTEST_LIMIT_SECONDS),
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_feedback(payloads: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    remote_inputs = {**payloads, "REQUEST.json": canonical_json_bytes(request)}
    for name, payload in remote_inputs.items():
        path = run_root / "inputs" / name
        if path.is_file():
            if sha256_file(path) != sha256_bytes(payload):
                raise PO1DispatchError(f"resume input differs: {path}")
        else:
            atomic_bytes(path, payload)
    retained_volume.commit()

    command = [
        f"{UPSTREAM_LOCKED_VENV}/bin/python",
        "-u",
        str(REMOTE_WORKER),
        "--run-root",
        str(run_root),
        "--resume-from",
        str(request["resume_from"]),
    ]
    log_path = run_root / "worker.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(COMMIT_PERIOD_SECONDS)
            retained_volume.commit()
        returncode = process.wait()
    retained_volume.commit()
    final_path = run_root / "FINAL_RESULT.json"
    if returncode or not final_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
        return {
            "schema": "ddm_po1_modal_return.v1",
            "passed": False,
            "returncode": returncode,
            "axis": AXIS,
            "run_id": run_id,
            "volume_name": VOLUME_NAME,
            "volume_path": str(run_root),
            "worker_log_tail": tail,
            "resume_same_run_id": run_id,
            "score_claim": False,
            "promotion_eligible": False,
        }
    final_bytes = final_path.read_bytes()
    local_root = f"./modal_{run_id}"
    return {
        "schema": "ddm_po1_modal_return.v1",
        "passed": True,
        "returncode": 0,
        "axis": AXIS,
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "volume_download_commands": [
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/FINAL_RESULT.json {local_root}/FINAL_RESULT.json",
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/retained/pose_vectors/ {local_root}/retained/pose_vectors/",
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/retained/fields/ {local_root}/retained/fields/",
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/retained/raw/candidate/0.raw {local_root}/retained/raw/candidate/0.raw",
        ],
        "final_result_sha256": sha256_bytes(final_bytes),
        "artifacts": {"PO1_FINAL_RESULT.json": final_bytes},
        "score_claim": False,
        "promotion_eligible": False,
    }


def recover(output_dir: Path, *, timeout_seconds: float = 0.0) -> int:
    output_dir = output_dir.resolve()
    spawn = json.loads((output_dir / "modal_auth_eval_spawn.json").read_text())
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise PO1DispatchError(f"remote return is not a dict: {type(result).__name__}")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise PO1DispatchError(f"unsafe or non-bytes returned artifact: {name!r}")
        atomic_bytes(output_dir / name, payload)
    atomic_json(output_dir / "modal_po1_result.json", result)
    passed = bool(result.get("passed"))
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if passed else "failed",
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_pose_feedback_component",
        evidence_grade="contest-CUDA T4 frozen-PoseNet first6 repeat and SegNet fields n600",
        lane_id=str(spawn["lane_id"]),
        label=LANE_LABEL,
        gpu="T4",
        agent=str(spawn["claim_agent"]),
        harvest_result={key: value for key, value in result.items() if key != "worker_log_tail"},
    )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(spawn["lane_id"]),
            instance_job_id=str(spawn["instance_job_id"]),
            agent=str(spawn["claim_agent"]),
            force=True,
        ),
        status="completed_cuda_pose_feedback_recovered" if passed else "failed_cuda_pose_feedback_recovered",
        notes=f"PO1 recovered; call_id={call_id}; output={output_dir}",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


@app.local_entrypoint()
def main(
    archive: str = str(DEFAULT_ARCHIVE),
    runtime: str = str(DEFAULT_RUNTIME),
    expected_archive_bytes: int = CP135_BYTES,
    expected_archive_sha256: str = CP135_SHA256,
    output_dir: str = str(DEFAULT_OUTPUT),
    run_id: str = "ddm_po1_round1_cp135_20260813",
    resume_from: str = "ddm_po1_round1_cp135_20260813",
    round_ordinal: int = 1,
    lane_id: str = "ddm_po1_t4_pose_feedback_round1",
    instance_job_id: str = "modal:ddm_po1_round1_cp135_20260813",
    claim_agent: str = "main:ddm_po1",
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    if detach and not provider_detach_ack:
        raise SystemExit("FATAL: --detach requires --provider-detach-ack")
    if not k_arithmetic()["fits_30_minutes"]:
        raise SystemExit("FATAL: pinned K=1 projection exceeds the 30-minute budget")
    payloads, request = prepare_request(
        archive=Path(archive),
        runtime=Path(runtime),
        expected_archive_bytes=expected_archive_bytes,
        expected_archive_sha256=expected_archive_sha256,
        run_id=run_id,
        resume_from=resume_from,
        round_ordinal=round_ordinal,
    )
    output = Path(output_dir).resolve()
    persist_local_fire_inputs(output, payloads, request)
    spec = ClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        notes=(
            f"PO1 round {round_ordinal} retained T4 PoseNet feedback; run={run_id}; "
            f"archive={expected_archive_sha256}; volume={VOLUME_NAME}/{run_id}"
        ),
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=spec,
        status="active_cuda_pose_feedback_spawning",
    )
    assert_modal_single_flight(label=LANE_LABEL, lane_id=lane_id, repo_root=REPO)
    if detach:
        call = run_feedback.spawn(payloads, request)
        call_id = function_call_id(call)
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=lane_id,
            label=LANE_LABEL,
            platform="modal",
            gpu="T4",
            expected_axis="contest_cuda_pose_feedback_component",
            recipe="experiments/ddm_po1_modal_t4_pose_feedback.py::main",
            max_seconds=int(CONTEST_LIMIT_SECONDS),
            agent=claim_agent,
            base_archive_sha256=expected_archive_sha256,
            composed_archive_sha256=expected_archive_sha256,
            archive_count=ARCHIVE_COUNT,
            volume_name=VOLUME_NAME,
            volume_run_id=run_id,
        )
        write_spawn_metadata(
            out_dir=output,
            tool="experiments/ddm_po1_modal_t4_pose_feedback.py",
            app=APP_NAME,
            axis="contest_cuda_pose_feedback_component",
            call_id=call_id,
            local_request=request,
            result_json_name="modal_po1_result.json",
            recover_tool="experiments/ddm_po1_modal_t4_pose_feedback.py recover",
            extra={
                "lane_id": lane_id,
                "instance_job_id": instance_job_id,
                "claim_agent": claim_agent,
                "claim_platform": "modal",
                "volume_name": VOLUME_NAME,
                "volume_run_id": run_id,
            },
        )
        claim_modal_auth_eval_dispatch(
            repo_root=REPO,
            spec=ClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                force=True,
                notes=f"PO1 detached spawn accepted; call_id={call_id}; output={output}",
            ),
            status="active_cuda_pose_feedback_spawned",
        )
        print(f"DISPATCHED call_id={call_id}")
        print(f"RECOVER .venv/bin/python {__file__} recover --output-dir {output}")
        return

    result = run_feedback.remote(payloads, request)
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        atomic_bytes(output / name, payload)
    atomic_json(output / "modal_po1_result.json", result)
    passed = bool(result.get("passed"))
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=lane_id,
            instance_job_id=instance_job_id,
            agent=claim_agent,
            force=True,
        ),
        status="completed_cuda_pose_feedback_synchronous" if passed else "failed_cuda_pose_feedback_synchronous",
        notes=f"PO1 synchronous component job ended; output={output}; resume run={run_id}",
    )
    if not passed:
        raise SystemExit("FATAL: PO1 remote feedback failed; resume the same run id")


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover a detached PO1 Modal feedback round.")
    sub = parser.add_subparsers(dest="command", required=True)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "recover":
        return recover(args.output_dir, timeout_seconds=args.timeout_seconds)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_main_cli())
