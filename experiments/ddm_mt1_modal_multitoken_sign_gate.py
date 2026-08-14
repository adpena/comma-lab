#!/usr/bin/env python3
"""Governed sealed Modal dispatcher for the DDM MT1 #978 T4 sign gate.

``prepare`` is local and scorer-free.  It seals the exact parsed model,
stratified heldout IDs, receiver inputs, and retained CP135 frame-0 carrier.
MAIN alone may invoke the detached T4 entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import modal
import numpy as np

try:
    import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch
except ModuleNotFoundError:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch

try:
    from modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image
except ModuleNotFoundError:
    from experiments.modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image

from tac.deploy.modal.auth_eval import (
    ClaimSpec,
    claim_modal_auth_eval_dispatch,
    function_call_id,
    write_spawn_metadata,
)
from tac.deploy.modal.call_id_ledger import register_dispatched_call_id_fail_closed
from tac.deploy.modal.single_flight import assert_modal_single_flight
from tac.deploy.worker_dependency_closure import require_worker_dependency_closure

REPO: Final = Path(__file__).resolve().parents[1]
REMOTE_REPO: Final = Path("/workspace/pact")
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_mt1_t4_sign_gate_worker.py"
APP_NAME: Final = "comma-ddm-mt1-multitoken-sign-gate"
LANE_ID: Final = "lane_ddm_mt1_978_multitoken_screen_20260814"
INSTANCE_JOB_ID: Final = "modal:ddm_mt1_t4_sign_gate_20260814"
RUN_ID: Final = "ddm_mt1_t4_sign_gate_20260814"
LANE_LABEL: Final = "ddm_mt1_978_t4_sign_gate"
VOLUME_NAME: Final = js1b_dispatch.VOLUME_NAME
VOLUME_ROOT: Final = js1b_dispatch.VOLUME_ROOT
COMMIT_PERIOD_SECONDS: Final = 20.0
HARD_CAP_SECONDS: Final = 960
T4_USD_PER_HOUR_ASSUMPTION: Final = 0.60
ESTIMATED_HARD_CAP_USD: Final = HARD_CAP_SECONDS / 3600 * T4_USD_PER_HOUR_ASSUMPTION
TARGET_VENV_EXTRA_DEPENDENCIES: Final = ("pydantic==2.13.4", "Brotli==1.2.0")
TARGET_VENV_PAYLOAD_IMPORT_ROOTS: Final = (
    "ddm_ec1_implicit_edge_conditioning",
    "ddm_ec1_runtime",
    "modules",
    "runtime",
)

LOCAL_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained"
)
DEFAULT_OUTPUT: Final = LOCAL_ROOT / "t4_sign_gate_r1"
DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
DEFAULT_RUNTIME: Final = DEFAULT_ARCHIVE.parent
DEFAULT_BASE_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
DEFAULT_C1_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/"
    "c1_solved_tokens_n600.u8"
)
DEFAULT_GT_FIELD: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "contest_cuda/ddm_js1b_20260813b/retained/fields/gt_argmax_n600.npy"
)
DEFAULT_GT_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_n600.npy"
)
DEFAULT_MODEL: Final = LOCAL_ROOT / "stages/20_collateral_finish/ema.mt1.br"
DEFAULT_SELECTION: Final = LOCAL_ROOT / "inputs/SELECTION.json"


class MT1DispatchError(RuntimeError):
    """A seal, source, lane, dispatch, or recovery invariant failed."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value))


def persist_exact_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    expected = {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if path.is_file():
        if file_record(path) != expected:
            raise MT1DispatchError(f"sealed payload differs: {path}")
        return expected
    atomic_bytes(path, payload)
    return file_record(path)


def persist_exact_file(source: Path, target: Path) -> dict[str, Any]:
    source_record = file_record(source)
    expected = {**source_record, "path": str(target.resolve())}
    if target.is_file():
        if file_record(target) != expected:
            raise MT1DispatchError(f"sealed file differs: {target}")
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        shutil.copyfile(source, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        if partial.stat().st_size != source_record["bytes"] or sha256_file(partial) != source_record["sha256"]:
            raise MT1DispatchError(f"copied fire input differs: {source}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return file_record(target)


def require_record(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != str(record["sha256"])
    ):
        raise MT1DispatchError(f"sealed input differs: {path}")


def frame0_slave_payload(selection: dict[str, Any]) -> np.ndarray:
    pairs = [int(value) for value in selection["heldout"]]
    chunks: list[np.ndarray] = []
    observed: list[int] = []
    for batch_index in range(8):
        root = LOCAL_ROOT / f"endpoint/cp135_hard/batch_{batch_index:02d}"
        batch_pairs = np.load(root / "pairs.int16.npy", allow_pickle=False)
        slaves = np.load(root / "pose_slave.uint8.npy", allow_pickle=False)
        if slaves.shape != (len(batch_pairs), 874, 1164, 3):
            raise MT1DispatchError(f"retained frame-0 geometry differs: {root}")
        observed.extend(int(value) for value in batch_pairs)
        chunks.append(np.asarray(slaves, dtype=np.uint8))
    if observed != pairs:
        raise MT1DispatchError("retained CP135 frame-0 order differs from sealed heldout IDs")
    return np.concatenate(chunks, axis=0)


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    required = 1024**3
    receipt = {
        "schema": "ddm_mt1_t4_seal_storage_preflight.v1",
        "tier": str(output.resolve()),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; every fire input is retained with bytes and SHA-256",
    }
    atomic_json(output / "STORAGE_PREFLIGHT.json", receipt)
    if not receipt["passed"]:
        raise MT1DispatchError(f"sign-gate seal storage preflight failed: {receipt}")
    return receipt


def prepare(
    *,
    output: Path,
    archive: Path,
    runtime: Path,
    base_tokens: Path,
    c1_tokens: Path,
    gt_field: Path,
    gt_pose: Path,
    model: Path,
    selection_path: Path,
) -> dict[str, Any]:
    output = output.resolve()
    storage = storage_preflight(output)
    for source in (
        archive,
        base_tokens,
        c1_tokens,
        gt_field,
        gt_pose,
        model,
        selection_path,
        LOCAL_ROOT / "COMPARISON.json",
    ):
        if not source.is_file():
            raise MT1DispatchError(f"missing sign-gate source: {source}")
    comparison = json.loads((LOCAL_ROOT / "COMPARISON.json").read_text())
    selection = json.loads(selection_path.read_text())
    if comparison["positive_local_sign"]:
        local_disposition = "positive_local_sign_requires_t4_reproduction"
    else:
        local_disposition = "negative_local_sign_requires_t4_boundary_confirmation"
    runtime_bundle, runtime_manifest = js1b_dispatch.build_runtime_bundle(
        runtime, label="cp135"
    )
    fire_inputs = output / "fire_inputs"
    records = {
        "cp135.archive.zip": persist_exact_file(archive, fire_inputs / "cp135.archive.zip"),
        "cp135.runtime.zip": persist_exact_bytes(
            fire_inputs / "cp135.runtime.zip", runtime_bundle
        ),
        "base_tokens.npy": persist_exact_file(base_tokens, fire_inputs / "base_tokens.npy"),
        "c1_tokens.u8": persist_exact_file(c1_tokens, fire_inputs / "c1_tokens.u8"),
        "gt_argmax.npy": persist_exact_file(gt_field, fire_inputs / "gt_argmax.npy"),
        "gt_pose.npy": persist_exact_file(gt_pose, fire_inputs / "gt_pose.npy"),
        "selected_model.mt1.br": persist_exact_file(
            model, fire_inputs / "selected_model.mt1.br"
        ),
        "selection.json": persist_exact_file(
            selection_path, fire_inputs / "selection.json"
        ),
    }
    slaves = frame0_slave_payload(selection)
    slave_path = fire_inputs / "frame0_slave_n32.npy"
    partial = slave_path.with_name(f".{slave_path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as stream:
            np.save(stream, slaves, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if slave_path.is_file():
            if (
                slave_path.stat().st_size != partial.stat().st_size
                or sha256_file(slave_path) != sha256_file(partial)
            ):
                raise MT1DispatchError("retained sign-gate frame-0 payload differs")
        else:
            os.replace(partial, slave_path)
    finally:
        partial.unlink(missing_ok=True)
    records["frame0_slave_n32.npy"] = file_record(slave_path)
    dependency_closure = require_worker_dependency_closure(
        repo_root=REPO,
        worker_entrypoints=(
            REPO / "experiments/ddm_mt1_t4_sign_gate_worker.py",
            REPO / "experiments/ddm_mt1_978_multitoken_screen.py",
            REPO / "experiments/ddm_mt1_runtime/multitoken_representative.py",
            REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py",
            REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py",
            REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py",
            REPO / "experiments/ddm_qs1_frame0_schur_coupled_solve.py",
        ),
        target_lock_path=REPO / "upstream/uv.lock",
        extra_target_dependencies=TARGET_VENV_EXTRA_DEPENDENCIES,
        payload_provided_import_roots=TARGET_VENV_PAYLOAD_IMPORT_ROOTS,
    )
    closure_path = output / "preflight/WORKER_DEPENDENCY_CLOSURE.json"
    atomic_json(closure_path, dependency_closure)
    sources = {
        "dispatcher": file_record(Path(__file__)),
        "worker": file_record(REPO / "experiments/ddm_mt1_t4_sign_gate_worker.py"),
        "screen": file_record(REPO / "experiments/ddm_mt1_978_multitoken_screen.py"),
        "runtime": file_record(
            REPO / "experiments/ddm_mt1_runtime/multitoken_representative.py"
        ),
        "ec2_worker_template": file_record(
            REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py"
        ),
        "ec1_design_import_dependency": file_record(
            REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py"
        ),
        "ec1_runtime_import_dependency": file_record(
            REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py"
        ),
        "qs5_pose_surface_template": file_record(
            REPO / "experiments/ddm_qs1_frame0_schur_coupled_solve.py"
        ),
        "dependency_closure": file_record(closure_path),
    }
    request = {
        "schema": "ddm_mt1_t4_sign_gate_request.v1",
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "instance_job_id": INSTANCE_JOB_ID,
        "claim_agent": "main:ddm_mt1_t4_sign_gate",
        "resume_from": str(VOLUME_ROOT / RUN_ID),
        "payloads": records,
        "sources": sources,
        "runtime_manifest": runtime_manifest,
        "selection": {
            "seed": selection["seed"],
            "mode": selection["selection_mode"],
            "n_train": len(selection["train"]),
            "n_heldout": len(selection["heldout"]),
        },
        "local_comparison": file_record(LOCAL_ROOT / "COMPARISON.json"),
        "local_disposition": local_disposition,
        "axis": (
            "[contest-CUDA T4 frozen SegNet/PoseNet; stratified-random n32 heldout] "
            "COMPONENT-ONLY NON-PROMOTABLE"
        ),
        "score_claim": False,
    }
    request_path = output / "SEALED_REQUEST.json"
    persist_exact_bytes(request_path, canonical_json(request))
    request_record = file_record(request_path)
    command = (
        ".venv/bin/modal run --detach "
        "experiments/ddm_mt1_modal_multitoken_sign_gate.py::main "
        f"--sealed-request {request_path} "
        f"--fire-input-dir {fire_inputs} "
        f"--expected-request-sha256 {request_record['sha256']} "
        f"--output-dir {output / 'dispatch'} --detach --provider-detach-ack"
    )
    fire_order = {
        "schema": "ddm_mt1_t4_sign_gate_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole Modal scorer-lane router",
        "consumer_store": str(output),
        "fire_trigger": (
            "MAIN confirms the #978 T4 scorer lane is free, terminals the local build claim, "
            "and all sealed request/source hashes still match"
        ),
        "exact_argv": command,
        "sealed_request": request_record,
        "fire_inputs": records,
        "remote_retention": {
            "volume": VOLUME_NAME,
            "run_id": RUN_ID,
            "complete_remote_custody": True,
        },
        "hard_cap_seconds": HARD_CAP_SECONDS,
        "t4_usd_per_hour_assumption": T4_USD_PER_HOUR_ASSUMPTION,
        "hard_cap_cost_usd": ESTIMATED_HARD_CAP_USD,
        "schedule_derivation": {
            "endpoint_batches": 24,
            "arms": 3,
            "pairs_per_arm": 32,
            "batch_size": 4,
            "prior_local_cpu_observation": "completed inside one bounded interactive run",
            "hard_cap_rationale": "16 minutes bounds the 24-batch T4 sign-only endpoint",
        },
        "harvest": {
            "commands": [
                f"mkdir -p {output / 'harvest'}",
                (
                    f".venv/bin/modal volume get --force {VOLUME_NAME} "
                    f"{RUN_ID}/FINAL_RESULT.json {output / 'harvest/FINAL_RESULT.json'}"
                ),
            ],
            "complete_remote_custody_command": (
                f"mkdir -p {output / 'complete_remote_custody'} && "
                f".venv/bin/modal volume get --force {VOLUME_NAME} "
                f"{RUN_ID}/ {output / 'complete_remote_custody/'}"
            ),
        },
        "on_positive_t4_sign": (
            "consume SECOND_TRAIN_FIRE_ORDER only if it exists and all T4 gates are true; "
            "the local negative created no second train order"
        ),
        "does_not_dispatch": True,
        "score_claim": False,
    }
    fire_path = output / "SEALED_FIRE_ORDER.json"
    persist_exact_bytes(fire_path, canonical_json(fire_order))
    seal = {
        "schema": "ddm_mt1_t4_sign_gate_seal.v1",
        "sealed_request": request_record,
        "sealed_fire_order": file_record(fire_path),
        "fire_inputs": records,
        "sources": sources,
        "worker_dependency_closure": file_record(closure_path),
        "storage_preflight": storage,
        "modal_dispatched": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    persist_exact_bytes(output / "SEAL_RESULT.json", canonical_json(seal))
    return seal


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
worker_image = (
    eval_image.run_commands(
        shlex.join(
            [
                "/usr/local/bin/uv",
                "pip",
                "install",
                "--python",
                f"{UPSTREAM_LOCKED_VENV}/bin/python",
                *TARGET_VENV_EXTRA_DEPENDENCIES,
            ]
        )
    )
    .add_local_file(
        "experiments/ddm_mt1_t4_sign_gate_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_mt1_978_multitoken_screen.py",
        remote_path=str(REMOTE_REPO / "experiments/ddm_mt1_978_multitoken_screen.py"),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_ec2_oriented_adapter_trainer_worker.py",
        remote_path=str(
            REMOTE_REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py"
        ),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_ec1_implicit_edge_conditioning.py",
        remote_path=str(
            REMOTE_REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py"
        ),
        copy=False,
    )
    .add_local_dir(
        "experiments/ddm_ec1_runtime",
        remote_path=str(REMOTE_REPO / "experiments/ddm_ec1_runtime"),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_qs1_frame0_schur_coupled_solve.py",
        remote_path=str(
            REMOTE_REPO / "experiments/ddm_qs1_frame0_schur_coupled_solve.py"
        ),
        copy=False,
    )
    .add_local_file(
        "experiments/__init__.py",
        remote_path=str(REMOTE_REPO / "experiments/__init__.py"),
        copy=False,
    )
    .add_local_dir(
        "experiments/ddm_mt1_runtime",
        remote_path=str(REMOTE_REPO / "experiments/ddm_mt1_runtime"),
        copy=False,
    )
    .add_local_python_source(
        "ddm_mt1_modal_multitoken_sign_gate",
        "ddm_js1b_modal_cuda_argmax_field_materializer",
    )
)


@app.function(
    image=worker_image,
    gpu="T4",
    timeout=HARD_CAP_SECONDS,
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_sign_gate(payloads: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    run_root = VOLUME_ROOT / str(request["run_id"])
    run_root.mkdir(parents=True, exist_ok=True)
    for name, payload in {**payloads, "REQUEST.json": canonical_json(request)}.items():
        target = run_root / "inputs" / name
        if target.is_file():
            if target.read_bytes() != payload:
                raise MT1DispatchError(f"remote resume input differs: {target}")
        else:
            atomic_bytes(target, payload)
    retained_volume.commit()
    command = [
        f"{UPSTREAM_LOCKED_VENV}/bin/python",
        "-u",
        str(REMOTE_WORKER),
        "--run-root",
        str(run_root),
        "--resume-from",
        str(run_root),
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
    return {
        "schema": "ddm_mt1_t4_modal_return.v1",
        "returncode": returncode,
        "complete": returncode == 0 and final_path.is_file(),
        "run_id": request["run_id"],
        "volume": VOLUME_NAME,
        "final_result": file_record(final_path) if final_path.is_file() else None,
        "worker_log_tail": log_path.read_text(errors="replace")[-8_000:],
        "resume_same_run_id": request["run_id"],
        "score_claim": False,
    }


def load_sealed(
    path: Path,
    expected_sha256: str,
    fire_input_dir: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise MT1DispatchError("sealed request SHA-256 differs")
    request = json.loads(path.read_text())
    payloads: dict[str, bytes] = {}
    for name, record in request["payloads"].items():
        source = fire_input_dir / name
        require_record(source, record)
        payloads[name] = source.read_bytes()
    return payloads, request


@app.local_entrypoint()
def main(
    sealed_request: str,
    fire_input_dir: str,
    expected_request_sha256: str,
    output_dir: str,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    if not detach or not provider_detach_ack:
        raise MT1DispatchError("T4 sign gate requires detached provider acknowledgement")
    output = Path(output_dir).resolve()
    payloads, request = load_sealed(
        Path(sealed_request).resolve(),
        expected_request_sha256,
        Path(fire_input_dir).resolve(),
    )
    spec = ClaimSpec(
        lane_id=str(request["lane_id"]),
        instance_job_id=str(request["instance_job_id"]),
        agent=str(request["claim_agent"]),
        notes=f"MT1 #978 T4 sign gate; run={request['run_id']}",
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=spec,
        status="active_ddm_mt1_t4_sign_gate_spawning",
    )
    assert_modal_single_flight(label=LANE_LABEL, lane_id=LANE_ID, repo_root=REPO)
    call = run_sign_gate.spawn(payloads, request)
    call_id = function_call_id(call)
    register_dispatched_call_id_fail_closed(
        call_id=call_id,
        lane_id=LANE_ID,
        label=LANE_LABEL,
        platform="modal",
        gpu="T4",
        expected_axis="contest_cuda_mt1_n32_component_sign_gate",
        recipe="experiments/ddm_mt1_modal_multitoken_sign_gate.py",
        max_seconds=HARD_CAP_SECONDS,
        agent=str(request["claim_agent"]),
        base_archive_sha256=str(request["payloads"]["cp135.archive.zip"]["sha256"]),
        composed_archive_sha256=str(request["payloads"]["cp135.archive.zip"]["sha256"]),
        archive_count=1,
        volume_name=VOLUME_NAME,
        volume_run_id=str(request["run_id"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    write_spawn_metadata(
        out_dir=output,
        lane_id=LANE_ID,
        instance_job_id=INSTANCE_JOB_ID,
        call_id=call_id,
        status="spawned_detached",
        extra={"run_id": RUN_ID, "volume_name": VOLUME_NAME},
    )
    print(json.dumps({"call_id": call_id, "run_id": RUN_ID}, sort_keys=True))


def cli() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    prepare_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare_parser.add_argument("--base-tokens", type=Path, default=DEFAULT_BASE_TOKENS)
    prepare_parser.add_argument("--c1-tokens", type=Path, default=DEFAULT_C1_TOKENS)
    prepare_parser.add_argument("--gt-field", type=Path, default=DEFAULT_GT_FIELD)
    prepare_parser.add_argument("--gt-pose", type=Path, default=DEFAULT_GT_POSE)
    prepare_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prepare_parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    args = parser.parse_args()
    result = prepare(
        output=args.output,
        archive=args.archive,
        runtime=args.runtime,
        base_tokens=args.base_tokens,
        c1_tokens=args.c1_tokens,
        gt_field=args.gt_field,
        gt_pose=args.gt_pose,
        model=args.model,
        selection_path=args.selection,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
