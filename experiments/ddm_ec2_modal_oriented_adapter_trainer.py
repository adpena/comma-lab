#!/usr/bin/env python3
"""Governed Modal dispatcher for the EC2 CP135 latent-adapter trainer.

``prepare`` is local and scorer-free: it validates exact custody, runs a tiny
structural gate, persists immutable fire inputs, and seals an oriented-only
first fire order plus condition-gated control requests.  MAIN alone may invoke
the Modal entry points.  This arm never fires them itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import modal

# BARE-first is required when a Modal image reuses another module's remote
# function/import surface.  The package fallback is for repository-local tests.
try:
    import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch
except ModuleNotFoundError:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch

try:
    from ddm_ec2_oriented_adapter_trainer_worker import (
        ERROR_BALANCE_WEIGHT,
        STAGES,
        run_toy_gate,
    )
except ModuleNotFoundError:
    from experiments.ddm_ec2_oriented_adapter_trainer_worker import (
        ERROR_BALANCE_WEIGHT,
        STAGES,
        run_toy_gate,
    )

try:
    from modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image
except ModuleNotFoundError:
    from experiments.modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image

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
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py"
APP_NAME: Final = "comma-ddm-ec2-oriented-adapter-trainer"
LANE_LABEL: Final = "ddm_ec2_oriented_adapter_trainer"
AXIS: Final = "[contest-CUDA T4 frozen-SegNet, n600] COMPONENT-ONLY"
VOLUME_NAME: Final = js1b_dispatch.VOLUME_NAME
VOLUME_ROOT: Final = js1b_dispatch.VOLUME_ROOT
PRIOR_RUN: Final = VOLUME_ROOT / "ddm_js1b_20260813b"
PRIOR_GT_FIELD: Final = PRIOR_RUN / "retained/fields/gt_argmax_n600.npy"
PRIOR_BASE_FIELD: Final = PRIOR_RUN / "retained/fields/cp135_base_argmax_n600.npy"
COMMIT_PERIOD_SECONDS: Final = 20.0
HARD_CAP_SECONDS: Final = 10_800
ESTIMATED_T4_COST_USD: Final = 1.80

DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/submittable_custody_mirror_20260811/cp135_packet/adapted_runtime/archive.zip"
)
DEFAULT_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/submittable_custody_mirror_20260811/cp135_packet/adapted_runtime"
)
DEFAULT_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/cp135_base/retained/decoded_tokens_n600.npy"
)
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda"
)
EC1_FINAL: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/FINAL_RESULT.json"
)
EC1_FIRE_ORDER: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "edge_conditioned/ddm_ec1_20260814/MAIN_CUDA_FIRE_ORDER.json"
)
EC1_COMMIT: Final = "fa29eb9ea17d3bfd5138478470600f322050634d"
EC1_FINAL_SHA256: Final = "bb0a6582745492dc77e4dc8a6556248bea5cc4084b06de028a4b1aa2aec76bd3"
EC1_FIRE_ORDER_SHA256: Final = "0d403be3b5af461c9e6e8c9caf77066b126f22be853c51d85509d0bcc8a6185c"
CP135_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_BYTES: Final = 186_252
TOKENS_SHA256: Final = "03f5379d70e4bbd88e125cfbfb785cf5473315c70a5b78661fa426bb3e96e0f4"
TOKENS_BYTES: Final = 117_964_928
GT_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
}
BASE_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727",
}
BASE_FLIPS: Final = 34_970
BASE_D_POSE: Final = 6.885642960696714e-6


class EC2DispatchError(RuntimeError):
    """A seal, custody, single-flight, remote, or recovery invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


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


def require_exact(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file():
        raise EC2DispatchError(f"missing {label}: {path}")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise EC2DispatchError(f"{label} differs from pinned custody: {path}")


def source_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    record["git_status_scope"] = "sealed by exact bytes; commit may be unavailable in managed sandbox"
    return record


def storage_preflight(output: Path, required_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    result = {
        "schema": "ddm_ec2_local_storage_preflight.v1",
        "tier": str(output),
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
        "passed": usage.free >= required_bytes,
        "policy": "SSD-first; block rather than delete or move uncertified payloads",
    }
    atomic_json(output / "LOCAL_STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise EC2DispatchError(f"local fire-input preflight failed: free={usage.free}, required={required_bytes}")
    return result


def _persist_exact(path: Path, payload: bytes) -> dict[str, Any]:
    if path.is_file():
        if path.read_bytes() != payload:
            raise EC2DispatchError(f"sealed payload differs: {path}")
    else:
        atomic_bytes(path, payload)
    return file_record(path)


def schedule_derivation() -> dict[str, Any]:
    field_pass_seconds = 15 * 60
    seconds_per_pair = field_pass_seconds / 600
    training_multiplier = 3.0
    training_seconds = 1_800 * seconds_per_pair * training_multiplier
    projected = training_seconds + 900 + 900
    return {
        "schema": "ddm_ec2_schedule_derivation.v1",
        "measured_n600_field_pass_upper_anchor_seconds": field_pass_seconds,
        "field_pass_pairs": 600,
        "derived_seconds_per_pair": seconds_per_pair,
        "optimizer_steps": 1_800,
        "conservative_training_forward_backward_multiplier_assumption": training_multiplier,
        "derived_training_seconds": training_seconds,
        "full_n600_endpoint_reserve_seconds": 900,
        "checkpoint_package_reserve_seconds": 900,
        "projected_seconds": projected,
        "hard_cap_seconds": HARD_CAP_SECONDS,
        "fits_hard_cap": projected <= HARD_CAP_SECONDS,
        "estimated_t4_cost_usd": ESTIMATED_T4_COST_USD,
        "cost_rate_assumption_usd_per_t4_hour": 0.60,
    }


def make_request(
    *,
    family: str,
    run_id: str,
    runtime_manifest: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if family not in {"oriented", "class_only", "undirected"}:
        raise EC2DispatchError(f"unsupported sealed family: {family!r}")
    return {
        "schema": "ddm_ec2_modal_training_request.v1",
        "run_id": run_id,
        "resume_from": run_id,
        "family": family,
        "axis": AXIS,
        "lane_id": f"ddm_ec2_{family}_adapter_trainer",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": f"main:ddm_ec2:{family}",
        "base": {
            "archive": {"bytes": CP135_BYTES, "sha256": CP135_SHA256},
            "flips": BASE_FLIPS,
            "d_pose": BASE_D_POSE,
        },
        "payloads": payloads,
        "runtime_manifest": runtime_manifest,
        "prior_retained_fields": {
            "volume": VOLUME_NAME,
            "run_path": str(PRIOR_RUN),
            "gt": {"path": str(PRIOR_GT_FIELD), **GT_FIELD_RECORD},
            "base": {"path": str(PRIOR_BASE_FIELD), **BASE_FIELD_RECORD},
        },
        "training": {
            "same_cp135_semantic_object": True,
            "injection_site": "token_embed+coord_mix, then counted conditioner, then four CP135 TokenBlocks",
            "camera_roundtrip": "BILINEAR lift 384x512->874x1164, uint8 STE, BILINEAR down",
            "frozen_scorer": "upstream SegNet safetensors",
            "population": "deterministic stratified n600 permutation per stage; never prefix",
            "stages": list(STAGES),
            "loss_weight_derivation": {
                "pixels": 600 * 384 * 512,
                "base_flips": BASE_FLIPS,
                "inverse_error_prevalence": ERROR_BALANCE_WEIGHT,
                "stage_error_to_correct_total_mass_ratios": [4.0, 1.0, 0.25],
            },
            "ema": "canonical warmup EMA with decay derived from total steps",
            "retention": "all step pre-R/camera/scorer/logit/argmax/target payloads plus live+EMA checkpoints",
        },
        "schedule_derivation": schedule_derivation(),
        "ec1_pins": {
            "commit": EC1_COMMIT,
            "final_result_sha256": EC1_FINAL_SHA256,
            "fire_order_sha256": EC1_FIRE_ORDER_SHA256,
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def prepare(
    *,
    archive: Path,
    runtime: Path,
    tokens: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    require_exact(archive, size=CP135_BYTES, digest=CP135_SHA256, label="CP135 archive")
    require_exact(tokens, size=TOKENS_BYTES, digest=TOKENS_SHA256, label="decoded CP135 tokens")
    require_exact(EC1_FINAL, size=EC1_FINAL.stat().st_size, digest=EC1_FINAL_SHA256, label="EC1 FINAL_RESULT")
    require_exact(
        EC1_FIRE_ORDER,
        size=EC1_FIRE_ORDER.stat().st_size,
        digest=EC1_FIRE_ORDER_SHA256,
        label="EC1 fire order",
    )
    storage = storage_preflight(output, TOKENS_BYTES + 2 * 1024**3)
    toy = run_toy_gate(output / "toy_gate")
    runtime_bundle, runtime_manifest = js1b_dispatch.build_runtime_bundle(runtime, label="cp135")
    fire_inputs = output / "fire_inputs"
    inputs = {
        "cp135.archive.zip": archive.read_bytes(),
        "cp135.runtime.zip": runtime_bundle,
        "decoded_tokens_n600.npy": tokens.read_bytes(),
    }
    input_records = {name: _persist_exact(fire_inputs / name, payload) for name, payload in inputs.items()}
    source_records = {
        "dispatcher": source_record(Path(__file__)),
        "worker": source_record(REPO / "experiments/ddm_ec2_oriented_adapter_trainer_worker.py"),
        "ec1_design": source_record(REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py"),
        "ec1_runtime": source_record(REPO / "experiments/ddm_ec1_runtime/ec1_latent_conditioner.py"),
    }
    payload_records = {**input_records, "sources": source_records}
    oriented = make_request(
        family="oriented",
        run_id="ddm_ec2_oriented_20260814",
        runtime_manifest=runtime_manifest,
        payloads=payload_records,
    )
    oriented_path = output / "SEALED_ORIENTED_REQUEST.json"
    _persist_exact(oriented_path, canonical_json_bytes(oriented))
    oriented_sha = sha256_file(oriented_path)
    controls: dict[str, Any] = {}
    for family in ("class_only", "undirected"):
        request = make_request(
            family=family,
            run_id=f"ddm_ec2_{family}_control_20260814",
            runtime_manifest=runtime_manifest,
            payloads=payload_records,
        )
        path = output / f"SEALED_{family.upper()}_REQUEST.json"
        _persist_exact(path, canonical_json_bytes(request))
        controls[family] = file_record(path)

    fire_command = (
        ".venv/bin/modal run --detach "
        "experiments/ddm_ec2_modal_oriented_adapter_trainer.py::main "
        f"--sealed-request {oriented_path} "
        f"--fire-input-dir {fire_inputs} "
        f"--expected-request-sha256 {oriented_sha} "
        f"--output-dir {output / 'dispatch_oriented'} "
        "--detach --provider-detach-ack"
    )
    control_command = (
        ".venv/bin/modal run --detach "
        "experiments/ddm_ec2_modal_oriented_adapter_trainer.py::controls "
        f"--sealed-class-request {output / 'SEALED_CLASS_ONLY_REQUEST.json'} "
        f"--sealed-undirected-request {output / 'SEALED_UNDIRECTED_REQUEST.json'} "
        f"--fire-input-dir {fire_inputs} "
        f"--oriented-endpoint-result {output / 'oriented_harvest/selected/SELECTED_RESULT.json'} "
        f"--expected-class-request-sha256 {controls['class_only']['sha256']} "
        f"--expected-undirected-request-sha256 {controls['undirected']['sha256']} "
        f"--output-dir {output / 'dispatch_controls'} "
        "--detach --provider-detach-ack"
    )
    fire_order = {
        "schema": "ddm_ec2_sealed_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole Modal scorer-lane router",
        "consumer_store": str(output),
        "first_dispatch": {
            "family": "oriented",
            "sealed_request": file_record(oriented_path),
            "exact_argv": fire_command,
            "fire_trigger": (
                "MAIN confirms no active duplicate EC2 lane, Modal is available, and the "
                "sealed request/source hashes still match"
            ),
        },
        "second_dispatch": {
            "families": ["class_only", "undirected"],
            "sealed_requests": controls,
            "exact_argv": control_command,
            "fire_trigger": (
                "ONLY after the harvested oriented SELECTED_RESULT has "
                "clears_oriented_break_even=true; the controls entry point verifies it"
            ),
            "blocked_now": True,
        },
        "remote_retention": {
            "volume": VOLUME_NAME,
            "oriented_run": "ddm_ec2_oriented_20260814",
            "controls": [
                "ddm_ec2_class_only_control_20260814",
                "ddm_ec2_undirected_control_20260814",
            ],
        },
        "harvest": {
            "precondition": "create each destination directory before modal volume get",
            "oriented_commands": [
                f"mkdir -p {output / 'oriented_harvest/selected'}",
                (
                    f".venv/bin/modal volume get --force {VOLUME_NAME} "
                    "ddm_ec2_oriented_20260814/FINAL_RESULT.json "
                    f"{output / 'oriented_harvest/FINAL_RESULT.json'}"
                ),
                (
                    f".venv/bin/modal volume get --force {VOLUME_NAME} "
                    "ddm_ec2_oriented_20260814/stages/selected/ "
                    f"{output / 'oriented_harvest/selected/'}"
                ),
            ],
            "complete_remote_custody_command": (
                f"mkdir -p {output / 'complete_remote_custody'} && "
                f".venv/bin/modal volume get --force {VOLUME_NAME} "
                f"ddm_ec2_oriented_20260814/ {output / 'complete_remote_custody/'}"
            ),
        },
        "package_then_measure": {
            "package_command": (
                ".venv/bin/python experiments/ddm_ec1_implicit_edge_conditioning.py package "
                f"--module {output / 'oriented_harvest/selected/retained/ec1_latent.int8.br'} "
                f"--output {output / 'packaged_oriented'} --classification candidate_proposal"
            ),
            "then": "reuse the sealed RE1T/JS1B T4 Seg/Pose component chain on the exact packaged archive",
        },
        "source_records": source_records,
        "fire_inputs": input_records,
        "toy_gate": toy,
        "storage_preflight": storage,
        "does_not_dispatch": True,
    }
    fire_path = output / "SEALED_FIRE_ORDER.json"
    _persist_exact(fire_path, canonical_json_bytes(fire_order))
    seal = {
        "schema": "ddm_ec2_seal_result.v1",
        "sealed_fire_order": file_record(fire_path),
        "oriented_request": file_record(oriented_path),
        "control_requests": controls,
        "source_records": source_records,
        "fire_inputs": input_records,
        "toy_gate": toy,
        "storage_preflight": storage,
        "modal_dispatched": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    _persist_exact(output / "SEAL_RESULT.json", canonical_json_bytes(seal))
    return seal


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
trainer_image = (
    eval_image.add_local_file(
        "experiments/ddm_ec2_oriented_adapter_trainer_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_ec1_implicit_edge_conditioning.py",
        remote_path=str(REMOTE_REPO / "experiments/ddm_ec1_implicit_edge_conditioning.py"),
        copy=False,
    )
    .add_local_file(
        "experiments/__init__.py",
        remote_path=str(REMOTE_REPO / "experiments/__init__.py"),
        copy=False,
    )
    .add_local_dir(
        "experiments/ddm_ec1_runtime",
        remote_path=str(REMOTE_REPO / "experiments/ddm_ec1_runtime"),
        copy=False,
    )
    .add_local_python_source(
        "ddm_ec2_modal_oriented_adapter_trainer",
        "ddm_js1b_modal_cuda_argmax_field_materializer",
    )
)


def _retain_input(path: Path, payload: bytes) -> None:
    if path.is_file():
        if sha256_file(path) != sha256_bytes(payload):
            raise EC2DispatchError(f"remote resume input differs: {path}")
        return
    atomic_bytes(path, payload)


def _bind_prior_field(source: Path, destination: Path, record: dict[str, Any]) -> None:
    require_exact(source, size=int(record["bytes"]), digest=str(record["sha256"]), label="prior T4 field")
    if destination.is_file():
        require_exact(destination, size=int(record["bytes"]), digest=str(record["sha256"]), label="bound T4 field")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    require_exact(destination, size=int(record["bytes"]), digest=str(record["sha256"]), label="bound T4 field")


@app.function(
    image=trainer_image,
    gpu="T4",
    timeout=HARD_CAP_SECONDS,
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_trainer(payloads: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    """Retain immutable inputs, execute one family, and commit advancing checkpoints."""
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    remote_inputs = {**payloads, "REQUEST.json": canonical_json_bytes(request)}
    for name, payload in remote_inputs.items():
        _retain_input(run_root / "inputs" / name, payload)
    _bind_prior_field(PRIOR_GT_FIELD, run_root / "inputs/gt_argmax_n600.npy", GT_FIELD_RECORD)
    _bind_prior_field(
        PRIOR_BASE_FIELD,
        run_root / "inputs/base_argmax_n600.npy",
        BASE_FIELD_RECORD,
    )
    retained_volume.commit()
    # The locked venv is the matched scorer instrument (torch/weights/batch
    # identical to the js1b field materializer) but ships no pydantic, which
    # tac.training's legacy module needs at import (fc-01M006HSKX died on it).
    # Fail-closed self-install per the e4 brotli precedent; training-side only.
    # uv-built venvs ship without pip (r2 fc-01M006SQ2N died on `-m pip`);
    # the image symlinks uv at /usr/local/bin/uv — install through it instead.
    subprocess.run(
        ["/usr/local/bin/uv", "pip", "install", "--quiet",
         "--python", f"{UPSTREAM_LOCKED_VENV}/bin/python", "pydantic>=2.0,<3", "brotli>=1.0"],
        check=True,
    )
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
    retained_volume.commit()
    final_path = run_root / "FINAL_RESULT.json"
    if returncode or not final_path.is_file():
        return {
            "schema": "ddm_ec2_modal_return.v1",
            "training_complete": False,
            "returncode": returncode,
            "axis": AXIS,
            "family": request["family"],
            "run_id": run_id,
            "volume_name": VOLUME_NAME,
            "volume_path": str(run_root),
            "worker_log_tail": log_path.read_text(errors="replace")[-8_000:],
            "resume_same_run_id": run_id,
            "score_claim": False,
        }
    final_bytes = final_path.read_bytes()
    selected_bytes = (run_root / "stages/selected/SELECTED_RESULT.json").read_bytes()
    return {
        "schema": "ddm_ec2_modal_return.v1",
        "training_complete": True,
        "returncode": 0,
        "axis": AXIS,
        "family": request["family"],
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "final_result_sha256": sha256_bytes(final_bytes),
        "artifacts": {
            f"{request['family']}.FINAL_RESULT.json": final_bytes,
            f"{request['family']}.SELECTED_RESULT.json": selected_bytes,
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def _load_sealed(path: Path, expected_sha256: str, fire_input_dir: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    require_exact(path, size=path.stat().st_size, digest=expected_sha256, label="sealed request")
    request = json.loads(path.read_text())
    payloads: dict[str, bytes] = {}
    for name in ("cp135.archive.zip", "cp135.runtime.zip", "decoded_tokens_n600.npy"):
        record = request["payloads"][name]
        source = fire_input_dir / name
        require_exact(source, size=int(record["bytes"]), digest=str(record["sha256"]), label=f"fire input {name}")
        payloads[name] = source.read_bytes()
    return payloads, request


def _claim_and_spawn(
    *,
    payloads: dict[str, bytes],
    request: dict[str, Any],
    output: Path,
) -> str:
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "REQUEST.json", request)
    spec = ClaimSpec(
        lane_id=str(request["lane_id"]),
        instance_job_id=str(request["instance_job_id"]),
        agent=str(request["claim_agent"]),
        notes=f"EC2 {request['family']} trainer; run={request['run_id']}; volume={VOLUME_NAME}",
    )
    claim_modal_auth_eval_dispatch(repo_root=REPO, spec=spec, status="active_ec2_adapter_training_spawning")
    family_label = f"{LANE_LABEL}_{request['family']}"
    assert_modal_single_flight(label=family_label, lane_id=str(request["lane_id"]), repo_root=REPO)
    call = run_trainer.spawn(payloads, request)
    call_id = function_call_id(call)
    register_dispatched_call_id_fail_closed(
        call_id=call_id,
        lane_id=str(request["lane_id"]),
        label=family_label,
        platform="modal",
        gpu="T4",
        expected_axis="contest_cuda_ec2_adapter_trainer_component",
        recipe="experiments/ddm_ec2_modal_oriented_adapter_trainer.py",
        max_seconds=HARD_CAP_SECONDS,
        agent=str(request["claim_agent"]),
        base_archive_sha256=CP135_SHA256,
        composed_archive_sha256=CP135_SHA256,
        archive_count=1,
        volume_name=VOLUME_NAME,
        volume_run_id=str(request["run_id"]),
    )
    write_spawn_metadata(
        out_dir=output,
        tool="experiments/ddm_ec2_modal_oriented_adapter_trainer.py",
        app=APP_NAME,
        axis="contest_cuda_ec2_adapter_trainer_component",
        call_id=call_id,
        local_request=request,
        result_json_name="modal_ec2_result.json",
        recover_tool="experiments/ddm_ec2_modal_oriented_adapter_trainer.py recover",
        extra={
            "lane_id": request["lane_id"],
            "instance_job_id": request["instance_job_id"],
            "claim_agent": request["claim_agent"],
            "volume_name": VOLUME_NAME,
            "volume_run_id": request["run_id"],
        },
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(request["lane_id"]),
            instance_job_id=str(request["instance_job_id"]),
            agent=str(request["claim_agent"]),
            force=True,
            notes=f"EC2 detached trainer accepted; call_id={call_id}; output={output}",
        ),
        status="active_ec2_adapter_training_spawned",
    )
    return call_id


@app.local_entrypoint()
def main(
    sealed_request: str,
    fire_input_dir: str,
    expected_request_sha256: str,
    output_dir: str,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    """MAIN-only oriented first dispatch."""
    if not detach or not provider_detach_ack:
        raise SystemExit("FATAL: EC2 fire order requires --detach --provider-detach-ack")
    payloads, request = _load_sealed(Path(sealed_request), expected_request_sha256, Path(fire_input_dir))
    if request["family"] != "oriented":
        raise SystemExit("FATAL: first dispatch is physically restricted to oriented")
    call_id = _claim_and_spawn(payloads=payloads, request=request, output=Path(output_dir).resolve())
    print(f"DISPATCHED oriented call_id={call_id}")


@app.local_entrypoint()
def controls(
    sealed_class_request: str,
    sealed_undirected_request: str,
    fire_input_dir: str,
    oriented_endpoint_result: str,
    expected_class_request_sha256: str,
    expected_undirected_request_sha256: str,
    output_dir: str,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    """MAIN-only second dispatch; impossible before oriented break-even receipt."""
    if not detach or not provider_detach_ack:
        raise SystemExit("FATAL: control fire order requires --detach --provider-detach-ack")
    endpoint_path = Path(oriented_endpoint_result)
    endpoint = json.loads(endpoint_path.read_text())
    if not endpoint.get("clears_oriented_break_even"):
        raise SystemExit("FATAL: controls remain blocked; oriented break-even was not cleared")
    output = Path(output_dir).resolve()
    calls = {}
    for family, sealed, expected_sha in (
        ("class_only", Path(sealed_class_request), expected_class_request_sha256),
        ("undirected", Path(sealed_undirected_request), expected_undirected_request_sha256),
    ):
        payloads, request = _load_sealed(sealed, expected_sha, Path(fire_input_dir))
        if request["family"] != family:
            raise SystemExit(f"FATAL: {family} sealed request family differs")
        calls[family] = _claim_and_spawn(payloads=payloads, request=request, output=output / family)
    atomic_json(
        output / "CONTROL_DISPATCH.json",
        {
            "schema": "ddm_ec2_control_dispatch.v1",
            "oriented_break_even_receipt": file_record(endpoint_path),
            "calls": calls,
        },
    )
    print(json.dumps(calls, sort_keys=True))


def recover(output: Path, timeout_seconds: float = 0.0) -> int:
    output = output.resolve()
    spawn = json.loads((output / "modal_auth_eval_spawn.json").read_text())
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise EC2DispatchError("remote return is not a dict")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise EC2DispatchError(f"unsafe returned artifact: {name!r}")
        atomic_bytes(output / name, payload)
    atomic_json(output / "modal_ec2_result.json", result)
    complete = bool(result.get("training_complete"))
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if complete else "failed",
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_ec2_adapter_trainer_component",
        evidence_grade=AXIS,
        lane_id=str(spawn["lane_id"]),
        label=f"{LANE_LABEL}_{result.get('family', 'unknown')}",
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
        status="completed_ec2_training_recovered" if complete else "failed_ec2_training_recovered",
        notes=f"EC2 trainer recovered; call_id={call_id}; output={output}",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if complete else 1


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    prepare_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    prepare_parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(
                    archive=args.archive,
                    runtime=args.runtime,
                    tokens=args.tokens,
                    output=args.output,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        raise SystemExit(recover(args.output_dir, args.timeout_seconds))


if __name__ == "__main__":
    _cli()
