#!/usr/bin/env python3
"""Compose retained PR130 section candidates into one real receiver packet.

This scorer-free harness replaces sections inside the actual one-member PR130
archive, retains every materialized payload, double-builds deterministically,
and stages the real ``inflate.sh`` runtime.  It never estimates composition by
adding byte counts: the interaction result comes from one stat of one ZIP.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import io
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from tac.process_group_kill import run_in_process_group

if TYPE_CHECKING:
    import torch


REPO = Path(__file__).resolve().parents[1]
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_OUTPUT_ROOT = SSD_ROOT / "ddm_cp2_20260810"
DEFAULT_RECEIVER_PYTHON = SSD_ROOT / "ddm_pq1_runtime_20260809/venv/bin/python"
BASE_BYTES = 191_052
BASE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
BASE_ARCHIVE = SSD_ROOT / "ddm_pr130_reproduce_20260809/reproduction/archive.zip"
ORIGINAL_BYTES = 37_545_489
VIDEO_NAMES = REPO / "upstream/public_test_video_names.txt"

AI1_ROOT = SSD_ROOT / "ddm_ai1_20260809"
AI1_TEMPORAL_ROOT = AI1_ROOT / "temporal_v2"
REFERENCE_RUNTIME = AI1_TEMPORAL_ROOT / "submission_temporal_reversion"
CP2_RUNTIME = REPO / "experiments/ddm_cp2_runtime"
SM3_ROOT = SSD_ROOT / "ddm_sm3_20260810/final_v3/retained/candidates"

RUNTIME_REFERENCE_HASHES = {
    "carrier_codec.py": "d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed",
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_integer_sparse.py": "2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c",
    "inflate.sh": "bc92880ef9c038c6adfe4968a4b6206b8e565501e839634e1d6762a704421915",
    "integer_model_io.py": "6f91c91ed4785d203aa3570af362fbe9c6a64bb2249599f8554adb31174b80a5",
    "receiver.py": "7dd29117a0cac30b32eb21bcc0e7ee6e1a45bf7f4af8f52ed5e94231945cc111",
}

LZMA_FILTERS = [
    {
        "id": lzma.FILTER_LZMA2,
        "dict_size": 1 << 16,
        "lc": 0,
        "lp": 1,
        "pb": 0,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    path: Path
    bytes: int
    sha256: str
    delta_bytes: int
    expected_state: Path | None = None
    expected_state_sha256: str | None = None


TOKEN_CANDIDATES = {
    "base_range": CandidateSpec("base_range", BASE_ARCHIVE, BASE_BYTES, BASE_SHA256, 0),
    "ans_control": CandidateSpec(
        "ans_control",
        AI1_ROOT / "resumable_v1/retained/ans_control/archive.zip",
        188_932,
        "d966d666da1079354428bbc38e94d0b181dbb8b5af9d0759ff39d51766228532",
        -2_120,
    ),
    "temporal_reversion": CandidateSpec(
        "temporal_reversion",
        AI1_TEMPORAL_ROOT / "retained/temporal_reversion/archive.zip",
        188_636,
        "0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84",
        -2_416,
    ),
}

SEMANTIC_CANDIDATES = {
    "inherit": None,
    "legacy_q4": CandidateSpec(
        "legacy_q4",
        SM3_ROOT / "legacy_q4_control/semantic.bin",
        40_252,
        "9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99",
        0,
        SM3_ROOT / "legacy_q4_control/decoded_state.sm3state",
        "23ef13ea9f00217d3b09250096a6f4cb14a0312612be3b50fe62f5d05bdd7933",
    ),
    "sd1_selected_mixed_q3q4": CandidateSpec(
        "sd1_selected_mixed_q3q4",
        SM3_ROOT / "sd1_selected_mixed_q3q4/semantic.bin",
        39_090,
        "39002165c78ab707c15586110678671cd832101a970de5bd0f3b96824a2aa2cc",
        -848,
        SM3_ROOT / "sd1_selected_mixed_q3q4/decoded_state.sm3state",
        "db13e34c207bfc593e646253b8d6d9bf17f3c0b136efeef019d693ac9aee5840",
    ),
    "vector_scale_vq32": CandidateSpec(
        "vector_scale_vq32",
        SM3_ROOT / "vector_scale_vq32/semantic.bin",
        34_693,
        "8bf3ac4d6d37314c29ab06811dc00734e3d43f27c4399fa42e51e51f8834e7e0",
        -4_648,
        SM3_ROOT / "vector_scale_vq32/decoded_state.sm3state",
        "ca0ef6b348a9c24ffd637f0dc0b67b19eb66553a0f00272d4239e7feb0798875",
    ),
    "pointwise_lowrank_r32": CandidateSpec(
        "pointwise_lowrank_r32",
        SM3_ROOT / "pointwise_lowrank_r32/semantic.bin",
        32_774,
        "fec60231bf557389421fb04c1e1449a5cf1ad04feef954f72b45d70832390d62",
        -6_272,
        SM3_ROOT / "pointwise_lowrank_r32/decoded_state.sm3state",
        "5063e24b2c2374aff44db3f50eb8e000908034a54512dd4cbf3108d5aa59c01e",
    ),
}

RUNTIME_IMPORT_NAMES = (
    "carrier_codec",
    "hpac_integer",
    "hpac_integer_sparse",
    "integer_model_io",
    "receiver",
    "sm3r_receiver",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if executable:
        path.chmod(0o755)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_record(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError(f"{label} differs from its byte-and-SHA pin: {path}")


def require_ssd(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SSD_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"DDM-CP2 bulk evidence must stay on the SSD tier: {resolved}") from error
    return resolved


def storage_preflight(path: Path, minimum_free_bytes: int) -> dict[str, Any]:
    root = require_ssd(path)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"DDM-CP2 storage preflight refused {root}: free={free}, required={minimum_free_bytes}")
    return {"path": str(root), "free_bytes": free, "minimum_free_bytes": minimum_free_bytes}


def load_reference_receiver() -> ModuleType:
    path = REFERENCE_RUNTIME / "receiver.py"
    require_record(
        path,
        size=path.stat().st_size,
        digest=RUNTIME_REFERENCE_HASHES["receiver.py"],
        label="AI1 temporal receiver",
    )
    name = "ddm_cp2_reference_receiver"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned AI1 receiver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def read_stored_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"archive is not the exact one-member stored PR130 grammar: {path}")
        return archive.read(infos[0])


def deterministic_zip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    return output.getvalue()


def semantic_field(models_raw: bytes) -> tuple[bytes, int, int]:
    if len(models_raw) < 8:
        raise RuntimeError("model bundle is truncated before semantic/carrier lengths")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if semantic_end > len(models_raw) or carrier_end >= len(models_raw):
        raise RuntimeError("model bundle semantic/carrier geometry is invalid")
    return models_raw[8:semantic_end], semantic_end, carrier_bytes


def encode_archive(
    receiver: ModuleType,
    models_raw: bytes,
    tokens: bytes,
    *,
    token_codec: str,
    model_codec: str,
) -> tuple[bytes, bytes, bytes]:
    if model_codec != "legacy_lzma":
        raise RuntimeError("DDM-CP2 v1 supports the landed legacy-LZMA model section only")
    models = lzma.compress(models_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    payload = receiver.pack_payload(
        models,
        tokens,
        token_codec=token_codec,
        model_codec=model_codec,
    )
    archive = deterministic_zip(payload)
    return models, payload, archive


def retain_encoding(root: Path, encoding: tuple[bytes, bytes, bytes]) -> dict[str, Any]:
    models, payload, archive = encoding
    atomic_bytes(root / "models.xz", models)
    atomic_bytes(root / "payload.p", payload)
    atomic_bytes(root / "archive.zip", archive)
    return {
        "models": file_record(root / "models.xz"),
        "member": file_record(root / "payload.p"),
        "archive": file_record(root / "archive.zip"),
    }


def state_wire(state: Mapping[str, torch.Tensor]) -> bytes:
    output = bytearray(b"SM3STATE\x01")
    output.extend(struct.pack("<I", len(state)))
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        name_bytes = name.encode()
        shape = tuple(tensor.shape)
        data = memoryview(tensor.numpy().astype("<f4", copy=False)).cast("B")
        output.extend(struct.pack("<H", len(name_bytes)))
        output.extend(name_bytes)
        output.extend(struct.pack("<B", len(shape)))
        output.extend(struct.pack(f"<{len(shape)}I", *shape))
        output.extend(struct.pack("<Q", len(data)))
        output.extend(data)
    return bytes(output)


def copy_runtime(submission: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, digest in RUNTIME_REFERENCE_HASHES.items():
        source = REFERENCE_RUNTIME / name
        require_record(source, size=source.stat().st_size, digest=digest, label=f"runtime {name}")
        destination = submission / name
        atomic_bytes(destination, source.read_bytes(), executable=name == "inflate.sh")
        records[name] = {**file_record(destination), "source": str(source), "source_sha256": digest}
    for name in ("inflate.py", "sm3r_receiver.py"):
        source = CP2_RUNTIME / name
        destination = submission / name
        atomic_bytes(destination, source.read_bytes(), executable=name == "inflate.py")
        records[name] = {**file_record(destination), "source": str(source), "source_sha256": sha256_file(source)}
    dependency_manifest = {
        "schema": "ddm_cp2_runtime_dependencies.v1",
        "borrowed_runtime": str(REFERENCE_RUNTIME),
        "borrowed_files": sorted(RUNTIME_REFERENCE_HASHES),
        "cp2_files": ["inflate.py", "sm3r_receiver.py"],
        "third_party_dependencies_unchanged": ["constriction==0.5.0", "numpy", "torch"],
        "brotli_not_selected_by_current_legacy_lzma_packets": True,
        "score_claim": False,
        "files": records,
    }
    atomic_json(submission / "runtime-dependencies.json", dependency_manifest)
    records["runtime-dependencies.json"] = file_record(submission / "runtime-dependencies.json")
    return records


def load_cp2_inflate(submission: Path) -> ModuleType:
    prior_path = list(sys.path)
    prior_modules = {name: sys.modules.get(name) for name in RUNTIME_IMPORT_NAMES}
    for name in RUNTIME_IMPORT_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(submission))
    name = "ddm_cp2_runtime_inflate"
    spec = importlib.util.spec_from_file_location(name, submission / "inflate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the staged DDM-CP2 inflate runtime")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = prior_path
        for imported in RUNTIME_IMPORT_NAMES:
            sys.modules.pop(imported, None)
            if prior_modules[imported] is not None:
                sys.modules[imported] = prior_modules[imported]
    return module


def build(
    output: Path,
    *,
    semantic_id: str,
    token_id: str,
    resume_from: Path,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    output = require_ssd(output)
    preflight = storage_preflight(output, minimum_free_bytes)
    receipt_path = output / "build_receipt.json"
    if receipt_path.is_file():
        result = json.loads(receipt_path.read_text())
        if result.get("schema") == "ddm_cp2_composed_archive.v2" and result.get("complete") is True:
            for record in result["retained"].values():
                require_record(
                    Path(record["path"]), size=record["bytes"], digest=record["sha256"], label="retained build artifact"
                )
            return result

    token_spec = TOKEN_CANDIDATES[token_id]
    semantic_spec = SEMANTIC_CANDIDATES[semantic_id]
    require_record(
        token_spec.path, size=token_spec.bytes, digest=token_spec.sha256, label=f"token candidate {token_id}"
    )
    if semantic_spec is not None:
        require_record(
            semantic_spec.path,
            size=semantic_spec.bytes,
            digest=semantic_spec.sha256,
            label=f"semantic candidate {semantic_id}",
        )
        if semantic_spec.expected_state is not None:
            require_record(
                semantic_spec.expected_state,
                size=semantic_spec.expected_state.stat().st_size,
                digest=str(semantic_spec.expected_state_sha256),
                label=f"semantic state {semantic_id}",
            )

    state = {
        "schema": "ddm_cp2_resume.v1",
        "complete": False,
        "semantic_id": semantic_id,
        "token_id": token_id,
        "updated_at_utc": utc_now(),
    }
    if resume_from.exists():
        prior = json.loads(resume_from.read_text())
        if prior.get("semantic_id") != semantic_id or prior.get("token_id") != token_id:
            raise RuntimeError("DDM-CP2 resume selection differs")
    atomic_json(resume_from, state)

    receiver = load_reference_receiver()
    driver_member = read_stored_member(token_spec.path)
    driver_parts = receiver.split_payload(driver_member)
    decoded_driver = receiver.decode_models(driver_parts.models, model_codec=driver_parts.model_codec)
    driver_semantic, driver_semantic_end, carrier_bytes = semantic_field(decoded_driver.raw)
    selected_semantic = driver_semantic if semantic_spec is None else semantic_spec.path.read_bytes()
    composed_raw = (
        struct.pack("<II", len(selected_semantic), carrier_bytes)
        + selected_semantic
        + decoded_driver.raw[driver_semantic_end:]
    )

    source_root = output / "retained/sources"
    atomic_bytes(source_root / "driver.payload.p", driver_member)
    atomic_bytes(source_root / "driver.models.raw", decoded_driver.raw)
    atomic_bytes(source_root / "driver.semantic.bin", driver_semantic)
    atomic_bytes(source_root / "driver.tokens.bin", driver_parts.tokens)
    atomic_bytes(source_root / "selected.semantic.bin", selected_semantic)
    atomic_bytes(source_root / "composed.models.raw", composed_raw)
    atomic_bytes(source_root / "preserved.model_suffix.bin", decoded_driver.raw[driver_semantic_end:])

    control = encode_archive(
        receiver,
        decoded_driver.raw,
        driver_parts.tokens,
        token_codec=driver_parts.token_codec,
        model_codec=driver_parts.model_codec,
    )
    control_records = retain_encoding(output / "retained/control", control)
    if control[2] != token_spec.path.read_bytes():
        raise RuntimeError("driver rebuild is not byte-identical to its source archive")

    first = encode_archive(
        receiver,
        composed_raw,
        driver_parts.tokens,
        token_codec=driver_parts.token_codec,
        model_codec=driver_parts.model_codec,
    )
    second = encode_archive(
        receiver,
        composed_raw,
        driver_parts.tokens,
        token_codec=driver_parts.token_codec,
        model_codec=driver_parts.model_codec,
    )
    first_records = retain_encoding(output / "retained/build_a", first)
    second_records = retain_encoding(output / "retained/build_b", second)
    if first != second:
        raise RuntimeError("composed archive double-build differs")

    submission = output / "submission"
    atomic_bytes(submission / "archive.zip", first[2])
    atomic_bytes(submission / "archive/p", first[1])
    runtime_records = copy_runtime(submission)

    parsed_member = read_stored_member(submission / "archive.zip")
    parsed_parts = receiver.split_payload(parsed_member)
    parsed_models = receiver.decode_models(parsed_parts.models, model_codec=parsed_parts.model_codec)
    if parsed_models.raw != composed_raw or parsed_parts.tokens != driver_parts.tokens:
        raise RuntimeError("real outer receiver parse-back differs from composed fields")
    core_models, temporal = receiver.split_optional_temporal_reversion(parsed_models.raw)
    parsed_semantic, parsed_semantic_end, _ = semantic_field(core_models)
    if parsed_semantic != selected_semantic:
        raise RuntimeError("real outer receiver changed the selected semantic field")
    temporal_path = output / "retained/parseback/temporal_reversion.bin"
    if temporal is not None:
        atomic_bytes(temporal_path, temporal.packed)
    runtime = load_cp2_inflate(submission)
    semantic_model, basis, coeff = runtime.unpack_semantic_pose(core_models[: parsed_semantic_end + carrier_bytes])
    semantic_wire = state_wire(semantic_model.state_dict())
    atomic_bytes(output / "retained/parseback/semantic_state.sm3state", semantic_wire)
    atomic_bytes(
        output / "retained/parseback/basis.f32le",
        basis.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C"),
    )
    atomic_bytes(
        output / "retained/parseback/coeff.f32le",
        coeff.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C"),
    )
    if (
        semantic_spec is not None
        and semantic_spec.expected_state is not None
        and semantic_wire != semantic_spec.expected_state.read_bytes()
    ):
        raise RuntimeError("shipped SM3R loader state differs from SM3 packer state")

    actual_bytes = (output / "retained/build_a/archive.zip").stat().st_size
    semantic_delta = 0 if semantic_spec is None else semantic_spec.delta_bytes
    expected_additive_bytes = BASE_BYTES + token_spec.delta_bytes + semantic_delta
    interaction_gap = actual_bytes - expected_additive_bytes
    if interaction_gap < 0:
        interaction_class = "SUPERADDITIVE_SAVINGS"
    elif interaction_gap > 0:
        interaction_class = "SUBADDITIVE_SAVINGS"
    else:
        interaction_class = "EXACTLY_ADDITIVE"
    retained = {
        "archive": first_records["archive"],
        "archive_repeat": second_records["archive"],
        "member": first_records["member"],
        "member_repeat": second_records["member"],
        "models": first_records["models"],
        "models_repeat": second_records["models"],
        "driver_member": file_record(source_root / "driver.payload.p"),
        "driver_models_raw": file_record(source_root / "driver.models.raw"),
        "driver_semantic": file_record(source_root / "driver.semantic.bin"),
        "driver_tokens": file_record(source_root / "driver.tokens.bin"),
        "models_raw": file_record(source_root / "composed.models.raw"),
        "semantic": file_record(source_root / "selected.semantic.bin"),
        "tokens": file_record(source_root / "driver.tokens.bin"),
        "preserved_model_suffix": file_record(source_root / "preserved.model_suffix.bin"),
        "semantic_state": file_record(output / "retained/parseback/semantic_state.sm3state"),
        "basis": file_record(output / "retained/parseback/basis.f32le"),
        "coeff": file_record(output / "retained/parseback/coeff.f32le"),
        "submission_archive": file_record(submission / "archive.zip"),
        "submission_member": file_record(submission / "archive/p"),
    }
    if temporal is not None:
        retained["temporal_reversion"] = file_record(temporal_path)
    result = {
        "schema": "ddm_cp2_composed_archive.v2",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[scorer-free exact archive bytes; real shipped receiver parse-back]",
        "score_claim": False,
        "semantic_id": semantic_id,
        "token_id": token_id,
        "base": {"bytes": BASE_BYTES, "sha256": BASE_SHA256},
        "actual_archive_bytes": actual_bytes,
        "actual_archive_delta_bytes": actual_bytes - BASE_BYTES,
        "rate_only_delta_s": 25 * (actual_bytes - BASE_BYTES) / ORIGINAL_BYTES,
        "individual_deltas": {"tokens": token_spec.delta_bytes, "semantic": semantic_delta},
        "summed_individual_delta_bytes": token_spec.delta_bytes + semantic_delta,
        "additive_expected_archive_bytes": expected_additive_bytes,
        "interaction_gap_bytes_actual_minus_additive": interaction_gap,
        "interaction_class": interaction_class,
        "checks": {
            "driver_rebuild_byte_identical": True,
            "archive_double_build_byte_identical": True,
            "outer_receiver_models_raw_byte_identical": True,
            "outer_receiver_token_payload_byte_identical": True,
            "semantic_parseback_byte_identical": True,
            "shipped_semantic_state_equals_packer_state": semantic_spec is not None
            and semantic_spec.expected_state is not None,
            "shipped_sm3r_state_equals_packer_state": (
                selected_semantic.startswith(b"SM3R")
                and semantic_spec is not None
                and semantic_spec.expected_state is not None
            ),
            "real_inflate_sh_staged": True,
            "real_inflate_sh_executed": False,
        },
        "storage_preflight": preflight,
        "retained": retained,
        "control": control_records,
        "submission": str(submission.resolve()),
        "runtime_files": runtime_records,
        "sources": {
            "token_archive": file_record(token_spec.path),
            "semantic": None if semantic_spec is None else file_record(semantic_spec.path),
        },
    }
    atomic_json(receipt_path, result)
    state["complete"] = True
    state["updated_at_utc"] = utc_now()
    state["build_receipt"] = file_record(receipt_path)
    atomic_json(resume_from, state)
    return result


def inflate_candidate(
    output: Path,
    *,
    python: Path,
    timeout_seconds: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    output = require_ssd(output)
    preflight = storage_preflight(output, minimum_free_bytes)
    build_receipt = json.loads((output / "build_receipt.json").read_text())
    if build_receipt.get("complete") is not True:
        raise RuntimeError("candidate build is incomplete")
    run_root = output / "receiver_parseback"
    receipt_path = run_root / "inflate_receipt.json"
    if receipt_path.is_file():
        result = json.loads(receipt_path.read_text())
        raw = Path(result["raw"]["path"])
        require_record(raw, size=result["raw"]["bytes"], digest=result["raw"]["sha256"], label="retained inflated RAW")
        return result
    run_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_root / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError("a DDM-CP2 receiver parse-back already owns this candidate") from error

    submission = Path(build_receipt["submission"])
    checkpoint = run_root / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    staging = run_root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "inflate.log"
    python_path = python if python.is_absolute() else REPO / python
    python_path = Path(os.path.abspath(python_path))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise RuntimeError(f"receiver Python is not an executable file: {python_path}")
    env = os.environ.copy()
    env.update(
        {
            # Do not resolve the venv symlink: the resolved Homebrew interpreter
            # loses the venv prefix and therefore its contest-runtime packages.
            "PYTHON": str(python_path),
            "PR130_INFLATE_DEVICE": "cpu",
            "PR130_RUNTIME_DEPS_DIR": str((run_root / "runtime-deps").resolve()),
            "PR130_TOKEN_CACHE": str((checkpoint / "tokens.npz").resolve()),
            "PR130_TOKEN_RECEIPT": str((checkpoint / "tokens_receipt.json").resolve()),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = [
        str(submission / "inflate.sh"),
        str(submission / "archive"),
        str(staging),
        str(VIDEO_NAMES),
    ]
    started = dt.datetime.now(dt.UTC)
    try:
        with log_path.open("ab") as log:
            process = run_in_process_group(
                command,
                cwd=submission,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        if process.returncode != 0:
            raise RuntimeError(f"real inflate.sh failed with rc={process.returncode}; see {log_path}")
        raw = staging / "0.raw"
        expected_raw_bytes = 600 * 2 * 874 * 1_164 * 3
        if raw.stat().st_size != expected_raw_bytes:
            raise RuntimeError("real inflate.sh produced the wrong RAW geometry")
        result = {
            "schema": "ddm_cp2_real_inflate_parseback.v1",
            "complete": True,
            "written_at_utc": utc_now(),
            "axis": "[macOS-CPU scorer-free real inflate.sh parse-back]",
            "score_claim": False,
            "command": command,
            "environment": {
                "PYTHON": str(python_path),
                "PYTHON_resolved": str(python_path.resolve()),
                "PR130_INFLATE_DEVICE": "cpu",
                "PR130_RUNTIME_DEPS_DIR": env["PR130_RUNTIME_DEPS_DIR"],
                "PR130_TOKEN_CACHE": env["PR130_TOKEN_CACHE"],
                "PR130_TOKEN_RECEIPT": env["PR130_TOKEN_RECEIPT"],
            },
            "returncode": process.returncode,
            "wall_seconds": (dt.datetime.now(dt.UTC) - started).total_seconds(),
            "archive": build_receipt["retained"]["archive"],
            "raw": file_record(raw),
            "log": file_record(log_path),
            "token_checkpoint": file_record(checkpoint / "tokens.npz"),
            "token_receipt": file_record(checkpoint / "tokens_receipt.json"),
            "storage_preflight": preflight,
            "within_1800_second_inflate_limit": (dt.datetime.now(dt.UTC) - started).total_seconds() <= 1_800,
        }
        token_progress = checkpoint / "tokens.progress.npz"
        if token_progress.is_file():
            result["token_progress"] = file_record(token_progress)
        atomic_json(receipt_path, result)
        build_receipt["checks"]["real_inflate_sh_executed"] = True
        build_receipt["real_inflate_receipt"] = file_record(receipt_path)
        atomic_json(output / "build_receipt.json", build_receipt)
        return result
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "inflate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", choices=tuple(SEMANTIC_CANDIDATES), default="inherit")
    parser.add_argument("--tokens", choices=tuple(TOKEN_CANDIDATES), default="base_range")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_RECEIVER_PYTHON)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    parser.add_argument("--minimum-free-bytes", type=int, default=12 << 30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.minimum_free_bytes <= 0:
        parser.error("timeout and minimum-free-bytes must be positive")
    if args.command == "build" and args.resume_from is None:
        parser.error("build requires --resume-from")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build(
            args.output,
            semantic_id=args.semantic,
            token_id=args.tokens,
            resume_from=require_ssd(args.resume_from),
            minimum_free_bytes=args.minimum_free_bytes,
        )
    else:
        result = inflate_candidate(
            args.output,
            python=args.python,
            timeout_seconds=args.timeout_seconds,
            minimum_free_bytes=args.minimum_free_bytes,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
