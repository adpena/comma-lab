#!/usr/bin/env python3
"""Build and train the receiver-closed DDM WD2 width-distilled student.

This instrument is scorer-free.  ``verify-build`` derives the architecture
ladder, patches a retained copy of the exact e480b runtime, and proves the
inactive WANS1 parse path byte-identical.  ``prepare-teacher-cache`` retains
the frozen renderer's real receiver+uint8 output.  ``train`` performs seeded
QAT distillation against that cache with atomic resumable checkpoints and
retains every evaluated student packet, rendered output, and full archive.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import resource
import shutil
import struct
import sys
import time
import zipfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for root in (REPO, SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_wd2_student_receiver as receiver
from tac.admission_guard import assert_governed_admission

RETENTION_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_wd2_width_distillation")
TRAINING_ROOT = Path("/Volumes/APDataStore/pact/ddm_wd2_width_distillation")
SSD_ROOTS = (RETENTION_ROOT.parent, TRAINING_ROOT.parent)
SOURCE_ROOT = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac")
SOURCE_CANDIDATE = SOURCE_ROOT / "retained/candidates/s1p25_c1p0/brotli_q10"
SOURCE_ARCHIVE = SOURCE_CANDIDATE / "archive.zip"
SOURCE_MEMBER = SOURCE_CANDIDATE / "p"
SOURCE_MODEL = SOURCE_CANDIDATE / "models.rx1m"
SOURCE_TOKEN = SOURCE_CANDIDATE / "tokens.rc64"
SOURCE_RESIDUAL = SOURCE_CANDIDATE / "residual.compact.bin"
TOKEN_CACHE = SOURCE_ROOT / "inputs/mc36_spatial_tokens_uint8.pt"
TOKEN_CACHE_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/inputs/"
    "mc36_training_cache_receipt.json"
)
SOURCE_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac/e480b_submission_v2"
)

EXPECTED = {
    "archive": (183_502, "e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3"),
    "member": (183_402, "30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58"),
    "model": (70_557, "7cf390160189e8708faf3a7b09a76fc18cee85e45fdc7f71d30f725014417411"),
    "token": (112_749, "b981b8399f184795da7cd99b8ee44416bd672c8c4ed1672f1252b32a64c10627"),
    "residual": (96, "64bbf9dfd88d6eb50d111f72d968ab7e8f8dc0ab00fb675d8ed2ee8a410b73ac"),
    "token_cache": (117_967_085, "f53db4e8e65789d7d0442e97f8531bfb9765f41a2c37c8509c6ccdaeb8a6c888"),
}
EXPECTED_MZ2_SHA256 = "7ded1d52d634100e00e86eff6e3769fa3c366d15bdeb6f065605d962638567c1"
EXPECTED_DONOR_SHA256 = "ecc923430c2af084ba4c582fab86b57f1ffc461929569aa167730d9afbcd19f6"
CURRENT_ARCHIVE_BYTES = 183_502
CURRENT_SEMANTIC_STREAM_BYTES = 34_763
CURRENT_CARRIER_STREAM_BYTES = 22_161
RX1_WRAPPER_BYTES = 14
REQUIRED_RUNG_SAVINGS = 15_153
RUNG_SEMANTIC_STREAM_CEILING = 19_610
SUB015_RATE_ONLY_SEMANTIC_STREAM_CEILING = 19_606
BEST_RETAINED_STRUCTURAL_SAVINGS = 2_051
RATE_DENOMINATOR = 37_545_489
CURRENT_SCORE = 0.1600920261571558
SEED = 20260815
AXIS = "[macOS-CPU apparatus; scorer-free current-e480b byte/receiver proof]"

RX1_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
RX1_VERSION = 1
RX1_CODEC_BROTLI = 2
RX1_TABLE_ON = 0

CHECKPOINT_SCHEMA = "ddm_wd2_width_distillation_checkpoint.v1"
TEACHER_CACHE_SCHEMA = "ddm_wd2_teacher_receiver_cache.v1"


class WD2BuildError(RuntimeError):
    """The WD2 build, custody, or training contract failed closed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WD2BuildError(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    record = file_record(path)
    if expected is not None and (record["bytes"], record["sha256"]) != expected:
        raise WD2BuildError(f"input pin changed: {path}")
    return record


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def atomic_torch(path: Path, value: Any) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return atomic_bytes(path, buffer.getvalue())


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def storage_preflight(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    resolved = output.resolve()
    if not any(resolved == root or root in resolved.parents for root in SSD_ROOTS):
        raise WD2BuildError("WD2 output must remain on the governed SSD waterfall")
    free = shutil.disk_usage(output).free
    if free < minimum_free_bytes:
        raise WD2BuildError(
            f"WD2 needs {minimum_free_bytes} free bytes at {output}; observed {free}"
        )
    return {
        "root": str(resolved),
        "observed_free_bytes": free,
        "required_free_bytes": minimum_free_bytes,
        "status": "PASS",
        "cleanup_policy": (
            "certify-or-block; retained teacher outputs, checkpoints, candidate renders, "
            "packets, and archives are never auto-deleted"
        ),
        "tier": "VertigoDataTier" if RETENTION_ROOT.parent in resolved.parents else "APDataStore",
    }


def _source_streams() -> tuple[bytes, bytes, bytes]:
    model = SOURCE_MODEL.read_bytes()
    magic, version, codec, table_mode, reserved, hpac_size, semantic_size, carrier_size = (
        RX1_HEADER.unpack_from(model)
    )
    if (magic, version, codec, table_mode, reserved) != (
        RX1_MAGIC,
        RX1_VERSION,
        RX1_CODEC_BROTLI,
        RX1_TABLE_ON,
        0,
    ):
        raise WD2BuildError("source RX1 header changed")
    if (semantic_size, carrier_size) != (
        CURRENT_SEMANTIC_STREAM_BYTES,
        CURRENT_CARRIER_STREAM_BYTES,
    ):
        raise WD2BuildError("source semantic/carrier section lengths changed")
    offset = RX1_HEADER.size
    hpac = model[offset : offset + hpac_size]
    semantic = model[offset + hpac_size : offset + hpac_size + semantic_size]
    carrier = model[offset + hpac_size + semantic_size :]
    if len(carrier) != carrier_size:
        raise WD2BuildError("source RX1 field accounting changed")
    return hpac, semantic, carrier


def _pack_rx1_model(hpac: bytes, semantic: bytes, carrier: bytes) -> bytes:
    if max(len(hpac), len(semantic), len(carrier)) > 0xFFFF:
        raise WD2BuildError("WD2 RX1 section exceeds the v1 uint16 field")
    return RX1_HEADER.pack(
        RX1_MAGIC,
        RX1_VERSION,
        RX1_CODEC_BROTLI,
        RX1_TABLE_ON,
        0,
        len(hpac),
        len(semantic),
        len(carrier),
    ) + hpac + semantic + carrier


def _candidate_specs(byte_ceiling: int = SUB015_RATE_ONLY_SEMANTIC_STREAM_CEILING) -> list[receiver.StudentSpec]:
    """Derive iso-payload maxima; every discrete choice follows from the ceiling."""

    specs: list[receiver.StudentSpec] = []
    for form in ("dense", "flattened"):
        for depth in range(1, 5):
            best: receiver.StudentSpec | None = None
            for width in range(8, 129):
                try:
                    candidate = receiver.StudentSpec(
                        candidate_id=f"{form}_d{depth}_w{width}",
                        form=form,
                        width=width,
                        depth=depth,
                    )
                except ValueError:
                    continue
                if receiver.serialized_bytes_for_spec(candidate) <= byte_ceiling:
                    best = candidate
            if best is not None and not (
                form == "flattened" and depth == 1
            ):
                specs.append(best)
    for width, depth in ((56, 4), (64, 4), (72, 3), (80, 3), (96, 2), (104, 1)):
        best = None
        for rank in range(1, width + 1):
            candidate = receiver.StudentSpec(
                candidate_id=f"factorized_d{depth}_w{width}_r{rank}",
                form="factorized",
                width=width,
                depth=depth,
                rank=rank,
            )
            if receiver.serialized_bytes_for_spec(candidate) <= byte_ceiling:
                best = candidate
        if best is not None:
            specs.append(best)
    return specs


def design_receipt() -> dict[str, Any]:
    specs = _candidate_specs()
    rows = []
    for spec in specs:
        packet_bytes = receiver.serialized_bytes_for_spec(spec)
        rows.append(
            {
                **spec.as_dict(),
                "exact_uncompressed_packet_bytes": packet_bytes,
                "headroom_to_rung_stream_ceiling_bytes": (
                    RUNG_SEMANTIC_STREAM_CEILING - packet_bytes
                ),
                "headroom_to_sub015_rate_only_stream_ceiling_bytes": (
                    SUB015_RATE_ONLY_SEMANTIC_STREAM_CEILING - packet_bytes
                ),
                "selection_law": (
                    "maximum legal width or rank at fixed depth/form under the exact "
                    "rate-only sub-0.15 semantic packet ceiling"
                ),
            }
        )
    required_for_sub015 = math.floor(
        (CURRENT_SCORE - 0.15) * RATE_DENOMINATOR / 25
    ) + 1
    return {
        "schema": "ddm_wd2_student_design.v1",
        "axis": "[byte-only exact packet accounting; compressed stream remains to be measured]",
        "score_claim": False,
        "baseline": {
            "archive_bytes": CURRENT_ARCHIVE_BYTES,
            "semantic_stream_bytes": CURRENT_SEMANTIC_STREAM_BYTES,
            "carrier_stream_bytes": CURRENT_CARRIER_STREAM_BYTES,
            "wrapper_bytes": RX1_WRAPPER_BYTES,
            "pool_bytes": CURRENT_SEMANTIC_STREAM_BYTES
            + CURRENT_CARRIER_STREAM_BYTES
            + RX1_WRAPPER_BYTES,
            "score": CURRENT_SCORE,
        },
        "budget_derivation": {
            "charter_rung_savings_bytes": REQUIRED_RUNG_SAVINGS,
            "rung_pool_ceiling_bytes": (
                CURRENT_SEMANTIC_STREAM_BYTES
                + CURRENT_CARRIER_STREAM_BYTES
                + RX1_WRAPPER_BYTES
                - REQUIRED_RUNG_SAVINGS
            ),
            "fixed_carrier_plus_wrapper_bytes": CURRENT_CARRIER_STREAM_BYTES
            + RX1_WRAPPER_BYTES,
            "rung_semantic_stream_ceiling_bytes": RUNG_SEMANTIC_STREAM_CEILING,
            "rung_rate_only_delta_score": -REQUIRED_RUNG_SAVINGS
            * 25
            / RATE_DENOMINATOR,
            "rung_rate_only_projected_score": CURRENT_SCORE
            - REQUIRED_RUNG_SAVINGS * 25 / RATE_DENOMINATOR,
            "strict_sub015_rate_only_required_savings_bytes": required_for_sub015,
            "strict_sub015_rate_only_semantic_stream_ceiling_bytes": (
                CURRENT_SEMANTIC_STREAM_BYTES - required_for_sub015
            ),
            "strict_sub015_rate_only_archive_ceiling_bytes": (
                CURRENT_ARCHIVE_BYTES - required_for_sub015
            ),
            "distortion_term": (
                "UNKNOWN until exact n600 authority; decode fidelity is a guard and cannot "
                "be converted into contest-score units"
            ),
        },
        "capacity_law": (
            "An iso-payload width/depth envelope is enumerated because no HM1/QA83 law "
            "transfers distortion constants to this renderer. Dense, low-rank pointwise, "
            "and single-FiLM flattened forms are all first-class rather than selecting a "
            "proxy winner before training."
        ),
        "candidates": rows,
        "primary_candidate": "flattened_d4_w64",
        "primary_rationale": (
            "At the inherited four-block receptive depth it preserves full-rank pointwise "
            "mixing and buys width 64 by paying frame conditioning once; factorized_d4_w64 "
            "is the matched conditioning-rich control and dense_d4_w56 is the no-mechanism control."
        ),
        "falsifier": {
            "verdict_scope": "FAMILY on this exact frozen e480b teacher and WD2 packet family",
            "condition": (
                "After family-optimal training, if the smallest distortion-holding student "
                f"saves <= {BEST_RETAINED_STRUCTURAL_SAVINGS} exact archive bytes, width "
                "distillation is priced out by mz2's retained structural candidate."
            ),
        },
    }


def _load_teacher() -> receiver.StudentSemanticRenderer:
    mz2_path = REPO / "experiments/ddm_mz2_frozen_section_representation_attack.py"
    if sha256_file(mz2_path) != EXPECTED_MZ2_SHA256:
        raise WD2BuildError("MZ2 exact teacher decoder source changed")
    mz2 = importlib.import_module("experiments.ddm_mz2_frozen_section_representation_attack")
    records, _, _ = mz2._load_records()
    teacher_spec = receiver.StudentSpec("frozen_teacher", "dense", 96, 4)
    teacher = receiver.StudentSemanticRenderer(teacher_spec)
    state = OrderedDict(
        (
            record.schema.name,
            torch.from_numpy(np.ascontiguousarray(record.values, dtype=np.float32)),
        )
        for record in records
    )
    if tuple(state) != tuple(teacher.state_dict()):
        raise WD2BuildError("WD2 teacher topology differs from the frozen receiver schema")
    teacher.load_state_dict(state, strict=True)
    return teacher.eval()


def _load_tokens() -> torch.Tensor:
    require_file(TOKEN_CACHE, EXPECTED["token_cache"])
    payload = torch.load(TOKEN_CACHE, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "ddm_rx2_mc36_training_cache.v1"
        or payload.get("spatial_token_sha256")
        != "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
    ):
        raise WD2BuildError("MC36 token cache identity differs")
    tokens = payload.get("seg")
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (
        receiver.N,
        receiver.EVAL_H,
        receiver.EVAL_W,
    ):
        raise WD2BuildError("MC36 token cache geometry differs")
    return tokens.to(torch.uint8)


def _load_residual_parts(runtime_tree: Path, archive_path: Path) -> dict[str, bytes]:
    prior = {name: module for name, module in sys.modules.items() if name == "runtime" or name.startswith("runtime.")}
    for name in list(prior):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(runtime_tree.resolve()))
    try:
        module = importlib.import_module("runtime.residual_archive")
        parts = module.read_residual_archive(archive_path)
        return {
            "semantic_blob": bytes(parts.semantic_blob),
            "carrier_blob": bytes(parts.carrier_blob),
            "hpac_blob": bytes(parts.hpac_blob),
            "token_stream": bytes(parts.token_stream),
            "residual_payload": bytes(parts.residual_payload),
            "compressed_models": bytes(parts.compressed_models),
            "compensation_blob": bytes(parts.compensation_blob or b""),
        }
    finally:
        sys.path.pop(0)
        for name in [name for name in sys.modules if name == "runtime" or name.startswith("runtime.")]:
            sys.modules.pop(name, None)
        sys.modules.update(prior)


def verify_build(output: Path, minimum_free_bytes: int) -> dict[str, Any]:
    storage = storage_preflight(output, minimum_free_bytes)
    inputs = {
        "archive": require_file(SOURCE_ARCHIVE, EXPECTED["archive"]),
        "member": require_file(SOURCE_MEMBER, EXPECTED["member"]),
        "model": require_file(SOURCE_MODEL, EXPECTED["model"]),
        "token": require_file(SOURCE_TOKEN, EXPECTED["token"]),
        "residual": require_file(SOURCE_RESIDUAL, EXPECTED["residual"]),
        "token_cache": require_file(TOKEN_CACHE, EXPECTED["token_cache"]),
        "token_cache_receipt": require_file(TOKEN_CACHE_RECEIPT),
        "donor": require_file(REPO / "tools/train_ddm_cl1_hpac_capacity_mps.py"),
        "composer": require_file(REPO / "tools/fire_watched_continuation.py"),
        "builder": require_file(Path(__file__).resolve()),
        "receiver": require_file(Path(receiver.__file__).resolve()),
    }
    if inputs["donor"]["sha256"] != EXPECTED_DONOR_SHA256:
        raise WD2BuildError("sealed MPS convention donor changed")
    if not SOURCE_RUNTIME.is_dir():
        raise WD2BuildError("pinned e480b runtime tree is absent")

    design = design_receipt()
    atomic_json(output / "DESIGN_RECEIPT.json", design)

    inactive_root = output / "retained/inactive_base"
    source_archive_copy = atomic_bytes(inactive_root / "archive.zip", SOURCE_ARCHIVE.read_bytes())
    source_member_copy = atomic_bytes(inactive_root / "p", SOURCE_MEMBER.read_bytes())
    source_model_copy = atomic_bytes(inactive_root / "models.rx1m", SOURCE_MODEL.read_bytes())
    runtime_root = output / "retained/runtime_inactive_base"
    patch_receipt = receiver.patch_runtime_tree(SOURCE_RUNTIME, runtime_root)
    archive_binding = receiver.bind_archive(runtime_root, inactive_root / "archive.zip")

    original_parts = _load_residual_parts(SOURCE_RUNTIME, SOURCE_ARCHIVE)
    patched_parts = _load_residual_parts(runtime_root, runtime_root / "archive.zip")
    if original_parts != patched_parts:
        changed = sorted(name for name in original_parts if original_parts[name] != patched_parts[name])
        raise WD2BuildError(f"inactive receiver parse-back changed fields: {changed}")
    retained_parts = {}
    for name, value in patched_parts.items():
        retained_parts[name] = atomic_bytes(inactive_root / f"{name}.bin", value)

    tokens = _load_tokens()
    teacher = _load_teacher()
    current_renderer_dir = SOURCE_RUNTIME / "cpr1"
    current_name = "_ddm_wd2_current_renderer"
    sys.path.insert(0, str(current_renderer_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            current_name, current_renderer_dir / "inflate.py"
        )
        if spec is None or spec.loader is None:
            raise WD2BuildError("could not load current renderer")
        current = importlib.util.module_from_spec(spec)
        sys.modules[current_name] = current
        spec.loader.exec_module(current)
    finally:
        sys.path.pop(0)
    current_model = current.SemanticTokenRenderer(96).eval()
    current_model.load_state_dict(teacher.state_dict(), strict=True)
    with torch.no_grad():
        index = torch.tensor([0], dtype=torch.long)
        expected_master = receiver.camera_uint8(current_model, tokens[:1], index)
        observed_master = receiver.camera_uint8(teacher, tokens[:1], index)
    expected_bytes = expected_master.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
    observed_bytes = observed_master.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
    expected_record = atomic_bytes(
        inactive_root / "teacher_current_receiver_pair0000.rgb.u8", expected_bytes
    )
    observed_record = atomic_bytes(
        inactive_root / "teacher_wd2_topology_pair0000.rgb.u8", observed_bytes
    )
    if expected_bytes != observed_bytes:
        raise WD2BuildError("WD2 teacher topology is not uint8-identical to current receiver")

    result = {
        "schema": "ddm_wd2_width_distillation_build.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "scorer_invocations": 0,
        "training_launched": False,
        "modal_invocations": 0,
        "live_e960_touched": False,
        "storage": storage,
        "inputs": inputs,
        "design_receipt": file_record(output / "DESIGN_RECEIPT.json"),
        "inactive_receiver_proof": {
            "source_archive_copy": source_archive_copy,
            "source_member_copy": source_member_copy,
            "source_model_copy": source_model_copy,
            "runtime_patch": patch_receipt,
            "archive_binding": archive_binding,
            "parse_back_fields": retained_parts,
            "all_parse_back_fields_byte_identical": True,
            "current_receiver_pair0": expected_record,
            "wd2_teacher_pair0": observed_record,
            "pair0_payload_layout": "NHWC uint8 [1,874,1164,3]",
            "pair0_camera_uint8_byte_identical": True,
            "scope": (
                "exact e480b archive parse-back plus pair-0 semantic master realization; "
                "no full-video inflate was run"
            ),
        },
        "payload_retention": (
            "every newly materialized parse-back field and both realized pair-0 RGB payloads retained"
        ),
    }
    atomic_json(output / "BUILD_RECEIPT.json", result)
    return result


def _device(device_name: str) -> torch.device:
    if device_name not in {"cpu", "mps"}:
        raise WD2BuildError("WD2 device must be cpu or mps")
    if device_name == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise WD2BuildError("set PYTORCH_ENABLE_MPS_FALLBACK=0 for MPS")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise WD2BuildError("MPS is unavailable; CPU substitution is forbidden")
    return torch.device(device_name)


def _verify_launch_source_pins(
    expected_builder_sha256: str, expected_receiver_sha256: str
) -> None:
    observed_builder = sha256_file(Path(__file__).resolve())
    observed_receiver = sha256_file(Path(receiver.__file__).resolve())
    if observed_builder != expected_builder_sha256:
        raise WD2BuildError(
            f"sealed builder SHA changed: expected {expected_builder_sha256}, observed {observed_builder}"
        )
    if observed_receiver != expected_receiver_sha256:
        raise WD2BuildError(
            f"sealed receiver SHA changed: expected {expected_receiver_sha256}, observed {observed_receiver}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare_teacher_cache(
    output: Path,
    *,
    device_name: str,
    batch_size: int,
    checkpoint_every: int,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    assert_governed_admission("ddm_wd2_prepare_teacher_cache")
    if batch_size < 1 or checkpoint_every < 1:
        raise WD2BuildError("teacher-cache batch/checkpoint cadence must be positive")
    storage = storage_preflight(output, minimum_free_bytes)
    device = _device(device_name)
    _seed_everything(SEED)
    tokens = _load_tokens()
    teacher = _load_teacher().to(device)
    cache_path = output / "retained/teacher/teacher_master_camera.rgb.u8"
    progress_path = output / "retained/teacher/progress.json"
    expected_bytes = receiver.N * 3 * receiver.CAMERA_H * receiver.CAMERA_W
    binding = {
        "schema": TEACHER_CACHE_SCHEMA,
        "source_archive": EXPECTED["archive"][1],
        "token_cache": EXPECTED["token_cache"][1],
        "teacher_decoder_source": EXPECTED_MZ2_SHA256,
        "receiver_source": sha256_file(Path(receiver.__file__).resolve()),
        "geometry": [receiver.N, 3, receiver.CAMERA_H, receiver.CAMERA_W],
        "dtype": "uint8",
        "seed": SEED,
        "device": device_name,
    }
    binding_sha = receiver.canonical_json_sha256(binding)
    completed = 0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("binding_sha256") != binding_sha:
            raise WD2BuildError("teacher-cache resume binding changed")
        completed = int(progress.get("completed_frames", 0))
        if not cache_path.is_file() or cache_path.stat().st_size != expected_bytes:
            raise WD2BuildError("teacher-cache resume payload is absent or truncated")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as stream:
            stream.truncate(expected_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(
            progress_path,
            {
                "schema": "ddm_wd2_teacher_cache_progress.v1",
                "binding": binding,
                "binding_sha256": binding_sha,
                "completed_frames": 0,
                "complete": False,
            },
        )
    if not 0 <= completed <= receiver.N:
        raise WD2BuildError("teacher-cache completed frame count differs")
    mapped = np.memmap(
        cache_path,
        mode="r+",
        dtype=np.uint8,
        shape=(receiver.N, 3, receiver.CAMERA_H, receiver.CAMERA_W),
    )
    started = time.time()
    with torch.no_grad():
        for start in range(completed, receiver.N, batch_size):
            end = min(start + batch_size, receiver.N)
            indices = torch.arange(start, end, device=device)
            realized = receiver.camera_uint8(
                teacher, tokens[start:end].to(device), indices
            )
            mapped[start:end] = realized.cpu().numpy()
            mapped.flush()
            if end % checkpoint_every == 0 or end == receiver.N:
                stage = {
                    "schema": "ddm_wd2_teacher_cache_stage.v1",
                    "binding": binding,
                    "binding_sha256": binding_sha,
                    "completed_frames": end,
                    "complete": end == receiver.N,
                    "elapsed_s": time.time() - started,
                    "payload_path": str(cache_path.resolve()),
                    "payload_bytes": cache_path.stat().st_size,
                }
                atomic_json(
                    cache_path.parent / f"stage_end_frame_{end:04d}.json", stage
                )
                atomic_json(progress_path, stage)
                print(
                    json.dumps(
                        {
                            "epoch": end,
                            "phase": "teacher_cache",
                            "bpp": 0.0,
                            "top1_error": 0.0,
                            "top1_error_semantics": "teacher_self_identity",
                            "estimated_joint_bytes": CURRENT_ARCHIVE_BYTES,
                            "byte_authority": "PINNED_CURRENT_ARCHIVE",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    del mapped
    payload = file_record(cache_path)
    result = {
        "schema": TEACHER_CACHE_SCHEMA,
        "complete": True,
        "axis": f"[{platform.system()}-{device_name} scorer-free teacher receiver cache]",
        "score_claim": False,
        "storage": storage,
        "binding": binding,
        "binding_sha256": binding_sha,
        "payload": payload,
        "resumable_from_disk": True,
        "all_stage_checkpoints_preserved": True,
    }
    atomic_json(output / "TEACHER_CACHE_RESULT.json", result)
    return result


def _spec_by_id(candidate_id: str) -> receiver.StudentSpec:
    matches = [spec for spec in _candidate_specs() if spec.candidate_id == candidate_id]
    if len(matches) != 1:
        raise WD2BuildError(f"candidate id is not in the derived design: {candidate_id}")
    return matches[0]


def _teacher_cache_memmap(result_path: Path) -> tuple[np.memmap, dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != TEACHER_CACHE_SCHEMA or result.get("complete") is not True:
        raise WD2BuildError("teacher-cache result is incomplete")
    record = result.get("payload")
    path = Path(record["path"])
    if file_record(path) != record:
        raise WD2BuildError("teacher-cache payload differs from its final receipt")
    mapped = np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        shape=(receiver.N, 3, receiver.CAMERA_H, receiver.CAMERA_W),
    )
    return mapped, result


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


class DeploymentEMA:
    """Warm-started EMA whose entire causal state is checkpointed."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.shadow = _cpu_state_dict(model)

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        effective = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, value in model.state_dict().items():
            source = value.detach().cpu()
            if source.dtype.is_floating_point:
                self.shadow[name].mul_(effective).add_(source, alpha=1.0 - effective)
            else:
                self.shadow[name].copy_(source)

    def state(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if float(payload["decay"]) != self.decay:
            raise WD2BuildError("resume EMA decay differs")
        if set(payload["shadow"]) != set(self.shadow):
            raise WD2BuildError("resume EMA state keys differ")
        self.updates = int(payload["updates"])
        self.shadow = {
            name: value.detach().cpu().clone() for name, value in payload["shadow"].items()
        }


@contextlib.contextmanager
def ema_scope(model: nn.Module, ema: DeploymentEMA) -> Iterator[None]:
    live = _cpu_state_dict(model)
    model.load_state_dict(ema.shadow, strict=True)
    try:
        yield
    finally:
        model.load_state_dict(live, strict=True)


def _rng_state(device: torch.device, generator: torch.Generator) -> dict[str, Any]:
    device_state = (
        torch.mps.get_rng_state().cpu()
        if device.type == "mps"
        else torch.random.get_rng_state()
    )
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "device": device_state,
        "generator": generator.get_state(),
    }


def _restore_rng_state(
    payload: Mapping[str, Any], device: torch.device, generator: torch.Generator
) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(payload["device"])
    generator.set_state(payload["generator"])


def _run_identity(args: argparse.Namespace, spec: receiver.StudentSpec, ema_decay: float) -> dict[str, Any]:
    trainer_path = Path(__file__).resolve()
    return {
        "schema": "ddm_wd2_width_distillation_run_identity.v1",
        "trainer": str(trainer_path),
        "trainer_sha256": sha256_file(trainer_path),
        "receiver_sha256": sha256_file(Path(receiver.__file__).resolve()),
        "source_archive_sha256": EXPECTED["archive"][1],
        "token_cache_sha256": EXPECTED["token_cache"][1],
        "teacher_cache_result": file_record(args.teacher_cache_result),
        "spec": spec.as_dict(),
        "config": {
            name: getattr(args, name)
            for name in (
                "epochs",
                "batch_size",
                "accumulation_steps",
                "eval_every",
                "checkpoint_every",
                "lr",
                "weight_decay",
                "grad_clip",
                "seed",
                "device",
            )
        },
        "ema": {
            "decay": ema_decay,
            "derivation": "1 - 2 / total_optimizer_updates",
        },
        "loss": (
            "mean squared error after exact 384x512 semantic renderer, bilinear "
            "resize to 874x1164, clamp, and uint8 straight-through rounding"
        ),
        "quantization": "AWARE: fp16 vectors plus per-row/per-embedding-column signed int4 STE",
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "brotli": getattr(brotli, "__version__", "unknown"),
        },
        "authority": (
            "MPS is training research signal only; CPU packet parse-back is serialization authority"
        ),
    }


def _qat_camera(
    model: receiver.StudentSemanticRenderer,
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    state = receiver.fake_quantize_state(model)
    master = functional_call(model, state, (tokens.long(), indices))
    master = F.interpolate(
        master,
        size=(receiver.CAMERA_H, receiver.CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 255.0)
    rounded = master.round().clamp(0.0, 255.0)
    return master + (rounded - master).detach()


def retain_candidate(
    output: Path,
    candidate_id: str,
    model: receiver.StudentSemanticRenderer,
    rendered_path: Path,
) -> dict[str, Any]:
    packet = receiver.pack_student(model.cpu())
    restored = receiver.unpack_student(packet)
    if receiver.pack_student(restored) != packet:
        raise WD2BuildError("student packet parse-back is not byte-idempotent")
    semantic_stream = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
    hpac, _, carrier = _source_streams()
    model_blob = _pack_rx1_model(hpac, semantic_stream, carrier)
    member = model_blob + SOURCE_RESIDUAL.read_bytes() + SOURCE_TOKEN.read_bytes()
    archive = deterministic_zip(member)
    repeat = deterministic_zip(member)
    if archive != repeat:
        raise WD2BuildError("candidate archive repeat differs")
    root = output / "retained/candidates" / candidate_id
    payloads = {
        "student_packet": atomic_bytes(root / "semantic.wd2s", packet),
        "semantic_brotli_q11": atomic_bytes(root / "semantic.br", semantic_stream),
        "model": atomic_bytes(root / "models.rx1m", model_blob),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", repeat),
        "rendered_master": file_record(rendered_path),
    }
    submission = root / "submission"
    patch = receiver.patch_runtime_tree(SOURCE_RUNTIME, submission)
    binding = receiver.bind_archive(submission, root / "archive.zip")
    parsed = _load_residual_parts(submission, submission / "archive.zip")
    if parsed["semantic_blob"] != packet:
        raise WD2BuildError("patched full-container parse-back did not return the WD2 packet")
    return {
        "candidate_id": candidate_id,
        "payloads": payloads,
        "runtime_patch": patch,
        "archive_binding": binding,
        "student_packet_bytes": len(packet),
        "semantic_stream_bytes": payloads["semantic_brotli_q11"]["bytes"],
        "pool_bytes": payloads["semantic_brotli_q11"]["bytes"]
        + CURRENT_CARRIER_STREAM_BYTES
        + RX1_WRAPPER_BYTES,
        "archive_bytes": len(archive),
        "delta_archive_bytes_vs_current": len(archive) - CURRENT_ARCHIVE_BYTES,
        "rate_only_delta_score": (len(archive) - CURRENT_ARCHIVE_BYTES)
        * 25
        / RATE_DENOMINATOR,
        "repeat_byte_identical": True,
        "receiver_parse_back": True,
        "receiver_parse_back_semantic_sha256": sha256_bytes(parsed["semantic_blob"]),
        "score_claim": False,
    }


@torch.no_grad()
def evaluate_and_retain(
    output: Path,
    *,
    model: receiver.StudentSemanticRenderer,
    ema: DeploymentEMA,
    tokens: torch.Tensor,
    teacher: np.memmap,
    device: torch.device,
    epoch: int,
    eval_batch_size: int,
) -> dict[str, Any]:
    evaluation_path = output / "evaluations" / f"epoch_{epoch:04d}.json"
    if evaluation_path.is_file():
        prior = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if prior.get("epoch") == epoch and prior.get("byte_authority") == "EXACT_RETAINED_RECEIVER_CLOSED_ARCHIVE":
            return prior
        raise WD2BuildError(f"existing evaluation receipt is malformed: {evaluation_path}")
    base_candidate_id = f"{model.spec.candidate_id}_epoch_{epoch:04d}"
    attempts_root = output / "retained/evaluation_attempts" / base_candidate_id
    attempt = 0
    while (attempts_root / f"attempt_{attempt:04d}").exists():
        attempt += 1
    attempt_id = f"attempt_{attempt:04d}"
    candidate_id = f"{base_candidate_id}/{attempt_id}"
    render_path = attempts_root / attempt_id / "student_master.rgb.u8"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = receiver.N * 3 * receiver.CAMERA_H * receiver.CAMERA_W
    with render_path.open("wb") as stream:
        stream.truncate(expected_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    rendered = np.memmap(
        render_path,
        mode="r+",
        dtype=np.uint8,
        shape=(receiver.N, 3, receiver.CAMERA_H, receiver.CAMERA_W),
    )
    squared_error = 0
    unequal = 0
    elements = 0
    maximum_error = 0
    with ema_scope(model, ema):
        model.to(device).eval()
        deployment_packet = receiver.pack_student(model)
        deployment_record = atomic_bytes(
            attempts_root / attempt_id / "deployment_before_render.wd2s",
            deployment_packet,
        )
        deployment = receiver.unpack_student(deployment_packet).to(device).eval()
        for start in range(0, receiver.N, eval_batch_size):
            end = min(start + eval_batch_size, receiver.N)
            indices = torch.arange(start, end, device=device)
            student = receiver.camera_uint8(
                deployment,
                tokens[start:end].to(device),
                indices,
            ).cpu().numpy()
            target = np.asarray(teacher[start:end])
            rendered[start:end] = student
            rendered.flush()
            difference = student.astype(np.int16) - target.astype(np.int16)
            squared_error += int(np.square(difference.astype(np.int32)).sum(dtype=np.int64))
            unequal += int(np.count_nonzero(difference))
            elements += int(difference.size)
            maximum_error = max(maximum_error, int(np.abs(difference).max()))
        model.cpu()
        candidate = retain_candidate(output, candidate_id, model, render_path)
        if candidate["payloads"]["student_packet"]["sha256"] != deployment_record["sha256"]:
            raise WD2BuildError("retained deployment packet changed during evaluation")
    model.to(device)
    del rendered
    prior_top1 = []
    evaluations_root = output / "evaluations"
    if evaluations_root.is_dir():
        for prior_path in sorted(evaluations_root.glob("epoch_*.json")):
            try:
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                prior_epoch = int(prior["epoch"])
                prior_value = float(prior["top1_error"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if prior_epoch < epoch and math.isfinite(prior_value):
                prior_top1.append(prior_value)
    prior_best = min(prior_top1) if prior_top1 else None
    current_top1 = unequal / elements
    if prior_best is None:
        top1_regression_ratio = 1.0
    elif prior_best == 0.0:
        top1_regression_ratio = 1.0 if current_top1 == 0.0 else 2.0
    else:
        top1_regression_ratio = current_top1 / prior_best
    row = {
        "epoch": epoch,
        "phase": "distill_qat",
        "top1_error": current_top1,
        "top1_error_semantics": "teacher_uint8_byte_mismatch_fraction",
        "top1_regression_ratio": top1_regression_ratio,
        "top1_regression_ratio_semantics": "current top1_error divided by best prior retained evaluation; first row is 1",
        "decode_mse_uint8": squared_error / elements,
        "decode_max_abs_uint8": maximum_error,
        "bpp": candidate["student_packet_bytes"] * 8 / (receiver.N * receiver.EVAL_H * receiver.EVAL_W),
        "estimated_joint_bytes": candidate["archive_bytes"],
        "estimated_joint_bytes_semantics": "exact retained archive bytes; key name preserved for fit_hpac_descent_law",
        "byte_authority": "EXACT_RETAINED_RECEIVER_CLOSED_ARCHIVE",
        "evaluated_weights": "ema_shadow_quantized_parse_back",
        "candidate": candidate,
    }
    atomic_json(evaluation_path, row)
    print(json.dumps({key: value for key, value in row.items() if key != "candidate"}, sort_keys=True), flush=True)
    return row


def _checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    ema: DeploymentEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    device: torch.device,
    history: list[dict[str, Any]],
    run_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "stage": "distill_qat",
        "live_state_dict": _cpu_state_dict(model),
        "ema": ema.state(),
        "deployment_weights": "ema_shadow",
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "rng": _rng_state(device, generator),
        "history": history,
        "run_identity": run_identity,
        "run_identity_sha256": receiver.canonical_json_sha256(run_identity),
    }
    return atomic_torch(path, payload)


def train(args: argparse.Namespace) -> dict[str, Any]:
    assert_governed_admission("ddm_wd2_width_distillation_train")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise WD2BuildError("set PYTHONHASHSEED=0 and TAC_ADMISSION_ENFORCE=1")
    if min(
        args.epochs,
        args.batch_size,
        args.accumulation_steps,
        args.eval_every,
        args.checkpoint_every,
    ) < 1:
        raise WD2BuildError("training counts must be positive")
    if args.eval_batch_size < 1 or args.lr <= 0 or args.grad_clip <= 0:
        raise WD2BuildError("eval batch, learning rate, and grad clip must be positive")
    if args.batch_size > receiver.N:
        raise WD2BuildError("training batch size cannot exceed the n600 population")
    total_batches = math.ceil(receiver.N / args.batch_size)
    if total_batches % args.accumulation_steps:
        raise WD2BuildError(
            "batch count must divide accumulation steps exactly; partial gradient groups "
            "would change the declared effective-batch law"
        )
    if args.resume_from is not None and not args.resume_from.is_file():
        raise WD2BuildError(f"resume checkpoint is absent: {args.resume_from}")
    storage = storage_preflight(args.output, args.min_free_bytes)
    device = _device(args.device)
    _seed_everything(args.seed)
    spec = _spec_by_id(args.candidate_id)
    tokens = _load_tokens()
    teacher_cache, teacher_result = _teacher_cache_memmap(args.teacher_cache_result)
    model = receiver.StudentSemanticRenderer(spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
    )
    optimizer_updates = args.epochs * (total_batches // args.accumulation_steps)
    if optimizer_updates <= 2:
        raise WD2BuildError("run geometry is too short to derive EMA")
    ema_decay = 1.0 - 2.0 / optimizer_updates
    ema = DeploymentEMA(model, ema_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    run_identity = _run_identity(args, spec, ema_decay)
    history: list[dict[str, Any]] = []
    start_epoch = 0
    checkpoint_root = args.output / "checkpoints" / spec.candidate_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume.get("schema") != CHECKPOINT_SCHEMA:
            raise WD2BuildError("resume checkpoint schema differs")
        if resume.get("run_identity") != run_identity:
            raise WD2BuildError("resume run identity differs; source/config drift refused")
        model.load_state_dict(resume["live_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        ema.restore(resume["ema"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"])
        _restore_rng_state(resume["rng"], device, generator)
        preserved = checkpoint_root / "resume_parents" / f"{sha256_file(args.resume_from)}.pt"
        preserved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.resume_from, preserved)
        if sha256_file(preserved) != sha256_file(args.resume_from):
            raise WD2BuildError("resume parent preservation differs")

    # Real-shape memory preflight: one exact QAT forward/backward, no optimizer step.
    probe_tokens = tokens[: args.batch_size].to(device)
    probe_indices = torch.arange(args.batch_size, device=device)
    probe_target = torch.from_numpy(np.asarray(teacher_cache[: args.batch_size]).copy()).to(
        device=device, dtype=torch.float32
    )
    optimizer.zero_grad(set_to_none=True)
    probe_output = _qat_camera(model, probe_tokens, probe_indices)
    probe_loss = F.mse_loss(probe_output / 255.0, probe_target / 255.0)
    probe_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    probe_payload = probe_output.detach().round().to(torch.uint8).cpu().numpy().tobytes()
    probe_record = atomic_bytes(
        args.output / "retained/memory_preflight/qat_probe_master.rgb.u8", probe_payload
    )
    probe_packet_record = atomic_bytes(
        args.output / "retained/memory_preflight/student_before_training.wd2s",
        receiver.pack_student(model),
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = peak_rss if platform.system() == "Darwin" else peak_rss * 1024
    memory_receipt = {
        "schema": "ddm_wd2_real_shape_memory_preflight.v1",
        "device": args.device,
        "batch_size": args.batch_size,
        "candidate": spec.as_dict(),
        "loss_finite": math.isfinite(float(probe_loss.detach().cpu())),
        "peak_process_rss_bytes": peak_rss_bytes,
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory()) if device.type == "mps" else None
        ),
        "retained_probe_output": probe_record,
        "retained_student_packet": probe_packet_record,
        "optimizer_step_taken": False,
        "status": "PASS",
    }
    atomic_json(args.output / "MEMORY_PREFLIGHT.json", memory_receipt)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(receiver.N, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        epoch_loss = 0.0
        batches = 0
        for offset in range(0, receiver.N, args.batch_size):
            batch_indices_cpu = permutation[offset : offset + args.batch_size]
            batch_indices = batch_indices_cpu.to(device)
            batch_tokens = tokens[batch_indices_cpu].to(device)
            target = torch.from_numpy(
                np.asarray(teacher_cache[batch_indices_cpu.numpy()]).copy()
            ).to(device=device, dtype=torch.float32)
            student = _qat_camera(model, batch_tokens, batch_indices)
            loss = F.mse_loss(student / 255.0, target / 255.0)
            (loss / args.accumulation_steps).backward()
            pending += 1
            batches += 1
            epoch_loss += float(loss.detach().cpu())
            final_batch = offset + args.batch_size >= receiver.N
            if pending == args.accumulation_steps or final_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                pending = 0
        scheduler.step()
        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "phase": "distill_qat",
            "train_decode_mse_normalized": epoch_loss / batches,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        should_eval = epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            evaluation = evaluate_and_retain(
                args.output,
                model=model,
                ema=ema,
                tokens=tokens,
                teacher=teacher_cache,
                device=device,
                epoch=epoch,
                eval_batch_size=args.eval_batch_size,
            )
            epoch_row.update(
                {
                    key: evaluation[key]
                    for key in (
                        "top1_error",
                        "decode_mse_uint8",
                        "estimated_joint_bytes",
                    )
                }
            )
        history.append(epoch_row)
        should_checkpoint = (
            epoch % args.checkpoint_every == 0 or should_eval or epoch == args.epochs
        )
        if should_checkpoint:
            _checkpoint(
                checkpoint_root / f"distill_qat_epoch_{epoch:04d}.pt",
                epoch=epoch,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                device=device,
                history=history,
                run_identity=run_identity,
            )
    _checkpoint(
        checkpoint_root / f"distill_qat_stage_end_epoch_{args.epochs:04d}.pt",
        epoch=args.epochs,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        device=device,
        history=history,
        run_identity=run_identity,
    )
    del teacher_cache
    result = {
        "schema": "ddm_wd2_width_distillation_train.v1",
        "complete": True,
        "score_claim": False,
        "axis": f"[macOS-{args.device} training research signal; scorer-free]",
        "storage": storage,
        "run_identity": run_identity,
        "teacher_cache": teacher_result,
        "history": history,
        "memory_preflight": file_record(args.output / "MEMORY_PREFLIGHT.json"),
        "resumable_from_disk": True,
        "per_stage_checkpoint": file_record(
            checkpoint_root / f"distill_qat_stage_end_epoch_{args.epochs:04d}.pt"
        ),
        "all_evaluated_payloads_retained": True,
        "scorer_invocations": 0,
    }
    atomic_json(args.output / "TRAIN_RESULT.json", result)
    return result


def memory_probe(
    output: Path,
    *,
    candidate_id: str,
    device_name: str,
    minimum_free_bytes: int,
) -> dict[str, Any]:
    """Run the exact full-resolution graph once without changing weights."""

    storage = storage_preflight(output, minimum_free_bytes)
    _seed_everything(SEED)
    device = _device(device_name)
    tokens = _load_tokens()
    teacher = _load_teacher().to(device)
    model = receiver.StudentSemanticRenderer(_spec_by_id(candidate_id)).to(device)
    index = torch.tensor([0], device=device)
    with torch.no_grad():
        teacher_output = receiver.camera_uint8(teacher, tokens[:1].to(device), index)
    student_output = _qat_camera(model, tokens[:1].to(device), index)
    loss = F.mse_loss(student_output / 255.0, teacher_output.float() / 255.0)
    loss.backward()
    root = output / "retained/memory_probe"
    payloads = {
        "teacher": atomic_bytes(
            root / "teacher_pair0000.rgb.u8", teacher_output.cpu().numpy().tobytes()
        ),
        "student": atomic_bytes(
            root / "student_pair0000.rgb.u8",
            student_output.detach().round().to(torch.uint8).cpu().numpy().tobytes(),
        ),
        "student_packet": atomic_bytes(root / "student_untrained.wd2s", receiver.pack_student(model.cpu())),
    }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    result = {
        "schema": "ddm_wd2_memory_probe.v1",
        "complete": True,
        "axis": f"[macOS-{device_name} real-shape apparatus; no optimizer step]",
        "score_claim": False,
        "candidate": candidate_id,
        "batch_size": 1,
        "payload_layout": "NCHW uint8 [1,3,874,1164]",
        "peak_process_rss_bytes": peak_bytes,
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory()) if device.type == "mps" else None
        ),
        "loss_finite": math.isfinite(float(loss.detach().cpu())),
        "payloads": payloads,
        "builder": file_record(Path(__file__).resolve()),
        "receiver": file_record(Path(receiver.__file__).resolve()),
        "storage": storage,
        "training_launched": False,
        "optimizer_steps": 0,
    }
    atomic_json(output / "MEMORY_PROBE_RESULT.json", result)
    return result


def inventory(output: Path) -> dict[str, Any]:
    retained = output / "retained"
    files = [file_record(path) for path in sorted(retained.rglob("*")) if path.is_file()]
    value = {
        "schema": "ddm_wd2_retention_inventory.v1",
        "root": str(retained.resolve()),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
    }
    atomic_json(output / "RETENTION_INVENTORY.json", value)
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-build")
    verify.add_argument("--output", type=Path, default=RETENTION_ROOT)
    verify.add_argument("--min-free-bytes", type=int, default=1 << 30)

    cache = sub.add_parser("prepare-teacher-cache")
    cache.add_argument("--output", type=Path, default=RETENTION_ROOT)
    cache.add_argument("--device", choices=("cpu", "mps"), required=True)
    cache.add_argument("--batch-size", type=int, required=True)
    cache.add_argument("--checkpoint-every", type=int, required=True)
    cache.add_argument("--min-free-bytes", type=int, required=True)
    cache.add_argument("--expected-builder-sha256", required=True)
    cache.add_argument("--expected-receiver-sha256", required=True)

    probe = sub.add_parser("memory-probe")
    probe.add_argument("--output", type=Path, default=RETENTION_ROOT)
    probe.add_argument("--candidate-id", required=True)
    probe.add_argument("--device", choices=("cpu", "mps"), required=True)
    probe.add_argument("--min-free-bytes", type=int, required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--teacher-cache-result", type=Path, required=True)
    train_parser.add_argument("--candidate-id", required=True)
    train_parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    train_parser.add_argument("--epochs", type=int, required=True)
    train_parser.add_argument("--batch-size", type=int, required=True)
    train_parser.add_argument("--eval-batch-size", type=int, required=True)
    train_parser.add_argument("--accumulation-steps", type=int, required=True)
    train_parser.add_argument("--eval-every", type=int, required=True)
    train_parser.add_argument("--checkpoint-every", type=int, required=True)
    train_parser.add_argument("--lr", type=float, required=True)
    train_parser.add_argument("--weight-decay", type=float, required=True)
    train_parser.add_argument("--grad-clip", type=float, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--min-free-bytes", type=int, required=True)
    train_parser.add_argument("--resume-from", type=Path)
    train_parser.add_argument("--expected-builder-sha256", required=True)
    train_parser.add_argument("--expected-receiver-sha256", required=True)

    inv = sub.add_parser("inventory")
    inv.add_argument("--output", type=Path, default=RETENTION_ROOT)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "verify-build":
        result = verify_build(args.output, args.min_free_bytes)
    elif args.command == "prepare-teacher-cache":
        _verify_launch_source_pins(
            args.expected_builder_sha256, args.expected_receiver_sha256
        )
        result = prepare_teacher_cache(
            args.output,
            device_name=args.device,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            minimum_free_bytes=args.min_free_bytes,
        )
    elif args.command == "memory-probe":
        result = memory_probe(
            args.output,
            candidate_id=args.candidate_id,
            device_name=args.device,
            minimum_free_bytes=args.min_free_bytes,
        )
    elif args.command == "train":
        _verify_launch_source_pins(
            args.expected_builder_sha256, args.expected_receiver_sha256
        )
        result = train(args)
    else:
        result = inventory(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "history"}}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
