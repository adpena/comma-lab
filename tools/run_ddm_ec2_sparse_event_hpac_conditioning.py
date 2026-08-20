#!/usr/bin/env python3
"""Run the scorer-free EC2 sparse-event-conditioned HPAC falsifier.

EC2 is a thin subclass runner over the pinned CL1/XI1 integer HPAC substrate.
It preserves CL1's learned previous-partition prior and adds one genuinely local
learned convolution over the counted sparse EC1 coordinate mask.  No scorer is
imported or executed.  The terminal verdict is the byte size of the exact,
receiver-consumed container containing model.xz, Range tokens, and coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import lzma
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

import brotli
import constriction
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.admission_guard import assert_governed_admission  # noqa: E402

OUTPUT = Path("/Volumes/VertigoDataTier/pact/ddm_ec2")
RETAINED = OUTPUT / "retained"
INPUTS = RETAINED / "inputs"
CONTEXTS = RETAINED / "contexts"
TRAINING = RETAINED / "training"
SERIALIZED = RETAINED / "serialized"
QUEUE = OUTPUT / "queue"
STATE = OUTPUT / "state.json"
BUILD_RECEIPT = OUTPUT / "BUILD_RECEIPT.json"
READY_TO_FIRE = OUTPUT / "READY_TO_FIRE.json"
FULL_SCALE_RESULT = OUTPUT / "FULL_SCALE_RESULT.json"

XI1_PATH = ROOT / "tools/run_ddm_xi1_screw_conditioned_learned_prior.py"
CL1_PATH = ROOT / "tools/train_ddm_cl1_hpac_capacity.py"
EC1_PATH = ROOT / "experiments/ddm_ec1_event_coordinate_producer.py"
JS5_STORE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200"
)
JS5_STATE = JS5_STORE / "state.json"
JS5_INDEX = JS5_STORE / "proposal_index.jsonl"
CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt")  # GT_LINEAGE_OK: bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
INITIALIZER = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt")
CONTROL_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/lambda_1p0_uninterrupted_twin/training")
CONTROL_RANGE = CONTROL_ROOT / "serialized/terminal.range.bin"
CONTROL_RAW = CONTROL_ROOT / "serialized/terminal.raw.u8"
CONTROL_MODEL = CONTROL_ROOT / "serialized/terminal.model.bin.xz"
CONTROL_REPORT = CONTROL_ROOT / "reports/trainer.json"
XI2_SAFE_RUN = Path("/Volumes/APDataStore/pact/ddm_xi2_20260812/run/main_r2.safe_run.json")

SCHEMA = "ddm_ec2_sparse_event_hpac_conditioning.v1"
CHECKPOINT_SCHEMA = "ddm_ec2_sparse_event_hpac_checkpoint.v1"
AXIS = "[macOS-MPS research-signal training; real Range bytes; scorer-free]"
SEED = 20260716
FRAME_COUNT = 600
H, W, CLASSES = 384, 512, 5
PIXELS = FRAME_COUNT * H * W
EPOCHS = 60
RATE_LAMBDA = 1.0
CONTROL_CONTAINER_BAR_BYTES = 116_716
MAX_PASSING_CONTAINER_BYTES = CONTROL_CONTAINER_BAR_BYTES - 1
MEASURED_XI2_PEAK_RSS_MIB = 4_879.953
SAFE_RUN_RSS_MIB = 6_144
SAFE_RUN_PROJECTED_GIB = 6
COORD_MAGIC = b"EC2COORD1"
PACKAGE_MAGIC = b"EC2PKG1\0"
PACKAGE_HEADER = struct.Struct("<8sIII")
COORD_HEADER = struct.Struct("<HHII")
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

TRAIN_CONFIG = {
    "epochs": EPOCHS,
    "batch_size": 8,
    "eval_batch_size": 4,
    "eval_every": 2,
    "lr": 0.003,
    "lr_exponent": 0.0002,
    "lr_bits": 0.01,
    "bit_eps": 1e-6,
    "rate_lambda": RATE_LAMBDA,
    "qat_fraction": 0.5,
    "init_bits": 8.0,
    "channels": 64,
    "patch": 64,
    "delta": 2,
    "frame_dim": 8,
    "norm_mode": "none",
    "activation": "relu",
    "frame_scale": True,
    "weight_bound": 127,
    "activation_bound": 127,
    "weight_scales": True,
    "weight_exponent_min": -6,
    "spm": True,
    "norm_gates": False,
    "target_mode": "raw",
    "seed": SEED,
    "ema_target_seed_fraction": 0.01,
    "device": "mps",
    "context_mode": "previous_decoded_partition_plus_counted_sparse_event_coordinates",
    "event_channels": 1,
    "event_kernel": 3,
}

EXPECTED_SHA256 = {
    XI1_PATH: "f8ddbd9bb9479950148364d504484bc2fc278150b2eb48c3789a431c42d78882",
    CL1_PATH: "0c1e6464173d61c5a585450310977c13822ea662bf0bf9b59548491209f3d423",
    EC1_PATH: "c927106ef695cd7f162be8ee692e54b2178ff08ee0e8a3d55c9ea6bbbba85a78",
    JS5_STATE: "7234e715c3f3b5bd434be5709a1cdb2bc2975ea1f4c812d45578b8a92b5f724c",
    JS5_INDEX: "599a3ac0a9c7d7e62c162fcee595194d6d3cd79685d0ceabab92e0231bd9d47e",
    CACHE: "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195",
    INITIALIZER: "0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985",
    CONTROL_RANGE: "ac2c549c1f48756ad33c6c99af8563f2170db1de61cd50d0615d4c1a0cdd7b87",
    CONTROL_RAW: "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece",
    CONTROL_MODEL: "b74be4d5f4c8f7f1aec37577f2277d43ca44ef6d53e5b0138a8ce5e7d7e02325",
    CONTROL_REPORT: "9382d0736a3e001a342f180d7c04cbab0ff0efc7bf3fcf90c642270cdf3a9d7a",
    XI2_SAFE_RUN: "ee06cff3911b39ca708ba367eab787fb3a366e3ff0e926bbdc1f55f0b3a24784",
}

EXPECTED_INTAKE_SHA256 = {
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_integer_sparse.py": "2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c",
    "hpac_self_compress.py": "d63d67945a0719ebbe72da36e6c99909557219360d50e74f20577d68d678beec",
    "pack_hpac_self_compress.py": "e796d9249926f8c7dcc45a7cdf1f39e33d0b4409ffee275fbb9cd481a6f5f099",
}


class EC2Error(RuntimeError):
    """Fail-closed EC2 custody, resume, codec, or admission error."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EC2Error(f"required artifact is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise EC2Error(f"refusing to overwrite a different retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_record(path)


def retain_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist a materialized payload immediately and return its custody row."""

    return _atomic_bytes(path, payload)


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _atomic_bytes(path, payload, replace=replace)


def atomic_torch(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return _atomic_bytes(path, buffer.getvalue(), replace=replace)


def atomic_numpy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return retain_payload(path, buffer.getvalue())


def import_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EC2Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_xi1() -> ModuleType:
    if sha256_file(XI1_PATH) != EXPECTED_SHA256[XI1_PATH]:
        raise EC2Error("XI1 source pin changed")
    return import_path(XI1_PATH, "ddm_ec2_pinned_xi1")


def pin_inputs(xi1: ModuleType) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for path, expected in EXPECTED_SHA256.items():
        observed = file_record(path)
        if observed["sha256"] != expected:
            raise EC2Error(f"input pin changed for {path}: {observed['sha256']}")
        records[str(path)] = observed
    for name, expected in EXPECTED_INTAKE_SHA256.items():
        path = xi1.INTAKE_CODE / name
        observed = file_record(path)
        if observed["sha256"] != expected:
            raise EC2Error(f"PR130 source pin changed for {name}: {observed['sha256']}")
        records[str(path)] = observed
    state = json.loads(JS5_STATE.read_text(encoding="utf-8"))
    if (
        state.get("schema") != "ddm_js5_realized_acceptance_200_store.v1"
        or state.get("proposal_count") != 200
        or state.get("receiver_effective_count") != 200
        or state.get("acceptance_tested") is not False
    ):
        raise EC2Error("JS5 200-proposal store state changed")
    report = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    expected_control = {
        key: value
        for key, value in TRAIN_CONFIG.items()
        if key not in {"context_mode", "event_channels", "event_kernel"}
    }
    if report.get("run_identity", {}).get("training_config") != expected_control:
        raise EC2Error("banked CL1 control configuration changed")
    if CONTROL_RANGE.stat().st_size != CONTROL_CONTAINER_BAR_BYTES:
        raise EC2Error("banked CL1 Range byte count changed")
    if CONTROL_RAW.stat().st_size != PIXELS:
        raise EC2Error("banked CL1 raw-token geometry changed")
    return records


def storage_preflight(required_free_bytes: int = 8 << 30) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(OUTPUT)
    if usage.free < required_free_bytes:
        raise EC2Error(f"VertigoDataTier needs {required_free_bytes} free bytes; found {usage.free}")
    return {
        "path": str(OUTPUT),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": usage.free,
        "status": "PASS",
    }


def put_uvarint(output: bytearray, value: int) -> None:
    if value < 0:
        raise EC2Error("negative uvarint")
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)


def get_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload) and shift <= 63:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise EC2Error("invalid or truncated uvarint")


def decode_ec1_proposal(payload: bytes) -> tuple[int, int, int, int, np.ndarray]:
    if len(payload) < 13 or payload[:8] != b"EC1PROP1":
        raise EC2Error("EC1 proposal header differs")
    frame, source_class, target_class, event_type = struct.unpack_from("<HBBB", payload, 8)
    if frame >= FRAME_COUNT or source_class >= CLASSES or target_class >= CLASSES:
        raise EC2Error("EC1 proposal address differs")
    count, offset = get_uvarint(payload, 13)
    indices = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        gap, offset = get_uvarint(payload, offset)
        value = gap if position == 0 else previous + gap
        if value >= H * W or (position and value <= previous):
            raise EC2Error("EC1 proposal coordinates differ")
        indices[position] = value
        previous = value
    if offset != len(payload):
        raise EC2Error("EC1 proposal has trailing bytes")
    return frame, source_class, target_class, event_type, indices


def encode_coordinate_canonical(global_indices: np.ndarray) -> bytes:
    indices = np.asarray(global_indices, dtype=np.int64)
    if indices.ndim != 1 or (len(indices) and (np.diff(indices) <= 0).any()):
        raise EC2Error("coordinate indices must be a strictly increasing vector")
    output = bytearray(COORD_MAGIC)
    output.extend(COORD_HEADER.pack(H, W, FRAME_COUNT, len(indices)))
    previous = 0
    for position, index in enumerate(indices.tolist()):
        if index < 0 or index >= PIXELS:
            raise EC2Error("coordinate index is outside n600 geometry")
        put_uvarint(output, index if position == 0 else index - previous)
        previous = index
    return bytes(output)


def decode_coordinate_canonical(payload: bytes) -> np.ndarray:
    if not payload.startswith(COORD_MAGIC):
        raise EC2Error("coordinate canonical magic differs")
    offset = len(COORD_MAGIC)
    if len(payload) < offset + COORD_HEADER.size:
        raise EC2Error("coordinate canonical header is truncated")
    height, width, frames, count = COORD_HEADER.unpack_from(payload, offset)
    offset += COORD_HEADER.size
    if (height, width, frames) != (H, W, FRAME_COUNT):
        raise EC2Error("coordinate canonical geometry differs")
    indices = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        gap, offset = get_uvarint(payload, offset)
        value = gap if position == 0 else previous + gap
        if value >= PIXELS or (position and value <= previous):
            raise EC2Error("coordinate canonical deltas differ")
        indices[position] = value
        previous = value
    if offset != len(payload):
        raise EC2Error("coordinate canonical payload has trailing bytes")
    return indices


def frame_coordinate_payload(canonical: bytes, coder: str) -> bytes:
    if coder == "raw":
        return b"R" + canonical
    if coder == "brotli_q11":
        return b"B" + brotli.compress(canonical, quality=11, mode=brotli.MODE_GENERIC)
    if coder == "lzma_xz":
        return b"X" + lzma.compress(canonical, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    raise EC2Error(f"unknown coordinate coder: {coder}")


def unframe_coordinate_payload(payload: bytes) -> np.ndarray:
    if not payload:
        raise EC2Error("coordinate payload is empty")
    if payload[:1] == b"R":
        canonical = payload[1:]
    elif payload[:1] == b"B":
        canonical = brotli.decompress(payload[1:])
    elif payload[:1] == b"X":
        canonical = lzma.decompress(payload[1:], format=lzma.FORMAT_XZ)
    else:
        raise EC2Error("coordinate coder tag differs")
    return decode_coordinate_canonical(canonical)


def event_mask_from_indices(indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(PIXELS, dtype=np.uint8)
    mask[np.asarray(indices, dtype=np.int64)] = 1
    return mask.reshape(FRAME_COUNT, H, W)


def build_package(model_xz: bytes, range_payload: bytes, coordinates: bytes) -> bytes:
    return (
        PACKAGE_HEADER.pack(PACKAGE_MAGIC, len(model_xz), len(range_payload), len(coordinates))
        + model_xz
        + range_payload
        + coordinates
    )


def parse_package(payload: bytes) -> tuple[bytes, bytes, bytes]:
    if len(payload) < PACKAGE_HEADER.size:
        raise EC2Error("EC2 package is truncated")
    magic, model_bytes, range_bytes, coordinate_bytes = PACKAGE_HEADER.unpack_from(payload)
    if magic != PACKAGE_MAGIC:
        raise EC2Error("EC2 package magic differs")
    expected = PACKAGE_HEADER.size + model_bytes + range_bytes + coordinate_bytes
    if expected != len(payload) or not model_bytes or not range_bytes or not coordinate_bytes:
        raise EC2Error("EC2 package section lengths differ")
    offset = PACKAGE_HEADER.size
    model = payload[offset : offset + model_bytes]
    offset += model_bytes
    tokens = payload[offset : offset + range_bytes]
    offset += range_bytes
    return model, tokens, payload[offset:]


def _safe_store_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(JS5_STORE.resolve())
    except ValueError as exc:
        raise EC2Error(f"JS5 consumer path escaped its store: {path}") from exc
    for part in relative.parts:
        if part.startswith("._") or part == "__MACOSX" or unicodedata.normalize("NFC", part) != part:
            raise EC2Error(f"unsafe AppleDouble or non-NFC JS5 path: {relative}")
    return resolved


def prepare_coordinate_context() -> dict[str, Any]:
    lines = [json.loads(line) for line in JS5_INDEX.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 200:
        raise EC2Error(f"expected 200 JS5 proposal rows, found {len(lines)}")
    all_global: list[int] = []
    custody_rows: list[dict[str, Any]] = []
    for position, row in enumerate(lines):
        if (
            row.get("schema") != "ddm_ec1_receiver_proposal_attempt.v1"
            or row.get("receiver_effective") is not True
            or row.get("parse_back_exact") is not True
            or row.get("acceptance_tested") is not False
        ):
            raise EC2Error(f"JS5 proposal row {position} changed its scorer-free custody status")
        event_expected = row["consumer_payloads"]["event.ec1p"]
        event_path = _safe_store_path(Path(event_expected["path"]))
        event_record = file_record(event_path)
        if event_record["bytes"] != event_expected["bytes"] or event_record["sha256"] != event_expected["sha256"]:
            raise EC2Error(f"JS5 event custody changed at row {position}")
        receipt_expected = row["proposal_receipt"]
        receipt_path = _safe_store_path(Path(receipt_expected["path"]))
        receipt_record = file_record(receipt_path)
        if (
            receipt_record["bytes"] != receipt_expected["bytes"]
            or receipt_record["sha256"] != receipt_expected["sha256"]
        ):
            raise EC2Error(f"JS5 proposal receipt custody changed at row {position}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("proposal_id") != row.get("proposal_id"):
            raise EC2Error(f"JS5 proposal receipt identity changed at row {position}")
        frame, source, target, event_type, local = decode_ec1_proposal(event_path.read_bytes())
        if frame != row["pair"] or len(local) != row["site_count"]:
            raise EC2Error(f"JS5 event parse-back differs from index row {position}")
        all_global.extend((frame * H * W + local).tolist())
        custody_rows.append(
            {
                "ordinal": position,
                "proposal_id": row["proposal_id"],
                "frame": frame,
                "source_class_id": source,
                "target_class_id": target,
                "event_type_id": event_type,
                "site_count": len(local),
                "event": event_record,
                "proposal_receipt": receipt_record,
            }
        )
    unique = np.unique(np.asarray(all_global, dtype=np.int64))
    canonical = encode_coordinate_canonical(unique)
    if not np.array_equal(decode_coordinate_canonical(canonical), unique):
        raise EC2Error("canonical coordinate codec failed exact round trip")
    canonical_record = retain_payload(CONTEXTS / "coordinates.canonical.bin", canonical)
    candidate_rows: list[dict[str, Any]] = []
    suffix = {"raw": "raw", "brotli_q11": "br", "lzma_xz": "xz"}
    for coder in ("raw", "brotli_q11", "lzma_xz"):
        payload = frame_coordinate_payload(canonical, coder)
        primary = retain_payload(CONTEXTS / f"coordinates.{suffix[coder]}.ec2c", payload)
        repeat = retain_payload(
            CONTEXTS / f"coordinates.{suffix[coder]}.repeat.ec2c", frame_coordinate_payload(canonical, coder)
        )
        if primary["sha256"] != repeat["sha256"] or not np.array_equal(unframe_coordinate_payload(payload), unique):
            raise EC2Error(f"coordinate coder {coder} is not deterministic and exact")
        candidate_rows.append({"coder": coder, "payload": primary, "repeat": repeat, "roundtrip_exact": True})
    winner = min(candidate_rows, key=lambda row: (row["payload"]["bytes"], row["coder"]))
    event_mask = event_mask_from_indices(unique)
    mask_record = atomic_numpy(CONTEXTS / "event_mask_n600.uint8.npy", event_mask)
    manifest = {
        "schema": "ddm_ec2_js5_coordinate_custody.v1",
        "source_store": str(JS5_STORE),
        "proposal_rows_verified": len(custody_rows),
        "proposal_sites_with_multiplicity": len(all_global),
        "unique_coordinate_sites": len(unique),
        "duplicate_sites_collapsed": len(all_global) - len(unique),
        "rows": custody_rows,
        "canonical": canonical_record,
        "coordinate_coder_race": candidate_rows,
        "winner": winner,
        "training_mask": mask_record,
        "path_hygiene": "all consumed paths confined to JS5 store, NFC UTF-8, no AppleDouble/__MACOSX components",
        "context_legality": "video-derived EC1 coordinates are counted in every candidate complete container",
        "acceptance_claim": False,
        "score_claim": False,
    }
    atomic_json(CONTEXTS / "coordinate_custody_manifest.json", manifest)
    return manifest


def sparse_event_model_class(integer: ModuleType) -> type[torch.nn.Module]:
    class SparseEventIntegerHPAC(integer.IntegerHPAC):
        """CL1 HPAC plus one counted sparse local event-coordinate channel."""

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.conv_event = integer.IntegerConv2d(
                1,
                self.ch,
                3,
                padding=1,
                weight_bound=self.weight_bound,
                use_weight_scales=self.use_weight_scales,
                exponent_min=-6,
            )
            torch.nn.init.zeros_(self.conv_event.weight)
            torch.nn.init.zeros_(self.conv_event.bias)

        def prepare_frame_context(
            self,
            idx: torch.Tensor,
            previous_raw: torch.Tensor,
            event_mask: torch.Tensor,
        ) -> tuple[Any, ...]:
            shift, past, scale, spm = super().prepare_frame_context(idx, previous_raw)
            local = integer.requantize(
                self.conv_event(event_mask.unsqueeze(1).float()),
                0,
                -self.activation_bound,
                self.activation_bound,
            )
            past = integer.requantize(
                past + self._to_patches(local),
                0,
                -self.activation_bound,
                self.activation_bound,
            )
            return shift, past, scale, spm

        def forward(
            self,
            current: torch.Tensor,
            idx: torch.Tensor,
            previous_raw: torch.Tensor,
            event_mask: torch.Tensor,
        ) -> torch.Tensor:
            context = self.prepare_frame_context(idx, previous_raw, event_mask)
            return self.cached_context_logits(current, context)

    return SparseEventIntegerHPAC


def make_model(
    integer: ModuleType,
    compression: ModuleType,
    device: torch.device,
    *,
    self_compressed: bool,
    initialize: bool,
) -> torch.nn.Module:
    model_type = sparse_event_model_class(integer)
    model = model_type(
        channels=64,
        patch=64,
        delta=2,
        frame_dim=8,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    ).to(device)
    if self_compressed:
        compression.enable_self_compression(model, 8.0)
    if initialize:
        initial = torch.load(INITIALIZER, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(initial["state_dict"], strict=False)
        allowed_missing = {
            name for name in model.state_dict() if name.startswith("conv_event.") or name.endswith(".bit_depth")
        }
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != allowed_missing:
            raise EC2Error(f"HPAC initializer is incompatible with EC2: {incompatible}")
    return model


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    parameters = dict(model.named_parameters())
    bit_names = {name for name in parameters if name.endswith(".bit_depth")}
    exponent_names = {name for name in parameters if name.endswith(".exponent")}
    other_names = set(parameters) - bit_names - exponent_names
    if not bit_names or "conv_event.bit_depth" not in bit_names:
        raise EC2Error("EC2 learned bit-depth schema omits the local event channel")
    groups: list[dict[str, Any]] = [
        {"params": [parameters[name] for name in sorted(other_names)], "lr": 0.003, "eps": 1e-8},
        {
            "params": [parameters[name] for name in sorted(bit_names)],
            "lr": 0.01,
            "eps": 1e-6,
            "weight_decay": 0.0,
        },
    ]
    if exponent_names:
        groups.append({"params": [parameters[name] for name in sorted(exponent_names)], "lr": 0.0002, "eps": 1e-8})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _json_without_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": sha256_bytes(tensor.numpy().tobytes()),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": sha256_bytes(array.tobytes())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_without_tensors(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_without_tensors(item) for item in value]
    return value


def _checkpoint_digest(payload: dict[str, Any]) -> str:
    causal = {key: value for key, value in payload.items() if key not in {"causal_state_sha256", "resume_lineage"}}
    encoded = json.dumps(_json_without_tensors(causal), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    ema: Any,
    ema_policy: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    history: list[dict[str, Any]],
    source_identity: dict[str, Any],
    resume_lineage: list[dict[str, Any]],
) -> dict[str, Any]:
    phase = "initial" if epoch == 0 else ("continuous" if epoch <= EPOCHS // 2 else "discrete_qat")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "phase": phase,
        "train_config": TRAIN_CONFIG,
        "source_identity": source_identity,
        "live_state_dict": _cpu_tree(model.state_dict()),
        "ema_shadow": _cpu_tree(ema.state_dict()),
        "ema_policy": ema_policy,
        "ema_decay": ema.decay,
        "ema_updates": ema._num_updates,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "shuffle_generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "mps_rng_state": torch.mps.get_rng_state().cpu(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "history": history,
        "resume_lineage": resume_lineage,
        "deployment_weights": "terminal ema_shadow",
    }
    payload["causal_state_sha256"] = _checkpoint_digest(payload)
    return payload


def _periodic_paths() -> list[Path]:
    return sorted((TRAINING / "checkpoints/periodic").glob("epoch_*.pt"))


def resolve_resume(value: str) -> Path | None:
    if value == "auto":
        paths = _periodic_paths()
        return paths[-1] if paths else None
    path = Path(value)
    if not path.is_file():
        raise EC2Error(f"--resume-from is absent: {path}")
    try:
        path.resolve().relative_to(OUTPUT.resolve())
    except ValueError as exc:
        raise EC2Error("--resume-from must remain under the EC2 custody root") from exc
    return path


def _restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    source_identity: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise EC2Error("resume checkpoint schema changed")
    if checkpoint.get("causal_state_sha256") != _checkpoint_digest(checkpoint):
        raise EC2Error("resume checkpoint causal state hash is absent or does not verify")
    if checkpoint.get("train_config") != TRAIN_CONFIG or checkpoint.get("source_identity") != source_identity:
        raise EC2Error("resume checkpoint config/input/source identity changed")
    model.load_state_dict(checkpoint["live_state_dict"], strict=True)
    ema.shadow = {name: tensor.to(next(model.parameters()).device) for name, tensor in checkpoint["ema_shadow"].items()}
    ema._num_updates = int(checkpoint["ema_updates"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    generator.set_state(checkpoint["shuffle_generator_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"])
    torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint["epoch"]), list(checkpoint["history"]), list(checkpoint.get("resume_lineage", []))


def _source_identity(pins: dict[str, Any]) -> dict[str, Any]:
    return {
        "runner": file_record(Path(__file__).resolve()),
        "pinned_inputs": pins,
        "coordinate_payload": file_record(CONTEXTS / "coordinates.selected.ec2c"),
        "event_mask": file_record(CONTEXTS / "event_mask_n600.uint8.npy"),
        "train_config": TRAIN_CONFIG,
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    ema: Any,
    compression: ModuleType,
    target: torch.Tensor,
    previous: torch.Tensor,
    event_mask: torch.Tensor,
    ids: torch.Tensor,
) -> dict[str, Any]:
    live = _cpu_tree(model.state_dict())
    model.load_state_dict(ema.shadow, strict=True)
    compression.set_deployed_bit_depths(model, True)
    model.eval()
    nats = 0.0
    misses = 0
    for start in range(0, FRAME_COUNT, TRAIN_CONFIG["eval_batch_size"]):
        end = min(start + TRAIN_CONFIG["eval_batch_size"], FRAME_COUNT)
        logits = model(target[start:end], ids[start:end], previous[start:end], event_mask[start:end])
        nats += float(F.cross_entropy(logits, target[start:end], reduction="sum"))
        misses += int((logits.argmax(dim=1) != target[start:end]).sum().item())
    model.load_state_dict(live, strict=True)
    return {
        "bpp": nats / math.log(2) / PIXELS,
        "top1_error": misses / PIXELS,
        "estimated_token_bytes": math.ceil(nats / math.log(2) / 8),
        "estimated_model_bytes": math.ceil(compression.estimated_model_bits(model) / 8),
        "bit_depth_histogram": compression.bit_depth_histogram(model),
        "byte_authority": "ADVISORY_ESTIMATE_NOT_SERIALIZED",
        "evaluated_weights": "ema_shadow",
    }


def train_stage(args: argparse.Namespace, xi1: ModuleType, pins: dict[str, Any]) -> dict[str, Any]:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise EC2Error("EC2 training requires live Metal; CPU substitution is forbidden")
    integer, compression, _, _ = xi1.configure_hpac()
    device = torch.device("mps")
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"].long().to(device)
    previous = torch.zeros_like(target)
    previous[1:] = target[:-1]
    event_array = np.load(CONTEXTS / "event_mask_n600.uint8.npy", mmap_mode="r", allow_pickle=False)
    if event_array.shape != (FRAME_COUNT, H, W) or event_array.dtype != np.uint8:
        raise EC2Error("retained event-mask geometry changed")
    event_mask = torch.from_numpy(np.asarray(event_array)).to(device)
    ids = torch.arange(FRAME_COUNT, device=device)
    model = make_model(integer, compression, device, self_compressed=True, initialize=True)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.003 * 0.02)
    generator = torch.Generator(device=device).manual_seed(SEED)
    updates = EPOCHS * math.ceil(FRAME_COUNT / TRAIN_CONFIG["batch_size"])
    ema_policy = xi1.resolve_ema_policy(updates, target_seed_fraction=0.01)
    ema = xi1.EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    source_identity = _source_identity(pins)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    resume_lineage: list[dict[str, Any]] = []
    resume_from = resolve_resume(args.resume_from)
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        start_epoch, history, resume_lineage = _restore_checkpoint(
            checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            source_identity=source_identity,
        )
        parent = file_record(resume_from)
        if not any(row.get("sha256") == parent["sha256"] for row in resume_lineage):
            resume_lineage.append({**parent, "epoch": start_epoch})
    checkpoint_root = TRAINING / "checkpoints"
    latest = checkpoint_root / "latest.pt"
    if start_epoch == 0 and not latest.exists():
        initial = _checkpoint_payload(
            epoch=0,
            model=model,
            ema=ema,
            ema_policy=ema_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
            source_identity=source_identity,
            resume_lineage=resume_lineage,
        )
        atomic_torch(checkpoint_root / "initial_stage_start.pt", initial)
        atomic_torch(latest, initial, replace=True)
    started = time.time()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        discrete = epoch > EPOCHS // 2
        compression.set_deployed_bit_depths(model, discrete)
        permutation = torch.randperm(FRAME_COUNT, generator=generator, device=device)
        for start in range(0, FRAME_COUNT, TRAIN_CONFIG["batch_size"]):
            index = permutation[start : start + TRAIN_CONFIG["batch_size"]]
            logits = model(target[index], ids[index], previous[index], event_mask[index])
            task_loss = F.cross_entropy(logits, target[index])
            rate_loss = RATE_LAMBDA * math.log(2) * compression.variable_weight_bits(model, deployed=False) / PIXELS
            loss = task_loss + rate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()
        if epoch == 1 or epoch % TRAIN_CONFIG["eval_every"] == 0 or epoch == EPOCHS:
            metrics = _evaluate(model, ema, compression, target, previous, event_mask, ids)
            record = {
                "epoch": epoch,
                "phase": "discrete_qat" if discrete else "continuous",
                **metrics,
                "telemetry_process_elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            ema=ema,
            ema_policy=ema_policy,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
            source_identity=source_identity,
            resume_lineage=resume_lineage,
        )
        atomic_torch(checkpoint_root / f"periodic/epoch_{epoch:04d}.pt", payload)
        if epoch == EPOCHS // 2:
            atomic_torch(checkpoint_root / f"continuous_stage_end_epoch_{epoch:04d}.pt", payload)
        if epoch == EPOCHS:
            atomic_torch(checkpoint_root / f"qat_stage_end_epoch_{epoch:04d}.pt", payload)
        atomic_torch(latest, payload, replace=True)
    terminal = checkpoint_root / f"qat_stage_end_epoch_{EPOCHS:04d}.pt"
    result = {
        "schema": "ddm_ec2_training_result.v1",
        "status": "COMPLETE_TRAINING_ONLY",
        "terminal_checkpoint": file_record(terminal),
        "history": history,
        "train_config": TRAIN_CONFIG,
        "source_identity": source_identity,
        "resume_lineage": resume_lineage,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(TRAINING / "TRAINING_RESULT.json", result)
    return result


def pack_stage(xi1: ModuleType) -> dict[str, Any]:
    terminal_path = TRAINING / f"checkpoints/qat_stage_end_epoch_{EPOCHS:04d}.pt"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    if terminal.get("schema") != CHECKPOINT_SCHEMA or terminal.get("epoch") != EPOCHS:
        raise EC2Error("terminal checkpoint is absent or wrong")
    if terminal.get("causal_state_sha256") != _checkpoint_digest(terminal):
        raise EC2Error("terminal checkpoint causal state hash does not verify")
    integer, compression, packer, _ = xi1.configure_hpac()
    source = make_model(integer, compression, torch.device("cpu"), self_compressed=True, initialize=False).eval()
    source.load_state_dict(terminal["ema_shadow"], strict=True)
    compression.set_deployed_bit_depths(source, True)
    raw = packer.serialize_self_compressed(source)
    raw_record = retain_payload(SERIALIZED / "terminal.hpac.raw", raw)
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS)
    compressed_record = retain_payload(SERIALIZED / "terminal.model.bin.xz", compressed)
    repeat_record = retain_payload(
        SERIALIZED / "terminal.model.repeat.bin.xz",
        lzma.compress(raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS),
    )
    if compressed_record["sha256"] != repeat_record["sha256"]:
        raise EC2Error("packed model repeat differs")
    decoded_raw = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    if decoded_raw != raw:
        raise EC2Error("XZ model payload changed packed HPAC bytes")
    restored = make_model(integer, compression, torch.device("cpu"), self_compressed=False, initialize=False).eval()
    packer.deserialize_self_compressed(restored, decoded_raw)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    current = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    previous = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    events = torch.randint(0, 2, (2, H, W), generator=generator, dtype=torch.uint8)
    ids = torch.tensor([0, FRAME_COUNT - 1])
    max_diff = float((source(current, ids, previous, events) - restored(current, ids, previous, events)).abs().max())
    if max_diff != 0.0:
        raise EC2Error(f"packed terminal changed logits: {max_diff}")
    result = {
        "schema": "ddm_ec2_pack_result.v1",
        "terminal_checkpoint": file_record(terminal_path),
        "hpac_raw": raw_record,
        "model_xz": compressed_record,
        "model_xz_repeat": repeat_record,
        "repeat_exact": True,
        "xz_roundtrip_exact": True,
        "max_logit_abs_diff": max_diff,
        "deployment_weights": "terminal ema_shadow",
        "extra_counted_model_surface": "conv_event: IntegerConv2d(1,64,3x3), learned and included in model.xz",
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.pack.json", result)
    return result


def probability_table(selected: torch.Tensor, digest: Any) -> np.ndarray:
    codes = selected.mul(8).round().clamp(-32768, 32767).to(torch.int16).cpu().numpy()
    digest.update(codes.tobytes(order="C"))
    logits = codes.astype(np.float64) / 8
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _model_from_raw(xi1: ModuleType, raw: bytes, device: torch.device) -> tuple[Any, Any, Any]:
    integer, compression, packer, inflate = xi1.configure_hpac()
    model = make_model(integer, compression, torch.device("cpu"), self_compressed=False, initialize=False).eval()
    packer.deserialize_self_compressed(model, raw)
    model = model.to(device)
    masks = inflate.group_masks(device)
    sparse = inflate.SparseIntegerHPAC(model, H, W)
    return model, masks, sparse


def _selected_coordinate_record() -> dict[str, Any]:
    manifest = json.loads((CONTEXTS / "coordinate_custody_manifest.json").read_text(encoding="utf-8"))
    selected = manifest["winner"]["payload"]
    selected_path = Path(selected["path"])
    observed = file_record(selected_path)
    if observed["sha256"] != selected["sha256"] or observed["bytes"] != selected["bytes"]:
        raise EC2Error("selected coordinate payload custody changed")
    alias = CONTEXTS / "coordinates.selected.ec2c"
    alias_record = retain_payload(alias, selected_path.read_bytes())
    if alias_record["sha256"] != selected["sha256"]:
        raise EC2Error("selected coordinate alias differs")
    return alias_record


@torch.no_grad()
def _encode_once(destination: Path, *, xi1: ModuleType) -> dict[str, Any]:
    device = torch.device("mps")
    packed_raw = (SERIALIZED / "terminal.hpac.raw").read_bytes()
    model, masks, sparse = _model_from_raw(xi1, packed_raw, device)
    raw = torch.load(CACHE, map_location="cpu", weights_only=False)["seg"].to(torch.uint8).numpy()
    coordinate_payload = (CONTEXTS / "coordinates.selected.ec2c").read_bytes()
    event_mask = event_mask_from_indices(unframe_coordinate_payload(coordinate_payload))
    training_mask = np.load(CONTEXTS / "event_mask_n600.uint8.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(event_mask, training_mask):
        raise EC2Error("receiver-derived event mask differs from retained training mask")
    encoder = constriction.stream.queue.RangeEncoder()
    family = constriction.stream.model.Categorical(perfect=False)
    logit_digest = hashlib.sha256()
    context_digest = hashlib.sha256()
    previous: np.ndarray | None = None
    ideal_bits = 0.0
    started = time.time()
    for frame in range(FRAME_COUNT):
        prior_np = np.zeros((H, W), dtype=np.uint8) if previous is None else previous
        event_np = event_mask[frame]
        context_digest.update(prior_np.tobytes(order="C"))
        context_digest.update(event_np.tobytes(order="C"))
        prior = torch.from_numpy(prior_np.astype(np.int64)).view(1, H, W).to(device)
        event = torch.from_numpy(event_np).view(1, H, W).to(device)
        current = torch.zeros_like(prior)
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        prepared = model.prepare_frame_context(idx, prior, event)
        target = torch.from_numpy(raw[frame].astype(np.int64)).to(device)
        for group, mask in enumerate(masks):
            selected = sparse.selected_logits(current, prepared, group)
            table = probability_table(selected, logit_digest)
            symbols = target[mask].cpu().numpy().astype(np.int32)
            ideal_bits += float(-np.log2(table[np.arange(len(symbols)), symbols].astype(np.float64)).sum())
            encoder.encode(symbols, family, table)
            current[0, mask] = target[mask]
        previous = raw[frame]
        if frame == 0 or (frame + 1) % 25 == 0:
            atomic_json(
                SERIALIZED / f"{destination.name}.progress.json",
                {
                    "schema": "ddm_ec2_encode_progress.v1",
                    "destination": str(destination),
                    "encoded_frames": frame + 1,
                    "elapsed_seconds": time.time() - started,
                },
            )
    payload = encoder.get_compressed().tobytes()
    payload_record = retain_payload(destination, payload)
    return {
        "range": payload_record,
        "frames": FRAME_COUNT,
        "ideal_bpp": ideal_bits / PIXELS,
        "token_bpp": payload_record["bytes"] * 8 / PIXELS,
        "logit_sha256": logit_digest.hexdigest(),
        "context_payload_sha256": context_digest.hexdigest(),
        "coordinate_payload": file_record(CONTEXTS / "coordinates.selected.ec2c"),
        "receiver_mask_matches_training_mask": True,
        "elapsed_seconds": time.time() - started,
    }


def encode_stage(xi1: ModuleType) -> dict[str, Any]:
    _selected_coordinate_record()
    primary = _encode_once(SERIALIZED / "terminal.range.bin", xi1=xi1)
    repeat = _encode_once(SERIALIZED / "terminal.repeat.range.bin", xi1=xi1)
    if primary["range"]["sha256"] != repeat["range"]["sha256"]:
        raise EC2Error("full-scale Range repeat differs")
    if (
        primary["logit_sha256"] != repeat["logit_sha256"]
        or primary["context_payload_sha256"] != repeat["context_payload_sha256"]
    ):
        raise EC2Error("repeat encode changed logits or consumed contexts")
    result = {
        "schema": "ddm_ec2_encode_result.v1",
        "primary": primary,
        "repeat": repeat,
        "repeat_exact": True,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.encode.json", result)
    return result


def package_stage() -> dict[str, Any]:
    model = (SERIALIZED / "terminal.model.bin.xz").read_bytes()
    tokens = (SERIALIZED / "terminal.range.bin").read_bytes()
    manifest = json.loads((CONTEXTS / "coordinate_custody_manifest.json").read_text(encoding="utf-8"))
    candidate_rows: list[dict[str, Any]] = []
    for row in manifest["coordinate_coder_race"]:
        coordinate_record = row["payload"]
        coordinate_path = Path(coordinate_record["path"])
        observed = file_record(coordinate_path)
        if observed["sha256"] != coordinate_record["sha256"]:
            raise EC2Error("coordinate race candidate custody changed")
        package = build_package(model, tokens, coordinate_path.read_bytes())
        package_record = retain_payload(
            SERIALIZED / f"terminal.package.{row['coder']}.ec2pkg",
            package,
        )
        parsed_model, parsed_tokens, parsed_coordinates = parse_package(package)
        if parsed_model != model or parsed_tokens != tokens or parsed_coordinates != coordinate_path.read_bytes():
            raise EC2Error(f"complete package parse-back failed for {row['coder']}")
        candidate_rows.append(
            {
                "coordinate_coder": row["coder"],
                "coordinates": observed,
                "complete_package": package_record,
                "container_overhead_bytes": PACKAGE_HEADER.size,
                "parse_back_exact": True,
            }
        )
    winner = min(candidate_rows, key=lambda row: (row["complete_package"]["bytes"], row["coordinate_coder"]))
    selected_payload = Path(winner["complete_package"]["path"]).read_bytes()
    selected = retain_payload(SERIALIZED / "terminal.package.ec2pkg", selected_payload)
    repeat = retain_payload(
        SERIALIZED / "terminal.package.repeat.ec2pkg",
        build_package(model, tokens, Path(winner["coordinates"]["path"]).read_bytes()),
    )
    if selected["sha256"] != repeat["sha256"]:
        raise EC2Error("complete package deterministic repeat differs")
    result = {
        "schema": "ddm_ec2_package_race.v1",
        "candidates": candidate_rows,
        "winner": winner,
        "selected": selected,
        "repeat": repeat,
        "repeat_exact": True,
        "admission_boundary": "actual EC2PKG1 bytes including 20-byte section framing",
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.package.json", result)
    return result


@torch.no_grad()
def decode_stage(xi1: ModuleType) -> dict[str, Any]:
    package = (SERIALIZED / "terminal.package.ec2pkg").read_bytes()
    model_xz, range_payload, coordinate_payload = parse_package(package)
    packed_raw = lzma.decompress(model_xz, format=lzma.FORMAT_XZ)
    device = torch.device("mps")
    model, masks, sparse = _model_from_raw(xi1, packed_raw, device)
    event_mask = event_mask_from_indices(unframe_coordinate_payload(coordinate_payload))
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(range_payload, dtype=np.uint32))
    family = constriction.stream.model.Categorical(perfect=False)
    logit_digest = hashlib.sha256()
    context_digest = hashlib.sha256()
    partial = SERIALIZED / f"terminal.raw.attempt_{os.getpid()}.partial.u8"
    output = np.memmap(partial, mode="w+", dtype=np.uint8, shape=(FRAME_COUNT, H, W))
    previous: np.ndarray | None = None
    started = time.time()
    for frame in range(FRAME_COUNT):
        prior_np = np.zeros((H, W), dtype=np.uint8) if previous is None else previous
        event_np = event_mask[frame]
        context_digest.update(prior_np.tobytes(order="C"))
        context_digest.update(event_np.tobytes(order="C"))
        prior = torch.from_numpy(prior_np.astype(np.int64)).view(1, H, W).to(device)
        event = torch.from_numpy(event_np).view(1, H, W).to(device)
        current = torch.zeros_like(prior)
        idx = torch.tensor([frame], dtype=torch.long, device=device)
        prepared = model.prepare_frame_context(idx, prior, event)
        for group, mask in enumerate(masks):
            selected = sparse.selected_logits(current, prepared, group)
            table = probability_table(selected, logit_digest)
            symbols = decoder.decode(family, table).astype(np.int64)
            current[0, mask] = torch.from_numpy(symbols).to(device)
        decoded = current[0].to(torch.uint8).cpu().numpy()
        output[frame] = decoded
        previous = np.asarray(output[frame])
        if frame == 0 or (frame + 1) % 25 == 0:
            output.flush()
            atomic_json(
                SERIALIZED / "terminal.decode.progress.json",
                {
                    "schema": "ddm_ec2_decode_progress.v1",
                    "partial_path": str(partial),
                    "partial_bytes": partial.stat().st_size,
                    "decoded_frames": frame + 1,
                    "elapsed_seconds": time.time() - started,
                },
            )
    output.flush()
    del output
    decoder_maybe_exhausted = bool(decoder.maybe_exhausted())
    if not decoder_maybe_exhausted:
        raise EC2Error("EC2 Range decoder did not consume the submitted stream")
    destination = SERIALIZED / "terminal.raw.u8"
    if destination.exists():
        if sha256_file(destination) != sha256_file(partial):
            raise EC2Error("existing decoded raw differs from new exact package decode")
        repeat_destination = SERIALIZED / "terminal.raw.repeat.u8"
        if repeat_destination.exists():
            if sha256_file(repeat_destination) != sha256_file(partial):
                raise EC2Error("existing decoded repeat differs from new exact package decode")
            replay_destination = SERIALIZED / f"terminal.raw.replay_{time.time_ns()}.u8"
            os.replace(partial, replay_destination)
        else:
            os.replace(partial, repeat_destination)
    else:
        os.replace(partial, destination)
    decoded_record = file_record(destination)
    expected_raw = EXPECTED_SHA256[CONTROL_RAW]
    if decoded_record["bytes"] != PIXELS or decoded_record["sha256"] != expected_raw:
        raise EC2Error("EC2 decoded tokens differ from canonical raw partition")
    encode_result = json.loads((SERIALIZED / "terminal.encode.json").read_text(encoding="utf-8"))
    primary = encode_result["primary"]
    if logit_digest.hexdigest() != primary["logit_sha256"]:
        raise EC2Error("encoder and package decoder logit hashes differ")
    if context_digest.hexdigest() != primary["context_payload_sha256"]:
        raise EC2Error("encoder and package decoder context hashes differ")
    result = {
        "schema": "ddm_ec2_decode_result.v1",
        "consumed_complete_package": file_record(SERIALIZED / "terminal.package.ec2pkg"),
        "decoded_raw": decoded_record,
        "verified_exact": True,
        "range_decoder_maybe_exhausted": decoder_maybe_exhausted,
        "logit_hash_encode_decode_equal": True,
        "context_hash_encode_decode_equal": True,
        "receiver_path": "EC2PKG1 parse -> model XZ decode + counted coordinate decode + Range decode",
        "elapsed_seconds": time.time() - started,
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(SERIALIZED / "terminal.decode.json", result)
    return result


def package_admission_passes(container_bytes: int) -> bool:
    """The preregistered falsifier is strict inequality at complete-container scope."""

    return int(container_bytes) <= MAX_PASSING_CONTAINER_BYTES


def finalize_stage() -> dict[str, Any]:
    pack = json.loads((SERIALIZED / "terminal.pack.json").read_text(encoding="utf-8"))
    encode = json.loads((SERIALIZED / "terminal.encode.json").read_text(encoding="utf-8"))
    package = json.loads((SERIALIZED / "terminal.package.json").read_text(encoding="utf-8"))
    decode = json.loads((SERIALIZED / "terminal.decode.json").read_text(encoding="utf-8"))
    actual = int(package["selected"]["bytes"])
    passes = package_admission_passes(actual)
    result = {
        "schema": "ddm_ec2_full_scale_result.v1",
        "status": "ADMISSION_PASS_BYTE_ONLY" if passes else "FORMULATION_CLOSED_FULL_SCALE",
        "complete_container": package["selected"],
        "complete_container_bytes": actual,
        "delta_vs_preregistered_bar_bytes": actual - CONTROL_CONTAINER_BAR_BYTES,
        "sections": {
            "model_xz": pack["model_xz"],
            "tokens_range": encode["primary"]["range"],
            "counted_coordinates": package["winner"]["coordinates"],
            "container_framing_bytes": PACKAGE_HEADER.size,
        },
        "decode_exact": decode["verified_exact"],
        "deterministic_repeats": {
            "model": pack["repeat_exact"],
            "range": encode["repeat_exact"],
            "complete_package": package["repeat_exact"],
        },
        "falsifier": {
            "rule": "complete receiver-consumed package bytes < 116716",
            "banked_control_bar_bytes": CONTROL_CONTAINER_BAR_BYTES,
            "largest_passing_integer_bytes": MAX_PASSING_CONTAINER_BYTES,
            "fires": not passes,
            "failure_disposition": "FORMULATION_CLOSED_FULL_SCALE",
            "verdict_scope": "FORMULATION",
        },
        "xi1_counterexample_respected": (
            "CAP1 beat XI1's learned CPR1 by 85 bytes and counted geometric xi lost by 7745 bytes; "
            "EC2 therefore admits only if its entire counted context amortizes below the package bar"
        ),
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(FULL_SCALE_RESULT, result)
    return result


def cleanup_stage() -> dict[str, Any]:
    """Certify the EC2 bulk-retention decision; never delete required evidence."""

    rows: list[dict[str, Any]] = []
    manifest_path = RETAINED / "CLEANUP_RETENTION_MANIFEST.json"
    for path in sorted(RETAINED.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if "checkpoints" in path.parts:
            reason = "P0 immutable initial/periodic/per-stage resume checkpoint"
        elif path.name.startswith("event_mask_n600"):
            reason = "training-context proof derived from counted coordinates"
        elif path.suffix in {".ec2c", ".ec2pkg", ".xz"} or "range" in path.name:
            reason = "always-keep materialized candidate or deterministic-repeat payload"
        elif "raw" in path.name:
            reason = "byte-close exact decode evidence"
        elif ".partial" in path.name:
            reason = "uncertified crash residue; certify-or-block requires KEEP"
        else:
            reason = "small provenance/receipt or required source manifest"
        rows.append({**file_record(path), "action": "KEEP", "reason": reason})
    result = {
        "schema": "ddm_ec2_cleanup_retention_manifest.v1",
        "status": "CERTIFIED_KEEP_NO_SIGNAL_LOSS",
        "files": rows,
        "deleted": [],
        "policy": (
            "EC2's large files are mandatory counted-context, stage-checkpoint, candidate/repeat, "
            "or exact-decode evidence; no certified disposable bulk remains after atomic finalization"
        ),
    }
    atomic_json(manifest_path, result)
    return result


def _host_memory_bytes() -> int | None:
    completed = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
    value = completed.stdout.strip()
    return int(value) if value.isdigit() else None


def pinned_fire_command() -> str:
    return (
        "PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 "
        ".venv/bin/python tools/safe_run.py --rss-mb 6144 --projected-gib 6 --timeout 7200 "
        "--label ddm_ec2_sparse_event_hpac_n600 --status-receipt "
        "/Volumes/VertigoDataTier/pact/ddm_ec2/run/main.safe_run.json -- "
        ".venv/bin/python tools/run_ddm_ec2_sparse_event_hpac_conditioning.py --leg all --resume-from auto"
    )


def prepare_stage(xi1: ModuleType, pins: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    INPUTS.mkdir(parents=True, exist_ok=True)
    coordinate_manifest = prepare_coordinate_context()
    selected_coordinate = _selected_coordinate_record()
    retained_inputs = {
        "js5_state": retain_payload(INPUTS / "js5_state.json", JS5_STATE.read_bytes()),
        "js5_index": retain_payload(INPUTS / "proposal_index.jsonl", JS5_INDEX.read_bytes()),
    }
    xi2_memory = json.loads(XI2_SAFE_RUN.read_text(encoding="utf-8"))
    if xi2_memory.get("peak_rss_mib") != MEASURED_XI2_PEAK_RSS_MIB or xi2_memory.get("peak_rss_observed") is not True:
        raise EC2Error("XI2 measured memory anchor changed")
    memory = {
        "status": "PASS_FROM_MEASURED_MATCHED_FULL_SCALE_ANCHOR_PLUS_MARGIN",
        "measured_anchor": file_record(XI2_SAFE_RUN),
        "measured_xi2_peak_rss_mib": MEASURED_XI2_PEAK_RSS_MIB,
        "safe_run_rss_limit_mib": SAFE_RUN_RSS_MIB,
        "headroom_mib": SAFE_RUN_RSS_MIB - MEASURED_XI2_PEAK_RSS_MIB,
        "headroom_fraction_of_measured_peak": SAFE_RUN_RSS_MIB / MEASURED_XI2_PEAK_RSS_MIB - 1,
        "safe_run_projected_gib": SAFE_RUN_PROJECTED_GIB,
        "host_memory_bytes": _host_memory_bytes(),
        "boundary": "EC2 adds one retained uint8 n600 mask and one 1x64x3x3 learned convolution; live EC2 peak unmeasured until MAIN",
    }
    receipt = {
        "schema": "ddm_ec2_build_receipt.v1",
        "status": "READY_TO_FIRE",
        "runner_source": file_record(Path(__file__).resolve()),
        "architectural_choice": (
            "thin subclass runner; pinned CL1/XI1 integer HPAC and 60-epoch lambda-1 schedule remain unchanged; "
            "CL1 previous-partition conv_past remains active and a zero-start learned conv_event(1->64,3x3) adds "
            "the counted local coordinate mask before the causal patch trunk"
        ),
        "banked_control_untouched": {
            "range": file_record(CONTROL_RANGE),
            "decoded_raw": file_record(CONTROL_RAW),
            "model_xz": file_record(CONTROL_MODEL),
            "retrained_by_ec2": False,
        },
        "coordinate_custody": {
            "manifest": file_record(CONTEXTS / "coordinate_custody_manifest.json"),
            "proposal_rows_verified": coordinate_manifest["proposal_rows_verified"],
            "unique_coordinate_sites": coordinate_manifest["unique_coordinate_sites"],
            "selected_counted_payload": selected_coordinate,
            "legality": coordinate_manifest["context_legality"],
        },
        "retained_source_manifests": retained_inputs,
        "memory_preflight": memory,
        "storage_preflight": preflight,
        "payload_retention": (
            "all coordinate coder candidates and repeats, mask, epoch checkpoints, terminal model/range/package "
            "primaries and repeats, and decoded raw are retained under the Vertigo EC2 root"
        ),
        "automatic_disk_hygiene": (
            "cleanup_stage inventories every retained file with SHA-256 and a fail-closed KEEP reason; atomic "
            "success paths leave no disposable large scratch, while checkpoints/payloads are preservation-mandated"
        ),
        "utf8_appledouble_guard": (
            "all consumed store paths must be NFC UTF-8 and must not contain AppleDouble ._ names or __MACOSX; "
            "package framing contains only binary section lengths and never filesystem metadata"
        ),
        "pinned_fire_command": pinned_fire_command(),
        "falsifier": {
            "strict_rule": "complete EC2PKG1 receiver-consumed bytes < 116716",
            "largest_passing_integer_bytes": MAX_PASSING_CONTAINER_BYTES,
            "sections": ["model.xz", "tokens.range", "counted coordinates", "20-byte EC2PKG1 framing"],
            "failure_disposition": "FORMULATION_CLOSED_FULL_SCALE",
        },
        "input_pins": pins,
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(BUILD_RECEIPT, receipt)
    ready = {
        "schema": "ddm_ec2_ready_to_fire.v1",
        "status": "READY_TO_FIRE",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN Metal executor",
        "consumer_store": str(FULL_SCALE_RESULT),
        "fire_trigger": (
            "MAIN confirms the local Metal lane is free, torch.backends.mps.is_available() is true, all source/custody "
            "pins and the 6 GiB governed memory admission pass, then executes pinned_fire_command"
        ),
        "pinned_fire_command": pinned_fire_command(),
        "build_receipt": file_record(BUILD_RECEIPT),
    }
    atomic_json(READY_TO_FIRE, ready)
    atomic_json(QUEUE / "main_metal_fire_order.json", ready)
    return receipt


def update_state(*, leg: str, status: str, pins: dict[str, Any], preflight: dict[str, Any]) -> None:
    atomic_json(
        STATE,
        {
            "schema": "ddm_ec2_state.v1",
            "arm": "ddm_ec2",
            "leg": leg,
            "status": status,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_root": str(OUTPUT),
            "payload_policy": "retain every materialized payload with bytes and sha256",
            "input_pins": pins,
            "storage_preflight": preflight,
            "hardware": {
                "system": platform.system(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            },
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leg",
        choices=("prepare", "train", "pack", "encode", "package", "decode", "finalize", "cleanup", "all"),
        required=True,
    )
    parser.add_argument(
        "--resume-from",
        default="auto",
        help="checkpoint path or 'auto' for the newest immutable per-epoch checkpoint",
    )
    parser.add_argument("--required-free-bytes", type=int, default=8 << 30)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise EC2Error("PYTHONHASHSEED=0 is required")
    if os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise EC2Error("TAC_ADMISSION_ENFORCE=1 is required")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise EC2Error("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    if args.leg not in {"prepare", "finalize", "cleanup"}:
        assert_governed_admission("run_ddm_ec2_sparse_event_hpac_conditioning")
    preflight = storage_preflight(args.required_free_bytes)
    xi1 = load_xi1()
    pins = pin_inputs(xi1)
    update_state(leg=args.leg, status="running", pins=pins, preflight=preflight)
    if args.leg in {"prepare", "all"}:
        prepare_stage(xi1, pins, preflight)
    if args.leg in {"train", "all"}:
        train_stage(args, xi1, pins)
    if args.leg in {"pack", "all"}:
        pack_stage(xi1)
    if args.leg in {"encode", "all"}:
        if not torch.backends.mps.is_available():
            raise EC2Error("EC2 Range encode requires live Metal")
        encode_stage(xi1)
    if args.leg in {"package", "all"}:
        package_stage()
    if args.leg in {"decode", "all"}:
        if not torch.backends.mps.is_available():
            raise EC2Error("EC2 package decode requires live Metal")
        decode_stage(xi1)
    result: dict[str, Any] | None = None
    if args.leg in {"finalize", "all"}:
        result = finalize_stage()
    if args.leg in {"cleanup", "all"}:
        cleanup_stage()
    update_state(leg=args.leg, status="complete", pins=pins, preflight=preflight)
    print(json.dumps(result if result is not None else {"status": "COMPLETE", "leg": args.leg}, indent=2))


if __name__ == "__main__":
    main()
