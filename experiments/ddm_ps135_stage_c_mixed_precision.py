#!/usr/bin/env python3
"""Scorer-free Stage-C LC2 mixed-precision construction and orchestration.

This driver is a sequential consumer of ``ddm_ps135`` Leg A.  Its preflight
builds the real LC2 CX2/Brotli container for the legacy q4 control and the four
cumulative SD1M q3 rungs, parses each archive through the shipped LC2 receiver,
and retains every payload before any scorer lane is claimed.

The full-n600 compensation launch is intentionally fail-closed until the Leg-A
runner exposes two dependency-injection seams documented by
``scorer_launch_blocker()``.  Emitting a proxy selection or silently reusing the
legacy-q4 master bank would be a fake implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ps135_pose_resolve as pose
from experiments import ddm_sd1_semantic_rd_curve as sd1
from src.tac.witness_dsl.jrd_priors import JrdReusablePriorPolicy

AXIS = "[macOS-CPU advisory]"
N = 600
BYTE_CEILING = pose.LC2_ARCHIVE_BYTES
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/stage_c"
)
DEFAULT_RESUME = DEFAULT_OUTPUT_ROOT / "preflight" / "state.v2.json"
DEFAULT_BULK_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_ps135_20260810/stage_c"
)
DEFAULT_RUN_STATE = DEFAULT_OUTPUT_ROOT / "run" / "state.v1.json"
CHECKPOINT = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/"
    "artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)
TOKEN_TENSOR = Path(
    "/Volumes/APDataStore/pact/ddm_lc2_20260810/cold_decode/checkpoint/tokens.npz"
)
TOKEN_RECEIPT = TOKEN_TENSOR.with_name("tokens_receipt.json")
SD1_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen"
)
JRD_POINTER_MEMO = REPO / ".omx/research/jrd_pr110_pointer_completion_20260713.md"
JRD_HARVEST_MEMO = REPO / ".omx/research/jrd_reusable_priors_harvest_20260713.md"
JRD_HARVEST_JSON = REPO / ".omx/research/jrd_reusable_priors_harvest_20260713.json"
JRD_MEASUREMENT = (
    REPO
    / "experiments/results/jrd_pr110_pointer_completion_20260713T023300Z/"
    "measurement_receipt.json"
)

RUNG_TENSORS = (
    "blocks.3.film.weight",
    "blocks.2.film.weight",
    "frame_embed.weight",
    "blocks.1.film.weight",
)
RUNG_STEMS = (
    "greedy_prefix_01_blocks.3.film.weight_q3",
    "greedy_prefix_02_blocks.2.film.weight_q3",
    "greedy_prefix_03_frame_embed.weight_q3",
    "greedy_prefix_04_blocks.1.film.weight_q3",
)
EXPECTED_CANDIDATE_IDS = ("q4_legacy_control", *RUNG_STEMS)
CANDIDATE_RECEIPT_SCHEMA = "ddm_ps135_stage_c_candidate.v2"
MASTER_CHUNK_PAIRS = 24
MASTER_RENDER_THREADS = 2
MASTER_FRAME_BYTES = pose.CAMERA_H * pose.CAMERA_W * 3
MASTER_BANK_BYTES = pose.N * MASTER_FRAME_BYTES
MASTER_BANK_REQUIRED_FREE_BYTES = 3 * MASTER_BANK_BYTES + 1_000_000_000
Q4_PARITY_PAIR_INDEX = pose.N - 1
Q4_PARITY_RAW_FRAME_INDEX = 2 * Q4_PARITY_PAIR_INDEX + 1
Q4_PARITY_EXPECTED_SHA256 = (
    "3ddebcfe23c60e891b2c5b8cccb2df1fe261d5a2cb660b64c70b97aa4b562563"
)
Q4_PARITY_AXIS = "[receiver-render exact; scorer-free parity gate]"
Q4_LITERAL_DECODE_RECEIPT = (
    pose.LC2_ROOT / "retained" / "decode" / "decode_receipt.json"
)
Q4_PARITY_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(MASTER_RENDER_THREADS),
    "MKL_NUM_THREADS": str(MASTER_RENDER_THREADS),
    "PYTHONHASHSEED": "0",
}
# A bank binds the entire driver.  This may become FROZEN only in the same edit
# that lands and reviews the sequential Stage-C scorer state machine; changing
# this file afterward deliberately invalidates the parity receipt and bank.
SCORER_SEAM_STATUS = "QUEUED_NOT_FROZEN"
CANDIDATE_RECORD_LABELS = frozenset(
    {
        "semantic",
        "semantic_cx2",
        "semantic_brotli",
        "carrier",
        "carrier_cx2",
        "carrier_brotli",
        "hpac_cx2",
        "hpac_brotli",
        "tokens",
        "model_pack",
        "member",
        "archive",
        "semantic_cx2_repeat",
        "semantic_brotli_repeat",
        "carrier_cx2_repeat",
        "carrier_brotli_repeat",
        "hpac_cx2_repeat",
        "hpac_brotli_repeat",
        "model_pack_repeat",
        "member_repeat",
        "archive_repeat",
        "allocation",
    }
)


class StageCError(RuntimeError):
    """A Stage-C custody, construction, resume, or scope invariant failed."""


class RetainedMasterRenderError(StageCError):
    """A master render failed after retaining zero or more completed frames."""

    def __init__(
        self,
        message: str,
        *,
        records: Mapping[str, object],
        completed_frames: int,
    ) -> None:
        super().__init__(message)
        self.records = dict(records)
        self.completed_frames = completed_frames


@dataclasses.dataclass(frozen=True)
class ExpectedFile:
    path: Path
    bytes: int
    sha256: str


EXPECTED_FILES = (
    ExpectedFile(
        pose.LC2_ARCHIVE,
        187_226,
        "f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45",
    ),
    ExpectedFile(
        pose.LC2_INPUTS / "semantic.raw",
        40_252,
        "9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99",
    ),
    ExpectedFile(
        pose.LC2_INPUTS / "carrier.raw",
        23_054,
        "a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4",
    ),
    ExpectedFile(
        pose.LC2_SEARCH,
        140_514,
        "bc582ee2b42171aa628db0e03d98c6937d860ac0d30d2a7bf68db7ad95af4676",
    ),
    ExpectedFile(
        CHECKPOINT,
        282_352,
        "3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647",
    ),
    ExpectedFile(
        TOKEN_TENSOR,
        117_967_988,
        "fed6331aedf16ce231718ca8fdfa7ecaaf67b762abdb697066bb55747fa9fa52",
    ),
    ExpectedFile(
        TOKEN_RECEIPT,
        920,
        "4bc9aa52b291b576aeadb7f6a239e904f82cf668d836b4a960ca804004173e97",
    ),
    ExpectedFile(
        REPO / "experiments/ddm_sd1_semantic_rd_curve.py",
        42_000,
        "85faba437a7f57ac49442b31701720cb785d20bc709e4c1dd5f92774f6941520",
    ),
    ExpectedFile(
        REPO / "src/tac/witness_dsl/jrd_priors.py",
        6_582,
        "dce176635a9c26efdd618e1ea0c9a52baf7428e265629b50ae7c2be8e49fce00",
    ),
    ExpectedFile(
        JRD_POINTER_MEMO,
        11_582,
        "b9c87233378b1cd1e5d3d58467b797086e0e2dd34b2745af8a6299406a2d0493",
    ),
    ExpectedFile(
        JRD_HARVEST_MEMO,
        9_809,
        "a4e59dd64d94ea42e9b91b6193d8308639c50969b4c469f2749388156da01d0a",
    ),
    ExpectedFile(
        JRD_HARVEST_JSON,
        2_629,
        "21de7fb839789cca5e3998cbffa59b0ce7ce95ce3715af74b04793da75c7a02e",
    ),
    ExpectedFile(
        JRD_MEASUREMENT,
        41_110,
        "2cdb36ff2b842b72d284368de44a04883a50ab4db6d886ca7402631bcd5eabb8",
    ),
)

RUNG_ARCHIVE_PINS = (
    (190_868, "9107bc1c06f1298479f8f140117be4ebe14ee659138d19b7118b1fbb96fdaf20"),
    (190_700, "f96edef0cc78d6f4ac92f0f21fc1196016a29c1cb37d11e84ed09f2645ff94a7"),
    (190_376, "de36f7dd70bce85b91c414dba03c3024de6803e974c17d2ed2faea5ae8c6475f"),
    (190_204, "010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67"),
)
RUNG_ALLOCATION_PINS = (
    (883, "aa59a2a6a717afaed9ef1b17781e352d1bd223f0fee669fb652afb1de96eef44"),
    (883, "7df9447e18555de7f2370edbd248ffec4f4c5842ea6c3e707116af69ad3e2581"),
    (881, "e7e04d7f92682a5bd24a79dfa5f5f207340c2def559c2ff0ff6afc7d00870264"),
    (883, "d4b3a36ed4b4e4f1a7b6e59d904446152a7d4184d5757001e0416a963e994f35"),
)


@dataclasses.dataclass(frozen=True)
class SemanticCandidate:
    candidate_id: str
    allocation: OrderedDict[str, int]
    semantic_blob: bytes
    expected_state: OrderedDict[str, torch.Tensor]
    source_records: dict[str, object]


@dataclasses.dataclass(frozen=True)
class ArchiveProduct:
    semantic_transformed: bytes
    carrier_transformed: bytes
    hpac_transformed: bytes
    semantic_stream: bytes
    carrier_stream: bytes
    hpac_stream: bytes
    model_pack: bytes
    member: bytes
    archive: bytes
    parseback: dict[str, object]


def verify_expected_file(expected: ExpectedFile) -> dict[str, object]:
    pose.require_file(
        expected.path,
        label=f"Stage-C source {expected.path}",
        size=expected.bytes,
        digest=expected.sha256,
    )
    return pose.file_record(expected.path)


def source_bindings() -> dict[str, object]:
    """Verify immutable inputs and bind mutable code sources for resume."""

    pose.verify_input_pins()
    immutable = {
        str(item.path.resolve()): verify_expected_file(item) for item in EXPECTED_FILES
    }
    for index, stem in enumerate(RUNG_STEMS):
        archive = SD1_ROOT / "archives" / f"{stem}.zip"
        allocation = SD1_ROOT / "allocations" / f"{stem}.json"
        archive_bytes, archive_sha = RUNG_ARCHIVE_PINS[index]
        allocation_bytes, allocation_sha = RUNG_ALLOCATION_PINS[index]
        immutable[str(archive.resolve())] = verify_expected_file(
            ExpectedFile(archive, archive_bytes, archive_sha)
        )
        immutable[str(allocation.resolve())] = verify_expected_file(
            ExpectedFile(allocation, allocation_bytes, allocation_sha)
        )
    mutable_bound = {
        "stage_c_driver": pose.file_record(Path(__file__)),
        "pose_runner": pose.file_record(REPO / "experiments/ddm_ps135_pose_resolve.py"),
        "lc2_inflate": pose.file_record(pose.LC2_RUNTIME / "inflate.py"),
        "lc2_receiver": pose.file_record(pose.LC2_RUNTIME / "receiver.py"),
        "lc2_carrier_codec": pose.file_record(pose.LC2_RUNTIME / "carrier_codec.py"),
    }
    return {"immutable_verified": immutable, "resume_bound": mutable_bound}


def load_checkpoint_state() -> OrderedDict[str, torch.Tensor]:
    with open(CHECKPOINT, "rb") as fh:
        magic = fh.read(4)
    if not (magic.startswith(b"PK\x03\x04") or magic[:1] == b"\x80"):
        raise StageCError(
            f"semantic checkpoint {CHECKPOINT} is not a PyTorch pickle/zip "
            f"(magic {magic!r}); refusing torch.load"
        )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise StageCError("semantic checkpoint lacks a state_dict mapping")
    state = OrderedDict(
        (str(name), value.detach().cpu())
        for name, value in checkpoint["state_dict"].items()
    )
    names = sd1.quantized_names(state)
    if len(names) != 16 or sum(state[name].numel() for name in names) != 63_936:
        raise StageCError("semantic checkpoint tensor census differs from SD1")
    return state


def validate_cumulative_allocations(
    allocations: list[Mapping[str, int]], names: list[str]
) -> None:
    expected = OrderedDict((name, 4) for name in names)
    if len(allocations) != len(RUNG_TENSORS):
        raise StageCError("Stage-C needs exactly four cumulative allocations")
    for index, (allocation, changed_name) in enumerate(
        zip(allocations, RUNG_TENSORS, strict=True), 1
    ):
        expected[changed_name] = 3
        normalized = OrderedDict((name, int(allocation[name])) for name in names)
        if normalized != expected or set(allocation) != set(names):
            raise StageCError(f"rung {index} is not the registered cumulative q4-to-q3 prefix")


def semantic_candidates() -> list[SemanticCandidate]:
    state = load_checkpoint_state()
    names = sd1.quantized_names(state)
    q4 = OrderedDict((name, 4) for name in names)
    q4_blob, q4_expected = sd1.pack_semantic_state(state, q4, legacy_int4=True)
    lc2_semantic = (pose.LC2_INPUTS / "semantic.raw").read_bytes()
    if q4_blob != lc2_semantic:
        raise StageCError("LC2 semantic is not byte-identical to the SD1/PR130 q4 semantic")
    candidates = [
        SemanticCandidate(
            candidate_id="q4_legacy_control",
            allocation=q4,
            semantic_blob=q4_blob,
            expected_state=q4_expected,
            source_records={"lc2_semantic": pose.file_record(pose.LC2_INPUTS / "semantic.raw")},
        )
    ]
    allocations: list[Mapping[str, int]] = []
    for stem in RUNG_STEMS:
        allocation_path = SD1_ROOT / "allocations" / f"{stem}.json"
        archive_path = SD1_ROOT / "archives" / f"{stem}.zip"
        allocation_receipt = pose.load_json(allocation_path)
        allocation = OrderedDict(
            (name, int(allocation_receipt["allocation"][name])) for name in names
        )
        allocations.append(allocation)
        blob, expected_state = sd1.pack_semantic_state(
            state, allocation, legacy_int4=False
        )
        source_blob = sd1.read_base_archive(archive_path).semantic_blob
        if blob != source_blob:
            raise StageCError(f"packed semantic differs from retained SD1 source: {stem}")
        if (
            len(blob) != int(allocation_receipt["semantic_blob_bytes"])
            or pose.sha256_bytes(blob) != allocation_receipt["semantic_blob_sha256"]
        ):
            raise StageCError(f"semantic receipt differs from retained source: {stem}")
        candidates.append(
            SemanticCandidate(
                candidate_id=stem,
                allocation=allocation,
                semantic_blob=blob,
                expected_state=expected_state,
                source_records={
                    "allocation": pose.file_record(allocation_path),
                    "source_archive": pose.file_record(archive_path),
                },
            )
        )
    validate_cumulative_allocations(allocations, names)
    return candidates


def parse_stage_c_archive(
    archive: bytes,
    *,
    semantic: SemanticCandidate,
    carrier: bytes,
    source: pose.LC2Source,
) -> dict[str, object]:
    _, receiver, inflate = pose.import_runtime_modules()
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        entries = handle.infolist()
        if (
            len(entries) != 1
            or entries[0].filename != "p"
            or entries[0].compress_type != zipfile.ZIP_STORED
            or entries[0].flag_bits & 0x1
        ):
            raise StageCError("Stage-C archive outer ZIP grammar differs from LC2")
        member = handle.read("p")
        if handle.testzip() is not None:
            raise StageCError("Stage-C archive failed ZIP CRC")
    parts = receiver.split_payload(member)
    if parts.token_codec != "ans" or parts.model_codec != "split_brotli_cx2":
        raise StageCError("Stage-C archive selectors differ from LC2")
    if len(parts.models) < 12:
        raise StageCError("Stage-C split model pack is truncated")
    stream_lengths = struct.unpack_from("<III", parts.models)
    stream_end = 12 + sum(stream_lengths)
    if any(length <= 0 for length in stream_lengths) or stream_end != len(parts.models):
        raise StageCError("Stage-C split model pack lengths do not close exactly")
    semantic_stream = parts.models[12 : 12 + stream_lengths[0]]
    carrier_stream = parts.models[
        12 + stream_lengths[0] : 12 + stream_lengths[0] + stream_lengths[1]
    ]
    hpac_stream = parts.models[12 + stream_lengths[0] + stream_lengths[1] :]
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    if decoded.compressed_stream_bytes != stream_lengths:
        raise StageCError("Stage-C decoded stream lengths differ from the model pack")
    models_raw, temporal = receiver.split_optional_temporal_reversion(decoded.raw)
    if temporal is None or temporal.packed != source.temporal_packed:
        raise StageCError("Stage-C archive changed the temporal sidecar")
    if parts.tokens != source.tokens:
        raise StageCError("Stage-C archive changed the counted token payload")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", models_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if models_raw[8:semantic_end] != semantic.semantic_blob:
        raise StageCError("Stage-C semantic parse-back differs")
    if models_raw[semantic_end:carrier_end] != carrier:
        raise StageCError("Stage-C carrier parse-back differs")
    if models_raw[carrier_end:] != source.hpac_base:
        raise StageCError("Stage-C HPAC parse-back differs")
    semantic_model, basis, coefficients = inflate.unpack_semantic_pose(
        models_raw[:carrier_end]
    )
    sd1.assert_states_equal(semantic.expected_state, semantic_model.state_dict())
    allocation, _, format_name = inflate.semantic_allocation(
        semantic.semantic_blob, semantic_model.state_dict()
    )
    if OrderedDict((name, int(allocation[name])) for name in semantic.allocation) != semantic.allocation:
        raise StageCError("shipped LC2 runtime parsed a different SD1M allocation")
    parsed_carrier = pose.decode_carrier(carrier)
    expected_coefficients = (
        parsed_carrier.codes.astype(np.float32)
        * parsed_carrier.coefficient_scales[None, :]
    )
    if not np.array_equal(coefficients.numpy(), expected_coefficients):
        raise StageCError("Stage-C receiver coefficients differ after parse-back")
    hpac = inflate.load_hpac(models_raw[carrier_end:], torch.device("cpu"))
    return {
        "archive_bytes": len(archive),
        "archive_sha256": pose.sha256_bytes(archive),
        "member_bytes": len(member),
        "member_sha256": pose.sha256_bytes(member),
        "semantic_bytes": semantic_bytes,
        "semantic_sha256": pose.sha256_bytes(semantic.semantic_blob),
        "semantic_format": format_name,
        "semantic_allocation": dict(semantic.allocation),
        "semantic_tensor_count": len(semantic_model.state_dict()),
        "carrier_bytes": carrier_bytes,
        "carrier_sha256": pose.sha256_bytes(carrier),
        "carrier_basis_shape": list(basis.shape),
        "carrier_coefficients_shape": list(coefficients.shape),
        "hpac_tensor_count": len(hpac.state_dict()),
        "hpac_base_bytes": len(source.hpac_base),
        "hpac_base_sha256": pose.sha256_bytes(source.hpac_base),
        "semantic_brotli_bytes": len(semantic_stream),
        "semantic_brotli_sha256": pose.sha256_bytes(semantic_stream),
        "carrier_brotli_bytes": len(carrier_stream),
        "carrier_brotli_sha256": pose.sha256_bytes(carrier_stream),
        "hpac_brotli_bytes": len(hpac_stream),
        "hpac_brotli_sha256": pose.sha256_bytes(hpac_stream),
        "tokens_bytes": len(parts.tokens),
        "tokens_sha256": pose.sha256_bytes(parts.tokens),
        "temporal_sha256": pose.sha256_bytes(temporal.packed),
        "validation_scope": {
            "semantic": "PARSED_LOADED_AND_STATE_EQUAL",
            "carrier": "PARSED_AND_RECEIVER_COEFFICIENTS_EQUAL",
            "hpac": "PARSED_AND_LOADED",
            "temporal": "PARSED_AND_BYTE_EQUAL",
            "tokens": "BYTE_EQUAL_ONLY_NOT_ANS_FINISHED",
        },
        "model_sections_consumed": True,
        "token_payload_consumed": False,
        "token_terminal_finish_verified": False,
        "all_sections_consumed": False,
    }


def build_stage_c_archive(
    semantic: SemanticCandidate,
    carrier: bytes,
    source: pose.LC2Source,
    *,
    failure_root: Path,
) -> ArchiveProduct:
    _, receiver, _ = pose.import_runtime_modules()
    records: dict[str, object] = {
        "semantic": pose.persist_exact(
            failure_root / "semantic.raw", semantic.semantic_blob
        ),
        "carrier": pose.persist_exact(failure_root / "carrier.cpr1", carrier),
        "hpac_wire": pose.persist_exact(
            failure_root / "hpac_plus_temporal.raw", source.hpac_wire
        ),
        "tokens": pose.persist_exact(failure_root / "tokens.ans", source.tokens),
    }
    phase = "archive_cx2_encode"
    try:
        transformed = tuple(
            receiver.encode_cx2_model_sections(
                semantic.semantic_blob, carrier, source.hpac_wire
            )
        )
        transformed_names = (
            ("semantic_cx2", "semantic.signed_zigzag_lane2.raw"),
            ("carrier_cx2", "carrier.identity.raw"),
            ("hpac_cx2", "hpac_plus_temporal.xor80.raw"),
        )
        for (label, name), payload in zip(
            transformed_names, transformed, strict=False
        ):
            if not isinstance(payload, bytes):
                raise StageCError(f"CX2 {label} output is not bytes")
            records[label] = pose.persist_exact(failure_root / name, payload)
        if len(transformed) != 3:
            raise StageCError("CX2 transform did not return exactly three sections")

        phase = "archive_cx2_inverse"
        if receiver.decode_cx2_model_sections(*transformed) != (
            semantic.semantic_blob,
            carrier,
            source.hpac_wire,
        ):
            raise StageCError("CX2 transform failed its exact inverse")

        phase = "archive_semantic_brotli"
        semantic_stream = pose.brotli_compress(transformed[0], quality=10)
        records["semantic_brotli"] = pose.persist_exact(
            failure_root / "semantic.q10.br", semantic_stream
        )
        phase = "archive_carrier_brotli"
        carrier_stream = pose.brotli_compress(transformed[1], quality=9)
        records["carrier_brotli"] = pose.persist_exact(
            failure_root / "carrier.q9.br", carrier_stream
        )
        phase = "archive_hpac_brotli"
        hpac_stream = pose.brotli_compress(transformed[2], quality=10)
        records["hpac_brotli"] = pose.persist_exact(
            failure_root / "hpac_plus_temporal.q10.br", hpac_stream
        )

        phase = "archive_model_pack"
        model_pack = pose.split_pack(
            (semantic_stream, carrier_stream, hpac_stream)
        )
        records["model_pack"] = pose.persist_exact(
            failure_root / "models.split_pack.bin", model_pack
        )
        phase = "archive_member_pack"
        member = receiver.pack_payload(
            model_pack,
            source.tokens,
            token_codec="ans",
            model_codec="split_brotli_cx2",
        )
        records["member"] = pose.persist_exact(
            failure_root / "payload.p", member
        )
        phase = "archive_zip"
        archive = pose.deterministic_stored_zip(member)
        records["archive"] = pose.persist_exact(
            failure_root / "archive.zip", archive
        )

        phase = "archive_parseback"
        parseback = parse_stage_c_archive(
            archive,
            semantic=semantic,
            carrier=carrier,
            source=source,
        )
    except Exception as error:
        persist_typed_failure(
            failure_root,
            phase=phase,
            candidate_id=semantic.candidate_id,
            reason=f"{type(error).__name__}: {error}",
            records=records,
            details={
                "semantic_allocation": dict(semantic.allocation),
                "source_records": semantic.source_records,
                "completed_record_labels": sorted(records),
            },
        )
        raise
    product = ArchiveProduct(
        semantic_transformed=transformed[0],
        carrier_transformed=transformed[1],
        hpac_transformed=transformed[2],
        semantic_stream=semantic_stream,
        carrier_stream=carrier_stream,
        hpac_stream=hpac_stream,
        model_pack=model_pack,
        member=member,
        archive=archive,
        parseback=parseback,
    )
    pose.atomic_json(
        failure_root / "build_complete.json",
        {
            "schema": "ddm_ps135_stage_c_archive_build_attempt.v1",
            "complete": True,
            "score_claim": False,
            "candidate_id": semantic.candidate_id,
            "records": records,
            "parseback": parseback,
            "payloads_retained": True,
        },
    )
    return product


def persist_typed_failure(
    root: Path,
    *,
    phase: str,
    candidate_id: str,
    reason: str,
    records: Mapping[str, object],
    details: Mapping[str, object],
    payloads_retained: bool | None = None,
) -> dict[str, object]:
    retained_records = dict(records)
    retained_payload_flag = (
        bool(retained_records)
        if payloads_retained is None
        else payloads_retained
    )
    if retained_payload_flag and not retained_records:
        raise StageCError("cannot claim retained failure payloads without records")
    receipt = {
        "schema": "ddm_ps135_retained_failure.v1",
        "complete": False,
        "score_claim": False,
        "phase": phase,
        "candidate_id": candidate_id,
        "reason": reason,
        "records": retained_records,
        "details": dict(details),
        "payloads_retained": retained_payload_flag,
        "retention_scope": (
            "TYPED_RETAINED_RECORDS_PRESENT"
            if retained_records
            else "NO_COMPLETED_PAYLOAD_RECORDS"
        ),
        "disposition": "FAIL_CLOSED_RETAINED",
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = pose.sha256_bytes(payload)
    return pose.persist_exact(root / f"failure_{phase}_{digest[:16]}.json", payload)


def persist_archive_product(
    root: Path,
    product: ArchiveProduct,
    *,
    repeat: bool,
) -> dict[str, dict[str, object]]:
    """Persist every materialized wire payload for one archive construction."""

    suffix = "_repeat" if repeat else ""
    infix = ".repeat" if repeat else ""
    payloads = {
        f"semantic_cx2{suffix}": (
            root / f"semantic{infix}.signed_zigzag_lane2.raw",
            product.semantic_transformed,
        ),
        f"semantic_brotli{suffix}": (
            root / f"semantic{infix}.q10.br",
            product.semantic_stream,
        ),
        f"carrier_cx2{suffix}": (
            root / f"carrier{infix}.identity.raw",
            product.carrier_transformed,
        ),
        f"carrier_brotli{suffix}": (
            root / f"carrier{infix}.q9.br",
            product.carrier_stream,
        ),
        f"hpac_cx2{suffix}": (
            root / f"hpac_plus_temporal{infix}.xor80.raw",
            product.hpac_transformed,
        ),
        f"hpac_brotli{suffix}": (
            root / f"hpac_plus_temporal{infix}.q10.br",
            product.hpac_stream,
        ),
        f"model_pack{suffix}": (
            root / f"models{infix}.split_pack.bin",
            product.model_pack,
        ),
        f"member{suffix}": (root / f"payload{infix}.p", product.member),
        f"archive{suffix}": (root / f"archive{infix}.zip", product.archive),
    }
    return {
        label: pose.persist_exact(path, payload)
        for label, (path, payload) in payloads.items()
    }


def verify_retained_candidate(
    receipt_path: Path,
    *,
    semantic: SemanticCandidate | None = None,
    carrier: bytes | None = None,
    source: pose.LC2Source | None = None,
) -> dict[str, object]:
    receipt = pose.load_json(receipt_path)
    if receipt.get("schema") != CANDIDATE_RECEIPT_SCHEMA or receipt.get("complete") is not True:
        raise StageCError(f"invalid retained Stage-C candidate receipt: {receipt_path}")
    records = receipt.get("records")
    if not isinstance(records, dict):
        raise StageCError("candidate receipt lacks records")
    if set(records) != CANDIDATE_RECORD_LABELS:
        raise StageCError("candidate receipt does not bind every retained payload")
    for label, record in records.items():
        pose.verify_file_record_binding(record, label=f"Stage-C {label}")
    archive = Path(records["archive"]["path"]).read_bytes()
    repeat = Path(records["archive_repeat"]["path"]).read_bytes()
    if archive != repeat:
        raise StageCError("retained Stage-C archive repeat differs")
    if len(archive) > BYTE_CEILING or receipt.get("byte_ceiling_passes") is not True:
        raise StageCError("retained Stage-C archive exceeds its hard ceiling")
    allocation = pose.load_json(Path(records["allocation"]["path"]))
    if allocation != receipt.get("semantic_allocation"):
        raise StageCError("retained allocation payload and receipt differ")
    if semantic is not None:
        if receipt.get("candidate_id") != semantic.candidate_id:
            raise StageCError("retained candidate ID differs from its ordered rung")
        if receipt.get("source_records") != semantic.source_records:
            raise StageCError("retained candidate source records differ from current sources")
        if allocation != dict(semantic.allocation):
            raise StageCError("retained allocation differs from the registered rung")
        if Path(records["semantic"]["path"]).read_bytes() != semantic.semantic_blob:
            raise StageCError("retained semantic payload differs from the registered rung")
        if carrier is None or source is None:
            raise StageCError("candidate reparse needs its carrier and LC2 source")
        if Path(records["carrier"]["path"]).read_bytes() != carrier:
            raise StageCError("retained carrier differs from its bound candidate")
        if Path(records["tokens"]["path"]).read_bytes() != source.tokens:
            raise StageCError("retained token payload differs from the LC2 source")
        _, receiver, _ = pose.import_runtime_modules()
        transformed = receiver.encode_cx2_model_sections(
            semantic.semantic_blob,
            carrier,
            source.hpac_wire,
        )
        streams = (
            pose.brotli_compress(transformed[0], quality=10),
            pose.brotli_compress(transformed[1], quality=9),
            pose.brotli_compress(transformed[2], quality=10),
        )
        model_pack = pose.split_pack(streams)
        member = receiver.pack_payload(
            model_pack,
            source.tokens,
            token_codec="ans",
            model_codec="split_brotli_cx2",
        )
        expected_archive = pose.deterministic_stored_zip(member)
        expected_payloads = {
            "semantic_cx2": transformed[0],
            "semantic_brotli": streams[0],
            "carrier_cx2": transformed[1],
            "carrier_brotli": streams[1],
            "hpac_cx2": transformed[2],
            "hpac_brotli": streams[2],
            "model_pack": model_pack,
            "member": member,
            "archive": expected_archive,
        }
        for label, expected in expected_payloads.items():
            for retained_label in (label, f"{label}_repeat"):
                retained = Path(records[retained_label]["path"]).read_bytes()
                if retained != expected:
                    raise StageCError(
                        f"retained {retained_label} differs from recomputed LC2 wire bytes"
                    )
        reparsed = parse_stage_c_archive(
            archive,
            semantic=semantic,
            carrier=carrier,
            source=source,
        )
        if reparsed != receipt.get("parseback"):
            raise StageCError("retained archive reparse differs from its receipt")
        if (
            records["archive"]["bytes"] != reparsed["archive_bytes"]
            or records["archive"]["sha256"] != reparsed["archive_sha256"]
            or records["member"]["bytes"] != reparsed["member_bytes"]
            or records["member"]["sha256"] != reparsed["member_sha256"]
            or records["semantic"]["bytes"] != reparsed["semantic_bytes"]
            or records["semantic"]["sha256"] != reparsed["semantic_sha256"]
            or records["carrier"]["bytes"] != reparsed["carrier_bytes"]
            or records["carrier"]["sha256"] != reparsed["carrier_sha256"]
            or records["semantic_brotli"]["bytes"]
            != reparsed["semantic_brotli_bytes"]
            or records["semantic_brotli"]["sha256"]
            != reparsed["semantic_brotli_sha256"]
            or records["carrier_brotli"]["bytes"]
            != reparsed["carrier_brotli_bytes"]
            or records["carrier_brotli"]["sha256"]
            != reparsed["carrier_brotli_sha256"]
            or records["hpac_brotli"]["bytes"] != reparsed["hpac_brotli_bytes"]
            or records["hpac_brotli"]["sha256"]
            != reparsed["hpac_brotli_sha256"]
            or records["tokens"]["bytes"] != reparsed["tokens_bytes"]
            or records["tokens"]["sha256"] != reparsed["tokens_sha256"]
        ):
            raise StageCError("retained archive sections are not cross-bound")
    return receipt


def retain_candidate(
    root: Path,
    semantic: SemanticCandidate,
    carrier: bytes,
    source: pose.LC2Source,
) -> dict[str, object]:
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = verify_retained_candidate(
            receipt_path,
            semantic=semantic,
            carrier=carrier,
            source=source,
        )
        if receipt.get("candidate_id") != semantic.candidate_id:
            raise StageCError("retained candidate ID differs on resume")
        if receipt.get("semantic_allocation") != dict(semantic.allocation):
            raise StageCError("retained candidate allocation differs on resume")
        records = receipt["records"]
        if records["semantic"]["sha256"] != pose.sha256_bytes(semantic.semantic_blob):
            raise StageCError("retained candidate semantic differs on resume")
        if records["carrier"]["sha256"] != pose.sha256_bytes(carrier):
            raise StageCError("retained candidate carrier differs on resume")
        return receipt
    allocation_payload = (
        json.dumps(dict(semantic.allocation), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    records = {
        "semantic": pose.persist_exact(root / "semantic.raw", semantic.semantic_blob),
        "carrier": pose.persist_exact(root / "carrier.cpr1", carrier),
        "tokens": pose.persist_exact(root / "tokens.ans", source.tokens),
        "allocation": pose.persist_exact(root / "allocation.json", allocation_payload),
    }
    product = build_stage_c_archive(
        semantic,
        carrier,
        source,
        failure_root=root / "failures" / "primary_build",
    )
    records.update(persist_archive_product(root, product, repeat=False))
    repeat = build_stage_c_archive(
        semantic,
        carrier,
        source,
        failure_root=root / "failures" / "repeat_build",
    )
    records.update(persist_archive_product(root, repeat, repeat=True))
    if (
        records["archive"]["sha256"] != records["archive_repeat"]["sha256"]
        or records["archive"]["bytes"] != records["archive_repeat"]["bytes"]
        or product.parseback != repeat.parseback
    ):
        persist_typed_failure(
            root / "failures",
            phase="candidate_repeat_mismatch",
            candidate_id=semantic.candidate_id,
            reason="independent retained archive or parseback repeat differs",
            records=records,
            details={
                "primary_parseback": product.parseback,
                "repeat_parseback": repeat.parseback,
            },
        )
        raise StageCError("Stage-C independent archive repeat differs")
    receipt = {
        "schema": CANDIDATE_RECEIPT_SCHEMA,
        "complete": True,
        "written_at_utc": pose.utc_now(),
        "axis": "[archive-byte exact; scorer-free]",
        "score_claim": False,
        "candidate_id": semantic.candidate_id,
        "semantic_allocation": dict(semantic.allocation),
        "compensation_status": "NOT_RUN_SCORER_FREE_CONTAINER_PREFLIGHT",
        "byte_ceiling": BYTE_CEILING,
        "byte_ceiling_passes": len(product.archive) <= BYTE_CEILING,
        "parseback": product.parseback,
        "records": records,
        "source_records": semantic.source_records,
        "payloads_retained": True,
    }
    if not receipt["byte_ceiling_passes"]:
        persist_typed_failure(
            root / "failures",
            phase="candidate_byte_ceiling",
            candidate_id=semantic.candidate_id,
            reason="retained archive exceeds the hard LC2 byte ceiling",
            records=records,
            details={
                "archive_bytes": len(product.archive),
                "byte_ceiling": BYTE_CEILING,
            },
        )
        raise StageCError(f"preflight candidate exceeds LC2 byte ceiling: {semantic.candidate_id}")
    pose.atomic_json(receipt_path, receipt)
    return receipt


def retain_jrd_policy(root: Path) -> dict[str, object]:
    policy = JrdReusablePriorPolicy().compile_warm_start()
    if (
        policy.get("state") != "DORMANT_N1_SCREEN"
        or policy.get("active") is not False
        or policy.get("precision_actuation") != "REFUSED_PENDING_N600_CONFIRMATION"
    ):
        raise StageCError("JRD no-confirmation policy did not fail closed")
    receipt = {
        "schema": "ddm_ps135_stage_c_jrd_policy.v1",
        "complete": True,
        "written_at_utc": pose.utc_now(),
        "axis": "[prior-custody; scorer-free]",
        "score_claim": False,
        "compiled_no_confirmation": policy,
        "use_in_stage_c": (
            "measurement ordering only; PR110 n1 bit-plane ranges are not LC2 precision assignments"
        ),
        "terminal_click_policy": (
            "eligible only after compensated convergence if implemented as a distinct exact lattice traversal"
        ),
        "sources": {
            "pointer_memo": pose.file_record(JRD_POINTER_MEMO),
            "harvest_memo": pose.file_record(JRD_HARVEST_MEMO),
            "harvest_json": pose.file_record(JRD_HARVEST_JSON),
            "measurement": pose.file_record(JRD_MEASUREMENT),
            "policy_source": pose.file_record(REPO / "src/tac/witness_dsl/jrd_priors.py"),
        },
    }
    pose.atomic_json(root / "jrd" / "policy_no_confirmation.json", receipt)
    return receipt


def official_batch_geometry(batch_size: int = 16) -> list[int]:
    geometry = pose.official_batch_sizes(batch_size)
    if geometry != [16] * 37 + [8]:
        raise StageCError("official n600 batch geometry is not 37x16 plus final unpadded 8")
    return geometry


def semantic_rung_admission(
    *,
    previous_compensated_score: float,
    candidate_compensated_score: float,
    carrier_accepted_rows: int,
) -> dict[str, object]:
    """Keep semantic/rate admission distinct from carrier-row acceptance."""

    if carrier_accepted_rows < 0:
        raise StageCError("carrier accepted-row count cannot be negative")
    admitted = candidate_compensated_score < previous_compensated_score - 1e-15
    return {
        "semantic_rung_admitted": admitted,
        "previous_compensated_score": previous_compensated_score,
        "candidate_compensated_score": candidate_compensated_score,
        "delta_score": candidate_compensated_score - previous_compensated_score,
        "carrier_accepted_rows": carrier_accepted_rows,
        "law": (
            "semantic/rate rung admission is strict full-S improvement; carrier accepted rows "
            "are reported independently and may be zero"
        ),
    }


def scorer_launch_blocker() -> dict[str, object]:
    return {
        "schema": "ddm_ps135_stage_c_scorer_launch_blocker.v1",
        "receipt_complete": True,
        "complete": False,
        "status": "QUEUED_MASTER_BANKS_AND_SCORER_ORCHESTRATION",
        "stage_c_scorer_ready": False,
        "launch_allowed": False,
        "master_bank_launch_allowed": False,
        "blocker_active": True,
        "scorer_seam_status": SCORER_SEAM_STATUS,
        "verdict_scope": "IMPLEMENTATION",
        "score_claim": False,
        "bindings": {
            "driver": pose.file_record(Path(__file__)),
            "pose_runner": pose.file_record(
                REPO / "experiments/ddm_ps135_pose_resolve.py"
            ),
        },
        "closed_runner_seams": [
            {
                "function": "rate_aware_select/save_selected_pass",
                "status": "IMPLEMENTED_NOT_SCORER_FIRED",
                "mechanism": "archive_builder callback plus exact refresh of every eligible variant",
            },
            {
                "function": "pose_outputs/GN/JRD/exact_population_refresh",
                "status": "IMPLEMENTED_NOT_SCORER_FIRED",
                "mechanism": "MasterFrameProvider bound into GN and JRD resume protocols",
            },
        ],
        "remaining": [
            "implement the sequential Leg-A-bound four-rung scorer state machine and exact final decode",
            "freeze the scorer seam and rerun the whole-driver-bound q4 pair-599 parity gate",
            "materialize and repeat-verify all four SD1M master banks on APDataStore",
            "obtain the sole ps135 lane claim only after both preceding gates close",
        ],
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN/#995 Stage-C continuation",
        "consumer_store": str(DEFAULT_OUTPUT_ROOT),
        "fire_trigger": (
            "sequential scorer seam is frozen, q4 parity and four master manifests validate, "
            "Leg A RESULT/state are complete and bound, "
            "Vertigo has pass headroom, and the sole ps135 scorer claim is live"
        ),
    }


def require_scorer_launch_ready() -> None:
    blocker = scorer_launch_blocker()
    if blocker["launch_allowed"] is not True:
        raise StageCError(
            "Stage-C scorer launch is blocked: sequential orchestration is not frozen"
        )


def verify_preflight_resume_state(
    state: Mapping[str, object],
    bindings: Mapping[str, object],
    *,
    output_root: Path,
    resume_from: Path,
) -> None:
    if (
        state.get("schema") != "ddm_ps135_stage_c_preflight_state.v2"
        or state.get("sources") != bindings
        or state.get("output_root") != str(output_root)
        or state.get("resume_path") != str(resume_from)
        or state.get("manifest_path") != str(output_root / "PREFLIGHT.json")
        or state.get("expected_candidate_ids") != list(EXPECTED_CANDIDATE_IDS)
    ):
        raise StageCError("Stage-C preflight resume bindings differ")
    candidates = state.get("candidates")
    if not isinstance(candidates, list):
        raise StageCError("Stage-C preflight resume candidate list is malformed")
    complete = state.get("complete")
    if not isinstance(complete, bool):
        raise StageCError("Stage-C preflight resume completion flag is malformed")
    ids = [row.get("candidate_id") if isinstance(row, dict) else None for row in candidates]
    if ids != list(EXPECTED_CANDIDATE_IDS[: len(candidates)]):
        raise StageCError("Stage-C preflight candidates are not the exact ordered prefix")
    if complete and (
        len(candidates) != len(EXPECTED_CANDIDATE_IDS)
        or ids != list(EXPECTED_CANDIDATE_IDS)
    ):
        raise StageCError("completed Stage-C preflight lacks the exact five candidates")
    semantic_rows = semantic_candidates()
    source = pose.load_lc2_source()
    for index, row in enumerate(candidates):
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            raise StageCError("Stage-C preflight resume candidate row is malformed")
        pose.verify_file_record_binding(
            row.get("receipt"), label="Stage-C candidate receipt"
        )
        receipt = verify_retained_candidate(
            Path(row["receipt"]["path"]),
            semantic=semantic_rows[index],
            carrier=source.carrier,
            source=source,
        )
        if (
            row.get("archive_bytes") != receipt["records"]["archive"]["bytes"]
            or row.get("archive_sha256")
            != receipt["records"]["archive"]["sha256"]
        ):
            raise StageCError("Stage-C state row and candidate archive differ")
        if (
            receipt.get("byte_ceiling_passes") is not True
            or int(row["archive_bytes"]) > BYTE_CEILING
        ):
            raise StageCError("Stage-C resumed candidate fails the hard byte ceiling")
        if index == 0:
            q4_archive = Path(receipt["records"]["archive"]["path"]).read_bytes()
            if q4_archive != pose.LC2_ARCHIVE.read_bytes():
                raise StageCError("resumed q4 archive is not exact LC2 byte identity")
    if complete:
        manifest_path = pose.verify_file_record_binding(
            state.get("manifest"), label="Stage-C preflight manifest"
        )
        if manifest_path.resolve() != (output_root / "PREFLIGHT.json").resolve():
            raise StageCError("completed Stage-C manifest record names the wrong path")


def verify_completed_preflight_manifest(
    manifest: Mapping[str, object],
    state: Mapping[str, object],
    *,
    output_root: Path,
    resume_from: Path,
) -> None:
    candidates = manifest.get("candidates")
    ids = (
        [row.get("candidate_id") if isinstance(row, dict) else None for row in candidates]
        if isinstance(candidates, list)
        else []
    )
    if (
        manifest.get("schema") != "ddm_ps135_stage_c_preflight.v2"
        or manifest.get("complete") is not True
        or state.get("complete") is not True
        or manifest.get("candidates") != state.get("candidates")
        or manifest.get("candidate_count") != len(EXPECTED_CANDIDATE_IDS)
        or manifest.get("cumulative_rung_count") != len(RUNG_STEMS)
        or manifest.get("expected_candidate_ids") != list(EXPECTED_CANDIDATE_IDS)
        or ids != list(EXPECTED_CANDIDATE_IDS)
        or manifest.get("q4_identity") is not True
        or manifest.get("byte_ceiling") != BYTE_CEILING
        or manifest.get("all_byte_ceiling_pass") is not True
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("archive_bytes"), int)
            or row["archive_bytes"] > BYTE_CEILING
            for row in candidates or []
        )
        or manifest.get("output_root") != str(output_root)
        or manifest.get("resume_path") != str(resume_from)
        or manifest.get("manifest_path") != str(output_root / "PREFLIGHT.json")
    ):
        raise StageCError("completed Stage-C manifest differs from exact resume custody")


def preflight(output_root: Path, resume_from: Path) -> dict[str, object]:
    output_root = output_root.resolve()
    resume_from = resume_from.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    storage = pose.require_vertigo_free_space(
        output_root,
        required_free_bytes=100_000_000,
        stage="stage_c_scorer_free_preflight",
    )
    bindings = source_bindings()
    if resume_from.is_file():
        state = pose.load_json(resume_from)
        verify_preflight_resume_state(
            state,
            bindings,
            output_root=output_root,
            resume_from=resume_from,
        )
        if state.get("complete") is True:
            manifest = pose.load_json(output_root / "PREFLIGHT.json")
            verify_completed_preflight_manifest(
                manifest,
                state,
                output_root=output_root,
                resume_from=resume_from,
            )
            return manifest
    else:
        state = {
            "schema": "ddm_ps135_stage_c_preflight_state.v2",
            "complete": False,
            "written_at_utc": pose.utc_now(),
            "axis": "[archive-byte exact; scorer-free]",
            "score_claim": False,
            "sources": bindings,
            "output_root": str(output_root),
            "resume_path": str(resume_from),
            "manifest_path": str(output_root / "PREFLIGHT.json"),
            "expected_candidate_ids": list(EXPECTED_CANDIDATE_IDS),
            "storage_preflight": storage,
            "candidates": [],
        }
        pose.atomic_json(resume_from, state)

    source = pose.load_lc2_source()
    carrier = source.carrier
    built_ids = {row["candidate_id"] for row in state["candidates"]}
    for index, semantic in enumerate(semantic_candidates()):
        if semantic.candidate_id in built_ids:
            continue
        directory = (
            output_root
            / "preflight"
            / f"candidate_v2_{index:02d}_{semantic.candidate_id}"
        )
        receipt = retain_candidate(directory, semantic, carrier, source)
        if index == 0 and Path(receipt["records"]["archive"]["path"]).read_bytes() != pose.LC2_ARCHIVE.read_bytes():
            raise StageCError("q4 Stage-C preflight archive does not reproduce LC2 exactly")
        state["candidates"].append(
            {
                "candidate_id": semantic.candidate_id,
                "receipt": pose.file_record(directory / "receipt.json"),
                "archive_bytes": receipt["records"]["archive"]["bytes"],
                "archive_sha256": receipt["records"]["archive"]["sha256"],
            }
        )
        pose.atomic_json(resume_from, state)

    if len(state["candidates"]) != 5:
        raise StageCError("Stage-C preflight did not retain q4 plus four rungs")
    jrd = retain_jrd_policy(output_root)
    blocker = scorer_launch_blocker()
    pose.atomic_json(output_root / "SCORER_LAUNCH_BLOCKER.json", blocker)
    orchestration = {
        "schema": "ddm_ps135_stage_c_orchestration.v1",
        "complete": False,
        "status": "SCORER_FREE_PREFLIGHT_COMPLETE_SCORER_BLOCKED",
        "axis": AXIS,
        "score_claim": False,
        "batch_geometry": official_batch_geometry(),
        "rungs": [
            {
                "rung": index,
                "candidate_id": RUNG_STEMS[index - 1],
                "tensor_added": RUNG_TENSORS[index - 1],
                "status": "QUEUED_MASTER_BANK_NOT_SCORED",
                "owner": "MAIN/#995 Stage-C continuation",
                "consumer_store": str(DEFAULT_OUTPUT_ROOT / "run"),
                "fire_trigger": (
                    "this rung master manifest validates and the preceding compensated "
                    "rung artifact binding is complete"
                ),
            }
            for index in range(1, 5)
        ],
        "scorer_launch_blocker": pose.file_record(
            output_root / "SCORER_LAUNCH_BLOCKER.json"
        ),
    }
    pose.atomic_json(output_root / "ORCHESTRATION.json", orchestration)
    manifest = {
        "schema": "ddm_ps135_stage_c_preflight.v2",
        "complete": True,
        "written_at_utc": pose.utc_now(),
        "axis": "[archive-byte exact; scorer-free]",
        "score_claim": False,
        "q4_identity": True,
        "cumulative_rung_count": 4,
        "candidate_count": 5,
        "expected_candidate_ids": list(EXPECTED_CANDIDATE_IDS),
        "output_root": str(output_root),
        "resume_path": str(resume_from),
        "manifest_path": str(output_root / "PREFLIGHT.json"),
        "candidates": state["candidates"],
        "byte_ceiling": BYTE_CEILING,
        "all_byte_ceiling_pass": all(
            int(row["archive_bytes"]) <= BYTE_CEILING for row in state["candidates"]
        ),
        "jrd_policy": jrd,
        "orchestration": pose.file_record(output_root / "ORCHESTRATION.json"),
        "scorer_launch": "NOT_RUN",
        "modal_dispatched": False,
        "payloads_retained": True,
    }
    pose.atomic_json(output_root / "PREFLIGHT.json", manifest)
    state["complete"] = True
    state["manifest"] = pose.file_record(output_root / "PREFLIGHT.json")
    pose.atomic_json(resume_from, state)
    verify_preflight_resume_state(
        state,
        bindings,
        output_root=output_root,
        resume_from=resume_from,
    )
    verify_completed_preflight_manifest(
        manifest,
        state,
        output_root=output_root,
        resume_from=resume_from,
    )
    return manifest


def require_ap_bulk_space(destination: Path, *, required_free_bytes: int) -> dict[str, object]:
    root = Path("/Volumes/APDataStore/pact").resolve()
    resolved = destination.resolve()
    if not resolved.is_relative_to(root):
        raise StageCError("Stage-C master banks must route to APDataStore")
    probe = resolved if resolved.exists() else resolved.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    receipt = {
        "schema": "ddm_ps135_stage_c_bulk_storage.v1",
        "destination": str(resolved),
        "observed_free_bytes": free,
        "required_free_bytes": required_free_bytes,
        "passes": free >= required_free_bytes,
    }
    if not receipt["passes"]:
        raise StageCError(
            f"Stage-C master bank needs {required_free_bytes} free APDataStore bytes; "
            f"observed {free}"
        )
    return receipt


def current_token_transfer_binding(source: pose.LC2Source) -> dict[str, object]:
    token_record = verify_expected_file(
        next(item for item in EXPECTED_FILES if item.path == TOKEN_TENSOR)
    )
    receipt_record = verify_expected_file(
        next(item for item in EXPECTED_FILES if item.path == TOKEN_RECEIPT)
    )
    receipt = pose.load_json(TOKEN_RECEIPT)
    _, receiver, _ = pose.import_runtime_modules()
    with zipfile.ZipFile(pose.LC2_ARCHIVE) as archive:
        if archive.namelist() != ["p"] or archive.testzip() is not None:
            raise StageCError("q4 LC2 archive does not contain one valid member p")
        q4_member = archive.read("p")
    q4_parts = receiver.split_payload(q4_member)
    q4_decoded = receiver.decode_models(
        q4_parts.models, model_codec=q4_parts.model_codec
    )
    if q4_decoded.raw != source.models_raw_wire or q4_parts.tokens != source.tokens:
        raise StageCError("q4 member does not decode to the current LC2 source bytes")
    q4_member_sha = pose.sha256_bytes(q4_member)
    q4_models_raw_sha = pose.sha256_bytes(q4_decoded.raw)
    decoded_sha = receipt.get("decoded_token_sha256")
    token_payload_sha = pose.sha256_bytes(source.tokens)
    if (
        not isinstance(decoded_sha, str)
        or len(decoded_sha) != 64
        or receipt.get("finish_token_decode_returned") is not True
        or receipt.get("ans_final_state_empty") is not True
        or receipt.get("token_payload_sha256") != token_payload_sha
        or receipt.get("archive_member_sha256") != q4_member_sha
        or receipt.get("models_raw_sha256") != q4_models_raw_sha
    ):
        raise StageCError("retained token receipt failed its frozen transfer proof")
    return {
        "schema": "ddm_ps135_stage_c_token_transfer.v1",
        "reason": (
            "semantic and carrier do not enter decode_tokens; reuse is allowed only because "
            "the token payload, HPAC base, TM1 correction, decoded token hash, and terminal "
            "ANS proof are all frozen and bound"
        ),
        "token_checkpoint": token_record,
        "token_receipt": receipt_record,
        "decoded_token_sha256": decoded_sha,
        "q4_archive_member_sha256": q4_member_sha,
        "q4_models_raw_sha256": q4_models_raw_sha,
        "token_payload_sha256": token_payload_sha,
        "hpac_base_sha256": pose.sha256_bytes(source.hpac_base),
        "temporal_sha256": pose.sha256_bytes(source.temporal_packed),
        "finish_token_decode_returned": True,
        "ans_final_state_empty": True,
    }


def load_exact_token_tensor(source: pose.LC2Source) -> tuple[np.ndarray, dict[str, object]]:
    transfer = current_token_transfer_binding(source)
    receipt = pose.load_json(TOKEN_RECEIPT)
    with np.load(TOKEN_TENSOR, allow_pickle=False) as payload:
        if set(payload.files) != {
            "tokens",
            "token_sha256",
            "finish_token_decode_returned",
            "ans_final_state_empty",
            "archive_member_sha256",
            "models_raw_sha256",
            "token_payload_sha256",
            "token_codec",
        }:
            raise StageCError("retained token checkpoint fields differ")
        tokens = payload["tokens"].copy()
        token_sha = str(payload["token_sha256"].item())
        finish_returned = bool(payload["finish_token_decode_returned"].item())
        ans_empty = bool(payload["ans_final_state_empty"].item())
        token_payload_sha = str(payload["token_payload_sha256"].item())
        token_codec = str(payload["token_codec"].item())
        archive_member_sha = str(payload["archive_member_sha256"].item())
        models_raw_sha = str(payload["models_raw_sha256"].item())
    actual_token_sha = hashlib.sha256(tokens.tobytes(order="C")).hexdigest()
    if (
        tokens.shape != (pose.N, pose.SCORER_H, pose.SCORER_W)
        or tokens.dtype != np.uint8
        or token_sha != actual_token_sha
        or transfer["decoded_token_sha256"] != actual_token_sha
        or not finish_returned
        or not ans_empty
        or receipt.get("finish_token_decode_returned") is not True
        or receipt.get("ans_final_state_empty") is not True
        or token_payload_sha != pose.sha256_bytes(source.tokens)
        or receipt.get("token_payload_sha256") != token_payload_sha
        or token_codec != "ans"
        or archive_member_sha != transfer["q4_archive_member_sha256"]
        or models_raw_sha != transfer["q4_models_raw_sha256"]
    ):
        raise StageCError("retained token checkpoint failed its terminal exact proof")
    return tokens, transfer


def require_master_render_threads(threads: int) -> None:
    if type(threads) is not int or threads != MASTER_RENDER_THREADS:
        raise StageCError(
            "Stage-C master/parity rendering requires exactly "
            f"{MASTER_RENDER_THREADS} CPU threads"
        )


def q4_parity_receipt_path(output_root: Path) -> Path:
    driver_sha = pose.sha256_file(Path(__file__))
    return (
        output_root.resolve()
        / "parity"
        / "q4_pair_0599"
        / f"driver_{driver_sha}"
        / "parity_receipt.json"
    )


def require_q4_parity_runtime(threads: int) -> dict[str, object]:
    require_master_render_threads(threads)
    literal_receipt = pose.load_json(Q4_LITERAL_DECODE_RECEIPT)
    provenance = literal_receipt.get("provenance")
    if not isinstance(provenance, dict):
        raise StageCError("q4 literal-decode runtime provenance is malformed")
    observed_environment = {
        name: os.environ.get(name) for name in Q4_PARITY_ENVIRONMENT
    }
    if (
        sys.version.split()[0] != str(provenance.get("python", "")).split()[0]
        or np.__version__ != provenance.get("numpy")
        or torch.__version__ != provenance.get("torch")
        or observed_environment != Q4_PARITY_ENVIRONMENT
    ):
        raise StageCError(
            "q4 parity requires the retained Python/NumPy/Torch runtime and 2/2/0 environment"
        )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "environment": observed_environment,
        "threads": threads,
    }


def require_configured_master_torch(torch_module) -> None:
    if (
        torch_module.get_num_threads() != MASTER_RENDER_THREADS
        or torch_module.get_num_interop_threads() != 1
        or not torch_module.are_deterministic_algorithms_enabled()
    ):
        raise StageCError("configured Torch master runtime differs from the exact contract")


def _q4_expected_master_frame() -> tuple[bytes, dict[str, object]]:
    literal_receipt = pose.load_json(Q4_LITERAL_DECODE_RECEIPT)
    raw_record = literal_receipt.get("raw")
    provenance = literal_receipt.get("provenance")
    if not isinstance(raw_record, dict) or not isinstance(provenance, dict):
        raise StageCError("q4 literal-decode receipt is malformed")
    raw_path_value = raw_record.get("path")
    raw_bytes = raw_record.get("bytes")
    raw_sha256 = raw_record.get("sha256")
    if (
        not isinstance(raw_path_value, str)
        or type(raw_bytes) is not int
        or not isinstance(raw_sha256, str)
    ):
        raise StageCError("q4 literal-decode raw binding is malformed")
    literal_raw_path = Path(raw_path_value)
    current_raw_path = pose.LC2_RAW
    expected_raw_bytes = pose.N * 2 * MASTER_FRAME_BYTES
    if (
        raw_bytes != expected_raw_bytes
        or raw_bytes != pose.LC2_RAW_BYTES
        or raw_sha256 != pose.LC2_RAW_SHA256
        or not current_raw_path.is_file()
        or current_raw_path.stat().st_size != expected_raw_bytes
    ):
        raise StageCError("q4 literal-decode raw custody pin or geometry differs")
    offset = Q4_PARITY_RAW_FRAME_INDEX * MASTER_FRAME_BYTES
    with current_raw_path.open("rb") as stream:
        stream.seek(offset)
        expected = stream.read(MASTER_FRAME_BYTES)
    if (
        len(expected) != MASTER_FRAME_BYTES
        or pose.sha256_bytes(expected) != Q4_PARITY_EXPECTED_SHA256
    ):
        raise StageCError("q4 retained final odd frame differs from its exact pin")
    return expected, {
        "literal_decode_receipt": pose.file_record(Q4_LITERAL_DECODE_RECEIPT),
        "literal_raw_path": str(literal_raw_path.resolve()),
        "current_raw_path": str(current_raw_path.resolve()),
        "cold_store_relocated": (
            literal_raw_path.resolve() != current_raw_path.resolve()
        ),
        "raw_bytes": raw_bytes,
        "raw_receipt_sha256": raw_sha256,
        "current_raw_full_sha256_pin": pose.LC2_RAW_SHA256,
        "frame_index": Q4_PARITY_RAW_FRAME_INDEX,
        "frame_offset": offset,
        "frame_bytes": MASTER_FRAME_BYTES,
        "frame_sha256": Q4_PARITY_EXPECTED_SHA256,
        "provenance": provenance,
    }


def q4_parity_bindings(
    output_root: Path,
    source: pose.LC2Source,
    *,
    threads: int,
) -> dict[str, object]:
    require_master_render_threads(threads)
    runtime = require_q4_parity_runtime(threads)
    _, expected = _q4_expected_master_frame()
    return {
        "driver": pose.file_record(Path(__file__)),
        "pose_runner": pose.file_record(
            REPO / "experiments/ddm_ps135_pose_resolve.py"
        ),
        "inflate": pose.file_record(pose.LC2_RUNTIME / "inflate.py"),
        "receiver": pose.file_record(pose.LC2_RUNTIME / "receiver.py"),
        "q4_archive": pose.file_record(pose.LC2_ARCHIVE),
        "output_root": str(output_root.resolve()),
        "candidate_id": "q4_legacy_control",
        "semantic_sha256": pose.sha256_bytes(source.semantic),
        "carrier_sha256": pose.sha256_bytes(source.carrier),
        "hpac_wire_sha256": pose.sha256_bytes(source.hpac_wire),
        "tokens_sha256": pose.sha256_bytes(source.tokens),
        "token_transfer": current_token_transfer_binding(source),
        "expected_frame": expected,
        "runtime": runtime,
    }


def validate_q4_parity_receipt(
    output_root: Path,
    *,
    source: pose.LC2Source | None = None,
    threads: int = MASTER_RENDER_THREADS,
) -> tuple[Path, dict[str, object]]:
    """Require a complete, current, byte-exact q4 odd-frame parity receipt."""

    require_master_render_threads(threads)
    path = q4_parity_receipt_path(output_root)
    if not path.is_file():
        raise StageCError(
            "complete validated q4 pair-599 parity receipt is absent; master launch blocked"
        )
    receipt = pose.load_json(path)
    if (
        receipt.get("schema") != "ddm_ps135_q4_odd_master_parity.v2"
        or receipt.get("complete") is not True
        or receipt.get("parity") is not True
        or receipt.get("axis") != Q4_PARITY_AXIS
        or receipt.get("score_claim") is not False
        or receipt.get("candidate_id") != "q4_legacy_control"
        or receipt.get("pair_index") != Q4_PARITY_PAIR_INDEX
        or receipt.get("raw_frame_index") != Q4_PARITY_RAW_FRAME_INDEX
        or receipt.get("expected_sha256") != Q4_PARITY_EXPECTED_SHA256
        or receipt.get("mismatch_count") != 0
        or receipt.get("payloads_retained") is not True
        or receipt.get("output_root") != str(output_root.resolve())
        or receipt.get("receipt_path") != str(path.resolve())
    ):
        raise StageCError("q4 parity receipt is incomplete or has the wrong identity")
    actual_path = pose.verify_file_record_binding(
        receipt.get("actual"), label="q4 parity generated frame"
    )
    expected_actual = (
        path.parent / f"pair_{Q4_PARITY_PAIR_INDEX:04d}_master.uint8.raw"
    ).resolve()
    if (
        actual_path.resolve() != expected_actual
        or receipt["actual"]["bytes"] != MASTER_FRAME_BYTES
        or receipt["actual"]["sha256"] != Q4_PARITY_EXPECTED_SHA256
    ):
        raise StageCError("q4 parity generated frame differs from the exact retained frame")
    current_source = source if source is not None else pose.load_lc2_source()
    current_bindings = q4_parity_bindings(
        output_root.resolve(), current_source, threads=threads
    )
    if receipt.get("bindings") != current_bindings:
        raise StageCError("q4 parity receipt source/runtime/driver bindings are stale")
    checkpoint = receipt.get("render_checkpoint")
    checkpoint_path = pose.verify_file_record_binding(
        checkpoint, label="q4 parity render checkpoint"
    )
    expected_checkpoint_path = actual_path.with_name(
        f"{actual_path.name}.state.json"
    ).resolve()
    checkpoint_payload = pose.load_json(checkpoint_path)
    checkpoint_binding = checkpoint_payload.get("binding")
    expected_checkpoint_binding = {
        "driver": pose.file_record(Path(__file__)),
        "candidate_id": "q4_legacy_control",
        "attempt_kind": "parity",
        "pair_start": Q4_PARITY_PAIR_INDEX,
        "pair_end": Q4_PARITY_PAIR_INDEX + 1,
        "frame_bytes": MASTER_FRAME_BYTES,
        "render": current_bindings,
    }
    if (
        checkpoint_path.resolve() != expected_checkpoint_path
        or checkpoint_payload.get("schema")
        != "ddm_ps135_master_frame_attempt.v1"
        or checkpoint_payload.get("complete") is not True
        or checkpoint_payload.get("frames_committed") != 1
        or checkpoint_payload.get("payload") != receipt.get("actual")
        or checkpoint_binding != expected_checkpoint_binding
    ):
        raise StageCError("q4 parity render checkpoint is incomplete or stale")
    return path, receipt


def require_scorer_seam_frozen() -> None:
    if SCORER_SEAM_STATUS != "FROZEN":
        raise StageCError(
            "Stage-C master launch blocked until the sequential scorer seam is frozen; "
            f"current status is {SCORER_SEAM_STATUS}"
        )


def _preflight_candidate_receipt(
    output_root: Path,
    index: int,
    semantic: SemanticCandidate,
    source: pose.LC2Source,
) -> tuple[Path, dict[str, object]]:
    path = (
        output_root
        / "preflight"
        / f"candidate_v2_{index:02d}_{semantic.candidate_id}"
        / "receipt.json"
    )
    receipt = verify_retained_candidate(
        path,
        semantic=semantic,
        carrier=source.carrier,
        source=source,
    )
    return path, receipt


def master_chunk_ranges() -> list[tuple[int, int]]:
    return [
        (start, min(start + MASTER_CHUNK_PAIRS, pose.N))
        for start in range(0, pose.N, MASTER_CHUNK_PAIRS)
    ]


def master_bank_bindings(
    *,
    output_root: Path,
    index: int,
    semantic: SemanticCandidate,
    source: pose.LC2Source,
    preflight_path: Path,
    preflight_receipt: Mapping[str, object],
    parity_path: Path,
    parity_receipt: Mapping[str, object],
    token_transfer: Mapping[str, object],
    threads: int,
) -> dict[str, object]:
    require_master_render_threads(threads)
    require_scorer_seam_frozen()
    if (
        index < 0
        or index >= len(EXPECTED_CANDIDATE_IDS)
        or EXPECTED_CANDIDATE_IDS[index] != semantic.candidate_id
    ):
        raise StageCError("master-bank candidate index differs from the registered order")
    return {
        "driver": pose.file_record(Path(__file__)),
        "pose_runner": pose.file_record(REPO / "experiments/ddm_ps135_pose_resolve.py"),
        "inflate": pose.file_record(pose.LC2_RUNTIME / "inflate.py"),
        "receiver": pose.file_record(pose.LC2_RUNTIME / "receiver.py"),
        "carrier_codec": pose.file_record(pose.LC2_RUNTIME / "carrier_codec.py"),
        "route_sources": pose.stage_c_route_source_records(),
        "output_root": str(output_root.resolve()),
        "candidate_index": index,
        "candidate_receipt": pose.file_record(preflight_path),
        "candidate_archive": preflight_receipt["records"]["archive"],
        "candidate_semantic": preflight_receipt["records"]["semantic"],
        "candidate_allocation": preflight_receipt["records"]["allocation"],
        "q4_parity_receipt": pose.file_record(parity_path),
        "q4_parity_actual": parity_receipt["actual"],
        "q4_parity_expected_sha256": parity_receipt["expected_sha256"],
        "lc2_source": {
            "semantic_sha256": pose.sha256_bytes(source.semantic),
            "carrier_sha256": pose.sha256_bytes(source.carrier),
            "hpac_wire_sha256": pose.sha256_bytes(source.hpac_wire),
            "hpac_base_sha256": pose.sha256_bytes(source.hpac_base),
            "tokens_sha256": pose.sha256_bytes(source.tokens),
            "models_raw_sha256": pose.sha256_bytes(source.models_raw),
            "models_raw_wire_sha256": pose.sha256_bytes(source.models_raw_wire),
            "temporal_sha256": pose.sha256_bytes(source.temporal_packed),
        },
        "token_transfer": dict(token_transfer),
        "render_law": {
            "semantic_forward": "shipped SemanticTokenRenderer(tokens, pair_idx)",
            "resize": "torch bilinear 874x1164 align_corners=False",
            "round": "clamp[0,255] then round then uint8 BHWC",
        },
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
        },
        "threads": threads,
        "chunk_pairs": MASTER_CHUNK_PAIRS,
        "scorer_seam_status": SCORER_SEAM_STATUS,
    }


def current_master_bank_bindings(
    *,
    output_root: Path,
    index: int,
    semantic: SemanticCandidate,
    source: pose.LC2Source,
    threads: int,
) -> dict[str, object]:
    require_master_render_threads(threads)
    preflight_path, preflight_receipt = _preflight_candidate_receipt(
        output_root.resolve(), index, semantic, source
    )
    parity_path, parity_receipt = validate_q4_parity_receipt(
        output_root.resolve(), source=source, threads=threads
    )
    return master_bank_bindings(
        output_root=output_root,
        index=index,
        semantic=semantic,
        source=source,
        preflight_path=preflight_path,
        preflight_receipt=preflight_receipt,
        parity_path=parity_path,
        parity_receipt=parity_receipt,
        token_transfer=current_token_transfer_binding(source),
        threads=threads,
    )


def validate_master_chunks(
    chunks: object,
    *,
    bank_root: Path,
    require_complete: bool = True,
) -> list[Path]:
    expected = master_chunk_ranges()
    if (
        not isinstance(chunks, list)
        or len(chunks) > len(expected)
        or (require_complete and len(chunks) != len(expected))
    ):
        raise StageCError("master-bank manifest lacks the exact registered chunks")
    primary_paths: list[Path] = []
    for row, (start, end) in zip(chunks, expected[: len(chunks)], strict=True):
        if (
            not isinstance(row, dict)
            or type(row.get("pair_start")) is not int
            or type(row.get("pair_end")) is not int
            or row["pair_start"] != start
            or row["pair_end"] != end
            or row.get("shape") != [end - start, pose.CAMERA_H, pose.CAMERA_W, 3]
            or row.get("dtype") != "uint8"
            or row.get("layout") != "BHWC_pair_order"
        ):
            raise StageCError("master-bank chunk geometry differs from the registered grid")
        primary = pose.verify_file_record_binding(
            row.get("payload"), label="master-bank chunk"
        )
        repeat = pose.verify_file_record_binding(
            row.get("payload_repeat"), label="master-bank repeat chunk"
        )
        checkpoint_keys = {
            "payload_checkpoint",
            "payload_repeat_checkpoint",
        }
        present_checkpoint_keys = checkpoint_keys.intersection(row)
        if present_checkpoint_keys and present_checkpoint_keys != checkpoint_keys:
            raise StageCError("master-bank chunk has incomplete frame checkpoints")
        if present_checkpoint_keys:
            for label in sorted(checkpoint_keys):
                checkpoint_path = pose.verify_file_record_binding(
                    row[label], label=f"master-bank {label}"
                )
                checkpoint = pose.load_json(checkpoint_path)
                if (
                    checkpoint.get("schema")
                    != "ddm_ps135_master_frame_attempt.v1"
                    or checkpoint.get("complete") is not True
                    or checkpoint.get("frames_committed") != end - start
                ):
                    raise StageCError("master-bank frame checkpoint is incomplete")
        expected_primary = (
            bank_root / "chunks" / f"chunk_{start:04d}_{end:04d}.uint8.raw"
        ).resolve()
        expected_repeat = (
            bank_root
            / "chunks"
            / f"chunk_{start:04d}_{end:04d}.repeat.uint8.raw"
        ).resolve()
        expected_bytes = (end - start) * MASTER_FRAME_BYTES
        if (
            primary.resolve() != expected_primary
            or repeat.resolve() != expected_repeat
            or row["payload"]["bytes"] != expected_bytes
            or row["payload_repeat"]["bytes"] != expected_bytes
            or row["payload"]["sha256"] != row["payload_repeat"]["sha256"]
        ):
            raise StageCError("master-bank chunk payload custody differs")
        primary_paths.append(primary)
    return primary_paths


def ordered_chunk_concatenation(primary_paths: list[Path]) -> dict[str, object]:
    digest = hashlib.sha256()
    total = 0
    for path in primary_paths:
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
                total += len(block)
    return {
        "chunk_count": len(primary_paths),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def verify_master_bank_concatenation(
    bank_record: object,
    primary_paths: list[Path],
    *,
    bank_root: Path,
    expected_concatenation: object | None = None,
) -> dict[str, object]:
    bank_path = pose.verify_file_record_binding(
        bank_record, label="master-bank payload"
    )
    if bank_path.resolve() != (bank_root / "masters.uint8.raw").resolve():
        raise StageCError("master-bank payload names a noncanonical path")
    concatenation = ordered_chunk_concatenation(primary_paths)
    if (
        concatenation["bytes"] != MASTER_BANK_BYTES
        or bank_record["bytes"] != concatenation["bytes"]
        or bank_record["sha256"] != concatenation["sha256"]
        or (
            expected_concatenation is not None
            and expected_concatenation != concatenation
        )
    ):
        raise StageCError("master bank differs from the ordered chunk concatenation")
    return concatenation


def _ordered_prefix_digest(chunks: list[dict[str, object]], count: int) -> dict[str, object]:
    return ordered_chunk_concatenation(
        [Path(row["payload"]["path"]) for row in chunks[:count]]
    )


def reconcile_master_assembly(
    assembly: Path,
    state_path: Path,
    state: dict[str, object],
) -> int:
    """Recover the fsync-before-state crash window without losing tail bytes."""

    chunks = state.get("chunks")
    completed = state.get("assembly_chunks")
    if (
        not isinstance(chunks, list)
        or type(completed) is not int
        or completed < 0
        or completed > len(chunks)
    ):
        raise StageCError("master assembly checkpoint is malformed")
    committed = _ordered_prefix_digest(chunks, completed)
    recoveries = state.setdefault("assembly_recoveries", [])
    if not isinstance(recoveries, list):
        raise StageCError("master assembly recovery ledger is malformed")
    recovery_root = assembly.parent / "recovery"
    if not assembly.exists():
        if completed:
            receipt = {
                "schema": "ddm_ps135_master_assembly_recovery.v1",
                "reason": "rebuildable assembly scratch absent; retained chunks restart assembly",
                "prior_assembly_chunks": completed,
                "retained_chunk_count": len(chunks),
                "payload_lost": False,
            }
            recovery_path = recovery_root / f"missing_after_{completed:04d}.json"
            pose.atomic_json(recovery_path, receipt)
            recoveries.append(pose.file_record(recovery_path))
            state["assembly_chunks"] = 0
            pose.atomic_json(state_path, state)
            return 0
        return 0
    actual_bytes = assembly.stat().st_size
    committed_bytes = int(committed["bytes"])
    if actual_bytes < committed_bytes:
        raise StageCError("master assembly is shorter than its committed checkpoint")
    digest = hashlib.sha256()
    with assembly.open("rb") as stream:
        remaining = committed_bytes
        while remaining:
            block = stream.read(min(8 << 20, remaining))
            if not block:
                raise StageCError("master assembly committed prefix is truncated")
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != committed["sha256"]:
        raise StageCError("master assembly differs within its committed prefix")
    if actual_bytes > committed_bytes:
        with assembly.open("rb") as stream:
            stream.seek(committed_bytes)
            tail = stream.read()
        tail_sha = pose.sha256_bytes(tail)
        tail_record = pose.persist_exact(
            recovery_root / f"uncheckpointed_after_{completed:04d}_{tail_sha[:16]}.raw",
            tail,
        )
        receipt = {
            "schema": "ddm_ps135_master_assembly_recovery.v1",
            "reason": "retained and replayed fsynced bytes written before state advance",
            "prior_assembly_chunks": completed,
            "committed_prefix_bytes": committed_bytes,
            "uncheckpointed_tail": tail_record,
        }
        recovery_path = recovery_root / f"tail_after_{completed:04d}_{tail_sha[:16]}.json"
        pose.atomic_json(recovery_path, receipt)
        recoveries.append(pose.file_record(recovery_path))
        with assembly.open("r+b") as stream:
            stream.truncate(committed_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        pose.atomic_json(state_path, state)
    return completed


def retain_master_chunk(
    bank_root: Path,
    *,
    candidate_id: str,
    start: int,
    end: int,
    render_chunk: Callable[[int, int], bytes] | None = None,
    render_to_path: Callable[
        [int, int, Path, str], tuple[dict[str, object], dict[str, object]]
    ]
    | None = None,
) -> dict[str, object]:
    """Persist primary and repeat renderings before comparing their identities."""

    if (render_chunk is None) == (render_to_path is None):
        raise StageCError("master chunk requires exactly one render interface")
    expected_bytes = (end - start) * MASTER_FRAME_BYTES
    primary_path = (
        bank_root / "chunks" / f"chunk_{start:04d}_{end:04d}.uint8.raw"
    )
    repeat_path = (
        bank_root
        / "chunks"
        / f"chunk_{start:04d}_{end:04d}.repeat.uint8.raw"
    )

    def materialize(
        path: Path, attempt_kind: str
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if render_to_path is not None:
            record, checkpoint = render_to_path(
                start, end, path, attempt_kind
            )
            pose.verify_file_record_binding(
                record, label=f"master {attempt_kind} streamed payload"
            )
            pose.verify_file_record_binding(
                checkpoint, label=f"master {attempt_kind} stream checkpoint"
            )
            return record, checkpoint
        assert render_chunk is not None
        payload = render_chunk(start, end)
        return pose.persist_exact(path, payload), None

    try:
        record, primary_checkpoint = materialize(primary_path, "primary")
    except Exception as error:
        retained = getattr(error, "records", {})
        persist_typed_failure(
            bank_root / "failures",
            phase="master_primary_render",
            candidate_id=candidate_id,
            reason=f"{type(error).__name__}: {error}",
            records=retained if isinstance(retained, Mapping) else {},
            details={
                "pair_start": start,
                "pair_end": end,
                "frames_committed": getattr(error, "completed_frames", 0),
            },
            payloads_retained=(
                isinstance(retained, Mapping)
                and any(label in retained for label in ("partial", "returned_frame"))
            ),
        )
        raise
    try:
        repeat_record, repeat_checkpoint = materialize(repeat_path, "repeat")
    except Exception as error:
        retained = {"primary": record}
        error_records = getattr(error, "records", {})
        if isinstance(error_records, Mapping):
            retained.update(
                {f"repeat_{label}": value for label, value in error_records.items()}
            )
        persist_typed_failure(
            bank_root / "failures",
            phase="master_repeat_render",
            candidate_id=candidate_id,
            reason=f"{type(error).__name__}: {error}",
            records=retained,
            details={
                "pair_start": start,
                "pair_end": end,
                "frames_committed": getattr(error, "completed_frames", 0),
            },
            payloads_retained=True,
        )
        raise
    if (
        record["bytes"] != expected_bytes
        or repeat_record["bytes"] != expected_bytes
        or record["bytes"] != repeat_record["bytes"]
        or record["sha256"] != repeat_record["sha256"]
    ):
        persist_typed_failure(
            bank_root / "failures",
            phase="master_chunk_geometry_or_repeat",
            candidate_id=candidate_id,
            reason="retained master primary/repeat geometry or identity differs",
            records={"primary": record, "repeat": repeat_record},
            details={
                "pair_start": start,
                "pair_end": end,
                "expected_bytes": expected_bytes,
                "primary_bytes": record["bytes"],
                "repeat_bytes": repeat_record["bytes"],
            },
        )
        raise StageCError("independent retained master chunk repeat differs")
    row = {
        "pair_start": start,
        "pair_end": end,
        "shape": [end - start, pose.CAMERA_H, pose.CAMERA_W, 3],
        "dtype": "uint8",
        "layout": "BHWC_pair_order",
        "payload": record,
        "payload_repeat": repeat_record,
    }
    if primary_checkpoint is not None and repeat_checkpoint is not None:
        row["payload_checkpoint"] = primary_checkpoint
        row["payload_repeat_checkpoint"] = repeat_checkpoint
    return row


def render_semantic_master_frame(
    model,
    token_tensor,
    torch_module,
    *,
    pair: int,
) -> bytes:
    """Execute the one-frame shipped semantic forward and rounding law."""

    import torch.nn.functional as functional

    if pair < 0 or pair >= pose.N:
        raise StageCError("semantic master pair index is out of range")
    idx = torch_module.tensor([pair], dtype=torch_module.long)
    master_eval = model(token_tensor[pair : pair + 1].long(), idx)
    master = functional.interpolate(
        master_eval,
        size=(pose.CAMERA_H, pose.CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 255.0).round()
    payload = (
        master.to(torch_module.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .tobytes(order="C")
    )
    if len(payload) != MASTER_FRAME_BYTES:
        raise StageCError("semantic master frame has the wrong byte geometry")
    return payload


def render_semantic_master_chunk(
    model,
    token_tensor,
    torch_module,
    *,
    start: int,
    end: int,
) -> bytes:
    """Execute the shipped semantic forward and exact master-frame rounding law."""

    return b"".join(
        render_semantic_master_frame(
            model, token_tensor, torch_module, pair=pair
        )
        for pair in range(start, end)
    )


def snapshot_master_attempt_failure(
    payload_path: Path,
    state_path: Path,
    *,
    tag: str,
) -> dict[str, object]:
    """Copy mutable resume files to immutable, hash-addressed failure custody."""

    records: dict[str, object] = {}
    failure_root = payload_path.parent / "failures"
    if payload_path.is_file() and payload_path.stat().st_size:
        payload = payload_path.read_bytes()
        digest = pose.sha256_bytes(payload)
        records["partial"] = pose.persist_exact(
            failure_root / f"{payload_path.name}.{tag}.{digest[:16]}.partial.raw",
            payload,
        )
    if state_path.is_file():
        state_payload = state_path.read_bytes()
        digest = pose.sha256_bytes(state_payload)
        records["attempt_state"] = pose.persist_exact(
            failure_root / f"{payload_path.name}.{tag}.{digest[:16]}.state.json",
            state_payload,
        )
    return records


def stream_semantic_master_chunk_attempt(
    model,
    token_tensor,
    torch_module,
    *,
    candidate_id: str,
    attempt_kind: str,
    start: int,
    end: int,
    payload_path: Path,
    render_binding: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Append/fsync/checkpoint every rendered frame and resume exact prefixes."""

    if attempt_kind not in {"primary", "repeat", "parity"}:
        raise StageCError("unregistered semantic master attempt kind")
    if start < 0 or end <= start or end > pose.N:
        raise StageCError("semantic master attempt range is invalid")
    state_path = payload_path.with_name(f"{payload_path.name}.state.json")
    binding = {
        "driver": pose.file_record(Path(__file__)),
        "candidate_id": candidate_id,
        "attempt_kind": attempt_kind,
        "pair_start": start,
        "pair_end": end,
        "frame_bytes": MASTER_FRAME_BYTES,
        "render": dict(render_binding),
    }
    if state_path.is_file():
        state = pose.load_json(state_path)
        if (
            state.get("schema") != "ddm_ps135_master_frame_attempt.v1"
            or state.get("binding") != binding
            or type(state.get("frames_committed")) is not int
            or state["frames_committed"] < 0
            or state["frames_committed"] > end - start
            or not isinstance(state.get("recoveries"), list)
        ):
            raise StageCError("semantic master frame-attempt state is stale or malformed")
    else:
        if payload_path.exists():
            payload = payload_path.read_bytes()
            digest = pose.sha256_bytes(payload)
            record = pose.persist_exact(
                payload_path.parent
                / "failures"
                / f"{payload_path.name}.orphan.{digest[:16]}.partial.raw",
                payload,
            )
            raise RetainedMasterRenderError(
                "semantic master payload exists without its checkpoint",
                records={"partial": record},
                completed_frames=0,
            )
        state = {
            "schema": "ddm_ps135_master_frame_attempt.v1",
            "complete": False,
            "binding": binding,
            "frames_committed": 0,
            "next_pair": start,
            "payload_bytes": 0,
            "prefix_sha256": hashlib.sha256(b"").hexdigest(),
            "recoveries": [],
        }
        pose.atomic_json(state_path, state)
        pose.atomic_bytes(payload_path, b"")

    committed = state["frames_committed"]
    committed_bytes = committed * MASTER_FRAME_BYTES
    if committed == 0 and not payload_path.exists():
        recovery = {
            "schema": "ddm_ps135_master_frame_recovery.v1",
            "complete": True,
            "reason": "recreated empty payload after state-before-payload initialization crash",
            "committed_frames": 0,
            "committed_bytes": 0,
            "payload_lost": False,
        }
        recovery_payload = (
            json.dumps(recovery, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        recovery_sha = pose.sha256_bytes(recovery_payload)
        recovery_record = pose.persist_exact(
            payload_path.parent
            / "recovery"
            / f"{payload_path.name}.empty_init.{recovery_sha[:16]}.json",
            recovery_payload,
        )
        if recovery_record not in state["recoveries"]:
            state["recoveries"].append(recovery_record)
            pose.atomic_json(state_path, state)
        pose.atomic_bytes(payload_path, b"")
    if not payload_path.is_file() or payload_path.stat().st_size < committed_bytes:
        records = snapshot_master_attempt_failure(
            payload_path, state_path, tag="shorter_than_checkpoint"
        )
        raise RetainedMasterRenderError(
            "semantic master partial is shorter than its checkpoint",
            records=records,
            completed_frames=committed,
        )
    with payload_path.open("rb") as stream:
        committed_prefix = stream.read(committed_bytes)
    if pose.sha256_bytes(committed_prefix) != state.get("prefix_sha256"):
        raise RetainedMasterRenderError(
            "semantic master committed prefix differs from its checkpoint",
            records=snapshot_master_attempt_failure(
                payload_path, state_path, tag="prefix_mismatch"
            ),
            completed_frames=committed,
        )
    actual_bytes = payload_path.stat().st_size
    if actual_bytes > committed_bytes:
        with payload_path.open("rb") as stream:
            stream.seek(committed_bytes)
            tail = stream.read()
        tail_record = pose.persist_exact(
            payload_path.parent
            / "recovery"
            / f"{payload_path.name}.after_{committed:04d}.{pose.sha256_bytes(tail)[:16]}.raw",
            tail,
        )
        recovery = {
            "schema": "ddm_ps135_master_frame_recovery.v1",
            "complete": True,
            "reason": "fsynced tail existed beyond the atomic frame checkpoint",
            "committed_frames": committed,
            "committed_bytes": committed_bytes,
            "tail": tail_record,
        }
        recovery_payload = (
            json.dumps(recovery, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        recovery_sha = pose.sha256_bytes(recovery_payload)
        recovery_record = pose.persist_exact(
            payload_path.parent
            / "recovery"
            / (
                f"{payload_path.name}.after_{committed:04d}."
                f"{recovery_sha[:16]}.json"
            ),
            recovery_payload,
        )
        if recovery_record not in state["recoveries"]:
            state["recoveries"].append(recovery_record)
        with payload_path.open("r+b") as stream:
            stream.truncate(committed_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        pose.atomic_json(state_path, state)

    if state.get("complete") is True:
        if committed != end - start:
            raise StageCError("completed semantic master attempt lacks all frames")
        record = pose.file_record(payload_path)
        if state.get("payload") != record:
            raise StageCError("completed semantic master attempt payload changed")
        return record, pose.file_record(state_path)

    digest = hashlib.sha256(committed_prefix)
    failed_pair = start + committed
    returned_record: dict[str, object] | None = None
    try:
        with payload_path.open("ab") as stream:
            for pair in range(start + committed, end):
                failed_pair = pair
                frame = render_semantic_master_frame(
                    model, token_tensor, torch_module, pair=pair
                )
                if len(frame) != MASTER_FRAME_BYTES:
                    returned_record = pose.persist_exact(
                        payload_path.parent
                        / "failures"
                        / f"{payload_path.name}.returned_pair_{pair:04d}.raw",
                        frame,
                    )
                    raise StageCError("rendered semantic master frame geometry differs")
                stream.write(frame)
                stream.flush()
                os.fsync(stream.fileno())
                digest.update(frame)
                state["frames_committed"] = pair - start + 1
                state["next_pair"] = pair + 1
                state["payload_bytes"] = state["frames_committed"] * MASTER_FRAME_BYTES
                state["prefix_sha256"] = digest.hexdigest()
                pose.atomic_json(state_path, state)
    except Exception as error:
        records = snapshot_master_attempt_failure(
            payload_path, state_path, tag=f"pair_{failed_pair:04d}"
        )
        if returned_record is not None:
            records["returned_frame"] = returned_record
        raise RetainedMasterRenderError(
            f"pair {failed_pair} render failed: {type(error).__name__}: {error}",
            records=records,
            completed_frames=int(state["frames_committed"]),
        ) from error

    record = pose.file_record(payload_path)
    state["complete"] = True
    state["payload"] = record
    pose.atomic_json(state_path, state)
    return record, pose.file_record(state_path)


def materialize_q4_odd_master_parity(
    output_root: Path,
    *,
    threads: int = MASTER_RENDER_THREADS,
) -> dict[str, object]:
    """Retain and validate the exact q4 pair-599 master without a scorer."""

    require_master_render_threads(threads)
    output_root = output_root.resolve()
    receipt_path = q4_parity_receipt_path(output_root)
    if receipt_path.is_file():
        _, receipt = validate_q4_parity_receipt(
            output_root, threads=threads
        )
        return receipt
    root = receipt_path.parent
    storage = pose.require_vertigo_free_space(
        root,
        required_free_bytes=4 * MASTER_FRAME_BYTES + 10_000_000,
        stage="ddm_ps135_q4_pair599_parity",
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        source = pose.load_lc2_source()
        q4 = semantic_candidates()[0]
        if (
            q4.candidate_id != "q4_legacy_control"
            or q4.semantic_blob != source.semantic
        ):
            raise StageCError("registered q4 semantic differs from the LC2 source")
        bindings = q4_parity_bindings(output_root, source, threads=threads)
    except Exception as error:
        persist_typed_failure(
            root / "failures",
            phase="q4_odd_master_parity_preflight",
            candidate_id="q4_legacy_control",
            reason=f"{type(error).__name__}: {error}",
            records={},
            details={
                "threads": threads,
                "observed_environment": {
                    name: os.environ.get(name) for name in Q4_PARITY_ENVIRONMENT
                },
            },
            payloads_retained=False,
        )
        raise
    pose.atomic_json(
        root / "attempt_started.json",
        {
            "schema": "ddm_ps135_q4_odd_master_parity_attempt.v1",
            "complete": False,
            "score_claim": False,
            "receipt_path": str(receipt_path.resolve()),
            "bindings": bindings,
            "storage_preflight": storage,
            "argv": list(sys.argv),
            "payloads_retained": False,
        },
    )
    records: dict[str, object] = {}
    try:
        expected, _ = _q4_expected_master_frame()
        tokens, _ = load_exact_token_tensor(source)
        semantic_bytes, carrier_bytes = struct.unpack_from("<II", source.models_raw)
        model_end = 8 + semantic_bytes + carrier_bytes
        _, _, inflate = pose.import_runtime_modules()
        model, _, _ = inflate.unpack_semantic_pose(source.models_raw[:model_end])
        sd1.assert_states_equal(q4.expected_state, model.state_dict())
        torch_module = pose.configure_torch(threads)
        require_configured_master_torch(torch_module)
        model = model.eval().to(torch_module.device("cpu"))
        token_tensor = torch_module.from_numpy(tokens)
        actual_path = root / f"pair_{Q4_PARITY_PAIR_INDEX:04d}_master.uint8.raw"
        with torch_module.inference_mode():
            actual_record, checkpoint_record = stream_semantic_master_chunk_attempt(
                model,
                token_tensor,
                torch_module,
                candidate_id=q4.candidate_id,
                attempt_kind="parity",
                start=Q4_PARITY_PAIR_INDEX,
                end=Q4_PARITY_PAIR_INDEX + 1,
                payload_path=actual_path,
                render_binding=bindings,
            )
        records = {
            "actual": actual_record,
            "render_checkpoint": checkpoint_record,
        }
        actual = actual_path.read_bytes()
        mismatch_count = int(
            np.count_nonzero(
                np.frombuffer(actual, dtype=np.uint8)
                != np.frombuffer(expected, dtype=np.uint8)
            )
        )
        parity = (
            len(actual) == MASTER_FRAME_BYTES
            and pose.sha256_bytes(actual) == Q4_PARITY_EXPECTED_SHA256
            and mismatch_count == 0
        )
        receipt = {
            "schema": "ddm_ps135_q4_odd_master_parity.v2",
            "complete": parity,
            "parity": parity,
            "axis": Q4_PARITY_AXIS,
            "score_claim": False,
            "written_at_utc": pose.utc_now(),
            "output_root": str(output_root),
            "receipt_path": str(receipt_path.resolve()),
            "candidate_id": q4.candidate_id,
            "pair_index": Q4_PARITY_PAIR_INDEX,
            "raw_frame_index": Q4_PARITY_RAW_FRAME_INDEX,
            "raw_byte_offset": Q4_PARITY_RAW_FRAME_INDEX * MASTER_FRAME_BYTES,
            "shape": [pose.CAMERA_H, pose.CAMERA_W, 3],
            "dtype": "uint8",
            "expected_sha256": Q4_PARITY_EXPECTED_SHA256,
            "actual": actual_record,
            "render_checkpoint": checkpoint_record,
            "mismatch_count": mismatch_count,
            "bindings": bindings,
            "argv": list(sys.argv),
            "payloads_retained": True,
        }
        pose.atomic_json(receipt_path, receipt)
        if not parity:
            persist_typed_failure(
                root / "failures",
                phase="q4_odd_master_parity_mismatch",
                candidate_id=q4.candidate_id,
                reason="retained generated q4 pair-599 master differs from literal decode",
                records=records,
                details={
                    "expected_sha256": Q4_PARITY_EXPECTED_SHA256,
                    "mismatch_count": mismatch_count,
                },
            )
            raise StageCError("q4 pair-599 master parity failed")
    except Exception as error:
        error_records = getattr(error, "records", {})
        if isinstance(error_records, Mapping):
            records.update(error_records)
        if not receipt_path.is_file():
            retained_payload_labels = {"actual", "partial", "returned_frame"}
            persist_typed_failure(
                root / "failures",
                phase="q4_odd_master_parity_render",
                candidate_id=q4.candidate_id,
                reason=f"{type(error).__name__}: {error}",
                records=records,
                details={
                    "pair_index": Q4_PARITY_PAIR_INDEX,
                    "frames_committed": getattr(error, "completed_frames", 0),
                },
                payloads_retained=any(
                    label in retained_payload_labels for label in records
                ),
            )
        raise
    _, validated = validate_q4_parity_receipt(
        output_root, source=source, threads=threads
    )
    return validated


def materialize_master_bank(
    output_root: Path,
    bulk_root: Path,
    *,
    index: int,
    semantic: SemanticCandidate,
    threads: int,
) -> dict[str, object]:
    """Retain the exact shipped semantic-renderer masters in resumable chunks."""

    require_master_render_threads(threads)
    if semantic.candidate_id not in EXPECTED_CANDIDATE_IDS:
        raise StageCError("unregistered semantic candidate cannot materialize masters")
    output_root = output_root.resolve()
    source = pose.load_lc2_source()
    parity_path, parity_receipt = validate_q4_parity_receipt(
        output_root, source=source, threads=threads
    )
    require_scorer_seam_frozen()
    bank_root = bulk_root.resolve() / "master_banks" / semantic.candidate_id
    bank_root.mkdir(parents=True, exist_ok=True)
    storage = require_ap_bulk_space(
        bank_root,
        required_free_bytes=MASTER_BANK_REQUIRED_FREE_BYTES,
    )
    preflight_path, preflight_receipt = _preflight_candidate_receipt(
        output_root, index, semantic, source
    )
    tokens, token_transfer = load_exact_token_tensor(source)
    bindings = master_bank_bindings(
        output_root=output_root,
        index=index,
        semantic=semantic,
        source=source,
        preflight_path=preflight_path,
        preflight_receipt=preflight_receipt,
        parity_path=parity_path,
        parity_receipt=parity_receipt,
        token_transfer=token_transfer,
        threads=threads,
    )
    state_path = bank_root / "state.v3.json"
    if state_path.is_file():
        state = pose.load_json(state_path)
        if (
            state.get("schema") != "ddm_ps135_stage_c_master_state.v3"
            or state.get("candidate_id") != semantic.candidate_id
            or state.get("candidate_index") != index
            or state.get("output_root") != str(output_root)
            or state.get("bindings") != bindings
            or state.get("bank_root") != str(bank_root)
        ):
            raise StageCError("Stage-C master-bank resume bindings differ")
    else:
        state = {
            "schema": "ddm_ps135_stage_c_master_state.v3",
            "complete": False,
            "candidate_id": semantic.candidate_id,
            "candidate_index": index,
            "output_root": str(output_root),
            "bank_root": str(bank_root),
            "bindings": bindings,
            "storage_preflight": storage,
            "chunks": [],
            "assembly_chunks": 0,
            "assembly_recoveries": [],
        }
        pose.atomic_json(state_path, state)

    manifest_path = bank_root / "manifest.json"
    if state.get("complete") is True:
        pose.verify_file_record_binding(state.get("manifest"), label="master manifest")
        return load_master_bank_manifest(manifest_path, semantic=semantic)

    _, _, inflate = pose.import_runtime_modules()
    semantic_pose = (
        struct.pack("<II", len(semantic.semantic_blob), len(source.carrier))
        + semantic.semantic_blob
        + source.carrier
    )
    model, _, _ = inflate.unpack_semantic_pose(semantic_pose)
    sd1.assert_states_equal(semantic.expected_state, model.state_dict())
    torch_module = pose.configure_torch(threads)
    require_configured_master_torch(torch_module)
    model = model.eval().to(torch_module.device("cpu"))
    token_tensor = torch_module.from_numpy(tokens)
    expected_chunks = master_chunk_ranges()
    validate_master_chunks(
        state["chunks"], bank_root=bank_root, require_complete=False
    )

    with torch_module.inference_mode():
        for start, end in expected_chunks[len(state["chunks"]):]:
            def render_to_path(
                pair_start: int,
                pair_end: int,
                payload_path: Path,
                attempt_kind: str,
            ) -> tuple[dict[str, object], dict[str, object]]:
                return stream_semantic_master_chunk_attempt(
                    model,
                    token_tensor,
                    torch_module,
                    candidate_id=semantic.candidate_id,
                    attempt_kind=attempt_kind,
                    start=pair_start,
                    end=pair_end,
                    payload_path=payload_path,
                    render_binding=bindings,
                )

            state["chunks"].append(
                retain_master_chunk(
                    bank_root,
                    candidate_id=semantic.candidate_id,
                    start=start,
                    end=end,
                    render_to_path=render_to_path,
                )
            )
            pose.atomic_json(state_path, state)

    assembly = bank_root / "masters.assembling.uint8.raw"
    bank_path = bank_root / "masters.uint8.raw"
    if bank_path.exists():
        if (
            assembly.exists()
            or type(state.get("assembly_chunks")) is not int
            or state["assembly_chunks"] != len(state["chunks"])
        ):
            raise StageCError("master-bank assembly has conflicting retained outputs")
    else:
        completed_assembly = reconcile_master_assembly(assembly, state_path, state)
        mode = "ab" if assembly.exists() else "wb"
        with assembly.open(mode) as output:
            for chunk_index in range(completed_assembly, len(state["chunks"])):
                chunk_path = Path(state["chunks"][chunk_index]["payload"]["path"])
                with chunk_path.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=8 << 20)
                output.flush()
                os.fsync(output.fileno())
                state["assembly_chunks"] = chunk_index + 1
                pose.atomic_json(state_path, state)
        if assembly.stat().st_size != MASTER_BANK_BYTES:
            raise StageCError("complete master assembly has the wrong byte count")
        os.replace(assembly, bank_path)
    primary_paths = validate_master_chunks(state["chunks"], bank_root=bank_root)
    bank_record = pose.file_record(bank_path)
    ordered_concatenation = verify_master_bank_concatenation(
        bank_record,
        primary_paths,
        bank_root=bank_root,
    )
    final_bindings = current_master_bank_bindings(
        output_root=output_root,
        index=index,
        semantic=semantic,
        source=source,
        threads=threads,
    )
    if final_bindings != bindings:
        raise StageCError("master-bank sources changed during materialization")
    manifest = {
        "schema": "ddm_ps135_stage_c_master_bank.v3",
        "complete": True,
        "written_at_utc": pose.utc_now(),
        "axis": "[receiver-render exact; scorer-free]",
        "score_claim": False,
        "candidate_id": semantic.candidate_id,
        "candidate_index": index,
        "output_root": str(output_root),
        "bank_root": str(bank_root),
        "pair_count": pose.N,
        "shape": [pose.N, pose.CAMERA_H, pose.CAMERA_W, 3],
        "dtype": "uint8",
        "layout": "master_only_pair_order",
        "payload": bank_record,
        "ordered_concatenation": ordered_concatenation,
        "chunks": state["chunks"],
        "bindings": final_bindings,
        "payloads_retained": True,
    }
    pose.atomic_json(manifest_path, manifest)
    state["complete"] = True
    state["payload"] = bank_record
    state["manifest"] = pose.file_record(manifest_path)
    pose.atomic_json(state_path, state)
    return manifest


def load_master_bank_manifest(
    manifest_path: Path,
    *,
    semantic: SemanticCandidate,
) -> dict[str, object]:
    manifest = pose.load_json(manifest_path)
    candidate_index = manifest.get("candidate_index")
    output_root_value = manifest.get("output_root")
    bank_root = manifest_path.resolve().parent
    if (
        manifest.get("schema") != "ddm_ps135_stage_c_master_bank.v3"
        or manifest.get("complete") is not True
        or manifest.get("candidate_id") != semantic.candidate_id
        or type(candidate_index) is not int
        or candidate_index < 0
        or candidate_index >= len(EXPECTED_CANDIDATE_IDS)
        or EXPECTED_CANDIDATE_IDS[candidate_index] != semantic.candidate_id
        or not isinstance(output_root_value, str)
        or manifest.get("bank_root") != str(bank_root)
        or manifest.get("pair_count") != pose.N
        or manifest.get("shape") != [pose.N, pose.CAMERA_H, pose.CAMERA_W, 3]
        or manifest.get("dtype") != "uint8"
        or manifest.get("layout") != "master_only_pair_order"
    ):
        raise StageCError("master-bank manifest shape or identity differs")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or type(bindings.get("threads")) is not int:
        raise StageCError("master-bank manifest bindings are malformed")
    require_master_render_threads(bindings["threads"])
    require_scorer_seam_frozen()
    source = pose.load_lc2_source()
    current_bindings = current_master_bank_bindings(
        output_root=Path(output_root_value).resolve(),
        index=candidate_index,
        semantic=semantic,
        source=source,
        threads=bindings["threads"],
    )
    if bindings != current_bindings:
        raise StageCError("master-bank manifest source/runtime bindings are stale")
    primary_paths = validate_master_chunks(
        manifest.get("chunks"), bank_root=bank_root
    )
    verify_master_bank_concatenation(
        manifest.get("payload"),
        primary_paths,
        bank_root=bank_root,
        expected_concatenation=manifest.get("ordered_concatenation"),
    )
    return manifest


def master_provider_from_manifest(
    manifest_path: Path,
    *,
    semantic: SemanticCandidate,
) -> pose.MasterFrameProvider:
    manifest = load_master_bank_manifest(manifest_path, semantic=semantic)
    frames = np.memmap(
        Path(manifest["payload"]["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(pose.N, pose.CAMERA_H, pose.CAMERA_W, 3),
    )
    return pose.MasterFrameProvider(
        frames=frames,
        binding={
            "schema": "ddm_ps135_master_provider.v1",
            "layout": "master_only_pair_order",
            "candidate_id": semantic.candidate_id,
            "semantic_sha256": pose.sha256_bytes(semantic.semantic_blob),
            "manifest": pose.file_record(manifest_path),
            "payload": manifest["payload"],
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser(
        "preflight", help="retain q4 plus four real LC2 mixed archives without a scorer"
    )
    pre.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    pre.add_argument("--resume-from", type=Path, default=DEFAULT_RESUME)
    parity = subparsers.add_parser(
        "q4-parity",
        help="retain and validate the q4 pair-599 literal-decode master gate",
    )
    parity.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parity.add_argument(
        "--threads",
        type=int,
        choices=(MASTER_RENDER_THREADS,),
        default=MASTER_RENDER_THREADS,
    )
    parity.set_defaults(resume_from=None)
    masters = subparsers.add_parser(
        "materialize-masters",
        help="retain the four receiver-realized SD1M master banks without a scorer",
    )
    masters.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    masters.add_argument("--bulk-root", type=Path, default=DEFAULT_BULK_ROOT)
    masters.add_argument(
        "--threads",
        type=int,
        choices=(MASTER_RENDER_THREADS,),
        default=MASTER_RENDER_THREADS,
    )
    masters.add_argument(
        "--candidate-id",
        choices=("all", *RUNG_STEMS),
        default="all",
    )
    masters.set_defaults(resume_from=None)
    block = subparsers.add_parser(
        "scorer-blocker", help="print the exact runner seams required before launch"
    )
    block.set_defaults(output_root=None, resume_from=None)
    return root


def main() -> None:
    args = parser().parse_args()
    started = time.time()
    if args.command == "preflight":
        result = preflight(args.output_root, args.resume_from)
    elif args.command == "q4-parity":
        require_master_render_threads(args.threads)
        result = materialize_q4_odd_master_parity(
            args.output_root, threads=args.threads
        )
    elif args.command == "materialize-masters":
        require_master_render_threads(args.threads)
        candidates = semantic_candidates()[1:]
        selected = [
            (index, candidate)
            for index, candidate in enumerate(candidates, 1)
            if args.candidate_id in {"all", candidate.candidate_id}
        ]
        manifests = [
            materialize_master_bank(
                args.output_root,
                args.bulk_root,
                index=index,
                semantic=candidate,
                threads=args.threads,
            )
            for index, candidate in selected
        ]
        result = {
            "schema": "ddm_ps135_stage_c_master_materialization.v1",
            "complete": len(manifests) == len(selected),
            "axis": "[receiver-render exact; scorer-free]",
            "score_claim": False,
            "candidate_ids": [candidate.candidate_id for _, candidate in selected],
            "manifests": manifests,
            "scorer_launch": "NOT_RUN",
            "payloads_retained": True,
        }
    elif args.command == "scorer-blocker":
        result = scorer_launch_blocker()
    else:
        raise StageCError(f"unregistered Stage-C command: {args.command}")
    result = {**result, "command_elapsed_seconds": time.time() - started}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
