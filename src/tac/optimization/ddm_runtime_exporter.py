# SPDX-License-Identifier: MIT
"""Deterministic compiler from the sealed DDM v15/J2 state to a runtime packet."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import struct
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import brotli
except ImportError:  # E4's only authorized coder-fallback trigger.
    brotli = None  # type: ignore[assignment]

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.canonical_equations.ddm_runtime_export_identity_20260723 import (
    semantic_paint_jacobian_summary,
)
from tac.optimization import ddm_runtime_receiver as runtime
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    ReceiverGrammarAdmission,
    ReceiverGrammarStream,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.direct_description_carrier_compose import (
    CLASS_ORDER,
    REALIZATION_PAINT_ORDER,
    REALIZATION_STATIC_RULE_MEMBER,
    ROLE_CLASS_IDS,
    WORLDSHEET_G1_MEMBER,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_g1_worldsheet import lift_g1_movable_worldsheet
from tac.optimization.direct_description_joint_descent import derive_lane_program_seeds
from tac.optimization.direct_description_measurement_ladder import (
    rfc8785_canonicalize,
)

SCHEMA = "ddm_e1_runtime_archive.v1"
E2_SCHEMA = "ddm_e2_runtime_archive.v1"
E3_SCHEMA = "ddm_e3_runtime_archive.v1"
E4_SCHEMA = "ddm_e4_runtime_archive.v1"
E4_WS1_SCHEMA = "ddm_e4_ws1_runtime_archive.v1"
IC1_SCHEMA = "ddm_ic1_runtime_archive.v1"
IC2_SCHEMA = "ddm_ic2_runtime_archive.v1"
RATE_DOCTRINE_SCHEMA = "ddm_four_clause_rate_doctrine.v1"
CONFIG_SCHEMA = "DDME1RuntimeExporterConfigV1"
E2_CONFIG_SCHEMA = "DDME2RuntimeExporterConfigV1"
E3_CONFIG_SCHEMA = "DDME3RuntimeExporterConfigV1"
E4_CONFIG_SCHEMA = "DDME4RuntimeExporterConfigV1"
E4_WS1_CONFIG_SCHEMA = "DDME4WS1RuntimeExporterConfigV1"
IC1_CONFIG_SCHEMA = "DDMIC1RuntimeExporterConfigV1"
IC2_CONFIG_SCHEMA = "DDMIC2RuntimeExporterConfigV1"
RESULT_SCHEMA = "ddm_e1_runtime_export_receipt.v1"
E2_RESULT_SCHEMA = "ddm_e2_runtime_export_receipt.v1"
E3_RESULT_SCHEMA = "ddm_e3_runtime_export_receipt.v1"
E4_RESULT_SCHEMA = "ddm_e4_runtime_export_receipt.v1"
E4_WS1_RESULT_SCHEMA = "ddm_e4_ws1_runtime_export_receipt.v1"
IC1_RESULT_SCHEMA = "ddm_ic1_runtime_export_receipt.v1"
IC2_RESULT_SCHEMA = "ddm_ic2_runtime_export_receipt.v1"
SOURCE_BYTES = 133_941
SOURCE_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
STATE_BYTES = 134_211
STATE_SHA256 = "3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd"
STATE_NAME = "v15_j2_lane_seed_theta0"
EXPECTED_DOF_COUNT = 368
EXPECTED_MEMBERS = ("manifest.json", "base/chart.ddb", "semantic/composed.dds")
EXPECTED_WS1_MEMBERS = ("manifest.json", "state/ws1.ddj5")
EXTENSION_SLOTS = (
    {
        "active_member": None,
        "block": "D1",
        "frame": "DDE1B",
        "member_prefix": "amplitude/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_amplitude_field_section.v1",
    },
    {
        "active_member": None,
        "block": "D2",
        "frame": "DDE1B",
        "member_prefix": "tolerance/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_per_element_tolerance_dual_section.v1",
    },
    {
        "active_member": None,
        "block": "D5",
        "frame": "DDE1B",
        "member_prefix": "texture_quotient/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_texture_quotient_residual_stats_section.v1",
    },
    {
        "active_member": None,
        "block": "D4",
        "frame": "DDE1B",
        "member_prefix": "coder/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_coder_probability_parameters_section.v1",
    },
    {
        "active_member": None,
        "block": "D6",
        "frame": "DDE1B",
        "member_prefix": "realization/",
        "rate_custody": "independent_member_bytes_sha256",
        "schema": "ddm_realization_map_metadata_section.v1",
    },
)
REFINEMENT_CONTRACT = {
    "apparatus_validity_stamps_required": True,
    "best_so_far_checkpoint": True,
    "block_order": ["L", "D2", "D1", "D4", "D6", "D5"],
    "block_order_policy": ("dependency_topological_then_coarse_to_fine_then_largest_marginal_first"),
    "curve_authority": "verification_receipt",
    "cycle_index": 0,
    "fixed_budget": True,
    "global_reinvestment": True,
    "schema": "ddm_joint_fixed_budget_refinement.v1",
    "status": "OPEN_SUCCESSOR_CYCLES_OWED",
    "stop_law": "full_joint_cycle_no_net_gain_at_constant_bytes",
    "train_least_order": ["derive", "solve", "fit", "train_last"],
}
LANGUAGE_VERSION = "ddm_composed_language.v1"
WS1_GRAMMAR_VERSION = "ddm_ws1_receiver_closed_warm_start.v1"
BLOB_MAGIC = b"DDE1B"
BLOB_HEADER = struct.Struct(">5sBBBBQQ32s")
BROTLI_Q11_CODER = "brotli_q11"
E3_LZMA1_CODER = "lzma1_raw_d1m_lc3_lp0_pb2"
CODER_IDS = {BROTLI_Q11_CODER: 1, E3_LZMA1_CODER: 2}
ZIP_LOCAL_HEADER = struct.Struct("<4s5H3I2H")
ZIP_EOCD = struct.Struct("<4s4H2IH")
PAIR_H = 384
PAIR_W = 512
CAMERA_H = 874
CAMERA_W = 1164
FRAMES_PER_PAIR = 2
CHANNELS = 3
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SOURCE = Path(__file__).with_name("ddm_runtime_receiver.py")


class ExporterError(ValueError):
    """The export contract or source custody failed closed."""


@dataclass(frozen=True, slots=True)
class ExactRuntimePacketPrice:
    """Full stored-ZIP price after a caller-supplied exact parse-back."""

    archive_bytes: int
    archive_sha256: str
    member_payload_bytes: int
    container_bytes: int
    byte_home_ledger: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExactRuntimeMarginalPrice:
    """Candidate minus control bytes under identical deterministic framing."""

    control: ExactRuntimePacketPrice
    candidate: ExactRuntimePacketPrice
    delta_archive_bytes: int
    parseback_verified: Literal[True] = True


class DDME1RuntimeExporterConfigV1(BaseModel):
    """Typed local-only Build #636 compiler program."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal[
        "DDME1RuntimeExporterConfigV1",
        "DDME2RuntimeExporterConfigV1",
        "DDME3RuntimeExporterConfigV1",
        "DDME4RuntimeExporterConfigV1",
    ] = Field(default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema")
    run_id: Literal[
        "ddm_e1_runtime_exporter_n600_20260723",
        "ddm_e2_pose_stream_and_doctrine_export_20260723",
        "ddm_e3_inflate_compose_and_depclose_20260723",
        "ddm_e4_brotli_declared_dep_20260724",
    ] = "ddm_e1_runtime_exporter_n600_20260723"
    source_archive_path: StrictStr
    source_archive_bytes: Literal[133941] = SOURCE_BYTES
    source_archive_sha256: Literal["759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"] = SOURCE_SHA256
    state_name: Literal["v15_j2_lane_seed_theta0"] = STATE_NAME
    state_archive_bytes: Literal[134211] = STATE_BYTES
    state_archive_sha256: Literal["3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd"] = STATE_SHA256
    output_directory: StrictStr
    proof_root: StrictStr
    batch_pairs: Literal[16] = 16
    minimum_free_bytes: StrictInt = Field(ge=8 * 1024 * 1024 * 1024)
    seed: Literal[0] = 0
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDME1RuntimeExporterConfigV1:
        expected_schema = {
            "ddm_e1_runtime_exporter_n600_20260723": CONFIG_SCHEMA,
            "ddm_e2_pose_stream_and_doctrine_export_20260723": E2_CONFIG_SCHEMA,
            "ddm_e3_inflate_compose_and_depclose_20260723": E3_CONFIG_SCHEMA,
            "ddm_e4_brotli_declared_dep_20260724": E4_CONFIG_SCHEMA,
        }[self.run_id]
        if self.schema_ != expected_schema:
            raise ValueError("run_id and exporter config schema disagree")
        if Path(self.source_archive_path).is_absolute():
            raise ValueError("source_archive_path must be repository-relative")
        if Path(self.output_directory).is_absolute():
            raise ValueError("output_directory must be repository-relative")
        if not Path(self.proof_root).is_absolute():
            raise ValueError("proof_root must be absolute SSD custody")
        if not (
            self.proof_root.startswith("/Volumes/VertigoDataTier/pact/")
            or self.proof_root.startswith("/Volumes/APDataStore/pact/")
        ):
            raise ValueError("proof_root must use the governed SSD waterfall")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def compile_argv(self, config_path: str) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/export_ddm_runtime.py",
            "--config",
            config_path,
        )


class DDME4WS1GrammarStreamConfigV1(BaseModel):
    """Config-side declaration of one contiguous WS1 grammar stream."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    bytes: StrictInt = Field(gt=0)
    name: StrictStr = Field(min_length=1)
    offset: StrictInt = Field(ge=0)
    receiver_consumer: StrictStr = Field(min_length=8)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class DDME4WS1RuntimeExporterConfigV1(BaseModel):
    """Explicit typed admission route for receiver-closed WS1/J5 states."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDME4WS1RuntimeExporterConfigV1"] = Field(
        default=E4_WS1_CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_e5_e4_ws1_exporter_adapter_20260724"] = "ddm_e5_e4_ws1_exporter_adapter_20260724"
    candidate: Literal["W_seg", "W_joint"]
    grammar_version: Literal["ddm_ws1_receiver_closed_warm_start.v1"] = WS1_GRAMMAR_VERSION
    source_archive_path: StrictStr
    source_archive_bytes: StrictInt = Field(gt=0)
    source_archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stream_manifest: list[DDME4WS1GrammarStreamConfigV1] = Field(min_length=2)
    state_name: StrictStr = Field(min_length=8)
    output_directory: StrictStr
    proof_root: StrictStr
    batch_pairs: Literal[32] = 32
    minimum_free_bytes: StrictInt = Field(ge=8 * 1024 * 1024 * 1024)
    seed: Literal[0] = 0
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDME4WS1RuntimeExporterConfigV1:
        if len(self.stream_manifest) != 2:
            raise ValueError("WS1 grammar route requires exactly two outer streams")
        if not self.state_name.startswith(self.candidate + ":"):
            raise ValueError("state_name must begin with the typed candidate")
        source_path = Path(self.source_archive_path)
        if source_path.is_absolute() and not (
            self.source_archive_path.startswith("/Volumes/VertigoDataTier/pact/")
            or self.source_archive_path.startswith("/Volumes/APDataStore/pact/")
        ):
            raise ValueError("absolute WS1 source_archive_path must use governed SSD custody")
        if Path(self.output_directory).is_absolute():
            raise ValueError("output_directory must be repository-relative")
        if not Path(self.proof_root).is_absolute():
            raise ValueError("proof_root must be absolute SSD custody")
        if not (
            self.proof_root.startswith("/Volumes/VertigoDataTier/pact/")
            or self.proof_root.startswith("/Volumes/APDataStore/pact/")
        ):
            raise ValueError("proof_root must use the governed SSD waterfall")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def compile_argv(self, config_path: str) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/export_ddm_runtime.py",
            "--config",
            config_path,
        )


class DDMIC1RuntimeExporterConfigV1(DDME4WS1RuntimeExporterConfigV1):
    """Typed W_joint -> PA1 composition without widening the E5 route."""

    schema_: Literal["DDMIC1RuntimeExporterConfigV1"] = Field(
        default=IC1_CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_ic1_incumbent_compose_and_buy_row_20260724"] = "ddm_ic1_incumbent_compose_and_buy_row_20260724"
    candidate: Literal["W_joint"] = "W_joint"
    composition_order: Literal["W_joint_then_PA1_frame0"] = "W_joint_then_PA1_frame0"
    amplitude_transform: Literal["ddm_pa1_scorer_only_bn_inverse_frame0_receiver_v1"] = runtime.PA1_TRANSFORM_ID
    batch_pairs: Literal[16] = 16


class DDMIC2RuntimeExporterConfigV1(DDME4WS1RuntimeExporterConfigV1):
    """Typed W_seg -> PA1 incumbent composition without a W_joint fallback."""

    schema_: Literal["DDMIC2RuntimeExporterConfigV1"] = Field(
        default=IC2_CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_ic2_optimal_incumbent_pose_typed_20260724"] = (
        "ddm_ic2_optimal_incumbent_pose_typed_20260724"
    )
    candidate: Literal["W_seg"] = "W_seg"
    composition_order: Literal["W_seg_then_PA1_frame0"] = "W_seg_then_PA1_frame0"
    amplitude_transform: Literal["ddm_pa1_scorer_only_bn_inverse_frame0_receiver_v1"] = runtime.PA1_TRANSFORM_ID
    batch_pairs: Literal[16] = 16


RuntimeExporterConfig = (
    DDME1RuntimeExporterConfigV1
    | DDME4WS1RuntimeExporterConfigV1
    | DDMIC1RuntimeExporterConfigV1
    | DDMIC2RuntimeExporterConfigV1
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _require_repo_path(value: str, *, label: str) -> Path:
    resolved = (REPO_ROOT / value).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ExporterError(f"{label} escaped the repository") from exc
    return resolved


def _require_ws1_source_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return _require_repo_path(value, label="source_archive_path")
    resolved = path.resolve()
    if not (
        str(resolved).startswith("/Volumes/VertigoDataTier/pact/")
        or str(resolved).startswith("/Volumes/APDataStore/pact/")
    ):
        raise ExporterError("absolute WS1 source archive escaped governed SSD custody")
    return resolved


def _publish_or_verify(path: Path, payload: bytes, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ExporterError(f"refusing to overwrite differing artifact: {path}")
    else:
        temporary = path.with_name(path.name + f".partial.{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if executable:
        path.chmod(0o755)
    return path


def _e4_coder() -> Literal["brotli_q11", "lzma1_raw_d1m_lc3_lp0_pb2"]:
    """Select E4's primary coder, falling back only after import failure."""

    return BROTLI_Q11_CODER if brotli is not None else E3_LZMA1_CODER


def _ws1_e4_coder() -> Literal["brotli_q11"]:
    """Fail closed when WS1's inner grammar makes E4 fallback impossible."""

    coder = _e4_coder()
    if coder != BROTLI_Q11_CODER:
        raise ExporterError(
            "WS1 raw-LZMA1 fallback is inadmissible: the reconstructed "
            "receiver-closed source grammar itself contains Brotli-coded "
            "streams, so a Brotli ImportError cannot produce a dependency-"
            "closed decoder"
        )
    return BROTLI_Q11_CODER


def _compress_blob(
    raw: bytes,
    *,
    coder: Literal["brotli_q11", "lzma1_raw_d1m_lc3_lp0_pb2"],
) -> bytes:
    if coder == BROTLI_Q11_CODER:
        if brotli is None:
            raise ExporterError("Brotli coder selected after its import failed")
        return brotli.compress(raw, quality=11)
    if coder == E3_LZMA1_CODER:
        return lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=runtime.LZMA_FILTERS,
        )
    raise ExporterError(f"unknown blob coder: {coder}")


def _frame_blob(
    raw: bytes,
    *,
    kind: int,
    dimensions: tuple[int, ...] = (),
    coder: Literal[
        "brotli_q11",
        "lzma1_raw_d1m_lc3_lp0_pb2",
    ] = E3_LZMA1_CODER,
) -> bytes:
    if kind not in (0, 1) or len(dimensions) > 8:
        raise ExporterError("blob framing kind/rank is invalid")
    if kind == 0 and dimensions:
        raise ExporterError("opaque blob must be rank zero")
    product = 1
    for dimension in dimensions:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ExporterError("blob dimensions must be positive integers")
        product *= dimension
    if kind == 1 and product != len(raw):
        raise ExporterError("uint8 blob dimensions disagree with raw length")
    coded = _compress_blob(raw, coder=coder)
    return (
        BLOB_HEADER.pack(
            BLOB_MAGIC,
            1,
            CODER_IDS[coder],
            kind,
            len(dimensions),
            len(raw),
            len(coded),
            hashlib.sha256(raw).digest(),
        )
        + (struct.pack(f">{len(dimensions)}I", *dimensions) if dimensions else b"")
        + coded
    )


def _ordered_redundancy_matrix(
    raw_streams: dict[str, bytes],
    *,
    coder: Literal[
        "brotli_q11",
        "lzma1_raw_d1m_lc3_lp0_pb2",
    ] = E3_LZMA1_CODER,
) -> list[dict[str, Any]]:
    """Measure bytes(B)-bytes(B|A decoded) with the selected real coder."""

    if tuple(raw_streams) != EXPECTED_MEMBERS[1:]:
        raise ExporterError("redundancy streams are incomplete or reordered")
    standalone = {name: len(_compress_blob(payload, coder=coder)) for name, payload in raw_streams.items()}
    rows: list[dict[str, Any]] = []
    for conditioner, conditioner_payload in raw_streams.items():
        conditioner_bytes = standalone[conditioner]
        for stream, stream_payload in raw_streams.items():
            if conditioner == stream:
                continue
            conditioned = (
                len(
                    _compress_blob(
                        conditioner_payload + stream_payload,
                        coder=coder,
                    )
                )
                - conditioner_bytes
            )
            rows.append(
                {
                    "conditioned_bytes": conditioned,
                    "conditioner": conditioner,
                    "first_rung": True,
                    "redundancy_bytes": standalone[stream] - conditioned,
                    "standalone_bytes": standalone[stream],
                    "stream": stream,
                }
            )
    return rows


def _validate_rate_doctrine_manifest(value: Any) -> None:
    """Fail closed if any counted stream lacks one audit-triple row."""

    if (
        not isinstance(value, dict)
        or value.get("schema") != RATE_DOCTRINE_SCHEMA
        or [row.get("member") for row in value.get("streams", [])] != list(EXPECTED_MEMBERS[1:])
    ):
        raise ExporterError("four-clause stream audit is missing or reordered")
    triple_keys = {
        "scorer_visibility",
        "sensitivity_priced_tolerance",
        "three_layer_decomposition",
    }
    for row in value["streams"]:
        triple = row.get("audit_triple")
        if (
            not isinstance(triple, dict)
            or set(triple) != triple_keys
            or any(not isinstance(triple[key], dict) or not triple[key] for key in triple_keys)
            or row.get("first_rung") is not True
            or not isinstance(row.get("non_redundancy"), dict)
            or not row["non_redundancy"]
        ):
            raise ExporterError(f"incomplete stream audit: {row.get('member')}")
    expected = {(left, right) for left in EXPECTED_MEMBERS[1:] for right in EXPECTED_MEMBERS[1:] if left != right}
    matrix = value.get("ordered_redundancy_matrix")
    if (
        not isinstance(matrix, list)
        or {(row.get("conditioner"), row.get("stream")) for row in matrix if isinstance(row, dict)} != expected
    ):
        raise ExporterError("ordered redundancy matrix is incomplete")


def _rate_doctrine_manifest(
    *,
    chart_raw: bytes,
    chart_member: bytes,
    semantic_raw: bytes,
    semantic_member: bytes,
    coder: Literal[
        "brotli_q11",
        "lzma1_raw_d1m_lc3_lp0_pb2",
    ] = E3_LZMA1_CODER,
) -> dict[str, Any]:
    """Build the complete four-clause audit without promoting an adverse row."""

    matrix = _ordered_redundancy_matrix(
        {
            "base/chart.ddb": chart_raw,
            "semantic/composed.dds": semantic_raw,
        },
        coder=coder,
    )
    codec_label = "Brotli-Q11" if coder == BROTLI_Q11_CODER else "stdlib raw LZMA1 dict=1MiB lc=3 lp=0 pb=2"
    rows = [
        {
            "audit_triple": {
                "scorer_visibility": {
                    "authority_surfaces": [
                        "PoseNet:frame0+frame1",
                        "SegNet:frame1",
                    ],
                    "frame0_seg_facts": 0,
                    "instrument": ("DDMRuntimePerturbationV1 counted-to-output and output-to-single-owner checks"),
                    "status": "PARTIAL_STREAM_LEVEL_COORDINATE_FIELD_OWED",
                },
                "sensitivity_priced_tolerance": {
                    "current_quantization": "exact int16 chart coordinates",
                    "metric": ("Fisher-margin plus realized inner-Jacobian deltaS/byte"),
                    "score_byte_dual": "25/37545489",
                    "status": "BLOCKED_UNMEASURED_PER_COORDINATE_TOLERANCE",
                },
                "three_layer_decomposition": {
                    "coder": {
                        "coded_bytes": len(chart_member),
                        "codec": codec_label,
                        "raw_bytes": len(chart_raw),
                    },
                    "descriptive_form": (
                        "two-frame 12x16 RGB chart: anchors + axial gradients + conditioned residuals"
                    ),
                    "inherently_compact_dofs": {
                        "count": len(chart_raw) // 2,
                        "gauge_quotient": "not_yet_measured",
                        "storage_dtype": "int16",
                    },
                },
            },
            "candidate_admissible": False,
            "first_rung": True,
            "member": "base/chart.ddb",
            "non_redundancy": {
                "canonical_dimension_home": "pair x frame x 12x16 chart",
                "conditioned_on": [],
                "single_owner_facts": [
                    "base photometric chart",
                    "within-pair low-frequency appearance difference",
                ],
            },
            "verdict_scope": (
                "Current exact-int16 chart formulation only; the compact-chart "
                "family remains open pending receiver-closed tolerance rows."
            ),
        },
        {
            "audit_triple": {
                "scorer_visibility": {
                    "authority_surfaces": ["SegNet:frame1"],
                    "frame0_seg_facts": 0,
                    "receiver_policy": "frame1_only_seg_free_frame0",
                    "status": "STRUCTURAL_FRAME_HOME_PROVEN",
                },
                "sensitivity_priced_tolerance": {
                    "current_quantization": "exact categorical code per scorer cell",
                    "metric": ("rank4 flip-distance x Fisher margin x realized inner-Jacobian"),
                    "score_byte_dual": "25/37545489",
                    "status": "BLOCKED_UNMEASURED_BOUNDARY_TOLERANCE_FIELD",
                },
                "three_layer_decomposition": {
                    "coder": {
                        "coded_bytes": len(semantic_member),
                        "codec": codec_label,
                        "raw_bytes": len(semantic_raw),
                    },
                    "descriptive_form": ("one frame1-only categorical semantic plane reused at camera resolution"),
                    "inherently_compact_dofs": {
                        "count": len(semantic_raw),
                        "gauge_quotient": (
                            "frame0 semantic plane eliminated; region/boundary grammar factorization still owed"
                        ),
                        "storage_dtype": "uint8",
                    },
                },
            },
            "candidate_admissible": False,
            "first_rung": True,
            "member": "semantic/composed.dds",
            "non_redundancy": {
                "canonical_dimension_home": "pair x frame1 x scorer cell",
                "conditioned_on": ["base/chart.ddb"],
                "single_owner_facts": [
                    "semantic role assignment",
                    "role colour prototype reference",
                ],
            },
            "verdict_scope": (
                "Current exact-cell semantic formulation only; boundary/event "
                "descriptions and sensitivity-priced coarsening remain open."
            ),
        },
    ]
    result = {
        "candidate_admissible": all(bool(row["candidate_admissible"]) for row in rows)
        and all(int(row["redundancy_bytes"]) <= 0 for row in matrix),
        "correction_policy": {
            "counted_correction_streams": [],
            "description_owned_facts_reencoded": False,
            "status": "PASS_ZERO_BYTE_CORRECTION_STREAM",
        },
        "ordered_redundancy_matrix": matrix,
        "schema": RATE_DOCTRINE_SCHEMA,
        "single_owner_facts": [
            {
                "dimension_home": "pair x frame x 12x16 chart",
                "fact": "base photometric chart",
                "first_rung": True,
                "owner": "base/chart.ddb",
            },
            {
                "dimension_home": "pair x frame1 x scorer cell",
                "fact": "semantic role assignment",
                "first_rung": True,
                "owner": "semantic/composed.dds",
            },
        ],
        "streams": rows,
        "verdict_scope": (
            "E2 exported stream formulations only. Adverse audit rows block candidate admission, not the DDM family."
        ),
    }
    _validate_rate_doctrine_manifest(result)
    return result


def _pose_contract(receiver: Any) -> dict[str, Any]:
    pose6 = np.asarray(receiver.pose6_codes)
    if pose6.shape != (600, 6) or pose6.dtype != np.uint8:
        raise ExporterError("nested Pose6 custody changed")
    return {
        "classification": "ABSENT_FROM_COMPOSED_PACKET",
        "compact_inverse_status": "BLOCKED_NOT_PRESENT",
        "exact_lattice_control": {
            "archive_bytes": 409_526_925,
            "d_pose_n64": "0.000060022091887905524",
            "receipt": (".omx/research/mdl_polytope_member_solve_receipt_20260721.json"),
            "role": "high_byte_receiver_closed_control_not_pose_member",
        },
        "nested_pose6": {
            "bytes": int(pose6.nbytes),
            "consumption": "inter_pair_worldsheet_only_before_export",
            "dtype": "uint8",
            "sha256": _sha256(pose6.tobytes(order="C")),
            "shape": list(pose6.shape),
        },
        "packet_member": None,
        "receiver_bijection": {
            "counted_pose_to_output": "NOT_APPLICABLE_ZERO_PACKET_BYTES",
            "output_pose_effect_to_owner": "FAIL_NO_COMPACT_INVERSE_OWNER",
        },
        "verdict_scope": (
            "E1/E2 composed runtime packet boundary only. Exact two-plane "
            "lattice feasibility stands; compact code-to-photometry inversion "
            "is the missing production object."
        ),
    }


def _block_versions(
    *,
    chart_sha256: str,
    semantic_sha256: str,
    amplitude_enabled: bool = False,
    coder: Literal[
        "brotli_q11",
        "lzma1_raw_d1m_lc3_lp0_pb2",
    ] = E3_LZMA1_CODER,
) -> list[dict[str, Any]]:
    member_inputs = [
        {"member": "base/chart.ddb", "sha256": chart_sha256},
        {"member": "semantic/composed.dds", "sha256": semantic_sha256},
    ]
    active_rows = {
        "L": ("ddm_L_composed_semantic.v1", member_inputs),
        "D4": (
            "ddm_D4_brotli_q11_measure.v1" if coder == BROTLI_Q11_CODER else "ddm_D4_lzma1_raw_d1m_measure.v1",
            member_inputs,
        ),
        "D6": ("ddm_D6_camera_realization.v1", member_inputs),
    }
    if amplitude_enabled:
        active_rows["D1"] = (
            "ddm_D1_pa1_scorer_stat_affine_free.v1",
            member_inputs,
        )
    rows = []
    for block in REFINEMENT_CONTRACT["block_order"]:
        if block in active_rows:
            version, inputs = active_rows[block]
            rows.append(
                {
                    "block": block,
                    "consumption_time_policy": "verify_input_member_hashes_or_refuse",
                    "input_members": [dict(row) for row in inputs],
                    "status": "active",
                    "validity_horizon": {
                        "kind": "exact_input_member_hash_equality",
                        "reason": "byte_identity_requires_zero_untracked_input_drift",
                    },
                    "version": version,
                }
            )
        else:
            rows.append(
                {
                    "block": block,
                    "consumption_time_policy": "refuse_unstamped_activation",
                    "input_members": [],
                    "status": "inactive",
                    "validity_horizon": {
                        "kind": "inactive_no_cached_quantity",
                        "reason": "no_section_bytes_or_cached_state_consumed",
                    },
                    "version": f"ddm_{block}_inactive.v1",
                }
            )
    return rows


def _deterministic_zip(
    members: dict[str, bytes],
    *,
    expected_members: tuple[str, ...] = EXPECTED_MEMBERS,
) -> bytes:
    if tuple(members) != expected_members:
        raise ExporterError("runtime archive members are incomplete or reordered")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for name in expected_members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            info.extra = b""
            info.comment = b""
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def _zip_home_ledger(
    archive: bytes,
    *,
    expected_members: tuple[str, ...] = EXPECTED_MEMBERS,
) -> list[dict[str, Any]]:
    """Partition every stored-ZIP byte into one member-local or container home."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            infos = handle.infolist()
    except zipfile.BadZipFile as exc:
        raise ExporterError("runtime ZIP is malformed") from exc
    if [row.filename for row in infos] != list(expected_members):
        raise ExporterError("runtime ZIP member order changed")
    eocd_offset = archive.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or eocd_offset + ZIP_EOCD.size != len(archive):
        raise ExporterError("runtime ZIP lacks one terminal comment-free EOCD")
    eocd = ZIP_EOCD.unpack_from(archive, eocd_offset)
    central_bytes, central_offset = int(eocd[5]), int(eocd[6])
    if (
        eocd[0] != b"PK\x05\x06"
        or eocd[1:3] != (0, 0)
        or eocd[3] != len(infos)
        or eocd[4] != len(infos)
        or eocd[7] != 0
        or central_offset + central_bytes != eocd_offset
    ):
        raise ExporterError("runtime ZIP central-directory custody mismatch")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for info in infos:
        if info.header_offset != cursor:
            raise ExporterError("runtime ZIP has an unowned local-byte gap")
        local = ZIP_LOCAL_HEADER.unpack_from(archive, info.header_offset)
        if (
            local[0] != b"PK\x03\x04"
            or info.compress_type != zipfile.ZIP_STORED
            or info.date_time != (1980, 1, 1, 0, 0, 0)
            or info.extra
            or info.comment
        ):
            raise ExporterError("runtime ZIP local member is noncanonical")
        name_bytes, extra_bytes = int(local[-2]), int(local[-1])
        payload_start = info.header_offset + ZIP_LOCAL_HEADER.size + name_bytes + extra_bytes
        payload_end = payload_start + info.compress_size
        if (
            archive[
                info.header_offset + ZIP_LOCAL_HEADER.size : info.header_offset + ZIP_LOCAL_HEADER.size + name_bytes
            ].decode("ascii")
            != info.filename
            or info.compress_size != info.file_size
        ):
            raise ExporterError("runtime ZIP local identity mismatch")
        rows.append(
            {
                "home_bytes": payload_end - info.header_offset,
                "local_range": {
                    "bytes": payload_end - info.header_offset,
                    "start": info.header_offset,
                    "stop": payload_end,
                },
                "member": info.filename,
                "member_payload_range": {
                    "bytes": info.file_size,
                    "start": payload_start,
                    "stop": payload_end,
                },
                "owner": f"member:{info.filename}",
            }
        )
        cursor = payload_end
    if cursor != central_offset:
        raise ExporterError("runtime ZIP local ranges do not reach the central directory")
    rows.append(
        {
            "home_bytes": len(archive) - central_offset,
            "member": None,
            "owner": "container:central_directory_and_eocd",
            "range": {
                "bytes": len(archive) - central_offset,
                "start": central_offset,
                "stop": len(archive),
            },
        }
    )
    if sum(int(row["home_bytes"]) for row in rows) != len(archive):
        raise ExporterError("runtime ZIP byte homes are not bijective")
    return rows


def price_exact_runtime_packet(
    members: Mapping[str, bytes],
    *,
    parseback: Callable[[bytes], bool],
) -> ExactRuntimePacketPrice:
    """Price one complete E3 packet, including framing and ZIP container bytes.

    ``members`` must already contain the final framed payloads and a manifest
    whose hashes match them.  ``parseback`` is the public semantic gate: the
    caller must invoke the actual receiver/parser and return exact ``True``.
    Raw LZMA/member byte counts alone are intentionally not exposed as an
    admissible price.
    """

    if tuple(members) != EXPECTED_MEMBERS:
        raise ExporterError("runtime packet members are incomplete or reordered")
    normalized: dict[str, bytes] = {}
    for name in EXPECTED_MEMBERS:
        payload = members[name]
        if not isinstance(payload, bytes):
            raise ExporterError(f"runtime member {name!r} must be exact bytes")
        normalized[name] = payload
    archive = _deterministic_zip(normalized)
    homes = tuple(_zip_home_ledger(archive))
    if parseback(archive) is not True:
        raise ExporterError("runtime packet parse-back did not return exact True")
    member_payload_bytes = sum(len(normalized[name]) for name in EXPECTED_MEMBERS)
    return ExactRuntimePacketPrice(
        archive_bytes=len(archive),
        archive_sha256=_sha256(archive),
        member_payload_bytes=member_payload_bytes,
        container_bytes=len(archive) - member_payload_bytes,
        byte_home_ledger=homes,
    )


def price_exact_runtime_marginal(
    control_members: Mapping[str, bytes],
    candidate_members: Mapping[str, bytes],
    *,
    parseback: Callable[[bytes], bool],
) -> ExactRuntimeMarginalPrice:
    """Return exact candidate-minus-control E3 bytes after both parse-backs."""

    control = price_exact_runtime_packet(control_members, parseback=parseback)
    candidate = price_exact_runtime_packet(candidate_members, parseback=parseback)
    return ExactRuntimeMarginalPrice(
        control=control,
        candidate=candidate,
        delta_archive_bytes=candidate.archive_bytes - control.archive_bytes,
    )


def _runtime_cleanliness(runtime: bytes) -> dict[str, Any]:
    text = runtime.decode("utf-8", "strict")
    tree = ast.parse(text)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    allowed_roots = {
        "__future__",
        "base64",
        "collections.abc",
        "hashlib",
        "json",
        "lzma",
        "math",
        "numpy",
        "os",
        "pathlib",
        "shutil",
        "struct",
        "sys",
        "tac.optimization.ddm_cc3_mixed_coder_receiver",
        "tac.optimization.ddm_pc1_pose_stream",
        "time",
        "tac.optimization.ddm_ws1_warm_start",
        "typing",
        "brotli",
        "torch",
        "torch.nn.functional",
    }
    unexpected = sorted(name for name in imports if name not in allowed_roots)
    forbidden_tokens = sorted(
        token
        for token in (
            "safetensors",
            "lstars",
            "gt_poses",
            "ground_truth_argmax",
            SOURCE_SHA256,
            STATE_SHA256,
        )
        if token in text
    )
    long_hex_literals = sorted(set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", text)))
    if unexpected or forbidden_tokens or long_hex_literals:
        raise ExporterError(
            f"runtime cleanliness failed: imports={unexpected}, tokens={forbidden_tokens}, hex={long_hex_literals}"
        )
    bundle_active = b'WS1_SOURCE_BUNDLE_B85 = b""' not in runtime
    return {
        "allowed_dependency_roots": [
            "torch",
            "brotli",
            *(["embedded:tac.optimization.ddm_ws1_warm_start"] if bundle_active else []),
        ],
        "forbidden_tokens": forbidden_tokens,
        "imports": sorted(imports),
        "long_hex_literals": long_hex_literals,
        "runtime_sha256": _sha256(runtime),
        "status": "PASS",
    }


def _compile_seed_state(source_archive: bytes) -> tuple[bytes, Any, dict[str, Any]]:
    members, _ = parse_carrier_compose_archive(source_archive)
    source_receiver = receive_carrier_compose_archive(source_archive, verify_member_effects=False)
    lane_seeds = derive_lane_program_seeds(source_receiver)
    lane_records = tuple(row.counted_record() for row in lane_seeds)
    state_archive, _ = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=members[WORLDSHEET_G1_MEMBER],
        lane_programs=lane_records,
        realization_profile=source_receiver.realization_profile,
        realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id=source_receiver.realization_static_rule_id,
        scorer_solved_templates=source_receiver.scorer_solved_templates,
    )
    if (len(state_archive), _sha256(state_archive)) != (STATE_BYTES, STATE_SHA256):
        raise ExporterError("J2 seed-state archive differs from its sealed identity")
    g1 = lift_g1_movable_worldsheet(members[WORLDSHEET_G1_MEMBER])
    template_count = (
        0 if source_receiver.scorer_solved_templates is None else len(source_receiver.scorer_solved_templates.templates)
    )
    dof_count = 2 * len(g1.tracks) + 4 * len(lane_seeds) + 3 * template_count
    if dof_count != EXPECTED_DOF_COUNT:
        raise ExporterError(f"receiver-effective DOF count drifted: {dof_count}")
    receiver = receive_carrier_compose_archive(state_archive, verify_member_effects=False)
    return (
        state_archive,
        receiver,
        {
            "island_translation_dofs": 2 * len(g1.tracks),
            "lane_program_dofs": 4 * len(lane_seeds),
            "shared_template_dofs": 3 * template_count,
            "total": dof_count,
        },
    )


def _validate_templates(receiver: Any) -> None:
    bank = receiver.scorer_solved_templates
    if bank is None:
        return
    profile = receiver.realization_profile
    if profile is None:
        raise ExporterError("template state lacks the counted realization profile")
    for template in bank.templates:
        colour = bytes(int(value) for value in profile.colour_for(template.role))
        if template.patch_height != 1 or template.patch_width != 1 or template.rgb_u8 != colour:
            raise ExporterError(
                "current exporter materialization requires each shared template to equal its one-cell role colour"
            )


def _compose_semantic_state(
    receiver: Any,
    *,
    batch_pairs: int,
    semantic_frame_policy: Literal[
        "both_frames",
        "frame1_only_seg_free_frame0",
    ] = "both_frames",
    amplitude_enabled: bool = False,
) -> tuple[np.ndarray, int, str, dict[str, Any] | None]:
    if receiver.realization_profile is None:
        raise ExporterError("seed state lacks a counted realization profile")
    _validate_templates(receiver)
    labels = np.zeros((600, PAIR_H, PAIR_W), dtype=np.uint8)
    layer_by_role = {row.role: row for row in receiver.layers}
    if tuple(layer_by_role) != (
        "UndrivableBoundary",
        "Road",
        "Lane",
        "MyCar",
        "Movable",
    ):
        raise ExporterError("source role layer order drifted")

    def render_source_batch(indexes: tuple[int, ...]) -> np.ndarray:
        camera = receiver.render_camera_pairs(indexes)
        if semantic_frame_policy == "frame1_only_seg_free_frame0":
            from tac.through_r.resolution_chain import render_grid_to_camera_uint8

            base_grid = receiver.predictor.baseline.render_pairs(indexes)
            for local in range(len(indexes)):
                camera[local, 0] = render_grid_to_camera_uint8(base_grid[local, 0])
        return np.ascontiguousarray(camera)

    base_digest = hashlib.sha256()
    base_total = 0
    moment_rows: list[dict[str, Any]] = []
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        indexes = tuple(range(start, stop))
        semantic_cells = np.full((stop - start, PAIR_H, PAIR_W), -1, dtype=np.int16)
        for code, role in enumerate(REALIZATION_PAINT_ORDER, start=1):
            layer = layer_by_role[role]
            for local, pair_id in enumerate(indexes):
                mask = receiver._mask_for_layer(layer, pair_id, replace_g1_movable=True)
                labels[pair_id, mask] = code
                semantic_cells[local, mask] = ROLE_CLASS_IDS[role]
        if receiver.realization_static_rule_codes is not None:
            rules = receiver.realization_static_rule_codes
            source = rules // 5
            target = rules % 5
            for local, pair_id in enumerate(indexes):
                admitted = (rules >= 0) & (semantic_cells[local] == source)
                for target_id, class_role in enumerate(CLASS_ORDER):
                    target_mask = admitted & (target == target_id)
                    role = "UndrivableBoundary" if class_role == "Undrivable" else class_role
                    labels[pair_id, target_mask] = REALIZATION_PAINT_ORDER.index(role) + 1
        camera = render_source_batch(indexes)
        payload = camera.tobytes(order="C")
        base_digest.update(payload)
        base_total += len(payload)
        if amplitude_enabled:
            moment_rows.append(runtime._pose_moment_row(torch.from_numpy(camera)))
    expected = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    if base_total != expected:
        raise ExporterError("source receiver raw byte count differs from geometry")
    if not amplitude_enabled:
        return labels, base_total, base_digest.hexdigest(), None

    moments = runtime._merge_pose_moments(moment_rows)
    gain, bias = runtime._derive_pa1_affine(moments)
    composed_digest = hashlib.sha256()
    composed_total = 0
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        indexes = tuple(range(start, stop))
        corrected = runtime._apply_pa1_frame0_affine(
            torch.from_numpy(render_source_batch(indexes)),
            gain,
            bias,
        )
        payload = corrected.numpy().tobytes(order="C")
        composed_digest.update(payload)
        composed_total += len(payload)
    if composed_total != expected:
        raise ExporterError("PA1-composed raw byte count differs from geometry")
    amplitude = {
        "base_output": {
            "bytes": base_total,
            "sha256": base_digest.hexdigest(),
        },
        "bias_f32": bias.tolist(),
        "gain_f32": gain.tolist(),
        "moments": moments,
        "payload_bytes": 0,
        "rate_class": "FREE",
        "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
        "transform_id": runtime.PA1_TRANSFORM_ID,
    }
    return labels, composed_total, composed_digest.hexdigest(), amplitude


def _chart_payload(receiver: Any) -> tuple[bytes, dict[str, Any]]:
    baseline = receiver.predictor.baseline
    arrays = (
        ("anchors", np.asarray(baseline.anchors, dtype="<i2"), [600, 2, 3]),
        ("gradients", np.asarray(baseline.gradients, dtype="<i2"), [600, 2, 2, 3]),
        (
            "residuals",
            np.asarray(baseline.residuals, dtype="<i2"),
            [600, 2, 12, 16, 3],
        ),
    )
    chunks: list[bytes] = []
    rows: list[dict[str, Any]] = []
    offset = 0
    for name, array, shape in arrays:
        if list(array.shape) != shape:
            raise ExporterError(f"chart {name} shape drifted")
        payload = np.ascontiguousarray(array).tobytes(order="C")
        chunks.append(payload)
        rows.append({"name": name, "offset": offset, "shape": shape})
        offset += len(payload)
    raw = b"".join(chunks)
    if len(raw) != 1_404_000:
        raise ExporterError("chart raw length drifted")
    return raw, {"byteorder": "little", "dtype": "int16", "streams": rows}


_WS1_SOURCE_BUNDLE_CACHE: bytes | None = None
_RUNTIME_PYDANTIC_COMPAT = b'''"""Runtime-only Pydantic compatibility for receiver modules.

The bundled receiver executes dataclass decoders, not repository authoring
configs.  When Pydantic is unavailable, these inert definitions let the
authoring-only config class bodies load without adding an undeclared runtime
dependency.  Normal repository imports continue to use real Pydantic.
"""
try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        Field,
        StrictBool,
        StrictInt,
        StrictStr,
        ValidationError,
        field_validator,
        model_validator,
    )
except ImportError:
    class BaseModel:
        def __init__(self, **values):
            for name in getattr(type(self), "__annotations__", {}):
                if hasattr(type(self), name):
                    setattr(self, name, getattr(type(self), name))
            for name, value in values.items():
                setattr(self, name, value)

        def model_dump(self, **_values):
            return {
                name: getattr(self, name)
                for name in getattr(type(self), "__annotations__", {})
                if hasattr(self, name)
            }

        @classmethod
        def model_validate(cls, values, **_options):
            return cls(**values)

    class ValidationError(Exception):
        pass

    StrictBool = bool
    StrictInt = int
    StrictStr = str

    def ConfigDict(**values):
        return values

    def Field(default=None, *, default_factory=None, **_values):
        return default_factory() if default_factory is not None else default

    def _identity_decorator(*_args, **_kwargs):
        return lambda value: value

    field_validator = _identity_decorator
    model_validator = _identity_decorator
'''


def _local_module_path(module_name: str) -> Path | None:
    relative = Path(*module_name.split("."))
    module_path = REPO_ROOT / "src" / relative.with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_path = REPO_ROOT / "src" / relative / "__init__.py"
    return package_path if package_path.is_file() else None


def _ws1_runtime_source_bundle() -> bytes:
    """Build a deterministic source ZIP for the generic WS1 receiver closure."""

    global _WS1_SOURCE_BUNDLE_CACHE
    if _WS1_SOURCE_BUNDLE_CACHE is not None:
        return _WS1_SOURCE_BUNDLE_CACHE
    queue = [
        "tac.optimization.ddm_cc3_mixed_coder_receiver",
        "tac.optimization.ddm_ws1_warm_start",
    ]
    visited: set[str] = set()
    modules: dict[str, Path] = {}
    while queue:
        module_name = queue.pop()
        if module_name in visited:
            continue
        visited.add(module_name)
        path = _local_module_path(module_name)
        if path is None:
            continue
        if path.name == "__init__.py":
            # Runtime packages are namespace containers.  Executing the
            # repository's convenience re-export initializers would pull in
            # unrelated campaign modules that this receiver never consumes.
            continue
        modules[module_name] = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = module_name.split(".")
                    parent = parts[: -node.level]
                    base = ".".join(parent + (node.module or "").split("."))
                elif node.module:
                    base = node.module
                else:
                    base = ""
                if base:
                    imported.append(base)
                    imported.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            queue.extend(name for name in imported if name.startswith("tac"))
    paths = set(modules.values())
    package_names: set[str] = set()
    for path in paths:
        relative = path.relative_to(REPO_ROOT / "src")
        parts = relative.parts[:-1]
        for index in range(1, len(parts) + 1):
            package_names.add("/".join(parts[:index]) + "/__init__.py")
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(package_names):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(
                info,
                b"# Deliberately empty runtime namespace package.\\n",
            )
        compat_info = zipfile.ZipInfo(
            "tac/runtime_pydantic_compat.py",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        compat_info.compress_type = zipfile.ZIP_DEFLATED
        compat_info.external_attr = 0o100644 << 16
        compat_info.create_system = 3
        archive.writestr(compat_info, _RUNTIME_PYDANTIC_COMPAT)
        for path in sorted(paths, key=lambda row: row.relative_to(REPO_ROOT / "src").as_posix()):
            name = path.relative_to(REPO_ROOT / "src").as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            source = path.read_bytes().replace(
                b"from pydantic import",
                b"from tac.runtime_pydantic_compat import",
            )
            archive.writestr(info, source)
    _WS1_SOURCE_BUNDLE_CACHE = buffer.getvalue()
    return _WS1_SOURCE_BUNDLE_CACHE


def _ws1_runtime_payload() -> bytes:
    source = RUNTIME_SOURCE.read_bytes()
    marker = b'WS1_SOURCE_BUNDLE_B85 = b""'
    if source.count(marker) != 1:
        raise ExporterError("WS1 runtime source-bundle marker changed")
    encoded = base64.b85encode(_ws1_runtime_source_bundle())
    return source.replace(marker, b"WS1_SOURCE_BUNDLE_B85 = " + repr(encoded).encode("ascii"))


def cc3_runtime_payload() -> bytes:
    """Return the canonical E3/E4/E5 runtime with the CC3 receiver bridge."""

    payload = _ws1_runtime_payload()
    _runtime_cleanliness(payload)
    return payload


def _ws1_grammar_state(
    source_archive: bytes,
    config: DDME4WS1RuntimeExporterConfigV1,
) -> tuple[Any, ReceiverGrammarAdmission, dict[str, int]]:
    """Strictly parse one typed WS1 state without any literal SHA allowlist."""

    from tac.optimization.ddm_ws1_warm_start import (
        parse_ws1_warm_start_archive,
        receive_ws1_warm_start_archive,
    )

    parsed = parse_ws1_warm_start_archive(source_archive)
    if parsed.exact_reemit() != source_archive:
        raise ExporterError("WS1 source parse/re-emit changed archive bytes")
    if parsed.candidate != config.candidate:
        raise ExporterError("WS1 parsed candidate differs from typed config")
    payload_consumer = (
        "apply_temporal_affine+reassert_frame1"
        if config.candidate == "W_seg"
        else "signed_distance_geometry+apply_local_statistics"
    )
    derived_streams = (
        ReceiverGrammarStream(
            name="nested_preuint8_archive",
            offset=0,
            bytes=len(parsed.base_archive),
            sha256=_sha256(parsed.base_archive),
            receiver_consumer="receive_preuint8_q8_archive",
        ),
        ReceiverGrammarStream(
            name="warm_start_payload",
            offset=len(parsed.base_archive),
            bytes=len(parsed.payload),
            sha256=_sha256(parsed.payload),
            receiver_consumer=payload_consumer,
        ),
    )
    configured = ReceiverGrammarAdmission(
        grammar_version=config.grammar_version,
        archive_bytes=config.source_archive_bytes,
        archive_sha256=config.source_archive_sha256,
        streams=tuple(ReceiverGrammarStream.from_dict(row.model_dump(mode="json")) for row in config.stream_manifest),
    )
    configured.verify_archive(source_archive)
    derived = ReceiverGrammarAdmission(
        grammar_version=config.grammar_version,
        archive_bytes=len(source_archive),
        archive_sha256=_sha256(source_archive),
        streams=derived_streams,
    )
    if configured != derived:
        raise ExporterError("configured WS1 grammar streams differ from parser derivation")

    carrier_members, _ = parse_carrier_compose_archive(parsed.carrier_archive)
    carrier_receiver = receive_carrier_compose_archive(
        parsed.carrier_archive,
        verify_member_effects=False,
    )
    g1 = lift_g1_movable_worldsheet(carrier_members[WORLDSHEET_G1_MEMBER])
    lane_seeds = derive_lane_program_seeds(carrier_receiver)
    template_count = (
        0
        if carrier_receiver.scorer_solved_templates is None
        else len(carrier_receiver.scorer_solved_templates.templates)
    )
    dofs = {
        "island_translation_dofs": 2 * len(g1.tracks),
        "lane_program_dofs": 4 * len(lane_seeds),
        "shared_template_dofs": 3 * template_count,
    }
    dofs["total"] = sum(dofs.values())
    if dofs["total"] != EXPECTED_DOF_COUNT:
        raise ExporterError(f"WS1 receiver-effective DOF count drifted: {dofs['total']}")
    return receive_ws1_warm_start_archive(source_archive), derived, dofs


def _measure_ws1_output_identity(
    receiver: Any,
    *,
    batch_pairs: int,
    checkpoint_root: Path,
    state_sha256: str,
) -> tuple[int, str, dict[str, Any]]:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected_total = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    free_bytes = shutil.disk_usage(checkpoint_root).free
    if free_bytes < expected_total + 256 * 1024 * 1024:
        raise ExporterError("WS1 output-identity checkpoint storage preflight failed")
    digest = hashlib.sha256()
    total = 0
    stage_rows: list[dict[str, Any]] = []
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        raw_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.raw"
        row_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.json"
        expected_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        if row_path.is_file() and raw_path.is_file():
            row = json.loads(row_path.read_bytes())
            if (
                row
                != {
                    "bytes": row.get("bytes"),
                    "pair_start": start,
                    "pair_stop": stop,
                    "sha256": row.get("sha256"),
                    "state_sha256": state_sha256,
                }
                or row["bytes"] != expected_bytes
                or _sha256_file(raw_path) != (row["bytes"], row["sha256"])
            ):
                raise ExporterError("WS1 output-identity checkpoint differs")
        elif row_path.exists() or raw_path.exists():
            raise ExporterError("WS1 output-identity checkpoint is incomplete")
        else:
            camera = receiver.render_camera_pairs(tuple(range(start, stop)))
            if (
                camera.dtype != np.uint8
                or camera.shape
                != (
                    stop - start,
                    FRAMES_PER_PAIR,
                    CAMERA_H,
                    CAMERA_W,
                    CHANNELS,
                )
                or not camera.flags.c_contiguous
            ):
                raise ExporterError("WS1 receiver output geometry or dtype changed")
            payload = camera.tobytes(order="C")
            temporary = raw_path.with_name(raw_path.name + f".partial.{os.getpid()}")
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, raw_path)
            row = {
                "bytes": len(payload),
                "pair_start": start,
                "pair_stop": stop,
                "sha256": _sha256(payload),
                "state_sha256": state_sha256,
            }
            _publish_or_verify(
                row_path,
                rfc8785_canonicalize(row) + b"\n",
            )
        with raw_path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        stage_rows.append(row)
    if total != expected_total:
        raise ExporterError("WS1 receiver output byte count differs from geometry")
    return (
        total,
        digest.hexdigest(),
        {
            "all_stage_checkpoints_preserved": True,
            "checkpoint_root": str(checkpoint_root),
            "stage_count": len(stage_rows),
        },
    )


def _measure_ws1_pa1_output_identity(
    receiver: Any,
    *,
    batch_pairs: int,
    checkpoint_root: Path,
    state_sha256: str,
) -> tuple[int, str, dict[str, Any], dict[str, Any]]:
    """Compose the sealed PA1 frame-0 transform on one WS1 receiver."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected_total = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    free_bytes = shutil.disk_usage(checkpoint_root).free
    if free_bytes < expected_total * 2 + 256 * 1024 * 1024:
        raise ExporterError("IC1 base/composed checkpoint storage preflight failed")
    composition_binding = _sha256((state_sha256 + "\0" + runtime.PA1_TRANSFORM_ID).encode("ascii"))
    base_rows: list[dict[str, Any]] = []
    base_paths: list[Path] = []
    moment_rows: list[dict[str, Any]] = []
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        base_path = checkpoint_root / f"base_pairs_{start:04d}_{stop:04d}.raw"
        base_state_path = checkpoint_root / f"base_pairs_{start:04d}_{stop:04d}.json"
        expected_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        row = runtime._load_preserved_stage(
            stage_path=base_path,
            state_path=base_state_path,
            manifest_sha256=state_sha256,
            start=start,
            stop=stop,
            expected_bytes=expected_bytes,
        )
        if row is None:
            camera = receiver.render_camera_pairs(tuple(range(start, stop)))
            rendered = torch.from_numpy(camera)
            if (
                rendered.dtype != torch.uint8
                or tuple(rendered.shape)
                != (
                    stop - start,
                    FRAMES_PER_PAIR,
                    CAMERA_H,
                    CAMERA_W,
                    CHANNELS,
                )
                or not rendered.is_contiguous()
            ):
                raise ExporterError("IC1 source receiver rendered noncanonical camera bytes")
            row = runtime._write_or_adopt_rendered_stage(
                stage_path=base_path,
                state_path=base_state_path,
                rendered=rendered,
                manifest_sha256=state_sha256,
                start=start,
                stop=stop,
            )
        base_rows.append(row)
        base_paths.append(base_path)
        moment_rows.append(
            runtime._load_or_measure_pose_moments(
                stage_path=base_path,
                state_path=(checkpoint_root / f"base_pairs_{start:04d}_{stop:04d}.moments.json"),
                stage_row=row,
                manifest_sha256=composition_binding,
                start=start,
                stop=stop,
            )
        )
    gain, bias, amplitude = runtime._load_or_write_pa1_affine(
        path=checkpoint_root / "pa1_affine.json",
        manifest_sha256=composition_binding,
        moments=runtime._merge_pose_moments(moment_rows),
    )

    digest = hashlib.sha256()
    total = 0
    composed_rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, 600, batch_pairs)):
        stop = min(start + batch_pairs, 600)
        stage_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.raw"
        state_path = checkpoint_root / f"pairs_{start:04d}_{stop:04d}.json"
        expected_bytes = (stop - start) * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
        row = runtime._load_preserved_stage(
            stage_path=stage_path,
            state_path=state_path,
            manifest_sha256=composition_binding,
            start=start,
            stop=stop,
            expected_bytes=expected_bytes,
        )
        if row is None:
            corrected = runtime._apply_pa1_frame0_affine(
                runtime._load_stage_tensor(
                    stage_path=base_paths[index],
                    start=start,
                    stop=stop,
                ),
                gain,
                bias,
            )
            row = runtime._write_or_adopt_rendered_stage(
                stage_path=stage_path,
                state_path=state_path,
                rendered=corrected,
                manifest_sha256=composition_binding,
                start=start,
                stop=stop,
            )
        with stage_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        composed_rows.append(row)
    if total != expected_total:
        raise ExporterError("IC1 composed output byte count differs from geometry")
    return (
        total,
        digest.hexdigest(),
        {
            "all_stage_checkpoints_preserved": True,
            "base_stage_count": len(base_rows),
            "checkpoint_root": str(checkpoint_root),
            "composed_stage_count": len(composed_rows),
        },
        amplitude,
    )


def _inflate_sh(*, bootstrap_opencv: bool = False) -> bytes:
    bootstrap = (
        b"""# IC2 declares and bootstraps its two non-Torch runtime wheels.
if ! "$PYBIN" - <<'PY'
import brotli
import cv2
if cv2.__version__ != "4.11.0":
    raise ImportError(f"unexpected OpenCV runtime: {cv2.__version__}")
PY
then
  "$PYBIN" -m pip install --disable-pip-version-check --no-input --no-cache-dir \
    "Brotli==1.2.0" "opencv-python-headless==4.11.0.86"
fi
"$PYBIN" - <<'PY'
import brotli
import cv2
if cv2.__version__ != "4.11.0":
    raise ImportError(f"IC2 OpenCV bootstrap failed: {cv2.__version__}")
PY
"""
        if bootstrap_opencv
        else b""
    )
    return (
        b"""#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The archive manifest is the runtime-tree dependency declaration authority.
# PYTHON selects the locked environment provisioned from that declaration.
PYBIN="${PYTHON:-python3}"
"""
        + bootstrap
        + b"""exec "$PYBIN" "$HERE/inflate.py" "$1" "$2" "$3"
"""
    )


def cc3_inflate_sh() -> bytes:
    """Reuse the declared E3/E4/E5 locked-environment launcher for CC3."""

    return _inflate_sh()


def export_ws1_runtime(
    config: DDME4WS1RuntimeExporterConfigV1 | DDMIC1RuntimeExporterConfigV1 | DDMIC2RuntimeExporterConfigV1,
    *,
    config_path: Path,
) -> tuple[dict[str, Any], Path]:
    """Export one typed WS1/J5 state through E4's exact coder path."""

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    is_ic1 = isinstance(config, DDMIC1RuntimeExporterConfigV1)
    is_ic2 = isinstance(config, DDMIC2RuntimeExporterConfigV1)
    amplitude_enabled = is_ic1 or is_ic2
    coder = _ws1_e4_coder()
    source_path = _require_ws1_source_path(config.source_archive_path)
    output_dir = _require_repo_path(
        config.output_directory,
        label="output_directory",
    )
    proof_root = Path(config.proof_root)
    proof_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(proof_root).free
    if free_bytes < config.minimum_free_bytes:
        raise ExporterError(f"storage preflight failed: {free_bytes} < {config.minimum_free_bytes}")
    source_archive = source_path.read_bytes()
    if (len(source_archive), _sha256(source_archive)) != (
        config.source_archive_bytes,
        config.source_archive_sha256,
    ):
        raise ExporterError("typed WS1 source archive custody mismatch")
    receiver, grammar_admission, dofs = _ws1_grammar_state(
        source_archive,
        config,
    )
    amplitude: dict[str, Any] | None = None
    if amplitude_enabled:
        (
            output_bytes,
            output_sha256,
            output_identity_resume,
            amplitude,
        ) = _measure_ws1_pa1_output_identity(
            receiver,
            batch_pairs=config.batch_pairs,
            checkpoint_root=(proof_root / "output_identity" / f"{config.source_archive_sha256}.pa1_frame0"),
            state_sha256=config.source_archive_sha256,
        )
    else:
        (
            output_bytes,
            output_sha256,
            output_identity_resume,
        ) = _measure_ws1_output_identity(
            receiver,
            batch_pairs=config.batch_pairs,
            checkpoint_root=(proof_root / "output_identity" / config.source_archive_sha256),
            state_sha256=config.source_archive_sha256,
        )
    dependencies = (
        ["numpy", "torch", "brotli", "cv2"]
        if is_ic2
        else ["numpy", "scipy", "torch", "brotli"]
    )
    state_member = _frame_blob(source_archive, kind=0, coder=coder)
    state_member_sha256 = _sha256(state_member)
    manifest: dict[str, Any] = {
        "dependencies": dependencies,
        "false_authority": {
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        },
        "geometry": {
            "camera_hw": [CAMERA_H, CAMERA_W],
            "channels": CHANNELS,
            "frames_per_pair": FRAMES_PER_PAIR,
            "pair_count": 600,
            "scorer_hw": [PAIR_H, PAIR_W],
        },
        "grammar_admission": grammar_admission.to_dict(),
        "output": {
            "bytes": output_bytes,
            "sha256": output_sha256,
        },
        "schema": IC2_SCHEMA if is_ic2 else IC1_SCHEMA if is_ic1 else E4_WS1_SCHEMA,
        "sections": [
            {
                "bytes": len(state_member),
                "member": "state/ws1.ddj5",
                "sha256": state_member_sha256,
                "typed_stream_tag": TypedStreamTag(
                    type=StreamType.SKELETON,
                    layer_home=LayerHome.L1_PROGRAM,
                    evaluate_py_recursion_level_cited=("L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"),
                    counted_bytes=len(state_member),
                    free_receiver_code=True,
                ).to_dict(),
            }
        ],
        "state": {
            "batch_pairs": config.batch_pairs,
            "name": config.state_name,
            "receiver_effective_dofs": dofs["total"],
        },
    }
    if amplitude_enabled:
        manifest["amplitude_transform"] = {
            "application_frame": 0,
            "composition_order": config.composition_order,
            "payload_bytes": 0,
            "rate_class": "FREE",
            "target_derivation": ("frozen_posenet_first_stem_conv_and_bn_only_video_independent"),
            "transform_id": config.amplitude_transform,
        }
    members = {
        "manifest.json": rfc8785_canonicalize(manifest),
        "state/ws1.ddj5": state_member,
    }
    archive = _deterministic_zip(
        members,
        expected_members=EXPECTED_WS1_MEMBERS,
    )
    if (
        _deterministic_zip(
            members,
            expected_members=EXPECTED_WS1_MEMBERS,
        )
        != archive
    ):
        raise ExporterError("WS1 runtime archive compiler is nondeterministic")
    homes = _zip_home_ledger(
        archive,
        expected_members=EXPECTED_WS1_MEMBERS,
    )
    with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
        parsed_members = {name: handle.read(name) for name in EXPECTED_WS1_MEMBERS}
    parsed_manifest = runtime._validate_ws1_manifest(
        json.loads(parsed_members["manifest.json"]),
        parsed_members,
    )
    reconstructed, consumed_streams = runtime._reconstruct_ws1_state(
        parsed_manifest,
        parsed_members,
    )
    if reconstructed != source_archive:
        raise ExporterError("WS1 packet parse-back changed source archive bytes")

    runtime_payload = _ws1_runtime_payload()
    cleanliness = _runtime_cleanliness(runtime_payload)
    script = _inflate_sh(bootstrap_opencv=is_ic2)
    archive_path = _publish_or_verify(output_dir / "archive.zip", archive)
    runtime_path = _publish_or_verify(
        output_dir / "inflate.py",
        runtime_payload,
        executable=True,
    )
    script_path = _publish_or_verify(
        output_dir / "inflate.sh",
        script,
        executable=True,
    )
    delta_bytes = len(archive) - len(source_archive)
    output_identity = {
        "bytes": output_bytes,
        "resume": output_identity_resume,
        "sha256": output_sha256,
        "status": (
            "IC2_W_SEG_THEN_PA1_SOURCE_EXACT_MEASURED_PACKAGED_RECEIVER_PENDING"
            if is_ic2
            else "IC1_W_JOINT_THEN_PA1_SOURCE_EXACT_MEASURED_PACKAGED_RECEIVER_PENDING"
            if is_ic1
            else "WS1_SOURCE_RECEIVER_EXACT_MEASURED_PACKAGED_RECEIVER_PENDING"
        ),
    }
    if amplitude_enabled:
        output_identity["amplitude_transform"] = amplitude
    result = {
        "archive": {
            "bytes": len(archive),
            "compiler_determinism_x2": True,
            "member_homes": homes,
            "member_order": list(EXPECTED_WS1_MEMBERS),
            "path": str(archive_path.relative_to(REPO_ROOT)),
            "receiver_byte_home_bijection": True,
            "sha256": _sha256(archive),
        },
        "cleanup": {
            "bulk_proof_root": str(proof_root),
            "certify_or_block": "no source or proof bytes deleted",
            "source_archive_mutated": False,
        },
        "config": config.model_dump(mode="json", by_alias=True),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "dofs": dofs,
        "evidence_axis": EVIDENCE_AXIS,
        "execution_allowed": False,
        "grammar_admission": {
            **grammar_admission.to_dict(),
            "packet_parseback_source_byte_identical": True,
            "streams_consumed": list(consumed_streams),
        },
        "output_identity": output_identity,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "rate": {
            "delta_s_rate_term_vs_unpacked": (25.0 * delta_bytes / 37_545_489),
            "packed_archive_bytes": len(archive),
            "packed_minus_unpacked_bytes": delta_bytes,
            "unpacked_state_bytes": len(source_archive),
        },
        "research_only": True,
        "runtime": {
            "cleanliness": cleanliness,
            "coder": {
                "codec_id": CODER_IDS[coder],
                "fallback_trigger": ("ImportError" if coder == E3_LZMA1_CODER else None),
                "primary": BROTLI_Q11_CODER,
                "selected": coder,
            },
            "dependencies": dependencies,
            "inflate_py": {
                "bytes": len(runtime_payload),
                "path": str(runtime_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(runtime_payload),
            },
            "inflate_sh": {
                "bytes": len(script),
                "path": str(script_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(script),
            },
            "source_bundle": {
                "bytes": len(_ws1_runtime_source_bundle()),
                "sha256": _sha256(_ws1_runtime_source_bundle()),
            },
        },
        "schema": (
            IC2_RESULT_SCHEMA
            if is_ic2
            else IC1_RESULT_SCHEMA
            if is_ic1
            else E4_WS1_RESULT_SCHEMA
        ),
        "score_claim": False,
        "seed": config.seed,
        "source": {
            "bytes": len(source_archive),
            "path": config.source_archive_path,
            "sha256": _sha256(source_archive),
        },
        "state": {
            "bytes": len(source_archive),
            "name": config.state_name,
            "sha256": _sha256(source_archive),
        },
        "storage_preflight": {
            "minimum_free_bytes": config.minimum_free_bytes,
            "proof_root": str(proof_root),
            "status": "PASS",
        },
        "typed_config_sha256": config.typed_config_hash(),
    }
    receipt_path = _publish_or_verify(
        output_dir.parent
        / (
            "ddm_ic2_runtime_export_receipt.json"
            if is_ic2
            else "ddm_ic1_runtime_export_receipt.json"
            if is_ic1
            else "ddm_e4_ws1_runtime_export_receipt.json"
        ),
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def export_runtime(
    config: DDME1RuntimeExporterConfigV1,
    *,
    config_path: Path,
) -> tuple[dict[str, Any], Path]:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    is_e2 = config.run_id == "ddm_e2_pose_stream_and_doctrine_export_20260723"
    is_e3 = config.run_id == "ddm_e3_inflate_compose_and_depclose_20260723"
    is_e4 = config.run_id == "ddm_e4_brotli_declared_dep_20260724"
    is_pose_stream = is_e2 or is_e3 or is_e4
    archive_schema = E4_SCHEMA if is_e4 else E3_SCHEMA if is_e3 else E2_SCHEMA if is_e2 else SCHEMA
    coder = _e4_coder() if is_e4 else E3_LZMA1_CODER
    dependencies = ["torch", "brotli"] if coder == BROTLI_Q11_CODER else ["torch"]
    semantic_frame_policy = "frame1_only_seg_free_frame0" if is_pose_stream else "both_frames"
    source_path = _require_repo_path(config.source_archive_path, label="source_archive_path")
    output_dir = _require_repo_path(config.output_directory, label="output_directory")
    proof_root = Path(config.proof_root)
    proof_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(proof_root).free
    if free_bytes < config.minimum_free_bytes:
        raise ExporterError(f"storage preflight failed: {free_bytes} < {config.minimum_free_bytes}")
    source_archive = source_path.read_bytes()
    if (len(source_archive), _sha256(source_archive)) != (
        config.source_archive_bytes,
        config.source_archive_sha256,
    ):
        raise ExporterError("sealed v15 source archive custody mismatch")

    state_archive, receiver, dofs = _compile_seed_state(source_archive)
    labels, raw_bytes, raw_sha256, amplitude_transform = _compose_semantic_state(
        receiver,
        batch_pairs=config.batch_pairs,
        semantic_frame_policy=semantic_frame_policy,
        amplitude_enabled=is_e3 or is_e4,
    )
    chart_raw, chart_layout = _chart_payload(receiver)
    chart_member = _frame_blob(chart_raw, kind=0, coder=coder)
    semantic_raw = labels.tobytes(order="C")
    semantic_member = _frame_blob(
        semantic_raw,
        kind=1,
        dimensions=tuple(labels.shape),
        coder=coder,
    )
    profile = receiver.realization_profile
    assert profile is not None
    palette = [[0, 0, 0], *[list(map(int, row)) for row in profile.role_rgb_u8]]
    chart_sha256 = _sha256(chart_member)
    semantic_sha256 = _sha256(semantic_member)
    paint_jacobian = semantic_paint_jacobian_summary(
        labels,
        palette,
        camera_hw=(CAMERA_H, CAMERA_W),
        frames_per_pair=FRAMES_PER_PAIR,
    )
    paint_jacobian["input_hashes"] = {
        "semantic_member_sha256": semantic_sha256,
        "state_archive_sha256": _sha256(state_archive),
    }
    paint_jacobian["validity_horizon"] = {
        "kind": "exact_input_hash_equality",
        "consumption_time_policy": "verify_or_rederive",
    }
    sections = [
        {
            "bytes": len(chart_member),
            "member": "base/chart.ddb",
            "sha256": chart_sha256,
            "typed_stream_tag": TypedStreamTag(
                type=StreamType.FIBER,
                layer_home=LayerHome.L2_CHART,
                evaluate_py_recursion_level_cited=("L2_chart -> L3_raster -> L4_scorer_feature -> L5_verdict"),
                counted_bytes=len(chart_member),
                free_receiver_code=True,
            ).to_dict(),
        },
        {
            "bytes": len(semantic_member),
            "member": "semantic/composed.dds",
            "sha256": semantic_sha256,
            "typed_stream_tag": TypedStreamTag(
                type=StreamType.SKELETON,
                layer_home=LayerHome.L1_PROGRAM,
                evaluate_py_recursion_level_cited=("L1_program -> L3_raster -> L4_scorer_feature -> L5_verdict"),
                counted_bytes=len(semantic_member),
                free_receiver_code=True,
            ).to_dict(),
        },
    ]
    manifest: dict[str, Any] = {
        "archive": {
            "source_bytes": len(source_archive),
            "source_sha256": _sha256(source_archive),
            "state_bytes": len(state_archive),
            "state_sha256": _sha256(state_archive),
        },
        "block_versions": _block_versions(
            chart_sha256=chart_sha256,
            semantic_sha256=semantic_sha256,
            amplitude_enabled=is_e3 or is_e4,
            coder=coder,
        ),
        "chart": chart_layout,
        "dependencies": dependencies,
        "false_authority": {
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        },
        "extension_slots": [dict(row) for row in EXTENSION_SLOTS],
        "geometry": {
            "camera_hw": [CAMERA_H, CAMERA_W],
            "chart_grid_hw": [12, 16],
            "chart_hw": [32, 32],
            "channels": CHANNELS,
            "frames_per_pair": FRAMES_PER_PAIR,
            "pair_count": 600,
            "scorer_hw": [PAIR_H, PAIR_W],
        },
        "language_version": LANGUAGE_VERSION,
        "output": {
            "bytes": raw_bytes,
            "palette_rgb_u8": palette,
            "sha256": raw_sha256,
        },
        "refinement": dict(REFINEMENT_CONTRACT),
        "schema": archive_schema,
        "sections": sections,
        "state": {
            "batch_pairs": config.batch_pairs,
            "name": config.state_name,
            "receiver_effective_dofs": dofs,
        },
    }
    if is_pose_stream:
        manifest.update(
            {
                "pose_contract": _pose_contract(receiver),
                "rate_doctrine": _rate_doctrine_manifest(
                    chart_raw=chart_raw,
                    chart_member=chart_member,
                    semantic_raw=semantic_raw,
                    semantic_member=semantic_member,
                    coder=coder,
                ),
                "semantic_frame_policy": semantic_frame_policy,
            }
        )
    manifest_payload = rfc8785_canonicalize(manifest)
    members = {
        "manifest.json": manifest_payload,
        "base/chart.ddb": chart_member,
        "semantic/composed.dds": semantic_member,
    }
    archive = _deterministic_zip(members)
    replay = _deterministic_zip(members)
    if replay != archive:
        raise ExporterError("runtime archive compiler is nondeterministic")
    homes = _zip_home_ledger(archive)
    if sum(int(row["home_bytes"]) for row in homes) != len(archive):
        raise ExporterError("runtime archive byte homes are not bijective")
    runtime_payload = RUNTIME_SOURCE.read_bytes()
    cleanliness = _runtime_cleanliness(runtime_payload)
    script = _inflate_sh()

    archive_path = _publish_or_verify(output_dir / "archive.zip", archive)
    runtime_path = _publish_or_verify(output_dir / "inflate.py", runtime_payload, executable=True)
    script_path = _publish_or_verify(output_dir / "inflate.sh", script, executable=True)
    result = {
        "archive": {
            "bytes": len(archive),
            "compiler_determinism_x2": True,
            "member_homes": homes,
            "member_order": list(EXPECTED_MEMBERS),
            "path": str(archive_path.relative_to(REPO_ROOT)),
            "receiver_byte_home_bijection": True,
            "sha256": _sha256(archive),
        },
        "cleanup": {
            "bulk_proof_root": str(proof_root),
            "certify_or_block": "no source or proof bytes deleted",
            "source_archive_mutated": False,
        },
        "config": config.model_dump(mode="json", by_alias=True),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "dofs": dofs,
        "evidence_axis": EVIDENCE_AXIS,
        "execution_allowed": False,
        "output_identity": {
            "bytes": raw_bytes,
            "sha256": raw_sha256,
            "status": (
                "E4_PA1_COMPOSED_SOURCE_MEASURED_PACKAGED_RECEIVER_PENDING"
                if is_e4
                else "E3_PA1_COMPOSED_SOURCE_MEASURED_PACKAGED_RECEIVER_PENDING"
                if is_e3
                else "E2_FRAME1_ONLY_SOURCE_MEASURED_PACKAGED_RECEIVER_PENDING"
                if is_e2
                else "SOURCE_RECEIVER_MEASURED_PACKAGED_RECEIVER_PENDING"
            ),
        },
        "pointer_moved": False,
        "paint_jacobian": paint_jacobian,
        "research_only": True,
        "runtime": {
            "cleanliness": cleanliness,
            "coder": {
                "codec_id": CODER_IDS[coder],
                "fallback_trigger": ("ImportError" if coder == E3_LZMA1_CODER and is_e4 else None),
                "primary": BROTLI_Q11_CODER,
                "selected": coder,
            },
            "dependencies": dependencies,
            "inflate_py": {
                "bytes": len(runtime_payload),
                "path": str(runtime_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(runtime_payload),
            },
            "inflate_sh": {
                "bytes": len(script),
                "path": str(script_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(script),
            },
        },
        "schema": (
            E4_RESULT_SCHEMA if is_e4 else E3_RESULT_SCHEMA if is_e3 else E2_RESULT_SCHEMA if is_e2 else RESULT_SCHEMA
        ),
        "score_claim": False,
        "seed": config.seed,
        "source": {
            "bytes": len(source_archive),
            "path": config.source_archive_path,
            "sha256": _sha256(source_archive),
        },
        "state": {
            "bytes": len(state_archive),
            "name": config.state_name,
            "sha256": _sha256(state_archive),
        },
        "storage_preflight": {
            "minimum_free_bytes": config.minimum_free_bytes,
            "proof_root": str(proof_root),
            "status": "PASS",
        },
        "typed_config_sha256": config.typed_config_hash(),
    }
    result["rate_partition"] = {
        "COUNTED": {
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
        },
        "FREE": {
            "bytes": 0,
            "objects": [runtime.PA1_TRANSFORM_ID] if is_e3 or is_e4 else [],
        },
        "NULL": {"blocks": ["D2", "D5"], "bytes": 0},
    }
    if is_pose_stream:
        result.update(
            {
                "pose_contract": manifest["pose_contract"],
                "rate_doctrine": manifest["rate_doctrine"],
            }
        )
    if is_e3 or is_e4:
        result["amplitude_transform"] = amplitude_transform
    receipt_path = _publish_or_verify(
        output_dir.parent
        / (
            "ddm_e4_runtime_export_receipt.json"
            if is_e4
            else "ddm_e3_runtime_export_receipt.json"
            if is_e3
            else "ddm_e2_runtime_export_receipt.json"
            if is_e2
            else "ddm_e1_runtime_export_receipt.json"
        ),
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def load_config(path: Path) -> RuntimeExporterConfig:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExporterError("exporter config is malformed JSON") from exc
    canonical = rfc8785_canonicalize(value) + b"\n"
    if payload != canonical:
        raise ExporterError("exporter config must be canonical JSON plus one newline")
    if value.get("schema") == IC2_CONFIG_SCHEMA:
        return DDMIC2RuntimeExporterConfigV1.model_validate(value, strict=True)
    if value.get("schema") == IC1_CONFIG_SCHEMA:
        return DDMIC1RuntimeExporterConfigV1.model_validate(value, strict=True)
    if value.get("schema") == E4_WS1_CONFIG_SCHEMA:
        return DDME4WS1RuntimeExporterConfigV1.model_validate(value, strict=True)
    return DDME1RuntimeExporterConfigV1.model_validate(value, strict=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ExporterError("config path must be inside the repository") from exc
    config = load_config(config_path)
    if isinstance(
        config,
        (
            DDME4WS1RuntimeExporterConfigV1,
            DDMIC1RuntimeExporterConfigV1,
            DDMIC2RuntimeExporterConfigV1,
        ),
    ):
        result, receipt_path = export_ws1_runtime(
            config,
            config_path=config_path,
        )
    else:
        result, receipt_path = export_runtime(config, config_path=config_path)
    print(
        json.dumps(
            {
                "archive": result["archive"],
                "receipt_path": str(receipt_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "DDME1RuntimeExporterConfigV1",
    "DDME4WS1GrammarStreamConfigV1",
    "DDME4WS1RuntimeExporterConfigV1",
    "DDMIC1RuntimeExporterConfigV1",
    "DDMIC2RuntimeExporterConfigV1",
    "ExporterError",
    "export_runtime",
    "export_ws1_runtime",
    "load_config",
    "main",
]
