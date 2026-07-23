# SPDX-License-Identifier: MIT
"""Deterministic compiler from the sealed DDM v15/J2 state to a runtime packet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.canonical_equations.ddm_runtime_export_identity_20260723 import (
    semantic_paint_jacobian_summary,
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
RATE_DOCTRINE_SCHEMA = "ddm_four_clause_rate_doctrine.v1"
CONFIG_SCHEMA = "DDME1RuntimeExporterConfigV1"
E2_CONFIG_SCHEMA = "DDME2RuntimeExporterConfigV1"
RESULT_SCHEMA = "ddm_e1_runtime_export_receipt.v1"
E2_RESULT_SCHEMA = "ddm_e2_runtime_export_receipt.v1"
SOURCE_BYTES = 133_941
SOURCE_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
STATE_BYTES = 134_211
STATE_SHA256 = "3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd"
STATE_NAME = "v15_j2_lane_seed_theta0"
EXPECTED_DOF_COUNT = 368
EXPECTED_MEMBERS = ("manifest.json", "base/chart.ddb", "semantic/composed.dds")
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
    "block_order_policy": (
        "dependency_topological_then_coarse_to_fine_then_largest_marginal_first"
    ),
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
BLOB_MAGIC = b"DDE1B"
BLOB_HEADER = struct.Struct(">5sBBBBQQ32s")
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


class DDME1RuntimeExporterConfigV1(BaseModel):
    """Typed local-only Build #636 compiler program."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal[
        "DDME1RuntimeExporterConfigV1",
        "DDME2RuntimeExporterConfigV1",
    ] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal[
        "ddm_e1_runtime_exporter_n600_20260723",
        "ddm_e2_pose_stream_and_doctrine_export_20260723",
    ] = (
        "ddm_e1_runtime_exporter_n600_20260723"
    )
    source_archive_path: StrictStr
    source_archive_bytes: Literal[133941] = SOURCE_BYTES
    source_archive_sha256: Literal[
        "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
    ] = SOURCE_SHA256
    state_name: Literal["v15_j2_lane_seed_theta0"] = STATE_NAME
    state_archive_bytes: Literal[134211] = STATE_BYTES
    state_archive_sha256: Literal[
        "3d5ab9786cc3d3eedd9a5fd1d878aea8186fbcf450ffcb781862db63ac2ca0cd"
    ] = STATE_SHA256
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
        expected_schema = (
            E2_CONFIG_SCHEMA
            if self.run_id == "ddm_e2_pose_stream_and_doctrine_export_20260723"
            else CONFIG_SCHEMA
        )
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
        return _sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        )

    def compile_argv(self, config_path: str) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/export_ddm_runtime.py",
            "--config",
            config_path,
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


def _frame_blob(raw: bytes, *, kind: int, dimensions: tuple[int, ...] = ()) -> bytes:
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
    coded = brotli.compress(raw, quality=11)
    return (
        BLOB_HEADER.pack(
            BLOB_MAGIC,
            1,
            1,
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
) -> list[dict[str, Any]]:
    """Measure bytes(B)-bytes(B|A decoded) with one fixed Brotli-Q11 coder."""

    if tuple(raw_streams) != EXPECTED_MEMBERS[1:]:
        raise ExporterError("redundancy streams are incomplete or reordered")
    standalone = {
        name: len(brotli.compress(payload, quality=11))
        for name, payload in raw_streams.items()
    }
    rows: list[dict[str, Any]] = []
    for conditioner, conditioner_payload in raw_streams.items():
        conditioner_bytes = standalone[conditioner]
        for stream, stream_payload in raw_streams.items():
            if conditioner == stream:
                continue
            conditioned = (
                len(
                    brotli.compress(
                        conditioner_payload + stream_payload,
                        quality=11,
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
        or [row.get("member") for row in value.get("streams", [])]
        != list(EXPECTED_MEMBERS[1:])
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
            or any(
                not isinstance(triple[key], dict) or not triple[key]
                for key in triple_keys
            )
            or row.get("first_rung") is not True
            or not isinstance(row.get("non_redundancy"), dict)
            or not row["non_redundancy"]
        ):
            raise ExporterError(f"incomplete stream audit: {row.get('member')}")
    expected = {
        (left, right)
        for left in EXPECTED_MEMBERS[1:]
        for right in EXPECTED_MEMBERS[1:]
        if left != right
    }
    matrix = value.get("ordered_redundancy_matrix")
    if (
        not isinstance(matrix, list)
        or {
            (row.get("conditioner"), row.get("stream"))
            for row in matrix
            if isinstance(row, dict)
        }
        != expected
    ):
        raise ExporterError("ordered redundancy matrix is incomplete")


def _rate_doctrine_manifest(
    *,
    chart_raw: bytes,
    chart_member: bytes,
    semantic_raw: bytes,
    semantic_member: bytes,
) -> dict[str, Any]:
    """Build the complete four-clause audit without promoting an adverse row."""

    matrix = _ordered_redundancy_matrix(
        {
            "base/chart.ddb": chart_raw,
            "semantic/composed.dds": semantic_raw,
        }
    )
    rows = [
        {
            "audit_triple": {
                "scorer_visibility": {
                    "authority_surfaces": [
                        "PoseNet:frame0+frame1",
                        "SegNet:frame1",
                    ],
                    "frame0_seg_facts": 0,
                    "instrument": (
                        "DDMRuntimePerturbationV1 counted-to-output and "
                        "output-to-single-owner checks"
                    ),
                    "status": "PARTIAL_STREAM_LEVEL_COORDINATE_FIELD_OWED",
                },
                "sensitivity_priced_tolerance": {
                    "current_quantization": "exact int16 chart coordinates",
                    "metric": (
                        "Fisher-margin plus realized inner-Jacobian deltaS/byte"
                    ),
                    "score_byte_dual": "25/37545489",
                    "status": "BLOCKED_UNMEASURED_PER_COORDINATE_TOLERANCE",
                },
                "three_layer_decomposition": {
                    "coder": {
                        "coded_bytes": len(chart_member),
                        "codec": "Brotli-Q11",
                        "raw_bytes": len(chart_raw),
                    },
                    "descriptive_form": (
                        "two-frame 12x16 RGB chart: anchors + axial gradients "
                        "+ conditioned residuals"
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
                    "metric": (
                        "rank4 flip-distance x Fisher margin x realized "
                        "inner-Jacobian"
                    ),
                    "score_byte_dual": "25/37545489",
                    "status": "BLOCKED_UNMEASURED_BOUNDARY_TOLERANCE_FIELD",
                },
                "three_layer_decomposition": {
                    "coder": {
                        "coded_bytes": len(semantic_member),
                        "codec": "Brotli-Q11",
                        "raw_bytes": len(semantic_raw),
                    },
                    "descriptive_form": (
                        "one frame1-only categorical semantic plane reused at "
                        "camera resolution"
                    ),
                    "inherently_compact_dofs": {
                        "count": len(semantic_raw),
                        "gauge_quotient": (
                            "frame0 semantic plane eliminated; region/boundary "
                            "grammar factorization still owed"
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
        "candidate_admissible": all(
            bool(row["candidate_admissible"]) for row in rows
        )
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
            "E2 exported stream formulations only. Adverse audit rows block "
            "candidate admission, not the DDM family."
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
            "receipt": (
                ".omx/research/mdl_polytope_member_solve_receipt_20260721.json"
            ),
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


def _block_versions(*, chart_sha256: str, semantic_sha256: str) -> list[dict[str, Any]]:
    member_inputs = [
        {"member": "base/chart.ddb", "sha256": chart_sha256},
        {"member": "semantic/composed.dds", "sha256": semantic_sha256},
    ]
    active_rows = {
        "L": ("ddm_L_composed_semantic.v1", member_inputs),
        "D4": ("ddm_D4_brotli_q11_measure.v1", member_inputs),
        "D6": ("ddm_D6_camera_realization.v1", member_inputs),
    }
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


def _deterministic_zip(members: dict[str, bytes]) -> bytes:
    if tuple(members) != EXPECTED_MEMBERS:
        raise ExporterError("runtime archive members are incomplete or reordered")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for name in EXPECTED_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            info.extra = b""
            info.comment = b""
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def _zip_home_ledger(archive: bytes) -> list[dict[str, Any]]:
    """Partition every stored-ZIP byte into one member-local or container home."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
            infos = handle.infolist()
    except zipfile.BadZipFile as exc:
        raise ExporterError("runtime ZIP is malformed") from exc
    if [row.filename for row in infos] != list(EXPECTED_MEMBERS):
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
                info.header_offset
                + ZIP_LOCAL_HEADER.size : info.header_offset
                + ZIP_LOCAL_HEADER.size
                + name_bytes
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
        "hashlib",
        "json",
        "os",
        "pathlib",
        "shutil",
        "struct",
        "sys",
        "time",
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
            "runtime cleanliness failed: "
            f"imports={unexpected}, tokens={forbidden_tokens}, hex={long_hex_literals}"
        )
    return {
        "allowed_dependency_roots": ["torch", "brotli"],
        "forbidden_tokens": forbidden_tokens,
        "imports": sorted(imports),
        "long_hex_literals": long_hex_literals,
        "runtime_sha256": _sha256(runtime),
        "status": "PASS",
    }


def _compile_seed_state(source_archive: bytes) -> tuple[bytes, Any, dict[str, Any]]:
    members, _ = parse_carrier_compose_archive(source_archive)
    source_receiver = receive_carrier_compose_archive(
        source_archive, verify_member_effects=False
    )
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
        0
        if source_receiver.scorer_solved_templates is None
        else len(source_receiver.scorer_solved_templates.templates)
    )
    dof_count = 2 * len(g1.tracks) + 4 * len(lane_seeds) + 3 * template_count
    if dof_count != EXPECTED_DOF_COUNT:
        raise ExporterError(f"receiver-effective DOF count drifted: {dof_count}")
    receiver = receive_carrier_compose_archive(
        state_archive, verify_member_effects=False
    )
    return state_archive, receiver, {
        "island_translation_dofs": 2 * len(g1.tracks),
        "lane_program_dofs": 4 * len(lane_seeds),
        "shared_template_dofs": 3 * template_count,
        "total": dof_count,
    }


def _validate_templates(receiver: Any) -> None:
    bank = receiver.scorer_solved_templates
    if bank is None:
        return
    profile = receiver.realization_profile
    if profile is None:
        raise ExporterError("template state lacks the counted realization profile")
    for template in bank.templates:
        colour = bytes(int(value) for value in profile.colour_for(template.role))
        if (
            template.patch_height != 1
            or template.patch_width != 1
            or template.rgb_u8 != colour
        ):
            raise ExporterError(
                "current exporter materialization requires each shared template "
                "to equal its one-cell role colour"
            )


def _compose_semantic_state(
    receiver: Any,
    *,
    batch_pairs: int,
    semantic_frame_policy: Literal[
        "both_frames",
        "frame1_only_seg_free_frame0",
    ] = "both_frames",
) -> tuple[np.ndarray, int, str]:
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
    digest = hashlib.sha256()
    total = 0
    for start in range(0, 600, batch_pairs):
        stop = min(start + batch_pairs, 600)
        indexes = tuple(range(start, stop))
        semantic_cells = np.full((stop - start, PAIR_H, PAIR_W), -1, dtype=np.int16)
        for code, role in enumerate(REALIZATION_PAINT_ORDER, start=1):
            layer = layer_by_role[role]
            for local, pair_id in enumerate(indexes):
                mask = receiver._mask_for_layer(
                    layer, pair_id, replace_g1_movable=True
                )
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
                    role = (
                        "UndrivableBoundary"
                        if class_role == "Undrivable"
                        else class_role
                    )
                    labels[pair_id, target_mask] = (
                        REALIZATION_PAINT_ORDER.index(role) + 1
                    )
        camera = receiver.render_camera_pairs(indexes)
        if semantic_frame_policy == "frame1_only_seg_free_frame0":
            from tac.through_r.resolution_chain import render_grid_to_camera_uint8

            base_grid = receiver.predictor.baseline.render_pairs(indexes)
            for local in range(stop - start):
                camera[local, 0] = render_grid_to_camera_uint8(
                    base_grid[local, 0]
                )
        payload = np.ascontiguousarray(camera).tobytes(order="C")
        digest.update(payload)
        total += len(payload)
    expected = 600 * FRAMES_PER_PAIR * CAMERA_H * CAMERA_W * CHANNELS
    if total != expected:
        raise ExporterError("source receiver raw byte count differs from geometry")
    return labels, total, digest.hexdigest()


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


def _inflate_sh() -> bytes:
    return b"""#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/inflate.py" "$1" "$2" "$3"
"""


def export_runtime(
    config: DDME1RuntimeExporterConfigV1,
    *,
    config_path: Path,
) -> tuple[dict[str, Any], Path]:
    is_e2 = config.run_id == "ddm_e2_pose_stream_and_doctrine_export_20260723"
    archive_schema = E2_SCHEMA if is_e2 else SCHEMA
    semantic_frame_policy = (
        "frame1_only_seg_free_frame0" if is_e2 else "both_frames"
    )
    source_path = _require_repo_path(
        config.source_archive_path, label="source_archive_path"
    )
    output_dir = _require_repo_path(
        config.output_directory, label="output_directory"
    )
    proof_root = Path(config.proof_root)
    proof_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(proof_root).free
    if free_bytes < config.minimum_free_bytes:
        raise ExporterError(
            f"storage preflight failed: {free_bytes} < {config.minimum_free_bytes}"
        )
    source_archive = source_path.read_bytes()
    if (len(source_archive), _sha256(source_archive)) != (
        config.source_archive_bytes,
        config.source_archive_sha256,
    ):
        raise ExporterError("sealed v15 source archive custody mismatch")

    state_archive, receiver, dofs = _compile_seed_state(source_archive)
    labels, raw_bytes, raw_sha256 = _compose_semantic_state(
        receiver,
        batch_pairs=config.batch_pairs,
        semantic_frame_policy=semantic_frame_policy,
    )
    chart_raw, chart_layout = _chart_payload(receiver)
    chart_member = _frame_blob(chart_raw, kind=0)
    semantic_raw = labels.tobytes(order="C")
    semantic_member = _frame_blob(
        semantic_raw,
        kind=1,
        dimensions=tuple(labels.shape),
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
        },
        {
            "bytes": len(semantic_member),
            "member": "semantic/composed.dds",
            "sha256": semantic_sha256,
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
            chart_sha256=chart_sha256, semantic_sha256=semantic_sha256
        ),
        "chart": chart_layout,
        "dependencies": ["torch", "brotli"],
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
    if is_e2:
        manifest.update(
            {
                "pose_contract": _pose_contract(receiver),
                "rate_doctrine": _rate_doctrine_manifest(
                    chart_raw=chart_raw,
                    chart_member=chart_member,
                    semantic_raw=semantic_raw,
                    semantic_member=semantic_member,
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
    runtime = RUNTIME_SOURCE.read_bytes()
    cleanliness = _runtime_cleanliness(runtime)
    script = _inflate_sh()

    archive_path = _publish_or_verify(output_dir / "archive.zip", archive)
    runtime_path = _publish_or_verify(output_dir / "inflate.py", runtime, executable=True)
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
                "E2_FRAME1_ONLY_SOURCE_MEASURED_PACKAGED_RECEIVER_PENDING"
                if is_e2
                else "SOURCE_RECEIVER_MEASURED_PACKAGED_RECEIVER_PENDING"
            ),
        },
        "pointer_moved": False,
        "paint_jacobian": paint_jacobian,
        "research_only": True,
        "runtime": {
            "cleanliness": cleanliness,
            "dependencies": ["torch", "brotli"],
            "inflate_py": {
                "bytes": len(runtime),
                "path": str(runtime_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(runtime),
            },
            "inflate_sh": {
                "bytes": len(script),
                "path": str(script_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(script),
            },
        },
        "schema": E2_RESULT_SCHEMA if is_e2 else RESULT_SCHEMA,
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
    if is_e2:
        result.update(
            {
                "pose_contract": manifest["pose_contract"],
                "rate_doctrine": manifest["rate_doctrine"],
            }
        )
    receipt_path = _publish_or_verify(
        output_dir.parent
        / (
            "ddm_e2_runtime_export_receipt.json"
            if is_e2
            else "ddm_e1_runtime_export_receipt.json"
        ),
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def load_config(path: Path) -> DDME1RuntimeExporterConfigV1:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExporterError("exporter config is malformed JSON") from exc
    canonical = rfc8785_canonicalize(value) + b"\n"
    if payload != canonical:
        raise ExporterError("exporter config must be canonical JSON plus one newline")
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
    "ExporterError",
    "export_runtime",
    "load_config",
    "main",
]
