#!/usr/bin/env python3
"""Census exact PBR2 teacher debt against bounded, already-landed primitives.

This tool is an encoder-side falsifier/acquisition measurement.  It decodes the
exact counted V9 predictor and the exact PBR2 teacher packet, assigns every
predictor/teacher mismatch cell to one and only one PBR2 stratum, and reports
finite atom/byte accounting.  PBR2 and the reconstructed frozen target are
exhaustive teacher data and are forbidden from every candidate payload.

The 5.08 GB GT cache is deliberately not opened.  Its SHA-256 is cross-checked
between the sealed PBR2 materialization receipt and the sealed n600 partition
census receipt.  No RGB/scorer transition is run, so score value per byte is
reported as unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_dsl.factorized_v9_predictor import (  # noqa: E402
    PREDICTOR_CONTRACT_ID,
    receive_factorized_v9_predictor,
)
from tac.witness_dsl.progressive_geometry_residual import (  # noqa: E402
    apply_progressive_geometry_residual,
    decode_progressive_geometry_residual,
    packet_accounting,
)

SCHEMA = "tac.g1_teacher_atom_census.v1"
RECEIPT_SCHEMA = "tac.g1_teacher_atom_census_receipt.v1"
PAIR_START = 448
PAIR_COUNT = 64
HEIGHT = 384
WIDTH = 512
EXPECTED_PROGRAM_SHA256 = "56b563f2f9fb442508134bfb144eb1dc67a07675c93e0c56ec3a569f649bac9a"
EXPECTED_PBR2_SHA256 = "3372eee1d989012fb3293c7abe08eac233c874bf485e5ea15c5bd26d7306f0a1"
EXPECTED_PBR2_BYTES = 78_665
EXPECTED_GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_GT_CACHE_BYTES = 5_078_017_610

AUTHORITY_AXIS = "[research-only exact teacher semantic census; n64 formulation scope]"
VERDICT_SCOPE = "formulation: exact PBR2 teacher descriptiveness, not candidate acquisition or score"
CANDIDATE_LINEAGE_PROHIBITION = (
    "PBR1/PBR2, reconstructed target labels, owner masks, and target-derived events remain encoder-only "
    "teachers and MUST NOT enter P+G+A candidate bytes or generated decoder source"
)

RESEARCH_ROOT = REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
DEFAULT_PROGRAM = (
    REPO
    / ".omx/research/ddm_v9_carrier_compose_n64_603_613_20260722T122800Z"
    / "ddm_v9_carrier_compose_n64.not_a_candidate.zip.receipt-bytes"
)
DEFAULT_PBR2 = RESEARCH_ROOT / "c0b_pbr2_progressive_geometry_n64.pbr2"
DEFAULT_PBR2_RECEIPT = RESEARCH_ROOT / "c0b_pbr2_progressive_geometry_n64.json"
DEFAULT_N600_RECEIPT = RESEARCH_ROOT / "v9_target_partition_grammar_census_n600.json"
DEFAULT_V14_RECEIPT = (
    REPO
    / ".omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z"
    / "ddm_v14_realization_fidelity_n600_receipt.json"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "g1_teacher_atom_census_n64_20260726.json"

IMPLEMENTATION_PATHS = (
    "tools/measure_g1_teacher_atom_census.py",
    "tools/tests/test_measure_g1_teacher_atom_census.py",
    "src/tac/witness_dsl/factorized_v9_predictor.py",
    "src/tac/witness_dsl/progressive_geometry_residual.py",
    "src/tac/witness_dsl/c0b_counted_receiver_codec.py",
    "src/tac/witness_dsl/v10_two_plane_timing_receiver.py",
    "src/tac/boundary_math/legal_frame_bridge.py",
    "src/tac/boundary_math/road_undriv_bulk_field.py",
)

FAMILY_NAMES = (
    "same_coordinate_temporal_repeat",
    "connected_island_row_spans",
    "singleton_sparse_tail",
)
FORBIDDEN_EXHAUSTIVE_KEYS = frozenset(
    {
        "target_labels",
        "target_semantics",
        "target_label_bytes",
        "correction_masks",
        "pbr2_payload",
        "pbr2_raw_sections",
        "teacher_payload_base64",
        "target_payload_base64",
        "owner_masks_base64",
    }
)

BODY_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "authority_axis",
        "verdict_scope",
        "research_only",
        "candidate_payload_allowed",
        "score_claim",
        "promotion_eligible",
        "pointer_moved",
        "candidate_lineage_prohibition",
        "git_head",
        "semantic_argv",
        "inputs",
        "implementation_custody",
        "runtime_custody",
        "measurement",
        "teacher_packet_byte_ownership",
        "full_n600_teacher_grammar_crosslink",
        "v14_exact_anchor_dispositions",
        "remaining_debt",
        "storage_and_cleanup",
    }
)
MEASUREMENT_KEYS = frozenset(
    {
        "geometry",
        "total_semantic_cells",
        "predictor_debt_cells",
        "predictor_debt_fraction",
        "predictor_semantic_sha256",
        "teacher_semantic_sha256",
        "exclusive_owner_code_sha256",
        "owner_code_legend",
        "ownership_masks_materialized_in_receipt",
        "exclusive_ownership_closed",
        "pairwise_owner_intersection_cells",
        "families",
        "temporal_support",
        "palette_gauge_value_structure",
        "shared_vs_independent_plane_envelopes",
        "per_pair",
        "candidate_admissible_remaining_debt_cells",
        "teacher_reconstruction_remaining_debt_cells",
    }
)
FAMILY_KEYS = frozenset(
    {
        "family",
        "teacher_owned_cells",
        "teacher_coverage_numerator",
        "teacher_coverage_denominator",
        "teacher_coverage_fraction",
        "candidate_admissible_owned_cells",
        "candidate_admissible_parameter_bytes",
        "target_boundary_4_neighbor_cells",
        "target_interior_cells",
        "value_conditioned_row_span_atoms_derived",
        "packet_record_atoms_exact",
        "packet_span_atoms_exact",
        "canonical_raw_parameter_bytes_exact",
        "teacher_entropy_payload_bytes_exact",
        "teacher_entropy_codec",
        "raw_parameter_sha256",
        "teacher_payload_sha256",
        "errors_before",
        "errors_after",
        "predictor_to_teacher_transition_matrix",
        "per_pair_owned_cells",
        "score_value_per_byte",
        "score_value_per_byte_reason",
        "teacher_only",
    }
)


class CensusError(RuntimeError):
    """Fail-closed teacher-lineage, ownership, custody, or receipt error."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CensusError("receipt value is not finite canonical ASCII JSON") from exc


def sha256_bytes(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_stable_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read one small artifact generation and fail if it mutates during read."""

    supplied = Path(os.path.abspath(path))
    try:
        supplied_stat = supplied.lstat()
        if stat.S_ISLNK(supplied_stat.st_mode) or not stat.S_ISREG(supplied_stat.st_mode):
            raise CensusError(f"artifact must be a non-symlink regular file: {supplied}")
        descriptor = os.open(supplied, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(descriptor)
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = supplied.lstat()
    except OSError as exc:
        raise CensusError(f"cannot read required artifact: {supplied}") from exc
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_id = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    path_id = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if before_id != after_id or after_id != path_id or len(payload) != after.st_size:
        raise CensusError(f"artifact mutated while being read: {supplied}")
    return payload, {
        "path": _display_path(supplied),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def snapshot_file(path: Path) -> dict[str, Any]:
    """Hash a stable file without retaining its bytes."""

    resolved = path.resolve()
    try:
        before = resolved.stat()
        digest = sha256_file(resolved)
        after = resolved.stat()
    except OSError as exc:
        raise CensusError(f"cannot snapshot required file: {resolved}") from exc
    before_id = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_id = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_id != after_id:
        raise CensusError(f"file mutated while being hashed: {resolved}")
    return {"path": _display_path(resolved), "bytes": after.st_size, "sha256": digest}


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CensusError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CensusError(f"{label} must contain one JSON object")
    return value


def _assert_no_exhaustive_payload(value: Any, *, path: str = "body") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {
                "payload",
                "blob",
                "data",
                "base64",
                "encoded_payload",
                "raw_byte_values",
            }:
                raise CensusError(f"receipt contains an unowned generic payload field at {path}.{key}")
            if key in FORBIDDEN_EXHAUSTIVE_KEYS:
                raise CensusError(f"receipt attempts to retain exhaustive teacher payload at {path}.{key}")
            lowered = str(key).lower()
            if "base64" in lowered and any(
                marker in lowered for marker in ("teacher", "target", "owner", "correction", "pbr")
            ):
                raise CensusError(f"receipt attempts to alias exhaustive teacher payload at {path}.{key}")
            _assert_no_exhaustive_payload(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_exhaustive_payload(child, path=f"{path}[{index}]")


def _exact_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CensusError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise CensusError(f"{label} fields differ")


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise CensusError(f"{label} must be a nonnegative exact integer")
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    result = _exact_nonnegative_int(value, label)
    if result == 0:
        raise CensusError(f"{label} must be positive")
    return result


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _exact_transition_matrix(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 5:
        raise CensusError(f"{label} must be an exact 5x5 integer matrix")
    matrix: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 5:
            raise CensusError(f"{label} must be an exact 5x5 integer matrix")
        matrix.append([_exact_nonnegative_int(cell, label) for cell in row])
    return matrix


def _validate_receipt_body(body: Mapping[str, Any]) -> None:
    """Re-derive the authority, lineage, and closed census arithmetic."""

    _exact_keys(body, BODY_KEYS, "receipt body")
    required_authority = {
        "schema": SCHEMA,
        "purpose": "real-n64 predictor-conditioned teacher atom census over the exact committed V9 predictor and PBR2",
        "authority_axis": AUTHORITY_AXIS,
        "verdict_scope": VERDICT_SCOPE,
        "research_only": True,
        "candidate_payload_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "candidate_lineage_prohibition": CANDIDATE_LINEAGE_PROHIBITION,
    }
    for field, expected in required_authority.items():
        if canonical_json_bytes(body.get(field)) != canonical_json_bytes(expected):
            raise CensusError(f"receipt body {field} authority/lineage differs")
    git_head = body.get("git_head")
    if (
        not isinstance(git_head, str)
        or len(git_head) != 40
        or any(character not in "0123456789abcdef" for character in git_head)
    ):
        raise CensusError("receipt git_head is malformed")
    argv = body.get("semantic_argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise CensusError("receipt semantic_argv is incomplete")

    inputs = _exact_mapping(body.get("inputs"), "receipt inputs")
    _exact_keys(
        inputs,
        frozenset(
            {
                "predictor_program",
                "pbr2_teacher_packet",
                "pbr2_materialization_receipt",
                "n600_partition_grammar_receipt",
                "v14_receiver_closed_anchor_receipt",
                "gt_cache_custody_inherited_without_bulk_read",
                "predictor_renderer_source_manifest",
                "predictor_renderer_source_manifest_sha256",
            }
        ),
        "receipt inputs",
    )
    predictor = _exact_mapping(inputs.get("predictor_program"), "predictor program custody")
    pbr2 = _exact_mapping(inputs.get("pbr2_teacher_packet"), "PBR2 custody")
    gt_cache = _exact_mapping(inputs.get("gt_cache_custody_inherited_without_bulk_read"), "GT-cache custody")
    if predictor.get("sha256") != EXPECTED_PROGRAM_SHA256:
        raise CensusError("receipt predictor program custody differs")
    if pbr2.get("sha256") != EXPECTED_PBR2_SHA256 or pbr2.get("bytes") != EXPECTED_PBR2_BYTES:
        raise CensusError("receipt PBR2 packet custody differs")
    if (
        gt_cache.get("sha256") != EXPECTED_GT_CACHE_SHA256
        or gt_cache.get("bytes") != EXPECTED_GT_CACHE_BYTES
        or gt_cache.get("cross_receipt_match") is not True
        or gt_cache.get("cache_opened_by_this_tool") is not False
        or gt_cache.get("cache_rehashed_by_this_tool") is not False
    ):
        raise CensusError("receipt GT-cache inherited custody differs")
    if not _valid_sha256(inputs.get("predictor_renderer_source_manifest_sha256")):
        raise CensusError("receipt predictor renderer source identity is malformed")

    implementation = _exact_mapping(body.get("implementation_custody"), "implementation custody")
    if frozenset(implementation) != frozenset(IMPLEMENTATION_PATHS):
        raise CensusError("implementation custody file set differs")
    for path, raw in implementation.items():
        row = _exact_mapping(raw, f"implementation custody {path}")
        if row.get("path") != path or not _valid_sha256(row.get("sha256")):
            raise CensusError(f"implementation custody differs for {path}")
        _exact_positive_int(row.get("bytes"), f"implementation custody bytes for {path}")
    runtime = _exact_mapping(body.get("runtime_custody"), "runtime custody")
    _exact_keys(
        runtime,
        frozenset(
            {
                "python",
                "platform",
                "numpy",
                "byteorder",
                "zlib_compile",
                "zlib_runtime_version",
                "rng_used",
                "python_executable",
                "modules",
            }
        ),
        "runtime custody",
    )
    if (
        any(
            not isinstance(runtime.get(field), str) or not runtime.get(field)
            for field in (
                "python",
                "platform",
                "numpy",
                "byteorder",
                "zlib_compile",
                "zlib_runtime_version",
            )
        )
        or runtime.get("rng_used") is not False
    ):
        raise CensusError("runtime custody scalar fields differ")
    python_executable = _exact_mapping(runtime.get("python_executable"), "Python executable custody")
    _exact_keys(
        python_executable,
        frozenset({"path", "bytes", "sha256"}),
        "Python executable custody",
    )
    if not isinstance(python_executable.get("path"), str) or not python_executable.get("path"):
        raise CensusError("Python executable custody path is malformed")
    _exact_positive_int(python_executable.get("bytes"), "Python executable custody bytes")
    if not _valid_sha256(python_executable.get("sha256")):
        raise CensusError("Python executable custody SHA-256 is malformed")
    modules = _exact_mapping(runtime.get("modules"), "runtime module custody")
    expected_modules = frozenset(
        {"numpy_init", "numpy_multiarray_runtime", "bz2_runtime", "lzma_runtime", "zlib_runtime"}
    )
    _exact_keys(modules, expected_modules, "runtime module custody")
    for name, raw_module in modules.items():
        module = _exact_mapping(raw_module, f"runtime module {name}")
        if module.get("file_backed") is True:
            _exact_keys(
                module,
                frozenset({"module", "file_backed", "path", "bytes", "sha256"}),
                f"runtime module {name}",
            )
            if not isinstance(module.get("path"), str) or not module.get("path"):
                raise CensusError(f"runtime module {name} path is malformed")
            _exact_positive_int(module.get("bytes"), f"runtime module {name} bytes")
            if not _valid_sha256(module.get("sha256")):
                raise CensusError(f"runtime module {name} SHA-256 is malformed")
        elif module.get("file_backed") is False:
            _exact_keys(
                module,
                frozenset(
                    {
                        "module",
                        "file_backed",
                        "origin",
                        "standalone_sha256",
                        "covered_by_python_executable_sha256",
                    }
                ),
                f"runtime module {name}",
            )
            if module.get("standalone_sha256") != "not-applicable-statically-linked" or module.get(
                "covered_by_python_executable_sha256"
            ) != python_executable.get("sha256"):
                raise CensusError(f"runtime module {name} static-link custody differs")
        else:
            raise CensusError(f"runtime module {name} file_backed flag is not exact")

    measurement = _exact_mapping(body.get("measurement"), "measurement")
    _exact_keys(measurement, MEASUREMENT_KEYS, "measurement")
    geometry = measurement.get("geometry")
    if (
        not isinstance(geometry, list)
        or geometry != [PAIR_COUNT, HEIGHT, WIDTH]
        or any(type(value) is not int for value in geometry)
    ):
        raise CensusError("measurement geometry differs")
    total_cells = _exact_positive_int(measurement.get("total_semantic_cells"), "measurement total cells")
    if total_cells != PAIR_COUNT * HEIGHT * WIDTH:
        raise CensusError("measurement total-cell arithmetic differs")
    debt = _exact_positive_int(measurement.get("predictor_debt_cells"), "measurement predictor debt")
    if debt > total_cells or measurement.get("predictor_debt_fraction") != _ratio_string(debt, total_cells):
        raise CensusError("measurement predictor-debt arithmetic differs")
    for field in ("predictor_semantic_sha256", "teacher_semantic_sha256", "exclusive_owner_code_sha256"):
        if not _valid_sha256(measurement.get(field)):
            raise CensusError(f"measurement {field} is malformed")
    if (
        measurement.get("owner_code_legend")
        != {"0": "predictor-correct", "1": FAMILY_NAMES[0], "2": FAMILY_NAMES[1], "3": FAMILY_NAMES[2]}
        or measurement.get("ownership_masks_materialized_in_receipt") is not False
        or measurement.get("exclusive_ownership_closed") is not True
        or measurement.get("pairwise_owner_intersection_cells") != 0
        or measurement.get("candidate_admissible_remaining_debt_cells") != debt
        or measurement.get("teacher_reconstruction_remaining_debt_cells") != 0
    ):
        raise CensusError("measurement ownership/candidate debt boundary differs")

    raw_families = measurement.get("families")
    if not isinstance(raw_families, list) or len(raw_families) != len(FAMILY_NAMES):
        raise CensusError("measurement family set differs")
    families: list[Mapping[str, Any]] = []
    expected_before = debt
    owned_total = 0
    raw_bytes_total = 0
    payload_bytes_total = 0
    family_pair_cells: dict[str, list[int]] = {}
    for expected_name, raw_family in zip(FAMILY_NAMES, raw_families, strict=True):
        family = _exact_mapping(raw_family, f"family {expected_name}")
        _exact_keys(family, FAMILY_KEYS, f"family {expected_name}")
        if family.get("family") != expected_name or family.get("teacher_only") is not True:
            raise CensusError(f"family {expected_name} identity/lineage differs")
        owned = _exact_nonnegative_int(family.get("teacher_owned_cells"), f"family {expected_name} owned cells")
        before = _exact_nonnegative_int(family.get("errors_before"), f"family {expected_name} errors_before")
        after = _exact_nonnegative_int(family.get("errors_after"), f"family {expected_name} errors_after")
        boundary = _exact_nonnegative_int(
            family.get("target_boundary_4_neighbor_cells"), f"family {expected_name} boundary cells"
        )
        interior = _exact_nonnegative_int(family.get("target_interior_cells"), f"family {expected_name} interior cells")
        matrix = _exact_transition_matrix(
            family.get("predictor_to_teacher_transition_matrix"), f"family {expected_name} transition matrix"
        )
        if (
            before != expected_before
            or before - after != owned
            or family.get("teacher_coverage_numerator") != owned
            or family.get("teacher_coverage_denominator") != debt
            or family.get("teacher_coverage_fraction") != _ratio_string(owned, debt)
            or family.get("candidate_admissible_owned_cells") != 0
            or family.get("candidate_admissible_parameter_bytes") != 0
            or boundary + interior != owned
            or sum(sum(row) for row in matrix) != owned
            or family.get("score_value_per_byte") != "unmeasured"
        ):
            raise CensusError(f"family {expected_name} ownership/authority arithmetic differs")
        for field in (
            "value_conditioned_row_span_atoms_derived",
            "packet_record_atoms_exact",
            "packet_span_atoms_exact",
        ):
            _exact_nonnegative_int(family.get(field), f"family {expected_name} {field}")
        raw_bytes_total += _exact_positive_int(
            family.get("canonical_raw_parameter_bytes_exact"), f"family {expected_name} raw bytes"
        )
        payload_bytes_total += _exact_positive_int(
            family.get("teacher_entropy_payload_bytes_exact"), f"family {expected_name} payload bytes"
        )
        if not _valid_sha256(family.get("raw_parameter_sha256")) or not _valid_sha256(
            family.get("teacher_payload_sha256")
        ):
            raise CensusError(f"family {expected_name} payload identity is malformed")
        per_pair_owned = family.get("per_pair_owned_cells")
        if not isinstance(per_pair_owned, list) or len(per_pair_owned) != PAIR_COUNT:
            raise CensusError(f"family {expected_name} per-pair coverage differs")
        cells: list[int] = []
        for local_pair, raw_row in enumerate(per_pair_owned):
            row = _exact_mapping(raw_row, f"family {expected_name} per-pair row")
            _exact_keys(row, frozenset({"source_pair_id", "cells"}), f"family {expected_name} per-pair row")
            if row.get("source_pair_id") != PAIR_START + local_pair:
                raise CensusError(f"family {expected_name} source-pair order differs")
            cells.append(_exact_nonnegative_int(row.get("cells"), f"family {expected_name} pair cells"))
        if sum(cells) != owned:
            raise CensusError(f"family {expected_name} per-pair cells do not close")
        family_pair_cells[expected_name] = cells
        expected_before = after
        owned_total += owned
        families.append(family)
    if expected_before != 0 or owned_total != debt:
        raise CensusError("family ownership does not close exact predictor debt")

    per_pair = measurement.get("per_pair")
    per_pair_keys = frozenset({"local_pair_id", "source_pair_id", "predictor_debt_cells", *FAMILY_NAMES})
    if not isinstance(per_pair, list) or len(per_pair) != PAIR_COUNT:
        raise CensusError("measurement per-pair population differs")
    per_pair_debt_total = 0
    for local_pair, raw_row in enumerate(per_pair):
        row = _exact_mapping(raw_row, "measurement per-pair row")
        _exact_keys(row, per_pair_keys, "measurement per-pair row")
        if row.get("local_pair_id") != local_pair or row.get("source_pair_id") != PAIR_START + local_pair:
            raise CensusError("measurement per-pair coordinate differs")
        pair_debt = _exact_nonnegative_int(row.get("predictor_debt_cells"), "measurement pair debt")
        family_sum = 0
        for family_name in FAMILY_NAMES:
            value = _exact_nonnegative_int(row.get(family_name), f"measurement pair {family_name}")
            if value != family_pair_cells[family_name][local_pair]:
                raise CensusError("measurement family/per-pair join differs")
            family_sum += value
        if pair_debt != family_sum:
            raise CensusError("measurement per-pair debt does not close")
        per_pair_debt_total += pair_debt
    if per_pair_debt_total != debt:
        raise CensusError("measurement per-pair population does not close total debt")

    temporal = _exact_mapping(measurement.get("temporal_support"), "temporal support")
    palette = _exact_mapping(measurement.get("palette_gauge_value_structure"), "palette/gauge structure")
    plane = _exact_mapping(
        measurement.get("shared_vs_independent_plane_envelopes"), "shared/independent plane envelope"
    )
    if (
        temporal.get("same_coordinate_repeat_supported") is not True
        or temporal.get("same_coordinate_rule_exactly_verified") is not True
        or temporal.get("same_coordinate_repeat_cells") != families[0].get("teacher_owned_cells")
        or temporal.get("motion_aligned_repeat_supported_by_exact_teacher_packet") is not False
        or temporal.get("motion_aligned_repeat_coverage") != "unmeasured"
        or palette.get("conditional_value_alphabet_coverage_numerator") != debt
        or palette.get("conditional_value_alphabet_coverage_denominator") != debt
        or palette.get("logit_common_gauge_coverage") != "unmeasured"
        or palette.get("rgb_palette_coverage") != "unmeasured"
        or plane.get("candidate_byte_claim") is not False
        or plane.get("shared_plane_validity") != "unmeasured"
        or plane.get("independent_plane_validity") != "unmeasured"
    ):
        raise CensusError("measurement extrapolation/authority boundary differs")
    palette_matrix = _exact_transition_matrix(
        palette.get("predictor_to_teacher_transition_matrix"), "aggregate transition matrix"
    )
    if sum(sum(row) for row in palette_matrix) != debt:
        raise CensusError("aggregate transition matrix does not close predictor debt")

    byte_ownership = _exact_mapping(body.get("teacher_packet_byte_ownership"), "teacher packet byte ownership")
    _exact_keys(
        byte_ownership,
        frozenset(
            {
                "packet_bytes_exact",
                "packet_sha256",
                "shared_prefix_and_header_bytes_exact",
                "family_payload_bytes_exact",
                "family_payload_bytes_sum_without_overlap",
                "crc_bytes_exact",
                "packet_byte_ownership_closed",
                "canonical_raw_family_parameter_bytes_exact",
                "candidate_admissible_bytes",
                "candidate_archive_blocker",
            }
        ),
        "teacher packet byte ownership",
    )
    packet_bytes = _exact_positive_int(byte_ownership.get("packet_bytes_exact"), "teacher packet bytes")
    shared_bytes = _exact_nonnegative_int(
        byte_ownership.get("shared_prefix_and_header_bytes_exact"), "teacher shared/header bytes"
    )
    crc_bytes = _exact_nonnegative_int(byte_ownership.get("crc_bytes_exact"), "teacher CRC bytes")
    if (
        packet_bytes != EXPECTED_PBR2_BYTES
        or byte_ownership.get("packet_sha256") != EXPECTED_PBR2_SHA256
        or byte_ownership.get("family_payload_bytes_exact") != payload_bytes_total
        or byte_ownership.get("canonical_raw_family_parameter_bytes_exact") != raw_bytes_total
        or shared_bytes + payload_bytes_total + crc_bytes != packet_bytes
        or byte_ownership.get("family_payload_bytes_sum_without_overlap") is not True
        or byte_ownership.get("packet_byte_ownership_closed") is not True
        or byte_ownership.get("candidate_admissible_bytes") != 0
    ):
        raise CensusError("teacher packet byte/candidate ownership differs")

    remaining = _exact_mapping(body.get("remaining_debt"), "remaining debt")
    _exact_keys(
        remaining,
        frozenset(
            {
                "teacher_semantic_debt_after_all_pbr2_strata",
                "candidate_admissible_semantic_debt",
                "candidate_admissible_atom_count",
                "next_evidence_needed",
            }
        ),
        "remaining debt",
    )
    if (
        remaining.get("teacher_semantic_debt_after_all_pbr2_strata") != 0
        or remaining.get("candidate_admissible_semantic_debt") != debt
        or remaining.get("candidate_admissible_atom_count") != 0
    ):
        raise CensusError("remaining teacher/candidate debt differs")
    storage = _exact_mapping(body.get("storage_and_cleanup"), "storage and cleanup")
    if any(
        storage.get(field) is not False
        for field in (
            "bulk_cache_accessed",
            "bulk_artifact_created",
            "ssd_preflight_required",
            "auto_cleanup_required",
        )
    ):
        raise CensusError("storage/cleanup authority differs")
    v14 = _exact_mapping(body.get("v14_exact_anchor_dispositions"), "V14 crosslink")
    if (
        v14.get("evidence_axis") != "[macOS-CPU frozen-scorer advisory]"
        or v14.get("aggregation_into_family_claim") is not False
        or v14.get("score_value_per_byte") != "unmeasured"
    ):
        raise CensusError("V14 crosslink authority differs")


def make_receipt_envelope(body: Mapping[str, Any]) -> dict[str, Any]:
    body_dict = dict(body)
    if (
        body_dict.get("schema") != SCHEMA
        or body_dict.get("research_only") is not True
        or body_dict.get("candidate_payload_allowed") is not False
        or body_dict.get("score_claim") is not False
        or body_dict.get("promotion_eligible") is not False
        or body_dict.get("pointer_moved") is not False
    ):
        raise CensusError("receipt body does not preserve research-only/no-score custody")
    _assert_no_exhaustive_payload(body_dict)
    _validate_receipt_body(body_dict)
    encoded = canonical_json_bytes(body_dict)
    return {
        "schema": RECEIPT_SCHEMA,
        "body": body_dict,
        "body_sha256": sha256_bytes(encoded),
    }


def _strict_rebuild_receipt_body(body: Mapping[str, Any]) -> None:
    args = parse_args([])
    rebuilt = build_real_receipt(args)["body"]
    rebuilt["git_head"] = body.get("git_head")
    if canonical_json_bytes(body) != canonical_json_bytes(rebuilt):
        raise CensusError("receipt body differs from exact canonical source reconstruction")


def validate_receipt(receipt: Mapping[str, Any], *, reopen_sources: bool = True) -> None:
    if frozenset(receipt) != {"schema", "body", "body_sha256"}:
        raise CensusError("receipt envelope fields differ")
    if receipt.get("schema") != RECEIPT_SCHEMA or not isinstance(receipt.get("body"), dict):
        raise CensusError("receipt envelope schema/body differs")
    expected = sha256_bytes(canonical_json_bytes(receipt["body"]))
    if receipt.get("body_sha256") != expected:
        raise CensusError("receipt body SHA-256 differs")
    _assert_no_exhaustive_payload(receipt["body"])
    _validate_receipt_body(receipt["body"])
    if reopen_sources:
        _strict_rebuild_receipt_body(receipt["body"])
    regenerated = make_receipt_envelope(receipt["body"])
    if dict(receipt) != regenerated:
        raise CensusError("receipt is not in canonical envelope form")


def validate_frozen_teacher_contract(
    header: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
) -> None:
    """Reject synthetic, candidate-admitted, or non-exact exhaustive teachers."""

    if (
        header.get("target_semantic_lineage") != "frozen_gt_argmax"
        or header.get("pbr2_reconstructs_exact_gt_argmax") is not True
    ):
        raise CensusError("teacher must be the exact frozen_gt_argmax lineage, never a synthetic exhaustive target")
    required_header = {
        "pbr2_is_target_derived": True,
        "target_derived_residual_promotion_admitted": False,
        "research_only": True,
        "candidate_archive_admissible": False,
        "exact_target_semantic_reconstruction": True,
        "score_claim": False,
        "promotion_eligible": False,
        "decode_scorer_dependency": False,
    }
    if any(header.get(key) is not expected for key, expected in required_header.items()):
        raise CensusError("PBR2 header relaxed exhaustive-teacher no-fake custody")
    if (
        materialization_receipt.get("candidate_payload_allowed") is not False
        or materialization_receipt.get("research_only") is not True
        or materialization_receipt.get("score_claim") is not False
        or materialization_receipt.get("promotion_eligible") is not False
    ):
        raise CensusError("PBR2 materialization receipt relaxed candidate prohibition")
    closure = materialization_receipt.get("receiver_closure")
    if not isinstance(closure, Mapping) or closure.get("candidate_payload_allowed") is not False:
        raise CensusError("PBR2 receiver closure does not forbid teacher payload")


def derive_exclusive_ownership(
    predictor: np.ndarray,
    staged_outputs: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return final target and disjoint ownership masks for ordered staged outputs."""

    base = np.ascontiguousarray(predictor, dtype=np.uint8)
    if base.ndim != 3 or not staged_outputs:
        raise CensusError("ownership requires pair x height x width predictor and staged outputs")
    stages = tuple(np.ascontiguousarray(value, dtype=np.uint8) for value in staged_outputs)
    if any(value.shape != base.shape for value in stages):
        raise CensusError("staged semantic geometry differs from predictor")
    target = stages[-1]
    correction = base != target
    owner_code = np.zeros(base.shape, dtype=np.uint8)
    masks: list[np.ndarray] = []
    previous = base
    for index, current in enumerate(stages, start=1):
        changed = current != previous
        if np.any(changed & ~correction):
            raise CensusError("a stratum changed a cell outside final correction debt")
        if np.any(changed & (current != target)):
            raise CensusError("a stratum wrote a non-final teacher value")
        if np.any(changed & (owner_code != 0)):
            raise CensusError("strata double-owned at least one correction cell")
        owner_code[changed] = np.uint8(index)
        masks.append(np.ascontiguousarray(changed))
        previous = current
    if not np.array_equal(previous, target):
        raise CensusError("final staged output does not equal target")
    if not np.array_equal(owner_code != 0, correction):
        raise CensusError("exclusive ownership does not cover exact correction debt")
    if int(sum(np.count_nonzero(mask) for mask in masks)) != int(np.count_nonzero(correction)):
        raise CensusError("exclusive ownership count does not close")
    return target, tuple(masks)


def _boundary_4(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary


def _value_conditioned_row_spans(mask: np.ndarray, values: np.ndarray) -> int:
    starts = np.ascontiguousarray(mask, dtype=bool).copy()
    starts[:, :, 1:] &= ~(mask[:, :, :-1] & (values[:, :, 1:] == values[:, :, :-1]))
    return int(np.count_nonzero(starts))


def _transition_matrix(predictor: np.ndarray, target: np.ndarray, mask: np.ndarray) -> list[list[int]]:
    codes = predictor[mask].astype(np.int64) * 5 + target[mask].astype(np.int64)
    return np.bincount(codes, minlength=25).reshape(5, 5).astype(np.int64).tolist()


def _ratio_string(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        raise CensusError("ratio denominator must be positive")
    return f"{numerator / denominator:.12f}"


def _family_measurement(
    *,
    name: str,
    mask: np.ndarray,
    predictor: np.ndarray,
    target: np.ndarray,
    target_boundary: np.ndarray,
    stratum: Mapping[str, Any],
    source_pair_start: int,
) -> dict[str, Any]:
    cells = int(np.count_nonzero(mask))
    total = int(np.count_nonzero(predictor != target))
    boundary_cells = int(np.count_nonzero(mask & target_boundary))
    per_pair = np.count_nonzero(mask, axis=(1, 2)).astype(np.int64)
    row_spans = _value_conditioned_row_spans(mask, target)
    return {
        "family": name,
        "teacher_owned_cells": cells,
        "teacher_coverage_numerator": cells,
        "teacher_coverage_denominator": total,
        "teacher_coverage_fraction": _ratio_string(cells, total),
        "candidate_admissible_owned_cells": 0,
        "candidate_admissible_parameter_bytes": 0,
        "target_boundary_4_neighbor_cells": boundary_cells,
        "target_interior_cells": cells - boundary_cells,
        "value_conditioned_row_span_atoms_derived": row_spans,
        "packet_record_atoms_exact": int(stratum["record_count"]),
        "packet_span_atoms_exact": int(stratum["span_count"]),
        "canonical_raw_parameter_bytes_exact": int(stratum["raw_bytes"]),
        "teacher_entropy_payload_bytes_exact": int(stratum["payload_bytes"]),
        "teacher_entropy_codec": str(stratum["codec"]),
        "raw_parameter_sha256": str(stratum["raw_sha256"]),
        "teacher_payload_sha256": str(stratum["payload_sha256"]),
        "errors_before": int(stratum["errors_before"]),
        "errors_after": int(stratum["errors_after"]),
        "predictor_to_teacher_transition_matrix": _transition_matrix(predictor, target, mask),
        "per_pair_owned_cells": [
            {"source_pair_id": source_pair_start + index, "cells": int(value)}
            for index, value in enumerate(per_pair.tolist())
        ],
        "score_value_per_byte": "unmeasured",
        "score_value_per_byte_reason": (
            "no exact scorer transition attributes a candidate-admissible byte section to this teacher family"
        ),
        "teacher_only": True,
    }


def measure_teacher_atoms(
    predictor: np.ndarray,
    staged_outputs: Sequence[np.ndarray],
    strata: Sequence[Mapping[str, Any]],
    *,
    source_pair_start: int,
) -> dict[str, Any]:
    """Measure exclusive PBR2 families from already-decoded semantic stages."""

    if len(staged_outputs) != 3 or len(strata) != 3:
        raise CensusError("teacher census requires exactly three ordered PBR2 strata")
    target, masks = derive_exclusive_ownership(predictor, staged_outputs)
    correction = predictor != target
    total = int(np.count_nonzero(correction))
    if total <= 0:
        raise CensusError("teacher census requires nonzero predictor debt")
    target_boundary = _boundary_4(target)
    families = [
        _family_measurement(
            name=name,
            mask=mask,
            predictor=predictor,
            target=target,
            target_boundary=target_boundary,
            stratum=stratum,
            source_pair_start=source_pair_start,
        )
        for name, mask, stratum in zip(FAMILY_NAMES, masks, strata, strict=True)
    ]
    for family, stratum in zip(families, strata, strict=True):
        if family["teacher_owned_cells"] != int(stratum["corrected_cells"]):
            raise CensusError("derived family ownership differs from strict PBR2 accounting")

    temporal = masks[0]
    temporal_rule = np.zeros(temporal.shape, dtype=bool)
    temporal_rule[1:] = (
        correction[1:] & correction[:-1] & (predictor[1:] == predictor[:-1]) & (target[1:] == target[:-1])
    )
    if not np.array_equal(temporal, temporal_rule):
        raise CensusError("temporal owner differs from exact same-coordinate repeat rule")

    owner_codes = np.zeros(predictor.shape, dtype=np.uint8)
    for index, mask in enumerate(masks, start=1):
        owner_codes[mask] = np.uint8(index)
    total_cells = int(predictor.size)
    transition = _transition_matrix(predictor, target, correction)
    observed_transitions = sum(
        int(transition[left][right] > 0) for left in range(5) for right in range(5) if left != right
    )
    per_pair_rows = []
    for local_pair in range(predictor.shape[0]):
        row = {
            "local_pair_id": local_pair,
            "source_pair_id": source_pair_start + local_pair,
            "predictor_debt_cells": int(np.count_nonzero(correction[local_pair])),
        }
        for name, mask in zip(FAMILY_NAMES, masks, strict=True):
            row[name] = int(np.count_nonzero(mask[local_pair]))
        per_pair_rows.append(row)

    return {
        "geometry": list(predictor.shape),
        "total_semantic_cells": total_cells,
        "predictor_debt_cells": total,
        "predictor_debt_fraction": _ratio_string(total, total_cells),
        "predictor_semantic_sha256": sha256_bytes(memoryview(np.ascontiguousarray(predictor)).cast("B")),
        "teacher_semantic_sha256": sha256_bytes(memoryview(np.ascontiguousarray(target)).cast("B")),
        "exclusive_owner_code_sha256": sha256_bytes(memoryview(owner_codes).cast("B")),
        "owner_code_legend": {
            "0": "predictor-correct",
            "1": FAMILY_NAMES[0],
            "2": FAMILY_NAMES[1],
            "3": FAMILY_NAMES[2],
        },
        "ownership_masks_materialized_in_receipt": False,
        "exclusive_ownership_closed": sum(row["teacher_owned_cells"] for row in families) == total,
        "pairwise_owner_intersection_cells": 0,
        "families": families,
        "temporal_support": {
            "same_coordinate_repeat_supported": True,
            "same_coordinate_repeat_cells": int(np.count_nonzero(temporal)),
            "same_coordinate_rule_exactly_verified": True,
            "motion_aligned_repeat_supported_by_exact_teacher_packet": False,
            "motion_aligned_repeat_coverage": "unmeasured",
            "reason": (
                "PBR2 binds only unchanged scorer coordinates; the V9 semantic adapter exposes no exact motion field"
            ),
        },
        "palette_gauge_value_structure": {
            "predictor_to_teacher_transition_matrix": transition,
            "observed_off_diagonal_class_transition_atoms": observed_transitions,
            "possible_off_diagonal_class_transition_atoms": 20,
            "conditional_value_alphabet_coverage_numerator": total,
            "conditional_value_alphabet_coverage_denominator": total,
            "conditional_value_alphabet_note": (
                "all teacher values use the bounded five-class alphabet only after a spatial selector owns location"
            ),
            "logit_common_gauge_coverage": "unmeasured",
            "logit_common_gauge_reason": "PBR2 carries class IDs, not pre-argmax logits or gauge coordinates",
            "rgb_palette_coverage": "unmeasured",
            "rgb_palette_reason": "PBR2 contains no RGB or scorer transition",
        },
        "shared_vs_independent_plane_envelopes": {
            "semantic_teacher_planes_observed": 1,
            "shared_plane_validity": "unmeasured",
            "independent_plane_validity": "unmeasured",
            "one_exhaustive_semantic_plane_raw_bytes_teacher_only": total_cells,
            "two_independent_exhaustive_semantic_planes_raw_bytes_teacher_only": 2 * total_cells,
            "raw_bytes_avoided_if_sharing_were_valid_teacher_only": total_cells,
            "global_five_class_rgb_palette_raw_bytes": 5 * 3,
            "c0b_pair_independent_two_plane_palette_raw_bytes": predictor.shape[0] * 2 * 5 * 3,
            "pair_palette_shared_between_two_planes_raw_bytes_arithmetic_envelope": predictor.shape[0] * 5 * 3,
            "candidate_byte_claim": False,
            "reason": (
                "C0B supports a shared semantic partition and pair-plane palettes, while V10 supports independent "
                "planes; this one-plane semantic teacher cannot establish which receiver is valid"
            ),
        },
        "per_pair": per_pair_rows,
        "candidate_admissible_remaining_debt_cells": total,
        "teacher_reconstruction_remaining_debt_cells": 0,
    }


def _validate_n600_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if frozenset(value) != {"schema", "body", "body_sha256"}:
        raise CensusError("n600 grammar receipt envelope fields differ")
    body = value.get("body")
    if not isinstance(body, Mapping):
        raise CensusError("n600 grammar receipt body differs")
    if value.get("body_sha256") != sha256_bytes(canonical_json_bytes(body)):
        raise CensusError("n600 grammar receipt body hash differs")
    if (
        body.get("research_only") is not True
        or body.get("score_claim") is not False
        or body.get("promotion_eligible") is not False
    ):
        raise CensusError("n600 grammar receipt authority changed")
    return body


def _crosslink_v14(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = value.get("diagnostics")
    rows = diagnostics.get("lane_windows") if isinstance(diagnostics, Mapping) else None
    if not isinstance(rows, list):
        raise CensusError("V14 receipt lacks exact lane-window diagnostics")
    expected_ids = (448, 472, 496, 511)
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("source_pair_id") in expected_ids]
    if tuple(int(row["source_pair_id"]) for row in selected) != expected_ids:
        raise CensusError("V14 anchor pair disposition set differs")
    return [
        {
            "source_pair_id": int(row["source_pair_id"]),
            "local_pair_id": int(row["local_pair_id"]),
            "fixed_islands_d_seg": str(row["fixed_islands_d_seg"]),
            "fixed_both_d_seg": str(row["fixed_both_d_seg"]),
            "delta_d_seg": str(row["delta_d_seg"]),
            "family_attribution": False,
        }
        for row in selected
    ]


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise CensusError("git HEAD is not one SHA-1")
    return value


def _runtime_custody() -> dict[str, Any]:
    import _bz2
    import _lzma

    import numpy.core._multiarray_umath as multiarray

    python_executable = snapshot_file(Path(sys.executable).resolve())

    def module_custody(module: Any) -> dict[str, Any]:
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and Path(path).is_file():
            return {"module": module.__name__, "file_backed": True, **snapshot_file(Path(path))}
        spec = getattr(module, "__spec__", None)
        return {
            "module": module.__name__,
            "file_backed": False,
            "origin": str(getattr(spec, "origin", "unknown")),
            "standalone_sha256": "not-applicable-statically-linked",
            "covered_by_python_executable_sha256": python_executable["sha256"],
        }

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "byteorder": sys.byteorder,
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "rng_used": False,
        "python_executable": python_executable,
        "modules": {
            "numpy_init": module_custody(np),
            "numpy_multiarray_runtime": module_custody(multiarray),
            "bz2_runtime": module_custody(_bz2),
            "lzma_runtime": module_custody(_lzma),
            "zlib_runtime": module_custody(zlib),
        },
    }


def _semantic_argv(args: argparse.Namespace) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/measure_g1_teacher_atom_census.py",
        "--predictor-program",
        _display_path(args.predictor_program),
        "--expected-program-sha256",
        args.expected_program_sha256,
        "--pbr2",
        _display_path(args.pbr2),
        "--expected-pbr2-sha256",
        args.expected_pbr2_sha256,
        "--pbr2-receipt",
        _display_path(args.pbr2_receipt),
        "--n600-grammar-receipt",
        _display_path(args.n600_grammar_receipt),
        "--v14-receipt",
        _display_path(args.v14_receipt),
        "--expected-gt-cache-sha256",
        args.expected_gt_cache_sha256,
        "--expected-pair-start",
        str(args.expected_pair_start),
        "--expected-pair-count",
        str(args.expected_pair_count),
        "--output",
        _display_path(args.output),
    ]


def build_real_receipt(args: argparse.Namespace) -> dict[str, Any]:
    program, program_custody = read_stable_bytes(args.predictor_program)
    pbr2_payload, pbr2_custody = read_stable_bytes(args.pbr2)
    pbr_receipt_payload, pbr_receipt_custody = read_stable_bytes(args.pbr2_receipt)
    n600_payload, n600_custody = read_stable_bytes(args.n600_grammar_receipt)
    v14_payload, v14_custody = read_stable_bytes(args.v14_receipt)

    if program_custody["sha256"] != args.expected_program_sha256:
        raise CensusError("exact committed V9 program SHA-256 differs")
    if pbr2_custody["sha256"] != args.expected_pbr2_sha256:
        raise CensusError("exact committed PBR2 SHA-256 differs")

    pbr_receipt = _load_json_bytes(pbr_receipt_payload, label="PBR2 materialization receipt")
    n600_receipt = _load_json_bytes(n600_payload, label="n600 grammar receipt")
    v14_receipt = _load_json_bytes(v14_payload, label="V14 receipt")
    n600_body = _validate_n600_receipt(n600_receipt)
    decoded_pbr2 = decode_progressive_geometry_residual(pbr2_payload)
    header = decoded_pbr2.header
    validate_frozen_teacher_contract(header, pbr_receipt)

    pbr2_row = pbr_receipt.get("pbr2")
    gt_cache = pbr_receipt.get("inputs", {}).get("gt_cache")
    n600_cache = n600_body.get("input_custody")
    if not isinstance(pbr2_row, Mapping) or not isinstance(gt_cache, Mapping) or not isinstance(n600_cache, Mapping):
        raise CensusError("teacher materialization receipts lack required custody rows")
    if (
        pbr2_row.get("packet_sha256") != pbr2_custody["sha256"]
        or pbr2_row.get("packet_bytes") != pbr2_custody["bytes"]
        or gt_cache.get("sha256") != args.expected_gt_cache_sha256
        or gt_cache.get("bytes") != EXPECTED_GT_CACHE_BYTES
        or n600_cache.get("sha256") != args.expected_gt_cache_sha256
        or n600_cache.get("bytes") != EXPECTED_GT_CACHE_BYTES
    ):
        raise CensusError("program/PBR2/GT-cache custody does not cross-close")

    receiver = receive_factorized_v9_predictor(program, repository_root=REPO)
    if receiver.program_sha256 != program_custody["sha256"]:
        raise CensusError("fresh V9 receiver program identity differs")
    pair_start = int(header["source_pair_start"])
    pair_stop = int(header["source_pair_stop_exclusive"])
    if (
        pair_start != args.expected_pair_start
        or pair_stop - pair_start != args.expected_pair_count
        or receiver.source_pair_start != pair_start
        or receiver.pair_count != args.expected_pair_count
        or receiver.source_manifest_sha256 != header["predictor_renderer_sha256"]
    ):
        raise CensusError("exact V9/PBR2 pair window or renderer identity differs")
    predictor = receiver.decode_all_semantics(batch_size=16)
    if predictor.shape != (args.expected_pair_count, HEIGHT, WIDTH):
        raise CensusError("fresh V9 predictor geometry differs")

    apply_common = {
        "payload": pbr2_payload,
        "predictor_program": program,
        "predictor_contract_id": PREDICTOR_CONTRACT_ID,
        "predictor_renderer_sha256": receiver.source_manifest_sha256,
        "predictor_labels": predictor,
        "source_pair_ids": receiver.source_pair_ids,
    }
    stages = tuple(apply_progressive_geometry_residual(**apply_common, max_strata=index) for index in (1, 2, 3))
    accounting = packet_accounting(pbr2_payload)
    measurement = measure_teacher_atoms(
        predictor,
        stages,
        accounting["strata"],
        source_pair_start=pair_start,
    )
    if (
        measurement["predictor_semantic_sha256"] != header["predictor_semantic_sha256"]
        or measurement["teacher_semantic_sha256"] != header["target_semantic_sha256"]
        or measurement["predictor_debt_cells"] != header["pbr2_event_count"]
    ):
        raise CensusError("fresh semantic census differs from sealed PBR2 identities")

    packet_shared_bytes = int(accounting["packet_prefix_header_bytes"]) + int(accounting["header_bytes"])
    family_payload_bytes = sum(int(row["payload_bytes"]) for row in accounting["strata"])
    if packet_shared_bytes + family_payload_bytes + int(accounting["crc_bytes"]) != int(accounting["packet_bytes"]):
        raise CensusError("PBR2 packet byte ownership does not close")

    aggregate = n600_body.get("measurements", {}).get("aggregate", {})
    implementation = {path: snapshot_file(REPO / path) for path in IMPLEMENTATION_PATHS}
    body = {
        "schema": SCHEMA,
        "purpose": (
            "real-n64 predictor-conditioned teacher atom census over the exact committed V9 predictor and PBR2"
        ),
        "authority_axis": AUTHORITY_AXIS,
        "verdict_scope": VERDICT_SCOPE,
        "research_only": True,
        "candidate_payload_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "candidate_lineage_prohibition": CANDIDATE_LINEAGE_PROHIBITION,
        "git_head": _git_head(),
        "semantic_argv": _semantic_argv(args),
        "inputs": {
            "predictor_program": program_custody,
            "pbr2_teacher_packet": pbr2_custody,
            "pbr2_materialization_receipt": pbr_receipt_custody,
            "n600_partition_grammar_receipt": n600_custody,
            "v14_receiver_closed_anchor_receipt": v14_custody,
            "gt_cache_custody_inherited_without_bulk_read": {
                "path": str(gt_cache["path"]),
                "member": str(gt_cache["source_member"]),
                "bytes": int(gt_cache["bytes"]),
                "sha256": str(gt_cache["sha256"]),
                "cross_receipt_match": True,
                "cache_opened_by_this_tool": False,
                "cache_rehashed_by_this_tool": False,
            },
            "predictor_renderer_source_manifest": receiver.source_manifest,
            "predictor_renderer_source_manifest_sha256": receiver.source_manifest_sha256,
        },
        "implementation_custody": implementation,
        "runtime_custody": _runtime_custody(),
        "measurement": measurement,
        "teacher_packet_byte_ownership": {
            "packet_bytes_exact": int(accounting["packet_bytes"]),
            "packet_sha256": str(accounting["packet_sha256"]),
            "shared_prefix_and_header_bytes_exact": packet_shared_bytes,
            "family_payload_bytes_exact": family_payload_bytes,
            "family_payload_bytes_sum_without_overlap": True,
            "crc_bytes_exact": int(accounting["crc_bytes"]),
            "packet_byte_ownership_closed": True,
            "canonical_raw_family_parameter_bytes_exact": sum(int(row["raw_bytes"]) for row in accounting["strata"]),
            "candidate_admissible_bytes": 0,
            "candidate_archive_blocker": str(accounting["candidate_archive_blocker"]),
        },
        "full_n600_teacher_grammar_crosslink": {
            "receipt_body_sha256": str(n600_receipt["body_sha256"]),
            "evidence_rows": int(aggregate["evidence_rows"]),
            "total_sites": int(aggregate["total_sites"]),
            "target_row_runs_exact": int(aggregate["row_runs"]["sum"]),
            "successive_pair_end_changed_sites_exact": int(aggregate["temporal_changed_sites"]["sum"]),
            "successive_pair_end_changed_fraction": str(aggregate["temporal_changed_fraction"]),
            "purpose": "cross-link only; this tool does not duplicate the settled full-cache census",
        },
        "v14_exact_anchor_dispositions": {
            "evidence_axis": str(v14_receipt.get("evidence_axis")),
            "rows": _crosslink_v14(v14_receipt),
            "aggregation_into_family_claim": False,
            "score_value_per_byte": "unmeasured",
        },
        "remaining_debt": {
            "teacher_semantic_debt_after_all_pbr2_strata": 0,
            "candidate_admissible_semantic_debt": int(measurement["predictor_debt_cells"]),
            "candidate_admissible_atom_count": 0,
            "next_evidence_needed": (
                "an ORIGINAL bounded generator must acquire correction cells without copying teacher payload, then "
                "survive receiver/R and an exact scorer transition before score-value-per-byte exists"
            ),
        },
        "storage_and_cleanup": {
            "bulk_cache_accessed": False,
            "bulk_artifact_created": False,
            "ssd_preflight_required": False,
            "reason": "only sealed 52-79 KB packets, receipts, source files, and one small JSON receipt are read/written",
            "auto_cleanup_required": False,
        },
    }
    return make_receipt_envelope(body)


def write_once_receipt(
    path: Path,
    receipt: Mapping[str, Any],
    *,
    reopen_sources: bool = True,
) -> None:
    validate_receipt(receipt, reopen_sources=reopen_sources)
    payload = canonical_json_bytes(receipt) + b"\n"
    target = Path(os.path.abspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing, _ = read_stable_bytes(target)
        if existing != payload:
            raise CensusError(f"refusing to overwrite a different receipt: {target}")
        reopened = _load_json_bytes(existing, label="existing census receipt")
        validate_receipt(reopened, reopen_sources=reopen_sources)
        final, _ = read_stable_bytes(target)
        if final != existing:
            raise CensusError(f"existing census receipt changed during validation: {target}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            concurrent, _ = read_stable_bytes(target)
            if concurrent != payload:
                raise CensusError(f"concurrent receipt differs: {target}") from None
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    reopened_payload, _ = read_stable_bytes(target)
    reopened = _load_json_bytes(reopened_payload, label="written census receipt")
    validate_receipt(reopened, reopen_sources=reopen_sources)
    final, _ = read_stable_bytes(target)
    if final != reopened_payload:
        raise CensusError(f"written census receipt changed during validation: {target}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor-program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--expected-program-sha256", default=EXPECTED_PROGRAM_SHA256)
    parser.add_argument("--pbr2", type=Path, default=DEFAULT_PBR2)
    parser.add_argument("--expected-pbr2-sha256", default=EXPECTED_PBR2_SHA256)
    parser.add_argument("--pbr2-receipt", type=Path, default=DEFAULT_PBR2_RECEIPT)
    parser.add_argument("--n600-grammar-receipt", type=Path, default=DEFAULT_N600_RECEIPT)
    parser.add_argument("--v14-receipt", type=Path, default=DEFAULT_V14_RECEIPT)
    parser.add_argument("--expected-gt-cache-sha256", default=EXPECTED_GT_CACHE_SHA256)
    parser.add_argument("--expected-pair-start", type=int, default=PAIR_START)
    parser.add_argument("--expected-pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_real_receipt(args)
    write_once_receipt(args.output, receipt)
    print(json.dumps({"output": _display_path(args.output), "body_sha256": receipt["body_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
