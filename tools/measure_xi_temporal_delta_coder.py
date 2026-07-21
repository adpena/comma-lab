#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure Task #574's xi-keyed temporal lane-description coder at n64/n600.

The tool consumes the settled S4 archive and corrected BEV-v2 cross/within pose
stages without rebuilding either.  It writes every bulky/checkpoint artifact to
the Vertigo SSD, refuses quarantined archives, proves repository-decoder exact
reconstruction, and materializes a deterministic full-archive projection.  The
projection is intentionally labelled non-receiver-closed until S4's standalone
runtime learns the XTDL1 codec.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import os
import platform
import shutil
import struct
import sys
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.artifact_quarantine import assert_not_quarantined_archive  # noqa: E402
from tac.boundary_math import analytic_lane_render_band as lane_codec  # noqa: E402
from tac.boundary_math.warp_real_luma_frame0 import xi_from_pose_calibration  # noqa: E402
from tac.lie._se3_numpy import exp_se3, log_se3  # noqa: E402
from tac.optimization.s4_archive_composer import (  # noqa: E402
    SECTION_ORDER,
    SectionBytes,
    canonical_json_bytes,
    deterministic_archive,
    parse_sections,
    serialize_sections,
)
from tac.optimization.xi_temporal_delta_coder import (  # noqa: E402
    PMF_SEED,
    decode_lane_xi_temporal,
    decode_lane_xi_temporal_grid,
    encode_quantized_lane_xi_temporal,
    quantized_lane_grid_from_lbnd2,
    semantic_quantized_lane_sha256,
)

DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721/archive.zip"
)
DEFAULT_BEV: Final = Path("/Volumes/VertigoDataTier/pact/evidence/bev_staticity_v2_20260721/canonical_v1")
DEFAULT_SCALE_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/advected_screw6_20260721/advected_screw6_chartlevel/receipt.json"
)
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/evidence/xi_temporal_574_20260721")
DEFAULT_S4_BUILD_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721/build_receipt.json"
)
DEFAULT_S4_MEASUREMENT_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/measurement_s4_20260721/measurement_receipt.json"
)
DEFAULT_S4_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721/runtime/inflate.py"
)
EXPECTED_ARCHIVE_SHA256: Final = "d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed"
EXPECTED_ARCHIVE_BYTES: Final = 451_191
EXPECTED_S4_BUILD_RECEIPT_SHA256: Final = "f9f2f9b63ea5c1b1dd3972752012e0a9279aca705064f1a8fa89c231bd51f590"
EXPECTED_S4_MEASUREMENT_RECEIPT_SHA256: Final = (
    "244d7b8fa695068755cd64a47572d58f2212f215e287e5d1ee6e6182384ea428"
)
EXPECTED_S4_RUNTIME_SHA256: Final = "eef055896474b8327baf57ace016c37fe651f4c22534e2442ebc44da8c3f40b0"
EXPECTED_SCALE_RECEIPT_SHA256: Final = "e257bc6b9fe3a899e6b1d9b6a4d5b0496129dc4831a3aa25b40d7cf27346b450"
EXPECTED_BEV_STAGE_TREE_SHA256: Final = "ac7ec2703c5fac51542dcb86a5d381912ae6374a5adc560ded6e82bedca79e9c"
EXPECTED_BEV_STAGE_CONFIG_SHA256: Final = "99f23df35d720b5f006842a882c15f69439e11ca876b3f527e62118cd5703120"
EXPECTED_XI_FP64_SHA256: Final = "5269b47a0fb5d8cb10dabc860da8be05511cebf396861e9a0aac35267a23462d"
EXPECTED_DESCRIPTION_BASE_COMPONENTS: Final = 216_207
PHASE_BOXES: Final = (216_300, 154_600)
LZMA_FILTERS: Final = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
PAIR_COUNTS: Final = (64, 600)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
STRATUM_NAMES: Final = ("cell", "boundary", "critical")


class MeasureXiTemporalError(RuntimeError):
    """A source-custody, exact-decode, storage, or accounting gate failed."""


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _storage_preflight(output: Path) -> dict[str, Any]:
    resolved = output.resolve()
    allowed = Path("/Volumes/VertigoDataTier/pact/evidence/xi_temporal_574_20260721")
    if resolved != allowed and allowed not in resolved.parents:
        raise MeasureXiTemporalError(f"bulk evidence must remain below {allowed}")
    resolved.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(resolved).free
    required = 1 << 30
    if free < required:
        raise MeasureXiTemporalError(f"storage preflight failed: {free} < {required}")
    volatile_record = {
        "status": "PASS",
        "path": str(resolved),
        "free_bytes": free,
        "required_free_bytes": required,
        "cleanup": (
            "atomic same-directory scratch is deleted on success/failure; durable stage checkpoints "
            "and projected archive are preserved; no source bytes are moved or deleted"
        ),
    }
    _atomic_json(resolved / "checkpoints" / "storage_preflight.json", volatile_record)
    return {
        key: value for key, value in volatile_record.items() if key != "free_bytes"
    } | {
        "volatile_observation_path": str(resolved / "checkpoints" / "storage_preflight.json"),
        "determinism_note": "live free_bytes is checked and checkpointed but excluded from this receipt",
    }


def _source_authority() -> dict[str, Any]:
    build_path = DEFAULT_S4_BUILD_RECEIPT.resolve(strict=True)
    measurement_path = DEFAULT_S4_MEASUREMENT_RECEIPT.resolve(strict=True)
    runtime_path = DEFAULT_S4_RUNTIME.resolve(strict=True)
    if _sha_file(build_path) != EXPECTED_S4_BUILD_RECEIPT_SHA256:
        raise MeasureXiTemporalError("S4 build-receipt custody hash drifted")
    if _sha_file(measurement_path) != EXPECTED_S4_MEASUREMENT_RECEIPT_SHA256:
        raise MeasureXiTemporalError("S4 measurement-receipt custody hash drifted")
    if _sha_file(runtime_path) != EXPECTED_S4_RUNTIME_SHA256:
        raise MeasureXiTemporalError("S4 standalone-runtime custody hash drifted")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    advisory = measurement["advisory_eval"]
    measured = advisory["stages"]["evaluate"]["measured"]
    if (
        build.get("archive", {}).get("sha256") != EXPECTED_ARCHIVE_SHA256
        or build.get("research_only") is not True
        or build.get("promotion_eligible") is not False
        or advisory.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256
        or advisory.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES
        or measured.get("archive_bytes") != EXPECTED_ARCHIVE_BYTES
    ):
        raise MeasureXiTemporalError("S4 source authority does not match the pinned research-only archive")
    return {
        "build_receipt_path": str(build_path),
        "build_receipt_sha256": EXPECTED_S4_BUILD_RECEIPT_SHA256,
        "measurement_receipt_path": str(measurement_path),
        "measurement_receipt_sha256": EXPECTED_S4_MEASUREMENT_RECEIPT_SHA256,
        "runtime_path": str(runtime_path),
        "runtime_sha256": EXPECTED_S4_RUNTIME_SHA256,
        "research_only": True,
        "promotion_eligible": False,
        "advisory_axis": measurement["advisory_eval"]["axis"],
        "advisory_d_seg": float(measured["d_seg"]),
        "advisory_d_pose": float(measured["d_pose"]),
        "solved_pointer_object": False,
        "source_object_blocker": "SOURCE_S4_NOT_SOLVED_POINTER_OBJECT_DESCRIPTION_STREAM",
    }


def _runtime_custody() -> dict[str, Any]:
    import _lzma

    import _brotli

    python_executable = Path(sys.executable).resolve(strict=True)

    def _native_module(name: str, module: Any) -> dict[str, Any]:
        origin = getattr(module, "__file__", None) or getattr(importlib.util.find_spec(name), "origin", None)
        if origin and origin not in {"built-in", "frozen"} and Path(origin).is_file():
            path = Path(origin).resolve(strict=True)
            return {"origin": str(path), "sha256": _sha_file(path)}
        return {
            "origin": str(origin or "built-in"),
            "linked_python_executable_sha256": _sha_file(python_executable),
        }

    multiarray = np.core._multiarray_umath  # type: ignore[attr-defined]
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(python_executable),
            "executable_sha256": _sha_file(python_executable),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "numpy": {
            "version": np.__version__,
            "multiarray": _native_module("numpy.core._multiarray_umath", multiarray),
        },
        "zlib": {
            "compile_version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        },
        "liblzma_python_extension": {
            **_native_module("_lzma", _lzma),
            "filters": LZMA_FILTERS,
        },
        "brotli": {
            "version": getattr(brotli, "__version__", "unknown"),
            **_native_module("_brotli", _brotli),
        },
        "zip": {"method": "deflate9", "metadata": "canonical_s4_deterministic_archive"},
    }


def _load_archive(path: Path) -> tuple[bytes, tuple[SectionBytes, ...], dict[str, Any]]:
    archive = path.resolve(strict=True)
    assert_not_quarantined_archive(archive, context="Task #574 fresh S4 source open", repo_root=REPO)
    actual_sha = _sha_file(archive)
    if actual_sha != EXPECTED_ARCHIVE_SHA256 or archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise MeasureXiTemporalError("source archive is not the settled S4 custody object")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != 1 or members[0].filename != "0.bin":
            raise MeasureXiTemporalError("source archive is not the one-member S4 grammar")
        monolith = zipped.read("0.bin")
    sections = parse_sections(monolith)
    if serialize_sections(sections) != monolith:
        raise MeasureXiTemporalError("source S4 parse/re-serialize drift")
    return (
        monolith,
        sections,
        {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": actual_sha,
            "member_bytes": len(monolith),
            "member_sha256": _sha_bytes(monolith),
            "quarantine_gate": "PASS",
        },
    )


def _section_map(sections: tuple[SectionBytes, ...]) -> dict[str, SectionBytes]:
    return {row.name: row for row in sections}


def _decode_base(
    base: bytes,
) -> tuple[bytes, bytes, bytes, bytes, np.ndarray, np.ndarray, dict[str, Any], Any]:
    if len(base) < 8:
        raise MeasureXiTemporalError("PBASE3 is truncated")
    static_size, lane_size = struct.unpack_from("<II", base)
    if len(base) != 8 + static_size + lane_size:
        raise MeasureXiTemporalError("PBASE3 length mismatch")
    static_encoded = base[8 : 8 + static_size]
    lane_encoded = base[8 + static_size :]
    static_raw = brotli.decompress(static_encoded)
    lane_raw = lzma.decompress(lane_encoded, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    _decoded_lines, lane_header = lane_codec.deserialize_lane_band_any(lane_raw)
    q_lane, presence, exact_header = quantized_lane_grid_from_lbnd2(lane_raw)
    if exact_header != lane_header:
        raise MeasureXiTemporalError("LBND2 object and exact-lattice decoders disagree on the header")
    if int(lane_header["rd"]["n_pairs"]) != 600:
        raise MeasureXiTemporalError("settled lane description does not cover n600")
    if lane_header["rd"].get("pack_mode") != "coherent_slot":
        raise MeasureXiTemporalError("settled lane description is not the coherent-slot chart")
    config = lane_codec.render_config_from_header(lane_header)
    return static_encoded, static_raw, lane_encoded, lane_raw, q_lane, presence, lane_header, config


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(bytes.fromhex(_sha_file(path)))
    return digest.hexdigest()


def _corrected_full_chasles(
    bev_root: Path,
    scale_receipt: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    receipt_path = scale_receipt.resolve(strict=True)
    scale_receipt_sha = _sha_file(receipt_path)
    if scale_receipt_sha != EXPECTED_SCALE_RECEIPT_SHA256:
        raise MeasureXiTemporalError("full-screw scale receipt hash drifted from sealed custody")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    lawrefs = receipt["config"]["lawrefs"]
    translation_scale = float(lawrefs["translation_scale"]["value"])
    rotation_scale = float(lawrefs["rotation_scale"]["value"])
    pitch_rad = float(lawrefs["ground_pitch"]["resolved_value"])
    if (translation_scale, rotation_scale, pitch_rad) != (0.16, 1.0, -0.05):
        raise MeasureXiTemporalError("full-screw LawRef values drifted from the sealed custody receipt")

    stage_root = bev_root.resolve(strict=True) / "absolute_pose_stages"
    paths = [stage_root / f"frame_{index:04d}.json" for index in range(600)]
    if not all(path.is_file() for path in paths):
        raise MeasureXiTemporalError("corrected BEV-v2 absolute-pose stages do not cover n600")
    xi_rows: list[np.ndarray] = []
    stage_config_sha: str | None = None
    for index, path in enumerate(paths):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("schema") != "bev_staticity_absolute_pose_frame.v2" or int(row["frame"]) != index:
            raise MeasureXiTemporalError(f"BEV-v2 stage schema/frame mismatch at {path}")
        if stage_config_sha is None:
            stage_config_sha = str(row["config_sha256"])
        elif row["config_sha256"] != stage_config_sha:
            raise MeasureXiTemporalError("BEV-v2 stage config hashes are inconsistent")
        if index == 0:
            # Pair zero is the P-frame keyframe and has no prior pair.
            xi_rows.append(np.zeros(6, dtype=np.float64))
            continue
        cross = xi_from_pose_calibration(
            np.asarray(row["raw_cross_target"], dtype=np.float64),
            translation_scale,
            rotation_scale,
            pitch_rad,
            whole_ground=True,
        )
        within = xi_from_pose_calibration(
            np.asarray(row["raw_within_target"], dtype=np.float64),
            translation_scale,
            rotation_scale,
            pitch_rad,
            whole_ground=True,
        )
        # BEV-v2 custody: A_f1[t] = A_f1[t-1] exp(xi_cross[t]) exp(xi_within[t]).
        xi_rows.append(np.asarray(log_se3(exp_se3(cross) @ exp_se3(within)), dtype=np.float64))
    xi = np.stack(xi_rows)
    nonzero = np.count_nonzero(xi, axis=0).astype(int)
    if xi.shape != (600, 6) or not np.all(np.isfinite(xi)) or np.any(nonzero != 599):
        raise MeasureXiTemporalError("corrected full screw failed six-coordinate n600 coverage")
    stage_tree_sha = _tree_hash(paths)
    xi_sha = _sha_bytes(np.ascontiguousarray(xi, dtype="<f8").tobytes())
    if (
        stage_tree_sha != EXPECTED_BEV_STAGE_TREE_SHA256
        or stage_config_sha != EXPECTED_BEV_STAGE_CONFIG_SHA256
        or xi_sha != EXPECTED_XI_FP64_SHA256
    ):
        raise MeasureXiTemporalError("corrected full-screw source or derived xi hash drifted from sealed custody")
    return xi, {
        "schema": "xi_temporal_574_corrected_composed_full_screw_custody.v1",
        "source": "BEV-v2 raw exact cross-pair target plus raw exact within-pair target",
        "composition": "log_se3(exp_se3(xi_cross[t]) @ exp_se3(xi_within[t]))",
        "frame0": "zero keyframe by temporal-coder convention",
        "stage_root": str(stage_root),
        "stage_count": 600,
        "stage_tree_sha256": stage_tree_sha,
        "stage_config_sha256": stage_config_sha,
        "scale_receipt_path": str(receipt_path),
        "scale_receipt_sha256": scale_receipt_sha,
        "translation_scale": translation_scale,
        "rotation_scale": rotation_scale,
        "pitch_rad": pitch_rad,
        "coordinate_order": ["rho_x", "rho_y", "rho_z", "omega_x", "omega_y", "omega_z"],
        "nonzero_count_by_coordinate": nonzero.tolist(),
        "xi_fp64_sha256": xi_sha,
        "authority": "NumPy-fp64 tac.lie composition; rate/description authority only",
    }


def _lzma(payload: bytes) -> bytes:
    return lzma.compress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


def _rate_term(byte_count: int) -> float:
    return 25.0 * byte_count / 37_545_489.0


def _active_prefix_grid(
    q_lane: np.ndarray,
    presence: np.ndarray,
    pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_presence = np.asarray(presence[:pairs], dtype=bool)
    source_q = np.asarray(q_lane[:pairs], dtype=np.int64)
    active_slots = np.flatnonzero(np.any(source_presence, axis=0))
    slot_dims = lane_codec._RD_D_SLOT
    q_3d = source_q.reshape(pairs, source_presence.shape[1], slot_dims)
    return (
        np.ascontiguousarray(q_3d[:, active_slots, :].reshape(pairs, len(active_slots) * slot_dims)),
        np.ascontiguousarray(source_presence[:, active_slots]),
    )


def _serialize_exact_coherent_prefix(
    q_lane: np.ndarray,
    presence: np.ndarray,
    lane_header: dict[str, Any],
    config: Any,
) -> bytes:
    steps = np.asarray(lane_header["rd"]["base_steps"], dtype=np.float64)
    f_near = float(lane_header["rd"]["f_near"])
    slots = int(presence.shape[1])
    ever = np.asarray(presence[0], dtype=bool).copy()
    births = 0
    for row in presence[1:]:
        births += int(np.count_nonzero(row & ~ever))
        ever |= row
    extra = {
        key: value
        for key, value in lane_header["rd"].items()
        if key not in {"K", "d_slot", "n_pairs", "base_steps", "f_near"}
    }
    extra.update(
        {
            "pack_mode": "coherent_slot",
            "coherent_fit": "none",
            "n_tracks": slots,
            "n_births": births,
            "n_deaths": 0,
        }
    )
    steps_full = np.tile(steps, slots) if slots else np.zeros(0, dtype=np.float64)
    matrix = np.asarray(q_lane, dtype=np.float64) * steps_full
    return lane_codec._serialize_matrix_lbnd2(
        matrix,
        presence,
        slots,
        config,
        steps,
        f_near=f_near,
        extra_rd=extra,
    )


def _measure_prefix(
    *,
    pairs: int,
    lane_header: dict[str, Any],
    config: Any,
    full_xi: np.ndarray,
    source_q_lane: np.ndarray,
    source_presence: np.ndarray,
    original_lane_raw: bytes,
    original_lane_encoded: bytes,
    output: Path,
    fingerprint: str,
) -> dict[str, Any]:
    stage_path = output / "checkpoints" / f"stage_n{pairs}.json"
    xi_path = output / "bundles" / f"lane_xi_n{pairs}.xtdl1"
    identity_path = output / "bundles" / f"lane_identity_context_n{pairs}.xtdl1"
    if stage_path.is_file() and xi_path.is_file() and identity_path.is_file():
        existing = json.loads(stage_path.read_text(encoding="utf-8"))
        recorded = existing.get("artifacts", {})
        if (
            existing.get("input_fingerprint") == fingerprint
            and _sha_file(xi_path) == recorded.get("xi", {}).get("sha256")
            and _sha_file(identity_path) == recorded.get("identity", {}).get("sha256")
        ):
            decode_lane_xi_temporal(xi_path.read_bytes())
            decode_lane_xi_temporal(identity_path.read_bytes())
            return existing

    prefix_xi = full_xi[:pairs]
    prefix_q_lane, prefix_presence = _active_prefix_grid(source_q_lane, source_presence, pairs)
    steps = np.asarray(lane_header["rd"]["base_steps"], dtype=np.float64)
    f_near = float(lane_header["rd"]["f_near"])
    semantic_sha = semantic_quantized_lane_sha256(
        prefix_q_lane,
        prefix_presence,
        steps,
        f_near,
        config,
        pack_mode="coherent_slot",
    )

    coherent_raw = _serialize_exact_coherent_prefix(prefix_q_lane, prefix_presence, lane_header, config)
    if pairs == 600 and coherent_raw != original_lane_raw:
        raise MeasureXiTemporalError("n600 coherent-slot re-serialization differs from settled LBND2 bytes")
    baseline_raw = original_lane_raw if pairs == 600 else coherent_raw
    baseline_terminal = original_lane_encoded if pairs == 600 else _lzma(baseline_raw)
    baseline_q, baseline_presence, baseline_header = quantized_lane_grid_from_lbnd2(baseline_raw)
    if (
        baseline_header["rd"].get("pack_mode") != "coherent_slot"
        or not np.array_equal(baseline_q, prefix_q_lane)
        or not np.array_equal(baseline_presence, prefix_presence)
    ):
        raise MeasureXiTemporalError(f"n{pairs} LBND2 baseline exact coherent-slot lattice replay failed")

    xi_artifact = encode_quantized_lane_xi_temporal(
        prefix_q_lane,
        prefix_presence,
        config,
        prefix_xi,
        base_steps=steps,
        f_near=f_near,
        predictor="planar3_from_composed_screw",
        seed=PMF_SEED,
        pack_mode="coherent_slot",
    )
    identity_artifact = encode_quantized_lane_xi_temporal(
        prefix_q_lane,
        prefix_presence,
        config,
        prefix_xi,
        base_steps=steps,
        f_near=f_near,
        predictor="identity",
        seed=PMF_SEED,
        pack_mode="coherent_slot",
    )
    for label, artifact in (("xi", xi_artifact), ("identity", identity_artifact)):
        decoded_q, decoded_presence, decoded_header = decode_lane_xi_temporal_grid(artifact.payload)
        if (
            decoded_header["semantic"]["grid_sha256"] != semantic_sha
            or not np.array_equal(decoded_q, prefix_q_lane)
            or not np.array_equal(decoded_presence, prefix_presence)
        ):
            raise MeasureXiTemporalError(f"n{pairs} {label} artifact failed exact slot-labelled grid replay")
    _atomic_bytes(xi_path, xi_artifact.payload)
    _atomic_bytes(identity_path, identity_artifact.payload)
    xi_terminal = _lzma(xi_artifact.payload)
    identity_terminal = _lzma(identity_artifact.payload)

    rows = []
    for name, raw_bytes, terminal_bytes, artifact in (
        ("LBND2_identity_settled", len(baseline_raw), len(baseline_terminal), None),
        (
            "XTDL1_identity_xi_context_control",
            len(identity_artifact.payload),
            len(identity_terminal),
            identity_artifact,
        ),
        (
            "XTDL1_planar3_from_composed_screw_predictor",
            len(xi_artifact.payload),
            len(xi_terminal),
            xi_artifact,
        ),
    ):
        rows.append(
            {
                "arm": name,
                "description_wire_bytes_before_terminal": raw_bytes,
                "terminal_lzma_bytes": terminal_bytes,
                "rate_term": _rate_term(terminal_bytes),
                "delta_terminal_bytes_vs_lbnd2": terminal_bytes - len(baseline_terminal),
                "ratio_vs_lbnd2": terminal_bytes / len(baseline_terminal),
                "shared_pmf_estimated_payload_bytes": (
                    artifact.estimated_payload_bytes if artifact is not None else None
                ),
                "shared_pmf_model_bytes": artifact.model_bytes if artifact is not None else None,
                "shared_pmf_range_payload_bytes": artifact.range_payload_bytes if artifact is not None else None,
                "counted_full_xi_payload_bytes": artifact.xi_payload_bytes if artifact is not None else None,
                "presence_bytes": artifact.presence_bytes if artifact is not None else None,
                "selected_xi_context_bins": (
                    artifact.header["context"]["selected_bins"] if artifact is not None else None
                ),
                "semantic_lane_sha256": semantic_sha,
                "repository_codec_semantic_grid_exact": True,
            }
        )
    result = {
        "schema": "xi_temporal_574_prefix_measurement.v1",
        "pairs": pairs,
        "seed": PMF_SEED,
        "input_fingerprint": fingerprint,
        "rows": rows,
        "admission": {
            "repository_codec_semantic_grid_exact": True,
            "standalone_s4_receiver_closed": False,
            "xi_beats_settled_lbnd2": len(xi_terminal) < len(baseline_terminal),
            "identity_context_beats_settled_lbnd2": len(identity_terminal) < len(baseline_terminal),
            "verdict_scope": (
                "3-DOF planar (rho_z,rho_x,omega_y) projection of the corrected composed full "
                "screw plus shared-PMF entropy on the settled coherent-slot ground-frame LBND chart"
            ),
        },
        "artifacts": {
            "xi": {"path": str(xi_path), "bytes": xi_path.stat().st_size, "sha256": _sha_file(xi_path)},
            "identity": {
                "path": str(identity_path),
                "bytes": identity_path.stat().st_size,
                "sha256": _sha_file(identity_path),
            },
        },
    }
    _atomic_json(stage_path, result)
    return result


def _component_inventory(payload: bytes) -> dict[str, Any]:
    offset = 0
    rows: defaultdict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"packets": 0, "encoded_bytes": 0, "decoded_bytes": 0, "sites": 0}
    )
    total_packets = total_raw = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise MeasureXiTemporalError("PCOMP3 packet prefix is truncated")
        size = struct.unpack_from("<I", payload, offset)[0]
        start = offset
        offset += 4
        if size <= 0 or offset + size > len(payload):
            raise MeasureXiTemporalError("PCOMP3 packet size is invalid")
        raw = zlib.decompress(payload[offset : offset + size])
        offset += size
        if len(raw) < 12:
            raise MeasureXiTemporalError("PCOMP3 raw record is truncated")
        _frame, class_id, stratum_id, count, _first = struct.unpack_from("<HBBII", raw)
        if class_id >= len(CLASS_NAMES) or stratum_id >= len(STRATUM_NAMES) or count <= 0:
            raise MeasureXiTemporalError("PCOMP3 class/stratum metadata is invalid")
        row = rows[(class_id, stratum_id)]
        row["packets"] += 1
        row["encoded_bytes"] += offset - start
        row["decoded_bytes"] += len(raw)
        row["sites"] += int(count)
        total_packets += 1
        total_raw += len(raw)
    if offset != len(payload):
        raise MeasureXiTemporalError("PCOMP3 packet stream has trailing bytes")
    per_stratum = []
    for (class_id, stratum_id), values in sorted(rows.items()):
        per_stratum.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "stratum_id": stratum_id,
                "stratum": STRATUM_NAMES[stratum_id],
                **values,
                "candidate_encoded_bytes": values["encoded_bytes"],
                "delta_bytes": 0,
                "status": "UNCHANGED_BY_NARROW_LANE_FORMULATION",
            }
        )
    return {
        "packet_count": total_packets,
        "encoded_bytes": len(payload),
        "decoded_bytes": total_raw,
        "per_class_per_stratum": per_stratum,
    }


def _inventory(
    sections: tuple[SectionBytes, ...],
    lane_header: dict[str, Any],
    lane_raw: bytes,
) -> dict[str, Any]:
    by_name = _section_map(sections)
    components = _component_inventory(by_name["components.pcomp3"].payload)
    base = by_name["base.pbase3"].payload
    static_size, lane_size = struct.unpack_from("<II", base)
    return {
        "schema": "xi_temporal_574_stream_inventory.v1",
        "families": [
            {
                "family": "PPCS_seed",
                "encoded_bytes": len(by_name["seed.ppcs"].payload),
                "status": "UNCHANGED; counted planar trajectory already present; corrected full screw is not",
            },
            {
                "family": "PXQ1_static_quotient",
                "encoded_bytes": static_size,
                "status": "UNCHANGED; null/range projection settled upstream",
            },
            {
                "family": "LBND2_lane_coherent_slot",
                "encoded_bytes": lane_size,
                "decoded_bytes": len(lane_raw),
                "logical_pairs": int(lane_header["rd"]["n_pairs"]),
                "slots": int(lane_header["rd"]["K"]),
                "pack_mode": lane_header["rd"].get("pack_mode", "unknown"),
                "front_end": "lane_track_and_smooth coherent-slot path already consumed by settled S4",
                "status": "TREATMENT_TARGET",
            },
            {
                "family": "PCR3_causal",
                "encoded_bytes": len(by_name["causal.pcr3"].payload),
                "status": "EMPTY_SELECTED_ZERO_PARAMETER_POLICY",
            },
            {
                "family": "PCE3_events",
                "encoded_bytes": len(by_name["events.pce3"].payload),
                "decoded_bytes": by_name["events.pce3"].decoded_bytes,
                "status": "UNCHANGED; already adjacent-frame LAP/XOR INTER grammar",
            },
            {
                "family": "PCOMP3_components",
                "encoded_bytes": len(by_name["components.pcomp3"].payload),
                "decoded_bytes": by_name["components.pcomp3"].decoded_bytes,
                "status": "UNCHANGED; semantic persistent-ID encoder is a distinct owed parser surface",
            },
            {
                "family": "Movable_site_coder",
                "encoded_bytes": 0,
                "status": "BUILT_REUSE_SURFACE_BUT_NO_MOVABLE_PCOMP3_RECORDS_ADMITTED",
            },
        ],
        "components": components,
        "unique_home_caveat": (
            "LBND is a shared analytic Lane generator and has no custodied V9 cell/boundary/critical "
            "byte split. PCOMP3 rows below are exact unique-home packet bytes and remain unchanged."
        ),
    }


def _project_archive(
    *,
    sections: tuple[SectionBytes, ...],
    static_encoded: bytes,
    static_raw: bytes,
    xi_bundle: bytes,
    output: Path,
    fingerprint: str,
) -> dict[str, Any]:
    stage_path = output / "checkpoints" / "stage_projection.json"
    projected_dir = output / "projected_full_archive"
    archive_path = projected_dir / "archive.zip"
    if stage_path.is_file() and archive_path.is_file():
        existing = json.loads(stage_path.read_text(encoding="utf-8"))
        archive_row = existing.get("archive", {})
        if (
            existing.get("input_fingerprint") == fingerprint
            and archive_path.stat().st_size == archive_row.get("bytes")
            and _sha_file(archive_path) == archive_row.get("sha256")
        ):
            with zipfile.ZipFile(archive_path) as zipped:
                parse_sections(zipped.read("0.bin"))
            return existing

    by_name = _section_map(sections)
    xi_terminal = _lzma(xi_bundle)
    candidate_base = struct.pack("<II", len(static_encoded), len(xi_terminal)) + static_encoded + xi_terminal
    candidate_rows: list[SectionBytes] = []
    for name in SECTION_ORDER[1:]:
        original = by_name[name]
        if name == "base.pbase3":
            candidate_rows.append(
                SectionBytes(
                    name,
                    candidate_base,
                    "mixed",
                    len(static_raw) + len(xi_bundle),
                )
            )
        else:
            candidate_rows.append(original)

    manifest = json.loads(by_name["manifest.json"].payload.decode("ascii"))
    manifest["section_registry"] = [
        {
            "name": row.name,
            "codec": row.codec,
            "encoded_bytes": len(row.payload),
            "decoded_bytes": row.decoded_bytes,
            "sha256": _sha_bytes(row.payload),
            "registry_version": row.registry_version,
        }
        for row in candidate_rows
    ]
    manifest["runtime"]["standalone_receiver_closed"] = False
    manifest["xi_temporal_delta_coder"] = {
        "schema": "xi_temporal_lane_bundle.v1",
        "bundle_bytes": len(xi_bundle),
        "bundle_sha256": _sha_bytes(xi_bundle),
        "terminal_lzma_bytes": len(xi_terminal),
        "repository_decoder_bit_exact": True,
        "standalone_s4_receiver_integration": "OWED_MAIN",
        "implementation_fingerprint": fingerprint,
        "score_claim": False,
    }
    manifest["limitations"]["current_pointer_moved"] = False
    manifest["limitations"]["score_claim"] = False
    manifest["limitations"]["promotion_eligible"] = False
    manifest["limitations"]["xi_temporal_projection"] = (
        "exact one-member ZIP accounting; repository decode exact; standalone receiver support not executed "
        "because the pinned runtime dispatches LBND2 rather than XTDL1"
    )
    manifest_bytes = canonical_json_bytes(manifest)
    all_rows = [SectionBytes("manifest.json", manifest_bytes, "raw", len(manifest_bytes)), *candidate_rows]
    monolith = serialize_sections(all_rows)
    if serialize_sections(parse_sections(monolith)) != monolith:
        raise MeasureXiTemporalError("projected S4 container parse-back failed")
    _atomic_bytes(projected_dir / "0.bin", monolith)
    archive_row = deterministic_archive(archive_path, monolith)

    parsed = _section_map(parse_sections(monolith))
    projected_base = parsed["base.pbase3"].payload
    s_len, l_len = struct.unpack_from("<II", projected_base)
    decoded_bundle = lzma.decompress(
        projected_base[8 + s_len : 8 + s_len + l_len],
        format=lzma.FORMAT_RAW,
        filters=LZMA_FILTERS,
    )
    if decoded_bundle != xi_bundle:
        raise MeasureXiTemporalError("projected archive did not recover exact XTDL1 bytes")
    decoded_q, decoded_presence, _header = decode_lane_xi_temporal_grid(decoded_bundle)
    if decoded_q.shape[0] != 600 or decoded_presence.shape[0] != 600:
        raise MeasureXiTemporalError("projected archive XTDL1 decode did not cover n600")

    description_bytes = len(candidate_base) + len(parsed["components.pcomp3"].payload)
    nonmanifest_logical = sum(len(parsed[name].payload) for name in SECTION_ORDER[1:])
    result = {
        "schema": "xi_temporal_574_projected_archive.v1",
        "input_fingerprint": fingerprint,
        "archive": archive_row,
        "source_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "delta_archive_bytes": archive_row["bytes"] - EXPECTED_ARCHIVE_BYTES,
        "description_base_plus_components": {
            "before_bytes": EXPECTED_DESCRIPTION_BASE_COMPONENTS,
            "after_bytes": description_bytes,
            "delta_bytes": description_bytes - EXPECTED_DESCRIPTION_BASE_COMPONENTS,
            "definition": "base.pbase3 + components.pcomp3 only",
        },
        "all_nonmanifest_sections": {
            "before_bytes": sum(len(by_name[name].payload) for name in SECTION_ORDER[1:]),
            "after_bytes": nonmanifest_logical,
            "delta_bytes": nonmanifest_logical - sum(len(by_name[name].payload) for name in SECTION_ORDER[1:]),
        },
        "phase_boxes": [
            {
                "box_bytes": box,
                "description_after_minus_box": description_bytes - box,
                "projected_archive_minus_box": archive_row["bytes"] - box,
            }
            for box in PHASE_BOXES
        ],
        "parse_back": {
            "s4_container_exact": True,
            "xtdl1_repository_decode_exact": True,
            "standalone_s4_receiver_closed": False,
            "standalone_probe_status": "NOT_RUN_UNSUPPORTED_CODEC_DERIVED_FROM_PINNED_RUNTIME_SOURCE",
            "standalone_runtime_sha256": EXPECTED_S4_RUNTIME_SHA256,
            "blocker": "STANDALONE_S4_XTDL1_CODEC_INTEGRATION_OWED_MAIN",
        },
        "composition_law": (
            "A(theta)=len(DetZip9(SerializeS4(updated_manifest,seed,base_xtdl1,causal,events,components)))"
        ),
        "invalid_shortcut_refused": "451191 - 216207 + D_new",
    }
    _atomic_json(stage_path, result)
    return result


def _source_commit() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise MeasureXiTemporalError("could not resolve a full source commit")
    return value


def measure(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    storage = _storage_preflight(output)
    source_authority = _source_authority()
    runtime_custody = _runtime_custody()
    monolith, sections, source_archive = _load_archive(args.archive)
    source_archive["authority"] = source_authority
    by_name = _section_map(sections)
    (
        static_encoded,
        static_raw,
        lane_encoded,
        lane_raw,
        source_q_lane,
        source_presence,
        lane_header,
        config,
    ) = _decode_base(by_name["base.pbase3"].payload)
    full_xi, xi_custody = _corrected_full_chasles(args.bev_root, args.scale_receipt)

    implementation_paths = [
        "src/tac/optimization/xi_temporal_delta_coder.py",
        "tools/measure_xi_temporal_delta_coder.py",
        "src/tac/shared_pmf_model.py",
        "src/tac/lossless/range_coder.py",
        "src/tac/boundary_math/analytic_lane_render_band.py",
        "src/tac/boundary_math/lane_sdf_component.py",
        "src/tac/boundary_math/lane_track_and_smooth.py",
        "src/tac/boundary_math/ego_xi_trajectory.py",
        "src/tac/boundary_math/warp_real_luma_frame0.py",
        "src/tac/boundary_math/xi_pose_coder.py",
        "src/tac/boundary_math/xi_spline_residual_coder.py",
        "src/tac/lie/_se3_numpy.py",
        "src/tac/optimization/s4_archive_composer.py",
    ]
    implementation_sources = {path: _sha_file(REPO / path) for path in implementation_paths}
    fingerprint = _sha_bytes(
        canonical_json_bytes(
            {
                "source_archive_sha256": source_archive["sha256"],
                "source_authority": source_authority,
                "xi_source_sha256": xi_custody["xi_fp64_sha256"],
                "seed": PMF_SEED,
                "implementation_sources": implementation_sources,
                "runtime_custody": runtime_custody,
            }
        )
    )

    inventory = _inventory(sections, lane_header, lane_raw)
    _atomic_json(output / "checkpoints" / "stage_inventory.json", inventory)
    prefix_results = {}
    for pairs in PAIR_COUNTS:
        prefix_results[f"n{pairs}"] = _measure_prefix(
            pairs=pairs,
            lane_header=lane_header,
            config=config,
            full_xi=full_xi,
            source_q_lane=source_q_lane,
            source_presence=source_presence,
            original_lane_raw=lane_raw,
            original_lane_encoded=lane_encoded,
            output=output,
            fingerprint=fingerprint,
        )
    xi_n600_path = Path(prefix_results["n600"]["artifacts"]["xi"]["path"])
    projection = _project_archive(
        sections=sections,
        static_encoded=static_encoded,
        static_raw=static_raw,
        xi_bundle=xi_n600_path.read_bytes(),
        output=output,
        fingerprint=fingerprint,
    )

    canonical_per_stratum_tool = REPO / "tools/measure_per_stratum_recursive_fractal_optimal.py"
    receipt = {
        "schema": "xi_temporal_delta_coder_574_measurement.v1",
        "task": 574,
        "seed": PMF_SEED,
        "axis": "[macOS-CPU advisory] rate/description only; not contest score authority",
        "input_fingerprint": fingerprint,
        "delegation_base_commit": _source_commit(),
        "source_archive": source_archive,
        "source_monolith": {"bytes": len(monolith), "sha256": _sha_bytes(monolith)},
        "xi_custody": xi_custody,
        "implementation_sources": implementation_sources,
        "runtime_custody": runtime_custody,
        "storage": storage,
        "stream_inventory": inventory,
        "measurements": prefix_results,
        "projected_archive": projection,
        "canonical_per_stratum": {
            "required_tool": str(canonical_per_stratum_tool.relative_to(REPO)),
            "present_at_delegation_base": canonical_per_stratum_tool.is_file(),
            "run": False,
            "blocker": "CANONICAL_PER_STRATUM_TOOL_ABSENT_AT_DELEGATION_BASE",
            "quarantine_note": (
                "the later tool's settled control consumes quarantined M1 bytes; this lane does not "
                "recreate or silently substitute that authority"
            ),
            "existing_v9_status": "NO_VERDICT_RECEIVER_RATE_CUSTODY",
            "this_receipt_contribution": (
                "exact PCOMP3 unique-home packet inventory plus shared-Lane generator rate; MAIN "
                "must wire the receipt into the canonical tool before claiming per-stratum closure"
            ),
        },
        "verdict": {
            "formulation": (
                "3-DOF planar (rho_z,rho_x,omega_y) projection of the corrected composed full "
                "screw on the settled ground-frame coherent-slot LBND chart, shared-PMF residual coding"
            ),
            "n64_repository_codec_semantic_grid_exact": True,
            "n600_repository_codec_semantic_grid_exact": True,
            "standalone_s4_receiver_closed": False,
            "xi_rate_win": bool(prefix_results["n600"]["admission"]["xi_beats_settled_lbnd2"]),
            "scope": (
                "NARROW_LANE_FORMULATION_NEGATIVE_IF_FALSE; solved-object stream unmeasured; full-6D and "
                "non-Lane temporal coding remain open"
            ),
            "source_object_blocker": "SOURCE_S4_NOT_SOLVED_POINTER_OBJECT_DESCRIPTION_STREAM",
            "unimplemented_stream_families": [
                "PPCS_seed_temporal_recode",
                "PCE3_event_temporal_recode",
                "PCOMP3_persistent_object_temporal_recode",
                "Movable_site_temporal_recode",
            ],
            "promotion_eligible": False,
            "pointer_moved": False,
            "pointer": "0.1910828242 [contest-CPU]",
        },
        "main_landing_review_required": True,
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--bev-root", type=Path, default=DEFAULT_BEV)
    parser.add_argument("--scale-receipt", type=Path, default=DEFAULT_SCALE_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    measure(parse_args())
