#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Join the exact v19c endpoint to the DDM SN1 error-source tensor.

This is an advisory local CPU analysis.  It replays the exact SHA-pinned v19c
receiver in canonical batches, checks every camera/argmax/error row against the
preserved strict v19c replay, and then performs only joins and typed reductions.
Historical G3, G4, and v14 evidence remains explicitly labelled as a proxy or
cross-check; it is never promoted to current causal truth.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.analysis.ddm_sn1_error_source_tensor import (  # noqa: E402
    BOUNDARY_DISTANCE_BANDS,
    CLASS_NAMES,
    CURVATURE_BANDS,
    HEIGHT,
    MARGIN_BANDS,
    N600_SITES,
    PAINT_FLOOR_MECHANISMS,
    SOURCE_NAMES,
    TEMPORAL_PATTERNS,
    WIDTH,
    ErrorSource,
    ErrorSourceTensorError,
    boundary_distance_bands,
    classify_error_sources,
    curvature_bands,
    d2_margin_bands,
    decode_group_key,
    encode_group_key,
    paint_floor_mechanism_codes,
    source_budget,
    summarize_components,
    survival_wall_149,
    temporal_pattern_codes,
)
from tac.analysis.scorer_native_diff import (  # noqa: E402
    ScorerNativeDiffError,
    analytic_scorer_knowledge,
    finalize_scorer_native_product,
    measure_scorer_native_product,
)
from tac.analysis.segnet_amplitude_telemetry import (  # noqa: E402
    SegNetAmplitudeTelemetryError,
    measure_paired_segnet_amplitude,
)
from tac.optimization import direct_description_carrier_compose as _dcc  # noqa: E402
from tac.optimization import direct_description_coupled_margin as _ddcm  # noqa: E402
from tac.optimization import direct_description_preuint8_channel as _ddq8  # noqa: E402
from tac.optimization.ddm_description_vocabulary import (  # noqa: E402
    decode_boundary_worldsheet_spline,
    decode_joint_ground_vocabulary,
    decode_persistent_level_set,
)
from tac.optimization.direct_description_g1_worldsheet import (  # noqa: E402
    G1MovableWorldsheetMetadata,
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)

AXIS: Final = "[macOS-CPU frozen-SegNet+PoseNet advisory]"
SCHEMA: Final = "ddm_sn1_error_source_tensor_receipt.v1"
TARGET_CLASS_IDS: Final = (0, 2, 4)
EXPECTED_SOURCE_ERRORS: Final = {
    "Road": 1_935_140,
    "Undrivable": 293_434,
    "MyCar": 37_237,
}
EXPECTED_RESIDUAL_ERRORS: Final = sum(EXPECTED_SOURCE_ERRORS.values())
APPROVED_SSD_ROOTS: Final = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
IMPLEMENTATION_FILES: Final = {
    "runner": Path(__file__).resolve(),
    "tensor_library": (REPO_ROOT / "src/tac/analysis/ddm_sn1_error_source_tensor.py").resolve(),
    "amplitude_library": (REPO_ROOT / "src/tac/analysis/segnet_amplitude_telemetry.py").resolve(),
    "scorer_native_diff_library": (REPO_ROOT / "src/tac/analysis/scorer_native_diff.py").resolve(),
    "full_screw_mapping": (REPO_ROOT / "src/tac/optimization/predict_project_receiver.py").resolve(),
    "full_screw_geometry": (REPO_ROOT / "src/tac/boundary_math/warp_real_luma_frame0.py").resolve(),
}


class TensorBuildError(RuntimeError):
    """Raised when custody, replay, or tensor accounting fails."""


class DDMErrorSourceTensorConfigV1(BaseModel):
    """Strict typed config for the n600 join."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema: str = "ddm_sn1_error_source_tensor_config.v1"
    run_id: str
    source_repo_root: Path
    target_cache_path: Path
    v19c_receipt_path: Path
    v19c_final_archive_path: Path
    v19c_strict_batch_directory: Path
    g2_receipt_path: Path
    g2_aggregate_path: Path
    g3_atlas_path: Path
    g4_arrays_path: Path
    dv1_receipt_path: Path
    dv1_summary_path: Path
    dv1_selected_payload_path: Path
    v14_receipt_path: Path
    e1_receipt_path: Path
    sided_tolerance_path: Path
    inverse_receipt_path: Path
    survival_wall_149_path: Path
    advected_screw6_receipt_path: Path
    upstream_root: Path
    output_directory: Path
    scratch_directory: Path
    source_sha256: dict[str, str]
    batch_size: int = Field(default=16, ge=1, le=16)
    seed: int = 210
    torch_threads: int = Field(default=4, ge=1, le=16)
    telemetry_microbatch_size: int = Field(default=2, ge=1, le=4)
    expected_residual_errors: int = EXPECTED_RESIDUAL_ERRORS
    expected_residual_errors_by_class: dict[str, int] = Field(default_factory=lambda: dict(EXPECTED_SOURCE_ERRORS))
    strict_replay_digest_chain_sha256: str

    @model_validator(mode="after")
    def _validate_contract(self) -> DDMErrorSourceTensorConfigV1:
        if self.schema != "ddm_sn1_error_source_tensor_config.v1":
            raise ValueError("config schema differs")
        if self.expected_residual_errors != EXPECTED_RESIDUAL_ERRORS:
            raise ValueError("expected residual endpoint differs from binding")
        if self.expected_residual_errors_by_class != EXPECTED_SOURCE_ERRORS:
            raise ValueError("expected residual class endpoint differs from binding")
        if set(self.source_sha256) != set(SOURCE_PATH_FIELDS):
            raise ValueError("source_sha256 keys do not exactly match source paths")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.source_sha256.values()
        ):
            raise ValueError("source SHA-256 value is malformed")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        return value

    def typed_config_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.canonical_payload())).hexdigest()


SOURCE_PATH_FIELDS: Final = (
    "target_cache_path",
    "v19c_receipt_path",
    "v19c_final_archive_path",
    "g2_receipt_path",
    "g2_aggregate_path",
    "g3_atlas_path",
    "g4_arrays_path",
    "dv1_receipt_path",
    "dv1_summary_path",
    "dv1_selected_payload_path",
    "v14_receipt_path",
    "e1_receipt_path",
    "sided_tolerance_path",
    "inverse_receipt_path",
    "survival_wall_149_path",
    "advected_screw6_receipt_path",
    "upstream_modules_path",
    "segnet_weights_path",
    "posenet_weights_path",
)


@dataclass(frozen=True, slots=True)
class PairCovariates:
    score_rank: int
    g3_tail_bucket: str
    scene_event_labels: tuple[str, ...]


class LazyG1Mask:
    """Lossless per-pair G1 rasterization without a full n600 mask tensor."""

    def __init__(self, payload: bytes) -> None:
        lift = lift_g1_movable_worldsheet(payload)
        if encode_lifted_g1_movable_worldsheet(lift) != payload:
            raise TensorBuildError("G1 lift/re-emission changed bytes")
        templates = {row.template_ref: row.relative_vertices_xy for row in lift.templates}
        rows: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
        for knot in lift.knots:
            relative = templates[knot.template_ref]
            rows[int(knot.pair_index)].append(
                tuple(
                    (
                        int(knot.center_x) + int(x_value),
                        int(knot.center_y) + int(y_value),
                    )
                    for x_value, y_value in relative
                )
            )
        self._rows = {pair_id: tuple(sorted(polygons)) for pair_id, polygons in rows.items()}
        self.pair_count = int(lift.pair_count)
        self.max_slots = int(lift.max_slots)
        self.knots = len(lift.knots)
        self.templates = len(lift.templates)
        self.payload_sha256 = hashlib.sha256(payload).hexdigest()

    def __getitem__(self, pair_id: int) -> np.ndarray:
        value = int(pair_id)
        if not 0 <= value < self.pair_count:
            raise IndexError(value)
        output = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for polygon in self._rows.get(value, ()):
            cv2.fillPoly(
                output,
                [np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)],
                1,
            )
        return output.astype(bool)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n",
    )


def atomic_gzip_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write canonical JSONL as deterministic gzip and return both identities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    content_digest = hashlib.sha256()
    content_bytes = 0
    try:
        with temporary.open("wb") as handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=handle,
                mtime=0,
            ) as compressed:
                for row in rows:
                    line = canonical_json_bytes(row) + b"\n"
                    content_digest.update(line)
                    content_bytes += len(line)
                    compressed.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **path_identity(path),
        "compression": "gzip_level_9_mtime_0",
        "jsonl_bytes": content_bytes,
        "jsonl_sha256": content_digest.hexdigest(),
        "row_count": len(rows),
    }


def path_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def implementation_identity() -> dict[str, Any]:
    files = {name: path_identity(path) for name, path in IMPLEMENTATION_FILES.items()}
    return {
        "schema": "ddm_sn1_error_source_tensor_implementation.v1",
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def load_config(path: Path) -> DDMErrorSourceTensorConfigV1:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TensorBuildError(f"config is unreadable: {path}") from exc
    try:
        return DDMErrorSourceTensorConfigV1.model_validate(value)
    except ValueError as exc:
        raise TensorBuildError(f"typed config is invalid: {exc}") from exc


def source_paths(config: DDMErrorSourceTensorConfigV1) -> dict[str, Path]:
    return {
        "target_cache_path": config.target_cache_path,
        "v19c_receipt_path": config.v19c_receipt_path,
        "v19c_final_archive_path": config.v19c_final_archive_path,
        "g2_receipt_path": config.g2_receipt_path,
        "g2_aggregate_path": config.g2_aggregate_path,
        "g3_atlas_path": config.g3_atlas_path,
        "g4_arrays_path": config.g4_arrays_path,
        "dv1_receipt_path": config.dv1_receipt_path,
        "dv1_summary_path": config.dv1_summary_path,
        "dv1_selected_payload_path": config.dv1_selected_payload_path,
        "v14_receipt_path": config.v14_receipt_path,
        "e1_receipt_path": config.e1_receipt_path,
        "sided_tolerance_path": config.sided_tolerance_path,
        "inverse_receipt_path": config.inverse_receipt_path,
        "survival_wall_149_path": config.survival_wall_149_path,
        "advected_screw6_receipt_path": config.advected_screw6_receipt_path,
        "upstream_modules_path": config.upstream_root / "modules.py",
        "segnet_weights_path": config.upstream_root / "models/segnet.safetensors",
        "posenet_weights_path": config.upstream_root / "models/posenet.safetensors",
    }


def validate_sources(config: DDMErrorSourceTensorConfigV1) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for name, path in source_paths(config).items():
        if not path.is_file():
            raise TensorBuildError(f"source is absent: {name}={path}")
        identities[name] = path_identity(path)
        if identities[name]["sha256"] != config.source_sha256[name]:
            raise TensorBuildError(f"{name} SHA drift: {identities[name]['sha256']} != {config.source_sha256[name]}")
    strict_rows = sorted(config.v19c_strict_batch_directory.glob("batch_*.json"))
    if len(strict_rows) != 38:
        raise TensorBuildError("v19c strict batch directory must contain 38 receipts")
    chain = hashlib.sha256(
        "".join(
            json.loads(path.read_text())["camera_sha256"]
            + json.loads(path.read_text())["cells_sha256"]
            + json.loads(path.read_text())["pose6_sha256"]
            for path in strict_rows
        ).encode()
    ).hexdigest()
    if chain != config.strict_replay_digest_chain_sha256:
        raise TensorBuildError("v19c strict replay digest chain drifted")
    identities["v19c_strict_batch_directory"] = {
        "path": str(config.v19c_strict_batch_directory),
        "batch_count": len(strict_rows),
        "digest_chain_sha256": chain,
    }
    return identities


def storage_preflight(config: DDMErrorSourceTensorConfigV1) -> dict[str, Any]:
    config.scratch_directory.mkdir(parents=True, exist_ok=True)
    resolved = config.scratch_directory.resolve()
    first = next((root for root in APPROVED_SSD_ROOTS if root.exists()), None)
    if first is None or not (resolved == first or first in resolved.parents):
        raise TensorBuildError("scratch does not use the first available SSD tier")
    required = 8 * 1024 * 1024 * 1024
    free = shutil.disk_usage(resolved).free
    if free < required:
        raise TensorBuildError("SSD storage preflight failed")
    return {
        "status": "PASS",
        "selected_root": str(first),
        "scratch_directory": str(resolved),
        "required_free_bytes": required,
        "observed_free_bytes": free,
        "large_artifact_policy": "CERTIFY_OR_BLOCK",
        "cleanup": (
            "Tensor and resumable batch JSONL are atomically emitted as deterministic "
            "gzip; bulky stage checkpoints are written directly to the selected SSD "
            "scratch tier through a source-path symlink and certified at finalization."
        ),
    }


def checkpoint_cold_store_destination(
    *,
    config: DDMErrorSourceTensorConfigV1,
    config_hash: str,
    implementation: Mapping[str, Any],
) -> Path:
    return (
        config.scratch_directory
        / (
            f"stage_checkpoints_{config_hash[:16]}_"
            f"{implementation['bundle_sha256'][:16]}"
        )
    ).resolve()


def prepare_stage_checkpoint_directory(
    *,
    config: DDMErrorSourceTensorConfigV1,
    config_hash: str,
    implementation: Mapping[str, Any],
    checkpoint_directory: Path,
) -> None:
    """Put new bulky checkpoints on SSD before the first stage is written."""

    destination = checkpoint_cold_store_destination(
        config=config,
        config_hash=config_hash,
        implementation=implementation,
    )
    if checkpoint_directory.is_symlink():
        if checkpoint_directory.resolve() != destination:
            raise TensorBuildError("stage-checkpoint symlink destination drifted")
        return
    if checkpoint_directory.exists():
        # A legacy/interrupted local tree remains valid and will be moved only
        # after a complete content manifest exists.
        return
    if destination.exists():
        raise TensorBuildError(
            f"unbound checkpoint cold-store destination exists: {destination}"
        )
    destination.mkdir(parents=True)
    checkpoint_directory.symlink_to(destination, target_is_directory=True)


def checkpoint_tree_identity(root: Path) -> dict[str, Any]:
    """Content-bind one checkpoint tree using sorted relative file names."""

    resolved = root.resolve()
    paths = sorted(path for path in resolved.rglob("*") if path.is_file())
    lines: list[str] = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(resolved)
        size = path.stat().st_size
        total_bytes += size
        lines.append(f"{sha256_file(path)}  {relative.as_posix()}\n")
    return {
        "tree_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "file_count": len(paths),
        "bytes": total_bytes,
        "tree_hash_contract": (
            "sha256 of concatenated sorted '<file_sha256>  <relative_path>\\n' "
            "records over regular files"
        ),
    }


def externalize_stage_checkpoints(
    *,
    config: DDMErrorSourceTensorConfigV1,
    config_hash: str,
    implementation: Mapping[str, Any],
    checkpoint_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Certify and losslessly move bulky resumable stages to the SSD tier."""

    destination = checkpoint_cold_store_destination(
        config=config,
        config_hash=config_hash,
        implementation=implementation,
    )
    manifest_path = output_directory / "stage_checkpoint_cold_store_manifest.json"
    if checkpoint_directory.is_symlink():
        if checkpoint_directory.resolve() != destination:
            raise TensorBuildError("cold-store checkpoint symlink drifted")
        tree = checkpoint_tree_identity(destination)
    else:
        if destination.exists():
            raise TensorBuildError(
                f"checkpoint cold-store destination exists: {destination}"
            )
        tree = checkpoint_tree_identity(checkpoint_directory)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            Path(manifest["cold_store_destination"]).resolve() != destination
            or tree["tree_sha256"] != manifest["tree_sha256"]
        ):
            raise TensorBuildError("cold-store checkpoint custody drifted")
        return {
            **manifest,
            "manifest_artifact": path_identity(manifest_path),
        }
    if not checkpoint_directory.is_symlink() and destination.exists():
        raise TensorBuildError(f"checkpoint cold-store destination exists: {destination}")
    manifest = {
        "schema": "ddm_sn1_stage_checkpoint_cold_store.v1",
        "original_path": str(checkpoint_directory),
        "cold_store_destination": str(destination),
        **tree,
        "command": [
            "/Users/adpena/Projects/pact/.venv/bin/python",
            "-u",
            "tools/build_ddm_sn1_error_source_tensor.py",
            "--config",
            ".omx/research/configs/ddm_sn1_error_source_tensor_20260723.json",
        ],
        "typed_config_sha256": config_hash,
        "implementation_bundle_sha256": implementation["bundle_sha256"],
        "source_archive_sha256": config.source_sha256[
            "v19c_final_archive_path"
        ],
        "target_cache_sha256": config.source_sha256["target_cache_path"],
        "false_authority_flags": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
        "reason": (
            "Preserve all crash-resume and per-stage measurement checkpoints "
            "without leaving 4+ GiB of rebuildable bulk on the source tier."
        ),
        "rebuildable": True,
        "move_policy": "LOSSLESS_SSD_MOVE_WITH_ORIGINAL_PATH_SYMLINK",
    }
    atomic_json(manifest_path, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_directory.is_symlink():
        shutil.move(str(checkpoint_directory), str(destination))
        checkpoint_directory.symlink_to(destination, target_is_directory=True)
    if checkpoint_directory.resolve() != destination:
        raise TensorBuildError("checkpoint cold-store symlink creation failed")
    return {
        **manifest,
        "manifest_artifact": path_identity(manifest_path),
    }


def stable_storage_identity(storage: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude volatile capacity telemetry from the crash-resume identity."""

    return {key: value for key, value in storage.items() if key != "observed_free_bytes"}


def stable_resume_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only volatile observations before comparing run identities."""

    normalized = dict(identity)
    normalized["storage_preflight"] = stable_storage_identity(
        normalized["storage_preflight"],
    )
    return normalized


def stable_measurement_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a checkpoint run independently of a later aggregate finalizer."""

    normalized = stable_resume_identity(identity)
    normalized.pop("implementation", None)
    return normalized


def stored_npy_memmap(path: Path, key: str) -> np.memmap:
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
            raise TensorBuildError(f"{path}:{member} is not ZIP_STORED")
        local_header = int(info.header_offset)
    with path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise TensorBuildError(f"bad ZIP local header for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def lazy_g1_decode(
    payload: bytes,
    *,
    expected_pairs: int | None = None,
) -> tuple[LazyG1Mask, G1MovableWorldsheetMetadata]:
    value = LazyG1Mask(payload)
    if expected_pairs is not None and value.pair_count != expected_pairs:
        raise TensorBuildError("lazy G1 pair count differs")
    metadata = G1MovableWorldsheetMetadata(
        pair_count=value.pair_count,
        max_slots=value.max_slots,
        births=0,
        persists=0,
        deaths=0,
        vertices=0,
        production_counted_bytes={},
        payload_bytes=len(payload),
        payload_sha256=value.payload_sha256,
        decoded_mask_errors=None,
        decoded_clean_rest_dseg=None,
    )
    return value, metadata


def receive_v19c_lazy(archive: bytes) -> Any:
    """Decode exact v19c bytes with lazy G1 rasterization.

    The source archive already has a preserved strict full-receiver proof.  This
    join validates the lazy decoder against every preserved camera and cell
    batch, so no lazy output is trusted merely because parsing succeeded.
    """

    outer, _outer_homes = _ddq8.parse_preuint8_q8_archive(archive)
    coupled_archive = outer[_ddq8.BASE_MEMBER]
    coupled, _coupled_homes = _ddcm.parse_coupled_margin_archive(coupled_archive)
    carrier_archive = coupled[_ddcm.BASE_MEMBER]
    original = _dcc.decode_g1_movable_worldsheet
    _dcc.decode_g1_movable_worldsheet = lazy_g1_decode
    try:
        carrier = _dcc.receive_carrier_compose_archive(
            carrier_archive,
            verify_member_effects=False,
        )
    finally:
        _dcc.decode_g1_movable_worldsheet = original
    coupled_program = _ddcm.decode_coupled_margin_program(coupled[_ddcm.PROGRAM_MEMBER])
    _ddcm._validate_program_against_receiver(coupled_program, carrier)
    coupled_receiver = _ddcm.CoupledMarginReceiverV1(
        coupled_archive,
        carrier,
        coupled_program,
        {},
    )
    preuint8_program = _ddq8.decode_preuint8_q8_program(outer[_ddq8.PROGRAM_MEMBER])
    _ddq8._validate_program(preuint8_program, coupled_receiver)
    return _ddq8.PreUint8Q8ReceiverV1(
        archive,
        coupled_receiver,
        preuint8_program,
        {
            "lazy_g1": True,
            "strict_validation": ("every camera/cell/error batch checked against preserved v19c strict replay"),
        },
    )


def load_segnet(config: DDMErrorSourceTensorConfigV1) -> Any:
    import torch
    from safetensors.torch import load_file

    if str(config.upstream_root) not in sys.path:
        sys.path.insert(0, str(config.upstream_root))
    from modules import SegNet

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    model = SegNet().eval().cpu()
    model.load_state_dict(
        load_file(
            str(config.upstream_root / "models/segnet.safetensors"),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_posenet(config: DDMErrorSourceTensorConfigV1) -> Any:
    import torch
    from safetensors.torch import load_file

    if str(config.upstream_root) not in sys.path:
        sys.path.insert(0, str(config.upstream_root))
    from modules import PoseNet

    torch.manual_seed(config.seed)
    model = PoseNet().eval().cpu()
    model.load_state_dict(
        load_file(
            str(config.upstream_root / "models/posenet.safetensors"),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def forward_segnet(model: Any, camera_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    value = np.asarray(camera_pairs)
    if (
        value.dtype != np.uint8
        or value.ndim != 5
        or value.shape[1:]
        != (
            2,
            874,
            1164,
            3,
        )
    ):
        raise TensorBuildError("SegNet join requires uint8 [B,2,874,1164,3]")
    tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits = model(model.preprocess_input(tensor))
        cells = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        logits_numpy = logits.cpu().numpy().astype(np.float32)
    return np.ascontiguousarray(cells), np.ascontiguousarray(logits_numpy)


def scorer_native_products_for_batch(
    *,
    segnet: Any,
    posenet: Any,
    camera_pairs: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    gt_pose6: np.ndarray,
    full_xi: np.ndarray,
    pitch_rad: float,
    microbatch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure both frozen scorer trajectories without changing verdict replay."""

    import torch

    painted = np.asarray(camera_pairs)
    ground_f0 = np.asarray(gt_f0)
    ground_f1 = np.asarray(gt_f1)
    poses = np.asarray(gt_pose6)
    xi = np.asarray(full_xi)
    batch_size = painted.shape[0]
    if (
        painted.dtype != np.uint8
        or painted.shape != (batch_size, 2, 874, 1164, 3)
        or ground_f0.shape != (batch_size, 874, 1164, 3)
        or ground_f1.shape != ground_f0.shape
        or ground_f0.dtype != np.uint8
        or ground_f1.dtype != np.uint8
        or poses.shape != (batch_size, 6)
        or xi.shape != (batch_size, 6)
    ):
        raise TensorBuildError("scorer-native batch geometry differs")
    seg_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    for begin in range(0, batch_size, microbatch_size):
        end = min(begin + microbatch_size, batch_size)
        painted_raw = (
            torch.from_numpy(np.ascontiguousarray(painted[begin:end]))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .float()
        )
        gt_pair_numpy = np.stack(
            (ground_f0[begin:end], ground_f1[begin:end]),
            axis=1,
        )
        gt_raw = (
            torch.from_numpy(np.ascontiguousarray(gt_pair_numpy))
            .permute(0, 1, 4, 2, 3)
            .contiguous()
            .float()
        )

        def repeated_frame(value: torch.Tensor, index: int) -> torch.Tensor:
            selected = value[:, index : index + 1]
            return selected.repeat(1, 2, 1, 1, 1)

        seg_groups = {
            "painted_f0": segnet.preprocess_input(
                repeated_frame(painted_raw, 0)
            ),
            "painted_f1": segnet.preprocess_input(
                repeated_frame(painted_raw, 1)
            ),
            "gt_f0": segnet.preprocess_input(repeated_frame(gt_raw, 0)),
            "gt_f1": segnet.preprocess_input(repeated_frame(gt_raw, 1)),
        }
        seg_output, seg_batch = measure_scorer_native_product(
            segnet,
            scorer="segnet",
            grouped_inputs=seg_groups,
            contrasts={
                "painted_f0_vs_gt_f0": {
                    "painted_f0": 1.0,
                    "gt_f0": -1.0,
                },
                "painted_f1_vs_gt_f1": {
                    "painted_f1": 1.0,
                    "gt_f1": -1.0,
                },
                "painted_temporal": {
                    "painted_f1": 1.0,
                    "painted_f0": -1.0,
                },
                "gt_temporal": {"gt_f1": 1.0, "gt_f0": -1.0},
                "temporal_residual": {
                    "painted_f1": 1.0,
                    "painted_f0": -1.0,
                    "gt_f1": -1.0,
                    "gt_f0": 1.0,
                },
            },
            xi=xi[begin:end],
            pitch_rad=pitch_rad,
            transport_groups=("gt_f0", "gt_f1"),
        )
        seg_rows.append(seg_batch)

        pose_output, pose_batch = measure_scorer_native_product(
            posenet,
            scorer="posenet",
            grouped_inputs={
                "painted_pair": posenet.preprocess_input(painted_raw),
                "gt_pair": posenet.preprocess_input(gt_raw),
            },
            contrasts={
                "painted_pair_vs_gt_pair": {
                    "painted_pair": 1.0,
                    "gt_pair": -1.0,
                }
            },
        )
        pose_tensor = pose_output["pose"]
        split = end - begin
        painted_pose6 = pose_tensor[:split, :6].detach().cpu().numpy()
        replay_pose6 = pose_tensor[split:, :6].detach().cpu().numpy()
        expected_pose6 = poses[begin:end].astype(np.float32)
        pose_batch["pose6"] = {
            "coordinate_count": int(painted_pose6.size),
            "painted_vs_gt_sse": float(
                np.square(painted_pose6 - replay_pose6, dtype=np.float64).sum()
            ),
            "gt_cache_replay_max_abs": float(
                np.max(
                    np.abs(
                        replay_pose6.astype(np.float64)
                        - expected_pose6.astype(np.float64)
                    )
                )
            ),
            "painted_pose6": painted_pose6.astype(np.float64).tolist(),
            "gt_replay_pose6": replay_pose6.astype(np.float64).tolist(),
            "gt_cache_pose6": expected_pose6.astype(np.float64).tolist(),
            "authority": (
                "official frozen PoseNet first-six output; microbatch geometry "
                "recorded and advisory"
            ),
        }
        pose_rows.append(pose_batch)
        del (
            painted_raw,
            gt_raw,
            seg_groups,
            seg_output,
            pose_output,
            pose_tensor,
        )
    return seg_rows, pose_rows


def summarize_pose6_product(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce the frozen PoseNet first-six output without changing its geometry."""

    if not batches:
        raise TensorBuildError("PoseNet first-six summary requires batches")
    coordinate_count = sum(
        int(batch["pose6"]["coordinate_count"]) for batch in batches
    )
    pair_count = sum(int(batch["batch_size"]) for batch in batches)
    if coordinate_count != pair_count * 6:
        raise TensorBuildError("PoseNet first-six coordinate accounting differs")
    sse = sum(float(batch["pose6"]["painted_vs_gt_sse"]) for batch in batches)
    return {
        "pair_count": pair_count,
        "coordinate_count": coordinate_count,
        "painted_vs_gt_sse": sse,
        "d_pose_first_six_mse": sse / coordinate_count,
        "gt_cache_replay_max_abs": max(
            float(batch["pose6"]["gt_cache_replay_max_abs"])
            for batch in batches
        ),
        "batch_geometry": (
            "painted and GT pairs concatenated within each configured telemetry "
            "microbatch"
        ),
        "axis": AXIS,
        "score_claim": False,
    }


def compact_scorer_native_context(
    product: Mapping[str, Any],
    *,
    primary_contrast: str,
    full_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep a bounded decision context in every tensor row.

    Full per-layer/channel/spatial/frequency values remain in the content-bound
    product artifact instead of being duplicated into every error cell.
    """

    layers = list(product["layers"])
    if not layers or any(
        primary_contrast not in layer["contrasts"] for layer in layers
    ):
        raise TensorBuildError("scorer-native primary contrast is absent")
    peak = max(
        layers,
        key=lambda layer: float(layer["contrasts"][primary_contrast]["rms"]),
    )
    peak_rms = float(peak["contrasts"][primary_contrast]["rms"])
    onset = next(
        (
            layer
            for layer in layers
            if float(layer["contrasts"][primary_contrast]["rms"])
            >= 0.1 * peak_rms
        ),
        layers[0],
    )
    expansion = max(
        layers,
        key=lambda layer: float(
            layer["directional_secant"]["local_expansion_vs_previous_relay"]
            or 0.0
        ),
    )

    def auxiliary_peak(name: str) -> dict[str, Any] | None:
        rows = list(product[name])
        if not rows:
            return None
        selected = max(
            rows,
            key=lambda row: float(row["painted_vs_gt"]["mean_rms"]),
        )
        return {
            "layer": selected["layer"],
            **selected["painted_vs_gt"],
        }

    peak_frequency = peak["contrasts"][primary_contrast][
        "frequency_energy_by_channel"
    ]
    frequency_totals = (
        {
            name: sum(float(value) for value in values)
            for name, values in peak_frequency.items()
        }
        if peak_frequency is not None
        else None
    )
    transport_rows = [
        layer for layer in layers if layer["xi_advected_transport"] is not None
    ]
    transport_peak = (
        max(
            transport_rows,
            key=lambda layer: float(layer["xi_advected_transport"]["rms"]),
        )
        if transport_rows
        else None
    )
    return {
        "schema": product["schema"],
        "scorer": product["scorer"],
        "pair_count": int(product["pair_count"]),
        "primary_contrast": primary_contrast,
        "divergence_onset": {
            "layer": onset["layer"],
            "rms": onset["contrasts"][primary_contrast]["rms"],
            "threshold": 0.1 * peak_rms,
        },
        "divergence_peak": {
            "layer": peak["layer"],
            "rms": peak_rms,
            "uniform_shift_fraction": peak["contrasts"][primary_contrast][
                "uniform_shift_fraction"
            ],
            "geometry_residual_fraction": peak["contrasts"][
                primary_contrast
            ]["geometry_residual_fraction"],
            "frequency_energy": frequency_totals,
        },
        "directional_expansion_peak": {
            "layer": expansion["layer"],
            **expansion["directional_secant"],
        },
        "batchnorm_peak": auxiliary_peak("batchnorm"),
        "se_gate_peak": auxiliary_peak("se_gates"),
        "layer_scale_peak": auxiliary_peak("layer_scales"),
        "xi_advected_transport_peak": (
            {
                "layer": transport_peak["layer"],
                **transport_peak["xi_advected_transport"],
            }
            if transport_peak is not None
            else None
        ),
        "relay_ranking_top5": list(product["relay_ranking"][:5]),
        "product_axes": product["product_axes"],
        "limitations": product["limitations"],
        "full_artifact": dict(full_artifact),
        "causal_status": (
            "MEASURED_SCORER_NATIVE_ASSOCIATION_AND_DIRECTIONAL_SECANT; "
            "NO_INTERMEDIATE_HEAD_PULLBACK_OR_FULL_JACOBIAN_CUSTODY"
        ),
    }


def forward_segnet_paired_amplitude(
    model: Any,
    *,
    camera_pairs: np.ndarray,
    gt_frames: np.ndarray,
    reference_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay painted and GT trajectories together and reduce internal statistics."""

    import torch

    painted = np.asarray(camera_pairs)
    ground_truth = np.asarray(gt_frames)
    labels = np.asarray(reference_labels)
    if (
        painted.dtype != np.uint8
        or painted.ndim != 5
        or painted.shape[1:] != (2, 874, 1164, 3)
    ):
        raise TensorBuildError("painted paired telemetry input differs")
    if ground_truth.dtype != np.uint8 or ground_truth.shape != (
        painted.shape[0],
        874,
        1164,
        3,
    ):
        raise TensorBuildError("GT paired telemetry input differs")
    if labels.shape != (painted.shape[0], HEIGHT, WIDTH):
        raise TensorBuildError("paired telemetry reference labels differ")
    gt_pairs = np.repeat(ground_truth[:, None], 2, axis=1)
    combined = np.concatenate((painted, gt_pairs), axis=0)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(combined))
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .float()
    )
    model_input = model.preprocess_input(tensor)
    painted_logits, gt_logits, amplitude = measure_paired_segnet_amplitude(
        model,
        model_input,
        split_count=painted.shape[0],
        reference_labels=torch.from_numpy(np.ascontiguousarray(labels)),
    )
    cells = painted_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    gt_cells = gt_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    if not np.array_equal(gt_cells, labels.astype(np.uint8)):
        raise TensorBuildError("GT paired telemetry replay differs from target-cache labels")
    return (
        np.ascontiguousarray(cells),
        np.ascontiguousarray(painted_logits.cpu().numpy().astype(np.float32)),
        amplitude,
    )


def exact_head_norms(model: Any) -> dict[str, float]:
    weight = model.segmentation_head[0].weight.detach().cpu().numpy()
    flattened = weight.reshape(len(CLASS_NAMES), -1).astype(np.float64)
    return {
        f"{CLASS_NAMES[winner]}->{CLASS_NAMES[rival]}": float(np.linalg.norm(flattened[winner] - flattened[rival]))
        for winner in range(len(CLASS_NAMES))
        for rival in range(len(CLASS_NAMES))
        if winner != rival
    }


def load_sided_thresholds(path: Path) -> dict[str, tuple[float | None, float | None]]:
    rows: dict[str, tuple[float | None, float | None]] = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema") == "sdwl1.sided_tolerance.row.v1" and row["temporal_stratum"] == "n600_full":
                q10 = row["d2_quantiles"]["q10"]
                q90 = row["d2_quantiles"]["q90"]
                rows[str(row["orientation"])] = (
                    None if q10 is None else float(q10),
                    None if q90 is None else float(q90),
                )
    if set(rows) != {f"{left}->{right}" for left in CLASS_NAMES for right in CLASS_NAMES if left != right}:
        raise TensorBuildError("sided-tolerance table does not contain 20 full-n600 rows")
    return rows


def load_g3_covariates(path: Path) -> dict[int, PairCovariates]:
    rows: dict[int, PairCovariates] = {}
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            pair_id = int(value["pair_index"])
            rank = int(value["score_rank"])
            labels = tuple(sorted(str(row) for row in value["scene_covariates"]["scene_event_labels"]))
            bucket = "TOP24" if rank <= 24 else "TOP64_REMAINDER" if rank <= 64 else "G3_TAIL"
            rows[pair_id] = PairCovariates(rank, bucket, labels)
    if set(rows) != set(range(600)):
        raise TensorBuildError("G3 atlas pair coverage differs from n600")
    return rows


def load_dv1_fields(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sections = decode_joint_ground_vocabulary(path.read_bytes())
    by_kind = {row.kind: row for row in sections}
    if set(by_kind) != {"persistent_level_set", "boundary_worldsheet_spline"}:
        raise TensorBuildError("selected DV1 payload is not spline_plus_events")
    static = decode_persistent_level_set(by_kind["persistent_level_set"].envelope)
    spline, metadata = decode_boundary_worldsheet_spline(by_kind["boundary_worldsheet_spline"].envelope)
    if spline.shape != (600, HEIGHT, WIDTH):
        raise TensorBuildError("DV1 spline pair coverage differs")
    return static, spline, asdict(metadata)


def semantic_fields(
    *,
    carrier: Any,
    predicted: np.ndarray,
    pair_ids: Sequence[int],
    static_field: np.ndarray,
    spline_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current and tested-enriched semantic cell requests."""

    raw = np.full(predicted.shape, -1, dtype=np.int8)
    layer_by_role = {row.role: row for row in carrier.layers}
    for role in _dcc.REALIZATION_PAINT_ORDER:
        layer = layer_by_role[role]
        for local_index, pair_id in enumerate(pair_ids):
            mask = carrier._mask_for_layer(
                layer,
                int(pair_id),
                replace_g1_movable=True,
            )
            raw[local_index, mask] = int(_dcc.ROLE_CLASS_IDS[role])
    current = predicted.copy()
    described = raw >= 0
    current[described] = raw[described].astype(np.uint8)
    rules = carrier.realization_static_rule_codes
    if rules is not None:
        source = rules // 5
        target = rules % 5
        for local_index in range(len(pair_ids)):
            admitted = (rules >= 0) & (raw[local_index] == source)
            current[local_index, admitted] = target[admitted].astype(np.uint8)

    ground_domain = ((static_field == 0) | (static_field == 2)) & (np.arange(HEIGHT)[:, None] >= 96)
    enriched = predicted.copy()
    for local_index, pair_id in enumerate(pair_ids):
        road = spline_field[int(pair_id)]
        enriched[local_index, ground_domain & road] = 0
        enriched[local_index, ground_domain & ~road] = 2
        role_overlay = np.isin(raw[local_index], np.asarray((1, 3, 4), dtype=np.int8))
        enriched[local_index, role_overlay] = raw[local_index, role_overlay].astype(np.uint8)
        if rules is not None:
            source = rules // 5
            target = rules % 5
            admitted = (rules >= 0) & (enriched[local_index] == source)
            enriched[local_index, admitted] = target[admitted].astype(np.uint8)
    return current, enriched


def strict_batch_receipt(
    directory: Path,
    start: int,
    stop: int,
) -> dict[str, Any]:
    path = directory / f"batch_{start:04d}_{stop:04d}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TensorBuildError(f"strict batch receipt is unreadable: {path}") from exc


def verify_batch(
    *,
    strict: Mapping[str, Any],
    camera: np.ndarray,
    cells: np.ndarray,
    labels: np.ndarray,
    pair_ids: Sequence[int],
    archive_sha256: str,
) -> dict[str, Any]:
    errors = cells != labels
    class_rows = {
        class_name: {
            "errors": int(np.count_nonzero(errors & (labels == class_id))),
            "sites": int(np.count_nonzero(labels == class_id)),
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    observed = {
        "archive_sha256": archive_sha256,
        "source_pair_ids": [int(value) for value in pair_ids],
        "camera_sha256": sha256_array(camera),
        "cells_sha256": sha256_array(cells),
        "errors": int(np.count_nonzero(errors)),
        "sites": int(errors.size),
        "class_rows": class_rows,
    }
    for key, value in observed.items():
        if strict[key] != value:
            raise TensorBuildError(f"lazy v19c replay differs for {key}: {pair_ids}")
    return observed


def first_rung_and_move(source_name: str) -> tuple[str, str, str]:
    if source_name == ErrorSource.NEVER_DESCRIBED.name:
        return (
            "VOCABULARY",
            "compile spline_plus_events or persistent ground partition into the receiver",
            "shared boundary worldsheet before any point correction",
        )
    if source_name == ErrorSource.DESCRIBED_BUT_REALIZATION_LOST.name:
        return (
            "CHART_OR_PARAMETER",
            "re-solve class-sided paint, pre-uint8 dither, and placement through R",
            "asymmetric SDWL1 placement with e1/e2 exporter-realizable parameters",
        )
    return (
        "POINT_CORRECTION_LAST",
        "apply #366 only to measured leftover after vocabulary and chart moves",
        "receiver-closed sparse or lattice re-solve; no global irreducibility claim",
    )


def inverse_by_orientation(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["orientation"]): row for row in receipt["rows"]}


def merge_survival_wall_149(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge exact per-pair #149 counts without averaging fractions."""

    if not rows:
        raise TensorBuildError("#149 survival-wall merge requires rows")

    def merge_key(key: str) -> dict[str, int | float | None]:
        sites = sum(int(row[key]["sites"]) for row in rows)
        errors = sum(int(row[key]["errors"]) for row in rows)
        return {
            "sites": sites,
            "errors": errors,
            "error_fraction": errors / sites if sites else None,
        }

    by_target: dict[str, dict[str, int | float | None]] = {}
    for class_name in CLASS_NAMES:
        sites = sum(
            int(row["by_target_class"][class_name]["sites"])
            for row in rows
        )
        errors = sum(
            int(row["by_target_class"][class_name]["errors"])
            for row in rows
        )
        by_target[class_name] = {
            "sites": sites,
            "errors": errors,
            "error_fraction": errors / sites if sites else None,
        }
    return {
        "all_classes": merge_key("all_classes"),
        "by_target_class": by_target,
    }


def row_for_group(
    *,
    pair_id: int,
    covariates: PairCovariates,
    key: int,
    group_mask: np.ndarray,
    d2: np.ndarray,
    boundary_distance: np.ndarray,
    recurrence: np.ndarray,
    inverse_rows: Mapping[str, Mapping[str, Any]],
    dv1_joint_bytes: int,
    g2_target_partition: Mapping[str, Any],
    survival_wall_pair: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = decode_group_key(key)
    source_name = SOURCE_NAMES[decoded["source"]]
    target_name = CLASS_NAMES[decoded["target"]]
    predicted_name = CLASS_NAMES[decoded["predicted"]]
    orientation = f"{predicted_name}->{target_name}"
    geometry = summarize_components(group_mask)
    rung, solved_move, delineation = first_rung_and_move(source_name)
    inverse = inverse_rows.get(orientation)
    if source_name == ErrorSource.NEVER_DESCRIBED.name:
        byte_price = {
            "bytes": dv1_joint_bytes,
            "scope": "shared full-n600 real-coded DV1 joint section; not allocated per cluster",
            "epistemic_status": "MEASURED_REAL_CODER_SHARED_PRICE",
        }
    else:
        byte_price = {
            "bytes": None,
            "scope": "receiver-closed per-cluster bytes remain unmeasured",
            "epistemic_status": "OPEN_PRICE",
        }
    error_count = int(np.count_nonzero(group_mask))
    return {
        "schema": "ddm_sn1_error_source_tensor.row.v1",
        "pair_id": pair_id,
        "g3": {
            "score_rank": covariates.score_rank,
            "tail_bucket": covariates.g3_tail_bucket,
            "scene_event_labels": list(covariates.scene_event_labels),
            "authority": ("historical G3 score/covariate join; not current-v19c causal measurement"),
        },
        "source": source_name,
        "stratum": target_name,
        "ordered_pair": orientation,
        "decision_layer": "segmentation_head",
        "upstream_telemetry_peak_reference": "encoder.model.blocks.2.2.se.pre",
        "d2_band": MARGIN_BANDS[decoded["margin_band"]],
        "curvature_band": CURVATURE_BANDS[decoded["curvature_band"]],
        "temporal_pattern": TEMPORAL_PATTERNS[decoded["temporal_pattern"]],
        "boundary_distance_band": BOUNDARY_DISTANCE_BANDS[
            decoded["boundary_distance_band"]
        ],
        "curve_availability": (
            "CONTINUOUS_LANE_CURVE_AVAILABLE"
            if target_name == "Lane" or predicted_name == "Lane"
            else "NO_CONTINUOUS_LANE_CURVE"
        ),
        "paint_floor_mechanism": PAINT_FLOOR_MECHANISMS[
            decoded["paint_floor_mechanism"]
        ],
        "error_count": error_count,
        "global_d_seg": error_count / N600_SITES,
        "d2": {
            "min": float(d2[group_mask].min()),
            "median": float(np.median(d2[group_mask])),
            "max": float(d2[group_mask].max()),
            "authority": "current v19c logits divided by exact frozen-head normal",
        },
        "boundary_distance": {
            "min": float(boundary_distance[group_mask].min()),
            "median": float(np.median(boundary_distance[group_mask])),
            "max": float(boundary_distance[group_mask].max()),
            "units": "SegNet output cells",
            "authority": "current target-label two-sided 4-neighbour boundary",
        },
        "survival_wall_149_pair": survival_wall_pair,
        "historical_g4_recurrence": {
            "min": int(recurrence[group_mask].min()),
            "median": float(np.median(recurrence[group_mask])),
            "max": int(recurrence[group_mask].max()),
            "authority": ("historical v12 same-transition recurrence proxy; not current-v19c truth"),
        },
        "cluster_geometry": asdict(geometry),
        "menu": {
            "preference": rung,
            "solved_move": solved_move,
            "scale_delineation": delineation,
            "byte_price": byte_price,
            "g2_chart_marginal": {
                "target_energy_fraction": float(g2_target_partition[target_name]["energy_fraction"]),
                "selected_coder_bytes": int(g2_target_partition[target_name]["selected_coder_bytes"]),
                "joint_cell_status": "NO_JOINT_PER_CELL_CUSTODY",
                "use": (
                    "class-level chart-priority marginal only; never interpreted "
                    "as proof that this cluster is chart-expressible"
                ),
            },
            "realized_inverse_demo": (
                None
                if inverse is None
                else {
                    "pair_id": int(inverse["pair_id"]),
                    "segment_pixel_count": int(inverse["segment_pixel_count"]),
                    "desired_rival_realized_count": int(inverse["desired_rival_realized_count"]),
                    "all_argmax_changes": int(inverse["all_argmax_changes"]),
                    "off_segment_argmax_changes": int(inverse["off_segment_argmax_changes"]),
                    "max_linf_uint8": int(inverse["max_linf_uint8"]),
                    "payload_price_status": ("sparse NPZ is feasibility evidence, not compressed receiver bytes"),
                }
            ),
        },
        "verdict_scope": (
            "One exhaustive current-v19c residual-error join row. The structural-hard "
            "label means not requested by current semantics or tested spline_plus_events; "
            "it does not close richer description, chart, or inverse families. The "
            "paint-floor mechanism is observable-axis adjudication, not hidden-state "
            "causal proof."
        ),
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def write_batch_rows(
    *,
    output_directory: Path,
    start: int,
    stop: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = output_directory / "stage_checkpoints" / f"batch_{start:04d}_{stop:04d}.jsonl.gz"
    return atomic_gzip_jsonl(path, rows)


def load_batch_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def build_menu(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregate: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["source"]),
            str(row["stratum"]),
            str(row["ordered_pair"]),
            str(row["d2_band"]),
            str(row["curvature_band"]),
            str(row["temporal_pattern"]),
            str(row["boundary_distance_band"]),
            str(row["curve_availability"]),
            str(row["paint_floor_mechanism"]),
            str(row["g3"]["tail_bucket"]),
        )
        target = aggregate.setdefault(
            key,
            {
                "schema": "ddm_sn1_error_source_menu.row.v1",
                "source": key[0],
                "stratum": key[1],
                "ordered_pair": key[2],
                "d2_band": key[3],
                "curvature_band": key[4],
                "temporal_pattern": key[5],
                "boundary_distance_band": key[6],
                "curve_availability": key[7],
                "paint_floor_mechanism": key[8],
                "g3_tail_bucket": key[9],
                "error_count": 0,
                "pair_ids": set(),
                "menu": row["menu"],
                "evidence_axis": AXIS,
                "score_claim": False,
            },
        )
        target["error_count"] += int(row["error_count"])
        target["pair_ids"].add(int(row["pair_id"]))
    preference_order = {
        "VOCABULARY": 0,
        "CHART_OR_PARAMETER": 1,
        "POINT_CORRECTION_LAST": 2,
    }
    result = []
    for value in aggregate.values():
        pair_ids = sorted(value.pop("pair_ids"))
        value["pair_count"] = len(pair_ids)
        value["pair_ids"] = pair_ids
        value["global_d_seg"] = int(value["error_count"]) / N600_SITES
        result.append(value)
    result.sort(
        key=lambda row: (
            preference_order[row["menu"]["preference"]],
            -int(row["error_count"]),
            row["stratum"],
            row["ordered_pair"],
        )
    )
    for rank, row in enumerate(result, start=1):
        row["menu_rank"] = rank
    return result


def markdown_budget(budget: Mapping[str, Any]) -> str:
    lines = [
        "# DDM SN1 v19c residual error-source budget",
        "",
        f"Axis: `{AXIS}`. `score_claim=false`; pointer unmoved.",
        "",
        "| source | stratum | errors | global d_seg | conditional error rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in budget["rows"]:
        conditional = row["conditional_error_rate"]
        lines.append(
            f"| {row['source']} | {row['stratum']} | {row['errors']:,} | "
            f"{row['global_d_seg']:.12f} | "
            f"{'n/a' if conditional is None else f'{conditional:.12f}'} |"
        )
    lines.extend(
        [
            "",
            f"Total: **{budget['total_errors']:,} errors**, "
            f"global d_seg contribution **{budget['global_d_seg']:.12f}**.",
            "",
            "The three sources are exclusive and exhaustive at the SHA-pinned v19c "
            "endpoint. `STRUCTURALLY_HARD_IRREDUCIBLE` is scoped only to the current "
            "semantic program plus the tested DV1 `spline_plus_events` extension.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--finalize-only-from-bundle",
        help=(
            "Recompute aggregate artifacts from a complete checkpoint tree "
            "produced by this exact implementation bundle."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    config_hash = config.typed_config_sha256()
    sources = validate_sources(config)
    storage = storage_preflight(config)
    implementation = implementation_identity()
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output / "stage_checkpoints"
    identity_path = checkpoint_directory / "00_identity.json"
    finalize_only_bundle = args.finalize_only_from_bundle
    if finalize_only_bundle is not None and (
        len(finalize_only_bundle) != 64
        or any(char not in "0123456789abcdef" for char in finalize_only_bundle)
    ):
        raise TensorBuildError("finalize-only bundle SHA-256 is malformed")
    identity = {
        "schema": "ddm_sn1_error_source_tensor_identity.v1",
        "typed_config_sha256": config_hash,
        "implementation": implementation,
        "sources": sources,
        "storage_preflight": storage,
        "axis": AXIS,
        "score_claim": False,
    }
    if finalize_only_bundle is not None:
        if not identity_path.is_file():
            raise TensorBuildError("finalize-only checkpoint identity is absent")
        preserved_identity = json.loads(identity_path.read_text())
        measurement_implementation = preserved_identity["implementation"]
        if (
            measurement_implementation["bundle_sha256"]
            != finalize_only_bundle
            or stable_measurement_identity(preserved_identity)
            != stable_measurement_identity(identity)
        ):
            raise TensorBuildError("finalize-only measurement identity drifted")
    else:
        measurement_implementation = implementation
    prepare_stage_checkpoint_directory(
        config=config,
        config_hash=config_hash,
        implementation=measurement_implementation,
        checkpoint_directory=checkpoint_directory,
    )
    if finalize_only_bundle is not None:
        pass
    elif identity_path.exists():
        preserved_identity = json.loads(identity_path.read_text())
        if stable_resume_identity(preserved_identity) != stable_resume_identity(identity):
            raise TensorBuildError("resume identity drifted")
    else:
        preserved_identity = identity
        atomic_json(identity_path, identity)
    receipt_storage = preserved_identity["storage_preflight"]

    labels_all = stored_npy_memmap(config.target_cache_path, "lstars")
    if labels_all.shape != (600, HEIGHT, WIDTH) or labels_all.dtype != np.int64:
        raise TensorBuildError("target-cache lstars custody differs")
    gt_f1_all = stored_npy_memmap(config.target_cache_path, "gt_f1")
    if gt_f1_all.shape != (600, 874, 1164, 3) or gt_f1_all.dtype != np.uint8:
        raise TensorBuildError("target-cache gt_f1 custody differs")
    gt_f0_all = stored_npy_memmap(config.target_cache_path, "gt_f0")
    if gt_f0_all.shape != gt_f1_all.shape or gt_f0_all.dtype != np.uint8:
        raise TensorBuildError("target-cache gt_f0 custody differs")
    gt_pose6_all = stored_npy_memmap(config.target_cache_path, "gt_poses")
    if gt_pose6_all.shape != (600, 6):
        raise TensorBuildError("target-cache gt_poses custody differs")
    g4 = np.load(config.g4_arrays_path, allow_pickle=False, mmap_mode="r")
    transition_counts = np.asarray(g4["transition_counts"])
    if transition_counts.shape != (25, HEIGHT, WIDTH):
        raise TensorBuildError("G4 transition-count geometry differs")
    g3 = load_g3_covariates(config.g3_atlas_path)
    g2_receipt = json.loads(config.g2_receipt_path.read_text())
    g2_aggregate = json.loads(config.g2_aggregate_path.read_text())
    g2_target_partition = g2_aggregate["active_target_class_partition"]
    if set(g2_target_partition) != {
        "partition_energy_total",
        "partition_selected_coder_bytes_total",
        *CLASS_NAMES,
    }:
        raise TensorBuildError("G2 target-class partition custody differs")
    static_field, spline_field, spline_metadata = load_dv1_fields(config.dv1_selected_payload_path)
    dv1_receipt = json.loads(config.dv1_receipt_path.read_text())
    dv1_summary = json.loads(config.dv1_summary_path.read_text())
    selected = dv1_summary["selected_joint_composition"]
    if (
        selected["candidate_id"] != "spline_plus_events"
        or int(selected["new_joint_section_bytes"]) != 1610
        or selected["new_joint_section_sha256"] != config.source_sha256["dv1_selected_payload_path"]
        or not any(row["sha256"] == config.source_sha256["dv1_summary_path"] for row in dv1_receipt["outputs"])
    ):
        raise TensorBuildError("DV1 selected joint receipt differs")
    inverse_receipt = json.loads(config.inverse_receipt_path.read_text())
    inverse_rows = inverse_by_orientation(inverse_receipt)
    survival_prior = json.loads(config.survival_wall_149_path.read_text())
    try:
        survival_prior_value = float(
            survival_prior["rows"]["mp128"]["avg_boundary_band_flip"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TensorBuildError("#149 survival-wall prior is malformed") from exc
    if (
        survival_prior.get("schema") != "curve_core_dseg_feasibility_gate.v1"
        or survival_prior.get("producer")
        != "experiments/probe_curve_core_dseg_feasibility_gate.py"
        or int(survival_prior["method"]["n_frames"]) != 3
    ):
        raise TensorBuildError("#149 survival-wall prior custody differs")
    advected_receipt = json.loads(config.advected_screw6_receipt_path.read_text())
    if (
        advected_receipt.get("schema")
        != "advected_screw6_chartlevel_durable_receipt.v1"
        or advected_receipt.get("verdict")
        != "N16_TWO_AXIS_GATE_FAIL_STOP_BEFORE_N64"
    ):
        raise TensorBuildError("full-screw feature-transport prior custody differs")
    sided_thresholds = load_sided_thresholds(config.sided_tolerance_path)
    v19c_receipt = json.loads(config.v19c_receipt_path.read_text())
    strict_summary = json.loads(
        (config.v19c_receipt_path.parent / "stage_checkpoints/02_n600_saturation_curve.json").read_text()
    )["strict_final_full_n600_replay"]
    if (
        strict_summary["digest_chain_sha256"] != config.strict_replay_digest_chain_sha256
        or strict_summary["archive_sha256"] != config.source_sha256["v19c_final_archive_path"]
    ):
        raise TensorBuildError("v19c strict replay summary differs")

    archive = config.v19c_final_archive_path.read_bytes()
    receiver = receive_v19c_lazy(archive)
    model = load_segnet(config)
    pose_model = load_posenet(config)
    from tac.optimization.predict_project_receiver import (
        counted_full_screw_xi_series,
    )
    from tac.optimization.predictor_upgrade_xi_chart import (
        load_g1_worldsheet_motion,
    )

    _, _, pitch_rad, pitch_custody = load_g1_worldsheet_motion(
        REPO_ROOT
    )
    full_xi, full_xi_custody = counted_full_screw_xi_series(
        np.asarray(gt_pose6_all, dtype=np.float64),
        translation_scale=0.16,
        rotation_scale=1.0,
        pitch_rad=pitch_rad,
        source_sha256=config.source_sha256["target_cache_path"],
    )
    head_norms = exact_head_norms(model)
    carrier = receiver.base.base
    batch_manifests: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    seg_product_batches: list[dict[str, Any]] = []
    pose_product_batches: list[dict[str, Any]] = []

    segnet_analytic = analytic_scorer_knowledge(
        model,
        scorer="segnet",
        weights_sha256=config.source_sha256["segnet_weights_path"],
    )
    posenet_analytic = analytic_scorer_knowledge(
        pose_model,
        scorer="posenet",
        weights_sha256=config.source_sha256["posenet_weights_path"],
    )
    pose_inventory = posenet_analytic["module_inventory"]
    if (
        int(pose_inventory.get("LayerScale2d", 0)) != 24
        or int(pose_inventory.get("BatchNorm1d", 0)) != 8
        or int(pose_inventory.get("SEModule", 0)) != 1
        or int(pose_inventory.get("GELUTanh", 0)) != 19
    ):
        raise TensorBuildError("frozen PoseNet amplitude-module inventory differs")
    analytic_files = {
        "segnet": atomic_gzip_jsonl(
            output / "segnet_analytic_knowledge.jsonl.gz",
            [segnet_analytic],
        ),
        "posenet": atomic_gzip_jsonl(
            output / "posenet_analytic_knowledge.jsonl.gz",
            [posenet_analytic],
        ),
    }

    for start in range(0, 600, config.batch_size):
        stop = min(start + config.batch_size, 600)
        batch_receipt_path = checkpoint_directory / f"batch_{start:04d}_{stop:04d}.json"
        if batch_receipt_path.exists():
            prior = json.loads(batch_receipt_path.read_text())
            if (
                prior["typed_config_sha256"] != config_hash
                or prior["implementation_bundle_sha256"]
                != measurement_implementation["bundle_sha256"]
                or prior["source_pair_ids"] != list(range(start, stop))
            ):
                raise TensorBuildError("batch resume identity drifted")
            row_path = Path(prior["row_file"]["path"])
            if not row_path.is_file() or sha256_file(row_path) != prior["row_file"]["sha256"]:
                raise TensorBuildError("batch row file drifted")
            seg_product_path = Path(prior["segnet_product_file"]["path"])
            pose_product_path = Path(prior["posenet_product_file"]["path"])
            for product_name, product_path in (
                ("SegNet", seg_product_path),
                ("PoseNet", pose_product_path),
            ):
                if (
                    not product_path.is_file()
                    or sha256_file(product_path)
                    != prior[
                        f"{product_name.lower()}_product_file"
                    ]["sha256"]
                ):
                    raise TensorBuildError(
                        f"batch {product_name} product file drifted"
                    )
            rows = load_batch_rows(row_path)
            seg_product_rows = load_batch_rows(seg_product_path)
            pose_product_rows = load_batch_rows(pose_product_path)
            expected_product_rows = math.ceil(
                (stop - start) / config.telemetry_microbatch_size
            )
            if (
                len(seg_product_rows) != expected_product_rows
                or len(pose_product_rows) != expected_product_rows
            ):
                raise TensorBuildError("batch scorer-native row count differs")
            if sum(int(row["error_count"]) for row in rows) != int(prior["residual_errors"]):
                raise TensorBuildError("resumed batch row accounting differs")
            all_rows.extend(rows)
            seg_product_batches.extend(seg_product_rows)
            pose_product_batches.extend(pose_product_rows)
            batch_manifests.append(prior)
            continue

        if finalize_only_bundle is not None:
            raise TensorBuildError(
                f"finalize-only checkpoint is incomplete at pairs {start}:{stop}"
            )
        pair_ids = tuple(range(start, stop))
        camera = receiver.render_camera_pairs(pair_ids)
        labels = np.asarray(labels_all[start:stop], dtype=np.uint8)
        gt_f0 = np.asarray(gt_f0_all[start:stop], dtype=np.uint8)
        gt_f1 = np.asarray(gt_f1_all[start:stop], dtype=np.uint8)
        gt_pose6 = np.asarray(gt_pose6_all[start:stop], dtype=np.float64)
        cells, logits = forward_segnet(model, camera)
        seg_product_rows, pose_product_rows = scorer_native_products_for_batch(
            segnet=model,
            posenet=pose_model,
            camera_pairs=camera,
            gt_f0=gt_f0,
            gt_f1=gt_f1,
            gt_pose6=gt_pose6,
            full_xi=full_xi[start:stop],
            pitch_rad=pitch_rad,
            microbatch_size=config.telemetry_microbatch_size,
        )
        seg_product_file = atomic_gzip_jsonl(
            checkpoint_directory
            / f"segnet_product_{start:04d}_{stop:04d}.jsonl.gz",
            seg_product_rows,
        )
        pose_product_file = atomic_gzip_jsonl(
            checkpoint_directory
            / f"posenet_product_{start:04d}_{stop:04d}.jsonl.gz",
            pose_product_rows,
        )
        strict = strict_batch_receipt(
            config.v19c_strict_batch_directory,
            start,
            stop,
        )
        replay = verify_batch(
            strict=strict,
            camera=camera,
            cells=cells,
            labels=labels,
            pair_ids=pair_ids,
            archive_sha256=config.source_sha256["v19c_final_archive_path"],
        )
        current_semantic, enriched_semantic = semantic_fields(
            carrier=carrier,
            predicted=cells,
            pair_ids=pair_ids,
            static_field=static_field,
            spline_field=spline_field,
        )
        batch_rows: list[dict[str, Any]] = []
        batch_survival_rows: list[dict[str, Any]] = []
        batch_source_counts: Counter[str] = Counter()
        for local_index, pair_id in enumerate(pair_ids):
            source, residual = classify_error_sources(
                target=labels[local_index],
                predicted=cells[local_index],
                current_semantic=current_semantic[local_index],
                enriched_semantic=enriched_semantic[local_index],
            )
            d2, margin = d2_margin_bands(
                predicted=cells[local_index],
                target=labels[local_index],
                logits=logits[local_index],
                head_norms=head_norms,
                sided_thresholds=sided_thresholds,
            )
            curvature = curvature_bands(labels[local_index])
            boundary_distance, boundary_band = boundary_distance_bands(
                labels[local_index]
            )
            mechanism, _curve_available = paint_floor_mechanism_codes(
                target=labels[local_index],
                predicted=cells[local_index],
                margin_band=margin,
                boundary_distance_band=boundary_band,
            )
            pair_survival = survival_wall_149(
                target=labels[local_index],
                predicted=cells[local_index],
            )
            batch_survival_rows.append(pair_survival)
            rows_grid, columns_grid = np.indices((HEIGHT, WIDTH))
            recurrence = transition_counts[
                cells[local_index] * 5 + labels[local_index],
                rows_grid,
                columns_grid,
            ]
            covariates = g3[pair_id]
            temporal = temporal_pattern_codes(
                recurrence=recurrence,
                event_adjacent=bool(covariates.scene_event_labels),
            )
            group_keys = encode_group_key(
                source=source,
                target=labels[local_index],
                predicted=cells[local_index],
                margin_band=margin,
                curvature_band=curvature,
                temporal_pattern=temporal,
                boundary_distance_band=boundary_band,
                paint_floor_mechanism=mechanism,
            )
            unique = np.unique(group_keys[residual])
            for key in unique:
                group_mask = residual & (group_keys == key)
                row = row_for_group(
                    pair_id=pair_id,
                    covariates=covariates,
                    key=int(key),
                    group_mask=group_mask,
                    d2=d2,
                    boundary_distance=boundary_distance,
                    recurrence=recurrence,
                    inverse_rows=inverse_rows,
                    dv1_joint_bytes=int(selected["new_joint_section_bytes"]),
                    g2_target_partition=g2_target_partition,
                    survival_wall_pair=pair_survival,
                )
                batch_rows.append(row)
                batch_source_counts[str(row["source"])] += int(row["error_count"])
            if int(np.count_nonzero(residual)) != sum(
                int(row["error_count"]) for row in batch_rows if int(row["pair_id"]) == pair_id
            ):
                raise TensorBuildError(f"pair {pair_id} tensor accounting does not close")
        row_file = write_batch_rows(
            output_directory=output,
            start=start,
            stop=stop,
            rows=batch_rows,
        )
        batch_receipt = {
            "schema": "ddm_sn1_error_source_tensor.batch.v1",
            "typed_config_sha256": config_hash,
            "implementation_bundle_sha256": measurement_implementation[
                "bundle_sha256"
            ],
            "source_pair_ids": list(pair_ids),
            "strict_replay": replay,
            "residual_errors": sum(batch_source_counts.values()),
            "source_counts": dict(sorted(batch_source_counts.items())),
            "row_file": row_file,
            "segnet_product_file": seg_product_file,
            "posenet_product_file": pose_product_file,
            "survival_wall_149": merge_survival_wall_149(
                batch_survival_rows
            ),
            "axis": AXIS,
            "score_claim": False,
        }
        atomic_json(batch_receipt_path, batch_receipt)
        batch_manifests.append(batch_receipt)
        seg_product_batches.extend(seg_product_rows)
        pose_product_batches.extend(pose_product_rows)
        all_rows.extend(batch_rows)
        print(
            f"batch {start:04d}:{stop:04d} residual={batch_receipt['residual_errors']} rows={len(batch_rows)}",
            flush=True,
        )
        del (
            camera,
            cells,
            logits,
            labels,
            gt_f0,
            gt_f1,
            gt_pose6,
            current_semantic,
            enriched_semantic,
            seg_product_rows,
            pose_product_rows,
        )

    segnet_product = finalize_scorer_native_product(seg_product_batches)
    posenet_product = finalize_scorer_native_product(pose_product_batches)
    if (
        int(segnet_product["pair_count"]) != 600
        or int(posenet_product["pair_count"]) != 600
    ):
        raise TensorBuildError("scorer-native product does not cover n600")
    segnet_product_artifact = atomic_gzip_jsonl(
        output / "segnet_scorer_native_product_n600.jsonl.gz",
        [segnet_product],
    )
    posenet_product_artifact = atomic_gzip_jsonl(
        output / "posenet_scorer_native_product_n600.jsonl.gz",
        [posenet_product],
    )
    pose6_summary = summarize_pose6_product(pose_product_batches)
    segnet_context = compact_scorer_native_context(
        segnet_product,
        primary_contrast="painted_f1_vs_gt_f1",
        full_artifact=segnet_product_artifact,
    )
    posenet_context = compact_scorer_native_context(
        posenet_product,
        primary_contrast="painted_pair_vs_gt_pair",
        full_artifact=posenet_product_artifact,
    )
    scorer_native_context = {
        "segnet": segnet_context,
        "posenet": posenet_context,
        "posenet_first_six": pose6_summary,
        "analytic_weight_derived": analytic_files,
        "telemetry_microbatch_size": config.telemetry_microbatch_size,
        "full_screw_transport": {
            "prior_receipt": sources["advected_screw6_receipt_path"],
            "prior_verdict": advected_receipt["verdict"],
            "pitch_radians": pitch_rad,
            "pitch_custody": pitch_custody,
            "xi_custody": full_xi_custody,
            "verdict_scope": (
                "stored-PoseNet full-screw ground-plane proxy; the prior n16 "
                "formulation failed its two-axis rate gate, not the family"
            ),
        },
        "pixel_diff_status": (
            "CROSS_REFERENCE_ONLY; scorer-native feature difference is primary"
        ),
        "scope": (
            "global n600 painted-vs-GT frozen SegNet and PoseNet grouped "
            "microbatch forward; advisory and not contest-score authority"
        ),
    }
    scorer_native_row_reference = {
        "scope": "GLOBAL_N600_SCORER_NATIVE_PRODUCT",
        "segnet_artifact_sha256": segnet_product_artifact["sha256"],
        "posenet_artifact_sha256": posenet_product_artifact["sha256"],
        "primary_currency": "SCORER_NATIVE_FEATURE_DIFFERENCE",
        "pixel_diff_status": "CROSS_REFERENCE_ONLY",
    }
    for row in all_rows:
        row["scorer_native_divergence"] = scorer_native_row_reference

    survival_current = merge_survival_wall_149(
        [manifest["survival_wall_149"] for manifest in batch_manifests]
    )
    current_wall_fraction = float(
        survival_current["all_classes"]["error_fraction"]
    )
    survival_measurement = {
        "schema": "ddm_sn1_survival_wall_149.v1",
        "current_v19c_n600": survival_current,
        "historical_mp128_prior": {
            "boundary_band_flip": survival_prior_value,
            "source_sha256": config.source_sha256["survival_wall_149_path"],
            "axis": survival_prior["axis_tag"],
            "n_frames": int(survival_prior["method"]["n_frames"]),
            "scope": (
                "three-frame mp128 curve/sine-family receiver; contextual prior, "
                "not pooled with current v19c"
            ),
        },
        "current_minus_prior_fraction": current_wall_fraction
        - survival_prior_value,
        "current_over_prior_ratio": current_wall_fraction
        / survival_prior_value,
        "boundary_definition": (
            "two-sided 4-neighbour target-label boundary followed by one "
            "binary-dilation iteration"
        ),
        "comparability": (
            "same boundary-band/error-fraction formula; different receiver and "
            "600-frame versus three-frame scope"
        ),
        "axis": AXIS,
        "score_claim": False,
    }
    survival_path = output / "survival_wall_149_n600.json"
    atomic_json(survival_path, survival_measurement)

    mechanism_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in all_rows:
        mechanism_counts[
            (
                str(row["source"]),
                str(row["stratum"]),
                str(row["paint_floor_mechanism"]),
                str(row["boundary_distance_band"]),
                str(row["curve_availability"]),
            )
        ] += int(row["error_count"])
    mechanism_rows = [
        {
            "source": key[0],
            "stratum": key[1],
            "paint_floor_mechanism": key[2],
            "boundary_distance_band": key[3],
            "curve_availability": key[4],
            "errors": errors,
            "global_d_seg": errors / N600_SITES,
        }
        for key, errors in sorted(mechanism_counts.items())
    ]
    mechanism_budget = {
        "schema": "ddm_sn1_paint_floor_mechanism_budget.v1",
        "rows": mechanism_rows,
        "described_but_realization_lost": {
            mechanism: sum(
                int(row["errors"])
                for row in mechanism_rows
                if row["source"]
                == ErrorSource.DESCRIBED_BUT_REALIZATION_LOST.name
                and row["paint_floor_mechanism"] == mechanism
            )
            for mechanism in PAINT_FLOOR_MECHANISMS
        },
        "total_errors": sum(int(row["errors"]) for row in mechanism_rows),
        "adjudication_status": (
            "OBSERVABLE_AXIS_ASSOCIATION_NOT_HIDDEN_STATE_CAUSAL_PROOF"
        ),
        "axis": AXIS,
        "score_claim": False,
    }
    mechanism_path = output / "paint_floor_mechanism_budget.json"
    atomic_json(mechanism_path, mechanism_budget)

    header = {
        "schema": "ddm_sn1_error_source_tensor.header.v1",
        "run_id": config.run_id,
        "typed_config_sha256": config_hash,
        "implementation": implementation,
        "measurement_implementation": measurement_implementation,
        "source_custody": sources,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_vocabulary": list(SOURCE_NAMES),
        "paint_floor_mechanism_vocabulary": list(PAINT_FLOOR_MECHANISMS),
        "boundary_distance_vocabulary": list(BOUNDARY_DISTANCE_BANDS),
        "decision_layer": "segmentation_head",
        "scorer_native_divergence": scorer_native_context,
        "survival_wall_149": survival_measurement,
        "mechanism_adjudication": (
            "boundary distance plus Lane-curve availability plus sided D2 margin; "
            "association only"
        ),
        "historical_proxy_policy": {
            "g2": "class-level energy/byte marginal only; no joint per-cell custody",
            "g3": "pair rank/tail/event covariate only",
            "g4": "v12 same-transition recurrence only",
            "v14": "realization-leak cross-check receipt only",
        },
    }
    tensor_path = output / "error_source_tensor_n600.jsonl.gz"
    tensor_artifact = atomic_gzip_jsonl(tensor_path, [header, *all_rows])

    counts: dict[str, dict[str, int]] = {
        source_name: dict.fromkeys(EXPECTED_SOURCE_ERRORS, 0) for source_name in SOURCE_NAMES
    }
    target_sites = dict.fromkeys(CLASS_NAMES, 0)
    for manifest in batch_manifests:
        for class_name, row in manifest["strict_replay"]["class_rows"].items():
            target_sites[class_name] += int(row["sites"])
    for row in all_rows:
        counts[str(row["source"])][str(row["stratum"])] += int(row["error_count"])
    budget = source_budget(counts=counts, target_sites=target_sites)
    observed_class = {
        class_name: sum(counts[source][class_name] for source in SOURCE_NAMES) for class_name in EXPECTED_SOURCE_ERRORS
    }
    if (
        int(budget["total_errors"]) != config.expected_residual_errors
        or observed_class != config.expected_residual_errors_by_class
    ):
        raise TensorBuildError(
            f"v19c residual budget differs: total={budget['total_errors']}, classes={observed_class}"
        )
    budget.update(
        {
            "axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "expected_endpoint": config.expected_residual_errors,
        }
    )
    budget_path = output / "error_source_budget.json"
    atomic_json(budget_path, budget)
    atomic_bytes(
        output / "error_source_budget.md",
        markdown_budget(budget).encode(),
    )

    menu = build_menu(all_rows)
    menu_path = output / "error_source_solve_menu.jsonl"
    atomic_bytes(
        menu_path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in menu),
    )
    source_totals = {source: sum(counts[source].values()) for source in SOURCE_NAMES}
    vocabulary_rank = [
        {
            "rank": 1,
            "move_family": "VOCABULARY",
            "candidate": "spline_plus_events plus persistent ground partition",
            "measured_error_mass": source_totals[ErrorSource.NEVER_DESCRIBED.name],
            "measured_shared_bytes": int(selected["new_joint_section_bytes"]),
            "error_mass_per_shared_byte": (
                source_totals[ErrorSource.NEVER_DESCRIBED.name] / int(selected["new_joint_section_bytes"])
            ),
            "realization_status": "SEMANTIC_REACH_ONLY_RECEIVER_REALIZATION_OWED",
        },
        {
            "rank": 2,
            "move_family": "CHART_OR_PARAMETER",
            "candidate": "SDWL1 asymmetric paint/preuint8 placement",
            "measured_error_mass": source_totals[ErrorSource.DESCRIBED_BUT_REALIZATION_LOST.name],
            "measured_shared_bytes": None,
            "error_mass_per_shared_byte": None,
            "realization_status": "THREE_SEGMENT_FEASIBILITY_WITH_HIGH_COLLATERAL",
        },
        {
            "rank": 3,
            "move_family": "POINT_CORRECTION_LAST",
            "candidate": "#366 receiver-closed sparse/lattice re-solve",
            "measured_error_mass": source_totals[ErrorSource.STRUCTURALLY_HARD_IRREDUCIBLE.name],
            "measured_shared_bytes": None,
            "error_mass_per_shared_byte": None,
            "realization_status": ("ONLY_MEASURED_LEFTOVER_AFTER_CURRENT_AND_TESTED_VOCABULARY"),
        },
    ]
    vocabulary_path = output / "vocabulary_gap_ranking.json"
    atomic_json(
        vocabulary_path,
        {
            "schema": "ddm_sn1_vocabulary_gap_ranking.v1",
            "rows": vocabulary_rank,
            "preference": "vocabulary > chart_or_parameter > point_correction",
            "axis": AXIS,
            "score_claim": False,
        },
    )

    checkpoint_cold_store = externalize_stage_checkpoints(
        config=config,
        config_hash=config_hash,
        implementation=measurement_implementation,
        checkpoint_directory=checkpoint_directory,
        output_directory=output,
    )
    v14 = json.loads(config.v14_receipt_path.read_text())
    e1 = json.loads(config.e1_receipt_path.read_text())
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "typed_config": config.canonical_payload(),
        "typed_config_sha256": config_hash,
        "implementation": implementation,
        "measurement_implementation": measurement_implementation,
        "finalization": {
            "mode": (
                "COMPLETE_CHECKPOINT_TREE_REFINALIZATION"
                if finalize_only_bundle is not None
                else "IN_PROCESS_AFTER_MEASUREMENT"
            ),
            "command": [
                "/Users/adpena/Projects/pact/.venv/bin/python",
                "-u",
                "tools/build_ddm_sn1_error_source_tensor.py",
                "--config",
                ".omx/research/configs/ddm_sn1_error_source_tensor_20260723.json",
                *(
                    [
                        "--finalize-only-from-bundle",
                        finalize_only_bundle,
                    ]
                    if finalize_only_bundle is not None
                    else []
                ),
            ],
        },
        "source_custody": sources,
        "storage_preflight": receipt_storage,
        "stage_checkpoint_cold_store": checkpoint_cold_store,
        "v19c_join": {
            "commit_reference": "6be800b1f3",
            "strict_batch_count": len(batch_manifests),
            "strict_digest_chain_sha256": (config.strict_replay_digest_chain_sha256),
            "endpoint_residual_errors": int(budget["total_errors"]),
            "endpoint_residual_errors_by_class": observed_class,
            "all_lazy_camera_and_cell_batches_match_strict_replay": True,
        },
        "source_budget": budget,
        "source_totals": source_totals,
        "paint_floor_mechanism_budget": mechanism_budget,
        "scorer_native_telemetry": {
            "segnet": segnet_context,
            "posenet": posenet_context,
            "posenet_first_six": pose6_summary,
            "analytic_weight_derived": analytic_files,
            "full_screw_transport": scorer_native_context[
                "full_screw_transport"
            ],
        },
        "survival_wall_149": survival_measurement,
        "dv1": {
            "selected_candidate": selected["candidate_id"],
            "new_joint_section_bytes": int(selected["new_joint_section_bytes"]),
            "new_joint_section_sha256": selected["new_joint_section_sha256"],
            "spline_metadata": spline_metadata,
            "authority": "semantic-cell reach; RGB realization and Pose are open",
        },
        "inverse": {
            "segment_count": int(inverse_receipt["segment_count"]),
            "majority_transition_realized_count": int(inverse_receipt["majority_transition_realized_count"]),
            "high_collateral_warning": True,
        },
        "cross_checks": {
            "g2_receipt_sha256": config.source_sha256["g2_receipt_path"],
            "g2_receipt_schema": g2_receipt["schema"],
            "g2_aggregate_sha256": config.source_sha256["g2_aggregate_path"],
            "g2_authority": ("class-level energy/byte marginal; no per-cell chart-expressibility claim"),
            "v14_receipt_sha256": config.source_sha256["v14_receipt_path"],
            "v14_schema": v14["schema"],
            "v14_authority": "historical realization-fidelity cross-check only",
            "e1_receipt_sha256": config.source_sha256["e1_receipt_path"],
            "e1_schema": e1["schema"],
            "e1_authority": "exporter identity and receiver survival cross-check only",
        },
        "artifacts": {
            "tensor": tensor_artifact,
            "budget_json": path_identity(budget_path),
            "budget_markdown": path_identity(output / "error_source_budget.md"),
            "solve_menu": path_identity(menu_path),
            "vocabulary_gap_ranking": path_identity(vocabulary_path),
            "paint_floor_mechanism_budget": path_identity(mechanism_path),
            "segnet_scorer_native_product": segnet_product_artifact,
            "posenet_scorer_native_product": posenet_product_artifact,
            "segnet_analytic_knowledge": analytic_files["segnet"],
            "posenet_analytic_knowledge": analytic_files["posenet"],
            "survival_wall_149": path_identity(survival_path),
            "stage_checkpoint_cold_store_manifest": checkpoint_cold_store[
                "manifest_artifact"
            ],
        },
        "triality": {
            "dsl": "DDMErrorSourceTensorConfigV1",
            "dag": ".omx/research/ddm_sn1_error_source_tensor_603_DAG_FEED_20260723.md",
            "equations": (".omx/research/ddm_sn1_error_source_tensor_canonical_equations_20260723.md"),
        },
        "verdict": "SOLVE_FIRST_ERROR_SOURCE_BUDGET_MEASURED",
        "verdict_scope": (
            "Exact n600 v19c residual-error partition under current receiver semantics "
            "and one SHA-pinned DV1 spline_plus_events extension. Historical G3/G4/v14 "
            "fields are joins, not current causal measurements. Scorer-native BN/SE, "
            "relay, spectral, temporal, transport, and paint-floor mechanism fields "
            "are associations/directional secants, not per-cell causal proof or a "
            "full Jacobian spectrum. No score, Pose admission, byte admission, "
            "promotion, or global "
            "family-impossibility claim is made."
        ),
        "axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "v19c_receipt_schema": v19c_receipt["schema"],
    }
    receipt_path = output / "ddm_sn1_error_source_tensor_receipt.json"
    atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "total_errors": budget["total_errors"],
                "source_totals": source_totals,
                "menu_rows": len(menu),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        TensorBuildError,
        ErrorSourceTensorError,
        SegNetAmplitudeTelemetryError,
        ScorerNativeDiffError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
