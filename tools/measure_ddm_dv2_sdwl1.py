#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure SDWL1 on a frozen ZIP_STORED NPZ without materializing source arrays.

The output is a deterministic, resumable local-CPU research receipt.  It is not
a witness, candidate archive, evaluator run, score, or promotion artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.ddm_dv2_sdwl1 import (  # noqa: E402
    FactInventory,
    SentenceLayout,
    SentenceOptions,
    TemporalMode,
    decode_independent_descriptions,
    decode_sentence,
    decompress_outer_payload,
    extract_fact_inventory,
    measure_serialization,
)

DEFAULT_SOURCE_CACHE: Final = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_OUTPUT_DIR: Final = Path(".omx/research/ddm_dv2_sdwl1_n600_20260723")
EXPECTED_SOURCE_BYTES: Final = 5_078_017_610
EXPECTED_SOURCE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
DEFAULT_PAIR_COUNT: Final = 600
DEFAULT_MIN_FREE_BYTES: Final = 128 * 1024**2
SOURCE_MEMBERS: Final = ("lstars", "margins", "gt_poses")
COUNTERFACTUAL_FIELDS: Final = (
    "explicit_frame_indices",
    "repeated_provenance",
    "redundant_event_masks",
    "split_topology_vocabulary",
)
STAGE_CUSTODY: Final = "stage_00_source_custody.json"
STAGE_INVENTORY: Final = "stage_10_fact_inventory.json"
INVENTORY_PAYLOAD: Final = "payloads/fact_inventory.npy"
FINAL_RECEIPT: Final = "receipt.json"


class MeasurementError(ValueError):
    """Raised when measurement custody or resumability fails closed."""


@dataclass(frozen=True)
class StoredNpyMember:
    """One directly memory-mapped ZIP_STORED NPY member and its custody."""

    array: np.memmap
    custody: dict[str, Any]


@dataclass(frozen=True)
class RowSpec:
    """One deterministic measurement row."""

    row_id: str
    kind: str
    layout: SentenceLayout
    temporal_mode: TemporalMode
    counterfactual: str | None = None

    def options(self) -> SentenceOptions:
        kwargs = {name: name == self.counterfactual for name in COUNTERFACTUAL_FIELDS}
        return SentenceOptions(
            layout=self.layout,
            temporal_mode=self.temporal_mode,
            **kwargs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "counterfactual": self.counterfactual,
            "kind": self.kind,
            "layout": self.layout.value,
            "row_id": self.row_id,
            "temporal_mode": self.temporal_mode.value,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic, human-readable JSON receipt bytes."""

    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Publish immutable evidence bytes by fsynced sibling replace."""

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


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MeasurementError(f"invalid JSON receipt {path}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"receipt {path} is not a JSON object")
    return value


def _zip_local_name(raw_name: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        return raw_name.decode(encoding)
    except UnicodeDecodeError as exc:
        raise MeasurementError("invalid local ZIP member name encoding") from exc


def stored_npy_memmap(npz_path: Path, key: str) -> StoredNpyMember:
    """Directly map one unique ZIP_STORED NPY member read-only."""

    member_name = key if key.endswith(".npy") else f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            matches = [info for info in archive.infolist() if info.filename == member_name]
            if len(matches) != 1:
                raise MeasurementError(f"{npz_path}:{member_name} must occur exactly once; found {len(matches)}")
            info = matches[0]
            if info.compress_type != zipfile.ZIP_STORED or info.file_size != info.compress_size:
                raise MeasurementError(f"{npz_path}:{member_name} is not ZIP_STORED")
            if info.flag_bits & 0x1:
                raise MeasurementError(f"{npz_path}:{member_name} is encrypted")
            local_header_offset = int(info.header_offset)
            central = {
                "compress_type": int(info.compress_type),
                "crc32": f"{info.CRC:08x}",
                "file_size": int(info.file_size),
                "header_offset": local_header_offset,
                "member": member_name,
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise MeasurementError(f"invalid source NPZ {npz_path}") from exc

    with npz_path.open("rb") as handle:
        handle.seek(local_header_offset)
        raw_header = handle.read(30)
        if len(raw_header) != 30:
            raise MeasurementError(f"truncated local ZIP header for {member_name}")
        fields = struct.unpack("<IHHHHHIIIHH", raw_header)
        (
            signature,
            _version,
            flags,
            compression,
            _mtime,
            _mdate,
            _crc,
            _compressed_size,
            _file_size,
            name_length,
            extra_length,
        ) = fields
        if signature != 0x04034B50 or compression != zipfile.ZIP_STORED:
            raise MeasurementError(f"invalid local ZIP_STORED header for {member_name}")
        raw_name = handle.read(name_length)
        if _zip_local_name(raw_name, flags) != member_name:
            raise MeasurementError(f"central/local ZIP name drift for {member_name}")
        handle.seek(extra_length, os.SEEK_CUR)
        npy_start = handle.tell()
        try:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        except (EOFError, ValueError) as exc:
            raise MeasurementError(f"invalid NPY header for {member_name}") from exc
        data_offset = handle.tell()
    dtype = np.dtype(dtype)
    element_count = int(np.prod(shape, dtype=np.int64))
    data_bytes = element_count * dtype.itemsize
    if data_offset + data_bytes != npy_start + central["file_size"]:
        raise MeasurementError(f"NPY payload size drift for {member_name}")
    array = np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=shape,
        order="F" if fortran else "C",
    )
    custody = {
        **central,
        "data_bytes": data_bytes,
        "data_offset": data_offset,
        "dtype": dtype.str,
        "fortran_order": bool(fortran),
        "npy_version": list(version),
        "shape": [int(value) for value in shape],
        "source_access": "direct_zip_stored_npy_read_only_memmap",
    }
    return StoredNpyMember(array=array, custody=custody)


def durable_receipt_path(path: Path) -> str:
    """Use repository-relative evidence paths when the artifact is in-tree."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def storage_preflight(output_dir: Path, *, required_free_bytes: int) -> dict[str, Any]:
    """Check the explicitly requested output tier before any artifact write."""

    if required_free_bytes <= 0:
        raise MeasurementError("required free bytes must be positive")
    ancestor = output_dir.resolve()
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    usage = shutil.disk_usage(ancestor)
    passed = usage.free >= required_free_bytes
    result = {
        "available_free_bytes": usage.free,
        "bulk_source_copied": False,
        "output_dir": durable_receipt_path(output_dir),
        "preflight_path": durable_receipt_path(ancestor),
        "required_free_bytes": required_free_bytes,
        "status": "PASS" if passed else "BLOCK",
    }
    if not passed:
        raise MeasurementError(f"storage preflight refused: {usage.free} free < {required_free_bytes} required")
    return result


def implementation_custody() -> dict[str, Any]:
    paths = (
        REPO / "src/tac/optimization/ddm_dv2_sdwl1.py",
        Path(__file__).resolve(),
        REPO / "src/tac/optimization/arith_selfcomp_rate_coders.py",
    )
    files = {
        str(path.relative_to(REPO)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    }
    return {
        "files": files,
        "runtime": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "zlib_compile": zlib.ZLIB_VERSION,
            "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        },
    }


def source_custody(
    source_cache: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    pair_count: int,
    storage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.memmap]]:
    """Verify full source bytes/hash and map only the three admitted members."""

    resolved = source_cache.resolve(strict=True)
    observed_bytes = resolved.stat().st_size
    if observed_bytes != expected_bytes:
        raise MeasurementError(f"source cache bytes {observed_bytes} != expected {expected_bytes}")
    observed_sha256 = sha256_file(resolved)
    if observed_sha256 != expected_sha256:
        raise MeasurementError(f"source cache SHA-256 {observed_sha256} != expected {expected_sha256}")
    members = {name: stored_npy_memmap(resolved, name) for name in SOURCE_MEMBERS}
    arrays = {name: item.array for name, item in members.items()}
    lstars = arrays["lstars"]
    margins = arrays["margins"]
    gt_poses = arrays["gt_poses"]
    if lstars.ndim != 3 or margins.shape != lstars.shape:
        raise MeasurementError("lstars/margins must share (pairs,height,width) geometry")
    if gt_poses.shape != (lstars.shape[0], 6):
        raise MeasurementError("gt_poses must have shape (source_pairs,6)")
    if pair_count <= 0 or pair_count > lstars.shape[0]:
        raise MeasurementError(f"n-pairs must be in [1,{lstars.shape[0]}]; got {pair_count}")
    if gt_poses.dtype.kind != "f" or gt_poses.dtype.itemsize != 8:
        raise MeasurementError("gt_poses source member must be float64")
    if not np.issubdtype(lstars.dtype, np.integer):
        raise MeasurementError("lstars source member must be integer")
    if not np.issubdtype(margins.dtype, np.floating):
        raise MeasurementError("margins source member must be floating")
    custody = {
        "access": "complete_file_hash_then_direct_zip_stored_npy_read_only_memmap",
        "bytes": observed_bytes,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "members": {name: item.custody for name, item in members.items()},
        "mutated": False,
        "pair_count": pair_count,
        "path": str(resolved),
        "sha256": observed_sha256,
        "storage_preflight": storage,
    }
    return custody, arrays


def _receipt_binding(
    source: dict[str, Any],
    implementation: dict[str, Any],
    pair_count: int,
) -> dict[str, Any]:
    return {
        "implementation": implementation,
        "pair_count": pair_count,
        "schema": "sdwl1.measurement_binding.v1",
        "source_bytes": source["bytes"],
        "source_sha256": source["sha256"],
    }


def _validate_or_write_stage(
    path: Path,
    value: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    if resume and path.is_file():
        prior = load_json_object(path)
        if prior != value:
            raise MeasurementError(f"resume stage binding drift at {path}")
        return prior
    if path.exists():
        raise MeasurementError(f"refusing to overwrite stage receipt without --resume: {path}")
    atomic_json(path, value)
    return value


def build_or_resume_inventory(
    output_dir: Path,
    arrays: dict[str, np.memmap],
    *,
    pair_count: int,
    binding: dict[str, Any],
    resume: bool,
) -> tuple[FactInventory, dict[str, Any]]:
    payload_path = output_dir / INVENTORY_PAYLOAD
    stage_path = output_dir / STAGE_INVENTORY
    if resume and stage_path.is_file():
        stage = load_json_object(stage_path)
        if stage.get("binding") != binding:
            raise MeasurementError("inventory stage binding drift")
        payload = stage.get("payload")
        if not isinstance(payload, dict) or payload.get("path") != INVENTORY_PAYLOAD:
            raise MeasurementError("inventory stage payload metadata is malformed")
        if payload_path.stat().st_size != payload.get("bytes"):
            raise MeasurementError("preserved fact-inventory payload byte drift")
        if sha256_file(payload_path) != payload.get("sha256"):
            raise MeasurementError("preserved fact-inventory payload hash drift")
        tensor = np.load(payload_path, mmap_mode="r", allow_pickle=False)
        inventory = FactInventory(
            tensor=tensor,
            source_height=int(stage["source_geometry"][0]),
            source_width=int(stage["source_geometry"][1]),
            semantic_sha256=str(stage["semantic_sha256"]),
        )
        return inventory, stage

    inventory = extract_fact_inventory(
        arrays["lstars"][:pair_count],
        arrays["margins"][:pair_count],
        arrays["gt_poses"][:pair_count],
    )
    if payload_path.is_file():
        if not resume:
            raise MeasurementError(f"refusing to overwrite inventory payload {payload_path}")
        preserved = np.load(payload_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(preserved, inventory.tensor):
            raise MeasurementError("partial inventory payload differs from deterministic rebuild")
    else:
        atomic_npy(payload_path, inventory.tensor)
    stage = {
        "binding": binding,
        "described_record_count": inventory.described_record_count,
        "described_scalar_fact_count": inventory.described_scalar_fact_count,
        "pair_count": inventory.pair_count,
        "payload": {
            "bytes": payload_path.stat().st_size,
            "path": INVENTORY_PAYLOAD,
            "sha256": sha256_file(payload_path),
        },
        "schema": "sdwl1.fact_inventory_stage.v1",
        "semantic_sha256": inventory.semantic_sha256,
        "source_geometry": [inventory.source_height, inventory.source_width],
    }
    if stage_path.exists():
        raise MeasurementError(f"refusing to overwrite inventory stage {stage_path}")
    atomic_json(stage_path, stage)
    return inventory, stage


def measurement_specs() -> tuple[RowSpec, ...]:
    """Return exhaustive layout/mode rows and one-at-a-time MDL controls."""

    specs: list[RowSpec] = []
    for layout in SentenceLayout:
        specs.append(
            RowSpec(
                row_id=f"independent_{layout.value}_absolute",
                kind="independent",
                layout=layout,
                temporal_mode=TemporalMode.ABSOLUTE,
            )
        )
    for layout in SentenceLayout:
        for mode in TemporalMode:
            specs.append(
                RowSpec(
                    row_id=f"whole_{layout.value}_{mode.value}",
                    kind="whole",
                    layout=layout,
                    temporal_mode=mode,
                )
            )
            for counterfactual in COUNTERFACTUAL_FIELDS:
                specs.append(
                    RowSpec(
                        row_id=(f"whole_{layout.value}_{mode.value}__mdl_{counterfactual}"),
                        kind="counterfactual",
                        layout=layout,
                        temporal_mode=mode,
                        counterfactual=counterfactual,
                    )
                )
    return tuple(specs)


def _decode_row_payload(
    payload: bytes,
    spec: RowSpec,
    inventory: FactInventory,
) -> None:
    inner = decompress_outer_payload(payload)
    decoded = decode_independent_descriptions(inner) if spec.kind == "independent" else decode_sentence(inner)
    if decoded.semantic_sha256 != inventory.semantic_sha256 or not np.array_equal(decoded.tensor, inventory.tensor):
        raise MeasurementError(f"{spec.row_id} payload failed exact semantic parse-back")


def _measurement_dict(measurement: Any) -> dict[str, Any]:
    result = asdict(measurement)
    result.pop("outer_payload")
    return result


def build_or_resume_row(
    output_dir: Path,
    inventory: FactInventory,
    spec: RowSpec,
    *,
    binding: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    payload_rel = f"payloads/{spec.row_id}.zlib"
    row_rel = f"rows/{spec.row_id}.json"
    payload_path = output_dir / payload_rel
    row_path = output_dir / row_rel
    if resume and row_path.is_file():
        row = load_json_object(row_path)
        if row.get("binding") != binding or row.get("spec") != spec.as_dict():
            raise MeasurementError(f"resume row binding drift for {spec.row_id}")
        if row.get("outer_payload") != {
            "bytes": payload_path.stat().st_size,
            "path": payload_rel,
            "sha256": sha256_file(payload_path),
        }:
            raise MeasurementError(f"resume payload custody drift for {spec.row_id}")
        _decode_row_payload(payload_path.read_bytes(), spec, inventory)
        return row
    if row_path.exists() or (payload_path.exists() and not resume):
        raise MeasurementError(f"refusing to overwrite row without --resume: {spec.row_id}")
    measurement = (
        measure_serialization(inventory, independent_layout=spec.layout)
        if spec.kind == "independent"
        else measure_serialization(inventory, options=spec.options())
    )
    if not measurement.exact_parseback:
        raise MeasurementError(f"{spec.row_id} did not parse back exactly")
    if payload_path.is_file():
        persisted = payload_path.read_bytes()
        if persisted != measurement.outer_payload:
            raise MeasurementError(f"partial payload differs from deterministic rebuild for {spec.row_id}")
    else:
        atomic_bytes(payload_path, measurement.outer_payload)
    persisted = payload_path.read_bytes()
    _decode_row_payload(persisted, spec, inventory)
    row = {
        "binding": binding,
        "measurement": _measurement_dict(measurement),
        "outer_payload": {
            "bytes": len(persisted),
            "path": payload_rel,
            "sha256": sha256_bytes(persisted),
        },
        "schema": "sdwl1.measurement_row.v1",
        "semantic_sha256": inventory.semantic_sha256,
        "spec": spec.as_dict(),
    }
    atomic_json(row_path, row)
    return row


def annotate_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add full-payload temporal gains and same-semantics MDL deltas."""

    independent = {
        row["spec"]["layout"]: row["measurement"]["outer_deflate_bytes"]
        for row in rows
        if row["spec"]["kind"] == "independent"
    }
    base = {
        (row["spec"]["layout"], row["spec"]["temporal_mode"]): row["measurement"]["outer_deflate_bytes"]
        for row in rows
        if row["spec"]["kind"] == "whole"
    }
    annotated: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        spec = row["spec"]
        outer_bytes = row["measurement"]["outer_deflate_bytes"]
        if spec["kind"] == "independent":
            row["temporal_sharing_gain_bytes"] = None
            row["temporal_sharing_gain_fraction"] = None
        else:
            reset_bytes = independent[spec["layout"]]
            row["temporal_sharing_gain_bytes"] = reset_bytes - outer_bytes
            row["temporal_sharing_gain_fraction"] = (reset_bytes - outer_bytes) / reset_bytes
        if spec["kind"] == "counterfactual":
            control_bytes = base[(spec["layout"], spec["temporal_mode"])]
            delta = outer_bytes - control_bytes
            row["same_semantics_control_outer_bytes"] = control_bytes
            row["mdl_delta_outer_bytes"] = delta
            row["dimension_admitted"] = delta < 0
        else:
            row["same_semantics_control_outer_bytes"] = None
            row["mdl_delta_outer_bytes"] = None
            row["dimension_admitted"] = None
        annotated.append(row)
    return annotated


def summarize_selection(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the complete-payload dimension and MDL verdicts."""

    base_rows = [row for row in rows if row["spec"]["kind"] == "whole"]
    selected = min(
        base_rows,
        key=lambda row: (
            row["measurement"]["outer_deflate_bytes"],
            row["spec"]["row_id"],
        ),
    )
    selected_layout = selected["spec"]["layout"]
    selected_mode = selected["spec"]["temporal_mode"]
    layout_candidates = []
    for layout in SentenceLayout:
        candidates = [row for row in base_rows if row["spec"]["layout"] == layout.value]
        best = min(
            candidates,
            key=lambda row: (
                row["measurement"]["outer_deflate_bytes"],
                row["spec"]["row_id"],
            ),
        )
        layout_candidates.append(
            {
                "best_outer_deflate_bytes": best["measurement"]["outer_deflate_bytes"],
                "best_temporal_mode": best["spec"]["temporal_mode"],
                "layout": layout.value,
                "selected": layout.value == selected_layout,
            }
        )
    temporal_candidates = [
        {
            "outer_deflate_bytes": row["measurement"]["outer_deflate_bytes"],
            "selected": row["spec"]["temporal_mode"] == selected_mode,
            "temporal_mode": row["spec"]["temporal_mode"],
        }
        for row in base_rows
        if row["spec"]["layout"] == selected_layout
    ]
    counterfactuals = [
        row
        for row in rows
        if row["spec"]["kind"] == "counterfactual"
        and row["spec"]["layout"] == selected_layout
        and row["spec"]["temporal_mode"] == selected_mode
    ]
    mdl = [
        {
            "admitted": row["dimension_admitted"],
            "counterfactual": row["spec"]["counterfactual"],
            "mdl_delta_outer_bytes": row["mdl_delta_outer_bytes"],
            "outer_deflate_bytes": row["measurement"]["outer_deflate_bytes"],
            "verdict": "ADMIT" if row["dimension_admitted"] else "PRUNE",
        }
        for row in counterfactuals
    ]
    production_counts = selected["measurement"]["production_counts"]
    zero_use = {
        family: sorted(name for name, count in production_counts[family].items() if count == 0)
        for family in ("subjects", "predicates", "modifiers")
    }
    selection = {
        "admitted_counterfactual_dimensions": sorted(item["counterfactual"] for item in mdl if item["admitted"]),
        "layout_candidates": layout_candidates,
        "pruned_counterfactual_dimensions": sorted(item["counterfactual"] for item in mdl if not item["admitted"]),
        "selected_layout": selected_layout,
        "selected_temporal_mode": selected_mode,
        "temporal_candidates_for_selected_layout": temporal_candidates,
        "zero_use_vocabulary_pruned": zero_use,
    }
    return selected, {"dimension_selection": selection, "mdl_pruning": mdl}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-pairs", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument(
        "--expected-source-bytes",
        type=int,
        default=EXPECTED_SOURCE_BYTES,
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_BYTES,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    current_storage = storage_preflight(
        output_dir,
        required_free_bytes=args.min_free_bytes,
    )
    custody_path = output_dir / STAGE_CUSTODY
    if args.resume and custody_path.is_file():
        prior_custody = load_json_object(custody_path)
        prior_source = prior_custody.get("source")
        if not isinstance(prior_source, dict) or not isinstance(prior_source.get("storage_preflight"), dict):
            raise MeasurementError("preserved source custody lacks storage preflight")
        storage = prior_source["storage_preflight"]
        for field in ("output_dir", "required_free_bytes"):
            if current_storage[field] != storage.get(field):
                raise MeasurementError(f"resume storage-preflight {field} drift")
    else:
        storage = current_storage
    implementation = implementation_custody()
    source, arrays = source_custody(
        args.source_cache,
        expected_bytes=args.expected_source_bytes,
        expected_sha256=args.expected_source_sha256,
        pair_count=args.n_pairs,
        storage=storage,
    )
    binding = _receipt_binding(source, implementation, args.n_pairs)
    custody_stage = {
        "binding": binding,
        "research_only": True,
        "schema": "sdwl1.source_custody_stage.v1",
        "source": source,
    }
    _validate_or_write_stage(
        custody_path,
        custody_stage,
        resume=args.resume,
    )
    inventory, inventory_stage = build_or_resume_inventory(
        output_dir,
        arrays,
        pair_count=args.n_pairs,
        binding=binding,
        resume=args.resume,
    )
    raw_rows = [
        build_or_resume_row(
            output_dir,
            inventory,
            spec,
            binding=binding,
            resume=args.resume,
        )
        for spec in measurement_specs()
    ]
    rows = annotate_comparisons(raw_rows)
    selected, selection_summary = summarize_selection(rows)
    receipt = {
        "axis": "[macOS-CPU frozen-scorer advisory]",
        "candidate_archive": False,
        "checkpoint_policy": (
            "atomic source/inventory/row receipts and preserved payloads; "
            "--resume hash-binds and strictly reparses every stage"
        ),
        "cleanup": {
            "bulk_artifacts_created": False,
            "policy": "preserve small receipts/payloads; source remained read-only",
            "source_copied": False,
        },
        "coverage": {
            "counterfactual_fields": list(COUNTERFACTUAL_FIELDS),
            "independent_baseline_definition": (
                "600 separately framed absolute SDWL1 descriptions with arithmetic "
                "state reset per pair, then one complete zlib9 outer payload"
            ),
            "layouts": [layout.value for layout in SentenceLayout],
            "row_count": len(rows),
            "temporal_modes": [mode.value for mode in TemporalMode],
        },
        "described_fraction": 1,
        **selection_summary,
        "execution_allowed": False,
        "grammar": {
            "name": "Scorer-Derived Worldsheet Language v1",
            "pair_record_count": inventory.described_record_count // inventory.pair_count,
            "pair_scalar_fact_count": (inventory.described_scalar_fact_count // inventory.pair_count),
            "short_name": "SDWL1",
        },
        "implementation": implementation,
        "inventory_stage": inventory_stage,
        "main_landing_review_required": True,
        "pointer_moved": False,
        "production_counts": selected["measurement"]["production_counts"],
        "promotion_eligible": False,
        "rederive": {
            "argv": [
                ".venv/bin/python",
                "tools/measure_ddm_dv2_sdwl1.py",
                "--source-cache",
                str(args.source_cache),
                "--output-dir",
                str(args.output_dir),
                "--n-pairs",
                str(args.n_pairs),
            ],
            "resume_flag": "--resume",
        },
        "research_only": True,
        "rows": rows,
        "schema": "sdwl1.n600_measurement_receipt.v1",
        "score_claim": False,
        "selected_base_row": {
            "outer_deflate_bytes": selected["measurement"]["outer_deflate_bytes"],
            "outer_deflate_sha256": selected["measurement"]["outer_deflate_sha256"],
            "row_id": selected["spec"]["row_id"],
        },
        "source_custody": source,
        "syntax": {
            "arithmetic_coder": ("tac.optimization.arith_selfcomp_rate_coders.encode_spatial_context_arithmetic"),
            "collection_magic": "SDWL1IC\\x00",
            "json": "UTF-8 canonical sorted-key compact JSON",
            "outer_coder": "zlib level 9 over the complete framed object",
            "packet_magic": "SDWL1PK\\x00",
            "section_frame": "tag:u32 + payload_bytes:u64 + payload_sha256:32B",
            "strict_parseback": True,
            "version": 1,
        },
        "verdict_scope": (
            "exact declared SDWL1 fact inventory and complete outer-zlib payloads only; "
            "not pixels, witness closure, evaluator output, or contest score"
        ),
    }
    receipt_path = output_dir / FINAL_RECEIPT
    receipt_bytes = canonical_json_bytes(receipt)
    if args.resume and receipt_path.is_file():
        if receipt_path.read_bytes() != receipt_bytes:
            raise MeasurementError("final receipt changed under --resume")
    else:
        if receipt_path.exists():
            raise MeasurementError(f"refusing to overwrite final receipt {receipt_path}")
        atomic_bytes(receipt_path, receipt_bytes)
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = run(args)
    except (MeasurementError, OSError, KeyError, ValueError) as exc:
        print(f"SDWL1 measurement refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "receipt_sha256": sha256_file(args.output_dir / FINAL_RECEIPT),
                "row_count": receipt["coverage"]["row_count"],
                "selected_base_row": receipt["selected_base_row"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
