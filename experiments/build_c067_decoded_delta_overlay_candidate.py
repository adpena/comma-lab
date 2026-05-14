#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a byte-closed CDO1 decoded-mask overlay candidate archive.

The CDO1 payload is a charged sidecar applied after the base mask stream is
decoded.  This builder does not run scorers and always records
``score_claim=false``; exact CUDA auth eval of the emitted archive is required
for any score statement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import lzma
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import zlib


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "c067_decoded_delta_overlay_candidate_v1"
TOOL = "experiments/build_c067_decoded_delta_overlay_candidate.py"
CDO1_MAGIC = b"CDO1"
CDO1_VERSION = 1
CDO1_HEADER_STRUCT = struct.Struct("<4sHI")
CDO1_MAX_HEADER_JSON_BYTES = 1 << 20
CDO1_PAIR_INDEX_BASIS_VALUES = {
    "half_frame_pair_index",
    "video_frame_pair_index",
}
DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c067_decoded_delta_overlay_candidate_20260502"
DEFAULT_OUTPUT_ARCHIVE = DEFAULT_OUTPUT_DIR / "archive.zip"
DEFAULT_MANIFEST_JSON = DEFAULT_OUTPUT_DIR / "c067_decoded_delta_overlay_candidate_manifest.json"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"
CDO1_MEMBER_BY_COMPRESSOR = {
    "raw": "masks.cdo1",
    "zlib": "masks.cdo1.zlib",
    "lzma_xz": "masks.cdo1.xz",
    "brotli": "masks.cdo1.br",
}
PACKED_PAYLOAD_MEMBERS = ("p", "renderer_payload.bin.br")
DEFAULT_PACKED_OUTPUT_PAYLOAD_FORMAT = "rpk1_json"
MEMBER_ORDER = (
    "renderer.bin",
    "masks.mkv",
    "grayscale.mkv",
    "masks.alpha4.mkv",
    "masks.amrc",
    "masks.nrv",
    "masks.cmg2",
    "masks.cmg3",
    "masks.cdo1",
    "masks.cdo1.zlib",
    "masks.cdo1.xz",
    "masks.cdo1.br",
    "optimized_poses.bin",
    "optimized_poses.pt",
    "optimized_embedding.pt",
    "poses.pt",
    "zoom_scalars.bin",
    "foveation_params.bin",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe_member_name(name: str) -> str:
    parts = Path(name).parts
    if not name or name.startswith("/") or ".." in parts or len(parts) != 1:
        raise ValueError(f"unsafe archive member path: {name!r}")
    return name


def _read_zip_members(path: Path) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"base archive missing: {path}")
    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                raise ValueError(f"directory member not allowed: {info.filename!r}")
            name = _safe_member_name(info.filename)
            if name in members:
                raise ValueError(f"duplicate archive member: {name!r}")
            if name.startswith("masks.cdo1"):
                raise ValueError(f"base archive already contains CDO1 overlay member: {name!r}")
            data = zf.read(info)
            members[name] = data
            inventory.append(
                {
                    "name": name,
                    "size_bytes": len(data),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "sha256": _sha256_bytes(data),
                }
            )
    return members, inventory


def _logical_inventory(members: dict[str, bytes], *, source: str) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "size_bytes": len(data),
            "sha256": _sha256_bytes(data),
            "logical_source": source,
        }
        for name, data in _ordered_members(members)
    ]


def _unpack_packed_source_archive(
    path: Path,
    *,
    zip_members: dict[str, bytes],
    repo_root: Path,
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    unpacker = _load_module(UNPACKER_PATH, "_cdo1_overlay_unpacker")
    with tempfile.TemporaryDirectory(prefix="cdo1_base_unpack_") as tmp:
        archive_dir = Path(tmp)
        for name, data in zip_members.items():
            (archive_dir / _safe_member_name(name)).write_bytes(data)
        summary = unpacker.unpack_renderer_payload(archive_dir)
        logical: dict[str, bytes] = {}
        for row in summary.get("members", []):
            if not isinstance(row, dict):
                raise ValueError("packed payload unpack summary contains a non-object member row")
            name = _safe_member_name(str(row.get("name", "")))
            member_path = archive_dir / name
            if not member_path.is_file():
                raise ValueError(f"packed payload summary member was not materialized: {name}")
            data = member_path.read_bytes()
            expected_sha = str(row.get("sha256", ""))
            actual_sha = _sha256_bytes(data)
            if expected_sha and expected_sha != actual_sha:
                raise ValueError(
                    f"unpacked member SHA mismatch for {name}: "
                    f"{actual_sha} != {expected_sha}"
                )
            logical[name] = data
    return logical, _logical_inventory(logical, source="packed_payload_unpacked"), {
        "base_archive_layout": "packed_payload",
        "zip_payload_members": sorted(zip_members),
        "unpacked_by": str(UNPACKER_PATH.relative_to(repo_root)),
        "unpack_summary": summary,
    }


def _validate_runtime_members(members: dict[str, bytes]) -> None:
    if "renderer.bin" not in members:
        raise ValueError("base archive missing renderer.bin")
    if not any(
        name in members
        for name in (
            "masks.mkv",
            "grayscale.mkv",
            "masks.alpha4.mkv",
            "masks.amrc",
            "masks.nrv",
            "masks.cmg2",
            "masks.cmg3",
        )
    ):
        raise ValueError("base archive missing a supported base mask payload")
    if not any(
        name in members
        for name in ("optimized_poses.bin", "optimized_poses.pt", "zoom_scalars.bin")
    ):
        raise ValueError("base archive missing pose or zoom payload")


def _read_source_archive(
    path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    zip_members, zip_inventory = _read_zip_members(path)
    packed_names = [name for name in PACKED_PAYLOAD_MEMBERS if name in zip_members]
    has_logical_renderer = "renderer.bin" in zip_members
    if packed_names and not has_logical_renderer:
        members, inventory, contract = _unpack_packed_source_archive(
            path,
            zip_members=zip_members,
            repo_root=repo_root,
        )
    else:
        members = zip_members
        inventory = zip_inventory
        contract = {
            "base_archive_layout": "expanded_members",
            "zip_payload_members": packed_names,
        }
    _validate_runtime_members(members)
    contract["source_zip_inventory"] = zip_inventory
    return members, inventory, contract


def _compress_cdo1_payload(raw: bytes, compressor: str) -> tuple[str, bytes]:
    if not raw.startswith(b"CDO1"):
        raise ValueError("CDO1 overlay payload must start with CDO1 magic")
    if compressor not in CDO1_MEMBER_BY_COMPRESSOR:
        raise ValueError(f"unsupported CDO1 compressor {compressor!r}")
    if compressor == "raw":
        return CDO1_MEMBER_BY_COMPRESSOR[compressor], raw
    if compressor == "zlib":
        return CDO1_MEMBER_BY_COMPRESSOR[compressor], zlib.compress(raw, level=9)
    if compressor == "lzma_xz":
        return CDO1_MEMBER_BY_COMPRESSOR[compressor], lzma.compress(
            raw,
            format=lzma.FORMAT_XZ,
            preset=9 | lzma.PRESET_EXTREME,
        )
    try:
        import brotli  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("brotli compressor requested but brotli is unavailable") from exc
    return CDO1_MEMBER_BY_COMPRESSOR[compressor], brotli.compress(raw, quality=11)


def _decode_cdo1_payload_header(raw: bytes) -> dict[str, Any]:
    if len(raw) < CDO1_HEADER_STRUCT.size:
        raise ValueError("CDO1 overlay payload is shorter than the fixed header")
    magic, version, header_len = CDO1_HEADER_STRUCT.unpack(raw[: CDO1_HEADER_STRUCT.size])
    if magic != CDO1_MAGIC:
        raise ValueError(f"CDO1 overlay payload has bad magic {magic!r}")
    if int(version) != CDO1_VERSION:
        raise ValueError(f"unsupported CDO1 overlay payload version {version}")
    if header_len <= 0 or header_len > CDO1_MAX_HEADER_JSON_BYTES:
        raise ValueError(f"CDO1 overlay header length outside strict bounds: {header_len}")
    header_start = CDO1_HEADER_STRUCT.size
    header_end = header_start + int(header_len)
    if header_end > len(raw):
        raise ValueError("CDO1 overlay header extends past payload")
    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("CDO1 overlay header must be a JSON object")
    pair_index_basis = header.get("pair_index_basis")
    if pair_index_basis not in CDO1_PAIR_INDEX_BASIS_VALUES:
        raise ValueError(
            "CDO1 overlay header missing valid pair_index_basis "
            f"{sorted(CDO1_PAIR_INDEX_BASIS_VALUES)}"
        )
    selected_pairs = header.get("selected_pair_indices")
    if (
        not isinstance(selected_pairs, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selected_pairs)
        or selected_pairs != sorted(selected_pairs)
    ):
        raise ValueError("CDO1 overlay header selected_pair_indices must be sorted ints")
    shape = header.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError(f"CDO1 overlay header has invalid shape {shape!r}")
    frame_count = int(shape[0])
    max_pair = max(selected_pairs, default=-1)
    if pair_index_basis == "half_frame_pair_index" and max_pair >= frame_count:
        raise ValueError(
            "CDO1 half_frame_pair_index selected pair exceeds decoded mask frame count"
        )
    if pair_index_basis == "video_frame_pair_index" and max_pair >= max(1, frame_count // 2):
        raise ValueError(
            "CDO1 video_frame_pair_index selected pair exceeds decoded mask pair count"
        )
    return header


def _cdo1_header_manifest(header: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": header.get("schema"),
        "shape": header.get("shape"),
        "pair_index_basis": header.get("pair_index_basis"),
        "selected_pair_indices": header.get("selected_pair_indices"),
        "selected_pixel_count": header.get("selected_pixel_count"),
        "run_count": header.get("run_count"),
        "base_mask_tensor_sha256": header.get("base_mask_tensor_sha256"),
        "reconstructed_mask_u8_sha256": header.get("reconstructed_mask_u8_sha256"),
    }


def _load_cdo1_raw_payload(
    *,
    overlay_payload: Path | None,
    overlay_spec_json: Path | None,
    repo_root: Path,
) -> tuple[bytes, dict[str, Any], str | None]:
    if (overlay_payload is None) == (overlay_spec_json is None):
        raise ValueError("provide exactly one of overlay_payload or overlay_spec_json")
    if overlay_payload is not None:
        raw = overlay_payload.read_bytes()
        header = _decode_cdo1_payload_header(raw)
        return raw, {
            "source": "payload",
            "path": str(overlay_payload.resolve()),
            "payload_bytes": len(raw),
            "payload_sha256": _sha256_bytes(raw),
            "payload_header": _cdo1_header_manifest(header),
            "pair_index_basis": header["pair_index_basis"],
            "selected_pair_indices": header["selected_pair_indices"],
        }, None

    assert overlay_spec_json is not None
    spec = json.loads(overlay_spec_json.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"{overlay_spec_json} must contain a JSON object")
    payload_path_text = spec.get("payload_path")
    if not isinstance(payload_path_text, str) or not payload_path_text:
        raise ValueError(f"{overlay_spec_json} lacks payload_path")
    payload_path = Path(payload_path_text)
    if not payload_path.is_absolute():
        payload_path = repo_root / payload_path
    raw = payload_path.read_bytes()
    expected_sha = spec.get("payload_sha256")
    actual_sha = _sha256_bytes(raw)
    if expected_sha is not None and expected_sha != actual_sha:
        raise ValueError(f"overlay spec payload SHA mismatch: {actual_sha} != {expected_sha}")
    header = _decode_cdo1_payload_header(raw)
    spec_header = spec.get("payload_header")
    if isinstance(spec_header, dict):
        for key in ("pair_index_basis", "selected_pair_indices", "shape"):
            if spec_header.get(key) != header.get(key):
                raise ValueError(
                    f"overlay spec payload_header[{key!r}] disagrees with CDO1 header"
                )
    recommended = spec.get("recommended_compressor")
    return raw, {
        "source": "safe_spec",
        "spec_path": str(overlay_spec_json.resolve()),
        "payload_path": str(payload_path.resolve()),
        "payload_bytes": len(raw),
        "payload_sha256": actual_sha,
        "spec_sha256": _sha256_file(overlay_spec_json),
        "payload_header": _cdo1_header_manifest(header),
        "pair_index_basis": header["pair_index_basis"],
        "selected_pair_indices": header["selected_pair_indices"],
    }, recommended if isinstance(recommended, str) else None


def _ordered_members(members: dict[str, bytes]) -> list[tuple[str, bytes]]:
    selected: list[str] = []
    for name in MEMBER_ORDER:
        if name in members:
            selected.append(name)
    selected.extend(sorted(name for name in members if name not in selected))
    return [(name, members[name]) for name in selected]


def _write_candidate_archive(path: Path, ordered: list[tuple[str, bytes]]) -> None:
    from tac.submission_archive import write_deterministic_zip_member

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in ordered:
            write_deterministic_zip_member(zf, name, data, compresslevel=9)


def _write_packed_candidate_archive(
    *,
    output_archive: Path,
    ordered: list[tuple[str, bytes]],
    payload_member_name: str,
    payload_format: str,
    brotli_quality: int,
) -> dict[str, Any]:
    packer = _load_module(PACKER_PATH, "_cdo1_overlay_packer")
    with tempfile.TemporaryDirectory(prefix="cdo1_expanded_candidate_") as tmp:
        expanded_archive = Path(tmp) / "expanded_candidate.zip"
        _write_candidate_archive(expanded_archive, ordered)
        expanded_sha = _sha256_file(expanded_archive)
        expanded_bytes = expanded_archive.stat().st_size
        pack_report = packer.build_packed_archive(
            expanded_archive,
            output_archive,
            brotli_quality=brotli_quality,
            payload_member_name=payload_member_name,
            payload_format=payload_format,
        )
    return {
        "packed_output": True,
        "packer": str(PACKER_PATH.relative_to(REPO_ROOT)),
        "payload_member_name": payload_member_name,
        "payload_format": payload_format,
        "brotli_quality": brotli_quality,
        "expanded_candidate_archive_bytes": expanded_bytes,
        "expanded_candidate_archive_sha256": expanded_sha,
        "packed_archive_report": pack_report,
    }


def _archive_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        for info in zf.infolist():
            data = zf.read(info)
            rows.append(
                {
                    "name": info.filename,
                    "size_bytes": int(info.file_size),
                    "compressed_size_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "compress_type": int(info.compress_type),
                    "permissions_octal": oct((info.external_attr >> 16) & 0o777),
                    "sha256": _sha256_bytes(data),
                }
            )
    return rows


def build_candidate(
    *,
    base_archive: Path = DEFAULT_BASE_ARCHIVE,
    overlay_payload: Path | None = None,
    overlay_spec_json: Path | None = None,
    output_archive: Path = DEFAULT_OUTPUT_ARCHIVE,
    manifest_json: Path | None = DEFAULT_MANIFEST_JSON,
    overlay_compressor: str | None = None,
    pack_output_payload: bool = False,
    packed_payload_member_name: str = "p",
    packed_payload_format: str = DEFAULT_PACKED_OUTPUT_PAYLOAD_FORMAT,
    brotli_quality: int = 11,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    source_members, source_inventory, source_contract = _read_source_archive(
        base_archive,
        repo_root=repo_root,
    )
    raw_payload, payload_source, recommended_compressor = _load_cdo1_raw_payload(
        overlay_payload=overlay_payload,
        overlay_spec_json=overlay_spec_json,
        repo_root=repo_root,
    )
    compressor = overlay_compressor or recommended_compressor or "lzma_xz"
    overlay_member, overlay_bytes = _compress_cdo1_payload(raw_payload, compressor)
    members = dict(source_members)
    members[overlay_member] = overlay_bytes
    ordered = _ordered_members(members)
    if pack_output_payload:
        output_packaging = _write_packed_candidate_archive(
            output_archive=output_archive,
            ordered=ordered,
            payload_member_name=packed_payload_member_name,
            payload_format=packed_payload_format,
            brotli_quality=brotli_quality,
        )
    else:
        _write_candidate_archive(output_archive, ordered)
        output_packaging = {"packed_output": False}
    validation = validate_archive(
        output_archive,
        manifest=detect_pose_manifest(output_archive),
        strict=True,
    )
    if not validation.valid:
        raise ValueError(validation.summary())
    report = {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "auth_eval_required": True,
        "base_archive": {
            "path": str(base_archive.resolve()),
            "bytes": base_archive.stat().st_size,
            "sha256": _sha256_file(base_archive),
            "members": source_inventory,
            "runtime_member_contract": source_contract,
        },
        "cdo1_overlay": {
            "archive_member": overlay_member,
            "compressor": compressor,
            "raw_bytes": len(raw_payload),
            "raw_sha256": _sha256_bytes(raw_payload),
            "compressed_bytes": len(overlay_bytes),
            "compressed_sha256": _sha256_bytes(overlay_bytes),
            "source": payload_source,
            "pair_index_basis": payload_source["pair_index_basis"],
            "selected_pair_indices": payload_source["selected_pair_indices"],
            "payload_header": payload_source["payload_header"],
        },
        "output_archive": {
            "path": str(output_archive.resolve()),
            "bytes": output_archive.stat().st_size,
            "sha256": _sha256_file(output_archive),
            "delta_bytes_vs_base_archive": output_archive.stat().st_size - base_archive.stat().st_size,
            "inventory": _archive_inventory(output_archive),
            "packaging": output_packaging,
        },
        "runtime_contract": {
            "base_mask_sha_checked_by_cdo1_header": True,
            "reconstructed_mask_sha_checked_by_cdo1_header": True,
            "exact_score_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
        },
    }
    if manifest_json is not None:
        _write_json(manifest_json, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overlay-payload", type=Path)
    group.add_argument("--overlay-spec-json", type=Path)
    parser.add_argument("--overlay-compressor", choices=sorted(CDO1_MEMBER_BY_COMPRESSOR))
    parser.add_argument(
        "--pack-output-payload",
        action="store_true",
        help="Repack logical runtime members into a single renderer payload member after adding CDO1.",
    )
    parser.add_argument(
        "--packed-payload-member-name",
        choices=("p", "renderer_payload.bin.br"),
        default="p",
    )
    parser.add_argument("--packed-payload-format", default=DEFAULT_PACKED_OUTPUT_PAYLOAD_FORMAT)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--output-archive", type=Path, default=DEFAULT_OUTPUT_ARCHIVE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_candidate(
        base_archive=args.base_archive,
        overlay_payload=args.overlay_payload,
        overlay_spec_json=args.overlay_spec_json,
        output_archive=args.output_archive,
        manifest_json=args.manifest_json,
        overlay_compressor=args.overlay_compressor,
        pack_output_payload=args.pack_output_payload,
        packed_payload_member_name=args.packed_payload_member_name,
        packed_payload_format=args.packed_payload_format,
        brotli_quality=args.brotli_quality,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "archive": report["output_archive"],
                "cdo1_overlay": report["cdo1_overlay"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
