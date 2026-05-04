#!/usr/bin/env python3
"""Repack a Quantizr-faithful archive with top-submission renderer codecs.

This is a build-only, no-score transform.  It refuses non-JointFrameGenerator
renderer payloads so OWV3/ASYM archives cannot be accidentally re-labeled as
top-submission packer candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_renderer_packed_payload_archive import (
    PAYLOAD_FORMAT_PR64_LEN_TABLE,
    PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    PAYLOAD_FORMAT_RPK1_JSON,
    POSE_FP16_COL_DELTA_CODEC,
    POSE_FP16_VELOCITY_ONLY_CODEC,
    POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
    POSE_QP1_CODEC,
    POSE_QPOSE14_COL_DELTA_CODEC,
    build_packed_archive,
)
from submissions.robust_current.unpack_renderer_payload import unpack_renderer_payload
from tac.quantizr_faithful_export import load_qfai
from tac.quantizr_qzs3_codec import (
    decode_qzs3_state_dict,
    encode_qzs3_state_dict,
    encode_qzs4_block_search_state_dict,
)
from tac.quantizr_torch_fp4_codec import (
    encode_torch_fp4_state_dict,
    load_torch_fp4_bytes,
)

FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ORIGINAL_VIDEO_BYTES = 37_545_489
RUNTIME_MEMBERS = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
OPTIONAL_GEOMETRY_MEMBERS = ("zoom_scalars.bin",)
PACKED_PAYLOAD_MEMBERS = ("renderer_payload.bin", "renderer_payload.bin.br", "p")
RENDERER_CODEC_QZS3 = "qzs3"
RENDERER_CODEC_QZS4 = "qzs4"
RENDERER_CODEC_TORCH_FP4 = "torch_fp4"
SUBMISSION_LAYOUT_MULTI_MEMBER = "multi_member"
SUBMISSION_LAYOUT_RPK1_SINGLE_BLOB = "rpk1_single_blob"
SUBMISSION_LAYOUT_PR64_SINGLE_BLOB = "pr64_single_blob"
SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB = "pr64_mask_first_single_blob"

_SINGLE_BLOB_PAYLOAD_FORMATS = {
    SUBMISSION_LAYOUT_RPK1_SINGLE_BLOB: PAYLOAD_FORMAT_RPK1_JSON,
    SUBMISSION_LAYOUT_PR64_SINGLE_BLOB: PAYLOAD_FORMAT_PR64_LEN_TABLE,
    SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB: PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _member_metadata(data: bytes, *, member_name: str) -> dict[str, Any]:
    return {
        "member_name": member_name,
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _members_metadata(members: dict[str, bytes], names: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        name: _member_metadata(members[name], member_name=name)
        for name in names
        if name in members
    }


def _safe_zip_members(source_archive: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(source_archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            path = Path(name)
            if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
                raise ValueError(f"unsafe archive member path: {name!r}")
            if name in members:
                raise ValueError(f"duplicate archive member: {name!r}")
            members[name] = zf.read(info)
    return members


def _merge_optional_geometry_members(
    runtime_members: dict[str, bytes],
    source_members: dict[str, bytes],
) -> None:
    for name in OPTIONAL_GEOMETRY_MEMBERS:
        if name not in source_members:
            continue
        existing = runtime_members.get(name)
        if existing is not None and existing != source_members[name]:
            raise ValueError(f"conflicting optional geometry member: {name!r}")
        runtime_members[name] = source_members[name]


def _runtime_members_from_source_archive(source_archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Return normal runtime members from a multi-member or packed payload archive."""

    members = _safe_zip_members(source_archive)
    if all(name in members for name in RUNTIME_MEMBERS):
        runtime_members = {name: members[name] for name in RUNTIME_MEMBERS}
        _merge_optional_geometry_members(runtime_members, members)
        return runtime_members, {
            "source_archive_contract": "runtime_member_zip",
            "unpacked_from_packed_payload": False,
            "required_runtime_members": _members_metadata(runtime_members, RUNTIME_MEMBERS),
            "optional_geometry_members": _members_metadata(runtime_members, OPTIONAL_GEOMETRY_MEMBERS),
        }

    payload_names = [name for name in PACKED_PAYLOAD_MEMBERS if name in members]
    if len(payload_names) != 1:
        missing = [name for name in RUNTIME_MEMBERS if name not in members]
        raise ValueError(
            "source archive missing required runtime members "
            f"{missing} and did not contain exactly one packed payload member "
            f"{PACKED_PAYLOAD_MEMBERS}; found payload members {payload_names}"
        )

    payload_name = payload_names[0]
    with tempfile.TemporaryDirectory(prefix="qzs_repack_unpack_") as tmpdir:
        archive_dir = Path(tmpdir)
        (archive_dir / payload_name).write_bytes(members[payload_name])
        summary = unpack_renderer_payload(archive_dir)
        runtime_members = {
            name: (archive_dir / name).read_bytes()
            for name in (*RUNTIME_MEMBERS, *OPTIONAL_GEOMETRY_MEMBERS)
            if (archive_dir / name).is_file()
        }
    _merge_optional_geometry_members(runtime_members, members)
    missing = [name for name in RUNTIME_MEMBERS if name not in runtime_members]
    if missing:
        raise ValueError(
            f"packed payload member {payload_name!r} did not unpack required "
            f"runtime members: {missing}"
        )
    return runtime_members, {
        "source_archive_contract": "packed_payload_zip",
        "unpacked_from_packed_payload": True,
        "packed_payload_member": payload_name,
        "packed_payload_unpack_summary": summary,
        "required_runtime_members": _members_metadata(runtime_members, RUNTIME_MEMBERS),
        "optional_geometry_members": _members_metadata(runtime_members, OPTIONAL_GEOMETRY_MEMBERS),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _stored_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    info.create_system = 3
    info.extra = b""
    info.comment = b""
    return info


def _rewrite_single_blob_archive_with_geometry(
    output_archive: Path,
    geometry_members: dict[str, bytes],
) -> dict[str, Any]:
    if not geometry_members:
        return {}
    existing_members = _safe_zip_members(output_archive)
    if len(existing_members) != 1:
        raise ValueError(
            "single-blob geometry preservation expected one packed payload "
            f"member before adding geometry, got {sorted(existing_members)}"
        )
    payload_name, payload_bytes = next(iter(existing_members.items()))
    if payload_name not in PACKED_PAYLOAD_MEMBERS:
        raise ValueError(f"unexpected packed payload member: {payload_name!r}")

    ordered_geometry_names = [name for name in OPTIONAL_GEOMETRY_MEMBERS if name in geometry_members]
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(_stored_zip_info(payload_name), payload_bytes)
        for name in ordered_geometry_names:
            zf.writestr(
                _zip_info(name),
                geometry_members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return {
        "preservation_mode": "separate_charged_archive_members",
        "payload_member": payload_name,
        "member_order": [payload_name, *ordered_geometry_names],
        "members": _members_metadata(geometry_members, ordered_geometry_names),
    }


def _zip_overhead_breakdown(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
    compressed_member_bytes = sum(info.compress_size for info in infos)
    uncompressed_member_bytes = sum(info.file_size for info in infos)
    archive_bytes = path.stat().st_size
    return {
        "archive_bytes": archive_bytes,
        "member_count": len(infos),
        "compressed_member_bytes": compressed_member_bytes,
        "uncompressed_member_bytes": uncompressed_member_bytes,
        "zip_container_overhead_bytes": archive_bytes - compressed_member_bytes,
        "member_names": [info.filename for info in infos],
    }


def _submission_path_metadata(
    *,
    layout: str,
    output_archive: Path,
    pose_codec: str,
    pose_residual_topk: int,
) -> dict[str, Any]:
    return {
        "layout": layout,
        "zip_overhead": _zip_overhead_breakdown(output_archive),
        "charged_accounting": {
            "archive_bytes": output_archive.stat().st_size,
            "archive_sha256": _sha256_path(output_archive),
            "pose_codec": pose_codec,
            "pose_residual_topk": int(pose_residual_topk),
            "all_score_affecting_bits_inside_archive": True,
            "score_claim": False,
        },
        "full_pipeline_hooks": {
            "archive_member_overhead_screen": True,
            "single_blob_layout_screen": layout != SUBMISSION_LAYOUT_MULTI_MEMBER,
            "deterministic_runtime_simplification_screen": True,
            "rl_bandit_multipass_atom_selection_ready": True,
        },
        "stackability": {
            "pr65_postprocess_qpost_sidecar": "stackable_if_qpost.bin_is_charged_and_runtime_reads_it",
            "mask_grammar_lanes": "stackable_if_mask_member_contract_and_decoder_are_inside_archive",
            "pose_atoms": "stackable_with_qp1_col0_or_pvr1_residual_atoms_after_exact_eval",
            "renderer_atoms": "qzs4_keeps_QZS3_wire_format_for_existing_decoder",
        },
    }


def _parse_block_sizes(raw: str | None) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return (16, 24, 32, 48, 64, 96, 128)
    block_sizes = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not block_sizes:
        raise ValueError("block-size list must not be empty")
    return block_sizes


def _qzs3_renderer_bytes(
    renderer_raw: bytes,
    *,
    block_size: int = 32,
) -> tuple[bytes, dict[str, Any]]:
    magic = renderer_raw[:4]
    if magic == b"QZS3":
        state = decode_qzs3_state_dict(renderer_raw, device="cpu")
        source_block_size = int.from_bytes(renderer_raw[4:6], "little")
        if int(block_size) == source_block_size:
            return renderer_raw, {
                "source_renderer_format": "QZS3",
                "action": "validated_existing_qzs3",
                "block_size": int(block_size),
                "source_block_size": source_block_size,
            }
        qzs3 = encode_qzs3_state_dict(state, block_size=block_size)
        decode_qzs3_state_dict(qzs3, device="cpu")
        return qzs3, {
            "source_renderer_format": "QZS3",
            "action": "reencoded_qzs3_from_qzs3_state_dict",
            "block_size": int(block_size),
            "source_block_size": source_block_size,
        }
    if magic == b"QFAI":
        with tempfile.NamedTemporaryFile(suffix=".qfai.bin") as tmp:
            tmp.write(renderer_raw)
            tmp.flush()
            model = load_qfai(tmp.name, device="cpu")
        source_format = "QFAI"
        action = "encoded_qzs3_from_qfai_state_dict"
    else:
        try:
            model = load_torch_fp4_bytes(renderer_raw, device="cpu")
        except Exception as exc:
            raise ValueError(
                "QZS3 repack requires a JointFrameGenerator renderer.bin in "
                f"QFAI, QZS3, or Torch-FP4 form; got magic {magic!r}"
            ) from exc
        source_format = "Torch-FP4"
        action = "encoded_qzs3_from_pr63_torch_fp4_state_dict"
    qzs3 = encode_qzs3_state_dict(model, block_size=block_size)
    decode_qzs3_state_dict(qzs3, device="cpu")
    return qzs3, {
        "source_renderer_format": source_format,
        "action": action,
        "block_size": int(block_size),
    }


def _qzs4_renderer_bytes(
    renderer_raw: bytes,
    *,
    block_sizes: tuple[int, ...],
) -> tuple[bytes, dict[str, Any]]:
    magic = renderer_raw[:4]
    if magic == b"QZS3":
        state = decode_qzs3_state_dict(renderer_raw, device="cpu")
        source_format = "QZS3"
        action = "qzs4_block_search_from_qzs3_state_dict"
    elif magic == b"QFAI":
        with tempfile.NamedTemporaryFile(suffix=".qfai.bin") as tmp:
            tmp.write(renderer_raw)
            tmp.flush()
            state = load_qfai(tmp.name, device="cpu").state_dict()
        source_format = "QFAI"
        action = "qzs4_block_search_from_qfai_state_dict"
    else:
        try:
            state = load_torch_fp4_bytes(renderer_raw, device="cpu").state_dict()
        except Exception as exc:
            raise ValueError(
                "QZS4 repack requires a JointFrameGenerator renderer.bin in "
                f"QFAI, QZS3, or Torch-FP4 form; got magic {magic!r}"
            ) from exc
        source_format = "Torch-FP4"
        action = "qzs4_block_search_from_pr63_torch_fp4_state_dict"
    payload, search_meta = encode_qzs4_block_search_state_dict(
        state,
        block_sizes=block_sizes,
    )
    decode_qzs3_state_dict(payload, device="cpu")
    return payload, {
        "source_renderer_format": source_format,
        "action": action,
        **search_meta,
    }


def _torch_fp4_renderer_bytes(renderer_raw: bytes) -> tuple[bytes, dict[str, Any]]:
    magic = renderer_raw[:4]
    if magic == b"QFAI":
        with tempfile.NamedTemporaryFile(suffix=".qfai.bin") as tmp:
            tmp.write(renderer_raw)
            tmp.flush()
            model = load_qfai(tmp.name, device="cpu")
        payload = encode_torch_fp4_state_dict(model)
        load_torch_fp4_bytes(payload, device="cpu")
        return payload, {
            "source_renderer_format": "QFAI",
            "action": "encoded_pr63_torch_fp4_from_qfai_state_dict",
        }
    if magic == b"QZS3":
        state = decode_qzs3_state_dict(renderer_raw, device="cpu")
        payload = encode_torch_fp4_state_dict(state)
        load_torch_fp4_bytes(payload, device="cpu")
        return payload, {
            "source_renderer_format": "QZS3",
            "action": "encoded_pr63_torch_fp4_from_qzs3_state_dict",
        }
    try:
        load_torch_fp4_bytes(renderer_raw, device="cpu")
    except Exception as exc:
        raise ValueError(
            "Torch-FP4 repack requires a JointFrameGenerator renderer.bin in "
            f"QFAI, QZS3, or Torch-FP4 form; got magic {magic!r}"
        ) from exc
    return renderer_raw, {
        "source_renderer_format": "Torch-FP4",
        "action": "validated_existing_pr63_torch_fp4",
    }


def _renderer_bytes(
    renderer_raw: bytes,
    renderer_codec: str,
    *,
    qzs3_block_size: int = 32,
    qzs4_block_sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128),
) -> tuple[bytes, dict[str, Any]]:
    if renderer_codec == RENDERER_CODEC_QZS3:
        data, meta = _qzs3_renderer_bytes(renderer_raw, block_size=qzs3_block_size)
    elif renderer_codec == RENDERER_CODEC_QZS4:
        data, meta = _qzs4_renderer_bytes(renderer_raw, block_sizes=qzs4_block_sizes)
    elif renderer_codec == RENDERER_CODEC_TORCH_FP4:
        data, meta = _torch_fp4_renderer_bytes(renderer_raw)
    else:
        raise ValueError(f"unsupported renderer codec: {renderer_codec!r}")
    return data, {"renderer_codec": renderer_codec, **meta}


def build_archive(
    source_archive: Path,
    output_archive: Path,
    *,
    renderer_codec: str = RENDERER_CODEC_QZS3,
    qzs3_block_size: int = 32,
    qzs4_block_sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128),
    preserve_geometry_members: bool = True,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_archive = output_archive.resolve()
    members, source_contract = _runtime_members_from_source_archive(source_archive)

    renderer_payload, renderer_meta = _renderer_bytes(
        members["renderer.bin"],
        renderer_codec,
        qzs3_block_size=qzs3_block_size,
        qzs4_block_sizes=qzs4_block_sizes,
    )
    output_members = {
        "renderer.bin": renderer_payload,
        "masks.mkv": members["masks.mkv"],
        "optimized_poses.bin": members["optimized_poses.bin"],
    }
    source_geometry_members = {
        name: members[name] for name in OPTIONAL_GEOMETRY_MEMBERS if name in members
    }
    if preserve_geometry_members:
        output_members.update(source_geometry_members)
    output_member_order = [
        *RUNTIME_MEMBERS,
        *(name for name in OPTIONAL_GEOMETRY_MEMBERS if name in output_members),
    ]

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in output_member_order:
            zf.writestr(_zip_info(name), output_members[name])

    source_bytes = source_archive.stat().st_size
    output_bytes = output_archive.stat().st_size
    return {
        "schema_version": 1,
        "tool": "repack_quantizr_faithful_qzs3_archive",
        "score_claim": False,
        "evidence_grade": "empirical_byte_only_until_cuda_auth_eval",
        "source_archive": str(source_archive),
        "source_archive_bytes": source_bytes,
        "source_archive_sha256": _sha256_path(source_archive),
        "source_runtime_contract": source_contract,
        "output_archive": str(output_archive),
        "output_archive_bytes": output_bytes,
        "output_archive_sha256": _sha256_path(output_archive),
        "archive_byte_delta": output_bytes - source_bytes,
        "formula_rate_score_delta": 25.0 * float(output_bytes - source_bytes) / ORIGINAL_VIDEO_BYTES,
        "runtime_member_order": list(output_member_order),
        "geometry_preservation": {
            "optional_geometry_member_names": list(OPTIONAL_GEOMETRY_MEMBERS),
            "source_geometry_members": _members_metadata(
                source_geometry_members,
                list(source_geometry_members),
            ),
            "output_geometry_members": _members_metadata(
                output_members,
                list(source_geometry_members),
            ),
            "preserved": all(name in output_members for name in source_geometry_members),
        },
        "submission_path": _submission_path_metadata(
            layout=SUBMISSION_LAYOUT_MULTI_MEMBER,
            output_archive=output_archive,
            pose_codec="raw",
            pose_residual_topk=0,
        ),
        "members": {
            name: {
                "member_name": name,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
            for name, data in output_members.items()
        },
        "renderer": {
            **renderer_meta,
            "source_bytes": len(members["renderer.bin"]),
            "source_sha256": _sha256_bytes(members["renderer.bin"]),
            "output_renderer_bytes": len(renderer_payload),
            "output_renderer_sha256": _sha256_bytes(renderer_payload),
        },
    }


def build_submission_archive(
    source_archive: Path,
    output_archive: Path,
    *,
    renderer_codec: str = RENDERER_CODEC_QZS3,
    qzs3_block_size: int = 32,
    qzs4_block_sizes: tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128),
    submission_layout: str = SUBMISSION_LAYOUT_MULTI_MEMBER,
    pose_codec: str = "raw",
    pose_residual_topk: int = 0,
    brotli_quality: int = 11,
) -> dict[str, Any]:
    if submission_layout == SUBMISSION_LAYOUT_MULTI_MEMBER:
        if pose_codec != "raw" or pose_residual_topk != 0:
            raise ValueError(
                "pose_codec transforms require a single-blob payload layout so "
                "the contest runtime has codec metadata or magic bytes to decode"
            )
        return build_archive(
            source_archive,
            output_archive,
            renderer_codec=renderer_codec,
            qzs3_block_size=qzs3_block_size,
            qzs4_block_sizes=qzs4_block_sizes,
        )
    if submission_layout not in _SINGLE_BLOB_PAYLOAD_FORMATS:
        raise ValueError(f"unsupported submission layout: {submission_layout!r}")
    with tempfile.TemporaryDirectory(prefix="qzs_repack_") as tmp:
        intermediate = Path(tmp) / "runtime_members.zip"
        renderer_meta = build_archive(
            source_archive,
            intermediate,
            renderer_codec=renderer_codec,
            qzs3_block_size=qzs3_block_size,
            qzs4_block_sizes=qzs4_block_sizes,
            preserve_geometry_members=False,
        )
        packed_meta = build_packed_archive(
            intermediate,
            output_archive,
            brotli_quality=brotli_quality,
            pose_codec=pose_codec,
            pose_residual_topk=pose_residual_topk,
            payload_member_name="p",
            payload_format=_SINGLE_BLOB_PAYLOAD_FORMATS[submission_layout],
        )
        source_geometry_members = {}
        source_contract_geometry = renderer_meta.get("geometry_preservation", {}).get(
            "source_geometry_members",
            {},
        )
        if source_contract_geometry:
            runtime_members, _source_contract = _runtime_members_from_source_archive(source_archive.resolve())
            source_geometry_members = {
                name: runtime_members[name]
                for name in OPTIONAL_GEOMETRY_MEMBERS
                if name in runtime_members
            }
        geometry_archive_meta = _rewrite_single_blob_archive_with_geometry(
            output_archive,
            source_geometry_members,
        )
        if geometry_archive_meta:
            packed_meta = {
                **packed_meta,
                "output_archive_before_geometry_bytes": packed_meta["output_archive_bytes"],
                "output_archive_before_geometry_sha256": packed_meta["output_archive_sha256"],
            }
    return {
        "schema_version": 1,
        "tool": "repack_quantizr_faithful_qzs3_archive",
        "score_claim": False,
        "evidence_grade": "empirical_byte_only_until_cuda_auth_eval",
        "source_archive": str(source_archive.resolve()),
        "source_archive_bytes": source_archive.resolve().stat().st_size,
        "source_archive_sha256": _sha256_path(source_archive.resolve()),
        "output_archive": str(output_archive.resolve()),
        "output_archive_bytes": output_archive.resolve().stat().st_size,
        "output_archive_sha256": _sha256_path(output_archive.resolve()),
        "archive_byte_delta": output_archive.resolve().stat().st_size - source_archive.resolve().stat().st_size,
        "formula_rate_score_delta": 25.0
        * float(output_archive.resolve().stat().st_size - source_archive.resolve().stat().st_size)
        / ORIGINAL_VIDEO_BYTES,
        "renderer_stage": renderer_meta,
        "packed_payload_stage": packed_meta,
        "geometry_preservation": {
            "optional_geometry_member_names": list(OPTIONAL_GEOMETRY_MEMBERS),
            "source_geometry_members": renderer_meta.get("geometry_preservation", {}).get(
                "source_geometry_members",
                {},
            ),
            "output_geometry_members": geometry_archive_meta.get("members", {}),
            "preserved": (
                set(renderer_meta.get("geometry_preservation", {}).get("source_geometry_members", {}))
                <= set(geometry_archive_meta.get("members", {}))
            ),
            "archive_member_order": geometry_archive_meta.get("member_order"),
            "preservation_mode": geometry_archive_meta.get("preservation_mode"),
        },
        "submission_path": _submission_path_metadata(
            layout=submission_layout,
            output_archive=output_archive.resolve(),
            pose_codec=pose_codec,
            pose_residual_topk=pose_residual_topk,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, default=None)
    parser.add_argument(
        "--renderer-codec",
        choices=(RENDERER_CODEC_QZS3, RENDERER_CODEC_QZS4, RENDERER_CODEC_TORCH_FP4),
        default=RENDERER_CODEC_QZS3,
        help="JointFrameGenerator renderer codec to write into renderer.bin.",
    )
    parser.add_argument(
        "--qzs3-block-size",
        type=int,
        default=32,
        help="FP4 block size for --renderer-codec qzs3.",
    )
    parser.add_argument(
        "--qzs4-block-sizes",
        default="16,24,32,48,64,96,128",
        help="Comma-separated FP4 block-size candidates for --renderer-codec qzs4.",
    )
    parser.add_argument(
        "--submission-layout",
        choices=(
            SUBMISSION_LAYOUT_MULTI_MEMBER,
            SUBMISSION_LAYOUT_RPK1_SINGLE_BLOB,
            SUBMISSION_LAYOUT_PR64_SINGLE_BLOB,
            SUBMISSION_LAYOUT_PR64_MASK_FIRST_SINGLE_BLOB,
        ),
        default=SUBMISSION_LAYOUT_MULTI_MEMBER,
        help="Archive container shape to screen: normal members or charged single-blob payload.",
    )
    parser.add_argument(
        "--pose-codec",
        choices=(
            "raw",
            POSE_FP16_COL_DELTA_CODEC,
            POSE_QPOSE14_COL_DELTA_CODEC,
            POSE_QP1_CODEC,
            POSE_FP16_VELOCITY_ONLY_CODEC,
            POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
        ),
        default="raw",
        help="Pose codec for single-blob payload layouts.",
    )
    parser.add_argument(
        "--pose-residual-topk",
        type=int,
        default=0,
        help=f"Top-K residual atoms for {POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC}.",
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    args = parser.parse_args()

    output_archive = args.output_archive or (args.output_dir / "archive.zip")
    provenance = build_submission_archive(
        args.source_archive,
        output_archive,
        renderer_codec=args.renderer_codec,
        qzs3_block_size=args.qzs3_block_size,
        qzs4_block_sizes=_parse_block_sizes(args.qzs4_block_sizes),
        submission_layout=args.submission_layout,
        pose_codec=args.pose_codec,
        pose_residual_topk=args.pose_residual_topk,
        brotli_quality=args.brotli_quality,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output_dir / "build_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
