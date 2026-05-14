#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local QBF1 block-FP renderer transplants from a C067 archive.

This is a byte-screen builder only. It decodes the logical ``renderer.bin``
from an existing C067/public-floor packed archive, repacks the
JointFrameGenerator state as QBF1 block-FP bytes, and writes deterministic
single-member archive candidates. The output is not score evidence until the
exact bytes pass CUDA auth eval through the canonical contest path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"

SCHEMA = "blockfp_c067_archive_builder_v1"
SUMMARY_SCHEMA = "blockfp_c067_archive_summary_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
RENDERER_MEMBER_NAME = "renderer.bin"
POSE_MEMBER_NAME = "optimized_poses.bin"
MASK_MEMBER_NAME = "masks.mkv"
SUPPORTED_RENDERER_MAGICS = (b"QZS3", b"MQZ1", b"QBF1")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKER = _load_module(PACKER_PATH, "_blockfp_c067_packed_payload_builder")
UNPACKER = _load_module(UNPACKER_PATH, "_blockfp_c067_runtime_unpacker")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def parse_block_sizes(value: str) -> tuple[int, ...]:
    """Parse a comma-separated block-size list for argparse and tests."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("block size list must not be empty")
    out: list[int] = []
    for item in items:
        try:
            block_size = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid block size {item!r} in {value!r}"
            ) from exc
        if block_size <= 0 or block_size > 4096:
            raise argparse.ArgumentTypeError(
                f"block size must be in [1, 4096], got {block_size}"
            )
        if block_size not in out:
            out.append(block_size)
    return tuple(out)


def _read_safe_zip_members(source_archive: Path) -> dict[str, bytes]:
    """Read safe top-level members from a source ZIP archive."""

    try:
        with zipfile.ZipFile(source_archive, "r") as zf:
            members: dict[str, bytes] = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                PACKER._validate_member_name(info.filename)
                if info.filename.startswith(".") or info.filename.startswith("__MACOSX"):
                    raise ValueError(
                        "hidden/resource-fork archive member is not allowed: "
                        f"{info.filename!r}"
                    )
                if info.filename in members:
                    raise ValueError(f"duplicate source archive member: {info.filename}")
                members[info.filename] = zf.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {source_archive}") from exc
    if not members:
        raise ValueError(f"source archive is empty: {source_archive}")
    return members


def _parse_packed_payload_member(
    member_name: str,
    data: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Parse packed renderer payload bytes through the runtime parser."""

    candidates: list[tuple[str, bytes]] = []
    if member_name == UNPACKER.PAYLOAD_BIN:
        candidates.append((member_name, data))
    elif member_name in {UNPACKER.PAYLOAD_BR, UNPACKER.PAYLOAD_SHORT_BR}:
        try:
            import brotli
        except ImportError as exc:
            raise RuntimeError(f"{member_name} requires brotli to parse") from exc
        try:
            candidates.append((f"{member_name}:brotli", brotli.decompress(data)))
        except brotli.error:
            if member_name != UNPACKER.PAYLOAD_SHORT_BR:
                raise
        if member_name == UNPACKER.PAYLOAD_SHORT_BR:
            candidates.append((member_name, data))
    else:
        raise ValueError(f"{member_name!r} is not a packed renderer payload member")

    errors: list[str] = []
    for label, payload in candidates:
        try:
            return UNPACKER._parse_payload(payload)
        except Exception as exc:  # pragma: no cover - only surfaced on total failure
            errors.append(f"{label}: {exc}")
    raise ValueError(
        f"could not parse packed renderer payload member {member_name!r}: "
        f"{'; '.join(errors)}"
    )


def extract_runtime_members(source_archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Extract logical runtime members from direct or packed source archives."""

    source_archive = source_archive.resolve()
    zip_members = _read_safe_zip_members(source_archive)
    packed_names = [
        name
        for name in (
            UNPACKER.PAYLOAD_BIN,
            UNPACKER.PAYLOAD_BR,
            UNPACKER.PAYLOAD_SHORT_BR,
        )
        if name in zip_members
    ]
    if RENDERER_MEMBER_NAME not in zip_members and packed_names:
        if len(zip_members) != 1:
            raise ValueError(
                "packed source archives must contain only the renderer payload "
                f"member; got {sorted(zip_members)}"
            )
        payload_name = packed_names[0]
        header, members = _parse_packed_payload_member(payload_name, zip_members[payload_name])
        return dict(members), {
            "source_archive_packaging": "packed_renderer_payload",
            "source_zip_members": [payload_name],
            "source_payload_member": payload_name,
            "source_payload_member_bytes": len(zip_members[payload_name]),
            "source_payload_member_sha256": _sha256_bytes(zip_members[payload_name]),
            "source_payload_header_schema": header.get("schema"),
            "source_payload_header_payload_format": header.get("payload_format"),
            "source_logical_member_names": list(members),
        }

    return zip_members, {
        "source_archive_packaging": "direct_runtime_members",
        "source_zip_members": list(zip_members),
        "source_payload_member": None,
        "source_payload_member_bytes": None,
        "source_payload_member_sha256": None,
        "source_payload_header_schema": None,
        "source_payload_header_payload_format": None,
        "source_logical_member_names": list(zip_members),
    }


def _renderer_wire_format(renderer_bytes: bytes) -> str:
    magic = renderer_bytes[:4]
    if magic == b"QZS3":
        return "QZS3"
    if magic == b"MQZ1":
        return "MQZ1"
    if magic == b"QBF1":
        return "QBF1"
    return f"unsupported_magic_{magic!r}"


def _decode_joint_frame_state(renderer_bytes: bytes) -> dict[str, Any]:
    """Decode supported JointFrameGenerator renderer bytes to a state dict."""

    magic = renderer_bytes[:4]
    if magic not in SUPPORTED_RENDERER_MAGICS:
        raise ValueError(
            "Block-FP C067 builder only supports JointFrameGenerator renderer "
            "payloads with QZS3, MQZ1, or QBF1 magic; got "
            f"{magic!r}. This is a structural blocker, not score evidence."
        )
    if magic == b"QZS3":
        from tac.quantizr_qzs3_codec import decode_qzs3_state_dict

        return decode_qzs3_state_dict(renderer_bytes, device="cpu")
    if magic == b"MQZ1":
        from tac.quantizr_qzs3_codec import decode_mixed_qzs_block_state_dict

        return decode_mixed_qzs_block_state_dict(renderer_bytes, device="cpu")

    from tac.qbf1_renderer_codec import decode_qbf1_state_dict

    return decode_qbf1_state_dict(renderer_bytes, device="cpu")


def _ordered_three_member_payload(members: dict[str, bytes]) -> list[tuple[str, bytes]]:
    required = (RENDERER_MEMBER_NAME, MASK_MEMBER_NAME, POSE_MEMBER_NAME)
    missing = [name for name in required if name not in members]
    if missing:
        raise ValueError(f"C067 block-FP transplant requires members {required}; missing={missing}")
    extras = sorted(name for name in members if name not in required)
    if extras:
        raise ValueError(
            "public PR64 mask-first output is scoped to renderer/mask/pose "
            f"members only; extras={extras}. Use RPK1 after reviewing the "
            "additional runtime contract."
        )
    return [(name, members[name]) for name in required]


def _build_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    source_archive_sha256: str,
    payload_format: str,
) -> tuple[bytes, dict[str, Any]]:
    if payload_format == PACKER.PAYLOAD_FORMAT_RPK1_JSON:
        return PACKER.build_renderer_payload(
            ordered_members,
            source_archive_sha256=source_archive_sha256,
            pose_codec="raw",
        )
    if payload_format == PACKER.PAYLOAD_FORMAT_RP2_FIXED3:
        return PACKER.build_compact_renderer_payload(
            ordered_members,
            source_archive_sha256=source_archive_sha256,
            pose_codec="raw",
        )
    if payload_format == PACKER.PAYLOAD_FORMAT_PR64_LEN_TABLE:
        return PACKER.build_pr64_len_table_payload(
            ordered_members,
            source_archive_sha256=source_archive_sha256,
            pose_codec="raw",
        )
    if payload_format == PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE:
        return PACKER.build_public_pr64_mask_first_len_table_payload(
            ordered_members,
            source_archive_sha256=source_archive_sha256,
            pose_codec="raw",
        )
    raise ValueError(f"unsupported payload format: {payload_format!r}")


def _verify_output_archive(
    archive_path: Path,
    *,
    payload_member_name: str,
    expected_renderer: bytes,
) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if names != [payload_member_name]:
            raise ValueError(
                f"output archive must contain only {payload_member_name!r}; got {names}"
            )
        payload_member = zf.read(payload_member_name)
    header, members = _parse_packed_payload_member(payload_member_name, payload_member)
    renderer = members.get(RENDERER_MEMBER_NAME)
    if renderer != expected_renderer:
        raise ValueError("runtime unpacker did not reconstruct transformed renderer bytes")
    return {
        "output_archive_members": names,
        "runtime_parse_header_schema": header.get("schema"),
        "runtime_parse_payload_format": header.get("payload_format"),
        "runtime_parse_logical_members": list(members),
    }


def build_blockfp_c067_archives(
    *,
    source_archive: Path,
    output_dir: Path,
    block_sizes: tuple[int, ...] = (1024,),
    brotli_quality: int = 11,
    payload_member_name: str = PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    payload_format: str = PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    force: bool = False,
) -> dict[str, Any]:
    """Build deterministic QBF1 block-FP archive candidates and manifests."""

    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    if payload_member_name not in PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES:
        raise ValueError(
            f"payload_member_name must be one of {PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES}, "
            f"got {payload_member_name!r}"
        )
    if payload_format not in PACKER.ALLOWED_PAYLOAD_FORMATS:
        raise ValueError(
            f"payload_format must be one of {PACKER.ALLOWED_PAYLOAD_FORMATS}, "
            f"got {payload_format!r}"
        )
    block_sizes = parse_block_sizes(",".join(str(item) for item in block_sizes))

    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"output directory is non-empty; pass --force to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_archive.read_bytes()
    source_sha = _sha256_bytes(source_bytes)
    runtime_members, source_packaging = extract_runtime_members(source_archive)
    if RENDERER_MEMBER_NAME not in runtime_members:
        raise ValueError("source archive does not contain logical renderer.bin")
    source_renderer = runtime_members[RENDERER_MEMBER_NAME]
    source_renderer_sha = _sha256_bytes(source_renderer)
    state = _decode_joint_frame_state(source_renderer)

    import brotli
    from tac.qbf1_renderer_codec import pack_qbf1_state_dict, qbf1_byte_accounting

    manifests: list[dict[str, Any]] = []
    for block_size in block_sizes:
        transformed_renderer = pack_qbf1_state_dict(state, block_size=block_size)
        accounting = qbf1_byte_accounting(transformed_renderer)
        candidate_members = dict(runtime_members)
        candidate_members[RENDERER_MEMBER_NAME] = transformed_renderer
        ordered = _ordered_three_member_payload(candidate_members)
        payload, payload_header = _build_payload(
            ordered,
            source_archive_sha256=source_sha,
            payload_format=payload_format,
        )
        compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
        if brotli.decompress(compressed) != payload:
            raise RuntimeError("Brotli round-trip mismatch for block-FP payload")

        candidate_id = f"qbf1_b{block_size:04d}"
        candidate_dir = output_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        archive_path = candidate_dir / "archive.zip"
        PACKER.write_deterministic_payload_archive(
            archive_path,
            compressed,
            payload_member_name=payload_member_name,
        )
        runtime_unpack_check = _verify_output_archive(
            archive_path,
            payload_member_name=payload_member_name,
            expected_renderer=transformed_renderer,
        )

        archive_bytes = archive_path.stat().st_size
        delta_bytes = archive_bytes - len(source_bytes)
        renderer_delta_bytes = len(transformed_renderer) - len(source_renderer)
        manifest = {
            "schema": SCHEMA,
            "candidate_id": candidate_id,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
            "canonical_score_source_required": (
                "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                "experiments/contest_auth_eval.py --device cuda"
            ),
            "source_archive": {
                "path": str(source_archive),
                "bytes": len(source_bytes),
                "sha256": source_sha,
                **source_packaging,
            },
            "renderer_source_member": {
                "member_name": RENDERER_MEMBER_NAME,
                "wire_format": _renderer_wire_format(source_renderer),
                "bytes": len(source_renderer),
                "sha256": source_renderer_sha,
                "state_tensor_count": len(state),
            },
            "transformed_renderer_payload": {
                "member_name": RENDERER_MEMBER_NAME,
                "wire_format": "QBF1",
                "transform": "qzs3_or_mqz1_or_qbf1_decoded_state_to_qbf1_block_fp",
                "block_size": block_size,
                "bytes": len(transformed_renderer),
                "sha256": _sha256_bytes(transformed_renderer),
                "delta_bytes_vs_source_renderer": renderer_delta_bytes,
                "byte_accounting": {
                    "header_nbytes": accounting.header_nbytes,
                    "metadata_nbytes": accounting.metadata_nbytes,
                    "payload_nbytes": accounting.payload_nbytes,
                    "tensor_payload_nbytes": accounting.tensor_payload_nbytes,
                    "total_nbytes": accounting.total_nbytes,
                },
            },
            "output_archive": {
                "path": str(archive_path),
                "bytes": archive_bytes,
                "sha256": _sha256_file(archive_path),
                "delta_bytes_vs_source_archive": delta_bytes,
                "formula_only_rate_delta_vs_source_archive": (
                    25.0 * delta_bytes / ORIGINAL_VIDEO_BYTES
                ),
            },
            "packed_payload": {
                "payload_format": payload_format,
                "payload_member": payload_member_name,
                "payload_raw_bytes": len(payload),
                "payload_compressed_bytes": len(compressed),
                "brotli_quality": brotli_quality,
                "header": payload_header,
                **runtime_unpack_check,
            },
            "runtime_contract": {
                "unpacker": "submissions/robust_current/unpack_renderer_payload.py",
                "inflate_loader": (
                    "submissions/robust_current/inflate_renderer.py::_load_renderer "
                    "QBF1 branch"
                ),
                "renderer_loader_import": "tac.qbf1_renderer_codec.load_qbf1",
                "scorer_imports_at_inflate_time": False,
                "sidecars_required": False,
                "score_affecting_payload_charged_in_archive": True,
            },
            "byte_screen": {
                "local_archive_byte_win": delta_bytes < 0,
                "local_renderer_byte_win": renderer_delta_bytes < 0,
                "score_claim": False,
                "promotion_eligible": False,
                "exact_cuda_auth_eval_required_for_score": True,
            },
        }
        (candidate_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
        manifests.append(manifest)

    best = min(manifests, key=lambda item: item["output_archive"]["bytes"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": len(source_bytes),
            "sha256": source_sha,
        },
        "block_sizes": list(block_sizes),
        "best_by_output_archive_bytes": {
            "candidate_id": best["candidate_id"],
            "bytes": best["output_archive"]["bytes"],
            "sha256": best["output_archive"]["sha256"],
            "delta_bytes_vs_source_archive": best["output_archive"][
                "delta_bytes_vs_source_archive"
            ],
            "formula_only_rate_delta_vs_source_archive": best["output_archive"][
                "formula_only_rate_delta_vs_source_archive"
            ],
        },
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "archive_path": item["output_archive"]["path"],
                "archive_bytes": item["output_archive"]["bytes"],
                "archive_sha256": item["output_archive"]["sha256"],
                "renderer_bytes": item["transformed_renderer_payload"]["bytes"],
                "renderer_sha256": item["transformed_renderer_payload"]["sha256"],
                "delta_bytes_vs_source_archive": item["output_archive"][
                    "delta_bytes_vs_source_archive"
                ],
                "local_archive_byte_win": item["byte_screen"]["local_archive_byte_win"],
                "manifest_path": str(
                    output_dir / item["candidate_id"] / "build_manifest.json"
                ),
            }
            for item in manifests
        ],
    }
    (output_dir / "blockfp_c067_summary.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-sizes", type=parse_block_sizes, default=(1024,))
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--payload-member-name",
        choices=PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES,
        default=PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    )
    parser.add_argument(
        "--payload-format",
        choices=PACKER.ALLOWED_PAYLOAD_FORMATS,
        default=PACKER.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_blockfp_c067_archives(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        block_sizes=args.block_sizes,
        brotli_quality=args.brotli_quality,
        payload_member_name=args.payload_member_name,
        payload_format=args.payload_format,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
