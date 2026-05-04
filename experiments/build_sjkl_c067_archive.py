#!/usr/bin/env python3
"""Build a bounded SJ-KL payload archive from a C067/public-floor archive.

This is an archive-construction slice only.  It injects a charged ``sjkl.bin``
logical runtime member into an existing renderer payload archive and emits
deterministic contest archive bytes.  The result is not score evidence until
the exact archive bytes pass CUDA auth eval through the canonical contest path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"

SCHEMA = "sjkl_c067_archive_builder_v1"
SJKL_MEMBER_NAME = "sjkl.bin"
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_MAX_SJKL_BYTES = 32 * 1024


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKER = _load_module(PACKER_PATH, "_sjkl_c067_packed_payload_builder")
UNPACKER = _load_module(UNPACKER_PATH, "_sjkl_c067_runtime_unpacker")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _default_max_sjkl_bytes() -> int:
    raw = os.environ.get("SJKL_MAX_BYTES")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_SJKL_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"SJKL_MAX_BYTES must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"SJKL_MAX_BYTES must be positive, got {value}")
    return value


def verify_runtime_apply_proof(runtime_path: Path | None = None) -> dict[str, Any]:
    """Verify the inflate runtime can fail if charged ``sjkl.bin`` is skipped."""
    runtime_path = runtime_path or (REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py")
    if not runtime_path.is_file():
        raise ValueError(f"missing SJ-KL runtime apply proof source: {runtime_path}")
    text = runtime_path.read_text()
    required_tokens = (
        "SJKL_REQUIRE_APPLIED",
        "_finalize_sjkl_application_contract",
        "_apply_sjkl_residual_to_pairs",
        "applied_pair_count",
        "charged sjkl.bin did not affect",
    )
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(
            "SJ-KL runtime-apply proof is absent; missing tokens in "
            f"{runtime_path}: {missing}"
        )
    return {
        "path": str(runtime_path.resolve()),
        "sha256": _sha256_file(runtime_path),
        "require_env": "SJKL_REQUIRE_APPLIED=1",
        "proof_tokens": list(required_tokens),
        "verified": True,
    }


def _read_safe_zip_members(source_archive: Path) -> dict[str, bytes]:
    """Read safe top-level members from a source archive."""
    try:
        with zipfile.ZipFile(source_archive, "r") as zf:
            members: dict[str, bytes] = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                PACKER._validate_member_name(info.filename)
                if info.filename.startswith(".") or info.filename.startswith("__MACOSX"):
                    raise ValueError(f"hidden/resource-fork archive member is not allowed: {info.filename!r}")
                if info.filename in members:
                    raise ValueError(f"duplicate source archive member: {info.filename}")
                members[info.filename] = zf.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {source_archive}") from exc
    if not members:
        raise ValueError(f"source archive is empty: {source_archive}")
    return members


def _parse_packed_payload_member(member_name: str, data: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Parse a packed renderer payload member using the runtime parser."""
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
            header, members = UNPACKER._parse_payload(payload)
        except Exception as exc:  # pragma: no cover - only reported on total parse failure
            errors.append(f"{label}: {exc}")
            continue
        return header, members
    raise ValueError(f"could not parse packed renderer payload member {member_name!r}: {'; '.join(errors)}")


def extract_runtime_members(source_archive: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Extract logical runtime members from direct or packed source archive bytes."""
    source_archive = source_archive.resolve()
    zip_members = _read_safe_zip_members(source_archive)
    packed_names = [
        name
        for name in (UNPACKER.PAYLOAD_BIN, UNPACKER.PAYLOAD_BR, UNPACKER.PAYLOAD_SHORT_BR)
        if name in zip_members
    ]
    if "renderer.bin" not in zip_members and packed_names:
        if len(zip_members) != 1:
            raise ValueError(
                "packed source archives must contain only the renderer payload member; "
                f"got {sorted(zip_members)}"
            )
        payload_name = packed_names[0]
        header, members = _parse_packed_payload_member(payload_name, zip_members[payload_name])
        return dict(members), {
            "source_archive_packaging": "packed_renderer_payload",
            "source_zip_members": [payload_name],
            "source_payload_member": payload_name,
            "source_payload_header_schema": header.get("schema"),
            "source_payload_header_payload_format": header.get("payload_format"),
            "source_logical_member_names": list(members),
        }

    return zip_members, {
        "source_archive_packaging": "direct_runtime_members",
        "source_zip_members": list(zip_members),
        "source_payload_member": None,
        "source_payload_header_schema": None,
        "source_payload_header_payload_format": None,
        "source_logical_member_names": list(zip_members),
    }


def _minimal_rpk1_payload(ordered_members: list[tuple[str, bytes]]) -> tuple[bytes, dict[str, Any]]:
    """Build the smallest RPK1 header accepted by the runtime unpacker."""
    header = {
        "schema": PACKER.SCHEMA,
        "members": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
            for name, data in ordered_members
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = (
        PACKER.MAGIC
        + struct.pack(PACKER.HEADER_STRUCT, len(header_bytes))
        + header_bytes
        + b"".join(data for _name, data in ordered_members)
    )
    return payload, header


def _candidate_member_orders(ordered_members: list[tuple[str, bytes]]) -> list[list[tuple[str, bytes]]]:
    """Return deterministic candidate orders for RPK1 compression screening."""
    canonical = list(ordered_members)
    members = dict(ordered_members)
    fixed4 = ("renderer.bin", "masks.mkv", "optimized_poses.bin", SJKL_MEMBER_NAME)
    if set(members) != set(fixed4):
        return [canonical]
    return [
        [(name, members[name]) for name in order]
        for order in itertools.permutations(fixed4)
    ]


def _build_smallest_runtime_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    brotli_quality: int,
) -> tuple[bytes, bytes, dict[str, Any], dict[str, Any]]:
    """Choose the smallest deterministic runtime-compatible RPK1 payload."""
    import brotli

    candidates: list[dict[str, Any]] = []
    best: tuple[int, tuple[str, ...], bytes, bytes, dict[str, Any]] | None = None
    for candidate_order in _candidate_member_orders(ordered_members):
        payload, header = _minimal_rpk1_payload(candidate_order)
        compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
        if brotli.decompress(compressed) != payload:
            raise RuntimeError("Brotli round-trip mismatch for SJ-KL renderer payload")
        order_names = tuple(name for name, _data in candidate_order)
        candidate = {
            "member_order": list(order_names),
            "payload_raw_bytes": len(payload),
            "payload_compressed_bytes": len(compressed),
        }
        candidates.append(candidate)
        key = (len(compressed), order_names)
        if best is None or key < (best[0], best[1]):
            best = (len(compressed), order_names, payload, compressed, header)

    assert best is not None
    _best_size, best_order, best_payload, best_compressed, best_header = best
    order_rank = {name: idx for idx, name in enumerate(best_order)}
    chosen_members = sorted(ordered_members, key=lambda item: order_rank[item[0]])
    selection = {
        "header_minimized": True,
        "member_order_optimized": len(candidates) > 1,
        "chosen_member_order": list(best_order),
        "candidate_count": len(candidates),
        "candidates": sorted(
            candidates,
            key=lambda item: (item["payload_compressed_bytes"], item["member_order"]),
        )[:8],
    }
    # Rebuild the header from chosen_members so manifest order and payload
    # custody come from the same object.
    best_payload, best_header = _minimal_rpk1_payload(chosen_members)
    if len(best_payload) != len(best[2]):
        raise RuntimeError("RPK1 payload rebuild length changed after order selection")
    return best_payload, best_compressed, best_header, selection


def _zip_info(name: str, *, compress_type: int = zipfile.ZIP_STORED) -> zipfile.ZipInfo:
    PACKER._validate_member_name(name)
    info = zipfile.ZipInfo(name, date_time=PACKER.FIXED_ZIP_TIMESTAMP)
    info.compress_type = compress_type
    info.create_system = 3
    info.external_attr = 0o644 << 16
    info.extra = b""
    info.comment = b""
    return info


def _build_top_level_sibling_archive(
    *,
    source_archive: Path,
    zip_members: dict[str, bytes],
    sjkl_bytes: bytes,
    output_archive: Path,
    sjkl_zip_compression: str,
) -> dict[str, Any]:
    """Write ``p``/``renderer_payload`` plus top-level charged ``sjkl.bin``.

    C067/PR67-style archives already ship a compact fixed-slice payload in the
    top-level ``p`` member. Rewrapping that payload into generic RPK1 JSON costs
    hundreds of bytes before scorer benefit is considered. This sibling layout
    preserves the source payload bytes exactly and charges only the SJ-KL member
    plus unavoidable ZIP metadata.
    """
    packed_names = [
        name
        for name in (UNPACKER.PAYLOAD_SHORT_BR, UNPACKER.PAYLOAD_BR, UNPACKER.PAYLOAD_BIN)
        if name in zip_members
    ]
    if len(packed_names) != 1 or len(zip_members) != 1:
        raise ValueError(
            "top_level_sibling layout requires a source archive with exactly one "
            f"packed renderer payload member; got {sorted(zip_members)}"
        )
    payload_name = packed_names[0]
    if sjkl_zip_compression == "stored":
        sjkl_compress_type = zipfile.ZIP_STORED
        sjkl_compresslevel = None
    elif sjkl_zip_compression == "deflated":
        sjkl_compress_type = zipfile.ZIP_DEFLATED
        sjkl_compresslevel = 9
    else:
        raise ValueError(
            "sjkl_zip_compression must be 'stored' or 'deflated', "
            f"got {sjkl_zip_compression!r}"
        )

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_archive, "w") as zf:
        zf.writestr(_zip_info(payload_name, compress_type=zipfile.ZIP_STORED), zip_members[payload_name])
        kwargs: dict[str, Any] = {"compress_type": sjkl_compress_type}
        if sjkl_compresslevel is not None:
            kwargs["compresslevel"] = sjkl_compresslevel
        zf.writestr(_zip_info(SJKL_MEMBER_NAME, compress_type=sjkl_compress_type), sjkl_bytes, **kwargs)

    with zipfile.ZipFile(output_archive, "r") as zf:
        infos = zf.infolist()
        if [info.filename for info in infos] != [payload_name, SJKL_MEMBER_NAME]:
            raise RuntimeError("unexpected top-level sibling archive member order")
        if zf.read(payload_name) != zip_members[payload_name]:
            raise RuntimeError("source packed payload bytes changed in sibling archive")
        if zf.read(SJKL_MEMBER_NAME) != sjkl_bytes:
            raise RuntimeError("sjkl.bin bytes changed in sibling archive")
        zip_summary = [
            {
                "name": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "date_time": list(info.date_time),
                "external_attr_mode": info.external_attr >> 16,
            }
            for info in infos
        ]

    source_size = source_archive.stat().st_size
    archive_bytes = output_archive.stat().st_size
    return {
        "layout": "top_level_sjkl_sibling",
        "source_payload_member": payload_name,
        "preserves_source_payload_bytes": True,
        "sjkl_zip_compression": sjkl_zip_compression,
        "archive_delta_bytes_vs_source_archive": archive_bytes - source_size,
        "formula_only_rate_delta_vs_source_archive": 25.0 * (archive_bytes - source_size) / ORIGINAL_VIDEO_BYTES,
        "zip_members": zip_summary,
    }


def build_sjkl_archive(
    *,
    source_archive: Path,
    sjkl_bin: Path,
    output_dir: Path,
    brotli_quality: int = 11,
    payload_member_name: str = "p",
    archive_layout: str = "repack_rpk1",
    sjkl_zip_compression: str = "stored",
    force: bool = False,
    replace_existing_sjkl: bool = False,
    max_sjkl_bytes: int | None = None,
    require_runtime_apply_proof: bool = True,
) -> dict[str, Any]:
    """Inject ``sjkl.bin`` and write deterministic packed archive bytes."""
    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    if payload_member_name not in PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES:
        raise ValueError(
            f"payload_member_name must be one of {PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES}, "
            f"got {payload_member_name!r}"
        )
    if archive_layout not in {"repack_rpk1", "top_level_sibling"}:
        raise ValueError(
            "archive_layout must be 'repack_rpk1' or 'top_level_sibling', "
            f"got {archive_layout!r}"
        )

    source_archive = source_archive.resolve()
    sjkl_bin = sjkl_bin.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = source_archive.read_bytes()
    sjkl_bytes = sjkl_bin.read_bytes()
    if not sjkl_bytes:
        raise ValueError(f"sjkl payload is empty: {sjkl_bin}")
    max_sjkl_bytes = _default_max_sjkl_bytes() if max_sjkl_bytes is None else int(max_sjkl_bytes)
    if max_sjkl_bytes <= 0:
        raise ValueError(f"max_sjkl_bytes must be positive, got {max_sjkl_bytes}")
    if len(sjkl_bytes) > max_sjkl_bytes:
        raise ValueError(
            f"sjkl payload exceeds byte cap: {len(sjkl_bytes)} > {max_sjkl_bytes}; "
            "set SJKL_MAX_BYTES or --max-sjkl-bytes only for an explicit diagnostic"
        )
    runtime_apply_proof = verify_runtime_apply_proof() if require_runtime_apply_proof else {
        "verified": False,
        "disabled_by_explicit_flag": True,
    }

    zip_members = _read_safe_zip_members(source_archive)
    runtime_members, source_packaging = extract_runtime_members(source_archive)
    existing_sjkl = runtime_members.get(SJKL_MEMBER_NAME)
    if existing_sjkl is not None and existing_sjkl != sjkl_bytes and not replace_existing_sjkl:
        raise ValueError(
            "source archive already contains a different sjkl.bin; "
            "pass --replace-existing-sjkl to replace it explicitly"
        )
    runtime_members[SJKL_MEMBER_NAME] = sjkl_bytes

    archive_path = output_dir / "archive.zip"
    if archive_layout == "top_level_sibling":
        payload_header = {
            "schema": "source_payload_preserved_plus_top_level_sjkl_v1",
            "members": [
                {
                    "name": name,
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                    "codec": "preserved_source_member",
                }
                for name, data in sorted(zip_members.items())
            ]
            + [
                {
                    "name": SJKL_MEMBER_NAME,
                    "bytes": len(sjkl_bytes),
                    "sha256": _sha256_bytes(sjkl_bytes),
                    "codec": "top_level_archive_member",
                }
            ],
        }
        sibling_summary = _build_top_level_sibling_archive(
            source_archive=source_archive,
            zip_members=zip_members,
            sjkl_bytes=sjkl_bytes,
            output_archive=archive_path,
            sjkl_zip_compression=sjkl_zip_compression,
        )
        payload = b""
        compressed = b""
        payload_selection = {
            "header_minimized": False,
            "member_order_optimized": False,
            "candidate_count": 1,
            "chosen_member_order": [sibling_summary["source_payload_member"], SJKL_MEMBER_NAME],
            "layout_summary": sibling_summary,
        }
        output_archive_members = [sibling_summary["source_payload_member"], SJKL_MEMBER_NAME]
        payload_member_for_manifest = sibling_summary["source_payload_member"]
    else:
        ordered = PACKER.ordered_runtime_members(runtime_members)
        payload, compressed, payload_header, payload_selection = _build_smallest_runtime_payload(
            ordered,
            brotli_quality=brotli_quality,
        )
        PACKER.write_deterministic_payload_archive(
            archive_path,
            compressed,
            payload_member_name=payload_member_name,
        )
        output_archive_members = [payload_member_name]
        payload_member_for_manifest = payload_member_name

    archive_bytes = archive_path.stat().st_size
    source_size = len(source_bytes)
    delta_bytes = archive_bytes - source_size
    if archive_layout == "top_level_sibling":
        logical_member_names = list(runtime_members)
    else:
        logical_member_names = [member["name"] for member in payload_header["members"]]
    manifest = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_size,
            "sha256": _sha256_bytes(source_bytes),
            **source_packaging,
        },
        "sjkl_payload": {
            "path": str(sjkl_bin),
            "member_name": SJKL_MEMBER_NAME,
            "bytes": len(sjkl_bytes),
            "sha256": _sha256_bytes(sjkl_bytes),
            "max_bytes": max_sjkl_bytes,
            "replaced_existing": bool(existing_sjkl is not None and existing_sjkl != sjkl_bytes),
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_bytes,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_source_archive": delta_bytes,
            "formula_only_rate_delta_vs_source_archive": 25.0 * delta_bytes / ORIGINAL_VIDEO_BYTES,
        },
        "payload_member_names": {
            "output_archive_members": output_archive_members,
            "output_packed_payload_member": payload_member_for_manifest,
            "output_logical_runtime_members": logical_member_names,
            "sjkl_member": SJKL_MEMBER_NAME,
        },
        "packed_payload": {
            "archive_layout": archive_layout,
            "payload_format": PACKER.PAYLOAD_FORMAT_RPK1_JSON
            if archive_layout == "repack_rpk1"
            else "preserve_source_payload_plus_top_level_sjkl",
            "payload_member": payload_member_for_manifest,
            "payload_raw_bytes": len(payload),
            "payload_compressed_bytes": len(compressed),
            "brotli_quality": brotli_quality,
            "header": payload_header,
            "source_archive_sha256_recorded_in_manifest": _sha256_bytes(source_bytes),
            "compression_selection": payload_selection,
        },
        "runtime_contract": {
            "unpacker": "submissions/robust_current/unpack_renderer_payload.py",
            "inflate_loader": "submissions/robust_current/inflate_renderer.py::_load_sjkl_residual_from_archive_dir",
            "sidecars_required": False,
            "score_affecting_payload_charged_in_archive": True,
            "require_runtime_apply_proof": bool(require_runtime_apply_proof),
            "runtime_apply_proof": runtime_apply_proof,
        },
    }
    (output_dir / "sjkl_c067_archive_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--sjkl-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--payload-member-name",
        choices=PACKER.ALLOWED_PAYLOAD_MEMBER_NAMES,
        default=PACKER.SHORT_PAYLOAD_MEMBER_NAME,
    )
    parser.add_argument(
        "--archive-layout",
        choices=("repack_rpk1", "top_level_sibling"),
        default="repack_rpk1",
        help=(
            "repack_rpk1 rewrites logical runtime members into one generic packed "
            "payload. top_level_sibling preserves an existing single packed payload "
            "member and adds charged sjkl.bin as a top-level archive member."
        ),
    )
    parser.add_argument(
        "--sjkl-zip-compression",
        choices=("stored", "deflated"),
        default="stored",
        help="ZIP method for top-level sibling sjkl.bin; ignored by repack_rpk1.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--replace-existing-sjkl", action="store_true")
    parser.add_argument(
        "--max-sjkl-bytes",
        type=int,
        default=None,
        help=(
            "Fail closed if sjkl.bin exceeds this many bytes. Defaults to "
            "SJKL_MAX_BYTES or 32768."
        ),
    )
    parser.add_argument(
        "--allow-missing-runtime-apply-proof",
        action="store_true",
        help="Explicit diagnostic escape hatch; output remains non-promotable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_sjkl_archive(
        source_archive=args.source_archive,
        sjkl_bin=args.sjkl_bin,
        output_dir=args.output_dir,
        brotli_quality=args.brotli_quality,
        payload_member_name=args.payload_member_name,
        archive_layout=args.archive_layout,
        sjkl_zip_compression=args.sjkl_zip_compression,
        force=args.force,
        replace_existing_sjkl=args.replace_existing_sjkl,
        max_sjkl_bytes=args.max_sjkl_bytes,
        require_runtime_apply_proof=not args.allow_missing_runtime_apply_proof,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
