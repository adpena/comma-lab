#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Losslessly repack a single-member Brotli payload archive.

This is a byte-only transform for public-floor-style archives whose only ZIP
member is a Brotli blob named ``p``. It never changes the decompressed payload;
it only searches or applies deterministic Brotli container parameters, writes a
stored ZIP with fixed metadata, and emits custody JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import brotli


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_MEMBER = "p"
PRODUCER = "experiments/repack_single_payload_brotli.py"
SCHEMA_VERSION = 1
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
MODE_NAMES = {
    0: "generic",
    1: "text",
    2: "font",
}


class RepackError(ValueError):
    """Raised when the archive is not a safe single-member Brotli payload."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_single_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise RepackError(f"unsafe archive member path: {name!r}")
    parts = Path(name).parts
    if len(parts) != 1 or any(part in {"", ".", ".."} for part in parts):
        raise RepackError(f"unsafe archive member path: {name!r}")
    if name.startswith(".") or name == "__MACOSX":
        raise RepackError(f"hidden/system archive member path: {name!r}")
    return name


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_single_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def read_single_member_payload(archive: Path, *, member_name: str = DEFAULT_MEMBER) -> bytes:
    """Read the exact single member from a deterministic one-blob archive."""
    member_name = _safe_single_member_name(member_name)
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member_name]:
            raise RepackError(f"expected single member {member_name!r}; got {names!r}")
        return zf.read(infos[0])


def write_single_member_archive(path: Path, *, member_name: str, payload: bytes) -> None:
    """Write a deterministic stored-ZIP archive containing exactly one member."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(member_name), payload)


def _compress(raw: bytes, *, quality: int, mode: int, lgwin: int, lgblock: int) -> bytes:
    kwargs: dict[str, int] = {
        "quality": quality,
        "mode": mode,
        "lgwin": lgwin,
    }
    # The python-brotli API uses 0 as the explicit "auto" block size; record it
    # rather than omitting it so the archive can be reproduced exactly.
    kwargs["lgblock"] = lgblock
    return brotli.compress(raw, **kwargs)


def repack_archive(
    *,
    source_archive: Path,
    output_archive: Path,
    manifest_json: Path,
    member_name: str = DEFAULT_MEMBER,
    quality: int = 11,
    mode: int = 2,
    lgwin: int = 18,
    lgblock: int = 0,
    require_improvement: bool = True,
) -> dict[str, Any]:
    """Losslessly recompress a single-member archive and write custody metadata."""
    if quality < 0 or quality > 11:
        raise RepackError(f"quality out of Brotli range: {quality}")
    if mode not in MODE_NAMES:
        raise RepackError(f"unsupported Brotli mode {mode}; expected one of {sorted(MODE_NAMES)}")
    if lgwin < 10 or lgwin > 24:
        raise RepackError(f"lgwin out of Brotli range: {lgwin}")
    if lgblock < 0 or lgblock > 24:
        raise RepackError(f"lgblock out of Brotli range: {lgblock}")

    source_archive = source_archive.resolve()
    output_archive = output_archive.resolve()
    manifest_json = manifest_json.resolve()

    source_payload = read_single_member_payload(source_archive, member_name=member_name)
    try:
        raw_payload = brotli.decompress(source_payload)
    except brotli.error as exc:
        raise RepackError(
            f"archive member {member_name!r} is not a Brotli-compressed payload"
        ) from exc
    repacked_payload = _compress(
        raw_payload,
        quality=quality,
        mode=mode,
        lgwin=lgwin,
        lgblock=lgblock,
    )
    if brotli.decompress(repacked_payload) != raw_payload:
        raise RepackError("repacked Brotli payload does not round-trip to source raw payload")
    if require_improvement and len(repacked_payload) >= len(source_payload):
        raise RepackError(
            "repack did not improve payload size: "
            f"source={len(source_payload)} repacked={len(repacked_payload)}"
        )

    write_single_member_archive(output_archive, member_name=member_name, payload=repacked_payload)
    archive_bytes = output_archive.stat().st_size
    source_archive_bytes = source_archive.stat().st_size
    payload_delta = len(repacked_payload) - len(source_payload)
    archive_delta = archive_bytes - source_archive_bytes
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_lossless_byte_transform",
        "required_score_truth": "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda",
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive_bytes,
            "sha256": _sha256_file(source_archive),
        },
        "output_archive": {
            "path": str(output_archive),
            "bytes": archive_bytes,
            "sha256": _sha256_file(output_archive),
        },
        "member_name": member_name,
        "source_payload": {
            "bytes": len(source_payload),
            "sha256": _sha256_bytes(source_payload),
        },
        "raw_payload": {
            "bytes": len(raw_payload),
            "sha256": _sha256_bytes(raw_payload),
        },
        "repacked_payload": {
            "bytes": len(repacked_payload),
            "sha256": _sha256_bytes(repacked_payload),
        },
        "brotli_params": {
            "quality": quality,
            "mode": mode,
            "mode_name": MODE_NAMES[mode],
            "lgwin": lgwin,
            "lgblock": lgblock,
        },
        "lossless_payload_roundtrip": True,
        "source_payload_decompresses": True,
        "payload_delta_bytes": payload_delta,
        "archive_delta_bytes": archive_delta,
        "formula_only_rate_score_delta": archive_delta * RATE_SCORE_PER_BYTE,
        "determinism": {
            "zip_compress_type": "ZIP_STORED",
            "zip_timestamp": list(FIXED_ZIP_TIMESTAMP),
            "zip_permissions": "0644",
            "single_member_order": [member_name],
        },
        "non_promotable_warning": (
            "This is a lossless byte transform only. It cannot move the "
            "frontier until exact CUDA auth eval validates the closed archive."
        ),
    }
    if not math.isfinite(float(manifest["formula_only_rate_score_delta"])):
        raise RepackError("non-finite rate score delta")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--member-name", default=DEFAULT_MEMBER)
    parser.add_argument("--quality", type=int, default=11)
    parser.add_argument("--mode", type=int, default=2)
    parser.add_argument("--lgwin", type=int, default=18)
    parser.add_argument("--lgblock", type=int, default=0)
    parser.add_argument(
        "--allow-non-improvement",
        action="store_true",
        help="Write the archive even when the Brotli parameter set is not smaller.",
    )
    args = parser.parse_args(argv)
    manifest = repack_archive(
        source_archive=args.source_archive,
        output_archive=args.output_archive,
        manifest_json=args.manifest_json,
        member_name=args.member_name,
        quality=args.quality,
        mode=args.mode,
        lgwin=args.lgwin,
        lgblock=args.lgblock,
        require_improvement=not args.allow_non_improvement,
    )
    print(json.dumps({
        "output_archive": manifest["output_archive"],
        "archive_delta_bytes": manifest["archive_delta_bytes"],
        "formula_only_rate_score_delta": manifest["formula_only_rate_score_delta"],
        "promotion_eligible": manifest["promotion_eligible"],
        "score_claim": manifest["score_claim"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
