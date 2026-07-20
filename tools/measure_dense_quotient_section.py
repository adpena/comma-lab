#!/usr/bin/env python3
"""ZIP a dense quotient field with fixed metadata and emit runtime-bound custody.

This measures one formulation only: storing the explicit float32 spatial field
consumed by ``pdw2_spatial_receiver``.  It is a section-rate receipt, not a
contest archive or through-R score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

SCHEMA = "dense_quotient_field_zip_measurement.v1"
MEMBER_NAME = "quotient_features.f32.npy"
CHUNK_BYTES = 8 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"receipt overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"stale receipt temporary requires review: {partial}")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        partial.write_text(payload)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_custody(*, command_argv: tuple[str, ...]) -> dict:
    if not command_argv or any(not isinstance(value, str) or not value for value in command_argv):
        raise ValueError("command_argv must contain non-empty strings")
    repo = Path(__file__).resolve().parents[1]
    executable = Path(sys.executable).resolve(strict=True)
    zipfile_source = Path(zipfile.__file__).resolve(strict=True)
    zlib_file = getattr(zlib, "__file__", None)
    zlib_binary = None if zlib_file is None else Path(zlib_file).resolve(strict=True)
    return {
        "command_argv": list(command_argv),
        "git_head": _git_head(repo),
        "producer_tool_path": str(Path(__file__).resolve()),
        "producer_tool_sha256": sha256_file(Path(__file__).resolve()),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_path": str(executable),
            "executable_sha256": sha256_file(executable),
            "zipfile_path": str(zipfile_source),
            "zipfile_sha256": sha256_file(zipfile_source),
        },
        "zlib": {
            "compiled_version": zlib.ZLIB_VERSION,
            "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "linkage": "built-in" if zlib_binary is None else "extension-module",
            "binary_path": None if zlib_binary is None else str(zlib_binary),
            "binary_sha256": None if zlib_binary is None else sha256_file(zlib_binary),
            "identity_covered_by_python_executable_sha256": zlib_binary is None,
        },
        "environment": {
            key: os.environ.get(key)
            for key in ("LANG", "LC_ALL", "PYTHONHASHSEED", "SOURCE_DATE_EPOCH", "TZ")
        },
        "reproducibility_scope": (
            "fixed ZIP metadata plus identical bound Python/zipfile/zlib runtime; "
            "cross-runtime deflate identity is not inferred"
        ),
    }


def measure_dense_section(
    *,
    source: Path,
    output_zip: Path,
    receipt: Path,
    required_source_sha256: str | None,
    command_argv: tuple[str, ...],
) -> dict:
    source = source.expanduser().resolve(strict=True)
    output_zip = output_zip.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"source must be a file: {source}")
    if output_zip.exists() or receipt.exists():
        raise FileExistsError("output ZIP/receipt overwrite refused")
    if output_zip.parent != receipt.parent:
        raise ValueError("ZIP and receipt must share one durable evidence directory")
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    source_bytes = source.stat().st_size
    required_free = max(2_000_000_000, source_bytes + 1_000_000_000)
    free_before = shutil.disk_usage(output_zip.parent).free
    if free_before < required_free:
        raise OSError(f"storage preflight refused: free={free_before} required={required_free}")
    producer_custody = _runtime_custody(command_argv=command_argv)

    source_sha256 = sha256_file(source)
    if required_source_sha256 is not None and source_sha256 != required_source_sha256:
        raise ValueError(f"source SHA-256 mismatch: expected {required_source_sha256}, got {source_sha256}")

    partial = output_zip.with_name(f".{output_zip.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"stale ZIP temporary requires review: {partial}")
    started = time.monotonic()
    try:
        info = zipfile.ZipInfo(MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        info._compresslevel = 9
        with (
            source.open("rb") as source_handle,
            zipfile.ZipFile(
                partial,
                "x",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=True,
            ) as archive,
            archive.open(info, "w", force_zip64=True) as member,
        ):
            shutil.copyfileobj(source_handle, member, length=CHUNK_BYTES)
        with zipfile.ZipFile(partial, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME:
                raise ValueError("dense section ZIP member contract mismatch")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"dense section ZIP CRC failure: {bad}")
            member = infos[0]
            if member.file_size != source_bytes:
                raise ValueError("dense section ZIP uncompressed size mismatch")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, output_zip)
    finally:
        # The partial is certified true scratch: it is incomplete by definition,
        # is never an authority path, and is success/failure cleaned.
        if partial.exists():
            partial.unlink()

    result = {
        "schema": SCHEMA,
        "evidence_label": "MEASURED_EXACT_ZIP_SECTION_BYTES_NOT_CONTEST_ARCHIVE",
        "source_path": str(source),
        "source_bytes": source_bytes,
        "source_sha256": source_sha256,
        "output_zip_path": str(output_zip),
        "zip_bytes": output_zip.stat().st_size,
        "zip_sha256": sha256_file(output_zip),
        "member_name": MEMBER_NAME,
        "member_uncompressed_bytes": member.file_size,
        "member_compressed_bytes": member.compress_size,
        "compression": "zip_deflate9_deterministic_metadata_zip64",
        "zip_crc_test": "PASS",
        "storage_preflight": {
            "free_bytes_before": free_before,
            "required_free_bytes": required_free,
            "passed": True,
        },
        "wall_seconds": time.monotonic() - started,
        "platform": platform.platform(),
        "producer_custody": producer_custody,
        "through_r_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
        "formulation_scope": "explicit dense float32 quotient-feature field only",
    }
    try:
        _atomic_json(receipt, result)
    except Exception:
        # This invocation created a deterministic rebuildable output but did
        # not certify it. Remove only that new output so a retry can produce
        # an atomic ZIP+receipt pair; source data is never touched.
        output_zip.unlink(missing_ok=True)
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--required-source-sha256")
    process_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(process_args)
    result = measure_dense_section(
        source=args.source,
        output_zip=args.output_zip,
        receipt=args.receipt,
        required_source_sha256=args.required_source_sha256,
        command_argv=(
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            *process_args,
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
