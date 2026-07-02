#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic archive.zip from a torch-vehicle best_archive.bin payload."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive_zip(input_bin: Path, output_zip: Path, *, member_name: str) -> dict:
    payload = input_bin.read_bytes()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output_zip, "w") as zf:
        zf.writestr(info, payload)
    archive_bytes = output_zip.read_bytes()
    return {
        "schema": "torch_vehicle_d2_archive_zip_manifest.v1",
        "input_bin": str(input_bin),
        "input_bytes": len(payload),
        "input_sha256": _sha256(payload),
        "archive_zip": str(output_zip),
        "archive_zip_bytes": len(archive_bytes),
        "archive_zip_sha256": _sha256(archive_bytes),
        "member_name": member_name,
        "member_compression": "stored",
        "deterministic_mtime": "1980-01-01T00:00:00Z",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bin", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--member-name", default="0.bin")
    parser.add_argument("--manifest-json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build_archive_zip(
        args.input_bin, args.output_zip, member_name=args.member_name
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.manifest_json is not None:
        args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
