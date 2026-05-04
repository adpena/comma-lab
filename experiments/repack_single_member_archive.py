#!/usr/bin/env python3
"""Deterministically repack a one-member contest archive under a new name."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_single_member(path: Path) -> tuple[zipfile.ZipInfo, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise SystemExit(f"{path} has {len(infos)} file members; expected exactly one")
        info = infos[0]
        return info, zf.read(info.filename)


def write_single_member(
    output: Path,
    member_name: str,
    payload: bytes,
    *,
    compress_type: int,
    compresslevel: int | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=FIXED_DATE_TIME)
    info.compress_type = compress_type
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(output, "w", strict_timestamps=True) as zf:
        zf.writestr(info, payload, compress_type=compress_type, compresslevel=compresslevel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--member-name", required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--compression",
        choices=("preserve", "stored", "deflated"),
        default="preserve",
    )
    args = parser.parse_args()

    source_info, payload = read_single_member(args.input)
    if args.compression == "preserve":
        compress_type = source_info.compress_type
    elif args.compression == "stored":
        compress_type = zipfile.ZIP_STORED
    else:
        compress_type = zipfile.ZIP_DEFLATED
    compresslevel = 9 if compress_type == zipfile.ZIP_DEFLATED else None
    write_single_member(
        args.output,
        args.member_name,
        payload,
        compress_type=compress_type,
        compresslevel=compresslevel,
    )
    record = {
        "input": str(args.input),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "source_member_name": source_info.filename,
        "source_member_bytes": len(payload),
        "source_member_sha256": hashlib.sha256(payload).hexdigest(),
        "target_member_name": args.member_name,
        "compression": args.compression,
        "compress_type": compress_type,
        "byte_delta": args.output.stat().st_size - args.input.stat().st_size,
        "score_claim": False,
        "evidence_grade": "byte_repack_only",
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
