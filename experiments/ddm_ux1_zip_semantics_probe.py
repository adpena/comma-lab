#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute scorer-free ZIP edge probes against the exact AFR1 payload.

The upstream wrapper extracts with ``unzip -o`` and charges the outer file's
exact size.  AFR1 already uses a one-byte member name, so the only remaining
header-length question is whether Info-ZIP accepts an asymmetric empty local
filename while retaining the central-directory name ``p``.  This probe also
executes the stricter empty-central-name control.

Every candidate archive, extraction result, stdout/stderr log, and result row
is retained.  No scorer or renderer runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "ddm_ux1_zip_semantics_probe.v1"
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50
CENTRAL_DIRECTORY_SIGNATURE = 0x02014B50
EOCD_SIGNATURE = 0x06054B50


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _layout(data: bytes) -> dict[str, int]:
    if struct.unpack_from("<I", data, 0)[0] != LOCAL_FILE_HEADER_SIGNATURE:
        raise ValueError("archive does not start with a local file header")
    eocd = data.rfind(struct.pack("<I", EOCD_SIGNATURE))
    if eocd < 0 or eocd + 22 != len(data):
        raise ValueError("archive must have one terminal, comment-free EOCD")
    central_size = struct.unpack_from("<I", data, eocd + 12)[0]
    central_offset = struct.unpack_from("<I", data, eocd + 16)[0]
    if struct.unpack_from("<I", data, central_offset)[0] != CENTRAL_DIRECTORY_SIGNATURE:
        raise ValueError("central-directory offset does not point to its signature")
    local_name_length = struct.unpack_from("<H", data, 26)[0]
    local_extra_length = struct.unpack_from("<H", data, 28)[0]
    central_name_length = struct.unpack_from("<H", data, central_offset + 28)[0]
    central_extra_length = struct.unpack_from("<H", data, central_offset + 30)[0]
    central_comment_length = struct.unpack_from("<H", data, central_offset + 32)[0]
    return {
        "local_name_length": local_name_length,
        "local_extra_length": local_extra_length,
        "central_name_length": central_name_length,
        "central_extra_length": central_extra_length,
        "central_comment_length": central_comment_length,
        "central_offset": central_offset,
        "central_size": central_size,
        "eocd_offset": eocd,
    }


def _drop_local_name(data: bytes) -> bytes:
    layout = _layout(data)
    name_length = layout["local_name_length"]
    if name_length != 1 or data[30:31] != b"p":
        raise ValueError("probe expects the exact AFR1 one-byte local name p")
    candidate = bytearray(data[:30] + data[31:])
    struct.pack_into("<H", candidate, 26, 0)
    new_eocd = layout["eocd_offset"] - 1
    struct.pack_into("<I", candidate, new_eocd + 16, layout["central_offset"] - 1)
    return bytes(candidate)


def _drop_central_name(data: bytes) -> bytes:
    layout = _layout(data)
    central = layout["central_offset"]
    name_start = central + 46
    if layout["central_name_length"] != 1 or data[name_start : name_start + 1] != b"p":
        raise ValueError("probe expects the exact AFR1 one-byte central name p")
    candidate = bytearray(data[:name_start] + data[name_start + 1 :])
    struct.pack_into("<H", candidate, central + 28, 0)
    new_eocd = layout["eocd_offset"] - 1
    struct.pack_into("<I", candidate, new_eocd + 12, layout["central_size"] - 1)
    return bytes(candidate)


def _zipfile_probe(archive: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive) as opened:
            names = opened.namelist()
            payload = opened.read(names[0]) if len(names) == 1 and names[0] else None
        return {
            "passed": payload is not None,
            "names": names,
            "payload_sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
            "error": None,
        }
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        return {
            "passed": False,
            "names": None,
            "payload_sha256": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def _execute_variant(
    *,
    name: str,
    data: bytes,
    output_dir: Path,
    reference_payload_sha256: str,
) -> dict[str, Any]:
    variant_dir = output_dir / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    archive = variant_dir / "archive.zip"
    archive.write_bytes(data)
    extracted = variant_dir / "extracted"
    extracted.mkdir(exist_ok=True)
    completed = subprocess.run(
        ["unzip", "-o", str(archive), "-d", str(extracted)],
        text=True,
        capture_output=True,
        check=False,
    )
    (variant_dir / "unzip.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (variant_dir / "unzip.stderr.log").write_text(completed.stderr, encoding="utf-8")
    payload = extracted / "p"
    payload_sha256 = _sha256(payload) if payload.is_file() else None
    row = {
        "schema": f"{SCHEMA}.variant",
        "candidate_id": name,
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
            "layout": _layout(data),
        },
        "infozip": {
            "command": ["unzip", "-o", str(archive), "-d", str(extracted)],
            "returncode": completed.returncode,
            "stdout_path": str(variant_dir / "unzip.stdout.log"),
            "stderr_path": str(variant_dir / "unzip.stderr.log"),
            "payload_path": str(payload) if payload.is_file() else None,
            "payload_sha256": payload_sha256,
            "payload_byte_identical": payload_sha256 == reference_payload_sha256,
        },
        "python_zipfile_control": _zipfile_probe(archive),
        "scorer_loaded": False,
        "renderer_ran": False,
        "score_claim": False,
    }
    _atomic_json(variant_dir / "RESULT.json", row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    original_layout = _layout(data)
    with zipfile.ZipFile(source) as opened:
        if opened.namelist() != ["p"]:
            raise SystemExit("source archive must contain exactly member p")
        source_payload = opened.read("p")
    reference_payload_sha256 = hashlib.sha256(source_payload).hexdigest()

    local_empty = _drop_local_name(data)
    central_empty = _drop_central_name(data)
    both_empty = _drop_central_name(local_empty)
    candidates = {
        "control_original": data,
        "probe_local_filename_empty": local_empty,
        "probe_central_filename_empty": central_empty,
        "probe_both_filenames_empty": both_empty,
    }
    rows = [
        _execute_variant(
            name=name,
            data=candidate,
            output_dir=output_dir,
            reference_payload_sha256=reference_payload_sha256,
        )
        for name, candidate in candidates.items()
    ]
    source_bytes = len(data)
    for row in rows:
        row["archive_delta_bytes"] = int(row["archive"]["bytes"]) - source_bytes

    local_row = next(row for row in rows if row["candidate_id"] == "probe_local_filename_empty")
    central_rows = [
        row
        for row in rows
        if row["candidate_id"] in {
            "probe_central_filename_empty",
            "probe_both_filenames_empty",
        }
    ]
    local_survives = (
        local_row["infozip"]["returncode"] == 0
        and local_row["infozip"]["payload_byte_identical"] is True
    )
    central_survives = any(
        row["infozip"]["returncode"] == 0
        and row["infozip"]["payload_byte_identical"] is True
        for row in central_rows
    )
    result = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU scorer-free exact archive/extractor semantics]",
        "source_archive": {
            "path": str(source),
            "bytes": source_bytes,
            "sha256": _sha256(source),
            "layout": original_layout,
            "payload_sha256": reference_payload_sha256,
        },
        "variants": rows,
        "probe_local_filename_empty": {
            "verdict": "OPEN-executed-probe" if local_survives else "BOUNDED-CLOSED",
            "survives_upstream_infozip_extraction": local_survives,
            "priced_archive_delta_bytes": -1,
            "runtime_followup": (
                "free verifier pin/ZipFile check must be updated before a receiver-closed rider"
                if local_survives
                else None
            ),
        },
        "probe_empty_central_name": {
            "verdict": "OPEN-executed-probe" if central_survives else "BOUNDED-CLOSED",
            "survives_upstream_infozip_extraction": central_survives,
            "priced_archive_delta_bytes": -1,
        },
        "score_claim": False,
        "promotable": False,
        "scorer_loaded": False,
        "modal_dispatched": False,
        "payload_policy": "all candidate archives and every extracted payload retained",
    }
    _atomic_json(output_dir / "RESULT.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
