# SPDX-License-Identifier: MIT
"""Deterministic ZIP runtime-closure repair helpers for archive-bound candidates."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA = "hprc_runtime_closure_repair_report.v1"
_FIXED_ZIP_DATE = (2026, 1, 1, 0, 0, 0)


def repair_embedded_runtime_zip_closure(
    *,
    source_archive: str | Path,
    output_archive: str | Path,
    add_members: dict[str, str | Path],
    replace_members: dict[str, str | Path] | None = None,
    report_path: str | Path,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Copy a submission ZIP and add missing runtime source members.

    This is for custody-preserving runtime closure only: member payloads are
    copied byte-for-byte, added source files are hashed, and the report keeps
    score authority false until receiver proof and exact CPU/CUDA eval land.
    """

    source = Path(source_archive).expanduser().resolve(strict=False)
    output = Path(output_archive).expanduser().resolve(strict=False)
    report = Path(report_path).expanduser().resolve(strict=False)
    if output.exists() and not allow_overwrite:
        raise FileExistsError(f"output archive exists: {output}")
    if report.exists() and not allow_overwrite:
        raise FileExistsError(f"repair report exists: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    additions = {
        _normalize_member_name(member): Path(path).expanduser().resolve(strict=False)
        for member, path in add_members.items()
    }
    replacements = {
        _normalize_member_name(member): Path(path).expanduser().resolve(strict=False)
        for member, path in (replace_members or {}).items()
    }
    blockers: list[str] = []
    added_rows: list[dict[str, Any]] = []
    replaced_rows: list[dict[str, Any]] = []
    copied_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(source, "r") as zin:
        existing_names = {info.filename for info in zin.infolist() if not info.is_dir()}
        for member, path in additions.items():
            if member in existing_names:
                blockers.append(f"member_already_present:{member}")
            if not path.is_file():
                blockers.append(f"add_member_source_missing:{member}")
        for member, path in replacements.items():
            if member not in existing_names:
                blockers.append(f"replace_member_missing:{member}")
            if not path.is_file():
                blockers.append(f"replace_member_source_missing:{member}")
        if blockers:
            payload = _report_payload(
                source_archive=source,
                output_archive=output,
                added_rows=[],
                replaced_rows=[],
                copied_rows=[],
                blockers=blockers,
            )
            report.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            return {**payload, "report_path": report.as_posix()}
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
            for info in sorted(zin.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                data = zin.read(info.filename)
                if info.filename in replacements:
                    replacement = replacements[info.filename].read_bytes()
                    _writestr(zout, info.filename, replacement)
                    replaced_rows.append(
                        {
                            "member": info.filename,
                            "source_path": replacements[info.filename].as_posix(),
                            "old_bytes": len(data),
                            "old_sha256": _sha256_bytes(data),
                            "bytes": len(replacement),
                            "sha256": _sha256_bytes(replacement),
                        }
                    )
                    continue
                _writestr(zout, info.filename, data)
                copied_rows.append(
                    {
                        "member": info.filename,
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                    }
                )
            for member, path in sorted(additions.items()):
                data = path.read_bytes()
                _writestr(zout, member, data)
                added_rows.append(
                    {
                        "member": member,
                        "source_path": path.as_posix(),
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                    }
                )

    payload = _report_payload(
        source_archive=source,
        output_archive=output,
        added_rows=added_rows,
        replaced_rows=replaced_rows,
        copied_rows=copied_rows,
        blockers=[],
    )
    report.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {**payload, "report_path": report.as_posix()}


def _report_payload(
    *,
    source_archive: Path,
    output_archive: Path,
    added_rows: list[dict[str, Any]],
    replaced_rows: list[dict[str, Any]],
    copied_rows: list[dict[str, Any]],
    blockers: list[str],
) -> dict[str, Any]:
    exists = output_archive.is_file()
    return {
        "schema": HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_archive": {
            "path": source_archive.as_posix(),
            "bytes": source_archive.stat().st_size if source_archive.is_file() else None,
            "sha256": _sha256_file(source_archive) if source_archive.is_file() else None,
        },
        "output_archive": {
            "path": output_archive.as_posix(),
            "bytes": output_archive.stat().st_size if exists else None,
            "sha256": _sha256_file(output_archive) if exists else None,
        },
        "copied_member_count": len(copied_rows),
        "added_members": added_rows,
        "replaced_members": replaced_rows,
        "copied_members": copied_rows,
        "runtime_closure_repair_ready_for_receiver_proof": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _writestr(zout: zipfile.ZipFile, member: str, data: bytes) -> None:
    info = zipfile.ZipInfo(_normalize_member_name(member), date_time=_FIXED_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zout.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _normalize_member_name(value: str) -> str:
    member = str(value).strip().replace("\\", "/")
    if not member or member.startswith("/") or "/../" in f"/{member}/":
        raise ValueError(f"unsafe ZIP member name: {value!r}")
    return member


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "HPRC_RUNTIME_CLOSURE_REPAIR_REPORT_SCHEMA",
    "repair_embedded_runtime_zip_closure",
]
