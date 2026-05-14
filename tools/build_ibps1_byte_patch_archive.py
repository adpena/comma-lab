# SPDX-License-Identifier: MIT
"""Build deterministic IBPS1 byte-patch candidate archives.

This is the byte-closed bridge from Z1 MDL byte probes to an exact-evaluable
candidate: mutate selected bytes in the inflate-time ``0.bin`` member and
write a new archive.zip plus manifest. It never loads scorers and makes no
score claim; callers must run auth eval on the produced packet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.c6_e4_mdl_ibps.archive import parse_ibps1_archive_bytes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_patch_spec(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise ValueError(f"patch spec must be SECTION:OFFSET, got {spec!r}")
    section, offset_s = spec.split(":", 1)
    section = section.strip()
    if not section:
        raise ValueError(f"missing section in patch spec {spec!r}")
    try:
        offset = int(offset_s, 0)
    except ValueError as exc:
        raise ValueError(f"invalid offset in patch spec {spec!r}") from exc
    if offset < 0:
        raise ValueError(f"offset must be >= 0 in patch spec {spec!r}")
    return section, offset


def _read_zip_single_member(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if "0.bin" in names:
            name = "0.bin"
        elif len(names) == 1:
            name = names[0]
        else:
            raise ValueError(f"{path} has no 0.bin and is not single-member: {names}")
        return name, zf.read(name)


def _write_deterministic_zip(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr(info, payload)


def build_patch_archive(
    *,
    source_archive: Path,
    output_archive: Path,
    patch_specs: list[str],
    xor_value: int = 0xFF,
) -> dict[str, object]:
    if not 0 <= xor_value <= 0xFF:
        raise ValueError("xor_value must be in 0..255")
    member_name, inner = _read_zip_single_member(source_archive)
    sections = parse_ibps1_archive_bytes(inner)
    buf = bytearray(inner)
    applied = []
    for raw in patch_specs:
        section, rel_offset = _parse_patch_spec(raw)
        if section not in sections:
            raise ValueError(f"unknown section {section!r}; known={sorted(sections)}")
        start, length = sections[section]
        if rel_offset >= length:
            raise ValueError(
                f"offset {rel_offset} out of range for {section} length {length}"
            )
        abs_offset = start + rel_offset
        before = buf[abs_offset]
        after = before ^ xor_value
        buf[abs_offset] = after
        applied.append({
            "section": section,
            "relative_offset": rel_offset,
            "absolute_offset": abs_offset,
            "before": before,
            "after": after,
            "xor_value": xor_value,
        })
    patched_inner = bytes(buf)
    # Fail closed if header/section lengths drift. Brotli decoder validity is
    # checked by downstream inflate/auth-eval; this validates grammar custody.
    parse_ibps1_archive_bytes(patched_inner)
    _write_deterministic_zip(output_archive, member_name, patched_inner)
    output_bytes = output_archive.read_bytes()
    return {
        "source_archive": str(source_archive),
        "source_archive_size": source_archive.stat().st_size,
        "source_archive_sha256": _sha256_bytes(source_archive.read_bytes()),
        "member_name": member_name,
        "source_inner_sha256": _sha256_bytes(inner),
        "output_archive": str(output_archive),
        "output_archive_size": len(output_bytes),
        "output_archive_sha256": _sha256_bytes(output_bytes),
        "output_inner_sha256": _sha256_bytes(patched_inner),
        "patches": applied,
        "score_claim": False,
        "decision_grade_required": "exact auth eval after packet build",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--output-archive", type=Path, required=True)
    p.add_argument("--patch", action="append", required=True,
                   help="Patch spec SECTION:OFFSET, e.g. decoder_blob:34576")
    p.add_argument("--xor-value", type=lambda s: int(s, 0), default=0xFF)
    p.add_argument("--manifest", type=Path,
                   help="Manifest JSON path; defaults beside output archive.")
    args = p.parse_args(argv)
    try:
        manifest = build_patch_archive(
            source_archive=args.source_archive,
            output_archive=args.output_archive,
            patch_specs=args.patch,
            xor_value=args.xor_value,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    manifest_path = args.manifest or args.output_archive.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
