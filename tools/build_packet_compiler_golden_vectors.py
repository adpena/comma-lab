#!/usr/bin/env python3
"""Build deterministic golden vectors for the submission packet compiler.

The vectors are a Python-reference conformance surface for future Rust, Zig,
C, or assembly packet/compiler ports. They preserve ZIP header facts, member
SHA-256s, charged byte accounting, and fail-closed negative cases without
making score, promotion, or dispatch claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.submission_archive import (  # noqa: E402
    DETERMINISTIC_ZIP_DATE_TIME,
    DETERMINISTIC_ZIP_FILE_MODE,
)
from tac.submission_packet_compiler import TARGET_PROFILES, inspect_packet  # noqa: E402

INDEX_NAME = "packet_compiler_golden_vectors_index.json"
VECTOR_SCHEMA_VERSION = "packet_compiler_golden_vector.v1"
INDEX_SCHEMA_VERSION = "packet_compiler_golden_vector_index.v1"
TOOL_NAME = "tools/build_packet_compiler_golden_vectors.py"
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
EOCD_SIG = 0x06054B50
CENTRAL_DIRECTORY_SIG = 0x02014B50
BUILTIN_STORED_SHA256 = "7cae837c71aa1abbc55b52dcdb51487a847725bb97cb507d5761ac23c344bf86"


@dataclass(frozen=True)
class PacketVectorSpec:
    label: str
    packet_path: Path
    normalized_packet_path: str
    description: str
    expected_contest_shape: bool | None = None
    expected_blockers: tuple[str, ...] = ()
    fixture_archive_path: str | None = None


@dataclass(frozen=True)
class VectorArtifact:
    label: str
    description: str
    vector_path: Path
    vector_sha256: str
    archive_path: str | None
    archive_bytes: int | None
    archive_sha256: str | None
    expected_blockers: tuple[str, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _relative_output_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _validate_label(label: str) -> str:
    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"invalid vector label {label!r}; use [A-Za-z0-9_.-]+")
    return label


def _archive_path_for_packet(packet_path: Path) -> Path:
    return packet_path if packet_path.is_file() else packet_path / "archive.zip"


def _stable_archive_path(normalized_packet_path: str, packet_path: Path) -> str:
    return normalized_packet_path if packet_path.is_file() else f"{normalized_packet_path}/archive.zip"


def _write_deterministic_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, payload in members:
                info = zipfile.ZipInfo(name, date_time=DETERMINISTIC_ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = (0o100000 | DETERMINISTIC_ZIP_FILE_MODE) << 16
                info.create_system = 3
                zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def _write_fixture_packet(root: Path, members: list[tuple[str, bytes]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_deterministic_archive(root / "archive.zip", members)
    inflate = root / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\npython inflate.py\n", encoding="utf-8")
    inflate.chmod(0o755)
    (root / "inflate.py").write_text("print('packet compiler golden vector fixture')\n", encoding="utf-8")


def _build_builtin_specs(output_dir: Path) -> list[PacketVectorSpec]:
    fixture_root = output_dir / "fixtures"
    stored_root = fixture_root / "stored_single_member_ok"
    duplicate_root = fixture_root / "duplicate_member_fail_closed"

    _write_fixture_packet(stored_root, [("x", b"payload-bytes")])
    stored_sha = _sha256_file(stored_root / "archive.zip")
    if stored_sha != BUILTIN_STORED_SHA256:
        raise RuntimeError(
            "stored fixture archive drifted: "
            f"expected {BUILTIN_STORED_SHA256}, observed {stored_sha}"
        )

    _write_fixture_packet(duplicate_root, [("x", b"a"), ("x", b"b")])

    return [
        PacketVectorSpec(
            label="stored_single_member_ok",
            packet_path=stored_root,
            normalized_packet_path="fixtures/stored_single_member_ok",
            fixture_archive_path="fixtures/stored_single_member_ok/archive.zip",
            description="Canonical stored single-member contest packet fixture.",
            expected_contest_shape=True,
            expected_blockers=(),
        ),
        PacketVectorSpec(
            label="duplicate_member_fail_closed",
            packet_path=duplicate_root,
            normalized_packet_path="fixtures/duplicate_member_fail_closed",
            fixture_archive_path="fixtures/duplicate_member_fail_closed/archive.zip",
            description="Negative fixture with duplicate ZIP member names; must fail closed.",
            expected_contest_shape=False,
            expected_blockers=("archive:duplicate_archive_member:x",),
        ),
    ]


def _find_eocd(raw_zip: bytes) -> dict[str, int]:
    if len(raw_zip) < 22:
        raise ValueError("end_of_central_directory_not_found")
    min_offset = max(0, len(raw_zip) - 65_557)
    for offset in range(len(raw_zip) - 22, min_offset - 1, -1):
        if struct.unpack_from("<I", raw_zip, offset)[0] != EOCD_SIG:
            continue
        comment_len = struct.unpack_from("<H", raw_zip, offset + 20)[0]
        end = offset + 22 + comment_len
        if end != len(raw_zip):
            continue
        (
            signature,
            disk_number,
            central_directory_disk,
            entries_this_disk,
            total_entries,
            central_directory_bytes,
            central_directory_offset,
            comment_bytes,
        ) = struct.unpack_from("<IHHHHIIH", raw_zip, offset)
        return {
            "signature": signature,
            "offset": offset,
            "bytes": 22 + comment_bytes,
            "disk_number": disk_number,
            "central_directory_disk": central_directory_disk,
            "entries_this_disk": entries_this_disk,
            "total_entries": total_entries,
            "central_directory_bytes": central_directory_bytes,
            "central_directory_offset": central_directory_offset,
            "comment_bytes": comment_bytes,
        }
    raise ValueError("end_of_central_directory_not_found")


def _parse_central_directory(raw_zip: bytes, eocd: dict[str, int]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    offset = eocd["central_directory_offset"]
    end = offset + eocd["central_directory_bytes"]
    while offset < end:
        if offset + 46 > len(raw_zip):
            raise ValueError(f"central_directory_header_out_of_range:{offset}")
        (
            signature,
            version_made_by,
            version_needed,
            flag_bits,
            compress_type,
            mod_time,
            mod_date,
            crc32,
            compressed_bytes,
            uncompressed_bytes,
            name_bytes,
            extra_bytes,
            comment_bytes,
            disk_start,
            internal_attr,
            external_attr,
            local_header_offset,
        ) = struct.unpack_from("<IHHHHHHIIIHHHHHII", raw_zip, offset)
        if signature != CENTRAL_DIRECTORY_SIG:
            raise ValueError(f"bad_central_directory_signature:{offset}")
        header_bytes = 46 + name_bytes + extra_bytes + comment_bytes
        header_end = offset + header_bytes
        if header_end > len(raw_zip):
            raise ValueError(f"central_directory_entry_out_of_range:{offset}")
        name_start = offset + 46
        name_end = name_start + name_bytes
        encoding = "utf-8" if flag_bits & (1 << 11) else "cp437"
        entries.append(
            {
                "central_header_offset": offset,
                "central_header_bytes": header_bytes,
                "central_header_sha256": _sha256(raw_zip[offset:header_end]),
                "version_made_by": version_made_by,
                "version_needed": version_needed,
                "flag_bits": flag_bits,
                "compress_type": compress_type,
                "mod_time": mod_time,
                "mod_date": mod_date,
                "crc32": f"{crc32:08x}",
                "compressed_bytes": compressed_bytes,
                "uncompressed_bytes": uncompressed_bytes,
                "name": raw_zip[name_start:name_end].decode(encoding),
                "name_bytes": name_bytes,
                "extra_bytes": extra_bytes,
                "comment_bytes": comment_bytes,
                "disk_start": disk_start,
                "internal_attr": internal_attr,
                "external_attr": external_attr,
                "local_header_offset": local_header_offset,
            }
        )
        offset = header_end
    if offset != end:
        raise ValueError("central_directory_size_mismatch")
    return entries


def _build_zip_header_manifest(archive_path: Path, archive_manifest: dict[str, Any] | None) -> dict[str, Any]:
    if archive_manifest is None or not archive_path.is_file():
        return {
            "schema_version": "packet_compiler_zip_header_manifest.v1",
            "available": False,
            "blockers": ["archive_zip_missing"],
        }

    raw_zip = archive_path.read_bytes()
    blockers: list[str] = []
    central_entries: list[dict[str, Any]] = []
    eocd_record: dict[str, Any] | None = None
    try:
        eocd = _find_eocd(raw_zip)
        eocd_offset = eocd["offset"]
        eocd_record = {
            **eocd,
            "signature_hex": f"{eocd['signature']:08x}",
            "sha256": _sha256(raw_zip[eocd_offset : eocd_offset + eocd["bytes"]]),
        }
        central_entries = _parse_central_directory(raw_zip, eocd)
    except ValueError as exc:
        blockers.append(str(exc))

    member_rows: list[dict[str, Any]] = []
    charged_payload_bytes = 0
    accounted_bytes = 0
    members = archive_manifest.get("members", [])
    if not isinstance(members, list):
        members = []
        blockers.append("archive_members_not_list")

    for index, member in enumerate(members):
        if not isinstance(member, dict):
            blockers.append(f"member_not_object:{index}")
            continue
        local = member.get("local_header", {})
        if not isinstance(local, dict):
            local = {}
        central = central_entries[index] if index < len(central_entries) else None
        local_header_offset = int(member.get("header_offset") or 0)
        data_offset = member.get("data_offset")
        compressed_bytes = int(member.get("compressed_bytes") or 0)
        local_header_bytes = 30 + int(local.get("name_bytes") or 0) + int(local.get("extra_bytes") or 0)
        payload_offset = int(data_offset) if isinstance(data_offset, int) else None
        charged_payload_bytes += compressed_bytes
        accounted_bytes += local_header_bytes + compressed_bytes
        if central is not None:
            accounted_bytes += int(central["central_header_bytes"])
        member_rows.append(
            {
                "member_order_index": index,
                "name": member.get("name"),
                "local_header": {
                    "offset": local_header_offset,
                    "bytes": local_header_bytes,
                    "sha256": _sha256(raw_zip[local_header_offset : local_header_offset + local_header_bytes]),
                    "name_bytes": int(local.get("name_bytes") or 0),
                    "extra_bytes": int(local.get("extra_bytes") or 0),
                },
                "compressed_payload": {
                    "offset": payload_offset,
                    "bytes": compressed_bytes,
                    "sha256": member.get("compressed_payload_sha256"),
                    "charged_payload_bytes": compressed_bytes,
                },
                "uncompressed_payload": {
                    "bytes": int(member.get("uncompressed_bytes") or 0),
                    "sha256": member.get("payload_sha256"),
                },
                "central_directory_header": central,
                "crc32": member.get("crc32"),
                "compress_type": member.get("compress_type"),
                "blockers": member.get("blockers", []),
            }
        )

    if eocd_record is not None:
        accounted_bytes += int(eocd_record["bytes"])

    archive_bytes = int(archive_manifest.get("bytes") or len(raw_zip))
    return {
        "schema_version": "packet_compiler_zip_header_manifest.v1",
        "available": True,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_manifest.get("sha256"),
        "zip_strict": archive_manifest.get("zip_strict"),
        "duplicate_member_names": archive_manifest.get("duplicate_member_names", []),
        "eocd": eocd_record,
        "members": member_rows,
        "charged_byte_accounting": {
            "contest_charged_archive_bytes": archive_bytes,
            "charged_member_payload_bytes": charged_payload_bytes,
            "zip_container_overhead_bytes": archive_bytes - charged_payload_bytes,
            "accounted_zip_span_bytes": accounted_bytes,
            "accounted_zip_span_matches_archive_bytes": accounted_bytes == archive_bytes,
            "notes": [
                "Contest rate term charges archive.zip bytes.",
                "charged_member_payload_bytes is the sum of compressed ZIP member payload spans.",
            ],
        },
        "blockers": blockers,
    }


def _normalise_compiler_manifest_paths(
    manifest: dict[str, Any],
    *,
    packet_path: Path,
    normalized_packet_path: str,
) -> dict[str, Any]:
    normalized = json.loads(_canonical_json(manifest))
    stable_archive = _stable_archive_path(normalized_packet_path, packet_path)
    normalized["packet"]["path"] = normalized_packet_path
    if normalized.get("archive") is not None:
        normalized["archive"]["path"] = stable_archive
    if normalized.get("inflate_sh") is not None and not packet_path.is_file():
        normalized["inflate_sh"]["path"] = f"{normalized_packet_path}/inflate.sh"
    native = normalized.get("native_zipwire")
    if isinstance(native, dict) and native.get("archive_path") is not None:
        native["archive_path"] = stable_archive
    return normalized


def _expectation_section(
    manifest: dict[str, Any],
    spec: PacketVectorSpec,
) -> dict[str, Any]:
    blockers = tuple(manifest["contest_compliance"]["blockers"])
    missing = [blocker for blocker in spec.expected_blockers if blocker not in blockers]
    unexpected_shape = (
        spec.expected_contest_shape is not None
        and bool(manifest["contest_compliance"]["contest_compliant_packet_shape"])
        != spec.expected_contest_shape
    )
    if missing:
        raise RuntimeError(f"{spec.label}: missing expected blockers: {missing}")
    if unexpected_shape:
        raise RuntimeError(
            f"{spec.label}: expected contest_compliant_packet_shape="
            f"{spec.expected_contest_shape}, observed "
            f"{manifest['contest_compliance']['contest_compliant_packet_shape']}"
        )
    return {
        "expected_contest_compliant_packet_shape": spec.expected_contest_shape,
        "expected_blockers": list(spec.expected_blockers),
        "observed_expected_blockers": [blocker for blocker in spec.expected_blockers if blocker in blockers],
        "must_fail_closed": spec.expected_contest_shape is False,
        "status": "matched",
    }


def build_vector_for_packet(
    spec: PacketVectorSpec,
    *,
    target_profile: str,
    embed_archive_hex_max_bytes: int,
) -> dict[str, Any]:
    manifest = inspect_packet(spec.packet_path, target_profile=target_profile)
    compiler_manifest = _normalise_compiler_manifest_paths(
        manifest,
        packet_path=spec.packet_path,
        normalized_packet_path=spec.normalized_packet_path,
    )
    archive_path = _archive_path_for_packet(spec.packet_path)
    header_manifest = _build_zip_header_manifest(archive_path, compiler_manifest.get("archive"))
    raw_archive = archive_path.read_bytes() if archive_path.is_file() else b""
    archive_hex = raw_archive.hex() if 0 < len(raw_archive) <= embed_archive_hex_max_bytes else None
    expectation = _expectation_section(compiler_manifest, spec)
    compiler_manifest_text = _canonical_json(compiler_manifest)
    header_manifest_text = _canonical_json(header_manifest)
    return {
        "schema_version": VECTOR_SCHEMA_VERSION,
        "label": spec.label,
        "description": spec.description,
        "oracle": "tac.submission_packet_compiler.inspect_packet",
        "oracle_status": "python_reference",
        "tool": TOOL_NAME,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "input_archive": {
            "path": spec.fixture_archive_path or _stable_archive_path(spec.normalized_packet_path, spec.packet_path),
            "bytes": len(raw_archive) if raw_archive else None,
            "sha256": _sha256(raw_archive) if raw_archive else None,
            "hex": archive_hex,
            "hex_embedded": archive_hex is not None,
            "hex_embed_max_bytes": embed_archive_hex_max_bytes,
        },
        "compiler_manifest_sha256": _sha256(compiler_manifest_text.encode("utf-8")),
        "compiler_manifest": compiler_manifest,
        "zip_header_manifest_sha256": _sha256(header_manifest_text.encode("utf-8")),
        "zip_header_manifest": header_manifest,
        "expectation": expectation,
        "native_port_contract": {
            "reference_language": "python",
            "compare_surfaces": [
                "compiler_manifest.archive",
                "compiler_manifest.golden_vectors.member_vectors",
                "zip_header_manifest",
                "zip_header_manifest.charged_byte_accounting",
            ],
            "must_match": [
                "archive bytes and SHA-256",
                "member order, names, local header names, offsets, CRCs, sizes, and payload SHA-256s",
                "duplicate-member blockers and zip_strict false negative cases",
                "charged archive bytes and compressed member payload byte accounting",
            ],
            "not_score_evidence": True,
        },
    }


def build_golden_vector_suite(
    *,
    output_dir: Path,
    packet_specs: list[PacketVectorSpec] | None = None,
    include_builtin_fixtures: bool = True,
    target_profile: str = "contest_one_video_replay",
    embed_archive_hex_max_bytes: int = 4096,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[PacketVectorSpec] = []
    if include_builtin_fixtures:
        specs.extend(_build_builtin_specs(output_dir))
    if packet_specs:
        specs.extend(packet_specs)
    if not specs:
        raise ValueError("no packet specs supplied")

    artifacts: list[VectorArtifact] = []
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _validate_label(spec.label)
        vector = build_vector_for_packet(
            spec,
            target_profile=target_profile,
            embed_archive_hex_max_bytes=embed_archive_hex_max_bytes,
        )
        vector_text = _canonical_json(vector)
        vector_path = vector_dir / f"{spec.label}.json"
        vector_path.write_text(vector_text, encoding="utf-8")
        archive = vector["input_archive"]
        artifacts.append(
            VectorArtifact(
                label=spec.label,
                description=spec.description,
                vector_path=vector_path,
                vector_sha256=_sha256(vector_text.encode("utf-8")),
                archive_path=archive.get("path"),
                archive_bytes=archive.get("bytes"),
                archive_sha256=archive.get("sha256"),
                expected_blockers=spec.expected_blockers,
            )
        )

    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "oracle": "tac.submission_packet_compiler.inspect_packet",
        "oracle_status": "python_reference",
        "target_profile": target_profile,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatchable": False,
        "ready_for_exact_eval_dispatch": False,
        "vector_count": len(artifacts),
        "vectors": [
            {
                "label": artifact.label,
                "description": artifact.description,
                "vector_path": _relative_output_path(artifact.vector_path, output_dir),
                "vector_sha256": artifact.vector_sha256,
                "archive_path": artifact.archive_path,
                "archive_bytes": artifact.archive_bytes,
                "archive_sha256": artifact.archive_sha256,
                "expected_blockers": list(artifact.expected_blockers),
            }
            for artifact in artifacts
        ],
        "suite_sha256": None,
    }
    index_for_hash = dict(index)
    index_for_hash["suite_sha256"] = ""
    index["suite_sha256"] = _sha256(_canonical_json(index_for_hash).encode("utf-8"))
    (output_dir / INDEX_NAME).write_text(_canonical_json(index), encoding="utf-8")
    return index


def _parse_packet_arg(value: str) -> PacketVectorSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--packet must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    _validate_label(label)
    packet_path = Path(raw_path)
    return PacketVectorSpec(
        label=label,
        packet_path=packet_path,
        normalized_packet_path=f"inputs/{label}",
        description=f"User-supplied packet {label}.",
        expected_contest_shape=None,
        expected_blockers=(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="directory to write vectors and fixtures")
    parser.add_argument(
        "--packet",
        action="append",
        type=_parse_packet_arg,
        default=[],
        metavar="LABEL=PATH",
        help="additional packet directory or archive.zip to inspect",
    )
    parser.add_argument(
        "--no-builtin-fixtures",
        action="store_true",
        help="only emit vectors for --packet inputs",
    )
    parser.add_argument(
        "--embed-archive-hex-max-bytes",
        type=int,
        default=4096,
        help="embed raw archive hex only when the archive is at or below this byte count",
    )
    parser.add_argument(
        "--target-profile",
        choices=TARGET_PROFILES,
        default="contest_one_video_replay",
        help="target profile passed through to tac.submission_packet_compiler.inspect_packet",
    )
    args = parser.parse_args(argv)

    try:
        index = build_golden_vector_suite(
            output_dir=args.output_dir,
            packet_specs=args.packet,
            include_builtin_fixtures=not args.no_builtin_fixtures,
            target_profile=args.target_profile,
            embed_archive_hex_max_bytes=args.embed_archive_hex_max_bytes,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        f"[packet-compiler-golden-vectors] vectors={index['vector_count']} "
        f"index={args.output_dir / INDEX_NAME} suite_sha256={index['suite_sha256']}"
    )
    for row in index["vectors"]:
        print(f"  - {row['label']}: {row['vector_path']} sha256={row['vector_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
