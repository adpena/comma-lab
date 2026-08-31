#!/usr/bin/env python3
"""Retained scorer-free DCF1 control and RX1 self-delimiting-header proof.

This experiment is deliberately narrow.  It first reproduces the exact LB1 ZIP
bytes, then removes only the 14-byte RX1 length header and proves that three
consecutive Brotli streams can recover their own boundaries.  Every materialized
payload is retained in the selected SSD store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli

SOURCE_DEFAULT = Path(
    "/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/retained/"
    "candidate_lb1_joint22_patch192.zip"
)
OUT_DEFAULT = Path(
    "/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/retained"
)
SOURCE_SHA256 = "5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9"
SOURCE_BYTES = 180_083
RX1_HEADER = struct.Struct("<4sBBBBHHH")
EXPECTED_HEADER = {
    "magic": "RX1M",
    "version": 1,
    "codec": 2,
    "table_mode": 0,
    "reserved": 0x1A,
    "hpac_bytes": 13_515,
    "semantic_bytes": 30_856,
    "carrier_bytes": 22_010,
}
SECTION_NAMES = ("hpac", "semantic", "carrier")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite non-identical retained payload: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def clone_zip_info(source: zipfile.ZipInfo) -> zipfile.ZipInfo:
    target = zipfile.ZipInfo(source.filename, source.date_time)
    for attribute in (
        "compress_type",
        "comment",
        "extra",
        "create_system",
        "create_version",
        "extract_version",
        "reserved",
        "flag_bits",
        "volume",
        "internal_attr",
        "external_attr",
    ):
        setattr(target, attribute, getattr(source, attribute))
    return target


def write_repacked_zip(
    path: Path,
    *,
    source_info: zipfile.ZipInfo,
    source_comment: bytes,
    member: bytes,
) -> None:
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with zipfile.ZipFile(temporary, "x") as archive:
        archive.comment = source_comment
        archive.writestr(clone_zip_info(source_info), member)
    os.replace(temporary, path)


def discover_brotli_stream(payload: bytes, offset: int) -> tuple[bytes, bytes, int]:
    decoder = brotli.Decompressor()
    decoded = bytearray()
    cursor = offset
    while cursor < len(payload):
        decoded.extend(decoder.process(payload[cursor : cursor + 1]))
        cursor += 1
        if decoder.is_finished():
            return payload[offset:cursor], bytes(decoded), cursor
    raise RuntimeError(f"unterminated Brotli stream at offset {offset}")


def parse_header(member: bytes) -> dict[str, Any]:
    values = RX1_HEADER.unpack_from(member)
    parsed = {
        "magic": values[0].decode("ascii"),
        "version": values[1],
        "codec": values[2],
        "table_mode": values[3],
        "reserved": values[4],
        "hpac_bytes": values[5],
        "semantic_bytes": values[6],
        "carrier_bytes": values[7],
    }
    if parsed != EXPECTED_HEADER:
        raise RuntimeError(f"LB1 RX1 header drifted: {parsed!r}")
    return parsed


def stage_preflight(source: Path, out: Path, repo: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != SOURCE_BYTES or sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("source archive does not match pinned LB1 bytes")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "stage_00_preflight.json"
    if checkpoint.is_file():
        retained = read_json(checkpoint)
        if retained.get("source", {}).get("sha256") != SOURCE_SHA256:
            raise RuntimeError("retained preflight belongs to a different source archive")
        return retained
    free_bytes = shutil.disk_usage(out).free
    required_free = 8 * SOURCE_BYTES + (1 << 20)
    if free_bytes < required_free:
        raise RuntimeError(
            f"storage preflight failed: free={free_bytes}, required={required_free}"
        )
    receipt = {
        "schema": "ddm_dcf1_stage_preflight.v1",
        "source": describe(source),
        "output_store": str(out),
        "free_bytes_at_preflight": free_bytes,
        "required_free_bytes": required_free,
        "repo_git_head": git_head(repo),
        "experiment_source": describe(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    atomic_json(checkpoint, receipt)
    return receipt


def stage_control(source: Path, out: Path) -> tuple[bytes, zipfile.ZipInfo, bytes, dict[str, Any]]:
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p":
            raise RuntimeError("LB1 archive is not the expected one-member ZIP")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("LB1 contest ZIP member is unexpectedly compressed")
        member = archive.read(info)
        comment = archive.comment

    control = out / "control_repacked.zip"
    write_repacked_zip(control, source_info=info, source_comment=comment, member=member)
    control_repeat = out / "control_repacked.repeat.zip"
    write_repacked_zip(control_repeat, source_info=info, source_comment=comment, member=member)
    if sha256_file(control) != SOURCE_SHA256 or sha256_file(control_repeat) != SOURCE_SHA256:
        raise RuntimeError("repacked control is not byte-identical to pinned LB1")
    receipt = {
        "schema": "ddm_dcf1_stage_control.v1",
        "source": describe(source),
        "member": {"bytes": len(member), "sha256": sha256_bytes(member)},
        "control": describe(control),
        "control_repeat": describe(control_repeat),
        "byte_identical": True,
    }
    atomic_write(out / "control_member.bin", member)
    atomic_json(out / "stage_10_control.json", receipt)
    return member, info, comment, receipt


def stage_flatten(
    source: Path,
    out: Path,
    member: bytes,
    info: zipfile.ZipInfo,
    comment: bytes,
) -> dict[str, Any]:
    parsed_header = parse_header(member)
    explicit_offset = RX1_HEADER.size
    explicit_streams: dict[str, bytes] = {}
    for name in SECTION_NAMES:
        stream_bytes = int(parsed_header[f"{name}_bytes"])
        explicit_streams[name] = member[explicit_offset : explicit_offset + stream_bytes]
        explicit_offset += stream_bytes
    explicit_tail = member[explicit_offset:]
    if len(explicit_tail) <= 96:
        raise RuntimeError("RX1 residual/token tail is truncated")

    headerless = member[RX1_HEADER.size :]
    cursor = 0
    stream_rows: list[dict[str, Any]] = []
    discovered_streams: list[bytes] = []
    for name in SECTION_NAMES:
        stream, decoded, cursor = discover_brotli_stream(headerless, cursor)
        if stream != explicit_streams[name]:
            raise RuntimeError(f"self-delimited {name} boundary differs from RX1 length")
        discovered_streams.append(stream)
        stream_path = out / f"{name}.stream.br"
        raw_path = out / f"{name}.decoded.bin"
        atomic_write(stream_path, stream)
        atomic_write(raw_path, decoded)
        stream_rows.append(
            {
                "name": name,
                "stream": describe(stream_path),
                "decoded": describe(raw_path),
                "end_offset_in_headerless_member": cursor,
                "boundary_identity": True,
            }
        )

    discovered_tail = headerless[cursor:]
    if discovered_tail != explicit_tail:
        raise RuntimeError("self-delimited tail differs from RX1 length parse")
    tail_path = out / "residual_and_token_tail.bin"
    atomic_write(tail_path, discovered_tail)
    headerless_path = out / "headerless_member.bin"
    atomic_write(headerless_path, headerless)

    reconstructed = member[: RX1_HEADER.size] + b"".join(discovered_streams) + discovered_tail
    if reconstructed != member:
        raise RuntimeError("self-delimited reconstruction is not byte-identical")
    reconstructed_path = out / "reconstructed_rx1_member.bin"
    atomic_write(reconstructed_path, reconstructed)

    candidate = out / "candidate_headerless_rx1.zip"
    candidate_repeat = out / "candidate_headerless_rx1.repeat.zip"
    write_repacked_zip(candidate, source_info=info, source_comment=comment, member=headerless)
    write_repacked_zip(candidate_repeat, source_info=info, source_comment=comment, member=headerless)
    candidate_description = describe(candidate)
    repeat_description = describe(candidate_repeat)
    if (
        candidate_description["bytes"],
        candidate_description["sha256"],
    ) != (
        repeat_description["bytes"],
        repeat_description["sha256"],
    ):
        raise RuntimeError("headerless candidate repeat is not byte-identical")
    if candidate.stat().st_size != source.stat().st_size - RX1_HEADER.size:
        raise RuntimeError("headerless archive did not save exactly the RX1 header size")

    receipt = {
        "schema": "ddm_dcf1_stage_flatten.v1",
        "axis": "scorer-free exact archive bytes and exact RX1 component reconstruction",
        "score_claim": False,
        "header": parsed_header,
        "header_bytes_removed": RX1_HEADER.size,
        "source_archive": describe(source),
        "candidate_archive": candidate_description,
        "candidate_repeat_archive": repeat_description,
        "archive_byte_delta": candidate.stat().st_size - source.stat().st_size,
        "self_delimited_streams": stream_rows,
        "tail": describe(tail_path),
        "tail_split": {
            "fixed_residual_bytes": 96,
            "token_stream_bytes": len(discovered_tail) - 96,
        },
        "headerless_member": describe(headerless_path),
        "reconstructed_original_member": describe(reconstructed_path),
        "original_member_reconstruction_identity": True,
        "receiver_status": (
            "research parser proof only; native LB1 receiver still requires RX1M and therefore "
            "must be patched and byte-identity-tested before composition or scoring"
        ),
    }
    atomic_json(out / "stage_20_flatten.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify/reuse byte-identical retained stage outputs instead of replacing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    out = args.out.resolve()
    preflight = stage_preflight(source, out, repo)
    member, info, comment, control = stage_control(source, out)
    flatten = stage_flatten(source, out, member, info, comment)
    result = {
        "schema": "ddm_dcf1_duplicate_carry_factorization_result.v1",
        "status": "COMPLETE",
        "resume_requested": bool(args.resume),
        "preflight": preflight,
        "control": control,
        "flatten": flatten,
        "conclusion": (
            "The exact LB1 RX1 length header is losslessly derivable from three consecutive "
            "Brotli end markers plus the fixed receiver grammar, yielding a retained deterministic "
            "14-byte-smaller research archive. Native receiver identity remains unproved."
        ),
    }
    atomic_json(out / "DCF1_RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
