#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile native QMA9 metadata/decode timing without score claims.

This tool is a local CPU-only observability helper for PR85/PR92-style QMA9
mask payloads.  It can read a direct ``*.qma9`` stream or extract the mask
slice from a compact v5 single-member bundle, then drive the native Rust
``qma-codec`` CLI for metadata and decode timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_RS = REPO_ROOT / "runtime-rs"
DEFAULT_INPUT = REPO_ROOT / "experiments/results/public_pr92_intake_20260504_codex/archive.zip"
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/pr92_qma9_native_decode_profile_20260504_codex/timing_report.json"
)
TOOL = "experiments/profile_qma9_native_decode.py"
SCHEMA = "qma9_native_decode_timing_report_v1"
QMA9_HEADER_BYTES = 20
QMA9_MAGIC = b"QMA9"
COMPACT_V5_HEADER_BYTES = 24
COMPACT_V5_FIXED_BIAS_BYTES = 223
COMPACT_V5_FIXED_REGION_BYTES = 273


class QMA9NativeProfileError(RuntimeError):
    """Raised when input custody, bundle parsing, or native decode fails closed."""


@dataclass(frozen=True)
class QMA9Header:
    frame_count: int
    width: int
    height: int
    bitstream_bytes: int
    header_bytes: int
    packed_bytes: int
    decoded_mask_bytes: int


@dataclass(frozen=True)
class CompactV5MicroHeader:
    header_bytes: int
    mask_bytes: int
    model_bytes: int
    pose_bytes: int
    post_bytes: int
    shift_bytes: int
    frac_bytes: int
    frac2_bytes: int
    frac3_bytes: int
    bias_bytes: int
    region_bytes: int
    randmulti_bytes: int
    packed_bytes: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def parse_qma9_header(payload: bytes) -> QMA9Header:
    if len(payload) < QMA9_HEADER_BYTES:
        raise QMA9NativeProfileError("QMA9 payload is shorter than its 20-byte header")
    magic = payload[:4]
    if magic != QMA9_MAGIC:
        raise QMA9NativeProfileError(f"expected QMA9 magic, got {magic!r}")
    frame_count = int.from_bytes(payload[4:8], "little")
    width = int.from_bytes(payload[8:12], "little")
    height = int.from_bytes(payload[12:16], "little")
    bitstream_bytes = int.from_bytes(payload[16:20], "little")
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise QMA9NativeProfileError("QMA9 dimensions must be positive")
    packed_bytes = QMA9_HEADER_BYTES + bitstream_bytes
    if packed_bytes > len(payload):
        raise QMA9NativeProfileError(
            f"QMA9 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    decoded_mask_bytes = frame_count * width * height
    return QMA9Header(
        frame_count=frame_count,
        width=width,
        height=height,
        bitstream_bytes=bitstream_bytes,
        header_bytes=QMA9_HEADER_BYTES,
        packed_bytes=packed_bytes,
        decoded_mask_bytes=decoded_mask_bytes,
    )


def _u24le(payload: bytes, offset: int) -> int:
    return int.from_bytes(payload[offset : offset + 3], "little")


def parse_compact_v5_micro_header(payload: bytes) -> CompactV5MicroHeader:
    if len(payload) < COMPACT_V5_HEADER_BYTES:
        raise QMA9NativeProfileError(
            f"compact v5 bundle is shorter than {COMPACT_V5_HEADER_BYTES} bytes"
        )
    mask_bytes = _u24le(payload, 0)
    model_bytes = _u24le(payload, 3)
    pose_bytes = _u24le(payload, 6)
    post_bytes = _u24le(payload, 9)
    shift_bytes = _u24le(payload, 12)
    frac_bytes = _u24le(payload, 15)
    frac2_bytes = _u24le(payload, 18)
    frac3_bytes = _u24le(payload, 21)
    if (
        mask_bytes <= 1000
        or model_bytes <= 1000
        or pose_bytes <= 100
        or post_bytes == 0
        or post_bytes >= 10_000
        or shift_bytes == 0
        or shift_bytes >= 10_000
        or frac_bytes == 0
        or frac_bytes >= 10_000
        or frac2_bytes == 0
        or frac2_bytes >= 10_000
        or frac3_bytes == 0
        or frac3_bytes >= 10_000
    ):
        raise QMA9NativeProfileError("compact v5 bundle has invalid segment lengths")
    fixed_prefix = (
        COMPACT_V5_HEADER_BYTES
        + mask_bytes
        + model_bytes
        + pose_bytes
        + post_bytes
        + shift_bytes
        + frac_bytes
        + frac2_bytes
        + frac3_bytes
        + COMPACT_V5_FIXED_BIAS_BYTES
        + COMPACT_V5_FIXED_REGION_BYTES
    )
    if fixed_prefix >= len(payload):
        raise QMA9NativeProfileError(
            f"compact v5 bundle declares at least {fixed_prefix + 1} bytes but has {len(payload)}"
        )
    return CompactV5MicroHeader(
        header_bytes=COMPACT_V5_HEADER_BYTES,
        mask_bytes=mask_bytes,
        model_bytes=model_bytes,
        pose_bytes=pose_bytes,
        post_bytes=post_bytes,
        shift_bytes=shift_bytes,
        frac_bytes=frac_bytes,
        frac2_bytes=frac2_bytes,
        frac3_bytes=frac3_bytes,
        bias_bytes=COMPACT_V5_FIXED_BIAS_BYTES,
        region_bytes=COMPACT_V5_FIXED_REGION_BYTES,
        randmulti_bytes=len(payload) - fixed_prefix,
        packed_bytes=len(payload),
    )


def _read_zip_member(path: Path, member: str | None) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        selected_name = member
        if selected_name is None:
            if "x" in names:
                selected_name = "x"
            elif len(names) == 1:
                selected_name = names[0]
            else:
                raise QMA9NativeProfileError(
                    f"{path} has multiple file members {names!r}; pass --member"
                )
        matches = [info for info in infos if info.filename == selected_name]
        if len(matches) != 1:
            raise QMA9NativeProfileError(
                f"{path} must contain exactly one member {selected_name!r}; got {names!r}"
            )
        info = matches[0]
        data = zf.read(info)
    return data, {
        "archive_path": repo_rel(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_name": info.filename,
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": sha256_bytes(data),
        "zip_compress_type": int(info.compress_type),
        "all_file_members": names,
    }


def extract_qma9_payload(
    path: Path,
    *,
    member: str | None = None,
    input_format: str = "auto",
) -> tuple[bytes, dict[str, Any]]:
    source: dict[str, Any] = {
        "path": repo_rel(path),
        "input_format": input_format,
        "path_bytes": path.stat().st_size,
        "path_sha256": sha256_file(path),
    }
    is_zip = zipfile.is_zipfile(path)
    if input_format == "zip" and not is_zip:
        raise QMA9NativeProfileError(f"--input-format zip requires a ZIP file: {path}")
    if input_format in {"auto", "zip"} and is_zip:
        raw, zip_meta = _read_zip_member(path, member)
        source["zip"] = zip_meta
        bundle_bytes = raw
    else:
        if member is not None:
            raise QMA9NativeProfileError("--member is only valid for ZIP inputs")
        bundle_bytes = path.read_bytes()

    if input_format == "qma9" or (
        input_format in {"auto", "zip"} and bundle_bytes.startswith(QMA9_MAGIC)
    ):
        header = parse_qma9_header(bundle_bytes)
        return bundle_bytes[: header.packed_bytes], {
            **source,
            "extraction": "direct_qma9",
            "qma9_offset": 0,
            "qma9_bytes": header.packed_bytes,
        }

    if input_format in {"auto", "compact-v5-micro", "zip"}:
        compact = parse_compact_v5_micro_header(bundle_bytes)
        start = compact.header_bytes
        end = start + compact.mask_bytes
        mask = bundle_bytes[start:end]
        header = parse_qma9_header(mask)
        if header.packed_bytes != len(mask):
            raise QMA9NativeProfileError(
                f"compact mask slice has {len(mask)} bytes but QMA9 header declares {header.packed_bytes}"
            )
        return mask, {
            **source,
            "extraction": "compact_v5_micro_mask_slice",
            "compact_v5_micro_header": asdict(compact),
            "qma9_offset": start,
            "qma9_bytes": len(mask),
        }

    raise QMA9NativeProfileError(f"unsupported input format for QMA9 extraction: {input_format}")


def build_qma_codec(profile: str) -> Path:
    cmd = ["cargo", "build", "--quiet", "-p", "qma-codec"]
    if profile == "release":
        cmd.insert(2, "--release")
    elif profile != "debug":
        raise QMA9NativeProfileError(f"unknown build profile: {profile}")
    subprocess.run(cmd, cwd=RUNTIME_RS, check=True, timeout=240)
    decoder = RUNTIME_RS / "target" / ("release" if profile == "release" else "debug") / "qma-codec"
    if not decoder.is_file():
        raise QMA9NativeProfileError(f"native qma-codec build did not produce {decoder}")
    return decoder


def resolve_decoder(decoder: Path | None, build_profile: str) -> Path:
    if decoder is not None:
        if not decoder.is_file():
            raise QMA9NativeProfileError(f"qma-codec decoder is not a file: {decoder}")
        return decoder
    return build_qma_codec(build_profile)


def run_native_metadata(decoder: Path, qma9_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    proc = subprocess.run(
        [str(decoder), "metadata", str(qma9_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise QMA9NativeProfileError(
            f"native qma-codec metadata failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    metadata = json.loads(proc.stdout)
    return metadata, {
        "elapsed_seconds": elapsed,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def run_native_decode(
    decoder: Path,
    qma9_path: Path,
    *,
    header: QMA9Header,
    prefix_frames: int | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qma9-native-decode-") as tmp:
        raw_path = Path(tmp) / "mask.raw"
        frames = prefix_frames if prefix_frames is not None else header.frame_count
        cmd = [
            str(decoder),
            "decode",
            str(qma9_path),
            str(raw_path),
        ]
        if prefix_frames is not None:
            cmd += ["--prefix-frames", str(prefix_frames)]
        cmd += [
            "--expected-frames",
            str(frames),
            "--expected-width",
            str(header.width),
            "--expected-height",
            str(header.height),
        ]
        started = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            raise QMA9NativeProfileError(
                f"native qma-codec decode failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        raw = raw_path.read_bytes()
    expected_bytes = frames * header.width * header.height
    if len(raw) != expected_bytes:
        raise QMA9NativeProfileError(f"native decode wrote {len(raw)} bytes, expected {expected_bytes}")
    return {
        "elapsed_seconds": elapsed,
        "written_frames": frames,
        "decoded_bytes": len(raw),
        "decoded_sha256": sha256_bytes(raw),
        "encoded_bytes_per_second": header.packed_bytes / elapsed if elapsed > 0 else None,
        "decoded_bytes_per_second": len(raw) / elapsed if elapsed > 0 else None,
        "stderr": proc.stderr.strip(),
    }


def profile_qma9_native_decode(
    *,
    input_path: Path,
    output_json: Path,
    decoder_path: Path | None,
    build_profile: str,
    member: str | None,
    input_format: str,
    prefix_frames: int | None,
    repeat: int,
    expected_decoded_sha256: str | None,
) -> dict[str, Any]:
    if repeat <= 0:
        raise QMA9NativeProfileError("--repeat must be positive")
    qma9_payload, source = extract_qma9_payload(
        input_path,
        member=member,
        input_format=input_format,
    )
    header = parse_qma9_header(qma9_payload)
    if prefix_frames is not None and (prefix_frames <= 0 or prefix_frames > header.frame_count):
        raise QMA9NativeProfileError(
            f"--prefix-frames {prefix_frames} must be in [1, {header.frame_count}]"
        )
    decoder = resolve_decoder(decoder_path, build_profile)
    with tempfile.TemporaryDirectory(prefix="qma9-native-profile-") as tmp:
        qma9_path = Path(tmp) / "mask.qma9"
        qma9_path.write_bytes(qma9_payload)
        native_metadata, metadata_timing = run_native_metadata(decoder, qma9_path)
        timings = [
            run_native_decode(
                decoder,
                qma9_path,
                header=header,
                prefix_frames=prefix_frames,
            )
            for _ in range(repeat)
        ]

    hashes = [row["decoded_sha256"] for row in timings]
    unique_hashes = sorted(set(hashes))
    determinism_passed = len(unique_hashes) == 1
    expected_match = (
        None
        if expected_decoded_sha256 is None
        else determinism_passed and unique_hashes[0] == expected_decoded_sha256
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "input": source,
        "qma9_payload": {
            "bytes": len(qma9_payload),
            "sha256": sha256_bytes(qma9_payload),
            "header": asdict(header),
        },
        "native_decoder": {
            "path": repo_rel(decoder),
            "sha256": sha256_file(decoder),
            "build_profile": build_profile if decoder_path is None else "explicit_decoder_path",
        },
        "native_metadata": native_metadata,
        "metadata_timing": metadata_timing,
        "decode": {
            "storage_order": "frame_major_header_width_by_header_height",
            "prefix_frames": prefix_frames,
            "repeat": repeat,
            "timings": timings,
        },
        "determinism": {
            "passed": determinism_passed,
            "unique_decoded_sha256": unique_hashes,
            "expected_decoded_sha256": expected_decoded_sha256,
            "matches_expected_decoded_sha256": expected_match,
        },
        "no_op_detection": {
            "decoded_output_nonempty": all(row["decoded_bytes"] > 0 for row in timings),
            "decoded_sha256_equals_encoded_sha256": any(
                row["decoded_sha256"] == sha256_bytes(qma9_payload) for row in timings
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--decoder", type=Path, default=None)
    parser.add_argument("--build-profile", choices=("release", "debug"), default="release")
    parser.add_argument("--member", default=None)
    parser.add_argument(
        "--input-format",
        choices=("auto", "qma9", "zip", "compact-v5-micro"),
        default="auto",
    )
    parser.add_argument("--prefix-frames", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--expected-decoded-sha256", default=None)
    args = parser.parse_args(argv)

    try:
        report = profile_qma9_native_decode(
            input_path=args.input,
            output_json=args.output_json,
            decoder_path=args.decoder,
            build_profile=args.build_profile,
            member=args.member,
            input_format=args.input_format,
            prefix_frames=args.prefix_frames,
            repeat=args.repeat,
            expected_decoded_sha256=args.expected_decoded_sha256,
        )
    except QMA9NativeProfileError as exc:
        print(f"qma9-native-profile: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    expected_match = report["determinism"]["matches_expected_decoded_sha256"]
    if expected_match is False:
        return 1
    return 0 if report["determinism"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
