#!/usr/bin/env python3
"""Extract and profile the PR85 QMA9 mask-token source.

This is a local custody/planning tool for future PR85-owned entropy coding.
It never runs a scorer, never dispatches work, and never mutates dispatch
state.  The output proves the charged QMA9 mask segment identity and, when
requested, emits the decoded uint8 token tensor that an HPAC/arithmetic coder
would consume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.qma9_range_mask_contract import (  # noqa: E402
    QMA9ContractError,
    decode_qma9_mask,
    encode_qma9_mask,
    parse_qma9_header,
    sha256_bytes,
    sha256_file,
)


TOOL = "experiments/profile_pr85_qma9_token_source.py"
SCHEMA = "pr85_qma9_mask_token_source_profile_v1"
DEFAULT_PR85_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_ARCHIVE = DEFAULT_PR85_DIR / "archive.zip"
DEFAULT_PROFILE_JSON = DEFAULT_PR85_DIR / "profile_pr85_bundle.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PR85_DIR / "qma9_token_source"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "pr85_qma9_token_source_profile.json"
DEFAULT_CPP_DECODER = DEFAULT_PR85_DIR / "replay_submission/range_mask_codec.cpp"

SEGMENT_ORDER = (
    "mask",
    "model",
    "pose",
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
FIXED_V5_BIAS_BYTES = 223
FIXED_V5_REGION_BYTES = 273
EXPECTED_PR85_ARCHIVE_BYTES = 236_328
EXPECTED_PR85_ARCHIVE_SHA256 = (
    "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
)
EXPECTED_PR85_MASK_BYTES = 159_011


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _u24le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def _read_single_member_archive(archive: Path, *, member: str) -> tuple[bytes, dict[str, Any]]:
    with ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member]:
            raise QMA9ContractError(f"{archive} must contain exactly member {member!r}; got {names!r}")
        info = infos[0]
        raw = zf.read(info)
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_file(archive)
    member_sha = sha256_bytes(raw)
    return raw, {
        "path": _repo_rel(archive),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "expected_archive_bytes": EXPECTED_PR85_ARCHIVE_BYTES,
        "expected_archive_sha256": EXPECTED_PR85_ARCHIVE_SHA256,
        "known_public_pr85_v5_match": (
            archive_bytes == EXPECTED_PR85_ARCHIVE_BYTES
            and archive_sha == EXPECTED_PR85_ARCHIVE_SHA256
        ),
        "member_name": info.filename,
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": member_sha,
        "zip_compress_type": int(info.compress_type),
    }


def parse_pr85_v5_segments(raw: bytes) -> dict[str, tuple[int, bytes]]:
    if len(raw) < 24:
        raise QMA9ContractError("PR85 bundle is too short for the 24-byte v5 micro header")
    lengths = {
        "mask": _u24le(raw, 0),
        "model": _u24le(raw, 3),
        "pose": _u24le(raw, 6),
        "post": _u24le(raw, 9),
        "shift": _u24le(raw, 12),
        "frac": _u24le(raw, 15),
        "frac2": _u24le(raw, 18),
        "frac3": _u24le(raw, 21),
        "bias": FIXED_V5_BIAS_BYTES,
        "region": FIXED_V5_REGION_BYTES,
    }
    pos = 24
    segments: dict[str, tuple[int, bytes]] = {}
    for name in SEGMENT_ORDER[:-1]:
        size = int(lengths[name])
        if size <= 0:
            raise QMA9ContractError(f"invalid PR85 segment length for {name}: {size}")
        end = pos + size
        if end > len(raw):
            raise QMA9ContractError(f"truncated PR85 segment {name}")
        segments[name] = (pos, raw[pos:end])
        pos = end
    if pos >= len(raw):
        raise QMA9ContractError("PR85 bundle is missing randmulti tail")
    segments["randmulti"] = (pos, raw[pos:])
    return segments


def _load_expected_mask_from_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("segments", []):
        if row.get("name") == "mask":
            return {
                "profile_json": _repo_rel(path),
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
    raise QMA9ContractError(f"{path} does not contain a PR85 mask segment row")


def _compile_cpp_decoder(source: Path, output_dir: Path) -> tuple[Path, list[str]]:
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise QMA9ContractError("no C++ compiler found on PATH for PR85 runtime decoder")
    binary = output_dir / "pr85_qma9_token_decoder"
    cmd = [compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise QMA9ContractError(f"PR85 QMA9 decoder compile failed: {proc.stderr[-4000:]}")
    return binary, cmd


def _run_cpp_decode(*, source: Path, qma9_path: Path, raw_path: Path, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    binary, compile_cmd = _compile_cpp_decoder(source, output_dir)
    run_cmd = [str(binary), "decode", str(qma9_path), str(raw_path)]
    proc = subprocess.run(
        run_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
    )
    if proc.returncode != 0:
        raise QMA9ContractError(f"PR85 QMA9 decoder run failed: {proc.stderr[-4000:]}")
    return {
        "implementation": "pr85_replay_cpp_decode",
        "source": _repo_rel(source),
        "compile_command": [_repo_rel(Path(compile_cmd[0])) or compile_cmd[0], *compile_cmd[1:]],
        "run_command": [_repo_rel(binary), "decode", _repo_rel(qma9_path), _repo_rel(raw_path)],
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _decode_tokens(
    *,
    qma9_payload: bytes,
    qma9_path: Path,
    raw_path: Path,
    implementation: str,
    cpp_decoder: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if implementation == "python":
        decoded = decode_qma9_mask(qma9_payload)
        raw_path.write_bytes(decoded.data)
        return {
            "implementation": "src/tac/qma9_range_mask_contract.py::decode_qma9_mask",
            "storage_order": decoded.storage_order,
            "decode_header": asdict(decoded.header),
        }
    if implementation == "cpp":
        detail = _run_cpp_decode(
            source=cpp_decoder,
            qma9_path=qma9_path,
            raw_path=raw_path,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        detail["storage_order"] = "frame_major_header_width_by_header_height"
        return detail
    raise QMA9ContractError(f"unknown decode implementation: {implementation}")


def _profile_raw_tokens(raw_path: Path, *, shape: tuple[int, int, int]) -> dict[str, Any]:
    raw = raw_path.read_bytes()
    expected = shape[0] * shape[1] * shape[2]
    if len(raw) != expected:
        raise QMA9ContractError(f"decoded token bytes {len(raw)} != expected tensor elements {expected}")
    counts = Counter(raw)
    invalid = sorted(int(value) for value in counts if value > 4)
    observed_min = min(counts) if counts else None
    observed_max = max(counts) if counts else None
    return {
        "path": _repo_rel(raw_path),
        "dtype": "uint8",
        "semantic": "QMA9 decoded mask class token ids",
        "shape": list(shape),
        "element_count": expected,
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "range_contract": {"min": 0, "max": 4, "sentinel_internal_only": 5},
        "observed_range": {"min": observed_min, "max": observed_max},
        "class_counts": {str(key): int(counts.get(key, 0)) for key in range(5)},
        "invalid_symbol_values": invalid,
    }


def build_profile(
    *,
    archive: Path,
    member: str,
    profile_json: Path | None,
    output_dir: Path,
    output_json: Path,
    extract_raw_tokens: bool,
    decode_implementation: str,
    cpp_decoder: Path,
    cpp_timeout_seconds: int,
    verify_reencode: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_bundle, archive_identity = _read_single_member_archive(archive, member=member)
    segments = parse_pr85_v5_segments(raw_bundle)
    mask_offset, mask_payload = segments["mask"]
    qma9_header = parse_qma9_header(mask_payload)
    expected_mask = _load_expected_mask_from_profile(profile_json)
    qma9_path = output_dir / "pr85_mask_segment.qma9"
    qma9_path.write_bytes(mask_payload)

    segment_rows = []
    for name in SEGMENT_ORDER:
        offset, data = segments[name]
        segment_rows.append(
            {
                "name": name,
                "offset": offset,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "magic_hex": data[:8].hex(),
                "is_mask_token_source": name == "mask",
            }
        )

    mask_identity = {
        "name": "mask",
        "offset": mask_offset,
        "bytes": len(mask_payload),
        "sha256": sha256_bytes(mask_payload),
        "expected_public_pr85_mask_bytes": EXPECTED_PR85_MASK_BYTES,
        "expected_profile": expected_mask,
        "matches_expected_public_pr85_mask_bytes": len(mask_payload) == EXPECTED_PR85_MASK_BYTES,
        "matches_profile_json": (
            expected_mask is not None
            and len(mask_payload) == expected_mask["bytes"]
            and sha256_bytes(mask_payload) == expected_mask["sha256"]
        ),
        "extracted_qma9_path": _repo_rel(qma9_path),
        "extracted_qma9_sha256": sha256_file(qma9_path),
        "header": asdict(qma9_header),
    }

    shape = (qma9_header.frame_count, qma9_header.width, qma9_header.height)
    raw_tokens: dict[str, Any] = {
        "extracted": False,
        "dtype": "uint8",
        "semantic": "QMA9 decoded mask class token ids",
        "shape": list(shape),
        "range_contract": {"min": 0, "max": 4, "sentinel_internal_only": 5},
        "reason_not_extracted": "run with --extract-raw-tokens to write the HPAC token-source tensor",
    }
    decode_detail: dict[str, Any] = {"performed": False}
    reencode_check: dict[str, Any] = {
        "performed": False,
        "byte_exact": False,
        "reason": "run with --verify-reencode after raw token extraction for independent QMA9 byte parity",
    }
    if extract_raw_tokens:
        raw_path = output_dir / "pr85_qma9_tokens_u8_storage_order.bin"
        decode_detail = _decode_tokens(
            qma9_payload=mask_payload,
            qma9_path=qma9_path,
            raw_path=raw_path,
            implementation=decode_implementation,
            cpp_decoder=cpp_decoder,
            output_dir=output_dir,
            timeout_seconds=cpp_timeout_seconds,
        )
        raw_tokens = _profile_raw_tokens(raw_path, shape=shape)
        raw_tokens["extracted"] = True
        raw_tokens["decode_implementation"] = decode_detail["implementation"]
        if verify_reencode:
            raw = raw_path.read_bytes()
            reencoded = encode_qma9_mask(
                raw,
                frame_count=qma9_header.frame_count,
                width=qma9_header.width,
                height=qma9_header.height,
            )
            reencode_path = output_dir / "pr85_qma9_reencoded_from_tokens.qma9"
            reencode_path.write_bytes(reencoded)
            reencode_check = {
                "performed": True,
                "path": _repo_rel(reencode_path),
                "bytes": len(reencoded),
                "sha256": sha256_bytes(reencoded),
                "byte_exact": reencoded == mask_payload,
                "source_mask_sha256": sha256_bytes(mask_payload),
            }

    exactness = {
        "mask_segment_byte_exact": True,
        "mask_segment_exact_basis": "single-member PR85 archive slice plus profile/known identity checks",
        "raw_tensor_extracted": bool(raw_tokens.get("extracted")),
        "raw_tensor_exact": bool(raw_tokens.get("extracted")) and not raw_tokens.get("invalid_symbol_values"),
        "raw_tensor_exact_basis": (
            "decoded from charged QMA9 bytes using the selected deterministic decoder"
            if raw_tokens.get("extracted")
            else "not proven because raw token extraction was not requested"
        ),
        "qma9_reencode_byte_exact": bool(reencode_check.get("byte_exact")),
    }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "evidence_grade": "empirical/local_token_source_profile",
        "archive_identity": archive_identity,
        "bundle_format": "pr85_v5_micro_24bit_lengths_fixed_bias_region",
        "segments": segment_rows,
        "mask_segment_identity": mask_identity,
        "token_source": raw_tokens,
        "decode": decode_detail,
        "reencode_check": reencode_check,
        "exactness": exactness,
        "next_gates_for_entropy_code_replacement": [
            {
                "gate": "fit_pr85_entropy_model_on_this_token_source",
                "required": True,
                "status": "blocked_until_model_artifact_exists",
            },
            {
                "gate": "decode_entropy_stream_to_same_token_sha256",
                "required": True,
                "status": "blocked_until_candidate_entropy_stream_exists",
                "expected_token_sha256": raw_tokens.get("sha256"),
            },
            {
                "gate": "archive_replacement_changes_only_mask_stream_and_runtime_decoder",
                "required": True,
                "status": "blocked_until_byte_closed_candidate_archive_exists",
            },
            {
                "gate": "runtime_output_parity_before_exact_eval",
                "required": True,
                "status": "blocked_until_candidate_runtime_consumes_archive_bytes",
            },
            {
                "gate": "dispatch_claim_before_any_remote_or_gpu_eval",
                "required": True,
                "status": "not_unlocked_by_this_profile",
            },
        ],
        "dispatch_unlocked": False,
        "dispatch_unlock_reason": "This artifact profiles/extracts token source only; no replacement coder, archive parity, runtime parity, or dispatch claim exists.",
    }
    _write_json(output_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--member", default="x")
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--extract-raw-tokens", action="store_true")
    parser.add_argument("--decode-implementation", choices=("cpp", "python"), default="cpp")
    parser.add_argument("--cpp-decoder", type=Path, default=DEFAULT_CPP_DECODER)
    parser.add_argument("--cpp-timeout-seconds", type=int, default=300)
    parser.add_argument("--verify-reencode", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = build_profile(
        archive=args.archive,
        member=args.member,
        profile_json=args.profile_json,
        output_dir=args.output_dir,
        output_json=args.output_json,
        extract_raw_tokens=bool(args.extract_raw_tokens),
        decode_implementation=args.decode_implementation,
        cpp_decoder=args.cpp_decoder,
        cpp_timeout_seconds=int(args.cpp_timeout_seconds),
        verify_reencode=bool(args.verify_reencode),
    )
    print(f"wrote {args.output_json}")
    print(
        "score_claim=false dispatch_performed=false "
        f"raw_tensor_extracted={str(profile['exactness']['raw_tensor_extracted']).lower()} "
        f"dispatch_unlocked={str(profile['dispatch_unlocked']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
