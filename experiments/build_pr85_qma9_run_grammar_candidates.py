#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR85 QMA9 QRG1 row-run grammar byte-screen candidates.

This is a planning/local-byte-screen path only. It reads the exact PR85 QMA9
token source, compiles it into deterministic QRG1 row-run payloads, verifies
local decode-to-token SHA parity, and writes ``candidate_summary.json``. The
current robust runtime does not support QRG1, so every candidate remains
dispatch-locked and no GPU, scorer, remote job, or score claim is performed.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import parse_pr85_bundle  # noqa: E402
from tac.qma9_range_mask_contract import parse_qma9_header  # noqa: E402
from tac.qma9_run_grammar import (  # noqa: E402
    DEFAULT_QRG1_MODE_NAMES,
    ORIGINAL_VIDEO_BYTES,
    QMA9RunGrammarError,
    encode_qrg1_run_grammar,
    decode_qrg1_run_grammar,
    parse_modes,
    qrg1_runtime_custody_contract,
    sha256_bytes,
    sha256_file,
)


TOOL = "experiments/build_pr85_qma9_run_grammar_candidates.py"
SCHEMA = "pr85_qma9_run_grammar_candidates_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qma9_run_grammar_candidates_20260504_worker"

EXPECTED_PR85_ARCHIVE_BYTES = 236_328
EXPECTED_PR85_ARCHIVE_SHA256 = (
    "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
)
EXPECTED_PR85_QMA9_BYTES = 159_011
EXPECTED_PR85_QMA9_SHA256 = (
    "4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179"
)
EXPECTED_PR85_TOKEN_SHA256 = (
    "c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a"
)
EXPECTED_PR85_TOKEN_BYTES = 117_964_800
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QMA9RunGrammarError(f"source PR85 archive is missing: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise QMA9RunGrammarError(f"PR85 archive must contain exactly ['x']; got {names!r}")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise QMA9RunGrammarError("PR85 archive member 'x' must be ZIP_STORED")
        raw = zf.read(info)
    archive_bytes = int(path.stat().st_size)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": archive_bytes,
            "archive_sha256": sha256_file(path),
            "member_name": "x",
            "member_bytes": len(raw),
            "member_sha256": sha256_bytes(raw),
            "zip_overhead_bytes": archive_bytes - len(raw),
        },
        raw,
    )


def _validate_source_anchor(
    source_meta: Mapping[str, Any],
    *,
    expected_archive_sha256: str | None,
    expected_archive_bytes: int | None,
) -> None:
    if expected_archive_bytes is not None and int(source_meta["archive_bytes"]) != int(expected_archive_bytes):
        raise QMA9RunGrammarError(
            f"source archive bytes {source_meta['archive_bytes']} != expected {expected_archive_bytes}"
        )
    if expected_archive_sha256 is not None and str(source_meta["archive_sha256"]) != str(expected_archive_sha256):
        raise QMA9RunGrammarError(
            f"source archive sha256 {source_meta['archive_sha256']} != expected {expected_archive_sha256}"
        )


def _validate_token_source(
    token_source: Path,
    *,
    expected_token_sha256: str,
    expected_token_bytes: int | None,
) -> tuple[bytes, dict[str, Any]]:
    if not token_source.is_file():
        raise QMA9RunGrammarError(f"decoded token source is missing: {token_source}")
    raw = token_source.read_bytes()
    token_sha = sha256_bytes(raw)
    if expected_token_bytes is not None and len(raw) != int(expected_token_bytes):
        raise QMA9RunGrammarError(
            f"decoded token source bytes {len(raw)} != expected {expected_token_bytes}"
        )
    if token_sha != expected_token_sha256:
        raise QMA9RunGrammarError(
            f"decoded token source sha256 {token_sha} != expected {expected_token_sha256}"
        )
    return raw, {
        "path": _repo_rel(token_source),
        "bytes": len(raw),
        "sha256": token_sha,
        "expected_bytes": expected_token_bytes,
        "expected_sha256": expected_token_sha256,
        "dtype": "uint8",
        "semantic": "PR85 QMA9 decoded mask class token ids",
    }


def _zip_member_bytes(member_name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.date_time = FIXED_ZIP_TIMESTAMP
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _candidate_record(
    *,
    mode_name: str,
    payload: bytes,
    encoded_stats: Mapping[str, Any],
    decoded_header: Mapping[str, Any],
    decoded_stats: Mapping[str, Any],
    decoded_sha256: str,
    token_sha256: str,
    source_qma9_bytes: int,
    source_archive_bytes: int,
    write_payloads: bool,
    out_dir: Path,
) -> dict[str, Any]:
    delta_mask_bytes = len(payload) - int(source_qma9_bytes)
    runtime_contract = qrg1_runtime_custody_contract()
    local_byte_win = delta_mask_bytes < 0
    token_parity = decoded_sha256 == token_sha256
    if not token_parity:
        raise QMA9RunGrammarError(
            f"QRG1 mode {mode_name} decoded token sha256 {decoded_sha256} != expected {token_sha256}"
        )

    payload_path = None
    if write_payloads:
        payload_file = out_dir / "payloads" / f"{mode_name}.qrg1"
        payload_file.parent.mkdir(parents=True, exist_ok=True)
        payload_file.write_bytes(payload)
        payload_path = _repo_rel(payload_file)

    rejection_reasons: list[str] = []
    if local_byte_win:
        rejection_reasons.append("byte_positive_qrg1_payload_requires_runtime_support_before_archive_dispatch")
    else:
        rejection_reasons.append("no_local_byte_win_vs_pr85_qma9_159011B")
    rejection_reasons.append("robust_current_runtime_does_not_decode_qrg1")

    return {
        "candidate_id": mode_name,
        "mode": mode_name,
        "payload_magic": "QRG1",
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "payload_path": payload_path,
        "delta_bytes_vs_pr85_qma9_159011B": delta_mask_bytes,
        "projected_archive_bytes_if_other_streams_unchanged": int(source_archive_bytes) + delta_mask_bytes,
        "projected_archive_delta_bytes_vs_pr85": delta_mask_bytes,
        "rate_score_delta_if_components_unchanged_formula_only": (
            delta_mask_bytes * 25.0 / ORIGINAL_VIDEO_BYTES
        ),
        "payload_changed_vs_source_qma9": True,
        "archive_relevant_state_change": True,
        "local_byte_win": local_byte_win,
        "runtime_supported": False,
        "dispatch_unlocked": False,
        "safe_for_remote_dispatch": False,
        "archive": None,
        "archive_rejection_reasons": rejection_reasons,
        "token_parity": {
            "verified": token_parity,
            "decoded_token_sha256": decoded_sha256,
            "expected_token_sha256": token_sha256,
            "method": "src/tac/qma9_run_grammar.py decode_qrg1_run_grammar",
        },
        "runtime_custody_contract": runtime_contract,
        "encoded_stats": dict(encoded_stats),
        "decoded_header": dict(decoded_header),
        "decoded_stats": dict(decoded_stats),
    }


def build_run_grammar_candidates(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    token_source: Path = DEFAULT_TOKEN_SOURCE,
    out_dir: Path = DEFAULT_OUT_DIR,
    modes: tuple[str, ...] = DEFAULT_QRG1_MODE_NAMES,
    expected_archive_sha256: str | None = EXPECTED_PR85_ARCHIVE_SHA256,
    expected_archive_bytes: int | None = EXPECTED_PR85_ARCHIVE_BYTES,
    expected_qma9_bytes: int | None = EXPECTED_PR85_QMA9_BYTES,
    expected_qma9_sha256: str | None = EXPECTED_PR85_QMA9_SHA256,
    expected_token_sha256: str = EXPECTED_PR85_TOKEN_SHA256,
    expected_token_bytes: int | None = EXPECTED_PR85_TOKEN_BYTES,
    write_payloads: bool = False,
) -> dict[str, Any]:
    """Run the PR85 QRG1 row-run byte screen and write summary JSON."""

    parsed_modes = parse_modes(modes)
    source_meta, source_bundle_raw = _read_single_x_archive(archive)
    _validate_source_anchor(
        source_meta,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_bytes=expected_archive_bytes,
    )
    token_raw, token_meta = _validate_token_source(
        token_source,
        expected_token_sha256=expected_token_sha256,
        expected_token_bytes=expected_token_bytes,
    )
    bundle = parse_pr85_bundle(source_bundle_raw)
    source_mask = bytes(bundle.segments["mask"])
    source_qma9 = parse_qma9_header(source_mask)
    source_qma9_sha = sha256_bytes(source_mask)
    if expected_qma9_bytes is not None and len(source_mask) != int(expected_qma9_bytes):
        raise QMA9RunGrammarError(
            f"source QMA9 segment bytes {len(source_mask)} != expected {expected_qma9_bytes}"
        )
    if expected_qma9_sha256 is not None and source_qma9_sha != expected_qma9_sha256:
        raise QMA9RunGrammarError(
            f"source QMA9 segment sha256 {source_qma9_sha} != expected {expected_qma9_sha256}"
        )
    if int(source_qma9.decoded_mask_bytes) != int(token_meta["bytes"]):
        raise QMA9RunGrammarError(
            f"QMA9 decoded bytes {source_qma9.decoded_mask_bytes} != token source bytes {token_meta['bytes']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for mode in parsed_modes:
        encoded = encode_qrg1_run_grammar(
            token_raw,
            frame_count=source_qma9.frame_count,
            width=source_qma9.width,
            height=source_qma9.height,
            mode=mode,
        )
        decoded = decode_qrg1_run_grammar(encoded.payload)
        candidates.append(
            _candidate_record(
                mode_name=mode.name,
                payload=encoded.payload,
                encoded_stats=encoded.stats,
                decoded_header=asdict(decoded.header),
                decoded_stats=decoded.stats,
                decoded_sha256=decoded.sha256,
                token_sha256=str(token_meta["sha256"]),
                source_qma9_bytes=len(source_mask),
                source_archive_bytes=int(source_meta["archive_bytes"]),
                write_payloads=write_payloads,
                out_dir=out_dir,
            )
        )

    best_payload = min(candidates, key=lambda row: int(row["payload_bytes"])) if candidates else None
    byte_positive = [row for row in candidates if bool(row["local_byte_win"])]
    best_byte_positive = (
        min(byte_positive, key=lambda row: int(row["delta_bytes_vs_pr85_qma9_159011B"]))
        if byte_positive
        else None
    )
    runtime_supported_byte_positive: list[dict[str, Any]] = []

    blockers = [
        "robust_current_runtime_does_not_accept_qrg1_run_grammar",
        "dispatch_locked_until_qrg1_runtime_output_parity_and_fresh_lane_claim",
    ]
    if not byte_positive:
        blockers.extend(
            [
                "no_byte_positive_qrg1_row_run_candidate",
                "qma9_adaptive9bin_beats_screened_row_run_payloads",
            ]
        )
    else:
        blockers.append("byte_positive_qrg1_candidate_runtime_unsupported")

    runtime_requirements = qrg1_runtime_custody_contract()["required_runtime_changes_before_dispatch"]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "dispatch_unlocked": False,
        "source_archive": source_meta,
        "token_source": token_meta,
        "source_qma9": {
            "segment_bytes": len(source_mask),
            "segment_sha256": source_qma9_sha,
            "reference_label": "PR85 QMA9 159011B",
            "header": asdict(source_qma9),
            "bundle_format": bundle.format,
            "bundle_header_bytes": bundle.header_bytes,
        },
        "mode_count": len(parsed_modes),
        "candidate_count": len(candidates),
        "byte_positive_candidate_count": len(byte_positive),
        "runtime_supported_byte_positive_candidate_count": len(runtime_supported_byte_positive),
        "best_payload_candidate": best_payload,
        "best_byte_positive_candidate": best_byte_positive,
        "best_runtime_supported_candidate": None,
        "best_bytes_vs_pr85_qma9_159011B": {
            "reference_bytes": EXPECTED_PR85_QMA9_BYTES,
            "best_payload_bytes": best_payload["payload_bytes"] if best_payload else None,
            "best_delta_bytes": (
                best_payload["delta_bytes_vs_pr85_qma9_159011B"] if best_payload else None
            ),
            "best_mode": best_payload["mode"] if best_payload else None,
        },
        "candidates": candidates,
        "blockers": sorted(set(blockers)),
        "required_runtime_changes_if_byte_positive": runtime_requirements,
        "fail_closed": {
            "emitted": True,
            "reason": (
                "no screened QRG1 row-run payload beat the PR85 QMA9 159011B mask segment"
                if not byte_positive
                else "best local byte-positive QRG1 payload is not supported by current robust runtime"
            ),
            "source_qma9_payload_bytes": len(source_mask),
            "best_payload_mode": best_payload["mode"] if best_payload else None,
            "best_payload_bytes": best_payload["payload_bytes"] if best_payload else None,
            "best_delta_bytes_vs_pr85_qma9_159011B": (
                best_payload["delta_bytes_vs_pr85_qma9_159011B"] if best_payload else None
            ),
            "best_byte_positive_mode": best_byte_positive["mode"] if best_byte_positive else None,
            "best_runtime_supported_mode": None,
            "byte_economics": {
                "archive_bytes_change_equals_mask_payload_delta_for_pr85_v5": True,
                "points_per_byte": 25.0 / ORIGINAL_VIDEO_BYTES,
                "best_rate_score_delta_if_components_unchanged": (
                    best_payload["rate_score_delta_if_components_unchanged_formula_only"]
                    if best_payload
                    else None
                ),
            },
        },
        "dispatch_gate": {
            "dispatch_unlocked": False,
            "candidate_archive_exists": False,
            "safe_for_remote_dispatch": False,
            "exact_eval_dispatch_allowed": False,
            "reason": "Local-only QRG1 byte screen; current runtime cannot decode QRG1.",
        },
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--token-source", type=Path, default=DEFAULT_TOKEN_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--modes", default=",".join(DEFAULT_QRG1_MODE_NAMES))
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_PR85_ARCHIVE_SHA256)
    parser.add_argument("--expected-archive-bytes", type=int, default=EXPECTED_PR85_ARCHIVE_BYTES)
    parser.add_argument("--expected-qma9-bytes", type=int, default=EXPECTED_PR85_QMA9_BYTES)
    parser.add_argument("--expected-qma9-sha256", default=EXPECTED_PR85_QMA9_SHA256)
    parser.add_argument("--expected-token-sha256", default=EXPECTED_PR85_TOKEN_SHA256)
    parser.add_argument("--expected-token-bytes", type=int, default=EXPECTED_PR85_TOKEN_BYTES)
    parser.add_argument(
        "--allow-source-archive-mismatch",
        action="store_true",
        help="Disable PR85 archive/QMA9 byte/SHA anchor checks for synthetic local tests only.",
    )
    parser.add_argument(
        "--write-payloads",
        action="store_true",
        help="Write screened .qrg1 payloads under the output directory. Summary JSON is always written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_run_grammar_candidates(
        archive=args.archive,
        token_source=args.token_source,
        out_dir=args.out_dir,
        modes=tuple(mode.name for mode in parse_modes(args.modes)),
        expected_archive_sha256=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_sha256,
        expected_archive_bytes=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_bytes,
        expected_qma9_bytes=None if args.allow_source_archive_mismatch else args.expected_qma9_bytes,
        expected_qma9_sha256=None
        if args.allow_source_archive_mismatch
        else args.expected_qma9_sha256,
        expected_token_sha256=args.expected_token_sha256,
        expected_token_bytes=args.expected_token_bytes,
        write_payloads=args.write_payloads,
    )
    print(f"wrote {args.out_dir / 'candidate_summary.json'}")
    best = summary["best_payload_candidate"]
    best_positive = summary["best_byte_positive_candidate"]
    print(
        "score_claim=false dispatch_performed=false "
        f"dispatch_unlocked={str(summary['dispatch_unlocked']).lower()} "
        f"byte_positive_candidate_count={summary['byte_positive_candidate_count']} "
        f"runtime_supported_byte_positive_candidate_count={summary['runtime_supported_byte_positive_candidate_count']}"
    )
    if best is None:
        print("best_payload_candidate=None")
    else:
        print(
            "best_payload_candidate "
            f"mode={best['mode']} bytes={best['payload_bytes']} "
            f"delta_vs_pr85_qma9_159011B={best['delta_bytes_vs_pr85_qma9_159011B']} "
            f"sha256={best['payload_sha256']}"
        )
    if best_positive is not None:
        print(
            "best_byte_positive_candidate "
            f"mode={best_positive['mode']} bytes={best_positive['payload_bytes']} "
            "runtime_supported=false"
        )
    if summary["blockers"]:
        print("blockers=" + ";".join(summary["blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
