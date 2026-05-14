#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR85 QMA9 alternate-grammar/table-reduction screens.

This is a local-only byte screen. It compiles the audited PR85 replay
range-mask codec, re-encodes the exact PR85 QMA9 token source through selected
alternate grammar/table modes, records strict token-parity custody from the
codec self-decode check, and fails closed for every candidate the current
``submissions/robust_current`` runtime cannot decode.

No scorer, GPU, remote job, lane dispatch, or score claim is performed.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import pack_pr85_bundle, parse_pr85_bundle  # noqa: E402
from tac.qma9_alt_grammar import (  # noqa: E402
    DEFAULT_ALT_GRAMMAR_MODES,
    ORIGINAL_VIDEO_BYTES,
    QMA9AltGrammarError,
    compile_local_codec,
    mode_family,
    parse_modes,
    run_codec_mode,
    runtime_custody_contract,
    sha256_bytes,
    sha256_file,
)
from tac.qma9_range_mask_contract import parse_qma9_header  # noqa: E402


TOOL = "experiments/build_pr85_qma9_alt_grammar_candidates.py"
SCHEMA = "pr85_qma9_alt_grammar_candidates_v1"
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_TOKEN_SOURCE = (
    REPO_ROOT
    / "experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/"
    "pr85_qma9_tokens_u8_storage_order.bin"
)
DEFAULT_REPLAY_CODEC_CPP = (
    REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/replay_submission/range_mask_codec.cpp"
)
DEFAULT_LIVE_RUNTIME_CPP = REPO_ROOT / "submissions/robust_current/range_mask_codec.cpp"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_qma9_alt_grammar_candidates_20260504"
DEFAULT_NATIVE_SUMMARY = (
    REPO_ROOT
    / "experiments/results/pr85_qma9_native_grammar_candidates_20260504_orchestrator/"
    "candidate_summary.json"
)
DEFAULT_OPPORTUNITY_MATRIX = (
    REPO_ROOT
    / "experiments/results/pr85_full_stack_opportunity_matrix_20260504_orchestrator_final2/"
    "pr85_full_stack_opportunity_matrix.json"
)

EXPECTED_PR85_ARCHIVE_BYTES = 236_328
EXPECTED_PR85_ARCHIVE_SHA256 = (
    "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
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
        raise QMA9AltGrammarError(f"source PR85 archive is missing: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise QMA9AltGrammarError(f"PR85 archive must contain exactly ['x']; got {names!r}")
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise QMA9AltGrammarError("PR85 archive member 'x' must be ZIP_STORED")
        raw = zf.read(info)
    archive_bytes = path.stat().st_size
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
        raise QMA9AltGrammarError(
            f"source archive bytes {source_meta['archive_bytes']} != expected {expected_archive_bytes}"
        )
    if expected_archive_sha256 is not None and str(source_meta["archive_sha256"]) != str(expected_archive_sha256):
        raise QMA9AltGrammarError(
            f"source archive sha256 {source_meta['archive_sha256']} != expected {expected_archive_sha256}"
        )


def _validate_token_source(
    token_source: Path,
    *,
    expected_token_sha256: str,
    expected_token_bytes: int | None,
) -> dict[str, Any]:
    if not token_source.is_file():
        raise QMA9AltGrammarError(f"decoded token source is missing: {token_source}")
    token_bytes = token_source.stat().st_size
    token_sha = sha256_file(token_source)
    if expected_token_bytes is not None and token_bytes != int(expected_token_bytes):
        raise QMA9AltGrammarError(
            f"decoded token source bytes {token_bytes} != expected {expected_token_bytes}"
        )
    if token_sha != expected_token_sha256:
        raise QMA9AltGrammarError(
            f"decoded token source sha256 {token_sha} != expected {expected_token_sha256}"
        )
    return {
        "path": _repo_rel(token_source),
        "bytes": token_bytes,
        "sha256": token_sha,
        "expected_bytes": expected_token_bytes,
        "expected_sha256": expected_token_sha256,
    }


def _zip_member_bytes(member_name: str, payload: bytes) -> bytes:
    if member_name != "x":
        raise QMA9AltGrammarError(f"candidate archive member must be 'x', got {member_name!r}")
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.date_time = FIXED_ZIP_TIMESTAMP
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(info, payload)
    return buffer.getvalue()


def _write_runtime_supported_archive(path: Path, x_payload: bytes) -> dict[str, Any]:
    first = _zip_member_bytes("x", x_payload)
    second = _zip_member_bytes("x", x_payload)
    if first != second:
        raise QMA9AltGrammarError("deterministic ZIP construction produced non-identical bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(first)
    return {
        "path": _repo_rel(path),
        "bytes": len(first),
        "sha256": sha256_bytes(first),
        "member_name": "x",
        "member_bytes": len(x_payload),
        "member_sha256": sha256_bytes(x_payload),
        "zip_overhead_bytes": len(first) - len(x_payload),
        "zip_storage": "stored",
        "deterministic_rewrite_identical": True,
    }


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": _repo_rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _pack_with_candidate_mask(bundle_raw: bytes, candidate_mask: bytes) -> tuple[bytes, str]:
    bundle = parse_pr85_bundle(bundle_raw)
    segments = dict(bundle.segments)
    segments["mask"] = candidate_mask
    header_mode = "explicit_30" if bundle.format == "pr85_explicit_30byte_lengths" else "v5"
    return pack_pr85_bundle(segments, header_mode=header_mode), header_mode


def _candidate_record(
    *,
    run: Any,
    source_mask: bytes,
    source_bundle_raw: bytes,
    source_archive_bytes: int,
    source_archive_sha256: str,
    live_runtime_cpp: Path,
    replay_codec_cpp: Path,
    payload_bytes: bytes,
    out_dir: Path,
) -> dict[str, Any]:
    delta_mask = int(run.payload_bytes) - len(source_mask)
    projected_archive_bytes = int(source_archive_bytes) + delta_mask
    payload_changed = payload_bytes != source_mask
    contract = runtime_custody_contract(
        mode=run.mode,
        payload_magic=run.payload_magic,
        live_runtime_cpp=live_runtime_cpp,
        replay_codec_cpp=replay_codec_cpp,
    )
    local_byte_win = delta_mask < 0 and payload_changed
    runtime_supported = bool(contract["live_runtime_supported"])
    archive_meta = None
    archive_rejection_reasons: list[str] = []
    if local_byte_win and runtime_supported:
        candidate_x, header_mode = _pack_with_candidate_mask(source_bundle_raw, payload_bytes)
        archive_path = out_dir / run.mode / "archive.zip"
        archive_meta = _write_runtime_supported_archive(archive_path, candidate_x)
        if int(archive_meta["bytes"]) >= int(source_archive_bytes):
            archive_path.unlink(missing_ok=True)
            archive_meta = None
            archive_rejection_reasons.append("runtime_supported_candidate_archive_not_byte_positive")
    elif local_byte_win:
        archive_rejection_reasons.append("byte_positive_payload_requires_runtime_edit_before_archive_dispatch")
        header_mode = None
    else:
        archive_rejection_reasons.append("no_local_byte_win")
        header_mode = None

    record = {
        "candidate_id": run.mode,
        "mode": run.mode,
        "mode_family": mode_family(run.mode),
        "payload_path": _repo_rel(run.payload_path),
        "payload_magic": run.payload_magic,
        "payload_bytes": run.payload_bytes,
        "payload_sha256": run.payload_sha256,
        "bitstream_bytes": run.bitstream_bytes,
        "model_or_header_bytes": run.model_bytes,
        "delta_bytes_vs_source_qma9": delta_mask,
        "projected_archive_bytes_if_other_streams_unchanged": projected_archive_bytes,
        "projected_archive_delta_bytes_vs_pr85": delta_mask,
        "rate_score_delta_if_components_unchanged_formula_only": delta_mask * 25.0 / ORIGINAL_VIDEO_BYTES,
        "payload_changed_vs_source_qma9": payload_changed,
        "archive_relevant_state_change": payload_changed,
        "token_parity": {
            "verified": True,
            "method": "compiled replay codec self-decode equality before payload write",
            "raw_bytes": run.raw_bytes,
        },
        "runtime_custody_contract": contract,
        "local_byte_win": local_byte_win,
        "runtime_supported": runtime_supported,
        "dispatch_unlocked": False,
        "safe_for_remote_dispatch": False,
        "source_archive_sha256": source_archive_sha256,
        "archive": archive_meta,
        "pr85_bundle_header_mode_if_archive_written": header_mode,
        "archive_rejection_reasons": archive_rejection_reasons,
        "codec_stdout": dict(run.stdout_json),
        "run_command": list(run.run_command),
    }
    if archive_meta is not None:
        record["archive_delta_bytes_vs_pr85"] = int(archive_meta["bytes"]) - int(source_archive_bytes)
    return record


def build_alt_grammar_candidates(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    token_source: Path = DEFAULT_TOKEN_SOURCE,
    out_dir: Path = DEFAULT_OUT_DIR,
    replay_codec_cpp: Path = DEFAULT_REPLAY_CODEC_CPP,
    live_runtime_cpp: Path = DEFAULT_LIVE_RUNTIME_CPP,
    modes: tuple[str, ...] = DEFAULT_ALT_GRAMMAR_MODES,
    expected_archive_sha256: str | None = EXPECTED_PR85_ARCHIVE_SHA256,
    expected_archive_bytes: int | None = EXPECTED_PR85_ARCHIVE_BYTES,
    expected_token_sha256: str = EXPECTED_PR85_TOKEN_SHA256,
    expected_token_bytes: int | None = EXPECTED_PR85_TOKEN_BYTES,
    timeout_seconds_per_mode: int = 300,
    native_summary: Path = DEFAULT_NATIVE_SUMMARY,
    opportunity_matrix: Path = DEFAULT_OPPORTUNITY_MATRIX,
) -> dict[str, Any]:
    """Run the PR85 alternate grammar/table-reduction screen and write JSON."""

    modes = parse_modes(modes)
    source_meta, source_bundle_raw = _read_single_x_archive(archive)
    _validate_source_anchor(
        source_meta,
        expected_archive_sha256=expected_archive_sha256,
        expected_archive_bytes=expected_archive_bytes,
    )
    token_meta = _validate_token_source(
        token_source,
        expected_token_sha256=expected_token_sha256,
        expected_token_bytes=expected_token_bytes,
    )
    bundle = parse_pr85_bundle(source_bundle_raw)
    source_mask = bytes(bundle.segments["mask"])
    source_qma9 = parse_qma9_header(source_mask)
    if int(source_qma9.decoded_mask_bytes) != int(token_meta["bytes"]):
        raise QMA9AltGrammarError(
            f"QMA9 decoded bytes {source_qma9.decoded_mask_bytes} != token source bytes {token_meta['bytes']}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    codec_meta: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="pr85-qma9-alt-") as tmp:
        codec = compile_local_codec(replay_codec_cpp, Path(tmp))
        codec_meta = {
            "source": _repo_rel(codec.source),
            "source_sha256": codec.source_sha256,
            "compile_command": list(codec.compile_command),
        }
        for mode in modes:
            payload_path = payload_dir / f"{mode}.bin"
            run = run_codec_mode(
                codec,
                raw_tokens=token_source,
                frame_count=source_qma9.frame_count,
                width=source_qma9.width,
                height=source_qma9.height,
                output_path=payload_path,
                mode=mode,
                timeout_seconds=timeout_seconds_per_mode,
            )
            payload = payload_path.read_bytes()
            candidates.append(
                _candidate_record(
                    run=run,
                    source_mask=source_mask,
                    source_bundle_raw=source_bundle_raw,
                    source_archive_bytes=int(source_meta["archive_bytes"]),
                    source_archive_sha256=str(source_meta["archive_sha256"]),
                    live_runtime_cpp=live_runtime_cpp,
                    replay_codec_cpp=replay_codec_cpp,
                    payload_bytes=payload,
                    out_dir=out_dir,
                )
            )

    alt_candidates = [row for row in candidates if row["mode"] != "adaptive9bin"]
    best_alt = min(alt_candidates, key=lambda row: int(row["payload_bytes"])) if alt_candidates else None
    byte_positive = [row for row in alt_candidates if bool(row["local_byte_win"])]
    runtime_supported_byte_positive = [
        row for row in byte_positive if bool(row["runtime_custody_contract"]["live_runtime_supported"])
    ]
    baseline_records = [row for row in candidates if row["mode"] == "adaptive9bin"]
    baseline_record = baseline_records[0] if baseline_records else None

    blockers: list[str] = []
    if not byte_positive:
        blockers.append("no_byte_positive_alt_grammar_candidate")
    if not runtime_supported_byte_positive:
        blockers.append("no_runtime_supported_byte_positive_alt_grammar_candidate")
    blockers.append("robust_current_runtime_accepts_only_qma9_adaptive9bin")
    if best_alt is not None and int(best_alt["delta_bytes_vs_source_qma9"]) >= 0:
        blockers.append("qma9_adaptive9bin_remains_smallest_observed_full_stream")
    if baseline_record is not None and int(baseline_record["payload_bytes"]) != len(source_mask):
        blockers.append("local_replay_adaptive9bin_does_not_reproduce_source_payload_size")

    best_byte_positive = (
        min(byte_positive, key=lambda row: int(row["delta_bytes_vs_source_qma9"]))
        if byte_positive
        else None
    )
    best_runtime_supported = (
        min(runtime_supported_byte_positive, key=lambda row: int(row["projected_archive_delta_bytes_vs_pr85"]))
        if runtime_supported_byte_positive
        else None
    )
    runtime_change_template = [
        "port the selected alternate grammar decoder into submissions/robust_current/range_mask_codec.cpp",
        "use a distinct magic or charged mode-id so QMA9 adaptive9bin custody cannot silently misdecode alternate bits",
        "extend robust_current inflate/unpack mask detection for the selected magic",
        "add exact raw-token SHA parity tests before any GPU eval",
        "record updated runtime tree SHA in contest_auth_eval provenance",
    ]

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
        "input_artifacts": {
            "runtime_supported_trim_screen": _artifact_meta(native_summary),
            "opportunity_matrix": _artifact_meta(opportunity_matrix),
        },
        "source_qma9": {
            "segment_bytes": len(source_mask),
            "segment_sha256": sha256_bytes(source_mask),
            "header": asdict(source_qma9),
            "bundle_format": bundle.format,
            "bundle_header_bytes": bundle.header_bytes,
        },
        "local_screen_codec": codec_meta,
        "live_runtime": {
            "source": _repo_rel(live_runtime_cpp),
            "source_sha256": sha256_file(live_runtime_cpp) if live_runtime_cpp.is_file() else None,
            "accepted_current_contract": "QMA9 adaptive9bin only",
        },
        "mode_count": len(candidates),
        "candidate_count": len(candidates),
        "alternate_candidate_count": len(alt_candidates),
        "byte_positive_candidate_count": len(byte_positive),
        "runtime_supported_byte_positive_candidate_count": len(runtime_supported_byte_positive),
        "best_alt_candidate": best_alt,
        "best_byte_positive_candidate": best_byte_positive,
        "best_runtime_supported_candidate": best_runtime_supported,
        "candidates": candidates,
        "blockers": sorted(set(blockers)),
        "fail_closed": {
            "emitted": best_runtime_supported is None,
            "reason": (
                "no screened alternate grammar was byte-positive; runtime edits are not economically justified"
                if not byte_positive
                else "byte-positive alternate grammar is not supported by robust_current runtime"
            ),
            "source_qma9_payload_bytes": len(source_mask),
            "best_alt_mode": best_alt["mode"] if best_alt else None,
            "best_alt_payload_bytes": best_alt["payload_bytes"] if best_alt else None,
            "best_alt_delta_bytes_vs_source_qma9": (
                best_alt["delta_bytes_vs_source_qma9"] if best_alt else None
            ),
            "best_byte_positive_mode": best_byte_positive["mode"] if best_byte_positive else None,
            "best_runtime_supported_mode": best_runtime_supported["mode"] if best_runtime_supported else None,
            "required_runtime_change_if_future_byte_win_exists": runtime_change_template,
            "byte_economics": {
                "archive_bytes_change_equals_mask_payload_delta_for_pr85_v5": True,
                "points_per_byte": 25.0 / ORIGINAL_VIDEO_BYTES,
                "best_alt_rate_score_delta_if_components_unchanged": (
                    best_alt["rate_score_delta_if_components_unchanged_formula_only"] if best_alt else None
                ),
            },
        },
        "dispatch_gate": {
            "dispatch_unlocked": False,
            "candidate_archive_exists": best_runtime_supported is not None,
            "safe_for_remote_dispatch": False,
            "exact_eval_dispatch_allowed": False,
            "reason": "Local-only task; no GPU/remote dispatch or score claim is allowed.",
        },
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--token-source", type=Path, default=DEFAULT_TOKEN_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-codec-cpp", type=Path, default=DEFAULT_REPLAY_CODEC_CPP)
    parser.add_argument("--live-runtime-cpp", type=Path, default=DEFAULT_LIVE_RUNTIME_CPP)
    parser.add_argument("--modes", default=",".join(DEFAULT_ALT_GRAMMAR_MODES))
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_PR85_ARCHIVE_SHA256)
    parser.add_argument("--expected-archive-bytes", type=int, default=EXPECTED_PR85_ARCHIVE_BYTES)
    parser.add_argument("--expected-token-sha256", default=EXPECTED_PR85_TOKEN_SHA256)
    parser.add_argument("--expected-token-bytes", type=int, default=EXPECTED_PR85_TOKEN_BYTES)
    parser.add_argument("--timeout-seconds-per-mode", type=int, default=300)
    parser.add_argument("--native-summary", type=Path, default=DEFAULT_NATIVE_SUMMARY)
    parser.add_argument("--opportunity-matrix", type=Path, default=DEFAULT_OPPORTUNITY_MATRIX)
    parser.add_argument(
        "--allow-source-archive-mismatch",
        action="store_true",
        help="Disable PR85 archive byte/SHA anchor checks for synthetic local tests only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_alt_grammar_candidates(
        archive=args.archive,
        token_source=args.token_source,
        out_dir=args.out_dir,
        replay_codec_cpp=args.replay_codec_cpp,
        live_runtime_cpp=args.live_runtime_cpp,
        modes=parse_modes(args.modes),
        expected_archive_sha256=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_sha256,
        expected_archive_bytes=None
        if args.allow_source_archive_mismatch
        else args.expected_archive_bytes,
        expected_token_sha256=args.expected_token_sha256,
        expected_token_bytes=args.expected_token_bytes,
        timeout_seconds_per_mode=args.timeout_seconds_per_mode,
        native_summary=args.native_summary,
        opportunity_matrix=args.opportunity_matrix,
    )
    print(f"wrote {args.out_dir / 'candidate_summary.json'}")
    best_alt = summary["best_alt_candidate"]
    print(
        "score_claim=false dispatch_performed=false "
        f"dispatch_unlocked={str(summary['dispatch_unlocked']).lower()} "
        f"byte_positive_candidate_count={summary['byte_positive_candidate_count']} "
        f"runtime_supported_byte_positive_candidate_count={summary['runtime_supported_byte_positive_candidate_count']}"
    )
    if best_alt is None:
        print("best_alt_candidate=None")
    else:
        print(
            "best_alt_candidate "
            f"mode={best_alt['mode']} bytes={best_alt['payload_bytes']} "
            f"delta_vs_source_qma9={best_alt['delta_bytes_vs_source_qma9']} "
            f"sha256={best_alt['payload_sha256']}"
        )
    if summary["blockers"]:
        print("blockers=" + ";".join(summary["blockers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
