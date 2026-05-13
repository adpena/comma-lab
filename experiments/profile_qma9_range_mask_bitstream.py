#!/usr/bin/env python3
"""Profile PR81/QMA9 range-mask arithmetic coding at bitstream level.

This tool is local forensics only. It reads charged archive bytes, runs a
bounded pure-Python decoder trace for state checkpoints, optionally compiles a
local C++ profiler for full-stream entropy/cost accounting, and emits planning
candidate screens. It never invokes the contest scorer and never dispatches GPU
or remote jobs.
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.qma9_range_mask_contract import (
    QMA9_FIRST_ROW_SPECIALIZATION_MODES,
    PayloadSegment,
    QMA9ContractError,
    QMA9Split,
    analyze_qma9_vertical_block_copy_opportunities,
    decode_qma9_first_row_specialization_mask,
    decode_qma9_mask,
    decode_qma9_prefix_frames,
    decode_qma9_vertical_block_escape_mask,
    encode_qma9_first_row_specialization_mask,
    encode_qma9_mask,
    encode_qma9_vertical_block_escape_mask,
    parse_qma9_first_row_specialization_header,
    parse_qma9_header,
    parse_qma9_vertical_block_escape_header,
    read_single_member_zip,
    sha256_bytes,
    sha256_file,
    slice_payload_segments,
    split_qma9_pr81_payload,
    trace_qma9_prefix,
)

TOOL = "experiments/profile_qma9_range_mask_bitstream.py"
SCHEMA = "qma9_range_mask_bitstream_forensics_v1"
DEFAULT_PR81_DIR = REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex"
DEFAULT_PR81_ARCHIVE = DEFAULT_PR81_DIR / "archive.zip"
DEFAULT_PR81_INFLATE = DEFAULT_PR81_DIR / "replay_submission/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/qma9_range_mask_deconstruction_20260503_codex"
DEFAULT_CPP_PROFILER = REPO_ROOT / "experiments/qma9_range_mask_cpp_profiler.cpp"
ORIGINAL_VIDEO_BYTES = 37_545_489
PUBLIC_QMA9_QP1_POSE_STREAM_BYTES = 899
PUBLIC_QMA9_RANGE_MASK_BYTES = 159_011
PUBLIC_QMA9_REORDERED_MODEL_BYTES = 55_725


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


def _eval_int_expr(node: ast.AST, env: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_int_expr(node.left, env)
        right = _eval_int_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
    raise ValueError(f"unsupported integer expression: {ast.dump(node)}")


def parse_split_constants(path: Path) -> dict[str, int]:
    """Read PR81/PR84-style payload constants from an inflate.py or profile JSON."""

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        constants = payload.get("split_constants", payload)
        if not isinstance(constants, dict):
            raise QMA9ContractError(f"{path} does not contain split_constants")
        env = {str(key): int(value) for key, value in constants.items() if str(key).isupper()}
        wanted = {
            "RANGE_MASK_BYTES",
            "SPLIT_MODEL_REORDERED_BYTES",
            "POSE_STREAM_BYTES",
        }
        missing = sorted(wanted.difference(env))
        if missing:
            raise QMA9ContractError(f"{path} is missing split constants: {missing}")
        env.setdefault(
            "PACKED_PAYLOAD_BYTES",
            env["RANGE_MASK_BYTES"]
            + env["SPLIT_MODEL_REORDERED_BYTES"]
            + env["POSE_STREAM_BYTES"]
            + int(env.get("ROUTER_ACTION_BYTES", 0)),
        )
        env.setdefault("ROUTER_ACTION_BYTES", 0)
        return {key: int(env[key]) for key in sorted(env) if key in wanted | {"PACKED_PAYLOAD_BYTES", "ROUTER_ACTION_BYTES"}}

    tree = ast.parse(path.read_text())
    env: dict[str, int] = {}
    # Catalog #168 fix 2026-05-12: handle both bare Assign and AnnAssign forms.
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        elif (isinstance(stmt, ast.AnnAssign)
              and stmt.value is not None
              and isinstance(stmt.target, ast.Name)):
            target_name = stmt.target.id
            value_node = stmt.value
        else:
            continue
        try:
            env[target_name] = _eval_int_expr(value_node, env)
        except ValueError:
            continue

    if "SPLIT_MODEL_REORDERED_BYTES" not in env:
        reordered_parts = (
            "SPLIT_MODEL_PACKED_REORDERED_BR_BYTES",
            "SPLIT_MODEL_SCALES_REORDERED_BR_BYTES",
            "SPLIT_MODEL_TAIL_REORDERED_BR_BYTES",
        )
        if all(key in env for key in reordered_parts):
            env["SPLIT_MODEL_REORDERED_BYTES"] = sum(env[key] for key in reordered_parts)

    if "POSE_STREAM_BYTES" not in env:
        if (
            env.get("RANGE_MASK_BYTES") == PUBLIC_QMA9_RANGE_MASK_BYTES
            and env.get("SPLIT_MODEL_REORDERED_BYTES") == PUBLIC_QMA9_REORDERED_MODEL_BYTES
        ):
            env["POSE_STREAM_BYTES"] = PUBLIC_QMA9_QP1_POSE_STREAM_BYTES
        else:
            raise QMA9ContractError(
                f"{path} is missing POSE_STREAM_BYTES and does not match the known "
                "PR84/QMA9 fixed-slice shape"
            )

    env.setdefault("ROUTER_ACTION_BYTES", 0)
    env.setdefault(
        "PACKED_PAYLOAD_BYTES",
        env.get("RANGE_MASK_BYTES", 0)
        + env.get("SPLIT_MODEL_REORDERED_BYTES", 0)
        + env.get("POSE_STREAM_BYTES", 0)
        + env.get("ROUTER_ACTION_BYTES", 0),
    )

    wanted = {
        "RANGE_MASK_BYTES",
        "SPLIT_MODEL_REORDERED_BYTES",
        "POSE_STREAM_BYTES",
        "PACKED_PAYLOAD_BYTES",
    }
    missing = sorted(wanted.difference(env))
    if missing:
        raise QMA9ContractError(f"{path} is missing split constants: {missing}")
    return {key: int(env[key]) for key in sorted(wanted | {"ROUTER_ACTION_BYTES"})}


def _split_qma9_public_payload(payload: bytes, constants: dict[str, int]) -> QMA9Split:
    """Split PR81/PR84 QMA9 payloads, with PR84's missing router allowed."""

    range_mask_bytes = int(constants["RANGE_MASK_BYTES"])
    model_bytes = int(constants["SPLIT_MODEL_REORDERED_BYTES"])
    pose_bytes = int(constants["POSE_STREAM_BYTES"])
    base_bytes = range_mask_bytes + model_bytes + pose_bytes
    declared_router_bytes = int(constants.get("ROUTER_ACTION_BYTES", 0))
    remaining = len(payload) - base_bytes
    if remaining < 0:
        raise QMA9ContractError(
            f"payload has {len(payload)} bytes but QMA9/model/pose constants require {base_bytes}"
        )
    if remaining not in (0, declared_router_bytes):
        raise QMA9ContractError(
            f"payload has {remaining} trailing router/action bytes; expected 0 or {declared_router_bytes}"
        )
    if remaining == declared_router_bytes and declared_router_bytes > 0:
        return split_qma9_pr81_payload(
            payload,
            range_mask_bytes=range_mask_bytes,
            model_bytes=model_bytes,
            pose_bytes=pose_bytes,
            router_bytes=declared_router_bytes,
        )

    specs = [
        ("range_mask.qma9", range_mask_bytes, "qma9_adaptive9_binary_range_mask"),
        ("split_model_reordered.br_bundle", model_bytes, "brotli_reordered_qzs3_model_bundle"),
        ("optimized_poses.qp1.br", pose_bytes, "brotli_qp1_pose_stream"),
    ]
    segments = slice_payload_segments(payload, specs)
    return QMA9Split(
        range_mask=payload[segments[0].offset:segments[0].offset + segments[0].size_bytes],
        model=payload[segments[1].offset:segments[1].offset + segments[1].size_bytes],
        pose=payload[segments[2].offset:segments[2].offset + segments[2].size_bytes],
        router=b"",
        segments=(
            *segments,
            PayloadSegment(
                name="router_actions.3bit",
                offset=base_bytes,
                size_bytes=0,
                sha256=sha256_bytes(b""),
                codec="absent_pr84_no_router_actions",
            ),
        ),
    )


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _compile_cpp_profiler(source: Path, output_dir: Path) -> tuple[Path, list[str]]:
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise QMA9ContractError("no C++ compiler found on PATH")
    binary = output_dir / "qma9_range_mask_cpp_profiler"
    cmd = [compiler, "-O3", "-std=c++17", str(source), "-o", str(binary)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise QMA9ContractError(f"C++ profiler compile failed: {proc.stderr[-2000:]}")
    return binary, cmd


def _run_cpp_profile(*, source: Path, qma9_path: Path, output_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qma9_cpp_build_") as tmp:
        build_dir = Path(tmp)
        binary, compile_cmd = _compile_cpp_profiler(source, build_dir)
        cpp_json = output_dir / "qma9_range_mask_cpp_full_profile.json"
        run_cmd = [str(binary), str(qma9_path), str(cpp_json)]
        proc = subprocess.run(
            run_cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            raise QMA9ContractError(f"C++ profiler run failed: {proc.stderr[-2000:]}")
    profile = json.loads(cpp_json.read_text())
    return {
        "status": "ok",
        "compile_command": compile_cmd,
        "run_command": [_repo_rel(Path(run_cmd[0])) or run_cmd[0], _repo_rel(qma9_path), _repo_rel(cpp_json)],
        "profile_json": _repo_rel(cpp_json),
        "profile": profile,
    }


def _top_indices(values: list[float], *, limit: int = 12) -> list[dict[str, Any]]:
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:limit]
    return [{"index": int(index), "estimated_bytes": float(value)} for index, value in ranked]


def _candidate_screens(cpp_profile: dict[str, Any] | None, qma9_bytes: int) -> list[dict[str, Any]]:
    if not cpp_profile:
        return []
    stage_counts = cpp_profile["stage_counts"]
    stage_bits = cpp_profile["stage_estimated_bits"]
    fallback = cpp_profile["fallback_extra_predictor_matches"]
    fallback_events = int(fallback["fallback_events"])
    avg_class_bits = float(stage_bits["class_fallback"]) / max(1, int(stage_counts["class_fallback"]))
    screens: list[dict[str, Any]] = []
    for name in ("up_left", "up_right", "prev_right", "prev_down", "up2", "left2"):
        matches = int(fallback[name])
        added_gate_cost_bits = fallback_events
        gross_saved_bits = matches * avg_class_bits
        net_saved_bits = gross_saved_bits - added_gate_cost_bits
        screens.append(
            {
                "candidate_id": f"qma9_extra_{name}_fallback_gate",
                "type": "deterministic_decoder_algorithm_change",
                "target": "range_mask.qma9",
                "evidence_grade": "empirical/planning_only",
                "dispatchable": False,
                "reason_not_dispatchable": "requires encoder/runtime implementation, raw output parity, archive closure, and CUDA auth eval",
                "profile_signal": {
                    "fallback_events": fallback_events,
                    "matching_fallback_pixels": matches,
                    "avg_current_class_fallback_bits": avg_class_bits,
                    "rough_net_saved_bits_after_one_binary_gate": net_saved_bits,
                    "rough_net_saved_bytes": net_saved_bits / 8.0,
                },
            }
        )
    run = cpp_profile["run_structure"]
    avg_bits_per_pixel = float(cpp_profile["estimated_model_bits"]) / max(1, int(cpp_profile["header"]["decoded_pixels"]))
    long_runs = int(run["buckets"]["16_31"]) + int(run["buckets"]["32_63"]) + int(run["buckets"]["64_plus"])
    long_run_tail_pixels_lower_bound = int(run["buckets"]["16_31"]) * 15 + int(run["buckets"]["32_63"]) * 31 + int(run["buckets"]["64_plus"]) * 63
    screens.append(
        {
            "candidate_id": "qma9_horizontal_run_escape_len16",
            "type": "deterministic_run_length_escape",
            "target": "range_mask.qma9",
            "evidence_grade": "empirical/planning_only",
            "dispatchable": False,
            "reason_not_dispatchable": "requires finite policy search over escape thresholds plus runtime parity",
            "profile_signal": {
                "long_runs_len16_plus": long_runs,
                "tail_pixels_lower_bound": long_run_tail_pixels_lower_bound,
                "avg_model_bits_per_pixel": avg_bits_per_pixel,
                "rough_gross_tail_bits": long_run_tail_pixels_lower_bound * avg_bits_per_pixel,
            },
        }
    )
    screens.append(
        {
            "candidate_id": "qma9_context_backoff_prune_sparse9",
            "type": "adaptive_context_model_change",
            "target": "range_mask.qma9",
            "evidence_grade": "empirical/planning_only",
            "dispatchable": False,
            "reason_not_dispatchable": "requires A/B bitstream rebuild and exact raw mask equality before scoring",
            "profile_signal": {
                "top_contexts": cpp_profile["top_contexts"][:8],
                "range_mask_payload_bytes": qma9_bytes,
                "hypothesis": "many 9-neighbor contexts are cold; deterministic lower-order backoff may reduce cold-start bits without side bytes",
            },
        }
    )
    return sorted(
        screens,
        key=lambda row: float(row.get("profile_signal", {}).get("rough_net_saved_bytes", 0.0)),
        reverse=True,
    )


def _decode_raw_for_byte_search(
    *,
    qma9_payload: bytes,
    frames: int | None,
    raw_mask_path: Path | None,
) -> tuple[bytes, dict[str, Any]]:
    header = parse_qma9_header(qma9_payload)
    if raw_mask_path is not None:
        raw = raw_mask_path.read_bytes()
        expected_frames = header.frame_count if frames is None else int(frames)
        expected_bytes = expected_frames * header.width * header.height
        if len(raw) != expected_bytes:
            raise QMA9ContractError(f"{raw_mask_path} has {len(raw)} raw bytes, expected {expected_bytes}")
        return raw, {
            "source": _repo_rel(raw_mask_path),
            "source_kind": "external_raw_mask_storage_order",
            "decoded_by_tool": False,
            "frames": expected_frames,
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
        }

    if frames is None:
        decoded = decode_qma9_mask(qma9_payload)
        return decoded.data, {
            "source": "archive_range_mask.qma9",
            "source_kind": "full_qma9_decode",
            "decoded_by_tool": True,
            "frames": header.frame_count,
            "raw_bytes": len(decoded.data),
            "raw_sha256": decoded.sha256,
        }

    decoded_frames = int(frames)
    raw = decode_qma9_prefix_frames(qma9_payload, frame_count=decoded_frames)
    return raw, {
        "source": "archive_range_mask.qma9",
        "source_kind": "prefix_qma9_decode",
        "decoded_by_tool": True,
        "frames": decoded_frames,
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
    }


def _candidate_decision(
    *,
    mode_id: str,
    payload: bytes | None,
    raw: bytes,
    decoded_raw: bytes | None,
    reference_qma9_bytes: int,
    source_archive_bytes: int,
    archive_relevant_state_change: bool,
    planning_only_reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    parity_ok = decoded_raw == raw if decoded_raw is not None else False
    rejection_reasons: list[str] = []
    if error:
        rejection_reasons.append("candidate_encode_or_decode_failed")
    if not parity_ok:
        rejection_reasons.append("raw_mask_parity_failed")
    if not archive_relevant_state_change:
        rejection_reasons.append("no_archive_relevant_state_change")

    candidate_bytes = len(payload) if payload is not None else None
    delta_vs_reference = candidate_bytes - reference_qma9_bytes if candidate_bytes is not None else None
    byte_screen_win = delta_vs_reference is not None and delta_vs_reference < 0
    if not byte_screen_win:
        rejection_reasons.append("no_local_byte_screen_win")

    accepted = parity_ok and archive_relevant_state_change and byte_screen_win and error is None
    no_op_status = "state_changed" if archive_relevant_state_change else "no_archive_relevant_state_change"
    if not accepted and not rejection_reasons:
        rejection_reasons.append("not_accepted")
    return {
        "mode_id": mode_id,
        "payload_bytes": candidate_bytes,
        "payload_sha256": sha256_bytes(payload) if payload is not None else None,
        "no_op_status": no_op_status,
        "raw_mask_parity": bool(parity_ok),
        "archive_relevant_state_change": bool(archive_relevant_state_change),
        "delta_bytes_vs_source_range_mask": delta_vs_reference,
        "projected_archive_bytes_if_other_streams_unchanged": (
            source_archive_bytes + delta_vs_reference if delta_vs_reference is not None else None
        ),
        "rate_score_delta_if_components_unchanged": (
            delta_vs_reference * 25.0 / ORIGINAL_VIDEO_BYTES if delta_vs_reference is not None else None
        ),
        "accepted_for_exact_eval_candidate": bool(accepted),
        "selectable": bool(accepted),
        "rejection_reasons": [] if accepted else rejection_reasons,
        "planning_only_reason": planning_only_reason,
        "error": error,
    }


def build_byte_search_profile(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    frames: int | None,
    qmb1_block_widths: tuple[int, ...],
    qmf1_first_row_modes: tuple[int, ...] = tuple(sorted(QMA9_FIRST_ROW_SPECIALIZATION_MODES)),
    raw_mask_path: Path | None = None,
    write_candidates: bool = True,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, custody = read_single_member_zip(archive_path)
    split = _split_qma9_public_payload(payload, constants)
    source_header = parse_qma9_header(split.range_mask)
    if frames is not None:
        frames = int(frames)
        if frames <= 0:
            raise QMA9ContractError("byte-search frames must be positive or all")
        if frames > source_header.frame_count:
            raise QMA9ContractError(f"byte-search frames {frames} exceeds source frame count {source_header.frame_count}")

    raw, raw_source = _decode_raw_for_byte_search(qma9_payload=split.range_mask, frames=frames, raw_mask_path=raw_mask_path)
    search_frames = int(raw_source["frames"])
    reference_bytes = source_header.packed_bytes if frames is None else None
    if reference_bytes is None:
        reference_payload = encode_qma9_mask(raw, frame_count=search_frames, width=source_header.width, height=source_header.height)
        reference_bytes = len(reference_payload)
    else:
        reference_payload = split.range_mask

    search_dir = output_dir / "byte_search"
    candidate_dir = search_dir / "candidates"
    if write_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    if raw_mask_path is None and frames is None:
        baseline_decoded = raw
    else:
        baseline_decoded = (
            decode_qma9_mask(reference_payload).data
            if frames is None
            else decode_qma9_prefix_frames(reference_payload, frame_count=search_frames)
        )
    baseline = _candidate_decision(
        mode_id="qma9_reference_reencode",
        payload=reference_payload,
        raw=raw,
        decoded_raw=baseline_decoded,
        reference_qma9_bytes=reference_bytes,
        source_archive_bytes=custody.archive_bytes,
        archive_relevant_state_change=sha256_bytes(reference_payload) != source_header.payload_sha256,
        planning_only_reason="reference mode only; exact-eval selection requires a non-reference local byte win",
    )
    baseline["mode_family"] = "qma9"
    baseline["role"] = "reference"
    candidates.append(baseline)
    if write_candidates:
        (candidate_dir / f"{baseline['mode_id']}.qma9").write_bytes(reference_payload)

    for block_width in qmb1_block_widths:
        mode_id = f"qmb1_vertical_block_escape_bw{int(block_width)}"
        record: dict[str, Any]
        try:
            candidate_payload = encode_qma9_vertical_block_escape_mask(
                raw,
                frame_count=search_frames,
                width=source_header.width,
                height=source_header.height,
                block_width=int(block_width),
            )
            decoded = decode_qma9_vertical_block_escape_mask(candidate_payload)
            candidate_header = parse_qma9_vertical_block_escape_header(candidate_payload)
            opportunities = analyze_qma9_vertical_block_copy_opportunities(
                raw,
                frame_count=search_frames,
                width=source_header.width,
                height=source_header.height,
                block_width=int(block_width),
            )
            record = _candidate_decision(
                mode_id=mode_id,
                payload=candidate_payload,
                raw=raw,
                decoded_raw=decoded.data,
                reference_qma9_bytes=reference_bytes,
                source_archive_bytes=custody.archive_bytes,
                archive_relevant_state_change=int(opportunities["copied_blocks"]) > 0,
                planning_only_reason=(
                    "local byte screen only; runtime integration, lane claim, archive closure, "
                    "and CUDA auth eval are still required"
                ),
            )
            record.update(
                {
                    "mode_family": "qmb1_vertical_block_escape",
                    "header": asdict(candidate_header),
                    "opportunities": opportunities,
                }
            )
            if write_candidates:
                (candidate_dir / f"{mode_id}.qmb1").write_bytes(candidate_payload)
        except Exception as exc:  # pragma: no cover - exercised through manifest robustness, not expected path
            record = _candidate_decision(
                mode_id=mode_id,
                payload=None,
                raw=raw,
                decoded_raw=None,
                reference_qma9_bytes=reference_bytes,
                source_archive_bytes=custody.archive_bytes,
                archive_relevant_state_change=False,
                planning_only_reason="candidate failed before parity proof",
                error=str(exc),
            )
            record["mode_family"] = "qmb1_vertical_block_escape"
        candidates.append(record)

    for mode_id in qmf1_first_row_modes:
        mode_id = int(mode_id)
        mode_name = QMA9_FIRST_ROW_SPECIALIZATION_MODES.get(mode_id)
        candidate_mode_id = f"qmf1_first_row_{mode_id}_{mode_name or 'unknown'}"
        try:
            candidate_payload = encode_qma9_first_row_specialization_mask(
                raw,
                frame_count=search_frames,
                width=source_header.width,
                height=source_header.height,
                mode_id=mode_id,
            )
            decoded = decode_qma9_first_row_specialization_mask(candidate_payload)
            candidate_header = parse_qma9_first_row_specialization_header(candidate_payload)
            record = _candidate_decision(
                mode_id=candidate_mode_id,
                payload=candidate_payload,
                raw=raw,
                decoded_raw=decoded.data,
                reference_qma9_bytes=reference_bytes,
                source_archive_bytes=custody.archive_bytes,
                archive_relevant_state_change=sha256_bytes(candidate_payload) != source_header.payload_sha256,
                planning_only_reason=(
                    "local first-row/context specialization byte screen only; runtime integration, lane claim, "
                    "archive closure, and CUDA auth eval are still required"
                ),
            )
            record.update(
                {
                    "mode_family": "qmf1_first_row_specialization",
                    "header": asdict(candidate_header),
                    "specialization": {
                        "mode_name": candidate_header.mode_name,
                        "first_row_pixels": int(search_frames) * int(source_header.height),
                        "modified_rows_per_frame": 1,
                        "skips_static_up_gate": True,
                    },
                }
            )
            if write_candidates:
                (candidate_dir / f"{candidate_mode_id}.qmf1").write_bytes(candidate_payload)
        except Exception as exc:  # pragma: no cover - manifest robustness mirrors QMB1 path
            record = _candidate_decision(
                mode_id=candidate_mode_id,
                payload=None,
                raw=raw,
                decoded_raw=None,
                reference_qma9_bytes=reference_bytes,
                source_archive_bytes=custody.archive_bytes,
                archive_relevant_state_change=False,
                planning_only_reason="candidate failed before parity proof",
                error=str(exc),
            )
            record.update(
                {
                    "mode_family": "qmf1_first_row_specialization",
                    "specialization": {
                        "mode_name": mode_name,
                        "first_row_pixels": int(search_frames) * int(source_header.height),
                        "modified_rows_per_frame": 1,
                        "skips_static_up_gate": True,
                    },
                }
            )
        candidates.append(record)

    accepted = [row for row in candidates if row["accepted_for_exact_eval_candidate"]]
    accepted_sorted = sorted(accepted, key=lambda row: int(row["delta_bytes_vs_source_range_mask"]))
    manifest = {
        "schema": "qma9_range_mask_byte_search_profile_v1",
        "tool": TOOL,
        "evidence_grade": "empirical/local_byte_screen",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "segments": [asdict(segment) for segment in split.segments],
        "source_qma9_header": asdict(source_header),
        "raw_mask": raw_source,
        "mode_matrix": {
            "reference_mode": "qma9_reference_reencode",
            "qmb1_block_widths": [int(value) for value in qmb1_block_widths],
            "qmf1_first_row_modes": [int(value) for value in qmf1_first_row_modes],
            "candidate_count": len(candidates),
            "accepted_local_byte_wins": len(accepted_sorted),
        },
        "selection_guard": {
            "safe_to_choose_exact_eval_candidate_after_local_screen_win": bool(accepted_sorted),
            "required_before_dispatch": [
                "candidate accepted_for_exact_eval_candidate=true",
                "runtime consumes the candidate payload from archive bytes",
                "dispatch lane claim exists before any remote/GPU eval",
                "exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py",
            ],
        },
        "best_local_byte_win": accepted_sorted[0] if accepted_sorted else None,
        "candidates": candidates,
    }
    if write_candidates:
        _write_json(search_dir / "qma9_range_mask_byte_search_profile.json", manifest)
    return manifest


def build_profile(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    cpp_profiler: Path,
    pure_python_max_pixels: int,
    checkpoint_pixels: tuple[int, ...],
    skip_cpp_full: bool,
    cpp_timeout_seconds: int,
    run_byte_search: bool = False,
    byte_search_frames: int | None = None,
    qmb1_block_widths: tuple[int, ...] = (),
    qmf1_first_row_modes: tuple[int, ...] = tuple(sorted(QMA9_FIRST_ROW_SPECIALIZATION_MODES)),
    raw_mask_path: Path | None = None,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, custody = read_single_member_zip(archive_path)
    split = _split_qma9_public_payload(payload, constants)
    qma9 = parse_qma9_header(split.range_mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    qma9_path = output_dir / "range_mask_pr81.qma9"
    qma9_path.write_bytes(split.range_mask)
    py_trace = trace_qma9_prefix(
        split.range_mask,
        max_pixels=pure_python_max_pixels,
        checkpoint_pixels=checkpoint_pixels,
    )
    cpp_result: dict[str, Any] | None = None
    if not skip_cpp_full:
        cpp_result = _run_cpp_profile(
            source=cpp_profiler,
            qma9_path=qma9_path,
            output_dir=output_dir,
            timeout_seconds=cpp_timeout_seconds,
        )
    cpp_profile = cpp_result["profile"] if cpp_result and cpp_result.get("status") == "ok" else None
    profile = {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": "empirical/planning_only",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "segments": [asdict(segment) for segment in split.segments],
        "qma9_header": asdict(qma9),
        "extracted_qma9_path": _repo_rel(qma9_path),
        "extracted_qma9_sha256": sha256_file(qma9_path),
        "pure_python_prefix_trace": py_trace,
        "cpp_full_profile": cpp_result if cpp_result is not None else {"status": "skipped"},
        "cost_hotspots": {
            "top_frame_indices_by_estimated_bytes": _top_indices(cpp_profile["per_frame_estimated_bytes"]) if cpp_profile else [],
            "top_row_indices_by_estimated_bytes": _top_indices(cpp_profile["per_row_index_estimated_bytes"]) if cpp_profile else [],
        },
        "candidate_screens": _candidate_screens(cpp_profile, qma9.packed_bytes),
        "notes": [
            "Pure-Python trace records arithmetic decoder state on actual PR81 payload bytes.",
            "C++ full profile is local-only instrumentation of the same public QMA9 arithmetic model.",
            "Candidate screens are byte/entropy planning artifacts, not score or component claims.",
            f"Any future rate comparison must use 25 * archive_bytes / {ORIGINAL_VIDEO_BYTES} and exact CUDA auth eval.",
        ],
    }
    if run_byte_search:
        profile["byte_search"] = build_byte_search_profile(
            archive_path=archive_path,
            split_constants_path=split_constants_path,
            output_dir=output_dir,
            frames=byte_search_frames,
            qmb1_block_widths=qmb1_block_widths,
            qmf1_first_row_modes=qmf1_first_row_modes,
            raw_mask_path=raw_mask_path,
            write_candidates=True,
        )
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR81_ARCHIVE)
    parser.add_argument("--split-constants-py", type=Path, default=DEFAULT_PR81_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "qma9_range_mask_bitstream_profile.json")
    parser.add_argument("--cpp-profiler", type=Path, default=DEFAULT_CPP_PROFILER)
    parser.add_argument("--pure-python-max-pixels", type=int, default=512 * 384)
    parser.add_argument("--checkpoint-pixels", default="0,1,2,4095,65535,196607")
    parser.add_argument("--skip-cpp-full", action="store_true")
    parser.add_argument("--cpp-timeout-seconds", type=int, default=300)
    parser.add_argument("--run-byte-search", action="store_true")
    parser.add_argument(
        "--byte-search-frames",
        default="all",
        help="Frame count for local byte search, or 'all' for full-stream mask decode/re-encode.",
    )
    parser.add_argument("--qmb1-block-widths", default="2,4,8,16,32,64")
    parser.add_argument(
        "--qmf1-first-row-modes",
        default="1,2,3",
        help="Comma-separated local QMF1 first-row specialization mode ids, or empty to disable.",
    )
    parser.add_argument("--raw-mask", type=Path, default=None, help="Optional raw storage-order mask bytes to avoid QMA9 decode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    byte_search_frames = None if str(args.byte_search_frames).lower() == "all" else int(args.byte_search_frames)
    profile = build_profile(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        output_dir=args.output_dir,
        cpp_profiler=args.cpp_profiler,
        pure_python_max_pixels=args.pure_python_max_pixels,
        checkpoint_pixels=_parse_int_list(args.checkpoint_pixels),
        skip_cpp_full=args.skip_cpp_full,
        cpp_timeout_seconds=args.cpp_timeout_seconds,
        run_byte_search=args.run_byte_search,
        byte_search_frames=byte_search_frames,
        qmb1_block_widths=_parse_int_list(args.qmb1_block_widths),
        qmf1_first_row_modes=_parse_int_list(args.qmf1_first_row_modes),
        raw_mask_path=args.raw_mask,
    )
    _write_json(args.output_json, profile)
    print(f"wrote {args.output_json}")
    print(f"score_claim={profile['score_claim']} dispatch_performed={profile['dispatch_performed']}")
    print(f"qma9_bytes={profile['qma9_header']['packed_bytes']} candidate_screens={len(profile['candidate_screens'])}")
    if args.run_byte_search:
        byte_search = profile["byte_search"]
        print(
            "byte_search "
            f"candidates={byte_search['mode_matrix']['candidate_count']} "
            f"accepted_local_byte_wins={byte_search['mode_matrix']['accepted_local_byte_wins']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
