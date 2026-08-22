#!/usr/bin/env python3
"""Run the scorer-free RC1 terminal temporal-program VQ byte sweep.

Every materialized candidate, coder variant, receiver output, repeat, negative
control, and checkpoint is retained below the caller-provided APDataStore root.
The resulting archives are research shadow containers, not evaluator-runnable
submission archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.rc1_terminal_program_vq import (
    DX2_ARCHIVE_BYTES,
    DX2_ARCHIVE_SHA256,
    DX2_TOKEN_BYTES,
    DX2_TOKEN_SHA256,
    DX2_TOKEN_STREAM_BYTES,
    PAYLOAD_HEADER,
    SHADOW_HEADER,
    STRICT_SUB012_ARCHIVE_BYTES,
    EncodedVariant,
    RC1FormatError,
    TokenVQModel,
    build_payload,
    build_shadow_outer,
    canonicalize_model,
    decoded_sha256,
    encode_assignment_variants,
    encode_codebook_variants,
    extract_dx2_shadow_sections,
    fit_nested_debt_k_modes,
    iter_decoded_frames,
    parse_payload,
    parse_shadow_outer,
)

AXIS = "[macOS-CPU scorer-free real-coder token-representation n600]"
SOURCE_SHAPE = (600, 384, 512)
DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_rc1_rate_crush")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DEFAULT_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip")
RUNNER_SOURCE = Path(__file__).resolve()
MODULE_SOURCE = RUNNER_SOURCE.parents[1] / "src/tac/optimization/rc1_terminal_program_vq.py"


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def retain_producer_sources(out_dir: Path) -> list[dict[str, Any]]:
    """Retain the exact code bytes that produced a measurement run."""
    root = out_dir / "retained" / "producer_source"
    records = []
    for source in (RUNNER_SOURCE, MODULE_SOURCE):
        retained = root / source.name
        source_bytes = source.read_bytes()
        if retained.is_file():
            if retained.read_bytes() != source_bytes:
                raise RC1FormatError(
                    f"producer source changed for {out_dir}; use a fresh output directory"
                )
        else:
            atomic_bytes(retained, source_bytes)
        records.append({"workspace": file_record(source), "retained": file_record(retained)})
    return records


def persist_refused_mutation(
    path: Path,
    payload: bytes,
    offset: int,
    parser: Any,
) -> tuple[dict[str, Any], bool]:
    """Persist one deterministic one-bit mutation and require receiver refusal."""
    if not 0 <= offset < len(payload):
        raise RC1FormatError(f"mutation offset {offset} is outside payload length {len(payload)}")
    mutated = bytearray(payload)
    mutated[offset] ^= 1
    atomic_bytes(path, bytes(mutated))
    try:
        parser(bytes(mutated))
    except RC1FormatError:
        refused = True
    else:
        refused = False
    return file_record(path), refused


def deterministic_zip(path: Path, member: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_inputs(tokens_path: Path, archive_path: Path) -> None:
    tokens = file_record(tokens_path)
    archive = file_record(archive_path)
    if tokens["bytes"] != DX2_TOKEN_BYTES or tokens["sha256"] != DX2_TOKEN_SHA256:
        raise RC1FormatError(f"DX2 token custody differs: {tokens}")
    if archive["bytes"] != DX2_ARCHIVE_BYTES or archive["sha256"] != DX2_ARCHIVE_SHA256:
        raise RC1FormatError(f"DX2 archive custody differs: {archive}")


def storage_preflight(out_dir: Path, candidate_count: int) -> dict[str, Any]:
    resolved = out_dir.resolve()
    allowed = Path("/Volumes/APDataStore/pact").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise RC1FormatError("RC1 retained output must live under /Volumes/APDataStore/pact")
    usage = shutil.disk_usage(allowed)
    predicted = candidate_count * (DX2_TOKEN_BYTES + 64 * 1024 * 1024) + 512 * 1024 * 1024
    reserve = 8 * 1024 * 1024 * 1024
    if usage.free < predicted + reserve:
        raise RC1FormatError(
            f"storage preflight refused: free={usage.free}, predicted={predicted}, reserve={reserve}"
        )
    receipt = {
        "schema": "ddm_rc1_storage_preflight.v1",
        "path": str(resolved),
        "free_bytes": usage.free,
        "predicted_bytes": predicted,
        "reserve_bytes": reserve,
        "candidate_count": candidate_count,
        "passed": True,
    }
    atomic_json(out_dir / "STORAGE_PREFLIGHT.json", receipt)
    return receipt


def build_source_index(tokens_path: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = out_dir / "retained" / "source_index"
    programs_path = root / "unique_programs.u8"
    counts_path = root / "unique_counts.u32"
    inverse_path = root / "site_unique_ids.u32"
    manifest_path = root / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["files"]:
            if file_record(Path(record["path"])) != record:
                raise RC1FormatError("source-index checkpoint failed custody")
        programs = np.memmap(
            programs_path,
            dtype=np.uint8,
            mode="r",
            shape=(manifest["unique_programs"], SOURCE_SHAPE[0]),
        )
        counts = np.memmap(counts_path, dtype="<u4", mode="r", shape=(manifest["unique_programs"],))
        inverse = np.memmap(inverse_path, dtype="<u4", mode="r", shape=(SOURCE_SHAPE[1] * SOURCE_SHAPE[2],))
        return programs, counts, inverse

    root.mkdir(parents=True, exist_ok=True)
    source = np.memmap(tokens_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    from collections import Counter

    counter: Counter[bytes] = Counter()
    for y_start in range(0, SOURCE_SHAPE[1], 32):
        block = np.ascontiguousarray(source[:, y_start : y_start + 32, :].transpose(1, 2, 0)).reshape(-1, SOURCE_SHAPE[0])
        counter.update(map(bytes, block))
    keys = sorted(counter)
    key_to_index = {key: index for index, key in enumerate(keys)}
    unique = np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), SOURCE_SHAPE[0])
    counts = np.asarray([counter[key] for key in keys], dtype="<u4")
    inverse = np.empty(SOURCE_SHAPE[1] * SOURCE_SHAPE[2], dtype="<u4")
    cursor = 0
    for y_start in range(0, SOURCE_SHAPE[1], 32):
        block = np.ascontiguousarray(source[:, y_start : y_start + 32, :].transpose(1, 2, 0)).reshape(-1, SOURCE_SHAPE[0])
        for row in block:
            inverse[cursor] = key_to_index[bytes(row)]
            cursor += 1
    if cursor != inverse.size or int(counts.sum()) != inverse.size:
        raise RC1FormatError("source-index population accounting differs")
    atomic_bytes(programs_path, unique.tobytes())
    atomic_bytes(counts_path, counts.tobytes())
    atomic_bytes(inverse_path, inverse.tobytes())
    files = [file_record(path) for path in (programs_path, counts_path, inverse_path)]
    manifest = {
        "schema": "ddm_rc1_source_index.v1",
        "source": file_record(tokens_path),
        "source_shape": list(SOURCE_SHAPE),
        "unique_programs": len(keys),
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    return (
        np.memmap(programs_path, dtype=np.uint8, mode="r", shape=unique.shape),
        np.memmap(counts_path, dtype="<u4", mode="r", shape=counts.shape),
        np.memmap(inverse_path, dtype="<u4", mode="r", shape=inverse.shape),
    )


def persist_variants(root: Path, kind: str, variants: list[EncodedVariant]) -> list[dict[str, Any]]:
    records = []
    for variant in variants:
        path = root / "coder_race" / kind / f"{variant.method_id:03d}__{variant.name}.bin"
        atomic_bytes(path, variant.payload)
        record = file_record(path)
        record.update({"method_id": variant.method_id, "name": variant.name, "raw_bytes": variant.raw_bytes})
        records.append(record)
    return records


def token_diagnostics(
    source_path: Path,
    model: TokenVQModel,
    decoded_path: Path,
) -> dict[str, Any]:
    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = decoded_path.with_name(decoded_path.name + f".tmp.{os.getpid()}")
    decoded = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=SOURCE_SHAPE)
    confusion = np.zeros((5, 5), dtype=np.int64)
    per_frame = np.empty(SOURCE_SHAPE[0], dtype=np.int64)
    digest = hashlib.sha256()
    try:
        for time_index, frame in enumerate(iter_decoded_frames(model)):
            decoded[time_index] = frame
            digest.update(memoryview(np.ascontiguousarray(frame)))
            truth = np.asarray(source[time_index])
            per_frame[time_index] = np.count_nonzero(frame != truth)
            np.add.at(confusion, (truth.reshape(-1), frame.reshape(-1)), 1)
        decoded.flush()
        del decoded
        os.replace(temporary, decoded_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    total = int(per_frame.sum())
    class_iou = []
    for class_id in range(5):
        true_positive = int(confusion[class_id, class_id])
        union = int(confusion[class_id].sum() + confusion[:, class_id].sum() - true_positive)
        class_iou.append(true_positive / union if union else 1.0)
    return {
        "diagnostic_only": True,
        "source_token_hamming_mismatches": total,
        "source_token_hamming_fraction": total / DX2_TOKEN_BYTES,
        "source_token_agreement": 1.0 - total / DX2_TOKEN_BYTES,
        "per_frame_mismatch_min": int(per_frame.min()),
        "per_frame_mismatch_median": float(np.median(per_frame)),
        "per_frame_mismatch_max": int(per_frame.max()),
        "confusion_true_rows_predicted_columns": confusion.tolist(),
        "per_class_iou": class_iou,
        "decoded_token_sha256": digest.hexdigest(),
        "decoded_tokens": file_record(decoded_path),
    }


def build_candidate(
    *,
    target_k: int,
    iterations: int,
    programs: np.ndarray,
    counts: np.ndarray,
    inverse: np.ndarray,
    source_tokens: Path,
    base_archive: Path,
    out_dir: Path,
    producer_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = f"k{target_k:03d}_i{iterations}"
    root = out_dir / "retained" / "candidates" / candidate_id
    result_path = root / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if result.get("producer_sources") != producer_sources:
            raise RC1FormatError(f"candidate {candidate_id} producer source differs")
        for record in result["custody_files"]:
            if file_record(Path(record["path"])) != record:
                raise RC1FormatError(f"candidate {candidate_id} resume custody differs")
        return result

    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    assignments_path = root / "model" / ("assignments.u8" if target_k <= 256 else "assignments.u16le")
    codebook_path = root / "model" / "codebook.u8"
    model_checkpoint_path = root / "model" / "CHECKPOINT.json"
    if model_checkpoint_path.is_file():
        model_checkpoint = json.loads(model_checkpoint_path.read_text())
        if (
            model_checkpoint.get("target_k") != target_k
            or model_checkpoint.get("iterations") != iterations
            or file_record(assignments_path) != model_checkpoint.get("assignments")
            or file_record(codebook_path) != model_checkpoint.get("codebook")
        ):
            raise RC1FormatError(f"candidate {candidate_id} model checkpoint differs")
        model = TokenVQModel(
            assignments=np.memmap(
                assignments_path,
                dtype=np.uint8 if model_checkpoint["actual_k"] <= 256 else "<u2",
                mode="r",
                shape=SOURCE_SHAPE[1:],
            ),
            codebook=np.memmap(
                codebook_path,
                dtype=np.uint8,
                mode="r",
                shape=(model_checkpoint["actual_k"], SOURCE_SHAPE[0]),
            ),
        )
        model.validate()
        fit = model_checkpoint["fit"]
    else:
        codebook, unique_assignments, fit = fit_nested_debt_k_modes(
            programs,
            counts,
            target_k,
            iterations=iterations,
        )
        site_assignments = unique_assignments[np.asarray(inverse, dtype=np.uint32)].reshape(SOURCE_SHAPE[1:])
        assignment_dtype = np.uint8 if len(codebook) <= 256 else np.uint16
        model = canonicalize_model(
            TokenVQModel(
                assignments=np.ascontiguousarray(site_assignments, dtype=assignment_dtype),
                codebook=np.ascontiguousarray(codebook, dtype=np.uint8),
            )
        )
        atomic_bytes(assignments_path, model.assignments.tobytes())
        atomic_bytes(codebook_path, model.codebook.tobytes())
        model_checkpoint = {
            "schema": "ddm_rc1_model_checkpoint.v1",
            "target_k": target_k,
            "actual_k": model.codebook.shape[0],
            "iterations": iterations,
            "fit": fit,
            "assignments": file_record(assignments_path),
            "codebook": file_record(codebook_path),
        }
        atomic_json(model_checkpoint_path, model_checkpoint)

    assignment_variants = encode_assignment_variants(model.assignments, model.codebook.shape[0])
    codebook_variants = encode_codebook_variants(model.codebook)
    assignment_records = persist_variants(root, "assignments", assignment_variants)
    codebook_records = persist_variants(root, "codebook", codebook_variants)
    assignment_winner = min(assignment_variants, key=lambda item: (len(item.payload), item.method_id))
    codebook_winner = min(codebook_variants, key=lambda item: (len(item.payload), item.method_id))

    decoded_path = root / "receiver" / "decoded_tokens.u8"
    diagnostics = token_diagnostics(source_tokens, model, decoded_path)
    if diagnostics["decoded_token_sha256"] != decoded_sha256(model):
        raise RC1FormatError("independent decoded-token digest differs")
    payload = build_payload(model, assignment_winner, codebook_winner, diagnostics["decoded_token_sha256"])
    payload_path = root / "receiver" / "tokens.rc1v"
    repeat_path = root / "receiver" / "tokens.repeat.rc1v"
    atomic_bytes(payload_path, payload)
    atomic_bytes(repeat_path, build_payload(model, assignment_winner, codebook_winner, diagnostics["decoded_token_sha256"]))
    if payload_path.read_bytes() != repeat_path.read_bytes():
        raise RC1FormatError("RC1 payload repeat differs")
    parsed_model, parsed_digest = parse_payload(payload_path.read_bytes())
    if parsed_digest != diagnostics["decoded_token_sha256"] or decoded_sha256(parsed_model) != parsed_digest:
        raise RC1FormatError("RC1 receiver parse-back differs")

    sections = extract_dx2_shadow_sections(base_archive)
    outer = build_shadow_outer(sections, payload)
    parsed_sections, shadow_model, shadow_digest = parse_shadow_outer(outer)
    if parsed_sections != sections or shadow_digest != parsed_digest or decoded_sha256(shadow_model) != parsed_digest:
        raise RC1FormatError("RC1 shadow receiver parse-back differs")
    archive_path = root / "shadow" / "archive.zip"
    archive_repeat_path = root / "shadow" / "archive.repeat.zip"
    deterministic_zip(archive_path, outer)
    deterministic_zip(archive_repeat_path, build_shadow_outer(sections, repeat_path.read_bytes()))
    if archive_path.read_bytes() != archive_repeat_path.read_bytes():
        raise RC1FormatError("RC1 shadow archive repeat differs")

    negative_root = root / "negative_controls"
    negative_root.mkdir(parents=True, exist_ok=True)
    assignment_negative_path = negative_root / "payload_assignment_bitflip.rc1v"
    assignment_negative, assignment_refused = persist_refused_mutation(
        assignment_negative_path,
        payload,
        PAYLOAD_HEADER.size,
        parse_payload,
    )
    codebook_negative_path = negative_root / "payload_codebook_bitflip.rc1v"
    codebook_negative, codebook_refused = persist_refused_mutation(
        codebook_negative_path,
        payload,
        PAYLOAD_HEADER.size + len(assignment_winner.payload),
        parse_payload,
    )
    shadow_offsets = {
        "semantic": SHADOW_HEADER.size,
        "carrier": SHADOW_HEADER.size + len(sections.semantic),
        "residual": SHADOW_HEADER.size + len(sections.semantic) + len(sections.carrier),
    }
    shadow_negatives: dict[str, dict[str, Any]] = {}
    shadow_refusals: dict[str, bool] = {}
    for section, offset in shadow_offsets.items():
        path = negative_root / f"shadow_{section}_bitflip.p"
        record, refused = persist_refused_mutation(path, outer, offset, parse_shadow_outer)
        shadow_negatives[section] = record
        shadow_refusals[section] = refused
    trailing_path = negative_root / "payload_trailing_byte.rc1v"
    atomic_bytes(trailing_path, payload + b"\x00")
    try:
        parse_payload(payload + b"\x00")
    except RC1FormatError:
        trailing_refused = True
    else:
        trailing_refused = False
    if not all((assignment_refused, codebook_refused, trailing_refused, *shadow_refusals.values())):
        raise RC1FormatError("RC1 mutation controls did not all refuse")

    archive_record = file_record(archive_path)
    payload_record = file_record(payload_path)
    token_only_bar = DX2_TOKEN_STREAM_BYTES - (DX2_ARCHIVE_BYTES - STRICT_SUB012_ARCHIVE_BYTES)
    result = {
        "schema": "ddm_rc1_candidate_result.v1",
        "candidate_id": candidate_id,
        "axis": AXIS,
        "score_claim": False,
        "evaluator_runnable": False,
        "target_k": target_k,
        "actual_k": model.codebook.shape[0],
        "iterations": iterations,
        "producer_sources": producer_sources,
        "fit": fit,
        "diagnostics": diagnostics,
        "assignment_winner": {
            "method_id": assignment_winner.method_id,
            "name": assignment_winner.name,
            "bytes": len(assignment_winner.payload),
        },
        "codebook_winner": {
            "method_id": codebook_winner.method_id,
            "name": codebook_winner.name,
            "bytes": len(codebook_winner.payload),
        },
        "rc1_payload": payload_record,
        "token_only_bar_bytes": token_only_bar,
        "token_only_bar_pass": payload_record["bytes"] <= token_only_bar,
        "shadow_archive": archive_record,
        "shadow_archive_strict_sub012_byte_bar": STRICT_SUB012_ARCHIVE_BYTES,
        "shadow_archive_byte_bar_pass": archive_record["bytes"] <= STRICT_SUB012_ARCHIVE_BYTES,
        "shadow_archive_delta_vs_dx2_bytes": archive_record["bytes"] - DX2_ARCHIVE_BYTES,
        "measured_share_of_42382_byte_demand": min(
            1.0,
            max(0.0, (DX2_TOKEN_STREAM_BYTES - payload_record["bytes"]) / (DX2_ARCHIVE_BYTES - STRICT_SUB012_ARCHIVE_BYTES)),
        ),
        "receiver_contract": {
            "active_sections_consumed": ["copied_semantic", "copied_carrier", "copied_residual", "assignment_map", "temporal_codebook"],
            "exact_length_accounting": True,
            "canonical_reencode": True,
            "payload_repeat_identical": True,
            "archive_repeat_identical": True,
            "decoded_token_digest_verified": True,
            "mutation_refusals": {
                "payload_assignment_bitflip": assignment_refused,
                "payload_codebook_bitflip": codebook_refused,
                **{f"shadow_{section}_bitflip": refused for section, refused in shadow_refusals.items()},
                "payload_trailing_byte": trailing_refused,
            },
            "shipping_integration": False,
            "full_rgb_render": False,
        },
        "wall_seconds": time.monotonic() - started,
        "coder_race": {"assignments": assignment_records, "codebook": codebook_records},
        "negative_controls": [
            assignment_negative,
            codebook_negative,
            *shadow_negatives.values(),
            file_record(trailing_path),
        ],
    }
    custody_paths = [
        assignments_path,
        codebook_path,
        root / "model" / "CHECKPOINT.json",
        decoded_path,
        payload_path,
        repeat_path,
        archive_path,
        archive_repeat_path,
        assignment_negative_path,
        codebook_negative_path,
        *(negative_root / f"shadow_{section}_bitflip.p" for section in shadow_offsets),
        trailing_path,
    ]
    custody_paths.extend(Path(record["path"]) for record in assignment_records + codebook_records)
    result["custody_files"] = [file_record(path) for path in custody_paths]
    atomic_json(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or any(k <= 0 or k > 65_535 for k in args.k) or len(set(args.k)) != len(args.k):
        raise SystemExit("invalid RC1 k/iteration schedule")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    verify_inputs(args.tokens, args.base_archive)
    storage = storage_preflight(args.out_dir, len(args.k))
    producer_sources = retain_producer_sources(args.out_dir)
    run_manifest = {
        "schema": "ddm_rc1_run_manifest.v1",
        "axis": AXIS,
        "score_claim": False,
        "seed": 20260822,
        "determinism": "no stochastic operations; stable lexical and smallest-index tie breaks",
        "argv": [str(item) for item in os.sys.argv],
        "cwd": str(Path.cwd()),
        "tokens": file_record(args.tokens),
        "base_archive": file_record(args.base_archive),
        "out_dir": str(args.out_dir),
        "k": args.k,
        "iterations": args.iterations,
        "storage_preflight": storage,
        "producer_sources": producer_sources,
        "resumability": "source-index checkpoint plus one terminal RESULT.json per k; completed stages rehash before reuse",
        "payload_retention": "all raw learned state, real-coder variants, selected/repeat payloads, decoded tokens, shadow archives, and negative controls retained",
    }
    atomic_json(args.out_dir / "RUN_MANIFEST.json", run_manifest)
    programs, counts, inverse = build_source_index(args.tokens, args.out_dir)
    results = []
    for k in args.k:
        result = build_candidate(
            target_k=k,
            iterations=args.iterations,
            programs=programs,
            counts=counts,
            inverse=inverse,
            source_tokens=args.tokens,
            base_archive=args.base_archive,
            out_dir=args.out_dir,
            producer_sources=producer_sources,
        )
        results.append(result)
        atomic_json(
            args.out_dir / "STATE.json",
            {
                "schema": "ddm_rc1_run_state.v1",
                "completed_candidates": [row["candidate_id"] for row in results],
                "pending_k": [value for value in args.k if f"k{value:03d}_i{args.iterations}" not in {row["candidate_id"] for row in results}],
            },
        )
        print(
            json.dumps(
                {
                    "candidate": result["candidate_id"],
                    "payload_bytes": result["rc1_payload"]["bytes"],
                    "shadow_archive_bytes": result["shadow_archive"]["bytes"],
                    "token_agreement": result["diagnostics"]["source_token_agreement"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema": "ddm_rc1_rate_crush_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
        "source": {"tokens": file_record(args.tokens), "archive": file_record(args.base_archive)},
        "candidates": results,
        "best_payload": min(results, key=lambda row: row["rc1_payload"]["bytes"])["candidate_id"],
        "best_agreement_under_token_only_bar": max(
            (row for row in results if row["token_only_bar_pass"]),
            key=lambda row: row["diagnostics"]["source_token_agreement"],
            default=None,
        ),
        "best_agreement_under_shadow_bar": max(
            (row for row in results if row["shadow_archive_byte_bar_pass"]),
            key=lambda row: row["diagnostics"]["source_token_agreement"],
            default=None,
        ),
        "boundaries": [
            "token Hamming and IoU are diagnostics, not scorer or score evidence",
            "shadow archives are real counted research containers but are not integrated with shipped inflate.py",
            "no scorer, full RGB render, upstream evaluator, Modal call, or authority row ran",
        ],
    }
    selected = summary["best_agreement_under_token_only_bar"]
    if selected is None:
        raise RC1FormatError("no RC1 candidate passed the conservative token-only byte bar")
    selected_result = args.out_dir / "retained" / "candidates" / selected["candidate_id"] / "RESULT.json"
    fire_order = {
        "schema": "ddm_rc1_main_fire_order.v1",
        "disposition": "QUEUED_WITH_FIRE_ORDER",
        "owner": "MAIN",
        "axis_to_run": "[contest-CUDA T4 n600] and/or [contest-CPU 1:1 n600]",
        "selected_candidate": selected["candidate_id"],
        "selected_candidate_result": file_record(selected_result),
        "selected_rc1_payload": selected["rc1_payload"],
        "selected_shadow_archive": selected["shadow_archive"],
        "consumer_store": str(args.out_dir / "main_fire"),
        "dispatch_argv": None,
        "fire_trigger": (
            "MAIN owns an idle unique n600 scorer lane and a fresh owned runtime has integrated "
            "the selected RC1 payload into the real DX2 full-RGB receiver with parse-back, repeat, "
            "all-paid-section mutation refusal, and retained exact archive custody passing"
        ),
        "blocked_by": [
            "the research shadow receiver stops at reconstructed categorical tokens",
            "the selected representation has not been integrated into shipped inflate.py bytes",
            "no full-RGB render, SegNet/PoseNet pass, or exact evaluator run has occurred",
        ],
        "required_action": (
            "Build the integration only in a fresh MAIN-owned receiver surface, retain its exact "
            "archive and repeat, then evaluate that exact archive; fold immediately if the recomputed "
            "score does not improve the canonical pointer or the archive exceeds 137986 bytes"
        ),
    }
    fire_order_path = args.out_dir / "SEALED_FIRE_ORDER.json"
    atomic_json(fire_order_path, fire_order)
    summary["sealed_fire_order"] = file_record(fire_order_path)
    atomic_json(args.out_dir / "RESULT.json", summary)
    print(json.dumps({"result": file_record(args.out_dir / "RESULT.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
