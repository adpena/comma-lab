#!/usr/bin/env python3
"""Fit and retain the scorer-free NR1 task-cell quotient prebuild.

This runner never invokes a scorer and never touches the active r9 directory.
Every fitted surface, coder loser, deterministic repeat, decoded token field,
and renderer-smoke output is retained under APDataStore before it is reduced to
a scalar.  Stage manifests make the run resumable with ``--resume-from``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.nr1_taskcell_quotient import (
    CodedCandidate,
    NR1FormatError,
    Section,
    build_packet,
    coder_candidates,
    decode_packet,
    encode_raw_sections,
    physical_attribution,
    replace_qevent,
)

AXIS = "[macOS-CPU scorer-free real-coder task-token n600]"
SOURCE_SHAPE = (600, 384, 512)
DX2_ARCHIVE_BYTES = 180_368
DX2_ARCHIVE_SHA256 = "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
DX2_TOKEN_BYTES = 117_964_800
DX2_TASK_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
DX2_TOKEN_STREAM_BYTES = 113_777
TOKEN_ONLY_CEILING_BYTES = 71_395
QUOTIENT_PLUS_HPAC_CEILING_BYTES = 84_910
STRICT_ARCHIVE_CEILING_BYTES = 137_986
STRICT_RATE_CUT_BYTES = 42_382

DEFAULT_OUTPUT = Path("/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
DEFAULT_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/archive.zip"
)
DEFAULT_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
DEFAULT_TEACHER = Path("/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730")
RUNNER_SOURCE = Path(__file__).resolve()
MODULE_SOURCE = RUNNER_SOURCE.parents[1] / "src/tac/optimization/nr1_taskcell_quotient.py"


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def bytes_record(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if file_record(path) != {"path": str(path), **bytes_record(payload)}:
            raise NR1FormatError(f"refusing to overwrite different retained payload: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def deterministic_zip(path: Path, member: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member)
        payload = temporary.read_bytes()
        temporary.unlink()
        atomic_bytes(path, payload)
    finally:
        if temporary.exists():
            temporary.unlink()


def storage_preflight(out_dir: Path) -> dict[str, Any]:
    allowed = DEFAULT_OUTPUT.resolve().parent
    resolved = out_dir.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise NR1FormatError("NR1 retained output must be below /Volumes/APDataStore/pact")
    usage = shutil.disk_usage(allowed)
    predicted = 4 * DX2_TOKEN_BYTES + 512 * 1024 * 1024
    reserve = 8 * 1024 * 1024 * 1024
    if usage.free < predicted + reserve:
        raise NR1FormatError(
            f"storage preflight refused: free={usage.free}, predicted={predicted}, reserve={reserve}"
        )
    receipt_path = out_dir / "STORAGE_PREFLIGHT.json"
    if receipt_path.is_file():
        retained = json.loads(receipt_path.read_text())
        if (
            retained.get("schema") != "ddm_nr1_storage_preflight.v1"
            or retained.get("path") != str(resolved)
            or retained.get("predicted_bytes") != predicted
            or retained.get("reserve_bytes") != reserve
            or retained.get("passed") is not True
        ):
            raise NR1FormatError("retained storage preflight does not bind this resumable run")
        return retained
    receipt = {
        "schema": "ddm_nr1_storage_preflight.v1",
        "path": str(resolved),
        "free_bytes": usage.free,
        "predicted_bytes": predicted,
        "reserve_bytes": reserve,
        "passed": True,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def verify_inputs(tokens_path: Path, archive_path: Path, teacher_root: Path) -> dict[str, Any]:
    tokens = file_record(tokens_path)
    archive = file_record(archive_path)
    if tokens["bytes"] != DX2_TOKEN_BYTES or tokens["sha256"] != DX2_TASK_FIELD_SHA256:
        raise NR1FormatError(f"DX2 token custody differs: {tokens}")
    if archive["bytes"] != DX2_ARCHIVE_BYTES or archive["sha256"] != DX2_ARCHIVE_SHA256:
        raise NR1FormatError(f"DX2 archive custody differs: {archive}")
    teacher_manifest = teacher_root / "field_pass_manifest.json"
    manifest = json.loads(teacher_manifest.read_text())
    if manifest.get("pair_count") != SOURCE_SHAPE[0] or len(manifest.get("pairs", [])) != SOURCE_SHAPE[0]:
        raise NR1FormatError("teacher manifest does not cover all 600 pairs")
    return {
        "tokens": tokens,
        "archive": archive,
        "teacher_manifest": file_record(teacher_manifest),
        "teacher_authority": manifest.get("authority"),
        "teacher_score_claim": manifest.get("score_claim"),
    }


def retain_producer_sources(out_dir: Path) -> list[dict[str, Any]]:
    records = []
    for source in (RUNNER_SOURCE, MODULE_SOURCE):
        retained = out_dir / "retained" / "producer_source" / source.name
        atomic_bytes(retained, source.read_bytes())
        records.append({"workspace": file_record(source), "retained": file_record(retained)})
    return records


def select_task_priority_events(
    source: np.ndarray,
    decoded_base: np.ndarray,
    teacher_root: Path,
    event_limit: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select low-margin mismatches from a manifest-bound secondary teacher."""
    manifest_path = teacher_root / "field_pass_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates_index: list[np.ndarray] = []
    candidates_margin: list[np.ndarray] = []
    checked_files = []
    mismatches = 0
    plane = SOURCE_SHAPE[1] * SOURCE_SHAPE[2]
    for record in manifest["pairs"]:
        pair = int(record["pair_id"])
        path = teacher_root / record["path"]
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != record["sha256"]:
            raise NR1FormatError(f"teacher pair custody differs: {path}")
        with np.load(io.BytesIO(blob), allow_pickle=False) as payload:
            margin = np.asarray(payload["distill_margin"], dtype=np.float32)
        if margin.shape != SOURCE_SHAPE[1:]:
            raise NR1FormatError(f"teacher margin geometry differs: {path}")
        local = np.flatnonzero(source[pair].reshape(-1) != decoded_base[pair].reshape(-1))
        mismatches += int(local.size)
        if local.size:
            local_count = min(event_limit, int(local.size))
            scores = margin.reshape(-1)[local]
            keep = (
                np.arange(local.size)
                if local_count == local.size
                else np.argpartition(scores, local_count - 1)[:local_count]
            )
            candidates_index.append((local[keep] + pair * plane).astype(np.uint32))
            candidates_margin.append(scores[keep].astype(np.float32))
        checked_files.append({"pair_id": pair, "path": str(path), "sha256": digest})
    all_index = np.concatenate(candidates_index)
    all_margin = np.concatenate(candidates_margin)
    count = min(event_limit, int(all_index.size))
    keep = np.argpartition(all_margin, count - 1)[:count]
    order = np.argsort(all_index[keep], kind="stable")
    selected_index = all_index[keep][order]
    selected_margin = all_margin[keep][order]
    if len(np.unique(selected_index)) != len(selected_index):
        raise NR1FormatError("task-priority selector produced duplicate coordinates")
    receipt = {
        "schema": "ddm_nr1_task_priority_selector.v1",
        "selection": "lowest C1 distill margin among quotient token mismatches",
        "authority": manifest.get("authority"),
        "score_claim": False,
        "endpoint_status": "secondary C1 field; not the current frozen primary endpoint",
        "manifest": file_record(manifest_path),
        "checked_pair_files": checked_files,
        "base_mismatches": mismatches,
        "selected_events": len(selected_index),
        "event_limit": event_limit,
        "margin_min": float(selected_margin.min()),
        "margin_max": float(selected_margin.max()),
    }
    return selected_index, selected_margin, receipt


def fit_stage(
    out_dir: Path,
    tokens_path: Path,
    teacher_root: Path,
    tile_height: int,
    tile_width: int,
    codebook_size: int,
    event_limit: int,
) -> tuple[dict[Section, bytes], np.memmap, dict[str, Any]]:
    root = out_dir / "retained" / "fit"
    manifest_path = root / "FIT_MANIFEST.json"
    base_path = root / "decoded_before_events.u8"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        raw_sections = {
            Section(name): (root / "sections" / f"{name}.raw").read_bytes()
            for name in (section.value for section in Section)
        }
        for record in manifest["retained_files"]:
            if file_record(Path(record["path"])) != record:
                raise NR1FormatError("fit checkpoint failed retained custody")
        decoded_base = np.memmap(base_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
        return raw_sections, decoded_base, manifest

    source = np.memmap(tokens_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    raw_sections, decoded_base_array = encode_raw_sections(
        source,
        tile_height=tile_height,
        tile_width=tile_width,
        codebook_size=codebook_size,
    )
    atomic_bytes(base_path, decoded_base_array.tobytes())
    decoded_base = np.memmap(base_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    event_indices, event_margins, selector = select_task_priority_events(
        source,
        decoded_base,
        teacher_root,
        event_limit,
    )
    event_index_path = root / "teacher_fit" / "selected_event_indices.u32"
    event_margin_path = root / "teacher_fit" / "selected_event_margins.f32"
    atomic_bytes(event_index_path, np.asarray(event_indices, dtype="<u4").tobytes())
    atomic_bytes(event_margin_path, np.asarray(event_margins, dtype="<f4").tobytes())
    atomic_json(root / "teacher_fit" / "TEACHER_MANIFEST.json", selector)
    raw_sections = replace_qevent(raw_sections, source, decoded_base, event_indices)
    section_records = []
    for section, raw in raw_sections.items():
        path = root / "sections" / f"{section.value}.raw"
        atomic_bytes(path, raw)
        section_records.append(file_record(path))
    retained_files = [
        file_record(base_path),
        file_record(event_index_path),
        file_record(event_margin_path),
        *section_records,
    ]
    manifest = {
        "schema": "ddm_nr1_fit_stage.v1",
        "axis": AXIS,
        "config": {
            "shape": SOURCE_SHAPE,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "codebook_size_requested": codebook_size,
            "event_limit": event_limit,
        },
        "source": file_record(tokens_path),
        "teacher": selector,
        "retained_files": retained_files,
        "disposition": "SCORER_FREE_FIT_COMPLETE_NOT_RECEIVER_CLOSED",
    }
    atomic_json(manifest_path, manifest)
    return raw_sections, decoded_base, manifest


def coder_stage(
    out_dir: Path,
    raw_sections: dict[Section, bytes],
) -> tuple[bytes, dict[str, Any]]:
    root = out_dir / "retained" / "coder"
    manifest_path = root / "CODER_MANIFEST.json"
    packet_path = root / "nr1_packet.bin"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["retained_files"]:
            if file_record(Path(record["path"])) != record:
                raise NR1FormatError("coder checkpoint failed retained custody")
        return packet_path.read_bytes(), manifest

    selected: dict[Section, CodedCandidate] = {}
    races = {}
    retained_files = []
    for section, raw in raw_sections.items():
        rows = []
        candidates = coder_candidates(raw)
        for candidate in candidates:
            path = root / section.value / f"{candidate.coder.name}.bin"
            repeat = root / section.value / f"{candidate.coder.name}.repeat.bin"
            atomic_bytes(path, candidate.payload)
            repeated_candidates = coder_candidates(raw)
            repeated = next(item for item in repeated_candidates if item.coder is candidate.coder)
            atomic_bytes(repeat, repeated.payload)
            if path.read_bytes() != repeat.read_bytes():
                raise NR1FormatError(f"{section.value}/{candidate.coder.name} is nondeterministic")
            records = [file_record(path), file_record(repeat)]
            retained_files.extend(records)
            rows.append(
                {
                    "coder": candidate.coder.name,
                    "raw_bytes": len(raw),
                    "coded": records[0],
                    "repeat": records[1],
                }
            )
        winner = min(candidates, key=lambda item: (len(item.payload), int(item.coder)))
        selected[section] = winner
        races[section.value] = {
            "candidates": rows,
            "selected": winner.coder.name,
            "selected_bytes": len(winner.payload),
        }

    packet = build_packet(raw_sections, *SOURCE_SHAPE, selected=selected)
    repeat_packet = build_packet(raw_sections, *SOURCE_SHAPE, selected=selected)
    repeat_path = root / "nr1_packet.repeat.bin"
    atomic_bytes(packet_path, packet)
    atomic_bytes(repeat_path, repeat_packet)
    if packet != repeat_packet:
        raise NR1FormatError("deterministic packet repeat differs")
    zip_path = root / "nr1_packet.zip"
    zip_repeat_path = root / "nr1_packet.repeat.zip"
    deterministic_zip(zip_path, packet)
    deterministic_zip(zip_repeat_path, packet)
    if zip_path.read_bytes() != zip_repeat_path.read_bytes():
        raise NR1FormatError("deterministic ZIP repeat differs")
    retained_files.extend(
        [
            file_record(packet_path),
            file_record(repeat_path),
            file_record(zip_path),
            file_record(zip_repeat_path),
        ]
    )
    attribution = {
        section.value: {"start": start, "end": end, "bytes": end - start}
        for section, (start, end) in physical_attribution(packet).items()
    }
    manifest = {
        "schema": "ddm_nr1_coder_stage.v1",
        "axis": AXIS,
        "races": races,
        "packet": file_record(packet_path),
        "packet_zip": file_record(zip_path),
        "physical_attribution": attribution,
        "logical_bytes_sum": sum(row["bytes"] for row in attribution.values()),
        "retained_files": retained_files,
        "disposition": "REAL_CODED_PACKET_NOT_SUBMISSION_ARCHIVE",
    }
    atomic_json(manifest_path, manifest)
    return packet, manifest


def decode_stage(
    out_dir: Path,
    packet: bytes,
    source_path: Path,
) -> tuple[np.memmap, dict[str, Any]]:
    root = out_dir / "retained" / "decode"
    manifest_path = root / "DECODE_MANIFEST.json"
    output_path = root / "received_tokens.u8"
    repeat_path = root / "received_tokens.repeat.u8"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["retained_files"]:
            if file_record(Path(record["path"])) != record:
                raise NR1FormatError("decode checkpoint failed retained custody")
        return np.memmap(output_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE), manifest

    result = decode_packet(packet)
    atomic_bytes(output_path, result.tokens.tobytes())
    repeat = decode_packet(packet)
    atomic_bytes(repeat_path, repeat.tokens.tobytes())
    if output_path.read_bytes() != repeat_path.read_bytes():
        raise NR1FormatError("independent token decode repeats differ")
    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    received = np.memmap(output_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
    mismatch_count = 0
    class_confusion = np.zeros((5, 5), dtype=np.uint64)
    for pair in range(SOURCE_SHAPE[0]):
        left = np.asarray(source[pair]).reshape(-1)
        right = np.asarray(received[pair]).reshape(-1)
        mismatch_count += int(np.count_nonzero(left != right))
        np.add.at(class_confusion, (left, right), 1)
    retained_files = [file_record(output_path), file_record(repeat_path)]
    manifest = {
        "schema": "ddm_nr1_decode_stage.v1",
        "axis": AXIS,
        "packet": bytes_record(packet),
        "received_tokens": retained_files[0],
        "repeat": retained_files[1],
        "source": file_record(source_path),
        "mismatch_count": mismatch_count,
        "token_count": int(np.prod(SOURCE_SHAPE)),
        "token_agreement": 1.0 - mismatch_count / int(np.prod(SOURCE_SHAPE)),
        "class_confusion": class_confusion.tolist(),
        "consumption_trace": {
            "QPARAM": result.trace.qparam,
            "QCTX": result.trace.qctx,
            "QPAIR": result.trace.qpair,
            "QEVENT": result.trace.qevent,
        },
        "retained_files": retained_files,
        "disposition": "TOKEN_RECEIVER_CHECKED_NOT_FULL_RAW_CLOSED",
    }
    atomic_json(manifest_path, manifest)
    return received, manifest


def renderer_smoke_stage(
    out_dir: Path,
    runtime_root: Path,
    archive_path: Path,
    source_path: Path,
    received: np.ndarray,
) -> dict[str, Any]:
    """Run actual DX2 semantic weights on pair zero for a consumer counterfactual."""
    root = out_dir / "retained" / "renderer_smoke"
    manifest_path = root / "RENDERER_SMOKE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["retained_files"]:
            if file_record(Path(record["path"])) != record:
                raise NR1FormatError("renderer-smoke checkpoint failed retained custody")
        return manifest

    import torch

    cpr1 = runtime_root / "cpr1"
    sys.path.insert(0, str(cpr1))
    sys.path.insert(0, str(runtime_root))
    try:
        spec = importlib.util.spec_from_file_location("nr1_dx2_renderer", cpr1 / "inflate.py")
        if spec is None or spec.loader is None:
            raise NR1FormatError("could not load the DX2 semantic renderer")
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        from runtime.residual_archive import read_residual_archive

        parts = read_residual_archive(archive_path)
        semantic = renderer.SemanticTokenRenderer(96).eval()
        state = renderer.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
        if state is None:
            raise NR1FormatError("DX2 semantic payload is not directly supported by renderer smoke")
        semantic.load_state_dict(state, strict=True)
        torch.set_num_threads(4)
        source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=SOURCE_SHAPE)
        index = torch.tensor([0], dtype=torch.long)
        with torch.inference_mode():
            source_output = (
                semantic(torch.from_numpy(np.asarray(source[0:1]).copy()).long(), index)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            quotient_output = (
                semantic(torch.from_numpy(np.asarray(received[0:1]).copy()).long(), index)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .numpy()
            )
    finally:
        sys.path = [entry for entry in sys.path if entry not in {str(cpr1), str(runtime_root)}]
    source_output_path = root / "dx2_tokens_pair0_semantic_output.u8"
    quotient_output_path = root / "nr1_tokens_pair0_semantic_output.u8"
    atomic_bytes(source_output_path, source_output.tobytes())
    atomic_bytes(quotient_output_path, quotient_output.tobytes())
    if np.array_equal(source_output, quotient_output):
        raise NR1FormatError("actual semantic renderer was unchanged by NR1 token counterfactual")
    retained_files = [file_record(source_output_path), file_record(quotient_output_path)]
    manifest = {
        "schema": "ddm_nr1_renderer_smoke.v1",
        "axis": AXIS,
        "pair": 0,
        "runtime_root": str(runtime_root),
        "runtime_sources": [
            file_record(cpr1 / "inflate.py"),
            file_record(cpr1 / "ddm_mp2_semantic_receiver.py"),
        ],
        "archive": file_record(archive_path),
        "semantic_blob": bytes_record(parts.semantic_blob),
        "changed_output_values": int(np.count_nonzero(source_output != quotient_output)),
        "retained_files": retained_files,
        "disposition": "ACTUAL_RENDERER_PAIR0_CONSUMPTION_SMOKE_NOT_FULL_RAW_CLOSURE",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def write_final_receipts(
    out_dir: Path,
    inputs: dict[str, Any],
    source_records: list[dict[str, Any]],
    fit: dict[str, Any],
    coder: dict[str, Any],
    decode: dict[str, Any],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    packet_bytes = int(coder["packet"]["bytes"])
    zip_bytes = int(coder["packet_zip"]["bytes"])
    result = {
        "schema": "ddm_nr1_taskcell_quotient_prebuild.v1",
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
        "primary_endpoint_frozen": False,
        "matched_distortion_measured": False,
        "full_raw_receiver_closed": False,
        "inputs": inputs,
        "producer_sources": source_records,
        "config": fit["config"],
        "packet": coder["packet"],
        "packet_zip": coder["packet_zip"],
        "bytes": {
            "quotient_packet": packet_bytes,
            "quotient_packet_zip": zip_bytes,
            "current_token_stream": DX2_TOKEN_STREAM_BYTES,
            "token_only_ceiling": TOKEN_ONLY_CEILING_BYTES,
            "quotient_plus_hpac_ceiling": QUOTIENT_PLUS_HPAC_CEILING_BYTES,
            "strict_archive_ceiling": STRICT_ARCHIVE_CEILING_BYTES,
            "strict_rate_cut_required": STRICT_RATE_CUT_BYTES,
            "packet_vs_current_token_stream": packet_bytes - DX2_TOKEN_STREAM_BYTES,
            "packet_vs_token_only_ceiling": packet_bytes - TOKEN_ONLY_CEILING_BYTES,
            "packet_vs_quotient_plus_hpac_ceiling": packet_bytes - QUOTIENT_PLUS_HPAC_CEILING_BYTES,
        },
        "token_agreement": decode["token_agreement"],
        "token_mismatch_count": decode["mismatch_count"],
        "consumption_trace": decode["consumption_trace"],
        "renderer_smoke": smoke,
        "claims": {
            "measured": [
                "real coder bytes for all four paid surfaces",
                "canonical parse/repack and deterministic packet/ZIP repeats",
                "full n600 task-token decode and exact-once surface consumption",
                "actual DX2 semantic renderer pair-zero output changes under NR1 tokens",
            ],
            "not_measured": [
                "matched Seg/Pose distortion",
                "3,662,409,600-byte raw inflate",
                "contest score or sub-0.12 status",
                "fresh minimal-environment shipping closure",
            ],
        },
        "disposition": "SCORER_FREE_EXECUTABLE_PREBUILD_NOT_RECEIVER_CLOSED",
        "pointer": "UNMOVED",
    }
    atomic_json(out_dir / "RESULT.json", result)
    shipping = {
        "schema": "ddm_nr1_shipping_allowlist.v1",
        "status": "PREBUILD_ONLY_NO_SHIPPING_ARCHIVE_EXISTS",
        "hypothetical_counted_member": "p",
        "allowed_logical_sections": [section.value for section in Section],
        "packet": coder["packet"],
        "forbidden_from_shipping": [
            inputs["tokens"]["path"],
            inputs["teacher_manifest"]["path"],
            *(record["workspace"]["path"] for record in source_records),
            "scorer weights, logits, margins, ground truth, caches, stale coefficients",
        ],
        "rule_118_claim": "generic decoder code free; all fitted video-derived surfaces counted",
    }
    atomic_json(out_dir / "SHIPPING_ALLOWLIST.json", shipping)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "vq8_k64_e8192_v1")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--tile-height", type=int, default=8)
    parser.add_argument("--tile-width", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--event-limit", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_from is not None and args.resume_from.resolve() != args.output.resolve():
        raise NR1FormatError("--resume-from must name the same retained output as --output")
    started = time.time()
    storage_preflight(args.output)
    inputs = verify_inputs(args.tokens, args.archive, args.teacher_root)
    sources = retain_producer_sources(args.output)
    raw_sections, _, fit = fit_stage(
        args.output,
        args.tokens,
        args.teacher_root,
        args.tile_height,
        args.tile_width,
        args.codebook_size,
        args.event_limit,
    )
    packet, coder = coder_stage(args.output, raw_sections)
    received, decode = decode_stage(args.output, packet, args.tokens)
    smoke = renderer_smoke_stage(
        args.output,
        args.runtime_root,
        args.archive,
        args.tokens,
        received,
    )
    result = write_final_receipts(args.output, inputs, sources, fit, coder, decode, smoke)
    print(
        json.dumps(
            {
                "result": str(args.output / "RESULT.json"),
                "packet_bytes": result["bytes"]["quotient_packet"],
                "token_agreement": result["token_agreement"],
                "elapsed_seconds": time.time() - started,
                "disposition": result["disposition"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
