#!/usr/bin/env python3
"""QS2 real-coder rate rung for the measured six-pair QS1 object.

This runner never launches Modal and never re-runs SegNet.  It consumes the
retained QS1 semantic object and compensation codes, races a receiver-consumed
joint sparse compensation overlay through the actual Brotli/ZIP container,
measures the small local PoseNet quantization ladder, retains every payload,
and seals either one dual-axis fire order or a measured non-fire receipt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_cp135_rate_compose as cp135
from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs2_compensation_overlay_runtime as overlay_codec

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs2_20260813")
QS1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qs1_20260813")
QS1_ARCHIVE: Final = (
    QS1_STORE
    / "compile_workspace/retained/candidates/qs1_combined_unique_pairs/primary/"
    "qs1_objects/archive.zip"
)
QS1_RUNTIME: Final = (
    QS1_STORE
    / "compile_workspace/retained/candidates/qs1_combined_unique_pairs/primary/"
    "adapted_runtime"
)
QS1_COMPILED: Final = (
    QS1_STORE
    / "compile_workspace/retained/candidates/qs1_combined_unique_pairs/primary/"
    "QS1_COMPILED_RESULT.json"
)
QS1_SCREEN: Final = QS1_STORE / "COMPILED_SCREEN.json"
QS1_VERDICT: Final = REPO / ".omx/research/ddm_qs1_dual_axis_verdict_20260813.md"
CP135_ARCHIVE: Final = qs1.CP135_ARCHIVE
CP135_BYTES: Final = 186_252
CP135_SHA256: Final = qs1.CP135_ARCHIVE_SHA256
BROTLI: Final = Path("/opt/homebrew/bin/brotli")
SPLIT_HEADER: Final = struct.Struct("<HHH")
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489
SEG_DELTA_S_QS1: Final = -2.712674e-5
POSE_DELTA_S_QS1: Final = 1.126177e-7
BREAKEVEN_FLIPS_PER_BYTE: Final = 0.785
REMOTE_FIELD_SHA256: Final = (
    "ad1e3dcc0a57c53f0757773a018335924afc26992f398c23ec084eecace7ed20"
)
REMOTE_FIELD_VOLUME: Final = "comma-ddm-js1b-argmax-retained"
REMOTE_FIELD_PATH: Final = (
    "ddm_qs1_dual_axis_20260813_r2/retained/fields/candidate_argmax_n600.npy"
)
RUNTIME_OVERLAY_SOURCE: Final = (
    REPO / "experiments/ddm_qs2_compensation_overlay_runtime.py"
)
STORAGE_EXPECTED_BYTES: Final = 8 * 1024**3
STORAGE_RESERVE_BYTES: Final = 8 * 1024**3
AXIS: Final = "[macOS-CPU scorer-free rate + local frozen CPU-PoseNet advisory]"


class QS2Error(RuntimeError):
    """A source pin, retained payload, coder, receiver, or admission gate failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_replace(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        if executable:
            path.chmod(0o755)
    finally:
        partial.unlink(missing_ok=True)


def storage_preflight(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    required = max(0, STORAGE_EXPECTED_BYTES - retained) + STORAGE_RESERVE_BYTES
    free = shutil.disk_usage(output).free
    result = {
        "schema": "ddm_qs2_storage_preflight.v1",
        "tier": str(output.resolve()),
        "already_retained_bytes": retained,
        "expected_total_bytes": STORAGE_EXPECTED_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "free_bytes": free,
        "passed": free >= required,
        "cleanup_policy": "certify-or-block; no payload deletion",
    }
    qs1.atomic_json(output / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise QS2Error(f"SSD storage preflight failed: free={free}, required={required}")
    return result


def _zip_member(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise QS2Error(f"archive member census differs: {archive_path}")
        return archive.read("p")


def deterministic_zip(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def _split_member(member: bytes) -> tuple[tuple[bytes, bytes, bytes], bytes]:
    if len(member) < SPLIT_HEADER.size:
        raise QS2Error("split-model member is truncated")
    lengths = SPLIT_HEADER.unpack_from(member)
    if min(lengths) <= 0:
        raise QS2Error("split-model stream length is zero")
    model_end = SPLIT_HEADER.size + sum(lengths)
    if model_end >= len(member):
        raise QS2Error("split-model member has no residual/token suffix")
    offset = SPLIT_HEADER.size
    streams = []
    for length in lengths:
        streams.append(member[offset : offset + length])
        offset += length
    return (streams[0], streams[1], streams[2]), member[model_end:]


def _brotli_decompress(payload: bytes) -> bytes:
    completed = subprocess.run(
        [str(BROTLI), "-d", "-c"], input=payload, check=False, capture_output=True
    )
    if completed.returncode:
        raise QS2Error(
            f"Brotli decompression failed: {completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _brotli_compress(payload: bytes, quality: int) -> bytes:
    completed = subprocess.run(
        [str(BROTLI), "-q", str(quality), "-c"],
        input=payload,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise QS2Error(
            f"Brotli q{quality} failed: {completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def source_preflight(output: Path) -> dict[str, Any]:
    sources = {
        "cp135_archive": qs1.require_file(
            CP135_ARCHIVE, expected_bytes=CP135_BYTES, expected_sha256=CP135_SHA256
        ),
        "qs1_archive": qs1.require_file(
            QS1_ARCHIVE,
            expected_bytes=186_329,
            expected_sha256="e474d4528aa2917db1433f8ef0ef63a943a15a511628542f98af45d8c972db9d",
        ),
        "qs1_compiled": qs1.require_file(QS1_COMPILED),
        "qs1_screen": qs1.require_file(QS1_SCREEN),
        "qs1_verdict": qs1.require_file(QS1_VERDICT),
        "qs1_engine": qs1.require_file(
            REPO / "experiments/ddm_qs1_frame0_schur_coupled_solve.py",
            expected_sha256="7b299a9b7f520027b2dcc37b1f46081925c7de0b1173f9c0015b123f19c08c5d",
        ),
        "rate_rung_runner": qs1.require_file(Path(__file__).resolve()),
        "overlay_runtime": qs1.require_file(RUNTIME_OVERLAY_SOURCE),
        "brotli": qs1.require_file(BROTLI),
        "dispatcher": qs1.require_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "dual_axis_worker": qs1.require_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker": qs1.require_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    }
    result = {
        "schema": "ddm_qs2_source_preflight.v1",
        "sources": sources,
        "seed": 135,
        "axis": AXIS,
        "no_modal_fire": True,
        "no_segnet_rerun": True,
        "resume_from": str(output.resolve()),
        "passed": True,
    }
    qs1.retain_json(output / "checkpoints/stage_00_source_preflight_r2.json", result)
    return result


def retained_selected_rows() -> list[dict[str, Any]]:
    screen = json.loads(QS1_SCREEN.read_text())
    proposal_ids = [str(value) for value in screen["selected_proposal_ids"]]
    rows: list[dict[str, Any]] = []
    for proposal_id in proposal_ids:
        path = QS1_STORE / "retained/proposals" / proposal_id / "RESULT.json"
        row = json.loads(path.read_text())
        if row.get("schema") != "ddm_qs1_schur_pair_result.v1":
            raise QS2Error(f"QS1 pair result schema differs: {path}")
        row["result_record"] = qs1.file_record(path)
        rows.append(row)
    rows.sort(key=lambda row: int(row["pair"]))
    pairs = [int(row["pair"]) for row in rows]
    if pairs != [105, 176, 178, 517, 523, 532]:
        raise QS2Error(f"sealed QS1 pair census differs: {pairs}")
    return rows


def exact_deltas(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray([int(row["pair"]) for row in rows], dtype=np.int16)
    deltas = np.asarray(
        [row["solve"]["final_code_delta"] for row in rows], dtype=np.int32
    )
    return pairs, deltas


def deadzone_quantize(deltas: np.ndarray, step: int) -> np.ndarray:
    if step not in (1, 2, 3, 4):
        raise QS2Error("quantization step must be one of the sealed rungs")
    values = np.asarray(deltas, dtype=np.int32)
    if step == 1:
        return values.copy()
    return np.sign(values) * (np.abs(values) // step) * step


def postmortem_or_block(output: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    field_root = output / "retained/fields"
    candidate_path = field_root / "candidate_argmax_n600.npy"
    if not candidate_path.is_file():
        blocker = {
            "schema": "ddm_qs2_per_pair_postmortem_blocker.v1",
            "status": "BLOCKED_MISSING_RETAINED_FIELD_LOCAL_COPY",
            "required_remote_volume": REMOTE_FIELD_VOLUME,
            "required_remote_path": REMOTE_FIELD_PATH,
            "required_sha256": REMOTE_FIELD_SHA256,
            "local_destination": str(candidate_path.resolve()),
            "download_argv": [
                ".venv/bin/modal",
                "volume",
                "get",
                REMOTE_FIELD_VOLUME,
                REMOTE_FIELD_PATH,
                str(field_root.resolve()),
            ],
            "sandbox_observation": (
                "api.modal.com DNS resolution is unavailable; no field was downloaded"
            ),
            "no_rederive": True,
            "waterfill_status": "NOT_COMPUTED",
            "selected_pairs": [int(row["pair"]) for row in rows],
            "measured_whole_candidate": {
                "changed_pixels": 189,
                "net_realized_flips": 32,
                "realization_efficiency": 32 / 189,
                "coded_bytes": 77,
                "flips_per_byte": 32 / 77,
            },
            "score_claim": False,
        }
        qs1.retain_json(output / "PER_PAIR_POSTMORTEM_BLOCKER.json", blocker)
        return blocker
    record = qs1.require_file(candidate_path, expected_sha256=REMOTE_FIELD_SHA256)
    raise QS2Error(
        "candidate field is present but the matched base/GT retained fields are not yet "
        f"bound for exact post-mortem: {record}"
    )


def _candidate_rate_sources() -> tuple[bytes, bytes, bytes, bytes]:
    qs1_streams, qs1_suffix = _split_member(_zip_member(QS1_ARCHIVE))
    cp135_streams, _ = _split_member(_zip_member(CP135_ARCHIVE))
    packed_base_carrier = _brotli_decompress(cp135_streams[2])
    if len(packed_base_carrier) != 22_183:
        raise QS2Error(
            f"CP135 packed carrier source length differs: {len(packed_base_carrier)}"
        )
    return qs1_streams[0], qs1_streams[1], packed_base_carrier, qs1_suffix


def build_rate_candidate(
    *,
    output: Path,
    label: str,
    pair_indices: np.ndarray,
    deltas: np.ndarray,
    carrier_quality: int,
    sources: tuple[bytes, bytes, bytes, bytes],
) -> dict[str, Any]:
    candidate_root = output / "retained/rate_race" / label / f"q{carrier_quality:02d}"
    result_path = candidate_root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    active = np.any(deltas != 0, axis=1)
    active_pairs = np.asarray(pair_indices[active], dtype=np.int16)
    active_deltas = np.asarray(deltas[active], dtype=np.int32)
    if not active_pairs.size:
        raise QS2Error("quantized candidate removed every compensation row")
    overlay = overlay_codec.encode_compensation_overlay(active_pairs, active_deltas)
    decoded_pairs, decoded_deltas = overlay_codec.decode_compensation_overlay(overlay)
    if not np.array_equal(decoded_pairs, active_pairs) or not np.array_equal(
        decoded_deltas, active_deltas
    ):
        raise QS2Error("joint sparse overlay parse-back differs")
    stream_a, stream_b, base_carrier, suffix = sources
    carrier_source = base_carrier + overlay
    stream_c = _brotli_compress(carrier_source, carrier_quality)
    if max(len(stream_a), len(stream_b), len(stream_c)) >= 1 << 16:
        raise QS2Error("split-model stream exceeds u16")
    models = SPLIT_HEADER.pack(len(stream_a), len(stream_b), len(stream_c))
    models += stream_a + stream_b + stream_c
    member = models + suffix
    archive = deterministic_zip(member)
    overlay_record = qs1.retain_bytes(candidate_root / "compensation.q2c1", overlay)
    source_record = qs1.retain_bytes(
        candidate_root / "carrier_selector_plus_overlay.raw", carrier_source
    )
    stream_record = qs1.retain_bytes(
        candidate_root / f"carrier_selector_plus_overlay.q{carrier_quality:02d}.br",
        stream_c,
    )
    models_record = qs1.retain_bytes(candidate_root / "split_models.bin", models)
    member_record = qs1.retain_bytes(candidate_root / "p", member)
    archive_record = qs1.retain_bytes(candidate_root / "archive.zip", archive)
    result = {
        "schema": "ddm_qs2_rate_candidate.v1",
        "label": label,
        "carrier_quality": carrier_quality,
        "pair_indices": active_pairs.astype(int).tolist(),
        "deltas": active_deltas.astype(int).tolist(),
        "nonzero_coordinates": int(np.count_nonzero(active_deltas)),
        "overlay": overlay_record,
        "carrier_source": source_record,
        "carrier_stream": stream_record,
        "split_models": models_record,
        "member": member_record,
        "archive": archive_record,
        "archive_delta_bytes_vs_cp135": archive_record["bytes"] - CP135_BYTES,
        "bytes_per_active_pair": (
            (archive_record["bytes"] - CP135_BYTES) / len(active_pairs)
        ),
        "rate_delta_s": (archive_record["bytes"] - CP135_BYTES) * RATE_S_PER_BYTE,
        "all_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.retain_json(result_path, result)
    return result


def run_rate_race(
    output: Path, rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    pairs, deltas = exact_deltas(rows)
    sources = _candidate_rate_sources()
    all_rows: list[dict[str, Any]] = []
    winners: dict[str, dict[str, Any]] = {}
    for step in (1, 2, 3, 4):
        label = f"deadzone_step_{step}"
        quantized = deadzone_quantize(deltas, step)
        candidates = [
            build_rate_candidate(
                output=output,
                label=label,
                pair_indices=pairs,
                deltas=quantized,
                carrier_quality=quality,
                sources=sources,
            )
            for quality in range(12)
        ]
        winner = min(
            candidates,
            key=lambda row: (
                int(row["archive"]["bytes"]), int(row["carrier_quality"])
            ),
        )
        all_rows.extend(candidates)
        winners[label] = winner
    summary = {
        "schema": "ddm_qs2_real_coder_race.v1",
        "selection_mode": "minimum exact retained archive bytes; lower quality breaks ties",
        "candidate_denominator": len(all_rows),
        "quantization_steps": 4,
        "carrier_quality_denominator_per_step": 12,
        "winners": winners,
        "target_bytes_per_pair": 6.8,
        "all_payloads_retained": True,
        "axis": "[macOS-CPU exact byte/container measurement]",
        "score_claim": False,
    }
    qs1.retain_json(output / "RATE_RACE_RESULT.json", summary)
    qs1.retain_json(output / "checkpoints/stage_20_rate_race.json", summary)
    return all_rows, winners


def _selected_vector(row: dict[str, Any], step: int) -> np.ndarray:
    root = QS1_STORE / "retained/proposals" / str(row["proposal_id"])
    if step == 1:
        path = root / "FINAL_POSE_VECTOR.float32.npy"
    else:
        raise QS2Error("only exact QS1 vectors can be loaded without evaluation")
    return np.load(path, allow_pickle=False)


def measure_pose_curve(
    output: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    checkpoint = output / "checkpoints/stage_30_pose_curve.json"
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text())
    import torch

    torch.manual_seed(135)
    np.random.seed(135)
    torch.use_deterministic_algorithms(True)
    surface, _ = qs1.CP135Surface.load()
    posenet = qs1.load_posenet()
    base_pose = np.load(qs1.CP135_BASE_POSE, allow_pickle=False)
    gt_pose = np.load(qs1.GT_POSE, allow_pickle=False)
    base_dpose = float(np.mean(np.square(base_pose.astype(np.float64) - gt_pose)))
    pairs, exact = exact_deltas(rows)
    curve = []
    for step in (1, 2, 3, 4):
        quantized = deadzone_quantize(exact, step)
        pair_vectors = []
        vector_records = []
        for row_index, row in enumerate(rows):
            pair = int(row["pair"])
            proposal_id = str(row["proposal_id"])
            if step == 1:
                vector_path = (
                    QS1_STORE
                    / "retained/proposals"
                    / proposal_id
                    / "FINAL_POSE_VECTOR.float32.npy"
                )
                vector = _selected_vector(row, step)
                vector_records.append(qs1.file_record(vector_path))
            else:
                codes = surface.codes[pair] + quantized[row_index]
                if np.array_equal(codes, surface.codes[pair]):
                    source_path = (
                        QS1_STORE
                        / "retained/proposals"
                        / proposal_id
                        / "stage_20_event_leak/ALL_POSE_VECTORS.float32.npy"
                    )
                    vector = np.load(source_path, allow_pickle=False)[0]
                    vector_records.append(qs1.file_record(source_path))
                else:
                    proposal_root = qs1.JS6_BANK / "proposals" / proposal_id
                    master = np.load(
                        proposal_root / "candidate_camera.uint8.npy", allow_pickle=False
                    )
                    stage_root = (
                        output
                        / "retained/pose_curve"
                        / f"deadzone_step_{step}"
                        / proposal_id
                    )
                    vector = qs1.evaluate_codes(
                        surface=surface,
                        posenet=posenet,
                        codes=(codes,),
                        master=master,
                        pair=pair,
                        stage_root=stage_root,
                    )[0]
                    vector_records.append(
                        qs1.file_record(stage_root / "ALL_POSE_VECTORS.float32.npy")
                    )
            pair_vectors.append(np.asarray(vector, dtype=np.float32))
        candidate_pose = base_pose.copy()
        candidate_pose[pairs.astype(np.int64)] = np.stack(pair_vectors)
        candidate_record = qs1.retain_npy(
            output
            / "retained/pose_curve"
            / f"deadzone_step_{step}"
            / "candidate_first6_n600.float32.npy",
            candidate_pose,
        )
        dpose = float(
            np.mean(np.square(candidate_pose.astype(np.float64) - gt_pose.astype(np.float64)))
        )
        pose_delta_s = float((10.0 * dpose) ** 0.5 - (10.0 * base_dpose) ** 0.5)
        curve.append(
            {
                "step": step,
                "base_dpose": base_dpose,
                "candidate_dpose": dpose,
                "pose_delta_s": pose_delta_s,
                "pose_gate_s": 2e-7,
                "pose_gate_passed": pose_delta_s <= 2e-7,
                "candidate_pose_vectors": candidate_record,
                "pair_vector_sources": vector_records,
                "axis": "[macOS-CPU advisory frozen CPU-PoseNet, six changed pairs over n600]",
            }
        )
    result = {
        "schema": "ddm_qs2_pose_quantization_curve.v1",
        "rows": curve,
        "remote_qs1_reference": {
            "step": 1,
            "pose_delta_s": POSE_DELTA_S_QS1,
            "axis": "[contest-CUDA T4 component instrument, n600]",
        },
        "selection_rule": (
            "coarsest step with local pose_delta_s <= 2e-7; T4 remains the final authority"
        ),
        "score_claim": False,
    }
    qs1.retain_json(output / "POSE_QUANTIZATION_CURVE.json", result)
    qs1.retain_json(checkpoint, result)
    return result


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if new in source and old not in source:
        return
    if source.count(old) != 1:
        raise QS2Error(f"runtime patch surface differs: {path}: {old[:80]!r}")
    updated = source.replace(old, new)
    _atomic_replace(path, updated.encode(), executable=os.access(path, os.X_OK))


def patch_runtime(runtime_root: Path) -> dict[str, Any]:
    overlay_target = runtime_root / "runtime/compensation_overlay.py"
    overlay_payload = RUNTIME_OVERLAY_SOURCE.read_bytes()
    if overlay_target.is_file():
        if overlay_target.read_bytes() != overlay_payload:
            raise QS2Error("adapted runtime overlay source differs")
    else:
        qs1.retain_bytes(overlay_target, overlay_payload)

    archive_parser = runtime_root / "runtime/residual_archive.py"
    _replace_once(
        archive_parser,
        "from .carrier_repack import pack_frame0_selector_carrier\n",
        "from .carrier_repack import pack_frame0_selector_carrier\n"
        "from .compensation_overlay import (\n"
        "    MAGIC as COMPENSATION_MAGIC,\n"
        "    split_selector_compensation,\n"
        ")\n",
    )
    _replace_once(
        archive_parser,
        "    if len(carrier) == PACKED_CAP1_SECTION_BYTES:\n"
        "        carrier = _restore_packed_cap1_metadata(carrier)\n"
        "    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:\n"
        "        return None\n",
        "    if len(carrier) == PACKED_CAP1_SECTION_BYTES:\n"
        "        carrier = _restore_packed_cap1_metadata(carrier)\n"
        "    elif (\n"
        "        len(carrier) > PACKED_CAP1_SECTION_BYTES\n"
        "        and carrier[PACKED_CAP1_SECTION_BYTES:].startswith(COMPENSATION_MAGIC)\n"
        "    ):\n"
        "        carrier = (\n"
        "            _restore_packed_cap1_metadata(carrier[:PACKED_CAP1_SECTION_BYTES])\n"
        "            + carrier[PACKED_CAP1_SECTION_BYTES:]\n"
        "        )\n"
        "    elif len(carrier) != CANONICAL_CAP1_SECTION_BYTES:\n"
        "        return None\n",
    )
    _replace_once(
        archive_parser,
        "    selector = SPARSE_SELECTOR_PREFIX + models[cap1_end:]\n"
        "    try:\n"
        "        decode_cap1(cap1, frames=600, dimensions=12)\n"
        "        decode_selector(selector)\n"
        "        carrier = pack_frame0_selector_carrier(cap1, selector)\n",
        "    selector_tail = SPARSE_SELECTOR_PREFIX + models[cap1_end:]\n"
        "    try:\n"
        "        selector, compensation = split_selector_compensation(selector_tail)\n"
        "        decode_cap1(cap1, frames=600, dimensions=12)\n"
        "        decode_selector(selector)\n"
        "        carrier = pack_frame0_selector_carrier(cap1, selector)\n",
    )
    _replace_once(
        archive_parser,
        "    return semantic, carrier, hpac\n",
        "    return semantic, carrier, hpac, compensation\n",
    )
    _replace_once(
        archive_parser,
        "    compressed_models: bytes\n    token_codec: str = \"rc64\"\n",
        "    compressed_models: bytes\n"
        "    compensation_blob: bytes | None = None\n"
        "    token_codec: str = \"rc64\"\n",
    )
    _replace_once(
        archive_parser,
        "    semantic, carrier, hpac = _decode_models(models)\n",
        "    semantic, carrier, hpac, compensation = _decode_models(models)\n",
    )
    _replace_once(
        archive_parser,
        "        compressed_models=compressed,\n    )\n",
        "        compressed_models=compressed,\n"
        "        compensation_blob=compensation,\n"
        "    )\n",
    )

    inflater = runtime_root / "runtime/f26_inflate.py"
    _replace_once(
        inflater,
        "from .carrier_repack import (\n",
        "from .compensation_overlay import apply_compensation_overlay\n"
        "from .carrier_repack import (\n",
    )
    _replace_once(
        inflater,
        "    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)\n"
        "    semantic = renderer.SemanticTokenRenderer(96)\n",
        "    _, basis, coefficients = renderer.unpack_semantic_pose(semantic_pose)\n"
        "    compensation_report = None\n"
        "    if parts.compensation_blob is not None:\n"
        "        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W\n"
        "        _, _, coefficient_scales, encoded = renderer.decode_compact_carrier(\n"
        "            canonical_carrier,\n"
        "            basis_count=basis_count,\n"
        "            frames=renderer.N,\n"
        "            dimensions=renderer.CARRIER_DIM,\n"
        "        )\n"
        "        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)\n"
        "        base_codes = np.cumsum(delta, axis=0) & 0xFFF\n"
        "        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes).astype(np.int32)\n"
        "        expected_base = torch.from_numpy(base_codes).float() * torch.from_numpy(coefficient_scales)[None]\n"
        "        if not torch.equal(coefficients, expected_base):\n"
        "            raise InflationError(\"compensation base-code reconstruction differs\")\n"
        "        candidate_codes = apply_compensation_overlay(base_codes, parts.compensation_blob)\n"
        "        coefficients = torch.from_numpy(candidate_codes).float() * torch.from_numpy(coefficient_scales)[None]\n"
        "        compensation_report = {\n"
        "            \"payload_bytes\": len(parts.compensation_blob),\n"
        "            \"payload_sha256\": hashlib.sha256(parts.compensation_blob).hexdigest(),\n"
        "            \"changed_coordinates\": int(np.count_nonzero(candidate_codes != base_codes)),\n"
        "        }\n"
        "    semantic = renderer.SemanticTokenRenderer(96)\n",
    )
    _replace_once(
        inflater,
        "        \"residual_schema\": parts.schema,\n",
        "        \"residual_schema\": parts.schema,\n"
        "        \"compensation\": compensation_report,\n",
    )
    return {
        "overlay": qs1.file_record(overlay_target),
        "archive_parser": qs1.file_record(archive_parser),
        "inflater": qs1.file_record(inflater),
    }


def runtime_parseback(
    *, runtime_root: Path, archive: Path, expected_overlay: dict[str, Any]
) -> dict[str, Any]:
    code = (
        "import hashlib,json,sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from runtime.residual_archive import read_residual_archive; "
        "p=read_residual_archive(Path(sys.argv[2])); "
        "o=p.compensation_blob; "
        "print(json.dumps({'bytes':len(o or b''),'sha256':hashlib.sha256(o or b'').hexdigest(),"
        "'token_sha256':hashlib.sha256(p.token_stream).hexdigest(),"
        "'semantic_sha256':hashlib.sha256(p.semantic_blob).hexdigest(),"
        "'hpac_sha256':hashlib.sha256(p.hpac_blob).hexdigest()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(runtime_root), str(archive)],
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise QS2Error(f"adapted runtime parse-back failed: {completed.stderr}")
    result = json.loads(completed.stdout)
    if result["bytes"] != expected_overlay["bytes"] or result["sha256"] != expected_overlay["sha256"]:
        raise QS2Error("adapted runtime compensation parse-back differs")
    return result


def compile_runtime_candidate(
    output: Path,
    winner: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    root = output / "candidate"
    result_path = root / "COMPILE_RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    source_archive = Path(winner["archive"]["path"])
    archive_payload = source_archive.read_bytes()
    primary = qs1.retain_bytes(root / "archive.zip", archive_payload)
    repeated_payload = deterministic_zip(_zip_member(source_archive))
    repeated = qs1.retain_bytes(root / "archive.repeat.zip", repeated_payload)
    if primary["sha256"] != repeated["sha256"]:
        raise QS2Error("independent deterministic archive repeat differs")
    runtime_root = root / "adapted_runtime"
    from experiments import ddm_jo1_joint_probability_object as jo1

    runtime_result = jo1.copy_runtime(runtime_root, archive_payload)
    patches = patch_runtime(runtime_root)
    parseback = runtime_parseback(
        runtime_root=runtime_root,
        archive=runtime_root / "archive.zip",
        expected_overlay=winner["overlay"],
    )
    base_codes = qs1._load_cp135_carrier_codes()
    pairs, expected_deltas = exact_deltas(rows)
    overlay_payload = Path(winner["overlay"]["path"]).read_bytes()
    actual_codes = overlay_codec.apply_compensation_overlay(base_codes, overlay_payload)
    expected_codes = base_codes.copy()
    expected_codes[pairs.astype(np.int64)] += expected_deltas
    if not np.array_equal(actual_codes, expected_codes):
        raise QS2Error("receiver overlay does not reproduce the sealed QS1 code lattice")
    code_record = qs1.retain_npy(root / "candidate_codes.int32.npy", actual_codes)
    runtime_tree = cp135.tree_record(runtime_root)
    result = {
        "schema": "ddm_qs2_compiled_candidate.v1",
        "archive": primary,
        "archive_repeat": repeated,
        "archive_repeat_byte_identical": True,
        "archive_delta_bytes_vs_cp135": primary["bytes"] - CP135_BYTES,
        "bytes_per_pair": (primary["bytes"] - CP135_BYTES) / len(rows),
        "overlay": winner["overlay"],
        "candidate_codes": code_record,
        "exact_qs1_code_lattice_reproduced": True,
        "semantic_and_token_object": qs1.file_record(QS1_ARCHIVE),
        "runtime_root": str(runtime_root.resolve()),
        "runtime_copy": runtime_result,
        "runtime_patches": patches,
        "runtime_parseback": parseback,
        "runtime_tree": runtime_tree,
        "all_payloads_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.retain_json(result_path, result)
    qs1.retain_json(output / "checkpoints/stage_40_candidate_compile.json", result)
    return result


def admission(compiled: dict[str, Any]) -> dict[str, Any]:
    delta_bytes = int(compiled["archive_delta_bytes_vs_cp135"])
    rate_delta_s = delta_bytes * RATE_S_PER_BYTE
    complete_delta_s = SEG_DELTA_S_QS1 + POSE_DELTA_S_QS1 + rate_delta_s
    return {
        "schema": "ddm_qs2_preencoded_admission.v1",
        "seg_delta_s": SEG_DELTA_S_QS1,
        "pose_delta_s": POSE_DELTA_S_QS1,
        "rate_delta_bytes": delta_bytes,
        "rate_delta_s": rate_delta_s,
        "complete_delta_s": complete_delta_s,
        "admitted": complete_delta_s < 0.0,
        "flips_per_compensation_byte": 32 / delta_bytes if delta_bytes > 0 else None,
        "breakeven_flips_per_byte": BREAKEVEN_FLIPS_PER_BYTE,
        "component_source": "QS1 matched T4 component instrument on the same semantic and code lattice",
        "transport_invariance": (
            "receiver code/token parse-back is exact; the fresh T4 dual-axis run remains the verdict"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def validate_re1t_request(request: dict[str, Any]) -> None:
    """Mirror the unchanged worker's fail-closed request boundary before sealing."""
    if (
        request.get("schema") != "ddm_qs1_t4_dual_axis_request.v1"
        or request.get("resume_from") != request.get("run_id")
        or request.get("retain_pose_vectors") is not True
        or request.get("local_pose_delta") != 0.0
        or request.get("pose_unmeasured") is not True
        or request.get("score_claim") is not False
        or request.get("promotion_eligible") is not False
    ):
        raise QS2Error("sealed request differs from the unchanged RE1T worker contract")
    if set(request.get("inputs", {})) != {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "POSE_SCREEN_RESULT.json",
    }:
        raise QS2Error("sealed request input census differs from the RE1T worker")


def seal_fire_order(output: Path, compiled: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b

    run_id = "ddm_qs2_dual_axis_20260813_r2"
    fire_root = output / "fire_order"
    input_root = fire_root / "fire_inputs"
    archive_path = Path(compiled["archive"]["path"])
    runtime_root = Path(compiled["runtime_root"])
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        runtime_root, label="ddm_qs2_joint_sparse_compensation"
    )
    screen_payload = (json.dumps(screen, indent=2, sort_keys=True) + "\n").encode()
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "POSE_SCREEN_RESULT.json": screen_payload,
    }
    for name, payload in payloads.items():
        qs1.retain_bytes(input_root / name, payload)
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "transport_schema_note": "QS2 reuses the unchanged self-claiming QS1/js6b dual-axis transport",
        "run_id": run_id,
        "resume_from": run_id,
        "lane_id": "ddm_qs2_compensation_rate_n600_20260813",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "MAIN",
        "seed": 1234,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": qs1.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": runtime_manifest,
        "inputs": {name: js1b.payload_record(payload) for name, payload in payloads.items()},
        # The unchanged RE1T worker requires these Pose-unknown transport
        # placeholders.  The retained POSE_SCREEN_RESULT is advisory input;
        # the worker measures fresh Pose vectors on T4.
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": hashlib.sha256(git_status).hexdigest(),
        "dispatcher_source_sha256": qs1.sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": qs1.sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": qs1.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    validate_re1t_request(request)
    request_record = qs1.retain_json(fire_root / "SEALED_REQUEST_r2.json", request)
    dispatch_output = output / "dispatch" / run_id
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        request_record["path"],
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str(dispatch_output.resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_qs2_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(output.resolve()),
        "fire_trigger": (
            "MAIN verifies no active n600 exact-eval/Modal lane and all sealed input SHAs, then "
            "executes exact_command_argv; the dispatcher self-claims the lane"
        ),
        "fresh_run_id": run_id,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "budget_ledger": "#381",
        "remote_scope": (
            "one exact candidate decode plus n600 frozen T4 SegNet argmax field and official "
            "PoseNet first-six vectors with a repeat"
        ),
        "post_harvest_admission": (
            "recompute matched complete candidate delta S; accept only net realized delta S < 0"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.atomic_json(output / "SEALED_FIRE_ORDER.json", order)
    return order


def finalize(
    output: Path,
    *,
    postmortem: dict[str, Any],
    winners: dict[str, dict[str, Any]],
    pose_curve: dict[str, Any],
    compiled: dict[str, Any],
) -> dict[str, Any]:
    screen = admission(compiled)
    qs1.retain_json(output / "ADMISSION.json", screen)
    if screen["admitted"]:
        disposition = "QUEUED-WITH-A-FIRE-ORDER"
        sealed = seal_fire_order(output, compiled, screen)
        no_fire = None
    else:
        disposition = "FOLDED"
        sealed = None
        no_fire_value = {
            "schema": "ddm_qs2_sealed_no_fire_order.v1",
            "sealed": True,
            "disposition": disposition,
            "owner": "MAIN",
            "consumer_store": str(output.resolve()),
            "measured_rate_curve": winners,
            "admission": screen,
            "fire_trigger": (
                "reopen only if a real receiver-consumed compensation representation moves the "
                "measured flips-per-byte asymptote strictly above 0.785"
            ),
            "score_claim": False,
        }
        no_fire = qs1.retain_json(output / "SEALED_NO_FIRE_ORDER.json", no_fire_value)
    result = {
        "schema": "ddm_qs2_final_result.v1",
        "axis": AXIS,
        "disposition": disposition,
        "per_pair_postmortem": postmortem,
        "waterfill_computed": False,
        "waterfill_blocker": "retained T4 argmax field is not locally accessible",
        "rate_winners": winners,
        "pose_curve": pose_curve,
        "compiled_candidate": compiled,
        "admission": screen,
        "fire_order": sealed,
        "no_fire_order": no_fire,
        "modal_fired": False,
        "segnet_rerun": False,
        "all_payloads_retained": True,
        "pointer_moved": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    qs1.atomic_json(output / "FINAL_RESULT.json", result)
    qs1.atomic_json(output / "checkpoints/stage_90_final.json", result)
    return result


def run(output: Path = OUTPUT) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve():
        raise QS2Error(f"output must be the governed SSD store: {OUTPUT}")
    storage_preflight(output)
    source_preflight(output)
    rows = retained_selected_rows()
    postmortem = postmortem_or_block(output, rows)
    _, winners = run_rate_race(output, rows)
    pose_curve = measure_pose_curve(output, rows)
    exact_winner = winners["deadzone_step_1"]
    compiled = compile_runtime_candidate(output, exact_winner, rows)
    return finalize(
        output,
        postmortem=postmortem,
        winners=winners,
        pose_curve=pose_curve,
        compiled=compiled,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume_from.resolve() != args.output.resolve():
        raise QS2Error("--resume-from must equal --output")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "RUN.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QS2Error("another QS2 process holds the governed run lock") from error
        result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
