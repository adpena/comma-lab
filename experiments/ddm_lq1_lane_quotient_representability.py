#!/usr/bin/env python3
"""Measure Lane representability in retained RC1 and NR1 quotient objects.

This is a scorer-free diagnostic.  It neither fits nor recuts either retained
payload.  RC1 is tested with two assignments against its existing K=2,048
codebook:

* the globally nearest full-program Hamming assignment, which isolates RC1's
  recorded no-global-reassignment policy; and
* a Lane-recall oracle, which maximizes correct Lane time slots per spatial
  program and uses full-program agreement and then codeword index as tie
  breaks.  This is an upper bound, not a receiver or score result.

Every final assignment field, confusion matrix, diversity field, and per-class
decomposition is retained with byte and SHA-256 custody.  The expensive oracle
stage is resumable by deterministic program chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = Path("/Volumes/APDataStore/pact/ddm_lq1_lane_quotient_representability/measurement_v2")
AP_ROOT = Path("/Volumes/APDataStore/pact")
VERTIGO_ROOT = Path("/Volumes/VertigoDataTier/pact")

RC1_ROOT = Path("/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4")
RC1_RETAINED = RC1_ROOT / "retained"
RC1_CANDIDATE = RC1_RETAINED / "candidates/k2048_i3"
RC1_SOURCE_INDEX = RC1_RETAINED / "source_index"
CB2_ROOT = Path("/Volumes/APDataStore/pact/ddm_cb2_class_balanced_dictionary/measurement_v5")
NR1_ROOT = Path("/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1")

SHAPE = (600, 384, 512)
UNIQUE_PROGRAMS = 30_428
CODEBOOK_SIZE = 2_048
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
LANE_ID = 1
AXIS = "[macOS-CPU scorer-free retained-token n600]"

EXPECTED_SHA256 = {
    "cb2_memo": "e9bcae8776732cc12e0bce5196008c3f636bb9efa73233be3787d76e9c3e410d",
    "cb2_result": "5e27148484152aa9553eafb6a9cc96412064c07b613b69c930f1cb4d3c007682",
    "rc1_memo": "dfb239fcda4a749925326500b8821637d969e204d7eaf9191d64fc7a524e7c8d",
    "rc1_module": "6c2ea6f324ea32b21d8cc079bb327c6af97e283cc963ec610859f1f2b0cbfbc9",
    "rc1_payload": "eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164",
    "rc1_codebook": "d4e5f28b27bef4fca622108db92403942fe1d72470af40ebf11f8ceb308921bf",
    "rc1_assignments": "34c4eaf615d8030a0afd877cc3c2f5896e1e4ed56e0460a8a7932d247f5f2053",
    "rc1_shadow": "6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a",
    "source_tokens": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",  # gitleaks:allow -- public SHA-256 content digest
    "unique_programs": "47ceb559e1feda4d89cceb1f519ec9f2462e746dad44d0e78c45d9136e1f9955",
    "unique_counts": "69873bf282c3791109041db78538dd3b791d4de098f8c2f8e149540a47a13de6",
    "site_unique_ids": "7762a7808d91c85514da169c88659eb6d3112546582b0c0b834ba975550e1464",
    "nr1_memo": "e1ae945821f60d0c0fc2de062b6325c2773fde24125dbb1975862bc3c296c64d",
    "nr1_module": "66500b813eeafeaf264d57ecb47ef68360956ec1bdb040043456f3d6f101cbb6",
    "nr1_result": "d3e7d58c286c82813d0356f3681b76e940d3bb206e88eaad8feebe0c68ace623",
    "nr1_decode_manifest": "570b59dd7dd28441c4405a37ad8dd10753aae07e5b55f652d12ece9b5e1dfeb5",
    "nr1_packet": "a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28",
    "nr1_received": "d416895a250ce79be7f485188d4f7dfd1690a269a250063c2f6bc5f48cf8b8d8",
}

EXPECTED_CB2_CONFUSION = np.asarray(
    [
        [27_107_897, 6_275, 163_704, 72_077, 56_935],
        [583_728, 101_792, 2_034, 2_985, 556],
        [122_729, 0, 58_226_927, 63_563, 3],
        [153_010, 1, 136_668, 1_170_638, 141],
        [55_866, 22, 11, 23, 29_937_215],
    ],
    dtype=np.int64,
)
EXPECTED_CB2_CAPACITY = np.asarray([548_935, 64_539, 329_201, 244_863, 41_262])
EXPECTED_NR1_CONFUSION = np.asarray(
    [
        [26_991_462, 54_959, 215_168, 58_647, 86_652],
        [576_672, 101_672, 9_428, 1_648, 1_675],
        [218_470, 23, 58_142_705, 52_021, 3],
        [75_301, 6, 95_993, 1_289_050, 108],
        [110_331, 1_515, 10, 203, 29_881_078],
    ],
    dtype=np.int64,
)


class LQ1Refusal(RuntimeError):
    """Fail closed when inherited custody or a scientific invariant drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LQ1Refusal(f"required file is absent: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_file(
    path: Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    record = file_record(path)
    if size is not None and record["bytes"] != size:
        raise LQ1Refusal(f"byte drift for {path}: {record['bytes']} != {size}")
    if sha256 is not None and record["sha256"] != sha256:
        raise LQ1Refusal(f"SHA-256 drift for {path}: {record['sha256']} != {sha256}")
    return record


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def storage_preflight(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    allowed = [path.resolve() for path in (AP_ROOT, VERTIGO_ROOT)]
    if not any(resolved == base or base in resolved.parents for base in allowed):
        raise LQ1Refusal("receipt root must be on an authorized SSD tier")
    storage_base = next(base for base in allowed if resolved == base or base in resolved.parents)
    usage = shutil.disk_usage(storage_base)
    required = 256 * 1024 * 1024
    reserve = 8 * 1024 * 1024 * 1024
    if usage.free < required + reserve:
        raise LQ1Refusal(f"storage preflight refused: free={usage.free}, required={required}, reserve={reserve}")
    row = {
        "schema": "ddm_lq1_storage_preflight.v1",
        "receipt_root": str(resolved),
        "storage_tier": str(storage_base),
        "free_bytes": usage.free,
        "required_bytes": required,
        "reserve_bytes": reserve,
        "passed": True,
        "payload_plan": (
            "retain all oracle assignments, confusion matrices, diversity fields, per-class rows, "
            "producer source, and resumable progress arrays; do not materialize RGB or scorer fields"
        ),
    }
    atomic_json(root / "STORAGE_PREFLIGHT.json", row)
    return row


def checkpoint(root: Path, step: int, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / "checkpoints" / f"{step:02d}_{name}.json"
    row = {"schema": "ddm_lq1_stage_checkpoint.v1", "step": step, "stage": name, **payload}
    atomic_json(path, row)
    return file_record(path)


def verify_custody(root: Path) -> tuple[dict[str, Any], Path]:
    source_manifest_path = RC1_SOURCE_INDEX / "MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_tokens = Path(source_manifest["source"]["path"])
    inputs = {
        "cb2_memo": require_file(
            REPO / ".omx/research/ddm_cb2_class_balanced_dictionary_20260822.md",
            sha256=EXPECTED_SHA256["cb2_memo"],
        ),
        "cb2_result": require_file(CB2_ROOT / "RESULT.json", sha256=EXPECTED_SHA256["cb2_result"]),
        "rc1_memo": require_file(
            REPO / ".omx/research/ddm_rc1_rate_crush_20260822.md",
            sha256=EXPECTED_SHA256["rc1_memo"],
        ),
        "rc1_module": require_file(
            REPO / "src/tac/optimization/rc1_terminal_program_vq.py",
            sha256=EXPECTED_SHA256["rc1_module"],
        ),
        "rc1_payload": require_file(
            RC1_CANDIDATE / "receiver/tokens.rc1v",
            size=59_884,
            sha256=EXPECTED_SHA256["rc1_payload"],
        ),
        "rc1_codebook": require_file(
            RC1_CANDIDATE / "model/codebook.u8",
            size=1_228_800,
            sha256=EXPECTED_SHA256["rc1_codebook"],
        ),
        "rc1_assignments": require_file(
            RC1_CANDIDATE / "model/assignments.u16le",
            size=393_216,
            sha256=EXPECTED_SHA256["rc1_assignments"],
        ),
        "rc1_shadow": require_file(
            RC1_CANDIDATE / "shadow/archive.zip",
            size=113_006,
            sha256=EXPECTED_SHA256["rc1_shadow"],
        ),
        "source_manifest": file_record(source_manifest_path),
        "source_tokens": require_file(
            source_tokens,
            size=int(np.prod(SHAPE)),
            sha256=EXPECTED_SHA256["source_tokens"],
        ),
        "unique_programs": require_file(
            RC1_SOURCE_INDEX / "unique_programs.u8",
            size=UNIQUE_PROGRAMS * SHAPE[0],
            sha256=EXPECTED_SHA256["unique_programs"],
        ),
        "unique_counts": require_file(
            RC1_SOURCE_INDEX / "unique_counts.u32",
            size=UNIQUE_PROGRAMS * 4,
            sha256=EXPECTED_SHA256["unique_counts"],
        ),
        "site_unique_ids": require_file(
            RC1_SOURCE_INDEX / "site_unique_ids.u32",
            size=SHAPE[1] * SHAPE[2] * 4,
            sha256=EXPECTED_SHA256["site_unique_ids"],
        ),
        "nr1_memo": require_file(
            REPO / ".omx/research/ddm_nr1_taskcell_quotient_prebuild_20260822.md",
            sha256=EXPECTED_SHA256["nr1_memo"],
        ),
        "nr1_module": require_file(
            REPO / "src/tac/optimization/nr1_taskcell_quotient.py",
            sha256=EXPECTED_SHA256["nr1_module"],
        ),
        "nr1_result": require_file(NR1_ROOT / "RESULT.json", sha256=EXPECTED_SHA256["nr1_result"]),
        "nr1_decode_manifest": require_file(
            NR1_ROOT / "retained/decode/DECODE_MANIFEST.json",
            sha256=EXPECTED_SHA256["nr1_decode_manifest"],
        ),
        "nr1_packet": require_file(
            NR1_ROOT / "retained/coder/nr1_packet.bin",
            size=69_004,
            sha256=EXPECTED_SHA256["nr1_packet"],
        ),
        "nr1_received": require_file(
            NR1_ROOT / "retained/decode/received_tokens.u8",
            size=int(np.prod(SHAPE)),
            sha256=EXPECTED_SHA256["nr1_received"],
        ),
    }
    producer_copy = root / "retained/producer_source" / Path(__file__).name
    source_bytes = Path(__file__).read_bytes()
    if producer_copy.exists() and producer_copy.read_bytes() != source_bytes:
        raise LQ1Refusal("producer source drifted inside an existing receipt root")
    atomic_bytes(producer_copy, source_bytes)
    inputs["producer_source"] = file_record(producer_copy)
    return inputs, source_tokens


def recover_current_unique_assignments(
    inverse: np.ndarray,
    site_assignments: np.ndarray,
) -> np.ndarray:
    minimum = np.full(UNIQUE_PROGRAMS, np.iinfo(np.uint16).max, dtype=np.uint16)
    maximum = np.zeros(UNIQUE_PROGRAMS, dtype=np.uint16)
    np.minimum.at(minimum, inverse, site_assignments)
    np.maximum.at(maximum, inverse, site_assignments)
    if np.any(minimum != maximum):
        raise LQ1Refusal("one source program maps to multiple retained RC1 assignments")
    return minimum


def confusion_from_unique_assignments(
    programs: np.ndarray,
    counts: np.ndarray,
    codebook: np.ndarray,
    assignments: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    confusion = np.zeros((5, 5), dtype=np.int64)
    for start in range(0, len(programs), chunk_size):
        stop = min(start + chunk_size, len(programs))
        truth = np.asarray(programs[start:stop], dtype=np.uint8)
        prediction = np.asarray(codebook[np.asarray(assignments[start:stop])], dtype=np.uint8)
        keys = (5 * truth.astype(np.int16) + prediction.astype(np.int16)).reshape(-1)
        weights = np.repeat(np.asarray(counts[start:stop], dtype=np.float64), truth.shape[1])
        block = np.bincount(keys, weights=weights, minlength=25).reshape(5, 5)
        if not np.all(block == np.floor(block)):
            raise LQ1Refusal("weighted confusion lost integer exactness")
        confusion += block.astype(np.int64)
    return confusion


def confusion_from_token_files(source_path: Path, prediction_path: Path) -> np.ndarray:
    source = np.memmap(source_path, dtype=np.uint8, mode="r", shape=SHAPE)
    prediction = np.memmap(prediction_path, dtype=np.uint8, mode="r", shape=SHAPE)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for pair in range(SHAPE[0]):
        truth = np.asarray(source[pair], dtype=np.int16).reshape(-1)
        pred = np.asarray(prediction[pair], dtype=np.int16).reshape(-1)
        if np.any(truth >= 5) or np.any(pred >= 5):
            raise LQ1Refusal("token file contains a class outside [0,4]")
        confusion += np.bincount(5 * truth + pred, minlength=25).reshape(5, 5)
    return confusion


def per_class_rows(confusion: np.ndarray, surface: str) -> list[dict[str, Any]]:
    total = int(confusion.sum())
    mismatches = total - int(np.trace(confusion))
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        true_count = int(confusion[class_id].sum())
        predicted_count = int(confusion[:, class_id].sum())
        correct = int(confusion[class_id, class_id])
        class_mismatches = true_count - correct
        union = true_count + predicted_count - correct
        rows.append(
            {
                "surface": surface,
                "evidence": "PROXY-NOT-SCORE",
                "class_id": class_id,
                "class_name": class_name,
                "true_positions": true_count,
                "position_denominator": total,
                "correct_positions": correct,
                "agreement_given_true_class": correct / true_count,
                "mismatched_positions": class_mismatches,
                "all_mismatch_denominator": mismatches,
                "share_of_all_token_mismatches": class_mismatches / mismatches,
                "predicted_positions": predicted_count,
                "iou_proxy_not_score": correct / union,
            }
        )
    return rows


def open_oracle_progress(root: Path) -> tuple[dict[str, np.memmap], int]:
    progress = root / "retained/oracle_progress"
    progress.mkdir(parents=True, exist_ok=True)
    specifications = {
        "full_assignment": (np.uint16, np.iinfo(np.uint16).max),
        "full_matches": (np.uint16, 0),
        "lane_assignment": (np.uint16, np.iinfo(np.uint16).max),
        "lane_matches": (np.uint16, 0),
        "lane_assignment_full_matches": (np.uint16, 0),
    }
    arrays: dict[str, np.memmap] = {}
    state_path = progress / "STATE.json"
    next_program = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != "ddm_lq1_oracle_progress.v1":
            raise LQ1Refusal("oracle progress schema drifted")
        next_program = int(state["next_program"])
    for name, (dtype, fill) in specifications.items():
        path = progress / f"{name}.npy"
        if path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != (UNIQUE_PROGRAMS,) or array.dtype != np.dtype(dtype):
                raise LQ1Refusal(f"oracle progress array drifted: {path}")
        else:
            if next_program:
                raise LQ1Refusal("oracle state exists without every progress array")
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=(UNIQUE_PROGRAMS,),
            )
            array[:] = fill
            array.flush()
        arrays[name] = array
    if not 0 <= next_program <= UNIQUE_PROGRAMS:
        raise LQ1Refusal("oracle progress offset is outside the program population")
    return arrays, next_program


def measure_oracles(
    root: Path,
    programs: np.ndarray,
    codebook: np.ndarray,
    *,
    program_chunk: int,
    time_chunk: int,
) -> dict[str, np.memmap]:
    arrays, next_program = open_oracle_progress(root)
    state_path = root / "retained/oracle_progress/STATE.json"
    for start in range(next_program, len(programs), program_chunk):
        stop = min(start + program_chunk, len(programs))
        batch = np.asarray(programs[start:stop], dtype=np.uint8)
        matches = np.zeros((stop - start, len(codebook)), dtype=np.uint16)
        lane_matches = np.zeros_like(matches)
        for t0 in range(0, batch.shape[1], time_chunk):
            t1 = min(t0 + time_chunk, batch.shape[1])
            source_block = batch[:, t0:t1, None]
            codebook_block = np.asarray(codebook[:, t0:t1], dtype=np.uint8).T[None, :, :]
            equal = source_block == codebook_block
            matches += np.sum(equal, axis=1, dtype=np.uint16)
            lane_equal = (source_block == LANE_ID) & (codebook_block == LANE_ID)
            lane_matches += np.sum(lane_equal, axis=1, dtype=np.uint16)

        full_assignment = np.argmax(matches, axis=1).astype(np.uint16)
        rows = np.arange(stop - start)
        best_lane = np.max(lane_matches, axis=1)
        lane_eligible = lane_matches == best_lane[:, None]
        tie_scores = np.where(lane_eligible, matches.astype(np.int32), -1)
        lane_assignment = np.argmax(tie_scores, axis=1).astype(np.uint16)
        no_lane = np.count_nonzero(batch == LANE_ID, axis=1) == 0
        lane_assignment[no_lane] = full_assignment[no_lane]

        arrays["full_assignment"][start:stop] = full_assignment
        arrays["full_matches"][start:stop] = matches[rows, full_assignment]
        arrays["lane_assignment"][start:stop] = lane_assignment
        arrays["lane_matches"][start:stop] = lane_matches[rows, lane_assignment]
        arrays["lane_assignment_full_matches"][start:stop] = matches[rows, lane_assignment]
        for array in arrays.values():
            array.flush()
        atomic_json(
            state_path,
            {
                "schema": "ddm_lq1_oracle_progress.v1",
                "next_program": stop,
                "program_denominator": UNIQUE_PROGRAMS,
                "program_chunk": program_chunk,
                "time_chunk": time_chunk,
                "tie_breaks": {
                    "full_hamming": "maximum full-program matches, then smallest codeword index",
                    "lane_oracle": (
                        "maximum matches on true-Lane time slots, then maximum full-program "
                        "matches, then smallest codeword index"
                    ),
                },
            },
        )
    return arrays


def coverage_depths(mass: np.ndarray) -> dict[str, int]:
    positive = np.sort(np.asarray(mass[mass > 0], dtype=np.int64))[::-1]
    total = int(positive.sum())
    cumulative = np.cumsum(positive)
    return {
        f"programs_for_{int(threshold * 100)}pct_lane_token_mass": int(
            np.searchsorted(cumulative, math.ceil(threshold * total), side="left") + 1
        )
        for threshold in (0.5, 0.9, 0.95, 0.99)
    }


def entropy_perplexity(mass: np.ndarray) -> tuple[float, float]:
    positive = np.asarray(mass[mass > 0], dtype=np.float64)
    probability = positive / positive.sum()
    entropy = float(-np.sum(probability * np.log2(probability)))
    return entropy, float(2.0**entropy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--program-chunk", type=int, default=128)
    parser.add_argument("--time-chunk", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.program_chunk <= 0 or args.time_chunk <= 0:
        raise SystemExit("oracle chunk sizes must be positive")
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight(args.out_dir)
    inputs, source_tokens_path = verify_custody(args.out_dir)
    custody_checkpoint = checkpoint(
        args.out_dir,
        1,
        "inherited_custody_complete",
        {"status": "complete", "inputs": inputs, "storage": storage},
    )

    programs = np.memmap(
        RC1_SOURCE_INDEX / "unique_programs.u8",
        dtype=np.uint8,
        mode="r",
        shape=(UNIQUE_PROGRAMS, SHAPE[0]),
    )
    counts = np.memmap(
        RC1_SOURCE_INDEX / "unique_counts.u32",
        dtype="<u4",
        mode="r",
        shape=(UNIQUE_PROGRAMS,),
    )
    inverse = np.memmap(
        RC1_SOURCE_INDEX / "site_unique_ids.u32",
        dtype="<u4",
        mode="r",
        shape=(SHAPE[1] * SHAPE[2],),
    )
    codebook = np.memmap(
        RC1_CANDIDATE / "model/codebook.u8",
        dtype=np.uint8,
        mode="r",
        shape=(CODEBOOK_SIZE, SHAPE[0]),
    )
    site_assignments = np.memmap(
        RC1_CANDIDATE / "model/assignments.u16le",
        dtype="<u2",
        mode="r",
        shape=(SHAPE[1] * SHAPE[2],),
    )
    if int(counts.sum()) != SHAPE[1] * SHAPE[2]:
        raise LQ1Refusal("source program counts do not sum to the spatial-site denominator")
    if not np.array_equal(np.bincount(np.asarray(inverse), minlength=UNIQUE_PROGRAMS), np.asarray(counts)):
        raise LQ1Refusal("source program counts differ from the retained site index")

    current_unique = recover_current_unique_assignments(inverse, site_assignments)
    current_confusion = confusion_from_unique_assignments(programs, counts, codebook, current_unique)
    direct_current_confusion = confusion_from_token_files(
        source_tokens_path, RC1_CANDIDATE / "receiver/decoded_tokens.u8"
    )
    if not np.array_equal(current_confusion, direct_current_confusion):
        raise LQ1Refusal("unique-program and direct RC1 confusion derivations differ")
    if not np.array_equal(current_confusion, EXPECTED_CB2_CONFUSION):
        raise LQ1Refusal("CB2 confusion rows drifted")
    capacity = np.bincount(np.asarray(codebook).reshape(-1), minlength=5)
    if not np.array_equal(capacity, EXPECTED_CB2_CAPACITY):
        raise LQ1Refusal("CB2 capacity rows drifted")
    current_confusion_path = args.out_dir / "retained/confusion/rc1_current.npy"
    atomic_npy(current_confusion_path, current_confusion)
    cb2_reproduction = {
        "source_position_denominator": int(current_confusion.sum()),
        "capacity_slot_denominator": int(codebook.size),
        "capacity_slots_by_class": capacity.tolist(),
        "agreement_rows": per_class_rows(current_confusion, "RC1 retained assignment"),
        "matches_cb2_exactly": True,
        "evidence": "PROXY-NOT-SCORE",
    }
    cb2_checkpoint = checkpoint(
        args.out_dir,
        2,
        "cb2_rows_reproduced",
        {
            "status": "complete",
            "reproduction": cb2_reproduction,
            "confusion": file_record(current_confusion_path),
        },
    )

    oracle = measure_oracles(
        args.out_dir,
        programs,
        codebook,
        program_chunk=args.program_chunk,
        time_chunk=args.time_chunk,
    )
    full_assignment = np.asarray(oracle["full_assignment"])
    lane_assignment = np.asarray(oracle["lane_assignment"])
    if np.any(full_assignment >= CODEBOOK_SIZE) or np.any(lane_assignment >= CODEBOOK_SIZE):
        raise LQ1Refusal("oracle assignment contains an out-of-range codeword")

    current_matches = np.count_nonzero(np.asarray(programs) != np.asarray(codebook[current_unique]), axis=1)
    current_matches = SHAPE[0] - current_matches
    current_lane_matches = np.count_nonzero(
        (np.asarray(programs) == LANE_ID) & (np.asarray(codebook[current_unique]) == LANE_ID),
        axis=1,
    )
    if np.any(np.asarray(oracle["full_matches"]) < current_matches):
        raise LQ1Refusal("full-Hamming oracle is worse than the retained assignment")
    if np.any(np.asarray(oracle["lane_matches"]) < current_lane_matches):
        raise LQ1Refusal("Lane oracle is worse than the retained assignment on true-Lane slots")

    full_confusion = confusion_from_unique_assignments(programs, counts, codebook, full_assignment)
    lane_confusion = confusion_from_unique_assignments(programs, counts, codebook, lane_assignment)
    confusion_paths: dict[str, Path] = {
        "rc1_full_hamming_oracle": args.out_dir / "retained/confusion/rc1_full_hamming_oracle.npy",
        "rc1_lane_oracle": args.out_dir / "retained/confusion/rc1_lane_oracle.npy",
    }
    atomic_npy(confusion_paths["rc1_full_hamming_oracle"], full_confusion)
    atomic_npy(confusion_paths["rc1_lane_oracle"], lane_confusion)

    full_site = full_assignment[np.asarray(inverse)]
    lane_site = lane_assignment[np.asarray(inverse)]
    full_site_path = args.out_dir / "retained/oracle_assignments/full_hamming_site.u16le"
    lane_site_path = args.out_dir / "retained/oracle_assignments/lane_recall_site.u16le"
    atomic_bytes(full_site_path, np.asarray(full_site, dtype="<u2").tobytes())
    atomic_bytes(lane_site_path, np.asarray(lane_site, dtype="<u2").tobytes())

    lane_counts = np.count_nonzero(np.asarray(programs) == LANE_ID, axis=1).astype(np.uint16)
    lane_mass = lane_counts.astype(np.int64) * np.asarray(counts, dtype=np.int64)
    if int(lane_mass.sum()) != 691_095:
        raise LQ1Refusal("source Lane position denominator drifted")
    codebook_lookup = {bytes(row): index for index, row in enumerate(codebook)}
    exact_codeword = np.asarray([codebook_lookup.get(bytes(row), -1) for row in programs], dtype=np.int32)
    source_lane_masks = np.packbits(np.asarray(programs) == LANE_ID, axis=1, bitorder="little")
    codebook_lane_masks = np.packbits(np.asarray(codebook) == LANE_ID, axis=1, bitorder="little")
    lane_bearing = lane_counts > 0
    unique_lane_masks, lane_mask_inverse = np.unique(source_lane_masks[lane_bearing], axis=0, return_inverse=True)
    codebook_lane_mask_keys = {
        bytes(mask) for mask, word in zip(codebook_lane_masks, codebook, strict=True) if np.any(word == LANE_ID)
    }
    lane_mask_in_codebook = np.asarray(
        [bytes(row) in codebook_lane_mask_keys for row in source_lane_masks], dtype=np.bool_
    )
    diversity_path = args.out_dir / "retained/diversity/intrinsic_program_fields.npz"
    atomic_npz(
        diversity_path,
        source_lane_counts=lane_counts,
        source_lane_token_mass=lane_mass,
        exact_codeword_index=exact_codeword,
        source_lane_masks_packbits=source_lane_masks,
        source_lane_mask_in_codebook=lane_mask_in_codebook,
        unique_lane_masks_packbits=unique_lane_masks,
        lane_bearing_mask_inverse=lane_mask_inverse.astype(np.uint32),
        codebook_lane_masks_packbits=codebook_lane_masks,
        current_unique_assignment=current_unique,
        full_hamming_unique_assignment=full_assignment,
        lane_recall_unique_assignment=lane_assignment,
        current_full_matches=current_matches.astype(np.uint16),
        current_lane_matches=current_lane_matches.astype(np.uint16),
        full_oracle_matches=np.asarray(oracle["full_matches"]),
        lane_oracle_matches=np.asarray(oracle["lane_matches"]),
        lane_oracle_full_matches=np.asarray(oracle["lane_assignment_full_matches"]),
    )
    entropy, perplexity = entropy_perplexity(lane_mass)
    intrinsic_diversity = {
        "source_unique_full_programs_with_lane": int(np.count_nonzero(lane_bearing)),
        "source_unique_full_program_denominator": UNIQUE_PROGRAMS,
        "source_spatial_sites_with_lane": int(np.sum(np.asarray(counts)[lane_bearing])),
        "source_spatial_site_denominator": SHAPE[1] * SHAPE[2],
        "source_lane_positions": int(lane_mass.sum()),
        "source_token_position_denominator": int(np.prod(SHAPE)),
        "source_distinct_lane_binary_masks": len(unique_lane_masks),
        "codebook_words_with_lane": int(np.count_nonzero(np.any(codebook == LANE_ID, axis=1))),
        "codebook_word_denominator": CODEBOOK_SIZE,
        "codebook_distinct_lane_binary_masks": len(codebook_lane_mask_keys),
        "source_full_programs_exactly_in_codebook": int(np.count_nonzero(exact_codeword >= 0)),
        "source_lane_bearing_full_programs_exactly_in_codebook": int(
            np.count_nonzero((exact_codeword >= 0) & lane_bearing)
        ),
        "source_lane_bearing_full_program_denominator": int(np.count_nonzero(lane_bearing)),
        "source_lane_positions_from_exactly_present_full_programs": int(lane_mass[exact_codeword >= 0].sum()),
        "source_lane_positions_with_exact_lane_mask_in_codebook": int(lane_mass[lane_mask_in_codebook].sum()),
        "lane_token_weighted_program_entropy_bits": entropy,
        "lane_token_weighted_effective_program_perplexity": perplexity,
        **coverage_depths(lane_mass),
        "evidence": "PROXY-NOT-SCORE",
    }

    nr1_confusion = confusion_from_token_files(source_tokens_path, NR1_ROOT / "retained/decode/received_tokens.u8")
    if not np.array_equal(nr1_confusion, EXPECTED_NR1_CONFUSION):
        raise LQ1Refusal("NR1 K32 confusion differs from its retained decode manifest")
    nr1_confusion_path = args.out_dir / "retained/confusion/nr1_k32.npy"
    atomic_npy(nr1_confusion_path, nr1_confusion)

    decompositions = {
        "rc1_current": per_class_rows(current_confusion, "RC1 retained assignment"),
        "rc1_full_hamming_oracle": per_class_rows(full_confusion, "RC1 full-Hamming oracle assignment"),
        "rc1_lane_oracle": per_class_rows(lane_confusion, "RC1 Lane-recall oracle assignment"),
        "nr1_k32": per_class_rows(nr1_confusion, "NR1 K32 retained receiver"),
    }
    decomposition_path = args.out_dir / "retained/per_class_decomposition.json"
    atomic_json(
        decomposition_path,
        {
            "schema": "ddm_lq1_per_class_decomposition.v1",
            "axis": AXIS,
            "warning": "Every agreement and IoU value is PROXY-NOT-SCORE; no scorer ran.",
            "surfaces": decompositions,
        },
    )

    current_mismatches = int(current_confusion.sum() - np.trace(current_confusion))
    full_mismatches = int(full_confusion.sum() - np.trace(full_confusion))
    lane_mismatches = int(lane_confusion.sum() - np.trace(lane_confusion))
    current_lane_correct = int(current_confusion[LANE_ID, LANE_ID])
    full_lane_correct = int(full_confusion[LANE_ID, LANE_ID])
    lane_oracle_correct = int(lane_confusion[LANE_ID, LANE_ID])
    lane_denominator = int(current_confusion[LANE_ID].sum())
    lane_oracle_agreement = lane_oracle_correct / lane_denominator
    full_oracle_agreement = full_lane_correct / lane_denominator
    current_agreement = current_lane_correct / lane_denominator
    nr1_lane_denominator = int(nr1_confusion[LANE_ID].sum())
    nr1_lane_agreement = int(nr1_confusion[LANE_ID, LANE_ID]) / nr1_lane_denominator

    oracle_diagnosis = {
        "lane_position_denominator": lane_denominator,
        "current_lane_correct": current_lane_correct,
        "current_lane_agreement": current_agreement,
        "full_hamming_oracle_lane_correct": full_lane_correct,
        "full_hamming_oracle_lane_agreement": full_oracle_agreement,
        "lane_oracle_lane_correct": lane_oracle_correct,
        "lane_oracle_lane_agreement": lane_oracle_agreement,
        "current_total_mismatches": current_mismatches,
        "full_hamming_oracle_total_mismatches": full_mismatches,
        "lane_oracle_total_mismatches": lane_mismatches,
        "full_hamming_oracle_mismatches_removed": current_mismatches - full_mismatches,
        "full_hamming_oracle_lane_matches_recovered": full_lane_correct - current_lane_correct,
        "lane_oracle_lane_matches_recovered": lane_oracle_correct - current_lane_correct,
        "lane_oracle_total_mismatch_delta": lane_mismatches - current_mismatches,
        "unique_programs_reassigned_by_full_hamming_oracle": int(np.count_nonzero(full_assignment != current_unique)),
        "unique_program_denominator": UNIQUE_PROGRAMS,
        "spatial_sites_reassigned_by_full_hamming_oracle": int(
            np.sum(np.asarray(counts)[full_assignment != current_unique])
        ),
        "spatial_site_denominator": SHAPE[1] * SHAPE[2],
        "unique_programs_reassigned_by_lane_oracle": int(np.count_nonzero(lane_assignment != current_unique)),
        "spatial_sites_reassigned_by_lane_oracle": int(np.sum(np.asarray(counts)[lane_assignment != current_unique])),
        "oracle_construction": {
            "full_hamming": (
                "for each of 30,428 source temporal programs, choose the existing K=2,048 "
                "codeword with the most categorical matches over all 600 time slots"
            ),
            "lane_recall": (
                "for each source temporal program, choose the existing codeword with the most "
                "class-1 matches on true-Lane time slots; break ties by full 600-slot agreement "
                "and then smallest codeword index; programs without Lane use the full-Hamming oracle"
            ),
        },
        "evidence": "PROXY-NOT-SCORE",
    }
    assignment_mechanism = {
        "source_fact": "RC1 fit records global_reassignment_to_added_programs=false",
        "cure": (
            "globally reassign all source programs against all existing K=2,048 codewords, then "
            "solve the Lane-versus-collateral assignment objective before any refit or larger K"
        ),
        "incremental_codebook_words": 0,
        "incremental_raw_assignment_bytes": 0,
        "new_payload_sections": 0,
        "realized_compressed_assignment_delta_bytes": None,
        "compressed_byte_boundary": (
            "unknown by charter: LQ1 retained the oracle field but did not recut the live RC1 payload"
        ),
    }
    if lane_oracle_agreement > 0.9:
        prior_outcome = "REFUTED"
    elif lane_oracle_agreement < 0.6 and nr1_lane_agreement < 0.6:
        prior_outcome = "SUPPORTED"
    else:
        prior_outcome = "INCONCLUSIVE"
    prior_law = {
        "prediction": (
            "RC1 Lane failure is representational; Lane oracle stays below 60%, and NR1 K32 shows the same collapse"
        ),
        "falsifier": "RC1 Lane oracle agreement exceeds 90%",
        "outcome": prior_outcome,
        "verdict_scope": (
            "FORMULATION: retained RC1 K=2,048 codebook plus its assignment/objective; NR1 is a "
            "cross-instance agreement check, not an oracle-containment proof for every quotient family member"
        ),
        "family_closed": lane_oracle_agreement < 0.6 and nr1_lane_agreement < 0.6,
        "family_closure_reason": (
            "RC1 oracle containment is low and NR1 Lane also collapses"
            if lane_oracle_agreement < 0.6 and nr1_lane_agreement < 0.6
            else (
                "RC1's existing codebook can represent Lane under oracle assignment, so family closure is forbidden"
                if lane_oracle_agreement > 0.9
                else "the pre-registered representability thresholds do not resolve family closure"
            )
        ),
    }

    confusion_records = {
        "rc1_current": file_record(current_confusion_path),
        "rc1_full_hamming_oracle": file_record(confusion_paths["rc1_full_hamming_oracle"]),
        "rc1_lane_oracle": file_record(confusion_paths["rc1_lane_oracle"]),
        "nr1_k32": file_record(nr1_confusion_path),
    }
    retained = {
        "confusions": confusion_records,
        "full_hamming_site_assignment": file_record(full_site_path),
        "lane_recall_site_assignment": file_record(lane_site_path),
        "intrinsic_program_fields": file_record(diversity_path),
        "per_class_decomposition": file_record(decomposition_path),
        "oracle_progress": [
            file_record(path)
            for path in sorted((args.out_dir / "retained/oracle_progress").glob("*"))
            if path.is_file()
        ],
    }
    oracle_checkpoint = checkpoint(
        args.out_dir,
        3,
        "rc1_oracles_and_diversity_complete",
        {
            "status": "complete",
            "oracle_diagnosis": oracle_diagnosis,
            "intrinsic_diversity": intrinsic_diversity,
            "retained": retained,
        },
    )
    nr1_checkpoint = checkpoint(
        args.out_dir,
        4,
        "nr1_per_class_complete",
        {
            "status": "complete",
            "rows": decompositions["nr1_k32"],
            "confusion": confusion_records["nr1_k32"],
        },
    )

    scientific = {
        "axis": AXIS,
        "score_claim": False,
        "evidence_warning": "All token agreement and IoU values are PROXY-NOT-SCORE.",
        "cb2_reproduction": cb2_reproduction,
        "intrinsic_diversity": intrinsic_diversity,
        "assignment_and_objective_geometry": oracle_diagnosis,
        "nr1_k32_per_class": decompositions["nr1_k32"],
        "assignment_mechanism_and_byte_cost": assignment_mechanism,
        "prior_law_verdict": prior_law,
    }
    result = {
        "schema": "ddm_lq1_lane_quotient_representability_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
        "out_dir": str(args.out_dir),
        "argv": [str(value) for value in os.sys.argv],
        "cwd": str(Path.cwd()),
        "determinism": "no stochastic operations; explicit smallest-index oracle tie breaks",
        "resumability": ("five retained oracle progress arrays plus atomic next-program checkpoint after every chunk"),
        "inputs": inputs,
        "scientific": scientific,
        "retained": retained,
        "checkpoints": [
            custody_checkpoint,
            cb2_checkpoint,
            oracle_checkpoint,
            nr1_checkpoint,
        ],
        "boundaries": [
            "no scorer, RGB receiver, upstream evaluator, Metal, MPS, CUDA, or Modal ran",
            "no RC1 or NR1 payload was edited, refit, or recut",
            "oracle assignments are upper-bound diagnostic fields, not encoder outputs",
            "NR1 per-class agreement does not test NR1 oracle containment",
            "the exact compressed byte delta of a replacement RC1 assignment stream remains unmeasured",
        ],
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.out_dir / "RESULT.json", result)
    checkpoint(
        args.out_dir,
        5,
        "terminal_result_sealed",
        {
            "status": "complete",
            "result": file_record(args.out_dir / "RESULT.json"),
            "prior_law_outcome": prior_law["outcome"],
            "frontier_moved": False,
        },
    )
    print(json.dumps(scientific, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
