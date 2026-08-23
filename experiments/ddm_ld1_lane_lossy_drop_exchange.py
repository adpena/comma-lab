#!/usr/bin/env python3
"""Build and reconcile the scorer-free half of the LD1 Lane-lossy curve.

The swept formulation is deliberately narrow and distortion-bearing: among
positions where both DALI GT and the shipped decoded field are Lane, rank by
BL1's retained shipped-symbol cost and progressively merge Lane into Road.
Every rung therefore changes the decoded field.  The shipped DX2 runtime and
its 19-member HPAC/RC64 law are not modified; ``ddm_jg2_tail_reencode.py``
prices each retained field separately with a real full-stream re-encode.

This program does not run SegNet and cannot adjudicate the joint score.  It
materializes all fields and masks needed by the exclusive scorer-lane owner,
and its ``finalize-rate`` stage refuses to call a byte delta measured unless
the n600 encoder control reproduces the shipped stream byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VERTIGO = Path("/Volumes/VertigoDataTier/pact")
LOCAL_RECEIPT_ROOT = (
    REPO / ".omx" / "tmp" / "arm_receipts_local" / "ddm_ld1_lane_lossy_drop_exchange"
)
DEFAULT_STORE = LOCAL_RECEIPT_ROOT / "measurement_v1"
BL1 = VERTIGO / "ddm_bl1_per_position_bit_allocation" / "measurement_v1"
MS9 = VERTIGO / "ddm_ms9_dx2_seg_manufactured_fraction"
TO2_INPUT = VERTIGO / "ddm_to2_token_ordering_race" / "measurement_v1" / "retained" / "input"
DX2_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")

ARCHIVE = DX2_RUNTIME / "archive.zip"
STREAM = TO2_INPUT / "dx2_token_stream_rc64.bin"
TOKENS = BL1 / "retained" / "fields" / "decoded_tokens_instrumented.u8"
COST = BL1 / "retained" / "fields" / "position_rc64_frequency_cost_bits.f64le.bin"
GT = VERTIGO / "ddm_qs3_20260813" / "retained" / "inputs" / "gt_argmax_n600.npy"
BL1_RESULT = BL1 / "RESULT.json"
BL1_MANIFEST = BL1 / "MANIFEST.json"
MS9_RESULT = MS9 / "MS9_FIELD_REPLAY.json"
MS9_MANIFEST = MS9 / "MASK_MANIFEST.json"

N, HEIGHT, WIDTH = 600, 384, 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
PACKED_BYTES = (POSITIONS + 7) // 8
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
RUNG_COUNTS = (2_500, 5_000, 10_000, 20_000, 40_000, 60_000)
LANE, ROAD = 1, 0
S_PER_BYTE = 25.0 / 37_545_489.0
S_PER_FLIP = 100.0 / POSITIONS
DEMAND_BYTES = 42_382
MS9_SURVIVAL = 2_264 / 9_182
MIN_RESERVE_BYTES = 8 << 30
ESTIMATED_BYTES_PER_REMAINING_RUNG = 256 << 20
ESTIMATED_FIXED_REMAINING_BYTES = 256 << 20

EXPECTED = {
    "archive": (180_368, "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"),
    "stream": (113_777, "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"),
    "tokens": (117_964_800, "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"),
    "cost": (943_718_400, "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"),
    "gt": (117_964_928, "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"),
    "bl1_result": (318_937, "f8835acf27c3b46bf95f7cd1954e08d72d591854f8f78ac6c902889a064b6621"),
    "bl1_manifest": (56_421, "0b2ca8ec51738b6e7ee5940d262be7226457fcd5a4f8e56f4bfb5b98184a59ac"),
    "ms9_result": (37_911, "c14494194bda3d0dba30c1a5e5813d4a01646571239352398edfa29f8f79ddd5"),
    "ms9_manifest": (24_980, "2df0abbae76a1234f8af0e5a08bd857254cf9a6299f63a61b9ae06021b329cdd"),
}


class Ld1Error(RuntimeError):
    """Fail-closed custody, field, or reconciliation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify(path: Path, key: str) -> dict[str, object]:
    expected_bytes, expected_sha = EXPECTED[key]
    if not path.is_file():
        raise Ld1Error(f"required source is absent: {path}")
    fact = file_fact(path)
    if fact["bytes"] != expected_bytes or fact["sha256"] != expected_sha:
        raise Ld1Error(f"source drift for {key}: {fact}")
    return fact


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_raw(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(np.ascontiguousarray(value).tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def popcount_packbits(path: Path) -> int:
    table = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    total = 0
    payload = np.memmap(path, dtype=np.uint8, mode="r")
    for start in range(0, payload.size, 1 << 24):
        total += int(table[np.asarray(payload[start : start + (1 << 24)])].sum())
    return total


def source_facts() -> dict[str, object]:
    return {
        "archive": verify(ARCHIVE, "archive"),
        "stream": verify(STREAM, "stream"),
        "tokens": verify(TOKENS, "tokens"),
        "cost": verify(COST, "cost"),
        "gt": verify(GT, "gt"),
        "bl1_result": verify(BL1_RESULT, "bl1_result"),
        "bl1_manifest": verify(BL1_MANIFEST, "bl1_manifest"),
        "ms9_result": verify(MS9_RESULT, "ms9_result"),
        "ms9_manifest": verify(MS9_MANIFEST, "ms9_manifest"),
        "implementation": file_fact(Path(__file__)),
    }


def reproduce_sources() -> dict[str, object]:
    gt = np.load(GT, mmap_mode="r", allow_pickle=False).reshape(-1)
    tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(POSITIONS,))
    cost = np.memmap(COST, dtype="<f8", mode="r", shape=(POSITIONS,))
    lane_mask = np.asarray(gt == LANE)
    lane_positions = int(lane_mask.sum())
    lane_bits = float(np.asarray(cost[lane_mask], dtype=np.float64).sum())
    total_bits = float(np.asarray(cost, dtype=np.float64).sum())

    bl1 = json.loads(BL1_RESULT.read_text())
    lane_rows = [row for row in bl1["gt_class_join"] if row["class_id"] == LANE]
    if len(lane_rows) != 1:
        raise Ld1Error("BL1 result does not carry exactly one Lane GT row")
    expected_lane = lane_rows[0]
    if lane_positions != 690_754 or not math.isclose(lane_bits, 305_463.96947306144, abs_tol=1e-8):
        raise Ld1Error("direct BL1 Lane-row reproduction drifted")
    if not math.isclose(total_bits, 910_209.2806090603, abs_tol=1e-8):
        raise Ld1Error("direct BL1 total-bit reproduction drifted")
    if lane_positions != expected_lane["positions"] or not math.isclose(
        lane_bits, expected_lane["bits"], abs_tol=1e-8
    ):
        raise Ld1Error("direct Lane row disagrees with BL1 RESULT.json")

    ms9_manifest = json.loads(MS9_MANIFEST.read_text())
    manifest_rows = {row["name"]: row for row in ms9_manifest["masks"]}
    wanted = {
        "final_error": 23_757,
        "representation_error": 9_182,
        "representation_survived_final_error": 2_264,
        "manufactured_final_error": 21_493,
        "representation_corrected": 6_918,
        "class_1_lane_representation_error": 1_907,
        "class_1_lane_representation_survived_final_error": 571,
    }
    replayed: dict[str, Any] = {}
    for name, expected_count in wanted.items():
        row = manifest_rows[name]
        path = Path(row["path"])
        fact = file_fact(path)
        if fact["bytes"] != row["bytes"] or fact["sha256"] != row["sha256"]:
            raise Ld1Error(f"MS9 mask drift for {name}")
        count = popcount_packbits(path)
        if count != expected_count:
            raise Ld1Error(f"MS9 {name} count drift: {count} != {expected_count}")
        replayed[name] = {**fact, "count": count, "denominator": POSITIONS}
    if replayed["representation_error"]["count"] != (
        replayed["representation_survived_final_error"]["count"]
        + replayed["representation_corrected"]["count"]
    ):
        raise Ld1Error("MS9 transmitted-error split is not additive")
    if replayed["final_error"]["count"] != (
        replayed["representation_survived_final_error"]["count"]
        + replayed["manufactured_final_error"]["count"]
    ):
        raise Ld1Error("MS9 final-error split is not additive")

    eligible = int(np.count_nonzero((gt == LANE) & (tokens == LANE)))
    return {
        "schema": "ddm_ld1_source_reproduction.v1",
        "axis": "[macOS-CPU advisory / scorer-free retained-field replay]",
        "gt_lineage": "DALI_NVDEC authority GT inherited and pinned through MS9/DX2 custody",
        "denominator_positions": POSITIONS,
        "bl1_lane_row": {
            "positions": lane_positions,
            "area_fraction": lane_positions / POSITIONS,
            "bits": lane_bits,
            "bit_fraction": lane_bits / total_bits,
            "bytes_equivalent": lane_bits / 8.0,
            "bits_per_position": lane_bits / lane_positions,
            "enrichment_over_mean": (lane_bits / lane_positions) / (total_bits / POSITIONS),
            "eligible_correct_lane_positions": eligible,
        },
        "bl1_total_frequency_cost_bits": total_bits,
        "ms9_split_replayed_from_masks": replayed,
        "ms9_survival_fraction": 2_264 / 9_182,
        "ms9_lane_survival_fraction": 571 / 1_907,
        "ms9_manufactured_fraction_of_final": 21_493 / 23_757,
    }


def require_local_store(store: Path) -> Path:
    """Bind every LD1 output to the charter's explicit local-disk opt-in."""
    resolved = store.resolve()
    root = LOCAL_RECEIPT_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise Ld1Error(
            f"LD1 outputs must stay under the chartered local receipt root {root}; "
            f"refusing {resolved} (both SSD tiers are full)"
        )
    return resolved


def storage_preflight(store: Path) -> dict[str, object]:
    store = require_local_store(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(store.parent)
    complete = len(list((store / "retained" / "rungs").glob("*/RUNG.json")))
    remaining = len(RUNG_COUNTS) - complete
    if remaining < 0:
        raise Ld1Error("more completed rung receipts exist than the registered ladder")
    estimated_new = remaining * ESTIMATED_BYTES_PER_REMAINING_RUNG + ESTIMATED_FIXED_REMAINING_BYTES
    required = estimated_new + MIN_RESERVE_BYTES
    receipt = {
        "schema": "ddm_ld1_storage_preflight.v1",
        "tier": "LOCAL_DISK_EXPLICIT_OPT_IN",
        "chartered_root": str(LOCAL_RECEIPT_ROOT),
        "free_bytes": usage.free,
        "complete_rung_receipts_seen": complete,
        "remaining_rungs": remaining,
        "estimated_bytes_per_remaining_rung": ESTIMATED_BYTES_PER_REMAINING_RUNG,
        "estimated_fixed_remaining_bytes": ESTIMATED_FIXED_REMAINING_BYTES,
        "estimated_new_bytes": estimated_new,
        "minimum_reserve_bytes": MIN_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "large_payload_policy": (
            "all LD1 outputs remain under the local chartered receipt root; "
            "both SSD tiers were measured full and receive no LD1 writes"
        ),
    }
    atomic_json(store / "PREFLIGHT.json", receipt)
    if not receipt["passed"]:
        raise Ld1Error(f"local-disk storage preflight refused: {receipt}")
    return receipt


def quarantine_stale_partials(store: Path) -> list[dict[str, object]]:
    """Preserve interrupted materializer payloads and refuse live concurrency."""
    quarantined: list[dict[str, object]] = []
    retained = store / "retained"
    if not retained.exists():
        return quarantined
    for path in sorted(retained.rglob("*.partial.*")):
        try:
            pid = int(path.name.rsplit(".partial.", 1)[1])
        except (IndexError, ValueError) as error:
            raise Ld1Error(f"unrecognized partial artifact name: {path}") from error
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        else:
            alive = True
        if alive:
            raise Ld1Error(f"another LD1 materializer still owns {path} (pid {pid})")
        before = file_fact(path)
        destination = retained / "failed_attempts" / f"pid_{pid}" / (
            path.name.rsplit(".partial.", 1)[0] + ".interrupted_partial"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_fact(destination)["sha256"] != before["sha256"]:
                raise Ld1Error(f"failed-attempt custody collision at {destination}")
            destination = destination.with_name(destination.name + ".repeat")
            if destination.exists():
                raise Ld1Error(f"repeated failed-attempt custody collision at {destination}")
            os.replace(path, destination)
        else:
            os.replace(path, destination)
        quarantined.append({
            "source": before,
            "retained_as": file_fact(destination),
            "producer_pid": pid,
            "disposition": "RETAINED_INTERRUPTED_PARTIAL_NO_MEASUREMENT_CLAIM",
        })
    if quarantined:
        atomic_json(
            store / "INTERRUPTED_PARTIALS.json",
            {
                "schema": "ddm_ld1_interrupted_partials.v1",
                "artifacts": quarantined,
                "cleanup": "none; bytes moved losslessly under failed-attempt custody",
            },
        )
    return quarantined


def rung_tag(count: int) -> str:
    return f"lane2road_topcost_k{count:06d}"


def validate_existing_rung(path: Path, tag: str, count: int) -> dict[str, object] | None:
    """Return a complete rung receipt, or refuse a corrupt/ambiguous resume."""
    if not path.is_file():
        return None
    row = json.loads(path.read_text())
    if row.get("tag") != tag or row.get("tokens_changed") != count:
        raise Ld1Error(f"existing rung receipt is bound to another stage: {path}")
    artifact_rows = [
        row["candidate_field"],
        row["selected_indices"],
        row["transmitted_error_field"],
        row["edits_for_exact_reencoder"],
        *[entry["artifact"] for entry in row["transmitted_errors_added_by_gt_class"]],
    ]
    for expected in artifact_rows:
        path_value = Path(expected["path"])
        actual = file_fact(path_value)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise Ld1Error(f"existing rung artifact drifted: {path_value}")
    return row


def materialize(store: Path) -> dict[str, object]:
    interrupted = quarantine_stale_partials(store)
    preflight = storage_preflight(store)
    sources = source_facts()
    reproduction = reproduce_sources()
    atomic_json(store / "PREFLIGHT.json", preflight)
    atomic_json(store / "SOURCE_REPRODUCTION.json", {**reproduction, "sources": sources})

    tokens = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(POSITIONS,))
    gt = np.load(GT, mmap_mode="r", allow_pickle=False).reshape(-1)
    cost = np.memmap(COST, dtype="<f8", mode="r", shape=(POSITIONS,))
    eligible = np.flatnonzero((gt == LANE) & (tokens == LANE)).astype(np.uint64)
    eligible_cost = np.asarray(cost[eligible], dtype=np.float64)
    # Descending shipped-symbol cost; flat raster index breaks exact ties.
    order = np.lexsort((eligible, -eligible_cost))
    ranked = eligible[order]
    ranked_cost = eligible_cost[order]
    ranking_path = store / "retained" / "selection" / "eligible_correct_lane_ranked_by_cost.u64.npy"
    atomic_npy(ranking_path, ranked)
    ranking_cost_path = store / "retained" / "selection" / "eligible_cost_ranked.f64.npy"
    atomic_npy(ranking_cost_path, ranked_cost)

    rows: list[dict[str, object]] = []
    fields_dir = store / "retained" / "fields"
    masks_dir = store / "retained" / "transmitted_masks"
    edits_dir = store / "retained" / "edits"
    selection_dir = store / "retained" / "selection"
    class_denominators = [int(np.count_nonzero(gt == class_id)) for class_id in range(len(CLASS_NAMES))]
    for count in RUNG_COUNTS:
        tag = rung_tag(count)
        rung_receipt = store / "retained" / "rungs" / tag / "RUNG.json"
        existing = validate_existing_rung(rung_receipt, tag, count)
        if existing is not None:
            rows.append(existing)
            continue
        chosen = np.sort(ranked[:count])
        chosen_path = selection_dir / f"selected_{tag}.u64.npy"
        atomic_npy(chosen_path, chosen)

        field_path = fields_dir / f"tokens_{tag}.u8"
        temporary = field_path.with_name(field_path.name + f".partial.{os.getpid()}")
        field_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=(POSITIONS,))
        candidate[:] = tokens
        candidate[chosen] = ROAD
        candidate.flush()
        del candidate
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, field_path)
        candidate = np.memmap(field_path, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
        changed = np.asarray(candidate.reshape(-1) != tokens)
        if int(changed.sum()) != count:
            raise Ld1Error(f"{tag}: materialized field changed the wrong number of symbols")
        if np.any(np.asarray(gt.reshape(-1)[changed]) != LANE):
            raise Ld1Error(f"{tag}: a changed symbol is outside DALI-GT Lane")
        if np.any(np.asarray(candidate.reshape(-1)[changed]) != ROAD):
            raise Ld1Error(f"{tag}: a changed symbol did not merge into Road")

        packed = np.packbits(changed, bitorder="little")
        transmitted_path = masks_dir / f"transmitted_errors_added_{tag}.n600.packbits"
        atomic_raw(transmitted_path, packed)
        class_masks: list[dict[str, object]] = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_path = masks_dir / (
                f"transmitted_errors_added_class_{class_id}_{class_name.lower()}_{tag}.n600.packbits"
            )
            atomic_raw(class_path, packed if class_id == LANE else np.zeros(PACKED_BYTES, dtype=np.uint8))
            class_masks.append({
                "class_id": class_id,
                "class_name": class_name,
                "count": count if class_id == LANE else 0,
                "denominator_gt_pixels": class_denominators[class_id],
                "artifact": file_fact(class_path),
            })

        changed_frames = np.flatnonzero(changed.reshape(N, PLANE).any(axis=1))
        edits_path = edits_dir / f"edits_{tag}.npz"
        edits_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_edits = edits_path.with_name(edits_path.name + f".partial.{os.getpid()}")
        with temporary_edits.open("wb") as handle:
            np.savez_compressed(handle, **{str(int(i)): candidate[int(i)] for i in changed_frames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_edits, edits_path)
        del candidate

        row = {
            "tag": tag,
            "rank": len(rows) + 1,
            "direction": "DOWN_ONLY_NESTED",
            "mechanism": "DALI-GT-correct Lane token merged into Road, highest BL1 shipped-symbol costs first",
            "tokens_changed": count,
            "transmitted_errors_added": count,
            "transmitted_errors_added_by_gt_class": class_masks,
            "incumbent_cost_bits_selected": float(ranked_cost[:count].sum()),
            "incumbent_cost_bytes_equivalent_selected": float(ranked_cost[:count].sum() / 8.0),
            "candidate_field": file_fact(field_path),
            "selected_indices": file_fact(chosen_path),
            "transmitted_error_field": file_fact(transmitted_path),
            "edits_for_exact_reencoder": file_fact(edits_path),
            "final_argmax_field": None,
            "final_argmax_status": "QUEUED_NO_SCORER_AUTHORITY",
        }
        atomic_json(rung_receipt, row)
        rows.append(row)

    result = {
        "schema": "ddm_ld1_materialized_ladder.v1",
        "status": "MATERIALIZED_SCORER_FREE",
        "axis": "[macOS-CPU advisory / scorer-free field materialization]",
        "score_claim": False,
        "gt_lineage": reproduction["gt_lineage"],
        "selection": {
            "eligible_positions": int(ranked.size),
            "nested": True,
            "tie_break": "ascending flat raster index after descending BL1 cost",
            "ranking": file_fact(ranking_path),
            "ranked_cost": file_fact(ranking_cost_path),
        },
        "rungs": rows,
        "interrupted_attempts_retained": interrupted,
        "argv": sys.argv,
    }
    atomic_json(store / "MATERIALIZED_LADDER.json", result)
    return result


def finalize_rate(store: Path) -> dict[str, object]:
    materialized = json.loads((store / "MATERIALIZED_LADDER.json").read_text())
    control_path = store / "rate" / "retained" / "S1_control_600.json"
    if not control_path.is_file():
        raise Ld1Error("n600 re-encoder control receipt is absent")
    control = json.loads(control_path.read_text())
    if not control.get("byte_identical"):
        raise Ld1Error("n600 re-encoder control is not byte-identical")
    if control.get("shipped_token_stream_sha256") != EXPECTED["stream"][1]:
        raise Ld1Error("n600 control is bound to the wrong shipped token stream")

    rows: list[dict[str, object]] = []
    for source_row in materialized["rungs"]:
        tag = source_row["tag"]
        receipt_path = store / "rate" / "retained" / f"S1_encode_{tag}.json"
        if not receipt_path.is_file():
            raise Ld1Error(f"rate receipt absent for {tag}")
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get("delta_trustworthy"):
            raise Ld1Error(f"rate receipt is unproven for {tag}")
        if receipt.get("tokens_changed") != source_row["tokens_changed"]:
            raise Ld1Error(f"rate receipt changed-count drift for {tag}")
        if receipt["pointer_archive"]["sha256"] != EXPECTED["archive"][1]:
            raise Ld1Error(f"rate receipt pointer drift for {tag}")
        delta = int(receipt["archive_delta_bytes"])
        saved = -delta
        rows.append({
            "tag": tag,
            "tokens_changed": int(source_row["tokens_changed"]),
            "transmitted_errors_added": int(source_row["transmitted_errors_added"]),
            "archive_bytes_base": int(receipt["archive_bytes_base"]),
            "archive_bytes_candidate": int(receipt["archive_bytes_candidate"]),
            "archive_delta_bytes": delta,
            "bytes_saved": saved,
            "share_of_42382_byte_demand": saved / DEMAND_BYTES,
            "realized_bits_saved_per_added_transmitted_error": saved * 8.0 / source_row["transmitted_errors_added"],
            "delta_S_rate": delta * S_PER_BYTE,
            "d_seg": None,
            "delta_d_seg": None,
            "final_flips_added": None,
            "introduced_error_survival_fraction": None,
            "rate_plus_seg_delta_S": None,
            "joint_status": "PENDING_EXCLUSIVE_N600_SCORER",
            "candidate_archive": receipt["candidate_archive"],
            "reencoded_stream": receipt["stream"],
            "bits_per_frame_ledger": receipt["bits_per_frame_ledger"],
            "rate_receipt": file_fact(receipt_path),
        })
    result = {
        "schema": "ddm_ld1_rate_curve.v1",
        "status": "MEASURED_RATE_ONLY_JOINT_ADJUDICATION_QUEUED",
        "axis": "[macOS-CPU advisory / scorer-free EXACT full-stream re-encode]",
        "score_claim": False,
        "control": file_fact(control_path),
        "shipped_allocation": {
            "archive_bytes": EXPECTED["archive"][0],
            "archive_sha256": EXPECTED["archive"][1],
            "transmitted_errors": 9_182,
            "final_errors": 23_757,
            "d_seg": 23_757 / POSITIONS,
            "seg_term": 100.0 * 23_757 / POSITIONS,
            "gt_lineage": "DALI_NVDEC authority GT inherited and pinned through MS9/DX2 custody",
        },
        "exchange_constants": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_final_flip": S_PER_FLIP,
            "bytes_per_final_flip": S_PER_FLIP / S_PER_BYTE,
            "demand_bytes": DEMAND_BYTES,
            "ms9_survival_fraction": MS9_SURVIVAL,
            "charter_delta_S_formula": "25*delta_archive_bytes/37545489 + 100*delta_d_seg",
            "pose_axis_included": False,
            "full_contest_score_claim": False,
        },
        "rows": rows,
        "adjudication": "WITHHELD: bytes alone cannot select a rung or close the formulation",
    }
    atomic_json(store / "RATE_CURVE.json", result)
    return result


def write_rate_commands(store: Path) -> dict[str, object]:
    """Persist the exact deterministic argv/env used for the scorer-free rate jobs."""
    materialized = json.loads((store / "MATERIALIZED_LADDER.json").read_text())
    rate_store = store / "rate"
    shared = [
        ".venv/bin/python",
        "experiments/ddm_jg2_tail_reencode.py",
        "--store",
        str(rate_store),
        "--runtime-root",
        str(DX2_RUNTIME),
        "--tokens",
        str(TOKENS),
        "--frames",
        str(N),
        "--checkpoint-every",
        "20",
        "--resume",
    ]
    commands = [{"tag": "control_600", "argv": [*shared[:2], "--stage", "control", *shared[2:]]}]
    for row in materialized["rungs"]:
        commands.append({
            "tag": row["tag"],
            "argv": [
                *shared[:2],
                "--stage",
                "encode",
                *shared[2:],
                "--pointer-archive",
                str(ARCHIVE),
                "--expect-pointer-sha256",
                EXPECTED["archive"][1],
                "--edits",
                row["edits_for_exact_reencoder"]["path"],
                "--tag",
                row["tag"],
                "--wait-for-control-seconds",
                "1800",
            ],
        })
    payload = {
        "schema": "ddm_ld1_rate_commands.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT full-stream re-encode]",
        "environment": {
            "TAC_JG2_RC64_SOURCE": (
                "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/"
                "experiment_book/src/cpr1_sub4/entropy/rc64_backend.c"
            ),
            "TAC_JG2_RC64_SOURCE_SHA256": "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6",
        },
        "commands": commands,
        "resumability": "full adaptive state and RC64 interval checkpoint every 20 frames",
        "output_tier": "LOCAL_DISK_EXPLICIT_OPT_IN",
    }
    atomic_json(store / "RATE_RUN_COMMANDS.json", payload)
    return payload


def write_queue(store: Path) -> dict[str, object]:
    rate_curve = store / "RATE_CURVE.json"
    payload = {
        "schema": "ddm_ld1_scorer_fire_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN / exclusive n600 scorer-lane custodian",
        "consumer": "LD1 joint Lane-drop exchange adjudication",
        "consumer_store": str(store / "scorer"),
        "blocked_by": "LD1 charter does not grant the sole n600 scorer lane",
        "fire_trigger": [
            "MAIN explicitly transfers the exclusive n600 scorer lane to LD1",
            "no other full-n600 scorer job is active",
            "RATE_CURVE.json and every candidate archive validate against their recorded SHA-256",
            "fresh local-disk storage preflight admits every retained final-argmax field and lossless raw-output custody route",
        ],
        "ordered_actions": [
            {
                "order": 1,
                "action": "score the six rungs serially, never concurrently, through tools/fire_local_advisory.py",
                "input": str(rate_curve),
            },
            {
                "order": 2,
                "action": (
                    "retain each rung's DALI-GT-aligned terminal SegNet argmax field, final-error field, "
                    "and five GT-class final-error masks; retain or losslessly certify the produced raw payload"
                ),
            },
            {
                "order": 3,
                "action": (
                    "join candidate labels, terminal argmax, and GT to report added transmitted errors, "
                    "added final flips, survival fraction, per-class d_seg, and the charter's rate+seg delta_S"
                ),
                "positive_controls": {
                    "baseline_final_errors": 23_757,
                    "baseline_representation_errors": 9_182,
                    "baseline_representation_survived": 2_264,
                    "baseline_manufactured": 21_493,
                },
            },
            {
                "order": 4,
                "action": "select the measured minimum joint delta_S or record the honest all-positive formulation-scoped close",
            },
        ],
    }
    atomic_json(store / "SCORER_FIRE_ORDER.json", payload)
    return payload


def write_manifest(store: Path) -> dict[str, object]:
    manifest_path = store / "MANIFEST.json"
    rows = []
    for path in sorted(store.rglob("*")):
        if not path.is_file() or path == manifest_path or ".partial." in path.name:
            continue
        rows.append({"relative_path": str(path.relative_to(store)), **file_fact(path)})
    payload = {
        "schema": "ddm_ld1_retention_manifest.v1",
        "tier": "LOCAL_DISK_EXPLICIT_OPT_IN",
        "root": str(store),
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "retention": "KEEP; no cleanup authorized without lossless replacement custody and a new manifest",
    }
    atomic_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=("materialize", "finalize-rate", "commands", "queue", "manifest"),
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.store = require_local_store(args.store)
    result = {
        "materialize": materialize,
        "finalize-rate": finalize_rate,
        "commands": write_rate_commands,
        "queue": write_queue,
        "manifest": write_manifest,
    }[args.stage](args.store)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
