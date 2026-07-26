#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure whether W_seg's semantic program is the exact C1/S2 baseline.

This encode-side, research-only diagnostic joins two original-work objects:

* the receiver-closed WS1 ``W_seg`` archive, decoded to its *semantic cells*
  through :mod:`tac.optimization.ddm_ws1_warm_start`; and
* the exact C1-to-M2 error inventory carried by the geometry-bound S2 packet.

The live M2 target labels are regenerated from the custodied raw bytes with
the frozen CPU SegNet in evaluator-sized batches.  Cached labels are compared
only as a diagnostic because the custodied bridge records three known cache
disagreements.  Replacing every S2 target site with its stored baseline class
then reconstructs the complete C1 baseline partition.  No target labels,
scorer state, or result from this tool is candidate payload.

Small write-once batch receipts make the n600 pass resumable without storing
another dense label table.  This tool does not build an archive, run the
official evaluator, claim a score, or move the frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization import ddm_ws1_warm_start as ws1  # noqa: E402
from tac.optimization.s2_partition_seed import (  # noqa: E402
    SEMANTIC_NAMES,
    PartitionEvent,
    PartitionEventSeed,
    decode_partition_seed,
    encode_partition_seed,
)
from tools import measure_v10_free_predictor_floor as scorer  # noqa: E402

SCHEMA = "tac.c0b_wseg_c1_baseline_join.v1"
STAGE_SCHEMA = "tac.c0b_wseg_c1_baseline_join_stage.v1"
BRIDGE_SCHEMA = "tac.c0b_s2_debt_bridge.v4"
DEBT_SCHEMA = "tac.coupled_witness_raw_debt.v2"
WS2_SCHEMA = "ddm_ws2_warm_start_custody_producer.v1"
PAIR_COUNT = 600
STAGE_PAIRS = 16
SCORER_HW = (384, 512)
CAMERA_HW = (874, 1164)

DEFAULT_BRIDGE = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/c1_s2_exact_debt_bridge_v4.json"
)
DEFAULT_WS2_RECEIPT = REPO / ".omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json"
DEFAULT_OUTPUT = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_bj1_wseg_c1_baseline_join.json"
)


class BaselineJoinError(RuntimeError):
    """Fail-closed custody, geometry, resume, or join error."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BaselineJoinError("value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BaselineJoinError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(memoryview(array).cast("B"))


def _with_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise BaselineJoinError(f"hash field already exists: {field}")
    result = dict(payload)
    result[field] = _sha256_bytes(_canonical(result))
    return result


def _validate_hash(payload: Mapping[str, Any], field: str) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise BaselineJoinError(f"{field} is absent or malformed")
    body = {key: value for key, value in payload.items() if key != field}
    if _sha256_bytes(_canonical(body)) != observed:
        raise BaselineJoinError(f"{field} differs from its receipt body")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineJoinError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineJoinError(f"{label} root must be an object")
    return value


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise BaselineJoinError(f"write-once output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise BaselineJoinError(f"concurrent write-once output differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _exact_file(path: Path, row: Mapping[str, Any], label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    size = row.get("bytes")
    digest = row.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or not resolved.is_file()
        or resolved.stat().st_size != size
        or _sha256_file(resolved) != digest
    ):
        raise BaselineJoinError(f"{label} custody differs")
    return {"path": str(resolved), "bytes": size, "sha256": digest}


def _validate_bridge(path: Path) -> tuple[dict[str, Any], dict[str, Any], PartitionEventSeed]:
    bridge = _load_json(path, "C0B/S2 bridge")
    _validate_hash(bridge, "receipt_sha256")
    if (
        bridge.get("schema") != BRIDGE_SCHEMA
        or bridge.get("score_claim") is not False
        or bridge.get("promotion_eligible") is not False
        or bridge.get("pointer_moved") is not False
    ):
        raise BaselineJoinError("bridge schema or authority flags differ")
    geometry = bridge.get("authority_geometry")
    identity = bridge.get("identity")
    s2_row = bridge.get("s2")
    debt_row = bridge.get("debt_receipt")
    if not all(isinstance(value, Mapping) for value in (geometry, identity, s2_row, debt_row)):
        raise BaselineJoinError("bridge custody sections are malformed")
    if (
        geometry.get("pair_count") != PAIR_COUNT
        or geometry.get("scorer_hw") != list(SCORER_HW)
        or geometry.get("total_seg_sites") != PAIR_COUNT * math.prod(SCORER_HW)
        or geometry.get("s2_geometry_exact") is not True
    ):
        raise BaselineJoinError("bridge authority geometry differs")
    debt_path = Path(str(debt_row.get("path", ""))).resolve()
    if (
        not debt_path.is_file()
        or debt_path.stat().st_size != debt_row.get("bytes")
        or _sha256_file(debt_path) != debt_row.get("sha256")
    ):
        raise BaselineJoinError("bridge debt receipt file custody differs")
    debt = _load_json(debt_path, "C1 live-target debt")
    _validate_hash(debt, "receipt_sha256")
    if debt.get("schema") != DEBT_SCHEMA or debt.get("receipt_sha256") != debt_row.get("receipt_sha256"):
        raise BaselineJoinError("bridge debt receipt body custody differs")
    packet_path = Path(str(s2_row.get("packet_path", ""))).resolve()
    packet = _exact_file(
        packet_path,
        {"bytes": s2_row.get("packet_bytes"), "sha256": s2_row.get("packet_sha256")},
        "S2 packet",
    )
    payload = packet_path.read_bytes()
    try:
        seed = decode_partition_seed(payload)
        reencoded = encode_partition_seed(seed)
    except ValueError as exc:
        raise BaselineJoinError("S2 packet parse-back failed") from exc
    if reencoded != payload:
        raise BaselineJoinError("S2 packet is not its canonical exact re-encoding")
    events = [[e.pair, e.row, e.col, e.target_class, e.baseline_class] for e in seed.events]
    if (
        seed.n_pairs != PAIR_COUNT
        or (seed.height, seed.width) != SCORER_HW
        or tuple(seed.semantic_class_ids) != tuple(range(len(SEMANTIC_NAMES)))
        or len(events) != identity.get("event_count")
        or _sha256_bytes(_canonical(events)) != identity.get("event_stream_sha256")
    ):
        raise BaselineJoinError("decoded S2 population differs from bridge identity")
    target_row = debt_row.get("target_raw")
    if not isinstance(target_row, Mapping):
        raise BaselineJoinError("bridge lacks exact live-target raw custody")
    target = _exact_file(Path(str(target_row.get("path", ""))), target_row, "M2 live-target raw")
    config = debt.get("config")
    scorer_custody = debt.get("scorer_custody")
    if not isinstance(config, Mapping) or not isinstance(scorer_custody, Mapping):
        raise BaselineJoinError("debt receipt lacks config/scorer custody")
    if (
        config.get("pair_count") != PAIR_COUNT
        or config.get("stage_pairs") != STAGE_PAIRS
        or config.get("scorer_hw") != list(SCORER_HW)
        or config.get("target_raw") != target_row
    ):
        raise BaselineJoinError("debt and bridge target geometry differ")
    cache = config.get("cache")
    if not isinstance(cache, Mapping):
        raise BaselineJoinError("debt receipt lacks canonical cache diagnostic custody")
    cache_custody = _exact_file(Path(str(cache.get("path", ""))), cache, "n600 diagnostic cache")
    runtime_files: dict[str, Any] = {}
    for key in ("modules_py", "frame_utils_py", "segnet_weights", "posenet_weights"):
        row = scorer_custody.get(key)
        if not isinstance(row, Mapping):
            raise BaselineJoinError(f"debt scorer custody lacks {key}")
        runtime_files[key] = _exact_file(Path(str(row.get("path", ""))), row, key)
    cache_mismatches = bridge.get("r2b", {}).get("cache_label_mismatches")
    if isinstance(cache_mismatches, bool) or not isinstance(cache_mismatches, int) or cache_mismatches < 0:
        raise BaselineJoinError("bridge cache-label discrepancy count is malformed")
    return (
        bridge,
        {
            "bridge": {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "receipt_sha256": bridge["receipt_sha256"],
            },
            "debt": {
                "path": str(debt_path),
                "bytes": debt_path.stat().st_size,
                "sha256": _sha256_file(debt_path),
                "receipt_sha256": debt["receipt_sha256"],
            },
            "packet": packet,
            "target_raw": target,
            "cache": cache_custody,
            "scorer_files": runtime_files,
            "expected_cache_label_mismatches": cache_mismatches,
        },
        seed,
    )


def _validate_wseg_receipt(path: Path) -> tuple[Any, dict[str, Any]]:
    receipt = _load_json(path, "WS2 custody receipt")
    if (
        receipt.get("schema") != WS2_SCHEMA
        or receipt.get("score_claim") is not False
        or receipt.get("research_only") is not True
        or receipt.get("promotion_eligible") is not False
    ):
        raise BaselineJoinError("WS2 receipt schema or authority flags differ")
    archive_custody = receipt.get("archive_custody")
    if not isinstance(archive_custody, Mapping) or not isinstance(archive_custody.get(ws1.W_SEG), Mapping):
        raise BaselineJoinError("WS2 receipt lacks W_seg archive custody")
    row = archive_custody[ws1.W_SEG]
    archive = _exact_file(
        Path(str(row.get("archive_path", ""))),
        {"bytes": row.get("archive_bytes"), "sha256": row.get("archive_sha256")},
        "W_seg archive",
    )
    if row.get("archive_path_sha256") != archive["sha256"]:
        raise BaselineJoinError("W_seg archive path digest differs")
    payload = Path(archive["path"]).read_bytes()
    try:
        receiver = ws1.receive_ws1_warm_start_archive(payload)
    except Exception as exc:
        raise BaselineJoinError("W_seg receiver parse-back failed") from exc
    if (
        receiver.parsed.candidate != ws1.W_SEG
        or receiver.parsed.candidate_id != ws1.W_SEG_CANDIDATE_ID
        or receiver.parsed.exact_reemit() != payload
        or dict(receiver.parsed.custody).get("archive_sha256") != archive["sha256"]
        or row.get("receiver_parse_reemit_byte_identical") is not True
        or row.get("ground_truth_argmax_present") is not False
        or row.get("scorer_weights_present") is not False
    ):
        raise BaselineJoinError("W_seg parsed receiver identity differs")
    return receiver, {
        "receipt": {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)},
        "archive": archive,
        "candidate": ws1.W_SEG,
        "candidate_id": ws1.W_SEG_CANDIDATE_ID,
        "carrier_archive_sha256": receiver.parsed.custody["carrier_archive_sha256"],
        "semantic_cell_api": "ddm_ws1_warm_start._semantic_cells(receiver.base, ids, base_camera, palette)",
        "final_rgb_scorer_cells_consulted": False,
    }


def _events_by_pair(seed: PartitionEventSeed) -> dict[int, tuple[PartitionEvent, ...]]:
    grouped: dict[int, list[PartitionEvent]] = defaultdict(list)
    for event in seed.events:
        grouped[event.pair].append(event)
    return {pair: tuple(events) for pair, events in grouped.items()}


def _zero_matrix() -> np.ndarray:
    return np.zeros((len(SEMANTIC_NAMES), len(SEMANTIC_NAMES)), dtype=np.int64)


def _matrix_rows(matrix: np.ndarray, *, left_name: str, right_name: str, count_name: str) -> list[dict[str, Any]]:
    value = np.asarray(matrix)
    if value.shape != (len(SEMANTIC_NAMES), len(SEMANTIC_NAMES)) or value.dtype.kind not in ("i", "u"):
        raise BaselineJoinError("interface matrix must be one integral five-by-five table")
    return [
        {
            f"{left_name}_class": left,
            f"{left_name}_name": SEMANTIC_NAMES[left],
            f"{right_name}_class": right,
            f"{right_name}_name": SEMANTIC_NAMES[right],
            count_name: int(value[left, right]),
        }
        for left in range(len(SEMANTIC_NAMES))
        for right in range(len(SEMANTIC_NAMES))
    ]


def _target_argmax(segnet: Any, torch: Any, frames: np.ndarray) -> np.ndarray:
    try:
        import einops
    except ImportError as exc:
        raise BaselineJoinError("einops is required by the frozen SegNet path") from exc
    array = np.asarray(frames)
    if array.ndim != 5 or array.shape[1:] != (2, *CAMERA_HW, 3) or array.dtype != np.uint8:
        raise BaselineJoinError("live-target frame batch geometry/dtype differs")
    pair = torch.from_numpy(np.ascontiguousarray(array)).float()
    inputs = einops.rearrange(pair, "b t h w c -> b t c h w")
    with torch.inference_mode():
        labels = segnet(segnet.preprocess_input(inputs)).argmax(dim=1).cpu().numpy()
    result = np.ascontiguousarray(labels.astype(np.uint8, copy=False))
    if result.shape != (array.shape[0], *SCORER_HW) or result.min() < 0 or result.max() >= len(SEMANTIC_NAMES):
        raise BaselineJoinError("live-target SegNet labels escaped frozen geometry/alphabet")
    return result


def _reconstruct_baseline(
    target: np.ndarray,
    *,
    pair_start: int,
    events: Mapping[int, Sequence[PartitionEvent]],
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    labels = np.asarray(target)
    if labels.ndim != 3 or labels.shape[1:] != SCORER_HW or labels.dtype.kind not in ("i", "u"):
        raise BaselineJoinError("target labels differ from the frozen scorer grid")
    baseline = labels.astype(np.uint8, copy=True)
    seeded = np.zeros(labels.shape, dtype=bool)
    rows: list[list[int]] = []
    for local, pair_id in enumerate(range(pair_start, pair_start + labels.shape[0])):
        for event in events.get(pair_id, ()):
            if int(labels[local, event.row, event.col]) != event.target_class:
                raise BaselineJoinError("live target class at an S2 site differs from packet custody")
            if seeded[local, event.row, event.col]:
                raise BaselineJoinError("S2 packet repeats a site within the stage")
            seeded[local, event.row, event.col] = True
            baseline[local, event.row, event.col] = event.baseline_class
            rows.append([event.pair, event.row, event.col, event.target_class, event.baseline_class])
    return baseline, seeded, rows


def _measure_arrays(
    *,
    pair_start: int,
    target: np.ndarray,
    cached_target: np.ndarray,
    wseg: np.ndarray,
    owned: np.ndarray,
    events: Mapping[int, Sequence[PartitionEvent]],
) -> dict[str, Any]:
    arrays = [np.asarray(value) for value in (target, cached_target, wseg, owned)]
    shape = arrays[0].shape
    if shape[1:] != SCORER_HW or any(value.shape != shape for value in arrays):
        raise BaselineJoinError("stage arrays do not share scorer-grid geometry")
    if arrays[3].dtype != np.bool_:
        raise BaselineJoinError("W_seg ownership field must be boolean")
    if any(value.dtype.kind not in ("i", "u") for value in arrays[:3]):
        raise BaselineJoinError("target/cache/W_seg cells must be integral")
    if any(value.size and (int(value.min()) < 0 or int(value.max()) >= len(SEMANTIC_NAMES)) for value in arrays[:3]):
        raise BaselineJoinError("stage cell alphabet differs")
    target_u8 = arrays[0].astype(np.uint8, copy=False)
    cache_u8 = arrays[1].astype(np.uint8, copy=False)
    wseg_u8 = arrays[2].astype(np.uint8, copy=False)
    baseline, seeded, event_rows = _reconstruct_baseline(target_u8, pair_start=pair_start, events=events)
    reapplied = baseline.copy()
    for pair, row, col, target_class, baseline_class in event_rows:
        local = pair - pair_start
        if int(reapplied[local, row, col]) != baseline_class:
            raise BaselineJoinError("reconstructed C1 baseline disagrees with S2 baseline class")
        reapplied[local, row, col] = target_class
    if not np.array_equal(reapplied, target_u8):
        raise BaselineJoinError("applying S2 to reconstructed C1 baseline does not recover target")

    mismatch = wseg_u8 != baseline
    seeded_mismatch = mismatch & seeded
    non_event_mismatch = mismatch & ~seeded
    interface = _zero_matrix()
    confusion = _zero_matrix()
    for left in range(len(SEMANTIC_NAMES)):
        left_mask = baseline == left
        for right in range(len(SEMANTIC_NAMES)):
            interface[left, right] = np.count_nonzero(left_mask & (target_u8 == right))
            confusion[left, right] = np.count_nonzero(left_mask & (wseg_u8 == right))
    per_class = []
    for class_id, class_name in enumerate(SEMANTIC_NAMES):
        class_mask = baseline == class_id
        class_sites = int(np.count_nonzero(class_mask))
        class_errors = int(np.count_nonzero(mismatch & class_mask))
        per_class.append(
            {
                "baseline_class": class_id,
                "baseline_name": class_name,
                "sites": class_sites,
                "wseg_mismatches": class_errors,
                "mismatch_fraction": class_errors / class_sites if class_sites else None,
            }
        )
    per_pair = []
    for local in range(shape[0]):
        per_pair.append(
            {
                "pair_id": pair_start + local,
                "s2_events": int(np.count_nonzero(seeded[local])),
                "seeded_expected_baseline_mismatches": int(np.count_nonzero(seeded_mismatch[local])),
                "non_event_residual": int(np.count_nonzero(non_event_mismatch[local])),
                "total_wseg_to_c1_baseline_mismatches": int(np.count_nonzero(mismatch[local])),
                "wseg_owned_sites": int(np.count_nonzero(arrays[3][local])),
                "target_cache_label_mismatches": int(np.count_nonzero(target_u8[local] != cache_u8[local])),
            }
        )
    return {
        "pair_start": pair_start,
        "pair_end_exclusive": pair_start + shape[0],
        "sites": int(np.prod(shape, dtype=np.int64)),
        "s2_events": len(event_rows),
        "seeded_expected_baseline_mismatches": int(np.count_nonzero(seeded_mismatch)),
        "non_event_residual": int(np.count_nonzero(non_event_mismatch)),
        "total_wseg_to_c1_baseline_mismatches": int(np.count_nonzero(mismatch)),
        "target_cache_label_mismatches": int(np.count_nonzero(target_u8 != cache_u8)),
        "wseg_owned_sites": int(np.count_nonzero(arrays[3])),
        "hashes": {
            "live_target_labels_sha256": _sha256_array(target_u8),
            "cached_target_labels_sha256": _sha256_array(cache_u8),
            "reconstructed_c1_baseline_sha256": _sha256_array(baseline),
            "wseg_semantic_cells_sha256": _sha256_array(wseg_u8),
            "wseg_owned_mask_sha256": _sha256_array(arrays[3]),
            "s2_event_rows_sha256": _sha256_bytes(_canonical(event_rows)),
        },
        "per_baseline_class": per_class,
        "c1_to_target_interface_25": _matrix_rows(
            interface, left_name="baseline", right_name="target", count_name="sites"
        ),
        "c1_to_wseg_confusion_25": _matrix_rows(confusion, left_name="baseline", right_name="wseg", count_name="sites"),
        "pairs": per_pair,
    }


def _merge_class_rows(stages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for class_id, class_name in enumerate(SEMANTIC_NAMES):
        rows = []
        for stage in stages:
            table = stage.get("per_baseline_class")
            if not isinstance(table, list) or len(table) != len(SEMANTIC_NAMES):
                raise BaselineJoinError("stage per-class table is incomplete")
            row = table[class_id]
            if (
                not isinstance(row, Mapping)
                or row.get("baseline_class") != class_id
                or row.get("baseline_name") != class_name
            ):
                raise BaselineJoinError("stage per-class registry differs")
            rows.append(row)
        sites = sum(int(row["sites"]) for row in rows)
        errors = sum(int(row["wseg_mismatches"]) for row in rows)
        result.append(
            {
                "baseline_class": class_id,
                "baseline_name": class_name,
                "sites": sites,
                "wseg_mismatches": errors,
                "mismatch_fraction": errors / sites if sites else None,
            }
        )
    return result


def _merge_matrix_rows(stages: Sequence[Mapping[str, Any]], key: str, *, right_name: str) -> list[dict[str, Any]]:
    matrix = _zero_matrix()
    for stage in stages:
        rows = stage.get(key)
        if not isinstance(rows, list) or len(rows) != 25:
            raise BaselineJoinError(f"stage {key} is not the complete 25-row table")
        for row in rows:
            left = row.get("baseline_class")
            right = row.get(f"{right_name}_class")
            sites = row.get("sites")
            if (
                any(isinstance(value, bool) or not isinstance(value, int) for value in (left, right, sites))
                or not 0 <= left < len(SEMANTIC_NAMES)
                or not 0 <= right < len(SEMANTIC_NAMES)
                or sites < 0
            ):
                raise BaselineJoinError(f"stage {key} row is malformed")
            matrix[left, right] += sites
    return _matrix_rows(matrix, left_name="baseline", right_name=right_name, count_name="sites")


def _aggregate(stages: Sequence[Mapping[str, Any]], *, config_sha256: str, expected_events: int) -> dict[str, Any]:
    expected_starts = list(range(0, PAIR_COUNT, STAGE_PAIRS))
    if len(stages) != len(expected_starts):
        raise BaselineJoinError("complete aggregation requires every batch-16 stage")
    for stage, start in zip(stages, expected_starts, strict=True):
        if stage.get("pair_start") != start or stage.get("pair_end_exclusive") != min(start + STAGE_PAIRS, PAIR_COUNT):
            raise BaselineJoinError("stage sequence is not the exact contiguous n600 partition")
        if stage.get("config_sha256") != config_sha256 or stage.get("schema") != STAGE_SCHEMA:
            raise BaselineJoinError("stage config/schema differs")
        _validate_hash(stage, "stage_sha256")
    pairs = [row for stage in stages for row in stage["pairs"]]
    if [row.get("pair_id") for row in pairs] != list(range(PAIR_COUNT)):
        raise BaselineJoinError("per-pair rows are not the canonical n600 sequence")
    totals = {
        key: sum(int(stage[key]) for stage in stages)
        for key in (
            "sites",
            "s2_events",
            "seeded_expected_baseline_mismatches",
            "non_event_residual",
            "total_wseg_to_c1_baseline_mismatches",
            "target_cache_label_mismatches",
            "wseg_owned_sites",
        )
    }
    if totals["sites"] != PAIR_COUNT * math.prod(SCORER_HW) or totals["s2_events"] != expected_events:
        raise BaselineJoinError("aggregated population differs from exact bridge geometry")
    if (
        totals["seeded_expected_baseline_mismatches"] + totals["non_event_residual"]
        != totals["total_wseg_to_c1_baseline_mismatches"]
    ):
        raise BaselineJoinError("seeded/non-event partition does not reconcile")
    chain_rows = [
        {
            "pair_range": [stage["pair_start"], stage["pair_end_exclusive"]],
            "stage_sha256": stage["stage_sha256"],
            "hashes": stage["hashes"],
        }
        for stage in stages
    ]
    per_class = _merge_class_rows(stages)
    target_interfaces = _merge_matrix_rows(stages, "c1_to_target_interface_25", right_name="target")
    wseg_confusion = _merge_matrix_rows(stages, "c1_to_wseg_confusion_25", right_name="wseg")
    if (
        sum(row["sites"] for row in per_class) != totals["sites"]
        or sum(row["wseg_mismatches"] for row in per_class) != totals["total_wseg_to_c1_baseline_mismatches"]
        or sum(row["sites"] for row in target_interfaces) != totals["sites"]
        or sum(row["sites"] for row in wseg_confusion) != totals["sites"]
        or sum(row["sites"] for row in target_interfaces if row["baseline_class"] != row["target_class"])
        != totals["s2_events"]
    ):
        raise BaselineJoinError("class/interface tables do not reconcile to the exact population")
    return totals | {
        "pair_count": PAIR_COUNT,
        "scorer_hw": list(SCORER_HW),
        "per_baseline_class": per_class,
        "c1_to_target_interface_25": target_interfaces,
        "c1_to_wseg_confusion_25": wseg_confusion,
        "pairs": pairs,
        "batch_digest_chain_sha256": _sha256_bytes(_canonical(chain_rows)),
        "batch_digest_chain": chain_rows,
    }


def _stage_path(stage_dir: Path, start: int, end: int) -> Path:
    return stage_dir / f"pairs-{start:04d}-{end - 1:04d}.json"


def _load_prefix(stage_dir: Path, *, config_sha256: str, resume: bool) -> list[dict[str, Any]]:
    if stage_dir.exists() and not stage_dir.is_dir():
        raise BaselineJoinError("stage path exists but is not a directory")
    entries = sorted(stage_dir.iterdir()) if stage_dir.exists() else []
    if any(not path.is_file() or not path.name.startswith("pairs-") or path.suffix != ".json" for path in entries):
        raise BaselineJoinError("stage directory contains an unexpected entry")
    files = entries
    if files and not resume:
        raise BaselineJoinError("stage directory is nonempty; pass --resume")
    stages = []
    expected_start = 0
    for path in files:
        stage = _load_json(path, "baseline-join stage")
        _validate_hash(stage, "stage_sha256")
        start = stage.get("pair_start")
        end = stage.get("pair_end_exclusive")
        if (
            stage.get("schema") != STAGE_SCHEMA
            or stage.get("config_sha256") != config_sha256
            or start != expected_start
            or end != min(expected_start + STAGE_PAIRS, PAIR_COUNT)
            or path != _stage_path(stage_dir, start, end)
        ):
            raise BaselineJoinError("resume stages are not one exact valid prefix")
        stages.append(stage)
        expected_start = end
    return stages


def _stage_custody(stage_dir: Path, stages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        start = stage.get("pair_start")
        end = stage.get("pair_end_exclusive")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
            raise BaselineJoinError("stage range is malformed")
        path = _stage_path(stage_dir, start, end)
        if not path.is_file():
            raise BaselineJoinError("preserved stage disappeared")
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "stage_sha256": stage["stage_sha256"],
                "pair_range": [start, end],
            }
        )
    return rows


def _input_file_rows(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bridge = config.get("bridge_custody")
    wseg = config.get("wseg_custody")
    implementation = config.get("implementation_custody")
    if not all(isinstance(value, Mapping) for value in (bridge, wseg, implementation)):
        raise BaselineJoinError("config input-custody sections are malformed")
    scorer_files = bridge.get("scorer_files")
    if not isinstance(scorer_files, Mapping):
        raise BaselineJoinError("config scorer-file custody is malformed")
    candidates: dict[str, Any] = {
        "bridge": bridge.get("bridge"),
        "debt": bridge.get("debt"),
        "s2_packet": bridge.get("packet"),
        "target_raw": bridge.get("target_raw"),
        "diagnostic_cache": bridge.get("cache"),
        "wseg_receipt": wseg.get("receipt"),
        "wseg_archive": wseg.get("archive"),
        "tool": implementation.get("tool"),
        "wseg_receiver_module": implementation.get("wseg_receiver_module"),
        "scorer_adapter_module": implementation.get("scorer_adapter_module"),
    }
    candidates.update({f"scorer_{key}": value for key, value in scorer_files.items()})
    observed: dict[str, dict[str, Any]] = {}
    for label, row in candidates.items():
        if not isinstance(row, Mapping):
            raise BaselineJoinError(f"config lacks bound file row: {label}")
        observed[label] = _exact_file(Path(str(row.get("path", ""))), row, label)
    return observed


def _input_end_barrier(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "tac.c0b_wseg_c1_baseline_join_end_barrier.v1",
        "files": _input_file_rows(config),
        "all_inputs_rehashed_after_last_stage": True,
    }


def _join_verdict(
    aggregate: Mapping[str, Any], *, expected_events: int
) -> tuple[str, dict[str, Any]]:
    total = int(aggregate["total_wseg_to_c1_baseline_mismatches"])
    seeded_mismatch = int(aggregate["seeded_expected_baseline_mismatches"])
    non_event = int(aggregate["non_event_residual"])
    if total == 0:
        verdict = "WSEG_SEMANTIC_PROGRAM_EXACTLY_INITIALIZES_C1_BASELINE"
    elif seeded_mismatch == 0:
        verdict = "WSEG_MATCHES_S2_BASELINE_AT_SEEDED_SITES_BUT_DIFFERS_OUTSIDE_THE_EVENT_SET"
    else:
        verdict = "WSEG_SEMANTIC_PROGRAM_DOES_NOT_INITIALIZE_THE_EXACT_C1_BASELINE"
    return verdict, {
        "exact_initializer": total == 0,
        "seeded_site_expected_baseline_matches": expected_events - seeded_mismatch,
        "seeded_site_expected_baseline_mismatches": seeded_mismatch,
        "non_event_residual": non_event,
        "total_wseg_to_c1_baseline_mismatches": total,
        "total_wseg_to_c1_baseline_mismatch_fraction": total / int(aggregate["sites"]),
        "s2_apply_back_recovers_target": True,
        "known_cache_discrepancies_reproduced": int(aggregate["target_cache_label_mismatches"]),
    }


def _validate_existing_final(
    result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    stage_dir: Path,
    expected_events: int,
    expected_cache_mismatches: int,
) -> None:
    if (
        result.get("schema") != SCHEMA
        or result.get("config_sha256") != config_sha256
        or result.get("config") != config
        or result.get("research_only") is not True
        or result.get("score_claim") is not False
        or result.get("promotion_eligible") is not False
        or result.get("pointer_moved") is not False
        or result.get("candidate_payload_admission") is not False
    ):
        raise BaselineJoinError("existing final receipt belongs to different scientific inputs")
    stages = _load_prefix(stage_dir, config_sha256=config_sha256, resume=True)
    aggregate = _aggregate(stages, config_sha256=config_sha256, expected_events=expected_events)
    if aggregate["target_cache_label_mismatches"] != expected_cache_mismatches:
        raise BaselineJoinError("preserved stages lost the bridge-bound cache discrepancy count")
    verdict, join = _join_verdict(aggregate, expected_events=expected_events)
    if (
        result.get("aggregate") != aggregate
        or result.get("verdict") != verdict
        or result.get("join") != join
        or result.get("stage_custody") != _stage_custody(stage_dir, stages)
    ):
        raise BaselineJoinError("existing final receipt differs from preserved exact stages")
    barrier = result.get("input_end_barrier")
    if not isinstance(barrier, Mapping) or barrier != _input_end_barrier(config):
        raise BaselineJoinError("existing final receipt lost or drifted its end-of-run input barrier")


def measure(args: argparse.Namespace) -> dict[str, Any]:
    bridge_path = args.bridge.expanduser().resolve()
    ws2_path = args.ws2_receipt.expanduser().resolve()
    output = args.output.expanduser().resolve()
    stage_dir = (args.stage_dir or output.with_name(f"{output.name}.stages")).expanduser().resolve()
    bridge, bridge_custody, seed = _validate_bridge(bridge_path)
    receiver, wseg_custody = _validate_wseg_receipt(ws2_path)
    if tuple(ws1.CLASS_ORDER) != tuple(SEMANTIC_NAMES) or tuple(scorer.CLASS_ORDER) != tuple(SEMANTIC_NAMES):
        raise BaselineJoinError("W_seg/S2/scorer semantic class order differs")
    config = {
        "schema": "tac.c0b_wseg_c1_baseline_join_config.v1",
        "pair_count": PAIR_COUNT,
        "stage_pairs": STAGE_PAIRS,
        "camera_hw": list(CAMERA_HW),
        "scorer_hw": list(SCORER_HW),
        "cpu_threads": int(args.cpu_threads),
        "bridge_custody": bridge_custody,
        "wseg_custody": wseg_custody,
        "implementation_custody": {
            "tool": {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).resolve().stat().st_size,
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "wseg_receiver_module": {
                "path": str(Path(ws1.__file__).resolve()),
                "bytes": Path(ws1.__file__).resolve().stat().st_size,
                "sha256": _sha256_file(Path(ws1.__file__).resolve()),
            },
            "scorer_adapter_module": {
                "path": str(Path(scorer.__file__).resolve()),
                "bytes": Path(scorer.__file__).resolve().stat().st_size,
                "sha256": _sha256_file(Path(scorer.__file__).resolve()),
            },
        },
        "algorithm": {
            "target": "frozen CPU SegNet argmax on exact M2 raw in evaluator-sized batch16",
            "baseline": "replace every exact S2 target site by packet baseline_class",
            "join": "compare W_seg receiver semantic cells with reconstructed C1 baseline over every scorer cell",
            "cached_labels": "diagnostic only; exact known discrepancy count must match bridge",
        },
        "target_tables_diagnostic_only": True,
        "target_tables_forbidden_in_candidate_payload": True,
    }
    if isinstance(args.cpu_threads, bool) or args.cpu_threads <= 0:
        raise BaselineJoinError("cpu_threads must be a positive integer")
    expected_upstream = Path(bridge_custody["scorer_files"]["modules_py"]["path"]).parent.resolve()
    if args.upstream.expanduser().resolve() != expected_upstream:
        raise BaselineJoinError("upstream path differs from the bridge-bound frozen scorer source")
    config_sha = _sha256_bytes(_canonical(config))
    if output.exists():
        result = _load_json(output, "final baseline-join receipt")
        _validate_hash(result, "receipt_sha256")
        _validate_existing_final(
            result,
            config=config,
            config_sha256=config_sha,
            stage_dir=stage_dir,
            expected_events=len(seed.events),
            expected_cache_mismatches=bridge_custody["expected_cache_label_mismatches"],
        )
        return result
    stages = _load_prefix(stage_dir, config_sha256=config_sha, resume=bool(args.resume))
    fields, cache_sha = scorer._load_cache(Path(bridge_custody["cache"]["path"]), require_canonical_hash=True)
    if cache_sha != bridge_custody["cache"]["sha256"]:
        raise BaselineJoinError("loaded cache digest differs from bridge custody")
    grouped = _events_by_pair(seed)
    target_raw = np.memmap(
        Path(bridge_custody["target_raw"]["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, 2, *CAMERA_HW, 3),
    )
    scorer_bundle: tuple[Any, Any, Any] | None = None
    palette = ws1._palette(receiver.base)
    start = stages[-1]["pair_end_exclusive"] if stages else 0
    while start < PAIR_COUNT:
        if scorer_bundle is None:
            scorer_bundle = scorer._load_scorers(args.upstream.expanduser().resolve(), int(args.cpu_threads))
        end = min(start + STAGE_PAIRS, PAIR_COUNT)
        ids = tuple(range(start, end))
        frames = np.asarray(target_raw[start:end]).copy()
        target = _target_argmax(scorer_bundle[0], scorer_bundle[2], frames)
        base_camera = receiver.base.render_camera_pairs(ids)
        semantic, owned = ws1._semantic_cells(receiver.base, ids, base_camera, palette)
        measured = _measure_arrays(
            pair_start=start,
            target=target,
            cached_target=np.asarray(fields["lstars"][start:end]),
            wseg=semantic,
            owned=owned,
            events=grouped,
        )
        stage = _with_hash(
            {
                "schema": STAGE_SCHEMA,
                "config_sha256": config_sha,
                **measured,
                "stage_complete": True,
            },
            "stage_sha256",
        )
        path = _stage_path(stage_dir, start, end)
        _write_once(path, stage)
        stages.append(stage)
        del frames, target, base_camera, semantic, owned
        start = end
    aggregate = _aggregate(stages, config_sha256=config_sha, expected_events=len(seed.events))
    expected_cache_mismatches = bridge_custody["expected_cache_label_mismatches"]
    if aggregate["target_cache_label_mismatches"] != expected_cache_mismatches:
        raise BaselineJoinError("live-target/cache discrepancy count differs from geometry-bound bridge")
    verdict, join = _join_verdict(aggregate, expected_events=len(seed.events))
    join["known_cache_discrepancies_reproduced"] = expected_cache_mismatches
    barrier = _input_end_barrier(config)
    result = _with_hash(
        {
            "schema": SCHEMA,
            "verdict": verdict,
            "verdict_scope": (
                "exact all-600 scorer-grid semantic-cell join between the original W_seg receiver program and "
                "the C1 baseline reconstructed from the exact M2 live-target labels plus geometry-bound S2 events; "
                "not a camera-space realization, archive, official evaluation, score, promotion, or family kill"
            ),
            "config": config,
            "config_sha256": config_sha,
            "aggregate": aggregate,
            "join": join,
            "stage_custody": _stage_custody(stage_dir, stages),
            "input_end_barrier": barrier,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "candidate_payload_admission": False,
            "target_tables_diagnostic_only": True,
            "target_tables_forbidden_in_candidate_payload": True,
            "pointer_context_consulted": False,
        },
        "receipt_sha256",
    )
    _write_once(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--ws2-receipt", type=Path, default=DEFAULT_WS2_RECEIPT)
    parser.add_argument("--upstream", type=Path, default=scorer.DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = measure(args)
    except BaselineJoinError as exc:
        print(f"[c0b-bj1] REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
