#!/usr/bin/env python3
"""Measure the exact n600 V9 target-partition grammar without invoking a scorer.

This is an encoder-side, research-only census of ``gt_n600.npz::lstars``.  The
target label table is supervision/diagnostic evidence only: it is forbidden as
an archive member, generated decoder source, or any other candidate payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402

SCHEMA = "tac.v9_target_partition_grammar_census.v1"
RECEIPT_SCHEMA = "tac.v9_target_partition_grammar_census_receipt.v1"
N_PAIRS = 600
HEIGHT = 384
WIDTH = 512
EXPECTED_SHAPE = (N_PAIRS, HEIGHT, WIDTH)
LABELS = (0, 1, 2, 3, 4)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
ADJACENCY_KEYS = tuple(f"{left}-{right}" for left in LABELS for right in LABELS if left < right)
SIZE_BUCKETS = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-15", 4, 15),
    ("16-63", 16, 63),
    ("64-255", 64, 255),
    ("256-1023", 256, 1023),
    ("1024+", 1024, None),
)
CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)
DEFAULT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUTPUT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "v9_target_partition_grammar_census_n600.json"
)


class CensusError(RuntimeError):
    """Fail-closed input, custody, or receipt error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file_custody(path: Path) -> dict[str, Any]:
    """Hash one stable file generation and fail closed on concurrent mutation."""
    path = path.resolve()
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise CensusError(f"cannot snapshot target cache: {path}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CensusError("target cache mutated while its custody hash was being read")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest}


def _summary(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise CensusError("cannot summarize an empty sequence")
    total = int(sum(values))
    return {
        "count": len(values),
        "sum": total,
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": total / len(values),
    }


def _optional_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "sum": 0, "min": None, "max": None, "mean": None}
    return _summary(values)


def validate_labels(
    labels: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE,
    expected_labels: Sequence[int] = LABELS,
) -> tuple[int, ...]:
    if tuple(labels.shape) != tuple(expected_shape):
        raise CensusError(f"lstars shape changed: expected {expected_shape}, got {tuple(labels.shape)}")
    if labels.dtype.kind not in "biu":
        raise CensusError(f"lstars must have an integer dtype, got {labels.dtype}")
    observed: set[int] = set()
    allowed = {int(value) for value in expected_labels}
    for frame_index in range(expected_shape[0]):
        frame_values = np.unique(np.asarray(labels[frame_index]))
        observed.update(int(value) for value in frame_values.tolist())
        unexpected = observed - allowed
        if unexpected:
            raise CensusError(f"lstars contains labels outside {sorted(allowed)}: {sorted(unexpected)}")
    if observed != allowed:
        raise CensusError(f"lstars label alphabet changed: expected {sorted(allowed)}, got {sorted(observed)}")
    return tuple(sorted(observed))


def _component_bucket_counts(sizes: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, low, high in SIZE_BUCKETS:
        selected = sizes >= low
        if high is not None:
            selected &= sizes <= high
        result[name] = int(np.count_nonzero(selected))
    return result


def _class_metrics(frame: np.ndarray, label: int, run_starts: np.ndarray) -> dict[str, Any]:
    mask = frame == label
    components, component_count = ndimage.label(mask, structure=CONNECTIVITY_8)
    if component_count:
        sizes = np.bincount(components.ravel(), minlength=component_count + 1)[1:]
    else:
        sizes = np.empty(0, dtype=np.int64)
    return {
        "label": label,
        "name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else f"class_{label}",
        "pixels": int(np.count_nonzero(mask)),
        "row_runs": int(np.count_nonzero(run_starts & mask)),
        "components_8_connected": int(component_count),
        "components_ge_16": int(np.count_nonzero(sizes >= 16)),
        "component_size_buckets": _component_bucket_counts(sizes),
    }


def _adjacency_counts(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    changed = left != right
    if not np.any(changed):
        return dict.fromkeys(ADJACENCY_KEYS, 0)
    low = np.minimum(left[changed], right[changed]).astype(np.int64, copy=False)
    high = np.maximum(left[changed], right[changed]).astype(np.int64, copy=False)
    codes = low * len(LABELS) + high
    counts = np.bincount(codes, minlength=len(LABELS) ** 2)
    return {
        key: int(counts[a * len(LABELS) + b])
        for key, (a, b) in zip(
            ADJACENCY_KEYS,
            ((a, b) for a in LABELS for b in LABELS if a < b),
            strict=True,
        )
    }


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _temporal_record(previous: np.ndarray, current: np.ndarray, current_index: int) -> dict[str, Any]:
    codes = previous.astype(np.int64, copy=False) * len(LABELS) + current.astype(np.int64, copy=False)
    matrix = np.bincount(codes.ravel(), minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
    changed = int(np.count_nonzero(previous != current))
    return {
        "previous_pair_index": current_index - 1,
        "current_pair_index": current_index,
        "previous_source_frame_index": 2 * (current_index - 1) + 1,
        "current_source_frame_index": 2 * current_index + 1,
        "source_frame_stride": 2,
        "changed_sites": changed,
        "transition_matrix_previous_to_current": matrix.astype(np.int64).tolist(),
    }


def measure_partition(
    labels: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE,
    expected_labels: Sequence[int] = LABELS,
) -> dict[str, Any]:
    alphabet = validate_labels(labels, expected_shape=expected_shape, expected_labels=expected_labels)
    if alphabet != LABELS:
        raise CensusError("the V9 census requires the canonical five-class alphabet 0..4")

    frames: list[dict[str, Any]] = []
    temporal: list[dict[str, Any]] = []
    label_stream_hash = hashlib.sha256()
    aggregate_horizontal = dict.fromkeys(ADJACENCY_KEYS, 0)
    aggregate_vertical = dict.fromkeys(ADJACENCY_KEYS, 0)
    aggregate_class = {
        label: {
            "pixels": 0,
            "row_runs": 0,
            "components_8_connected": 0,
            "components_ge_16": 0,
            "component_size_buckets": {name: 0 for name, _low, _high in SIZE_BUCKETS},
        }
        for label in LABELS
    }
    previous: np.ndarray | None = None

    for frame_index in range(expected_shape[0]):
        frame = np.ascontiguousarray(labels[frame_index], dtype=np.uint8)
        label_stream_hash.update(frame.tobytes(order="C"))
        run_starts = np.ones(frame.shape, dtype=bool)
        run_starts[:, 1:] = frame[:, 1:] != frame[:, :-1]
        class_rows = [_class_metrics(frame, label, run_starts) for label in LABELS]
        for row in class_rows:
            target = aggregate_class[int(row["label"])]
            for key in ("pixels", "row_runs", "components_8_connected", "components_ge_16"):
                target[key] += int(row[key])
            _merge_counts(target["component_size_buckets"], row["component_size_buckets"])

        horizontal = _adjacency_counts(frame[:, :-1], frame[:, 1:])
        vertical = _adjacency_counts(frame[:-1, :], frame[1:, :])
        frame_sites = frame.shape[0] * frame.shape[1]
        if sum(int(row["pixels"]) for row in class_rows) != frame_sites:
            raise CensusError(f"class pixel census does not close at pair {frame_index}")
        if sum(int(row["row_runs"]) for row in class_rows) != int(np.count_nonzero(run_starts)):
            raise CensusError(f"class row-run census does not close at pair {frame_index}")
        if sum(horizontal.values()) != int(np.count_nonzero(frame[:, :-1] != frame[:, 1:])):
            raise CensusError(f"horizontal adjacency census does not close at pair {frame_index}")
        if sum(vertical.values()) != int(np.count_nonzero(frame[:-1, :] != frame[1:, :])):
            raise CensusError(f"vertical adjacency census does not close at pair {frame_index}")
        _merge_counts(aggregate_horizontal, horizontal)
        _merge_counts(aggregate_vertical, vertical)
        frames.append(
            {
                "pair_index": frame_index,
                "source_pair_frame_indices": [2 * frame_index, 2 * frame_index + 1],
                "segmentation_source_frame_index": 2 * frame_index + 1,
                "row_runs": int(np.count_nonzero(run_starts)),
                "horizontal_boundaries": int(sum(horizontal.values())),
                "vertical_boundaries": int(sum(vertical.values())),
                "horizontal_class_adjacency": horizontal,
                "vertical_class_adjacency": vertical,
                "classes": class_rows,
            }
        )
        if previous is not None:
            transition = _temporal_record(previous, frame, frame_index)
            matrix = np.asarray(transition["transition_matrix_previous_to_current"], dtype=np.int64)
            if int(matrix.sum()) != frame_sites:
                raise CensusError(f"temporal transition census does not close at pair {frame_index}")
            if int(matrix.sum() - np.trace(matrix)) != int(transition["changed_sites"]):
                raise CensusError(f"temporal changed-site census does not close at pair {frame_index}")
            temporal.append(transition)
        previous = frame

    temporal_matrix = (
        np.sum(
            np.asarray([row["transition_matrix_previous_to_current"] for row in temporal], dtype=np.int64),
            axis=0,
        )
        if temporal
        else np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    )
    row_runs = [int(row["row_runs"]) for row in frames]
    horizontal_boundaries = [int(row["horizontal_boundaries"]) for row in frames]
    vertical_boundaries = [int(row["vertical_boundaries"]) for row in frames]
    changed_sites = [int(row["changed_sites"]) for row in temporal]
    frame_sites = expected_shape[1] * expected_shape[2]
    total_sites = len(frames) * frame_sites
    if sum(int(row["pixels"]) for row in aggregate_class.values()) != total_sites:
        raise CensusError("aggregate class pixel census does not close")
    return {
        "shape": list(expected_shape),
        "dtype_in_cache": str(labels.dtype),
        "canonical_label_dtype": "uint8",
        "canonical_label_stream_order": "C:pair,row,column",
        "canonical_uint8_label_stream_sha256": label_stream_hash.hexdigest(),
        "label_alphabet": list(alphabet),
        "class_names": list(CLASS_NAMES),
        "connectivity": "8-connected within each 2D pair-end label field",
        "component_size_buckets": [name for name, _low, _high in SIZE_BUCKETS],
        "temporal_interpretation": {
            "meaning": (
                "lstars[i] is the SegNet label field of the last frame of source pair i; successive "
                "census fields therefore compare source frames 1,3,5,... with stride two, not adjacent frames"
            ),
            "first_source_frame_index": 1,
            "last_source_frame_index": 2 * expected_shape[0] - 1,
            "source_frame_stride": 2,
        },
        "frames": frames,
        "successive_pair_end_temporal": temporal,
        "aggregate": {
            "evidence_rows": len(frames),
            "temporal_transition_rows": len(temporal),
            "total_sites": total_sites,
            "row_runs": _summary(row_runs),
            "horizontal_boundaries": _summary(horizontal_boundaries),
            "vertical_boundaries": _summary(vertical_boundaries),
            "horizontal_class_adjacency": aggregate_horizontal,
            "vertical_class_adjacency": aggregate_vertical,
            "class_adjacency_both_axes": {
                key: aggregate_horizontal[key] + aggregate_vertical[key] for key in ADJACENCY_KEYS
            },
            "classes": [{"label": label, "name": CLASS_NAMES[label], **aggregate_class[label]} for label in LABELS],
            "temporal_changed_sites": _optional_summary(changed_sites),
            "temporal_changed_fraction": (
                int(sum(changed_sites)) / (len(temporal) * frame_sites) if temporal else None
            ),
            "temporal_transition_matrix_previous_to_current": temporal_matrix.astype(np.int64).tolist(),
        },
    }


def build_receipt(
    *,
    cache_path: Path,
    measurements: dict[str, Any],
    cache_custody: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if measurements.get("shape") != list(EXPECTED_SHAPE):
        raise CensusError("receipt construction requires the exact n600 geometry")
    if len(measurements.get("frames", [])) != N_PAIRS:
        raise CensusError("receipt construction requires all 600 frame rows")
    if len(measurements.get("successive_pair_end_temporal", [])) != N_PAIRS - 1:
        raise CensusError("receipt construction requires all 599 temporal rows")
    aggregate = measurements.get("aggregate", {})
    if aggregate.get("evidence_rows") != N_PAIRS or aggregate.get("total_sites") != N_PAIRS * HEIGHT * WIDTH:
        raise CensusError("receipt aggregate does not prove full n600 evidence")
    cache_path = cache_path.resolve()
    custody = dict(cache_custody) if cache_custody is not None else snapshot_file_custody(cache_path)
    if custody.get("path") != str(cache_path):
        raise CensusError("target-cache custody path differs from the measured path")
    if (
        isinstance(custody.get("bytes"), bool)
        or not isinstance(custody.get("bytes"), int)
        or custody["bytes"] <= 0
        or not isinstance(custody.get("sha256"), str)
        or len(custody["sha256"]) != 64
    ):
        raise CensusError("target-cache custody is malformed")
    body = {
        "schema": SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "candidate_lineage_prohibition": (
            "gt_n600.npz::lstars is encoder-side diagnostic/supervision only; the target label table, "
            "its exact stream, and any lossless restatement are forbidden from archive payloads and decoder code"
        ),
        "authority_scope": (
            "full-n600 frozen-target partition census; no RGB realization, archive, evaluator, d_seg, d_pose, "
            "rate, or exact contest score is produced"
        ),
        "input_custody": {
            **custody,
            "member": "lstars.npy",
            "member_access": "open_stored_npy_memmap read-only ZIP_STORED",
            "start_end_barrier": True,
        },
        "implementation_custody": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "measurements": measurements,
    }
    return make_receipt_envelope(body)


def make_receipt_envelope(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != {"schema", "body", "body_sha256"}:
        raise CensusError("receipt envelope keys changed")
    if receipt["schema"] != RECEIPT_SCHEMA or not isinstance(receipt["body"], dict):
        raise CensusError("receipt schema or body changed")
    body = receipt["body"]
    if body.get("schema") != SCHEMA:
        raise CensusError("receipt body schema changed")
    for key, expected_value in {
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }.items():
        if body.get(key) is not expected_value:
            raise CensusError(f"receipt authority marker changed: {key}")
    if "forbidden from archive payloads" not in str(body.get("candidate_lineage_prohibition", "")):
        raise CensusError("receipt candidate-lineage prohibition is missing")
    expected = hashlib.sha256(canonical_json_bytes(receipt["body"])).hexdigest()
    if receipt["body_sha256"] != expected:
        raise CensusError("receipt body SHA-256 mismatch")


def write_once_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    validate_receipt(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(receipt) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CensusError(f"write-once receipt already exists: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run(cache_path: Path, output_path: Path) -> dict[str, Any]:
    cache_path = cache_path.resolve()
    if not cache_path.is_file():
        raise CensusError(f"target cache is missing: {cache_path}")
    start_custody = snapshot_file_custody(cache_path)
    labels = open_stored_npy_memmap(cache_path, "lstars")
    if not isinstance(labels, np.memmap) or labels.flags.writeable:
        raise CensusError("lstars must be a read-only stored-NPY memmap")
    measurements = measure_partition(labels)
    end_custody = snapshot_file_custody(cache_path)
    if start_custody != end_custody:
        raise CensusError("target cache drifted during the partition census")
    receipt = build_receipt(
        cache_path=cache_path,
        measurements=measurements,
        cache_custody=end_custody,
    )
    write_once_receipt(output_path.resolve(), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    receipt = run(args.gt_cache, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "body_sha256": receipt["body_sha256"],
                "evidence_rows": receipt["body"]["measurements"]["aggregate"]["evidence_rows"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
