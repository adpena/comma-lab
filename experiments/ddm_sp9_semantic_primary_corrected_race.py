#!/usr/bin/env python3
"""Exact ceiling gate for the SP9 direct multi-token semantic schema.

This instrument deliberately stops before coding.  It reads the already-retained
AFR1 semantic field and computes the exact empirical ideal length of a direct
temporal-delta run-event grammar.  No candidate payload, archive, parser,
receiver, RGB output, or scorer output is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INPUT = Path(
    "/Volumes/APDataStore/pact/ddm_jbp1_joint_batch_price/retained/fields/"
    "sfp1_null_empty.u8"
)
DEFAULT_OUTPUT = Path(
    "/Volumes/APDataStore/pact/ddm_sp9_semantic_primary_corrected_race/"
    "CEILING_RESULT_V2.json"
)
APDATASTORE_ROOT = Path("/Volumes/APDataStore")

FIELD_SHAPE = (600, 384, 512)
FIELD_BYTES = 117_964_800
FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
NUM_CLASSES = 5
SAME_AS_PREVIOUS = NUM_CLASSES
NUM_BUCKETS = 18

BASE_JOINT_POOL_BYTES = 126_926
BASE_ARCHIVE_BYTES = 180_002
FIXED_NON_POOL_BYTES = BASE_ARCHIVE_BYTES - BASE_JOINT_POOL_BYTES
FIRE_ARCHIVE_CEILING_BYTES = 137_986
RATE_CORNER_DEMAND_BYTES = 42_016
MIN_AP_FREE_BYTES = 3 * (1 << 29)  # 1.5 GiB
AXIS = "[macOS-CPU advisory / scorer-free exact conditional-entropy measurement]"


class Sp9CeilingError(RuntimeError):
    """The source, storage, or exact-accounting contract was not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return int(stat.f_bavail * stat.f_frsize)


def _entropy_bits(counts: Mapping[Any, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    log_total = math.log2(total)
    return math.fsum(
        count * (log_total - math.log2(count)) for count in counts.values()
    )


def _conditional_entropy_bits(
    transitions: Mapping[tuple[Any, Any], int],
    context_counts: Mapping[Any, int],
) -> float:
    return math.fsum(
        count * (math.log2(context_counts[previous]) - math.log2(count))
        for (previous, _current), count in transitions.items()
    )


def _sorted_counts(counts: Mapping[Any, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(counts):
        if isinstance(key, tuple):
            encoded_key: object = list(key)
        else:
            encoded_key = key
        rows.append({"key": encoded_key, "count": int(counts[key])})
    return rows


def _run_events(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if symbols.ndim != 1 or not symbols.size:
        raise Sp9CeilingError("each pair must contain a non-empty flat symbol stream")
    boundaries = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array([symbols.size], dtype=np.int64)))
    lengths = ends - starts
    if np.any(lengths <= 0):
        raise Sp9CeilingError("run partition contains a non-positive length")
    return symbols[starts].astype(np.uint8, copy=False), lengths


def analyze_field(field: np.ndarray, expected_shape: tuple[int, int, int]) -> dict[str, object]:
    if field.shape != expected_shape:
        raise Sp9CeilingError(f"field shape {field.shape} != expected {expected_shape}")

    joint_counts: Counter[tuple[int, int]] = Counter()
    joint_transitions: Counter[tuple[int, int, int, int]] = Counter()
    joint_context_counts: Counter[tuple[int, int]] = Counter()
    exact_event_counts: Counter[tuple[int, int]] = Counter()
    exact_transitions: Counter[tuple[int, int, int, int]] = Counter()
    exact_context_counts: Counter[tuple[int, int]] = Counter()
    observed_classes: set[int] = set()
    per_pair_runs: list[int] = []
    residual_bits = 0

    previous_frame: np.ndarray | None = None
    for pair_index in range(expected_shape[0]):
        current = np.asarray(field[pair_index], dtype=np.uint8)
        observed_classes.update(int(value) for value in np.unique(current))
        if current.size and (int(current.min()) < 0 or int(current.max()) >= NUM_CLASSES):
            raise Sp9CeilingError(
                f"pair {pair_index} contains a class outside [0,{NUM_CLASSES})"
            )

        flat = current.reshape(-1)
        if previous_frame is None:
            direct = flat.copy()
        else:
            previous_flat = previous_frame.reshape(-1)
            direct = flat.copy()
            direct[flat == previous_flat] = SAME_AS_PREVIOUS

        symbols, lengths = _run_events(direct)
        buckets = np.floor(np.log2(lengths)).astype(np.int64)
        if np.any(buckets < 0) or np.any(buckets >= NUM_BUCKETS):
            raise Sp9CeilingError("run length exceeds the direct grammar's bucket domain")

        events = [
            (int(symbol), int(length))
            for symbol, length in zip(symbols, lengths, strict=True)
        ]
        joints = [
            (int(symbol), int(bucket))
            for symbol, bucket in zip(symbols, buckets, strict=True)
        ]
        exact_event_counts.update(events)
        joint_counts.update(joints)
        residual_bits += int(buckets.sum(dtype=np.int64))
        per_pair_runs.append(len(events))

        for previous, current_event in pairwise(joints):
            joint_context_counts[previous] += 1
            joint_transitions[(*previous, *current_event)] += 1
        for previous, current_event in pairwise(events):
            exact_context_counts[previous] += 1
            exact_transitions[(*previous, *current_event)] += 1

        previous_frame = current

    expected_sites = int(np.prod(expected_shape))
    if observed_classes != set(range(NUM_CLASSES)):
        raise Sp9CeilingError(
            f"full field alphabet {sorted(observed_classes)} != {list(range(NUM_CLASSES))}"
        )
    total_runs = sum(per_pair_runs)
    if sum(joint_counts.values()) != total_runs or sum(exact_event_counts.values()) != total_runs:
        raise Sp9CeilingError("run-event count accounting does not close")
    if sum(joint_context_counts.values()) != total_runs - expected_shape[0]:
        raise Sp9CeilingError("pair-reset joint transition accounting does not close")
    if sum(exact_context_counts.values()) != total_runs - expected_shape[0]:
        raise Sp9CeilingError("pair-reset exact transition accounting does not close")

    joint_entropy_bits = _entropy_bits(joint_counts)
    primary_ideal_bits = joint_entropy_bits + residual_bits
    primary_ideal_bytes = primary_ideal_bits / 8.0
    primary_ceil_bytes = math.ceil(primary_ideal_bytes)

    joint_h1_bits = _conditional_entropy_bits(
        {
            ((p_symbol, p_bucket), (c_symbol, c_bucket)): count
            for (p_symbol, p_bucket, c_symbol, c_bucket), count in joint_transitions.items()
        },
        joint_context_counts,
    )
    exact_h1_bits = _conditional_entropy_bits(
        {
            ((p_symbol, p_length), (c_symbol, c_length)): count
            for (p_symbol, p_length, c_symbol, c_length), count in exact_transitions.items()
        },
        exact_context_counts,
    )
    joint_h1_plus_residual_bytes = (joint_h1_bits + residual_bits) / 8.0
    exact_h1_bytes = exact_h1_bits / 8.0

    # Give the formulation every possible advantage at the refusal gate: charge
    # zero bytes for its fitted static model and zero coder redundancy.  A real
    # container can only be larger than this exact empirical-entropy bound.
    projected_archive_bytes = FIXED_NON_POOL_BYTES + primary_ceil_bytes

    return {
        "field_sites": expected_sites,
        "observed_classes": sorted(observed_classes),
        "total_runs": total_runs,
        "pair_count": expected_shape[0],
        "per_pair_runs": per_pair_runs,
        "run_min": min(per_pair_runs),
        "run_max": max(per_pair_runs),
        "run_mean": total_runs / expected_shape[0],
        "observed_joint_symbols": len(joint_counts),
        "observed_exact_events": len(exact_event_counts),
        "observed_joint_transitions": len(joint_transitions),
        "observed_exact_transitions": len(exact_transitions),
        "joint_event_counts": _sorted_counts(joint_counts),
        "joint_transition_counts": _sorted_counts(joint_transitions),
        "joint_context_counts": _sorted_counts(joint_context_counts),
        "exact_event_counts": _sorted_counts(exact_event_counts),
        "exact_transition_counts": _sorted_counts(exact_transitions),
        "exact_context_counts": _sorted_counts(exact_context_counts),
        "primary": {
            "joint_mle_entropy_bits": joint_entropy_bits,
            "raw_residual_bits": residual_bits,
            "ideal_bits": primary_ideal_bits,
            "ideal_bytes": primary_ideal_bytes,
            "ideal_stream_ceil_bytes": primary_ceil_bytes,
            "model_bytes_assumed": 0,
            "coder_redundancy_bytes_assumed": 0,
            "optimistic_pool_bytes": primary_ceil_bytes,
            "optimistic_projected_archive_bytes": projected_archive_bytes,
        },
        "relaxations": {
            "joint_event_first_order_previous_event_free_at_pair_start": {
                "conditional_bits": joint_h1_bits,
                "raw_residual_bits": residual_bits,
                "ideal_bytes": joint_h1_plus_residual_bytes,
                "ceil_bytes": math.ceil(joint_h1_plus_residual_bytes),
                "model_and_framing_bytes_assumed": 0,
            },
            "exact_event_first_order_previous_event_free_at_pair_start": {
                "conditional_bits": exact_h1_bits,
                "ideal_bytes": exact_h1_bytes,
                "ceil_bytes": math.ceil(exact_h1_bytes),
                "model_and_framing_bytes_assumed": 0,
            },
        },
        "comparison": {
            "base_joint_pool_bytes": BASE_JOINT_POOL_BYTES,
            "base_archive_bytes": BASE_ARCHIVE_BYTES,
            "fixed_non_pool_bytes": FIXED_NON_POOL_BYTES,
            "fire_archive_ceiling_bytes": FIRE_ARCHIVE_CEILING_BYTES,
            "rate_corner_demand_bytes": RATE_CORNER_DEMAND_BYTES,
            "primary_ideal_stream_delta_vs_joint_pool_bytes": (
                primary_ceil_bytes - BASE_JOINT_POOL_BYTES
            ),
            "primary_optimistic_delta_vs_joint_pool_bytes": (
                primary_ceil_bytes - BASE_JOINT_POOL_BYTES
            ),
            "projected_archive_shortfall_vs_fire_ceiling_bytes": (
                projected_archive_bytes - FIRE_ARCHIVE_CEILING_BYTES
            ),
            "rate_cut_fraction_of_demand": (
                (BASE_ARCHIVE_BYTES - projected_archive_bytes) / RATE_CORNER_DEMAND_BYTES
            ),
            "most_favorable_relaxation_delta_vs_joint_pool_bytes": (
                math.ceil(exact_h1_bytes) - BASE_JOINT_POOL_BYTES
            ),
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _without_observed_free_bytes(payload: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    normalized["storage_preflight"].pop("available_bytes")
    return normalized


def execute_ceiling(
    input_path: Path,
    output_path: Path,
    *,
    expected_shape: tuple[int, int, int],
    expected_bytes: int,
    expected_sha256: str,
    minimum_free_bytes: int = MIN_AP_FREE_BYTES,
    free_space_probe: Callable[[Path], int] = available_bytes,
    storage_root: Path = APDATASTORE_ROOT,
) -> dict[str, object]:
    if not input_path.is_file():
        raise Sp9CeilingError(f"missing retained AFR1 field: {input_path}")
    shape_bytes = math.prod(expected_shape)
    if shape_bytes != expected_bytes:
        raise Sp9CeilingError(
            f"uint8 shape implies {shape_bytes} B != expected {expected_bytes} B"
        )
    actual_bytes = input_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise Sp9CeilingError(f"field bytes {actual_bytes} != expected {expected_bytes}")
    actual_sha256 = sha256_file(input_path)
    if actual_sha256 != expected_sha256:
        raise Sp9CeilingError(
            f"field SHA-256 {actual_sha256} != expected {expected_sha256}"
        )
    free_bytes = free_space_probe(storage_root)
    if free_bytes < minimum_free_bytes:
        raise Sp9CeilingError(
            f"APDataStore free space {free_bytes} B < required {minimum_free_bytes} B"
        )

    field = np.memmap(input_path, dtype=np.uint8, mode="r", shape=expected_shape)
    analysis = analyze_field(field, expected_shape)
    del field
    post_analysis_sha256 = sha256_file(input_path)
    if post_analysis_sha256 != actual_sha256 or input_path.stat().st_size != actual_bytes:
        raise Sp9CeilingError("retained AFR1 field changed during exact analysis")
    ceiling_refused = (
        analysis["primary"]["ideal_stream_ceil_bytes"] > BASE_JOINT_POOL_BYTES
    )
    receipt: dict[str, object] = {
        "schema": "ddm_sp9.semantic_primary_corrected_race.ceiling.v2",
        "axis": AXIS,
        "score_claim": False,
        "input": {
            "path": str(input_path),
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "shape": list(expected_shape),
        },
        "storage_preflight": {
            "root": str(storage_root),
            "available_bytes": free_bytes,
            "minimum_bytes": minimum_free_bytes,
            "passed": True,
        },
        "contract_b": {
            "name": "direct_temporal_delta_scanline_run_events",
            "grammar": (
                "per pair, class or same-as-previous symbols are converted to maximal "
                "scanline runs and emitted as delta-class, log2-length bucket, and residual"
            ),
            "model_class": "static full-corpus MLE over delta-class x run-length-bucket",
            "fitted_model_reason": (
                "full-corpus MLE is used because compress-time sees the complete field and "
                "it minimizes ideal length within this static global model class; every "
                "video-fitted model byte would be counted in a built container"
            ),
            "lossless": True,
            "num_classes": NUM_CLASSES,
            "same_as_previous_symbol": SAME_AS_PREVIOUS,
            "num_length_buckets": NUM_BUCKETS,
        },
        "analysis": analysis,
        "decision": {
            "type": "CEILING-REFUSED" if ceiling_refused else "CEILING-PASSED-BUILD-OWED",
            "verdict_scope": "FORMULATION" if ceiling_refused else "NONE",
            "reason": (
                "the exact ideal stream alone exceeds the shipped 126,926 B joint pool; "
                "even the free-model exact-event first-order relaxation exceeds it"
                if ceiling_refused
                else "the exact ideal stream is below the shipped 126,926 B joint pool"
            ),
            "coder_built": False,
            "archive_built": False,
            "parser_built": False,
            "receiver_built": False,
            "parse_back_run": False,
            "repeat_encode_run": False,
            "rgb_identity_run": False,
            "scorer_run": False,
            "modal_run": False,
            "metal_run": False,
            "scalar_only_payload_rule": (
                "compliant: this calculation reads an already-retained source field and "
                "never materializes encoded candidate bytes"
            ),
        },
    }
    if output_path.is_file():
        existing = json.loads(output_path.read_text())
        if _without_observed_free_bytes(existing) != _without_observed_free_bytes(receipt):
            raise Sp9CeilingError(f"immutable ceiling receipt changed on resume: {output_path}")
        return existing
    _atomic_json(output_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shape", nargs=3, type=int, default=FIELD_SHAPE)
    parser.add_argument("--expected-bytes", type=int, default=FIELD_BYTES)
    parser.add_argument("--expected-sha256", default=FIELD_SHA256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt = execute_ceiling(
        args.input,
        args.output,
        expected_shape=tuple(args.shape),
        expected_bytes=args.expected_bytes,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(receipt["analysis"]["comparison"], sort_keys=True))
    print(f"decision={receipt['decision']['type']} receipt={args.output}")


if __name__ == "__main__":
    main()
