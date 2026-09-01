#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove and price QXO1's target-implicit overwrite event grammar.

The historical S2 packet describes baseline-to-target syndrome events.  QX4's
receiver instead consumes a decoded QBT field and applies every event as an
unconditional overwrite.  Because the retained S2 parser requires strictly
site-sorted, unique events, QBT-relative target no-ops can be omitted and the
remaining writes can be regrouped by target without changing the output.

QXO1 first executes that semantic equivalence on a seeded random sample and on
the full retained population.  Only after both proofs pass does it serialize a
new grammar.  Targets are implicit in five fixed decoder stream slots; no
per-event target or historical baseline label is serialized.  Every derived
field, raw/coded payload, deterministic repeat, packet, archive, and decoded
output is retained under APDataStore custody.  This is scorer-free receiver and
rate evidence, never a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import struct
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tac.optimization.s2_partition_seed import PartitionEvent, decode_partition_seed

STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx1_qxo1_target_overwrite_grammar")
QX1_STORE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qx1")
QX3_STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx3")
QX4_STORE: Final = Path("/Volumes/APDataStore/pact/ddm_qx4")
QX1_CORE: Final = QX1_STORE / "retained/envelopes/core_without_events_exceptions/envelope.qxe"
EVENT_SOURCE: Final = (
    QX1_STORE / "retained/sections/08_events_exceptions_explicit_address_control/raw.bin"
)
QX3_RESULT: Final = QX3_STORE / "RESULT.json"
QX4_RESULT: Final = QX4_STORE / "RESULT.json"
QX4_REFERENCE: Final = QX4_STORE / "retained/receiver/event_applied_partition.primary.u8"
QX2_RUNNER: Final = REPO / "experiments/ddm_qx2_events_section_redesign.py"
QX3_RUNNER: Final = REPO / "experiments/ddm_qx3_receiver_closure.py"
QX4_RUNNER: Final = REPO / "experiments/ddm_qx4_decodable_conditioning_reprice.py"
S2_SOURCE: Final = REPO / "src/tac/optimization/s2_partition_seed.py"
PBR_SOURCE: Final = REPO / "src/tac/witness_dsl/predictor_bound_residual.py"

PINS: Final = {
    QX1_CORE: "4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95",
    EVENT_SOURCE: "df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc",
    QX3_RESULT: "f9a71967ec01aa8905aeb31806f29ebb40a8c9729d0db9c58d36a02a540d7867",
    QX4_RESULT: "a147b3d08c7f485a323be3d41388f72e095ef8d3989e8a813993f0f36679d8bf",
    QX4_REFERENCE: "9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7",
    QX2_RUNNER: "88457037f5cbc272b494306a1613f8c6e2abe3499fdf83164274e3db76b1311c",
    QX3_RUNNER: "238560265a040d942429c7e63e5fe78617a1719d67308e6fada0fbc994a6e272",
    QX4_RUNNER: "0defd490a952de0e0d7b3bc0740b0586f040589bcca93faa22930dfc443c626d",
    S2_SOURCE: "ed8ff4be2f578ea27513cfe20475511e63cdf4cfe43e2fbfa939bed5087af713",
    PBR_SOURCE: "4e7cf4b70e1cb2894c67986626c71ccb0f5ae6e89dca2c76693a9f455f27fd02",
}

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
FRAME_SITES: Final = HEIGHT * WIDTH
SITES: Final = N_PAIRS * FRAME_SITES
SOURCE_EVENTS: Final = 17_926
EXPECTED_OVERWRITES: Final = 8_749
EXPECTED_NOOPS: Final = 9_177
SAMPLE_SEED: Final = 20_260_901
SAMPLE_EVENTS: Final = 1_000
SECTION_CAP_BYTES: Final = 24_093
ARCHIVE_GATE_EXCLUSIVE: Final = 137_986
QX1_CORE_ARCHIVE_BYTES: Final = 113_844
QXE_SECTION_BYTES: Final = 48
MINIMUM_FREE_BYTES: Final = 2_000_000_000
AXIS: Final = "[scorer-free exact receiver/rate measurement]"

GRAMMAR_HEADER: Final = struct.Struct(">4sBBHIIIIII32s32s32s")
OVERWRITE_RECORD: Final = struct.Struct(">IB")
MAGIC: Final = b"QXO1"
VERSION: Final = 1

SOURCE_CITATIONS: Final = {
    "seed_unique_order": "src/tac/optimization/s2_partition_seed.py:130-145",
    "historical_seed_apply": "src/tac/optimization/s2_partition_seed.py:379-394",
    "predictor_bound_consumer": "src/tac/witness_dsl/predictor_bound_residual.py:323-357",
    "qx3_conditioning_decode": "experiments/ddm_qx3_receiver_closure.py:308-377",
    "qx3_historical_consumer": "experiments/ddm_qx3_receiver_closure.py:754-780",
    "qx4_overwrite_consumer": "experiments/ddm_qx4_decodable_conditioning_reprice.py:778-818",
}


class QXO1Error(RuntimeError):
    """A custody, semantic, grammar, parse-back, or retention gate failed."""


@dataclass(frozen=True, order=True)
class OverwriteEvent:
    """One actual QBT-state mutation; target is implicit in its grammar slot."""

    pair: int
    row: int
    col: int
    target_class: int


def sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def require_fact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise QXO1Error(f"{label} is absent: {path}")
    observed = fact(path)
    if observed["sha256"] != expected_sha256:
        raise QXO1Error(
            f"{label} SHA-256 drifted: {observed['sha256']} != {expected_sha256}"
        )
    return observed


def verify_fact(row: dict[str, Any], label: str) -> None:
    path = Path(row["path"])
    if not path.is_file() or fact(path) != row:
        raise QXO1Error(f"{label} retained fact drifted: {path}")


def field_frame(path: Path, pair: int) -> np.ndarray:
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=pair * FRAME_SITES,
        shape=(HEIGHT, WIDTH),
    )


def split_by_pair(events: Sequence[PartitionEvent | OverwriteEvent]) -> list[list[Any]]:
    by_pair: list[list[Any]] = [[] for _ in range(N_PAIRS)]
    for event in events:
        by_pair[event.pair].append(event)
    return by_pair


def sample_indices() -> tuple[int, ...]:
    """Seeded hash-random sample without replacement; never a prefix."""

    seed = struct.pack(">Q", SAMPLE_SEED)
    ranked = sorted(
        range(SOURCE_EVENTS),
        key=lambda index: hashlib.sha256(seed + struct.pack(">I", index)).digest(),
    )
    return tuple(sorted(ranked[:SAMPLE_EVENTS]))


def retain_sample_indices(store: Path, indices: Sequence[int]) -> dict[str, Any]:
    payload = b"".join(struct.pack(">I", index) for index in indices)
    path = store / "retained/semantic_proof/sample_event_indices.u32be"
    atomic_bytes(path, payload)
    return fact(path)


def actual_overwrites(
    conditioning_path: Path, events: Sequence[PartitionEvent]
) -> tuple[tuple[OverwriteEvent, ...], dict[str, Any]]:
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    overwrites: list[OverwriteEvent] = []
    noops = 0
    third_class = 0
    context_equals_historical_baseline = 0
    seen: set[int] = set()
    first_collision: list[int] | None = None
    for event in events:
        site = ((event.pair * HEIGHT) + event.row) * WIDTH + event.col
        if site in seen and first_collision is None:
            first_collision = [event.pair, event.row, event.col]
        seen.add(site)
        context = int(conditioning[event.pair, event.row, event.col])
        context_equals_historical_baseline += int(context == event.baseline_class)
        if context == event.target_class:
            noops += 1
        else:
            third_class += int(context != event.baseline_class)
            overwrites.append(
                OverwriteEvent(event.pair, event.row, event.col, event.target_class)
            )
    del conditioning
    return tuple(overwrites), {
        "source_events": len(events),
        "unique_sites": len(seen),
        "writer_collisions": len(events) - len(seen),
        "first_writer_collision": first_collision,
        "actual_overwrites": len(overwrites),
        "target_noops": noops,
        "context_equals_historical_baseline": context_equals_historical_baseline,
        "overwrite_from_third_class": third_class,
    }


def write_reference_output(
    conditioning_path: Path,
    events: Sequence[PartitionEvent],
    output_path: Path,
) -> dict[str, Any]:
    by_pair = split_by_pair(events)
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    for pair, pair_events in enumerate(by_pair):
        output[pair] = conditioning[pair]
        for event in pair_events:
            output[pair, event.row, event.col] = event.target_class
        if (pair + 1) % 30 == 0:
            output.flush()
    output.flush()
    del output, conditioning
    os.replace(partial, output_path)
    return fact(output_path)


def write_overwrite_output(
    conditioning_path: Path,
    events: Sequence[OverwriteEvent],
    output_path: Path,
) -> dict[str, Any]:
    by_pair = split_by_pair(events)
    conditioning = np.memmap(
        conditioning_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(SITES)
    output = np.memmap(partial, dtype=np.uint8, mode="r+", shape=(N_PAIRS, HEIGHT, WIDTH))
    for pair, pair_events in enumerate(by_pair):
        output[pair] = conditioning[pair]
        seen: set[int] = set()
        for event in pair_events:
            index = event.row * WIDTH + event.col
            if index in seen:
                raise QXO1Error("target-implicit decoder produced multiple writers for one site")
            seen.add(index)
            if int(output[pair, event.row, event.col]) == event.target_class:
                raise QXO1Error("target-implicit grammar retained a QBT-relative no-op")
            output[pair, event.row, event.col] = event.target_class
        if (pair + 1) % 30 == 0:
            output.flush()
    output.flush()
    del output, conditioning
    os.replace(partial, output_path)
    return fact(output_path)


def compare_fields(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = np.memmap(left_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    right = np.memmap(right_path, dtype=np.uint8, mode="r", shape=(N_PAIRS, HEIGHT, WIDTH))
    mismatches = 0
    first: dict[str, int] | None = None
    for pair in range(N_PAIRS):
        different = np.asarray(left[pair]) != np.asarray(right[pair])
        count = int(np.count_nonzero(different))
        mismatches += count
        if count and first is None:
            row, col = (int(value) for value in np.argwhere(different)[0])
            first = {
                "pair": pair,
                "row": row,
                "col": col,
                "left": int(left[pair, row, col]),
                "right": int(right[pair, row, col]),
            }
    del left, right
    return {
        "denominator_sites": SITES,
        "mismatches": mismatches,
        "byte_identical": mismatches == 0,
        "first_counterexample": first,
        "left": fact(left_path),
        "right": fact(right_path),
    }


def write_uleb(value: int, output: bytearray) -> int:
    if value < 0:
        raise QXO1Error("ULEB value cannot be negative")
    start = len(output)
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return len(output) - start


def encode_grammar(
    qx2: Any,
    conditioning_path: Path,
    overwrites: Sequence[OverwriteEvent],
) -> tuple[bytes, dict[str, Any]]:
    by_pair = split_by_pair(overwrites)
    body = bytearray()
    count_bytes = 0
    rank_gap_bytes = 0
    target_counts = [0] * 5
    for pair, pair_events in enumerate(by_pair):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
        order, _ = qx2.distance_order(distance)
        flat_context = conditioning.reshape(-1)
        by_target: list[list[int]] = [[] for _ in range(5)]
        for event in pair_events:
            index = event.row * WIDTH + event.col
            if int(flat_context[index]) == event.target_class:
                raise QXO1Error("encoder attempted to serialize a target no-op")
            by_target[event.target_class].append(index)
        inverse = np.empty(FRAME_SITES, dtype=np.int32)
        for target in range(5):
            eligible = order[flat_context[order] != target]
            inverse.fill(-1)
            inverse[eligible] = np.arange(eligible.size, dtype=np.int32)
            ranks = sorted(int(inverse[index]) for index in by_target[target])
            if any(rank < 0 for rank in ranks):
                raise QXO1Error("overwrite is absent from its decoder-derived target alphabet")
            count_bytes += write_uleb(len(ranks), body)
            target_counts[target] += len(ranks)
            previous = -1
            for rank in ranks:
                rank_gap_bytes += write_uleb(rank - previous - 1, body)
                previous = rank
    header = GRAMMAR_HEADER.pack(
        MAGIC,
        VERSION,
        0,
        0,
        N_PAIRS,
        HEIGHT,
        WIDTH,
        SOURCE_EVENTS,
        len(overwrites),
        len(body),
        bytes.fromhex(sha256_file(conditioning_path)),
        bytes.fromhex(PINS[EVENT_SOURCE]),
        bytes.fromhex(PINS[QX4_REFERENCE]),
    )
    return header + bytes(body), {
        "grammar": "target_implicit_distance_rank_overwrite_v1",
        "source_events": SOURCE_EVENTS,
        "target_noops_omitted": SOURCE_EVENTS - len(overwrites),
        "overwrite_records": len(overwrites),
        "target_group_counts": target_counts,
        "fixed_target_group_count": N_PAIRS * 5,
        "target_group_count_bytes": count_bytes,
        "distance_rank_gap_bytes": rank_gap_bytes,
        "body_bytes": len(body),
        "header_bytes": GRAMMAR_HEADER.size,
        "explicit_per_event_target_label_bytes": 0,
        "historical_baseline_label_bytes": 0,
        "target_information_is_implicit_in_fixed_group_slot": True,
    }


def decode_grammar(
    qx2: Any,
    payload: bytes,
    conditioning_path: Path,
) -> tuple[OverwriteEvent, ...]:
    if len(payload) < GRAMMAR_HEADER.size:
        raise QXO1Error("QXO1 grammar is truncated")
    (
        magic,
        version,
        flags,
        reserved,
        pairs,
        height,
        width,
        source_events,
        overwrite_count,
        body_len,
        conditioning_sha,
        source_sha,
        reference_sha,
    ) = GRAMMAR_HEADER.unpack_from(payload)
    if (
        magic != MAGIC
        or version != VERSION
        or flags
        or reserved
        or (pairs, height, width) != (N_PAIRS, HEIGHT, WIDTH)
        or source_events != SOURCE_EVENTS
        or overwrite_count != EXPECTED_OVERWRITES
        or GRAMMAR_HEADER.size + body_len != len(payload)
    ):
        raise QXO1Error("QXO1 grammar identity or geometry drifted")
    if conditioning_sha.hex() != sha256_file(conditioning_path):
        raise QXO1Error("QXO1 conditioning field identity drifted")
    if source_sha.hex() != PINS[EVENT_SOURCE] or reference_sha.hex() != PINS[QX4_REFERENCE]:
        raise QXO1Error("QXO1 source event or reference-output identity drifted")
    body = payload[GRAMMAR_HEADER.size :]
    offset = 0
    events: list[OverwriteEvent] = []
    for pair in range(N_PAIRS):
        conditioning = np.asarray(field_frame(conditioning_path, pair))
        _, distance = qx2.boundary_and_distance(conditioning)
        order, _ = qx2.distance_order(distance)
        flat_context = conditioning.reshape(-1)
        seen: set[int] = set()
        for target in range(5):
            eligible = order[flat_context[order] != target]
            count, offset = qx2.read_uleb(body, offset)
            previous = -1
            for _ in range(count):
                delta, offset = qx2.read_uleb(body, offset)
                rank = previous + delta + 1
                if rank >= eligible.size:
                    raise QXO1Error("QXO1 distance rank exceeds its target alphabet")
                index = int(eligible[rank])
                if index in seen:
                    raise QXO1Error("QXO1 grammar assigns two targets to one site")
                seen.add(index)
                row, col = divmod(index, WIDTH)
                events.append(OverwriteEvent(pair, row, col, target))
                previous = rank
    if offset != len(body) or len(events) != overwrite_count:
        raise QXO1Error("QXO1 decoder did not consume the exact body/overwrite denominator")
    return tuple(events)


def serialize_overwrites(events: Sequence[OverwriteEvent]) -> bytes:
    output = bytearray()
    for event in events:
        site = ((event.pair * HEIGHT) + event.row) * WIDTH + event.col
        output.extend(OVERWRITE_RECORD.pack(site, event.target_class))
    return bytes(output)


def extract_grammar_from_archive(qx3: Any, archive_bytes: bytes) -> tuple[bytes, bytes]:
    with zipfile.ZipFile(BytesIO(archive_bytes), "r") as archive:
        if archive.namelist() != ["state/qx1.qxe"]:
            raise QXO1Error("QXO1 archive member roster drifted")
        packet = archive.read("state/qx1.qxe")
    records, sections, _codecs = qx3.parse_qxe(packet, 8)
    core = qx3.QXE_HEADER.pack(b"QXE1", 1, 0, 7) + b"".join(records[:7])
    if sha256_bytes(core) != PINS[QX1_CORE]:
        raise QXO1Error("QXO1 archive core differs from the pinned QX1 core")
    return core, sections[8]


def preflight(store: Path) -> tuple[dict[str, Any], Any, Any, tuple[PartitionEvent, ...]]:
    if store.resolve() != STORE.resolve():
        raise QXO1Error(f"custody is pinned to {STORE}, not {store.resolve()}")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise QXO1Error(f"storage preflight failed: {free} < {MINIMUM_FREE_BYTES}")
    inputs = {str(path): require_fact(path, digest, path.name) for path, digest in PINS.items()}
    qx2 = importlib.import_module("experiments.ddm_qx2_events_section_redesign")
    qx3 = importlib.import_module("experiments.ddm_qx3_receiver_closure")
    seed = decode_partition_seed(EVENT_SOURCE.read_bytes())
    events = tuple(seed.events)
    if (seed.n_pairs, seed.height, seed.width, len(events)) != (
        N_PAIRS,
        HEIGHT,
        WIDTH,
        SOURCE_EVENTS,
    ):
        raise QXO1Error("retained source-event geometry drifted")
    qx3_result = json.loads(QX3_RESULT.read_text(encoding="utf-8"))
    qx4_result = json.loads(QX4_RESULT.read_text(encoding="utf-8"))
    if (
        qx3_result["derived_baseline"]["fresh_decode_vs_retained_native_mismatches"] != 0
        or qx4_result["receiver_proof"]["primary"]["applied_field"]["output"]["sha256"]
        != PINS[QX4_REFERENCE]
        or qx4_result["receiver_proof"]["primary"]["applied_field"]["context_equals_target_noop"]
        != EXPECTED_NOOPS
    ):
        raise QXO1Error("QX3/QX4 source receipts drifted from the chartered semantic opening")
    stage = {
        "schema": "ddm_qxo1_preflight.v1",
        "complete": True,
        "runner_source": fact(Path(__file__).resolve()),
        "axis": AXIS,
        "selection_mode": "seeded random n1000 then full n17926",
        "storage": {"path": str(store), "observed_free_bytes": free},
        "inputs": inputs,
        "source_citations": SOURCE_CITATIONS,
        "denominators": {"pairs": N_PAIRS, "sites": SITES, "source_events": SOURCE_EVENTS},
        "authority_boundaries": {
            "scorers_loaded": 0,
            "contest_eval_invocations": 0,
            "modal_invocations": 0,
            "metal_invocations": 0,
            "score_claim": False,
        },
    }
    atomic_json(store / "checkpoints/STAGE0_PREFLIGHT.json", stage)
    return stage, qx2, qx3, events


def semantics_proof(
    store: Path,
    conditioning_path: Path,
    events: tuple[PartitionEvent, ...],
) -> tuple[dict[str, Any], tuple[OverwriteEvent, ...]]:
    stage_path = store / "checkpoints/STAGE2_SEMANTICS_PROOF.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("runner_source_sha256") != sha256_file(Path(__file__).resolve()):
            raise QXO1Error("semantic-proof checkpoint was produced by different runner source")
        for row in stage["retained_payloads"]:
            verify_fact(row, "semantic-proof payload")
        overwrites, census = actual_overwrites(conditioning_path, events)
        if census != stage["full"]["event_census"]:
            raise QXO1Error("resumed semantic event census drifted")
        return stage, overwrites

    indices = sample_indices()
    sample_index_fact = retain_sample_indices(store, indices)
    selected = set(indices)
    sample_events = tuple(event for index, event in enumerate(events) if index in selected)
    sample_overwrites, sample_census = actual_overwrites(conditioning_path, sample_events)
    sample_reference_path = store / "retained/semantic_proof/sample_reference_qx4_semantics.u8"
    sample_targetfree_path = store / "retained/semantic_proof/sample_target_implicit.u8"
    sample_reference = write_reference_output(conditioning_path, sample_events, sample_reference_path)
    sample_targetfree = write_overwrite_output(
        conditioning_path, sample_overwrites, sample_targetfree_path
    )
    sample_comparison = compare_fields(sample_reference_path, sample_targetfree_path)

    retained = [sample_index_fact, sample_reference, sample_targetfree]
    full_overwrites: tuple[OverwriteEvent, ...] = ()
    full_output: dict[str, Any] | None = None
    full_comparison: dict[str, Any] | None = None
    full_census: dict[str, Any] | None = None
    if sample_comparison["byte_identical"]:
        full_overwrites, full_census = actual_overwrites(conditioning_path, events)
        full_output_path = store / "retained/semantic_proof/full_target_implicit.u8"
        full_output = write_overwrite_output(conditioning_path, full_overwrites, full_output_path)
        full_comparison = compare_fields(QX4_REFERENCE, full_output_path)
        retained.append(full_output)

    proof_passed = bool(
        sample_comparison["byte_identical"]
        and full_comparison is not None
        and full_comparison["byte_identical"]
        and full_census is not None
        and full_census["writer_collisions"] == 0
        and full_census["actual_overwrites"] == EXPECTED_OVERWRITES
        and full_census["target_noops"] == EXPECTED_NOOPS
    )
    stage = {
        "schema": "ddm_qxo1_semantics_proof.v1",
        "complete": True,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "proof_passed": proof_passed,
        "source_citations": SOURCE_CITATIONS,
        "application_contract": {
            "reference": "copy decoded QBT field, then overwrite source events in retained order",
            "new_object": "omit target-equal writes; regroup unique actual writes by implicit target slot",
            "last_writer_wins": True,
            "regrouping_safe_only_because_writer_collisions_are_zero": True,
            "historical_c1_syndrome_identity_preserved": False,
            "consumer_output_field_preserved": proof_passed,
        },
        "sample": {
            "seed": SAMPLE_SEED,
            "selection": "smallest SHA256(seed_be64 || event_index_be32), without replacement; applied in source order",
            "denominator_source_events": SOURCE_EVENTS,
            "selected_events": SAMPLE_EVENTS,
            "indices": sample_index_fact,
            "event_census": sample_census,
            "comparison": sample_comparison,
        },
        "full": {
            "denominator_source_events": SOURCE_EVENTS,
            "event_census": full_census,
            "output": full_output,
            "comparison": full_comparison,
        },
        "retained_payloads": retained,
        "first_counterexample": (
            sample_comparison["first_counterexample"]
            if not sample_comparison["byte_identical"]
            else None if full_comparison is None else full_comparison["first_counterexample"]
        ),
    }
    atomic_json(stage_path, stage)
    return stage, full_overwrites


def retain_grammar(
    store: Path,
    qx2: Any,
    qx3: Any,
    conditioning_path: Path,
    expected_overwrites: tuple[OverwriteEvent, ...],
    core: bytes,
) -> dict[str, Any]:
    stage_path = store / "checkpoints/STAGE3_GRAMMAR.json"
    if stage_path.is_file():
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("runner_source_sha256") != sha256_file(Path(__file__).resolve()):
            raise QXO1Error("grammar checkpoint was produced by different runner source")
        for row in stage["retained_payloads"]:
            verify_fact(row, "grammar payload")
        raw = Path(stage["raw"]["path"]).read_bytes()
        decoded = decode_grammar(qx2, raw, conditioning_path)
        if tuple(sorted(decoded)) != tuple(sorted(expected_overwrites)):
            raise QXO1Error("resumed grammar no longer reconstructs all actual overwrites")
        return stage

    root = store / "retained/grammar_v1"
    raw, anatomy = encode_grammar(qx2, conditioning_path, expected_overwrites)
    raw_path = root / "raw.qxo"
    atomic_bytes(raw_path, raw)
    decoded = decode_grammar(qx2, raw, conditioning_path)
    if tuple(sorted(decoded)) != tuple(sorted(expected_overwrites)):
        raise QXO1Error("raw QXO1 grammar did not reconstruct the actual overwrite object")
    expected_records_path = root / "verification_expected_overwrites.u32be_target_u8"
    decoded_records_path = root / "verification_decoded_overwrites.u32be_target_u8"
    atomic_bytes(expected_records_path, serialize_overwrites(sorted(expected_overwrites)))
    atomic_bytes(decoded_records_path, serialize_overwrites(sorted(decoded)))
    if expected_records_path.read_bytes() != decoded_records_path.read_bytes():
        raise QXO1Error("diagnostic overwrite record parse-back drifted")

    coder_rows: list[dict[str, Any]] = []
    for codec in qx2.CODECS:
        coded = qx2.compress(codec, raw)
        repeat = qx2.compress(codec, raw)
        coded_path = root / f"candidate.{codec}.bin"
        repeat_path = root / f"candidate.{codec}.repeat.bin"
        atomic_bytes(coded_path, coded)
        atomic_bytes(repeat_path, repeat)
        if coded != repeat or qx2.decompress(codec, coded) != raw:
            raise QXO1Error(f"{codec}: real-coder repeat or exact parse-back failed")
        coder_rows.append(
            {
                "codec": codec,
                "payload": fact(coded_path),
                "repeat": fact(repeat_path),
                "deterministic_repeat": True,
                "parseback_exact": True,
            }
        )
    winner = min(coder_rows, key=lambda row: (row["payload"]["bytes"], row["codec"]))
    winner_bytes = Path(winner["payload"]["path"]).read_bytes()
    packet = qx2.build_complete_envelope(core, raw, winner_bytes, winner["codec"])
    packet_repeat = qx2.build_complete_envelope(core, raw, winner_bytes, winner["codec"])
    archive = qx2.build_zip(packet)
    archive_repeat = qx2.build_zip(packet_repeat)
    packet_path = root / "complete.qxe"
    packet_repeat_path = root / "complete.repeat.qxe"
    archive_path = root / "archive.zip"
    archive_repeat_path = root / "archive.repeat.zip"
    atomic_bytes(packet_path, packet)
    atomic_bytes(packet_repeat_path, packet_repeat)
    atomic_bytes(archive_path, archive)
    atomic_bytes(archive_repeat_path, archive_repeat)
    if packet != packet_repeat or archive != archive_repeat:
        raise QXO1Error("QXO1 packet/archive determinism repeat failed")
    if len(archive) != QX1_CORE_ARCHIVE_BYTES + QXE_SECTION_BYTES + len(winner_bytes):
        raise QXO1Error("QXO1 exact archive arithmetic does not close")

    decoder_passes: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = [
        fact(raw_path),
        fact(expected_records_path),
        fact(decoded_records_path),
        *(row[key] for row in coder_rows for key in ("payload", "repeat")),
        fact(packet_path),
        fact(packet_repeat_path),
        fact(archive_path),
        fact(archive_repeat_path),
    ]
    for label, archive_payload in (("primary", archive), ("repeat", archive_repeat)):
        decoded_core, decoded_raw = extract_grammar_from_archive(qx3, archive_payload)
        fresh_conditioning, conditioning_stage = qx3.derive_decoder_baseline(store, decoded_core)
        decoded_events = decode_grammar(qx2, decoded_raw, fresh_conditioning)
        decoded_record_path = root / f"decoded_overwrites.{label}.u32be_target_u8"
        atomic_bytes(decoded_record_path, serialize_overwrites(decoded_events))
        output_path = root / f"decoded_field.{label}.u8"
        output = write_overwrite_output(fresh_conditioning, decoded_events, output_path)
        comparison = compare_fields(QX4_REFERENCE, output_path)
        if not comparison["byte_identical"]:
            raise QXO1Error(f"{label}: grammar output differs from QX4 overwrite semantics")
        retained.extend((fact(decoded_record_path), output))
        decoder_passes.append(
            {
                "label": label,
                "conditioning": fact(fresh_conditioning),
                "conditioning_derivation": conditioning_stage,
                "decoded_overwrites": len(decoded_events),
                "decoded_records": fact(decoded_record_path),
                "output": output,
                "reference_comparison": comparison,
                "encoder_only_inputs_used_by_receiver": [],
            }
        )
    if (
        decoder_passes[0]["decoded_records"]["sha256"]
        != decoder_passes[1]["decoded_records"]["sha256"]
        or decoder_passes[0]["output"]["sha256"] != decoder_passes[1]["output"]["sha256"]
    ):
        raise QXO1Error("QXO1 double-decode output is not byte-identical")

    stage = {
        "schema": "ddm_qxo1_grammar_v1.v1",
        "complete": True,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "grammar_id": "target_implicit_distance_rank_overwrite_v1",
        "raw": fact(raw_path),
        "representation_anatomy": anatomy,
        "coders": coder_rows,
        "winner_codec": winner["codec"],
        "winner_payload": winner["payload"],
        "complete_packet": fact(packet_path),
        "complete_packet_repeat": fact(packet_repeat_path),
        "archive": fact(archive_path),
        "archive_repeat": fact(archive_repeat_path),
        "decoder_passes": decoder_passes,
        "exact_parseback": True,
        "double_decode_identity": True,
        "section_cap_bytes": SECTION_CAP_BYTES,
        "delta_bytes_vs_section_cap": winner["payload"]["bytes"] - SECTION_CAP_BYTES,
        "archive_gate_exclusive": ARCHIVE_GATE_EXCLUSIVE,
        "delta_bytes_vs_largest_legal_archive": len(archive) - (ARCHIVE_GATE_EXCLUSIVE - 1),
        "retained_payloads": retained,
    }
    atomic_json(stage_path, stage)
    return stage


def failure_result(
    store: Path, stage0: dict[str, Any], conditioning: dict[str, Any], proof: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": "ddm_qxo1_target_overwrite_grammar.v1",
        "complete": True,
        "verdict": "FAMILY-CLOSED-COMPLETELY",
        "verdict_scope": "FAMILY: QX historical-event and target-overwrite routes on the pinned QBT field",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "stage0": stage0,
        "conditioning": conditioning,
        "semantics_proof": proof,
        "grammar_built": False,
        "first_counterexample": proof["first_counterexample"],
        "follow_on": {
            "disposition": "FOLDED",
            "reason": "consumer-semantics proof failed before grammar admission",
        },
    }


def run(store: Path) -> dict[str, Any]:
    started = time.perf_counter()
    stage0, qx2, qx3, events = preflight(store)
    core = QX1_CORE.read_bytes()
    conditioning_path, conditioning_stage = qx3.derive_decoder_baseline(store, core)
    if sha256_file(conditioning_path) != "afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd":
        raise QXO1Error("fresh QX1/QBT decoder field drifted")
    proof, overwrites = semantics_proof(store, conditioning_path, events)
    if not proof["proof_passed"]:
        result = failure_result(store, stage0, conditioning_stage, proof)
    else:
        grammar = retain_grammar(store, qx2, qx3, conditioning_path, overwrites, core)
        section_cleared = grammar["winner_payload"]["bytes"] <= SECTION_CAP_BYTES
        archive_cleared = grammar["archive"]["bytes"] < ARCHIVE_GATE_EXCLUSIVE
        under_gate = section_cleared and archive_cleared
        result = {
            "schema": "ddm_qxo1_target_overwrite_grammar.v1",
            "complete": True,
            "verdict": "UNDER-GATE" if under_gate else "FAMILY-CLOSED-COMPLETELY",
            "verdict_scope": (
                "INSTANCE: target-implicit overwrite grammar v1 on pinned decoded QBT"
                if under_gate
                else "FAMILY: QX explicit forms plus the target-implicit overwrite successor on pinned decoded QBT"
            ),
            "axis": AXIS,
            "score_claim": False,
            "pointer_moved": False,
            "selection_mode": "seeded random n1000 proof then full n17926 proof and full real-coder race",
            "denominators": {
                "pairs": N_PAIRS,
                "sites": SITES,
                "source_events": SOURCE_EVENTS,
                "actual_overwrites": EXPECTED_OVERWRITES,
                "target_noops": EXPECTED_NOOPS,
            },
            "stage0": stage0,
            "conditioning": conditioning_stage,
            "semantics_proof": proof,
            "grammar": grammar,
            "section_gate": {
                "maximum_payload_bytes": SECTION_CAP_BYTES,
                "observed_payload_bytes": grammar["winner_payload"]["bytes"],
                "cleared": section_cleared,
                "delta_bytes_vs_cap": grammar["winner_payload"]["bytes"] - SECTION_CAP_BYTES,
                "qx1_core_archive_bytes": QX1_CORE_ARCHIVE_BYTES,
                "section_header_bytes": QXE_SECTION_BYTES,
                "complete_archive_bytes": grammar["archive"]["bytes"],
                "strict_archive_bytes_lt": ARCHIVE_GATE_EXCLUSIVE,
                "archive_cleared": archive_cleared,
                "exact_arithmetic": (
                    f"{QX1_CORE_ARCHIVE_BYTES} + {QXE_SECTION_BYTES} + "
                    f"{grammar['winner_payload']['bytes']} = {grammar['archive']['bytes']}"
                ),
            },
            "cross_half": {
                "held_by_this_object": "rate-shape: receiver-closed target-implicit overwrite section",
                "other_half": "distortion: BR2's measured refusal of the retained born instance remains standing",
                "byte_win_changes": "the QX rate-feasibility map on this pinned decoded field",
                "byte_win_does_not_change": "BR2 distortion, any Seg/Pose component, or the exact frontier",
            },
            "follow_on": (
                {
                    "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                    "owner": "MAIN n600 scorer-realization scheduler",
                    "consumer_store": str(store / "RESULT.json"),
                    "fire_trigger": (
                        "MAIN binds this exact archive and double-decoded field to the retained born-object "
                        "realization path under BR2's payload-retaining n600 protocol, while preserving all "
                        "grammar/core hashes and explicitly carrying BR2's standing distortion refusal"
                    ),
                }
                if under_gate
                else {
                    "disposition": "FOLDED",
                    "reason": "the new semantic object remains over the complete byte gate",
                }
            ),
            "authority_boundaries": {
                "scorers_loaded": 0,
                "contest_eval_invocations": 0,
                "modal_invocations": 0,
                "metal_invocations": 0,
                "distortion_measured_here": False,
                "contest_score_measured": False,
                "rate_and_receiver_exactness_measured": True,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    atomic_json(store / "RESULT.json", result)
    manifest = {
        "schema": "ddm_qxo1_run_manifest.v1",
        "complete": True,
        "result": fact(store / "RESULT.json"),
        "source": fact(Path(__file__).resolve()),
        "command": f"{sys.executable} {Path(__file__).resolve()} --resume-from {store}",
        "sample_seed": SAMPLE_SEED,
        "retention": "all materialized fields, raw/coded/repeat payloads, packets, archives, and decode outputs retained",
        "cleanup": "none fired; all QXO1 payloads remain under AP custody",
    }
    atomic_json(store / "RUN_MANIFEST.json", manifest)
    atomic_json(
        store / "checkpoints/STAGE4_COMPLETE.json",
        {
            "schema": "ddm_qxo1_complete.v1",
            "complete": True,
            "verdict": result["verdict"],
            "result": fact(store / "RESULT.json"),
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, default=STORE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.resume_from)
    summary = {
        "verdict": result["verdict"],
        "proof_passed": result["semantics_proof"]["proof_passed"],
    }
    if result.get("grammar"):
        summary.update(
            {
                "payload_bytes": result["grammar"]["winner_payload"]["bytes"],
                "archive_bytes": result["grammar"]["archive"]["bytes"],
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
