#!/usr/bin/env python3
"""Screen the Road-rooted five-class peel ladder and confirm its top chain.

The screen is intentionally a bound instrument, not a reference-form verdict.  It
factorizes the retained n600 five-class field into four binary planes with Road as
the implicit base.  A peel order is fine-to-coarse; decoding traverses it in
reverse.  Every rung conditions on a reduced view of the already-decoded coarse
field plus causal symbols from the current binary plane.

The reduced context is derived from the PP1 temporal-context winner and from the
extra non-causal information supplied by the ladder:

* current plane: left, up, and previous-frame same-site (ternary: 0/1/unavailable);
* decoded coarse field: left, right, up, down, previous-frame, and next-frame
  exact class values (base six including an out-of-bounds sentinel).

All 24 peel orders are ranked from the 32 unique (decoded-set, target-class)
states.  Each state retains its active binary plane, sufficient count table, and
JSON receipt.  The top chain is then encoded with the pinned D3 RC64 mechanism,
compiled at alphabet two, using an exact KT-prequential adaptive extension.  The
decoder control replays source-derived context keys, so it proves real-coder
round-trip at this screen fidelity but is deliberately NOT receiver-closed
decode-identity; the reference-form successor owns that proof and byte-close.

No scorer, renderer, archive candidate, Modal job, or score claim is produced.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2

DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_d3c_class_pyramid")
SOURCE_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/decoded_tokens_instrumented.u8"
)
SOURCE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
D3_FOUR_STREAM = Path(
    "/Volumes/APDataStore/pact/ddm_d3_alphabet_merge/retained/encode/"
    "token_stream_alphabet4_n600.bin"
)
D3_FOUR_STREAM_SHA256 = "84fa2f499fb6c052cf6a43f8cae98c227ac32412ce1495cc715aa5af94b8692d"
D3B_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/RESULT.json"
)

N_FRAMES = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
FIELD_BYTES = N_FRAMES * PLANE
ROAD = 0
LANE = 1
UNDRIVABLE = 2
MOVABLE = 3
MYCAR = 4
PEEL_CLASSES = (LANE, UNDRIVABLE, MOVABLE, MYCAR)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_SLUGS = ("road", "lane", "undrivable", "movable", "mycar")

SCREEN_AXIS = "[macOS-CPU advisory / scorer-free KT conditional-bound screen, n600]"
CONFIRM_AXIS = "[macOS-CPU advisory / scorer-free real RC64 bytes, n600]"
JOINT_SUBSYSTEM_BYTES = 127_292
D3_FOUR_STREAM_BYTES = 49_696
MINIMUM_FREE_BYTES = 2 << 30
CONTEXT_SPEC = "plane_l_u_prev__coarse_l_r_u_d_prev_next_r1"
PLANE_CONTEXT_BASE = 3
COARSE_CONTEXT_BASE = 6
PLANE_CONTEXT_FEATURES = 3
COARSE_CONTEXT_FEATURES = 6
N_CONTEXTS = (PLANE_CONTEXT_BASE**PLANE_CONTEXT_FEATURES) * (
    COARSE_CONTEXT_BASE**COARSE_CONTEXT_FEATURES
)
PRIOR_PEEL_ORDER = (LANE, MOVABLE, MYCAR, UNDRIVABLE)


class D3CError(RuntimeError):
    """A custody, screen, or real-coder invariant failed closed."""


@dataclass(frozen=True)
class RungState:
    decoded: tuple[int, ...]
    target: int

    @property
    def decoded_mask(self) -> int:
        mask = 0
        for value in self.decoded:
            mask |= 1 << value
        return mask

    @property
    def state_id(self) -> str:
        decoded = "none" if not self.decoded else "-".join(
            CLASS_SLUGS[value] for value in self.decoded
        )
        return f"decoded_{decoded}__target_{CLASS_SLUGS[self.target]}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def progress(**record: Any) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def require_file(path: Path, expected_bytes: int, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise D3CError(f"{label} is absent: {path}")
    observed = file_fact(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise D3CError(f"{label} custody pin failed: {observed}")
    return observed


def verify_inputs(store: Path) -> dict[str, Any]:
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise D3CError(
            f"storage preflight failed: {free} B free < {MINIMUM_FREE_BYTES} B"
        )
    d3b: dict[str, Any] | None = None
    if D3B_RESULT.is_file():
        payload = json.loads(D3B_RESULT.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        hpac = next(
            (row for row in rows if row.get("config", {}).get("name") == "hpac_conditional"),
            None,
        )
        if payload.get("complete") and hpac is not None:
            identity = hpac.get("receiver_identity", {})
            if not identity.get("byte_identical"):
                raise D3CError("live D3B result claims completion without byte identity")
            d3b = {
                "result": file_fact(D3B_RESULT),
                "hpac_conditional_lane_packet_bytes": hpac["lane_packet"]["bytes"],
                "hpac_conditional_lane_body_bytes": hpac["lane_rc64_body"]["bytes"],
                "token_subsystem_bytes": hpac["token_subsystem_bytes"],
                "delta_vs_127292_bytes": hpac["delta_vs_127292_bytes"],
                "source_field_sha256": identity["source_field_sha256"],
                "read_only_recall": True,
            }
            if d3b["source_field_sha256"] != SOURCE_FIELD_SHA256:
                raise D3CError("live D3B result uses a different source field")
    return {
        "storage": {
            "path": str(store),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "observed_free_bytes": free,
            "status": "PASS",
        },
        "source_field": require_file(
            SOURCE_FIELD, FIELD_BYTES, SOURCE_FIELD_SHA256, "retained five-class field"
        ),
        "d3_four_stream": require_file(
            D3_FOUR_STREAM,
            D3_FOUR_STREAM_BYTES,
            D3_FOUR_STREAM_SHA256,
            "D3 four-symbol stream",
        ),
        "live_d3b_result": d3b,
    }


def source_field() -> np.memmap:
    return np.memmap(
        SOURCE_FIELD,
        dtype=np.uint8,
        mode="r",
        shape=(N_FRAMES, HEIGHT, WIDTH),
    )


def shift_spatial(values: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    output = np.full(values.shape, fill, dtype=values.dtype)
    y0 = max(0, -dy)
    y1 = min(HEIGHT, HEIGHT - dy)
    x0 = max(0, -dx)
    x1 = min(WIDTH, WIDTH - dx)
    output[:, y0:y1, x0:x1] = values[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    return output


def shift_temporal(values: np.ndarray, delta: int, fill: int) -> np.ndarray:
    output = np.full(values.shape, fill, dtype=values.dtype)
    if delta == -1:
        output[1:] = values[:-1]
    elif delta == 1:
        output[:-1] = values[1:]
    else:  # pragma: no cover - guarded by the fixed context specification
        raise ValueError(f"unsupported temporal delta: {delta}")
    return output


def support_for(field: np.ndarray, decoded: Iterable[int]) -> np.ndarray:
    support = np.ones(field.shape, dtype=bool)
    for value in decoded:
        support &= field != value
    return support


def coarse_field(field: np.ndarray, decoded: Iterable[int]) -> np.ndarray:
    coarse = np.full(field.shape, ROAD, dtype=np.uint8)
    for value in decoded:
        coarse[field == value] = value
    return coarse


def _masked_plane_feature(
    plane: np.ndarray,
    support: np.ndarray,
    *,
    spatial: tuple[int, int] | None = None,
    temporal: int | None = None,
) -> np.ndarray:
    if (spatial is None) == (temporal is None):
        raise ValueError("exactly one shift kind is required")
    if spatial is not None:
        shifted_plane = shift_spatial(plane, *spatial, fill=0)
        shifted_support = shift_spatial(support, *spatial, fill=0)
    else:
        assert temporal is not None
        shifted_plane = shift_temporal(plane, temporal, fill=0)
        shifted_support = shift_temporal(support, temporal, fill=0)
    return np.where(shifted_support, shifted_plane, np.uint8(2)).astype(np.uint8)


def build_state_arrays(
    field: np.ndarray, state: RungState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active context keys, symbols, and the full support mask."""
    support = support_for(field, state.decoded)
    plane = np.asarray(field == state.target, dtype=np.uint8)
    coarse = coarse_field(field, state.decoded)

    plane_features = (
        _masked_plane_feature(plane, support, spatial=(0, -1)),
        _masked_plane_feature(plane, support, spatial=(-1, 0)),
        _masked_plane_feature(plane, support, temporal=-1),
    )
    coarse_features = (
        shift_spatial(coarse, 0, -1, fill=5),
        shift_spatial(coarse, 0, 1, fill=5),
        shift_spatial(coarse, -1, 0, fill=5),
        shift_spatial(coarse, 1, 0, fill=5),
        shift_temporal(coarse, -1, fill=5),
        shift_temporal(coarse, 1, fill=5),
    )

    context = np.zeros(field.shape, dtype=np.uint32)
    for feature in plane_features:
        context *= PLANE_CONTEXT_BASE
        context += feature
    for feature in coarse_features:
        context *= COARSE_CONTEXT_BASE
        context += feature
    if int(context.max()) >= N_CONTEXTS:
        raise D3CError("context encoder exceeded its derived finite range")
    keys = np.ascontiguousarray(context[support], dtype=np.uint32)
    symbols = np.ascontiguousarray(plane[support], dtype=np.uint8)
    return keys, symbols, support


def exact_binary_lengths(counts: np.ndarray) -> tuple[float, float, int]:
    totals = counts.sum(axis=1)
    occupied = totals > 0
    table = counts[occupied].astype(np.float64)
    total = totals[occupied].astype(np.float64)
    positive = table > 0
    repeated_total = np.repeat(total, positive.sum(axis=1))
    cells = table[positive]
    hindsight_bits = float(
        np.sum(cells * (np.log2(repeated_total) - np.log2(cells)))
    )
    half = 0.5
    per_context = (
        gammaln(total + 1.0)
        - gammaln(1.0)
        - (gammaln(table + half) - gammaln(half)).sum(axis=1)
    )
    kt_bits = float(per_context.sum() / np.log(2.0))
    return hindsight_bits, kt_bits, int(occupied.sum())


def state_paths(store: Path, state: RungState) -> dict[str, Path]:
    root = store / "retained/screen"
    return {
        "receipt": root / "receipts" / f"{state.state_id}.json",
        "plane": root / "planes" / f"{state.state_id}.packbits",
        "counts": root / "counts" / f"{state.state_id}.npz",
    }


def reusable_state_receipt(store: Path, state: RungState) -> dict[str, Any] | None:
    paths = state_paths(store, state)
    if not paths["receipt"].is_file():
        return None
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "ddm_d3c_screen_rung.v1"
        or receipt.get("source_field_sha256") != SOURCE_FIELD_SHA256
        or receipt.get("context_spec") != CONTEXT_SPEC
        or receipt.get("state_id") != state.state_id
    ):
        return None
    for key in ("active_plane", "sufficient_counts"):
        fact = receipt.get(key)
        if not isinstance(fact, dict):
            return None
        path = Path(fact["path"])
        if not path.is_file() or file_fact(path) != fact:
            return None
    return receipt


def screen_state(store: Path, field: np.ndarray, state: RungState, resume: bool) -> dict[str, Any]:
    if resume:
        reused = reusable_state_receipt(store, state)
        if reused is not None:
            progress(stage="screen", event="reuse", state=state.state_id)
            return reused
    started = time.perf_counter()
    keys, symbols, support = build_state_arrays(field, state)
    joint = keys.astype(np.int64) * 2 + symbols.astype(np.int64)
    dense_counts = np.bincount(joint, minlength=N_CONTEXTS * 2).reshape(N_CONTEXTS, 2)
    hindsight_bits, kt_bits, occupied = exact_binary_lengths(dense_counts)
    occupied_mask = dense_counts.sum(axis=1) > 0
    occupied_ids = np.flatnonzero(occupied_mask).astype(np.uint32)
    sparse_counts = dense_counts[occupied_mask].astype(np.uint32)

    paths = state_paths(store, state)
    plane_payload = np.packbits(symbols, bitorder="big").tobytes()
    atomic_bytes(paths["plane"], plane_payload)
    atomic_npz(
        paths["counts"],
        context_id=occupied_ids,
        count_zero=sparse_counts[:, 0],
        count_one=sparse_counts[:, 1],
    )
    receipt = {
        "schema": "ddm_d3c_screen_rung.v1",
        "complete": True,
        "axis": SCREEN_AXIS,
        "score_claim": False,
        "rank_only": True,
        "state_id": state.state_id,
        "decoded_classes": [CLASS_NAMES[value] for value in state.decoded],
        "decoded_class_ids": list(state.decoded),
        "target_class": CLASS_NAMES[state.target],
        "target_class_id": state.target,
        "source_field_sha256": SOURCE_FIELD_SHA256,
        "context_spec": CONTEXT_SPEC,
        "context_fidelity": {
            "plane_causal": ["left", "up", "previous_frame_same_site"],
            "decoded_coarse_noncausal": [
                "left",
                "right",
                "up",
                "down",
                "previous_frame_same_site",
                "next_frame_same_site",
            ],
            "radius": 1,
            "derivation": (
                "reduced from PP1 temporal_o8_prev5 to control KT sparsity; symmetric "
                "coarse-field neighbors are legal because the coarser field is fully decoded"
            ),
            "reference_form": False,
        },
        "denominators": {
            "field_sites": FIELD_BYTES,
            "support_symbols": int(symbols.size),
            "target_one_symbols": int(symbols.sum()),
            "target_fraction_within_support": float(symbols.mean()),
            "occupied_contexts": occupied,
            "possible_contexts": N_CONTEXTS,
        },
        "hindsight_bits": hindsight_bits,
        "hindsight_bytes": hindsight_bits / 8.0,
        "kt_bits": kt_bits,
        "kt_bytes": kt_bits / 8.0,
        "active_plane": file_fact(paths["plane"]),
        "active_plane_unpacked_bits": int(symbols.size),
        "sufficient_counts": file_fact(paths["counts"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["receipt"], receipt)
    progress(
        stage="screen",
        event="measured",
        state=state.state_id,
        kt_bytes=receipt["kt_bytes"],
        support=receipt["denominators"]["support_symbols"],
        elapsed_seconds=receipt["elapsed_seconds"],
    )
    return receipt


def kendall_distance(order: tuple[int, ...], reference: tuple[int, ...]) -> int:
    positions = {value: index for index, value in enumerate(reference)}
    mapped = [positions[value] for value in order]
    return sum(
        mapped[left] > mapped[right]
        for left in range(len(mapped))
        for right in range(left + 1, len(mapped))
    )


def all_chains() -> list[tuple[int, ...]]:
    chains = list(itertools.permutations(PEEL_CLASSES))
    chains.sort(key=lambda order: (kendall_distance(order, PRIOR_PEEL_ORDER), order))
    if len(chains) != 24 or len(set(chains)) != 24:
        raise D3CError("Road-rooted peel enumeration did not produce exactly 24 chains")
    return chains


def states_for_chains(chains: Iterable[tuple[int, ...]]) -> list[RungState]:
    unique: dict[tuple[tuple[int, ...], int], RungState] = {}
    for peel_order in chains:
        decoded: tuple[int, ...] = ()
        for target in reversed(peel_order):
            state = RungState(decoded=tuple(sorted(decoded)), target=target)
            unique[(state.decoded, target)] = state
            decoded = tuple(sorted((*decoded, target)))
    states = list(unique.values())
    states.sort(key=lambda state: (len(state.decoded), state.decoded, state.target))
    if len(states) != 32:
        raise D3CError(f"expected 32 unique rung states, found {len(states)}")
    return states


def chain_id(peel_order: tuple[int, ...]) -> str:
    return "peel_" + "-".join(CLASS_SLUGS[value] for value in peel_order)


def build_chain_row(
    peel_order: tuple[int, ...], receipts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    decoded: tuple[int, ...] = ()
    rungs = []
    total_kt = 0.0
    total_hindsight = 0.0
    for decode_index, target in enumerate(reversed(peel_order), start=1):
        state = RungState(decoded=tuple(sorted(decoded)), target=target)
        receipt = receipts[state.state_id]
        total_kt += receipt["kt_bytes"]
        total_hindsight += receipt["hindsight_bytes"]
        rungs.append(
            {
                "decode_rung": decode_index,
                "state_id": state.state_id,
                "decoded_before": [CLASS_NAMES[value] for value in state.decoded],
                "target": CLASS_NAMES[target],
                "support_symbols": receipt["denominators"]["support_symbols"],
                "target_one_symbols": receipt["denominators"]["target_one_symbols"],
                "occupied_contexts": receipt["denominators"]["occupied_contexts"],
                "hindsight_bytes": receipt["hindsight_bytes"],
                "kt_bytes": receipt["kt_bytes"],
                "receipt": receipt["active_plane"]["path"].replace(
                    "/planes/", "/receipts/"
                ).replace(".packbits", ".json"),
            }
        )
        decoded = tuple(sorted((*decoded, target)))
    return {
        "chain_id": chain_id(peel_order),
        "peel_order_fine_to_coarse": [CLASS_NAMES[value] for value in peel_order],
        "peel_order_class_ids": list(peel_order),
        "decode_order_coarse_to_fine": [CLASS_NAMES[value] for value in reversed(peel_order)],
        "implicit_base": "Road",
        "base_coarse_field_bound_bytes": 0.0,
        "hindsight_total_bytes": total_hindsight,
        "kt_total_bytes": total_kt,
        "delta_vs_127292_bytes": total_kt - JOINT_SUBSYSTEM_BYTES,
        "rungs": rungs,
    }


def completed_screen_if_current(store: Path) -> dict[str, Any] | None:
    path = store / "SCREEN_RESULT.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("schema") != "ddm_d3c_screen.v1"
        or result.get("source_field_sha256") != SOURCE_FIELD_SHA256
        or result.get("context_spec") != CONTEXT_SPEC
        or result.get("swept_chains") != 24
        or result.get("unique_rung_states") != 32
    ):
        return None
    state_receipts = result.get("state_receipts", [])
    if len(state_receipts) != 32:
        return None
    for receipt_fact in state_receipts:
        path = Path(receipt_fact["path"])
        if not path.is_file() or file_fact(path) != receipt_fact:
            return None
        receipt = json.loads(path.read_text(encoding="utf-8"))
        state = RungState(
            decoded=tuple(receipt.get("decoded_class_ids", [])),
            target=int(receipt.get("target_class_id", -1)),
        )
        if reusable_state_receipt(store, state) is None:
            return None
    return result


def stage_prepare(store: Path) -> dict[str, Any]:
    custody = verify_inputs(store)
    field = source_field()
    histogram = np.bincount(field.reshape(-1), minlength=5)
    if histogram.tolist() != [27_406_888, 691_095, 58_413_222, 1_460_458, 29_993_137]:
        raise D3CError(f"source class histogram drifted: {histogram.tolist()}")
    paths = []
    for value in PEEL_CLASSES:
        destination = store / "retained/source" / f"class_{CLASS_SLUGS[value]}.packbits"
        payload = np.packbits(np.asarray(field == value).reshape(-1), bitorder="big").tobytes()
        atomic_bytes(destination, payload)
        paths.append(
            {
                "class_id": value,
                "class_name": CLASS_NAMES[value],
                "ones": int(histogram[value]),
                "plane": file_fact(destination),
                "unpacked_bits": FIELD_BYTES,
            }
        )
    result = {
        "schema": "ddm_d3c_prepare.v1",
        "complete": True,
        "axis": SCREEN_AXIS,
        "score_claim": False,
        "custody": custody,
        "source_histogram": {
            CLASS_NAMES[index]: int(value) for index, value in enumerate(histogram)
        },
        "field_sites": FIELD_BYTES,
        "road_root_derivation": (
            "the charter specifies 24 chains over four peels; 4! fixes one base. Road is "
            "the D3/lm1 spatial-embedding host and is therefore the derived implicit base"
        ),
        "peel_classes": [CLASS_NAMES[value] for value in PEEL_CLASSES],
        "retained_source_planes": paths,
        "script": file_fact(Path(__file__)),
    }
    atomic_json(store / "PREPARE_RESULT.json", result)
    return result


def stage_screen(store: Path, resume: bool) -> dict[str, Any]:
    if resume:
        completed = completed_screen_if_current(store)
        if completed is not None:
            return completed
    prepare = stage_prepare(store)
    field = source_field()
    chains = all_chains()
    states = states_for_chains(chains)
    receipts: dict[str, dict[str, Any]] = {}
    for index, state in enumerate(states, start=1):
        receipts[state.state_id] = screen_state(store, field, state, resume)
        atomic_json(
            store / "SCREEN_CHECKPOINT.json",
            {
                "schema": "ddm_d3c_screen_checkpoint.v1",
                "source_field_sha256": SOURCE_FIELD_SHA256,
                "context_spec": CONTEXT_SPEC,
                "completed_states": index,
                "expected_states": len(states),
                "last_state": state.state_id,
            },
        )
    rows = [build_chain_row(chain, receipts) for chain in chains]
    rows.sort(key=lambda row: (row["kt_total_bytes"], row["chain_id"]))
    best = rows[0]
    worst = rows[-1]
    lane_single_state = RungState(
        decoded=tuple(sorted((UNDRIVABLE, MOVABLE, MYCAR))), target=LANE
    )
    lane_bound = receipts[lane_single_state.state_id]["kt_bytes"]
    d3_single_peel_bound = D3_FOUR_STREAM_BYTES + lane_bound
    prediction_order_span = (
        (worst["kt_total_bytes"] - best["kt_total_bytes"]) / best["kt_total_bytes"]
    )
    result = {
        "schema": "ddm_d3c_screen.v1",
        "complete": True,
        "axis": SCREEN_AXIS,
        "score_claim": False,
        "rank_only": True,
        "promotion_eligible": False,
        "source_field_sha256": SOURCE_FIELD_SHA256,
        "context_spec": CONTEXT_SPEC,
        "context_fidelity": receipts[states[0].state_id]["context_fidelity"],
        "root_class": "Road",
        "root_class_id": ROAD,
        "swept_chains": len(rows),
        "deferred_chains": 0,
        "unique_rung_states": len(states),
        "screen_order": "all 24 exhausted; initial ordering by distance from prior Lane-Movable-MyCar-Undrivable",
        "state_receipts": [file_fact(state_paths(store, state)["receipt"]) for state in states],
        "joint_subsystem_reference_bytes": JOINT_SUBSYSTEM_BYTES,
        "d3_single_peel_screen": {
            "four_symbol_real_stream_bytes": D3_FOUR_STREAM_BYTES,
            "lane_plane_kt_bound_bytes": lane_bound,
            "total_bytes": d3_single_peel_bound,
            "state_id": lane_single_state.state_id,
            "model_and_packet_framing_excluded": True,
        },
        "live_d3b_real_reference": prepare["custody"]["live_d3b_result"],
        "prior_law_tests": {
            "order_span_fraction_vs_best": prediction_order_span,
            "order_moves_at_least_10pct": prediction_order_span >= 0.10,
            "best_below_d3_single_peel_screen": best["kt_total_bytes"] < d3_single_peel_bound,
            "best_minus_d3_single_peel_screen_bytes": best["kt_total_bytes"]
            - d3_single_peel_bound,
        },
        "best_chain": best,
        "worst_chain": worst,
        "chains_ranked": rows,
    }
    atomic_json(store / "SCREEN_RESULT.json", result)
    return result


KT_EXTENSION = r'''

static uint32_t d3c_kt_one_frequency(uint32_t count_zero, uint32_t count_one) {
    uint64_t numerator = 2u * (uint64_t)count_one + 1u;
    uint64_t denominator = 2u * ((uint64_t)count_zero + (uint64_t)count_one) + 2u;
    uint64_t frequency = (numerator * RC64_TOTAL + denominator / 2u) / denominator;
    if (frequency < 1u) frequency = 1u;
    if (frequency >= RC64_TOTAL) frequency = RC64_TOTAL - 1u;
    return (uint32_t)frequency;
}

int d3c_encoder_encode_kt(
    void *opaque,
    const int32_t *symbols,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_zero,
    uint32_t *counts_one,
    size_t context_count
) {
    size_t index;
    if (!opaque || (!symbols && count) || (!context_keys && count) ||
        !counts_zero || !counts_one || !context_count) return -20;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        int status;
        if (key >= context_count || symbols[index] < 0 || symbols[index] > 1) return -21;
        row[1] = d3c_kt_one_frequency(counts_zero[key], counts_one[key]);
        row[0] = (uint32_t)(RC64_TOTAL - row[1]);
        status = rc64_encoder_encode(opaque, symbols + index, row, 1u);
        if (status) return status - 30;
        if (symbols[index]) counts_one[key]++;
        else counts_zero[key]++;
    }
    return 0;
}

int d3c_decoder_decode_kt(
    void *opaque,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_zero,
    uint32_t *counts_one,
    size_t context_count,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    if (!decoder || (!context_keys && count) || !counts_zero || !counts_one ||
        !context_count || (!symbols && count)) return -40;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        int status;
        if (key >= context_count) return -41;
        row[1] = d3c_kt_one_frequency(counts_zero[key], counts_one[key]);
        row[0] = (uint32_t)(RC64_TOTAL - row[1]);
        status = rc64_decoder_decode_row(decoder, row, symbols + index);
        if (status) return status - 50;
        if (symbols[index]) counts_one[key]++;
        else counts_zero[key]++;
    }
    return 0;
}
'''


def compile_rc64_binary(store: Path) -> tuple[Any, Path, dict[str, Any]]:
    route_b = jg2.load_route_b()
    route_b.ALPHABET = 2
    build = store / "retained/confirm/build/rc64_alphabet2_kt"
    build.mkdir(parents=True, exist_ok=True)
    base = jg2.resolve_rc64_base(route_b, build)
    source = base.read_text(encoding="utf-8")
    needle = "#define RC64_ALPHABET 5u"
    if source.count(needle) != 1:
        raise D3CError("pinned D3 RC64 base lost its unique alphabet macro")
    generated = build / "rc64_backend_alphabet2_kt.c"
    library = build / "librc64_alphabet2_kt.dylib"
    generated_payload = (
        source.replace(needle, "#define RC64_ALPHABET 2u")
        + "\n"
        + route_b.RC64_CHECKPOINT_EXTENSION
        + "\n"
        + KT_EXTENSION
    ).encode("utf-8")
    atomic_bytes(generated, generated_payload)
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        "-Wall",
        "-Wextra",
        str(generated),
        "-o",
        str(library),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stderr.strip():
        raise D3CError(f"RC64 compiler emitted warnings: {completed.stderr}")
    return route_b, library, {
        "mechanism": "D3 pinned RC64 source with alphabet macro changed from 5 to 2",
        "argv": command,
        "base_source": file_fact(base),
        "generated_source": file_fact(generated),
        "library": file_fact(library),
    }


def bind_kt(library: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    library.d3c_encoder_encode_kt.argtypes = [
        ctypes.c_void_p,
        i32p,
        u32p,
        ctypes.c_size_t,
        u32p,
        u32p,
        ctypes.c_size_t,
    ]
    library.d3c_encoder_encode_kt.restype = ctypes.c_int
    library.d3c_decoder_decode_kt.argtypes = [
        ctypes.c_void_p,
        u32p,
        ctypes.c_size_t,
        u32p,
        u32p,
        ctypes.c_size_t,
        i32p,
    ]
    library.d3c_decoder_decode_kt.restype = ctypes.c_int


def encode_kt_payload(route_b: Any, library: Path, keys: np.ndarray, symbols: np.ndarray) -> bytes:
    encoder = route_b.NativeRc64Encoder(library)
    bind_kt(encoder.library)
    counts_zero = np.zeros(N_CONTEXTS, dtype=np.uint32)
    counts_one = np.zeros(N_CONTEXTS, dtype=np.uint32)
    source = np.ascontiguousarray(symbols, dtype=np.int32)
    context = np.ascontiguousarray(keys, dtype=np.uint32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    status = encoder.library.d3c_encoder_encode_kt(
        encoder.context,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        context.ctypes.data_as(u32p),
        len(source),
        counts_zero.ctypes.data_as(u32p),
        counts_one.ctypes.data_as(u32p),
        N_CONTEXTS,
    )
    if status:
        raise D3CError(f"D3 RC64 KT encode failed with status {status}")
    return encoder.finish()


def decode_kt_payload(
    route_b: Any, library: Path, payload: bytes, keys: np.ndarray
) -> np.ndarray:
    decoder = route_b.NativeRc64Decoder(library, payload)
    bind_kt(decoder.library)
    counts_zero = np.zeros(N_CONTEXTS, dtype=np.uint32)
    counts_one = np.zeros(N_CONTEXTS, dtype=np.uint32)
    context = np.ascontiguousarray(keys, dtype=np.uint32)
    output = np.empty(len(context), dtype=np.int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    status = decoder.library.d3c_decoder_decode_kt(
        decoder.context,
        context.ctypes.data_as(u32p),
        len(context),
        counts_zero.ctypes.data_as(u32p),
        counts_one.ctypes.data_as(u32p),
        N_CONTEXTS,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status:
        raise D3CError(f"D3 RC64 KT decode failed with status {status}")
    return output


def confirm_paths(store: Path, decode_rung: int, state: RungState) -> dict[str, Path]:
    stem = f"rung_{decode_rung:02d}__{state.state_id}"
    root = store / "retained/confirm"
    return {
        "payload": root / "payloads" / f"{stem}.rc64",
        "repeat": root / "payloads" / f"{stem}.repeat.rc64",
        "decoded_active": root / "decoded_active" / f"{stem}.packbits",
        "receipt": root / "receipts" / f"{stem}.json",
    }


def reusable_confirm_rung(
    store: Path, decode_rung: int, state: RungState, chain: str
) -> dict[str, Any] | None:
    paths = confirm_paths(store, decode_rung, state)
    if not paths["receipt"].is_file():
        return None
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "ddm_d3c_confirm_rung.v2"
        or receipt.get("source_field_sha256") != SOURCE_FIELD_SHA256
        or receipt.get("context_spec") != CONTEXT_SPEC
        or receipt.get("chain_id") != chain
        or receipt.get("state_id") != state.state_id
    ):
        return None
    for key in ("payload", "determinism_repeat", "decoded_active_symbols"):
        fact = receipt.get(key)
        if not isinstance(fact, dict):
            return None
        path = Path(fact["path"])
        if not path.is_file() or file_fact(path) != fact:
            return None
    return receipt


def stage_confirm(store: Path, resume: bool) -> dict[str, Any]:
    screen = completed_screen_if_current(store)
    if screen is None:
        screen = stage_screen(store, resume)
    top = screen["best_chain"]
    chain = top["chain_id"]
    result_path = store / "CONFIRM_RESULT.json"
    if resume and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rungs = result.get("rungs", [])
        if (
            result.get("schema") == "ddm_d3c_confirm.v2"
            and result.get("complete")
            and result.get("chain_id") == chain
            and result.get("source_field_sha256") == SOURCE_FIELD_SHA256
            and result.get("context_spec") == CONTEXT_SPEC
            and len(rungs) == 4
            and all(
                reusable_confirm_rung(
                    store,
                    int(row["decode_rung"]),
                    RungState(
                        decoded=tuple(
                            sorted(CLASS_NAMES.index(name) for name in row["decoded_before"])
                        ),
                        target=CLASS_NAMES.index(row["target_class"]),
                    ),
                    chain,
                )
                is not None
                for row in rungs
            )
        ):
            return result
    route_b, library, build = compile_rc64_binary(store)
    field = source_field()
    peel_order = tuple(top["peel_order_class_ids"])
    decoded: tuple[int, ...] = ()
    rows = []
    started = time.perf_counter()
    for decode_rung, target in enumerate(reversed(peel_order), start=1):
        state = RungState(decoded=tuple(sorted(decoded)), target=target)
        if resume:
            reused = reusable_confirm_rung(store, decode_rung, state, chain)
            if reused is not None:
                rows.append(reused)
                decoded = tuple(sorted((*decoded, target)))
                continue
        keys, symbols, _ = build_state_arrays(field, state)
        payload = encode_kt_payload(route_b, library, keys, symbols)
        repeat = encode_kt_payload(route_b, library, keys, symbols)
        if payload != repeat:
            raise D3CError(f"real-coder determinism repeat differs for {state.state_id}")
        decoded_symbols = decode_kt_payload(route_b, library, payload, keys)
        if not np.array_equal(decoded_symbols, symbols.astype(np.int32)):
            mismatch = int(np.count_nonzero(decoded_symbols != symbols))
            raise D3CError(
                f"source-context replay round-trip differs for {state.state_id}: {mismatch}"
            )
        paths = confirm_paths(store, decode_rung, state)
        atomic_bytes(paths["payload"], payload)
        atomic_bytes(paths["repeat"], repeat)
        decoded_payload = np.packbits(decoded_symbols.astype(np.uint8), bitorder="big").tobytes()
        atomic_bytes(paths["decoded_active"], decoded_payload)
        screen_receipt = json.loads(
            state_paths(store, state)["receipt"].read_text(encoding="utf-8")
        )
        receipt = {
            "schema": "ddm_d3c_confirm_rung.v2",
            "complete": True,
            "axis": CONFIRM_AXIS,
            "score_claim": False,
            "chain_id": chain,
            "decode_rung": decode_rung,
            "state_id": state.state_id,
            "source_field_sha256": SOURCE_FIELD_SHA256,
            "context_spec": CONTEXT_SPEC,
            "target_class": CLASS_NAMES[target],
            "decoded_before": [CLASS_NAMES[value] for value in state.decoded],
            "support_symbols": int(symbols.size),
            "target_one_symbols": int(symbols.sum()),
            "screen_kt_bytes": screen_receipt["kt_bytes"],
            "payload": file_fact(paths["payload"]),
            "determinism_repeat": file_fact(paths["repeat"]),
            "decoded_active_symbols": file_fact(paths["decoded_active"]),
            "decoded_active_symbols_unpacked_bits": int(decoded_symbols.size),
            "decoded_active_symbols_order": (
                "C-order source sites restricted to the rung support; not a full spatial plane"
            ),
            "source_context_replay_roundtrip": True,
            "receiver_closed": False,
            "receiver_closure_blocker": (
                "context keys were replayed from the retained source/coarse field; the successor "
                "must regenerate them from prior decoded rungs and perform independent packet parse-back"
            ),
        }
        atomic_json(paths["receipt"], receipt)
        rows.append(receipt)
        decoded = tuple(sorted((*decoded, target)))
        atomic_json(
            store / "CONFIRM_CHECKPOINT.json",
            {
                "schema": "ddm_d3c_confirm_checkpoint.v1",
                "chain_id": chain,
                "completed_rungs": len(rows),
                "expected_rungs": 4,
                "last_state": state.state_id,
            },
        )
        progress(
            stage="confirm",
            event="real_coder_rung",
            state=state.state_id,
            payload_bytes=len(payload),
            screen_kt_bytes=screen_receipt["kt_bytes"],
        )
    total = sum(row["payload"]["bytes"] for row in rows)
    result = {
        "schema": "ddm_d3c_confirm.v2",
        "complete": True,
        "axis": CONFIRM_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "reference_form": False,
        "receiver_closed": False,
        "full_five_class_decode_identity_tested": False,
        "chain_id": chain,
        "source_field_sha256": SOURCE_FIELD_SHA256,
        "context_spec": CONTEXT_SPEC,
        "peel_order_fine_to_coarse": top["peel_order_fine_to_coarse"],
        "decode_order_coarse_to_fine": top["decode_order_coarse_to_fine"],
        "implicit_base": "Road",
        "screen_kt_total_bytes": top["kt_total_bytes"],
        "real_rc64_payload_total_bytes": total,
        "real_minus_screen_kt_bytes": total - top["kt_total_bytes"],
        "delta_vs_127292_bytes": total - JOINT_SUBSYSTEM_BYTES,
        "payload_framing": "each rung is an independent R6D1 RC64 payload including magic and u32 padding",
        "counted_context_parameters_bytes": 0,
        "rungs": rows,
        "rc64_build": build,
        "roundtrip_scope": (
            "all four real payloads decoded exactly under source-derived context replay; this is a "
            "coder control, not independent receiver closure or full packet decode identity"
        ),
        "successor_owed": [
            "reference-form HPAC/F26 plus derived geometry/temporal adaptive-mixer confirmation",
            "receiver-generated context keys from previously decoded rungs",
            "independent full-packet parse-back and exact five-class decode identity",
            "byte-closed composed archive if the reference row clears its bar",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(result_path, result)
    return result


def write_manifest(store: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(store.rglob("*")):
        if not path.is_file() or path.name.startswith("._") or path.name == "MANIFEST.json":
            continue
        entries.append(file_fact(path))
    manifest = {
        "schema": "ddm_d3c_manifest.v1",
        "source_field_sha256": SOURCE_FIELD_SHA256,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    atomic_json(store / "MANIFEST.json", manifest)
    return manifest


def self_test() -> None:
    global N_FRAMES, HEIGHT, WIDTH, PLANE, FIELD_BYTES
    original = (N_FRAMES, HEIGHT, WIDTH, PLANE, FIELD_BYTES)
    try:
        N_FRAMES, HEIGHT, WIDTH = 3, 3, 4
        PLANE = HEIGHT * WIDTH
        FIELD_BYTES = N_FRAMES * PLANE
        field = np.array(
            [
                [[0, 0, 1, 1], [0, 2, 2, 1], [4, 4, 3, 3]],
                [[0, 1, 1, 1], [0, 2, 3, 3], [4, 4, 4, 3]],
                [[0, 0, 1, 2], [0, 2, 2, 3], [4, 4, 3, 3]],
            ],
            dtype=np.uint8,
        )
        state = RungState(decoded=(UNDRIVABLE,), target=LANE)
        keys, symbols, support = build_state_arrays(field, state)
        assert len(keys) == int(support.sum()) == len(symbols)
        assert int(keys.max()) < N_CONTEXTS
        joint = keys.astype(np.int64) * 2 + symbols
        counts = np.bincount(joint, minlength=N_CONTEXTS * 2).reshape(N_CONTEXTS, 2)
        hindsight, kt, occupied = exact_binary_lengths(counts)
        assert hindsight >= 0.0 and kt > 0.0 and occupied > 0
        chains = all_chains()
        assert len(chains) == 24 and len(states_for_chains(chains)) == 32
    finally:
        N_FRAMES, HEIGHT, WIDTH, PLANE, FIELD_BYTES = original


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "screen", "confirm", "all", "self-test"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only sha-verified completed stage/rung receipts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "self-test":
        self_test()
        print("ddm_d3c self-test: PASS")
        return 0
    store = args.store.resolve()
    if args.stage == "prepare":
        stage_prepare(store)
    elif args.stage == "screen":
        stage_screen(store, args.resume)
    elif args.stage == "confirm":
        stage_confirm(store, args.resume)
    else:
        stage_screen(store, args.resume)
        stage_confirm(store, args.resume)
    manifest = write_manifest(store)
    progress(stage=args.stage, event="complete", manifest=file_fact(store / "MANIFEST.json"), files=manifest["file_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
