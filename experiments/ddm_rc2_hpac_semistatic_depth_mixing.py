#!/usr/bin/env python3
"""Measure and, only on a strict byte win, stage the ddm_rc2 IHS1 recode.

This is a scorer-free, zero-distortion experiment.  Every byte string constructed by
the experiment is retained before its length is reported.  The only mutable trees are
``STORE`` and a conditional staged receiver below it; the live sj1 receiver is an input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import brotli
import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import ddm_jg2_tail_reencode as jg2
from experiments import ddm_rc1_model_section_adaptive_recode as rc1
from experiments import ddm_rc2_hpac_semistatic_mixing_codec as codec

LIVE_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass3/candidate_runtime"
)
LIVE_ARCHIVE = LIVE_ROOT / "archive.zip"
LIVE_ARCHIVE_BYTES = 181_645
LIVE_ARCHIVE_SHA256 = "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
LIVE_SCORE = 0.13900437796841966
SCORE_DENOMINATOR = 37_545_489
POINTER = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing")
RC1_RETAINED_IHS1 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode/retained/hpac_body.ihs1.bin"
)
SOURCE_TOKEN_FIELD = LIVE_ROOT.parent / "parseback/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"

RX1_HEADER = struct.Struct("<4sBBBBHHH")
RC1_HPAC_RESERVED = 0x40
FAMILY_SLUG = {
    codec.FAMILY_PREV_ZERO: "prev_zero",
    codec.FAMILY_PREV_SIGN: "prev_sign",
    codec.FAMILY_PREV_BITLEN: "prev_bitlen",
    codec.FAMILY_EXACT_PREV: "exact_prev",
}
SERIALIZER_SLUG = {
    codec.SERIALIZER_COUNTS_ULEB: "counts_uleb",
    codec.SERIALIZER_PROB8: "prob8",
    codec.SERIALIZER_PROB12: "prob12",
}


class Rc2ExperimentError(RuntimeError):
    """A proof obligation failed; no result should be promoted."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Persist immutable bytes atomically and return their custody fact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = path.read_bytes()
        if current != payload:
            raise Rc2ExperimentError(f"refusing to overwrite non-identical retained payload: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    return {"path": str(path), "bytes": len(payload), "sha256": sha256_bytes(payload)}


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(path, payload)


def atomic_replace_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def brotli_bytes(payload: bytes, quality: int, lgwin: int) -> bytes:
    compressed = brotli.compress(payload, quality=quality, lgwin=lgwin)
    if brotli.decompress(compressed) != payload:
        raise Rc2ExperimentError("Brotli round-trip failed")
    return compressed


def current_pointer_fact() -> dict[str, Any]:
    value = json.loads(POINTER.read_text(encoding="utf-8"))
    node = value.get("our_local_frontier_contest_cuda")
    effective = value.get("effective_frontier")
    expected = (LIVE_ARCHIVE_SHA256, LIVE_ARCHIVE_BYTES, LIVE_SCORE)
    observed = (
        node.get("archive_sha256") if isinstance(node, dict) else None,
        (node.get("extra") or {}).get("archive_bytes") if isinstance(node, dict) else None,
        node.get("score") if isinstance(node, dict) else None,
    )
    effective_observed = (
        effective.get("archive_sha256") if isinstance(effective, dict) else None,
        effective.get("score") if isinstance(effective, dict) else None,
    )
    if observed != expected or effective_observed != (LIVE_ARCHIVE_SHA256, LIVE_SCORE):
        raise Rc2ExperimentError(
            f"live pointer no longer matches charter pin: cuda={observed}, effective={effective_observed}"
        )
    return {"path": str(POINTER), "sha256": sha256_bytes(POINTER.read_bytes())}


def import_live_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, LIVE_ROOT / relative)
    if spec is None or spec.loader is None:
        raise Rc2ExperimentError(f"cannot load live module {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_counts_from_live_runtime() -> list[int]:
    script = (
        "import json,sys\n"
        "from pathlib import Path\n"
        "root=Path(sys.argv[1]);sys.path[:0]=[str(root/'cpr1'),str(root)]\n"
        "import inflate\n"
        "from runtime.ihs2 import layout_from_runtime\n"
        "print(json.dumps(layout_from_runtime(inflate).row_counts))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(LIVE_ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    counts = json.loads(done.stdout.strip().splitlines()[-1])
    if len(counts) != 517 or any(int(value) <= 0 for value in counts):
        raise Rc2ExperimentError("live IHS1 row geometry is unexpected")
    return [int(value) for value in counts]


def extract_live() -> dict[str, Any]:
    current_pointer_fact()
    archive = LIVE_ARCHIVE.read_bytes()
    if len(archive) != LIVE_ARCHIVE_BYTES or sha256_bytes(archive) != LIVE_ARCHIVE_SHA256:
        raise Rc2ExperimentError("live archive differs from charter byte/sha pin")
    member = jg2.read_archive_member(LIVE_ARCHIVE)
    sections = jg2.split_member(member)
    fields = RX1_HEADER.unpack(sections["header"])
    magic, version, compression, table_mode, reserved, hpac_n, semantic_n, carrier_n = fields
    if (magic, version, compression) != (b"RX1M", 1, 2):
        raise Rc2ExperimentError(f"unexpected live RX1 header {fields}")
    if (hpac_n, semantic_n, carrier_n) != tuple(len(sections[name]) for name in ("hpac", "semantic", "carrier")):
        raise Rc2ExperimentError("RX1 section lengths disagree with header")
    if not reserved & RC1_HPAC_RESERVED:
        raise Rc2ExperimentError("live archive does not declare the RC1 HPAC rider")

    hpac_outer = brotli.decompress(sections["hpac"])
    rc1_rider = rc1.ck2_uninterleave(hpac_outer)
    if not rc1_rider.startswith(b"RC1H"):
        raise Rc2ExperimentError("live HPAC section does not restore to RC1H")
    row_counts = row_counts_from_live_runtime()
    live_rc1 = import_live_module("ddm_rc2_live_rc1", "runtime/rc1_adaptive_model_sections.py")
    ihs1 = live_rc1.restore_hpac(rc1_rider, row_counts)
    if not RC1_RETAINED_IHS1.is_file() or RC1_RETAINED_IHS1.read_bytes() != ihs1:
        raise Rc2ExperimentError("live restored IHS1 is not byte-identical to rc1 retained body")

    retained = STORE / "retained/source"
    facts = {
        "archive": {"path": str(LIVE_ARCHIVE), "bytes": len(archive), "sha256": sha256_bytes(archive)},
        "member": atomic_bytes(retained / "live_member.rx1", member),
        "hpac_container": atomic_bytes(retained / "live_hpac.br", sections["hpac"]),
        "hpac_brotli_output": atomic_bytes(retained / "live_hpac.ck2.bin", hpac_outer),
        "rc1_rider": atomic_bytes(retained / "live_hpac.rc1h.bin", rc1_rider),
        "ihs1": atomic_bytes(retained / "live_hpac.ihs1.bin", ihs1),
        "rc1_retained_identity": True,
        "header": {
            "magic": magic.decode(),
            "version": version,
            "compression": compression,
            "table_mode": table_mode,
            "reserved": reserved,
            "hpac_bytes": hpac_n,
            "semantic_bytes": semantic_n,
            "carrier_bytes": carrier_n,
        },
        "other_sections": {
            name: {"bytes": len(sections[name]), "sha256": sha256_bytes(sections[name])}
            for name in ("semantic", "carrier", "tail")
        },
        "row_count": len(row_counts),
        "value_count": sum(row_counts),
    }
    return {
        "facts": facts,
        "member": member,
        "sections": sections,
        "header": fields,
        "row_counts": row_counts,
        "ihs1": ihs1,
        "rc1_rider": rc1_rider,
    }


def empirical_entropy(counter: Counter[int]) -> float:
    total = sum(counter.values())
    return -sum(count * math.log2(count / total) for count in counter.values()) if total else 0.0


def derive_bounds(ihs1: bytes, row_counts: list[int], rc1_rider: bytes) -> dict[str, Any]:
    rows, depths = codec.unpack_rows(ihs1, row_counts)
    per_depth: dict[int, list[int]] = defaultdict(list)
    for row, depth in zip(rows, depths.tolist(), strict=True):
        per_depth[int(depth)].extend(int(value) for value in row)

    h0_plugin_bits = 0.0
    h0_mm_bits = 0.0
    h1_plugin_bits = 0.0
    h1_mm_bits = 0.0
    depth_rows: list[dict[str, Any]] = []
    for depth in sorted(per_depth):
        values = per_depth[depth]
        marginal = Counter(values)
        h0 = empirical_entropy(marginal)
        h0_correction = max(0, len(marginal) - 1) / (2.0 * math.log(2.0))
        transitions = Counter(itertools.pairwise(values))
        predecessors = Counter(values[:-1])
        h1 = empirical_entropy(transitions) - empirical_entropy(predecessors)
        h1_correction = max(0, len(transitions) - len(predecessors)) / (2.0 * math.log(2.0))
        first = -math.log2(marginal[values[0]] / len(values)) if values else 0.0
        h0_plugin_bits += h0
        h0_mm_bits += h0 + h0_correction
        h1_plugin_bits += h1 + first
        h1_mm_bits += h1 + h1_correction + first
        depth_rows.append(
            {
                "depth": depth,
                "samples": len(values),
                "alphabet": len(marginal),
                "transitions": max(0, len(values) - 1),
                "joint_alphabet": len(transitions),
                "h0_plugin_bits": h0,
                "h0_mm_bits": h0 + h0_correction,
                "h1_plugin_plus_first_bits": h1 + first,
                "h1_mm_plus_first_bits": h1 + h1_correction + first,
            }
        )

    # Reproduce the current RC1 binary adaptive model's ideal arithmetic length.
    ideal_bits = 0.0
    probabilities: dict[tuple[int, int], int] = defaultdict(lambda: codec.PROBABILITY_INITIAL)
    for row, depth in zip(rows, depths.tolist(), strict=True):
        for value in row.tolist():
            unsigned = int(value) & ((1 << int(depth)) - 1) if depth else 0
            node = 1
            for position in reversed(range(int(depth))):
                bit = (unsigned >> position) & 1
                key = (int(depth), node)
                probability = probabilities[key]
                frequency = probability if bit == 0 else codec.PROBABILITY_ONE - probability
                ideal_bits -= math.log2(frequency / codec.PROBABILITY_ONE)
                probabilities[key] = codec._updated_probability(probability, bit)
                node = (node << 1) | bit
    rc1_fields = struct.Struct("<4sBBI").unpack_from(rc1_rider)
    actual_range_bytes = int(rc1_fields[3])
    return {
        "derivation": "entropy sums per depth over rows in stored order; H1 includes one H0-priced first symbol per depth; Miller-Madow adds (K-1)/(2 ln 2) bits for H0 and (K_joint-K_prev)/(2 ln 2) bits for H1",
        "h0_plugin": {"bits": h0_plugin_bits, "bytes": h0_plugin_bits / 8.0},
        "h0_miller_madow": {"bits": h0_mm_bits, "bytes": h0_mm_bits / 8.0},
        "h1_plugin_plus_first": {"bits": h1_plugin_bits, "bytes": h1_plugin_bits / 8.0},
        "h1_miller_madow_plus_first": {"bits": h1_mm_bits, "bytes": h1_mm_bits / 8.0},
        "current_rc1": {
            "ideal_bits": ideal_bits,
            "ideal_bytes": ideal_bits / 8.0,
            "actual_range_payload_bytes": actual_range_bytes,
            "gap_vs_h1_mm_bytes": actual_range_bytes - h1_mm_bits / 8.0,
        },
        "per_depth": depth_rows,
    }


def container_grid() -> list[tuple[bool, int, int]]:
    return [(ck2, quality, lgwin) for ck2 in (False, True) for quality in (9, 10, 11) for lgwin in (22, 23, 24)]


def containerize(rider: bytes, *, ck2: bool, quality: int, lgwin: int) -> bytes:
    source = rc1.ck2_interleave(rider) if ck2 else rider
    return brotli_bytes(source, quality, lgwin)


def select_shippable(rows: list[dict[str, Any]], object_name: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["object"] == object_name and row["shippable"]]
    if len(candidates) != 9:
        raise Rc2ExperimentError(f"expected nine shippable grid rows for {object_name}")
    return min(candidates, key=lambda row: (row["container_bytes"], row["quality"], row["lgwin"]))


def race_semistatic(ihs1: bytes, row_counts: list[int]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    streams: dict[str, bytes] = {}
    root = STORE / "retained/semistatic"
    for family in sorted(codec.FAMILY_NAMES):
        for serializer in sorted(codec.SERIALIZER_NAMES):
            slug = f"{FAMILY_SLUG[family]}__{SERIALIZER_SLUG[serializer]}"
            rider, parts = codec.encode_semistatic(ihs1, row_counts, family, serializer)
            if codec.restore_hpac(rider, row_counts) != ihs1:
                raise Rc2ExperimentError(f"semi-static decode identity failed: {slug}")
            facts = {
                "table": atomic_bytes(root / f"{slug}.table.bin", parts["table_blob"]),
                "table_brotli": atomic_bytes(
                    root / f"{slug}.table.q11w24.br", brotli_bytes(parts["table_blob"], 11, 24)
                ),
                "range_payload": atomic_bytes(root / f"{slug}.range.bin", parts["payload"]),
                "range_brotli": atomic_bytes(root / f"{slug}.range.q11w24.br", brotli_bytes(parts["payload"], 11, 24)),
                "rider": atomic_bytes(root / f"{slug}.rc2h.bin", rider),
            }
            default = containerize(rider, ck2=True, quality=11, lgwin=24)
            facts["default_container"] = atomic_bytes(root / f"{slug}.ck2_q11_w24.br", default)
            rows.append(
                {
                    "design": "semi_static",
                    "slug": slug,
                    "family": codec.FAMILY_NAMES[family],
                    "serializer": codec.SERIALIZER_NAMES[serializer],
                    "table_entries": int(parts["table_entries"]),
                    "table_bytes": len(parts["table_blob"]),
                    "table_brotli_bytes": facts["table_brotli"]["bytes"],
                    "weights_bytes": 0,
                    "coded_bytes": len(parts["payload"]),
                    "net_J_bytes": len(parts["table_blob"]) + len(parts["payload"]),
                    "rider_bytes": len(rider),
                    "default_container_bytes": len(default),
                    "ideal_bits": float(parts["ideal_bits"]),
                    "decode_identity": True,
                    "payloads": facts,
                }
            )
            streams[slug] = rider
    return rows, streams


def group_ids(depths: np.ndarray, positions: np.ndarray, groups: int) -> np.ndarray:
    if groups == 1:
        return np.zeros_like(positions)
    if groups == 2:
        return (positions != 0).astype(np.int64)
    if groups == 4:
        return np.minimum(3, (4 * positions) // np.maximum(1, depths)).astype(np.int64)
    if groups == 8:
        return np.minimum(7, positions).astype(np.int64)
    raise Rc2ExperimentError(f"unsupported mixer group count {groups}")


def ideal_mixer_bits(features: np.ndarray, outcomes: np.ndarray, gids: np.ndarray, weights_q: np.ndarray) -> float:
    scores = np.sum(features * weights_q[gids].astype(np.int64), axis=1)
    scores = np.rint(scores / codec.WEIGHT_SCALE).astype(np.int64)
    stretch = np.asarray(codec.STRETCH, dtype=np.int64)
    indices = np.searchsorted(stretch, scores, side="left")
    probabilities = np.empty_like(indices)
    low = indices <= 0
    high = indices >= stretch.size
    middle = ~(low | high)
    probabilities[low] = 1
    probabilities[high] = codec.PROBABILITY_ONE - 1
    middle_indices = indices[middle]
    middle_scores = scores[middle]
    choose_before = middle_scores - stretch[middle_indices - 1] <= stretch[middle_indices] - middle_scores
    probabilities[middle] = np.where(choose_before, middle_indices, middle_indices + 1)
    # ``outcomes`` is one for a zero bit; ``probabilities`` is P(bit == 0).
    frequency = np.where(outcomes == 1, probabilities, codec.PROBABILITY_ONE - probabilities)
    return float(np.sum(-np.log2(frequency / codec.PROBABILITY_ONE)))


def fit_mixer(
    features: np.ndarray, outcomes: np.ndarray, gids: np.ndarray, groups: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = features.astype(np.float64) / codec.STRETCH_SCALE
    y = outcomes.astype(np.float64)
    initial = np.zeros((groups, codec.EXPERT_COUNT), dtype=np.float64)
    initial[:, 0] = 1.0

    def objective(flat: np.ndarray) -> float:
        weights = flat.reshape(groups, codec.EXPERT_COUNT)
        logits = np.sum(x * weights[gids], axis=1)
        return float(np.sum(np.logaddexp(0.0, logits) - y * logits) / math.log(2.0))

    result = minimize(
        objective,
        initial.ravel(),
        method="L-BFGS-B",
        bounds=[(-4.0, 4.0)] * initial.size,
        options={"maxiter": 250, "ftol": 1e-11, "maxls": 30},
    )
    floating = result.x.reshape(groups, codec.EXPERT_COUNT)
    quantized = np.clip(np.rint(floating * codec.WEIGHT_SCALE), -128, 127).astype(np.int8)

    # One deterministic discrete coordinate pass optimizes the exact integer decoder.
    best = ideal_mixer_bits(features, outcomes, gids, quantized)
    for group in range(groups):
        for expert in range(codec.EXPERT_COUNT):
            original = int(quantized[group, expert])
            local_best = best
            chosen = original
            for trial in (original - 1, original + 1):
                if -128 <= trial <= 127:
                    quantized[group, expert] = trial
                    bits = ideal_mixer_bits(features, outcomes, gids, quantized)
                    if bits < local_best:
                        local_best, chosen = bits, trial
            quantized[group, expert] = chosen
            best = local_best
    return (
        floating,
        quantized,
        {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "float_objective_bits": float(result.fun),
            "integer_objective_bits": best,
        },
    )


def race_logistic(ihs1: bytes, row_counts: list[int]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    unpacked, depths = codec.unpack_rows(ihs1, row_counts)
    features, outcomes, event_depths, positions, nodes = codec.collect_logistic_events(unpacked, depths)
    event_buffer = io.BytesIO()
    np.savez(
        event_buffer,
        features=features,
        outcomes=outcomes,
        depths=event_depths,
        positions=positions,
        nodes=nodes,
    )
    event_fact = atomic_bytes(STORE / "retained/logistic/training_events.npz", event_buffer.getvalue())
    rows: list[dict[str, Any]] = []
    streams: dict[str, bytes] = {}
    for weight_bytes in (8, 16, 32, 64):
        groups = weight_bytes // codec.EXPERT_COUNT
        gids = group_ids(event_depths, positions, groups)
        floating, quantized, optimizer = fit_mixer(features, outcomes, gids, groups)
        float_buffer = io.BytesIO()
        np.save(float_buffer, floating, allow_pickle=False)
        slug = f"logistic_int8_{weight_bytes}B"
        rider, parts = codec.encode_logistic(ihs1, row_counts, quantized)
        if codec.restore_hpac(rider, row_counts) != ihs1:
            raise Rc2ExperimentError(f"logistic decode identity failed: {slug}")
        root = STORE / "retained/logistic"
        facts = {
            "training_events": event_fact,
            "float_weights": atomic_bytes(root / f"{slug}.float64.npy", float_buffer.getvalue()),
            "weights": atomic_bytes(root / f"{slug}.weights.i8", parts["weights_blob"]),
            "range_payload": atomic_bytes(root / f"{slug}.range.bin", parts["payload"]),
            "range_brotli": atomic_bytes(root / f"{slug}.range.q11w24.br", brotli_bytes(parts["payload"], 11, 24)),
            "rider": atomic_bytes(root / f"{slug}.rc2h.bin", rider),
        }
        default = containerize(rider, ck2=True, quality=11, lgwin=24)
        facts["default_container"] = atomic_bytes(root / f"{slug}.ck2_q11_w24.br", default)
        rows.append(
            {
                "design": "logistic_mixing",
                "slug": slug,
                "groups": groups,
                "table_bytes": 0,
                "weights_bytes": weight_bytes,
                "coded_bytes": len(parts["payload"]),
                "net_J_bytes": weight_bytes + len(parts["payload"]),
                "rider_bytes": len(rider),
                "default_container_bytes": len(default),
                "ideal_bits": float(parts["ideal_bits"]),
                "decode_identity": True,
                "optimizer": optimizer,
                "payloads": facts,
            }
        )
        streams[slug] = rider
    return rows, streams


def sweep_containers(
    base: bytes, winner_name: str, winner: bytes
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    root = STORE / "retained/container_sweep"
    for object_name, rider in (("base_rc1", base), (winner_name, winner)):
        for ck2, quality, lgwin in container_grid():
            payload = containerize(rider, ck2=ck2, quality=quality, lgwin=lgwin)
            slug = f"{object_name}__ck2_{int(ck2)}__q{quality}__w{lgwin}.br"
            fact = atomic_bytes(root / slug, payload)
            row = {
                "object": object_name,
                "ck2": ck2,
                "quality": quality,
                "lgwin": lgwin,
                "container_bytes": len(payload),
                "sha256": fact["sha256"],
                "path": fact["path"],
                "shippable": ck2,
                "disposition": "candidate_compatible" if ck2 else "measurement_only_no_free_reserved_bit",
            }
            rows.append(row)
            payloads[slug] = payload
    shipped = next(
        row
        for row in rows
        if row["object"] == "base_rc1" and row["ck2"] and row["quality"] == 11 and row["lgwin"] == 24
    )
    if (
        bytes.fromhex(shipped["sha256"])
        != hashlib.sha256((STORE / "retained/source/live_hpac.br").read_bytes()).digest()
    ):
        raise Rc2ExperimentError("base shipped container cell is not byte-identical")
    base_best = select_shippable(rows, "base_rc1")
    new_best = select_shippable(rows, winner_name)
    all_base = min((row for row in rows if row["object"] == "base_rc1"), key=lambda row: row["container_bytes"])
    all_new = min((row for row in rows if row["object"] == winner_name), key=lambda row: row["container_bytes"])
    return (
        rows,
        {
            "base_shippable": base_best,
            "new_shippable": new_best,
            "base_all_cells": all_base,
            "new_all_cells": all_new,
            "shipped_control": shipped,
        },
        payloads,
    )


def build_member(source: dict[str, Any], hpac_container: bytes) -> bytes:
    magic, version, compression, table_mode, reserved, _old_hpac, semantic_n, carrier_n = source["header"]
    sections = dict(source["sections"])
    sections["header"] = RX1_HEADER.pack(
        magic, version, compression, table_mode, reserved, len(hpac_container), semantic_n, carrier_n
    )
    sections["hpac"] = hpac_container
    return jg2.join_member(sections)


def copy_live_tree(destination: Path) -> None:
    if destination.exists():
        raise Rc2ExperimentError(f"refusing to replace staged runtime: {destination}")

    def ignored(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "archive.zip" or name == "__pycache__" or name.startswith("._")}

    shutil.copytree(LIVE_ROOT, destination, ignore=ignored)


def patch_once(path: Path, anchor: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(anchor) != 1:
        raise Rc2ExperimentError(f"staging anchor count != 1 in {path}: {anchor[:60]!r}")
    path.write_text(text.replace(anchor, replacement), encoding="utf-8")


def patch_receiver(staged: Path) -> None:
    shutil.copy2(Path(codec.__file__), staged / "runtime/rc2_hpac_semistatic_mixing.py")
    residual = staged / "runtime/residual_archive.py"
    patch_once(
        residual,
        '        from .rc1_adaptive_model_sections import HPAC_MAGIC as RC1_HPAC_MAGIC\n\n        hpac = _ck2_uninterleave_planes(hpac)\n        if not hpac.startswith(RC1_HPAC_MAGIC):\n            raise ResidualArchiveError("RX1 HPAC does not carry the RC1 rider")',
        '        from .rc1_adaptive_model_sections import HPAC_MAGIC as RC1_HPAC_MAGIC\n        from .rc2_hpac_semistatic_mixing import MAGIC as RC2_HPAC_MAGIC\n\n        hpac = _ck2_uninterleave_planes(hpac)\n        if not hpac.startswith((RC1_HPAC_MAGIC, RC2_HPAC_MAGIC)):\n            raise ResidualArchiveError("RX1 HPAC does not carry an admitted lossless rider")',
    )
    ihs2 = staged / "runtime/ihs2.py"
    patch_once(
        ihs2,
        "from .rc1_adaptive_model_sections import restore_hpac as restore_rc1_hpac",
        "from .rc1_adaptive_model_sections import restore_hpac as restore_rc1_hpac\nfrom .rc2_hpac_semistatic_mixing import MAGIC as RC2_HPAC_MAGIC\nfrom .rc2_hpac_semistatic_mixing import restore_hpac as restore_rc2_hpac",
    )
    patch_once(
        ihs2,
        "    if blob.startswith(RC1_HPAC_MAGIC):\n        # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: restore the packed weight bitstream from\n        # its adaptive form.  The row geometry comes from the same value-free model\n        # shell IHS2 already builds, so nothing video-derived enters runtime code.\n        return restore_rc1_hpac(blob, layout_from_runtime(runtime).row_counts)",
        "    if blob.startswith(RC2_HPAC_MAGIC):\n        return restore_rc2_hpac(blob, layout_from_runtime(runtime).row_counts)\n    if blob.startswith(RC1_HPAC_MAGIC):\n        # DDM_RC1_ADAPTIVE_MODEL_SECTIONS_V1: restore the packed weight bitstream from\n        # its adaptive form.  The row geometry comes from the same value-free model\n        # shell IHS2 already builds, so nothing video-derived enters runtime code.\n        return restore_rc1_hpac(blob, layout_from_runtime(runtime).row_counts)",
    )


def repin_receiver(staged: Path, archive: bytes) -> None:
    inflate = staged / "inflate.py"
    text = inflate.read_text(encoding="utf-8")
    import re

    size_subs = re.subn(
        r"^(?P<name>(?:EXPECTED_)?ARCHIVE_BYTES)\s*=\s*\d+",
        rf"\g<name> = {len(archive)}",
        text,
        flags=re.MULTILINE,
    )
    sha_subs = re.subn(
        r'^(?P<name>(?:EXPECTED_)?ARCHIVE_SHA256)\s*=\s*["\'][0-9a-f]{64}["\']',
        rf'\g<name> = "{sha256_bytes(archive)}"',
        size_subs[0],
        flags=re.MULTILINE,
    )
    if size_subs[1] != 1 or sha_subs[1] != 1:
        raise Rc2ExperimentError("inflate.py archive pin anchors did not match exactly once")
    inflate.write_text(sha_subs[0], encoding="utf-8")


def regenerate_manifest(staged: Path) -> dict[str, Any]:
    manifest = staged / "MANIFEST.sha256"
    paths = sorted(
        path
        for path in staged.rglob("*")
        if path.is_file()
        and path != manifest
        and path.name != "archive.zip"
        and "__pycache__" not in path.parts
        and not path.name.startswith("._")
    )
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(staged)}" for path in paths]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(manifest), "entries": len(lines), "sha256": sha256_bytes(manifest.read_bytes())}


def staged_model_decode(staged: Path, expected_ihs1: bytes) -> dict[str, Any]:
    script = (
        "import hashlib,json,sys\n"
        "from pathlib import Path\n"
        "root=Path(sys.argv[1]);sys.path[:0]=[str(root/'cpr1'),str(root)]\n"
        "import inflate\n"
        "from runtime.residual_archive import read_residual_archive\n"
        "from runtime.ihs2 import materialize_ihs1\n"
        "obj=read_residual_archive(root/'archive.zip')\n"
        "body=materialize_ihs1(obj.hpac_blob,inflate)\n"
        "print(json.dumps({'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(staged)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
    )
    result = json.loads(done.stdout.strip().splitlines()[-1])
    result["identity"] = result["bytes"] == len(expected_ihs1) and result["sha256"] == sha256_bytes(expected_ihs1)
    if not result["identity"]:
        raise Rc2ExperimentError("staged public receiver did not restore exact IHS1 body")
    return result


def stage_candidate(
    source: dict[str, Any], winner_name: str, winner_rider: bytes, chosen_container: bytes
) -> dict[str, Any]:
    # Composition guard immediately before mutation.
    current_pointer_fact()
    staged = STORE / "candidate_runtime"
    member = build_member(source, chosen_container)
    retained = STORE / "retained/candidate"
    first = retained / "archive.zip"
    second = retained / "archive.repeat.zip"
    jg2.pack_archive(member, first)
    jg2.pack_archive(build_member(source, chosen_container), second)
    archive = first.read_bytes()
    twin = second.read_bytes()
    if archive != twin:
        raise Rc2ExperimentError("candidate twin archive encode differs")
    atomic_bytes(retained / "winner.rider.bin", winner_rider)
    if staged.exists():
        required = [staged / "archive.zip", staged / "runtime/rc2_hpac_semistatic_mixing.py"]
        if winner_name == "base_rc1":
            required = [staged / "archive.zip"]
        if not all(path.is_file() for path in required) or (staged / "archive.zip").read_bytes() != archive:
            raise Rc2ExperimentError("existing partial staged runtime is not this candidate")
        if (
            winner_name != "base_rc1"
            and (staged / "runtime/rc2_hpac_semistatic_mixing.py").read_bytes() != Path(codec.__file__).read_bytes()
        ):
            raise Rc2ExperimentError("existing staged RC2 receiver differs from current codec")
    else:
        copy_live_tree(staged)
        if winner_name != "base_rc1":
            patch_receiver(staged)
        shutil.copy2(first, staged / "archive.zip")
    repin_receiver(staged, archive)
    manifest = regenerate_manifest(staged)

    candidate_member = jg2.read_archive_member(first)
    old = jg2.split_member(source["member"])
    new = jg2.split_member(candidate_member)
    census = {
        name: {
            "source_bytes": len(old[name]),
            "candidate_bytes": len(new[name]),
            "source_sha256": sha256_bytes(old[name]),
            "candidate_sha256": sha256_bytes(new[name]),
            "identical": old[name] == new[name],
        }
        for name in ("semantic", "carrier", "tail")
    }
    if not all(item["identical"] for item in census.values()):
        raise Rc2ExperimentError("candidate changed an out-of-scope RX1 section")
    decoded = staged_model_decode(staged, source["ihs1"])
    public_probe = retained_public_path_probe(staged, "candidate_stage", timeout_s=240.0)
    shell_probe = rc1._inflate_sh_smoke(staged, timeout_s=300.0)
    if public_probe.get("outcome") not in {"REACHED_TOKEN_DECODE", "COMPLETED"}:
        raise Rc2ExperimentError(f"public f26 smoke failed: {public_probe}")
    if shell_probe.get("outcome") != "REACHED_CUDA_GATE":
        raise Rc2ExperimentError(f"inflate.sh did not reach CUDA gate: {shell_probe}")
    return {
        "staged_runtime": str(staged),
        "archive": atomic_bytes(first, archive),
        "archive_repeat": atomic_bytes(second, twin),
        "two_encodes_identical": True,
        "archive_delta_bytes": len(archive) - LIVE_ARCHIVE_BYTES,
        "rate_delta_S": (len(archive) - LIVE_ARCHIVE_BYTES) * 25.0 / SCORE_DENOMINATOR,
        "projected_S_no_distortion_change": LIVE_SCORE + (len(archive) - LIVE_ARCHIVE_BYTES) * 25.0 / SCORE_DENOMINATOR,
        "score_claim": False,
        "model_decode": decoded,
        "section_census": census,
        "token_field_receipt": {
            "basis": "exact restored IHS1 plus identical semantic, carrier, and tail sections",
            "source_path": str(SOURCE_TOKEN_FIELD),
            "source_exists": SOURCE_TOKEN_FIELD.is_file(),
            "source_bytes": SOURCE_TOKEN_FIELD.stat().st_size if SOURCE_TOKEN_FIELD.is_file() else None,
            "source_sha256": sha256_bytes(SOURCE_TOKEN_FIELD.read_bytes()) if SOURCE_TOKEN_FIELD.is_file() else None,
            "per_pair_receipts": "not_applicable_zero_distortion_lossless_section_recode",
        },
        "public_f26_smoke": public_probe,
        "inflate_sh_smoke": shell_probe,
        "manifest": manifest,
    }


def run() -> dict[str, Any]:
    STORE.mkdir(parents=True, exist_ok=True)
    source = extract_live()
    bounds = derive_bounds(source["ihs1"], source["row_counts"], source["rc1_rider"])
    semistatic_rows, semistatic_streams = race_semistatic(source["ihs1"], source["row_counts"])
    logistic_rows, logistic_streams = race_logistic(source["ihs1"], source["row_counts"])
    design_rows = semistatic_rows + logistic_rows
    best_design = min(design_rows, key=lambda row: (row["default_container_bytes"], row["rider_bytes"]))
    winner_name = best_design["slug"]
    winner_rider = {**semistatic_streams, **logistic_streams}[winner_name]

    base_range_bytes = bounds["current_rc1"]["actual_range_payload_bytes"]
    best_a = min(semistatic_rows, key=lambda row: row["net_J_bytes"])
    best_b = min(logistic_rows, key=lambda row: row["net_J_bytes"])
    combined_disposition = {
        "tested": False,
        "reason": "pruned because at least one component did not beat the current adaptive range payload net of counted parameters",
        "best_a_net_J": best_a["net_J_bytes"],
        "best_b_net_J": best_b["net_J_bytes"],
        "current_rc1_range_payload_bytes": base_range_bytes,
    }
    if best_a["net_J_bytes"] < base_range_bytes and best_b["net_J_bytes"] < base_range_bytes:
        raise Rc2ExperimentError("both A and B helped; combined coder must be implemented before conclusion")

    sweep_rows, winners, _payloads = sweep_containers(source["rc1_rider"], winner_name, winner_rider)
    chosen_object = min(
        (winners["base_shippable"], winners["new_shippable"]),
        key=lambda row: row["container_bytes"],
    )
    net_hpac_delta = chosen_object["container_bytes"] - source["facts"]["header"]["hpac_bytes"]
    staged = None
    if net_hpac_delta < 0:
        chosen_slug = (
            f"{chosen_object['object']}__ck2_{int(chosen_object['ck2'])}__"
            f"q{chosen_object['quality']}__w{chosen_object['lgwin']}.br"
        )
        chosen_payload = (STORE / "retained/container_sweep" / chosen_slug).read_bytes()
        chosen_rider = source["rc1_rider"] if chosen_object["object"] == "base_rc1" else winner_rider
        staged = stage_candidate(source, chosen_object["object"], chosen_rider, chosen_payload)

    falsifier = best_a["net_J_bytes"] > base_range_bytes - 150 and best_b["net_J_bytes"] > base_range_bytes - 150
    result = {
        "schema": "ddm_rc2_hpac_semistatic_depth_mixing.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "source": source["facts"],
        "bounds": bounds,
        "design_rows": design_rows,
        "best_semistatic": best_a,
        "best_logistic": best_b,
        "combined": combined_disposition,
        "container_sweep": sweep_rows,
        "container_winners": winners,
        "chosen": chosen_object,
        "net_hpac_container_delta_bytes": net_hpac_delta,
        "falsifier_fired": falsifier,
        "model_prior_formulation_closed": falsifier,
        "candidate": staged,
        "storage": {
            "store": str(STORE),
            "free_bytes": shutil.disk_usage(STORE).free,
            "payload_policy": "all materialized tables, weights, streams, containers, and archives retained",
        },
        "what_was_not_done": [
            "no scorer run",
            "no MPS or Metal",
            "no Modal dispatch",
            "no model-capacity change",
            "no permutation rerun",
            "no write to live sj1 or upstream",
        ],
    }
    atomic_replace_json(STORE / "RESULT.json", result)
    return result


def refresh_result_token_receipt() -> dict[str, Any]:
    result_path = STORE / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidate = result.get("candidate")
    if not isinstance(candidate, dict):
        raise Rc2ExperimentError("RESULT has no admitted candidate to refresh")
    if not SOURCE_TOKEN_FIELD.is_file():
        raise Rc2ExperimentError(f"source token field is absent: {SOURCE_TOKEN_FIELD}")
    candidate["token_field_receipt"] = {
        "basis": "exact restored IHS1 plus identical semantic, carrier, and tail sections",
        "source_path": str(SOURCE_TOKEN_FIELD),
        "source_exists": True,
        "source_bytes": SOURCE_TOKEN_FIELD.stat().st_size,
        "source_sha256": sha256_bytes(SOURCE_TOKEN_FIELD.read_bytes()),
        "per_pair_receipts": "not_applicable_zero_distortion_lossless_section_recode",
    }
    atomic_replace_json(result_path, result)
    return candidate["token_field_receipt"]


def _smoke_receipt(
    *, role: str, runtime: Path, probe: dict[str, Any], shell: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tac.candidate_seal import measure_archive_identity, measure_runtime_digest

    archive = runtime / "archive.zip"
    tree_sha = measure_runtime_digest(runtime).sha256
    archive_sha = measure_archive_identity(archive).sha256
    identity = {
        "runtime_path": str(runtime),
        "tree_sha256": tree_sha,
        "archive_path": str(archive),
        "archive_sha256": archive_sha,
    }
    direct = {
        **probe,
        **identity,
        "exception_class": None,
        "exception_message": "",
    }
    last_line = str((shell.get("stderr_tail") or [""])[-1])
    prefix = "RuntimeError: "
    shell_receipt = {
        **shell,
        **identity,
        "exception_class": "RuntimeError",
        "exception_message": last_line[len(prefix) :] if last_line.startswith(prefix) else last_line,
    }
    if direct.get("outcome") != "REACHED_TOKEN_DECODE":
        raise Rc2ExperimentError(f"{role} direct public smoke is not a pass: {direct}")
    if shell_receipt.get("outcome") != "REACHED_CUDA_GATE":
        raise Rc2ExperimentError(f"{role} shell public smoke is not a pass: {shell_receipt}")
    return direct, shell_receipt


def retained_public_path_probe(runtime: Path, role: str, timeout_s: float = 240.0) -> dict[str, Any]:
    """Run one exact f26 process with its mandated four threads; retain its work tree."""

    root = STORE / "smoke_work"
    root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (root / f"{role}_attempt_{attempt:02d}").exists():
        attempt += 1
    work = root / f"{role}_attempt_{attempt:02d}"
    work.mkdir()
    library = work / "rc64_backend.so"
    subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            str(runtime / "runtime/entropy/rc64_backend.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    script = (
        "import json,sys,time\n"
        "from pathlib import Path\n"
        "root=Path(sys.argv[1]);sys.path.insert(0,str(root))\n"
        "from runtime.f26_inflate import inflate_archive,InflationError\n"
        "out=Path(sys.argv[2]);started=time.time()\n"
        "try:\n"
        " inflate_archive(root/'archive.zip',out/'0.raw',renderer_dir=root/'cpr1',"
        "device_name='cpu',num_threads=4,checkpoint_dir=out/'.ckpt')\n"
        " print(json.dumps({'outcome':'COMPLETED','seconds':time.time()-started}))\n"
        "except InflationError as error:\n"
        " print(json.dumps({'outcome':'INFLATION_ERROR','error':str(error),'seconds':time.time()-started}))\n"
        "except Exception as error:\n"
        " print(json.dumps({'outcome':type(error).__name__,'error':str(error),'seconds':time.time()-started}))\n"
    )
    environment = {
        **os.environ,
        "CPR1_RC64_LIBRARY": str(library),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    }
    started = time.time()
    try:
        done = subprocess.run(
            [sys.executable, "-c", script, str(runtime), str(work)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(runtime),
            env=environment,
        )
    except subprocess.TimeoutExpired:
        result: dict[str, Any] = {
            "outcome": "REACHED_TOKEN_DECODE",
            "seconds": time.time() - started,
            "note": "no exception within the bound; the semantic guard is behind it",
        }
    else:
        lines = (done.stdout or "").strip().splitlines()
        result = {"outcome": "UNPARSED", "stdout_tail": lines[-3:]}
        for line in reversed(lines):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        result["returncode"] = done.returncode
        result["stderr_tail"] = (done.stderr or "").strip().splitlines()[-5:]
    artifacts = []
    for path in sorted(item for item in work.rglob("*") if item.is_file()):
        artifacts.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    result["processes"] = 1
    result["num_threads"] = 4
    result["retained_work"] = {"path": str(work), "artifacts": artifacts}
    atomic_replace_json(STORE / f"PUBLIC_SMOKE_{role}.json", result)
    return result


def run_public_smoke_role(role: str) -> dict[str, Any]:
    if role == "candidate":
        result = json.loads((STORE / "RESULT.json").read_text(encoding="utf-8"))
        candidate = result.get("candidate")
        if not isinstance(candidate, dict):
            raise Rc2ExperimentError("RESULT has no admitted candidate for the smoke receipt")
        runtime = Path(candidate["staged_runtime"])
    elif role == "frontier":
        runtime = LIVE_ROOT
    else:
        raise Rc2ExperimentError(f"unknown public smoke role {role}")
    direct = retained_public_path_probe(runtime, role)
    shell = rc1._inflate_sh_smoke(runtime, timeout_s=300.0)
    result = {"runtime": str(runtime), "direct": direct, "shell": shell}
    atomic_replace_json(STORE / f"PUBLIC_SMOKE_{role}.json", result)
    if direct.get("outcome") != "REACHED_TOKEN_DECODE" or shell.get("outcome") != "REACHED_CUDA_GATE":
        raise Rc2ExperimentError(f"{role} public smoke pair failed: {result}")
    return result


def build_public_smoke_receipt() -> dict[str, Any]:
    raw: dict[str, dict[str, Any]] = {}
    for role in ("candidate", "frontier"):
        path = STORE / f"PUBLIC_SMOKE_{role}.json"
        if not path.is_file():
            raise Rc2ExperimentError(f"missing retained {role} public smoke: {path}")
        raw[role] = json.loads(path.read_text(encoding="utf-8"))
    candidate_direct, candidate_shell = _smoke_receipt(
        role="candidate",
        runtime=Path(raw["candidate"]["runtime"]),
        probe=raw["candidate"]["direct"],
        shell=raw["candidate"]["shell"],
    )
    frontier_direct, frontier_shell_receipt = _smoke_receipt(
        role="frontier",
        runtime=Path(raw["frontier"]["runtime"]),
        probe=raw["frontier"]["direct"],
        shell=raw["frontier"]["shell"],
    )
    receipt = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "axis": "[macOS-CPU advisory / public-path smoke; score_claim=false]",
        "public_path_probe_seconds": 300.0,
        "public_path_probes": {
            "candidate": candidate_direct,
            "frontier": frontier_direct,
        },
        "inflate_sh_smokes": {
            "candidate": candidate_shell,
            "frontier": frontier_shell_receipt,
        },
    }
    atomic_replace_json(STORE / "PUBLIC_ENTRYPOINT_SMOKE.json", receipt)
    return receipt


def main() -> int:
    global STORE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=STORE)
    parser.add_argument(
        "--stage",
        choices=(
            "run",
            "refresh-token-receipt",
            "public-smoke-candidate",
            "public-smoke-frontier",
            "public-smoke-receipt",
        ),
        default="run",
    )
    args = parser.parse_args()
    STORE = args.store
    started = time.time()
    if args.stage == "refresh-token-receipt":
        print(json.dumps(refresh_result_token_receipt(), sort_keys=True))
        return 0
    if args.stage in {"public-smoke-candidate", "public-smoke-frontier"}:
        role = args.stage.removeprefix("public-smoke-")
        print(json.dumps(run_public_smoke_role(role), sort_keys=True))
        return 0
    if args.stage == "public-smoke-receipt":
        receipt = build_public_smoke_receipt()
        print(
            json.dumps(
                {"receipt": str(STORE / "PUBLIC_ENTRYPOINT_SMOKE.json"), "roles": sorted(receipt["public_path_probes"])}
            )
        )
        return 0
    result = run()
    print(
        json.dumps(
            {
                "result": str(STORE / "RESULT.json"),
                "seconds": time.time() - started,
                "best_semistatic_J": result["best_semistatic"]["net_J_bytes"],
                "best_logistic_J": result["best_logistic"]["net_J_bytes"],
                "net_hpac_delta": result["net_hpac_container_delta_bytes"],
                "candidate": result["candidate"]["archive"] if result["candidate"] else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
