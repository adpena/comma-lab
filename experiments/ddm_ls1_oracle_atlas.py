#!/usr/bin/env python3
"""Frozen-history oracle diagnostics on the actual move44 receiver rows.

No fitted runtime model, scorer or candidate. Constant-cell empirical oracles
are upper bounds on in-sample savings in that cell family only; Miller--Madow
is an estimate, not a universal or finite-sample certified bound.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np

from experiments.ddm_ls1_shipped_surprise import (
    AXIS, FIELD, FIELD_SHA, H, K, N, ROOT, TOTAL, W, arrays, fact, record,
)
from experiments.ddm_tc1_context_statistics import LEVELS, contexts
from experiments.ddm_gdc3_geometry_law_probe import classify_plane, GEOMETRY_NAMES

PLANE = H * W
Y, X = np.indices((H, W))
GROUP = X % 64 + 2 * (Y % 64)
NAMES = ("tc1_joint", "receiver_lane", "granted_previous_row_lane")
GEOMETRIES = list(GEOMETRY_NAMES) + ["correct_prediction"]
EDGES = np.array([0, 1, 2, 4, 8, 16, 32])


def lane_row_distance(plane, causal):
    """Nearest Lane in the preceding row, clipped at 33; causal reads obey HPAC order.

Eight bins: 0, 1, 2, 3..4, 5..8, 9..16, 17..32, >=33/no observed Lane.
The oracle sees the whole preceding row; the receiver sees only earlier groups.
"""
    distance = np.full((H, W), 33, dtype=np.uint8)
    for dx in range(-32, 33):
        start, stop = max(0, -dx), min(W, W - dx)
        source = plane[:-1, start + dx:stop + dx] == 1
        if causal:
            source &= GROUP[:-1, start + dx:stop + dx] < GROUP[1:, start:stop]
        view = distance[1:, start:stop]
        np.minimum(view, np.where(source, abs(dx), 33).astype(np.uint8), out=view)
    return np.searchsorted(EDGES, distance).astype(np.uint8).reshape(-1)


def previous_frame_distance(previous):
    """Horizontal Lane distance in the complete preceding decoded plane, eight bins."""
    if previous is None:
        return np.full(PLANE, 7, dtype=np.uint8)
    xx = np.broadcast_to(np.arange(W), (H, W))
    lane = previous == 1
    left = np.maximum.accumulate(np.where(lane, xx, -W), axis=1)
    right = np.minimum.accumulate(np.where(lane, xx, 2 * W)[:, ::-1], axis=1)[:, ::-1]
    d = np.minimum(xx - left, right - xx)
    return np.searchsorted(EDGES, d).astype(np.uint8).reshape(-1)


def make_keys(freq, plane, previous, temporal_run):
    """Nested cells; class/geometry labels are attribution only, never predictors."""
    arg = freq.argmax(axis=1)
    miss = (TOTAL - freq[np.arange(PLANE), arg].astype(np.int64)) / TOTAL
    bucket = arg * 64 + np.minimum((-2 * np.log2(miss)).astype(np.int64), 63)
    joint = bucket.astype(np.uint64)
    causal_context = contexts(plane, previous, temporal_run)
    for name, levels in LEVELS.items():
        joint = joint * levels + causal_context[name].astype(np.uint64)
    past = previous_frame_distance(previous).astype(np.uint64)
    visible = lane_row_distance(plane, True).astype(np.uint64)
    granted = lane_row_distance(plane, False).astype(np.uint64)
    receiver = (joint * 8 + past) * 8 + visible
    return (joint, receiver, receiver * 8 + granted)


def read_frame(frame):
    path = ROOT / "rows" / f"frame_{frame:04d}.npz"
    if fact(path) != json.loads(path.with_suffix(".json").read_text()):
        raise ValueError("trace row checksum changed")
    with np.load(path, allow_pickle=False) as data:
        freq, truth, bits = data["frequencies"], data["symbols"], data["bits"]
    if freq.shape != (PLANE, K) or not np.all(freq.sum(axis=1, dtype=np.uint64) == TOTAL):
        raise ValueError("invalid receiver row")
    selected = freq[np.arange(PLANE), truth]
    np.testing.assert_array_equal(bits, -np.log2(selected.astype(np.float64) / TOTAL))
    return freq, truth, bits


def sparse_add(counts, code):
    keys, values = np.unique(code, return_counts=True)
    for key, value in zip(keys.tolist(), values.tolist(), strict=True):
        counts[key] = counts.get(key, 0) + value


def count_stage(resume_from):
    """One atomic full-state checkpoint every 25 frames; wait for producer receipts."""
    store = ROOT / "atlas"
    if Path(resume_from).resolve() != store.resolve():
        raise ValueError("wrong atlas resume root")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise ValueError("shipped field changed")
    binding = dict(producer=fact(Path(__file__)), field=fact(FIELD),
                   contexts=fact(REPO / "experiments/ddm_tc1_context_statistics.py"),
                   geometry=fact(REPO / "experiments/ddm_gdc3_geometry_law_probe.py"))
    bp = store / "BINDING.json"
    if bp.exists() and json.loads(bp.read_text()) != binding:
        raise ValueError("analysis source/input drift")
    record(bp, binding)
    target = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    latest = store / "LATEST.json"
    counts = [dict() for _ in NAMES]
    start = 0
    temporal = np.zeros((H, W), dtype=np.uint8)
    table_bits = np.zeros((K, 7))
    table_counts = np.zeros((K, 7), dtype=np.int64)
    pair_bits = np.zeros((N, K))
    row_bits = np.zeros((H, K))
    if latest.exists():
        receipt = json.loads(latest.read_text())
        path = Path(receipt["path"])
        if fact(path) != receipt:
            raise ValueError("atlas checkpoint hash changed")
        with np.load(path, allow_pickle=False) as data:
            start = int(data["frame"][0])
            temporal, table_bits, table_counts = data["temporal"], data["table_bits"], data["table_counts"]
            pair_bits, row_bits = data["pair_bits"], data["row_bits"]
            counts = [dict(zip(data[name + "_keys"].tolist(), data[name + "_counts"].tolist(), strict=True)) for name in NAMES]
    for frame in range(start, N):
        receipt = ROOT / "rows" / f"frame_{frame:04d}.json"
        deadline = time.monotonic() + 3600
        while not receipt.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("trace stalled; prior complete atlas checkpoint retained")
            time.sleep(2)
        freq, truth, bits = read_frame(frame)
        plane, previous = target[frame], None if frame == 0 else target[frame - 1]
        np.testing.assert_array_equal(truth.reshape(H, W), plane)
        geom = classify_plane(plane, freq.argmax(axis=1).reshape(H, W))["geometry"].reshape(-1)
        geom = np.where(geom < 0, 6, geom)
        code = truth.astype(np.int64) * 7 + geom
        table_bits += np.bincount(code, weights=bits, minlength=K * 7).reshape(K, 7)
        table_counts += np.bincount(code, minlength=K * 7).reshape(K, 7)
        pair_bits[frame] = np.bincount(truth, weights=bits, minlength=K)
        row_bits += np.bincount(Y.reshape(-1) * K + truth, weights=bits, minlength=H * K).reshape(H, K)
        keys = make_keys(freq, plane, previous, temporal)
        for i, key in enumerate(keys):
            sparse_add(counts[i], key * K + truth)
        # Keys are retained, eliminating any recomputation ambiguity at attribution.
        arrays(store / "cells" / f"frame_{frame:04d}.npz", dict(
            **{name: key for name, key in zip(NAMES, keys, strict=True)}, geometry=geom.astype(np.uint8)))
        temporal = (np.zeros((H, W), dtype=np.uint8) if frame == 0 else
                    np.where(plane == previous, np.minimum(temporal + 1, 7), 0).astype(np.uint8))
        if (frame + 1) % 25 == 0:
            values = dict(frame=np.array([frame + 1]), temporal=temporal, table_bits=table_bits,
                          table_counts=table_counts, pair_bits=pair_bits, row_bits=row_bits)
            for name, counts_i in zip(NAMES, counts, strict=True):
                kk = np.array(sorted(counts_i), dtype=np.uint64)
                values[name + "_keys"] = kk
                values[name + "_counts"] = np.array([counts_i[int(k)] for k in kk], dtype=np.int64)
            path = store / f"counts_{frame + 1:04d}.npz"
            arrays(path, values)
            record(latest, fact(path))
            print(json.dumps(dict(stage="atlas_counts", frames=frame + 1,
                                  occupied=[len(c) for c in counts])), flush=True)
    return store


def oracle_table(keys, values):
    """Exact plug-in MLE and explicitly allocated MM degrees, for these cells only."""
    cells, inverse = np.unique(keys // K, return_inverse=True)
    counts = np.zeros((len(cells), K), dtype=np.int64)
    counts[inverse, (keys % K).astype(np.int64)] = values
    totals = counts.sum(axis=1)
    nz = counts > 0
    support = nz.sum(axis=1)
    loss = np.zeros_like(counts, dtype=np.float64)
    np.log2(np.divide(totals[:, None], counts, out=np.ones_like(loss), where=nz), out=loss)
    # Each occupied symbol gets an equal share of its cell's K_c-1 degrees;
    # distribute this fixed diagnostic correction over its occurrences.
    correction = np.divide((1 - 1 / support)[:, None] / (2 * math.log(2)), counts,
                           out=np.zeros_like(loss), where=nz)
    plugin_bits = float((counts * loss).sum())
    correction_bits = float((counts * correction).sum())
    np.testing.assert_allclose(correction_bits, float((support - 1).sum()) / (2 * math.log(2)), rtol=1e-12)
    return cells, counts, loss, correction, dict(
        symbols=int(totals.sum()), occupied_cells=len(cells),
        singleton_cells=int(np.count_nonzero(totals == 1)),
        plugin_bits=plugin_bits, correction_bits=correction_bits,
        mm_bits=plugin_bits + correction_bits,
        table_fp16_probability_bytes=int(len(cells) * (K - 1) * 2),
        table_uint64_key_bytes=int(len(cells) * 8),
        correction="Miller-Madow; per-class allocation convention, not class-conditioned fit",
        scope="fixed cells, full n600 in-sample MLE; free oracle parameters, not a universal ceiling")


def cumulative(values):
    """Oracle-ranked signed gains; location/ranking information is granted, not free."""
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))[::-1]
    sums = ordered.cumsum() / 8
    indices = sorted(set([1, len(ordered)] + [n for n in (10, 100, 1000, 10000, 100000, 1000000) if n <= len(ordered)] +
                         [max(1, int(len(ordered) * f)) for f in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)]))
    return [dict(touched=n, oracle_signed_gain_bytes=float(sums[n - 1])) for n in indices]


def attribute(store):
    """Marginal log-likelihood differences against shipped probabilities, never averages."""
    while not (ROOT / "TRACE.json").exists():
        time.sleep(2)
    trace = json.loads((ROOT / "TRACE.json").read_text())
    if not trace["full_n600"] or not trace["field_identity"]:
        raise ValueError("missing n600 receiver identity")
    with np.load(store / "counts_0600.npz", allow_pickle=False) as data:
        tables = [oracle_table(data[name + "_keys"], data[name + "_counts"]) for name in NAMES]
        atlas = dict(class_geometry_bits=data["table_bits"].tolist(), class_geometry_counts=data["table_counts"].tolist(),
                     pair_class_bits=data["pair_bits"].tolist(), row_class_bits=data["row_bits"].tolist())
    if sum(map(sum, atlas["class_geometry_counts"])) != N * PLANE:
        raise ValueError("atlas coverage mismatch")
    np.testing.assert_allclose(sum(map(sum, atlas["class_geometry_bits"])), trace["bits"], rtol=1e-12)
    results = []
    for name, (cells, counts, loss, correction, summary) in zip(NAMES, tables, strict=True):
        class_geom = np.zeros((K, 7))
        pair_gain = np.zeros((N, K))
        row_gain = np.zeros((H, K))
        gain_path = store / (name + "_gain_bits.npy")
        new_gain = gain_path.with_suffix(".npy.new")
        gains = np.lib.format.open_memmap(new_gain, mode="w+", dtype=np.float64, shape=(N, PLANE))
        # The full arrays are required localization evidence, retained and hashed.
        for frame in range(N):
            saved = store / name / f"gain_{frame:04d}.npz"
            _, truth, bits = read_frame(frame)
            path = store / "cells" / f"frame_{frame:04d}.npz"
            if fact(path) != json.loads(path.with_suffix(".json").read_text()):
                raise ValueError("context payload changed")
            with np.load(path, allow_pickle=False) as data:
                key, geometry = data[name], data["geometry"]
            where = np.searchsorted(cells, key)
            np.testing.assert_array_equal(cells[where], key)
            if np.any(counts[where, truth] == 0):
                raise ValueError("oracle missing occupied symbol")
            if saved.exists():
                if fact(saved) != json.loads(saved.with_suffix(".json").read_text()):
                    raise ValueError("gain checkpoint changed")
                with np.load(saved, allow_pickle=False) as data:
                    gain = data["gain"]
                np.testing.assert_array_equal(gain, bits - loss[where, truth] - correction[where, truth])
            else:
                gain = bits - loss[where, truth] - correction[where, truth]
                arrays(saved, dict(gain=gain))
            gains[frame] = gain
            class_geom += np.bincount(truth.astype(np.int64) * 7 + geometry, weights=gain, minlength=K * 7).reshape(K, 7)
            pair_gain[frame] = np.bincount(truth, weights=gain, minlength=K)
            row_gain += np.bincount(Y.reshape(-1) * K + truth, weights=gain, minlength=H * K).reshape(H, K)
        gains.flush()
        if gain_path.exists():
            if fact(gain_path)["sha256"] != fact(new_gain)["sha256"]:
                raise ValueError("complete gain replay changed")
            # Do not delete retained bytes. The byte-identical repeat is evidence.
            new_gain.replace(store / (name + "_gain_bits.repeat.npy"))
        else:
            new_gain.replace(gain_path)
        measured_gain = float(class_geom.sum()) / 8
        np.testing.assert_allclose(measured_gain, (trace["bits"] - summary["mm_bits"]) / 8, atol=1e-7)
        # Positive-only per-symbol picking is an even stronger acausal selection
        # grant; do not equate its curve with any paid carrier or model.
        top = cumulative(gains)
        lane_values = []
        for frame in range(N):
            truth = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, PLANE))[frame]
            lane_values.append(np.asarray(gains[frame])[truth == 1].copy())
        lane = np.concatenate(lane_values)
        arrays(store / (name + "_localization.npz"), dict(pair_gain_bits=pair_gain, row_gain_bits=row_gain,
                                                         class_geometry_gain_bits=class_geom, lane_gain_bits=lane))
        summary.update(name=name, mm_gain_bytes=measured_gain,
                       plugin_gain_bytes=(trace["bits"] - summary["plugin_bits"]) / 8,
                       demand_bytes=25899, mm_shortfall_bytes=25899 - measured_gain,
                       class_geometry_gain_bytes=(class_geom / 8).tolist(),
                       pair_curve=cumulative(pair_gain.sum(axis=1)),
                       lane_pair_curve=cumulative(pair_gain[:, 1]),
                       row_curve=cumulative(row_gain.sum(axis=1)),
                       lane_row_curve=cumulative(row_gain[:, 1]),
                       symbol_curve=top, lane_symbol_curve=cumulative(lane),
                       gain_payload=fact(gain_path),
                       localization_warning="retrospective signed-gain ranking grants selection addresses; no counted selector or carrier")
        record(store / (name + "_RESULT.json"), summary)
        results.append(summary)
        del gains
        print(json.dumps(dict(stage="oracle_attribution", name=name, mm_gain_bytes=measured_gain)), flush=True)
    if any(results[i + 1]["plugin_bits"] > results[i]["plugin_bits"] + 1e-6 for i in range(2)):
        raise ValueError("nested plugin entropy increased")
    map_price = price_map(store)
    record(store / "RESULT.json", dict(axis=AXIS, score_claim=False, full_n600=True,
        symbols=N * PLANE, field=fact(FIELD), geometry_names=GEOMETRIES, atlas=atlas, oracles=results,
        note="Conditional entropy of exact specified observable/granted contexts; no universal family verdict, no physical carrier cost measured",
        trace=fact(ROOT / "TRACE.json"), producer=fact(Path(__file__)), map_price=map_price))


def price_map(store):
    """Actual encoded extra-map cost, with exact reconstruction and retained twins.

This is an achieved code length, not a map entropy lower bound. It pays only
for oracle context access, not the hindsight probability parameters.
"""
    raw_path = store / "granted_distance_delta.u8"
    raw_new = raw_path.with_suffix(".u8.new")
    changed = 0
    with raw_new.open("wb") as handle:
        for frame in range(N):
            path = store / "cells" / f"frame_{frame:04d}.npz"
            with np.load(path, allow_pickle=False) as data:
                visible = (data["receiver_lane"] % 8).astype(np.uint8)
                granted = (data["granted_previous_row_lane"] % 8).astype(np.uint8)
            delta = np.where(visible == granted, 0, granted + 1).astype(np.uint8)
            np.testing.assert_array_equal(np.where(delta == 0, visible, delta - 1), granted)
            changed += int(np.count_nonzero(delta))
            handle.write(delta.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    if raw_path.exists() and fact(raw_path)["sha256"] != fact(raw_new)["sha256"]:
        raise ValueError("map repeat differs")
    raw_new.replace(raw_path if not raw_path.exists() else store / "granted_distance_delta.repeat.u8")
    payloads = []
    for repeat in range(2):
        encoder = zlib.compressobj(9)
        path = store / f"granted_distance_delta_{repeat}.lsm"
        temporary = path.with_suffix(".lsm.new")
        with raw_path.open("rb") as source, temporary.open("wb") as dest:
            dest.write(b"LSM1" + struct.pack("<HHH", N, H, W))
            for chunk in iter(lambda: source.read(4 << 20), b""):
                dest.write(encoder.compress(chunk))
            dest.write(encoder.flush())
        if path.exists() and fact(path)["sha256"] != fact(temporary)["sha256"]:
            raise ValueError("encoded map changed")
        temporary.replace(path)
        payloads.append(fact(path))
    if payloads[0]["sha256"] != payloads[1]["sha256"]:
        raise ValueError("map coder nondeterminism")
    payload = Path(payloads[0]["path"]).read_bytes()
    if payload[:10] != b"LSM1" + struct.pack("<HHH", N, H, W):
        raise ValueError("map header invalid")
    decoded = zlib.decompress(payload[10:])
    decoded_path = store / "granted_distance_delta.decoded.u8"
    if not decoded_path.exists():
        decoded_path.write_bytes(decoded)
    if fact(decoded_path)["sha256"] != fact(raw_path)["sha256"]:
        raise ValueError("map decoded identity failed")
    return dict(raw=fact(raw_path), payloads=payloads, decoded=fact(decoded_path),
                changed_context_symbols=changed, cost_bytes=payloads[0]["bytes"],
                scope="actual lossless causal-distance override stream; oracle probability table unpriced; no candidate archive",
                cost_direction="achieved upper bound on map code length, not minimum possible cost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args()
    attribute(count_stage(args.resume_from))
