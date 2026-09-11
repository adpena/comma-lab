#!/usr/bin/env python3
"""Independent full-n600 recount and spatial localization of the LS1 atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np

from experiments.ddm_ls1_oracle_atlas import NAMES
from experiments.ddm_ls1_shipped_surprise import AXIS, FIELD, FIELD_SHA, ROOT, TOTAL, H, K, N, W, arrays, fact, record


def concentration(value):
    """Quantiles of positive gain, with all negative gain separately disclosed."""
    v = np.asarray(value).reshape(-1)
    order = np.argsort(-v, kind="stable")
    positive = np.maximum(v[order], 0)
    cumulative = np.cumsum(positive)
    gross = float(cumulative[-1])
    return {
        "net_bytes": float(v.sum() / 8),
        "positive_bytes": gross / 8,
        "negative_bytes": float(v[v < 0].sum() / 8),
        "positive_units": int(np.count_nonzero(v > 0)),
        "units": len(v),
        "positive_gain_quantiles": {
            str(f): int(np.searchsorted(cumulative, f * gross) + 1) for f in (0.5, 0.8, 0.9, 0.95)
        }
        if gross
        else {},
        "top20_indices": order[:20].tolist(),
        "units_to_25899_bytes": (int(np.searchsorted(cumulative, 25899 * 8) + 1) if gross >= 25899 * 8 else None),
    }


def verify(resume_from):
    if Path(resume_from).resolve() != (ROOT / "verification").resolve():
        raise ValueError("wrong verification root")
    out = Path(resume_from)
    result_path = ROOT / "atlas/RESULT.json"
    deadline = time.monotonic() + 3600
    while not result_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("atlas not complete; no partial verdict")
        time.sleep(2)
    result = json.loads(result_path.read_text())
    trace = json.loads((ROOT / "TRACE.json").read_text())
    field = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H * W))
    if fact(FIELD)["sha256"] != FIELD_SHA or result["symbols"] != field.size:
        raise ValueError("population mismatch")
    spatial_bits = np.zeros((H * W, K))
    spatial_counts = np.zeros((H * W, K), dtype=np.int64)
    class_geometry_bits = np.zeros((K, 7))
    class_geometry_counts = np.zeros((K, 7), dtype=np.int64)
    binary_bits = np.zeros(3)
    indices = np.arange(H * W)
    start = 0
    latest = out / "LATEST.json"
    binding = {"source": fact(Path(__file__)), "atlas": fact(result_path), "trace": fact(ROOT / "TRACE.json")}
    if latest.exists():
        receipt = json.loads(latest.read_text())
        if receipt["binding"] != binding or fact(receipt["state"]["path"]) != receipt["state"]:
            raise ValueError("verification checkpoint drift")
        with np.load(receipt["state"]["path"], allow_pickle=False) as data:
            start = int(data["frame"][0])
            spatial_bits, spatial_counts = data["spatial_bits"], data["spatial_counts"]
            class_geometry_bits, class_geometry_counts = data["class_geometry_bits"], data["class_geometry_counts"]
            binary_bits = data["binary_bits"]
    # Re-read the complete actual receiver rows, not a prefix or summary claim.
    for frame in range(start, N):
        path = ROOT / "rows" / f"frame_{frame:04d}.npz"
        if fact(path) != json.loads(path.with_suffix(".json").read_text()):
            raise ValueError("trace payload hash differs")
        with np.load(path, allow_pickle=False) as data:
            freq = data["frequencies"].astype(np.int64)
            symbol = data["symbols"]
            np.testing.assert_array_equal(symbol, field[frame])
            np.testing.assert_array_equal(freq.sum(axis=1), TOTAL)
            # Different arithmetic expression from the producer.
            bits = 31 - np.log2(freq[indices, symbol].astype(np.float64))
            np.testing.assert_allclose(bits, data["bits"], atol=4e-15, rtol=1e-12)
        spatial_bits[indices, symbol] += bits
        spatial_counts[indices, symbol] += 1
        arg = freq.argmax(axis=1)
        miss = symbol != arg
        pmax = freq[indices, arg] / TOTAL
        binary_bits[0] += float((-np.log2(pmax[~miss])).sum())
        binary_bits[1] += float((-np.log2(1 - pmax[miss])).sum())
        binary_bits[2] += float((bits[miss] + np.log2(1 - pmax[miss])).sum())
        with np.load(ROOT / "atlas/cells" / f"frame_{frame:04d}.npz", allow_pickle=False) as data:
            geometry = data["geometry"]
            np.testing.assert_array_equal(data[NAMES[1]] // 64, data[NAMES[0]])
            np.testing.assert_array_equal(data[NAMES[2]] // 8, data[NAMES[1]])
        np.testing.assert_array_equal(geometry == 6, ~miss)
        for c in range(K):
            for g in range(7):
                selected = (symbol == c) & (geometry == g)
                class_geometry_bits[c, g] += float(bits[selected].sum())
                class_geometry_counts[c, g] += int(np.count_nonzero(selected))
        if (frame + 1) % 25 == 0:
            path = out / f"stage_{frame + 1:04d}.npz"
            arrays(
                path,
                {
                    "frame": np.array([frame + 1]),
                    "spatial_bits": spatial_bits,
                    "spatial_counts": spatial_counts,
                    "class_geometry_bits": class_geometry_bits,
                    "class_geometry_counts": class_geometry_counts,
                    "binary_bits": binary_bits,
                },
            )
            record(latest, {"binding": binding, "state": fact(path)})
            print(json.dumps({"stage": "verification", "frames": frame + 1}), flush=True)
    np.testing.assert_allclose(class_geometry_bits, result["atlas"]["class_geometry_bits"], atol=1e-6, rtol=1e-10)
    np.testing.assert_array_equal(class_geometry_counts, result["atlas"]["class_geometry_counts"])
    np.testing.assert_allclose(binary_bits.sum(), trace["bits"], atol=1e-6)
    np.testing.assert_allclose(spatial_bits.sum(), trace["bits"], atol=1e-6)
    arrays(
        out / "spatial_atlas.npz",
        {"bits": spatial_bits.reshape(H, W, K), "counts": spatial_counts.reshape(H, W, K), "binary_bits": binary_bits},
    )
    localization = {}
    for name in NAMES:
        summary = json.loads((ROOT / "atlas" / (name + "_RESULT.json")).read_text())
        path = Path(summary["gain_payload"]["path"])
        if fact(path) != summary["gain_payload"]:
            raise ValueError("marginal-gain payload changed")
        gains = np.load(path, mmap_mode="r", allow_pickle=False)
        with np.load(ROOT / "atlas" / (name + "_localization.npz"), allow_pickle=False) as data:
            pairs = data["pair_gain_bits"]
            rows = data["row_gain_bits"]
            lane = data["lane_gain_bits"]
        np.testing.assert_allclose(gains.sum() / 8, summary["mm_gain_bytes"], atol=1e-6)
        localization[name] = {
            "pairs": concentration(pairs.sum(axis=1)),
            "lane_pairs": concentration(pairs[:, 1]),
            "rows": concentration(rows.sum(axis=1)),
            "lane_rows": concentration(rows[:, 1]),
            "lane_symbols": concentration(lane),
            "band128_319_gain_bytes": float(rows[128:320].sum() / 8),
            "lane_band128_319_gain_bytes": float(rows[128:320, 1].sum() / 8),
        }
    # Check the transmitted override against BOTH context maps, over all pairs.
    map_result = result["map_price"]
    import zlib

    payload = Path(map_result["payloads"][0]["path"]).read_bytes()
    decoded = zlib.decompress(payload[10:])
    if hashlib.sha256(decoded).hexdigest() != map_result["raw"]["sha256"]:
        raise ValueError("map decode differs")
    delta = np.frombuffer(decoded, dtype=np.uint8).reshape(N, H * W)
    for frame in range(N):
        with np.load(ROOT / "atlas/cells" / f"frame_{frame:04d}.npz", allow_pickle=False) as data:
            visible = data[NAMES[1]] % 8
            granted = data[NAMES[2]] % 8
        np.testing.assert_array_equal(np.where(delta[frame] == 0, visible, delta[frame].astype(np.int64) - 1), granted)
    result = {
        "axis": AXIS,
        "full_n600": True,
        "score_claim": False,
        "status": "PASS",
        "source": fact(Path(__file__)),
        "trace": fact(ROOT / "TRACE.json"),
        "atlas": fact(result_path),
        "spatial": fact(out / "spatial_atlas.npz"),
        "binary_bits": binary_bits.tolist(),
        "class_counts": class_geometry_counts.sum(axis=1).tolist(),
        "class_bits": class_geometry_bits.sum(axis=1).tolist(),
        "band128_319_bits": float(spatial_bits.reshape(H, W, K)[128:320].sum()),
        "localization": localization,
        "checks": [
            "full n600 field hash",
            "all 117964800 symbol frequencies and independent bit arithmetic",
            "class/geometry recount",
            "binary decomposition",
            "nested contexts",
            "all n600 map reconstruction",
            "complete marginal-gain hash and sum",
        ],
    }
    record(out / "RESULT.json", result)
    print(json.dumps({"status": "PASS", "symbols": N * H * W}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", required=True)
    verify(parser.parse_args().resume_from)
