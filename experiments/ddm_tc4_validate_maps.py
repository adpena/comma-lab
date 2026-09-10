#!/usr/bin/env python3
"""Real-field offline/prefix equivalence and future-symbol mutation controls.

Implementation controls only: no score, no estimate of byte savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments import ddm_tc4_maps as maps
from experiments import ddm_tc4_price as price


def run(count):
    pin = {"inputs": price.pin("map_controls"), "validator": price.fact(Path(__file__))}
    field = np.memmap(price.SOURCE / "field.u8", dtype=np.uint8, mode="r", shape=(600, 384, 512))
    rng = np.random.default_rng(20260910)
    selected = sorted(rng.choice(600, count, replace=False).tolist())
    work = price.ROOT / "map_controls"
    for frame in selected:
        path = work / f"frame_{frame:04d}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt["binding"] != pin or not receipt["online_offline_equal"] or not receipt["future_mutation_equal"]:
                raise ValueError("control restart drift")
            continue
        plane = field[frame]
        previous = None if not frame else field[frame - 1]
        prior = price.arrays(price.SOURCE / "prepare_v2/frames" / f"frame_{frame:04d}.npz")
        bins = prior["bins0"].reshape(384, 512)
        temporal = maps.previous_map(previous)
        complete = maps.context_maps(plane, temporal, bins, complete=True)
        prefix = np.full((384, 512), 5, dtype=np.uint8)
        online_bins = np.full((384, 512), 9, dtype=np.uint8)
        for group, positions in enumerate(price.POSITIONS):
            online = maps.context_maps(prefix, temporal, online_bins, positions)
            np.testing.assert_array_equal(online, complete[positions])
            if group in (0, 31, 63, 94, 126, 158, 189):
                changed = plane.copy().reshape(-1)
                unavailable = group <= price.base.GROUP
                changed[unavailable] = (changed[unavailable] + 1) % 5
                altered_bins = bins.copy().reshape(-1)
                altered_bins[unavailable] = (altered_bins[unavailable] + 1) % 9
                actual = maps.context_maps(
                    changed.reshape(384, 512), temporal, altered_bins.reshape(384, 512), positions, complete=True
                )
                np.testing.assert_array_equal(actual, online)
            prefix.reshape(-1)[positions] = plane.reshape(-1)[positions]
            online_bins.reshape(-1)[positions] = bins.reshape(-1)[positions]
        price.record(
            path,
            {
                "binding": pin,
                "frame": frame,
                "groups": 190,
                "maps": 5,
                "online_offline_equal": True,
                "future_mutation_equal": True,
                "plane_sha256": hashlib.sha256(plane.tobytes()).hexdigest(),
            },
        )
        print(json.dumps({"frame": frame, "online_offline_equal": True}), flush=True)
    return price.record(
        work / "RESULT.json",
        {
            "binding": pin,
            "selected": selected,
            "seed": 20260910,
            "all_passed": True,
            "scope": "real-field implementation controls; not byte/score estimates",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.resume_from.resolve() != price.ROOT.resolve() or not 0 < args.count <= 600:
        raise ValueError("invalid control root/count")
    print(json.dumps(run(args.count)), flush=True)
