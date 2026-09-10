"""Broader geometry-profile permutation class, scorer-free and fully retained.

Unlike the collar control, this class groups sites by their exact effect on the
required geometry statistics. No physical-video source distribution is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict

import ddm_eb1_entropy_bound as base
import numpy as np

ROOT = base.ROOT / "profile"
CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))


def profile_classes(plane):
    """Permute simple-point labels with identical area/B/E/CC effect signatures."""
    yy, xx = np.indices((base.H - 4, base.W - 4))
    yy, xx = yy + 2, xx + 2
    choose = (yy + 2 * xx) % 5 == 0
    ys, xs = yy[choose], xx[choose]
    collars = np.stack([plane[ys + dy, xs + dx] for dy, dx in base.OFFSETS], axis=1)
    centers = plane[ys, xs]
    potential = np.any(collars != centers[:, None], axis=1)
    groups = defaultdict(list)
    for y, x, collar, center in zip(ys[potential], xs[potential], collars[potential], centers[potential], strict=True):
        signature = tuple(map(int, collar))
        good = base.simple_labels(signature)
        if int(center) not in good or len(good) < 2:
            continue
        around = dict(zip(base.OFFSETS, signature, strict=True))
        neighbors = Counter(around[d] for d in CARDINAL)
        unsaturated = Counter()
        for dy, dx in CARDINAL:
            color = around[(dy, dx)]
            other = [(dy + ey, dx + ex) for ey, ex in CARDINAL if (dy + ey, dx + ex) != (0, 0)]
            if all(around[p] == color for p in other):
                unsaturated[color] += 1
        key = (
            int(y) // 64,
            *(neighbors[c] for c in range(5)),
            *(unsaturated[c] for c in range(5)),
            sum(1 << c for c in good),
        )
        groups[key].append((int(y), int(x)))
    natural, conservative = plane.copy(), plane.copy()
    rows, cardinality, switches, mutable = [], 1, 0, 0
    for key, positions in sorted(groups.items()):
        values = [int(plane[y, x]) for y, x in positions]
        counts = Counter(values)
        if len(counts) < 2:
            continue
        remaining, ways = len(values), 1
        for c in sorted(counts):
            ways *= math.comb(remaining, counts[c])
            remaining -= counts[c]
        cardinality *= ways
        mutable += len(values)
        for (y, x), c in zip(positions, values[1:] + values[:1], strict=True):
            natural[y, x] = c
        buckets = {c: [p for p, v in zip(positions, values, strict=True) if v == c] for c in counts}
        pairs = []
        while sum(bool(v) for v in buckets.values()) >= 2:
            a, b = sorted((c for c in buckets if buckets[c]), key=lambda c: (-len(buckets[c]), c))[:2]
            p, q = buckets[a].pop(), buckets[b].pop()
            pairs.append([p[0] * base.W + p[1], q[0] * base.W + q[1]])
            conservative[p], conservative[q] = plane[q], plane[p]
        k = len(pairs)
        assert k == min(len(values) // 2, len(values) - max(counts.values()))
        switches += k
        rows.append(
            {
                "band_neighbor_hist_unsaturated_hist_good_mask": list(key),
                "positions": [y * base.W + x for y, x in positions],
                "counts": {str(c): counts[c] for c in sorted(counts)},
                "paired_switches": pairs,
                "cardinality_hex": hex(ways),
            }
        )
    return (
        {
            "groups": rows,
            "cardinality_hex": hex(cardinality),
            "switches": switches,
            "mutable_sites": mutable,
            "qmax": max((len(r["counts"]) for r in rows), default=1),
        },
        natural,
        conservative,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=base.Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != (ROOT / "frames").resolve():
        raise ValueError("unexpected resume path")
    base.storage(ROOT)
    source = base.ROOT / "retained/gt_dali_labels.npy"
    bound = json.loads((base.ROOT / "retained/ARRAY_INPUTS.json").read_text())["gt"]
    if base.fact(source) != bound:
        raise ValueError("retained GT identity drift")
    base.record(
        ROOT / "INPUTS.json",
        {
            "gt": bound,
            "producer": base.fact(__file__),
            "base_producer": base.fact(base.__file__),
            "axis": base.AXIS,
            "reference_class_C": "Independent paired swaps within each profile-signature group; all other cells fixed.",
            "reference_class_N": "All observed-label permutations within each profile-signature group; all other cells fixed.",
            "cardinality_direction": "exact counts for these orbits; lower cardinality bounds for the full same-profile class",
            "score_claim": False,
            "retention": "KEEP all fields, counts and per-frame checkpoints",
        },
    )
    gt = np.load(source, mmap_mode="r", allow_pickle=False)
    start = time.monotonic()
    for t in range(base.N):
        path = args.resume_from / f"{t:03d}.json"
        if path.exists():
            continue
        refs, natural, conservative = profile_classes(gt[t])
        old = json.loads((base.ROOT / "retained/frames" / f"{t:03d}.json").read_text())
        checks = {}
        for tag, alternate in (("natural", natural), ("conservative", conservative)):
            base.persist_array(ROOT / tag / f"{t:03d}.npy", alternate)
            geom = base.geometry(alternate)
            for key in ("area", "boundary_cells", "boundary_edges", "pair_edges", "components"):
                assert geom[key] == old["gt"][key], (t, tag, key)
            checks[tag] = {"changed_cells": int(np.count_nonzero(alternate != gt[t])), "invariant_counts_equal": True}
        assert refs["switches"] >= old["reference_classes"]["switches"]
        assert int(refs["cardinality_hex"], 16) >= int(old["reference_classes"]["cardinality_hex"], 16)
        base.record(path, {"pair_index": t, "reference_classes": refs, "checks": checks})
        if (t + 1) % 20 == 0:
            print(json.dumps({"frames_completed": t + 1, "elapsed_seconds": time.monotonic() - start}), flush=True)
    rows = [json.loads((args.resume_from / f"{t:03d}.json").read_text()) for t in range(base.N)]
    cardinality = math.prod(int(r["reference_classes"]["cardinality_hex"], 16) for r in rows)
    switches = sum(r["reference_classes"]["switches"] for r in rows)
    mutable = sum(r["reference_classes"]["mutable_sites"] for r in rows)
    qmax = max(r["reference_classes"]["qmax"] for r in rows)
    bounds = [
        base.partition_description_rate_distortion_lower_bound_v1(cardinality, mutable, qmax, switches, d)
        for d in (6270, 12540, 25080)
    ]
    for b in bounds:
        # Worst-case covering numbers are monotone in the reference class.
        b["natural_minimax_lower_bits_certified"] = max(
            b["natural_lower_bits_certified"], b["conservative_lower_bits_certified"]
        )
        b["natural_minimax_lower_bytes"] = b["natural_minimax_lower_bits_certified"] / 8
        b["uniform_natural_expected_prefix_inherits_subset_bound"] = False
    result = {
        "axis": base.AXIS,
        "score_claim": False,
        "research_only": True,
        "reference_groups": sum(len(r["reference_classes"]["groups"]) for r in rows),
        "cardinality_hex": hex(cardinality),
        "switches": switches,
        "mutable_sites": mutable,
        "qmax": qmax,
        "bounds": bounds,
        "all_frame_invariants_passed": 600,
    }
    base.record(ROOT / "RESULT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cardinality_hex", "bounds")}), flush=True)
    print(json.dumps([{k: v for k, v in b.items() if "hex" not in k} for b in bounds]), flush=True)


if __name__ == "__main__":
    main()
