# SPDX-License-Identifier: MIT
"""ddm_ma1 - extract the MISS SECTOR of the live token field, in exact decode order.

WHY THIS EXISTS.  ``ddm_fx1`` priced the un-adapted within-miss relative law at a
**1,247.19 B ceiling** and nobody has attacked it; ``ddm_fx2`` re-confirmed the
decomposition (``hit_event + within_miss = total``, closing to the cent) and left
the sector untouched.  It is the largest UNBUILT priced item on the token stream.

THE SEPARABILITY THEOREM THIS SCRIPT RESTS ON, verified at source in
``experiments/ddm_rr4_free_corrector_v2.py``:

* ``coding_row`` (L271-293) scales every non-argmax column by ONE scalar,
  ``scale = (1 - q) / one_minus``.  So the coded probability of a non-argmax
  class ``k`` is ``row64[k] * scale`` and the RELATIVE law among the miss classes
  is exactly the neural prior's, ``r_k = row64[k] / one_minus``, unmodified.
* ``observe`` (L295-303) folds only ``hit = decoded == arg``.  The hit-event
  statistics NEVER see which class a miss landed on.

Therefore a model that reweights the miss-sector relative law changes NEITHER the
hit-event code length NOR the hit-event model's trajectory, and

    delta(total code length) == delta(within-miss code length), EXACTLY.

That is what makes this extraction admissible rather than a shortcut: the sector
is 223,694 records out of 117,964,800 positions, so a race that would cost four
minutes per architecture on the full field costs seconds here, and the number it
produces is the SAME number, not an estimate of it.  The full-field replay is
still run as the confirming control (``ddm_ma1_race_within_miss.py --confirm``).

ORDER IS PART OF THE CONTRACT.  Records are emitted frame-major, then by causal
group, matching ``coder_replay.replay_code_length``.  ``observe`` is batched per
group in the shipped driver, so a faithful replay must apply group-batched
updates; the ``group`` column is retained so the race can honour that.

WHAT IS CAUSAL.  Neighbour reads are gated by the real decode order: a neighbour
is already decoded exactly when its group index is strictly smaller (``ddm_fx2``
R1 measured up 98.6945%, up-right 98.6945%, left 98.6301%, up-left 97.3425%).
Positions whose neighbour is not yet decoded carry the sentinel ``UNKNOWN = 5``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.micro_edit.coder_replay import (  # noqa: E402
    HPAC_LOGIT_PRECISION,
    NUM_CLASSES,
    PLANE,
    ReplayAssets,
    _frame_probabilities,
    _group_positions,
)

HEIGHT = 384
WIDTH = 512
UNKNOWN = 5
PROB_EPS = 1e-9
U_BINS = 64
RUN_LEVELS = 8
RUN_CAP = 255
BOUNDARY_LEVELS = 5

_ASSETS = ReplayAssets(
    logits_i16=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/base_logits_int16_n600.i16"
    ),
    tokens_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
        ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
    ),
    boundary_u8=Path(
        "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/boundary_bucket_n600.u8"
    ),
    group_index_u8=Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/group_index.u8"),
    table_values_npy=Path("/Volumes/APDataStore/pact/ddm_me1/table_values.npy"),
)

# ddm_rr4 surprise-bin ladder, transcribed so the extract carries the same ubin
# the live corrector computes.  Exact powers of two and 1/sqrt(2); no logarithm.
_INV_SQRT2 = float(np.sqrt(0.5))


def _surprise_ascending() -> np.ndarray:
    table = np.empty(U_BINS - 1, dtype=np.float64)
    for k in range(1, U_BINS):
        value = float(np.ldexp(1.0, -(k // 2)))
        if k % 2:
            value *= _INV_SQRT2
        table[k - 1] = value
    return table[::-1].copy()


_SURPRISE_ASC = _surprise_ascending()


def _neighbour_offsets() -> dict[str, tuple[int, int]]:
    """The four causal offsets ddm_fx2 R1 measured, highest causality first."""
    return {"up": (0, -1), "upright": (1, -1), "left": (-1, 0), "upleft": (-1, -1)}


def _neighbour_index(dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat source index for each plane position, plus an in-bounds mask."""
    ys, xs = np.divmod(np.arange(PLANE), WIDTH)
    sx = xs + dx
    sy = ys + dy
    inside = (sx >= 0) & (sx < WIDTH) & (sy >= 0) & (sy < HEIGHT)
    src = np.where(inside, sy * WIDTH + sx, 0).astype(np.int64)
    return src, inside


def extract(frames: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_values = np.load(_ASSETS.table_values_npy).astype(np.float32)
    group_index = np.fromfile(_ASSETS.group_index_u8, dtype=np.uint8).astype(np.int64)

    # The extract sorts misses by ``group * (PLANE + 1) + flat`` rather than
    # walking the group list, so assert once that the two agree.  The shipped
    # driver visits groups in ascending index and, within a group, in ascending
    # flat position (stable argsort of a uint8 key).  If that ever stopped
    # holding, every online model raced on this sector would be replaying a
    # decode order the receiver does not use.
    walked = np.concatenate(_group_positions(group_index))
    sorted_key = np.argsort(
        group_index * (PLANE + 1) + np.arange(PLANE), kind="stable"
    )
    if not np.array_equal(walked, sorted_key):
        raise RuntimeError("extraction order does not match the shipped decode order")

    logits = np.memmap(_ASSETS.logits_i16, dtype=np.int16, mode="r")
    tokens = np.memmap(_ASSETS.tokens_u8, dtype=np.uint8, mode="r")
    boundary_all = np.memmap(_ASSETS.boundary_u8, dtype=np.uint8, mode="r")

    offsets = _neighbour_offsets()
    neighbour = {name: _neighbour_index(*off) for name, off in offsets.items()}
    # A neighbour is decoded before us exactly when its causal group is smaller.
    causal = {
        name: inside & (group_index[src] < group_index)
        for name, (src, inside) in neighbour.items()
    }

    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "frame", "group", "pos", "arg", "token", "base_class", "boundary",
            "ubin", "run", "agree1", "agree2", "prev1", "prev2",
            "nb_up", "nb_upright", "nb_left", "nb_upleft",
        )
    }
    rows: list[np.ndarray] = []
    p_max_col: list[np.ndarray] = []
    one_minus_col: list[np.ndarray] = []

    prev1 = np.zeros(PLANE, dtype=np.uint8)
    prev2 = np.zeros(PLANE, dtype=np.uint8)
    run = np.zeros(PLANE, dtype=np.int64)
    have_prev = False

    hit_bits = 0.0
    being_miss_bits = 0.0
    within_miss_bits = 0.0
    hit_n = 0
    miss_n = 0
    started = time.time()

    for frame in range(frames):
        base = (
            np.asarray(
                logits[frame * PLANE * NUM_CLASSES : (frame + 1) * PLANE * NUM_CLASSES]
            )
            .reshape(PLANE, NUM_CLASSES)
            .astype(np.float32)
            / HPAC_LOGIT_PRECISION
        )
        boundary = np.asarray(boundary_all[frame * PLANE : (frame + 1) * PLANE]).astype(np.int64)
        token = np.asarray(tokens[frame * PLANE : (frame + 1) * PLANE]).astype(np.int64)
        probability = _frame_probabilities(base, boundary, table_values)
        row64 = probability.astype(np.float64)
        predicted = base.argmax(axis=1).astype(np.int64)

        arg = row64.argmax(axis=1)
        idx = np.arange(PLANE)
        p_max = row64[idx, arg]
        one_minus = np.maximum(1.0 - p_max, PROB_EPS)
        chosen = np.maximum(row64[idx, token], 1e-300)

        below = np.searchsorted(_SURPRISE_ASC, one_minus, side="left")
        ubin = np.clip((U_BINS - 1) - below, 0, U_BINS - 1).astype(np.int64)
        if have_prev:
            agree1 = (prev1.astype(np.int64) == predicted).astype(np.int64)
            agree2 = (prev2.astype(np.int64) == predicted).astype(np.int64)
        else:
            agree1 = np.zeros(PLANE, dtype=np.int64)
            agree2 = np.zeros(PLANE, dtype=np.int64)
        run_level = np.minimum(run, RUN_LEVELS - 1)

        # Causal neighbour classes from THIS frame's already-decoded tokens.
        nb = {}
        for name, (src, _inside) in neighbour.items():
            values = np.full(PLANE, UNKNOWN, dtype=np.int64)
            mask = causal[name]
            values[mask] = token[src[mask]]
            nb[name] = values

        miss = token != arg
        bits = -np.log2(chosen)
        hit_bits += float(bits[~miss].sum())
        hit_n += int((~miss).sum())
        relative = np.maximum(chosen[miss] / one_minus[miss], 1e-300)
        within = float((-np.log2(relative)).sum())
        within_miss_bits += within
        being_miss_bits += float(bits[miss].sum()) - within
        miss_n += int(miss.sum())

        where = np.flatnonzero(miss)
        if where.size:
            # Emit in decode order: group ascending, and within a group the
            # shipped driver's own flat ordering (argsort-stable of the plane).
            order_key = group_index[where] * (PLANE + 1) + where
            where = where[np.argsort(order_key, kind="stable")]
            cols["frame"].append(np.full(where.size, frame, dtype=np.int16))
            cols["group"].append(group_index[where].astype(np.int16))
            cols["pos"].append(where.astype(np.int32))
            cols["arg"].append(arg[where].astype(np.int8))
            cols["token"].append(token[where].astype(np.int8))
            cols["base_class"].append(predicted[where].astype(np.int8))
            cols["boundary"].append(boundary[where].astype(np.int8))
            cols["ubin"].append(ubin[where].astype(np.int8))
            cols["run"].append(run_level[where].astype(np.int8))
            cols["agree1"].append(agree1[where].astype(np.int8))
            cols["agree2"].append(agree2[where].astype(np.int8))
            cols["prev1"].append(
                (prev1[where].astype(np.int64) if have_prev else np.full(where.size, UNKNOWN)).astype(np.int8)
            )
            cols["prev2"].append(
                (prev2[where].astype(np.int64) if have_prev else np.full(where.size, UNKNOWN)).astype(np.int8)
            )
            for name in offsets:
                cols[f"nb_{name}"].append(nb[name][where].astype(np.int8))
            rows.append(row64[where].copy())
            p_max_col.append(p_max[where].copy())
            one_minus_col.append(one_minus[where].copy())

        current = token.astype(np.uint8)
        if have_prev:
            run = np.where(current == prev1, np.minimum(run + 1, RUN_CAP), 0)
            prev2 = prev1
        prev1 = current.copy()
        have_prev = True

    payload = {k: np.concatenate(v) for k, v in cols.items() if v}
    payload["row64"] = np.concatenate(rows)
    payload["p_max"] = np.concatenate(p_max_col)
    payload["one_minus"] = np.concatenate(one_minus_col)

    # ALWAYS KEEP THE PAYLOAD: the sector itself, not only its measured length.
    sector = out_dir / f"miss_sector_n{frames}.npz"
    np.savez(sector, **payload)
    sha = hashlib.sha256(sector.read_bytes()).hexdigest()

    total_bits = hit_bits + being_miss_bits + within_miss_bits
    receipt = {
        "frames": frames,
        "positions": frames * PLANE,
        "hit_positions": hit_n,
        "miss_positions": miss_n,
        "miss_fraction_pct": 100.0 * miss_n / (hit_n + miss_n),
        "total_bytes": total_bits / 8.0,
        "hit_bytes": hit_bits / 8.0,
        "being_miss_bytes": being_miss_bits / 8.0,
        "within_miss_bytes": within_miss_bits / 8.0,
        "hit_event_bytes": (hit_bits + being_miss_bits) / 8.0,
        "elapsed_s": time.time() - started,
        "payload": str(sector),
        "payload_sha256": sha,
        "payload_bytes": sector.stat().st_size,
        "causal_fraction_pct": {
            name: 100.0 * float(mask.mean()) for name, mask in causal.items()
        },
    }
    (out_dir / f"MA1_EXTRACT_n{frames}.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default="/Volumes/APDataStore/pact/ddm_ma1/retained")
    args = ap.parse_args()

    receipt = extract(args.frames, Path(args.out))
    print(json.dumps({k: v for k, v in receipt.items() if k != "causal_fraction_pct"}, indent=2))
    print("causal fractions:", json.dumps(receipt["causal_fraction_pct"], indent=2))

    if args.frames == 600:
        print()
        print("CONTROLS vs ddm_fx1 §5 / ddm_fx2 §6")
        for label, got, want in (
            ("total (uncorrected HPAC)", receipt["total_bytes"], 112_109.57757858819),
            ("within-miss (the ceiling)", receipt["within_miss_bytes"], 1_247.19),
            ("hit-event", receipt["hit_event_bytes"], 110_862.39),
            ("being-a-miss", receipt["being_miss_bytes"], 77_241.46),
        ):
            print(f"  {label:28s} {got:14.5f}  target {want:14.5f}  delta {got - want:+10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
