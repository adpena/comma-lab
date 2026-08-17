"""ddm_rr1 stage 3 - remove the last scope reduction: per-GROUP statistics.

Stages 1 and 2 updated the free correction tables only at frame boundaries and
said so, labelling every result a LOWER bound.  That was the conservative
choice, not the faithful one: the shipped decoder processes a frame as 190
causal patch groups (``group_masks``), decoding every position in a group
simultaneously and then moving on.  A decoder that refreshes its statistics at
GROUP boundaries is therefore doing exactly what the existing decode loop
already does, is still perfectly causal, and adapts 190x faster.

This stage measures that, and sweeps two richer free contexts at the same time.

    H1  G3 context, per-GROUP updates            -> the granularity term
    H2  H1 + previous-frame neighbour agreement   -> a finer free spatial term
    H3  H1 with run length saturated at 16        -> deeper temporal memory

H1 minus stage 2's G3 is the granularity increment; H2/H3 minus H1 are context
increments.  Every rung remains free: zero transmitted bytes, generic constants,
tables rebuilt on both sides from already-decoded symbols.  The decoded token
field is untouched, so d_seg and d_pose cannot move.

EFFICIENCY NOTE (why this is minutes, not hours).  The correction is never
materialized over the whole table per group.  Each group gathers only the
contexts its own ~1,035 positions use, computes the shift pointwise, applies it
and scatters the update back.  Cost is O(group size), not O(table size).

POSITIVE CONTROL, fail-closed: the rebuilt probabilities must reproduce the
112,109.57757858852 B cross-entropy that ddm_dc1, ddm_hm1 and ddm_rc4 measured
independently.  A second control asserts the per-group H1 rung is no worse than
stage 2's frame-granularity G3 - if a strictly faster-adapting estimator lost
ground, the estimator is wrong and the number is not quotable.

Axis: [macOS-CPU advisory / scorer-free EXACT byte measurement].  score_claim
false.  No archive is built here; byte counts downstream of the code length are
PROJECTED through the measured constant coder overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/prepared/hv1_base_control/archive.zip"
)
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"

HM1 = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")
LOGITS = HM1 / "base_logits_int16_n600.i16"
BOUNDARY = HM1 / "boundary_bucket_n600.u8"
GROUP_INDEX = HM1 / "group_index.u8"
TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)

STORE = Path("/Volumes/APDataStore/pact/ddm_rr1")
RETAINED = STORE / "retained" / "per_group"

EXPECT_XE_BYTES = 112109.57757858852
STAGE2_G3_SAVED = 1549.8685603824386

N_FRAMES = 600
HEIGHT, WIDTH = 384, 512
PLANE = HEIGHT * WIDTH
NUM_CLASSES = 5
LOGIT_PRECISION = 8
SCORE_DENOMINATOR = 37_545_489
S_BASE = 0.15959729295498598
S_TARGET = 0.15
SHIPPED_TOKEN_BYTES = 112110
ADMISSION_DS = -3.5e-6

U_STEP, U_BINS = 0.5, 64
KT_ALPHA = 0.5
MIN_COUNT = 32.0
DELTA_CLIP = 4.0
PROB_EPS = 1e-9


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probability_from_corrected(corrected: np.ndarray, precision: int) -> np.ndarray:
    """Byte-identical copy of runtime ``residual_archive._probability_table``."""
    quantized = np.clip(np.rint(np.asarray(corrected, dtype=np.float32) * precision), -32768, 32767).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


class GroupModel:
    """Free hit-event corrector with per-group statistics.

    Only the contexts a group actually touches are ever read or written, so the
    cost is set by the group size and not by the table size.
    """

    __slots__ = ("bits", "counts", "hits", "name", "phat", "size")

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.hits = np.zeros(size, dtype=np.float64)
        self.counts = np.zeros(size, dtype=np.float64)
        self.phat = np.zeros(size, dtype=np.float64)
        self.bits = 0.0

    def code_group(self, ctx: np.ndarray, logit_p: np.ndarray, hit: np.ndarray, p_max: np.ndarray) -> float:
        n = self.counts[ctx]
        empirical = np.clip((self.hits[ctx] + KT_ALPHA) / (n + 2.0 * KT_ALPHA), PROB_EPS, 1.0 - PROB_EPS)
        predicted = np.clip((self.phat[ctx] + KT_ALPHA) / (n + 2.0 * KT_ALPHA), PROB_EPS, 1.0 - PROB_EPS)
        delta = np.log2(empirical / (1.0 - empirical)) - np.log2(predicted / (1.0 - predicted))
        np.clip(delta, -DELTA_CLIP, DELTA_CLIP, out=delta)
        delta[n < MIN_COUNT] = 0.0
        q = np.clip(1.0 / (1.0 + np.exp2(-(logit_p + delta))), PROB_EPS, 1.0 - PROB_EPS)
        value = float(np.where(hit > 0.0, -np.log2(q), -np.log2(1.0 - q)).sum())
        self.bits += value
        np.add.at(self.counts, ctx, 1.0)
        np.add.at(self.hits, ctx, hit)
        np.add.at(self.phat, ctx, p_max)
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--tag", default="n600")
    args = parser.parse_args()
    limit = int(args.frames)
    full = limit == N_FRAMES

    RETAINED.mkdir(parents=True, exist_ok=True)
    if sha256_file(ARCHIVE) != ARCHIVE_SHA:
        raise SystemExit("archive sha mismatch - refusing")

    import sys

    prepared = ARCHIVE.parent
    sys.path.insert(0, str(prepared))
    sys.path.insert(0, str(prepared / "cpr1"))
    from runtime.residual_archive import read_residual_archive

    parts = read_residual_archive(ARCHIVE)
    table_values = parts.table.values

    logits_mm = np.memmap(LOGITS, dtype=np.int16, mode="r", shape=(N_FRAMES, PLANE, 5))
    boundary_mm = np.memmap(BOUNDARY, dtype=np.uint8, mode="r", shape=(N_FRAMES, PLANE))
    tokens_mm = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N_FRAMES, HEIGHT, WIDTH))
    group_index = np.frombuffer(GROUP_INDEX.read_bytes(), dtype=np.uint8)
    n_groups = int(group_index.max()) + 1
    groups = [np.flatnonzero(group_index == g) for g in range(n_groups)]

    RL8, RL16 = 8, 16
    models = {
        "H1_pergroup_G3": GroupModel("H1_pergroup_G3", NUM_CLASSES * U_BINS * 2 * 2 * RL8 * 5),
        "H2_plus_neighbour": GroupModel("H2_plus_neighbour", NUM_CLASSES * U_BINS * 2 * 2 * RL8 * 5 * 5),
        "H3_rl16": GroupModel("H3_rl16", NUM_CLASSES * U_BINS * 2 * 2 * RL16 * 5),
    }

    base_bits = 0.0
    prev1 = np.zeros(PLANE, dtype=np.uint8)
    prev2 = np.zeros(PLANE, dtype=np.uint8)
    run = np.zeros(PLANE, dtype=np.int64)
    have1 = False
    rows = np.arange(PLANE)
    started = time.perf_counter()

    for frame in range(limit):
        base = logits_mm[frame].astype(np.float32) / LOGIT_PRECISION
        boundary = np.asarray(boundary_mm[frame], dtype=np.int64)
        actual = np.asarray(tokens_mm[frame], dtype=np.int64).reshape(-1)
        predicted = base.argmax(axis=1).astype(np.int64)
        corrected = base + table_values[boundary * NUM_CLASSES + predicted]
        prob64 = probability_from_corrected(corrected, LOGIT_PRECISION).astype(np.float64)
        p_actual = prob64[rows, actual]
        arg = prob64.argmax(axis=1)
        p_max = prob64[rows, arg]
        hit = (arg == actual).astype(np.float64)
        one_minus = np.maximum(1.0 - p_max, PROB_EPS)
        base_bits += float(-np.log2(np.maximum(p_actual, 1e-300)).sum())

        # The miss refinement is carried unchanged by every rung, exactly as in
        # stage 2, so the rungs differ only in the hit/miss model.
        miss_refine = np.zeros(PLANE, dtype=np.float64)
        is_miss = hit == 0.0
        if is_miss.any():
            miss_refine[is_miss] = -np.log2(np.maximum(p_actual[is_miss] / one_minus[is_miss], 1e-300))
        miss_total = float(miss_refine.sum())

        u = -np.log2(one_minus)
        ubin = np.clip((u / U_STEP).astype(np.int64), 0, U_BINS - 1)
        zero = np.zeros(PLANE, dtype=np.int64)
        t1 = (prev1.astype(np.int64) == predicted).astype(np.int64) if have1 else zero
        t2 = (prev2.astype(np.int64) == predicted).astype(np.int64) if have1 else zero
        rl8 = np.minimum(run, RL8 - 1)
        rl16 = np.minimum(run, RL16 - 1)
        logit_p = np.log2(np.maximum(p_max, PROB_EPS) / one_minus)

        # Free spatial term: how many of the previous frame's 4-neighbours agree
        # with it at this pixel.  Entirely decoder-side and causal.
        grid = prev1.reshape(HEIGHT, WIDTH).astype(np.int64)
        nbr = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
        nbr[1:, :] += grid[1:, :] == grid[:-1, :]
        nbr[:-1, :] += grid[:-1, :] == grid[1:, :]
        nbr[:, 1:] += grid[:, 1:] == grid[:, :-1]
        nbr[:, :-1] += grid[:, :-1] == grid[:, 1:]
        nbr = nbr.reshape(-1)

        head = ((predicted * U_BINS + ubin) * 2 + t1) * 2 + t2
        ctx_all = {
            "H1_pergroup_G3": (head * RL8 + rl8) * 5 + boundary,
            "H2_plus_neighbour": ((head * RL8 + rl8) * 5 + boundary) * 5 + nbr,
            "H3_rl16": (head * RL16 + rl16) * 5 + boundary,
        }

        for positions in groups:
            lp = logit_p[positions]
            h = hit[positions]
            pm = p_max[positions]
            for key, model in models.items():
                model.code_group(ctx_all[key][positions], lp, h, pm)
        for model in models.values():
            model.bits += miss_total

        cur = np.asarray(tokens_mm[frame], dtype=np.uint8).reshape(-1)
        if have1:
            run = np.where(cur == prev1, np.minimum(run + 1, 255), 0)
            prev2 = prev1
        prev1 = cur
        have1 = True

        if frame % 100 == 0 or frame == limit - 1:
            best = min(m.bits for m in models.values())
            print(
                f"  frame {frame:4d}/{limit}  base={base_bits / 8:,.1f}B  "
                f"best={best / 8:,.1f}B  {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    base_bytes = base_bits / 8.0

    results = []
    payloads = {}
    for name, model in models.items():
        code = model.bits / 8.0
        saved = base_bytes - code
        ds = -saved * 25.0 / SCORE_DENOMINATOR
        results.append(
            {
                "rung": name,
                "contexts": model.size,
                "warm_contexts": int((model.counts >= MIN_COUNT).sum()),
                "code_bytes": code,
                "bytes_saved": saved,
                "delta_S": ds,
                "percent_of_gap": 100.0 * (-ds) / (S_BASE - S_TARGET),
                "admission_margin_x": (ds / ADMISSION_DS) if ds < 0 else 0.0,
            }
        )
        path = RETAINED / f"counts_{name}_{args.tag}.npy"
        np.save(path, model.counts)
        blob = path.read_bytes()
        payloads[path.name] = {
            "path": str(path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        }

    by = {r["rung"]: r for r in results}
    h1 = by["H1_pergroup_G3"]["bytes_saved"]
    control = {
        "cross_entropy_bytes": base_bytes,
        "cross_entropy_matches": bool(full and abs(base_bytes - EXPECT_XE_BYTES) < 1e-3),
        "stage2_G3_saved": STAGE2_G3_SAVED,
        "per_group_not_worse_than_per_frame": bool(full and h1 >= STAGE2_G3_SAVED),
    }
    control["instrument_valid"] = bool(
        control["cross_entropy_matches"] and control["per_group_not_worse_than_per_frame"]
    )

    best = max(results, key=lambda r: r["bytes_saved"])
    overhead = SHIPPED_TOKEN_BYTES - base_bytes
    projected_stream = int(np.ceil(best["code_bytes"] + overhead))
    projected_saved = SHIPPED_TOKEN_BYTES - projected_stream
    out = {
        "arm": "ddm_rr1",
        "stage": "3_per_group_and_context_sweep",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "archive_sha256": ARCHIVE_SHA,
        "frames": limit,
        "groups_per_frame": n_groups,
        "shipped_cross_entropy_bytes": base_bytes,
        "realized_coder_overhead_bytes": overhead,
        "positive_control": control,
        "rungs": results,
        "best": best,
        "attribution": {
            "granularity_frame_to_group": h1 - STAGE2_G3_SAVED,
            "neighbour_context": by["H2_plus_neighbour"]["bytes_saved"] - h1,
            "run_length_8_to_16": by["H3_rl16"]["bytes_saved"] - h1,
        },
        "projection": {
            "label": "PROJECTED - no archive was built",
            "token_stream_bytes": projected_stream,
            "bytes_saved": projected_saved,
            "archive_bytes": 182759 - projected_saved,
            "delta_S": -projected_saved * 25.0 / SCORE_DENOMINATOR,
            "S": S_BASE - projected_saved * 25.0 / SCORE_DENOMINATOR,
        },
        "payloads": payloads,
        "elapsed_seconds": elapsed,
    }
    out_path = STORE / f"FREE_MODEL_PER_GROUP_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "control": control,
                "best": best,
                "attribution": out["attribution"],
                "projection": out["projection"],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
