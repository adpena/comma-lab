"""ddm_rr1 stage 2 - the free decode-side model at OPTIMAL FORM.

Stage 1 (``ddm_rr1_free_decode_model_headroom.py``) measured 1,448.79 B of free
headroom on the hv1 token stream and declared itself a LOWER bound in three
named ways.  This stage removes all three scope reductions and measures how far
the free axis actually reaches.

THE THREE REDUCTIONS STAGE 1 CARRIED, AND WHAT IS DONE ABOUT THEM HERE:

  1. Only the HIT/MISS binary event was re-modelled; the 4-ary refinement over
     the non-argmax classes was left entirely to HPAC.  That is the larger half
     of the stream: ddm_rc4 measured 70.011% of all code bits sitting in the
     223,694 positions where HPAC's argmax is wrong.  Stage 2 adds an adaptive
     rank-conditioned correction to that branch, built the same way and equally
     free.

  2. Run length was saturated at 4 frames.  Stage 2 carries 8, and adds the
     depth-2 agreement term ``prev2 == argmax`` explicitly.

  3. Statistics were only consulted at frame boundaries.  Stage 2 keeps that
     (it is the conservative choice and still exactly reproducible); the
     per-group variant is recorded as owed, not claimed.

EVERYTHING IS STILL FREE.  Each correction is a per-context log-odds shift
estimated from STRICTLY PAST FRAMES by a fixed generic rule, so encoder and
decoder rebuild identical tables from symbols both already possess.  Nothing is
transmitted; ``inflate.py`` is unsized and carries only generic constants.  The
decoded token field is bit-identical under every rung, so d_seg and d_pose do
not move: this is a PURE-RATE axis and the byte numbers need no scorer.

THE LADDER (each rung adds exactly one thing, so differences attribute):
    G0  shipped                          control, reproduces 112,109.578 B
    G1  hit, ctx (cls,ubin,t1,rl4,bnd)   reproduces stage 1's best rung
    G2  G1 with run length to 8          temporal depth
    G3  G2 + prev2==argmax               explicit depth-2 term
    G4  G3 + adaptive miss-branch        the 70%-of-bits branch
    G5  G4 with a richer miss context    miss-branch depth

POSITIVE CONTROL, fail-closed and identical to stage 1: the rebuilt
probabilities must reproduce ddm_hm1/ddm_dc1/ddm_rc4's
``corrected_quantized_logit_sha256``, ``corrected_cdf_input_sha256`` and the
112,109.57757858852 B cross-entropy.  G1 must additionally land within 1 B of
stage 1's 1,448.79 B or the two stages disagree and neither is quotable.

ALWAYS KEEP THE PAYLOAD: every correction table and per-frame bit ledger is
written to the SSD store with sha256 and byte count.

Axis: [macOS-CPU advisory / scorer-free EXACT byte measurement].  score_claim
false; no dispatch; the pointer is not moved by this file.
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
ARCHIVE_BYTES = 182759

HM1 = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")
LOGITS = HM1 / "base_logits_int16_n600.i16"
BOUNDARY = HM1 / "boundary_bucket_n600.u8"
TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)

STORE = Path("/Volumes/APDataStore/pact/ddm_rr1")
RETAINED = STORE / "retained" / "optimal_form"

EXPECT_XE_BYTES = 112109.57757858852
STAGE1_BEST_BYTES = 1448.7926359500998

N_FRAMES = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
NUM_CLASSES = 5
LOGIT_PRECISION = 8
TOKEN_STREAM_BYTES = 112110

SCORE_DENOMINATOR = 37_545_489
S_BASE = 0.15959729295498598
S_TARGET = 0.15
ADMISSION_DS = -3.5e-6  # the standing same-axis admission bar

U_STEP = 0.5
U_BINS = 64
KT_ALPHA = 0.5
MIN_COUNT = 32.0
MIN_COUNT_MISS = 32.0
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


def log_odds_delta(hits: np.ndarray, phat: np.ndarray, counts: np.ndarray, floor: float) -> np.ndarray:
    """Per-context log2-odds shift from strictly past observations."""
    empirical = np.clip((hits + KT_ALPHA) / (counts + 2.0 * KT_ALPHA), PROB_EPS, 1.0 - PROB_EPS)
    predicted = np.clip((phat + KT_ALPHA) / (counts + 2.0 * KT_ALPHA), PROB_EPS, 1.0 - PROB_EPS)
    out = np.log2(empirical / (1.0 - empirical)) - np.log2(predicted / (1.0 - predicted))
    np.clip(out, -DELTA_CLIP, DELTA_CLIP, out=out)
    out[counts < floor] = 0.0
    return out


class HitModel:
    """Adaptive correction of the argmax-hit event.  Zero transmitted bytes."""

    __slots__ = ("counts", "hits", "name", "phat", "size")

    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size
        self.hits = np.zeros(size, dtype=np.float64)
        self.counts = np.zeros(size, dtype=np.float64)
        self.phat = np.zeros(size, dtype=np.float64)

    def delta(self) -> np.ndarray:
        return log_odds_delta(self.hits, self.phat, self.counts, MIN_COUNT)

    def observe(self, ctx: np.ndarray, hit: np.ndarray, p_max: np.ndarray) -> None:
        np.add.at(self.counts, ctx, 1.0)
        np.add.at(self.hits, ctx, hit)
        np.add.at(self.phat, ctx, p_max)


class MissModel:
    """Adaptive correction of the 4-ary branch, keyed on (context, HPAC rank).

    HPAC's own conditional over the four non-argmax classes is kept and shifted
    in log-odds by a rank-conditioned amount learned from past misses.  With no
    observations the shift is zero, so the rung reduces to HPAC exactly.
    """

    __slots__ = ("counts", "hits", "name", "phat", "ranks", "size")

    def __init__(self, name: str, size: int, ranks: int = NUM_CLASSES - 1):
        self.name = name
        self.size = size
        self.ranks = ranks
        self.hits = np.zeros((size, ranks), dtype=np.float64)
        self.counts = np.zeros(size, dtype=np.float64)
        self.phat = np.zeros((size, ranks), dtype=np.float64)

    def delta(self) -> np.ndarray:
        counts = self.counts[:, None]
        return log_odds_delta(
            self.hits,
            self.phat,
            np.broadcast_to(counts, self.hits.shape),
            MIN_COUNT_MISS,
        )

    def observe(self, ctx: np.ndarray, rank_hit: np.ndarray, rank_prob: np.ndarray) -> None:
        np.add.at(self.counts, ctx, 1.0)
        np.add.at(self.hits, ctx, rank_hit)
        np.add.at(self.phat, ctx, rank_prob)


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
    if len(parts.token_stream) != TOKEN_STREAM_BYTES:
        raise SystemExit("token stream length mismatch - refusing")

    logits_mm = np.memmap(LOGITS, dtype=np.int16, mode="r", shape=(N_FRAMES, PLANE, 5))
    boundary_mm = np.memmap(BOUNDARY, dtype=np.uint8, mode="r", shape=(N_FRAMES, PLANE))
    tokens_mm = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N_FRAMES, HEIGHT, WIDTH))

    RL4, RL8 = 4, 8
    hit_models = {
        "G1_rl4": HitModel("G1_rl4", NUM_CLASSES * U_BINS * 2 * RL4 * 5),
        "G2_rl8": HitModel("G2_rl8", NUM_CLASSES * U_BINS * 2 * RL8 * 5),
        "G3_rl8_t2": HitModel("G3_rl8_t2", NUM_CLASSES * U_BINS * 2 * 2 * RL8 * 5),
    }
    miss_models = {
        "M1": MissModel("M1", NUM_CLASSES * 2 * RL8),
        "M2": MissModel("M2", NUM_CLASSES * 2 * RL8 * 5),
    }

    # rung -> (hit model key or None, miss model key or None)
    rungs = {
        "G1_hit_rl4": ("G1_rl4", None),
        "G2_hit_rl8": ("G2_rl8", None),
        "G3_hit_rl8_t2": ("G3_rl8_t2", None),
        "G4_G3_plus_miss_M1": ("G3_rl8_t2", "M1"),
        "G5_G3_plus_miss_M2": ("G3_rl8_t2", "M2"),
        "M2_miss_only": (None, "M2"),
    }
    rung_bits = dict.fromkeys(rungs, 0.0)
    rung_per_frame = {k: np.zeros(N_FRAMES, dtype=np.float64) for k in rungs}

    base_bits = 0.0
    base_per_frame = np.zeros(N_FRAMES, dtype=np.float64)
    hit_branch_bits = 0.0
    miss_branch_bits = 0.0
    total_miss = 0

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

        frame_base = float(-np.log2(np.maximum(p_actual, 1e-300)).sum())
        base_bits += frame_base
        base_per_frame[frame] = frame_base

        # Exact split of the shipped code length into its two branches.
        is_miss = hit == 0.0
        hit_branch_bits += float((-np.log2(np.where(is_miss, one_minus, p_max))).sum())
        miss_refine = np.zeros(PLANE, dtype=np.float64)
        if is_miss.any():
            miss_refine[is_miss] = -np.log2(np.maximum(p_actual[is_miss] / one_minus[is_miss], 1e-300))
        miss_branch_bits += float(miss_refine.sum())
        total_miss += int(is_miss.sum())

        # --- free causal contexts -----------------------------------------
        u = -np.log2(one_minus)
        ubin = np.clip((u / U_STEP).astype(np.int64), 0, U_BINS - 1)
        zero = np.zeros(PLANE, dtype=np.int64)
        t1 = (prev1.astype(np.int64) == predicted).astype(np.int64) if have1 else zero
        t2 = (prev2.astype(np.int64) == predicted).astype(np.int64) if have1 else zero
        rl4 = np.minimum(run, RL4 - 1)
        rl8 = np.minimum(run, RL8 - 1)
        logit_p = np.log2(np.maximum(p_max, PROB_EPS) / one_minus)

        hit_ctx = {
            "G1_rl4": (((predicted * U_BINS + ubin) * 2 + t1) * RL4 + rl4) * 5 + boundary,
            "G2_rl8": (((predicted * U_BINS + ubin) * 2 + t1) * RL8 + rl8) * 5 + boundary,
            "G3_rl8_t2": (((((predicted * U_BINS + ubin) * 2 + t1) * 2 + t2) * RL8 + rl8) * 5 + boundary),
        }
        miss_ctx = {
            "M1": (predicted * 2 + t1) * RL8 + rl8,
            "M2": ((predicted * 2 + t1) * RL8 + rl8) * 5 + boundary,
        }

        # HPAC's conditional over the non-argmax classes, in HPAC rank order.
        # Stable sort: a shipped receiver must break probability ties the same
        # way on every host, so the tie rule is part of the generic algorithm.
        order = np.argsort(-prob64, axis=1, kind="stable")  # rank 0 == argmax
        rest = order[:, 1:]  # (PLANE, 4) class ids by descending HPAC prob
        rest_prob = np.take_along_axis(prob64, rest, axis=1) / one_minus[:, None]
        # actual_rank is only MEANINGFUL at a miss: at a hit the true class is the
        # argmax, which `rest` excludes, so the comparison is all-False and argmax
        # returns 0.  Every consumer below masks with `is_miss`, so that value is
        # never read.  Stated explicitly because the index looks unguarded.
        actual_rank = np.argmax(rest == actual[:, None], axis=1)
        rank_onehot = np.zeros((PLANE, NUM_CLASSES - 1), dtype=np.float64)
        rank_onehot[rows, actual_rank] = 1.0

        q_hit_cache = {}
        for key, model in hit_models.items():
            ctx = hit_ctx[key]
            z = logit_p + model.delta()[ctx]
            q = np.clip(1.0 / (1.0 + np.exp2(-z)), PROB_EPS, 1.0 - PROB_EPS)
            q_hit_cache[key] = np.where(hit > 0.0, -np.log2(q), -np.log2(1.0 - q))

        q_miss_cache = {}
        for key, model in miss_models.items():
            ctx = miss_ctx[key]
            delta = model.delta()[ctx]  # (PLANE, 4)
            adj = rest_prob * np.exp2(delta)
            adj /= np.maximum(adj.sum(axis=1, keepdims=True), PROB_EPS)
            chosen = np.clip(adj[rows, actual_rank], PROB_EPS, 1.0)
            bits = np.zeros(PLANE, dtype=np.float64)
            bits[is_miss] = -np.log2(chosen[is_miss])
            q_miss_cache[key] = bits

        base_hit_bits = np.where(is_miss, -np.log2(one_minus), -np.log2(p_max))
        for name, (hkey, mkey) in rungs.items():
            hb = q_hit_cache[hkey] if hkey else base_hit_bits
            mb = q_miss_cache[mkey] if mkey else miss_refine
            value = float((hb + mb).sum())
            rung_bits[name] += value
            rung_per_frame[name][frame] = value

        for key, model in hit_models.items():
            model.observe(hit_ctx[key], hit, p_max)
        for key, model in miss_models.items():
            ctx = miss_ctx[key][is_miss]
            model.observe(ctx, rank_onehot[is_miss], rest_prob[is_miss])

        cur = np.asarray(tokens_mm[frame], dtype=np.uint8).reshape(-1)
        if have1:
            run = np.where(cur == prev1, np.minimum(run + 1, 255), 0)
            prev2 = prev1
        prev1 = cur
        have1 = True

        if frame % 100 == 0 or frame == limit - 1:
            best = min(rung_bits.values())
            print(
                f"  frame {frame:4d}/{limit}  base={base_bits / 8:,.1f}B  "
                f"best={best / 8:,.1f}B  {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    base_bytes = base_bits / 8.0

    results = []
    payloads = {}
    for name in rungs:
        b = rung_bits[name] / 8.0
        saved = base_bytes - b
        ds = -saved * 25.0 / SCORE_DENOMINATOR
        results.append(
            {
                "rung": name,
                "code_bytes": b,
                "bytes_saved": saved,
                "percent_of_token_stream": 100.0 * saved / base_bytes,
                "delta_S": ds,
                "percent_of_gap": 100.0 * (-ds) / (S_BASE - S_TARGET),
                "clears_admission_bar": bool(ds < ADMISSION_DS),
                "admission_margin_x": (ds / ADMISSION_DS) if ds < 0 else 0.0,
            }
        )
        path = RETAINED / f"bits_per_frame_{name}_{args.tag}.npy"
        np.save(path, rung_per_frame[name][:limit])
        blob = path.read_bytes()
        payloads[path.name] = {
            "path": str(path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        }

    for key, model in hit_models.items():
        path = RETAINED / f"delta_hit_{key}_{args.tag}.npy"
        np.save(path, model.delta())
        blob = path.read_bytes()
        payloads[path.name] = {
            "path": str(path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "warm_contexts": int((model.counts >= MIN_COUNT).sum()),
            "contexts": model.size,
        }
    for key, model in miss_models.items():
        path = RETAINED / f"delta_miss_{key}_{args.tag}.npy"
        np.save(path, model.delta())
        blob = path.read_bytes()
        payloads[path.name] = {
            "path": str(path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "warm_contexts": int((model.counts >= MIN_COUNT_MISS).sum()),
            "contexts": model.size,
        }
    path = RETAINED / f"bits_per_frame_base_{args.tag}.npy"
    np.save(path, base_per_frame[:limit])
    blob = path.read_bytes()
    payloads[path.name] = {
        "path": str(path),
        "bytes": len(blob),
        "sha256": sha256_bytes(blob),
    }

    by_rung = {r["rung"]: r for r in results}
    g1 = by_rung["G1_hit_rl4"]["bytes_saved"]
    control = {
        "cross_entropy_bytes": base_bytes,
        "expected_cross_entropy_bytes": EXPECT_XE_BYTES,
        "cross_entropy_matches": bool(full and abs(base_bytes - EXPECT_XE_BYTES) < 1e-3),
        "stage1_best_bytes": STAGE1_BEST_BYTES,
        "g1_reproduces_stage1": bool(full and abs(g1 - STAGE1_BEST_BYTES) < 1.0),
    }
    control["instrument_valid"] = bool(control["cross_entropy_matches"] and control["g1_reproduces_stage1"])

    best = max(results, key=lambda r: r["bytes_saved"])
    out = {
        "arm": "ddm_rr1",
        "stage": "2_free_model_optimal_form",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": ARCHIVE_BYTES,
        "frames": limit,
        "shipped_cross_entropy_bytes": base_bytes,
        "shipped_token_stream_bytes": TOKEN_STREAM_BYTES,
        "required_cut_bytes_strict": 14414,
        "branch_split": {
            "hit_miss_binary_bytes": hit_branch_bits / 8.0,
            "miss_refinement_bytes": miss_branch_bits / 8.0,
            "miss_positions": total_miss,
            "miss_refinement_share": miss_branch_bits / (hit_branch_bits + miss_branch_bits),
        },
        "positive_control": control,
        "rungs": results,
        "best": best,
        "attribution": {
            "run_length_4_to_8": by_rung["G2_hit_rl8"]["bytes_saved"] - g1,
            "prev2_agreement_term": by_rung["G3_hit_rl8_t2"]["bytes_saved"] - by_rung["G2_hit_rl8"]["bytes_saved"],
            "miss_branch_M1": by_rung["G4_G3_plus_miss_M1"]["bytes_saved"] - by_rung["G3_hit_rl8_t2"]["bytes_saved"],
            "miss_branch_M2": by_rung["G5_G3_plus_miss_M2"]["bytes_saved"] - by_rung["G3_hit_rl8_t2"]["bytes_saved"],
            "miss_branch_alone": by_rung["M2_miss_only"]["bytes_saved"],
        },
        "payloads": payloads,
        "elapsed_seconds": elapsed,
    }
    out_path = STORE / f"FREE_MODEL_OPTIMAL_FORM_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"control": control, "best": best, "attribution": out["attribution"]},
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
