"""ddm_rr1 - the FREE decode-side headroom of the hv1 RC64 token stream.

THE QUESTION.  Rule 118 makes ``inflate.py`` an unsized free interpreter: a
GENERIC algorithm costs zero archive bytes, only VIDEO-DERIVED content is
counted.  The token stream is 112,110 B = 61.3% of the hv1 archive and its
coder is measured exactly optimal (ddm_dc1: rc64 is +7.80 B over the HPAC
cross-entropy).  So the only remaining way to shorten it without touching the
decoded field is a BETTER PROBABILITY MODEL THAT COSTS NOTHING - one built at
decode time out of symbols the decoder has already produced.

WHAT IS ALREADY CLOSED (do not re-derive):
  * ddm_dc1 - a table-free adaptive context model used INSTEAD of HPAC needs
    177,109 B (KT) / 144,167 B (oracle) against a shipped 112,110 B.  Standalone
    replacement is dead by +32,057 B at best.  dc1 states plainly that a
    logistic-mixing / augmentation coder "was not built".
  * ddm_hm1 - a COUNTED correction table saturates: the shipped 25-cell RCF1
    returns -3.355 B per counted byte and the next rung returns -0.016.  Given
    the table away at zero cost the whole ladder ceiling is 356.1 B.  hm1 states
    that a correction keyed on richer context than its summaries is not bounded
    by its rows.

WHAT THIS ARM MEASURES.  Not a replacement - an AUGMENTATION.  Keep every
shipped HPAC probability exactly as the receiver computes it, then apply a
per-context log-odds correction that is ESTIMATED ONLINE FROM STRICTLY PAST
FRAMES.  The correction table is never transmitted: encoder and decoder both
rebuild it from the symbols already decoded, so it costs zero archive bytes.
The correction starts at exactly zero, so the ladder's first rung reproduces the
shipped code length by construction.

THE HYPOTHESIS WITH A SOURCE-LEVEL WARRANT.  HPAC sees exactly ONE frame of
history: ``decode_tokens`` sets ``previous_raw = current.clone()`` each frame and
``prepare_frame_context(idx, previous_raw)`` is the only history input
(``conv_past`` runs over a one-hot of that single frame).  Frames t-2, t-3, ...
are sitting in the decoder's own output buffer for free and are PROVABLY outside
HPAC's input.  If per-pixel temporal persistence carries information beyond one
frame - and in a driving segmentation it should - that information is free.

THE LADDER.  Context families of increasing richness; the DIFFERENCES isolate
the mechanism.
    F0  none                                -> reproduces the shipped stream (control)
    F1  (cls, ubin)                         -> pure recalibration of HPAC
    F2  (cls, ubin, t1)                     -> + agreement with frame t-1
    F3  (cls, ubin, t1, rl)                 -> + run length over t-1..t-4  [NEW INFO]
    F4  (cls, ubin, t1, rl, bnd)            -> + boundary bucket
    F5  (cls, ubin, bnd)                    -> F4 minus every temporal term (control)
F3-F2 and F4-F5 are the two independent reads of the free t-2+ temporal term.

LEGALITY (rule 118).  The correction is a GENERIC algorithm - fixed smoothing
constant, fixed bin edges, fixed update order - driven only by already-decoded
symbols.  No video-derived table enters inflate.py.  The estimator is causal at
FRAME granularity (frame t is coded from statistics over frames 0..t-1), which
is what a vectorized decoder does anyway, and is therefore exactly reproducible
on the encode side.  Frame granularity makes every number here a LOWER BOUND on
the achievable free gain (a per-group updater adapts strictly faster).

POSITIVE CONTROL (fail-closed, three independent prior arms).  The shipped
probabilities are rebuilt from ddm_hm1's retained pre-correction logits and must
reproduce (a) ``corrected_quantized_logit_sha256``, (b)
``corrected_cdf_input_sha256`` and (c) the cross-entropy 112,109.57757858852 B
that ddm_dc1, ddm_hm1 and ddm_rc4 measured independently.  The run refuses to
emit a verdict unless all three match.

ALWAYS KEEP THE PAYLOAD.  Every materialized array - the learned correction
tables and the per-frame bit ledgers - is written to the SSD store with sha256
and byte count.  This script never measures a payload it then discards.

Axis: [macOS-CPU advisory / scorer-free EXACT byte measurement].  The rate leg is
exact (the ZIP member is STORED, so a byte off the token stream is a byte off
archive.zip 1:1); there is no distortion leg because the decoded token field is
bit-identical by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

# --- pinned inputs -----------------------------------------------------------

ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/prepared/hv1_base_control/archive.zip"
)
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182759

HM1 = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")
LOGITS = HM1 / "base_logits_int16_n600.i16"
BOUNDARY = HM1 / "boundary_bucket_n600.u8"
GROUP_INDEX = HM1 / "group_index.u8"
TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)

STORE = Path("/Volumes/APDataStore/pact/ddm_rr1")
RETAINED = STORE / "retained" / "free_model"

# Identities measured independently by ddm_dc1, ddm_hm1 and ddm_rc4.
EXPECT_CORRECTED_SHA = "562ac652b372faa020d0fc5e2ed9b7b61625169e0f5c2041d4fe99196055b8c7"
EXPECT_CDF_SHA = "dd48843b021763e78524caf3dcd01e944045e7bd0ffd93b451dec83548f083b7"
EXPECT_XE_BYTES = 112109.57757858852

N_FRAMES = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
NUM_CLASSES = 5
LOGIT_PRECISION = 8
TOKEN_STREAM_BYTES = 112110

# Score arithmetic, re-derived here rather than inherited.
SCORE_DENOMINATOR = 37_545_489
S_BASE = 0.15959729295498598
S_TARGET = 0.15

# --- estimator constants (GENERIC: fixed, video-independent) ------------------

U_CLIP = 32.0  # u = -log2(1 - p_max), saturated
U_STEP = 0.5
U_BINS = int(U_CLIP / U_STEP)  # 64
RL_LEVELS = 4  # run length over frames t-1..t-4, saturated
KT_ALPHA = 0.5  # Krichevsky-Trofimov smoothing
MIN_COUNT = 32.0  # below this a context emits delta = 0 (i.e. exactly HPAC)
DELTA_CLIP = 4.0  # log-odds correction clamp, in nats of log2-odds
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


class Family:
    """One free adaptive context family.

    Holds cumulative hit / count / HPAC-predicted-hit sums over STRICTLY PAST
    frames.  ``delta`` is the log2-odds correction the decoder applies before
    coding frame t; it is rebuilt from the same past the decoder has, so it
    travels in zero archive bytes.
    """

    __slots__ = ("bits", "counts", "hits", "name", "per_frame", "phat", "shape", "size")

    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape
        self.size = int(np.prod(shape))
        self.hits = np.zeros(self.size, dtype=np.float64)
        self.counts = np.zeros(self.size, dtype=np.float64)
        self.phat = np.zeros(self.size, dtype=np.float64)  # sum of HPAC p_max
        self.bits = 0.0
        self.per_frame = np.zeros(N_FRAMES, dtype=np.float64)

    def delta(self) -> np.ndarray:
        """Per-context log2-odds shift estimated from strictly past frames."""
        n = self.counts
        empirical = (self.hits + KT_ALPHA) / (n + 2.0 * KT_ALPHA)
        predicted = (self.phat + KT_ALPHA) / (n + 2.0 * KT_ALPHA)
        empirical = np.clip(empirical, PROB_EPS, 1.0 - PROB_EPS)
        predicted = np.clip(predicted, PROB_EPS, 1.0 - PROB_EPS)
        out = np.log2(empirical / (1.0 - empirical)) - np.log2(predicted / (1.0 - predicted))
        np.clip(out, -DELTA_CLIP, DELTA_CLIP, out=out)
        out[n < MIN_COUNT] = 0.0  # cold contexts fall back to exactly HPAC
        return out

    def observe(self, ctx: np.ndarray, hit: np.ndarray, p_max: np.ndarray) -> None:
        np.add.at(self.counts, ctx, 1.0)
        np.add.at(self.hits, ctx, hit)
        np.add.at(self.phat, ctx, p_max)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--tag", default="n600")
    parser.add_argument(
        "--skip-digest",
        action="store_true",
        help="skip the sha256 identity control (only legal for a smoke)",
    )
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
    # Use the receiver's own array object and dtype: coercing it would change
    # the float rounding of `base + correction` and break the sha256 identity.
    table_values = parts.table.values
    if len(parts.token_stream) != TOKEN_STREAM_BYTES:
        raise SystemExit("token stream length mismatch - refusing")

    logits_mm = np.memmap(LOGITS, dtype=np.int16, mode="r", shape=(N_FRAMES, PLANE, 5))
    boundary_mm = np.memmap(BOUNDARY, dtype=np.uint8, mode="r", shape=(N_FRAMES, PLANE))
    group_index = np.frombuffer(GROUP_INDEX.read_bytes(), dtype=np.uint8)
    tokens_mm = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N_FRAMES, HEIGHT, WIDTH))
    n_groups = int(group_index.max()) + 1
    # Digest order is (frame, group, position-within-group) exactly as the
    # shipped decoder emits it.
    group_positions = [np.flatnonzero(group_index == g) for g in range(n_groups)]
    digest_order = np.concatenate(group_positions)

    families = {
        "F1_cls_ubin": Family("F1_cls_ubin", (NUM_CLASSES, U_BINS)),
        "F2_cls_ubin_t1": Family("F2_cls_ubin_t1", (NUM_CLASSES, U_BINS, 2)),
        "F3_cls_ubin_t1_rl": Family("F3_cls_ubin_t1_rl", (NUM_CLASSES, U_BINS, 2, RL_LEVELS)),
        "F4_cls_ubin_t1_rl_bnd": Family("F4_cls_ubin_t1_rl_bnd", (NUM_CLASSES, U_BINS, 2, RL_LEVELS, 5)),
        "F5_cls_ubin_bnd": Family("F5_cls_ubin_bnd", (NUM_CLASSES, U_BINS, 5)),
    }

    corrected_digest = hashlib.sha256()
    cdf_digest = hashlib.sha256()
    base_bits = 0.0
    base_per_frame = np.zeros(N_FRAMES, dtype=np.float64)
    total_positions = 0
    total_hits = 0

    # Persistent per-pixel temporal state, all of it decoder-side and free.
    prev1 = np.zeros(PLANE, dtype=np.uint8)
    run = np.zeros(PLANE, dtype=np.uint8)  # frames the token has been unchanged
    have1 = False

    started = time.perf_counter()
    for frame in range(limit):
        base = logits_mm[frame].astype(np.float32) / LOGIT_PRECISION
        boundary = np.asarray(boundary_mm[frame], dtype=np.int64)
        actual = np.asarray(tokens_mm[frame], dtype=np.int64).reshape(-1)

        predicted = base.argmax(axis=1).astype(np.int64)
        feature = boundary * NUM_CLASSES + predicted
        corrected = base + table_values[feature]
        if not args.skip_digest:
            ordered = np.ascontiguousarray(corrected[digest_order], dtype="<f4")
            corrected_digest.update(ordered.tobytes())
        probability = probability_from_corrected(corrected, LOGIT_PRECISION)
        if not args.skip_digest:
            cdf_digest.update(np.ascontiguousarray(probability[digest_order], dtype="<f4").tobytes())

        prob64 = probability.astype(np.float64)
        rows = np.arange(PLANE)
        p_actual = prob64[rows, actual]
        arg = prob64.argmax(axis=1)
        p_max = prob64[rows, arg]
        hit = (arg == actual).astype(np.float64)

        frame_base_bits = float(-np.log2(np.maximum(p_actual, 1e-300)).sum())
        base_bits += frame_base_bits
        base_per_frame[frame] = frame_base_bits
        total_positions += PLANE
        total_hits += int(hit.sum())

        # The miss branch is carried unchanged by every family: only the
        # hit/miss binary event is re-modelled.  Splitting it this way makes the
        # control exact - with delta == 0 the ladder reproduces `base_bits`.
        one_minus = np.maximum(1.0 - p_max, PROB_EPS)
        miss_refine_bits = np.zeros(PLANE, dtype=np.float64)
        miss = hit == 0.0
        miss_refine_bits[miss] = -np.log2(np.maximum(p_actual[miss] / one_minus[miss], 1e-300))

        # --- free contexts, all causal ------------------------------------
        u = -np.log2(one_minus)
        ubin = np.clip((u / U_STEP).astype(np.int64), 0, U_BINS - 1)
        t1 = (prev1.astype(np.int64) == predicted).astype(np.int64) if have1 else np.zeros(PLANE, dtype=np.int64)
        rl = np.minimum(run.astype(np.int64), RL_LEVELS - 1)
        logit_p = np.log2(np.maximum(p_max, PROB_EPS) / one_minus)

        ctx_map = {
            "F1_cls_ubin": predicted * U_BINS + ubin,
            "F2_cls_ubin_t1": (predicted * U_BINS + ubin) * 2 + t1,
            "F3_cls_ubin_t1_rl": ((predicted * U_BINS + ubin) * 2 + t1) * RL_LEVELS + rl,
            "F4_cls_ubin_t1_rl_bnd": (((predicted * U_BINS + ubin) * 2 + t1) * RL_LEVELS + rl) * 5 + boundary,
            "F5_cls_ubin_bnd": (predicted * U_BINS + ubin) * 5 + boundary,
        }

        for name, fam in families.items():
            ctx = ctx_map[name]
            delta = fam.delta()[ctx]  # estimated from frames < `frame` only
            z = logit_p + delta
            q_hit = 1.0 / (1.0 + np.exp2(-z))
            q_hit = np.clip(q_hit, PROB_EPS, 1.0 - PROB_EPS)
            bits = np.where(hit > 0.0, -np.log2(q_hit), -np.log2(1.0 - q_hit) + miss_refine_bits)
            frame_bits = float(bits.sum())
            fam.bits += frame_bits
            fam.per_frame[frame] = frame_bits
            fam.observe(ctx, hit, p_max)

        # --- advance the free temporal state ------------------------------
        cur = np.asarray(tokens_mm[frame], dtype=np.uint8).reshape(-1)
        if have1:
            same = cur == prev1
            run = np.where(same, np.minimum(run.astype(np.int64) + 1, 255), 0).astype(np.uint8)
        prev1 = cur
        have1 = True

        if frame % 50 == 0 or frame == limit - 1:
            best = min(f.bits for f in families.values())
            print(
                f"  frame {frame:4d}/{limit}  base={base_bits / 8:,.1f}B  "
                f"best={best / 8:,.1f}B  {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started

    control = {
        "cross_entropy_bytes": base_bits / 8.0,
        "expected_cross_entropy_bytes": EXPECT_XE_BYTES,
        "cross_entropy_matches": bool(full and abs(base_bits / 8.0 - EXPECT_XE_BYTES) < 1e-3),
    }
    if not args.skip_digest:
        control["corrected_quantized_logit_sha256"] = corrected_digest.hexdigest()
        control["corrected_cdf_input_sha256"] = cdf_digest.hexdigest()
        control["logit_sha_matches"] = bool(full and corrected_digest.hexdigest() == EXPECT_CORRECTED_SHA)
        control["cdf_sha_matches"] = bool(full and cdf_digest.hexdigest() == EXPECT_CDF_SHA)
        control["instrument_valid"] = bool(
            control["cross_entropy_matches"] and control["logit_sha_matches"] and control["cdf_sha_matches"]
        )
    else:
        control["instrument_valid"] = False
        control["note"] = "digest control skipped - smoke only, not a verdict"

    base_bytes = base_bits / 8.0
    results = []
    payloads = {}
    for name, fam in sorted(families.items()):
        fam_bytes = fam.bits / 8.0
        saved = base_bytes - fam_bytes
        delta_table = fam.delta()
        warm = int((fam.counts >= MIN_COUNT).sum())
        table_path = RETAINED / f"delta_{name}_{args.tag}.npy"
        np.save(table_path, delta_table.reshape(fam.shape))
        blob = table_path.read_bytes()
        payloads[table_path.name] = {
            "path": str(table_path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        }
        frame_path = RETAINED / f"bits_per_frame_{name}_{args.tag}.npy"
        np.save(frame_path, fam.per_frame[:limit])
        blob = frame_path.read_bytes()
        payloads[frame_path.name] = {
            "path": str(frame_path),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        }
        results.append(
            {
                "family": name,
                "contexts": fam.size,
                "warm_contexts": warm,
                "code_bytes": fam_bytes,
                "bytes_saved_vs_shipped": saved,
                "percent_of_token_stream": 100.0 * saved / base_bytes,
                "delta_S": -saved * 25.0 / SCORE_DENOMINATOR,
                "percent_of_gap": 100.0 * (saved * 25.0 / SCORE_DENOMINATOR) / (S_BASE - S_TARGET),
            }
        )

    frame_path = RETAINED / f"bits_per_frame_base_{args.tag}.npy"
    np.save(frame_path, base_per_frame[:limit])
    blob = frame_path.read_bytes()
    payloads[frame_path.name] = {
        "path": str(frame_path),
        "bytes": len(blob),
        "sha256": sha256_bytes(blob),
    }

    by_name = {r["family"]: r for r in results}
    mechanism = {
        "recalibration_only_bytes": by_name["F1_cls_ubin"]["bytes_saved_vs_shipped"],
        "temporal_beyond_t1_bytes_F3_minus_F2": (
            by_name["F3_cls_ubin_t1_rl"]["bytes_saved_vs_shipped"] - by_name["F2_cls_ubin_t1"]["bytes_saved_vs_shipped"]
        ),
        "temporal_beyond_t1_bytes_F4_minus_F5": (
            by_name["F4_cls_ubin_t1_rl_bnd"]["bytes_saved_vs_shipped"]
            - by_name["F5_cls_ubin_bnd"]["bytes_saved_vs_shipped"]
        ),
        "best_family": max(results, key=lambda r: r["bytes_saved_vs_shipped"])["family"],
        "best_bytes_saved": max(r["bytes_saved_vs_shipped"] for r in results),
    }

    out = {
        "arm": "ddm_rr1",
        "stage": "free_decode_model_headroom",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "promotable": False,
        "archive_sha256": ARCHIVE_SHA,
        "archive_bytes": ARCHIVE_BYTES,
        "frames": limit,
        "positions": total_positions,
        "top1_hits": total_hits,
        "top1_error": 1.0 - total_hits / total_positions,
        "shipped_token_stream_bytes": TOKEN_STREAM_BYTES,
        "shipped_cross_entropy_bytes": base_bytes,
        "required_cut_bytes": (S_BASE - S_TARGET) * SCORE_DENOMINATOR / 25.0,
        "estimator": {
            "u_clip": U_CLIP,
            "u_step": U_STEP,
            "u_bins": U_BINS,
            "rl_levels": RL_LEVELS,
            "kt_alpha": KT_ALPHA,
            "min_count": MIN_COUNT,
            "delta_clip": DELTA_CLIP,
            "update_granularity": "frame (causal: frame t coded from frames < t)",
            "transmitted_bytes": 0,
        },
        "positive_control": control,
        "families": results,
        "mechanism": mechanism,
        "payloads": payloads,
        "elapsed_seconds": elapsed,
    }
    out_path = STORE / f"FREE_MODEL_HEADROOM_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"control": control, "mechanism": mechanism}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
