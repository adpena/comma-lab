"""ddm_ma2 — reconcile the two circulating Lane byte figures on the dx2 token stream.

Two numbers circulate for "Lane's share of the shipped token stream" and one
committed memo calls them "two adjacent measurements of the same concentration"
(``ddm_af1_address_free_class_law_20260824.md`` :451-453):

* ``ddm_tba1``:74   -> **38,649.8 B**, 33.9700% of stream bits, area 0.5858%
* ``ddm_ld1``:44    -> **38,182.996184 B**, 33.5597511452%, area 0.5855594211%

This module tests the hypothesis that they are NOT two measurements of one
object but the cost of **two different index sets** over the same per-position
cost field:

* ``tba1`` sums cost over positions whose **DECODED TOKEN** is Lane.
* ``ld1``  sums cost over positions whose **DALI GT ARGMAX** is Lane.

Those sets differ exactly on the field/GT disagreements, which ``tba1``:84-86
notes in prose but never quantifies.  If the hypothesis holds, the difference
decomposes exactly as

    cost(decoded==Lane) - cost(GT==Lane)
        = cost(decoded==Lane & GT!=Lane) - cost(GT==Lane & decoded!=Lane)

and the two set cardinalities must reproduce ``ld1``:62's independently measured
intersection of **688,847** both-Lane positions.

Nothing here is a rate claim.  Every number is a sum over a retained coder-cost
field, per ``ddm_tba1`` §8: *"the bit map is a MAP, not a PRICE."*  No encoder is
run, no scorer is fired, no archive is mutated.

Custody is verified by SHA-256 before any arithmetic; a mismatch refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- pinned inputs, all read-only -----------------------------------------
# Charter law: never trust a path, verify the digest.  These three SHA-256
# values are transcribed from the committed memos that produced the files, so a
# silent re-materialization of any input cannot reach the arithmetic below.
COST_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation"
    "/measurement_v1/retained/fields/position_rc64_frequency_cost_bits.f64le.bin"
)
COST_SHA = "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"

TOKENS_PATH = Path(
    "/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation"
    "/measurement_v1/retained/fields/decoded_tokens_instrumented.u8"
)
TOKENS_SHA = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

# The contest-authority DALI/NVDEC Seg argmax, sha pinned by ddm_ar1b:38 /
# ddm_ae1:34 / ddm_ap1:60.  READ-ONLY out of jf1's custody (that store is
# write-sacred; this arm only reads it).
#
# NOTE, and it is load-bearing: the first path tried here was
# ``/Volumes/APDataStore/pact/ddm_cpu1/retained/cpu1_seg_argmax_n600.npy``,
# which has the right SIZE (117,964,928 B) and a plausible name but hashes to
# ``68f5ad96...`` -- a DIFFERENT GT LINEAGE (a CPU-scorer argmax, not DALI).
# The digest gate below caught it.  Size and filename are not identity.
GT_PATH = Path(
    "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local"
    "/ddm_jf1_joint_field_model_refit/inputs/gt/gt_argmax_n600.dali.npy"
)
GT_SHA = "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"

N_FRAMES = 600
H, W = 384, 512
N_POS = N_FRAMES * H * W  # 117,964,800

# Canonical comma10k order, MEASURED and recorded in CLAUDE.md.  Never
# re-derived by luma-sorting the comma10k class_values -- that gives the wrong
# order and has bitten this project three times.
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
LANE = 1

# Published figures this run adjudicates.
TBA1_LANE_BYTES = 38_649.8
LD1_LANE_BYTES = 38_182.996184
LD1_BOTH_LANE_POSITIONS = 688_847
STREAM_BYTES = 113_777
ATTRIBUTED_BITS = 910_209.280609  # tba1 §1, sum of all per-symbol costs

# Exchange rate CITED from ddm_tx1_toolbox_crosswalk_20260819.md §0, never
# re-derived (charter instruction).
LAMBDA_B = 6.658590e-07
DEMAND_BYTES = 42_382


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_custody(strict: bool = True) -> dict[str, dict[str, object]]:
    """Hash every input and refuse on mismatch before any arithmetic runs."""
    out: dict[str, dict[str, object]] = {}
    for name, path, want in (
        ("cost", COST_PATH, COST_SHA),
        ("tokens", TOKENS_PATH, TOKENS_SHA),
        ("gt", GT_PATH, GT_SHA),
    ):
        if not path.exists():
            raise SystemExit(f"REFUSE: missing input {name}: {path}")
        got = sha256_file(path)
        ok = got == want
        out[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": got,
            "expected_sha256": want,
            "match": ok,
        }
        print(f"  {name:7s} sha256 {got} match={ok}", flush=True)
        if strict and not ok:
            raise SystemExit(
                f"REFUSE: {name} digest mismatch\n  want {want}\n  got  {got}"
            )
    return out


def load_gt() -> np.ndarray:
    """Load the DALI GT argmax as a FLAT uint8 view of exactly N_POS labels.

    The stored array is ``(600, 384, 512)``.  The cost and token fields are
    flat, so the GT must be flattened to share their raster order before any
    positional join -- slicing the 3-D array directly selects FRAMES, not a
    flat window, which silently mis-joins every chunk.
    """
    gt = np.load(GT_PATH, mmap_mode="r")
    if gt.dtype != np.uint8:
        raise SystemExit(f"REFUSE: GT dtype {gt.dtype} != uint8")
    if int(gt.size) != N_POS:
        raise SystemExit(f"REFUSE: GT size {gt.size} != {N_POS}")
    if gt.ndim == 3 and gt.shape != (N_FRAMES, H, W):
        raise SystemExit(f"REFUSE: GT shape {gt.shape} != {(N_FRAMES, H, W)}")
    flat = gt.reshape(-1)
    if flat.ndim != 1 or flat.size != N_POS:
        raise SystemExit(f"REFUSE: GT flatten failed: {flat.shape}")
    return flat


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--store",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_ma2_lane_index_set/measurement_v1"),
        help="retention store (SSD tier per CLAUDE.md disk rules)",
    )
    ap.add_argument("--frame-chunk", type=int, default=25)
    ap.add_argument("--allow-digest-mismatch", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    if args.frame_chunk < 1:
        # step <= 0 makes range() raise or scan nothing; a zero-length scan
        # would still emit a report full of zeros, which is the vacuity==PASS
        # class.  Refuse at the argument instead.
        raise SystemExit(f"REFUSE: --frame-chunk {args.frame_chunk} < 1")
    store = args.store

    print("[custody] hashing pinned inputs", flush=True)
    custody = verify_custody(strict=not args.allow_digest_mismatch)

    # Created only after the gate passes, so a refused run leaves no stray dir.
    store.mkdir(parents=True, exist_ok=True)

    cost = np.memmap(COST_PATH, dtype="<f8", mode="r")
    tokens = np.memmap(TOKENS_PATH, dtype=np.uint8, mode="r")
    gt = load_gt()
    if cost.size != N_POS or tokens.size != N_POS:
        raise SystemExit(f"REFUSE: size {cost.size}/{tokens.size} != {N_POS}")

    px_per_frame = H * W
    step = args.frame_chunk * px_per_frame

    # Per-class accumulators on BOTH index axes.
    dec_count = np.zeros(5, dtype=np.int64)
    dec_bits = np.zeros(5, dtype=np.float64)
    gt_count = np.zeros(5, dtype=np.int64)
    gt_bits = np.zeros(5, dtype=np.float64)

    # The 2x2 Lane contingency, which is the object the reconciliation needs.
    both_lane_n, both_lane_bits = 0, 0.0
    dec_only_n, dec_only_bits = 0, 0.0
    gt_only_n, gt_only_bits = 0, 0.0
    total_bits = 0.0

    print(f"[scan] {N_POS} positions in {args.frame_chunk}-frame chunks", flush=True)
    for start in range(0, N_POS, step):
        stop = min(start + step, N_POS)
        c = np.asarray(cost[start:stop], dtype=np.float64)
        d = np.asarray(tokens[start:stop])
        g = np.asarray(gt[start:stop])
        total_bits += float(c.sum(dtype=np.float64))

        for k in range(5):
            dm = d == k
            dec_count[k] += int(dm.sum())
            dec_bits[k] += float(c[dm].sum(dtype=np.float64))
            gm = g == k
            gt_count[k] += int(gm.sum())
            gt_bits[k] += float(c[gm].sum(dtype=np.float64))

        dl = d == LANE
        gl = g == LANE
        m = dl & gl
        both_lane_n += int(m.sum())
        both_lane_bits += float(c[m].sum(dtype=np.float64))
        m = dl & ~gl
        dec_only_n += int(m.sum())
        dec_only_bits += float(c[m].sum(dtype=np.float64))
        m = gl & ~dl
        gt_only_n += int(m.sum())
        gt_only_bits += float(c[m].sum(dtype=np.float64))

    b = 8.0  # bits per byte
    dec_lane_bytes = dec_bits[LANE] / b
    gt_lane_bytes = gt_bits[LANE] / b

    # A run that bypassed the digest gate is DEGRADED and must say so in its own
    # artifact.  Per-input `match` flags are not enough: a reader scanning the
    # adjudication block would never see them, which is the silent-failure class.
    degraded = [k for k, v in custody.items() if not v["match"]]

    report: dict[str, object] = {
        "arm": "ddm_ma2",
        "axis": "[macOS-CPU advisory / scorer-free arithmetic over retained coder-cost field]",
        "score_claim": False,
        "custody_gate_bypassed": bool(degraded),
        "degraded_inputs": degraded,
        "what_this_is": (
            "A sum over a RETAINED coder-cost field, not a re-encode. Per ddm_tba1 "
            "§8 the bit map is a MAP, not a PRICE; no rate claim is made."
        ),
        "custody": custody,
        "n_positions": N_POS,
        "attributed_bits_this_run": total_bits,
        "attributed_bits_tba1": ATTRIBUTED_BITS,
        "attributed_bits_delta": total_bits - ATTRIBUTED_BITS,
        "physical_stream_bytes": STREAM_BYTES,
        "per_class": {
            CLASS_NAMES[k]: {
                "decoded_positions": int(dec_count[k]),
                "decoded_area_frac": float(dec_count[k]) / N_POS,
                "decoded_bytes": dec_bits[k] / b,
                "decoded_bit_share": dec_bits[k] / total_bits,
                "gt_positions": int(gt_count[k]),
                "gt_area_frac": float(gt_count[k]) / N_POS,
                "gt_bytes": gt_bits[k] / b,
                "gt_bit_share": gt_bits[k] / total_bits,
            }
            for k in range(5)
        },
        "lane_contingency": {
            "both_lane_positions": both_lane_n,
            "both_lane_bytes": both_lane_bits / b,
            "decoded_lane_only_positions": dec_only_n,
            "decoded_lane_only_bytes": dec_only_bits / b,
            "gt_lane_only_positions": gt_only_n,
            "gt_lane_only_bytes": gt_only_bits / b,
        },
        "adjudication": {
            "tba1_lane_bytes_published": TBA1_LANE_BYTES,
            "tba1_reproduced_decoded_axis": dec_lane_bytes,
            "tba1_abs_err": abs(dec_lane_bytes - TBA1_LANE_BYTES),
            "ld1_lane_bytes_published": LD1_LANE_BYTES,
            "ld1_reproduced_gt_axis": gt_lane_bytes,
            "ld1_abs_err": abs(gt_lane_bytes - LD1_LANE_BYTES),
            "ld1_both_lane_positions_published": LD1_BOTH_LANE_POSITIONS,
            "both_lane_positions_reproduced": both_lane_n,
            "both_lane_positions_exact_match": both_lane_n == LD1_BOTH_LANE_POSITIONS,
            "published_difference_bytes": TBA1_LANE_BYTES - LD1_LANE_BYTES,
            "difference_explained_by_disagreement_bytes": (dec_only_bits - gt_only_bits) / b,
            "identity_residual_bytes": (
                (dec_lane_bytes - gt_lane_bytes) - (dec_only_bits - gt_only_bits) / b
            ),
        },
        "demand_context": {
            "demand_bytes_fixed_distortion": DEMAND_BYTES,
            "lambda_B_S_per_byte_CITED_tx1_sec0": LAMBDA_B,
            "decoded_lane_pct_of_demand": 100.0 * dec_lane_bytes / DEMAND_BYTES,
            "gt_lane_pct_of_demand": 100.0 * gt_lane_bytes / DEMAND_BYTES,
        },
        "elapsed_s": time.time() - t0,
    }

    # `elapsed_s` is wall-clock and changes every run, so the file's own SHA-256
    # changes even when every measured number is bit-identical -- which makes the
    # file sha uncitable.  Publish a digest over the report with the volatile
    # keys removed; that one IS stable across runs and is the value a memo should
    # cite.  Volatile context is preserved in the file, just outside the hash.
    volatile = ("elapsed_s", "content_digest_excluding_volatile")
    stable = {k: v for k, v in report.items() if k not in volatile}
    report["content_digest_excluding_volatile"] = hashlib.sha256(
        json.dumps(stable, indent=2, sort_keys=True).encode()
    ).hexdigest()

    out = store / "LANE_INDEX_SET_RECONCILIATION.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[digest] stable content {report['content_digest_excluding_volatile']}", flush=True)
    print(json.dumps(report["adjudication"], indent=2))
    print(json.dumps(report["lane_contingency"], indent=2))
    print(f"[done] {time.time()-t0:.1f}s -> {out}", flush=True)
    print(f"[sha] result {sha256_file(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
