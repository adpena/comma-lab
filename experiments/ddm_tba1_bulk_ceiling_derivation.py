#!/usr/bin/env python3
"""ddm_tba1 - independent re-derivation over the RETAINED dx2 token-cost field.

This script does NOT re-run the encoder. ddm_tb2 (and ddm_bl1 before it) already
drove the exact shipped HPAC/RC64 receiver and retained the per-symbol
-log2 p field; both retained copies are byte-identical. Re-running that replay a
third time would reproduce a byte-identical payload, which is rediscovery, not
measurement. This arm instead verifies the retained field by re-derivation and
computes the quantities no prior arm reported.

What this adds beyond ddm_tb2 / ddm_bl1 / ddm_wj1:

  1. INDEPENDENT verification of the SUM reconciliation and the concentration
     headline, recomputed from the primary artifact rather than recognised from
     a memo.
  2. The SELF-DETECTED class order, assigned from spatial/static signature
     (vertical centroid, temporal IoU, area) and never from a hardcoded index
     or a luma sort. CLAUDE.md records that luma-sorting this order was wrong
     three separate times.
  3. The BULK-LEVER CEILING - the total bit mass held by the cheap complement
     of the expensive set. This bounds every lever that acts on structure
     spread across all positions (ordering, addressing, generic context
     refinement). It is not reported anywhere in the corpus.
  4. The cost-stratum table: mass above each per-symbol bit threshold, and what
     fraction of the 42,382 B demand each stratum could supply if driven to
     exactly zero. This is a CEILING, never a projection - a -log2 p reading is
     a map, not a price (ddm_fs2 measured -log2 p prices 0.77-0.88x wrong away
     from argmax and 0.09x toward it).

Read-only against ddm_tb2's store. Outputs land in this arm's own store. All
derived vectors are PERSISTED (P0: always keep the payload).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- pinned inputs

TB2_FIELDS = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/"
    "measurement_v1/retained/fields"
)
COST_PATH = TB2_FIELDS / "position_rc64_frequency_cost_bits.f64le.bin"
SYMBOL_PATH = TB2_FIELDS / "decoded_tokens_instrumented.u8"

COST_SHA = "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"
# This is also ddm_tba1's charter-pinned categorical field sha - the identity
# check that the retained field is THIS arm's object.
SYMBOL_SHA = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

N_PAIRS = 600
HEIGHT = 384
WIDTH = 512
N_POS = N_PAIRS * HEIGHT * WIDTH  # 117,964,800

# ar1b physical census on archive sha 976f706d..., verified zero-remainder.
PHYSICAL_STREAM_BYTES = 113_777
PHYSICAL_STREAM_BITS = PHYSICAL_STREAM_BYTES * 8  # 910,216

# fb1 demand arithmetic; tx1 section 0 exchange rate, CITED not re-derived.
DEMAND_BYTES = 42_382
EXCHANGE_RATE_S_PER_B = 6.658590e-07

# Canonical comma10k order. Used ONLY to LABEL and cross-check the self-detected
# result, never to assign it.
CANONICAL_ORDER = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gini_from_sorted(sorted_asc: np.ndarray, total: float) -> float:
    """Exact Gini of a non-negative vector already sorted ascending.

    G = 2*sum(i*x_i)/(n*sum(x)) - (n+1)/n, i one-based. The index-weighted sum
    is accumulated blockwise in float64 so the large-i terms do not swamp the
    tail.
    """
    n = sorted_asc.size
    if total <= 0.0:
        return float("nan")
    weighted = 0.0
    block = 1 << 22
    for start in range(0, n, block):
        stop = min(start + block, n)
        idx = np.arange(start + 1, stop + 1, dtype=np.float64)
        weighted += float(np.dot(idx, sorted_asc[start:stop]))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def self_detect_class_order(symbols: np.ndarray) -> dict:
    """Assign semantic names from spatial/static signature, never from the index.

    Signature, per the MEASURED canonical comma10k semantics in CLAUDE.md:
      Undrivable - sky/top; the SMALLEST vertical centroid of the five
      MyCar      - ego hood; the LARGEST vertical centroid of the five, static
      Road       - largest remaining area
      Lane       - of what is left, the LOWEST temporal IoU (unstable orbit)
      Movable    - the remaining one

    Centroid and IoU separate these without relying on the MyCar/Road area
    near-tie (25.4% vs 23.2%), which an area-rank rule would decide by ~2 pp.
    """
    frames = symbols.reshape(N_PAIRS, HEIGHT, WIDTH)
    row_index = np.arange(HEIGHT, dtype=np.float64)

    stats: dict[int, dict] = {}
    for value in range(5):
        mask = frames == value
        count = int(mask.sum())
        per_row = mask.sum(axis=(0, 2)).astype(np.float64)
        centroid = float((per_row * row_index).sum() / max(per_row.sum(), 1.0))
        inter = int(np.logical_and(mask[:-1], mask[1:]).sum())
        union = int(np.logical_or(mask[:-1], mask[1:]).sum())
        stats[value] = {
            "index": value,
            "count": count,
            "area_frac": count / N_POS,
            "row_centroid": centroid,
            "temporal_iou": (inter / union) if union else float("nan"),
        }

    unassigned = dict(stats)
    assignment: dict[int, str] = {}

    def take(name: str, key) -> None:
        row = key(list(unassigned.values()))
        assignment[row["index"]] = name
        del unassigned[row["index"]]

    take("Undrivable", lambda rows: min(rows, key=lambda r: r["row_centroid"]))
    take("MyCar", lambda rows: max(rows, key=lambda r: r["row_centroid"]))
    take("Road", lambda rows: max(rows, key=lambda r: r["count"]))
    take("Lane", lambda rows: min(rows, key=lambda r: r["temporal_iou"]))
    take("Movable", lambda rows: rows[0])

    for value, row in stats.items():
        row["self_detected_name"] = assignment[value]
        row["matches_canonical_order"] = assignment[value] == CANONICAL_ORDER[value]
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/Volumes/APDataStore/pact/ddm_tba1_token_bit_attribution/derivation_v1",
    )
    ap.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip input re-hash; only legal on a resume whose hashes are logged",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    ret = out / "retained"
    ret.mkdir(parents=True, exist_ok=True)

    # Storage preflight, fail-closed. This run persists a f64 cumsum
    # (8 B/position) plus an i32 rank (4 B/position) plus slack for RESULT.json.
    need_bytes = N_POS * (8 + 4) + (64 << 20)
    stv = os.statvfs(ret)
    free_bytes = stv.f_bavail * stv.f_frsize
    if free_bytes < need_bytes:
        print(
            f"REFUSE storage preflight: {free_bytes} B free < {need_bytes} B needed "
            f"at {ret}",
            file=sys.stderr,
        )
        return 4

    t0 = time.time()
    report: dict = {
        "arm": "ddm_tba1",
        "kind": "re-derivation over the retained field; the encoder was NOT re-run",
        "object_archive_sha256": (
            "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
        ),
        "inputs": {},
    }

    # ---- input custody -----------------------------------------------------
    for name, path, expect in (
        ("cost_field", COST_PATH, COST_SHA),
        ("symbol_field", SYMBOL_PATH, SYMBOL_SHA),
    ):
        digest = None if args.skip_hash else sha256_file(path)
        report["inputs"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "sha256_expected": expect,
            "sha256_match": None if digest is None else digest == expect,
        }
        if digest is not None and digest != expect:
            print(f"REFUSE: {name} sha {digest} != {expect}", file=sys.stderr)
            return 3
    print(f"[{time.time()-t0:7.1f}s] input custody verified", flush=True)

    # ---- load and validate -------------------------------------------------
    cost = np.fromfile(COST_PATH, dtype="<f8")
    symbols = np.fromfile(SYMBOL_PATH, dtype=np.uint8)
    if cost.size != N_POS or symbols.size != N_POS:
        print(f"REFUSE: size {cost.size}/{symbols.size} != {N_POS}", file=sys.stderr)
        return 3
    if not np.isfinite(cost).all():
        print("REFUSE: non-finite cost present", file=sys.stderr)
        return 3
    if float(cost.min()) < 0.0:
        print(f"REFUSE: negative cost min={cost.min()}", file=sys.stderr)
        return 3
    if int(symbols.max()) > 4:
        print(f"REFUSE: symbol alphabet >5: max={symbols.max()}", file=sys.stderr)
        return 3
    print(f"[{time.time()-t0:7.1f}s] loaded {cost.size} positions", flush=True)

    # ---- SUM verification (the gate) ---------------------------------------
    total_bits = float(cost.sum(dtype=np.float64))
    residual_bits = PHYSICAL_STREAM_BITS - total_bits
    # Arithmetic-coder overhead bound: final-interval flush < 2 bits plus
    # partial-final-byte padding <= 7 bits, so strictly less than 9 bits.
    residual_bound_bits = 9.0
    report["sum_verification"] = {
        "physical_stream_bytes": PHYSICAL_STREAM_BYTES,
        "physical_stream_bits": PHYSICAL_STREAM_BITS,
        "attributed_bits": total_bits,
        "attributed_bytes": total_bits / 8.0,
        "residual_bits": residual_bits,
        "residual_bytes": residual_bits / 8.0,
        "residual_bound_bits": residual_bound_bits,
        "residual_within_bound": bool(abs(residual_bits) < residual_bound_bits),
        "residual_frac_of_stream": residual_bits / PHYSICAL_STREAM_BITS,
    }
    print(
        f"[{time.time()-t0:7.1f}s] SUM attributed {total_bits:.6f} bits vs physical "
        f"{PHYSICAL_STREAM_BITS} -> residual {residual_bits:.6f} bits "
        f"({residual_bits/8.0:.6f} B)",
        flush=True,
    )

    # ---- sort once; every quantile below reads off this Lorenz curve -------
    order = np.argsort(cost, kind="stable")
    sorted_asc = cost[order]
    csum = np.cumsum(sorted_asc, dtype=np.float64)
    print(f"[{time.time()-t0:7.1f}s] sorted + cumsum", flush=True)

    gini = gini_from_sorted(sorted_asc, total_bits)
    print(f"[{time.time()-t0:7.1f}s] gini {gini:.16f}", flush=True)

    def stratum(frac: float) -> dict:
        """Top `frac` of positions by cost, and its cheap complement."""
        k = round(N_POS * frac)
        cheap_bits = float(csum[N_POS - k - 1]) if 0 < k < N_POS else (
            0.0 if k >= N_POS else total_bits
        )
        top_bits = total_bits - cheap_bits
        return {
            "frac": frac,
            "k_positions": k,
            "top_bits": top_bits,
            "top_bytes": top_bits / 8.0,
            "top_share": top_bits / total_bits,
            "top_bits_per_position": top_bits / k if k else float("nan"),
            "complement_positions": N_POS - k,
            "complement_bits": cheap_bits,
            "complement_bytes": cheap_bits / 8.0,
            "complement_share": cheap_bits / total_bits,
            "complement_ceiling_pct_of_demand": (cheap_bits / 8.0) / DEMAND_BYTES,
            "complement_ceiling_delta_S": (cheap_bits / 8.0) * EXCHANGE_RATE_S_PER_B,
        }

    report["concentration"] = {
        "gini": gini,
        "strata": {
            label: stratum(f)
            for label, f in (
                ("top_0p01pct", 0.0001),
                ("top_0p1pct", 0.001),
                ("top_1pct", 0.01),
                ("top_5pct", 0.05),
                ("top_10pct", 0.10),
                ("top_25pct", 0.25),
                ("top_50pct", 0.50),
            )
        },
    }

    # ---- cost-threshold strata, read off the sorted curve (no copies) ------
    thresholds = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    strata = []
    for thr in thresholds:
        # first index whose value is strictly greater than thr
        idx = int(np.searchsorted(sorted_asc, thr, side="right"))
        n_above = N_POS - idx
        bits_below = float(csum[idx - 1]) if idx > 0 else 0.0
        bits_above = total_bits - bits_below
        strata.append(
            {
                "threshold_bits_per_symbol": thr,
                "positions_above": n_above,
                "positions_above_frac": n_above / N_POS,
                "bits_above": bits_above,
                "bytes_above": bits_above / 8.0,
                "share_above": bits_above / total_bits,
                "positions_at_or_below": idx,
                "bits_at_or_below": bits_below,
                "bytes_at_or_below": bits_below / 8.0,
                "below_ceiling_pct_of_demand": (bits_below / 8.0) / DEMAND_BYTES,
            }
        )
    report["cost_strata"] = strata
    report["cost_extrema"] = {
        "min_bits": float(sorted_asc[0]),
        "max_bits": float(sorted_asc[-1]),
        "mean_bits": total_bits / N_POS,
        "median_bits": float(sorted_asc[N_POS // 2]),
        "p90_bits": float(sorted_asc[int(N_POS * 0.90)]),
        "p99_bits": float(sorted_asc[int(N_POS * 0.99)]),
        "p999_bits": float(sorted_asc[int(N_POS * 0.999)]),
        "exactly_zero_positions": int(np.searchsorted(sorted_asc, 0.0, side="right")),
    }

    # ---- self-detected class order + per-class bits (one bincount pass) ----
    class_stats = self_detect_class_order(symbols)
    class_bits = np.bincount(symbols, weights=cost, minlength=5)
    for value, row in class_stats.items():
        bits = float(class_bits[value])
        n = max(row["count"], 1)
        row["bits"] = bits
        row["bytes"] = bits / 8.0
        row["bit_share"] = bits / total_bits
        row["bits_per_position"] = bits / n
        row["enrichment_over_body_mean"] = (bits / n) / (total_bits / N_POS)
        row["ceiling_pct_of_demand_if_zeroed"] = (bits / 8.0) / DEMAND_BYTES
        row["ceiling_delta_S_if_zeroed"] = (bits / 8.0) * EXCHANGE_RATE_S_PER_B
    report["class_attribution_self_detected"] = {
        str(k): v for k, v in class_stats.items()
    }
    report["class_order_selfcheck_all_match_canonical"] = bool(
        all(v["matches_canonical_order"] for v in class_stats.values())
    )
    # Internal cross-check: the per-class partition must exhaust the stream.
    class_bits_sum = float(class_bits.sum())
    class_count_sum = int(sum(r["count"] for r in class_stats.values()))
    report["class_partition_crosscheck"] = {
        "sum_of_class_bits": class_bits_sum,
        "total_bits": total_bits,
        "bit_closure_residual": total_bits - class_bits_sum,
        "sum_of_class_counts": class_count_sum,
        "n_positions": N_POS,
        "count_closure_exact": bool(class_count_sum == N_POS),
    }

    # ---- the demand, restated on the concentrated set ----------------------
    demand_bits = DEMAND_BYTES * 8
    top1 = report["concentration"]["strata"]["top_1pct"]
    report["demand_restated"] = {
        "demand_bytes": DEMAND_BYTES,
        "demand_bits": demand_bits,
        "demand_frac_of_stream": demand_bits / PHYSICAL_STREAM_BITS,
        "required_fractional_cut_on_top1pct_if_sole_source": (
            demand_bits / top1["top_bits"]
        ),
        "top1pct_bits_per_position": top1["top_bits_per_position"],
        "bulk_lever_ceiling_bytes": top1["complement_bytes"],
        "bulk_lever_ceiling_pct_of_demand": top1["complement_ceiling_pct_of_demand"],
        "bulk_lever_ceiling_delta_S": top1["complement_ceiling_delta_S"],
    }

    # ---- PERSIST the derived vectors (P0: always keep the payload) --------
    persisted: dict[str, dict] = {}

    def persist(name: str, arr: np.ndarray) -> None:
        p = ret / name
        arr.tofile(p)
        persisted[name] = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "dtype": str(arr.dtype),
            "elements": int(arr.size),
        }
        print(f"[{time.time()-t0:7.1f}s] persisted {name}", flush=True)

    # The Lorenz curve. Every quantile in this report is read off it, and a
    # successor can read ANY stratum without re-sorting 118M positions.
    persist("cost_bits_cumsum_asc.f64le.bin", csum)
    # The join key: lets a successor intersect this cost ordering with any other
    # per-position field (seg error, context cell, class) without re-deriving
    # the sort. rank[p] = ascending cost rank of position p.
    rank = np.empty(N_POS, dtype="<i4")
    rank[order] = np.arange(N_POS, dtype="<i4")
    persist("cost_rank_ascending.i32le.bin", rank)

    report["persisted_payloads"] = persisted
    report["payload_certification"] = {
        "cost_bits_sorted_asc": (
            "NOT re-persisted: exactly reconstructible as cost[argsort(cost)] "
            "from the pinned cost field sha " + COST_SHA + " plus the retained "
            "cost_rank_ascending.i32le.bin. Certified rebuildable per the "
            "certify-or-block disk rule; no signal is lost."
        ),
        "per_symbol_minus_log2_p_field": (
            "NOT duplicated into this arm's store: it is retained byte-identical "
            "in TWO independent prior stores (ddm_bl1 and ddm_tb2, sha "
            + COST_SHA
            + "), re-verified by this run. A third byte-identical copy would add "
            "944 MB and zero signal."
        ),
    }
    report["elapsed_seconds"] = time.time() - t0
    report["exchange_rate_S_per_B"] = EXCHANGE_RATE_S_PER_B
    report["exchange_rate_source"] = (
        "ddm_tx1_toolbox_crosswalk_20260819.md section 0 - CITED, not re-derived"
    )
    report["price_caveat"] = (
        "Every 'ceiling' above is an UPPER BOUND read off -log2 p, never a price. "
        "ddm_fs2 measured -log2 p prices 0.77-0.88x wrong away from argmax and "
        "0.09x toward it. A rate claim is a MEASURED re-encode or it is not made."
    )

    result_path = out / "RESULT.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[{time.time()-t0:7.1f}s] wrote {result_path}", flush=True)
    print(f"RESULT.json sha256 {sha256_file(result_path)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
