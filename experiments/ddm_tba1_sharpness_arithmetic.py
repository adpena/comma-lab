#!/usr/bin/env python3
"""ddm_tba1 - the sharpness arithmetic over the dx2 token-cost map.

Three independently checkable pieces:

  A. COST x HARM JOIN, computed a third time and independently. ddm_wj1
     measured 90.96x count / 257.48x bit enrichment and ddm_tb2 reproduced it.
     This recomputes it from wj1's retained gross manufactured mask and the
     retained cost field, with its own top-k selection, so the number that
     carries the campaign's pattern-of-patterns law rests on three independent
     computations rather than one plus two citations.

  B. SELECTOR-COST REFERENCE. Any lever that treats a chosen subset of
     positions differently must tell the receiver WHICH positions. For a subset
     of size m out of N the i.i.d. indicator reference cost is N*H(m/N) bits.
     This is calibrated against ddm_ae1's MEASURED explicit-flag cost
     (130,228 B for m=93,580) so it is an empirically anchored reference, not a
     bare information-theoretic assertion.

  C. THE NET CEILING TABLE. For each cost stratum: the prize (every bit the
     stratum holds, i.e. the maximally generous assumption that the stratum
     could be coded for exactly zero) minus the selector cost of naming it.
     If even that upper bound is negative, the whole selector-based family is
     closed on this stream by arithmetic, before any coder is built.

Read-only against ddm_wj1's and ddm_tb2's stores. Derived vectors persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

TB2_FIELDS = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/"
    "measurement_v1/retained/fields"
)
COST_PATH = TB2_FIELDS / "position_rc64_frequency_cost_bits.f64le.bin"
COST_SHA = "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86"

WJ1_INPUTS = Path(
    "/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/"
    "ddm_wj1_cost_error_position_join/measurement_v1/retained/inputs"
)
WJ1_COST_PATH = WJ1_INPUTS / "position_rc64_frequency_cost_bits.f64le.bin"
GROSS_MANUFACTURED_PATH = (
    WJ1_INPUTS / "gross_manufactured_native_render_head.n600.packbits"
)

N_PAIRS, HEIGHT, WIDTH = 600, 384, 512
N_POS = N_PAIRS * HEIGHT * WIDTH

PHYSICAL_STREAM_BYTES = 113_777
DEMAND_BYTES = 42_382
EXCHANGE_RATE_S_PER_B = 6.658590e-07

# ddm_ae1 MEASURED anchor for the selector-cost calibration.
AE1_SET_POSITIONS = 93_580
AE1_MEASURED_FLAG_BYTES = 130_228
AE1_GROSS_EXCESS_BYTES = 26_645.297908
AE1_REPORTED_NET_BYTES = -103_582.702092


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def binary_entropy_bits(p: float) -> float:
    """H(p) in bits, the per-position i.i.d. indicator cost."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def selector_reference_bits(m: int, n: int = N_POS) -> float:
    """i.i.d. indicator reference cost, in bits, to name an m-of-n subset."""
    return n * binary_entropy_bits(m / n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/Volumes/APDataStore/pact/ddm_tba1_token_bit_attribution/sharpness_v1",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    ret = out / "retained"
    ret.mkdir(parents=True, exist_ok=True)

    stv = os.statvfs(ret)
    need = N_POS // 8 * 4 + (64 << 20)
    if stv.f_bavail * stv.f_frsize < need:
        print("REFUSE storage preflight", file=sys.stderr)
        return 4

    t0 = time.time()
    report: dict = {"arm": "ddm_tba1", "piece": "sharpness arithmetic", "inputs": {}}

    # ---- A. inputs + custody ----------------------------------------------
    for name, path in (
        ("tb2_cost_field", COST_PATH),
        ("wj1_cost_field_copy", WJ1_COST_PATH),
        ("wj1_gross_manufactured_mask", GROSS_MANUFACTURED_PATH),
    ):
        report["inputs"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    # wj1's copy of the cost field must be byte-identical to tb2's, else the
    # join below would be measuring a different object than the map.
    same = (
        report["inputs"]["wj1_cost_field_copy"]["sha256"]
        == report["inputs"]["tb2_cost_field"]["sha256"]
        == COST_SHA
    )
    report["cost_field_identical_across_stores"] = bool(same)
    if not same:
        print("REFUSE: wj1/tb2 cost fields differ; join would be cross-object",
              file=sys.stderr)
        return 3
    print(f"[{time.time()-t0:6.1f}s] custody verified; cost field identical",
          flush=True)

    cost = np.fromfile(COST_PATH, dtype="<f8")
    packed = np.fromfile(GROSS_MANUFACTURED_PATH, dtype=np.uint8)
    if cost.size != N_POS:
        print("REFUSE: cost size", file=sys.stderr)
        return 3

    # Bit-order is not self-describing in a raw packbits file, and popcount
    # CANNOT discriminate it: reversing bits within a byte leaves the count
    # unchanged. The convention is therefore SOURCED, not guessed - ddm_wj1's
    # retained writer and reader both pin bitorder="little"
    # (retained/provenance/ddm_wj1_cost_error_position_join.py lines 167, 500,
    # 502). The order-SENSITIVE cross-check is the top-1% intersection count,
    # asserted below against wj1's reported 26,016.
    MASK_BITORDER = "little"
    manufactured = np.unpackbits(packed, bitorder=MASK_BITORDER, count=N_POS).astype(
        bool
    )
    report["mask_bitorder"] = {
        "bitorder": MASK_BITORDER,
        "source": (
            "ddm_wj1 retained provenance ddm_wj1_cost_error_position_join.py "
            "lines 167/500/502 - sourced, not inferred"
        ),
        "decoded_support_positions": int(manufactured.sum()),
        "note": (
            "popcount is bit-order invariant and cannot validate this; the "
            "order-sensitive validator is wj1_top1pct_intersection_reproduced"
        ),
    }
    total_bits = float(cost.sum(dtype=np.float64))
    m_total = int(manufactured.sum())
    manufactured_bits = float(cost[manufactured].sum(dtype=np.float64))
    print(f"[{time.time()-t0:6.1f}s] manufactured support {m_total:,} positions",
          flush=True)

    report["manufactured_support"] = {
        "positions": m_total,
        "position_frac": m_total / N_POS,
        "bits": manufactured_bits,
        "bytes": manufactured_bits / 8.0,
        "share_of_stream_bits": manufactured_bits / total_bits,
    }

    # ---- A. the join, independently selected -------------------------------
    order_desc = np.argsort(cost, kind="stable")[::-1]
    joins = []
    for label, frac in (
        ("top_0p1pct", 0.001),
        ("top_1pct", 0.01),
        ("top_5pct", 0.05),
        ("top_10pct", 0.10),
    ):
        k = round(N_POS * frac)
        sel = order_desc[:k]
        hit = manufactured[sel]
        obs_positions = int(hit.sum())
        obs_bits = float(cost[sel][hit].sum(dtype=np.float64))
        exp_positions = m_total * (k / N_POS)
        stratum_bits = float(cost[sel].sum(dtype=np.float64))

        # TWO independence baselines. They answer different questions and give
        # different ratios; MEMORY records "the floor you divide by decides the
        # answer" (pc2) as a recurring genus, so both are reported and neither
        # is presented as the number.
        #
        #  baseline_A (ddm_wj1 / ddm_tb2 convention): if the manufactured
        #    positions were placed at random, exp_positions of them would land
        #    in this stratum, each carrying the stratum's MEAN cost.
        #  baseline_B (mass-share convention): the manufactured bit mass spread
        #    over strata in proportion to their POSITION share.
        exp_bits_a = exp_positions * (stratum_bits / k)
        exp_bits_b = manufactured_bits * (k / N_POS)

        joins.append(
            {
                "stratum": label,
                "k_positions": k,
                "boundary_cost_bits": float(cost[sel[-1]]),
                "boundary_ties_in_field": int((cost == cost[sel[-1]]).sum()),
                "stratum_bits": stratum_bits,
                "stratum_mean_bits_per_position": stratum_bits / k,
                "observed_manufactured_positions": obs_positions,
                "expected_manufactured_positions": exp_positions,
                "count_enrichment": obs_positions / exp_positions,
                "observed_manufactured_bits": obs_bits,
                "observed_manufactured_bytes": obs_bits / 8.0,
                "expected_bits_baseline_A_wj1_tb2": exp_bits_a,
                "bit_enrichment_baseline_A_wj1_tb2": obs_bits / exp_bits_a,
                "expected_bits_baseline_B_mass_share": exp_bits_b,
                "bit_enrichment_baseline_B_mass_share": obs_bits / exp_bits_b,
                "share_of_manufactured_bit_mass_captured": obs_bits
                / manufactured_bits,
            }
        )
        print(
            f"[{time.time()-t0:6.1f}s] {label}: {obs_positions:,} pos "
            f"({obs_positions/exp_positions:.4f}x count) / "
            f"{obs_bits/exp_bits_a:.4f}x bits[A] / "
            f"{obs_bits/exp_bits_b:.4f}x bits[B]",
            flush=True,
        )
    report["cost_harm_join"] = joins

    # Order-SENSITIVE validator. ddm_wj1 JOIN_RESULT.json reports, for top_1pct:
    # positions 26,016 / expected 286.02 / count 90.95867421858613 /
    # bits 54,734.65143519006 / expected_bits 212.57834526834395 /
    # bit enrichment 257.47990166212315. A wrong bit-order, a wrong mask, or a
    # different top-k convention all break this.
    WJ1_TOP1 = {
        "positions": 26_016,
        "count_enrichment": 90.95867421858613,
        "bits": 54_734.65143519006,
        "bit_enrichment_A": 257.47990166212315,
    }
    mine = next(j for j in joins if j["stratum"] == "top_1pct")
    report["wj1_reproduction_check"] = {
        "wj1_reported": WJ1_TOP1,
        "tba1_independent": {
            "positions": mine["observed_manufactured_positions"],
            "count_enrichment": mine["count_enrichment"],
            "bits": mine["observed_manufactured_bits"],
            "bit_enrichment_A": mine["bit_enrichment_baseline_A_wj1_tb2"],
        },
        "positions_exact_match": bool(
            mine["observed_manufactured_positions"] == WJ1_TOP1["positions"]
        ),
        "bits_rel_error": abs(mine["observed_manufactured_bits"] - WJ1_TOP1["bits"])
        / WJ1_TOP1["bits"],
        "bit_enrichment_A_rel_error": abs(
            mine["bit_enrichment_baseline_A_wj1_tb2"] - WJ1_TOP1["bit_enrichment_A"]
        )
        / WJ1_TOP1["bit_enrichment_A"],
    }
    print(
        f"[{time.time()-t0:6.1f}s] wj1 reproduction: positions "
        f"{mine['observed_manufactured_positions']:,} vs {WJ1_TOP1['positions']:,} "
        f"exact={report['wj1_reproduction_check']['positions_exact_match']}",
        flush=True,
    )

    # ---- B. selector-cost calibration against ae1's MEASURED flag cost -----
    ae1_ref_bits = selector_reference_bits(AE1_SET_POSITIONS)
    report["selector_cost_calibration"] = {
        "ae1_set_positions": AE1_SET_POSITIONS,
        "iid_reference_bits": ae1_ref_bits,
        "iid_reference_bytes": ae1_ref_bits / 8.0,
        "ae1_measured_flag_bytes": AE1_MEASURED_FLAG_BYTES,
        "measured_over_reference": AE1_MEASURED_FLAG_BYTES / (ae1_ref_bits / 8.0),
        "ae1_gross_prize_bytes": AE1_GROSS_EXCESS_BYTES,
        "reference_predicted_net_bytes": AE1_GROSS_EXCESS_BYTES
        - ae1_ref_bits / 8.0,
        "ae1_reported_net_bytes": AE1_REPORTED_NET_BYTES,
        "note": (
            "The i.i.d. reference is not a strict bound: a spatially correlated "
            "indicator can be coded below it. ae1's real flags achieved "
            "94.8% of the reference, i.e. this indicator is only ~5% "
            "compressible. The reference is therefore treated as a calibrated "
            "estimate, and every net below is reported with that caveat."
        ),
    }

    # ---- C. the net ceiling table -----------------------------------------
    sorted_asc = np.sort(cost, kind="stable")
    csum = np.cumsum(sorted_asc, dtype=np.float64)
    rows = []
    for thr in [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 30.0]:
        idx = int(np.searchsorted(sorted_asc, thr, side="right"))
        m = N_POS - idx
        if m <= 0:
            continue
        prize_bits = total_bits - (float(csum[idx - 1]) if idx > 0 else 0.0)
        sel_bits = selector_reference_bits(m)
        rows.append(
            {
                "threshold_bits_per_symbol": thr,
                "set_positions": m,
                "set_frac": m / N_POS,
                "prize_bits_if_stratum_were_free": prize_bits,
                "prize_bytes_if_stratum_were_free": prize_bits / 8.0,
                "prize_pct_of_demand": (prize_bits / 8.0) / DEMAND_BYTES,
                "selector_reference_bits": sel_bits,
                "selector_reference_bytes": sel_bits / 8.0,
                "net_ceiling_bytes": (prize_bits - sel_bits) / 8.0,
                "net_ceiling_positive": bool(prize_bits > sel_bits),
                "prize_over_selector": prize_bits / sel_bits,
            }
        )
        print(
            f"[{time.time()-t0:6.1f}s] thr>{thr:g}: m={m:,} prize="
            f"{prize_bits/8.0:,.1f} B selector={sel_bits/8.0:,.1f} B net="
            f"{(prize_bits-sel_bits)/8.0:,.1f} B",
            flush=True,
        )
    report["net_ceiling_table"] = rows
    report["any_stratum_net_positive"] = bool(
        any(r["net_ceiling_positive"] for r in rows)
    )
    # Best case over the whole family, and the break-even set size if one exists.
    best = max(rows, key=lambda r: r["net_ceiling_bytes"])
    report["best_stratum"] = best
    report["best_stratum_delta_S_if_realized"] = (
        best["net_ceiling_bytes"] * EXCHANGE_RATE_S_PER_B
    )

    # ---- persist derived vectors ------------------------------------------
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

    # The per-position expensive-set membership at each stratum, joined with the
    # manufactured flag. This is the object a successor needs to target anything
    # and it is a payload, not a statistic.
    k1 = round(N_POS * 0.01)
    top1_mask = np.zeros(N_POS, dtype=bool)
    top1_mask[order_desc[:k1]] = True
    persist("top1pct_expensive_mask.n600.packbits", np.packbits(top1_mask))
    persist(
        "top1pct_expensive_and_manufactured.n600.packbits",
        np.packbits(np.logical_and(top1_mask, manufactured)),
    )
    report["persisted_payloads"] = persisted

    report["elapsed_seconds"] = time.time() - t0
    report["exchange_rate_S_per_B"] = EXCHANGE_RATE_S_PER_B
    report["exchange_rate_source"] = (
        "ddm_tx1_toolbox_crosswalk_20260819.md section 0 - CITED, not re-derived"
    )
    report["price_caveat"] = (
        "Prizes and net ceilings are UPPER BOUNDS read off -log2 p, never "
        "prices. ddm_fs2 measured -log2 p prices 0.77-0.88x wrong away from "
        "argmax and 0.09x toward it. A rate claim is a MEASURED re-encode."
    )

    rp = out / "RESULT.json"
    rp.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"[{time.time()-t0:6.1f}s] wrote {rp}")
    print(f"RESULT.json sha256 {sha256_file(rp)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
