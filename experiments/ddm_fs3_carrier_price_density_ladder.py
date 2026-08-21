#!/usr/bin/env python3
"""ddm_fs3 leg 2 -- MEASURE the real archive price of a carrier-coefficient move.

WHAT IS BEING ADJUDICATED
-------------------------
``ddm_fs1`` sec.3 closed the pose-only-edit actuator against a table of edit
encodings.  One of its rows, ``jg1 re-solve midpoint 10.500 B/pair``, is a UNITS
MISREAD: ``ddm_na10:562`` says that move "moves 9-12 already-shipped
COEFFICIENTS", and 10.5 is the midpoint of that COUNT, entered into a column of
byte prices (fs1 ERRATUM E1, from rv17 wave-2 W2-F8).

The three prices that exist at source for the same move are mutually inconsistent
and none of them is a real re-encode of this move on this body:

* ``na10``  0.83 B/pair -- itself a 100x internal slip (5 B / 7,200 coefficients
  spread over 600 pairs is 0.0083 B/pair, not 0.83);
* ``up3`` sec.5  ~0.08 B/pair when the moved coefficients are ABSORBED as shipped,
  but ``+3 B`` per ISOLATED coefficient, i.e. ~27-36 B/pair for isolated moves;
* ``jg5``  ``+45 B`` measured BY BUILDING for 455 pairs of re-solved codes, which
  is 0.0989 B/pair -- but at 455/600 density.

At 0.83 B/pair the blanket-27 move nets -2.384e-05 S, a gain ~6.8x the admission
bar.  At 27-36 B/pair it stays dead.  So the whole verdict turns on a number
nobody has measured at the density that matters.

**The disagreement is not noise -- it is a DENSITY question.**  up3's "absorbed"
and "isolated" regimes are the two ends of one curve, and the blanket-27 move sits
at the sparse end while jg5's +45 B sits at the dense end.  This module measures
the curve instead of picking a point off it.

THE MEASUREMENT
---------------
The carrier is ``(600, 12)`` int12 codes behind a Rice payload inside a brotli
container.  ``ddm_up3_carrier_splice.build_archive`` rebuilds the archive carrying
any code array, copying the hpac stream, the semantic stream and the section tail
verbatim so ONLY the carrier moves, and it parses the finished bytes back through
the receiver and refuses to return them unless they decode to exactly those codes.

So the price of moving N pairs' coefficients is an exact ``archive.zip`` stat
delta.  This module measures it by REVERTING N of the shipped body's re-solved
pairs to their base codes, which is the same field difference as adding them and
therefore the same code length, and it does that at a ladder of N so the
absorbed-vs-isolated curve is visible rather than assumed.

CONTROL: rebuilding the body from its OWN codes must reproduce the shipped archive
byte-identically.  Without that, a byte delta cannot be attributed to the codes
rather than to the rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

N_PAIRS = 600
CARRIER_DIM = 12
SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR

#: fs1 sec.3's break-even budgets for the pose-only-edit actuator, re-derived there
#: at the live operating point.  The adjudication is against these two.
BREAKEVEN_MEDIAN_B_PER_PAIR = 1.429
BREAKEVEN_MEAN_B_PER_PAIR = 2.909

#: fs1's blanket move: the 27 pairs jg5 never edited.
BLANKET_PAIRS = 27

DEFAULT_BODY_RUNTIME = "/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5"
DEFAULT_BR1_CODES = (
    "/Volumes/APDataStore/pact/ddm_br1/retained/byte_close_n600/"
    "br1_candidate_codes.npy"
)


class Fs3LegTwoError(RuntimeError):
    """Fail-closed error."""


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_ladder(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_up3_carrier_splice as splice

    runtime = Path(args.body_runtime)
    archive_path = runtime / "archive.zip"
    shipped = archive_path.read_bytes()
    shipped_sha = sha256_of_bytes(shipped)
    shipped_bytes = len(shipped)
    if args.expect_body_sha256 and shipped_sha != args.expect_body_sha256:
        raise Fs3LegTwoError(
            f"body sha256 {shipped_sha} != expected {args.expect_body_sha256}"
        )

    body = splice.parse_shipped_body(runtime, verify_sha=False)
    shipped_codes = np.asarray(body.codes, dtype=np.int32)
    if shipped_codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Fs3LegTwoError(f"shipped codes have shape {shipped_codes.shape}")

    base_codes = np.load(args.br1_codes).astype(np.int32)
    if base_codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Fs3LegTwoError(f"base codes have shape {base_codes.shape}")

    moved = np.flatnonzero((shipped_codes != base_codes).any(axis=1))
    coeffs_moved = int((shipped_codes != base_codes).sum())
    if moved.size == 0:
        raise Fs3LegTwoError(
            "the shipped body carries the base codes unchanged; there is no "
            "carrier move to price"
        )

    out_dir = Path(args.out)
    (out_dir / "retained").mkdir(parents=True, exist_ok=True)

    # CONTROL: the body's own codes must rebuild the body's own bytes.
    identity = splice.build_archive(
        body, shipped_codes, runtime_dir=runtime, container_search=True
    )
    identity_ok = identity["archive_sha256"] == shipped_sha
    if not identity_ok and not args.allow_identity_drift:
        raise Fs3LegTwoError(
            "CONTROL FAILED: rebuilding the body from its own codes gives "
            f"{identity['archive_sha256']} ({identity['archive_size']} B), not "
            f"{shipped_sha} ({shipped_bytes} B); a byte delta could not be "
            "attributed to the codes"
        )
    identity_path = out_dir / "retained" / "archive_identity.zip"
    identity_path.write_bytes(identity["archive_bytes"])

    # The ladder reverts the FIRST n moved pairs (deterministic order, recorded) so
    # every rung is a strict subset of the next and the curve is nested.
    rungs = sorted({n for n in args.rungs if 0 < n <= moved.size} | {moved.size})
    rows: list[dict[str, Any]] = []
    for n in rungs:
        reverted = moved[:n]
        candidate = shipped_codes.copy()
        candidate[reverted] = base_codes[reverted]
        built = splice.build_archive(
            body,
            candidate,
            runtime_dir=runtime,
            container_search=True,
            verify=True,
        )
        # ALWAYS KEEP THE PAYLOAD -- every rung is retained, not only the headline.
        path = out_dir / "retained" / f"archive_revert_{n:04d}.zip"
        path.write_bytes(built["archive_bytes"])
        np.save(out_dir / "retained" / f"codes_revert_{n:04d}.npy", candidate)

        size = int(built["archive_size"])
        delta = shipped_bytes - size
        coeffs = int((shipped_codes[reverted] != base_codes[reverted]).sum())
        rows.append(
            {
                "pairs_reverted": int(n),
                "coefficients_reverted": coeffs,
                "archive_bytes": size,
                "delta_bytes_vs_shipped": delta,
                "bytes_per_pair": delta / n,
                "bytes_per_coefficient": delta / coeffs if coeffs else None,
                "archive_sha256": built["archive_sha256"],
                "path": str(path),
                "clears_median_breakeven": (delta / n) <= BREAKEVEN_MEDIAN_B_PER_PAIR,
                "clears_mean_breakeven": (delta / n) <= BREAKEVEN_MEAN_B_PER_PAIR,
            }
        )
        print(
            f"  revert {n:4d} pairs ({coeffs:5d} coeffs): {size:7d} B  "
            f"delta {delta:+6d}  {delta / n:8.4f} B/pair  "
            f"{delta / coeffs if coeffs else float('nan'):8.4f} B/coeff",
            flush=True,
        )

    blanket = min(rows, key=lambda r: abs(r["pairs_reverted"] - BLANKET_PAIRS))
    report = {
        "schema": "ddm_fs3_carrier_price_density_ladder.v1",
        "arm": "ddm_fs3",
        "leg": 2,
        "adjudicates": "ddm_fs1 ERRATUM E1 -- the jg1 re-solve encoding price",
        "axis": "EXACT -- archive.zip stat, parse-back verified per rung",
        "score_claim": False,
        "promotion_eligible": False,
        "body": {
            "runtime": str(runtime),
            "archive_sha256": shipped_sha,
            "archive_bytes": shipped_bytes,
        },
        "base_codes": {
            "path": str(args.br1_codes),
            "sha256": sha256_of_bytes(Path(args.br1_codes).read_bytes()),
        },
        "carrier_move_in_the_shipped_body": {
            "pairs_moved": int(moved.size),
            "coefficients_moved": coeffs_moved,
            "coefficients_per_moved_pair": coeffs_moved / int(moved.size),
        },
        "identity_control": {
            "rebuilt_sha256": identity["archive_sha256"],
            "rebuilt_bytes": int(identity["archive_size"]),
            "byte_identical_to_shipped": identity_ok,
            "verdict": "PASS" if identity_ok else "DRIFT_ALLOWED_BY_FLAG",
        },
        "breakeven_median_b_per_pair": BREAKEVEN_MEDIAN_B_PER_PAIR,
        "breakeven_mean_b_per_pair": BREAKEVEN_MEAN_B_PER_PAIR,
        "candidate_prices_at_source": {
            "na10_stated": 0.83,
            "na10_internal_arithmetic": 5.0 / 7200.0 * 12.0,
            "up3_absorbed": 0.08,
            "up3_isolated_low": 27.0,
            "up3_isolated_high": 36.0,
            "jg5_measured_455_pair_splice": 45.0 / 455.0,
            "fs1_misread_coefficient_count_entered_as_price": 10.5,
        },
        "ladder": rows,
        "nearest_rung_to_the_blanket_27": blanket,
    }
    out_json = out_dir / "FS3_CARRIER_PRICE_LADDER.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nidentity control: {'PASS' if identity_ok else 'DRIFT'}")
    print(
        f"nearest rung to the blanket-{BLANKET_PAIRS}: "
        f"{blanket['pairs_reverted']} pairs at {blanket['bytes_per_pair']:.4f} B/pair "
        f"-> median breakeven {BREAKEVEN_MEDIAN_B_PER_PAIR}: "
        f"{'CLEARS' if blanket['clears_median_breakeven'] else 'FAILS'}"
    )
    print(f"wrote {out_json}")
    return 0


def run_rescreen(args: argparse.Namespace) -> int:
    """E4 -- re-screen fs1's js6b rows with the population-defective rows DROPPED.

    fs1 sec.4 refused the pose actuator's population defect (a prior sampled from
    pairs selected for being credits, applied to a population that is not).  fs1
    sec.5's js6b re-screen carries the same class and did not apply the same cure:
    18 of its 200 rows sit on pairs jg5 never edited, and one of the two
    median-calibration admits is among them.

    The rate leg is also re-stated.  fs1 priced every row at qs2's 5.667 B/pair,
    sourced from ``ddm_na10:562`` -- the same line whose jg1 clause the erratum
    found misread.  Leg 2 MEASURES the carrier-compensation half of that price at
    essentially zero, so the screen is reported at both the original price and at a
    compensation-free price, with the edit-encoding half declared UNMEASURED so no
    admit can be cited off the cheaper column alone.
    """
    payload = json.loads(Path(args.fs1_receipt).read_text())
    screen = payload["js6b_compensated_rescreen"]
    rows = screen["per_row"]
    unedited = set(payload["unbanked_pair_indices"]["unedited"])
    dropped = set(payload["unbanked_pair_indices"]["dropped"])

    defective = [r for r in rows if int(r["pair"]) in unedited]
    clean = [r for r in rows if int(r["pair"]) not in unedited]

    s_per_byte = S_PER_ARCHIVE_BYTE
    original_b = float(screen["rate_model"]["bytes_per_pair"])

    def screen_rows(subset: list[dict[str, Any]], bytes_per_pair: float, c: float):
        out = []
        for row in subset:
            rate = bytes_per_pair * s_per_byte
            for bound in ("lower", "upper"):
                risk = float(row[f"pose_risk_{bound}_s"]) / c
                net = -float(row["optimistic_seg_value_s"]) + risk + rate
                if net < 0:
                    out.append(
                        {
                            "pair": int(row["pair"]),
                            "proposal_id": row["proposal_id"],
                            "bound": bound,
                            "net_delta_s": net,
                            "on_a_pair_jg5_dropped": int(row["pair"]) in dropped,
                        }
                    )
        return out

    result: dict[str, Any] = {
        "schema": "ddm_fs3_js6b_rescreen_population_cured.v1",
        "arm": "ddm_fs3",
        "leg": "2 / E4",
        "axis": "[macOS-CPU advisory, scorer-free arithmetic over fs1's retained rows]",
        "score_claim": False,
        "promotion_eligible": False,
        "source_receipt": {
            "path": str(args.fs1_receipt),
            "sha256": sha256_of_bytes(Path(args.fs1_receipt).read_bytes()),
        },
        "population_cure": {
            "rows_total": len(rows),
            "rows_on_pairs_jg5_never_edited": len(defective),
            "rows_retained": len(clean),
            "defective_pairs": sorted({int(r["pair"]) for r in defective}),
            "why": (
                "a pair jg5 never edited has no measured edit and no measured "
                "compensation on this vehicle; fs1 sec.4 refused exactly this "
                "transfer for the pose actuator and sec.5 did not apply the cure"
            ),
        },
        "rate_legs": {
            "fs1_original_b_per_pair": original_b,
            "fs1_original_provenance": screen["rate_model"]["provenance"],
            "leg2_measured_carrier_compensation_b_per_pair": (
                args.compensation_bytes_per_pair
            ),
            "edit_encoding_half": "UNMEASURED on this vehicle for js6b semantic cells",
        },
        "screens": {},
    }

    for label, bpp in (
        ("fs1_original_5p667_b_per_pair", original_b),
        ("compensation_free_leg2_price", args.compensation_bytes_per_pair),
    ):
        per_c = {}
        for c_label, c in (("c=1", 1.0), ("c=8.11338", 8.11338), ("c=13.7356", 13.7356)):
            before = screen_rows(rows, bpp, c)
            after = screen_rows(clean, bpp, c)
            per_c[c_label] = {
                "admits_before_population_cure": len(before),
                "admits_after_population_cure": len(after),
                "admits_removed_by_the_cure": len(before) - len(after),
                "best_net_delta_s_after_cure": (
                    min(r["net_delta_s"] for r in after) if after else None
                ),
                "admits_after_cure_on_pairs_jg5_dropped": sum(
                    1 for r in after if r["on_a_pair_jg5_dropped"]
                ),
            }
        result["screens"][label] = per_c

    control = result["screens"]["fs1_original_5p667_b_per_pair"]["c=1"]
    result["positive_control_c1_reproduces_js6b_zero_survivors"] = (
        control["admits_before_population_cure"] == 0
    )
    result["admission_boundary"] = screen["admission_boundary"]
    result["verdict"] = (
        "No js6b row may be cited as an admit. The population cure removes rows, and "
        "the cheaper rate column rests on a carrier-compensation price that leg 2 "
        "measured but an EDIT-encoding price that nobody has measured for js6b's "
        "semantic cells."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"js6b rows {len(rows)}; on pairs jg5 never edited: {len(defective)}; "
        f"retained {len(clean)}"
    )
    for label, per_c in result["screens"].items():
        print(f"  {label}")
        for c_label, row in per_c.items():
            print(
                f"    {c_label:12s} admits {row['admits_before_population_cure']:3d} "
                f"-> {row['admits_after_population_cure']:3d} "
                f"(cure removed {row['admits_removed_by_the_cure']})"
            )
    print(
        "positive control (c=1, original price, 0 survivors): "
        f"{result['positive_control_c1_reproduces_js6b_zero_survivors']}"
    )
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    ladder = sub.add_parser("ladder", help="measure the carrier price vs density")
    ladder.add_argument("--body-runtime", default=DEFAULT_BODY_RUNTIME)
    ladder.add_argument("--expect-body-sha256", default=None)
    ladder.add_argument("--br1-codes", default=DEFAULT_BR1_CODES)
    ladder.add_argument("--out", required=True)
    ladder.add_argument(
        "--rungs",
        type=int,
        nargs="+",
        default=[1, 3, 9, 27, 55, 110, 227],
        help="pair counts to revert; the full moved set is always appended",
    )
    ladder.add_argument("--allow-identity-drift", action="store_true")
    ladder.set_defaults(func=run_ladder)

    rescreen = sub.add_parser("rescreen", help="E4 -- js6b re-screen, population cured")
    rescreen.add_argument(
        "--fs1-receipt",
        default=(
            "/Volumes/APDataStore/pact/ddm_fs1/retained/"
            "FS1_COMPOSITION_LAW_HEADROOM.json"
        ),
    )
    rescreen.add_argument(
        "--compensation-bytes-per-pair",
        type=float,
        default=0.0,
        help="leg 2's measured carrier-compensation price",
    )
    rescreen.add_argument("--out", required=True)
    rescreen.set_defaults(func=run_rescreen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
