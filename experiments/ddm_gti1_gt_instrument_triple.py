# SPDX-License-Identifier: MIT
"""ddm_gti1 -- resolve the 12.1x disagreement between two numbers both labelled "lb1's token error".

WHAT THIS SETTLES
-----------------
``.omx/research/ddm_gestalt_the_chasm_not_the_cross_20260831.md`` S4a records two figures under one
label, 12.1x apart:

    vs DALI GT (measured, count_nonzero) ..  1,717 / 117,964,800 = 0.00146%  -> 93.1% slack
    PYAV instrument (carried in hot-state)   ~20,762             = 0.01760%  -> 17.0% slack

Only one of them can be "lb1's token error", and the campaign's accuracy-for-bytes trades inherit a
5.5x factor from the choice.  This script measures the THREE pairwise mismatch counts over the same
n600 field so the arithmetic is closed rather than argued:

    lb1  vs DALI_GT   -- reproduces gf1's 1,717 (POSITIVE CONTROL, not a new number)
    DALI vs PYAV_GT   -- reproduces the gl1/gl2 lineage separation ~20,671 (POSITIVE CONTROL)
    lb1  vs PYAV_GT   -- THE ANSWER, never previously measured

and then partitions every disagreeing site into the five mutually exclusive agreement patterns, so
the third count is *explained* by the first two rather than merely reported next to them.

WHY A PARTITION AND NOT JUST THREE SCALARS
------------------------------------------
Three Hamming counts alone cannot distinguish "lb1 has its own errors on top of the lineage split"
from "lb1's apparent PyAV error IS the lineage split".  The partition can: if the ``dali==pyav !=
lb1`` cell holds essentially all of the 1,717 and the ``lb1==dali != pyav`` cell holds essentially
all of the ~20,671, then the PyAV-side figure is dominated by a property of the two GROUND TRUTHS
and not by a property of lb1.  That is a structural claim, and it needs the cell counts to make it.

POSITIVE CONTROLS ARE MANDATORY AND REPORTED, NOT ASSERTED
----------------------------------------------------------
Both controls are recorded in the receipt with an explicit PASS/FAIL and the expected value they are
checked against.  They do NOT raise: a failing control is itself the finding (it would mean the
retained fields are not the ones the prior arms measured), and a crash would destroy the receipt
that proves it.  Refusing to crash is not refusing to fail -- ``controls_all_pass`` is written into
the receipt and is the first thing a reader sees.

Axis: ``[macOS-CPU scorer-free exact count]``.  ``score_claim=false``, ``promotable=false``.
No scorer is loaded, no archive is built, no pointer is touched.  This is a count over retained
bytes; it is not a d_seg and must never be read as one.  Token-field mismatch and d_seg are
different quantities on different sides of the render->SegNet round trip (``ddm_td1``: 1,717 token
errors coexist with 34,930.6 scored flips), and conflating them is the [[m99]] units error this
arm exists to correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

#: 600 pairs x 384 x 512 scored pixels.  Declared, not inferred from a file size, so a truncated or
#: differently-shaped input is caught rather than silently redefining the denominator ([[m50]]).
FIELD_POSITIONS = 117_964_800

#: Counts the prior arms measured, checked here as positive controls.
#: gf1 ``GF1_FAMILY_CAPACITY_CROSSCHECK.json`` -> ``targets.lb1_vs_gt_mismatch``.
EXPECTED_LB1_VS_DALI = 1_717
#: ``ddm_gl2`` S1: ``gt_segnet_argmax.u8`` (n600) vs the DALI ruler = 20,672 differing sites; gl1
#: measured the same separation at 20,670/20,671/20,672 across three artifact pairs, and a1s at
#: 20,673 against the qs3 npy.  The control therefore admits a small band rather than one integer:
#: the separation is a property of the decode pair, and the exact DALI artifact varies by store.
EXPECTED_DALI_VS_PYAV_BAND = (20_600, 20_750)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Stream the sha256 of a file.  Streaming keeps the 112 MiB fields off the heap."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _open_field(path: Path, *, name: str) -> np.memmap:
    """Memory-map a raw ``uint8`` label field, refusing on a wrong length.

    The length check is the denominator check.  A field of unexpected size would still produce a
    plausible-looking mismatch count against a different denominator, which is exactly how a
    percentage becomes wrong while every number in it is real.
    """
    size = path.stat().st_size
    if size != FIELD_POSITIONS:
        raise ValueError(
            f"{name}: {path} is {size:,} B but the scored field is {FIELD_POSITIONS:,} positions. "
            "Refusing to divide by a denominator I did not verify."
        )
    return np.memmap(path, dtype=np.uint8, mode="r")


def measure(
    lb1_path: Path,
    dali_path: Path,
    pyav_path: Path,
    *,
    chunk: int = 1 << 23,
) -> dict[str, Any]:
    """Pairwise mismatch counts plus the five-cell agreement partition, in one streaming pass."""
    lb1 = _open_field(lb1_path, name="lb1_field")
    dali = _open_field(dali_path, name="dali_gt")
    pyav = _open_field(pyav_path, name="pyav_gt")

    lb1_ne_dali = 0
    lb1_ne_pyav = 0
    dali_ne_pyav = 0
    # The five mutually exclusive agreement patterns.  They sum to FIELD_POSITIONS by construction,
    # and that sum is asserted below -- a partition whose parts do not sum to the whole is a bug,
    # not a rounding difference.
    all_equal = 0
    lb1_dali_agree_pyav_differs = 0
    lb1_pyav_agree_dali_differs = 0
    gts_agree_lb1_differs = 0
    all_three_distinct = 0

    for start in range(0, FIELD_POSITIONS, chunk):
        stop = min(start + chunk, FIELD_POSITIONS)
        a = np.asarray(lb1[start:stop])
        d = np.asarray(dali[start:stop])
        p = np.asarray(pyav[start:stop])

        ad = a == d
        ap = a == p
        dp = d == p

        lb1_ne_dali += int(ad.size - np.count_nonzero(ad))
        lb1_ne_pyav += int(ap.size - np.count_nonzero(ap))
        dali_ne_pyav += int(dp.size - np.count_nonzero(dp))

        all_equal += int(np.count_nonzero(ad & dp))
        lb1_dali_agree_pyav_differs += int(np.count_nonzero(ad & ~dp))
        lb1_pyav_agree_dali_differs += int(np.count_nonzero(ap & ~ad))
        gts_agree_lb1_differs += int(np.count_nonzero(dp & ~ad))
        all_three_distinct += int(np.count_nonzero(~ad & ~ap & ~dp))

    partition = {
        "all_three_equal": all_equal,
        "lb1==dali, pyav differs": lb1_dali_agree_pyav_differs,
        "lb1==pyav, dali differs": lb1_pyav_agree_dali_differs,
        "dali==pyav, lb1 differs": gts_agree_lb1_differs,
        "all_three_distinct": all_three_distinct,
    }
    partition_sum = sum(partition.values())

    def pct(n: int) -> float:
        return 100.0 * n / FIELD_POSITIONS

    return {
        "field_positions": FIELD_POSITIONS,
        "pairwise": {
            "lb1_vs_dali_gt": {"mismatches": lb1_ne_dali, "pct": pct(lb1_ne_dali)},
            "lb1_vs_pyav_gt": {"mismatches": lb1_ne_pyav, "pct": pct(lb1_ne_pyav)},
            "dali_gt_vs_pyav_gt": {"mismatches": dali_ne_pyav, "pct": pct(dali_ne_pyav)},
        },
        "agreement_partition": partition,
        "partition_sum": partition_sum,
        "partition_sums_to_field": partition_sum == FIELD_POSITIONS,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lb1-field",
        type=Path,
        default=Path("/Volumes/APDataStore/pact/ddm_dc1_20260816/retained/redecoded_tokens_n600.u8"),
        help="the token field gf1 measured as 'lb1_field' (default: gf1's own declared path)",
    )
    ap.add_argument(
        "--dali-gt",
        type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_bz2_bornsmall_capacity_ceiling/retained/targets/dali_gt_full_n600.u8"
        ),
        help="DALI-lineage GT argmax (default: gf1's own declared path)",
    )
    ap.add_argument(
        "--pyav-gt",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/targets_n600/gt_segnet_argmax.u8"
        ),
        help="PyAV/frame_utils-lineage GT argmax (registry sha 36c6be718916..., PRODUCER_DECLARED)",
    )
    ap.add_argument("--out", type=Path, required=True, help="receipt directory (APDataStore tier)")
    args = ap.parse_args(argv)

    t0 = time.time()
    inputs = {
        "lb1_field": args.lb1_field,
        "dali_gt": args.dali_gt,
        "pyav_gt": args.pyav_gt,
    }
    custody = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in inputs.items()
    }

    result = measure(args.lb1_field, args.dali_gt, args.pyav_gt)

    lb1_dali = result["pairwise"]["lb1_vs_dali_gt"]["mismatches"]
    dali_pyav = result["pairwise"]["dali_gt_vs_pyav_gt"]["mismatches"]
    lo, hi = EXPECTED_DALI_VS_PYAV_BAND
    controls = {
        "lb1_vs_dali_reproduces_gf1_1717": {
            "expected": EXPECTED_LB1_VS_DALI,
            "measured": lb1_dali,
            "pass": lb1_dali == EXPECTED_LB1_VS_DALI,
            "source": "GF1_FAMILY_CAPACITY_CROSSCHECK.json targets.lb1_vs_gt_mismatch",
        },
        "dali_vs_pyav_reproduces_lineage_separation": {
            "expected_band": [lo, hi],
            "measured": dali_pyav,
            "pass": lo <= dali_pyav <= hi,
            "source": "ddm_gl2 S1 (20,672) / ddm_gl1 (20,670-20,672) / ddm_a1s S7 (20,673)",
        },
        "partition_sums_to_field": {
            "expected": FIELD_POSITIONS,
            "measured": result["partition_sum"],
            "pass": bool(result["partition_sums_to_field"]),
            "source": "internal consistency of the five-cell partition",
        },
    }
    controls_all_pass = all(c["pass"] for c in controls.values())

    receipt: dict[str, Any] = {
        "schema": "ddm_gti1_gt_instrument_triple.v1",
        "axis": "[macOS-CPU scorer-free exact count]",
        "score_claim": False,
        "promotable": False,
        "quantity": (
            "token-field mismatch (count_nonzero) between raw uint8 label fields. NOT d_seg: d_seg "
            "is measured after the render->SegNet round trip and is a different, larger quantity "
            "(ddm_td1: 1,717 token errors vs 34,930.6 scored flips on the same vehicle)."
        ),
        "controls_all_pass": controls_all_pass,
        "controls": controls,
        "inputs": custody,
        **result,
        "elapsed_seconds": time.time() - t0,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "GTI1_GT_INSTRUMENT_TRIPLE.json"
    out_path.write_text(json.dumps(receipt, indent=2) + "\n")

    print(json.dumps(receipt, indent=2))
    print(f"\nreceipt -> {out_path}  sha256={sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
