#!/usr/bin/env python3
"""ddm_pd3: POSE-DIRECTED token pre-distortion, PASS 3, on the MOVE-51 field.

pd1 (`experiments/ddm_pd1_pose_directed.py`) is the reference form and pd2
(`experiments/ddm_pd2_pose_directed_pass2.py`) is the pass-2 binding; both are REUSED here
rather than re-implemented.  This module supplies only what is move-51 specific: (a) the
MOVE-51 binding every pd1/sj1 stage needs, (b) move 51's own pose base measured on the
decode the T4 row scored, (c) the economics at move 51's operating point -- which EXPIRE at
every pointer move and are therefore re-derived, never quoted -- and (d) the pass-3 walk
plan over three MEASURED strata.  The actuator, the screen, the refine, the pricer and the
admission are pd1's and sj1's, unchanged; the carry ranking is pd2's own helper functions
called with move-51 economics.

THE BINDING, stated once.  `ddm_pp1_pose_actuation` and `ddm_sj1_multipass_token_predistortion`
pin the live pointer at MOVE 49 in module globals evaluated at import time, and
`ddm_pd2_pose_directed_pass2` pins MOVE 50.  Move 51 is a real pointer move, so every one
of those globals is STALE for this arm.  This module re-binds them PROCESS-LOCALLY (never
on disk, never for any other process) and refuses to proceed unless every re-bound object
reproduces its declared sha256 and the row's own score arithmetic reproduces the packet's S
from its three components.  The shared modules are not edited: an arm that does not import
this one is unaffected.

Stages
------
``bind``    verify + re-bind move 51, write the receipt, touch nothing else.
``base``    move 51's own per-pair d_pose, n600, on the decode the T4 row scored.
``prereg``  the economics at move 51 + the three walk strata + the falsifiers.
``plan``    the shard assignment, ranked by MEASURED expected credit per pair.
``run``     bind, then dispatch a pd1 / pd2 / sj1 argv inside this process.
``carry``   per-bit carry: best proposal per pair, ranked by resolved-pose credit per BIT.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_pd2_pose_directed_pass2 as pd2  # the path above must be set first

N_PAIRS = pd2.N_PAIRS
EVAL_H, EVAL_W = pd2.EVAL_H, pd2.EVAL_W
CELL_COUNT = pd2.CELL_COUNT
S_PER_BYTE = pd2.S_PER_BYTE

#: pp1 measured these twelve as a FLOOR the frame_1 render cannot lower.  Excluded here on
#: pp1's measurement, exactly as pd1 and pd2 excluded them; nothing in this arm reopens them.
FLOOR_PAIRS = pd2.FLOOR_PAIRS

#: MOVE 51 -- the live row.  Lane ddm_pd2_pose_directed_pass2_first_measurement_20260913,
#: call fc-01M2CDRQXPWPYVJT0QAPHMJF1P, packet
#: `.omx/research/ddm_pd2_pose_directed_pass2_first_measurement_20260913_pointer_move_51_20260913.md`.
#: Every path below is another arm's custody and is opened READ-ONLY.
MOVE51_TREE = Path("/Volumes/APDataStore/pact/ddm_pd2/candidate/candidate_runtime")
MOVE51_ARCHIVE_SHA256 = "42e47d0bae1b0647d08db8a5061fe3eec6368b2169d89eb62b14f832fdb0978f"
MOVE51_ARCHIVE_BYTES = 179_285
MOVE51_SCORE_T4 = 0.1362333315680336
MOVE51_D_SEG_T4 = 0.00010305
MOVE51_D_POSE_T4 = 4.29e-06
#: move 51's OWN cold parse-back of its shipped bytes -- the decode the T4 row scored.
MOVE51_RAW = Path("/Volumes/APDataStore/pact/ddm_pd2/parseback/0.raw")
MOVE51_RAW_SHA256 = "b46351f8bee766481a3c1aca9c8d9781a34d11d0377c70aa145944353356202f"
MOVE51_RAW_BYTES = 3_662_409_600
#: the token field those bytes decode to (pd2's parse-back: decoded_field_matches_admitted).
MOVE51_FIELD = Path("/Volumes/APDataStore/pact/ddm_pd2/admission/field_admitted.npz")
MOVE51_FIELD_SHA256 = "e9c47f0f7c13c710ccc08c2014b58182df752374269acd02fca3db95c5563252"
#: the n600 argmax of that decode -- move 51's own seg receipt, 12,148 flipped cells.
MOVE51_ARGMAX = Path("/Volumes/APDataStore/pact/ddm_pd2/seg_final/argmax_n600.npy")
MOVE51_ARGMAX_SHA256 = "b0f35bd5ca12c86731f2df468b34a4e6bcc1572b3961a05fa759a347f6fc5799"
#: move 51's instrument seg leg on that decode, DALI lineage: 12,148 / (600*384*512).
MOVE51_FLIPS = 12_148
MOVE51_INSTRUMENT_D_SEG = 12_148 / CELL_COUNT
#: the forty pairs move 51 itself edited; their renders MOVED, so pass 3 re-searches them
#: from the new render rather than reusing a pd2 row measured on the move-50 render.
MOVE51_EDITED_PAIRS = Path("/Volumes/APDataStore/pact/ddm_pd2/admission/kept_pairs.json")

#: pd2's MEASURED price for an isolated one-token-per-pair edit on this coder and this
#: container: 16.630 bits per changed token on its 165-token field, 15.400 on the 40-token
#: subset it shipped (pd1 measured 16.98 on a 41-token field).  MEASURED on three fields at
#: this edit shape -- not a law.  Re-measured by this arm's own real encode before anything
#: is admitted; the search price only sets which pairs are WALKED.
PD2_MEASURED_BITS_PER_TOKEN_FULL_FIELD = 16.630
PD2_MEASURED_BITS_PER_TOKEN_SUBSET = 15.400

#: pd1's MEASURED absolute batch-1 vs batch-8 band over 40 pairs spanning the whole range:
#: max |gap| 2.2642595483754955e-09, gate = 10x.  The instrument did not change between
#: move 50 and move 51, so the band is pd2's unchanged; this arm re-verifies it against the
#: MOVE-51 base on pd2's own realized rows (falsifier F3's control).
BASE_BAND_ABS = pd2.BASE_BAND_ABS

#: pd2's MEASURED per-stratum admitted fraction at move 50 (its own search, K_refine 8).
#: tier 1: 38 admitted of 153 walked.  tier 2: 2 of 12.  Credited-set re-walk: 40 of 165.
#: These are PRIORS for the pass-3 ranking, measured on ONE field at ONE depth.
PD2_ADMITTED_FRACTION_TIER1 = 38.0 / 153.0
PD2_ADMITTED_FRACTION_TIER2 = 2.0 / 12.0
PD2_ADMITTED_FRACTION_CREDITED = 40.0 / 165.0

#: pd2's own search artefacts -- read-only, used to derive the three pass-3 strata.
PD2_STORE = Path("/Volumes/APDataStore/pact/ddm_pd2")


class Pd3Error(RuntimeError):
    """A pd3 input or invariant is not what the measured object says it is."""


sha256_file = pd2.sha256_file
composed_score = pd2.composed_score
read_rows = pd2.read_rows
load_pair_bits = pd2.load_pair_bits
credit_per_bit = pd2.credit_per_bit


def seg_s_per_cell() -> float:
    """S carried by ONE flipped cell on the T4 axis, through move 51's instrument ratio."""
    ratio = MOVE51_D_SEG_T4 / MOVE51_INSTRUMENT_D_SEG
    return 100.0 * ratio / CELL_COUNT


def pose_s_per_pair_unit(base_mean: float) -> float:
    """dS / d(one pair's d_pose) at this base mean.  EXPIRES at every pointer move."""
    return math.sqrt(10.0 / base_mean) / (2.0 * N_PAIRS)


def edited_pairs_at_move51() -> list[int]:
    return sorted(int(p) for p in json.loads(MOVE51_EDITED_PAIRS.read_text()))


# ----------------------------------------------------------------------------------
# THE BINDING
# ----------------------------------------------------------------------------------


def bind_move51(*, verify_raw: bool = False) -> dict[str, Any]:
    """Re-point pp1, sj1 and pd2 at MOVE 51, process-locally, fail closed if anything drifts.

    Three shared modules bind a pointer in globals evaluated at import time: pp1 and sj1 at
    move 49, pd2 at move 50.  Re-binding is therefore mandatory, and it is exactly the kind
    of silent rebinding that hands a successor a wrong number, so every object is checked
    against its declared sha BEFORE it is bound and the pointer's own score arithmetic is
    re-derived from its three components afterwards.
    """
    import ddm_pd1_pose_directed as pd1
    import ddm_pp1_pose_actuation as pp1
    import ddm_sj1_multipass_token_predistortion as sj1

    archive = MOVE51_TREE / "archive.zip"
    observed_bytes = archive.stat().st_size
    if observed_bytes != MOVE51_ARCHIVE_BYTES:
        raise Pd3Error(f"move-51 archive is {observed_bytes} B, not {MOVE51_ARCHIVE_BYTES}")
    observed_sha = sha256_file(archive)
    if observed_sha != MOVE51_ARCHIVE_SHA256:
        raise Pd3Error(f"move-51 archive sha is {observed_sha}, not {MOVE51_ARCHIVE_SHA256}")
    argmax_sha = sha256_file(MOVE51_ARGMAX)
    if argmax_sha != MOVE51_ARGMAX_SHA256:
        raise Pd3Error(f"move-51 argmax sha is {argmax_sha}, not {MOVE51_ARGMAX_SHA256}")
    field_sha = sha256_file(MOVE51_FIELD)
    if field_sha != MOVE51_FIELD_SHA256:
        raise Pd3Error(f"move-51 field sha is {field_sha}, not {MOVE51_FIELD_SHA256}")
    raw_sha = None
    raw_bytes = MOVE51_RAW.stat().st_size
    if raw_bytes != MOVE51_RAW_BYTES:
        raise Pd3Error(f"move-51 decode is {raw_bytes} B, not {MOVE51_RAW_BYTES}")
    if verify_raw:
        raw_sha = sha256_file(MOVE51_RAW)
        if raw_sha != MOVE51_RAW_SHA256:
            raise Pd3Error(f"move-51 decode sha is {raw_sha}, not {MOVE51_RAW_SHA256}")

    recomputed = composed_score(MOVE51_D_SEG_T4, MOVE51_D_POSE_T4, MOVE51_ARCHIVE_BYTES)
    if abs(recomputed - MOVE51_SCORE_T4) > 1e-15:
        raise Pd3Error(
            f"move 51's three components recompute to {recomputed!r}, not the packet's "
            f"{MOVE51_SCORE_T4!r}"
        )

    row = sj1.PointerRow(
        label="pd2_pose_directed_pass2_move51",
        tree=MOVE51_TREE,
        archive_sha256=MOVE51_ARCHIVE_SHA256,
        archive_bytes=MOVE51_ARCHIVE_BYTES,
        d_seg_t4=MOVE51_D_SEG_T4,
        d_pose_t4=MOVE51_D_POSE_T4,
        score_t4=MOVE51_SCORE_T4,
    )
    row.verify_arithmetic()

    for module in (sj1, pp1):
        module.POINTER_TREE = MOVE51_TREE
        module.POINTER_ARCHIVE = archive
        module.POINTER_ARCHIVE_SHA256 = MOVE51_ARCHIVE_SHA256
        module.POINTER_ARCHIVE_BYTES = MOVE51_ARCHIVE_BYTES
        module.POINTER_SCORE_T4 = MOVE51_SCORE_T4
        module.POINTER_D_SEG_T4 = MOVE51_D_SEG_T4
        module.POINTER_D_POSE_T4 = MOVE51_D_POSE_T4
    sj1.LIVE_POINTER = row
    pp1.POINTER = row
    pp1.LIVE_RAW = MOVE51_RAW
    pp1.LIVE_FIELD = MOVE51_FIELD
    pp1.LIVE_ARGMAX = MOVE51_ARGMAX

    # pd2's own move-50 constants are re-pointed too, because pd2 stages are reachable
    # through ``run --module pd2`` and its helpers read these at call time.
    pd2.MOVE50_TREE = MOVE51_TREE
    pd2.MOVE50_ARCHIVE_SHA256 = MOVE51_ARCHIVE_SHA256
    pd2.MOVE50_ARCHIVE_BYTES = MOVE51_ARCHIVE_BYTES
    pd2.MOVE50_SCORE_T4 = MOVE51_SCORE_T4
    pd2.MOVE50_D_SEG_T4 = MOVE51_D_SEG_T4
    pd2.MOVE50_D_POSE_T4 = MOVE51_D_POSE_T4
    pd2.MOVE50_RAW = MOVE51_RAW
    pd2.MOVE50_RAW_SHA256 = MOVE51_RAW_SHA256
    pd2.MOVE50_FIELD = MOVE51_FIELD
    pd2.MOVE50_ARGMAX = MOVE51_ARGMAX
    pd2.MOVE50_ARGMAX_SHA256 = MOVE51_ARGMAX_SHA256
    pd2.MOVE50_FLIPS = MOVE51_FLIPS
    pd2.MOVE50_INSTRUMENT_D_SEG = MOVE51_INSTRUMENT_D_SEG
    pd2.MOVE50_EDITED_PAIRS = str(MOVE51_EDITED_PAIRS)

    # pd1 carries MOVE-49 constants of its own, and `assemble` reads its seg-cell price from
    # them.  With `--carry-all` and one row per pair that price cannot change which row is
    # kept -- MEASURED: the move-49 ratio is 1.000663 against move 51's 1.000681, 0.0018 %
    # apart -- but leaving a stale pointer bound in a module this arm calls is exactly the
    # silent-rebinding hazard, so it is re-pointed and re-asserted like the others.
    pd1.POINTER_ARCHIVE_SHA256 = MOVE51_ARCHIVE_SHA256
    pd1.POINTER_ARCHIVE_BYTES = MOVE51_ARCHIVE_BYTES
    pd1.POINTER_SCORE_T4 = MOVE51_SCORE_T4
    pd1.POINTER_D_SEG_T4 = MOVE51_D_SEG_T4
    pd1.POINTER_D_POSE_T4 = MOVE51_D_POSE_T4
    pd1.INSTRUMENT_BASE_D_SEG = MOVE51_INSTRUMENT_D_SEG

    # Re-assert THROUGH the module that will be used, not through this one's copy of the
    # facts: if any global above were missed, this call still reports its predecessor.
    receipts = pp1.assert_pointer_identity()
    for name, expected in (
        ("pointer_archive_sha256", MOVE51_ARCHIVE_SHA256),
        ("live_raw", str(MOVE51_RAW)),
        ("live_field", str(MOVE51_FIELD)),
        ("live_argmax", str(MOVE51_ARGMAX)),
    ):
        if str(receipts[name]) != str(expected):
            raise Pd3Error(f"after binding, pp1 still reports {name}={receipts[name]!r}")
    if sj1.assert_carrier_is_pointer(MOVE51_TREE) != MOVE51_ARCHIVE_SHA256:
        raise Pd3Error("sj1's carrier anchor did not re-bind to move 51")
    if pd2.seg_s_per_cell() != seg_s_per_cell():
        raise Pd3Error("pd2's seg cell price did not re-bind to move 51")
    if pd1.seg_s_per_cell() != seg_s_per_cell():
        raise Pd3Error("pd1's seg cell price did not re-bind to move 51")
    receipts.update({
        "bound_move": 51,
        "argmax_sha256": argmax_sha,
        "field_sha256": field_sha,
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha,
        "score_recomputed_from_components": recomputed,
        "instrument_d_seg_move51": MOVE51_INSTRUMENT_D_SEG,
        "instrument_flips_move51": MOVE51_FLIPS,
        "edited_pairs_at_move51": edited_pairs_at_move51(),
    })
    return receipts


def cmd_bind(args) -> int:
    receipts = bind_move51(verify_raw=bool(args.verify_raw))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ddm_pd3_bind.v1",
        "axis": "[macOS-CPU advisory / identity only]",
        "score_claim": False,
        "receipts": receipts,
    }, indent=1, sort_keys=True))
    print(json.dumps(receipts, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: base -- move 51's own per-pair d_pose, n600
# ----------------------------------------------------------------------------------


def cmd_base(args) -> int:
    receipts = bind_move51(verify_raw=bool(args.verify_raw))
    import ddm_pp1_pose_actuation as pp1
    import ddm_up2_shipping_pose_solve as up2

    pp1.set_threads(args.threads)
    started = time.time()
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Pd3Error(f"move-51 carrier codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, _poses = up2.measure_pose(
        inst.posenet, inst.state, coefficients, inst.raw, inst.targets, indices,
        batch_size=args.batch_size,
    )
    mean = float(per_pair.mean())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = out_dir / "pose_base_move51.npy"
    np.save(per_pair_path, per_pair)
    np.save(out_dir / "codes_move51.npy", codes)
    order = np.argsort(-per_pair)
    report = {
        "schema": "ddm_pd3_pose_base.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "solver": (
            f"up2.measure_pose at batch {args.batch_size} on move 51's own cold parse-back"
        ),
        "receipts": receipts,
        "pairs": N_PAIRS,
        "d_pose_mean": mean,
        "pose_leg": math.sqrt(10.0 * mean),
        "t4_print": MOVE51_D_POSE_T4,
        "instrument_ratio_vs_t4_print": mean / MOVE51_D_POSE_T4,
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "top_12_share": float(per_pair[order[:12]].sum() / per_pair.sum()),
        "top_12_pairs": [int(p) for p in order[:12]],
        "per_pair_path": str(per_pair_path),
        "per_pair_sha256": sha256_file(per_pair_path),
        "codes_sha256": sha256_file(out_dir / "codes_move51.npy"),
        "elapsed_seconds": time.time() - started,
    }
    if args.control:
        control = np.load(args.control)
        if control.shape != per_pair.shape:
            raise Pd3Error(f"control vector has shape {control.shape}")
        edited = set(edited_pairs_at_move51())
        untouched = np.array([p for p in range(N_PAIRS) if p not in edited], dtype=np.int64)
        report["control"] = {
            "path": str(args.control),
            "what": (
                "pd2 measured its composed resolved pose on the OVERLAY renders; this base "
                "is measured on the SHIPPED decode, so a difference is the overlay-vs-decode "
                "class and is reported, not asserted away"
            ),
            "max_abs_difference_all_pairs": float(np.abs(control - per_pair).max()),
            "max_abs_difference_untouched_pairs": float(
                np.abs(control[untouched] - per_pair[untouched]).max()
            ),
            "mean_control": float(control.mean()),
            "bit_identical": bool(np.array_equal(control, per_pair)),
        }
    (out_dir / "POSE_BASE_MOVE51.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "receipts"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: prereg -- the economics at move 51, re-derived, and the three walk strata
# ----------------------------------------------------------------------------------


def pd2_walked_and_carried() -> tuple[set[int], set[int]]:
    """The pairs pd2 REALIZED a row on, and the pairs it CARRIED a credit for.

    Read from pd2's own retained rows rather than from its memo, so the strata are derived
    from the artefacts and a missing file is an error rather than a silent empty stratum.
    """
    walked: set[int] = set()
    row_files = sorted(PD2_STORE.glob("search/search_b_*.jsonl")) + sorted(
        PD2_STORE.glob("smoke/search_b_*.jsonl")
    )
    if not row_files:
        raise Pd3Error(f"no pd2 search rows under {PD2_STORE}")
    for row in read_rows(row_files):
        walked.add(int(row["pair"]))
    ranking_path = PD2_STORE / "assemble" / "CARRY_RANKING.json"
    ranking = json.loads(ranking_path.read_text())
    carried = {int(entry["pair"]) for entry in ranking["ranking"]}
    return walked, carried


def cmd_prereg(args) -> int:
    base = np.load(args.base_pose)
    if base.shape != (N_PAIRS,):
        raise Pd3Error(f"base pose vector has shape {base.shape}")
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    edited = edited_pairs_at_move51()
    walked, carried = pd2_walked_and_carried()

    prices = []
    for bits in sorted({
        8.93, PD2_MEASURED_BITS_PER_TOKEN_SUBSET, PD2_MEASURED_BITS_PER_TOKEN_FULL_FIELD,
        16.98, float(args.bits_per_token),
    }):
        rate_S = (bits / 8.0) * S_PER_BYTE
        need = rate_S / pose_unit
        able = [int(p) for p in np.where(base > need)[0] if int(p) not in FLOOR_PAIRS]
        prices.append({
            "bits_per_token": bits,
            "rate_S_per_token": rate_S,
            "required_per_pair_credit_d_pose": need,
            "non_floor_pairs_able_to_pay": len(able),
            "provenance": {
                8.93: "pass 8's clustered field, MEASURED there",
                PD2_MEASURED_BITS_PER_TOKEN_SUBSET:
                    "pd2's 40-token admitted subset, MEASURED by its own real encode",
                PD2_MEASURED_BITS_PER_TOKEN_FULL_FIELD:
                    "pd2's 165-token full field, MEASURED by its own real encode",
                16.98: "pd1's 41-token field, MEASURED by its own real encode",
            }.get(bits, "this arm's declared search price"),
        })

    rate_S = (float(args.bits_per_token) / 8.0) * S_PER_BYTE
    need_pose_only = rate_S / pose_unit
    # A tier-2 pair pays only if the same edit ALSO repairs one seg cell: the pose then has
    # to supply the rate cost MINUS one repaired cell.  DERIVED from the two prices above.
    need_with_one_repair = max(rate_S - seg_cell, 0.0) / pose_unit
    order = [int(p) for p in np.argsort(-base) if int(p) not in FLOOR_PAIRS]
    tier1 = [p for p in order if float(base[p]) > need_pose_only]
    tier2 = [
        p for p in order
        if need_with_one_repair < float(base[p]) <= need_pose_only
    ]

    strata = {
        "R_recredited_rewalk_deeper_K": {
            "pairs": [p for p in order if p in carried],
            "admitted_fraction_prior": PD2_ADMITTED_FRACTION_CREDITED,
            "prior_provenance": (
                "pd2 MEASURED 40 admitted of 165 carried at K_refine 8; the pass-3 gain on "
                "these pairs is the DELTA from a deeper refine and is UNMEASURED, so this "
                "prior is optimistic for them by construction"
            ),
        },
        "U1_unwalked_tier1": {
            "pairs": [p for p in tier1 if p not in walked],
            "admitted_fraction_prior": PD2_ADMITTED_FRACTION_TIER1,
            "prior_provenance": "pd2 MEASURED 38 admitted of 153 tier-1 pairs walked",
        },
        "U2_unwalked_tier2": {
            "pairs": [p for p in tier2 if p not in walked],
            "admitted_fraction_prior": PD2_ADMITTED_FRACTION_TIER2,
            "prior_provenance": "pd2 MEASURED 2 admitted of 12 tier-2 pairs walked",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ddm_pd3_prereg.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "base_pose_path": str(args.base_pose),
        "base_pose_sha256": sha256_file(Path(args.base_pose)),
        "base_mean": base_mean,
        "economics": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_seg_cell_T4_carried": seg_cell,
            "S_per_unit_of_one_pairs_d_pose": pose_unit,
            "note": "DERIVED at move 51's base mean; EXPIRES at the next pointer move",
        },
        "prices": prices,
        "search_price_bits_per_token": float(args.bits_per_token),
        "required_credit_pose_only": need_pose_only,
        "required_credit_with_one_seg_repair": need_with_one_repair,
        "hard_bound": (
            "a single-token edit cannot drive a pair's resolved d_pose below zero, so "
            "|credit| <= base and a pair whose base is under the required credit can never "
            "admit with one token at this price.  DERIVED, not measured."
        ),
        "tier1_size": len(tier1),
        "tier2_size": len(tier2),
        "pd2_walked": len(walked),
        "pd2_carried": len(carried),
        "strata": {k: {**v, "size": len(v["pairs"])} for k, v in strata.items()},
        "floor_pairs_excluded": list(FLOOR_PAIRS),
        "move51_edited_pairs_researched_from_new_renders": edited,
        "pre_registered_band_net_dS": [-6e-05, -2e-05],
        "falsifiers": [
            "F1 the admitted set projects net > -2e-05 on the RESOLVED pose with real-encode rate",
            "F2 the composition realizes < 0.8 of the per-pair credit sum",
            "F3 any scored row's batch-1 base is outside the MEASURED absolute band "
            f"{BASE_BAND_ABS:.6e} from the n600 base",
            "F4 the control encode does not reproduce move 51's own 119,055 B token stream",
            "F5 an admitting pair's carrier re-solve returns the codes it started from",
            "F6 the candidate's own parse-back decodes a field that is not the admitted field",
            "F7 the shipped-mode decode's seg leg disagrees with the admission's prediction",
            "F8 a deeper K_refine does not dominate pd2's own K=8 rows on the re-walked pairs",
        ],
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({
        k: (v if k != "strata" else {n: s["size"] for n, s in report["strata"].items()})
        for k, v in report.items()
        if k not in ("falsifiers",)
    }, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: plan -- shards ranked by MEASURED expected credit
# ----------------------------------------------------------------------------------


def cmd_plan(args) -> int:
    prereg = json.loads(Path(args.prereg).read_text())
    base = np.load(args.base_pose)
    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, stratum in prereg["strata"].items():
        prior = float(stratum["admitted_fraction_prior"])
        for pair in stratum["pairs"]:
            pair = int(pair)
            if pair in seen:
                continue
            seen.add(pair)
            ranked.append({
                "pair": pair,
                "stratum": name,
                "base_d_pose": float(base[pair]),
                "admitted_fraction_prior": prior,
                "expected_credit_d_pose": float(base[pair]) * prior,
            })
    ranked.sort(key=lambda entry: -entry["expected_credit_d_pose"])
    shards: dict[str, list[int]] = {str(i): [] for i in range(args.shard_count)}
    for index, entry in enumerate(ranked):
        shards[str(index % args.shard_count)].append(entry["pair"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ddm_pd3_search_plan.v1",
        "score_claim": False,
        "rule": (
            "expected credit = this pair's MEASURED move-51 base d_pose x pd2's MEASURED "
            "admitted fraction for its stratum; round-robin over that order across shards so "
            "a prefix stop keeps a near-uniform prefix of the global ranking"
        ),
        "shard_count": args.shard_count,
        "to_search": len(ranked),
        "by_stratum": {
            name: sum(1 for e in ranked if e["stratum"] == name)
            for name in prereg["strata"]
        },
        "ranked": ranked,
        "shards": shards,
    }, indent=1, sort_keys=True))
    print(json.dumps({
        "to_search": len(ranked),
        "shard_count": args.shard_count,
        "by_stratum": {
            name: sum(1 for e in ranked if e["stratum"] == name)
            for name in prereg["strata"]
        },
        "per_shard": {k: len(v) for k, v in shards.items()},
    }, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: run -- bind, then dispatch a pd1 / pd2 / sj1 argv inside THIS process
# ----------------------------------------------------------------------------------


def cmd_run(args) -> int:
    bind_move51(verify_raw=False)
    if args.module == "pd1":
        import ddm_pd1_pose_directed as target
    elif args.module == "pd2":
        target = pd2
    elif args.module == "joint":
        import ddm_sj1_joint_admission as target
    elif args.module == "pp1":
        import ddm_pp1_pose_actuation as target
    elif args.module == "tree":
        import ddm_pd1_candidate_tree as target
    else:  # pragma: no cover - argparse restricts the choices
        raise Pd3Error(f"unknown module {args.module}")
    argv = list(args.argv)
    if argv and argv[0] == "--":  # argparse.REMAINDER keeps the separator
        argv = argv[1:]
    if not argv:
        raise Pd3Error("run needs the target's own argv after --")
    return int(target.main(argv))


# ----------------------------------------------------------------------------------
# stage: carry -- best proposal per pair, ranked by resolved-pose credit per BIT
# ----------------------------------------------------------------------------------


def cmd_carry(args) -> int:
    """pd2's carry rule, called with move-51 economics.

    The ranking functions are pd2's own (`credit_per_bit`, `load_pair_bits`, `read_rows`);
    what changes is the price of a pose unit and of a seg cell, both of which EXPIRE at a
    pointer move.  The base-band gate then refuses any row whose batch-1 base is not move
    51's -- which is how "re-search the forty pairs move 51 edited from their NEW renders"
    is enforced structurally rather than by hand.
    """
    base = np.load(args.base_pose)
    if base.shape != (N_PAIRS,):
        raise Pd3Error(f"base pose vector has shape {base.shape}")
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    default_bits = float(args.bits_per_token)
    pair_bits = load_pair_bits(Path(args.ledger) if args.ledger else None)

    rows = read_rows([Path(p) for p in args.rows])
    scored: dict[int, dict[str, Any]] = {}
    skipped_band = 0
    skipped_unrefined = 0
    provenance: dict[int, str] = {}
    for row in rows:
        pair = int(row["pair"])
        if pair in FLOOR_PAIRS:
            continue
        if "credit_d_pose" not in row or "d_pose_base" not in row:
            skipped_unrefined += 1
            continue
        gap = abs(float(row["d_pose_base"]) - float(base[pair]))
        if gap > BASE_BAND_ABS:
            skipped_band += 1
            continue
        bits = pair_bits.get(pair, default_bits)
        credit = float(row["credit_d_pose"])
        benefit = -(credit * pose_unit) - float(row["d_cells"]) * seg_cell
        entry = dict(row)
        entry.update({
            "bits_used": bits,
            "bits_source": "measured_pair_ledger" if pair in pair_bits else "declared_price",
            "dS_benefit_pose_plus_seg": -benefit,
            "dS_rate": (bits / 8.0) * S_PER_BYTE,
            "dS_modelled_one_token": -benefit + (bits / 8.0) * S_PER_BYTE,
            "credit_S_per_bit": credit_per_bit(benefit, bits),
            "fraction_of_pair_d_pose": credit / float(base[pair]) if base[pair] > 0 else math.nan,
            "row_source": row.get("row_source", "pd3"),
        })
        best = scored.get(pair)
        if best is None or entry["credit_S_per_bit"] > best["credit_S_per_bit"]:
            scored[pair] = entry
            provenance[pair] = entry["row_source"]

    keep = dict(scored) if args.carry_all else {
        p: e for p, e in scored.items() if e["dS_modelled_one_token"] < 0.0
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(keep.values(), key=lambda e: -e["credit_S_per_bit"])
    winners_by_source: dict[str, int] = {}
    for pair in keep:
        winners_by_source[provenance[pair]] = winners_by_source.get(provenance[pair], 0) + 1
    (out_dir / "CARRY_RANKING.json").write_text(json.dumps({
        "schema": "ddm_pd3_carry_ranking.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "rule": "best proposal per pair by resolved-pose credit PER BIT; pairs ordered by it",
        "ledger": str(args.ledger) if args.ledger else None,
        "declared_bits_per_token": default_bits,
        "pose_unit_S_per_pair_d_pose": pose_unit,
        "seg_cell_S": seg_cell,
        "pairs_with_measured_pair_bits": sum(
            1 for e in keep.values() if e["bits_source"] == "measured_pair_ledger"
        ),
        "rows_read": len(rows),
        "rows_outside_base_band": skipped_band,
        "rows_screen_only_not_refined": skipped_unrefined,
        "pairs_scored": len(scored),
        "pairs_carried": len(keep),
        "winning_rows_by_source": winners_by_source,
        "sum_credit_d_pose": float(sum(e["credit_d_pose"] for e in keep.values())),
        "sum_d_cells": int(sum(int(e["d_cells"]) for e in keep.values())),
        "sum_dS_modelled": float(sum(e["dS_modelled_one_token"] for e in keep.values())),
        "ranking": [
            {
                "pair": int(e["pair"]),
                "cell": [int(v) for v in e["cell"]],
                "old": int(e["old"]), "new": int(e["new"]),
                "d_cells": int(e["d_cells"]),
                "credit_d_pose": float(e["credit_d_pose"]),
                "fraction_of_pair_d_pose": float(e["fraction_of_pair_d_pose"]),
                "bits_used": float(e["bits_used"]),
                "bits_source": e["bits_source"],
                "row_source": e["row_source"],
                # strict JSON has no Infinity: a free-or-paying edit is reported as a
                # typed rank class beside a null ratio, never as a non-finite literal.
                "credit_S_per_bit": (
                    float(e["credit_S_per_bit"])
                    if math.isfinite(e["credit_S_per_bit"]) else None
                ),
                "rank_class": (
                    "priced" if math.isfinite(e["credit_S_per_bit"])
                    else ("dominating_costs_no_bits" if e["credit_S_per_bit"] > 0
                          else "dominated_costs_no_bits")
                ),
                "dS_modelled_one_token": float(e["dS_modelled_one_token"]),
            }
            for e in ranked
        ],
    }, indent=1, sort_keys=True))
    selected = out_dir / "carry_rows.jsonl"
    selected.write_text("".join(
        json.dumps({k: v for k, v in e.items() if k != "credit_S_per_bit"}) + "\n"
        for e in ranked
    ))
    print(json.dumps({
        "pairs_scored": len(scored), "pairs_carried": len(keep),
        "rows_outside_base_band": skipped_band,
        "rows_screen_only_not_refined": skipped_unrefined,
        "winning_rows_by_source": winners_by_source,
        "sum_credit_d_pose": float(sum(e["credit_d_pose"] for e in keep.values())),
        "sum_d_cells": int(sum(int(e["d_cells"]) for e in keep.values())),
        "carry_rows": str(selected),
    }, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    bind = sub.add_parser("bind", help="verify + re-bind move 51, write the receipt")
    bind.add_argument("--out", type=Path, required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.set_defaults(func=cmd_bind)

    base = sub.add_parser("base", help="move 51's own per-pair d_pose, n600")
    base.add_argument("--out-dir", type=Path, required=True)
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=8)
    base.add_argument("--control", type=Path, default=None)
    base.add_argument("--verify-raw", action="store_true")
    base.set_defaults(func=cmd_base)

    pre = sub.add_parser("prereg", help="economics at move 51 + strata + falsifiers")
    pre.add_argument("--base-pose", type=Path, required=True)
    pre.add_argument(
        "--bits-per-token", type=float, default=PD2_MEASURED_BITS_PER_TOKEN_FULL_FIELD
    )
    pre.add_argument("--out", type=Path, required=True)
    pre.set_defaults(func=cmd_prereg)

    plan = sub.add_parser("plan", help="shards ranked by MEASURED expected credit")
    plan.add_argument("--prereg", type=Path, required=True)
    plan.add_argument("--base-pose", type=Path, required=True)
    plan.add_argument("--shard-count", type=int, default=8)
    plan.add_argument("--out", type=Path, required=True)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="bind, then dispatch a pd1 / pd2 / sj1 argv in-process")
    run.add_argument("--module", choices=("pd1", "pd2", "joint", "pp1", "tree"), required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    carry = sub.add_parser("carry", help="best proposal per pair by credit PER BIT")
    carry.add_argument("--rows", nargs="+", required=True)
    carry.add_argument("--base-pose", type=Path, required=True)
    carry.add_argument("--ledger", type=Path, default=None)
    carry.add_argument(
        "--bits-per-token", type=float, default=PD2_MEASURED_BITS_PER_TOKEN_FULL_FIELD
    )
    carry.add_argument("--carry-all", action="store_true")
    carry.add_argument("--out-dir", type=Path, required=True)
    carry.set_defaults(func=cmd_carry)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
