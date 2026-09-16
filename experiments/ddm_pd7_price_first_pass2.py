#!/usr/bin/env python3
"""ddm_pd7 -- the PRICE-FIRST generator, PASS 2, on MOVE 53's re-rendered token field.

WHY THIS EXISTS
---------------
pd6 landed move 53 by inverting the generation order: capture the shipped coder's own
probability rows, enumerate the CHEAP HALF of the plane first, and measure credit afterwards.
Two MEASURED facts make a second pass on the NEW field a different unit rather than a repeat:

1. **The field moved.**  101 of the 600 pairs carry new tokens and are therefore RE-RENDERED,
   and the repair family re-opens on a re-rendered pair
   (``repair_family_exhausts_per_object_20260910``: 375/489 new rows on re-rendered pairs, 0 on
   unchanged ones).
2. **Every price expired.**  The cheap half is a property of the PRIOR's context, and the edits
   changed that context (pc2/pc3: prices EXPIRE per move).  move 53's cheap half has never been
   enumerated.

pd6 also left measured headroom it could not spend: 279 pairs carried positive resolved-pose
credit and only 128 paid at their own price, and the sensitivity of the cheap threshold above
6 bits/token was never measured for credit.  This arm widens the enumeration to <= 8 bits and
reports the 6/8 split.

WHAT IS AND IS NOT NEW CODE
---------------------------
Every stage is pd6's, imported and called unchanged.  This module contributes exactly one
thing: it re-points pd4's move-52 constants at MOVE 53 *process-locally*, before any pd6 stage
runs, so that ``pd4.bind_move52`` -- which pd6's stages call -- binds move 53 through its own
unchanged verification path (archive sha, argmax sha, field sha, raw sha and bytes, and the
pointer's score re-derived from its three components).  This is the same silent-rebinding hazard
pd4's own docstring names, so the rebinding is explicit, is checked against declared shas, and
is re-asserted THROUGH pp1/sj1/pd1/pd2/pd3 rather than through this module's copy of the facts.

No scorer weights ship anywhere, no Modal call, no authorization, no fire, no packet, no pointer
write.  Only ``upstream/evaluate.py`` on shipped bytes is a score, and MAIN fires.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import ddm_pd6_price_first as pd6

pd4 = pd6.pd4

PD7_STORE = Path("/Volumes/APDataStore/pact/ddm_pd7")

# ----------------------------------------------------------------------------------
# MOVE 53 -- the live pointer, every value declared and every file pinned by sha
# ----------------------------------------------------------------------------------

MOVE53_TREE = PD7_STORE / "base/move53_runtime"
MOVE53_ARCHIVE_SHA256 = "aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957"
MOVE53_ARCHIVE_BYTES = 179_286
MOVE53_SCORE_T4 = 0.1361014714463198
MOVE53_D_SEG_T4 = 0.00010288
MOVE53_D_POSE_T4 = 4.14e-06

#: move 53's own token field -- pd6's ``setprice/set_02.npz``, copied into this arm's store.
MOVE53_FIELD = PD7_STORE / "base/field_move53.npz"
MOVE53_FIELD_SHA256 = "f3ab56ed0996ed1ebe247d3346e935cde4cb5ca6d846321cabe5cd76e64eb6e5"

#: the SegNet argmax of move 53's OWN COLD DECODE (pd6 ``seg_final2/argmax_n600.npy``).
MOVE53_ARGMAX = PD7_STORE / "base/argmax_move53.npy"
MOVE53_ARGMAX_SHA256 = "bed84af6ec58a32017432310ba79a1b2236bc88898421209e7f72ea56324bc5a"
MOVE53_FLIPS = 12_128

#: the per-pair n600 pose vector measured on move 53's own decode with its own carrier.
MOVE53_BASE_POSE = PD7_STORE / "base/pose_base_move53.npy"
MOVE53_BASE_POSE_SHA256 = "1c75d6539d4dc5a4c7da2773b4fe5c0d6d5b7df4befa800921fdc607c141094b"

#: move 53's cold decode.  pd6 measured it (936.7 s) and hashed it, then removed the payload
#: under its retention cap; this arm rebuilds the same bytes and re-verifies the sha.
MOVE53_RAW = PD7_STORE / "parseback_base/0.raw"
MOVE53_RAW_SHA256 = "8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b"
MOVE53_RAW_BYTES = 3_662_409_600

#: move 53's DECODED token plane.  pd6's parseback2 reported it; this arm reproduced it
#: independently, because the pricer's ``field_to_u8`` of ``field_move53.npz`` hashes to the
#: same value -- so the field npz bound here IS the plane the receiver decodes.
MOVE53_DECODED_TOKEN_SHA256 = "4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c"

MOVE53_EDITED_PAIRS = PD7_STORE / "base/edited_pairs_move53.json"

CELL_COUNT = pd4.CELL_COUNT
MOVE53_INSTRUMENT_D_SEG = MOVE53_FLIPS / CELL_COUNT


class Pd7Error(RuntimeError):
    """A pd7 input or invariant is not what the measured object says it is."""


def move53_facts() -> dict[str, Any]:
    """Every declared fact about the live row, in one place, for the receipts."""
    return {
        "tree": str(MOVE53_TREE),
        "archive_sha256": MOVE53_ARCHIVE_SHA256,
        "archive_bytes": MOVE53_ARCHIVE_BYTES,
        "score_t4": MOVE53_SCORE_T4,
        "d_seg_t4": MOVE53_D_SEG_T4,
        "d_pose_t4": MOVE53_D_POSE_T4,
        "field": str(MOVE53_FIELD),
        "field_sha256": MOVE53_FIELD_SHA256,
        "argmax": str(MOVE53_ARGMAX),
        "argmax_sha256": MOVE53_ARGMAX_SHA256,
        "instrument_flips": MOVE53_FLIPS,
        "instrument_d_seg": MOVE53_INSTRUMENT_D_SEG,
        "raw_sha256": MOVE53_RAW_SHA256,
        "raw_bytes": MOVE53_RAW_BYTES,
        "base_pose": str(MOVE53_BASE_POSE),
        "base_pose_sha256": MOVE53_BASE_POSE_SHA256,
        "decoded_token_plane_sha256": MOVE53_DECODED_TOKEN_SHA256,
    }


def rebind_pd4_to_move53() -> dict[str, Any]:
    """Re-point pd4's move-52 constants at MOVE 53, process-locally.

    pd4 binds move 52 in module globals evaluated at import time and every downstream stage
    (pd6's ``realize`` included) calls ``pd4.bind_move52``.  Rather than fork pd4 -- which would
    duplicate its verification path and make the two copies drift -- this function moves the
    constants and leaves pd4's checks in charge.  ``bind_move52`` then verifies EVERY one of
    them against the file on disk before binding anything, so a mistake here fails closed.
    """
    pd4.MOVE52_TREE = MOVE53_TREE
    pd4.MOVE52_ARCHIVE_SHA256 = MOVE53_ARCHIVE_SHA256
    pd4.MOVE52_ARCHIVE_BYTES = MOVE53_ARCHIVE_BYTES
    pd4.MOVE52_SCORE_T4 = MOVE53_SCORE_T4
    pd4.MOVE52_D_SEG_T4 = MOVE53_D_SEG_T4
    pd4.MOVE52_D_POSE_T4 = MOVE53_D_POSE_T4
    pd4.MOVE52_FIELD = MOVE53_FIELD
    pd4.MOVE52_FIELD_SHA256 = MOVE53_FIELD_SHA256
    pd4.MOVE52_ARGMAX = MOVE53_ARGMAX
    pd4.MOVE52_ARGMAX_SHA256 = MOVE53_ARGMAX_SHA256
    pd4.MOVE52_FLIPS = MOVE53_FLIPS
    pd4.MOVE52_INSTRUMENT_D_SEG = MOVE53_INSTRUMENT_D_SEG
    pd4.MOVE52_EDITED_PAIRS = MOVE53_EDITED_PAIRS
    pd4.MOVE52_DECODED_TOKEN_SHA256 = MOVE53_DECODED_TOKEN_SHA256
    pd4.PD3_RAW = MOVE53_RAW
    pd4.PD3_RAW_SHA256 = MOVE53_RAW_SHA256
    pd4.PD3_RAW_BYTES = MOVE53_RAW_BYTES
    return move53_facts()


def bound_seg_s_per_cell() -> float:
    """S carried by ONE flipped cell at MOVE 53's operating point, through its own ratio."""
    return 100.0 * (MOVE53_D_SEG_T4 / MOVE53_INSTRUMENT_D_SEG) / CELL_COUNT


def cmd_bind(args) -> int:
    """The binding receipt: rebind, let pd4 verify, then re-derive the pointer arithmetic."""
    facts = rebind_pd4_to_move53()
    receipts = pd4.bind_move52(verify_raw=bool(args.verify_raw), raw=Path(args.raw))
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    live = pointer["our_local_frontier_contest_cuda"]
    if live["archive_sha256"] != MOVE53_ARCHIVE_SHA256:
        raise Pd7Error(
            f"POINTER MOVED: the live row is {live['archive_sha256']}, not move 53's "
            f"{MOVE53_ARCHIVE_SHA256}; re-derive the base before pricing anything"
        )
    recomputed = pd4.composed_score(MOVE53_D_SEG_T4, MOVE53_D_POSE_T4, MOVE53_ARCHIVE_BYTES)
    if abs(recomputed - MOVE53_SCORE_T4) > 1e-15:
        raise Pd7Error(f"move 53 recomputes to {recomputed!r}, not {MOVE53_SCORE_T4!r}")
    if abs(pd4.seg_s_per_cell() - bound_seg_s_per_cell()) > 1e-18:
        raise Pd7Error("pd4's seg cell price did not re-bind to move 53")
    base_mean = float(np.load(MOVE53_BASE_POSE).mean())
    payload = {
        "schema": "ddm_pd7_bind.v1",
        "axis": "[macOS-CPU advisory / binding receipt, no score]",
        "score_claim": False,
        "move53": facts,
        "pd4_receipts": receipts,
        "live_pointer": {
            "archive_sha256": live["archive_sha256"],
            "score": live.get("score"),
            "lane_id": live.get("lane_id"),
            "re_derived_from": str(REPO / ".omx/state/canonical_frontier_pointer.json"),
        },
        "score_recomputed_from_components": recomputed,
        "seg_s_per_cell_move53": bound_seg_s_per_cell(),
        "base_pose_mean_n600": base_mean,
        "pose_s_per_pair_unit_move53": pd4.pose_s_per_pair_unit(base_mean),
        "s_per_byte": pd4.S_PER_BYTE,
        "floor_pairs_excluded": sorted(pd4.FLOOR_PAIRS),
        "rebinding_note": (
            "pd4's move-52 constants were re-pointed at move 53 PROCESS-LOCALLY by "
            "ddm_pd7_price_first_pass2.rebind_pd4_to_move53; pd4.bind_move52 then verified "
            "every one of them against the file on disk and re-asserted the binding THROUGH "
            "pp1/sj1/pd1/pd2/pd3.  No pd4, pd6 or shipped-runtime source was edited."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_headroom(args) -> int:
    """What credit the lever needs at MOVE 53's operating point -- DERIVED before any credit."""
    rebind_pd4_to_move53()
    base = np.load(MOVE53_BASE_POSE)
    base_mean = float(base.mean())
    pose_unit = pd4.pose_s_per_pair_unit(base_mean)
    seg_cell = bound_seg_s_per_cell()
    s_per_byte = pd4.S_PER_BYTE
    eligible = np.array([p for p in range(pd4.N_PAIRS) if p not in pd4.FLOOR_PAIRS])
    table = []
    for frac in (0.05, 0.10, 0.20):
        for bits in (3.0, 6.0, 8.0):
            gain = base[eligible] * frac * pose_unit
            cost = (bits / 8.0) * s_per_byte  # one token at this price, in S
            pays = gain > cost
            table.append({
                "recovered_fraction": frac, "bits_per_token": bits,
                "pairs_paying": int(pays.sum()),
                "net_dS": float((cost - gain)[pays].sum()),
            })
    budget = np.floor(base * pose_unit / seg_cell).astype(int)
    payload = {
        "schema": "ddm_pd7_headroom.v1",
        "axis": "[DERIVED at move 53's operating point; EXPIRES at the next pointer move]",
        "score_claim": False,
        "base_pose_mean_n600": base_mean,
        "s_per_archive_byte": s_per_byte,
        "s_per_seg_cell": seg_cell,
        "s_per_pair_pose_unit": pose_unit,
        "admit_bar_net_dS": -2e-05,
        "ladder": table,
        "seg_cell_budget_histogram": {
            "pairs_budget_0": int((budget[eligible] == 0).sum()),
            "pairs_budget_1": int((budget[eligible] == 1).sum()),
            "pairs_budget_2_or_more": int((budget[eligible] >= 2).sum()),
        },
        "method": (
            "a pair can credit at most its own d_pose; gain = base[pair]*frac*pose_unit, cost = "
            "(bits/8)*S_per_byte for one token.  The bar is -2e-05 S."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_baseband(args) -> int:
    """MEASURE the batch-1 vs n600 pose-base gap over ALL 600 pairs, and set the wave's gate.

    ``realize`` refuses a pair whose batch-1 base is further from the n600 base than the band.
    The band is therefore an INSTRUMENT tolerance, not a preference, and it belongs to the
    field it was measured on: move 53's base pose vector came from the candidate's own cold
    decode at batch 8, and the wave reads it at batch 1.  No prefix -- all 600 pairs.
    """
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_pp1_pose_actuation as pp1

    if float(args.headroom) < 1.0:
        raise Pd7Error(
            f"headroom {args.headroom} < 1 would set a band TIGHTER than the measured max gap, "
            "and realize raises on the first pair outside it -- the whole shard would die"
        )
    rebind_pd4_to_move53()
    pd4.bind_move52(verify_raw=False, raw=Path(args.raw))
    pp1.set_threads(int(args.threads))
    base = np.load(MOVE53_BASE_POSE)
    body = pp1.load_body(with_raw=True, with_segnet=False)
    inst = pp1.build_pose_instrument(body.raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    gaps = np.zeros(pd4.N_PAIRS, dtype=np.float64)
    batch1 = np.zeros(pd4.N_PAIRS, dtype=np.float64)
    for pair in range(pd4.N_PAIRS):
        batch1[pair] = float(br1.evaluate_codes(inst, pair, codes[pair][None])[0])
        gaps[pair] = abs(batch1[pair] - float(base[pair]))
    np.save(Path(args.out).with_suffix(".batch1.npy"), batch1)
    eligible = np.array([p for p in range(pd4.N_PAIRS) if p not in pd4.FLOOR_PAIRS])
    payload = {
        "schema": "ddm_pd7_baseband.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT lineage; batch 1 vs batch 8]",
        "score_claim": False,
        "pairs": int(pd4.N_PAIRS),
        "n600_base_path": str(MOVE53_BASE_POSE),
        "n600_base_mean": float(base.mean()),
        "batch1_mean": float(batch1.mean()),
        "gap_abs": {
            "max_all": float(gaps.max()),
            "max_non_floor": float(gaps[eligible].max()),
            "median": float(np.median(gaps)),
            "p99": float(np.quantile(gaps, 0.99)),
        },
        "band_abs_measured": float(gaps[eligible].max()) * float(args.headroom),
        "band_basis": ("the max gap over the NON-FLOOR pairs -- the only pairs the wave walks; "
                       "the all-pair max is reported beside it"),
        "band_headroom_factor": float(args.headroom),
        "meaning": (
            "the wave's own gate: a pair whose batch-1 base is further than the band from the "
            "n600 base is refused, because its credit would be measured against a base the "
            "shipped leg does not carry"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_price_merge(args) -> int:
    """pd4's merge, reading pd7's OWN sheet encodes.

    pd4's merge builds the field name ``pd4sheet{k:02d}``; pd6 rebound that to ``pd6sheet`` for
    exactly this reason and this arm rebinds it to ``pd7sheet``.  The rebinding is
    PROCESS-LOCAL and is restored in ``finally``; a ``pd4sheet`` name here would be a
    provenance lie, because the bits come from this arm's own encodes of its own sheets.
    """
    original = pd4._encode_bits

    def rebound(rlc1_root, field: str, tag: str):
        name = field.replace("pd4sheet", "pd7sheet") if field.startswith("pd4sheet") else field
        return original(rlc1_root, name, tag)

    pd4._encode_bits = rebound
    try:
        return pd4.cmd_price_merge(args)
    finally:
        pd4._encode_bits = original


def cmd_run(args) -> int:
    """pd4's ``run`` passthrough, bound to MOVE 53 -- the joint admission's own argv."""
    rebind_pd4_to_move53()
    pd4.bind_move52(verify_raw=False, raw=Path(args.raw))
    if args.module == "joint":
        import ddm_sj1_joint_admission as target
    elif args.module == "pp1":
        import ddm_pp1_pose_actuation as target
    elif args.module == "tree":
        import ddm_pd1_candidate_tree as target
    elif args.module == "pd4":
        target = pd4
    elif args.module == "pd5":
        import ddm_pd5_multitoken as target
    else:  # pragma: no cover - argparse restricts the choices
        raise Pd7Error(f"unknown module {args.module}")
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise Pd7Error("run needs the target's own argv after --")
    return int(target.main(argv))


def main(argv: list[str] | None = None) -> int:
    parser = pd6.build_parser()
    sub = parser._subparsers._group_actions[0]  # reuse pd6's subparser set verbatim
    bb = sub.add_parser("baseband", help="MEASURE the batch-1 vs n600 pose-base gap, all 600")
    bb.add_argument("--raw", required=True)
    bb.add_argument("--threads", type=int, default=4)
    bb.add_argument("--headroom", type=float, default=2.0)
    bb.add_argument("--out", required=True)
    bb.set_defaults(func=cmd_baseband)
    rn = sub.add_parser("run", help="run another module's argv under the MOVE 53 binding")
    rn.add_argument("--module", required=True,
                    choices=["joint", "pp1", "tree", "pd4", "pd5"])
    rn.add_argument("--raw", required=True)
    rn.add_argument("argv", nargs=argparse.REMAINDER)
    rn.set_defaults(func=cmd_run)
    # pd6's price-merge rebinds pd4's sheet name to pd6's; this arm's sheets are pd7's
    sub._name_parser_map["price-merge"].set_defaults(func=cmd_price_merge)
    bind = sub.add_parser("bind53", help="bind MOVE 53 and re-derive its arithmetic")
    bind.add_argument("--raw", required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.add_argument("--out", required=True)
    bind.set_defaults(func=cmd_bind)
    hr = sub.add_parser("headroom", help="what credit the lever needs at move 53")
    hr.add_argument("--out", required=True)
    hr.set_defaults(func=cmd_headroom)

    args = parser.parse_args(argv)
    if args.func is not cmd_bind:
        # every pd6 stage reads pd4's constants, directly or through bind_move52
        rebind_pd4_to_move53()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
