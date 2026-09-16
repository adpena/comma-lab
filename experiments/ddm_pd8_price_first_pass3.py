#!/usr/bin/env python3
"""ddm_pd8 -- the PRICE-FIRST generator, PASS 3, on MOVE 54's re-rendered token field.

WHY THIS EXISTS
---------------
pd6 (move 53) and pd7 (move 54) both landed by inverting the generation order: capture the
shipped coder's own probability rows, enumerate the CHEAP HALF of the plane first, and measure
credit afterwards.  pd7 MEASURED two facts that make a third pass a different unit rather than
a repeat, and that also predict it will be thinner:

1. **The field moved again.**  53 of the 600 pairs carry new tokens and are therefore
   RE-RENDERED, and the repair family re-opens on a re-rendered pair
   (``repair_family_exhausts_per_object_20260910``).
2. **The prices expired again, at the TOP of each pair's list.**  pd7 measured the rank-0 sheet
   at 3.075 bits/token against pd6's 2.163 (+42 %) while ranks 1-5 moved under 1.5 %.  The
   decay of the rank-0 sheet across three passes is this arm's cleanest single figure.
3. **The headroom shrinks 2-3 % per move**, because each pass lowers the base d_pose its
   successor must recover from.  pd7's own ``HEADROOM.json`` is the cell-for-cell comparand.

WHAT IS AND IS NOT NEW CODE
---------------------------
Every stage is pd6's (through pd7), imported and called unchanged.  This module contributes
exactly one thing: it re-points pd4's move-52 constants at MOVE 54 *process-locally*, before any
stage runs, so that ``pd4.bind_move52`` -- which every downstream stage calls -- binds move 54
through its own unchanged verification path (archive sha, argmax sha, field sha, raw sha and
bytes, and the pointer's score re-derived from its three components).  pd7's two documented
defects are carried as LAWS, not as code: ``setprice`` needs its ``setabsorb`` between
iterations (``measured_pairs_used`` must be non-zero before ``set_repeats_previous`` is
evidence of convergence), and the pricer's u8 bulk is prunable only after its LAST rlc1 call.

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

PD8_STORE = Path("/Volumes/APDataStore/pact/ddm_pd8")

# ----------------------------------------------------------------------------------
# MOVE 54 -- the live pointer, every value declared and every file pinned by sha
# ----------------------------------------------------------------------------------

MOVE54_TREE = PD8_STORE / "base/move54_runtime"
MOVE54_ARCHIVE_SHA256 = "5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6"
MOVE54_ARCHIVE_BYTES = 179_266
MOVE54_SCORE_T4 = 0.13605599532783202
MOVE54_D_SEG_T4 = 0.00010287
MOVE54_D_POSE_T4 = 4.1e-06

#: move 54's own token field -- pd7's ``setprice/set_03.npz``, copied into this arm's store.
MOVE54_FIELD = PD8_STORE / "base/field_move54.npz"
MOVE54_FIELD_SHA256 = "e1ec64e528df6200481dcbe57062bece28d8754550cb5c72565e58ca533e477b"

#: the SegNet argmax of move 54's OWN COLD DECODE (pd7 ``seg_cand/argmax_n600.npy``).
MOVE54_ARGMAX = PD8_STORE / "base/argmax_move54.npy"
MOVE54_ARGMAX_SHA256 = "6682b93da43c33b803711637fc104b977446cc757e9595db49a1184f11ef3ad3"
MOVE54_FLIPS = 12_127

#: the per-pair n600 pose vector measured on move 54's own decode with its own carrier
#: (pd7 ``POSE_ON_DECODE_set03c.npy``; mean 4.099227789357173e-06).
MOVE54_BASE_POSE = PD8_STORE / "base/pose_base_move54.npy"
MOVE54_BASE_POSE_SHA256 = "d46a971ef24ca41c5d88385a20b0882d14b5562f50726b1645f8d7d913509907"

#: move 54's cold decode.  pd7 measured it (1,213.9 s) and hashed it, then removed the payload
#: under its retention cap; this arm rebuilds the same bytes and re-verifies the sha.
MOVE54_RAW = PD8_STORE / "parseback_base/0.raw"
MOVE54_RAW_SHA256 = "ff43a9c97c72d0917ac4c2b856315648eddf3c0d3f69717eca132bfecf37a324"
MOVE54_RAW_BYTES = 3_662_409_600

#: move 54's DECODED token plane.  pd7's parseback reported it; this arm reproduces it
#: independently, because the pricer's ``field_to_u8`` of ``field_move54.npz`` hashes to the
#: same value -- so the field npz bound here IS the plane the receiver decodes.
MOVE54_DECODED_TOKEN_SHA256 = "ed69d961fe0b98c3ccad05c0f4eebcae8d39b438acf0b145a34016739668f7cf"

MOVE54_EDITED_PAIRS = PD8_STORE / "base/edited_pairs_move54.json"

CELL_COUNT = pd4.CELL_COUNT
MOVE54_INSTRUMENT_D_SEG = MOVE54_FLIPS / CELL_COUNT


class Pd8Error(RuntimeError):
    """A pd8 input or invariant is not what the measured object says it is."""


def move54_facts() -> dict[str, Any]:
    """Every declared fact about the live row, in one place, for the receipts."""
    return {
        "tree": str(MOVE54_TREE),
        "archive_sha256": MOVE54_ARCHIVE_SHA256,
        "archive_bytes": MOVE54_ARCHIVE_BYTES,
        "score_t4": MOVE54_SCORE_T4,
        "d_seg_t4": MOVE54_D_SEG_T4,
        "d_pose_t4": MOVE54_D_POSE_T4,
        "field": str(MOVE54_FIELD),
        "field_sha256": MOVE54_FIELD_SHA256,
        "argmax": str(MOVE54_ARGMAX),
        "argmax_sha256": MOVE54_ARGMAX_SHA256,
        "instrument_flips": MOVE54_FLIPS,
        "instrument_d_seg": MOVE54_INSTRUMENT_D_SEG,
        "raw_sha256": MOVE54_RAW_SHA256,
        "raw_bytes": MOVE54_RAW_BYTES,
        "base_pose": str(MOVE54_BASE_POSE),
        "base_pose_sha256": MOVE54_BASE_POSE_SHA256,
        "decoded_token_plane_sha256": MOVE54_DECODED_TOKEN_SHA256,
    }


def rebind_pd4_to_move54() -> dict[str, Any]:
    """Re-point pd4's move-52 constants at MOVE 54, process-locally.

    pd4 binds move 52 in module globals evaluated at import time and every downstream stage
    calls ``pd4.bind_move52``.  Rather than fork pd4 -- which would duplicate its verification
    path and let the two copies drift -- this function moves the constants and leaves pd4's
    checks in charge.  ``bind_move52`` then verifies EVERY one of them against the file on disk
    before binding anything, so a mistake here fails closed.
    """
    pd4.MOVE52_TREE = MOVE54_TREE
    pd4.MOVE52_ARCHIVE_SHA256 = MOVE54_ARCHIVE_SHA256
    pd4.MOVE52_ARCHIVE_BYTES = MOVE54_ARCHIVE_BYTES
    pd4.MOVE52_SCORE_T4 = MOVE54_SCORE_T4
    pd4.MOVE52_D_SEG_T4 = MOVE54_D_SEG_T4
    pd4.MOVE52_D_POSE_T4 = MOVE54_D_POSE_T4
    pd4.MOVE52_FIELD = MOVE54_FIELD
    pd4.MOVE52_FIELD_SHA256 = MOVE54_FIELD_SHA256
    pd4.MOVE52_ARGMAX = MOVE54_ARGMAX
    pd4.MOVE52_ARGMAX_SHA256 = MOVE54_ARGMAX_SHA256
    pd4.MOVE52_FLIPS = MOVE54_FLIPS
    pd4.MOVE52_INSTRUMENT_D_SEG = MOVE54_INSTRUMENT_D_SEG
    pd4.MOVE52_EDITED_PAIRS = MOVE54_EDITED_PAIRS
    pd4.MOVE52_DECODED_TOKEN_SHA256 = MOVE54_DECODED_TOKEN_SHA256
    pd4.PD3_RAW = MOVE54_RAW
    pd4.PD3_RAW_SHA256 = MOVE54_RAW_SHA256
    pd4.PD3_RAW_BYTES = MOVE54_RAW_BYTES
    return move54_facts()


def bound_seg_s_per_cell() -> float:
    """S carried by ONE flipped cell at MOVE 54's operating point, through its own ratio."""
    return 100.0 * (MOVE54_D_SEG_T4 / MOVE54_INSTRUMENT_D_SEG) / CELL_COUNT


def cmd_bind(args) -> int:
    """The binding receipt: rebind, let pd4 verify, then re-derive the pointer arithmetic."""
    facts = rebind_pd4_to_move54()
    receipts = pd4.bind_move52(verify_raw=bool(args.verify_raw), raw=Path(args.raw))
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    live = pointer["our_local_frontier_contest_cuda"]
    if live["archive_sha256"] != MOVE54_ARCHIVE_SHA256:
        raise Pd8Error(
            f"POINTER MOVED: the live row is {live['archive_sha256']}, not move 54's "
            f"{MOVE54_ARCHIVE_SHA256}; re-derive the base before pricing anything"
        )
    recomputed = pd4.composed_score(MOVE54_D_SEG_T4, MOVE54_D_POSE_T4, MOVE54_ARCHIVE_BYTES)
    if abs(recomputed - MOVE54_SCORE_T4) > 1e-15:
        raise Pd8Error(f"move 54 recomputes to {recomputed!r}, not {MOVE54_SCORE_T4!r}")
    if abs(pd4.seg_s_per_cell() - bound_seg_s_per_cell()) > 1e-18:
        raise Pd8Error("pd4's seg cell price did not re-bind to move 54")
    base_mean = float(np.load(MOVE54_BASE_POSE).mean())
    payload = {
        "schema": "ddm_pd8_bind.v1",
        "axis": "[macOS-CPU advisory / binding receipt, no score]",
        "score_claim": False,
        "move54": facts,
        "pd4_receipts": receipts,
        "live_pointer": {
            "archive_sha256": live["archive_sha256"],
            "score": live.get("score"),
            "lane_id": live.get("lane_id"),
            "re_derived_from": str(REPO / ".omx/state/canonical_frontier_pointer.json"),
        },
        "score_recomputed_from_components": recomputed,
        "seg_s_per_cell_move54": bound_seg_s_per_cell(),
        "base_pose_mean_n600": base_mean,
        "pose_s_per_pair_unit_move54": pd4.pose_s_per_pair_unit(base_mean),
        "s_per_byte": pd4.S_PER_BYTE,
        "floor_pairs_excluded": sorted(pd4.FLOOR_PAIRS),
        "rebinding_note": (
            "pd4's move-52 constants were re-pointed at move 54 PROCESS-LOCALLY by "
            "ddm_pd8_price_first_pass3.rebind_pd4_to_move54; pd4.bind_move52 then verified "
            "every one of them against the file on disk and re-asserted the binding THROUGH "
            "pp1/sj1/pd1/pd2/pd3.  No pd4, pd6, pd7 or shipped-runtime source was edited."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_headroom(args) -> int:
    """What credit the lever needs at MOVE 54's operating point -- DERIVED before any credit."""
    rebind_pd4_to_move54()
    base = np.load(MOVE54_BASE_POSE)
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
        "schema": "ddm_pd8_headroom.v1",
        "axis": "[DERIVED at move 54's operating point; EXPIRES at the next pointer move]",
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
    field it was measured on: move 54's base pose vector came from the candidate's own cold
    decode at batch 8, and the wave reads it at batch 1.  No prefix -- all 600 pairs.
    """
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_pp1_pose_actuation as pp1

    if float(args.headroom) < 1.0:
        raise Pd8Error(
            f"headroom {args.headroom} < 1 would set a band TIGHTER than the measured max gap, "
            "and realize raises on the first pair outside it -- the whole shard would die"
        )
    rebind_pd4_to_move54()
    pd4.bind_move52(verify_raw=False, raw=Path(args.raw))
    pp1.set_threads(int(args.threads))
    base = np.load(MOVE54_BASE_POSE)
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
        "schema": "ddm_pd8_baseband.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT lineage; batch 1 vs batch 8]",
        "score_claim": False,
        "pairs": int(pd4.N_PAIRS),
        "n600_base_path": str(MOVE54_BASE_POSE),
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
    """pd4's merge, reading pd8's OWN sheet encodes.

    pd4's merge builds the field name ``pd4sheet{k:02d}``; pd6 rebound that to ``pd6sheet``,
    pd7 to ``pd7sheet``, and this arm to ``pd8sheet``.  The rebinding is PROCESS-LOCAL and is
    restored in ``finally``; a ``pd4sheet`` name here would be a provenance lie, because the
    bits come from this arm's own encodes of its own sheets.
    """
    original = pd4._encode_bits

    def rebound(rlc1_root, field: str, tag: str):
        name = field.replace("pd4sheet", "pd8sheet") if field.startswith("pd4sheet") else field
        return original(rlc1_root, name, tag)

    pd4._encode_bits = rebound
    try:
        return pd4.cmd_price_merge(args)
    finally:
        pd4._encode_bits = original


def cmd_run(args) -> int:
    """pd4's ``run`` passthrough, bound to MOVE 54 -- the joint admission's own argv."""
    rebind_pd4_to_move54()
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
        raise Pd8Error(f"unknown module {args.module}")
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise Pd8Error("run needs the target's own argv after --")
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
    rn = sub.add_parser("run", help="run another module's argv under the MOVE 54 binding")
    rn.add_argument("--module", required=True,
                    choices=["joint", "pp1", "tree", "pd4", "pd5"])
    rn.add_argument("--raw", required=True)
    rn.add_argument("argv", nargs=argparse.REMAINDER)
    rn.set_defaults(func=cmd_run)
    # pd6's price-merge rebinds pd4's sheet name to pd6's; this arm's sheets are pd8's
    sub._name_parser_map["price-merge"].set_defaults(func=cmd_price_merge)
    bind = sub.add_parser("bind54", help="bind MOVE 54 and re-derive its arithmetic")
    bind.add_argument("--raw", required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.add_argument("--out", required=True)
    bind.set_defaults(func=cmd_bind)
    hr = sub.add_parser("headroom", help="what credit the lever needs at move 54")
    hr.add_argument("--out", required=True)
    hr.set_defaults(func=cmd_headroom)

    args = parser.parse_args(argv)
    if args.func is not cmd_bind:
        # every pd6 stage reads pd4's constants, directly or through bind_move52
        rebind_pd4_to_move54()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
