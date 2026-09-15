#!/usr/bin/env python3
"""ddm_pd4: POSE-DIRECTED token pre-distortion, PASS 4, on the MOVE-52 field -- the PRICE lever.

pd1 (``experiments/ddm_pd1_pose_directed.py``) is the reference form, pd2 and pd3 are the
pass-2/pass-3 bindings, and sj1's ``ddm_sj1_rlc1_price`` is the real-encode pricer.  All four
are REUSED here rather than re-implemented.  This module supplies only what pass 4 adds:

(a) the MOVE-52 binding every pd1/sj1/pd2/pd3 stage needs (those modules pin moves 49/50/51
    in globals evaluated at import time, so every one of them is STALE for this arm);
(b) move 52's own pose base, measured on THIS arm's own cold parse-back of the shipped bytes;
(c) the economics at move 52's operating point, re-derived because they EXPIRE at every
    pointer move ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]);
(d) **the price lever** -- the thing the charter is about:

    1. ``sheets``      turn ranked proposals into PRICE SHEETS: at most one proposal per pair
                       per sheet, so ONE real known-symbol encode prices every proposal in the
                       sheet at once, frame-locally.
    2. ``price-merge`` read each sheet's ``per_frame_bits`` back and attach the MEASURED real
                       marginal bits to each proposal -- per PROPOSAL, which pd3's per-PAIR
                       rlc1 ledger structurally could not do (pd3 memo Sect. 9.8).
    3. ``carry``       rank by resolved-pose credit per REAL bit and keep the best proposal
                       per pair under THAT ranking, not under best-credit.
(e) ``cluster-search`` -- the clustered proposal family: a SECOND token move in the
    8-neighbourhood of the pair's best single-token move, realized jointly (one render, one
    argmax, one carrier re-solve), never summed from two rows.

WHY FRAME-LOCAL PRICING IS A REAL PRICE, AND WHAT IT IS NOT
-----------------------------------------------------------
``per_frame_bits[f] = sum over the frame's coded positions of -log2 p_model(symbol)`` under
the SHIPPED receiver's own probability rows, driven through ``rx.decode_production_tokens``
with the true symbols injected.  It is not a first-order token price
([[first_order_token_price_is_a_ranking_never_a_charge...]]): it carries the adaptive model's
full response WITHIN the frame.  What it does not carry is the spill into later frames.
MEASURED on pd3's own full field (271 pairs / 271 tokens, its retained
``bits_rlc1_control.npy`` and ``bits_rlc1_pd3full.npy``): the signed total delta is 4,449.05
bits, of which 4,324.86 (97.21 %) lands on the edited frames and 124.19 (2.79 %) on unedited
ones; realized bytes / ideal bytes = 1.0016.  So a frame-local price is a 97.2 %-complete
REAL price on that field, and the residue is measured rather than assumed.  The CHARGE is
still the admitted subset's own full real encode with twins; this rail is the RANKING.

No scorer weights, no dispatch, no pointer write, no edit to any sister arm's tree.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_pd3_pose_directed_pass3 as pd3  # the path above must be set first

pd2 = pd3.pd2

N_PAIRS = pd3.N_PAIRS
EVAL_H, EVAL_W = pd3.EVAL_H, pd3.EVAL_W
CELL_COUNT = pd3.CELL_COUNT
S_PER_BYTE = pd3.S_PER_BYTE
FLOOR_PAIRS = pd3.FLOOR_PAIRS
BASE_BAND_ABS = pd3.BASE_BAND_ABS

#: MOVE 52 -- the live row.  Lane ddm_pd3_pose_directed_pass3_on_move51_20260913, Modal call
#: fc-01M2D1J97K1QPBQ4KASFH1SG96, packet
#: `.omx/research/ddm_pd3_..._pointer_move_52_20260913.md`.  Every path below is another arm's
#: custody and is opened READ-ONLY.
MOVE52_TREE = Path("/Volumes/APDataStore/pact/ddm_pd3/candidate/candidate_runtime")
MOVE52_ARCHIVE_SHA256 = "ae59c5109597968409c4b581337b8e23b5b3c4262fd3710c315cc83913b3b20e"
MOVE52_ARCHIVE_BYTES = 179_332
MOVE52_SCORE_T4 = 0.13620226906030858
MOVE52_D_SEG_T4 = 0.00010304
MOVE52_D_POSE_T4 = 4.21e-06
#: the token field move 52's own parse-back decodes to (pd3: decoded_field_matches_admitted).
MOVE52_FIELD = Path("/Volumes/APDataStore/pact/ddm_pd3/admission/field_admitted.npz")
MOVE52_FIELD_SHA256 = "0577bbb88e7218c70ae00876d5992de0859353c080776d4952b6cd3f2566e8d8"
#: the u8 token plane sha the receiver itself reports for that decode.
MOVE52_DECODED_TOKEN_SHA256 = "117951bbd6800949556b185566f8e9efbd83a7cc00ae542cc4febff842f12e11"
#: pd3's own cold parse-back of move 52's shipped bytes, and the n600 argmax of it.
PD3_RAW = Path("/Volumes/APDataStore/pact/ddm_pd3/parseback/0.raw")
PD3_RAW_SHA256 = "ccb89e3e73bf61ac334f50d6963e51e291d446d798c9a5bac5992da0beceeced"
PD3_RAW_BYTES = 3_662_409_600
MOVE52_ARGMAX = Path("/Volumes/APDataStore/pact/ddm_pd3/seg_final/argmax_n600.npy")
MOVE52_ARGMAX_SHA256 = "622b284eada5023ded0af3b7353b34d22bf494eabb86cce6f92006ab297dd79a"
#: move 52's instrument seg leg on that decode, DALI lineage.
MOVE52_FLIPS = 12_147
MOVE52_INSTRUMENT_D_SEG = 12_147 / CELL_COUNT
#: the twenty-six pairs move 52 itself edited; their renders MOVED, so pass 4 re-searches
#: them from the NEW render rather than reusing a pd3 row measured on the move-51 render.
MOVE52_EDITED_PAIRS = Path("/Volumes/APDataStore/pact/ddm_pd3/admission/kept_pairs.json")

#: this arm's OWN parse-back of the same archive bytes (written by the parseback stage), used
#: as the live raw once it has reproduced ``PD3_RAW_SHA256``.  Set by ``--raw``.
PD4_STORE = Path("/Volumes/APDataStore/pact/ddm_pd4")

#: MEASURED real-encode prices for this edit shape on this coder, each on ONE field, none a
#: law: pd1 16.98 (41-token field), pd2 16.630 (165-token field) / 15.400 (40-token subset),
#: pd3 16.443 (271-token field) / 12.923 (26-token subset).  Pass 8 measured 8.93 bits/token
#: on a CLUSTERED seg-repair field (163 tokens over 96 pairs).  This arm SEARCHES at pd3's
#: shipped-subset price and re-prices every proposal by its own real encode.
PD3_MEASURED_BITS_PER_TOKEN_FULL_FIELD = 16.443
PD3_MEASURED_BITS_PER_TOKEN_SUBSET = 12.923
PASS8_MEASURED_BITS_PER_TOKEN_CLUSTERED = 8.93

#: pd3's MEASURED per-stratum admitted fraction at move 51 (its own search, K_refine 12):
#: U1 3/8, R 19/163, U2 4/100.  PRIORS for the pass-4 ranking, measured on ONE field.
PD3_ADMITTED_FRACTION_TIER1_UNWALKED = 3.0 / 8.0
PD3_ADMITTED_FRACTION_CREDITED = 19.0 / 163.0
PD3_ADMITTED_FRACTION_TIER2 = 4.0 / 100.0

PD3_STORE = Path("/Volumes/APDataStore/pact/ddm_pd3")


class Pd4Error(RuntimeError):
    """A pd4 input or invariant is not what the measured object says it is."""


sha256_file = pd3.sha256_file
composed_score = pd2.composed_score
read_rows = pd2.read_rows
credit_per_bit = pd2.credit_per_bit


def seg_s_per_cell() -> float:
    """S carried by ONE flipped cell on the T4 axis, through move 52's instrument ratio."""
    ratio = MOVE52_D_SEG_T4 / MOVE52_INSTRUMENT_D_SEG
    return 100.0 * ratio / CELL_COUNT


def pose_s_per_pair_unit(base_mean: float) -> float:
    """dS / d(one pair's d_pose) at this base mean.  EXPIRES at every pointer move."""
    return math.sqrt(10.0 / base_mean) / (2.0 * N_PAIRS)


def edited_pairs_at_move52() -> list[int]:
    return sorted(int(p) for p in json.loads(MOVE52_EDITED_PAIRS.read_text()))


# ----------------------------------------------------------------------------------
# THE BINDING
# ----------------------------------------------------------------------------------


def bind_move52(*, verify_raw: bool = False, raw: Path | None = None) -> dict[str, Any]:
    """Re-point pp1, sj1, pd1, pd2 and pd3 at MOVE 52, process-locally, fail closed on drift.

    Five shared modules bind a pointer in globals evaluated at import time: pp1 and sj1 at
    move 49, pd2 at move 50, pd1 at move 49 and pd3 at move 51.  Re-binding is mandatory and
    is exactly the silent-rebinding hazard that hands a successor a wrong number, so every
    object is checked against its declared sha BEFORE it is bound and the pointer's own score
    arithmetic is re-derived from its three components afterwards.
    """
    import ddm_pd1_pose_directed as pd1
    import ddm_pp1_pose_actuation as pp1
    import ddm_sj1_multipass_token_predistortion as sj1

    live_raw = Path(raw) if raw is not None else PD3_RAW

    archive = MOVE52_TREE / "archive.zip"
    observed_bytes = archive.stat().st_size
    if observed_bytes != MOVE52_ARCHIVE_BYTES:
        raise Pd4Error(f"move-52 archive is {observed_bytes} B, not {MOVE52_ARCHIVE_BYTES}")
    observed_sha = sha256_file(archive)
    if observed_sha != MOVE52_ARCHIVE_SHA256:
        raise Pd4Error(f"move-52 archive sha is {observed_sha}, not {MOVE52_ARCHIVE_SHA256}")
    argmax_sha = sha256_file(MOVE52_ARGMAX)
    if argmax_sha != MOVE52_ARGMAX_SHA256:
        raise Pd4Error(f"move-52 argmax sha is {argmax_sha}, not {MOVE52_ARGMAX_SHA256}")
    field_sha = sha256_file(MOVE52_FIELD)
    if field_sha != MOVE52_FIELD_SHA256:
        raise Pd4Error(f"move-52 field sha is {field_sha}, not {MOVE52_FIELD_SHA256}")
    raw_bytes = live_raw.stat().st_size
    if raw_bytes != PD3_RAW_BYTES:
        raise Pd4Error(f"the bound decode is {raw_bytes} B, not {PD3_RAW_BYTES}")
    raw_sha = None
    if verify_raw:
        raw_sha = sha256_file(live_raw)
        if raw_sha != PD3_RAW_SHA256:
            raise Pd4Error(
                f"the bound decode sha is {raw_sha}, not move 52's {PD3_RAW_SHA256}"
            )

    recomputed = composed_score(MOVE52_D_SEG_T4, MOVE52_D_POSE_T4, MOVE52_ARCHIVE_BYTES)
    if abs(recomputed - MOVE52_SCORE_T4) > 1e-15:
        raise Pd4Error(
            f"move 52's three components recompute to {recomputed!r}, not the packet's "
            f"{MOVE52_SCORE_T4!r}"
        )

    row = sj1.PointerRow(
        label="pd3_pose_directed_pass3_move52",
        tree=MOVE52_TREE,
        archive_sha256=MOVE52_ARCHIVE_SHA256,
        archive_bytes=MOVE52_ARCHIVE_BYTES,
        d_seg_t4=MOVE52_D_SEG_T4,
        d_pose_t4=MOVE52_D_POSE_T4,
        score_t4=MOVE52_SCORE_T4,
    )
    row.verify_arithmetic()

    for module in (sj1, pp1):
        module.POINTER_TREE = MOVE52_TREE
        module.POINTER_ARCHIVE = archive
        module.POINTER_ARCHIVE_SHA256 = MOVE52_ARCHIVE_SHA256
        module.POINTER_ARCHIVE_BYTES = MOVE52_ARCHIVE_BYTES
        module.POINTER_SCORE_T4 = MOVE52_SCORE_T4
        module.POINTER_D_SEG_T4 = MOVE52_D_SEG_T4
        module.POINTER_D_POSE_T4 = MOVE52_D_POSE_T4
    sj1.LIVE_POINTER = row
    pp1.POINTER = row
    pp1.LIVE_RAW = live_raw
    pp1.LIVE_FIELD = MOVE52_FIELD
    pp1.LIVE_ARGMAX = MOVE52_ARGMAX

    pd2.MOVE50_TREE = MOVE52_TREE
    pd2.MOVE50_ARCHIVE_SHA256 = MOVE52_ARCHIVE_SHA256
    pd2.MOVE50_ARCHIVE_BYTES = MOVE52_ARCHIVE_BYTES
    pd2.MOVE50_SCORE_T4 = MOVE52_SCORE_T4
    pd2.MOVE50_D_SEG_T4 = MOVE52_D_SEG_T4
    pd2.MOVE50_D_POSE_T4 = MOVE52_D_POSE_T4
    pd2.MOVE50_RAW = live_raw
    pd2.MOVE50_RAW_SHA256 = PD3_RAW_SHA256
    pd2.MOVE50_FIELD = MOVE52_FIELD
    pd2.MOVE50_ARGMAX = MOVE52_ARGMAX
    pd2.MOVE50_ARGMAX_SHA256 = MOVE52_ARGMAX_SHA256
    pd2.MOVE50_FLIPS = MOVE52_FLIPS
    pd2.MOVE50_INSTRUMENT_D_SEG = MOVE52_INSTRUMENT_D_SEG
    pd2.MOVE50_EDITED_PAIRS = str(MOVE52_EDITED_PAIRS)

    pd3.MOVE51_TREE = MOVE52_TREE
    pd3.MOVE51_ARCHIVE_SHA256 = MOVE52_ARCHIVE_SHA256
    pd3.MOVE51_ARCHIVE_BYTES = MOVE52_ARCHIVE_BYTES
    pd3.MOVE51_SCORE_T4 = MOVE52_SCORE_T4
    pd3.MOVE51_D_SEG_T4 = MOVE52_D_SEG_T4
    pd3.MOVE51_D_POSE_T4 = MOVE52_D_POSE_T4
    pd3.MOVE51_RAW = live_raw
    pd3.MOVE51_RAW_SHA256 = PD3_RAW_SHA256
    pd3.MOVE51_FIELD = MOVE52_FIELD
    pd3.MOVE51_FIELD_SHA256 = MOVE52_FIELD_SHA256
    pd3.MOVE51_ARGMAX = MOVE52_ARGMAX
    pd3.MOVE51_ARGMAX_SHA256 = MOVE52_ARGMAX_SHA256
    pd3.MOVE51_FLIPS = MOVE52_FLIPS
    pd3.MOVE51_INSTRUMENT_D_SEG = MOVE52_INSTRUMENT_D_SEG
    pd3.MOVE51_EDITED_PAIRS = MOVE52_EDITED_PAIRS

    pd1.POINTER_ARCHIVE_SHA256 = MOVE52_ARCHIVE_SHA256
    pd1.POINTER_ARCHIVE_BYTES = MOVE52_ARCHIVE_BYTES
    pd1.POINTER_SCORE_T4 = MOVE52_SCORE_T4
    pd1.POINTER_D_SEG_T4 = MOVE52_D_SEG_T4
    pd1.POINTER_D_POSE_T4 = MOVE52_D_POSE_T4
    pd1.INSTRUMENT_BASE_D_SEG = MOVE52_INSTRUMENT_D_SEG

    # Re-assert THROUGH the modules that will be used, never through this one's copy of the
    # facts: if any global above were missed, this call still reports its predecessor.
    receipts = pp1.assert_pointer_identity()
    for name, expected in (
        ("pointer_archive_sha256", MOVE52_ARCHIVE_SHA256),
        ("live_raw", str(live_raw)),
        ("live_field", str(MOVE52_FIELD)),
        ("live_argmax", str(MOVE52_ARGMAX)),
    ):
        if str(receipts[name]) != str(expected):
            raise Pd4Error(f"after binding, pp1 still reports {name}={receipts[name]!r}")
    if sj1.assert_carrier_is_pointer(MOVE52_TREE) != MOVE52_ARCHIVE_SHA256:
        raise Pd4Error("sj1's carrier anchor did not re-bind to move 52")
    for label, module in (("pd1", pd1), ("pd2", pd2), ("pd3", pd3)):
        if module.seg_s_per_cell() != seg_s_per_cell():
            raise Pd4Error(f"{label}'s seg cell price did not re-bind to move 52")
    receipts.update({
        "bound_move": 52,
        "argmax_sha256": argmax_sha,
        "field_sha256": field_sha,
        "raw_path": str(live_raw),
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha,
        "score_recomputed_from_components": recomputed,
        "instrument_d_seg_move52": MOVE52_INSTRUMENT_D_SEG,
        "instrument_flips_move52": MOVE52_FLIPS,
        "edited_pairs_at_move52": edited_pairs_at_move52(),
        "seg_s_per_cell": seg_s_per_cell(),
    })
    return receipts


def cmd_bind(args) -> int:
    receipts = bind_move52(verify_raw=bool(args.verify_raw), raw=args.raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ddm_pd4_bind.v1",
        "axis": "[macOS-CPU advisory / identity only]",
        "score_claim": False,
        "receipts": receipts,
    }, indent=1, sort_keys=True))
    print(json.dumps(receipts, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: base -- move 52's own per-pair d_pose, n600, on THIS arm's own parse-back
# ----------------------------------------------------------------------------------


def cmd_base(args) -> int:
    receipts = bind_move52(verify_raw=bool(args.verify_raw), raw=args.raw)
    import ddm_pp1_pose_actuation as pp1
    import ddm_up2_shipping_pose_solve as up2

    pp1.set_threads(args.threads)
    started = time.time()
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Pd4Error(f"move-52 carrier codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, _poses = up2.measure_pose(
        inst.posenet, inst.state, coefficients, inst.raw, inst.targets, indices,
        batch_size=args.batch_size,
    )
    mean = float(per_pair.mean())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = out_dir / "pose_base_move52.npy"
    np.save(per_pair_path, per_pair)
    np.save(out_dir / "codes_move52.npy", codes)
    control = {}
    if args.control is not None:
        other = np.load(args.control)
        if other.shape == per_pair.shape:
            control = {
                "control_path": str(args.control),
                "max_abs_difference": float(np.abs(other - per_pair).max()),
                "control_mean": float(other.mean()),
            }
    order = np.argsort(-per_pair)
    report = {
        "schema": "ddm_pd4_pose_base.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "solver": (
            f"up2.measure_pose at batch {args.batch_size} on this arm's own cold parse-back "
            "of move 52's shipped bytes"
        ),
        "receipts": receipts,
        "pairs": N_PAIRS,
        "d_pose_mean": mean,
        "pose_leg": math.sqrt(10.0 * mean),
        "t4_print": MOVE52_D_POSE_T4,
        "instrument_ratio_vs_t4_print": mean / MOVE52_D_POSE_T4,
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "top12_share": float(per_pair[order[:12]].sum() / per_pair.sum()),
        "top12_pairs": [int(p) for p in order[:12]],
        "floor_pairs": sorted(int(p) for p in FLOOR_PAIRS),
        "pose_s_per_pair_unit": pose_s_per_pair_unit(mean),
        "per_pair_path": str(per_pair_path),
        "control": control,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "POSE_BASE_MOVE52.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "receipts"}, indent=1,
                     sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: run -- bind, then dispatch another module's argv inside this process
# ----------------------------------------------------------------------------------


def cmd_run(args) -> int:
    bind_move52(verify_raw=False, raw=args.raw)
    if args.module == "pd1":
        import ddm_pd1_pose_directed as target
    elif args.module == "pd2":
        target = pd2
    elif args.module == "pd3":
        target = pd3
    elif args.module == "joint":
        import ddm_sj1_joint_admission as target
    elif args.module == "pp1":
        import ddm_pp1_pose_actuation as target
    elif args.module == "tree":
        import ddm_pd1_candidate_tree as target
    else:  # pragma: no cover - argparse restricts the choices
        raise Pd4Error(f"unknown module {args.module}")
    argv = list(args.argv)
    if argv and argv[0] == "--":  # argparse.REMAINDER keeps the separator
        argv = argv[1:]
    if not argv:
        raise Pd4Error("run needs the target's own argv after --")
    return int(target.main(argv))


# ----------------------------------------------------------------------------------
# the proposal pool: every REALIZED row pd1/pd2/pd3 left behind, gated onto move 52
# ----------------------------------------------------------------------------------


def proposal_key(row: dict[str, Any]) -> tuple:
    """Identity of a proposal: the pair and the SET of (cell, new) moves it makes."""
    edits = row_edits(row)
    return (int(row["pair"]), tuple(sorted((r, c, n) for (r, c, _o, n) in edits)))


def row_edits(row: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """(row, col, old, new) for every token this row moves -- 1 for a single, 2 for a pair."""
    if "edits" in row:
        return [(int(e[0]), int(e[1]), int(e[2]), int(e[3])) for e in row["edits"]]
    cell = row["cell"]
    return [(int(cell[0]), int(cell[1]), int(row["old"]), int(row["new"]))]


def load_pool(
    row_paths: list[Path],
    base: np.ndarray,
    *,
    band_abs: float,
    exclude_pairs: set[int],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    """Every refined row whose pair-base still IS move 52's, keyed by pair, deduplicated.

    TWO gates, not one.  pd3 relied on the base-band gate ALONE to keep a predecessor's rows
    off the pairs the new pointer re-rendered ("enforced structurally rather than by hand").
    MEASURED here on move 52: of the 26 pairs move 52 itself edited, **25 are refused by the
    band but pair 569 is not** -- its predecessor rows sit 9.311e-10 from the new base, well
    inside the 2.264e-08 band, because that pair's own d_pose barely moved.  So the band is a
    96 %-effective proxy for "this render did not move", not a proof of it.  The explicit
    exclusion below is the cure; the band stays as the second gate.
    """
    stats = {"rows_read": 0, "unrefined": 0, "excluded_stale_row_on_edited_pair": 0,
             "outside_band": 0, "floor_pair": 0, "duplicates": 0, "fresh_rows": 0}
    # Read the files here rather than through ``read_rows`` so every row carries the STORE
    # it came from.  That distinction is load-bearing: the 26 pairs move 52 itself edited
    # were RE-SEARCHED by this arm from their new renders, so this arm's own rows on those
    # pairs are current and must be kept, while a predecessor's rows on the same pairs were
    # measured on a render that no longer exists and must go -- and the base band catches
    # only 25 of the 26 (pair 569 slips through at 9.311e-10).
    rows: list[dict[str, Any]] = []
    for path in row_paths:
        store = Path(path).resolve()
        fresh = "ddm_pd4" in store.parts
        source = next((part for part in store.parts if part.startswith("ddm_pd")), "unknown")
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("row_source", source)
            row["row_is_this_arms_own"] = fresh
            rows.append(row)
    stats["rows_read"] = len(rows)
    pool: dict[int, dict[tuple, dict[str, Any]]] = {}
    for row in rows:
        pair = int(row["pair"])
        if "credit_d_pose" not in row or "d_pose_base" not in row:
            stats["unrefined"] += 1
            continue
        if pair in FLOOR_PAIRS:
            stats["floor_pair"] += 1
            continue
        if pair in exclude_pairs and not row["row_is_this_arms_own"]:
            stats["excluded_stale_row_on_edited_pair"] += 1
            continue
        if row["row_is_this_arms_own"]:
            stats["fresh_rows"] += 1
        gap = abs(float(row["d_pose_base"]) - float(base[pair]))
        if gap > band_abs:
            stats["outside_band"] += 1
            continue
        entry = dict(row)
        entry["base_gap_abs_vs_n600"] = gap
        entry.setdefault("tokens_changed", len(row_edits(row)))
        key = proposal_key(entry)
        slot = pool.setdefault(pair, {})
        prior = slot.get(key)
        if prior is None:
            slot[key] = entry
        else:
            stats["duplicates"] += 1
            # Credit is resolved MINUS base, so a GAIN is negative: keep the strongest
            # (most negative) row, which is the deepest refine of that same proposal.
            if float(entry["credit_d_pose"]) < float(prior["credit_d_pose"]):
                slot[key] = entry
    return {p: list(v.values()) for p, v in pool.items()}, stats


def score_proposals(
    pool: dict[int, list[dict[str, Any]]], *, pose_unit: float, seg_cell: float
) -> None:
    """Attach the pose+seg benefit in S units.  Rate is NOT here: that is the price lever."""
    for entries in pool.values():
        for entry in entries:
            entry["benefit_S"] = (
                -(float(entry["credit_d_pose"]) * pose_unit)
                - float(entry["d_cells"]) * seg_cell
            )
            entry["tokens_changed"] = len(row_edits(entry))


# ----------------------------------------------------------------------------------
# stage: prereg -- the economics at move 52, the strata, the falsifiers
# ----------------------------------------------------------------------------------


def cmd_prereg(args) -> int:
    receipts = bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    if base.shape != (N_PAIRS,):
        raise Pd4Error(f"base pose vector has shape {base.shape}")
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    excluded = set(edited_pairs_at_move52())
    pool, stats = load_pool([Path(p) for p in args.rows], base,
                            band_abs=BASE_BAND_ABS, exclude_pairs=excluded)
    score_proposals(pool, pose_unit=pose_unit, seg_cell=seg_cell)

    paying = {p: [e for e in v if e["benefit_S"] > 0.0] for p, v in pool.items()}
    paying = {p: v for p, v in paying.items() if v}
    ladder = []
    for bits in (PD3_MEASURED_BITS_PER_TOKEN_FULL_FIELD, PD3_MEASURED_BITS_PER_TOKEN_SUBSET,
                 10.0, PASS8_MEASURED_BITS_PER_TOKEN_CLUSTERED, 7.0):
        fee = (bits / 8.0) * S_PER_BYTE
        pairs = [p for p, v in paying.items() if max(e["benefit_S"] for e in v) > fee]
        net = -sum(max(e["benefit_S"] for e in paying[p]) - fee for p in pairs)
        ladder.append({
            "bits_per_token": bits,
            "pairs_able_to_pay_with_best_benefit_proposal": len(pairs),
            "modelled_net_dS": net,
            "bars_of_the_2e_05_admit_bar": -net / 2e-05,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ddm_pd4_prereg.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "written_before": "any pass-4 search row, price sheet or admission existed",
        "receipts": receipts,
        "economics_at_move52_they_expire_at_every_pointer_move": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_flipped_seg_cell_T4_carried": seg_cell,
            "S_per_unit_of_one_pairs_d_pose": pose_unit,
            "base_pose_mean": base_mean,
            "base_pose_path": str(args.base_pose),
        },
        "measured_prices_for_this_edit_shape_none_a_law": {
            "pd1_41_token_field": 16.98,
            "pd2_165_token_field": 16.630,
            "pd2_40_token_subset": 15.400,
            "pd3_271_token_field": PD3_MEASURED_BITS_PER_TOKEN_FULL_FIELD,
            "pd3_26_token_subset": PD3_MEASURED_BITS_PER_TOKEN_SUBSET,
            "pass8_163_token_clustered_seg_repair_field": PASS8_MEASURED_BITS_PER_TOKEN_CLUSTERED,
        },
        "carried_pool": {
            "row_sources": [str(p) for p in args.rows],
            "gates": stats,
            "pairs_with_any_in_band_row": len(pool),
            "proposals_in_band": int(sum(len(v) for v in pool.values())),
            "pairs_with_a_paying_proposal": len(paying),
            "paying_proposals": int(sum(len(v) for v in paying.values())),
            "excluded_move52_edited_pairs": sorted(excluded),
            "floor_pairs_excluded": sorted(int(p) for p in FLOOR_PAIRS),
        },
        "price_ladder_on_the_carried_pool_only": ladder,
        "pre_registered_band_from_the_charter": [-2e-05, -6e-05],
        "falsifiers": [
            {"id": "F1", "claim": "the rlc1 pricer reproduces move 52's OWN archive",
             "fires_if": "the control encode's packed archive sha is not ae59c510…"},
            {"id": "F2", "claim": "the frame-local price is a real price",
             "fires_if": "on the final full field, the share of the signed delta bits landing "
                         "on EDITED frames is below 90 % (pd3's own field measured 97.21 %)"},
            {"id": "F3", "claim": "the per-proposal price is resolvable above its noise floor",
             "fires_if": "the spread of a HELD-FIXED pair's delta bits across sheets exceeds "
                         "the median |difference| between the best and second-best proposal "
                         "on a pair"},
            {"id": "F4", "claim": "the per-bit ranking changes the admitted set",
             "fires_if": "the per-bit winner equals the per-benefit winner on every pair"},
            {"id": "F5", "claim": "clustered edits are cheaper per token than isolated ones",
             "fires_if": "the measured bits/token of the clustered proposals is not below "
                         "the isolated proposals' on the same pairs"},
            {"id": "F6", "claim": "composition is realized, not summed",
             "fires_if": "the composed re-verified pose credit is under 0.99 of the per-pair sum"},
            {"id": "F7", "claim": "the admitted subset beats pd3's 12.923 bits/token",
             "fires_if": "the subset's own real-encode bits/token is >= 12.923 -- the charter's "
                         "price-lever falsifier"},
            {"id": "F8", "claim": "the admitted set clears the bar",
             "fires_if": "net dS on the resolved pose is above -2e-05"},
            {"id": "F9", "claim": "the seg leg on the shipped bytes equals the admission's",
             "fires_if": "the candidate's own cold parse-back argmax disagrees on any cell"},
            {"id": "F10", "claim": "the base is move 52's own decode",
             "fires_if": "this arm's parse-back raw sha is not ccb89e3e…"},
        ],
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("receipts", "falsifiers")}, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: plan -- the anchors the cluster family hangs on, and the shard assignment
# ----------------------------------------------------------------------------------


def cmd_plan(args) -> int:
    """Rank pairs by their best paying proposal; emit the anchors and a round-robin plan.

    Round-robin over shards (rather than contiguous blocks) so that a prefix stop leaves a
    near-uniform prefix of the GLOBAL ranking rather than the top of one shard -- pd3's rule,
    kept.
    """
    bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    excluded = set(edited_pairs_at_move52())
    pool, stats = load_pool([Path(p) for p in args.rows], base,
                            band_abs=BASE_BAND_ABS, exclude_pairs=excluded)
    score_proposals(pool, pose_unit=pose_unit, seg_cell=seg_cell)

    anchors: dict[str, Any] = {}
    ranked: list[tuple[float, int]] = []
    for pair, entries in pool.items():
        singles = [e for e in entries
                   if len(row_edits(e)) == 1 and e["benefit_S"] > 0.0]
        if not singles:
            continue
        best = max(singles, key=lambda e: e["benefit_S"])
        r, c, old, new = row_edits(best)[0]
        anchors[str(pair)] = {
            "cell": [r, c], "old": old, "new": new,
            "benefit_S": best["benefit_S"],
            "credit_d_pose": float(best["credit_d_pose"]),
            "d_cells": int(best["d_cells"]),
            "break_even_bits": best["benefit_S"] / S_PER_BYTE * 8.0,
            "row_source": best.get("row_source", "unknown"),
        }
        ranked.append((best["benefit_S"], pair))
    ranked.sort(reverse=True)
    order = [p for _b, p in ranked][: args.max_pairs] if args.max_pairs else [p for _b, p in ranked]
    shards: list[list[int]] = [[] for _ in range(args.shard_count)]
    for index, pair in enumerate(order):
        shards[index % args.shard_count].append(pair)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anchors_path = out.with_name(out.stem + "_anchors.json")
    anchors_path.write_text(json.dumps(anchors, indent=1, sort_keys=True))
    report = {
        "schema": "ddm_pd4_plan.v1",
        "axis": "[macOS-CPU advisory / planning only]",
        "score_claim": False,
        "gates": stats,
        "pairs_with_a_paying_single_token_anchor": len(anchors),
        "cluster_walk_order": order,
        "cluster_walk_pairs": len(order),
        "shards": {str(i): s for i, s in enumerate(shards)},
        "shard_count": args.shard_count,
        "anchors_path": str(anchors_path),
        "rule": "pairs ranked by their best PAYING single-token proposal's benefit; "
                "round-robin over shards so a prefix stop keeps a uniform prefix",
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("cluster_walk_order", "shards")}, indent=1, sort_keys=True))
    for index, shard in enumerate(shards):
        print(f"shard {index}: {len(shard)} pairs -> {','.join(str(p) for p in shard)}")
    return 0


# ----------------------------------------------------------------------------------
# stage: cluster-search -- a SECOND token, realized jointly with the first
# ----------------------------------------------------------------------------------


def cmd_cluster_search(args) -> int:
    """Two tokens on one pair, realized as ONE render / ONE argmax / ONE carrier re-solve.

    The anchor is the pair's best single-token proposal from the carried pool; the second
    token is drawn from the anchor's neighbourhood.  The row's credit is measured against the
    pair's OWN base, never against the anchor's -- a cluster is one object, and reporting it
    as anchor-plus-delta would be exactly the additivity this campaign refuses
    ([[composition_of_disjoint_token_edits_is_subadditive_on_seg_by_pair_overlap_20260910]]).
    """
    bind_move52(verify_raw=False, raw=args.raw)
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pd1_pose_directed as pd1
    import ddm_pp1_pose_actuation as pp1

    pp1.set_threads(args.threads)
    started = time.time()
    anchors = {int(k): v for k, v in json.loads(Path(args.anchors).read_text()).items()}
    pairs = [int(p) for p in args.pairs.split(",") if p.strip() != ""]
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    body = pp1.load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = pp1.build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_mean)
    shipped_argmax = np.load(pp1.LIVE_ARGMAX, mmap_mode="r")
    deltas = [int(d) for d in args.deltas.split(",")]
    token_classes = int(body.semantic.token_embed.num_embeddings)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"cluster_{args.shard_index}.jsonl"
    screen_path = out_dir / f"cluster_screen_{args.shard_index}.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.exists():
        done = {int(json.loads(line)["pair"])
                for line in rows_path.read_text().splitlines() if line.strip()}
    handle = rows_path.open("a")
    screen_handle = screen_path.open("a")
    counters = {"pairs": 0, "screened": 0, "refined": 0, "skipped_done": 0}

    for pair in pairs:
        if pair in done and args.resume:
            counters["skipped_done"] += 1
            continue
        anchor = anchors.get(pair)
        if anchor is None:
            print(f"pair {pair}: no anchor, skipped", flush=True)
            continue
        a_row, a_col = int(anchor["cell"][0]), int(anchor["cell"][1])
        a_old, a_new = int(anchor["old"]), int(anchor["new"])
        fe1.restore_pair_codes(body, pair)
        base_plane = body.tokens[pair].copy()
        if int(base_plane[a_row, a_col]) != a_old:
            raise Pd4Error(
                f"pair {pair} anchor cell ({a_row},{a_col}) holds "
                f"{int(base_plane[a_row, a_col])} but the anchor row says {a_old}"
            )
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        gap = abs(d_pose_base - float(base[pair]))
        if gap > args.base_band_abs:
            raise Pd4Error(
                f"pair {pair} batch-1 base {d_pose_base:.12e} is {gap:.3e} from the n600 base "
                f"{float(base[pair]):.12e}, outside the MEASURED band {args.base_band_abs:.3e}"
            )
        saliency = pp1.pose_saliency_on_token_grid(base_inst, pair,
                                                   np.asarray(raw[2 * pair + 1]))
        interior = pd1.interior_mask(np.asarray(shipped_argmax[pair]), args.interior_radius)
        radius = int(args.neighbour_radius)
        neighbours = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = a_row + dr, a_col + dc
                if not (0 <= rr < EVAL_H and 0 <= cc < EVAL_W):
                    continue
                if not bool(interior[rr, cc]):
                    continue
                neighbours.append((rr, cc))
        neighbours.sort(key=lambda rc: -float(saliency[rc[0], rc[1]]))
        print(f"pair {pair}: anchor ({a_row},{a_col}) {a_old}->{a_new}, base flips "
              f"{base_flips}, d_pose {d_pose_base:.4e}, {len(neighbours)} interior neighbours",
              flush=True)
        counters["pairs"] += 1

        screened: list[dict[str, Any]] = []
        for rr, cc in neighbours:
            old2 = int(base_plane[rr, cc])
            for delta in deltas:
                new2 = old2 + delta
                if not 0 <= new2 < token_classes:
                    continue
                body.tokens[pair][a_row, a_col] = a_new
                body.tokens[pair][rr, cc] = new2
                frame = fe1.render_pair(body, pair)
                moved_flips = fe1.flips_pair(
                    jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
                )
                body.tokens[pair][a_row, a_col] = a_old
                body.tokens[pair][rr, cc] = old2
                counters["screened"] += 1
                entry = {
                    "pair": pair,
                    "edits": [[a_row, a_col, a_old, a_new], [rr, cc, old2, new2]],
                    "cell": [a_row, a_col], "old": a_old, "new": a_new,
                    "second_cell": [rr, cc], "second_old": old2, "second_new": new2,
                    "tokens_changed": 2,
                    "base_flips": base_flips, "flips": moved_flips,
                    "d_cells": moved_flips - base_flips,
                    "saliency": float(saliency[rr, cc]),
                    "anchor_saliency": float(saliency[a_row, a_col]),
                    "interior": True, "family": "cluster",
                }
                screen_handle.write(json.dumps(entry) + "\n")
                if entry["d_cells"] <= args.max_cells:
                    screened.append(entry)
        screen_handle.flush()
        screened.sort(key=lambda e: (e["d_cells"], -e["saliency"]))
        for entry in screened[: args.refine]:
            rr, cc = entry["second_cell"]
            body.tokens[pair][a_row, a_col] = a_new
            body.tokens[pair][rr, cc] = entry["second_new"]
            frame = fe1.render_pair(body, pair)
            body.tokens[pair][a_row, a_col] = a_old
            body.tokens[pair][rr, cc] = entry["second_old"]
            overlay = pp1.MemoryOverlayRaw(raw, {2 * pair + 1: frame[0]})
            moved_inst = br1.Instrument(
                base_inst.state, overlay, base_inst.targets, base_inst.posenet,
                base_inst.blow, base_inst.gram, base_inst.bmat,
            )
            stale = float(br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0])
            refined = jg5.refine_pair(
                moved_inst, pair, live_codes[pair], dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
            )
            resolved = float(refined["final_d_pose"])
            counters["refined"] += 1
            credit = resolved - d_pose_base
            row = dict(entry)
            row.update({
                "d_pose_base": d_pose_base,
                "d_pose_stale": stale,
                "d_pose_resolved": resolved,
                "resolved_over_base": resolved / d_pose_base if d_pose_base > 0 else math.nan,
                "carrier_codes": [int(c) for c in refined["codes"]],
                "base_gap_abs_vs_n600": gap,
                "credit_d_pose": credit,
                "dS_seg": entry["d_cells"] * seg_cell,
                "dS_pose_resolved": credit * pose_unit,
                "row_source": "ddm_pd4_cluster",
            })
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"pair {pair} cluster ({a_row},{a_col})+({rr},{cc}) -> cells "
                  f"{row['d_cells']:+d}, resolved {resolved:.4e} "
                  f"({row['resolved_over_base']:.4f}x)", flush=True)
    handle.close()
    screen_handle.close()
    (out_dir / f"CLUSTER_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pd4_cluster_search.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "proposal": "second token in the anchor's neighbourhood, realized JOINTLY with it",
        "pairs": pairs, "counters": counters,
        "neighbour_radius": args.neighbour_radius, "deltas": deltas,
        "refine": args.refine, "max_cells": args.max_cells,
        "rows_path": str(rows_path), "screen_path": str(screen_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path), "counters": counters}))
    return 0


# ----------------------------------------------------------------------------------
# stage: smoke-timing -- declare K_refine, the pair budget and the PRICE method from
# a measurement, under the fleet condition this arm actually runs in
# ----------------------------------------------------------------------------------


def cmd_smoke_timing(args) -> int:
    """Two costs in one smoke: seconds per cluster-search pair, and seconds per PRICED
    proposal on the real-encode rail -- both measured with the other running."""
    shards = []
    for path in sorted(Path(args.smoke_dir).glob("CLUSTER_*.json")):
        payload = json.loads(path.read_text())
        counters = payload["counters"]
        shards.append({
            "shard": int(path.stem.split("_")[-1]),
            "pairs": payload["pairs"],
            "screened": int(counters["screened"]),
            "refined": int(counters["refined"]),
            "wall_seconds": float(payload["elapsed_seconds"]),
        })
    if not shards:
        raise Pd4Error(f"no CLUSTER_*.json under {args.smoke_dir}")
    walls = np.array([s["wall_seconds"] for s in shards], dtype=np.float64)
    refines = np.array([s["refined"] for s in shards], dtype=np.float64)
    pairs = float(sum(len(s["pairs"]) for s in shards))
    slowest = float(walls.max())
    throughput = pairs / (slowest / 60.0)  # pairs per minute for the whole fleet

    price = json.loads(Path(args.price_encode).read_text())
    priced_proposals = int(args.priced_proposals)
    price_seconds = float(args.price_seconds)

    report = {
        "schema": "ddm_pd4_smoke_timing.v1",
        "axis": "[macOS-CPU advisory / wall-clock and EXACT bits]",
        "score_claim": False,
        "fleet_condition": (
            f"{len(shards)} concurrent cluster-search shards at 2 threads each, CONCURRENT "
            "with the smoke price encode and the move-52 re-search -- the condition this arm "
            "runs in, not an idle machine"
        ),
        "shards": shards,
        "wall_per_pair_seconds": {
            "min": float(walls.min()), "max": float(walls.max()),
            "mean": float(walls.mean()), "median": float(np.median(walls)),
        },
        "seconds_per_refine": {
            "mean": float((walls / np.maximum(refines, 1)).mean()),
            "median": float(np.median(walls / np.maximum(refines, 1))),
        },
        "fleet_throughput_pairs_per_minute": throughput,
        "K_refine_DECLARED": args.k_refine,
        "K_refine_evidence": (
            "pd3 MEASURED K=12 strictly better than K=8 on 24 of 125 shared pairs and worse "
            "on none; this arm keeps 12 so the declared delta stays the PRICE, not the depth"
        ),
        "pair_budget_DECLARED": args.pair_budget,
        "pair_budget_projection_minutes": (args.pair_budget / throughput) if throughput else None,
        "price_method": {
            "rail": "ddm_sj1_rlc1_price.encode -- the SHIPPED receiver's own decode loop with "
                    "true symbols injected; per_frame_bits = sum -log2 p_model(symbol)",
            "unit_priced": "one SHEET = one 600-frame encode prices at most one proposal per "
                           "pair, frame-locally, in a single pass",
            "encode_wall_seconds": price_seconds,
            "proposals_priced_by_that_encode": priced_proposals,
            "seconds_per_priced_proposal": (
                price_seconds / priced_proposals if priced_proposals else None
            ),
            "smoke_field_tokens_changed": int(
                price.get("binding", {}).get("field_tokens_changed", priced_proposals)
            ),
            "encode_stream_bytes": int(price["stream_bytes"]),
            "encode_ideal_bytes": float(price["ideal_bytes"]),
            "encode_archives": [a["sha256"] for a in price["archives"]],
            "encode_archive_bytes": [a["bytes"] for a in price["archives"]],
            "twins_agree": price["archives"][0]["sha256"] == price["archives"][1]["sha256"],
            "output_lossless": bool(price["output_lossless"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: sheets -- turn ranked proposals into fields the shipped coder can price at once
# ----------------------------------------------------------------------------------


def cmd_sheets(args) -> int:
    """One sheet = at most ONE proposal per pair, so ONE real encode prices them all.

    Sheet k carries every pair's rank-k proposal, or its LAST proposal when it has fewer
    than k+1.  Repeating rather than dropping keeps the edit DENSITY identical across sheets
    -- the cross-frame spill a neighbour's edit causes is then common to every sheet and
    cancels in the per-pair comparison -- and it hands the arm a free control: a pair with
    only one proposal is held FIXED across all sheets, so the spread of its measured delta
    bits IS the price's noise floor (falsifier F3).
    """
    bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    excluded = set(edited_pairs_at_move52())
    pool, stats = load_pool([Path(p) for p in args.rows], base,
                            band_abs=BASE_BAND_ABS, exclude_pairs=excluded)
    score_proposals(pool, pose_unit=pose_unit, seg_cell=seg_cell)

    # Only proposals that can EVER pay are worth an encode slot.
    only = {int(p) for p in args.only_pairs.split(",") if p.strip()} if args.only_pairs else None
    ranked: dict[int, list[dict[str, Any]]] = {}
    for pair, entries in pool.items():
        if only is not None and pair not in only:
            continue
        paying = [e for e in entries if e["benefit_S"] > 0.0]
        if not paying:
            continue
        paying.sort(key=lambda e: -e["benefit_S"])
        ranked[pair] = paying[: args.per_pair]
    if not ranked:
        raise Pd4Error("no paying proposal survived the gates; there is nothing to price")

    source = np.load(args.control_npz)
    planes = {int(k): source[k] for k in source.files}
    if len(planes) != N_PAIRS:
        raise Pd4Error(f"control field has {len(planes)} planes, expected {N_PAIRS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_count = max(len(v) for v in ranked.values())
    manifest: list[dict[str, Any]] = []
    for k in range(sheet_count):
        field = {str(p): planes[p].copy() for p in range(N_PAIRS)}
        placed = 0
        for pair, entries in ranked.items():
            entry = entries[min(k, len(entries) - 1)]
            plane = field[str(pair)]
            for r, c, old, new in row_edits(entry):
                if int(plane[r, c]) != old:
                    raise Pd4Error(
                        f"pair {pair} cell ({r},{c}) holds {int(plane[r, c])} in the control "
                        f"field but the row says {old}"
                    )
                plane[r, c] = new
            placed += 1
            manifest.append({
                "sheet": k, "pair": pair, "rank": min(k, len(entries) - 1),
                "held_fixed": min(k, len(entries) - 1) != k,
                "key": list(proposal_key(entry)[1]),
                "edits": row_edits(entry),
                "tokens_changed": int(entry["tokens_changed"]),
                "d_cells": int(entry["d_cells"]),
                "credit_d_pose": float(entry["credit_d_pose"]),
                "benefit_S": float(entry["benefit_S"]),
                "row_source": entry.get("row_source", "unknown"),
                "family": entry.get("family", "single"),
                # the admission needs the pose legs and the re-solved carrier, so the whole
                # realized row travels with the price rather than only its identity.
                "d_pose_base": float(entry["d_pose_base"]),
                "d_pose_stale": float(entry["d_pose_stale"]),
                "d_pose_resolved": float(entry["d_pose_resolved"]),
                "base_gap_abs_vs_n600": float(entry.get("base_gap_abs_vs_n600", 0.0)),
                "carrier_codes": [int(c) for c in entry["carrier_codes"]],
            })
        path = out_dir / f"{args.name_prefix}_{k:02d}.npz"
        np.savez_compressed(path, **field)
        written = np.load(path)
        for pair in range(N_PAIRS):
            if not np.array_equal(written[str(pair)], field[str(pair)]):
                raise Pd4Error(f"sheet {k} does not round-trip plane {pair}")
        print(json.dumps({"sheet": k, "path": str(path), "pairs_edited": placed,
                          "bytes": path.stat().st_size}), flush=True)

    report = {
        "schema": "ddm_pd4_price_sheets.v1",
        "axis": "[macOS-CPU advisory / scorer-free field construction]",
        "score_claim": False,
        "rule": ("sheet k carries each pair's rank-k proposal, or its last when it has fewer; "
                 "edit density is therefore identical across sheets and a pair with one "
                 "proposal is a held-fixed price control"),
        "control_npz": str(args.control_npz),
        "gates": stats,
        "per_pair_cap": args.per_pair,
        "sheets": sheet_count,
        "pairs_priced": len(ranked),
        "proposals_priced": int(sum(len(v) for v in ranked.values())),
        "held_fixed_control_pairs": sorted(p for p, v in ranked.items() if len(v) == 1),
        "manifest": manifest,
    }
    (out_dir / "SHEETS.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"},
                     indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: price-merge -- the sheets' MEASURED per-frame bits become per-PROPOSAL prices
# ----------------------------------------------------------------------------------


def _encode_bits(rlc1_root: Path, field: str, tag: str) -> np.ndarray:
    path = Path(rlc1_root) / "encode" / field / tag / "ENCODE.json"
    if not path.exists():
        raise Pd4Error(f"{path} does not exist; that encode has not run")
    payload = json.loads(path.read_text())
    bits = np.asarray(payload["per_frame_bits"], dtype=np.float64)
    if bits.shape != (N_PAIRS,):
        raise Pd4Error(f"{path} carries {bits.shape} per-frame bits")
    if payload.get("output_lossless") is not True:
        raise Pd4Error(f"{path} is not output-lossless")
    return bits


def cmd_price_merge(args) -> int:
    """Delta bits per PROPOSAL, measured frame-locally by the shipped coder's own rows."""
    bind_move52(verify_raw=False, raw=args.raw)
    sheets = json.loads(Path(args.sheets).read_text())
    control = _encode_bits(args.rlc1_root, "control", args.control_tag)
    per_sheet = {
        int(k): _encode_bits(args.rlc1_root, f"pd4sheet{k:02d}", args.sheet_tag)
        for k in range(int(sheets["sheets"]))
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priced: dict[tuple, dict[str, Any]] = {}
    repeats: dict[tuple, list[float]] = {}
    for entry in sheets["manifest"]:
        pair, k = int(entry["pair"]), int(entry["sheet"])
        delta = float(per_sheet[k][pair] - control[pair])
        key = (pair, tuple(tuple(e) for e in entry["edits"]))
        repeats.setdefault(key, []).append(delta)
        row = dict(entry)
        row["real_delta_bits"] = delta
        row["bits_source"] = "sheet_encode_frame_local"
        if key not in priced:
            priced[key] = row
    # THE PRICE'S OWN NOISE FLOOR.  A pair with fewer proposals than there are sheets is
    # held FIXED across the remainder, so the SAME proposal is measured more than once in
    # fields that differ at other pairs.  The spread of those measurements is the spill a
    # neighbouring pair's edit causes, measured rather than assumed.  It is recorded, never
    # averaged away: the first measurement is the price.
    for key, values in repeats.items():
        if len(values) > 1:
            priced[key]["repeat_delta_bits"] = values
            priced[key]["repeat_spread_bits"] = float(max(values) - min(values))
    fixed_spread = {k: v for k, v in repeats.items() if len(v) > 1}

    rows = sorted(priced.values(), key=lambda r: (int(r["pair"]), int(r["rank"])))
    rows_path = out_dir / "priced_rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    spreads = [max(v) - min(v) for v in fixed_spread.values()]
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), []).append(row)
    within = [
        max(r["real_delta_bits"] for r in v) - min(r["real_delta_bits"] for r in v)
        for v in by_pair.values() if len(v) > 1
    ]
    singles = [r for r in rows if int(r["tokens_changed"]) == 1]
    clusters = [r for r in rows if int(r["tokens_changed"]) > 1]

    # SECOND, INDEPENDENT SPILL MEASUREMENT.  The smoke priced eight proposals in a field
    # that changes only those eight pairs.  The same proposals are priced again inside a
    # sheet that changes ~140.  The difference is the spill a crowd of neighbours causes on
    # a price, measured on a different object from the held-fixed repeats.
    cross: dict[str, Any] = {}
    if args.cross_check_field and args.cross_check_sheets:
        sparse = _encode_bits(args.rlc1_root, args.cross_check_field, args.sheet_tag)
        sparse_manifest = json.loads(Path(args.cross_check_sheets).read_text())
        # Match on the PROPOSAL, never on the rank: the sparse field was built before the
        # cluster rows existed, so a pair's rank-0 proposal is not the same object in both.
        sparse_keys = {
            (int(e["pair"]), tuple(tuple(v) for v in e["edits"])): int(e["pair"])
            for e in sparse_manifest["manifest"]
        }
        pairs_checked, diffs = [], []
        for row in rows:
            key = (int(row["pair"]), tuple(tuple(e) for e in row["edits"]))
            if key not in sparse_keys:
                continue
            pair = int(row["pair"])
            delta = float(sparse[pair] - control[pair])
            pairs_checked.append(pair)
            diffs.append(row["real_delta_bits"] - delta)
        if diffs:
            arr = np.asarray(diffs, dtype=np.float64)
            cross = {
                "field": args.cross_check_field,
                "matched_on": "the proposal's own (pair, edits) identity, not its rank",
                "pairs": pairs_checked,
                "sheet_minus_sparse_bits": [float(d) for d in arr],
                "max_abs_bits": float(np.abs(arr).max()),
                "median_abs_bits": float(np.median(np.abs(arr))),
                "mean_bits": float(arr.mean()),
            }

    def per_token(rows_in: list[dict[str, Any]]) -> dict[str, float]:
        if not rows_in:
            return {}
        bits = np.array([r["real_delta_bits"] for r in rows_in], dtype=np.float64)
        toks = np.array([int(r["tokens_changed"]) for r in rows_in], dtype=np.float64)
        return {
            "proposals": len(rows_in),
            "tokens": float(toks.sum()),
            "total_delta_bits": float(bits.sum()),
            "bits_per_token_pooled": float(bits.sum() / toks.sum()),
            "bits_per_token_median": float(np.median(bits / toks)),
            "bits_per_token_min": float((bits / toks).min()),
            "bits_per_token_max": float((bits / toks).max()),
        }

    report = {
        "schema": "ddm_pd4_price_merge.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT bit measurement through the shipped coder]",
        "score_claim": False,
        "method": (
            "per_frame_bits = sum over the frame's coded positions of -log2 p_model(symbol), "
            "driven through the SHIPPED receiver's decode loop with true symbols injected "
            "(ddm_sj1_rlc1_price.encode).  A proposal's price is its frame-local delta "
            "against the control encode of move 52's own field."
        ),
        "frame_local_scope": (
            "MEASURED on pd3's own 271-token field: 97.21 % of the signed delta bits land on "
            "the EDITED frames.  This price is that 97.2 %; the CHARGE is the admitted "
            "subset's own full real encode with twins."
        ),
        "control_encode": f"{args.rlc1_root}/encode/control/{args.control_tag}/ENCODE.json",
        "sheets": int(sheets["sheets"]),
        "proposals_priced": len(rows),
        "price_noise_floor_held_fixed_proposals": {
            "proposals_measured_more_than_once": len(spreads),
            "max_spread_bits": float(max(spreads)) if spreads else 0.0,
            "median_spread_bits": float(np.median(spreads)) if spreads else 0.0,
            "mean_spread_bits": float(np.mean(spreads)) if spreads else 0.0,
        },
        "within_pair_price_range_bits": {
            "pairs": len(within),
            "median": float(np.median(within)) if within else 0.0,
            "max": float(max(within)) if within else 0.0,
        },
        "sparse_vs_sheet_cross_check": cross,
        "single_token_proposals": per_token(singles),
        "clustered_proposals": per_token(clusters),
        "rows_path": str(rows_path),
    }
    (out_dir / "PRICE_MERGE.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: carry -- the price lever itself: best proposal per pair by credit per REAL bit
# ----------------------------------------------------------------------------------


def cmd_carry(args) -> int:
    """Rank by benefit / REAL bits, and report what that ranking changed against best-benefit."""
    bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    rows = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
            if line.strip()]
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        row["benefit_S"] = (
            -(float(row["credit_d_pose"]) * pose_unit) - float(row["d_cells"]) * seg_cell
        )
        row["dS_rate_real"] = (float(row["real_delta_bits"]) / 8.0) * S_PER_BYTE
        row["dS_modelled_real_price"] = -row["benefit_S"] + row["dS_rate_real"]
        row["credit_S_per_bit"] = credit_per_bit(row["benefit_S"], float(row["real_delta_bits"]))
        by_pair.setdefault(int(row["pair"]), []).append(row)

    per_bit_wins: dict[int, dict[str, Any]] = {}
    per_benefit_wins: dict[int, dict[str, Any]] = {}
    changed: list[dict[str, Any]] = []
    for pair, entries in by_pair.items():
        # Two free-and-paying rows both rank +inf; the tie breaks on benefit, never on
        # file order.
        by_bit = max(entries, key=lambda e: (e["credit_S_per_bit"], e["benefit_S"]))
        by_ben = max(entries, key=lambda e: e["benefit_S"])
        per_bit_wins[pair] = by_bit
        per_benefit_wins[pair] = by_ben
        if proposal_key_from_priced(by_bit) != proposal_key_from_priced(by_ben):
            changed.append({
                "pair": pair,
                "per_bit": {"rank": int(by_bit["rank"]),
                            "benefit_S": by_bit["benefit_S"],
                            "bits": by_bit["real_delta_bits"],
                            "dS": by_bit["dS_modelled_real_price"]},
                "per_benefit": {"rank": int(by_ben["rank"]),
                                "benefit_S": by_ben["benefit_S"],
                                "bits": by_ben["real_delta_bits"],
                                "dS": by_ben["dS_modelled_real_price"]},
                "dS_improvement": by_ben["dS_modelled_real_price"] - by_bit["dS_modelled_real_price"],
            })

    def net(wins: dict[int, dict[str, Any]]) -> dict[str, Any]:
        keep = [e for e in wins.values() if e["dS_modelled_real_price"] < 0.0]
        tokens = sum(int(e["tokens_changed"]) for e in keep)
        bits = sum(float(e["real_delta_bits"]) for e in keep)
        return {
            "pairs_that_pay": len(keep),
            "tokens": tokens,
            "delta_bits": bits,
            "bits_per_token": bits / tokens if tokens else math.nan,
            "modelled_net_dS": float(sum(e["dS_modelled_real_price"] for e in keep)),
        }

    carried = sorted(
        (e for e in per_bit_wins.values() if e["dS_modelled_real_price"] < 0.0),
        key=lambda e: e["dS_modelled_real_price"],
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    carry_path = out_dir / "carry_rows.jsonl"
    carry_path.write_text("".join(json.dumps(e) + "\n" for e in carried))
    report = {
        "schema": "ddm_pd4_carry.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "rule": "best proposal per pair by resolved-pose+seg benefit per REAL measured bit",
        "pose_unit_S_per_pair_d_pose": pose_unit,
        "seg_cell_S": seg_cell,
        "pairs_priced": len(by_pair),
        "ranking_by_real_bits": net(per_bit_wins),
        "ranking_by_benefit_only_pd3s_rule": net(per_benefit_wins),
        "pairs_where_the_rankings_disagree": len(changed),
        "disagreements": changed,
        "carry_rows": str(carry_path),
        "carried": len(carried),
    }
    (out_dir / "CARRY_RANKING.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"},
                     indent=1, sort_keys=True))
    return 0


def proposal_key_from_priced(row: dict[str, Any]) -> tuple:
    return (int(row["pair"]), tuple(sorted((int(e[0]), int(e[1]), int(e[3]))
                                           for e in row["edits"])))


# ----------------------------------------------------------------------------------
# stage: assemble -- carry rows -> the candidate field + the admission's pass rows
# ----------------------------------------------------------------------------------


def cmd_assemble(args) -> int:
    """pd1's assemble, generalized to MULTI-token pairs, with its revert guard kept.

    pd1's own ``assemble`` writes one ``cell``/``old``/``new`` per pair and stamps
    ``tokens_changed: 1``; a clustered pair needs a list.  The two invariants that matter are
    carried verbatim: every plane is written against the CARRY FIELD (a pair merely absent
    from an edit npz reverts to the ORIGINAL base and silently undoes what the live pointer
    banked -- MEASURED by sj1 pass 3 at 230 pairs / 2,337 tokens), and the field must
    round-trip.
    """
    bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    rows = [json.loads(line) for line in Path(args.carry_rows).read_text().splitlines()
            if line.strip()]
    source = np.load(args.carry_field)
    planes = {int(k): source[k].copy() for k in source.files}
    if len(planes) != N_PAIRS:
        raise Pd4Error(f"carry field has {len(planes)} planes, expected {N_PAIRS}")
    stale = base.copy()
    resolved = base.copy()
    pass_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: int(r["pair"])):
        pair = int(row["pair"])
        plane = planes[pair]
        edits = [(int(e[0]), int(e[1]), int(e[2]), int(e[3])) for e in row["edits"]]
        for r, c, old, new in edits:
            if int(plane[r, c]) != old:
                raise Pd4Error(
                    f"pair {pair} cell ({r},{c}) holds {int(plane[r, c])} in the carry field "
                    f"but the row says {old}; the search and the base disagree"
                )
            plane[r, c] = new
        stale[pair] = base[pair] + (float(row["d_pose_stale"]) - float(row["d_pose_base"]))
        resolved[pair] = base[pair] + float(row["credit_d_pose"])
        pass_rows.append({
            "pair": pair,
            "flips_repaired": -int(row["d_cells"]),
            "tokens_changed": len(edits),
            "edits": [list(e) for e in edits],
            "cell": [edits[0][0], edits[0][1]],
            "old": edits[0][2], "new": edits[0][3],
            "d_cells": int(row["d_cells"]),
            "credit_d_pose": float(row["credit_d_pose"]),
            "fraction_of_pair_d_pose": (
                float(row["credit_d_pose"]) / float(base[pair]) if base[pair] > 0 else math.nan
            ),
            "d_pose_base_n600": float(base[pair]),
            "d_pose_base_batch1": float(row["d_pose_base"]),
            "d_pose_stale_batch1": float(row["d_pose_stale"]),
            "d_pose_resolved_batch1": float(row["d_pose_resolved"]),
            "base_gap_abs_vs_n600": float(row.get("base_gap_abs_vs_n600", 0.0)),
            "carrier_codes": [int(c) for c in row["carrier_codes"]],
            "real_delta_bits": float(row["real_delta_bits"]),
            "dS_modelled_real_price": float(row["dS_modelled_real_price"]),
            "family": row.get("family", "single"),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field = out_dir / "field_candidate.npz"
    np.savez_compressed(field, **{str(p): plane for p, plane in sorted(planes.items())})
    written = np.load(field)
    edited = {int(r["pair"]) for r in pass_rows}
    for pair in range(N_PAIRS):
        if not np.array_equal(written[str(pair)], planes[pair]):
            raise Pd4Error(f"candidate field does not round-trip plane {pair}")
        if pair not in edited and not np.array_equal(written[str(pair)], source[str(pair)]):
            raise Pd4Error(
                f"pair {pair} is not edited but its plane differs from the carry field; "
                "that would revert banked edits"
            )
    (out_dir / "pass_rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pass_rows)
    )
    np.save(out_dir / "pose_stale.npy", stale)
    np.save(out_dir / "pose_resolved.npy", resolved)
    if args.codes_base:
        codes = np.load(args.codes_base).astype(np.int32).copy()
        if codes.shape[0] != N_PAIRS:
            raise Pd4Error(f"base carrier codes have shape {codes.shape}")
        for row in pass_rows:
            codes[row["pair"]] = np.asarray(row["carrier_codes"], dtype=np.int32)
        np.save(out_dir / "codes_resolved.npy", codes)
    bits_candidate = None
    if args.bits_control:
        control = np.load(args.bits_control).astype(np.float64).copy()
        if control.shape != (N_PAIRS,):
            raise Pd4Error(f"control bit ledger has shape {control.shape}")
        for row in pass_rows:
            control[row["pair"]] += float(row["real_delta_bits"])
        bits_candidate = out_dir / "bits_candidate_spliced.npy"
        np.save(bits_candidate, control)

    tokens = int(sum(r["tokens_changed"] for r in pass_rows))
    bits = float(sum(r["real_delta_bits"] for r in pass_rows))
    summary = {
        "schema": "ddm_pd4_assemble.v1",
        "axis": "[macOS-CPU advisory / field construction + measured bit splice]",
        "score_claim": False,
        "pairs": len(pass_rows),
        "tokens": tokens,
        "clustered_pairs": int(sum(1 for r in pass_rows if r["tokens_changed"] > 1)),
        "sum_real_delta_bits": bits,
        "bits_per_token": bits / tokens if tokens else math.nan,
        "modelled_bytes": bits / 8.0,
        "sum_credit_d_pose": float(sum(r["credit_d_pose"] for r in pass_rows)),
        "sum_d_cells": int(sum(r["d_cells"] for r in pass_rows)),
        "sum_dS_modelled_real_price": float(sum(r["dS_modelled_real_price"] for r in pass_rows)),
        "field": str(field),
        "bits_candidate_spliced": str(bits_candidate) if bits_candidate else None,
        "bits_candidate_scope": (
            "control + each pair's OWN sheet-measured frame-local delta; a splice of "
            "per-sheet real encodes, re-measured exactly by the admitted subset's own encode"
        ),
    }
    (out_dir / "ASSEMBLE.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    bind = sub.add_parser("bind", help="verify + re-bind move 52, write the receipt")
    bind.add_argument("--out", type=Path, required=True)
    bind.add_argument("--raw", type=Path, default=None)
    bind.add_argument("--verify-raw", action="store_true")
    bind.set_defaults(func=cmd_bind)

    base = sub.add_parser("base", help="move 52's own per-pair d_pose, n600")
    base.add_argument("--out-dir", type=Path, required=True)
    base.add_argument("--raw", type=Path, default=None)
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=8)
    base.add_argument("--control", type=Path, default=None)
    base.add_argument("--verify-raw", action="store_true")
    base.set_defaults(func=cmd_base)

    run = sub.add_parser("run", help="bind, then dispatch a sister module's argv in-process")
    run.add_argument("--module",
                     choices=("pd1", "pd2", "pd3", "joint", "pp1", "tree"), required=True)
    run.add_argument("--raw", type=Path, default=None)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    pre = sub.add_parser("prereg", help="economics at move 52 + the pool + the falsifiers")
    pre.add_argument("--base-pose", type=Path, required=True)
    pre.add_argument("--rows", nargs="+", required=True)
    pre.add_argument("--raw", type=Path, default=None)
    pre.add_argument("--out", type=Path, required=True)
    pre.set_defaults(func=cmd_prereg)

    pl = sub.add_parser("plan", help="anchors for the cluster family + the shard assignment")
    pl.add_argument("--base-pose", type=Path, required=True)
    pl.add_argument("--rows", nargs="+", required=True)
    pl.add_argument("--shard-count", type=int, default=8)
    pl.add_argument("--max-pairs", type=int, default=0)
    pl.add_argument("--raw", type=Path, default=None)
    pl.add_argument("--out", type=Path, required=True)
    pl.set_defaults(func=cmd_plan)

    cl = sub.add_parser("cluster-search", help="a SECOND token, realized jointly with the first")
    cl.add_argument("--pairs", required=True)
    cl.add_argument("--anchors", type=Path, required=True)
    cl.add_argument("--base-pose", type=Path, required=True)
    cl.add_argument("--base-band-abs", type=float, required=True)
    cl.add_argument("--raw", type=Path, default=None)
    cl.add_argument("--deltas", default="-1,1")
    cl.add_argument("--interior-radius", type=int, default=3)
    cl.add_argument("--neighbour-radius", type=int, default=1)
    cl.add_argument("--max-cells", type=int, default=2)
    cl.add_argument("--refine", type=int, default=12)
    cl.add_argument("--outer-rounds", type=int, default=40)
    cl.add_argument("--max-gn-iterations", type=int, default=400)
    cl.add_argument("--threads", type=int, default=2)
    cl.add_argument("--shard-index", type=int, default=0)
    cl.add_argument("--out-dir", type=Path, required=True)
    cl.add_argument("--resume", action="store_true")
    cl.set_defaults(func=cmd_cluster_search)

    st = sub.add_parser("smoke-timing", help="declare K_refine, the budget and the price method")
    st.add_argument("--smoke-dir", type=Path, required=True)
    st.add_argument("--price-encode", type=Path, required=True)
    st.add_argument("--price-seconds", type=float, required=True)
    st.add_argument("--priced-proposals", type=int, required=True)
    st.add_argument("--k-refine", type=int, default=12)
    st.add_argument("--pair-budget", type=int, required=True)
    st.add_argument("--out", type=Path, required=True)
    st.set_defaults(func=cmd_smoke_timing)

    sh = sub.add_parser("sheets", help="ranked proposals -> price sheets (one per pair each)")
    sh.add_argument("--rows", nargs="+", required=True)
    sh.add_argument("--base-pose", type=Path, required=True)
    sh.add_argument("--control-npz", type=Path, default=MOVE52_FIELD)
    sh.add_argument("--per-pair", type=int, default=6)
    sh.add_argument("--only-pairs", default="")
    sh.add_argument("--name-prefix", default="sheet")
    sh.add_argument("--raw", type=Path, default=None)
    sh.add_argument("--out-dir", type=Path, required=True)
    sh.set_defaults(func=cmd_sheets)

    pm = sub.add_parser("price-merge", help="read the sheets' real bits back onto the proposals")
    pm.add_argument("--sheets", type=Path, required=True)
    pm.add_argument("--rlc1-root", type=Path, required=True)
    pm.add_argument("--control-tag", default="primary")
    pm.add_argument("--sheet-tag", default="primary")
    pm.add_argument("--cross-check-field", default="")
    pm.add_argument("--cross-check-sheets", type=Path, default=None)
    pm.add_argument("--raw", type=Path, default=None)
    pm.add_argument("--out-dir", type=Path, required=True)
    pm.set_defaults(func=cmd_price_merge)

    ca = sub.add_parser("carry", help="best proposal per pair by credit per REAL bit")
    ca.add_argument("--priced", type=Path, required=True)
    ca.add_argument("--base-pose", type=Path, required=True)
    ca.add_argument("--raw", type=Path, default=None)
    ca.add_argument("--out-dir", type=Path, required=True)
    ca.set_defaults(func=cmd_carry)

    asm = sub.add_parser("assemble", help="carry rows -> candidate field + pass rows (multi-edit)")
    asm.add_argument("--carry-rows", type=Path, required=True)
    asm.add_argument("--base-pose", type=Path, required=True)
    asm.add_argument("--carry-field", type=Path, default=MOVE52_FIELD)
    asm.add_argument("--codes-base", type=Path, default=None)
    asm.add_argument("--bits-control", type=Path, default=None)
    asm.add_argument("--raw", type=Path, default=None)
    asm.add_argument("--out-dir", type=Path, required=True)
    asm.set_defaults(func=cmd_assemble)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
