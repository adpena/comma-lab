#!/usr/bin/env python3
"""ddm_pd2: POSE-DIRECTED token pre-distortion, PASS 2, on the MOVE-50 field.

pd1 (`experiments/ddm_pd1_pose_directed.py`) is the reference form and it is REUSED here
rather than re-implemented: this module supplies (a) the MOVE-50 binding every pd1/sj1
stage needs, (b) move 50's own pose base, (c) the economics at move 50's operating point
priced by pd1's MEASURED 16.98 bits/token rather than by its 1.9x-optimistic prior, and
(d) the per-bit carry rule.  The actuator, the screen, the refine and the admission are
pd1's and sj1's, unchanged.

THE BINDING, stated once.  `ddm_pp1_pose_actuation` and `ddm_sj1_multipass_token_predistortion`
pin the live pointer at MOVE 49 in module globals evaluated at import time.  Move 50 is a
real pointer move, so every one of those globals is STALE for this arm.  This module
re-binds them PROCESS-LOCALLY (never on disk, never for any other process) and refuses to
proceed unless every re-bound object reproduces its declared sha256 and the row's own score
arithmetic reproduces the packet's S from its three components.  The shared modules are not
edited: an arm that does not import this one is unaffected.

Stages
------
``bind``    verify + re-bind move 50, write the receipt, touch nothing else.
``base``    move 50's own per-pair d_pose, n600, on the decode the T4 row scored.
``prereg``  the economics at move 50 + the search population + the falsifiers.
``run``     bind, then dispatch a pd1 / sj1-joint-admission argv inside this process.
``carry``   per-bit carry: best proposal per pair, ranked by resolved-pose credit per BIT.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

N_PAIRS = 600
EVAL_H, EVAL_W = 384, 512
CELL_COUNT = N_PAIRS * EVAL_H * EVAL_W
SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_BYTE = 25.0 / SCORE_RATE_DENOMINATOR

#: pp1 measured these twelve as a FLOOR the frame_1 render cannot lower.  Excluded here on
#: pp1's measurement, exactly as pd1 excluded them; nothing in this arm reopens them.
FLOOR_PAIRS = (88, 87, 73, 316, 89, 70, 63, 448, 64, 91, 66, 67)

#: MOVE 50 -- the live row.  Lane ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913,
#: call fc-01M2BYVCXGYTFC2BN8WZSF5PAS, packet
#: `.omx/research/ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913_pointer_move_50_20260912.md`.
#: Every path below is another arm's custody and is opened READ-ONLY.
MOVE50_TREE = Path("/Volumes/APDataStore/pact/ddm_pd1/candidate/candidate_runtime")
MOVE50_ARCHIVE_SHA256 = "1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7"
MOVE50_ARCHIVE_BYTES = 179_195
MOVE50_SCORE_T4 = 0.13628342713679067
MOVE50_D_SEG_T4 = 0.00010294
MOVE50_D_POSE_T4 = 4.45e-06
#: move 50's OWN cold parse-back of its shipped bytes -- the decode the T4 row scored.
MOVE50_RAW = Path("/Volumes/APDataStore/pact/ddm_pd1/parseback/0.raw")
MOVE50_RAW_SHA256 = "135c9b3a580fcc84d74da8e2ac547f6ba06059b5e2148c8a2086ef843bd72fb7"
MOVE50_RAW_BYTES = 3_662_409_600
#: the token field those bytes decode to (pd1's parse-back: decoded_field_matches_admitted).
MOVE50_FIELD = Path("/Volumes/APDataStore/pact/ddm_pd1/admission/field_admitted.npz")
#: the n600 argmax of that decode -- move 50's own seg receipt, 12,135 flipped cells.
MOVE50_ARGMAX = Path("/Volumes/APDataStore/pact/ddm_pd1/seg_final/argmax_n600.npy")
MOVE50_ARGMAX_SHA256 = "b3db8fb550fa33ff71bcd2c0418622238ead71d641e9c392569851cf19f2b81f"
#: move 50's instrument seg leg on that decode, DALI lineage: 12,135 / (600*384*512).
MOVE50_FLIPS = 12_135
MOVE50_INSTRUMENT_D_SEG = 12_135 / CELL_COUNT
#: the twenty pairs move 50 itself edited; their renders MOVED, so pass 2 re-searches them
#: from the new render rather than reusing a pd1 row measured on the move-49 render.
MOVE50_EDITED_PAIRS = (
    "/Volumes/APDataStore/pact/ddm_pd1/admission/kept_pairs.json"
)

#: pd1's MEASURED price for an isolated one-token-per-pair edit on this coder and this
#: container: 16.98 bits per changed token (its own real encode of the 41-pair field;
#: the 20-pair subset measured 16.00).  pd1's PRE-registered 8.93 was 1.90x optimistic.
#: MEASURED, on one field, at this edit shape -- not a law.  Re-measured by this arm's own
#: real encode before anything is admitted.
PD1_MEASURED_BITS_PER_TOKEN = 16.98
#: pd1's MEASURED absolute batch-1 vs batch-8 band over 40 pairs spanning the whole range,
#: re-derived here from pd1's own 40 rows: max |gap| 2.2642595483754955e-09, gate = 10x.
BASE_BAND_ABS = 2.2642595483754955e-08


class Pd2Error(RuntimeError):
    """A pd2 input or invariant is not what the measured object says it is."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def composed_score(d_seg: float, d_pose: float, archive_bytes: float) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / SCORE_RATE_DENOMINATOR


def seg_s_per_cell() -> float:
    """S carried by ONE flipped cell on the T4 axis, through the instrument ratio."""
    ratio = MOVE50_D_SEG_T4 / MOVE50_INSTRUMENT_D_SEG
    return 100.0 * ratio / CELL_COUNT


def pose_s_per_pair_unit(base_mean: float) -> float:
    """dS / d(one pair's d_pose) at this base mean.  EXPIRES at every pointer move."""
    return math.sqrt(10.0 / base_mean) / (2.0 * N_PAIRS)


def edited_pairs_at_move50() -> list[int]:
    return sorted(int(p) for p in json.loads(Path(MOVE50_EDITED_PAIRS).read_text()))


# ----------------------------------------------------------------------------------
# THE BINDING
# ----------------------------------------------------------------------------------


def bind_move50(*, verify_raw: bool = False) -> dict[str, Any]:
    """Re-point pp1 and sj1 at MOVE 50, process-locally, and fail closed if anything drifts.

    The two shared modules bind the live pointer in globals evaluated at import time, and
    that pointer is move 49.  Re-binding is therefore mandatory, and it is exactly the kind
    of silent rebinding that hands a successor a wrong number, so every object is checked
    against its declared sha BEFORE it is bound and the pointer's own score arithmetic is
    re-derived from its three components afterwards.
    """
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ddm_pp1_pose_actuation as pp1
    import ddm_sj1_multipass_token_predistortion as sj1

    archive = MOVE50_TREE / "archive.zip"
    observed_bytes = archive.stat().st_size
    if observed_bytes != MOVE50_ARCHIVE_BYTES:
        raise Pd2Error(f"move-50 archive is {observed_bytes} B, not {MOVE50_ARCHIVE_BYTES}")
    observed_sha = sha256_file(archive)
    if observed_sha != MOVE50_ARCHIVE_SHA256:
        raise Pd2Error(f"move-50 archive sha is {observed_sha}, not {MOVE50_ARCHIVE_SHA256}")
    argmax_sha = sha256_file(MOVE50_ARGMAX)
    if argmax_sha != MOVE50_ARGMAX_SHA256:
        raise Pd2Error(f"move-50 argmax sha is {argmax_sha}, not {MOVE50_ARGMAX_SHA256}")
    raw_sha = None
    raw_bytes = MOVE50_RAW.stat().st_size
    if raw_bytes != MOVE50_RAW_BYTES:
        raise Pd2Error(f"move-50 decode is {raw_bytes} B, not {MOVE50_RAW_BYTES}")
    if verify_raw:
        raw_sha = sha256_file(MOVE50_RAW)
        if raw_sha != MOVE50_RAW_SHA256:
            raise Pd2Error(f"move-50 decode sha is {raw_sha}, not {MOVE50_RAW_SHA256}")

    recomputed = composed_score(MOVE50_D_SEG_T4, MOVE50_D_POSE_T4, MOVE50_ARCHIVE_BYTES)
    if abs(recomputed - MOVE50_SCORE_T4) > 1e-15:
        raise Pd2Error(
            f"move 50's three components recompute to {recomputed!r}, not the packet's "
            f"{MOVE50_SCORE_T4!r}"
        )

    row = sj1.PointerRow(
        label="pd1_pose_directed_move50",
        tree=MOVE50_TREE,
        archive_sha256=MOVE50_ARCHIVE_SHA256,
        archive_bytes=MOVE50_ARCHIVE_BYTES,
        d_seg_t4=MOVE50_D_SEG_T4,
        d_pose_t4=MOVE50_D_POSE_T4,
        score_t4=MOVE50_SCORE_T4,
    )
    row.verify_arithmetic()

    for module in (sj1, pp1):
        module.POINTER_TREE = MOVE50_TREE
        module.POINTER_ARCHIVE = archive
        module.POINTER_ARCHIVE_SHA256 = MOVE50_ARCHIVE_SHA256
        module.POINTER_ARCHIVE_BYTES = MOVE50_ARCHIVE_BYTES
        module.POINTER_SCORE_T4 = MOVE50_SCORE_T4
        module.POINTER_D_SEG_T4 = MOVE50_D_SEG_T4
        module.POINTER_D_POSE_T4 = MOVE50_D_POSE_T4
    sj1.LIVE_POINTER = row
    pp1.POINTER = row
    pp1.LIVE_RAW = MOVE50_RAW
    pp1.LIVE_FIELD = MOVE50_FIELD
    pp1.LIVE_ARGMAX = MOVE50_ARGMAX

    # Re-assert THROUGH the module that will be used, not through this one's copy of the
    # facts: if any global above were missed, this call still reports move 49.
    receipts = pp1.assert_pointer_identity()
    for name, expected in (
        ("pointer_archive_sha256", MOVE50_ARCHIVE_SHA256),
        ("live_raw", str(MOVE50_RAW)),
        ("live_field", str(MOVE50_FIELD)),
        ("live_argmax", str(MOVE50_ARGMAX)),
    ):
        if str(receipts[name]) != str(expected):
            raise Pd2Error(f"after binding, pp1 still reports {name}={receipts[name]!r}")
    if sj1.assert_carrier_is_pointer(MOVE50_TREE) != MOVE50_ARCHIVE_SHA256:
        raise Pd2Error("sj1's carrier anchor did not re-bind to move 50")
    receipts.update({
        "bound_move": 50,
        "argmax_sha256": argmax_sha,
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_sha,
        "score_recomputed_from_components": recomputed,
        "instrument_d_seg_move50": MOVE50_INSTRUMENT_D_SEG,
        "instrument_flips_move50": MOVE50_FLIPS,
        "edited_pairs_at_move50": edited_pairs_at_move50(),
    })
    return receipts


def cmd_bind(args) -> int:
    receipts = bind_move50(verify_raw=bool(args.verify_raw))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "ddm_pd2_bind.v1",
        "axis": "[macOS-CPU advisory / identity only]",
        "score_claim": False,
        "receipts": receipts,
    }, indent=1, sort_keys=True))
    print(json.dumps(receipts, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: base -- move 50's own per-pair d_pose, n600
# ----------------------------------------------------------------------------------


def cmd_base(args) -> int:
    receipts = bind_move50(verify_raw=bool(args.verify_raw))
    import ddm_pp1_pose_actuation as pp1
    import ddm_up2_shipping_pose_solve as up2

    pp1.set_threads(args.threads)
    started = time.time()
    raw = pp1.open_live_raw()
    inst = pp1.build_pose_instrument(raw)
    codes = np.asarray(inst.state.codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise Pd2Error(f"move-50 carrier codes have shape {codes.shape}")
    coefficients = up2.codes_to_coefficients(codes, inst.state.coefficient_scales)
    indices = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, _poses = up2.measure_pose(
        inst.posenet, inst.state, coefficients, inst.raw, inst.targets, indices,
        batch_size=args.batch_size,
    )
    mean = float(per_pair.mean())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = out_dir / "pose_base_move50.npy"
    np.save(per_pair_path, per_pair)
    np.save(out_dir / "codes_move50.npy", codes)
    order = np.argsort(-per_pair)
    report = {
        "schema": "ddm_pd2_pose_base.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT, n600]",
        "score_claim": False,
        "solver": (
            f"up2.measure_pose at batch {args.batch_size} on move 50's own cold parse-back"
        ),
        "receipts": receipts,
        "pairs": N_PAIRS,
        "d_pose_mean": mean,
        "pose_leg": math.sqrt(10.0 * mean),
        "t4_print": MOVE50_D_POSE_T4,
        "instrument_ratio_vs_t4_print": mean / MOVE50_D_POSE_T4,
        "d_pose_median": float(np.median(per_pair)),
        "d_pose_max": float(per_pair.max()),
        "top_12_share": float(per_pair[order[:12]].sum() / per_pair.sum()),
        "top_12_pairs": [int(p) for p in order[:12]],
        "per_pair_path": str(per_pair_path),
        "per_pair_sha256": sha256_file(per_pair_path),
        "codes_sha256": sha256_file(out_dir / "codes_move50.npy"),
        "elapsed_seconds": time.time() - started,
    }
    if args.control:
        control = np.load(args.control)
        if control.shape != per_pair.shape:
            raise Pd2Error(f"control vector has shape {control.shape}")
        report["control"] = {
            "path": str(args.control),
            "max_abs_difference": float(np.abs(control - per_pair).max()),
            "bit_identical": bool(np.array_equal(control, per_pair)),
        }
    (out_dir / "POSE_BASE_MOVE50.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "receipts"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: prereg -- the economics at move 50, priced by the MEASURED bit cost
# ----------------------------------------------------------------------------------


def cmd_prereg(args) -> int:
    base = np.load(args.base_pose)
    if base.shape != (N_PAIRS,):
        raise Pd2Error(f"base pose vector has shape {base.shape}")
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    edited = edited_pairs_at_move50()

    prices = []
    for bits in sorted({8.93, 16.0, PD1_MEASURED_BITS_PER_TOKEN, float(args.bits_per_token)}):
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
                16.0: "pd1's 20-pair admitted subset, MEASURED by its own real encode",
                PD1_MEASURED_BITS_PER_TOKEN: "pd1's 41-pair field, MEASURED by its own real encode",
            }.get(bits, "this arm's declared search price"),
        })

    rate_S = (float(args.bits_per_token) / 8.0) * S_PER_BYTE
    need = rate_S / pose_unit
    order = np.argsort(-base)
    population = [
        int(p) for p in order
        if int(p) not in FLOOR_PAIRS and float(base[int(p)]) > need
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "ddm_pd2_prereg.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "base_pose_path": str(args.base_pose),
        "base_pose_sha256": sha256_file(Path(args.base_pose)),
        "base_mean": base_mean,
        "economics": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_seg_cell_T4_carried": seg_cell,
            "S_per_unit_of_one_pairs_d_pose": pose_unit,
            "note": "DERIVED at move 50's base mean; EXPIRES at the next pointer move",
        },
        "prices": prices,
        "search_price_bits_per_token": float(args.bits_per_token),
        "hard_bound": (
            "a single-token edit cannot drive a pair's resolved d_pose below zero, so "
            "|credit| <= base and a pair whose base is under the required credit can never "
            "admit with one token at this price.  DERIVED, not measured."
        ),
        "population_size": len(population),
        "population_pairs_descending_d_pose": population,
        "floor_pairs_excluded": list(FLOOR_PAIRS),
        "move50_edited_pairs_researched_from_new_renders": edited,
        "pre_registered_band_net_dS": [-8e-05, -2e-05],
        "falsifiers": [
            "F1 the admitted set projects net > -2e-05 on the RESOLVED pose with real-encode rate",
            "F2 the composition realizes < 0.8 of the per-pair credit sum",
            "F3 any scored row's batch-1 base is outside the MEASURED absolute band "
            f"{BASE_BAND_ABS:.6e} from the n600 base",
            "F4 the control encode does not reproduce move 50's own 118,978 B token stream",
            "F5 an admitting pair's carrier re-solve returns the codes it started from",
            "F6 the candidate's own parse-back decodes a field that is not the admitted field",
            "F7 the shipped-mode decode's seg leg disagrees with the admission's prediction",
        ],
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("population_pairs_descending_d_pose",)}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: run -- bind, then dispatch a pd1 / sj1 argv inside THIS process
# ----------------------------------------------------------------------------------


def cmd_run(args) -> int:
    bind_move50(verify_raw=False)
    if args.module == "pd1":
        import ddm_pd1_pose_directed as target
    elif args.module == "joint":
        import ddm_sj1_joint_admission as target
    elif args.module == "pp1":
        import ddm_pp1_pose_actuation as target
    elif args.module == "tree":
        # pd1's candidate-tree builder calls sj1.patch_inflate_pins, which checks the tree's
        # CURRENT pins against the LIVE pointer; unbound it reports move 49 and refuses the
        # move-50 tree as "not the tree it claims to be".
        import ddm_pd1_candidate_tree as target
    else:  # pragma: no cover - argparse restricts the choices
        raise Pd2Error(f"unknown module {args.module}")
    argv = list(args.argv)
    if argv and argv[0] == "--":  # argparse.REMAINDER keeps the separator
        argv = argv[1:]
    if not argv:
        raise Pd2Error("run needs the target's own argv after --")
    return int(target.main(argv))


# ----------------------------------------------------------------------------------
# stage: carry -- best proposal per pair, ranked by resolved-pose credit per BIT
# ----------------------------------------------------------------------------------


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pair_bits(ledger_path: Path | None) -> dict[int, float]:
    """Per-pair MEASURED ideal-bit delta from an rlc1 ledger of a previous round.

    The ledger prices a PAIR, not a proposal: every proposal on one pair is charged that
    pair's measured delta.  Where two proposals on one pair really do cost different
    numbers of bits, this ranking cannot see the difference -- said plainly rather than
    dressed up as a per-proposal price.
    """
    if ledger_path is None:
        return {}
    ledger = json.loads(Path(ledger_path).read_text())
    return {int(row["pair"]): float(row["delta_bits"]) for row in ledger["rows"]}


def credit_per_bit(benefit: float, bits: float) -> float:
    """Rank key for benefit-per-bit that stays monotone when the edit COSTS no bits.

    ``benefit / bits`` inverts its own ordering the moment a measured per-pair ledger
    delta is zero or negative -- an edit that SHORTENS the stream while gaining pose is
    the best row there is, and dividing by its negative cost would rank it last.  Such a
    row dominates every priced row instead; a bit-costing row keeps the ratio.
    """
    if bits > 0.0:
        return benefit / bits
    return math.inf if benefit > 0.0 else -math.inf


def cmd_carry(args) -> int:
    base = np.load(args.base_pose)
    if base.shape != (N_PAIRS,):
        raise Pd2Error(f"base pose vector has shape {base.shape}")
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    default_bits = float(args.bits_per_token)
    pair_bits = load_pair_bits(Path(args.ledger) if args.ledger else None)

    rows = read_rows([Path(p) for p in args.rows])
    scored: dict[int, dict[str, Any]] = {}
    skipped_band = 0
    skipped_unrefined = 0
    for row in rows:
        pair = int(row["pair"])
        if pair in FLOOR_PAIRS:
            continue
        # A SCREEN row carries the seg cost only; it was never refined, so it has no
        # realized credit and cannot be ranked beside a refined one.  Skipped, and counted.
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
        })
        best = scored.get(pair)
        if best is None or entry["credit_S_per_bit"] > best["credit_S_per_bit"]:
            scored[pair] = entry

    keep = dict(scored) if args.carry_all else {
        p: e for p, e in scored.items() if e["dS_modelled_one_token"] < 0.0
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(keep.values(), key=lambda e: -e["credit_S_per_bit"])
    (out_dir / "CARRY_RANKING.json").write_text(json.dumps({
        "schema": "ddm_pd2_carry_ranking.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "rule": "best proposal per pair by resolved-pose credit PER BIT; pairs ordered by it",
        "ledger": str(args.ledger) if args.ledger else None,
        "declared_bits_per_token": default_bits,
        "pairs_with_measured_pair_bits": sum(
            1 for e in keep.values() if e["bits_source"] == "measured_pair_ledger"
        ),
        "rows_read": len(rows),
        "rows_outside_base_band": skipped_band,
        "rows_screen_only_not_refined": skipped_unrefined,
        "pairs_scored": len(scored),
        "pairs_carried": len(keep),
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
        "sum_credit_d_pose": float(sum(e["credit_d_pose"] for e in keep.values())),
        "sum_d_cells": int(sum(int(e["d_cells"]) for e in keep.values())),
        "carry_rows": str(selected),
    }, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    bind = sub.add_parser("bind", help="verify + re-bind move 50, write the receipt")
    bind.add_argument("--out", type=Path, required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.set_defaults(func=cmd_bind)

    base = sub.add_parser("base", help="move 50's own per-pair d_pose, n600")
    base.add_argument("--out-dir", type=Path, required=True)
    base.add_argument("--batch-size", type=int, default=8)
    base.add_argument("--threads", type=int, default=8)
    base.add_argument("--control", type=Path, default=None)
    base.add_argument("--verify-raw", action="store_true")
    base.set_defaults(func=cmd_base)

    pre = sub.add_parser("prereg", help="economics at move 50 + population + falsifiers")
    pre.add_argument("--base-pose", type=Path, required=True)
    pre.add_argument("--bits-per-token", type=float, default=PD1_MEASURED_BITS_PER_TOKEN)
    pre.add_argument("--out", type=Path, required=True)
    pre.set_defaults(func=cmd_prereg)

    run = sub.add_parser("run", help="bind, then dispatch a pd1 / sj1 argv in this process")
    run.add_argument("--module", choices=("pd1", "joint", "pp1", "tree"), required=True)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    carry = sub.add_parser("carry", help="best proposal per pair by credit PER BIT")
    carry.add_argument("--rows", nargs="+", required=True)
    carry.add_argument("--base-pose", type=Path, required=True)
    carry.add_argument("--ledger", type=Path, default=None)
    carry.add_argument("--bits-per-token", type=float, default=PD1_MEASURED_BITS_PER_TOKEN)
    carry.add_argument("--carry-all", action="store_true")
    carry.add_argument("--out-dir", type=Path, required=True)
    carry.set_defaults(func=cmd_carry)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
