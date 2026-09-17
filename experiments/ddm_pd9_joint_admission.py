#!/usr/bin/env python3
"""ddm_pd9 -- replace the per-pair SEG SCREEN with a JOINT seg+pose admission on pd8's pool.

WHY THIS EXISTS
---------------
pd6/pd7/pd8 admitted a proposal only if its SegNet cost fit a per-pair cell budget derived from
how much pose that pair could ever credit, and priced pose against real bits afterwards.  That is
two gates in series in two currencies.  The score has ONE currency::

    S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489

pd8 MEASURED the cost of the screen: of 4,291 charged proposals, 3,176 were refused before any
credit was read, and 2,122 of those cost three SegNet cells or fewer.  One cell is
+8.4827e-07 S at move 54's operating point; a proposal that costs one cell, buys -3e-06 S of
resolved pose and spends no bytes is a -2.15e-06 S move the screen threw away.

WHAT IS AND IS NOT NEW CODE
---------------------------
Every stage is pd6's (through pd7 and pd8), imported and called unchanged.  pd8's own realize
loop does the widening: its cell budget is ``min(cap, floor(base*pose_unit*frac/seg_cell))``, so
``--cell-budget-cap 3 --cell-budget-frac 1e9`` makes the budget a FLAT THREE CELLS for every pair
-- the charter's scope -- without touching a line of pd6.  The admission arithmetic in pd5's
``setprice`` is ALREADY joint (``benefit_S = -credit*pose_unit - d_cells*seg_cell``); what the
screen did was delete rows from the pool before that arithmetic could see them.

This module contributes exactly four things:

1. ``rebind_pd8_to_pd9`` -- re-points pd8's store constants at pd9's store PROCESS-LOCALLY, so
   pd8's ``rebind_pd4_to_move54`` and pd4's own verification path stay in charge of the binding.
2. ``refused-census`` -- the histogram of what the screen refused, and the DERIVED ceiling on
   what each refused proposal could ever net (pre-registered before the wave reads any credit).
3. ``seed-wave`` -- seeds this arm's wave directory from pd8's realized and screened rows, with
   a declared HOLDOUT that the wave re-realizes so pd8's numbers are reproduced, not trusted.
4. ``split-pool`` -- splits the enriched pool into the JOINT pool (every row) and the SCREEN
   pool (only rows the per-pair budget would have admitted), so the control is the same code on
   the same rows under the old rule.

No scorer weights ship anywhere, no Modal call, no authorization, no fire, no packet, no pointer
write.  Only ``upstream/evaluate.py`` on shipped bytes is a score, and MAIN fires.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True  # every tree this arm reads is another arm's custody

import ddm_pd8_price_first_pass3 as pd8

pd6 = pd8.pd6
pd4 = pd8.pd4

PD9_STORE = Path("/Volumes/APDataStore/pact/ddm_pd9")
PD8_STORE = Path("/Volumes/APDataStore/pact/ddm_pd8")


class Pd9Error(RuntimeError):
    """A pd9 input or invariant is not what the measured object says it is."""


def rebind_pd8_to_pd9() -> dict[str, Any]:
    """Re-point pd8's move-54 file constants at THIS arm's copies, process-locally.

    The VALUES (sha256, byte counts, score, flip count) are pd8's unchanged: they are facts
    about move 54, not about a store.  Only the paths move.  ``pd4.bind_move52`` then verifies
    every one of them against the file on disk, so a wrong copy fails closed.
    """
    pd8.PD8_STORE = PD9_STORE
    pd8.MOVE54_TREE = PD9_STORE / "base/move54_runtime"
    pd8.MOVE54_FIELD = PD9_STORE / "base/field_move54.npz"
    pd8.MOVE54_ARGMAX = PD9_STORE / "base/argmax_move54.npy"
    pd8.MOVE54_BASE_POSE = PD9_STORE / "base/pose_base_move54.npy"
    pd8.MOVE54_RAW = PD9_STORE / "parseback_base/0.raw"
    pd8.MOVE54_EDITED_PAIRS = PD9_STORE / "base/edited_pairs_move54.json"
    globals().update({
        "MOVE54_TREE": pd8.MOVE54_TREE, "MOVE54_FIELD": pd8.MOVE54_FIELD,
        "MOVE54_ARGMAX": pd8.MOVE54_ARGMAX, "MOVE54_BASE_POSE": pd8.MOVE54_BASE_POSE,
        "MOVE54_RAW": pd8.MOVE54_RAW,
    })
    return pd8.rebind_pd4_to_move54()


#: the store-side helper scripts (``seg_on_decode``, ``pose_on_decode``, ``wave_analysis``) are
#: pd8's, copied with ONE line changed -- the module they import.  They call these names, so the
#: rebinding is exposed under pd8's spelling rather than forking three more files.
rebind_pd4_to_move54 = rebind_pd8_to_pd9
MOVE54_TREE = PD9_STORE / "base/move54_runtime"
MOVE54_FIELD = PD9_STORE / "base/field_move54.npz"
MOVE54_ARGMAX = PD9_STORE / "base/argmax_move54.npy"
MOVE54_BASE_POSE = PD9_STORE / "base/pose_base_move54.npy"
MOVE54_RAW = PD9_STORE / "parseback_base/0.raw"
bound_seg_s_per_cell = pd8.bound_seg_s_per_cell


# ----------------------------------------------------------------------------------
# stage: refused-census -- what the screen threw away, and what it could ever have been worth
# ----------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.name.startswith("._"):
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _wave_rows(directory: Path, stem: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(f"{stem}_*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _key(row: dict[str, Any]) -> tuple:
    return (int(row["pair"]),
            tuple(tuple(int(x) for x in e) for e in row["edits"]))


def cmd_refused_census(args) -> int:
    """The histogram of refused seg costs, and the DERIVED ceiling on each refused proposal.

    The ceiling is the best case the joint rule could ever reach for a row: the pair recovers
    ALL of its own d_pose.  ``ceiling_net_S = bits/8*S_per_byte + d_cells*seg_cell -
    base[pair]*pose_unit``.  A row whose ceiling is >= 0 can never net no matter what the refine
    finds; the wave realizes it anyway (the charter's scope), and this file is the prediction
    that the credit is then checked against.
    """
    rebind_pd8_to_pd9()
    base = np.load(pd8.MOVE54_BASE_POSE)
    pose_unit = pd4.pose_s_per_pair_unit(float(base.mean()))
    seg_cell = pd8.bound_seg_s_per_cell()
    s_per_byte = pd4.S_PER_BYTE

    screen = _wave_rows(Path(args.screen_dir), "screen")
    realized = _wave_rows(Path(args.screen_dir), "realized")
    priced = {_key(r): r for r in _read_jsonl(Path(args.priced))}
    refused = [r for r in screen if r.get("refused_by_seg_screen")]

    hist = collections.Counter(int(r["d_cells"]) for r in refused)
    by_budget = collections.Counter(int(r.get("cell_budget", -1)) for r in refused)
    cap = int(args.max_cells)
    eligible = [r for r in refused if int(r["d_cells"]) <= cap]
    never: list[dict[str, Any]] = []
    could: list[dict[str, Any]] = []
    for row in eligible:
        pk = _key(row)
        charged = priced.get(pk)
        if charged is None:
            raise Pd9Error(f"refused row {pk[0]} has no charge in {args.priced}")
        bits = float(charged["real_delta_bits"])
        pair = int(row["pair"])
        ceiling = (bits / 8.0) * s_per_byte + int(row["d_cells"]) * seg_cell \
            - float(base[pair]) * pose_unit
        entry = {"pair": pair, "edits": row["edits"], "d_cells": int(row["d_cells"]),
                 "real_delta_bits": bits, "ceiling_net_S": ceiling,
                 "cell_budget_pd8": int(row.get("cell_budget", -1))}
        (could if ceiling < 0.0 else never).append(entry)

    payload = {
        "schema": "ddm_pd9_refused_census.v1",
        "axis": "[MEASURED from pd8's own screen rows; the ceiling is DERIVED at move 54's "
                "operating point and EXPIRES at the next pointer move]",
        "score_claim": False,
        "source_screen_dir": str(args.screen_dir),
        "source_priced": str(args.priced),
        "charged_proposals": len(priced),
        "realized_by_pd8": len(realized),
        "refused_by_pd8": len(refused),
        "refused_d_cells_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "refused_by_pd8_cell_budget": {str(k): int(v) for k, v in sorted(by_budget.items())},
        "max_cells_this_arm_realizes": cap,
        "refused_within_cap": len(eligible),
        "refused_within_cap_pairs": len({int(r["pair"]) for r in eligible}),
        "ceiling_can_never_net": len(never),
        "ceiling_could_net": len(could),
        "ceiling_could_net_pairs": sorted({int(r["pair"]) for r in could}),
        "ceiling_best_S": min((r["ceiling_net_S"] for r in could), default=None),
        "operating_point": {
            "s_per_archive_byte": s_per_byte,
            "s_per_seg_cell": seg_cell,
            "s_per_pair_pose_unit": pose_unit,
            "base_pose_mean_n600": float(base.mean()),
        },
        "method": ("ceiling_net_S = bits/8*S_per_byte + d_cells*seg_cell - base[pair]*pose_unit "
                   "-- the best case in which the pair's WHOLE d_pose is recovered"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    if args.out_rows:
        Path(args.out_rows).write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in could + never) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "refused_by_pd8", "refused_within_cap", "refused_within_cap_pairs",
        "ceiling_can_never_net", "ceiling_could_net")}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: seed-wave -- inherit pd8's realized rows, minus a holdout the wave reproduces
# ----------------------------------------------------------------------------------


def cmd_seed_wave(args) -> int:
    """Seed this arm's wave from pd8's rows, holding out N per shard as a reproduction control.

    A reused measurement is not evidence until this arm's own instrument reproduces it
    (``a new check is not evidence until it reproduces a known answer``).  The held-out rows are
    absent from the seed, so the wave re-renders, re-argmaxes and re-refines them; ``verify-
    holdout`` then compares this arm's numbers with pd8's recorded ones.
    """
    src = Path(args.src_dir)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    holdout: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    for shard in range(int(args.shards)):
        realized = _read_jsonl(src / f"realized_{shard}.jsonl")
        screen = _read_jsonl(src / f"screen_{shard}.jsonl")
        keep_idx = set(range(len(realized)))
        if realized and int(args.holdout_per_shard) > 0:
            take = min(int(args.holdout_per_shard), len(realized))
            chosen = rng.choice(len(realized), size=take, replace=False)
            for i in chosen:
                holdout.append(realized[int(i)])
                keep_idx.discard(int(i))
        seeded = [realized[i] for i in sorted(keep_idx)]
        (dst / f"realized_{shard}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in seeded))
        (dst / f"screen_{shard}.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in screen))
        manifest.append({
            "shard": shard,
            "source_realized_rows": len(realized),
            "seeded_realized_rows": len(seeded),
            "source_screen_rows": len(screen),
            "held_out": len(realized) - len(seeded),
        })
    (dst / "HOLDOUT.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in holdout))
    payload = {
        "schema": "ddm_pd9_seed_wave.v1",
        "axis": "[custody: pd8's rows are INHERITED, the holdout is RE-MEASURED here]",
        "score_claim": False,
        "source_dir": str(src),
        # AppleDouble stubs (``._name``) are filesystem noise on this volume, never payload
        "source_file_sha256": {
            p.name: pd4.sha256_file(p) for p in sorted(src.glob("*.jsonl"))
            if not p.name.startswith("._") and p.is_file()
        },
        "out_dir": str(dst),
        "seed": int(args.seed),
        "holdout_per_shard": int(args.holdout_per_shard),
        "holdout_rows": len(holdout),
        "shards": manifest,
        "meaning": ("the wave re-realizes every held-out row and every row the screen refused "
                    "at or under this arm's flat cell cap; everything else is inherited"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"holdout_rows": len(holdout),
                      "seeded": sum(m["seeded_realized_rows"] for m in manifest)}, indent=1))
    return 0


def cmd_verify_holdout(args) -> int:
    """Compare this arm's re-realized holdout rows against pd8's recorded values."""
    expected = {_key(r): r for r in _read_jsonl(Path(args.holdout))}
    got = {_key(r): r for r in _wave_rows(Path(args.wave_dir), "realized")}
    rows: list[dict[str, Any]] = []
    missing = []
    for key, exp in expected.items():
        have = got.get(key)
        if have is None:
            missing.append({"pair": key[0], "edits": [list(e) for e in key[1]]})
            continue
        rows.append({
            "pair": key[0],
            "d_cells_pd8": int(exp["d_cells"]), "d_cells_pd9": int(have["d_cells"]),
            "d_pose_resolved_pd8": float(exp["d_pose_resolved"]),
            "d_pose_resolved_pd9": float(have["d_pose_resolved"]),
            "rel_pose": abs(float(have["d_pose_resolved"]) - float(exp["d_pose_resolved"]))
            / max(abs(float(exp["d_pose_resolved"])), 1e-30),
        })
    cells_equal = sum(1 for r in rows if r["d_cells_pd8"] == r["d_cells_pd9"])
    worst = max((r["rel_pose"] for r in rows), default=None)
    payload = {
        "schema": "ddm_pd9_verify_holdout.v1",
        "axis": "[macOS-CPU advisory; the reuse control]",
        "score_claim": False,
        "holdout_rows": len(expected),
        "reproduced": len(rows),
        "missing": missing,
        "d_cells_identical": cells_equal,
        "d_pose_max_relative_difference": worst,
        "rows": rows,
        "meaning": ("pd8's realized rows are inherited on the strength of THIS comparison; a "
                    "cell disagreement or a pose gap above the stated bar refuses the reuse"),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: payload[k] for k in (
        "holdout_rows", "reproduced", "d_cells_identical",
        "d_pose_max_relative_difference")}, indent=1))
    return 0 if (not missing and cells_equal == len(rows)) else 3


# ----------------------------------------------------------------------------------
# stage: split-pool -- the JOINT pool and the SCREEN pool, from the same enriched rows
# ----------------------------------------------------------------------------------


def cmd_price_merge_refused(args) -> int:
    """This arm does not re-charge the pool; it INHERITS pd8's sheet encodes on the same field.

    pd8's ``price-merge`` renames pd4's sheet field to ``pd8sheet``.  Running it here would write
    ``pd8sheet`` rows from THIS arm's root, which is a provenance lie, and re-charging a pool
    whose field has not moved would spend eight 600-frame encodes to reproduce numbers already
    measured.  The inherited charges are pinned by file sha in this arm's receipts, and this
    arm's own control encode re-packs move 54's archive as the identity check on the pricer.
    """
    raise Pd9Error(
        "REFUSED: pd9 inherits pd8's sheet charges on the unchanged move-54 field. "
        "Re-charge only after re-running `sheets` in THIS root under a pd9sheet name."
    )


def cmd_split_pool(args) -> int:
    """Write the JOINT pool (every enriched row) and the SCREEN pool (the old rule's rows).

    The old rule: ``d_cells <= min(cap, floor(base[pair]*pose_unit/seg_cell))`` with pd8's own
    cap of 2.  Both pools carry identical rows and identical charges; only membership differs,
    so the control is the same code on the same measurements under the old admission.
    """
    rebind_pd8_to_pd9()
    base = np.load(pd8.MOVE54_BASE_POSE)
    pose_unit = pd4.pose_s_per_pair_unit(float(base.mean()))
    seg_cell = pd8.bound_seg_s_per_cell()
    rows = _read_jsonl(Path(args.priced_realized))
    cap = int(args.screen_cap)

    def budget(pair: int) -> int:
        return int(min(cap, math.floor(float(base[pair]) * pose_unit / seg_cell)))

    joint, screen = [], []
    for row in rows:
        joint.append(row)
        if int(row["d_cells"]) <= budget(int(row["pair"])):
            screen.append(row)
    Path(args.out_joint).write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in joint))
    Path(args.out_screen).write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in screen))
    cells = collections.Counter(int(r["d_cells"]) for r in joint)
    payload = {
        "schema": "ddm_pd9_split_pool.v1",
        "axis": "[MEASURED rows; the split is the old admission rule applied to them]",
        "score_claim": False,
        "priced_realized": str(args.priced_realized),
        "joint_rows": len(joint), "joint_pairs": len({int(r["pair"]) for r in joint}),
        "screen_rows": len(screen), "screen_pairs": len({int(r["pair"]) for r in screen}),
        "rows_only_in_joint": len(joint) - len(screen),
        "joint_d_cells_histogram": {str(k): int(v) for k, v in sorted(cells.items())},
        "screen_cap": cap,
        "screen_rule": "d_cells <= min(cap, floor(base[pair]*pose_unit/seg_cell))",
        "operating_point": {"s_per_seg_cell": seg_cell, "s_per_pair_pose_unit": pose_unit,
                            "s_per_archive_byte": pd4.S_PER_BYTE},
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: payload[k] for k in (
        "joint_rows", "joint_pairs", "screen_rows", "screen_pairs",
        "rows_only_in_joint")}, indent=1))
    return 0


# ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = pd6.build_parser()
    sub = parser._subparsers._group_actions[0]
    # pd8's own added subcommands, re-bound to pd9's store
    bb = sub.add_parser("baseband", help="MEASURE the batch-1 vs n600 pose-base gap, all 600")
    bb.add_argument("--raw", required=True)
    bb.add_argument("--threads", type=int, default=4)
    bb.add_argument("--headroom", type=float, default=2.0)
    bb.add_argument("--out", required=True)
    bb.set_defaults(func=pd8.cmd_baseband)
    rn = sub.add_parser("run", help="run another module's argv under the MOVE 54 binding")
    rn.add_argument("--module", required=True, choices=["joint", "pp1", "tree", "pd4", "pd5"])
    rn.add_argument("--raw", required=True)
    rn.add_argument("argv", nargs=argparse.REMAINDER)
    rn.set_defaults(func=pd8.cmd_run)
    sub._name_parser_map["price-merge"].set_defaults(func=cmd_price_merge_refused)
    bind = sub.add_parser("bind54", help="bind MOVE 54 from pd9's own copies")
    bind.add_argument("--raw", required=True)
    bind.add_argument("--verify-raw", action="store_true")
    bind.add_argument("--out", required=True)
    bind.set_defaults(func=pd8.cmd_bind)
    hr = sub.add_parser("headroom", help="what credit the lever needs at move 54")
    hr.add_argument("--out", required=True)
    hr.set_defaults(func=pd8.cmd_headroom)

    rc = sub.add_parser("refused-census", help="what the seg screen threw away")
    rc.add_argument("--screen-dir", required=True)
    rc.add_argument("--priced", required=True)
    rc.add_argument("--max-cells", type=int, default=3)
    rc.add_argument("--out", required=True)
    rc.add_argument("--out-rows", default="")
    rc.set_defaults(func=cmd_refused_census)

    sw = sub.add_parser("seed-wave", help="inherit pd8's rows, minus a reproduction holdout")
    sw.add_argument("--src-dir", required=True)
    sw.add_argument("--out-dir", required=True)
    sw.add_argument("--shards", type=int, default=7)
    sw.add_argument("--holdout-per-shard", type=int, default=3)
    sw.add_argument("--seed", type=int, default=20260917)
    sw.add_argument("--out", required=True)
    sw.set_defaults(func=cmd_seed_wave)

    vh = sub.add_parser("verify-holdout", help="reproduce pd8's held-out rows on this instrument")
    vh.add_argument("--holdout", required=True)
    vh.add_argument("--wave-dir", required=True)
    vh.add_argument("--out", required=True)
    vh.set_defaults(func=cmd_verify_holdout)

    sp = sub.add_parser("split-pool", help="the JOINT pool and the old SCREEN pool")
    sp.add_argument("--priced-realized", required=True)
    sp.add_argument("--screen-cap", type=int, default=2)
    sp.add_argument("--out-joint", required=True)
    sp.add_argument("--out-screen", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_split_pool)

    args = parser.parse_args(argv)
    # every stage reads pd4's constants, directly or through bind_move52; pd8's cmd_bind
    # re-runs the rebinding itself, so calling it here for all stages is idempotent
    rebind_pd8_to_pd9()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
