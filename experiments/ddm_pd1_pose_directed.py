#!/usr/bin/env python3
"""ddm_pd1: POSE-DIRECTED token pre-distortion over ALL pairs -- the re-solve credit as
the objective.

The actuator is ``ddm_pp1_pose_actuation.cmd_search_b`` (pose-saliency-ranked single-token
edits, each REALIZED through the shipped render, the frozen SegNet argmax and the carrier
re-solve).  pp1 ran it on the 12 floor pairs only.  This module does NOT re-implement the
actuator: it supplies the pair population, the economics that decide which pairs can pay
before a byte is spent, the per-pair best selection, and the admission inputs in the schema
``ddm_sj1_joint_admission.cmd_admit`` already reads.

Every number this module emits is recomputed from the search rows and the measured base
vector.  Nothing is modelled except where the label says MODELLED.

Stages
------
``prereg``    write the economics + falsifiers + strata BEFORE any search row exists.
``assemble``  search rows -> per-pair best proposal, gated, as admission inputs.
``verdict``   recompute the arm's numbers from the receipts.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # every pointer/sister tree this arm reads is custody

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

#: MOVE 49, the live row this arm stands on.  Asserted against the pointer state, never
#: retyped into a claim (`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]`).
POINTER_ARCHIVE_SHA256 = "73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214"
POINTER_ARCHIVE_BYTES = 179_153
POINTER_SCORE_T4 = 0.13632299781031237
POINTER_D_SEG_T4 = 0.00010287
POINTER_D_POSE_T4 = 4.55e-06
#: pass 7's own seg receipt for the shipped row: 12,127 flipped cells on the jg1 instrument.
INSTRUMENT_BASE_D_SEG = 0.00010280185275607639
#: pp1 measured these twelve as a FLOOR the frame_1 render cannot lower; excluded.
FLOOR_PAIRS = (88, 87, 73, 316, 89, 70, 63, 448, 64, 91, 66, 67)

S_PER_BYTE = 25.0 / SCORE_RATE_DENOMINATOR


class Pd1Error(RuntimeError):
    """A pd1 input or invariant is not what the measured object says it is."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seg_s_per_cell() -> float:
    """Score per flipped SegNet cell, carried onto T4 by the same-instrument ratio."""
    ratio_t4 = POINTER_D_SEG_T4 / INSTRUMENT_BASE_D_SEG
    return 100.0 * ratio_t4 / CELL_COUNT


def pose_s_per_pair_unit(base_mean: float) -> float:
    """d(score)/d(one pair's d_pose) at the operating point -- DERIVED, and it EXPIRES.

    ``sqrt(10 * mean)`` has derivative ``5 / sqrt(10 * mean)`` in the MEAN, and one pair
    contributes ``1/600`` of the mean.
    """
    return 5.0 / math.sqrt(10.0 * base_mean) / N_PAIRS


def load_base(path: Path) -> np.ndarray:
    vec = np.load(path)
    if vec.shape != (N_PAIRS,):
        raise Pd1Error(f"base pose vector has shape {vec.shape}, expected ({N_PAIRS},)")
    if not np.all(np.isfinite(vec)) or float(vec.min()) < 0.0:
        raise Pd1Error("base pose vector is not finite and non-negative")
    return vec.astype(np.float64)


def read_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ----------------------------------------------------------------------------------
# stage: prereg -- the economics, written before any search row exists
# ----------------------------------------------------------------------------------


def cmd_prereg(args) -> int:
    base = load_base(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()

    # The break-even a SINGLE-token edit must clear on its own pair, at three measured
    # token prices: pass 8's full-field 8.93 bits/token, pass 7's admitted-subset 5.01,
    # and a pessimistic 16 bits.  A pair whose whole d_pose is under the break-even
    # cannot admit at that price even if the edit erased its pose error entirely.
    prices = {}
    for name, bits in (("pass8_full_field", 8.93), ("pass7_subset", 5.0149), ("pessimistic", 16.0)):
        bytes_per_token = bits / 8.0
        need = bytes_per_token * S_PER_BYTE / pose_unit
        prices[name] = {
            "bits_per_token": bits,
            "bytes_per_token": bytes_per_token,
            "rate_S_per_token": bytes_per_token * S_PER_BYTE,
            "required_pair_credit_d_pose": need,
            "pairs_above_it": int((base > need).sum()),
            "pairs_above_it_excluding_floor": int(
                (base > need).sum() - sum(1 for p in FLOOR_PAIRS if base[p] > need)
            ),
        }

    order = np.argsort(-base)
    searchable = [int(p) for p in order if int(p) not in FLOOR_PAIRS]
    strata: list[dict[str, Any]] = []
    edges = [0, 20, 68, 148, 220, len(searchable)]
    rng = np.random.default_rng(args.seed)
    smoke: list[int] = []
    for index in range(len(edges) - 1):
        members = searchable[edges[index] : edges[index + 1]]
        take = min(args.smoke_per_stratum, len(members))
        picked = [int(p) for p in rng.choice(members, size=take, replace=False)]
        smoke.extend(picked)
        strata.append({
            "stratum": "ABCDE"[index],
            "rank_range_excluding_floor": [edges[index], edges[index + 1] - 1],
            "pairs": len(members),
            "d_pose_max": float(base[members[0]]),
            "d_pose_min": float(base[members[-1]]),
            "smoke_pairs": picked,
        })

    payload = {
        "schema": "ddm_pd1_prereg.v1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "promotable": False,
        "base": {
            "path": str(args.base_pose),
            "sha256": sha256_file(args.base_pose),
            "d_pose_mean": base_mean,
            "d_pose_median": float(np.median(base)),
            "d_pose_max": float(base.max()),
            "instrument_ratio_vs_t4_print": base_mean / POINTER_D_POSE_T4,
        },
        "economics_DERIVED_and_expiring": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_seg_cell": seg_cell,
            "S_per_unit_of_one_pair_d_pose": pose_unit,
            "admit_bar_net_dS": -2e-05,
            "container_break_lottery_sd_bytes": 34.8,
            "prices": prices,
        },
        "floor_pairs_excluded": list(FLOOR_PAIRS),
        "strata": strata,
        "smoke_pairs": sorted(smoke),
        "priors_that_disagree": {
            "pp1_floor_law": (
                "pp1 measured the best seg-neutral resolved fall at 0.25 % (token) / 0.45 % "
                "(frame_embed) on the 12 hard pairs, and its floor probe found any frame_1 "
                "render lands within +-3 % of the same resolved value. If that law holds on "
                "non-floor pairs, the per-pair credit is ~1 % of the pair's own d_pose and "
                "only pairs above ~6e-05 could pay -- all of which are floor pairs, so the "
                "admitted set is EMPTY and the family closes."
            ),
            "rp1_move42_credit": (
                "rp1 round 2's Lagrange-selected 160 pairs resolved to a mean d_pose BELOW "
                "base: 600*(4.64947689e-06 - 4.88709198e-06) = -1.4257e-04 over 160 pairs, "
                "i.e. -8.9e-07 of d_pose PER KEPT PAIR -- above the 6.0e-07 single-token "
                "break-even. If a pose-DIRECTED single edit reproduces even half of that on "
                "the pairs it is aimed at, 50-150 pairs admit and the sum is -5e-05..-1.5e-04."
            ),
            "which_decides": (
                "the stratified smoke: the best resolved fractional credit per pair, measured "
                "on non-floor pairs across the whole d_pose range. The two priors differ by "
                "more than 10x, so the smoke is decisive rather than confirmatory."
            ),
        },
        "falsifiers": {
            "F_charter": (
                "the Lagrange-admitted set projects net dS > -2e-05 on the RESOLVED pose with "
                "real-encode rate -> the family closes on this field at FORMULATION scope"
            ),
            "F_composition": "the composed object realizes < 0.8 of the per-pair sum",
            "F_base_gate": (
                "a search row's batch-1 per-pair base must agree with the n600 batch-8 base "
                "inside the MEASURED batch-shape band for the population TREATED (pp1 sec.10: "
                "the band is ABSOLUTE, not relative, and this arm treats pairs spanning 3,000x)"
            ),
            "F_control_encode": (
                "the RLC1 pricer must repack move 49's own field to 179,153 B sha 73e41a66... "
                "byte-identically, twice, before any candidate price is believed"
            ),
            "F_base_reproduction": (
                "the reproduced per-pair base must agree with pass 8's F7 value "
                "4.543568679770593e-06 and with pp1's own independently measured vector"
            ),
        },
        "stop_rules_on_the_smoke": {
            "A_stop": "implied full-search admitted yield < 0.5x the bar -> close, do not burn the search",
            "B_marginal": "0.5x .. 1.5x the bar -> run the full search, expect a marginal verdict",
            "C_run": "> 1.5x the bar -> run the full search",
            "how_implied_yield_is_computed": (
                "per sampled pair take the BEST realized proposal, score it as "
                "dS = seg_cells*S_per_seg_cell + credit*S_per_unit_pair_pose + 1*rate_S_per_token "
                "at the pass8_full_field price, keep it if dS < 0, then scale each stratum's "
                "kept sum by (stratum pairs / sampled pairs). MODELLED on rate; the real "
                "encode decides."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({
        "out": str(args.out),
        "base_mean": base_mean,
        "S_per_unit_of_one_pair_d_pose": pose_unit,
        "S_per_seg_cell": seg_cell,
        "prices": {k: (v["required_pair_credit_d_pose"], v["pairs_above_it_excluding_floor"])
                   for k, v in prices.items()},
        "smoke_pairs": sorted(smoke),
    }, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: assemble -- per-pair best proposal, gated, as admission inputs
# ----------------------------------------------------------------------------------


def _best_per_pair(
    rows: Sequence[dict[str, Any]],
    base: np.ndarray,
    *,
    band_abs: float,
    seg_cell: float,
    pose_unit: float,
    rate_S_per_token: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Pick, per pair, the proposal with the best MODELLED dS -- seg + pose + one token.

    The selection is made on the same arithmetic the admission will use, so a pair that
    cannot pay is never carried into the candidate field.  The pose term is carried as a
    DELTA on the n600 base, never as the batch-1 VALUE: pp1 sec.10 measured that batch
    shape moves the pose vector, and the delta form cancels that offset exactly.
    """
    rejected_band = 0
    considered = 0
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        pair = int(row["pair"])
        gap = abs(float(row["d_pose_base"]) - float(base[pair]))
        if gap > band_abs:
            rejected_band += 1
            continue
        considered += 1
        credit = float(row["d_pose_resolved"]) - float(row["d_pose_base"])
        d_cells = int(row["d_cells"])
        modelled = d_cells * seg_cell + credit * pose_unit + rate_S_per_token
        entry = dict(row)
        entry["base_gap_abs_vs_n600"] = gap
        entry["credit_d_pose"] = credit
        entry["fraction_of_pair_d_pose"] = credit / float(base[pair]) if base[pair] > 0 else math.nan
        entry["dS_modelled_one_token"] = modelled
        previous = best.get(pair)
        if previous is None or modelled < previous["dS_modelled_one_token"]:
            best[pair] = entry
    stats = {
        "rows_total": len(rows),
        "rows_inside_base_band": considered,
        "rows_rejected_by_base_band": rejected_band,
        "base_band_abs": band_abs,
        "pairs_with_a_row": len(best),
    }
    return best, stats


def cmd_assemble(args) -> int:
    base = load_base(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    rate_S_per_token = (args.bits_per_token / 8.0) * S_PER_BYTE
    rows = read_rows([Path(p) for p in args.rows])
    best, stats = _best_per_pair(
        rows, base, band_abs=args.base_band_abs, seg_cell=seg_cell,
        pose_unit=pose_unit, rate_S_per_token=rate_S_per_token,
    )
    for pair in FLOOR_PAIRS:
        if pair in best and not args.keep_floor_pairs:
            del best[pair]

    keep = {p: e for p, e in best.items() if e["dS_modelled_one_token"] < 0.0}
    if args.carry_all:
        keep = dict(best)

    source = np.load(args.carry_field)
    planes: dict[int, np.ndarray] = {int(k): source[k] for k in source.files}
    if len(planes) != N_PAIRS:
        raise Pd1Error(f"carry field has {len(planes)} planes, expected {N_PAIRS}")
    stale = base.copy()
    resolved = base.copy()
    pass_rows: list[dict[str, Any]] = []
    for pair, entry in sorted(keep.items()):
        plane = planes[pair].copy()
        row_index, col_index = (int(v) for v in entry["cell"])
        if int(plane[row_index, col_index]) != int(entry["old"]):
            raise Pd1Error(
                f"pair {pair} cell ({row_index},{col_index}) holds "
                f"{int(plane[row_index, col_index])} in the carry field but the search row "
                f"says {int(entry['old'])}; the search and the base disagree"
            )
        plane[row_index, col_index] = int(entry["new"])
        planes[pair] = plane
        # DELTA form: the n600 base plus the edit's own measured effect.
        stale[pair] = base[pair] + (float(entry["d_pose_stale"]) - float(entry["d_pose_base"]))
        resolved[pair] = base[pair] + float(entry["credit_d_pose"])
        pass_rows.append({
            "pair": pair,
            "flips_repaired": -int(entry["d_cells"]),
            "tokens_changed": 1,
            "cell": [row_index, col_index],
            "old": int(entry["old"]),
            "new": int(entry["new"]),
            "d_cells": int(entry["d_cells"]),
            "credit_d_pose": float(entry["credit_d_pose"]),
            "fraction_of_pair_d_pose": float(entry["fraction_of_pair_d_pose"]),
            "d_pose_base_n600": float(base[pair]),
            "d_pose_base_batch1": float(entry["d_pose_base"]),
            "d_pose_stale_batch1": float(entry["d_pose_stale"]),
            "d_pose_resolved_batch1": float(entry["d_pose_resolved"]),
            "base_gap_abs_vs_n600": float(entry["base_gap_abs_vs_n600"]),
            "carrier_codes": [int(c) for c in entry["carrier_codes"]],
            "dS_modelled_one_token": float(entry["dS_modelled_one_token"]),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    field = out_dir / "field_candidate.npz"
    np.savez_compressed(field, **{str(p): plane for p, plane in sorted(planes.items())})
    written = np.load(field)
    for pair in range(N_PAIRS):
        if not np.array_equal(written[str(pair)], planes[pair]):
            raise Pd1Error(f"candidate field does not round-trip plane {pair}")
    for pair in range(N_PAIRS):
        if pair not in keep and not np.array_equal(written[str(pair)], source[str(pair)]):
            raise Pd1Error(
                f"pair {pair} is not edited but its plane differs from the carry field; "
                "that would revert banked edits"
            )
    rows_path = out_dir / "pass_rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in pass_rows))
    np.save(out_dir / "pose_stale.npy", stale)
    np.save(out_dir / "pose_resolved.npy", resolved)

    codes_source = np.load(args.codes_base).astype(np.int32) if args.codes_base else None
    if codes_source is not None:
        if codes_source.shape[0] != N_PAIRS:
            raise Pd1Error(f"base carrier codes have shape {codes_source.shape}")
        codes = codes_source.copy()
        for entry in pass_rows:
            codes[entry["pair"]] = np.asarray(entry["carrier_codes"], dtype=np.int32)
        np.save(out_dir / "codes_resolved.npy", codes)

    credits = np.array([r["credit_d_pose"] for r in pass_rows], dtype=np.float64)
    cells = np.array([r["d_cells"] for r in pass_rows], dtype=np.int64)
    summary = {
        "schema": "ddm_pd1_assemble.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "row_stats": stats,
        "pairs_with_any_proposal": len(best),
        "pairs_carried": len(keep),
        "floor_pairs_kept": bool(args.keep_floor_pairs),
        "selection_rule": "best MODELLED dS per pair (seg + pose delta + one token), kept if < 0"
        if not args.carry_all else "every pair's best proposal carried, admission decides",
        "bits_per_token_modelled": args.bits_per_token,
        "sum_credit_d_pose": float(credits.sum()) if credits.size else 0.0,
        "sum_d_cells": int(cells.sum()) if cells.size else 0,
        "sum_dS_modelled": float(sum(r["dS_modelled_one_token"] for r in pass_rows)),
        "field_candidate": str(field),
        "field_candidate_sha256": sha256_file(field),
        "pass_rows": str(rows_path),
        "pose_stale": str(out_dir / "pose_stale.npy"),
        "pose_resolved": str(out_dir / "pose_resolved.npy"),
        "base_pose": str(args.base_pose),
        "base_pose_sha256": sha256_file(args.base_pose),
    }
    (out_dir / "ASSEMBLE.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "row_stats"}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: yield -- read a search's rows and report the credit/cost histogram
# ----------------------------------------------------------------------------------


def cmd_yield(args) -> int:
    base = load_base(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    rate_S_per_token = (args.bits_per_token / 8.0) * S_PER_BYTE
    rows = read_rows([Path(p) for p in args.rows])
    best, stats = _best_per_pair(
        rows, base, band_abs=args.base_band_abs, seg_cell=seg_cell,
        pose_unit=pose_unit, rate_S_per_token=rate_S_per_token,
    )
    per_pair = []
    for pair, entry in sorted(best.items()):
        per_pair.append({
            "pair": pair,
            "d_pose_base_n600": float(base[pair]),
            "best_credit_d_pose": float(entry["credit_d_pose"]),
            "best_fraction": float(entry["fraction_of_pair_d_pose"]),
            "d_cells": int(entry["d_cells"]),
            "dS_modelled": float(entry["dS_modelled_one_token"]),
            "admits_modelled": bool(entry["dS_modelled_one_token"] < 0.0),
            "floor_pair": pair in FLOOR_PAIRS,
        })
    fractions = np.array([r["best_fraction"] for r in per_pair], dtype=np.float64)
    admits = [r for r in per_pair if r["admits_modelled"]]
    # The fraction histogram, in the units pp1's floor law is stated in.
    edges = [-np.inf, -0.20, -0.10, -0.05, -0.02, -0.01, -0.005, 0.0, np.inf]
    histogram = []
    for index in range(len(edges) - 1):
        count = int(((fractions >= edges[index]) & (fractions < edges[index + 1])).sum())
        histogram.append({"from": float(edges[index]), "to": float(edges[index + 1]), "pairs": count})
    payload = {
        "schema": "ddm_pd1_yield.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "row_stats": stats,
        "pairs": len(per_pair),
        "pairs_admitting_modelled": len(admits),
        "sum_dS_modelled_over_admitting": float(sum(r["dS_modelled"] for r in admits)),
        "best_fraction_min": float(fractions.min()) if fractions.size else math.nan,
        "best_fraction_median": float(np.median(fractions)) if fractions.size else math.nan,
        "fraction_histogram": histogram,
        "bits_per_token_modelled": args.bits_per_token,
        "per_pair": per_pair,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "per_pair"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# stage: search -- screen on the BINDING leg first, refine only what can pay
# ----------------------------------------------------------------------------------


def interior_mask(argmax_plane: np.ndarray, radius: int) -> np.ndarray:
    """True where the pair's own argmax is uniform over a (2r+1)^2 window.

    The seg leg is the binding one: pp1 measured 99.4 % of the frame_embed lattice and
    11 of 12 token proposals costing cells, and one cell (8.48e-07 S) already exceeds the
    pose credit a whole pair usually has to give.  A cell deep inside a uniform argmax
    region is the one whose render perturbation has no boundary to move.  This is a
    PROPOSAL FILTER only: every survivor is still screened by the frozen argmax, and the
    filter's own value is measured by reporting the seg-neutral rate with and without it.
    """
    if radius <= 0:
        return np.ones(argmax_plane.shape, dtype=bool)
    plane = argmax_plane
    uniform = np.ones(plane.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(plane, dy, axis=0), dx, axis=1)
            uniform &= shifted == plane
    uniform[:radius, :] = False
    uniform[-radius:, :] = False
    uniform[:, :radius] = False
    uniform[:, -radius:] = False
    return uniform


def cmd_search(args) -> int:
    """Pose-ranked token edits: SCREEN every proposal on seg, REFINE only what can pay.

    The mechanism is pp1's actuator B unchanged -- the shipped render, the frozen SegNet
    argmax, ``jg5.refine_pair`` on the moved render.  What changes is the ORDERING and the
    proposal ranking, both of which pp1 already validated for actuator A ("an ORDERING
    change, not a mechanism change: every number the admission uses is still realized").
    """
    if str(Path(__file__).resolve().parents[0]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pp1_pose_actuation as pp1

    pp1.set_threads(args.threads)
    started = time.time()
    pairs = [int(p) for p in args.pairs.split(",")]
    base = load_base(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    rate_S_per_token = (args.bits_per_token / 8.0) * S_PER_BYTE
    body = pp1.load_body(with_raw=True, with_segnet=True)
    raw = body.raw
    base_inst = pp1.build_pose_instrument(raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(base_mean)
    shipped_argmax = np.load(pp1.LIVE_ARGMAX, mmap_mode="r")
    deltas = [int(d) for d in args.deltas.split(",")]
    # THE TOKEN ALPHABET IS THE RENDERER'S OWN, READ FROM IT.
    # pp1's search_b guards a proposal with ``0 <= new <= 255``, which is the uint8 storage
    # domain and NOT the symbol domain: the shipped ``token_embed`` has exactly 5 rows, so
    # a cell holding 4 with delta +1 indexes row 5 and torch raises IndexError inside the
    # embedding.  pp1 never met it because both its pairs' salient cells held 0 or 2.
    # Measured here on shard 7.  The cure reads ``num_embeddings`` rather than hardcoding a
    # class count, so a renderer with a different alphabet is handled rather than assumed.
    token_classes = int(body.semantic.token_embed.num_embeddings)
    if token_classes < 2:
        raise Pd1Error(f"token_embed has {token_classes} rows; there is no alternative symbol")
    field_max = int(max(int(body.tokens[p].max()) for p in pairs))
    if field_max >= token_classes:
        raise Pd1Error(
            f"the token field holds symbol {field_max} but token_embed has only "
            f"{token_classes} rows; the field and the renderer disagree"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"search_b_{args.shard_index}.jsonl"
    screen_path = out_dir / f"screen_{args.shard_index}.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.exists():
        done = {int(json.loads(line)["pair"]) for line in rows_path.read_text().splitlines() if line.strip()}
    handle = rows_path.open("a")
    screen_handle = screen_path.open("a")
    counters = {"screened": 0, "seg_neutral": 0, "refined": 0, "pairs": 0}
    for pair in pairs:
        if pair in done and args.resume:
            continue
        fe1.restore_pair_codes(body, pair)
        base_plane = body.tokens[pair].copy()
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        gap = abs(d_pose_base - float(base[pair]))
        if gap > args.base_band_abs:
            raise Pd1Error(
                f"pair {pair} batch-1 base {d_pose_base:.12e} is {gap:.3e} from the n600 "
                f"base {float(base[pair]):.12e}, outside the MEASURED band {args.base_band_abs:.3e}"
            )
        saliency = pp1.pose_saliency_on_token_grid(
            base_inst, pair, np.asarray(raw[2 * pair + 1])
        )
        interior = interior_mask(np.asarray(shipped_argmax[pair]), args.interior_radius)
        ranked = saliency.copy()
        ranked[~interior] = -np.inf
        flat = np.argsort(-ranked.ravel())[: args.cells]
        cells = [
            (int(i // EVAL_W), int(i % EVAL_W))
            for i in flat
            if np.isfinite(ranked.ravel()[i])
        ]
        print(
            f"pair {pair}: base flips {base_flips}, d_pose {d_pose_base:.4e}, "
            f"interior cells {int(interior.sum())}/{EVAL_H * EVAL_W}, "
            f"{len(cells)} proposals ranked",
            flush=True,
        )
        screened: list[dict[str, Any]] = []
        for row_index, col_index in cells:
            old = int(base_plane[row_index, col_index])
            for delta in deltas:
                new = old + delta
                if not 0 <= new < token_classes:
                    continue
                body.tokens[pair][row_index, col_index] = new
                frame = fe1.render_pair(body, pair)
                moved_flips = fe1.flips_pair(
                    jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
                )
                body.tokens[pair][row_index, col_index] = old
                counters["screened"] += 1
                entry = {
                    "pair": pair, "cell": [row_index, col_index], "old": old, "new": new,
                    "base_flips": base_flips, "flips": moved_flips,
                    "d_cells": moved_flips - base_flips,
                    "saliency": float(saliency[row_index, col_index]),
                    "interior": True,
                }
                screen_handle.write(json.dumps(entry) + "\n")
                if entry["d_cells"] <= args.max_cells:
                    counters["seg_neutral"] += int(entry["d_cells"] == 0)
                    screened.append(entry)
        screen_handle.flush()
        # Refine the cheapest-on-seg first: at one token the rate fee is fixed, so the
        # seg cost is the whole difference between candidates before the pose is known.
        screened.sort(key=lambda e: (e["d_cells"], -e["saliency"]))
        for entry in screened[: args.refine]:
            row_index, col_index = entry["cell"]
            body.tokens[pair][row_index, col_index] = entry["new"]
            frame = fe1.render_pair(body, pair)
            body.tokens[pair][row_index, col_index] = entry["old"]
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
                "dS_modelled_one_token": entry["d_cells"] * seg_cell + credit * pose_unit + rate_S_per_token,
            })
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"  pair {pair} ({row_index},{col_index}) {entry['old']}->{entry['new']}: "
                f"cells {entry['d_cells']:+d}, resolved {resolved:.4e} "
                f"({row['resolved_over_base']:.4f}x), dS {row['dS_modelled_one_token']:+.3e}",
                flush=True,
            )
        counters["pairs"] += 1
    handle.close()
    screen_handle.close()
    (out_dir / f"SEARCH_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pd1_search.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "receipts": body.receipts,
        "pairs": pairs,
        "cells_ranked_per_pair": args.cells,
        "deltas": deltas,
        "interior_radius": args.interior_radius,
        "max_cells_to_refine": args.max_cells,
        "refine_per_pair": args.refine,
        "bits_per_token_modelled": args.bits_per_token,
        "base_band_abs": args.base_band_abs,
        "counters": counters,
        "proposal": (
            "pose saliency |d(d_pose)/d(frame_1)| area-pooled to the token grid, masked to "
            "cells whose shipped argmax is uniform over a (2r+1)^2 window; screened on the "
            "frozen argmax, refined only where the seg cost can be paid"
        ),
        "base_pose_mean": base_mean,
        "rows_path": str(rows_path),
        "screen_path": str(screen_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path), "counters": counters}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    search = sub.add_parser("search", help="screen on seg, refine only what can pay")
    search.add_argument("--pairs", required=True)
    search.add_argument("--base-pose", type=Path, required=True)
    search.add_argument("--base-band-abs", type=float, required=True)
    search.add_argument("--cells", type=int, default=12)
    search.add_argument("--deltas", default="-1,1")
    search.add_argument("--interior-radius", type=int, default=2)
    search.add_argument("--max-cells", type=int, default=0)
    search.add_argument("--refine", type=int, default=2)
    search.add_argument("--bits-per-token", type=float, default=8.93)
    search.add_argument("--out-dir", type=Path, required=True)
    search.add_argument("--shard-index", type=int, default=0)
    search.add_argument("--outer-rounds", type=int, default=40)
    search.add_argument("--max-gn-iterations", type=int, default=400)
    search.add_argument("--threads", type=int, default=2)
    search.add_argument("--resume", action="store_true")
    search.set_defaults(func=cmd_search)

    pre = sub.add_parser("prereg", help="economics + falsifiers + strata, before any row")
    pre.add_argument("--base-pose", type=Path, required=True)
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--seed", type=int, default=20260913)
    pre.add_argument("--smoke-per-stratum", type=int, default=4)
    pre.set_defaults(func=cmd_prereg)

    asm = sub.add_parser("assemble", help="search rows -> admission inputs")
    asm.add_argument("--rows", nargs="+", required=True)
    asm.add_argument("--base-pose", type=Path, required=True)
    asm.add_argument("--carry-field", type=Path, required=True)
    asm.add_argument("--codes-base", type=Path, default=None)
    asm.add_argument("--out-dir", type=Path, required=True)
    asm.add_argument("--base-band-abs", type=float, required=True)
    asm.add_argument("--bits-per-token", type=float, default=8.93)
    asm.add_argument("--keep-floor-pairs", action="store_true")
    asm.add_argument("--carry-all", action="store_true")
    asm.set_defaults(func=cmd_assemble)

    yld = sub.add_parser("yield", help="credit/cost histogram from search rows")
    yld.add_argument("--rows", nargs="+", required=True)
    yld.add_argument("--base-pose", type=Path, required=True)
    yld.add_argument("--out", type=Path, required=True)
    yld.add_argument("--base-band-abs", type=float, required=True)
    yld.add_argument("--bits-per-token", type=float, default=8.93)
    yld.set_defaults(func=cmd_yield)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
