#!/usr/bin/env python3
"""ddm_pd5 -- MULTI-TOKEN joint proposals on the move-52 field, priced as a SET.

WHY THIS EXISTS
---------------
pd4 measured the price lever and came 9.2 % short: its admitted 18-pair / 24-token subset
priced at **12.000 bits per changed token** by its own real encode where the same pool needs
**10.891** to clear the -2e-05 bar.  It also measured *why* its ledger under-charged the
chosen subset by 36.3 %: the per-bit rule takes the argmin of a few noisy real prices per
pair, and the argmin of noisy draws is biased low.  Two mechanisms were left unpriced and
they are this arm's rung:

1. **Longer runs.**  MEASURED on pd4's own 122 matched cluster/anchor rows (this arm's
   read-only re-analysis of ``sheets/priced_rows.jsonl``; the receipt is
   ``PREREGISTRATION.json.prior_from_pd4_reanalysis``): the MARGINAL cost of the second token
   is **6.213 bits (median)** against **15.420** for an isolated one -- the coder's context
   model really does discount a neighbour.  If that discount survives to tokens 3 and 4 the
   price falls where pd4 needs it to.  What the same rows also say, and what this arm
   pre-registers as the thing that can kill the family, is that the second token's marginal
   credit is **+3.504e-08 d_pose at the median -- i.e. it makes the pair WORSE** -- and it
   improves the pair on only **41.0 %** of extensions, so a longer run buys cheap bits and
   usually spends pose to get them.
2. **Set pricing.**  Rank on the ledger, then re-encode the SELECTED SET as one field and
   re-run the sweep on the set's own real bits, iterating to a fixed point.  That is the
   cure pd4 named for its own 36.3 % selection bias.

WHAT THIS MODULE ADDS, AND WHAT IT REUSES
-----------------------------------------
The search here is pd4's ``cluster-search`` with ONE declared generalisation: the anchor is a
SEED RUN of any length rather than a single token, so one code path grows 1->2, 2->3 and
3->4 and every row is realized JOINTLY (one render, one frozen-argmax, one carrier re-solve)
and credited against the pair's OWN base, never summed from its parts
([[composition_of_disjoint_token_edits_is_subadditive_on_seg_by_pair_overlap_20260910]]).
Everything else is pd4's or sj1's, unchanged and imported: the move-52 binding
(``pd4.bind_move52``), the economics, the pool gates, the price sheets, the real-encode
pricer (``ddm_sj1_rlc1_price``) and the joint admission (``ddm_sj1_joint_admission``).

``load_pool_move52`` is pd4's ``load_pool`` with the "this arm's own store" marker lifted
from the hard-coded ``ddm_pd4`` to a parameter, because rows pd4 measured on move 52's own
renders are current for THIS arm too and the hard-coded marker would silently drop pd4's
re-search rows on the 26 pairs move 52 edited.  That is the only semantic change.

No scorer weights ship anywhere, no Modal call, no authorization, no fire, no packet, no
pointer write.  Only ``upstream/evaluate.py`` on shipped bytes is a score, and MAIN fires.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

import ddm_pd4_pose_directed_pass4 as pd4  # the path above must be set first

pd3 = pd4.pd3
N_PAIRS = pd4.N_PAIRS
EVAL_H, EVAL_W = pd4.EVAL_H, pd4.EVAL_W
CELL_COUNT = pd4.CELL_COUNT
FLOOR_PAIRS = pd4.FLOOR_PAIRS
BASE_BAND_ABS = pd4.BASE_BAND_ABS
S_PER_BYTE = pd4.S_PER_BYTE

PD4_STORE = Path("/Volumes/APDataStore/pact/ddm_pd4")
PD5_STORE = Path("/Volumes/APDataStore/pact/ddm_pd5")

#: stores whose rows were measured on MOVE 52's own renders and are therefore current here.
FRESH_STORES = ("ddm_pd4", "ddm_pd5")

row_edits = pd4.row_edits
#: a proposal's IDENTITY: its pair plus the set of (cell, new) moves it makes.
proposal_identity = pd4.proposal_key
score_proposals = pd4.score_proposals
seg_s_per_cell = pd4.seg_s_per_cell
pose_s_per_pair_unit = pd4.pose_s_per_pair_unit
edited_pairs_at_move52 = pd4.edited_pairs_at_move52
sha256_file = pd4.sha256_file


class Pd5Error(RuntimeError):
    """A pd5 input or invariant is not what the measured object says it is."""


# ----------------------------------------------------------------------------------
# the pool, with the "current for move 52" marker lifted to a parameter
# ----------------------------------------------------------------------------------


def load_pool_move52(
    row_paths: list[Path],
    base: np.ndarray,
    *,
    band_abs: float,
    exclude_pairs: set[int],
    fresh_stores: tuple[str, ...] = FRESH_STORES,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    """pd4's ``load_pool`` with the fresh-store marker as a parameter.

    pd4 hard-coded ``"ddm_pd4" in store.parts`` to mean "measured on move 52's renders".  For
    this arm both pd4's store and its own are current, and a row pd4 re-searched on one of
    the 26 pairs move 52 edited must survive; the base band alone does not carry it (pd4
    MEASURED the band as a 96.2 %-effective proxy -- pair 569 slips through at 9.311e-10).
    """
    stats = {"rows_read": 0, "unrefined": 0, "excluded_stale_row_on_edited_pair": 0,
             "outside_band": 0, "floor_pair": 0, "duplicates": 0, "fresh_rows": 0}
    rows: list[dict[str, Any]] = []
    for path in row_paths:
        store = Path(path).resolve()
        fresh = any(marker in store.parts for marker in fresh_stores)
        source = next((part for part in store.parts if part.startswith("ddm_")), "unknown")
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("row_source", source)
            row["row_is_current_for_move52"] = fresh
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
        if pair in exclude_pairs and not row["row_is_current_for_move52"]:
            stats["excluded_stale_row_on_edited_pair"] += 1
            continue
        if row["row_is_current_for_move52"]:
            stats["fresh_rows"] += 1
        gap = abs(float(row["d_pose_base"]) - float(base[pair]))
        if gap > band_abs:
            stats["outside_band"] += 1
            continue
        entry = dict(row)
        entry["base_gap_abs_vs_n600"] = gap
        entry.setdefault("tokens_changed", len(row_edits(row)))
        key = proposal_identity(entry)
        slot = pool.setdefault(pair, {})
        prior = slot.get(key)
        if prior is None:
            slot[key] = entry
        else:
            stats["duplicates"] += 1
            if float(entry["credit_d_pose"]) < float(prior["credit_d_pose"]):
                slot[key] = entry
    return {p: list(v.values()) for p, v in pool.items()}, stats


def _row_files(spec: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in spec:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        elif any(ch in item for ch in "*?["):
            out.extend(sorted(Path(item).parent.glob(Path(item).name)))
        else:
            out.append(path)
    # ``._name`` entries are ExFAT AppleDouble stubs, not rows: they glob, they are not
    # UTF-8, and reading one aborts the stage
    # ([[landing_an_arm_bundle_guard_on_file_count_and_skip_exfat_dot_underscore_stubs_20260910]]).
    kept = [p for p in out if p.exists() and p.stat().st_size > 0
            and not p.name.startswith("._")
            and "screen" not in p.name and "priced" not in p.name]
    if not kept:
        raise Pd5Error(f"no row files matched {spec}")
    return kept


# ----------------------------------------------------------------------------------
# stage: bind -- re-point every stale module at MOVE 52 and re-verify the inherited decode
# ----------------------------------------------------------------------------------


def cmd_bind(args) -> int:
    receipts = pd4.bind_move52(verify_raw=bool(args.verify_raw), raw=args.raw)
    payload = {
        "schema": "ddm_pd5_bind.v1",
        "axis": "[macOS-CPU advisory / binding receipt, no score]",
        "score_claim": False,
        "pd4_receipts": receipts,
        "decode_provenance": (
            "INHERITED: this arm did not re-decode move 52's archive.  pd3 and pd4 each "
            "parsed the SAME shipped bytes back cold on their own copies of the runtime "
            "tree and produced a bit-identical 0.raw (pd4 MEASURED max abs per-pair pose "
            "difference 0.000e+00 against pd3's vector over all 600 pairs).  This arm "
            "re-verifies that raw by sha256 and stands on it; a third decode would add "
            "3.66 GB to a store with 22 GiB free and could only reproduce the same sha."
        ),
    }
    if args.verify_raw:
        payload["raw_reverified"] = {
            "path": str(args.raw), "sha256": sha256_file(Path(args.raw)),
            "bytes": Path(args.raw).stat().st_size,
            "expect_sha256": pd4.PD3_RAW_SHA256, "expect_bytes": pd4.PD3_RAW_BYTES,
        }
        got = payload["raw_reverified"]
        if got["sha256"] != pd4.PD3_RAW_SHA256 or got["bytes"] != pd4.PD3_RAW_BYTES:
            raise Pd5Error(f"inherited raw does not reproduce move 52's decode: {got}")
    for name, path, expect in (
        ("pd4_control_archive", args.control_archive, pd4.MOVE52_ARCHIVE_SHA256),
    ):
        if path is None:
            continue
        sha = sha256_file(Path(path))
        payload[name] = {"path": str(path), "sha256": sha, "bytes": Path(path).stat().st_size}
        if sha != expect:
            raise Pd5Error(f"{name} sha {sha} != {expect}")
        if Path(path).stat().st_size != pd4.MOVE52_ARCHIVE_BYTES:
            raise Pd5Error(f"{name} is not {pd4.MOVE52_ARCHIVE_BYTES} B")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: prereg -- the prediction and the falsifiers, written before any pd5 search row
# ----------------------------------------------------------------------------------


def marginal_of_the_second_token(priced_rows: Path) -> dict[str, Any]:
    """pd4's OWN 122 clustered rows, re-read: what did token 2 cost and what did it buy?

    This is the prior that decides whether tokens 3 and 4 can be worth anything, and it is a
    read-only re-analysis of pd4's retained ``sheets/priced_rows.jsonl`` -- no new search, no
    new encode.  A cluster is matched to the single-token row whose edit is its FIRST edit,
    which is exactly the anchor pd4's cluster search grew it from.
    """
    rows = [json.loads(line) for line in Path(priced_rows).read_text().splitlines()
            if line.strip()]
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(int(row["pair"]), []).append(row)
    samples: list[dict[str, float]] = []
    for row in rows:
        if int(row.get("tokens_changed", 1)) != 2:
            continue
        first = tuple(int(x) for x in row["edits"][0])
        anchor = next((s for s in by_pair[int(row["pair"])]
                       if int(s.get("tokens_changed", 1)) == 1
                       and tuple(int(x) for x in s["edits"][0]) == first), None)
        if anchor is None:
            continue
        samples.append({
            "marginal_bits": float(row["real_delta_bits"]) - float(anchor["real_delta_bits"]),
            "marginal_credit_d_pose": float(row["credit_d_pose"])
            - float(anchor["credit_d_pose"]),
            "marginal_cells": float(row["d_cells"] - anchor["d_cells"]),
        })
    if not samples:
        raise Pd5Error(f"no matched cluster/anchor rows in {priced_rows}")

    def med(vals: list[float]) -> float:
        srt = sorted(vals)
        return srt[len(srt) // 2]

    singles = [r for r in rows if int(r.get("tokens_changed", 1)) == 1]
    return {
        "source": str(priced_rows),
        "matched_cluster_anchor_rows": len(samples),
        "isolated_token_bits_median": med([float(r["real_delta_bits"]) for r in singles]),
        "marginal_bits_median": med([s["marginal_bits"] for s in samples]),
        "marginal_bits_mean": sum(s["marginal_bits"] for s in samples) / len(samples),
        "marginal_credit_d_pose_median": med([s["marginal_credit_d_pose"] for s in samples]),
        "marginal_credit_improves_fraction": sum(
            1 for s in samples if s["marginal_credit_d_pose"] < 0.0) / len(samples),
        "marginal_cells_median": med([s["marginal_cells"] for s in samples]),
        "reading": (
            "token 2 is CHEAP (the coder's context model discounts a neighbour) and its "
            "pose credit is a coin flip biased the wrong way; a longer run therefore buys "
            "cheap bits and usually spends pose, and only the SEARCH over many extensions "
            "can decide whether the best one pays"
        ),
    }


def cmd_prereg(args) -> int:
    """The prediction, the falsifiers and the bar -- fixed before this arm searches."""
    receipts = pd4.bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    payload = {
        "schema": "ddm_pd5_prereg.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]",
        "score_claim": False,
        "written_before": "any pd5 search row, price sheet, set encode or admission existed",
        "base": {
            "pointer": "move 52", "archive_bytes": pd4.MOVE52_ARCHIVE_BYTES,
            "archive_sha256": pd4.MOVE52_ARCHIVE_SHA256,
            "score_t4": pd4.MOVE52_SCORE_T4, "pose_base_mean": base_mean,
            "instrument_d_seg": receipts["instrument_d_seg_move52"],
        },
        "economics_they_expire_at_every_pointer_move": {
            "S_per_archive_byte": S_PER_BYTE,
            "S_per_flipped_seg_cell_T4_carried": seg_s_per_cell(),
            "S_per_unit_of_one_pairs_d_pose": pose_s_per_pair_unit(base_mean),
        },
        "prior_from_pd4_reanalysis": marginal_of_the_second_token(Path(args.pd4_priced)),
        "pd4_measured": {
            "admitted_subset_bits_per_token": 12.0,
            "admitted_subset_net_dS": -1.029893e-05,
            "bar_price_derived_by_bisection": 10.891,
            "ledger_under_charge_fraction": 0.363,
            "single_token_bits_pooled": 15.839,
            "cluster_two_token_bits_pooled": 13.804,
            "cluster_family_walked_fraction": 0.45,
        },
        "prediction": (
            "The run-length lever lowers the per-token PRICE -- pd4's own rows already say "
            "the marginal token is ~2.5x cheaper than an isolated one -- but the marginal "
            "CREDIT is negative at the median, so the family's net gain is bounded by what "
            "the SEARCH can find among extensions, not by the discount.  Adding the whole "
            "2-token family to pd4's singles-only pool moved the LEDGER net from -1.144e-05 "
            "to -1.471e-05, i.e. -3.27e-06; if tokens 3 and 4 buy a comparable or smaller "
            "increment (extensions get worse, not better), the pass lands near -1.2e-05 to "
            "-1.6e-05 real and the -2e-05 bar does not fall.  This arm MEASURES that."
        ),
        "admit_bar_net_dS": -2e-05,
        "pre_registered_band_net_dS": [-2e-05, -5e-05],
        "falsifiers": {
            "F1": "the RLC1 pricer does not reproduce move 52's own 179,332 B archive byte-identically",
            "F2": "the base is not move 52's own decode (sha ccb89e3e...)",
            "F3": "the per-proposal price does not resolve above its measured noise floor",
            "F4": "3- and 4-token runs are NOT cheaper per token than 2-token runs",
            "F5": "the admitted SET's real bits/token is >= 11.0",
            "F6": "composition is summed rather than realized (realized fraction < 0.99)",
            "F7": "the set re-price differs from the ledger by > 10 % after iteration",
            "F8": "the admitted set's net vs move 52 is > -2e-05",
            "closure_rule": (
                "F5, F7 or F8 firing CLOSES the multi-token joint-proposal formulation on "
                "this object; the arm says so plainly and builds nothing past the admission"
            ),
        },
        "boundaries": [
            "no Modal, no authorize_*, no fire, no packet, no completion",
            "upstream/, the PR tree, sealed trees, contract code, receiver code, the "
            "renderer, the basis and the prior are never edited",
            "pd1-pd4 / sj1 / pp1 custody is opened read-only",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: seeds -- the run each pair grows from, and the walk order
# ----------------------------------------------------------------------------------


def cmd_seeds(args) -> int:
    """Per pair: the best REALIZED run of length ``--seed-tokens`` to extend by one token.

    Ranked by the pair's own benefit_S so the round-robin shard walk is a near-uniform prefix
    of the GLOBAL order and a prefix stop stays interpretable (pd4's STOP_RULE discipline).
    """
    pd4.bind_move52(verify_raw=False, raw=args.raw)
    base = np.load(args.base_pose)
    base_mean = float(base.mean())
    pose_unit = pose_s_per_pair_unit(base_mean)
    seg_cell = seg_s_per_cell()
    pool, stats = load_pool_move52(
        _row_files(args.rows), base, band_abs=BASE_BAND_ABS,
        exclude_pairs=set(edited_pairs_at_move52()),
    )
    score_proposals(pool, pose_unit=pose_unit, seg_cell=seg_cell)
    wanted = {int(t) for t in str(args.seed_tokens).split(",") if t.strip()}
    seeds: dict[str, Any] = {}
    skipped = {"no_row_of_that_length": 0, "no_paying_row": 0}
    for pair, entries in sorted(pool.items()):
        cands = [e for e in entries if int(e["tokens_changed"]) in wanted]
        if not cands:
            skipped["no_row_of_that_length"] += 1
            continue
        paying = [e for e in cands if e["benefit_S"] > 0.0]
        if not paying:
            skipped["no_paying_row"] += 1
            continue
        best = max(paying, key=lambda e: e["benefit_S"])
        seeds[str(pair)] = {
            "edits": row_edits(best),
            "tokens": int(best["tokens_changed"]),
            "benefit_S": float(best["benefit_S"]),
            "credit_d_pose": float(best["credit_d_pose"]),
            "d_cells": int(best["d_cells"]),
            "d_pose_base": float(best["d_pose_base"]),
            "family": best.get("family", "single"),
            "row_source": best.get("row_source", "unknown"),
        }
    order = sorted(seeds, key=lambda p: -seeds[p]["benefit_S"])
    shards = [[] for _ in range(int(args.shard_count))]
    for i, pair in enumerate(order):
        shards[i % len(shards)].append(int(pair))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.seeds_name).write_text(json.dumps(seeds, indent=1, sort_keys=True))
    report = {
        "schema": "ddm_pd5_seeds.v1",
        "axis": "[macOS-CPU advisory / planning only]",
        "score_claim": False,
        "seed_tokens_wanted": sorted(wanted),
        "pairs_seeded": len(seeds),
        "by_seed_length": {str(t): sum(1 for v in seeds.values() if v["tokens"] == t)
                           for t in sorted(wanted)},
        "skipped": skipped,
        "gates": stats,
        "walk_order": [int(p) for p in order],
        "shards": {str(i): s for i, s in enumerate(shards)},
        "seeds_path": str(out_dir / args.seeds_name),
    }
    (out_dir / args.report_name).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("walk_order", "shards")}, indent=1, sort_keys=True))
    for i, s in enumerate(shards):
        print(f"shard {i}: {','.join(str(p) for p in s)}", flush=True)
    return 0


# ----------------------------------------------------------------------------------
# stage: run-search -- ONE more token on a seed run, realized jointly
# ----------------------------------------------------------------------------------


def cmd_run_search(args) -> int:
    """Grow a seed run by one token.  ONE render, ONE argmax, ONE carrier re-solve per row.

    This is pd4's ``cluster-search`` with the anchor generalised from a single token to a run
    of any length.  The new cell is drawn from the union of the radius-``--neighbour-radius``
    neighbourhoods of the run's OWN cells, masked to the argmax interior, sorted by pose
    saliency, screened on the frozen argmax and refined K-deep by pd1/jg5's solver.  The
    credit is measured against the PAIR's base, so a 3-token row is one object, not an
    anchor plus a delta.
    """
    pd4.bind_move52(verify_raw=False, raw=args.raw)
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_fe1_frame_embedding_search as fe1
    import ddm_jg1_seg_solve as jg1
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_pd1_pose_directed as pd1
    import ddm_pp1_pose_actuation as pp1

    pp1.set_threads(args.threads)
    started = time.time()
    seeds = json.loads(Path(args.seeds).read_text())
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
    rows_path = out_dir / f"run_{args.shard_index}.jsonl"
    screen_path = out_dir / f"run_screen_{args.shard_index}.jsonl"
    done: set[int] = set()
    if args.resume and rows_path.exists():
        done = {int(json.loads(line)["pair"])
                for line in rows_path.read_text().splitlines() if line.strip()}
    handle = rows_path.open("a")
    screen_handle = screen_path.open("a")
    counters = {"pairs": 0, "screened": 0, "refined": 0, "skipped_done": 0, "no_seed": 0}

    for pair in pairs:
        if pair in done and args.resume:
            counters["skipped_done"] += 1
            continue
        seed = seeds.get(str(pair))
        if seed is None:
            counters["no_seed"] += 1
            print(f"pair {pair}: no seed, skipped", flush=True)
            continue
        seed_edits = [(int(r), int(c), int(o), int(n)) for r, c, o, n in seed["edits"]]
        fe1.restore_pair_codes(body, pair)
        base_plane = body.tokens[pair].copy()
        for r, c, old, _new in seed_edits:
            if int(base_plane[r, c]) != old:
                raise Pd5Error(
                    f"pair {pair} seed cell ({r},{c}) holds {int(base_plane[r, c])} but the "
                    f"seed row says {old}"
                )
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        d_pose_base = float(br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0])
        gap = abs(d_pose_base - float(base[pair]))
        if gap > args.base_band_abs:
            raise Pd5Error(
                f"pair {pair} batch-1 base {d_pose_base:.12e} is {gap:.3e} from the n600 base "
                f"{float(base[pair]):.12e}, outside the MEASURED band {args.base_band_abs:.3e}"
            )
        saliency = pp1.pose_saliency_on_token_grid(base_inst, pair,
                                                   np.asarray(raw[2 * pair + 1]))
        interior = pd1.interior_mask(np.asarray(shipped_argmax[pair]), args.interior_radius)
        radius = int(args.neighbour_radius)
        occupied = {(r, c) for r, c, _o, _n in seed_edits}
        neighbours: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for sr, sc, _o, _n in seed_edits:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = sr + dr, sc + dc
                    if not (0 <= rr < EVAL_H and 0 <= cc < EVAL_W):
                        continue
                    if (rr, cc) in occupied or (rr, cc) in seen:
                        continue
                    if not bool(interior[rr, cc]):
                        continue
                    seen.add((rr, cc))
                    neighbours.append((rr, cc))
        neighbours.sort(key=lambda rc: -float(saliency[rc[0], rc[1]]))
        seed_len = len(seed_edits)
        print(f"pair {pair}: seed {seed_len} tokens, base flips {base_flips}, d_pose "
              f"{d_pose_base:.4e}, {len(neighbours)} interior neighbours", flush=True)
        counters["pairs"] += 1

        def apply(edits: list[tuple[int, int, int, int]], *, forward: bool,
                  plane: np.ndarray = body.tokens[pair]) -> None:
            for r, c, old, new in edits:
                plane[r, c] = new if forward else old

        screened: list[dict[str, Any]] = []
        for rr, cc in neighbours:
            old2 = int(base_plane[rr, cc])
            for delta in deltas:
                new2 = old2 + delta
                if not 0 <= new2 < token_classes:
                    continue
                edits = [*seed_edits, (rr, cc, old2, new2)]
                apply(edits, forward=True)
                frame = fe1.render_pair(body, pair)
                moved_flips = fe1.flips_pair(
                    jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
                )
                apply(edits, forward=False)
                counters["screened"] += 1
                entry = {
                    "pair": pair,
                    "edits": [list(e) for e in edits],
                    "tokens_changed": len(edits),
                    "seed_tokens": seed_len,
                    "new_cell": [rr, cc], "new_old": old2, "new_new": new2,
                    "base_flips": base_flips, "flips": moved_flips,
                    "d_cells": moved_flips - base_flips,
                    "saliency": float(saliency[rr, cc]),
                    "interior": True, "family": f"run{len(edits)}",
                    "seed_family": seed.get("family", "single"),
                }
                screen_handle.write(json.dumps(entry) + "\n")
                if entry["d_cells"] <= args.max_cells:
                    screened.append(entry)
        screen_handle.flush()
        screened.sort(key=lambda e: (e["d_cells"], -e["saliency"]))
        for entry in screened[: args.refine]:
            edits = [(int(a), int(b), int(c), int(d)) for a, b, c, d in entry["edits"]]
            apply(edits, forward=True)
            frame = fe1.render_pair(body, pair)
            apply(edits, forward=False)
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
                "row_source": "ddm_pd5_run",
            })
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"pair {pair} run{len(edits)} +({entry['new_cell'][0]},"
                  f"{entry['new_cell'][1]}) -> cells {row['d_cells']:+d}, resolved "
                  f"{resolved:.4e} ({row['resolved_over_base']:.4f}x)", flush=True)
    handle.close()
    screen_handle.close()
    (out_dir / f"RUN_{args.shard_index}.json").write_text(json.dumps({
        "schema": "ddm_pd5_run_search.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage]",
        "score_claim": False,
        "proposal": "ONE more token in the seed run's own neighbourhood, realized JOINTLY",
        "seeds": str(args.seeds), "pairs": pairs, "counters": counters,
        "neighbour_radius": args.neighbour_radius, "deltas": deltas,
        "refine": args.refine, "max_cells": args.max_cells,
        "rows_path": str(rows_path), "screen_path": str(screen_path),
        "elapsed_seconds": time.time() - started,
    }, indent=1, sort_keys=True))
    print(json.dumps({"rows": str(rows_path), "counters": counters}))
    return 0


# ----------------------------------------------------------------------------------
# stage: smoke-timing -- the pair budget, DECLARED from a measurement under the real fleet
# ----------------------------------------------------------------------------------


def cmd_smoke_timing(args) -> int:
    """Seconds per pair and per refine, measured with the price encodes running alongside."""
    shards: list[dict[str, Any]] = []
    for path in sorted(Path(args.smoke_dir).rglob("RUN_*.json")):
        if path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text())
        shards.append({
            "receipt": str(path), "pairs": payload["pairs"],
            "screened": payload["counters"]["screened"],
            "refined": payload["counters"]["refined"],
            "wall_seconds": payload["elapsed_seconds"],
            "seed_length": "2->3" if "two_to_three" in str(path) else "1->2",
        })
    if not shards:
        raise Pd5Error(f"no RUN_*.json receipts under {args.smoke_dir}")
    walls = sorted(s["wall_seconds"] for s in shards)
    refined = sum(s["refined"] for s in shards)
    total_wall = sum(s["wall_seconds"] for s in shards)
    per_refine = total_wall / refined if refined else math.nan
    mean_wall = total_wall / len(shards)
    shard_count = int(args.shard_count)
    throughput = shard_count / mean_wall * 60.0 if mean_wall else math.nan
    budget = {}
    for name, count in json.loads(args.populations).items():
        budget[name] = {
            "pairs": count,
            "projected_minutes_mean_basis": count * mean_wall / shard_count / 60.0,
            "projected_minutes_slowest_shard_basis":
                count * walls[-1] / shard_count / 60.0,
        }
    payload = {
        "schema": "ddm_pd5_smoke_timing.v1",
        "axis": "[macOS-CPU advisory / wall-clock under the fleet condition this arm runs in]",
        "score_claim": False,
        "fleet_condition": args.fleet_condition,
        "shards": shards,
        "measured": {
            "shard_wall_seconds_min": walls[0], "shard_wall_seconds_max": walls[-1],
            "shard_wall_seconds_mean": mean_wall,
            "shard_wall_seconds_median": walls[len(walls) // 2],
            "refines": refined, "seconds_per_refine": per_refine,
            "pd4_seconds_per_refine_for_comparison": 35.0,
            "pairs_per_minute_at_shard_count": throughput,
            "shard_count_for_the_projection": shard_count,
        },
        "k_refine_declared": int(args.k_refine),
        "k_refine_rationale": (
            "K_refine 12 is pd4's and pd3's, held FIXED: pd3 MEASURED K=12 strictly better "
            "than K=8 on 24 of 125 shared pairs and worse on none.  Depth is not this arm's "
            "declared delta -- run LENGTH and SET pricing are -- so the depth stays put."
        ),
        "pair_budget": budget,
        "prefix_stop_rule": (
            "the walk order is round-robin over shards in descending benefit_S, so a stop "
            "leaves a near-uniform prefix of the GLOBAL order; any verdict drawn on a "
            "stopped wave is scoped to its measured coverage fraction, as pd4's was"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in payload.items() if k != "shards"},
                     indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: sheets -- pd4's price sheets, over a pool that also admits pd5's own rows
# ----------------------------------------------------------------------------------


def cmd_sheets(args) -> int:
    """pd4's ``sheets`` verbatim, run against ``load_pool_move52``.

    The sheet construction, the density rule and the held-fixed price control are pd4's and
    are NOT re-implemented here: the only thing this arm changes is which stores count as
    "measured on move 52's renders", so ``pd4.load_pool`` is swapped for the hour of this
    call and restored in ``finally``.  The swap is process-local -- it cannot reach any other
    arm's run -- and rebinding rather than copying is deliberate: a copied 90-line sheet
    builder would drift from pd4's and the price sheets would stop being comparable.
    """
    args.rows = [str(p) for p in _row_files(args.rows)]
    original = pd4.load_pool
    pd4.load_pool = load_pool_move52
    try:
        return pd4.cmd_sheets(args)
    finally:
        pd4.load_pool = original


def cmd_price_merge(args) -> int:
    """pd4's ``price-merge`` verbatim, reading THIS arm's sheet encodes.

    pd4's merge hard-codes the registry key ``pd4sheet{k:02d}``.  This arm's encodes live in
    its OWN price store under ``pd5sheet{k:02d}``, and naming them ``pd4sheet…`` inside a pd5
    store would be a provenance lie a later reader could not unpick.  So the NAME is remapped
    for the duration of the call and pd4's merge -- including its three noise controls -- runs
    unchanged.  Process-local; restored in ``finally``.
    """
    original = pd4._encode_bits

    def remapped(rlc1_root: Path, field: str, tag: str) -> np.ndarray:
        if field.startswith("pd4sheet"):
            field = args.field_prefix + field[len("pd4sheet"):]
        return original(rlc1_root, field, tag)

    pd4._encode_bits = remapped
    try:
        return pd4.cmd_price_merge(args)
    finally:
        pd4._encode_bits = original


# ----------------------------------------------------------------------------------
# stage: price-report -- bits/token BY RUN LENGTH, and the marginal of the added token
# ----------------------------------------------------------------------------------


def cmd_price_report(args) -> int:
    """The charter's headline: the real per-token price as a function of run length."""
    rows = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
            if line.strip()]
    for row in rows:
        row.setdefault("tokens_changed", len(row_edits(row)))
    by_len: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_len.setdefault(int(row["tokens_changed"]), []).append(row)

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        per = sorted(float(r["real_delta_bits"]) / int(r["tokens_changed"]) for r in group)
        tot = sum(float(r["real_delta_bits"]) for r in group)
        tok = sum(int(r["tokens_changed"]) for r in group)
        return {
            "proposals": len(group), "tokens": tok, "total_delta_bits": tot,
            "bits_per_token_pooled": tot / tok if tok else math.nan,
            "bits_per_token_median": per[len(per) // 2] if per else math.nan,
            "bits_per_token_min": per[0] if per else math.nan,
            "bits_per_token_max": per[-1] if per else math.nan,
        }

    index = {(int(r["pair"]), proposal_identity(r)[1]): r for r in rows}
    marginal: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        n = int(row["tokens_changed"])
        if n < 2:
            continue
        edits = row_edits(row)
        for drop in range(len(edits)):
            sub = [e for i, e in enumerate(edits) if i != drop]
            key = (int(row["pair"]), tuple(sorted((r, c, nn) for (r, c, _o, nn) in sub)))
            parent = index.get(key)
            if parent is None:
                continue
            marginal.setdefault(n, []).append({
                "pair": int(row["pair"]),
                "marginal_bits": float(row["real_delta_bits"])
                - float(parent["real_delta_bits"]),
                "marginal_credit_d_pose": float(row["credit_d_pose"])
                - float(parent["credit_d_pose"]),
                "marginal_cells": int(row["d_cells"]) - int(parent["d_cells"]),
            })
            break

    def med(values: list[float]) -> float:
        vals = sorted(values)
        return vals[len(vals) // 2] if vals else math.nan

    report = {
        "schema": "ddm_pd5_price_report.v1",
        "axis": "[macOS-CPU advisory / EXACT frame-local bits from the shipped coder]",
        "score_claim": False,
        "by_run_length": {str(k): stats(v) for k, v in sorted(by_len.items())},
        "marginal_of_the_added_token": {
            str(k): {
                "samples": len(v),
                "bits_median": med([m["marginal_bits"] for m in v]),
                "bits_mean": sum(m["marginal_bits"] for m in v) / len(v),
                "credit_d_pose_median": med([m["marginal_credit_d_pose"] for m in v]),
                "credit_improves_fraction": sum(
                    1 for m in v if m["marginal_credit_d_pose"] < 0.0) / len(v),
                "cells_median": med([float(m["marginal_cells"]) for m in v]),
            }
            for k, v in sorted(marginal.items()) if v
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# stage: setprice -- the CURE: admit on the SET's own real bits, iterated to a fixed point
# ----------------------------------------------------------------------------------


def _sweep(
    cands: dict[int, dict[str, Any]], *, bits: dict[int, float],
) -> tuple[list[int], dict[str, float]]:
    """Keep every pair whose own benefit beats its own rate charge.  No lambda search needed.

    The three legs are separable per pair once the rate leg is a per-pair MEASURED number:
    a pair is in the set iff ``benefit_S - bits/8 * S_PER_BYTE > 0``.  A lambda sweep over a
    shared budget would be the right tool if the rate were a shared resource; it is not --
    every admitted pair pays its own bits.
    """
    keep: list[int] = []
    total_benefit = 0.0
    total_bits = 0.0
    for pair, row in sorted(cands.items()):
        charge = bits[pair] / 8.0 * S_PER_BYTE
        if float(row["benefit_S"]) - charge > 0.0:
            keep.append(pair)
            total_benefit += float(row["benefit_S"])
            total_bits += bits[pair]
    return keep, {
        "benefit_S": total_benefit, "bits": total_bits,
        "net_S": -(total_benefit - total_bits / 8.0 * S_PER_BYTE),
    }


def cmd_setprice(args) -> int:
    """Build iteration ``--iteration``'s candidate SET field from the current bit estimates.

    Iteration 0 uses the sheet ledger.  Every later iteration replaces the estimate of every
    pair that appeared in the PREVIOUS iteration's set with that set's OWN measured
    frame-local delta bits, and re-runs the admission.  The loop stops when the set repeats
    (fixed point) or when the real total and the ledger total agree inside the measured noise.
    """
    pd4.bind_move52(verify_raw=False, raw=args.raw)
    rows = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
            if line.strip()]
    base = np.load(args.base_pose)
    pose_unit = pose_s_per_pair_unit(float(base.mean()))
    seg_cell = seg_s_per_cell()
    for row in rows:
        row.setdefault("tokens_changed", len(row_edits(row)))
        row["benefit_S"] = (-(float(row["credit_d_pose"]) * pose_unit)
                            - float(row["d_cells"]) * seg_cell)
    ledger = {(int(r["pair"]), proposal_identity(r)[1]): float(r["real_delta_bits"]) for r in rows}
    measured: dict[tuple, float] = {}
    history: list[dict[str, Any]] = []
    state_path = Path(args.state)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        history = state["history"]
        measured = {(int(k.split("|", 1)[0]), tuple(tuple(x) for x in json.loads(k.split("|", 1)[1]))): v
                    for k, v in state["measured_bits"].items()}
    # every pair's best candidate under the CURRENT estimates, then the admission
    def estimate(row: dict[str, Any]) -> float:
        key = (int(row["pair"]), proposal_identity(row)[1])
        return measured.get(key, ledger[key])

    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        charge = estimate(row) / 8.0 * S_PER_BYTE
        row["net_S_estimate"] = float(row["benefit_S"]) - charge
        pair = int(row["pair"])
        if pair not in best or row["net_S_estimate"] > best[pair]["net_S_estimate"]:
            best[pair] = row
    bits = {p: estimate(r) for p, r in best.items()}
    keep, totals = _sweep(best, bits=bits)
    chosen = {p: best[p] for p in keep}

    source = np.load(args.control_npz)
    planes = {int(k): source[k] for k in source.files}
    if len(planes) != N_PAIRS:
        raise Pd5Error(f"control field has {len(planes)} planes, expected {N_PAIRS}")
    field = {str(p): planes[p].copy() for p in range(N_PAIRS)}
    for pair, row in chosen.items():
        plane = field[str(pair)]
        for r, c, old, new in row_edits(row):
            if int(plane[r, c]) != old:
                raise Pd5Error(
                    f"pair {pair} cell ({r},{c}) holds {int(plane[r, c])} in the control "
                    f"field but the row says {old}"
                )
            plane[r, c] = new
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"set_{int(args.iteration):02d}.npz"
    np.savez_compressed(npz, **field)
    written = np.load(npz)
    for pair in range(N_PAIRS):
        if not np.array_equal(written[str(pair)], field[str(pair)]):
            raise Pd5Error(f"set field does not round-trip plane {pair}")

    entry = {
        "iteration": int(args.iteration),
        "pairs": keep,
        "tokens": int(sum(int(chosen[p]["tokens_changed"]) for p in keep)),
        "ledger_bits_total": float(sum(bits[p] for p in keep)),
        "measured_pairs_used": int(sum(
            1 for p in keep if (p, proposal_identity(chosen[p])[1]) in measured)),
        "estimated_net_S": float(totals["net_S"]),
        "field": str(npz),
        "field_sha256": sha256_file(npz),
        "rows": [{"pair": p, "edits": row_edits(chosen[p]),
                  "tokens": int(chosen[p]["tokens_changed"]),
                  "family": chosen[p].get("family", "single"),
                  "d_cells": int(chosen[p]["d_cells"]),
                  "credit_d_pose": float(chosen[p]["credit_d_pose"]),
                  "benefit_S": float(chosen[p]["benefit_S"]),
                  "bits_estimate": bits[p],
                  "bits_source": ("measured_in_a_prior_set"
                                  if (p, proposal_identity(chosen[p])[1]) in measured
                                  else "sheet_ledger")}
                 for p in keep],
    }
    prior = history[-1]["pairs"] if history else None
    entry["set_repeats_previous"] = prior is not None and prior == keep
    # idempotent per iteration: re-running an index REPLACES its entry rather than appending
    # a second one, so a resumed arm cannot double-count an iteration in the convergence log.
    history = [e for e in history if int(e["iteration"]) != int(args.iteration)]
    history.append(entry)
    history.sort(key=lambda e: int(e["iteration"]))
    state = {
        "schema": "ddm_pd5_setprice.v1",
        "axis": "[macOS-CPU advisory / field construction; bits are EXACT where measured]",
        "score_claim": False,
        "history": history,
        "measured_bits": {f"{k[0]}|{json.dumps([list(x) for x in k[1]])}": v
                          for k, v in measured.items()},
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in entry.items() if k != "rows"},
                     indent=1, sort_keys=True))
    return 0


def cmd_carry_from_set(args) -> int:
    """Emit ``carry_rows.jsonl`` for exactly the proposals a SET iteration chose.

    The realization renders the SPECIFIC edits it is given.  If the set iteration were free
    to switch a pair's proposal after the render, the composed pose leg would be measured on
    a render that does not exist -- the silent class this campaign keeps paying for.  So the
    carried rows are pinned to one iteration's choices, and every later iteration may only
    DROP pairs, never swap them.
    """
    state = json.loads(Path(args.state).read_text())
    entry = next(e for e in state["history"] if int(e["iteration"]) == int(args.iteration))
    chosen = {(int(r["pair"]), tuple(tuple(int(x) for x in e) for e in r["edits"]))
              for r in entry["rows"]}
    priced = [json.loads(line) for line in Path(args.priced).read_text().splitlines()
              if line.strip()]
    base = np.load(args.base_pose)
    pose_unit = pose_s_per_pair_unit(float(base.mean()))
    seg_cell = seg_s_per_cell()
    out: list[dict[str, Any]] = []
    for row in priced:
        key = (int(row["pair"]), tuple(tuple(int(x) for x in e) for e in row["edits"]))
        if key not in chosen:
            continue
        row["benefit_S"] = (-(float(row["credit_d_pose"]) * pose_unit)
                            - float(row["d_cells"]) * seg_cell)
        row["dS_rate_real"] = (float(row["real_delta_bits"]) / 8.0) * S_PER_BYTE
        row["dS_modelled_real_price"] = -row["benefit_S"] + row["dS_rate_real"]
        out.append(row)
    if len(out) != len(chosen):
        raise Pd5Error(f"{len(out)} priced rows matched {len(chosen)} chosen proposals")
    out.sort(key=lambda r: int(r["pair"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps({"carry_rows": str(args.out), "pairs": len(out),
                      "tokens": int(sum(int(r["tokens_changed"]) for r in out)),
                      "ledger_bits": float(sum(float(r["real_delta_bits"]) for r in out)),
                      "from_iteration": int(args.iteration)}, indent=1, sort_keys=True))
    return 0


def cmd_setabsorb(args) -> int:
    """Fold a SET encode's MEASURED frame-local bits back into the estimates."""
    state_path = Path(args.state)
    state = json.loads(state_path.read_text())
    entry = next(e for e in state["history"] if int(e["iteration"]) == int(args.iteration))
    control = np.load(args.bits_control)
    cand = np.asarray(json.loads(Path(args.encode).read_text())["per_frame_bits"],
                      dtype=np.float64)
    if cand.shape != (N_PAIRS,) or control.shape != (N_PAIRS,):
        raise Pd5Error("per-frame bit vectors must be n600")
    delta = cand - control
    measured = state["measured_bits"]
    per_pair = {}
    for row in entry["rows"]:
        pair = int(row["pair"])
        key = f"{pair}|{json.dumps(sorted([r, c, n] for r, c, _o, n in row['edits']))}"
        measured[key] = float(delta[pair])
        per_pair[str(pair)] = {"ledger": float(row["bits_estimate"]),
                               "measured": float(delta[pair])}
    ledger_total = float(sum(r["bits_estimate"] for r in entry["rows"]))
    real_total = float(sum(delta[int(r["pair"])] for r in entry["rows"]))
    signed_total = float(delta.sum())
    edited = {int(r["pair"]) for r in entry["rows"]}
    spill = float(sum(delta[p] for p in range(N_PAIRS) if p not in edited))
    encode_payload = json.loads(Path(args.encode).read_text())
    entry["absorbed"] = {
        "encode": str(args.encode),
        "stream_bytes": encode_payload.get("stream_bytes"),
        "ideal_bytes": encode_payload.get("ideal_bytes"),
        "output_lossless": encode_payload.get("output_lossless"),
        "decoded_field_sha256": encode_payload.get("decoded_field_sha256"),
        "archives": encode_payload.get("archives"),
        "ledger_bits_total": ledger_total,
        "real_frame_local_bits_total": real_total,
        "residual_bias_fraction": (real_total - ledger_total) / ledger_total
        if ledger_total else math.nan,
        "signed_whole_field_bits": signed_total,
        "spill_onto_unedited_frames_bits": spill,
        "spill_fraction_of_signed": spill / signed_total if signed_total else math.nan,
        "per_pair": per_pair,
    }
    state["measured_bits"] = measured
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True))
    print(json.dumps(entry["absorbed"], indent=1, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    bind = sub.add_parser("bind", help="re-point every stale module at move 52")
    bind.add_argument("--out", type=Path, required=True)
    bind.add_argument("--raw", type=Path, default=None)
    bind.add_argument("--verify-raw", action="store_true")
    bind.add_argument("--control-archive", type=Path, default=None)
    bind.set_defaults(func=cmd_bind)

    pre = sub.add_parser("prereg", help="the prediction + falsifiers, before any search row")
    pre.add_argument("--base-pose", type=Path, required=True)
    pre.add_argument("--pd4-priced", type=Path,
                     default=PD4_STORE / "sheets/priced_rows.jsonl")
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--raw", type=Path, default=None)
    pre.set_defaults(func=cmd_prereg)

    seeds = sub.add_parser("seeds", help="per-pair seed run + the shard walk order")
    seeds.add_argument("--rows", nargs="+", required=True)
    seeds.add_argument("--base-pose", type=Path, required=True)
    seeds.add_argument("--seed-tokens", default="1,2")
    seeds.add_argument("--shard-count", type=int, default=8)
    seeds.add_argument("--out-dir", type=Path, required=True)
    seeds.add_argument("--seeds-name", default="SEEDS.json")
    seeds.add_argument("--report-name", default="SEED_PLAN.json")
    seeds.add_argument("--raw", type=Path, default=None)
    seeds.set_defaults(func=cmd_seeds)

    run = sub.add_parser("run-search", help="grow a seed run by ONE token, realized jointly")
    run.add_argument("--pairs", required=True)
    run.add_argument("--seeds", type=Path, required=True)
    run.add_argument("--base-pose", type=Path, required=True)
    run.add_argument("--base-band-abs", type=float, default=BASE_BAND_ABS)
    run.add_argument("--raw", type=Path, default=None)
    run.add_argument("--refine", type=int, default=12)
    run.add_argument("--neighbour-radius", type=int, default=1)
    run.add_argument("--interior-radius", type=int, default=3)
    run.add_argument("--max-cells", type=int, default=2)
    run.add_argument("--deltas", default="-1,1")
    run.add_argument("--outer-rounds", type=int, default=3)
    run.add_argument("--max-gn-iterations", type=int, default=12)
    run.add_argument("--threads", type=int, default=2)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_run_search)

    st = sub.add_parser("smoke-timing", help="declare the pair budget from a measurement")
    st.add_argument("--smoke-dir", type=Path, required=True)
    st.add_argument("--shard-count", type=int, default=8)
    st.add_argument("--k-refine", type=int, default=12)
    st.add_argument("--populations", default='{"two_to_three": 51, "one_to_two": 105}')
    st.add_argument("--fleet-condition", default="")
    st.add_argument("--out", type=Path, required=True)
    st.set_defaults(func=cmd_smoke_timing)

    sh = sub.add_parser("sheets", help="pd4's price sheets over pd4's + pd5's rows")
    sh.add_argument("--rows", nargs="+", required=True)
    sh.add_argument("--base-pose", type=Path, required=True)
    sh.add_argument("--control-npz", type=Path, default=pd4.MOVE52_FIELD)
    sh.add_argument("--per-pair", type=int, default=6)
    sh.add_argument("--only-pairs", default="")
    sh.add_argument("--name-prefix", default="sheet")
    sh.add_argument("--raw", type=Path, default=None)
    sh.add_argument("--out-dir", type=Path, required=True)
    sh.set_defaults(func=cmd_sheets)

    pm = sub.add_parser("price-merge", help="pd4's merge, reading pd5's own sheet encodes")
    pm.add_argument("--sheets", type=Path, required=True)
    pm.add_argument("--rlc1-root", type=Path, required=True)
    pm.add_argument("--control-tag", default="primary")
    pm.add_argument("--sheet-tag", default="primary")
    pm.add_argument("--field-prefix", default="pd5sheet")
    pm.add_argument("--cross-check-field", default="")
    pm.add_argument("--cross-check-sheets", type=Path, default=None)
    pm.add_argument("--raw", type=Path, default=None)
    pm.add_argument("--out-dir", type=Path, required=True)
    pm.set_defaults(func=cmd_price_merge)

    pr = sub.add_parser("price-report", help="real bits/token BY RUN LENGTH + the marginal")
    pr.add_argument("--priced", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=cmd_price_report)

    sp = sub.add_parser("setprice", help="build iteration N's candidate SET field")
    sp.add_argument("--priced", type=Path, required=True)
    sp.add_argument("--base-pose", type=Path, required=True)
    sp.add_argument("--control-npz", type=Path, default=pd4.MOVE52_FIELD)
    sp.add_argument("--state", type=Path, required=True)
    sp.add_argument("--iteration", type=int, required=True)
    sp.add_argument("--out-dir", type=Path, required=True)
    sp.add_argument("--raw", type=Path, default=None)
    sp.set_defaults(func=cmd_setprice)

    cf = sub.add_parser("carry-from-set", help="carry_rows.jsonl pinned to one set iteration")
    cf.add_argument("--state", type=Path, required=True)
    cf.add_argument("--iteration", type=int, required=True)
    cf.add_argument("--priced", type=Path, required=True)
    cf.add_argument("--base-pose", type=Path, required=True)
    cf.add_argument("--out", type=Path, required=True)
    cf.set_defaults(func=cmd_carry_from_set)

    sa = sub.add_parser("setabsorb", help="fold a SET encode's measured bits into the state")
    sa.add_argument("--state", type=Path, required=True)
    sa.add_argument("--iteration", type=int, required=True)
    sa.add_argument("--encode", type=Path, required=True)
    sa.add_argument("--bits-control", type=Path, required=True)
    sa.set_defaults(func=cmd_setabsorb)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
