#!/usr/bin/env python3
"""ddm_rp1 sizing -- realized argmax-NEUTRALITY of rate-directed token changes.

THE QUESTION
------------
``ddm_rp1_rate_rank`` produces, for every token position, the shipped coder's exact
first-order saving from rewriting that position to the coder's own most-probable class.
That is a CEILING, not a yield: rewriting a token changes what the receiver renders, and
the score only holds if the re-rendered, re-segmented argmax is UNCHANGED.

This module measures the yield.  For a seeded pair sample it takes the top-ranked
proposals, applies each one ALONE, renders frame ``2p+1`` through the receiver's own
renderer at ``semantic_batch=1`` (batch 8 is byte-changing on this half -- ``ddm_up2``
sec.6), re-segments with the frozen CPU SegNet through the evaluator's own preprocess,
and ACCEPTS only if the argmax is identical on all 196,608 cells.

Not "flips fall".  Not "d_seg does not rise".  IDENTICAL.  A proposal that moves one cell
in a helpful direction is still refused here, because this arm's contract with the seg
leg is that the seg leg does not move at all -- that is what makes the rate delta the
whole delta, and it is the only version of this arm that composes with a live seg arm
working the same field.

The single-proposal loop measures the NEUTRAL FRACTION.  The composite re-render of the
accepted set is the ACCEPTANCE: a set can only be shipped as a set, so the set is what
gets verified.

``[macOS-CPU advisory]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W
PLANE = EVAL_H * EVAL_W


class Instrument:
    """Renderer + SegNet + the live field, held together so no caller mixes two bodies."""

    def __init__(self, *, threads: int, lineage: str = up2.LINEAGE_DALI) -> None:
        up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
        rp1.set_threads(threads)
        self.semantic = jg1.load_semantic_renderer(
            archive_path=rp1.ENCODER_TREE / "archive.zip",
            runtime_dir=rp1.ENCODER_TREE / "runtime",
        )
        self.net = jg1.load_segnet()
        self.gt = jg1.load_gt_seg_labels(lineage)
        self.base_field, self.live_field, self.field_receipts = rp1.load_live_field()

    def argmax_batch(self, planes: list[np.ndarray], pair: int) -> np.ndarray:
        """Realize a list of candidate planes for ONE pair; return their argmaxes.

        The render is per-variant at batch 1 (required for decode identity); only the
        SegNet forward is batched, which is the shape the evaluator itself feeds.
        """
        stack = np.stack(planes).astype(np.uint8)
        frames = jg1.render_frame1(
            self.semantic, stack, np.full(len(planes), int(pair), dtype=np.int64)
        )
        return jg1.argmax_from_camera_frames(self.net, frames)


#: The renderer's own receptive field, DERIVED from the receiver source (cpr1/inflate.py):
#: coord_mix 1x1 (r=0) -> TokenBlock depthwise 3x3 at dilations 1,1,2,4 -> head 3x3, so
#: r = 1+1+2+4+1 = 9 token cells.  A token whose proposed class ALREADY dominates that
#: window is one the render was mostly ignoring; this is the closed-form neutrality
#: predictor, and the sizing MEASURES whether it predicts.
RENDER_RADIUS_TOKENS = 9


def neighbourhood_agreement(plane: np.ndarray, pos: int, new_class: int) -> float:
    """Fraction of the render's own receptive window already equal to ``new_class``."""
    row, col = divmod(int(pos), EVAL_W)
    r0 = max(0, row - RENDER_RADIUS_TOKENS)
    r1 = min(EVAL_H, row + RENDER_RADIUS_TOKENS + 1)
    c0 = max(0, col - RENDER_RADIUS_TOKENS)
    c1 = min(EVAL_W, col + RENDER_RADIUS_TOKENS + 1)
    window = plane[r0:r1, c0:c1]
    return float((window == new_class).mean())


def select_proposals(
    order: np.ndarray, mode: str, top_k: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose which ranked proposals to realize.

    ``top`` takes the head of the ranking.  ``stratified`` also samples the middle and
    the tail, because a head-only sizing cannot tell "no neutral proposals exist" from
    "the neutral ones are ranked below the cut" -- the false-negative shape this arm's
    falsifier would otherwise walk into.
    """
    if mode == "top" or order.size <= top_k:
        return order[:top_k]
    head = max(1, top_k // 3)
    mid_n = max(1, top_k // 3)
    tail_n = top_k - head - mid_n
    picked = [order[:head]]
    mid_hi = min(order.size, max(head + 1, order.size // 10))
    if mid_hi > head:
        picked.append(
            rng.choice(order[head:mid_hi], size=min(mid_n, mid_hi - head), replace=False)
        )
    if order.size > mid_hi:
        picked.append(
            rng.choice(
                order[mid_hi:], size=min(tail_n, order.size - mid_hi), replace=False
            )
        )
    return np.concatenate(picked)


def load_candidates(rank_dir: Path) -> dict[str, np.ndarray]:
    with np.load(rank_dir / "candidates.npz", allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


#: Absolute rank-in-pair bucket edges.  The coverage-corrected projection weights each
#: bucket's MEASURED neutral fraction by the bucket's real population, so a sample that
#: only ever tested 32 of ~365 ranked proposals per pair cannot be read as if 32 were all
#: there is.
RANK_EDGES = [0, 4, 12, 32, 100, 400, 2000, 10**9]


def _coverage_corrected_bits_per_pair(
    rows: list[dict[str, Any]], edges: list[int]
) -> tuple[float, list[dict[str, Any]]]:
    """Sum over rank strata of (neutral fraction) x (population) x (mean neutral saving).

    Every term is measured: the fraction from the realized tests in that stratum, the
    population from each pair's own ranked candidate count, the saving from the coder.
    A stratum with zero tests contributes zero and is REPORTED as untested rather than
    silently extrapolated from its neighbours.
    """
    detail: list[dict[str, Any]] = []
    total = 0.0
    n_pairs = max(1, len(rows))
    for lo, hi in zip(edges[:-1], edges[1:]):
        tested = [
            t for r in rows for t in r["tested"] if lo <= t["rank_in_pair"] < hi
        ]
        population = sum(
            max(0, min(hi, int(r["candidates_available"])) - lo) for r in rows
        )
        if not tested:
            detail.append(
                {
                    "range": [lo, hi],
                    "tested": 0,
                    "population": population,
                    "untested_stratum": True,
                    "bits_contributed": 0.0,
                }
            )
            continue
        neutral = [t for t in tested if t["neutral"]]
        fraction = len(neutral) / len(tested)
        mean_saving = float(np.mean([t["saving_bits"] for t in neutral])) if neutral else 0.0
        bits = fraction * population * mean_saving / n_pairs
        total += bits
        detail.append(
            {
                "range": [lo, hi],
                "tested": len(tested),
                "neutral": len(neutral),
                "neutral_fraction": fraction,
                "population": population,
                "population_per_pair": population / n_pairs,
                "mean_neutral_saving_bits": mean_saving,
                "bits_per_pair_contributed": bits,
            }
        )
    return total, detail


def accept_by_bisection(
    inst: "Instrument",
    pair: int,
    plane: np.ndarray,
    base_argmax: np.ndarray,
    proposals: list[tuple[int, int]],
    *,
    max_verifies: int = 400,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Accept the largest verified-neutral subset, verifying CUMULATIVE sets only.

    Every ``verify`` call realizes ``accepted + group`` through the receiver's renderer
    and the frozen SegNet and requires argmax identity on all 196,608 cells.  Because a
    group is only merged after the cumulative set that CONTAINS it verified, the final
    accepted set is certified by an actual realization -- it is never assembled out of
    separately-verified parts, which is the interaction the composite check exists for.

    Cost is O(K) verifies only when almost everything fails; when most proposals are
    neutral the whole set passes on the first call.
    """
    stats = {"verifies": 0, "hit_cap": False}

    def verify(candidate: list[tuple[int, int]]) -> bool:
        stats["verifies"] += 1
        variant = plane.copy()
        flat = variant.reshape(-1)
        for pos, new_class in candidate:
            flat[pos] = np.uint8(new_class)
        return int((inst.argmax_batch([variant], pair)[0] != base_argmax).sum()) == 0

    accepted: list[tuple[int, int]] = []
    queue: list[list[tuple[int, int]]] = [list(proposals)] if proposals else []
    while queue:
        if stats["verifies"] >= max_verifies:
            stats["hit_cap"] = True
            break
        group = queue.pop(0)
        if verify(accepted + group):
            accepted = accepted + group
        elif len(group) > 1:
            mid = len(group) // 2
            queue.insert(0, group[mid:])
            queue.insert(0, group[:mid])
    if accepted and not verify(accepted):
        raise rp1.Rp1Error(
            f"pair {pair}: final accepted set failed its own realization -- refusing"
        )
    stats["accepted"] = len(accepted)
    stats["proposed"] = len(proposals)
    return accepted, stats


def _by_bucket(
    tested: list[dict[str, Any]], key: str, edges: list[float]
) -> list[dict[str, Any]]:
    """Neutral fraction conditioned on one covariate.  The DENOMINATOR is always shown."""
    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = [t for t in tested if lo <= t[key] < hi]
        if not rows:
            continue
        neutral = [t for t in rows if t["neutral"]]
        out.append(
            {
                "range": [lo, hi],
                "tested": len(rows),
                "neutral": len(neutral),
                "neutral_fraction": len(neutral) / len(rows),
                "mean_saving_bits": float(np.mean([t["saving_bits"] for t in rows])),
                "neutral_saving_bits_total": float(
                    sum(t["saving_bits"] for t in neutral)
                ),
            }
        )
    return out


def cmd_sizing(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rank_dir = Path(args.rank_dir)
    rank_receipt = json.loads((rank_dir / "RANK.json").read_text())
    if not rank_receipt["identity_control"]["byte_identical"]:
        raise rp1.Rp1Error(
            "the rank run's identity control did not pass; its ranking is not the "
            "shipped coder's prices"
        )

    pointer = rp1.verify_pointer()
    cand = load_candidates(rank_dir)
    saving = cand["bits_sym"].astype(np.float64) - cand["bits_best"].astype(np.float64)

    rng = np.random.default_rng(args.seed)
    if args.pair_list:
        pairs = np.array([int(x) for x in args.pair_list.split(",")], dtype=np.int64)
    elif args.pairs >= jg1.N_PAIRS:
        pairs = np.arange(jg1.N_PAIRS, dtype=np.int64)
    else:
        pairs = np.sort(rng.choice(jg1.N_PAIRS, size=args.pairs, replace=False))
    if args.shards > 1:
        # Interleaved, not blocked: a contiguous block of pairs is a prefix of a skewed
        # population ([[m88]]), so a shard that dies leaves a biased partial rather than
        # a uniform one.
        pairs = pairs[args.shard :: args.shards]

    inst = Instrument(threads=args.threads)
    started = time.perf_counter()

    rows: list[dict[str, Any]] = []
    #: pair -> [(pos, new_class)].  The EDITS are the resumable state, not the planes:
    #: a plane restored from a stale base is the silent-revert failure this arm inherits
    #: from sj1's own (fixed) subset writer.
    accepted_edits: dict[int, list[tuple[int, int]]] = {}
    wanted = {int(p) for p in pairs}
    ledger_path = out / "SIZING_ROWS.jsonl"
    if args.resume and ledger_path.is_file():
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if int(prior["pair"]) not in wanted:
                continue
            rows.append(prior)
            accepted_edits[int(prior["pair"])] = [
                (int(a["pos"]), int(a["best"])) for a in prior.get("accepted", [])
            ]
        done = {int(r["pair"]) for r in rows}
    else:
        done = set()
        ledger_path.write_text("")

    for pair in pairs:
        pair = int(pair)
        if pair in done:
            continue
        pair_started = time.perf_counter()
        plane = np.array(inst.live_field[pair], dtype=np.uint8)
        base_argmax = inst.argmax_batch([plane], pair)[0]
        gt_plane = inst.gt[pair]
        base_flips = int((base_argmax != gt_plane).sum())

        # BATCH CONTROL.  The base argmax is read at SegNet batch 1 and the proposals at
        # batch B.  If the forward were batch-dependent, every proposal would read as
        # non-neutral for a reason that has nothing to do with tokens -- the shape of a
        # false negative that would kill this arm on an instrument artefact.  So the
        # UNMODIFIED plane is put through the batched path and required to agree exactly.
        control = inst.argmax_batch([plane] * args.batch, pair)
        control_disagreements = [
            int((control[k] != base_argmax).sum()) for k in range(args.batch)
        ]
        if any(control_disagreements):
            raise rp1.Rp1Error(
                f"pair {pair}: SegNet is batch-dependent on this host "
                f"(batch-{args.batch} argmax differs from batch-1 at "
                f"{control_disagreements} cells); neutrality cannot be measured this way"
            )

        select = cand["frame"] == pair
        # sj1's own pre-distortion edits are excluded from PROPOSALS: reverting one is a
        # rate move whose whole purpose was a seg move, so it is non-neutral by
        # construction and would only burn realize cycles.  They are COUNTED so the
        # denominator stays honest.
        is_edit = cand["is_sj1_edit"][select].astype(bool)
        idx = np.flatnonzero(select)
        order = idx[np.argsort(-saving[idx])]
        order = order[~cand["is_sj1_edit"][order].astype(bool)]
        rank_of = {int(j): int(k) for k, j in enumerate(order)}
        chosen = select_proposals(order, args.sample_mode, args.top_k, rng)

        tested: list[dict[str, Any]] = []
        neutral_idx: list[int] = []
        if args.accept_mode == "bisect":
            proposals = [
                (int(cand["pos"][j]), int(cand["best"][j])) for j in chosen
            ]
            accepted_pairs, bisect_stats = accept_by_bisection(
                inst, pair, plane, base_argmax, proposals,
                max_verifies=args.max_verifies,
            )
            accepted_set = set(accepted_pairs)
            for j in chosen:
                key = (int(cand["pos"][j]), int(cand["best"][j]))
                is_in = key in accepted_set
                tested.append(
                    {
                        "pos": key[0],
                        "row": key[0] // EVAL_W,
                        "col": key[0] % EVAL_W,
                        "sym": int(cand["sym"][j]),
                        "best": key[1],
                        "saving_bits": float(saving[j]),
                        "rank_in_pair": rank_of[int(j)],
                        "neighbourhood_agreement": neighbourhood_agreement(
                            plane, key[0], key[1]
                        ),
                        "argmax_cells_changed": 0 if is_in else -1,
                        "neutral": is_in,
                    }
                )
                if is_in:
                    neutral_idx.append(int(j))
            composite = {
                "members": len(neutral_idx),
                "argmax_cells_changed": 0,
                "neutral": True,
                "mode": "bisect",
                **bisect_stats,
            }
            accepted_edits[pair] = [
                (int(cand["pos"][j]), int(cand["best"][j])) for j in neutral_idx
            ]
            row = {
                "pair": pair,
                "accepted": [
                    {"pos": int(cand["pos"][j]), "best": int(cand["best"][j])}
                    for j in neutral_idx
                ],
                "base_flipped_cells": base_flips,
                "segnet_batch_control_disagreements": control_disagreements,
                "candidates_in_pair": int(select.sum()),
                "candidates_available": int(order.size),
                "candidates_that_are_sj1_edits": int(is_edit.sum()),
                "proposals_tested": len(tested),
                "neutral_count": len(neutral_idx),
                "neutral_fraction": (len(neutral_idx) / len(tested)) if tested else 0.0,
                "accepted_saving_bits_first_order": float(
                    sum(saving[j] for j in neutral_idx)
                ),
                "tested_saving_bits_first_order": float(
                    sum(t["saving_bits"] for t in tested)
                ),
                "composite": composite,
                "seconds": time.perf_counter() - pair_started,
                "tested": tested,
            }
            rows.append(row)
            with ledger_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                json.dumps(
                    {
                        "pair": pair,
                        "proposed": len(tested),
                        "accepted": len(neutral_idx),
                        "verifies": bisect_stats["verifies"],
                        "accepted_bits": round(
                            row["accepted_saving_bits_first_order"], 1
                        ),
                        "seconds": round(row["seconds"], 1),
                    }
                ),
                flush=True,
            )
            continue
        for start in range(0, len(chosen), args.batch):
            block = chosen[start : start + args.batch]
            planes = []
            for j in block:
                variant = plane.copy()
                variant.reshape(-1)[int(cand["pos"][j])] = np.uint8(cand["best"][j])
                planes.append(variant)
            argmaxes = inst.argmax_batch(planes, pair)
            for k, j in enumerate(block):
                changed = int((argmaxes[k] != base_argmax).sum())
                tested.append(
                    {
                        "pos": int(cand["pos"][j]),
                        "row": int(cand["pos"][j]) // EVAL_W,
                        "col": int(cand["pos"][j]) % EVAL_W,
                        "sym": int(cand["sym"][j]),
                        "best": int(cand["best"][j]),
                        "saving_bits": float(saving[j]),
                        "rank_in_pair": rank_of[int(j)],
                        "neighbourhood_agreement": neighbourhood_agreement(
                            plane, int(cand["pos"][j]), int(cand["best"][j])
                        ),
                        "argmax_cells_changed": changed,
                        "neutral": changed == 0,
                    }
                )
                if changed == 0:
                    neutral_idx.append(int(j))

        # THE ACCEPTANCE: the set, realized together.  Members verified alone can
        # interact; the object that ships is the set, so the set is what is verified.
        composite = {"members": len(neutral_idx)}
        if neutral_idx:
            variant = plane.copy()
            flat = variant.reshape(-1)
            for j in neutral_idx:
                flat[int(cand["pos"][j])] = np.uint8(cand["best"][j])
            comp_argmax = inst.argmax_batch([variant], pair)[0]
            comp_changed = int((comp_argmax != base_argmax).sum())
            composite["argmax_cells_changed"] = comp_changed
            composite["neutral"] = comp_changed == 0
            if comp_changed == 0:
                pass
            else:
                # Fall back to a greedy re-accumulation, verifying after each add.  This
                # costs one realize per member but cannot ship an unverified set.
                keep: list[int] = []
                cur = plane.copy()
                for j in neutral_idx:
                    trial = cur.copy()
                    trial.reshape(-1)[int(cand["pos"][j])] = np.uint8(cand["best"][j])
                    trial_argmax = inst.argmax_batch([trial], pair)[0]
                    if int((trial_argmax != base_argmax).sum()) == 0:
                        cur = trial
                        keep.append(j)
                composite["greedy_members"] = len(keep)
                composite["greedy_used"] = True
                neutral_idx = keep
        else:
            composite["argmax_cells_changed"] = 0
            composite["neutral"] = True

        accepted_edits[pair] = [
            (int(cand["pos"][j]), int(cand["best"][j])) for j in neutral_idx
        ]
        row = {
            "pair": pair,
            "accepted": [
                {"pos": int(cand["pos"][j]), "best": int(cand["best"][j])}
                for j in neutral_idx
            ],
            "base_flipped_cells": base_flips,
            "segnet_batch_control_disagreements": control_disagreements,
            "candidates_in_pair": int(select.sum()),
            "candidates_available": int(order.size),
            "candidates_that_are_sj1_edits": int(is_edit.sum()),
            "proposals_tested": len(tested),
            "neutral_count": len(neutral_idx),
            "neutral_fraction": (len(neutral_idx) / len(tested)) if tested else 0.0,
            "accepted_saving_bits_first_order": float(
                sum(saving[j] for j in neutral_idx)
            ),
            "tested_saving_bits_first_order": float(
                sum(t["saving_bits"] for t in tested)
            ),
            "composite": composite,
            "seconds": time.perf_counter() - pair_started,
            "tested": tested,
        }
        rows.append(row)
        with ledger_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "pair": pair,
                    "tested": len(tested),
                    "neutral": len(neutral_idx),
                    "neutral_fraction": round(row["neutral_fraction"], 4),
                    "accepted_bits": round(row["accepted_saving_bits_first_order"], 1),
                    "seconds": round(row["seconds"], 1),
                }
            ),
            flush=True,
        )

    # THE PRICING FIELD CARRIES ALL 600 PLANES.  ``jg2.apply_edits`` splices planes onto
    # the PRISTINE cl2 field, so a pair merely ABSENT from this npz reverts all the way
    # back and silently undoes every edit the live pointer banked -- a valid field, an
    # invisible loss.  sj1 measured that exact revert at 2,337 tokens over 230 pairs
    # (commit 42d5fc651).  So the field is written whole and CHECKED whole.
    field = np.array(inst.live_field, dtype=np.uint8)
    total_edits = 0
    for pair, edits in accepted_edits.items():
        flat = field[pair].reshape(-1)
        for pos, new_class in edits:
            flat[pos] = np.uint8(new_class)
        total_edits += len(edits)
    changed_vs_live = int((field != inst.live_field).sum())
    if changed_vs_live != total_edits:
        raise rp1.Rp1Error(
            f"pricing field differs from the live field at {changed_vs_live} tokens but "
            f"{total_edits} edits were accepted -- the field is not the one measured"
        )
    field_path = out / "field_rp1_sizing.npz"
    np.savez_compressed(
        field_path, **{str(p): field[p] for p in range(jg1.N_PAIRS)}
    )

    tested_total = sum(r["proposals_tested"] for r in rows)
    neutral_total = sum(r["neutral_count"] for r in rows)
    accepted_bits = sum(r["accepted_saving_bits_first_order"] for r in rows)
    all_tested = [t for r in rows for t in r["tested"]]
    neutral_savings = [t["saving_bits"] for t in all_tested if t["neutral"]]
    refused_savings = [t["saving_bits"] for t in all_tested if not t["neutral"]]

    _cc_bits, _cc_detail = _coverage_corrected_bits_per_pair(rows, RANK_EDGES)
    summary = {
        "schema": "ddm_rp1_sizing.v1",
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
        "scope_reduction": (
            "SCOPE: pair sample, top-K per pair.  Token changes are pair-confined on the "
            "seg and pose legs, so a pair sample SIZES the pass (rw1 screening law); the "
            "RATE leg is not pair-confined and is priced only by a full 600-frame encode."
        ),
        "pointer": pointer,
        "rank_dir": str(rank_dir),
        "pairs": [int(p) for p in pairs],
        "seed": args.seed,
        "top_k_per_pair": args.top_k,
        "proposals_tested": tested_total,
        "neutral_count": neutral_total,
        "neutral_fraction": neutral_total / tested_total if tested_total else 0.0,
        "accepted_saving_bits_first_order": accepted_bits,
        "accepted_saving_bytes_first_order": accepted_bits / 8.0,
        "mean_saving_bits_neutral": float(np.mean(neutral_savings))
        if neutral_savings
        else 0.0,
        "mean_saving_bits_refused": float(np.mean(refused_savings))
        if refused_savings
        else 0.0,
        "sample_mode": args.sample_mode,
        "neutrality_by_rank_decile": _by_bucket(
            all_tested, "rank_in_pair", [float(e) for e in RANK_EDGES]
        ),
        "neutrality_by_saving_bits": _by_bucket(
            all_tested, "saving_bits", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 1e9]
        ),
        "neutrality_by_neighbourhood_agreement": _by_bucket(
            all_tested, "neighbourhood_agreement", [0.0, 0.5, 0.8, 0.95, 0.99, 1.01]
        ),
        "projection_a_sampled_bytes_n600": (
            accepted_bits / 8.0 * jg1.N_PAIRS / max(1, len(rows))
        ),
        "projection_b_coverage_corrected_bytes_n600": _cc_bits * jg1.N_PAIRS / 8.0,
        "projection_b_strata": _cc_detail,
        "projection_b_delta_S_if_realized": (
            -_cc_bits * jg1.N_PAIRS / 8.0 * rp1.S_PER_BYTE
        ),
        "stop_rule": {
            "threshold_bytes": 300,
            "binds_on": "projection_b_coverage_corrected_bytes_n600",
            "verdict": (
                "STOP"
                if _cc_bits * jg1.N_PAIRS / 8.0 < 300
                else "CONTINUE_TO_N600"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "pricing_field": {
            "path": str(field_path),
            "planes": jg1.N_PAIRS,
            "tokens_changed_vs_live_field": changed_vs_live,
            "tokens_changed_vs_pristine": int(
                (field != inst.base_field).sum()
            ),
        },
    }
    (out / "SIZING.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sizing = sub.add_parser("sizing")
    sizing.add_argument("--rank-dir", required=True)
    sizing.add_argument("--out", required=True)
    sizing.add_argument("--pairs", type=int, default=12)
    sizing.add_argument("--pair-list", default="")
    sizing.add_argument("--seed", type=int, default=20260909)
    sizing.add_argument("--top-k", type=int, default=32)
    sizing.add_argument("--batch", type=int, default=4)
    sizing.add_argument(
        "--sample-mode", choices=("top", "stratified"), default="stratified"
    )
    sizing.add_argument("--threads", type=int, default=3)
    sizing.add_argument(
        "--accept-mode",
        choices=("singles", "bisect"),
        default="singles",
        help="singles measures the per-proposal neutral fraction; bisect maximizes the "
        "accepted set per realization and is the n600 mode",
    )
    sizing.add_argument("--max-verifies", type=int, default=400)
    sizing.add_argument("--shards", type=int, default=1)
    sizing.add_argument("--shard", type=int, default=0)
    sizing.add_argument("--resume", action="store_true")
    sizing.set_defaults(func=cmd_sizing)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
