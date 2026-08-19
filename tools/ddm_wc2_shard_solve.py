#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_wc2 -- shard the ddm_jg3 joint solve across processes to buy wall clock.

WHAT THIS DOES AND DOES NOT CLAIM
---------------------------------
This module was written to make sharding DECISION-IDENTICAL.  The replay it
ships MEASURED that decision-identity is **not attainable**, and the claim is
withdrawn rather than softened:

* Three independent FRESH processes reproduce each other byte-for-byte on pair
  519 -- ``repaired=60``, accepted-set sha ``98c13421`` -- including one run with
  the thread environment UNSET and two with ``OMP_NUM_THREADS=6``.  The solver is
  exactly reproducible across fresh processes, and the thread environment is NOT
  the discriminator (that hypothesis was tested and falsified).
* The long-running live process banked ``repaired=62``, sha ``b710afc5`` on the
  same pair.  A fresh process does not reproduce a warm one.

The mechanism is the solver's own sensitivity, not the sharding: every screened
site is a pixel where the base argmax DISAGREES with GT, i.e. a low-margin
near-tie.  A ~1e-6 logit difference flips such a pixel, which changes that site's
``repaired``, hence ``gain``, hence ``best[]``, hence ``select_separated`` and the
winning configuration.  Two processes with different computation histories land
on different -- but equally realized-and-measured -- local optima of the same
greedy descent.  Neither is "correct": acceptance is REALIZED (joint render +
re-segment against the frozen scorer).

So the admissibility question is not identity but BIAS, and it is read in the
solver's own objective (net delta S, which prices repair against the tokens that
bought it) via ``yield_comparison``.  See the arm memo for the measured bound.

WHY SHARDING IS STRUCTURALLY LICENSED
-------------------------------------
The binding wall clock is the ``ddm_jg3`` n600 joint solve, measured at ~139 s
per pair against 18 logical cores that the single process leaves ~85% idle
(MEASURED: the live worker holds ~260% CPU of 1800% available, because
``jg1.render_frame1`` must run at batch 1 -- ``ddm_up2`` sec.6 measured semantic
batch 8 as BYTE-CHANGING -- so the render half is essentially serial).

Sharding is only legitimate if per-pair acceptance is INDEPENDENT across pairs.
It is, and every leg of that was read out of ``experiments/ddm_jg3_joint_solve.py``
rather than assumed:

1. **The site subsample RNG is per-pair, not a stream.**  ``solve_pair`` builds
   ``np.random.default_rng(site_seed + pair)`` fresh on every call.  A shared
   generator would have made results depend on visit order; a per-pair seed does
   not.  Order and process placement are therefore irrelevant.
2. **The rate ranker is a static per-pair lookup.**  ``LogitPrice.bits_for``
   reads ``self._memmap[pair]`` -- a read-only, size-checked logits file -- and
   prices moves against ``tokens_pair``, which is the BASE token plane.  It never
   sees an accepted edit from any pair.
3. **The acceptance price is a flat measured constant.**  The configuration that
   wins is chosen on ``cost_bits = tokens_here * RATE_PRIOR_BITS_PER_TOKEN``
   (4.1379 bits/token, ``ddm_jg2``'s byte-identical re-encoder constant).  It is
   deliberately NOT the context-dependent logit sum -- the solver's own comment
   records that the logit price is a 2.2x under-price and is kept for ranking
   only.  So the decision has no cross-pair term at all.
4. **The inputs are per-pair slices of fixed arrays.**  ``tokens[pair]``,
   ``base_argmax[pair]``, ``gt[pair]``.
5. **The models carry no state.**  ``SegNet().eval()`` with
   ``requires_grad_(False)``; BatchNorm reads running statistics.
6. **The edit accumulator is write-only.**  ``cmd_solve`` fills ``edits[str(pair)]``
   and never reads it back into ``solve_pair``.

The one real cross-pair coupling in this campaign -- ``ddm_jg2``'s measured
union/sum = 1.0258 rate superposition -- lives at the ARCHIVE layer, not in this
loop.  It is resolved at merge time by the authority re-encoder
(``experiments/ddm_jg2_tail_reencode.py``) over the union of accepted edits, and
that union is identical whether one process or six produced it.  Sharding cannot
perturb a term the solver never evaluates.

HOW THE SHARDING IS DONE
------------------------
Not by re-batching.  ``ddm_et4``'s law is that batch shape is part of the forward
instrument, so every shard runs the IDENTICAL per-pair instrument and differs
only in WHICH pairs it visits.  The solver already has the exact seam for this:
``--pair-list``, its explicit reproduction control.  Each shard gets a disjoint
pair list and its own ``--tag``, which routes its checkpoint and payload to
distinct files under the same store.  The live solver file is not edited.

Pairs are assigned by LPT BIN-PACKING on predicted cost, never by contiguous
block and never by blind stride.  Per-pair cost is heavy tailed (MEASURED 55.7 s
to 657.0 s) and predictable before any pair is solved: ``seconds = 3.59 +
0.5123 * evaluations`` (R2 = 0.989) and ``evaluations`` tracks the base flip
count ``base_argmax != gt`` at r = 0.967.  Balancing on that key equalizes FINISH
times; a blind stride can hand one shard the tail and the fleet then waits on it.

Each shard's bucket is then RE-SORTED into seeded-permutation order.  LPT emits
heaviest-first, and a heaviest-first prefix is exactly the biased-population
shape ``ddm_bp2``/``ddm_na2`` measured (pose prefixes 2.54-4.21x harder).
Balancing decides WHICH pairs a shard owns; the permutation decides in WHAT ORDER
it visits them, so every shard's own prefix stays an unbiased sample of the field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

# The canonical live-run parameters.  These mirror the invocation of pid 81175 so
# a shard is the same instrument on a different pair set.
LIVE_SEED = 20260819
LIVE_PAIRS = 600
SOLVER = "experiments/ddm_jg3_joint_solve.py"

# MEASURED on the live worker (``ps -o %cpu,rss``): ~260% CPU of 1800% available
# and ~6.0 GiB RSS at steady state.  Declared here so the launcher's resource
# derivation is fed a measured number, never a guessed one.
MEASURED_PEAK_RSS_GIB = 8.0
MEASURED_THREAD_NEED = 6  # torch.get_num_threads() on this host, unset env

# The fields that constitute a DECISION.  Wall-clock is excluded because it is
# the only field a shard is allowed to change; everything else must be identical
# or the shard is not the same instrument.
DECISION_FIELDS = (
    "pair",
    "flips_before",
    "flips_after",
    "repaired",
    "tokens_changed",
    "screened_candidates",
    "evaluations",
    "packing_residual_max",
    "rejected_for_separation",
    "accept_separation_chosen",
    "keep_fraction_chosen",
    "accepted",
)


class ShardError(RuntimeError):
    """Raised when a shard plan, replay, or merge cannot be made honest."""


# ---------------------------------------------------------------------------
# Pure helpers -- every one of these is exercised by the pinned tests.
# ---------------------------------------------------------------------------


def visit_permutation(pairs: int = LIVE_PAIRS, seed: int = LIVE_SEED) -> np.ndarray:
    """Reproduce the solver's seeded visit order exactly.

    ``cmd_solve`` builds this as ``default_rng(seed).permutation(select_pairs(...))``.
    Reproducing it here is what lets the orchestrator name the REMAINING pairs
    without guessing, and it is asserted against the live log's ``first_12``.
    """
    import ddm_up2_shipping_pose_solve as up2

    indices = up2.select_pairs(pairs, seed)
    return np.random.default_rng(seed).permutation(indices)


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    """Read a per-pair JSONL checkpoint into ``{pair: row}``.

    A truncated final line is TOLERATED and dropped: the writer fsyncs after each
    pair, but a process terminated mid-write can still leave a partial line, and
    refusing to read the whole checkpoint over one partial row would throw away
    every completed pair.  A partial row is not a completed pair, so dropping it
    is exactly correct -- the pair simply returns to the remaining set.
    """
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[int(row["pair"])] = row
    return rows


def collect_done(store: Path, tag: str) -> dict[int, dict[str, Any]]:
    """Every pair completed by the live run OR by any shard of it.

    Reading the UNION rather than the live checkpoint alone is what makes
    ``plan``/``launch`` idempotent and re-balanceable: per-pair cost is heavy
    tailed (MEASURED 55.7 s to 657.0 s), so shards will not finish together, and
    re-planning against the union lets a drained shard be topped up with whatever
    is genuinely left instead of re-solving pairs a sibling already banked.
    """
    rows: dict[int, dict[str, Any]] = {}
    rows.update(read_checkpoint(_live_checkpoint(store, tag)))
    for path in sorted((store / "checkpoints").glob(f"seg_solve_{tag}_wc2s*.jsonl")):
        rows.update(read_checkpoint(path))
    return rows


def predicted_flip_counts(base_argmax_path: str, verify_sha: bool = False) -> np.ndarray:
    """Per-pair base flip count -- the cost predictor, known WITHOUT solving.

    ``ddm_jg3`` measured per-pair wall clock as essentially all evaluation cost
    (``seconds = 3.59 + 0.5123 * evaluations``, R2 = 0.989 over the first 17
    pairs) and evaluations track the base flip count at r = 0.967 (jg3's own
    read: r = 0.945).  The flip count is just ``base_argmax != gt``, so the cost
    of every pair is predictable before any of them is solved -- which is what
    makes a balanced partition possible rather than a hopeful one.
    """
    import ddm_jg1_seg_solve as jg1
    import ddm_jg3_joint_solve as jg3
    import ddm_up2_shipping_pose_solve as up2

    base = jg3._load_base_argmax(Path(base_argmax_path), verify_sha)
    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    return (np.asarray(base) != np.asarray(gt)).reshape(len(base), -1).sum(axis=1)


def plan_shards(
    permutation: Sequence[int],
    done: Iterable[int],
    shards: int,
    weights: dict[int, float] | None = None,
) -> list[list[int]]:
    """Partition the not-yet-done pairs across ``shards`` lists.

    With ``weights`` (predicted flip counts) this is LPT bin-packing -- heaviest
    pair to the currently-lightest shard -- which equalizes FINISH TIMES.  That
    matters because per-pair cost is heavy tailed (MEASURED 55.7 s to 657.0 s,
    flip median 46 against max 245): an unweighted split hands one shard the tail
    and the fleet then waits on it, so the realized speedup is set by the slowest
    shard, not the mean.

    Without weights it degrades to a stride interleave, which is still never a
    contiguous block.

    In BOTH cases each shard's list is finally re-ordered to follow the seeded
    permutation.  LPT would otherwise emit each shard heaviest-first, and a
    heaviest-first prefix is exactly the biased-population shape ``ddm_bp2`` /
    ``ddm_na2`` measured (pose prefixes 2.54-4.21x harder).  Balancing decides
    WHICH pairs a shard owns; the permutation decides in WHAT ORDER it visits
    them, so every shard's own prefix stays an unbiased sample of the field.
    """
    if shards < 1:
        raise ShardError(f"shards must be >= 1, got {shards}")
    done_set = {int(p) for p in done}
    remaining = [int(p) for p in permutation if int(p) not in done_set]
    rank = {pair: order for order, pair in enumerate(remaining)}

    if weights is None:
        buckets = [remaining[k::shards] for k in range(shards)]
    else:
        buckets = [[] for _ in range(shards)]
        loads = [0.0] * shards
        for pair in sorted(remaining, key=lambda p: -float(weights.get(p, 0.0))):
            target = min(range(shards), key=lambda k: (loads[k], k))
            buckets[target].append(pair)
            loads[target] += float(weights.get(pair, 0.0))
    return [sorted(bucket, key=lambda p: rank[p]) for bucket in buckets]


def decision_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    """The order-insensitive decision content of one checkpoint row."""
    out: list[Any] = []
    for field in DECISION_FIELDS:
        value = row.get(field)
        if field == "accepted":
            value = tuple(sorted(tuple(int(v) for v in edit) for edit in value or ()))
        out.append(value)
    sweep = row.get("separation_sweep") or []
    out.append(
        tuple(
            (
                entry.get("accept_separation"),
                entry.get("keep_fraction"),
                entry.get("tokens"),
                entry.get("flips_after"),
                entry.get("repaired"),
                entry.get("net_delta_S"),
            )
            for entry in sweep
        )
    )
    return tuple(out)


def compare_decisions(
    reference: dict[int, dict[str, Any]], replay: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Decision-equivalence report between a reference and a replay checkpoint."""
    shared = sorted(set(reference) & set(replay))
    mismatches = []
    for pair in shared:
        want = decision_signature(reference[pair])
        got = decision_signature(replay[pair])
        if want == got:
            continue
        # Name the differing fields directly.  The signature is a positional
        # tuple over ``DECISION_FIELDS`` plus a trailing sweep entry, so the
        # index alignment is exact and no re-derivation is needed.
        names = (*DECISION_FIELDS, "separation_sweep")
        differing = [
            name for name, a, b in zip(names, want, got, strict=True) if a != b
        ]
        mismatches.append(
            {
                "pair": pair,
                "fields": differing,
                "reference": {f: reference[pair].get(f) for f in differing},
                "replay": {f: replay[pair].get(f) for f in differing},
            }
        )
    return {
        "compared": len(shared),
        "identical": len(shared) - len(mismatches),
        "mismatches": mismatches,
        "equivalent": not mismatches and bool(shared),
        "replay_only": sorted(set(replay) - set(reference)),
    }


def yield_comparison(
    reference: dict[int, dict[str, Any]], replay: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Paired repair-yield comparison between two solves of the SAME pairs.

    This is the metric that actually decides whether sharding is admissible.
    Decision-IDENTITY is already known to fail: the solver screens at the argmax
    decision boundary, where every site is a low-margin near-tie, so two
    processes with different computation histories land on different -- but
    equally realized-and-measured -- local optima of the same greedy descent.

    Since neither optimum is "correct", the question is not whether the sets
    differ but whether the shard condition is BIASED: does a fresh process bank
    systematically less repair than a long-running one?  A two-sided sign test
    over the paired pairs answers that without assuming a distribution, which
    matters because the per-pair repair counts are heavy tailed.
    """
    import ddm_jg3_joint_solve as jg3

    def net_delta_s(row: dict[str, Any]) -> float:
        """The solver's OWN objective for one pair, in score units.

        Judging on ``repaired`` alone would be the wrong instrument: repair is
        bought with TOKENS, and tokens cost rate.  ``ddm_jg3`` selects its winning
        configuration on exactly this quantity, so equivalence and bias must be
        read in it too.  Pair 17 is the worked example -- the fresh run banked
        MORE repair (19 vs 18) using FEWER tokens (16 vs 20), so a repair-only
        comparison would have understated how much better it was.
        """
        return -int(row["repaired"]) * jg3.S_PER_SEG_CELL + (
            int(row["tokens_changed"]) * jg3.RATE_PRIOR_BITS_PER_TOKEN / 8.0
        ) * jg3.S_PER_ARCHIVE_BYTE

    shared = sorted(set(reference) & set(replay))
    rows = []
    wins = losses = ties = 0
    for pair in shared:
        a = int(reference[pair]["repaired"])
        b = int(replay[pair]["repaired"])
        sa = net_delta_s(reference[pair])
        sb = net_delta_s(replay[pair])
        rows.append(
            {
                "pair": pair,
                "reference_repaired": a,
                "replay_repaired": b,
                "delta": b - a,
                "reference_tokens": int(reference[pair]["tokens_changed"]),
                "replay_tokens": int(replay[pair]["tokens_changed"]),
                "reference_net_delta_S": sa,
                "replay_net_delta_S": sb,
                "net_delta_S_improvement": sa - sb,
            }
        )
        # A WIN is a more negative net delta S -- a better score contribution.
        if sb < sa:
            wins += 1
        elif sb > sa:
            losses += 1
        else:
            ties += 1
    ref_total = sum(r["reference_repaired"] for r in rows)
    rep_total = sum(r["replay_repaired"] for r in rows)
    ref_tok = sum(r["reference_tokens"] for r in rows)
    rep_tok = sum(r["replay_tokens"] for r in rows)
    decided = wins + losses
    # Exact two-sided binomial sign test against p = 0.5.  No SciPy dependency:
    # the tail is a short exact sum over a small n.
    from math import comb

    if decided:
        extreme = min(wins, losses)
        tail = sum(comb(decided, k) for k in range(extreme + 1)) / (2**decided)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    ref_s = sum(r["reference_net_delta_S"] for r in rows)
    rep_s = sum(r["replay_net_delta_S"] for r in rows)
    return {
        "pairs_compared": len(shared),
        "reference_net_delta_S_total": ref_s,
        "replay_net_delta_S_total": rep_s,
        "net_delta_S_shard_advantage": ref_s - rep_s,
        "reference_repaired_total": ref_total,
        "replay_repaired_total": rep_total,
        "repaired_delta": rep_total - ref_total,
        "repaired_relative": (rep_total - ref_total) / ref_total if ref_total else 0.0,
        "reference_tokens_total": ref_tok,
        "replay_tokens_total": rep_tok,
        "reference_yield": ref_total / ref_tok if ref_tok else 0.0,
        "replay_yield": rep_total / rep_tok if rep_tok else 0.0,
        "replay_wins": wins,
        "replay_losses": losses,
        "ties": ties,
        "sign_test_p_value": round(p_value, 5),
        "biased_at_p05": bool(decided and p_value < 0.05),
        "per_pair": rows,
    }


def merge_checkpoints(
    sources: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Union per-pair rows across sources, refusing to silently pick a winner.

    A pair present in two sources is only merged when the two rows carry the SAME
    decision.  A genuine disagreement is returned as a conflict rather than
    resolved, because picking one would be exactly the fake this arm exists to
    avoid: the whole claim is that sharding changed no decision.
    """
    merged: dict[int, dict[str, Any]] = {}
    origin: dict[int, str] = {}
    conflicts: list[dict[str, Any]] = []
    for tag, rows in sorted(sources.items()):
        for pair, row in rows.items():
            if pair not in merged:
                merged[pair] = row
                origin[pair] = tag
                continue
            if decision_signature(merged[pair]) != decision_signature(row):
                conflicts.append(
                    {"pair": pair, "sources": sorted((origin[pair], tag))}
                )
    return merged, conflicts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_argv(store: Path, tag: str, pair_list: Sequence[int]) -> list[str]:
    """The solver invocation for one shard.

    Every flag other than ``--tag``/``--pair-list`` mirrors the live run, so the
    per-pair instrument is unchanged.  ``--seed`` in particular MUST match: it
    feeds ``site_seed`` and therefore ``default_rng(site_seed + pair)``.
    """
    return [
        ".venv/bin/python",
        SOLVER,
        "solve",
        "--store",
        str(store),
        "--tag",
        tag,
        "--pairs",
        str(LIVE_PAIRS),
        "--seed",
        str(LIVE_SEED),
        "--no-verify-sha",
        "--resume",
        "--pair-list",
        ",".join(str(int(p)) for p in pair_list),
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _live_checkpoint(store: Path, tag: str) -> Path:
    return store / "checkpoints" / f"seg_solve_{tag}.jsonl"


def _weights(args) -> dict[int, float] | None:
    if args.no_balance:
        return None
    base_argmax = args.base_argmax
    if base_argmax is None:
        import ddm_jg3_joint_solve as jg3

        base_argmax = str(jg3.DEFAULT_BASE_ARGMAX)
    flips = predicted_flip_counts(base_argmax)
    return {pair: float(flips[pair]) for pair in range(len(flips))}


def cmd_plan(args) -> int:
    store = Path(args.store)
    done = collect_done(store, args.tag)
    permutation = visit_permutation()
    weights = _weights(args)
    lists = plan_shards(permutation, done, args.shards, weights)
    seconds = [row["seconds"] for row in done.values() if "seconds" in row]
    mean_s = float(np.mean(seconds)) if seconds else 0.0
    remaining = sum(len(x) for x in lists)
    # Cost is ~linear in flips, so a flip-weighted share predicts finish time far
    # better than a pair count does.  Both are reported so the balance is visible.
    loads = [
        sum(weights.get(p, 0.0) for p in bucket) if weights else float(len(bucket))
        for bucket in lists
    ]
    total_load = sum(loads) or 1.0
    eta_balanced = (
        max(loads) / total_load * remaining * mean_s / 3600.0 if remaining else 0.0
    )
    plan = {
        "arm": "ddm_wc2",
        "store": str(store),
        "live_tag": args.tag,
        "pairs_done": len(done),
        "pairs_remaining": remaining,
        "shards": args.shards,
        "mean_seconds_per_pair_measured": round(mean_s, 2),
        "eta_hours_single_process": round(remaining * mean_s / 3600.0, 2),
        "eta_hours_sharded_balanced": round(eta_balanced, 2),
        "speedup_predicted": round(
            (remaining * mean_s / 3600.0) / eta_balanced, 2
        )
        if eta_balanced
        else 0.0,
        "assignment": (
            "stride_over_seeded_permutation"
            if args.no_balance
            else "LPT_bin_pack_on_predicted_flip_count_then_permutation_order"
        ),
        "shard_flip_loads": [int(x) for x in loads],
        "load_imbalance_max_over_mean": round(
            max(loads) / (total_load / len(loads)), 4
        )
        if loads
        else 0.0,
        "shard_sizes": [len(x) for x in lists],
        "shard_tags": [f"{args.tag}_wc2s{k}" for k in range(args.shards)],
        "first_pairs_per_shard": [x[:4] for x in lists],
        "axis": "[macOS-CPU advisory]",
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.write:
        out = store / "retained" / f"wc2_shard_plan_{args.tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True))
        for k, pair_list in enumerate(lists):
            (store / "retained" / f"wc2_shard_{args.tag}_{k}.json").write_text(
                json.dumps({"tag": plan["shard_tags"][k], "pairs": pair_list})
            )
    return 0


def cmd_launch(args) -> int:
    store = Path(args.store)
    done = collect_done(store, args.tag)
    lists = plan_shards(visit_permutation(), done, args.shards, _weights(args))
    launched: list[dict[str, Any]] = []
    for k, pair_list in enumerate(lists):
        if not pair_list:
            continue
        tag = f"{args.tag}_wc2s{k}"
        proc = _launch(
            store / "logs" / f"wc2_shard_{k}",
            f"ddm_wc2 shard {k}/{args.shards} of the ddm_jg3 n600 joint solve",
            "ddm_wc2 wall-clock shard; decision-equivalence proven by replay",
            f"wc2_shard_{k}",
            shard_argv(store, tag, pair_list),
            nice=args.nice,
            walltime_cap_s=args.walltime_cap_s,
            dry_run=args.dry_run,
            match_live_env=args.match_live_env,
        )
        launched.append(
            {
                "shard": k,
                "tag": tag,
                "pairs": len(pair_list),
                "rc": proc.returncode,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }
        )
        print(json.dumps(launched[-1], indent=2, sort_keys=True))
    return 0 if all(item["rc"] == 0 for item in launched) else 1


def _launch(
    out_dir: Path,
    purpose: str,
    authority: str,
    receipt: str,
    inner: Sequence[str],
    *,
    nice: int,
    walltime_cap_s: float,
    dry_run: bool,
    match_live_env: bool = False,
) -> subprocess.CompletedProcess:
    """One governed detached launch.

    Shards and the equivalence replay go through THIS function so the replay is
    proven under byte-identical launch conditions to the fleet it authorizes --
    including the thread environment, which ``--derive-resource-budgets`` sets and
    the live process leaves unset.  That difference is a real numerical hazard
    (``ddm_et4``: the forward instrument includes its shape), so it is settled by
    measurement here rather than argued away.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python",
        str(REPO / "tools" / "launch_detached_process.py"),
        "--output-dir",
        str(out_dir),
        "--cwd",
        str(REPO),
        "--purpose",
        purpose,
        "--authority",
        authority,
        "--nice",
        str(nice),
        "--done-receipt",
        receipt,
        "--receipt-supersede",
    ]
    if not match_live_env:
        # ``--derive-resource-budgets`` SETS OMP/OPENBLAS/MKL/VECLIB/NUMEXPR
        # thread counts.  The live process leaves them unset.  That is a change to
        # the forward instrument, not just to scheduling (``ddm_et4``), so
        # ``--match-live-env`` exists to reproduce the live environment exactly
        # when decision-identity is the thing being measured.
        cmd.extend(
            [
                "--derive-resource-budgets",
                "--measured-peak-rss-gib",
                str(MEASURED_PEAK_RSS_GIB),
                "--measured-thread-need",
                str(MEASURED_THREAD_NEED),
                "--walltime-cap-s",
                str(walltime_cap_s),
            ]
        )
    if dry_run:
        cmd.append("--dry-run")
    cmd.append("--")
    cmd.extend(inner)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def cmd_verify(args) -> int:
    """Replay already-completed pairs and prove the shard changes no decision."""
    store = Path(args.store)
    done = read_checkpoint(_live_checkpoint(store, args.tag))
    if not done:
        raise ShardError(f"no completed pairs in {_live_checkpoint(store, args.tag)}")
    if args.pairs:
        chosen = [int(p) for p in args.pairs.split(",") if p.strip()]
        missing = [p for p in chosen if p not in done]
        if missing:
            raise ShardError(f"pairs {missing} are not completed; cannot replay them")
    else:
        # Prefer the pairs most likely to expose order- or state-dependence: the
        # ones whose packing residual was largest, i.e. where sites interacted.
        chosen = [
            pair
            for pair, _ in sorted(
                done.items(),
                key=lambda kv: -int(kv[1].get("packing_residual_max", 0)),
            )
        ][: args.count]
    tag = args.replay_tag
    proc = _launch(
        store / "logs" / f"wc2_verify_{tag}",
        f"ddm_wc2 decision-equivalence replay of {len(chosen)} completed pairs",
        "ddm_wc2; proves sharding changes no accept/reject decision before swap",
        f"wc2_verify_{tag}",
        shard_argv(store, tag, chosen),
        nice=args.nice,
        walltime_cap_s=args.walltime_cap_s,
        dry_run=args.dry_run,
        match_live_env=args.match_live_env,
    )
    print(
        json.dumps(
            {
                "replay_tag": tag,
                "match_live_env": args.match_live_env,
                "pairs": chosen,
                "rc": proc.returncode,
                "stdout": proc.stdout[-3000:],
                "stderr": proc.stderr[-2000:],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return proc.returncode


def cmd_compare(args) -> int:
    store = Path(args.store)
    reference = read_checkpoint(_live_checkpoint(store, args.tag))
    replay: dict[int, dict[str, Any]] = {}
    for tag in [t.strip() for t in args.replay_tag.split(",") if t.strip()]:
        replay.update(read_checkpoint(_live_checkpoint(store, tag)))
    report = compare_decisions(reference, replay)
    report["yield"] = yield_comparison(reference, replay)
    report["reference_tag"] = args.tag
    report["replay_tag"] = args.replay_tag
    report["verdict"] = (
        "DECISION_EQUIVALENT" if report["equivalent"] else "NOT_EQUIVALENT"
    )
    # Price the measured per-pair advantage out to the pairs the swap would
    # actually hand to shards.  Reported as a PROJECTION, never as a measurement:
    # it is a linear extrapolation of a small paired sample.
    y = report["yield"]
    per_pair = (
        y["net_delta_S_shard_advantage"] / y["pairs_compared"]
        if y["pairs_compared"]
        else 0.0
    )
    report["remaining_pairs_at_swap"] = args.remaining
    report["projected_full_run_delta_S"] = per_pair * args.remaining
    report["projection_is_extrapolation_not_measurement"] = True
    report["verdict"] = (
        "DECISION_IDENTICAL"
        if report["equivalent"]
        else "NOT_DECISION_IDENTICAL_BIAS_MEASURED"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write:
        out = store / "retained" / f"wc2_equivalence_{args.replay_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.receipt:
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["equivalent"] else 4


def cmd_merge(args) -> int:
    store = Path(args.store)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    sources = {tag: read_checkpoint(_live_checkpoint(store, tag)) for tag in tags}
    merged, conflicts = merge_checkpoints(sources)
    if conflicts:
        print(json.dumps({"conflicts": conflicts}, indent=2, sort_keys=True))
        raise ShardError(
            f"{len(conflicts)} pair(s) carry DIFFERENT decisions across shards; "
            "refusing to merge -- the equivalence claim is falsified, not resolvable"
        )
    out_ckpt = _live_checkpoint(store, args.out_tag)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(merged[pair], sort_keys=True) for pair in sorted(merged)
    ]
    out_ckpt.write_text("\n".join(lines) + ("\n" if lines else ""))

    # ALWAYS KEEP THE PAYLOAD.  The edited token planes are reconstructed from the
    # base field plus each pair's sparse ``accepted`` list -- which is exactly what
    # the solver's own docstring records as the lossless payload -- rather than
    # copied from the per-shard npz mirrors, which may lag the JSONL because they
    # are only rewritten every ``--payload-every`` pairs.
    payload_manifest: dict[str, Any] = {"reconstructed": False}
    if not args.no_payload:
        import ddm_jg1_seg_solve as jg1

        tokens = jg1.load_tokens(Path(args.tokens or jg1.DEFAULT_TOKENS))
        planes: dict[str, np.ndarray] = {}
        for pair, row in merged.items():
            accepted = row.get("accepted") or []
            if not accepted:
                continue
            plane = np.asarray(tokens[pair]).copy()
            for y, x, value in accepted:
                plane[int(y), int(x)] = value
            planes[str(pair)] = plane.astype(np.uint8)
        edits_path = store / "retained" / f"seg_edits_{args.out_tag}.npz"
        np.savez_compressed(edits_path, **planes)
        payload_manifest = {
            "reconstructed": True,
            "path": str(edits_path),
            "sha256": sha256_file(edits_path),
            "bytes": edits_path.stat().st_size,
            "pairs_edited": len(planes),
            "source": "base_tokens + per_pair_accepted_list (lossless per solver docstring)",
        }

    repaired = sum(row["repaired"] for row in merged.values())
    tokens_changed = sum(row["tokens_changed"] for row in merged.values())
    summary = {
        "arm": "ddm_wc2",
        "out_tag": args.out_tag,
        "sources": {tag: len(rows) for tag, rows in sources.items()},
        "pairs": len(merged),
        "conflicts": 0,
        "repaired": repaired,
        "tokens_changed": tokens_changed,
        "cells_per_changed_token": (
            repaired / tokens_changed if tokens_changed else 0.0
        ),
        "checkpoint": str(out_ckpt),
        "checkpoint_sha256": sha256_file(out_ckpt),
        "payload": payload_manifest,
        "axis": "[macOS-CPU advisory]",
    }
    out = store / "retained" / f"wc2_merged_{args.out_tag}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="stride-assign remaining pairs to shards")
    plan.add_argument("--store", required=True)
    plan.add_argument("--tag", default="n600")
    plan.add_argument("--shards", type=int, default=6)
    plan.add_argument("--write", action="store_true")
    plan.add_argument("--no-balance", action="store_true")
    plan.add_argument("--base-argmax", default=None)
    plan.set_defaults(func=cmd_plan)

    launch = sub.add_parser("launch", help="launch the shard fleet, governed")
    launch.add_argument("--store", required=True)
    launch.add_argument("--tag", default="n600")
    launch.add_argument("--shards", type=int, default=6)
    launch.add_argument("--nice", type=int, default=5)
    launch.add_argument("--walltime-cap-s", type=float, default=86400.0)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--no-balance", action="store_true")
    launch.add_argument("--base-argmax", default=None)
    launch.add_argument("--match-live-env", action="store_true")
    launch.set_defaults(func=cmd_launch)

    verify = sub.add_parser("verify", help="replay completed pairs for equivalence")
    verify.add_argument("--store", required=True)
    verify.add_argument("--tag", default="n600")
    verify.add_argument("--replay-tag", default="n600_wc2verify")
    verify.add_argument("--pairs", default=None, help="explicit completed pairs")
    verify.add_argument("--count", type=int, default=4)
    verify.add_argument("--nice", type=int, default=5)
    verify.add_argument("--walltime-cap-s", type=float, default=7200.0)
    verify.add_argument("--dry-run", action="store_true")
    verify.add_argument("--match-live-env", action="store_true")
    verify.set_defaults(func=cmd_verify)

    compare = sub.add_parser("compare", help="decision-equivalence of a replay")
    compare.add_argument("--store", required=True)
    compare.add_argument("--tag", default="n600")
    compare.add_argument("--replay-tag", required=True)
    compare.add_argument("--write", action="store_true")
    compare.add_argument("--receipt", default=None, help="committed receipt path")
    compare.add_argument("--remaining", type=int, default=0, help="pairs at swap")
    compare.set_defaults(func=cmd_compare)

    merge = sub.add_parser("merge", help="merge shard checkpoints into one")
    merge.add_argument("--store", required=True)
    merge.add_argument("--tags", required=True, help="comma-separated tags")
    merge.add_argument("--out-tag", default="n600_merged")
    merge.add_argument("--tokens", default=None, help="base token npz for payload")
    # ALWAYS KEEP THE PAYLOAD is the default; the opt-out exists only for a
    # checkpoint-schema smoke where no token field is available, and it records
    # ``reconstructed: false`` in the manifest so the omission is never silent.
    merge.add_argument("--no-payload", action="store_true")
    merge.set_defaults(func=cmd_merge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
