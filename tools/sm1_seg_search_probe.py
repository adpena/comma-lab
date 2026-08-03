#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_sm1 — is the seg token solve UNDER-CONVERGED, BASIN-TRAPPED, or neither?

`ddm_pu2` measured that the POSE defect was SEARCH, not model, and split it into two
independent pathologies: (a) UNDER-CONVERGENCE (the solve stops while still descending)
and (b) BASIN-TRAPPING (the result depends on where you started). `ddm_xa1` §4 designed
the seg-side transfer test and priced it, but ran no scorer. This tool executes it.

It REUSES (does not rebuild) `tools/sb1_seg_batch.py`'s instrument -- `SegRuntime`
(parse -> in-place token edit -> render -> realized whole-pair SegNet argmax flips),
`_top_instances` (the ru1 atlas ranking) and `_best_single_quantum` (the exact 8-move
greedy probe the shipped QA03 solve uses). Three things are ADDED, each because the
shipped receipt schema cannot express it:

  1. PER-STEP flips inside `accepted_steps`. QA03 records `[ch, sign, new_code]` only,
     so the within-instance decay curve does not exist on disk and `xa1` Probe A could
     only compare ACROSS strata (different instances). This records `[ch, sign,
     new_code, flips]` so the decay curve is within-instance.
  2. MULTI-START. QA03 is single-start by construction (initial state is `rtp.codes`,
     the shipped base, and nothing else) and QA04's `--n-prop/--seed` subsamples the
     same 8 unit moves rather than re-initializing (xa1 Probe B). Starts here are
     genuine re-initializations of the cell's 4-code vector.
  3. A BOX-EXHAUSTIVE ORACLE. Greedy coordinate descent on a non-separable objective
     terminates at a COORDINATE-WISE local minimum, which need not be a joint one. The
     oracle enumerates every code vector within L-infinity radius R of the shipped base
     and returns the exact joint optimum inside that box, so "trapped" is measured
     against a known-capacity instrument instead of against another heuristic.

STRATA. `pu2` measured its multi-start defect as TAIL-SPECIFIC (tail median ratio
0.2013 vs non-tail control 0.9388), and `xa1` §4 makes a non-tail control MANDATORY:
QA03 selects a top-k tail by atlas rank, so without a control stratum this run cannot
distinguish "multi-start works" from "the tail is where everything works". `--control-k`
draws the control from atlas ranks BEYOND the tail cut.

SCOPE, stated up front because it bounds every number this tool can produce: the token
grid lives in the 767,812 B TR1 base (sha b9a7983b). The live-best 353,805 B v4d
container REFUSES `ddm_tr1_runtime.parse_archive` ("TR1 archive members/order differ"),
so any yield measured here is `verdict_scope: INSTANCE` on the TR1 base and does NOT
compose with the live row without a re-derivation.

Axis: [macOS-CPU advisory]; score_claim=false; promotion_eligible=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

try:  # repo-root import, then script-dir fallback (matches sb1_seg_batch)
    from tools.sb1_seg_batch import (
        LEVELS,
        SegRuntime,
        _atomic_write_text,
        _best_single_quantum,
        _sha,
        _top_instances,
        slot_holders,
    )
except ImportError:  # pragma: no cover - exercised as `python tools/<this>.py`
    from sb1_seg_batch import (  # type: ignore
        LEVELS,
        SegRuntime,
        _atomic_write_text,
        _best_single_quantum,
        _sha,
        _top_instances,
        slot_holders,
    )

from tac.single_writer_lock import single_writer_lock  # (sb1 import puts src on sys.path)

SEG_PX = 196608  # 512*384 SegNet internal resolution
# Live-best own-vehicle row this arm prices against (ddm_pu2, 2026-08-03).
LIVE_S = 0.7910689
LIVE_GAP = 0.6189279  # vs the PR130 demonstrated floor 0.172141


def _restore(rtp: SegRuntime, p: int, gy: int, gx: int) -> None:
    """Reset ONE cell's 4 codes to the pristine shipped base."""
    rtp.codes[p, gy, gx, :] = rtp.orig_codes[p, gy, gx, :].astype(rtp.codes.dtype)


def _set_codes(rtp: SegRuntime, p: int, gy: int, gx: int, vec) -> None:
    rtp.codes[p, gy, gx, :] = np.asarray(vec, dtype=rtp.codes.dtype)


def _descend(rtp: SegRuntime, p: int, gy: int, gx: int, cur: int, max_quanta: int):
    """Greedy coordinate descent, recording flips at EVERY accepted step.

    Identical move set and acceptance test to `sb1_seg_batch.qa03_gn_solve` (all 8 of
    4 channels x +/-1, strict improvement); the ONLY difference is that the realized
    flip count is stored per step so the decay curve is recoverable within-instance.
    """
    steps: list[list[int]] = []
    stop_reason = "no_move"
    n_eval = 0
    for _ in range(max_quanta):
        best = _best_single_quantum(rtp, p, gy, gx, cur)
        # _best_single_quantum evaluates every in-range (channel, sign); count them.
        n_eval += sum(
            1
            for ch in range(4)
            for sg in (-1, 1)
            if 0 <= int(rtp.codes[p, gy, gx, ch]) + sg < LEVELS
        )
        if best is None or best[0] >= cur:
            stop_reason = "converged" if steps else "no_move"
            break
        f, ch, sign = best
        rtp.codes[p, gy, gx, ch] += sign
        steps.append([int(ch), int(sign), int(rtp.codes[p, gy, gx, ch]), int(f)])
        cur = f
    else:
        stop_reason = "cap"
    return cur, steps, stop_reason, n_eval


def _starts(rng: np.random.Generator, base_vec: np.ndarray, n_starts: int, radius: int):
    """Genuine re-initializations of the cell's 4-code vector.

    start 0 is ALWAYS the shipped base -- it is the control arm and reproduces the
    shipped single-start solve exactly. The remainder alternate LOCAL (base +/- radius,
    a nearby basin) and GLOBAL (uniform over the whole lattice, `pu2`'s "a start 8.2x
    WORSE descends past the shipped point" case).
    """
    out = [("base", base_vec.copy())]
    k = 1
    while len(out) < n_starts:
        if k % 2 == 1:
            off = rng.integers(-radius, radius + 1, size=4)
            v = np.clip(base_vec + off, 0, LEVELS - 1)
            out.append((f"local_r{radius}_{k}", v.astype(base_vec.dtype)))
        else:
            v = rng.integers(0, LEVELS, size=4)
            out.append((f"uniform_{k}", v.astype(base_vec.dtype)))
        k += 1
    return out


def _box_configs(base_vec: np.ndarray, radius: int) -> list[tuple[int, ...]]:
    """Deterministic enumeration of the L-infinity box, clipped at lattice bounds."""
    axes = [
        range(max(0, int(base_vec[c]) - radius),
              min(LEVELS - 1, int(base_vec[c]) + radius) + 1)
        for c in range(4)
    ]
    return list(itertools.product(*axes))


def _box_oracle_chunk(rtp: SegRuntime, p: int, gy: int, gx: int,
                      configs: list[tuple[int, ...]], lo: int, hi: int):
    """EXACT joint optimum over ``configs[lo:hi]``.

    The full box is the instrument-capacity statement `m94` demands: "greedy is
    trapped" is only as strong as the reach of the thing that beat it, so the reach is
    ENUMERATED rather than sampled. Cost is (2R+1)^4 evals for the whole box, which at
    the measured 0.389 s/eval exceeds this environment's process-reaper window -- hence
    the chunking, so a kill costs one chunk rather than the instance.
    """
    best_f, best_v, n_eval = None, None, 0
    for vec in configs[lo:hi]:
        _set_codes(rtp, p, gy, gx, vec)
        f = rtp.pair_flips(p)
        n_eval += 1
        if best_f is None or f < best_f:
            best_f, best_v = f, list(vec)
    return best_f, best_v, n_eval


def run(args) -> None:
    holders = slot_holders()
    if not args.skip_slot_check and holders:
        raise SystemExit(f"[refuse] scorer slot live: {holders}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "sm1_instances.jsonl"

    # ---- strata -----------------------------------------------------------
    ranked = _top_instances(args.atlas, args.tail_k + args.control_k)
    tail_all = ranked[: args.tail_k]
    control_all = ranked[args.tail_k:]

    # QA03's own censoring set: instances that stopped ON the cap-4 bound. Those are
    # the ones `dc1` costed and `xa1` measured as still descending; they are the
    # UNDER-CONVERGENCE population. Read from the shipped receipt, not re-derived.
    censored: set[tuple[int, int, int]] = set()
    if args.qa03_instances and args.qa03_instances.exists():
        for line in args.qa03_instances.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if len(r["accepted_steps"]) >= args.qa03_cap:
                censored.add((r["pair"], r["cell"][0], r["cell"][1]))

    rng = np.random.default_rng(args.seed)

    def _pick(pool, n, want_censored=None):
        cand = [i for i in pool
                if want_censored is None
                or ((i[0], i[1], i[2]) in censored) == want_censored]
        if n >= len(cand):
            return cand
        # Deterministic spread over atlas rank, NOT a prefix (m88/m96: a prefix of a
        # rank-ordered population is a different population). Even strides over rank.
        idx = np.linspace(0, len(cand) - 1, n).round().astype(int)
        return [cand[i] for i in sorted(set(idx.tolist()))]

    selected = (
        [("tail_censored", i) for i in _pick(tail_all, args.n_tail, True)]
        + [("control_nontail", i) for i in _pick(control_all, args.n_control)]
    )

    done: set[tuple[str, int, int, int]] = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["stratum"], r["pair"], r["cell"][0], r["cell"][1]))

    # ---- PARTIALS: sub-instance checkpointing --------------------------------
    # An instance here costs ~8 min (4 descents + a 625-eval box). The first version
    # of this tool checkpointed per INSTANCE and, under a process reaper that fires
    # in minutes, banked NOTHING across three launches: the checkpoint granularity
    # exceeded the failure interval, which is the resumability non-negotiable's actual
    # requirement rather than merely "has a resume path". Arms and oracle chunks are
    # each ~1 min, so a kill now costs one arm.
    partials_path = out_dir / "sm1_partials.jsonl"
    partial: dict[tuple, dict] = {}
    if partials_path.exists():
        for line in partials_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = tuple(rec["key"])
            ent = partial.setdefault(key, {"base": None, "arms": {}, "oracle": None})
            if rec["kind"] == "base":
                ent["base"] = rec["pair_base_flips"]
            elif rec["kind"] == "arm":
                ent["arms"][rec["arm"]["start"]] = rec["arm"]
            elif rec["kind"] == "oracle_chunk":
                cur = ent["oracle"]
                if cur is None or rec["upto"] > cur["upto"]:
                    ent["oracle"] = rec
    if done or partial:
        print(f"[resume] {len(done)} instances complete, "
              f"{len(partial)} with partial work", flush=True)

    def _append_partial(rec: dict) -> None:
        with partials_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    rtp = SegRuntime(args.archive, args.gt_cache)
    t0 = time.time()
    n_eval_total = 0

    for stratum, (p, gy, gx, nflips) in selected:
        if (stratum, p, gy, gx) in done:
            continue
        key = (stratum, p, gy, gx)
        ent = partial.setdefault(key, {"base": None, "arms": {}, "oracle": None})
        _restore(rtp, p, gy, gx)
        base_vec = rtp.orig_codes[p, gy, gx, :].astype(np.int64).copy()
        if ent["base"] is None:
            ent["base"] = int(rtp.pair_flips(p))
            n_eval_total += 1
            _append_partial({"key": list(key), "kind": "base",
                             "pair_base_flips": ent["base"]})
        pair_base = ent["base"]

        arms = []
        for sname, svec in _starts(rng, base_vec, args.n_starts, args.start_radius):
            if sname in ent["arms"]:
                arms.append(ent["arms"][sname])
                continue
            _restore(rtp, p, gy, gx)
            _set_codes(rtp, p, gy, gx, svec)
            start_f = pair_base if sname == "base" else rtp.pair_flips(p)
            n_eval_total += 0 if sname == "base" else 1
            final_f, steps, stop_reason, n_ev = _descend(
                rtp, p, gy, gx, start_f, args.max_quanta)
            n_eval_total += n_ev
            arm = {
                "start": sname,
                "start_codes": [int(x) for x in svec],
                "start_flips": int(start_f),
                "final_flips": int(final_f),
                "final_codes": [int(x) for x in rtp.codes[p, gy, gx, :]],
                "net_vs_pair_base": int(pair_base - final_f),
                "n_steps": len(steps),
                "steps": steps,
                "stop_reason": stop_reason,
                "n_eval": int(n_ev),
            }
            ent["arms"][sname] = arm
            arms.append(arm)
            _append_partial({"key": list(key), "kind": "arm", "arm": arm})

        oracle = None
        if args.box_radius > 0:
            configs = _box_configs(base_vec, args.box_radius)
            state = ent["oracle"] or {"upto": 0, "best_flips": None,
                                      "best_codes": None, "n_eval": 0}
            while state["upto"] < len(configs):
                lo = state["upto"]
                hi = min(lo + args.oracle_chunk, len(configs))
                _restore(rtp, p, gy, gx)
                bf, bv, n_ev = _box_oracle_chunk(rtp, p, gy, gx, configs, lo, hi)
                n_eval_total += n_ev
                if state["best_flips"] is None or bf < state["best_flips"]:
                    state = {"upto": hi, "best_flips": int(bf), "best_codes": bv,
                             "n_eval": state["n_eval"] + n_ev}
                else:
                    state = {**state, "upto": hi,
                             "n_eval": state["n_eval"] + n_ev}
                ent["oracle"] = state
                _append_partial({"key": list(key), "kind": "oracle_chunk", **state})
            oracle = {
                "radius": args.box_radius,
                "n_configs": len(configs),
                "best_flips": state["best_flips"],
                "best_codes": state["best_codes"],
                "net_vs_pair_base": int(pair_base - state["best_flips"]),
                "n_eval": int(state["n_eval"]),
            }
        _restore(rtp, p, gy, gx)  # leave the working state pristine between instances

        best_arm = min(arms, key=lambda a: a["final_flips"])
        row = {
            "stratum": stratum,
            "pair": p,
            "cell": [gy, gx],
            "atlas_flips": int(nflips),
            "pair_base_flips": int(pair_base),
            "base_codes": [int(x) for x in base_vec],
            "arms": arms,
            "single_start_final": arms[0]["final_flips"],
            "multi_start_final": best_arm["final_flips"],
            "multi_start_winner": best_arm["start"],
            "box_oracle": oracle,
            "seed": args.seed,
            "max_quanta": args.max_quanta,
        }
        with jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[{stratum} p{p} ({gy},{gx}) atlas {nflips}] base {pair_base} "
            f"1start {arms[0]['final_flips']}({arms[0]['stop_reason']},"
            f"{arms[0]['n_steps']}st) multi {best_arm['final_flips']}"
            f"({best_arm['start']}) "
            + (f"box{args.box_radius} {oracle['best_flips']} " if oracle else "")
            + f"evals {n_eval_total} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    receipt = {
        "schema": "ddm_sm1_seg_search_probe.v1",
        "item": ("seg token solve: UNDER-CONVERGENCE (uncap + within-instance decay) "
                 "and BASIN-TRAPPING (multi-start + box-exhaustive oracle), with a "
                 "non-tail control stratum"),
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "live_best_S": LIVE_S,
        "live_best_gap": LIVE_GAP,
        "base_archive_sha256": _sha(rtp.base_archive),
        "base_archive_bytes": len(rtp.base_archive),
        "vehicle_scope_note": (
            "token grid lives in the 767,812 B TR1 base; the live-best 353,805 B v4d "
            "container REFUSES ddm_tr1_runtime.parse_archive, so every number here is "
            "INSTANCE-scoped to this base and does not compose with the live row "
            "without a re-derivation"),
        "atlas": str(args.atlas),
        "gt_cache": str(args.gt_cache),
        "seed": args.seed,
        "tail_k": args.tail_k,
        "control_k": args.control_k,
        "n_tail_requested": args.n_tail,
        "n_control_requested": args.n_control,
        "n_instances_selected": len(selected),
        "n_starts": args.n_starts,
        "start_radius": args.start_radius,
        "box_radius": args.box_radius,
        "max_quanta": args.max_quanta,
        "qa03_cap_reference": args.qa03_cap,
        "n_qa03_censored_known": len(censored),
        "scorer_evals_this_run": int(n_eval_total),
        "wall_seconds": time.time() - t0,
        "verdict_scope": "INSTANCE (this endpoint, this scorer, this atlas aim)",
    }
    _atomic_write_text(out_dir / "sm1_receipt.json",
                       json.dumps(receipt, indent=1, sort_keys=True))
    print(json.dumps(receipt, indent=1, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--atlas", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-slot-check", action="store_true")
    ap.add_argument("--qa03-instances", type=Path, default=None,
                    help="shipped QA03 instances.jsonl; rows at the cap define the "
                         "UNDER-CONVERGENCE (censored) population")
    ap.add_argument("--qa03-cap", type=int, default=4,
                    help="the cap the QA03 rows were WRITTEN under (they do not record "
                         "it; see the sb1 resume-inference defect)")
    ap.add_argument("--tail-k", type=int, default=120,
                    help="atlas rank cut that defines the tail (QA03 used 120)")
    ap.add_argument("--control-k", type=int, default=120,
                    help="how many ranks BEYOND the tail form the control pool")
    ap.add_argument("--n-tail", type=int, default=10)
    ap.add_argument("--n-control", type=int, default=10)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--start-radius", type=int, default=3)
    ap.add_argument("--box-radius", type=int, default=0,
                    help="L-infinity radius of the exhaustive oracle; 0 disables. "
                         "Cost is (2R+1)^4 scorer evals per instance.")
    ap.add_argument("--oracle-chunk", type=int, default=80,
                    help="box-oracle evals per checkpoint; keep the chunk shorter "
                         "than the environment's process-reaper interval")
    ap.add_argument("--max-quanta", type=int, default=48,
                    help="safety bound only; the intended terminator is convergence")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--skip-writer-lock", action="store_true",
                    help="bypass the single-writer lock (test harnesses only)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    with single_writer_lock(args.out_dir, label="sm1_seg_search_probe",
                            skip=args.skip_writer_lock):
        run(args)


if __name__ == "__main__":
    main()
