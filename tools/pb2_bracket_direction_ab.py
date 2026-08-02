#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb2 (#900) — price the ``break`` in the ddm_pw1 outward bracket.

WHAT IS BEING MEASURED
----------------------
``ddm_pw1`` removed two saturating menus from the live v4d pose solve and
moved the own-vehicle line ``0.9639878 -> 0.9476091``.  The bracket it
installed to do that (``experiments/ddm_v4d_resolve.py:215-221`` for dim0 and
``:314-319`` for beta; the same shape in
``tools/pw1_pose_menu_saturation_ab.py:90-100``) carries a ``break``::

    for sign in (1.0, -1.0):
        d = evaluate(x0 + sign * step0)
        if d < best_d:
            commit(sign); break        # <- -1.0 never evaluated

so the ``-`` direction is not a *sample of the continuum*, it is a *fallback*.
``ddm_lg2`` MEASURED from the shipped ``pw1_arms.jsonl`` receipt (reproduced
by this tool's ``--verify-input``, exactly): 94 pairs in arm A and 31 in arm B
commit to ``+`` with ``-`` never evaluated -- 109 distinct pairs, 15.89% of
the live arm-AB d_pose mass -- while among the pairs where ``-`` IS allowed to
compete it wins 60 to 31 in arm B, and pw1's own decomposition measured that
the dominant beta win (0.21963 of 0.30155 d_gain) requires ``g < -1.0``, i.e.
the negative side.  The probe ORDER is biased against the direction pw1
itself measured as dominant.

THE MEASUREMENT (pre-registered by ddm_lg2 §4; executed here unchanged)
----------------------------------------------------------------------
Drop the ``break``: evaluate BOTH entry probes for all 600 pairs, keep the
better, then continue the identical doubling expansion.  Monotone-safe by
construction -- the accept rule is unchanged (a strict decrease from the same
``d0``), so no arm can report a win it did not realize.

MONOTONE-SAFE IS NOT THE SAME AS DOMINANT, and this tool measured the
difference: the entry probe is a ONE-STEP-LOOKAHEAD greedy, so the direction
that LOSES the entry probe can still WIN the doubling continuation.  Both
variants are guaranteed no worse than their common starting point; neither is
guaranteed better than the other.  The receipt therefore reports three rows --
``asym`` (what ships), ``sym`` (the pre-registered change), and ``both`` (the
per-pair min, i.e. what a bracket expanding BOTH directions would reach at
roughly double the continuation cost) -- and ``entry_rule_does_not_dominate``
counts the pairs each way.

FALSIFIER (pre-registered, honored): if the symmetric search yields
``delta_d_pose >= -1e-6`` over the 109 short-circuited pairs, the probe-order
asymmetry is priced at ZERO on this vehicle and the row closes at
``verdict_scope: FORMULATION``.

HOW THE COMPARISON IS KEPT HONEST
---------------------------------
* Both bracket variants run against the SAME memoized evaluator per (pair,
  arm), so the delta is measured entirely inside one instrument and carries no
  cross-run floor.  The memo also means running both costs barely more than
  running the symmetric one alone.
* POSITIVE CONTROL: the asymmetric variant is pw1's semantics verbatim and
  MUST reproduce the shipped ``arm_a_d`` / ``arm_b_d`` / ``arm_ab_d`` from
  ``pw1_arms.jsonl``.  A control that cannot reproduce the thing it replays is
  not a control, so a mismatch fails the run.
* ``+`` wins exact ties (the entry scan is ``+`` first against a running best
  seeded at ``d0``), so any measured delta is a STRICT win for ``-``.
* Every emitted count reports its DENOMINATOR and an empty scope emits
  ``VACUOUS``, never ``PASS``.

HONEST BOUND, carried not dropped: the 109 short-circuited pairs have mean
d_pose 0.006685 vs the population 0.007645 -- they sit BELOW average.  This is
a broad, MILD under-search, not the tail (pw1 §7: 10 pairs carry 62.1% of the
mass).  Do not inflate it into a tail lever.

Axis: ``[macOS-CPU frozen-PoseNet advisory]`` NON-PROMOTABLE.
``score_claim=false``.  d_seg is untouched by construction: frame_1 is never
modified and SegNet reads ``x[:, -1]`` only (``upstream/modules.py:108``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "ddm_pb2_bracket_direction_ab.v1"
V4D = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
N_PAIRS = 600
# The live steps, read from experiments/ddm_v4d_resolve.py:70-72 (and mirrored
# by tools/pw1_pose_menu_saturation_ab.py:62-66).  Imported at runtime rather
# than re-typed -- see main().
FALLBACK_DIM0_STEP0 = 0.012
FALLBACK_BETA_STEP0 = 0.5
BETA_MAGS = (0.0, 0.5, 1.0)
# pw1 ran its arms with this safety cap, so the positive control must use it.
PW1_MAX_EXPAND = 12
# pw1_receipt.json total_scorer_evals for the same n600 arms (asym only).
PW1_TOTAL_EVALS = 4644
FALSIFIER_EPS = 1e-6


def _atomic_write(path: Path, payload: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# the two bracket variants -- identical except for the entry rule
# --------------------------------------------------------------------------- #
def _entry_candidates(x0: float, step0: float, quantize):
    """The (+, -) entry points, de-duplicated against x0 on the lattice."""
    out = []
    for sign in (1.0, -1.0):
        cand = x0 + sign * step0
        if quantize is not None:
            cand = quantize(cand)
        if cand == x0:
            continue
        out.append((sign, cand))
    return out


def _expand(evaluate, best_x, best_d, direction, step, max_expand, quantize):
    """The doubling continuation.  Identical for both variants."""
    n = 0
    probes = []
    for _ in range(max_expand):
        step *= 2.0
        cand = best_x + direction * step
        if quantize is not None:
            cand = quantize(cand)
        if cand == best_x:
            break
        dv = evaluate(cand)
        n += 1
        probes.append({"x": float(cand), "d": float(dv), "phase": "expand"})
        if dv >= best_d:
            break
        best_x, best_d = cand, dv
    return best_x, best_d, n, probes


def bracket_asym(evaluate, x0, d0, step0, max_expand, quantize=None):
    """pw1 semantics VERBATIM: first-improving entry probe wins and breaks.

    This is the POSITIVE CONTROL, not a proposal.  It must reproduce the
    shipped pw1 arm values.
    """
    best_x, best_d, direction = x0, d0, 0.0
    n = 0
    probes = []
    for sign, cand in _entry_candidates(x0, step0, quantize):
        dv = evaluate(cand)
        n += 1
        probes.append({"x": float(cand), "d": float(dv), "phase": "probe"})
        if dv < best_d:
            best_x, best_d, direction = cand, dv, sign
            break  # <- THE MEASURED DEFECT
    if direction == 0.0:
        return best_x, best_d, n, probes, direction
    bx, bd, ne, pe = _expand(evaluate, best_x, best_d, direction, step0,
                             max_expand, quantize)
    return bx, bd, n + ne, probes + pe, direction


def bracket_sym(evaluate, x0, d0, step0, max_expand, quantize=None):
    """Both entry probes ALWAYS; commit to the better.

    ``+`` is scanned first against a running best seeded at ``d0``, so ``+``
    wins exact ties and any delta vs :func:`bracket_asym` is a STRICT win for
    ``-``.  The accept rule (strict decrease from ``d0``) is unchanged, so this
    is monotone-safe: it cannot return a point worse than ``x0``.
    """
    n = 0
    probes = []
    scored = []
    for sign, cand in _entry_candidates(x0, step0, quantize):
        dv = evaluate(cand)
        n += 1
        probes.append({"x": float(cand), "d": float(dv), "phase": "probe"})
        scored.append((sign, cand, dv))
    best_x, best_d, direction = x0, d0, 0.0
    for sign, cand, dv in scored:  # (+, -) order => + wins ties
        if dv < best_d:
            best_x, best_d, direction = cand, dv, sign
    if direction == 0.0:
        return best_x, best_d, n, probes, direction
    bx, bd, ne, pe = _expand(evaluate, best_x, best_d, direction, step0,
                             max_expand, quantize)
    return bx, bd, n + ne, probes + pe, direction


class MemoEval:
    """Counts UNIQUE scorer forwards; both variants share one memo per arm."""

    def __init__(self, fn):
        self._fn = fn
        self._memo: dict[float, float] = {}
        self.n_forward = 0

    def __call__(self, x: float) -> float:
        key = float(x)
        if key not in self._memo:
            self._memo[key] = self._fn(key)
            self.n_forward += 1
        return self._memo[key]


# --------------------------------------------------------------------------- #
# input verification (zero compute) -- reproduces ddm_lg2 §4 from the receipt
# --------------------------------------------------------------------------- #
def verify_input(arms_jsonl: Path) -> dict:
    rows = [json.loads(ln) for ln in arms_jsonl.read_text().splitlines()
            if ln.strip()]
    n = len(rows)
    if n == 0:
        return {"verdict": "VACUOUS", "denominator": 0,
                "note": "no arm rows; an empty scope is VACUOUS, never PASS"}
    by = {int(r["pair"]): r for r in rows}

    def short(r, key):
        return len([p for p in r[key] if p["phase"] == "probe"]) == 1

    a = {int(r["pair"]) for r in rows if short(r, "arm_a_probes")}
    b = {int(r["pair"]) for r in rows if short(r, "arm_b_probes")}
    u = a | b
    # the live mass basis is the arm-AB (post-pw1) solution -- that is what we
    # actually hold at S=0.9476091, not the pre-pw1 shipped row.
    m = np.array([by[p]["arm_ab_d"] for p in sorted(by)], np.float64)
    tot = float(m.sum())
    # every short-circuit must be a genuine break (first probe improved), not
    # a lattice-collapse skip; otherwise the count means something else.
    genuine = sum(1 for p in a if by[p]["arm_a_probes"][0]["d"] < by[p]["d_ctrl"])
    genuine += sum(1 for p in b if by[p]["arm_b_probes"][0]["d"] < by[p]["d_ctrl"])
    return {
        "verdict": "UNTESTED_BINARY_COMMITMENT" if u else "VACUOUS",
        "denominator": n,
        "arm_a_short_circuit": len(a),
        "arm_b_short_circuit": len(b),
        "union_pairs": len(u),
        "arm_instances": len(a) + len(b),
        "genuine_break_of_arm_instances": genuine,
        "union_mass_frac_of_arm_ab": (
            float(sum(by[p]["arm_ab_d"] for p in u) / tot) if tot else None),
        "mean_d_pose_union": (
            float(sum(by[p]["arm_ab_d"] for p in u) / len(u)) if u else None),
        "mean_d_pose_population": float(m.mean()),
        "mass_basis": "arm_ab_d (the LIVE pw1 solution, S=0.9476091)",
    }


def falsifier_verdict(ctrl_max: float, n_in_scope: int,
                      pop_delta: float | None, per_pair_delta: float | None,
                      eps: float = FALSIFIER_EPS) -> str:
    """Adjudicate ddm_lg2's pre-registered falsifier.

    VERDICT CLEARANCE (L3): a verdict drawn from an instrument that cannot
    reproduce the thing it replays is inadmissible, so the positive control
    GATES the falsifier rather than sitting beside it as a field.

    The pre-registered rule ("delta_d_pose >= -1e-6 over the 109 pairs") is
    ambiguous between a per-scope-pair mean and the population mean that
    actually enters the score; the two differ by a factor 109/600.  The
    population reading is the EASIER one to call null, so BOTH are required to
    agree -- that removes any freedom to pick the convenient denominator after
    seeing the data.  A split reading is reported as its own verdict, not
    rounded to whichever side is preferred.
    """
    if ctrl_max > 0.0:
        return "INSTRUMENT_UNTRUSTED"
    if n_in_scope == 0 or pop_delta is None or per_pair_delta is None:
        return "VACUOUS"
    pop_null = pop_delta >= -eps
    per_null = per_pair_delta >= -eps
    if pop_null and per_null:
        return "NULL_PRICED_AT_ZERO"
    if not pop_null and not per_null:
        return "ASYMMETRY_PRICED"
    return "BELOW_SCORE_RESOLUTION"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="celldrop50")
    ap.add_argument("--final-jsonl", type=Path, default=V4D / "final_refine.jsonl")
    ap.add_argument("--pw1-arms", type=Path, default=V4D / "pw1" / "pw1_arms.jsonl")
    ap.add_argument("--out-dir", type=Path, default=V4D / "pb2")
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--max-minutes", type=float, default=180.0)
    ap.add_argument("--max-expand", type=int, default=PW1_MAX_EXPAND)
    ap.add_argument("--emit-final-jsonl", type=Path, default=None)
    ap.add_argument("--verify-input", action="store_true",
                    help="reproduce ddm_lg2 §4 from the pw1 receipt and exit "
                         "(zero compute, no scorer)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_input:
        print(json.dumps(verify_input(args.pw1_arms), indent=1))
        return

    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    torch.set_num_threads(1)
    import ddm_v4c_resolve as v4c
    import ddm_v4d_resolve as v4d

    # HIJACK guard (memory: shared_venv_editable_install_hijack_from_arm_worktree).
    # Identity is compared with os.path.samefile (device+inode), NOT by string.
    # MEASURED: this repo is reachable as both /Users/adpena/Projects/pact and
    # /Users/adpena/projects/pact on a case-insensitive volume; Path.resolve()
    # does not normalise case and os.path.normcase is a NO-OP on darwin, so both
    # a prefix test and a normcase test report that benign alias as a hijack.
    # samefile resolves the alias and still catches a genuine foreign worktree.
    import tac
    tac_src = Path(tac.__file__).resolve().parent.parent  # .../src
    want = (REPO / "src").resolve()
    if not (tac_src.exists() and want.exists()
            and os.path.samefile(tac_src, want)):
        raise SystemExit(f"tac hijack: {tac.__file__} is not under {want}")

    dim0_step0 = float(getattr(v4d, "DIM0_STEP0", FALLBACK_DIM0_STEP0))
    beta_step0 = float(getattr(v4d, "BETA_STEP0", FALLBACK_BETA_STEP0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shipped = {int(r["pair"]): r for r in
               (json.loads(ln) for ln in
                args.final_jsonl.read_text().splitlines() if ln.strip())}
    if len(shipped) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} shipped rows, got {len(shipped)}")
    pw1 = {int(r["pair"]): r for r in
           (json.loads(ln) for ln in
            args.pw1_arms.read_text().splitlines() if ln.strip())}
    if len(pw1) != N_PAIRS:
        raise SystemExit(f"expected {N_PAIRS} pw1 arm rows, got {len(pw1)}")

    oracle = v4c.build_oracle(args.base, s_r=1.0)
    comp = v4c.StaticComposer(oracle)
    photo = v4d.load_photo(v4d.PHOTO_JL)
    offset = v4d._dim0_offset(photo)

    def lattice(x: float) -> float:
        return offset + float(np.float16(x - offset))

    def score(pose, s_t, sel, f1_u8, f1_f, tp, a, b, g):
        """Realized d_pose at rolling-shutter magnitude g.

        Byte-for-byte the same compose path as
        ``tools/pw1_pose_menu_saturation_ab.py:164-180`` so the positive
        control can reproduce the shipped arm values.
        """
        beta = g * (1.0 if pose[5] >= 0.0 else -1.0)
        wg_t, wf_t = comp.warps(f1_f, pose, s_t, 1.0 - beta / 2.0)
        wg_b, wf_b = comp.warps(f1_f, pose, s_t, 1.0 + beta / 2.0)
        f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
        f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
        f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
        if a != 1.0 or b != 0.0:
            f0 = a * f0 + b
        p6 = comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    order = sorted(range(N_PAIRS), key=lambda p: -shipped[p]["d_final"])
    seq = order[: args.pairs]
    jl = args.out_dir / "pb2_arms.jsonl"
    cache = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[pb2] base={args.base} offset={offset:.6f} pairs={len(seq)} "
          f"cached={len(cache)} dim0_step0={dim0_step0} "
          f"beta_step0={beta_step0} max_expand={args.max_expand}", flush=True)

    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[pb2] wall cap; {len(cache)} done; rerun to resume",
                  flush=True)
            break
        sh = shipped[pidx]
        ref = pw1[pidx]
        pose = np.asarray(sh["p"], np.float64).copy()
        a, b = float(sh["a"]), float(sh["b"])
        sel = int(sh["selector"])
        if "beta_mag" in sh:
            g_ship = float(sh["beta_mag"])
        else:
            _bi = int(sh["beta_idx"])
            if not 0 <= _bi < len(BETA_MAGS):
                raise SystemExit(f"pair {pidx}: beta_idx={_bi} with no beta_mag")
            g_ship = float(BETA_MAGS[_bi])
        s_t = float(v4c._d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        ctx = (s_t, sel, f1_u8, f1_f, tp, a, b)

        d_ctrl = score(pose, *ctx, g_ship)
        n_fwd = 1

        # every per-pair variable is bound at DEFINITION time (default args),
        # never closed over the loop variable -- the same guard pw1 used.
        def dim0_eval_at(p, g, _ctx=ctx):
            def _ev(x):
                q = p.copy()
                q[0] = x
                return score(q, *_ctx, g)
            return _ev

        def beta_eval_at(p, _ctx=ctx):
            def _ev(g):
                return score(p, *_ctx, g)
            return _ev

        # ---- ARM A: dim0, both variants on one memo -----------------------
        mA = MemoEval(dim0_eval_at(pose, g_ship))
        xa_as, da_as, _, _, dir_a_as = bracket_asym(
            mA, float(pose[0]), d_ctrl, dim0_step0, args.max_expand, lattice)
        xa_sy, da_sy, _, _, dir_a_sy = bracket_sym(
            mA, float(pose[0]), d_ctrl, dim0_step0, args.max_expand, lattice)
        n_fwd += mA.n_forward

        # ---- ARM B: beta at the shipped dim0, both variants on one memo ---
        mB = MemoEval(beta_eval_at(pose))
        gb_as, db_as, _, _, dir_b_as = bracket_asym(
            mB, g_ship, d_ctrl, beta_step0, args.max_expand)
        gb_sy, db_sy, _, _, dir_b_sy = bracket_sym(
            mB, g_ship, d_ctrl, beta_step0, args.max_expand)
        n_fwd += mB.n_forward

        # ---- ARM AB: beta at each variant's own dim0 ----------------------
        pose_as = pose.copy()
        pose_as[0] = xa_as
        mAB_as = MemoEval(beta_eval_at(pose_as))
        gab_as, dab_as, _, _, _ = bracket_asym(
            mAB_as, g_ship, da_as, beta_step0, args.max_expand)
        n_fwd += mAB_as.n_forward
        if xa_sy == xa_as:
            # identical pose => the symmetric AB bracket shares the memo.
            mAB_sy = mAB_as
            before = mAB_sy.n_forward
            gab_sy, dab_sy, _, _, _ = bracket_sym(
                mAB_sy, g_ship, da_sy, beta_step0, args.max_expand)
            n_fwd += mAB_sy.n_forward - before
        else:
            pose_sy = pose.copy()
            pose_sy[0] = xa_sy
            mAB_sy = MemoEval(beta_eval_at(pose_sy))
            gab_sy, dab_sy, _, _, _ = bracket_sym(
                mAB_sy, g_ship, da_sy, beta_step0, args.max_expand)
            n_fwd += mAB_sy.n_forward

        # ---- POSITIVE CONTROL: the asym replay must reproduce pw1 ---------
        ctrl_err = max(abs(da_as - float(ref["arm_a_d"])),
                       abs(db_as - float(ref["arm_b_d"])),
                       abs(dab_as - float(ref["arm_ab_d"])))

        rec = {
            "pair": int(pidx), "rank": rank,
            "d_ctrl": d_ctrl,
            "pw1_canary_abs_err": abs(d_ctrl - float(ref["d_ctrl"])),
            "control_abs_err_vs_pw1_arms": ctrl_err,
            "dim0_shipped": float(pose[0]), "g_shipped": float(g_ship),
            "asym": {"a_dim0": float(xa_as), "a_d": float(da_as),
                     "b_g": float(gb_as), "b_d": float(db_as),
                     "ab_g": float(gab_as), "ab_d": float(dab_as),
                     "a_dir": dir_a_as, "b_dir": dir_b_as},
            "sym": {"a_dim0": float(xa_sy), "a_d": float(da_sy),
                    "b_g": float(gb_sy), "b_d": float(db_sy),
                    "ab_g": float(gab_sy), "ab_d": float(dab_sy),
                    "a_dir": dir_a_sy, "b_dir": dir_b_sy},
            "a_direction_flipped": bool(dir_a_as != dir_a_sy),
            "b_direction_flipped": bool(dir_b_as != dir_b_sy),
            "n_forward": int(n_fwd),
        }
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if len(cache) % 25 == 0 or rank < 3:
            arr = list(cache.values())
            print(f"[pb2 {len(cache):3d}/{len(seq)}] pair {pidx} "
                  f"asymAB {dab_as:.6f} symAB {dab_sy:.6f} | "
                  f"ctrl_max {max(r['control_abs_err_vs_pw1_arms'] for r in arr):.2e} "
                  f"flips {sum(r['a_direction_flipped'] or r['b_direction_flipped'] for r in arr)} "
                  f"{time.time()-t0:.0f}s", flush=True)

    fj.close()
    summarize(args, cache, shipped, pw1)


def summarize(args, cache: dict, shipped: dict, pw1: dict) -> None:
    rows = [cache[p] for p in sorted(cache)]
    n = len(rows)
    if n == 0:
        print(json.dumps({"schema": SCHEMA, "verdict": "VACUOUS",
                          "denominator": 0,
                          "note": "no rows measured; VACUOUS, never PASS"},
                         indent=1))
        return

    # the short-circuited scope, re-derived from the pw1 receipt
    def short(r, key):
        return len([p for p in r[key] if p["phase"] == "probe"]) == 1

    sc_a = {p for p, r in pw1.items() if short(r, "arm_a_probes")}
    sc_b = {p for p, r in pw1.items() if short(r, "arm_b_probes")}
    scope = sc_a | sc_b

    d_ship_all = np.array([shipped[p]["d_final"] for p in range(N_PAIRS)])

    def composed(variant: str, key: str) -> np.ndarray:
        d = d_ship_all.copy()
        for r in rows:
            d[r["pair"]] = min(d[r["pair"]], r[variant][key])
        return d

    # BOTH-CONTINUATIONS bound.  MEASURED: neither entry rule dominates -- the
    # entry probe is a ONE-STEP-LOOKAHEAD greedy, so the direction that loses
    # the entry probe can still win the doubling continuation (pair 326:
    # sym takes g=-1.5 on a better entry probe and ends 4.58e-5 WORSE than
    # asym's g=+1.5).  So "drop the break" is not automatically an
    # improvement, and the honest achievable bound is the per-pair min of the
    # two continuations -- what a bracket that expands BOTH directions would
    # get, at roughly double the continuation cost.
    def best_of_both(key: str) -> np.ndarray:
        d = d_ship_all.copy()
        for r in rows:
            d[r["pair"]] = min(d[r["pair"]], r["asym"][key], r["sym"][key])
        return d

    out_rows: dict[str, dict] = {}
    for key in ("a_d", "b_d", "ab_d"):
        da = composed("asym", key)
        ds = composed("sym", key)
        db = best_of_both(key)
        ma, ms, mb = float(da.mean()), float(ds.mean()), float(db.mean())
        out_rows[key] = {
            "asym_d_pose_mean": ma, "sym_d_pose_mean": ms,
            "both_d_pose_mean": mb,
            "asym_pose_contribution": float(np.sqrt(10.0 * ma)),
            "sym_pose_contribution": float(np.sqrt(10.0 * ms)),
            "both_pose_contribution": float(np.sqrt(10.0 * mb)),
            "delta_d_pose_mean": ms - ma,
            "delta_S_pose": float(np.sqrt(10.0 * ms) - np.sqrt(10.0 * ma)),
            "delta_S_pose_both_vs_asym": float(np.sqrt(10.0 * mb)
                                               - np.sqrt(10.0 * ma)),
        }

    # FALSIFIER, evaluated on the pre-registered scope: the 109 pairs whose
    # bracket short-circuited.  Reported as the mean over the WHOLE population
    # (the scored quantity) restricted to that scope's contribution.
    in_scope = [r for r in rows if r["pair"] in scope]
    scope_delta_ab = (
        float(sum(r["sym"]["ab_d"] - r["asym"]["ab_d"] for r in in_scope)
              / N_PAIRS) if in_scope else None)
    scope_delta_ab_per_pair = (
        float(np.mean([r["sym"]["ab_d"] - r["asym"]["ab_d"] for r in in_scope]))
        if in_scope else None)

    # THE SHIPPING ARM'S OWN SCOPE -- beyond ddm_lg2's pre-registration.
    # lg2 derived the 109-pair scope from arm A's and arm B's stored probes,
    # because those are the ones pw1 wrote.  But pw1 DISCARDED the arm-AB
    # probes (``g_ab, d_ab, nab, _ = bracket_out(...)``) and arm AB is the arm
    # that SHIPS, so its own bracket short-circuits are invisible in that
    # receipt and are NOT confined to the 109.  Measured here directly by
    # comparing the two variants' chosen magnitudes -- the same
    # "signal produced and discarded" residue lg2 flagged for ``ab_trace``.
    ab_div = [r for r in rows if r["sym"]["ab_g"] != r["asym"]["ab_g"]]
    ab_out = [r for r in ab_div if r["pair"] not in scope]
    ab_scope = {
        "ab_arm_diverged_pairs": len(ab_div),
        "ab_arm_diverged_outside_lg2_109": len(ab_out),
        "denominator_measured": n,
        "delta_d_pose_population_mean": float(
            sum(r["sym"]["ab_d"] - r["asym"]["ab_d"] for r in ab_div)
            / N_PAIRS) if ab_div else 0.0,
        "pre_registered": False,
        "note": ("pw1 discarded the arm-AB probes, so lg2's 109-pair scope "
                 "cannot see the SHIPPING arm's short-circuits; this row "
                 "EXTENDS the pre-registration and is labelled as such"),
    }

    ctrl_max = max(r["control_abs_err_vs_pw1_arms"] for r in rows)
    falsifier = falsifier_verdict(ctrl_max, len(in_scope), scope_delta_ab,
                                  scope_delta_ab_per_pair)
    # The whole-population reading, which is what actually enters the score.
    # Its per-item denominator is the DIVERGING set (the pairs where the two
    # variants chose differently); everywhere else the delta is identically 0
    # and averaging over them would only dilute.
    pop_delta_ab = out_rows["ab_d"]["delta_d_pose_mean"]
    per_div = (float(np.mean([r["sym"]["ab_d"] - r["asym"]["ab_d"]
                              for r in ab_div])) if ab_div else 0.0)
    falsifier_pop = falsifier_verdict(ctrl_max, n, pop_delta_ab, per_div)
    ab_scope["delta_d_pose_per_diverging_pair"] = per_div

    out = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False, "promotion_eligible": False,
        "pointer_moved": False, "research_only": True,
        "base": args.base,
        "denominator_pairs_measured": n,
        "denominator_pairs_population": N_PAIRS,
        "scope_short_circuited_pairs": len(scope),
        "scope_measured": len(in_scope),
        "positive_control_max_abs_err_vs_pw1_arms": ctrl_max,
        "positive_control_note": (
            "the asymmetric variant replays pw1's bracket verbatim and must "
            "reproduce arm_a_d/arm_b_d/arm_ab_d from pw1_arms.jsonl"),
        "positive_control_composed_check": {
            "asym_arm_ab_d_pose_mean": out_rows["ab_d"]["asym_d_pose_mean"],
            "pw1_receipt_arm_ab_d_pose_mean": 0.007645062472871804,
            "note": ("a second, independent control: the asym composed mean "
                     "must land on pw1's own receipt value"),
        },
        "pw1_canary_max_abs_err": max(r["pw1_canary_abs_err"] for r in rows),
        "arms": out_rows,
        "direction_flips": {
            "arm_a": int(sum(r["a_direction_flipped"] for r in rows)),
            "arm_b": int(sum(r["b_direction_flipped"] for r in rows)),
            "either": int(sum(r["a_direction_flipped"]
                              or r["b_direction_flipped"] for r in rows)),
        },
        "falsifier": {
            "pre_registered_by": "ddm_lg2 §4",
            "rule": ("delta_d_pose >= -1e-6 over the short-circuited scope "
                     "=> priced at zero, close at FORMULATION scope"),
            "eps": FALSIFIER_EPS,
            "scope_delta_d_pose_population_mean": scope_delta_ab,
            "scope_delta_d_pose_per_scope_pair": scope_delta_ab_per_pair,
            "delta_S_pose_arm_ab": out_rows["ab_d"]["delta_S_pose"],
            "both_readings_required": True,
            "verdict": falsifier,
        },
        "shipping_arm_scope_EXTENSION": ab_scope,
        "whole_population_verdict": falsifier_pop,
        "entry_rule_does_not_dominate": {
            "sym_better_pairs": int(sum(1 for r in rows
                                        if r["sym"]["ab_d"] < r["asym"]["ab_d"])),
            "asym_better_pairs": int(sum(1 for r in rows
                                         if r["asym"]["ab_d"] < r["sym"]["ab_d"])),
            "denominator": n,
            "note": ("the entry probe is a ONE-STEP-LOOKAHEAD greedy; the "
                     "loser of the entry probe can still win the doubling "
                     "continuation, so dropping the break is not "
                     "automatically an improvement"),
        },
        "total_scorer_forwards": int(sum(r["n_forward"] for r in rows)),
        "cost_accounting": {
            "pw1_total_scorer_evals_asym_only": PW1_TOTAL_EVALS,
            "pb2_total_forwards_both_variants": int(
                sum(r["n_forward"] for r in rows)),
            "marginal_forwards_for_the_symmetric_variant": int(
                sum(r["n_forward"] for r in rows)) - PW1_TOTAL_EVALS,
            "lg2_pre_registered_estimate": 125,
            "estimate_note": (
                "lg2 estimated +125 = one extra ENTRY probe per short-"
                "circuited arm-instance (94 arm A + 31 arm B).  The realised "
                "cost is larger because (a) the SHIPPING arm-AB bracket has "
                "its own short-circuits, which lg2's scope could not see, and "
                "(b) a flipped entry sends the doubling continuation down a "
                "different chain.  Comparable only at n=600 and only against "
                "pw1's own 4644."),
            "measurement_gap": (
                "per-probe traces are DISCARDED in the emitted rows, so the "
                "cost of the shipped change alone cannot be separated from "
                "the cost of replaying both variants; this figure is an "
                "UPPER BOUND.  Same 'produced and discarded' defect this "
                "unit flags in pw1's arm-AB probes -- named, not hidden."),
        },
        "generated_by": "tools/pb2_bracket_direction_ab.py",
        "note": ("both bracket variants share one memoized evaluator per "
                 "(pair, arm), so the delta is floor-free; '+' wins exact "
                 "ties so any delta is a STRICT win for '-'. d_seg untouched "
                 "(frame_1 never modified)."),
    }
    if ctrl_max > 0.0:
        out["positive_control_FAILED"] = True
        print("*** POSITIVE CONTROL FAILED: the asymmetric replay does not "
              f"reproduce pw1 (max abs err {ctrl_max:.3e}). No verdict from "
              "this run is admissible. ***", flush=True)

    if args.emit_final_jsonl:
        floor = max(r["pw1_canary_abs_err"] for r in rows)
        kept = 0
        with open(args.emit_final_jsonl, "w") as fh:
            for p in range(N_PAIRS):
                row = dict(shipped[p])
                if "beta_mag" in row:
                    mag = float(row["beta_mag"])
                else:
                    idx = int(row["beta_idx"])
                    if not 0 <= idx < len(BETA_MAGS):
                        raise SystemExit(
                            f"pair {p}: beta_idx={idx} with no beta_mag")
                    mag = float(BETA_MAGS[idx])
                r = cache.get(p)
                # BEST-OF-BOTH, not sym-only: neither entry rule dominates
                # (see ``entry_rule_does_not_dominate``), so emitting the
                # symmetric solution alone would ship a knowingly-inferior
                # candidate on the pairs where the '+' continuation won.
                if r is not None:
                    win = min(("asym", r["asym"]["ab_d"]),
                              ("sym", r["sym"]["ab_d"]), key=lambda t: t[1])
                    if win[1] < row["d_final"] - floor:
                        pose = list(row["p"])
                        pose[0] = r[win[0]]["a_dim0"]
                        row["p"] = pose
                        mag = float(r[win[0]]["ab_g"])
                        row["d_final"] = float(win[1])
                        row["source"] = f"pb2_bestof_{win[0]}_ab"
                        kept += 1
                row["beta_mag"] = mag
                row["beta_idx"] = (BETA_MAGS.index(mag) if mag in BETA_MAGS
                                   else -1)
                fh.write(json.dumps(row) + "\n")
        out["emitted_final_jsonl"] = str(args.emit_final_jsonl)
        out["emitted_rows_replaced"] = kept

    path = args.out_dir / "pb2_receipt.json"
    _atomic_write(path, json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))
    print(f"[pb2] receipt {path}", flush=True)


if __name__ == "__main__":
    main()
