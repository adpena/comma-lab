#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pu2 — the multi-start pose floor probe on the cx1 tail, through the SHIPPED receiver.

WHY THIS EXISTS (ddm_pu1 §8): four open questions collapse onto one measurement —
(a) is ``mq1``'s search headroom still available after ``pj2``; (b) the realised
B/pair of a tail solve; (c) is pair 74 a MODEL wall or a SEARCH wall; (d) is
``pz1``'s per-pair ALLOCATION correct (its mean was validated to 1.6e-5, but a
mis-allocation between pairs preserves the mean).

INSTRUMENT.  ``pu1`` specified a ~30-line re-point of ``ddm_pfs1``'s
``WarpPoseOracle`` (which is wired to the *pb1* vehicle).  That is not the
cheapest correct path: the live ``cx1`` receiver is ALREADY factored as
``inflate_runner.Decoder`` with ``.f1(i)`` / ``.f0(i, f1)``, so this probe drives
the SHIPPED decoder directly and never re-implements the reconstruction.  Every
candidate is rendered by the same code the evaluator runs.

LEGAL CANDIDATE SET = exactly what the v4d grammar can express, at IDENTICAL
stream widths (so a winner is a value change, never a format change):
  pose6    f16 (dim0 stored as an f16 RESIDUAL over ``config.pose_dim0_offset``)
  st_idx   index into the shipped ``st_vals`` table
  sel      {0,1}   single-plane / two-plane static compose
  (a,b)    f16 photometric auto-exposure
  beta_idx index into the shipped ``beta_mags`` rolling-shutter table
Candidates are ALWAYS quantized to the shipped dtype BEFORE scoring
(realized-acceptance discipline: no f64 ceilings are ever reported as reachable).

STAGES.
  A  categorical sweep at the shipped pose/ab over (sel x beta_idx x st_idx).
     Directly answers "did pj2 leave DISCRETE headroom?" with no gradient.
  B  damped Gauss-Newton over the continuous knobs (pose6, optionally a/b) from
     several starts, accepted at shipped quantization.
  Reported per pair, never pooled (pu1 §5.5: the tail has >=2 mechanisms).

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
No archive is rebuilt and no gate is fired here; the exact contest pointer
0.1910828242 [contest-CPU] is UNMOVED.  ONE full-n600 scorer job at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import numpy as np

np.seterr(all="ignore")

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CX1_SUB = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
               "submissions/v4d_cx1_pj2ix2")
PZ1_JSON = REPO / ".omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json"
SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803")

# cx1 evaluator row (report.txt, n600, the exact archive bytes) — recomputed, never assumed.
CX1_D_SEG = 0.00431179
CX1_D_POSE = 0.00255143
CX1_BYTES = 353_808
DEN = 37_545_489
N_PAIRS = 600
# PR130 = the BAR (lessons-only lineage; used ONLY as the gap denominator).
PR130_S = 0.172141

# pu1 §5.2 tail, mass-ordered.  Listed so the probe's target set is auditable.
TAIL_DEFAULT = [74, 67, 21, 523, 16, 71, 44, 42, 275, 18]


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * float(d_pose_mean)))


def score_of(d_seg: float, d_pose_mean: float, nbytes: int) -> float:
    return 100.0 * d_seg + contribution(d_pose_mean) + 25.0 * nbytes / DEN


class Cx1Oracle:
    """d_pose of the SHIPPED cx1 reconstruction for an arbitrary legal knob set.

    Drives ``inflate_runner.Decoder`` itself — the same object the evaluator's
    ``inflate.sh`` runs — so no reconstruction logic is duplicated here.
    """

    def __init__(self, sub: Path = CX1_SUB) -> None:
        import ddm_p3v2_optimal_form_pose_resolve as p3v2
        self.p3v2 = p3v2
        self.posenet, _ = p3v2.load_posenet()
        self.targets = p3v2.load_targets(N_PAIRS)
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
        import inflate_runner as ir
        self.ir = ir
        self.dec = ir.Decoder(sub / "archive")
        d = self.dec
        self.n_st = int(np.asarray(d.st_vals).size)
        self.n_beta = len(d.beta_mags)
        self.dim0_offset = float(d.dim0_offset) if d.dim0_offset is not None else None
        # shipped knob custody (copies; the decoder's arrays are mutated in place
        # by ``score`` and restored, so the originals must be held separately).
        self.ship_pose = np.array(d.p_best, np.float64, copy=True)
        self.ship_st = np.array(d.st_idx, np.int64, copy=True)
        self.ship_sel = np.array(d.sel, np.int64, copy=True)
        self.ship_ab = np.array(d.ab, np.float64, copy=True)
        self.ship_beta = np.array(d.beta_idx, np.int64, copy=True)
        self._f1: dict[int, np.ndarray] = {}
        self.n_forwards = 0

    # -- shipped quantization -------------------------------------------------
    def q_pose(self, pose: np.ndarray) -> np.ndarray:
        """Round to what the v4d pose member can actually store."""
        p = np.asarray(pose, np.float64).copy()
        if self.dim0_offset is None:
            return p.astype(np.float16).astype(np.float64)
        resid = np.float16(p[0] - self.dim0_offset).astype(np.float64)
        out = p.astype(np.float16).astype(np.float64)
        out[0] = self.dim0_offset + resid
        return out

    @staticmethod
    def q_ab(ab: np.ndarray) -> np.ndarray:
        return np.asarray(ab, np.float64).astype(np.float16).astype(np.float64)

    def f1(self, pidx: int) -> np.ndarray:
        if pidx not in self._f1:
            self._f1[pidx] = self.dec.f1(pidx)
        return self._f1[pidx]

    def score(self, pidx: int, pose=None, st_idx=None, sel=None, ab=None,
              beta_idx=None) -> float:
        """d_pose at the SHIPPED quantization for one legal knob set."""
        d = self.dec
        keep = (np.array(d.p_best[pidx], copy=True), int(d.st_idx[pidx]),
                int(d.sel[pidx]), np.array(d.ab[pidx], copy=True),
                int(d.beta_idx[pidx]))
        try:
            if pose is not None:
                d.p_best[pidx] = self.q_pose(pose)
            if st_idx is not None:
                d.st_idx[pidx] = int(st_idx)
            if sel is not None:
                d.sel[pidx] = int(sel)
            if ab is not None:
                d.ab[pidx] = self.q_ab(ab)
            if beta_idx is not None:
                d.beta_idx[pidx] = int(beta_idx)
            f1 = self.f1(pidx)
            f0 = d.f0(pidx, f1)
            self.n_forwards += 1
            return self.p3v2.d_pose_u8(self.posenet, f0, f1, self.targets[pidx])
        finally:
            (d.p_best[pidx], d.st_idx[pidx], d.sel[pidx], d.ab[pidx],
             d.beta_idx[pidx]) = keep

    def shipped(self, pidx: int) -> float:
        return self.score(pidx)


# --------------------------------------------------------------------------- #
# stage A — categorical sweep (no gradient; answers "discrete headroom?")
# --------------------------------------------------------------------------- #
def stage_a(oracle: Cx1Oracle, pidx: int, *, max_seconds: float = 0.0) -> dict:
    base = oracle.shipped(pidx)
    best = {"d_pose": base, "sel": int(oracle.ship_sel[pidx]),
            "beta_idx": int(oracle.ship_beta[pidx]),
            "st_idx": int(oracle.ship_st[pidx])}
    t0 = time.time()
    n_eval = 0
    for sel in range(2):
        for beta_idx in range(oracle.n_beta):
            for st_i in range(oracle.n_st):
                if (sel == best["sel"] and beta_idx == int(oracle.ship_beta[pidx])
                        and st_i == int(oracle.ship_st[pidx])):
                    continue
                if max_seconds and (time.time() - t0) > max_seconds:
                    return {"d_pose_shipped": base, "best": best, "n_eval": n_eval,
                            "truncated": True, "seconds": time.time() - t0}
                v = oracle.score(pidx, sel=sel, beta_idx=beta_idx, st_idx=st_i)
                n_eval += 1
                if v < best["d_pose"]:
                    best = {"d_pose": v, "sel": sel, "beta_idx": beta_idx,
                            "st_idx": st_i}
    return {"d_pose_shipped": base, "best": best, "n_eval": n_eval,
            "truncated": False, "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# stage B — damped GN over the continuous knobs, accepted at shipped quant
# --------------------------------------------------------------------------- #
# forward-difference steps, scaled to the shipped pose spread per dim.
FD_POSE = np.array([0.08, 0.004, 0.004, 0.0015, 0.0015, 0.004], np.float64)
FD_AB = np.array([0.004, 0.5], np.float64)


def _pose6_of(oracle: Cx1Oracle, pidx: int, pose, st_idx, sel, ab, beta_idx):
    d = oracle.dec
    keep = (np.array(d.p_best[pidx], copy=True), int(d.st_idx[pidx]),
            int(d.sel[pidx]), np.array(d.ab[pidx], copy=True),
            int(d.beta_idx[pidx]))
    try:
        d.p_best[pidx] = oracle.q_pose(pose)
        d.st_idx[pidx] = int(st_idx)
        d.sel[pidx] = int(sel)
        d.ab[pidx] = oracle.q_ab(ab)
        d.beta_idx[pidx] = int(beta_idx)
        f1 = oracle.f1(pidx)
        f0 = d.f0(pidx, f1)
        oracle.n_forwards += 1
        return oracle.p3v2.pose6_u8(oracle.posenet, f0, f1)
    finally:
        (d.p_best[pidx], d.st_idx[pidx], d.sel[pidx], d.ab[pidx],
         d.beta_idx[pidx]) = keep


def gn_solve(oracle: Cx1Oracle, pidx: int, *, pose0, st_idx, sel, ab0, beta_idx,
             relins: int, with_ab: bool, lm0: float = 1.0,
             deadline: float = 0.0) -> dict:
    """Damped GN on r(theta) = pose6(recon(theta)) - target, theta = pose6 [+ a,b].

    Every line-search candidate is quantized to the shipped dtype BEFORE it is
    scored, so the returned value is a REALIZED point, never a ceiling.
    """
    tp = oracle.targets[pidx]
    th_p = oracle.q_pose(pose0)
    th_ab = oracle.q_ab(ab0)
    nd = 8 if with_ab else 6
    cur6 = _pose6_of(oracle, pidx, th_p, st_idx, sel, th_ab, beta_idx)
    cur = float(np.mean((cur6 - tp) ** 2))
    start = cur
    lm = lm0
    for _ in range(relins):
        if deadline and time.time() > deadline:
            break
        J = np.zeros((6, nd), np.float64)
        for k in range(nd):
            pp, aa = th_p.copy(), th_ab.copy()
            if k < 6:
                pp[k] += FD_POSE[k]
                step_k = FD_POSE[k]
            else:
                aa[k - 6] += FD_AB[k - 6]
                step_k = FD_AB[k - 6]
            J[:, k] = (_pose6_of(oracle, pidx, pp, st_idx, sel, aa, beta_idx)
                       - cur6) / step_k
        r = cur6 - tp
        accepted = False
        for _damp in range(4):
            A = J.T @ J + lm * np.diag(np.maximum(np.diag(J.T @ J), 1e-8))
            try:
                step = np.linalg.solve(A, -(J.T @ r))
            except np.linalg.LinAlgError:
                break
            for scale in (1.0, 0.5, 0.25):
                pp = oracle.q_pose(th_p + scale * step[:6])
                aa = oracle.q_ab(th_ab + scale * step[6:]) if with_ab else th_ab
                c6 = _pose6_of(oracle, pidx, pp, st_idx, sel, aa, beta_idx)
                cval = float(np.mean((c6 - tp) ** 2))
                if cval < cur:
                    th_p, th_ab, cur6, cur = pp, aa, c6, cval
                    lm = max(lm * 0.3, 1e-4)
                    accepted = True
                    break
            if accepted:
                break
            lm *= 8.0
        if not accepted or cur < 1e-7:
            break
    return {"d_pose_start": start, "d_pose": cur,
            "pose": [float(x) for x in th_p], "ab": [float(x) for x in th_ab],
            "st_idx": int(st_idx), "sel": int(sel), "beta_idx": int(beta_idx)}


def multistart(oracle: Cx1Oracle, pidx: int, a_res: dict, *, relins: int,
               n_random: int, seed: int, with_ab: bool,
               deadline: float = 0.0) -> dict:
    """Starts: shipped point, stage-A categorical best, GT-target pose, randoms."""
    rng = np.random.default_rng(seed + pidx)
    ship_p = oracle.ship_pose[pidx]
    ship_ab = oracle.ship_ab[pidx]
    b = a_res["best"]
    starts: list[tuple[str, dict]] = [
        ("shipped_knobs", dict(pose0=ship_p, st_idx=int(oracle.ship_st[pidx]),
                               sel=int(oracle.ship_sel[pidx]), ab0=ship_ab,
                               beta_idx=int(oracle.ship_beta[pidx]))),
        ("stageA_best", dict(pose0=ship_p, st_idx=b["st_idx"], sel=b["sel"],
                             ab0=ship_ab, beta_idx=b["beta_idx"])),
    ]
    # GT-target pose with rotation dims zeroed (expmap(0)=I) — the pfs1 D1 start.
    tp0 = oracle.targets[pidx].copy()
    tp0[3:] = 0.0
    starts.append(("gt_target_rot0", dict(pose0=tp0, st_idx=b["st_idx"],
                                          sel=b["sel"], ab0=ship_ab,
                                          beta_idx=b["beta_idx"])))
    # random perturbations of the shipped pose, scaled to the population spread.
    spread = oracle.ship_pose.std(0)
    for j in range(n_random):
        pj = ship_p + rng.normal(0.0, 1.0, 6) * spread * 0.35
        starts.append((f"rand{j}", dict(pose0=pj, st_idx=b["st_idx"],
                                        sel=b["sel"], ab0=ship_ab,
                                        beta_idx=b["beta_idx"])))
    out = []
    best = None
    for name, kw in starts:
        if deadline and time.time() > deadline:
            out.append({"start": name, "skipped_deadline": True})
            continue
        r = gn_solve(oracle, pidx, relins=relins, with_ab=with_ab,
                     deadline=deadline, **kw)
        r["start"] = name
        out.append(r)
        if best is None or r["d_pose"] < best["d_pose"]:
            best = r
    return {"starts": out, "best": best}


# --------------------------------------------------------------------------- #
def run_smoke(args) -> None:
    """POSITIVE CONTROL: reproduce pz1's PER-PAIR d_pose through the shipped receiver.

    pu1 §7-R1-c: pz1's MEAN was validated to 1.6e-5, but a mis-allocation between
    pairs preserves the mean.  This settles the allocation, and it is the probe's
    own instrument check — an instrument that cannot return the negative is not
    an instrument.
    """
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    oracle = Cx1Oracle()
    boot = time.time() - t0
    pz1 = json.loads(PZ1_JSON.read_text())
    base = {int(r["pair"]): float(r["d_pose_base"]) for r in pz1["rows"]}
    pairs = [int(p) for p in args.pairs] if args.pairs else TAIL_DEFAULT[:4] + [0, 300]
    rows = []
    for p in pairs:
        t1 = time.time()
        got = oracle.shipped(p)
        dt = time.time() - t1
        exp = base[p]
        rel = abs(got - exp) / max(exp, 1e-12)
        rows.append({"pair": p, "pz1_d_pose_base": exp, "probe_d_pose": got,
                     "rel_err": rel, "seconds": dt})
        print(f"  pair {p:4d}  pz1={exp:.6f}  probe={got:.6f}  rel={rel:.3e}  "
              f"({dt:.2f}s)", flush=True)
    worst = max(r["rel_err"] for r in rows)
    receipt = {
        "schema": "ddm_pu2_positive_control.v1", "utc": _utc(),
        "git": _git_hash(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "boot_seconds": boot, "rows": rows, "worst_rel_err": worst,
        "verdict": ("ALLOCATION_VERIFIED" if worst < 1e-6 else
                    "ALLOCATION_REFUTED_OR_INSTRUMENT_DIFFERS"),
        "note": "reproduces pz1's PER-PAIR d_pose_base by driving the shipped "
                "cx1 inflate_runner.Decoder; settles pu1 §7-R1-c.",
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "pu2_positive_control.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "rows"}, indent=1),
          flush=True)


def run_probe(args) -> None:
    import torch
    torch.set_num_threads(4)
    oracle = Cx1Oracle()
    pairs = [int(p) for p in args.pairs] if args.pairs else TAIL_DEFAULT
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    jl = work / "pu2_floor_probe.partial.jsonl"
    cache: dict[int, dict] = {}
    if jl.exists() and args.resume:
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        print(f"[pu2] resume: {len(cache)} pairs cached", flush=True)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    for p in pairs:
        if p in cache:
            continue
        deadline = (t0 + args.max_seconds) if args.max_seconds else 0.0
        if deadline and time.time() > deadline:
            print(f"[pu2] --max-seconds before pair {p}; re-run --resume",
                  flush=True)
            break
        n0 = oracle.n_forwards
        tA = time.time()
        a_res = stage_a(oracle, p, max_seconds=args.stage_a_seconds)
        b_res = multistart(oracle, p, a_res, relins=args.relins,
                           n_random=args.n_random, seed=args.seed,
                           with_ab=args.with_ab, deadline=deadline)
        ship = a_res["d_pose_shipped"]
        floor = min(ship, a_res["best"]["d_pose"],
                    b_res["best"]["d_pose"] if b_res["best"] else ship)
        row = {
            "pair": p, "d_pose_shipped": ship,
            "d_pose_stage_a": a_res["best"]["d_pose"],
            "d_pose_floor": floor,
            "ratio_floor_over_shipped": floor / ship if ship > 0 else float("nan"),
            "stage_a": a_res, "stage_b": b_res,
            "n_forwards": oracle.n_forwards - n0,
            "seconds": time.time() - tA,
        }
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[p] = row
        print(f"[pu2 pair {p:4d}] shipped={ship:.6f} stageA={row['d_pose_stage_a']:.6f} "
              f"floor={floor:.6f} ratio={row['ratio_floor_over_shipped']:.4f} "
              f"fwd={row['n_forwards']} ({row['seconds']:.0f}s)", flush=True)
    fj.close()
    _summarize(cache, work, args)


def _summarize(cache: dict[int, dict], work: Path, args,
               out_name: str = "pu2_floor_probe_receipt.json") -> None:
    """n600 S-arithmetic from the measured per-pair floors.

    The population mean is REBUILT from pz1's full n600 array with the probed
    pairs replaced by their measured floors — never extrapolated from the subset
    (the #875 prefix law: a tail subset is a different population by
    construction, so only a full-population substitution is admissible).
    """
    if not cache:
        print("[pu2] no rows to summarize", flush=True)
        return
    pz1 = json.loads(PZ1_JSON.read_text())
    base = np.array([float(r["d_pose_base"]) for r in
                     sorted(pz1["rows"], key=lambda r: int(r["pair"]))], np.float64)
    if base.size != N_PAIRS:
        raise SystemExit(f"pz1 array is {base.size} pairs, expected {N_PAIRS}")
    mean0 = float(base.mean())
    new = base.copy()
    for p, row in cache.items():
        new[p] = float(row["d_pose_floor"])
    mean1 = float(new.mean())
    s0 = score_of(CX1_D_SEG, mean0, CX1_BYTES)
    s1 = score_of(CX1_D_SEG, mean1, CX1_BYTES)
    gap = s0 - PR130_S
    per_pair = []
    for p in sorted(cache):
        r = cache[p]
        d = r["d_pose_shipped"] - r["d_pose_floor"]
        # exact single-pair S effect, holding the other 599 at the shipped value.
        m_alt = base.copy()
        m_alt[p] = r["d_pose_floor"]
        ds = score_of(CX1_D_SEG, float(m_alt.mean()), CX1_BYTES) - s0
        per_pair.append({
            "pair": p, "d_pose_shipped": r["d_pose_shipped"],
            "d_pose_floor": r["d_pose_floor"],
            "d_pose_stage_a": r["d_pose_stage_a"],
            "ratio_floor_over_shipped": r["ratio_floor_over_shipped"],
            "delta_d_pose": d, "delta_S_alone": ds,
            "pct_of_gap_alone": 100.0 * (-ds) / gap,
            "breakeven_bytes_alone": (-ds) * DEN / 25.0,
            "best_start": (r["stage_b"]["best"] or {}).get("start"),
            "n_forwards": r["n_forwards"],
        })
    total_dS = s1 - s0
    receipt = {
        "schema": "ddm_pu2_floor_probe.v1", "utc": _utc(), "git": _git_hash(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_vehicle": "v4d_cx1_pj2ix2",
        "baseline": {"d_seg": CX1_D_SEG, "d_pose_mean": mean0,
                     "bytes": CX1_BYTES, "S": s0,
                     "pose_contribution": contribution(mean0)},
        "floor_denominator": {"name": "PR130 (the BAR; lessons-only lineage)",
                              "S": PR130_S, "gap": gap,
                              "bytes_per_pct_of_gap": gap * DEN / 25.0 / 100.0},
        "n_pairs_probed": len(cache),
        "d_pose_mean_after": mean1,
        "delta_S_total": total_dS,
        "pct_of_gap_total": 100.0 * (-total_dS) / gap,
        "breakeven_bytes_total": (-total_dS) * DEN / 25.0,
        "breakeven_bytes_per_pair": ((-total_dS) * DEN / 25.0 / len(cache)
                                     if cache else 0.0),
        "per_pair": per_pair,
        "settings": {"relins": args.relins, "n_random": args.n_random,
                     "with_ab": bool(args.with_ab), "seed": args.seed},
        "note": "floors are REALIZED at the shipped v4d quantization through the "
                "actual inflate_runner.Decoder; the population mean is rebuilt by "
                "substituting into pz1's full n600 array (never extrapolated).",
    }
    # NOTE (ddm_pu2 §6 Round 4): the terminal probe receipt and an interim
    # `--mode summarize` MUST NOT share a path.  They did, and a completion
    # waiter keyed on "the receipt exists" fired while the probe was still
    # running.  One path with two writers turns an existence test into a false
    # positive -- the "probe that cannot return the negative" class.
    (work / out_name).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


# --------------------------------------------------------------------------- #
# realised bytes — rebuild the ACTUAL archive.zip with the probed knobs
# --------------------------------------------------------------------------- #
KL1_MAGIC = b"KL1PWF01"
PW_MAGIC = b"PFS1WPD1"


def _encode_kl1_field(arr: np.ndarray) -> bytes:
    """Inverse of ``inflate_runner.decode_kl1_field`` (byte-plane f16 + brotli)."""
    import struct

    import brotli
    bits = np.ascontiguousarray(arr, np.float16).view(np.uint16)   # (n,d)
    n, d = bits.shape
    cm = np.ascontiguousarray(bits.T)                              # (d,n)
    raw = ((cm >> 8) & 0xFF).astype(np.uint8).tobytes() + (cm & 0xFF).astype(np.uint8).tobytes()
    comp = brotli.compress(raw, quality=11)
    return KL1_MAGIC + struct.pack("<HHI", n, d, len(comp)) + comp


def _encode_pose_warp(oracle: Cx1Oracle, pose, st_idx, sel, ab, beta_idx) -> bytes:
    """Inverse of ``inflate_runner.parse_pose_warp_v4d``."""
    import struct

    import brotli
    from ddm_r7_token_coder import encode_token_codes
    n = int(pose.shape[0])
    tp = np.asarray(pose, np.float64).copy()
    if oracle.dim0_offset is not None:
        tp[:, 0] = tp[:, 0] - oracle.dim0_offset          # store the residual
    tp_member = _encode_kl1_field(tp.astype(np.float16))
    st_coded = encode_token_codes(
        np.ascontiguousarray(np.asarray(st_idx, np.uint8)).reshape(n, 1, 1, 1),
        levels=oracle.n_st, codec="auto")
    sel_coded = brotli.compress(
        np.packbits(np.asarray(sel, np.uint8), bitorder="big").tobytes(), quality=11)
    ab_member = _encode_kl1_field(np.asarray(ab, np.float16))
    beta_coded = brotli.compress(np.asarray(beta_idx, np.uint8).tobytes(), quality=11)
    out = PW_MAGIC + struct.pack("<I", n)
    for sec in (tp_member, st_coded, sel_coded, ab_member, beta_coded):
        out += struct.pack("<I", len(sec)) + sec
    return out


def run_bytes(args) -> None:
    """Measure the REALISED archive byte delta of the probed knob set.

    Positive control first: re-encoding the SHIPPED values must reproduce the
    shipped ``pose_warp`` payload and the shipped ``archive.zip`` size, else the
    delta is reported under my encoder on BOTH sides and labelled as such.
    """
    import torch
    torch.set_num_threads(4)
    from tac.optimization.ddm_ix2_archive_container import (
        build_payload,
        build_single_member_zip,
        parse_payload,
    )
    oracle = Cx1Oracle()
    blob = (CX1_SUB / "archive" / "0.bin").read_bytes()
    bulk, sections = parse_payload(blob)
    config, renderer, selector, pose_warp_shipped = sections
    ship_zip = len(build_single_member_zip(build_payload(bulk, list(sections))))

    rebuilt = _encode_pose_warp(oracle, oracle.ship_pose, oracle.ship_st,
                                oracle.ship_sel, oracle.ship_ab, oracle.ship_beta)
    pw_identical = rebuilt == pose_warp_shipped
    ctl_zip = len(build_single_member_zip(
        build_payload(bulk, [config, renderer, selector, rebuilt])))

    jl = args.work_dir / "pu2_floor_probe.partial.jsonl"
    rows = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            rr = json.loads(ln)
            rows[int(rr["pair"])] = rr
    pose = oracle.ship_pose.copy()
    st = oracle.ship_st.copy()
    sel = oracle.ship_sel.copy()
    ab = oracle.ship_ab.copy()
    beta = oracle.ship_beta.copy()
    changed = []
    for p, r in rows.items():
        best = (r.get("stage_b") or {}).get("best")
        a_best = (r.get("stage_a") or {}).get("best") or {}
        cand = None
        if best and best["d_pose"] <= r["d_pose_floor"] + 1e-15:
            cand = best
        elif a_best and a_best.get("d_pose", 1e9) <= r["d_pose_floor"] + 1e-15:
            cand = {"pose": list(oracle.ship_pose[p]), "ab": list(oracle.ship_ab[p]),
                    "st_idx": a_best["st_idx"], "sel": a_best["sel"],
                    "beta_idx": a_best["beta_idx"]}
        if cand is None or r["d_pose_floor"] >= r["d_pose_shipped"]:
            continue
        pose[p] = oracle.q_pose(np.asarray(cand["pose"], np.float64))
        ab[p] = oracle.q_ab(np.asarray(cand["ab"], np.float64))
        st[p], sel[p], beta[p] = int(cand["st_idx"]), int(cand["sel"]), int(cand["beta_idx"])
        changed.append(p)
    new_pw = _encode_pose_warp(oracle, pose, st, sel, ab, beta)
    new_payload = build_payload(bulk, [config, renderer, selector, new_pw])
    new_zip = len(build_single_member_zip(new_payload))
    d_bytes = new_zip - ctl_zip

    # BYTE-CLOSED VERIFICATION: write the rebuilt container, decode it with a FRESH
    # Decoder, and re-score every changed pair.  A floor that the rebuilt archive
    # does not reproduce is not a result -- it is an encoder bug.
    verify = []
    if changed and not args.no_verify:
        vdir = args.work_dir / "verify_archive"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "0.bin").write_bytes(new_payload)
        dec2 = oracle.ir.Decoder(vdir)
        for p in changed:
            f1 = oracle.f1(p)
            f0 = dec2.f0(p, f1)
            got = oracle.p3v2.d_pose_u8(oracle.posenet, f0, f1, oracle.targets[p])
            want = float(rows[p]["d_pose_floor"])
            verify.append({"pair": p, "probe_floor": want, "decoded_d_pose": got,
                           "rel_err": abs(got - want) / max(want, 1e-12)})
    worst_v = max((v["rel_err"] for v in verify), default=0.0)
    receipt = {
        "schema": "ddm_pu2_realised_bytes.v1", "utc": _utc(), "git": _git_hash(),
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE", "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "positive_control": {
            "pose_warp_reencode_byte_identical": bool(pw_identical),
            "shipped_zip_bytes": ship_zip,
            "reencoded_shipped_zip_bytes": ctl_zip,
            "cx1_report_bytes": CX1_BYTES,
            "note": ("deltas are measured against the RE-ENCODED shipped archive "
                     "(same encoder both sides), so an encoder mismatch cancels"),
        },
        "pairs_changed": sorted(changed), "n_changed": len(changed),
        "archive_bytes_after": new_zip,
        "delta_bytes_total": d_bytes,
        "delta_bytes_per_changed_pair": (d_bytes / len(changed)) if changed else 0.0,
        "delta_S_rate": 25.0 * d_bytes / DEN,
        "byte_closed_verification": {
            "rows": verify, "worst_rel_err": worst_v,
            "verdict": ("FLOORS_REPRODUCE_FROM_REBUILT_ARCHIVE" if worst_v < 1e-9
                        else "REBUILT_ARCHIVE_DOES_NOT_REPRODUCE_FLOORS"),
            "note": "fresh Decoder over the rebuilt 0.bin; proves the winning knob "
                    "values survive the shipped f16/offset/token coding",
        },
    }
    (args.work_dir / "pu2_realised_bytes.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode",
                    choices=("smoke", "probe", "summarize", "bytes"), required=True)
    ap.add_argument("--work-dir", type=Path, default=SSD_OUT)
    ap.add_argument("--pairs", type=int, nargs="*", default=None)
    ap.add_argument("--relins", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--with-ab", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stage-a-seconds", type=float, default=0.0)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-verify", action="store_true",
                    help="bytes mode: skip the byte-closed decode verification")
    args = ap.parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "probe":
        run_probe(args)
    elif args.mode == "bytes":
        run_bytes(args)
    else:
        jl = args.work_dir / "pu2_floor_probe.partial.jsonl"
        cache = {}
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        _summarize(cache, args.work_dir, args,
                   out_name="pu2_interim_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
