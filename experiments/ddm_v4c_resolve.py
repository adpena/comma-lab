#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4c — base-adjudicated pose RE-SOLVE (QA59) + photometric rungs (QA62).

v4c composes on TOP of the measured v4b gate (n600 evaluate.py S 1.534258 =
d_pose 0.06365131 + d_seg 0.00553676 + 274,479 B) with four orthogonal moves:

  STAGE 1  BASE ADJUDICATION  (--mode transfer)
      Transfer the v4b shipped pose field (Knee-A-optimal, 600x6 f16 +
      selector) onto a candidate token BASE through the REALIZABLE v4b static
      two-plane compose and measure d_pose.  cell_drop50 (gr1) is seg+rate
      -0.066 better than Knee-A but its far-field is frozen on a DIFFERENT
      cell set (seg-|g| ordered, pose-blind).  The transfer measures the pose
      damage before re-solve; a tail subset re-solve directly measures the
      RECOVERED floor.  Decision by arithmetic (both bases priced).

  STAGE 2  POSE RE-SOLVE THROUGH THE STATIC MASK  (--mode solve, QA59/QA60)
      Per pair best-of{single-plane 6-DOF GN, two-plane STATIC GN} where the
      two-plane far/ground split is the receiver's DERIVED horizon v=round(cy)
      =437 (rule-118 free), NOT the GT mask the v4b ship poses were solved on.
      The shipped v4b poses are GT-mask-solved => suboptimal for the static
      receiver; re-solving THROUGH the static compose is a strict further win
      and (unlike v4b's tail-only two-plane) extends two-plane to ALL pairs
      (QA60 non-tail).  Poses kept at f64 for the dim0 offset-coding lattice.

  STAGE 3  PHOTOMETRIC RUNGS  (--mode photo, QA62)
      At the chosen pose, per pair solve rung-B auto-exposure (f0 := a*warp+b
      after warp, before uint8; camerad AE re-tunes frame-to-frame) via a
      2-param GN, and pick rung-A rolling-shutter row-shear beta (sign free
      from the shipped xi yaw dim; magnitude one global constant, ~0 bytes).
      Applied on the SAME realizable static compose the receiver runs.

Every d_pose realized through the frozen CPU-torch PoseNet + the vendored
receiver warp primitives (byte-identical to the engine).  Resumable JSONL.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  ONE n600 scorer job at a time
(PoseNet-only per-pair work).  `tac` HIJACK guarded: run with
PYTHONPATH=$PWD/src so `import tac` resolves to main src (positive control:
the Knee-A transfer reproduces the v4b gate d_pose 0.06365).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")  # match the proven ck1/qa44 instrument env

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, "experiments")
import ddm_pfs1_ep_warp_pose_solve as d2m  # WarpPoseOracle, solve_pair_gn, FD_STEPS, ST_GRID

BASES = {
    "kneeA": Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeA_safe_274k_archive.zip"),
    "celldrop50": Path("/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_cell_drop50_archive.zip"),
}
SHIP_TABLE = Path("/Volumes/VertigoDataTier/pact/ddm_v4b_20260730/v4b_ship_table.json")
D2_JL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730")
GATE_D_POSE_V4B = 0.06365131   # the measured v4b gate mean (evaluate.py rc=0)
RELINS = 4
GAIN_FD, BIAS_FD = 0.02, 2.0   # rung-B FD steps (a near 1.0, b in 0..255 units)
GN_RELINS_PHOTO = 4
RS_BETAS = [0.0, 0.5, 1.0, -0.5, -1.0]  # rolling-shutter shear sweep (b=0 == ctrl)
RS_GLOBAL_G = (0.5, 1.0)  # SHIPPABLE global shear magnitudes (sign from yaw, free)
SEG_D_KNEEA = 0.00553676       # MEASURED d_seg at the wr1 Knee-A gate (tokens unchanged)


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def contribution(dpm: float) -> float:
    return float(np.sqrt(10.0 * float(dpm)))


def geometric_horizon_row(K: np.ndarray) -> int:
    """Ground-plane vanishing row for pitch=0: n=[0,-1,0], v=round(cy)=437."""
    n = np.array([0.0, -1.0, 0.0], np.float64)
    ln = np.linalg.inv(K).T @ n
    return round(-ln[2] / ln[1])


def build_oracle(base: str, s_r: float) -> d2m.WarpPoseOracle:
    """pfs1 oracle with its packet OVERRIDDEN to the candidate base tokens.

    Mirrors ck1.build_kneeA_oracle exactly but parametrized over the archive
    (both Knee-A and cell_drop50 are the v3_warp sectioned grammar with
    manifest['tr1_metadata']).  No shared-file edit; a post-construction
    oracle.packet override only.  render_frame1_camera_uint8(packet, i) then
    renders the DROPPED-token f1 the evaluator would score for this base.
    """
    from ddm_r7_token_coder import decode_token_codes
    archive = BASES[base]
    oracle = d2m.WarpPoseOracle(s_r=s_r)
    rt = oracle.rt
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        codes = decode_token_codes(z.read("state/tokens.dr7t"))
        sections = {
            "tokens": rt._encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8)),
            "lotto_renderer": z.read("state/renderer.sec"),
            "selector": z.read("state/selector.sec"),
            "pose_stub": z.read("state/pose_stub.sec"),
        }
    packet_bytes = rt.build_packet(manifest["tr1_metadata"], sections)
    oracle.packet = rt.parse_packet(packet_bytes)
    return oracle


def _load_jl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[int(r["pair"])] = r
    return rows


def load_ship_table() -> dict[int, dict]:
    obj = json.loads(SHIP_TABLE.read_text())
    return {int(r["pair"]): r for r in obj["rows"]}


# --------------------------------------------------------------------------- #
# the REALIZABLE static two-plane compose (identical to inflate_runner_v4b)
# --------------------------------------------------------------------------- #
class StaticComposer:
    def __init__(self, oracle: d2m.WarpPoseOracle) -> None:
        recv = oracle.recv
        self.recv, self.o = recv, oracle
        self.K, self.Kinv, self.grid = oracle.K, oracle.Kinv, oracle.grid
        self.ch, self.cw = recv.CAMERA_H, recv.CAMERA_W
        self.v_row = geometric_horizon_row(self.K)
        self.far = (np.arange(self.ch)[:, None] < self.v_row) & np.ones((1, self.cw), bool)
        self.alpha_row = (np.arange(self.ch) / (self.ch - 1.0))[:, None, None]

    def warps(self, f1_f: np.ndarray, theta: np.ndarray, s_t: float,
              rot_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        hg = self.recv.pose_to_homography(theta, self.K, self.Kinv, s_t, rot_scale, 0.0)
        hf = self.recv.pose_to_homography(theta, self.K, self.Kinv, 0.0, rot_scale, 0.0)
        return (self.recv.warp_rgb(f1_f, hg, self.grid),
                self.recv.warp_rgb(f1_f, hf, self.grid))

    def warp_ground(self, f1_f: np.ndarray, theta: np.ndarray, s_t: float) -> np.ndarray:
        """Just the ground homography warp (single-plane path); half the warp
        cost of warps() when the far plane is unused."""
        hg = self.recv.pose_to_homography(theta, self.K, self.Kinv, s_t, 1.0, 0.0)
        return self.recv.warp_rgb(f1_f, hg, self.grid)

    def compose(self, warp_g: np.ndarray, warp_f: np.ndarray, selector: int, *,
                gain: float = 1.0, bias: float = 0.0) -> np.ndarray:
        """v4b static compose (NO hood — matches inflate_runner_v4b) + optional
        photometric gain/bias, then uint8.  selector 0 -> single (ground H),
        1 -> two-plane static (far H_inf above horizon)."""
        f0 = np.where(self.far[..., None], warp_f, warp_g) if selector else warp_g
        if gain != 1.0 or bias != 0.0:
            f0 = gain * f0 + bias
        return self.recv._to_uint8(f0)

    def score(self, f0_u8: np.ndarray, f1_u8: np.ndarray, tp: np.ndarray) -> float:
        p6 = self.o.p3v2.pose6_u8(self.o.posenet, f0_u8, f1_u8)
        return float(np.mean((p6 - tp) ** 2))


# --------------------------------------------------------------------------- #
# STAGE 1 — base adjudication: transfer v4b ship poses onto a base
# --------------------------------------------------------------------------- #
def run_transfer(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = args.base
    oracle = build_oracle(base, s_r=1.0)
    comp = StaticComposer(oracle)
    ship = load_ship_table()
    n = int(args.n_pairs)
    jl = OUT / f"transfer_{base}.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[v4c-xfer {base}] horizon v={comp.v_row}  n={n}  cached={len(cache)}",
          flush=True)
    for pidx in range(n):
        if pidx in cache:
            continue
        r = ship[pidx]
        tp = oracle.targets64[pidx].copy()
        sel = int(r["selector"])
        theta = q16(np.asarray(r["p"], np.float64))
        # the ship s_t for this pair (single/two selection stored the pose only;
        # s_t rides the pfs1 D1 grid at the pair's index — read from the D2 solve)
        s_t = float(_d2_row(pidx)["s_t"])
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        wg, wf = comp.warps(f1_f, theta, s_t)
        d = comp.score(comp.compose(wg, wf, sel), f1_u8, tp)
        rec = {"pair": int(pidx), "selector": sel, "s_t": s_t,
               "d_pose_ship_on_base": float(d),
               "d_ship_kneeA": float(r["d"])}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if pidx % 25 == 0 or pidx == n - 1:
            vals = np.asarray([cache[k]["d_pose_ship_on_base"] for k in cache])
            print(f"[v4c-xfer {base} {pidx:3d}] d={d:.4f} mean={vals.mean():.6f} "
                  f"({len(cache)}) {time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize_transfer(base, cache)


_D2_CACHE: dict[int, dict] | None = None


def _d2_row(pidx: int) -> dict:
    global _D2_CACHE
    if _D2_CACHE is None:
        _D2_CACHE = _load_jl(D2_JL)
    return _D2_CACHE[pidx]


def _summarize_transfer(base: str, cache: dict[int, dict]) -> None:
    n = len(cache)
    d_on_base = np.asarray([cache[k]["d_pose_ship_on_base"] for k in sorted(cache)])
    d_kneeA = np.asarray([cache[k]["d_ship_kneeA"] for k in sorted(cache)])
    receipt = {
        "schema": "ddm_v4c_transfer.v1", "utc": _utc(), "base": base,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n,
        "ship_pose_mean_d_pose_on_base": float(d_on_base.mean()),
        "ship_pose_contribution_on_base": contribution(float(d_on_base.mean())),
        "kneeA_ship_mean_reference": float(d_kneeA.mean()),
        "v4b_gate_d_pose_ref": GATE_D_POSE_V4B,
        "note": "Transfer of the v4b Knee-A-optimal ship poses onto this base "
                "through the realizable static compose. base=kneeA is the "
                "positive control (must reproduce 0.06365); base=celldrop50 is "
                "the pre-resolve pose damage from the base swap (recovery via "
                "re-solve TBD in --mode solve).",
    }
    (OUT / f"transfer_{base}_receipt_n{n}.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


# --------------------------------------------------------------------------- #
# STAGE 2 — pose re-solve through the STATIC compose (QA59/QA60)
# --------------------------------------------------------------------------- #
def _two_plane_static_gn(comp: StaticComposer, pidx: int, s_t: float,
                         tp: np.ndarray, f1_u8: np.ndarray, f1_f: np.ndarray,
                         starts: list[np.ndarray]) -> dict:
    """Damped GN over theta(6) minimizing d_pose through the STATIC two-plane
    compose (selector=1), acceptance at f16 quantization (monotone)."""
    def pose6(theta: np.ndarray) -> np.ndarray:
        wg, wf = comp.warps(f1_f, theta, s_t)
        return comp.o.p3v2.pose6_u8(comp.o.posenet, comp.compose(wg, wf, 1), f1_u8)

    def mse(p6: np.ndarray) -> float:
        return float(np.mean((p6 - tp) ** 2))

    best = None
    n_fwd = 0
    for start in starts:
        theta = q16(start)
        cur6 = pose6(theta)
        cur = mse(cur6)
        n_fwd += 1
        lm = 1.0
        for _ in range(RELINS):
            jac = np.zeros((6, 6), np.float64)
            for k in range(6):
                dp = theta.copy()
                dp[k] += d2m.FD_STEPS[k]
                jac[:, k] = (pose6(dp) - cur6) / d2m.FD_STEPS[k]
            n_fwd += 6
            r = cur6 - tp
            accepted = False
            for _damp in range(4):
                a = jac.T @ jac + lm * np.diag(np.maximum(np.diag(jac.T @ jac), 1e-8))
                try:
                    step = np.linalg.solve(a, -(jac.T @ r))
                except np.linalg.LinAlgError:
                    break
                for scale in (1.0, 0.5):
                    cand = q16(theta + scale * step)
                    c6 = pose6(cand)
                    n_fwd += 1
                    cv = mse(c6)
                    if cv < cur:
                        theta, cur6, cur = cand, c6, cv
                        accepted = True
                        break
                if accepted:
                    lm = max(lm * 0.33, 1e-3)
                    break
                lm *= 4.0
            if not accepted:
                break
        if best is None or cur < best[0]:
            best = (cur, theta.copy())
    return {"d_two_static": float(best[0]),
            "p_two_static": [float(v) for v in best[1]], "n_fwd_two": n_fwd}


def _single_plane_static_gn(comp: StaticComposer, pidx: int, s_t: float,
                            tp: np.ndarray, f1_u8: np.ndarray, f1_f: np.ndarray,
                            start: np.ndarray) -> dict:
    """Single-plane 6-DOF GN (selector=0, full ground H) — mirrors the ck1
    single solve but scored through comp (identical primitives)."""
    def pose6(theta: np.ndarray) -> np.ndarray:
        wg = comp.warp_ground(f1_f, theta, s_t)
        return comp.o.p3v2.pose6_u8(comp.o.posenet, comp.compose(wg, wg, 0), f1_u8)

    def mse(p6: np.ndarray) -> float:
        return float(np.mean((p6 - tp) ** 2))

    theta = q16(start)
    cur6 = pose6(theta)
    cur = mse(cur6)
    n_fwd = 1
    lm = 1.0
    for _ in range(RELINS):
        jac = np.zeros((6, 6), np.float64)
        for k in range(6):
            dp = theta.copy()
            dp[k] += d2m.FD_STEPS[k]
            jac[:, k] = (pose6(dp) - cur6) / d2m.FD_STEPS[k]
        n_fwd += 6
        r = cur6 - tp
        accepted = False
        for _damp in range(4):
            a = jac.T @ jac + lm * np.diag(np.maximum(np.diag(jac.T @ jac), 1e-8))
            try:
                step = np.linalg.solve(a, -(jac.T @ r))
            except np.linalg.LinAlgError:
                break
            for scale in (1.0, 0.5):
                cand = q16(theta + scale * step)
                c6 = pose6(cand)
                n_fwd += 1
                cv = mse(c6)
                if cv < cur:
                    theta, cur6, cur = cand, c6, cv
                    accepted = True
                    break
            if accepted:
                lm = max(lm * 0.33, 1e-3)
                break
            lm *= 4.0
        if not accepted:
            break
    return {"d_single_static": float(cur),
            "p_single_static": [float(v) for v in theta], "n_fwd_single": n_fwd}


def run_solve(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    base = args.base
    oracle = build_oracle(base, s_r=1.0)
    comp = StaticComposer(oracle)
    ship = load_ship_table()
    xfer = _load_jl(OUT / f"transfer_{base}.partial.jsonl")
    # order pairs by the transfer damage (hardest first) if available, else by
    # the Knee-A ship d_pose; solve tail-first so decisive pairs land early.
    if xfer:
        order = sorted(range(600), key=lambda p: -xfer[p]["d_pose_ship_on_base"])
    else:
        order = sorted(range(600), key=lambda p: -ship[p]["d"])
    seq = order[: int(args.k)] if args.k else order
    two_all = bool(args.two_all)
    tail_cut = int(args.tail_cut)
    jl = OUT / f"solve_{base}.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[v4c-solve {base}] horizon v={comp.v_row}  seq={len(seq)}  "
          f"two_all={two_all}  tail_cut={tail_cut}  cached={len(cache)}", flush=True)
    for rank, pidx in enumerate(seq):
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[v4c-solve] --max-seconds; {len(cache)} done; re-run --resume",
                  flush=True)
            break
        s_t = float(_d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        p_ship = np.asarray(ship[pidx]["p"], np.float64)
        # single-plane GN start = ship pose (rotation-zeroed if it was two-plane)
        p0_single = p_ship.copy()
        p0_single[3:] = 0.0
        single = _single_plane_static_gn(comp, pidx, s_t, tp, f1_u8, f1_f, p0_single)
        _xr = xfer.get(pidx) if xfer else None
        rec = {"pair": int(pidx), "s_t": s_t, "rank": rank,
               "d_ship_kneeA": float(ship[pidx]["d"]),
               "d_ship_on_base": float(_xr["d_pose_ship_on_base"]) if _xr else None,
               **single}
        best_d, best_p, best_sel = single["d_single_static"], single["p_single_static"], 0
        # two-plane on the tail (by rank) OR everywhere with --two-all (QA60)
        if two_all or rank < tail_cut:
            starts = [np.asarray(p_ship, np.float64),
                      np.asarray(single["p_single_static"], np.float64)]
            two = _two_plane_static_gn(comp, pidx, s_t, tp, f1_u8, f1_f, starts)
            rec.update(two)
            if two["d_two_static"] < best_d:
                best_d, best_p, best_sel = (two["d_two_static"],
                                            two["p_two_static"], 1)
        rec["d_best_static"] = float(best_d)
        rec["p_best_static"] = [float(v) for v in best_p]
        rec["selector"] = int(best_sel)
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        done = len(cache)
        if done % 5 == 0 or rank == len(seq) - 1:
            bv = np.asarray([cache[p]["d_best_static"] for p in cache])
            print(f"[v4c-solve {base} {done:3d}/{len(seq)}] pair {pidx} rank {rank} "
                  f"single {single['d_single_static']:.4f} best {best_d:.4f} "
                  f"(sel{best_sel}) mean {bv.mean():.5f} {time.time()-t0:.0f}s",
                  flush=True)
    fj.close()
    _summarize_solve(base, cache, ship)


def _summarize_solve(base: str, cache: dict[int, dict], ship: dict[int, dict]) -> None:
    n = len(cache)
    # composed field over 600: solved best where available, else v4b ship d_pose
    comp600 = {}
    for i in range(600):
        comp600[i] = cache[i]["d_best_static"] if i in cache else float(ship[i]["d"])
    arr = np.asarray([comp600[i] for i in range(600)])
    solved = np.asarray([cache[p]["d_best_static"] for p in cache]) if n else np.array([0.0])
    ship_solved = np.asarray([ship[p]["d"] for p in cache]) if n else np.array([0.0])
    n_two = sum(1 for p in cache if cache[p].get("selector") == 1)
    receipt = {
        "schema": "ddm_v4c_solve.v1", "utc": _utc(), "base": base,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_solved": n, "n_selector_two": n_two,
        "solved_mean_d_pose": float(solved.mean()),
        "ship_mean_on_solved_pairs": float(ship_solved.mean()),
        "solved_vs_ship_delta": float(solved.mean() - ship_solved.mean()),
        "composed_mean_d_pose_600": float(arr.mean()),
        "composed_contribution_600": contribution(float(arr.mean())),
        "note": "d_best_static = min(single-static GN, two-static GN) per pair, "
                "re-solved THROUGH the realizable static compose (QA59). "
                "unsolved-on-this-base pairs use the v4b ship d_pose as PROXY.",
    }
    (OUT / f"solve_{base}_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


#: damping levels tried per relinearization before the step search gives up.
#: This is a SAFETY BOUND, not a target -- see ``ab_damped_gn``.
AB_DAMP_LEVELS = 4


def _step_below_f16_resolution(a: float, b: float, step) -> bool:
    """True when neither half of ``step`` can move (a,b) at the SHIPPED precision.

    (a,b) are quantized to float16 before they are stored and re-scored, so a
    proposed step smaller than half a float16 ULP at the current point cannot
    change the shipped values no matter how the damping ladder continues.  When
    the ladder gives up on such a step the search is genuinely EXHAUSTED at the
    representable resolution -- that is a proof of local optimality on the
    shipped lattice, and is the only honest ``converged`` this solver has.

    Without this test the three exits are indistinguishable: the bare
    ``if not accepted: break`` fires identically whether the ladder proved a
    local optimum or merely ran out of levels.  Reads ``step`` only; it never
    changes control flow or any returned value.
    """
    if step is None:
        return False
    da, db = abs(float(step[0])), abs(float(step[1]))
    return (da <= 0.5 * float(np.spacing(np.float16(a)))
            and db <= 0.5 * float(np.spacing(np.float16(b))))


def ab_damped_gn(pose6, mse, a0: float, b0: float, cur6: np.ndarray,
                 curB: float, tp: np.ndarray, *,
                 relins: int = GN_RELINS_PHOTO,
                 damp_levels: int = AB_DAMP_LEVELS):
    """The photometric (a,b) damped-Gauss-Newton, with its TERMINATION EMITTED.

    Single implementation shared by the v4c rung-B site and the v4d
    ``_refit_ab`` site, which carried byte-identical copies of this loop.

    WHY THIS EXISTS (ddm_sv1, 2026-08-01).  The loop has three distinct ways to
    stop and the shipped code collapsed all three onto one ``if not accepted:
    break``, so a solve that RAN OUT was recorded identically to one that
    CONVERGED -- and the per-pair receipt recorded neither.  Measured on the
    hardest 60 pairs (the ones carrying 86.6% of post-GN pose mass), n600 v4c
    cache, canary EXACT against the shipped ``d_rungB``:

        converged    0%
        damp_cap    ~64%   the `range(damp_levels)` ladder ran out
        relin_cap   ~36%   `relins` exhausted while still accepting steps

    i.e. the shipped solve stopped on a BOUND on 100% of the mass-carrying
    pairs and its receipt could not say so.  ``stop_reason`` is now recorded so
    that censoring can never again be invisible.  Sister of the ddm_dc1 cure in
    ``tools/sb1_seg_batch.py`` and of the registered law
    ``tac.canonical_equations.ddm_pw1_menu_saturation_discriminator_v1``.

    The defaults are UNCHANGED: this landing is observability only and is
    byte-identical by construction.  Freeing the bounds changes the shipped
    solve and is staged behind an exact gate, not taken here.

    Returns ``(a, b, cur6, curB, trace)`` where ``trace`` carries
    ``stop_reason`` in {converged, damp_cap, relin_cap, singular},
    ``n_relin`` (relinearizations entered) and ``damp_used`` (damping levels
    tried in each).
    """
    a_p, b_p = a0, b0
    lm = 1.0
    stop = "relin_cap"
    n_relin = 0
    damp_used: list[int] = []
    for _ in range(relins):
        n_relin += 1
        p6a = pose6(a_p + GAIN_FD, b_p)
        p6b = pose6(a_p, b_p + BIAS_FD)
        jb = np.stack([(p6a - cur6) / GAIN_FD, (p6b - cur6) / BIAS_FD], 1)
        r = cur6 - tp
        accepted = False
        singular = False
        used = 0
        last_step = None
        for _damp in range(damp_levels):
            used += 1
            aa = jb.T @ jb + lm * np.diag(np.maximum(np.diag(jb.T @ jb), 1e-8))
            try:
                step = np.linalg.solve(aa, -(jb.T @ r))
            except np.linalg.LinAlgError:
                singular = True
                break
            last_step = step
            for scale in (1.0, 0.5):
                ca, cb = a_p + scale * step[0], b_p + scale * step[1]
                c6 = pose6(ca, cb)
                cv = mse(c6)
                if cv < curB:
                    a_p, b_p, cur6, curB = ca, cb, c6, cv
                    lm = max(lm * 0.3, 1e-4)
                    accepted = True
                    break
            if accepted:
                break
            lm *= 8.0
        damp_used.append(used)
        if not accepted:
            stop = ("singular" if singular
                    else "converged" if _step_below_f16_resolution(
                        a_p, b_p, last_step) else "damp_cap")
            break
    trace = {"stop_reason": stop, "n_relin": n_relin,
             "damp_used": damp_used,
             "relins_bound": int(relins), "damp_bound": int(damp_levels)}
    return a_p, b_p, cur6, curB, trace


def ab_stop_census(rows) -> dict:
    """Run-level census of ``ab_damped_gn`` terminations across per-pair rows.

    Reports the DENOMINATOR and the count that stopped on a bound rather than
    on the criterion.  Rows predating the trace are counted as ``unrecorded``
    so an old cache reads as UNKNOWN, never as converged.
    """
    total = 0
    by = {"converged": 0, "damp_cap": 0, "relin_cap": 0, "singular": 0,
          "unrecorded": 0}
    for row in rows:
        total += 1
        by[str(row.get("ab_stop", "unrecorded"))] = by.get(
            str(row.get("ab_stop", "unrecorded")), 0) + 1
    censored = by["damp_cap"] + by["relin_cap"] + by["singular"]
    return {"n_rows": total, "by_stop_reason": by,
            "n_stopped_on_bound": censored,
            "frac_stopped_on_bound": (censored / total) if total else 0.0,
            "note": "stopped_on_bound counts damp_cap+relin_cap+singular; any "
                    "value >0 means the solve is CENSORED at that bound"}


# --------------------------------------------------------------------------- #
# STAGE 3 — photometric rungs (QA62): rung-B (a,b) + rung-A row-shear
# --------------------------------------------------------------------------- #
def run_photo(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    base = args.base
    oracle = build_oracle(base, s_r=1.0)
    comp = StaticComposer(oracle)
    ship = load_ship_table()
    # pose source: the re-solved field (if present) else the v4b ship table
    solve_rows = _load_jl(OUT / f"solve_{base}.partial.jsonl")
    if args.pose_source == "resolve" and not solve_rows:
        raise SystemExit("pose-source=resolve but no solve rows found")
    n = int(args.n_pairs)
    jl = OUT / f"photo_{base}_{args.pose_source}.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[v4c-photo {base}/{args.pose_source}] n={n} cached={len(cache)}", flush=True)
    for pidx in range(n):
        if pidx in cache:
            continue
        s_t = float(_d2_row(pidx)["s_t"])
        tp = oracle.targets64[pidx].copy()
        if args.pose_source == "resolve" and pidx in solve_rows:
            theta = q16(np.asarray(solve_rows[pidx]["p_best_static"], np.float64))
            sel = int(solve_rows[pidx]["selector"])
        else:
            theta = q16(np.asarray(ship[pidx]["p"], np.float64))
            sel = int(ship[pidx]["selector"])
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        wg, wf = comp.warps(f1_f, theta, s_t)

        def pose6(gain: float, bias: float, *, _wg=wg, _wf=wf, _sel=sel,
                  _f1=f1_u8) -> np.ndarray:
            return comp.o.p3v2.pose6_u8(
                comp.o.posenet, comp.compose(_wg, _wf, _sel, gain=gain, bias=bias), _f1)

        def mse(p6: np.ndarray, *, _tp=tp) -> float:
            return float(np.mean((p6 - _tp) ** 2))

        cur6 = pose6(1.0, 0.0)
        d_ctrl = mse(cur6)
        # RUNG B: (a,b) GN
        a_p, b_p, cur6B, curB, ab_trace = ab_damped_gn(
            pose6, mse, 1.0, 0.0, cur6, d_ctrl, tp)
        # quantize (a,b) to f16 (the shipped precision) and re-score honestly
        a_q, b_q = float(np.float16(a_p)), float(np.float16(b_p))
        d_photoB = mse(pose6(a_q, b_q))
        # RUNG A: rolling-shutter row-shear at the (a,b)-corrected exposure.
        # ONLY the SHIPPABLE candidates: beta = g * sign(yaw) for global g (the
        # receiver derives the sign from pose[5] free; g is one manifest const).
        # Record d PER g so the build picks the global g* by exact arithmetic.
        yaw_sign = 1.0 if theta[5] >= 0.0 else -1.0
        rungA_by_g: dict[str, float] = {"0.0": float(d_photoB)}
        best_A, best_beta = d_photoB, 0.0
        for g in RS_GLOBAL_G:
            beta = g * yaw_sign
            wg_t, wf_t = comp.warps(f1_f, theta, s_t, 1.0 - beta / 2.0)
            wg_b, wf_b = comp.warps(f1_f, theta, s_t, 1.0 + beta / 2.0)
            f0_t = np.where(comp.far[..., None], wf_t, wg_t) if sel else wg_t
            f0_b = np.where(comp.far[..., None], wf_b, wg_b) if sel else wg_b
            f0 = (1.0 - comp.alpha_row) * f0_t + comp.alpha_row * f0_b
            if a_q != 1.0 or b_q != 0.0:
                f0 = a_q * f0 + b_q
            dv = mse(comp.o.p3v2.pose6_u8(comp.o.posenet, comp.recv._to_uint8(f0), f1_u8))
            rungA_by_g[str(g)] = float(dv)
            if dv < best_A:
                best_A, best_beta = dv, beta
        rec = {"pair": int(pidx), "selector": sel, "s_t": s_t,
               "d_ctrl": float(d_ctrl), "d_rungB": float(d_photoB),
               "a": a_q, "b": b_q, "d_rungAB": float(best_A),
               "rungA_beta": float(best_beta), "rungA_by_g": rungA_by_g,
               "yaw_sign": yaw_sign,
               "ab_stop": ab_trace["stop_reason"],
               "ab_relins": ab_trace["n_relin"],
               "ab_damp_used": ab_trace["damp_used"],
               "p": [float(v) for v in theta]}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if pidx % 25 == 0 or pidx == n - 1:
            arr = np.asarray([cache[k]["d_rungAB"] for k in cache])
            ctl = np.asarray([cache[k]["d_ctrl"] for k in cache])
            print(f"[v4c-photo {base} {pidx:3d}] ctrl {d_ctrl:.4f} B {d_photoB:.4f} "
                  f"(a{a_q:.3f},b{b_q:+.1f}) AB {best_A:.4f}(beta{best_beta:+.1f}) | "
                  f"mean ctrl {ctl.mean():.5f} -> AB {arr.mean():.5f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize_photo(base, args.pose_source, cache, ship)


def _summarize_photo(base: str, pose_source: str, cache: dict[int, dict],
                     ship: dict[int, dict]) -> None:
    n = len(cache)
    ab = {i: cache[i]["d_rungAB"] if i in cache else float(ship[i]["d"])
          for i in range(600)}
    ctrl = {i: cache[i]["d_ctrl"] if i in cache else float(ship[i]["d"])
            for i in range(600)}
    ab_arr = np.asarray([ab[i] for i in range(600)])
    ctrl_arr = np.asarray([ctrl[i] for i in range(600)])
    # SHIPPABLE global-g rung-A selection: one manifest constant, sign from yaw
    g_means = {}
    for g in ("0.0", *(str(x) for x in RS_GLOBAL_G)):
        vals = [cache[i]["rungA_by_g"].get(g, cache[i]["d_rungB"])
                if i in cache else float(ship[i]["d"]) for i in range(600)]
        g_means[g] = float(np.mean(vals))
    g_star = min(g_means, key=g_means.get)
    rate_base = 25.0 * BASES[base].stat().st_size / 37_545_489
    receipt = {
        "schema": "ddm_v4c_photo.v1", "utc": _utc(), "base": base,
        "pose_source": pose_source,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": n,
        "ctrl_mean_d_pose_600": float(ctrl_arr.mean()),
        "ctrl_contribution_600": contribution(float(ctrl_arr.mean())),
        "rungAB_mean_d_pose_600": float(ab_arr.mean()),
        "rungAB_contribution_600": contribution(float(ab_arr.mean())),
        "rungAB_delta_S_600": (contribution(float(ab_arr.mean()))
                               - contribution(float(ctrl_arr.mean()))),
        "n_rungA_active": int(sum(1 for i in cache if cache[i]["rungA_beta"] != 0.0)),
        "rungA_global_g_means": g_means,
        "rungA_global_g_star": g_star,
        "shippable_mean_d_pose_600_at_g_star": g_means[g_star],
        "shippable_contribution_600_at_g_star": contribution(g_means[g_star]),
        "rungB_ab_stop_census": ab_stop_census(cache.values()),
        "note": "rung B = f16 (a,b) auto-exposure applied on the realizable "
                "static compose; rung A = rolling-shutter row-shear (best beta). "
                "ctrl = compose at (a=1,b=0). ADVISORY; the n600 evaluate gate is "
                "the authority.  seg (Knee-A tokens) 0.00553676, rate ref "
                f"{rate_base:.6f}.",
    }
    (OUT / f"photo_{base}_{pose_source}_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("transfer", "solve", "photo"), required=True)
    ap.add_argument("--base", choices=tuple(BASES), required=True)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--k", type=int, default=0, help="solve: limit to first K in order")
    ap.add_argument("--tail-cut", type=int, default=112,
                    help="solve: run two-plane GN for the first N ranked pairs")
    ap.add_argument("--two-all", action="store_true",
                    help="solve: run two-plane GN for every pair (QA60)")
    ap.add_argument("--pose-source", choices=("shiptable", "resolve"),
                    default="shiptable", help="photo: pose field source")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.mode == "transfer":
        run_transfer(args)
    elif args.mode == "solve":
        run_solve(args)
    else:
        run_photo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
