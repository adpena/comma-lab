#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ck1 — pose RE-SOLVE on the Knee-A token base (the composed decision cell).

QA06 answered a MEASURED fact (real evaluate.py, rc=0, full n600): the Knee-A
wr1 archive standalone scored S 2.4097 vs ref 2.2566 = +0.153 net REJECT.  The
verdict is INSTANCE-scoped: the shipped pose member (per-pair warp pose = t_p,
receiver s_r=0) was solved on the FULL-token pfs1 base; the Knee-A base DROPPED
288 sky + 170 hood + 28 stable-road latent cells, which FROZE the far-field
content the warp/PoseNet reads -> d_pose 0.22144216 -> 0.28002128 (+0.185 S).
Knee law: the composed candidate must RE-SOLVE the pose on the frames it ships.

This tool points the p3v2/pfs1 WarpPoseOracle at the Knee-A archive (override
``oracle.packet`` post-construction; ZERO edits to the shared oracle file that
qa43 imports) and re-solves the per-pair pose ON that base:

  control   d_pose at the SHIPPED params (t_p, s_r=0, D1 s_t) on the Knee-A base
            for N pairs -> MUST reproduce the gate mean 0.28002128 (24-pair spot
            check first; substrate-identity gate; STOP on mismatch = confound).
  transfer  re-SCORE the FULL-base solved poses (d2 single p_star, qa43 two-plane
            p_two_star) on the Knee-A base (1 fwd/pair) -> a $0 lower-bound
            estimate of how much of the full-base pose solution survives the drop.
  solve     RE-SOLVE on the Knee-A base, tail-first (by full-base d_pose_solved):
            per pair best-of {single-plane 6-DOF GN (d2m.solve_pair_gn),
            two-plane near/far GN multi-start (qa43 compose)}.  Resumable JSONL.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  One n600 scorer job at a time
(this is PoseNet-only per-pair work; no SegNet n600 job).
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
    os.environ.setdefault(_tv, "4")

import numpy as np

np.seterr(all="ignore")

sys.path.insert(0, "experiments")
import ddm_pfs1_ep_warp_pose_solve as d2m  # WarpPoseOracle, solve_pair_gn, FD_STEPS, ST_GRID

KNEEA_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeA_safe_274k_archive.zip")
D2_JL = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl")
QA43_JL = Path("/Volumes/VertigoDataTier/pact/ddm_qa43_20260729/two_plane_probe_v2.partial.jsonl")
GT_NPZ = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_ck1_20260729")

GATE_D_POSE = 0.28002128   # the Knee-A realized gate mean (evaluate.py rc=0)
REF_D_POSE = 0.22144216    # pfs1 D1 reference gate mean
RELINS = 4


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def q16(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float64).astype(np.float16).astype(np.float64)


def contribution(dpm: float) -> float:
    return float(np.sqrt(10.0 * float(dpm)))


def _load_jl(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                rows[int(r["pair"])] = r
    return rows


def build_kneeA_oracle(s_r: float) -> "d2m.WarpPoseOracle":
    """Construct the pfs1 oracle then OVERRIDE its packet to the Knee-A tokens.

    The Knee-A archive is the ``ddm_pfs1_composed_archive.v3_warp`` grammar
    (sectioned members: state/tokens.dr7t + renderer.sec + selector.sec +
    pose_stub.sec + pose_warp.stp), NOT the pb1 TR1 monolithic packet the default
    __init__ loads.  We rebuild the TR1 packet from the sectioned members EXACTLY
    as the gate's inflate_runner.py does (decode_token_codes -> _encode_tokens ->
    build_packet), so ``oracle.f1(pidx) = render_frame1_camera_uint8(packet, pidx)``
    renders the DROPPED-token frame_1 the evaluator scored.  The run_control mode
    asserts this reproduces the gate d_pose (substrate identity).  No shared-file
    edit; a post-construction override of oracle.packet only.
    """
    from ddm_r7_token_coder import decode_token_codes
    oracle = d2m.WarpPoseOracle(s_r=s_r)
    rt = oracle.rt
    with zipfile.ZipFile(KNEEA_ARCHIVE) as z:
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


def parse_pose_warp_shipped() -> tuple[np.ndarray, np.ndarray]:
    """Read the EXACT shipped pose member (f16 tp + s_t index) from the Knee-A
    archive's state/pose_warp.stp — the gate warped f0 with THESE, so the control
    must reproduce them (not oracle.targets64/load_solved_st) for a true identity
    check.  Mirrors inflate_runner.parse_pose_warp."""
    import struct

    import brotli
    from ddm_r7_token_coder import decode_token_codes
    with zipfile.ZipFile(KNEEA_ARCHIVE) as z:
        payload = z.read("state/pose_warp.stp")
    if payload[:8] != b"PFS1WPB1":
        raise SystemExit("pose_warp magic differs")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off); off += 4
    (l1,) = struct.unpack_from("<I", payload, off); off += 4
    tp_coded = payload[off:off + l1]; off += l1
    (l2,) = struct.unpack_from("<I", payload, off); off += 4
    st_coded = payload[off:off + l2]; off += l2
    tp = np.frombuffer(brotli.decompress(tp_coded),
                       dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    st_idx = np.asarray(decode_token_codes(st_coded),
                        dtype=np.int64).reshape(-1)[:n_pairs]
    return tp, st_idx


def _tail_order() -> tuple[dict[int, dict], list[int]]:
    d2 = _load_jl(D2_JL)
    if len(d2) != 600:
        raise SystemExit(f"d2 solve incomplete: {len(d2)}/600")
    order = sorted(d2, key=lambda p: -d2[p]["d_pose_solved"])
    return d2, order


# --------------------------------------------------------------------------- #
# two-plane compose + GN (mirrors ddm_qa43_two_plane_parallax_probe on this base)
# --------------------------------------------------------------------------- #
def _two_plane_solver(oracle: "d2m.WarpPoseOracle"):
    recv, p3v2 = oracle.recv, oracle.p3v2
    lstars = np.load(GT_NPZ)["lstars"]
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)

    def solve(pidx: int, s_t: float, tp: np.ndarray, p_star: np.ndarray):
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        cls = lstars[pidx][np.ix_(yi, xi)]
        m_far = (cls == 2)[..., None]
        m_hood = cls == 4

        def compose_two(theta: np.ndarray) -> np.ndarray:
            th = np.asarray(theta, np.float64)
            hg = recv.pose_to_homography(th, oracle.K, oracle.Kinv, s_t, 1.0, 0.0)
            hf = recv.pose_to_homography(th, oracle.K, oracle.Kinv, 0.0, 1.0, 0.0)
            f0 = np.where(m_far, recv.warp_rgb(f1_f, hf, oracle.grid),
                          recv.warp_rgb(f1_f, hg, oracle.grid))
            f0[m_hood] = f1_f[m_hood]
            return recv._to_uint8(f0)

        def pose6_two(theta: np.ndarray) -> np.ndarray:
            return p3v2.pose6_u8(oracle.posenet, compose_two(theta), f1_u8)

        def mse(p6: np.ndarray) -> float:
            return float(np.mean((p6 - tp) ** 2))

        def gn_from(theta_init: np.ndarray) -> tuple[float, np.ndarray, int]:
            nf = 0
            theta = q16(theta_init)
            cur6 = pose6_two(theta)
            cur = mse(cur6)
            nf += 1
            lm = 1.0
            for _ in range(RELINS):
                jac = np.zeros((6, 6), np.float64)
                for k in range(6):
                    dp = theta.copy()
                    dp[k] += d2m.FD_STEPS[k]
                    jac[:, k] = (pose6_two(dp) - cur6) / d2m.FD_STEPS[k]
                nf += 6
                r = cur6 - tp
                accepted = False
                for _damp in range(4):
                    a = jac.T @ jac + lm * np.diag(
                        np.maximum(np.diag(jac.T @ jac), 1e-8))
                    try:
                        step = np.linalg.solve(a, -(jac.T @ r))
                    except np.linalg.LinAlgError:
                        break
                    for scale in (1.0, 0.5):
                        cand = q16(theta + scale * step)
                        c6 = pose6_two(cand)
                        nf += 1
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
            return cur, theta, nf

        p0 = tp.copy()
        p0[3:] = 0.0
        two_at_pstar = mse(pose6_two(q16(p_star)))
        cur_a, th_a, nf_a = gn_from(p0)
        cur_b, th_b, nf_b = gn_from(np.asarray(p_star, np.float64))
        cur, theta = (cur_a, th_a) if cur_a <= cur_b else (cur_b, th_b)
        return {"d_two_at_pstar": float(two_at_pstar),
                "d_two_from_p0": float(cur_a), "d_two_from_pstar": float(cur_b),
                "d_two_solved": float(cur), "p_two_star": [float(v) for v in theta],
                "n_fwd_two": int(1 + nf_a + nf_b)}

    return solve


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def run_control(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oracle = build_kneeA_oracle(s_r=0.0)   # gate ships s_r=0
    tp_ship, st_idx = parse_pose_warp_shipped()   # the EXACT shipped pose member
    n = int(args.n_pairs)
    jl = OUT / "ck1_control.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    for pidx in range(n):
        if pidx in cache:
            continue
        f1 = oracle.f1(pidx)
        tp = tp_ship[pidx]
        s_t = d2m.ST_GRID[int(st_idx[pidx])]
        d = oracle.d_pose_shipped(pidx, f1, tp, s_t)
        rec = {"pair": int(pidx), "d_pose_shipped_sr0": float(d), "s_t": float(s_t)}
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if pidx % 8 == 0 or pidx == n - 1:
            vals = np.asarray([cache[k]["d_pose_shipped_sr0"] for k in cache])
            print(f"[ck1-ctrl {pidx:3d}] d_pose={d:.5f} running_mean={vals.mean():.6f} "
                  f"(gate 0.28002128) n={len(cache)}", flush=True)
    fj.close()
    vals = np.asarray([cache[k]["d_pose_shipped_sr0"] for k in sorted(cache)])
    mean = float(vals.mean())
    receipt = {
        "schema": "ddm_ck1_control.v1", "utc": _utc(), "n_pairs": int(len(vals)),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "control_d_pose_mean": mean, "gate_d_pose_mean": GATE_D_POSE,
        "abs_delta_vs_gate": abs(mean - GATE_D_POSE),
        "substrate_identity_ok": bool(abs(mean - GATE_D_POSE) < 1.5e-3),
        "note": "s_r=0 t_p warp pose on the Knee-A base; reproduce the gate mean "
                "(instrument<->evaluator drift band ~1e-3). n<600 is a spot check "
                "-> mean will not equal the 600-gate mean exactly.",
    }
    (OUT / f"ck1_control_receipt_n{len(vals)}.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def run_transfer(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d2, order = _tail_order()
    qa43 = _load_jl(QA43_JL)
    tail = set(order[:112])
    oracle = build_kneeA_oracle(s_r=1.0)
    recv, p3v2 = oracle.recv, oracle.p3v2
    lstars = np.load(GT_NPZ)["lstars"]
    ch, cw = recv.CAMERA_H, recv.CAMERA_W
    yi = np.minimum((np.arange(ch) * 384) // ch, 383)
    xi = np.minimum((np.arange(cw) * 512) // cw, 511)
    n = int(args.n_pairs)
    jl = OUT / "ck1_transfer.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    for pidx in range(n):
        if pidx in cache:
            continue
        f1_u8 = oracle.f1(pidx)
        tp = oracle.targets64[pidx]
        s_t = float(d2[pidx]["s_t"])
        # transfer the full-base SINGLE-plane p_star (s_r active)
        d_single = oracle.d_pose_shipped(pidx, f1_u8, np.asarray(d2[pidx]["p_star"]), s_t)
        rec = {"pair": int(pidx), "s_t": s_t, "d_single_xfer": float(d_single)}
        # transfer the full-base TWO-plane p_two_star (tail only)
        if pidx in tail and pidx in qa43 and "p_two_star" in qa43[pidx]:
            f1_f = f1_u8.astype(np.float64)
            cls = lstars[pidx][np.ix_(yi, xi)]
            m_far = (cls == 2)[..., None]
            m_hood = cls == 4
            th = np.asarray(qa43[pidx]["p_two_star"], np.float64)
            hg = recv.pose_to_homography(th, oracle.K, oracle.Kinv, s_t, 1.0, 0.0)
            hf = recv.pose_to_homography(th, oracle.K, oracle.Kinv, 0.0, 1.0, 0.0)
            f0 = np.where(m_far, recv.warp_rgb(f1_f, hf, oracle.grid),
                          recv.warp_rgb(f1_f, hg, oracle.grid))
            f0[m_hood] = f1_f[m_hood]
            p6 = p3v2.pose6_u8(oracle.posenet, recv._to_uint8(f0), f1_u8)
            rec["d_two_xfer"] = float(np.mean((p6 - tp) ** 2))
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        if pidx % 40 == 0 or pidx == n - 1:
            print(f"[ck1-xfer {pidx:3d}] single={d_single:.5f} "
                  f"two={rec.get('d_two_xfer', float('nan')):.5f}", flush=True)
    fj.close()
    _summarize_transfer(cache, order)


def _summarize_transfer(cache: dict[int, dict], order: list[int]) -> None:
    n = len(cache)
    single = np.asarray([cache[i]["d_single_xfer"] for i in sorted(cache)])
    tail = [p for p in order[:112] if p in cache and "d_two_xfer" in cache[p]]
    # composed field: per pair min(single_xfer, two_xfer if tail)
    comp = {}
    for i in sorted(cache):
        v = cache[i]["d_single_xfer"]
        if "d_two_xfer" in cache[i]:
            v = min(v, cache[i]["d_two_xfer"])
        comp[i] = v
    comp_arr = np.asarray([comp[i] for i in sorted(comp)])
    receipt = {
        "schema": "ddm_ck1_transfer.v1", "utc": _utc(), "n_pairs": int(n),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "note": "re-SCORE of full-base solved poses on the Knee-A base (NOT a "
                "re-solve). Lower bound on the re-solve; a valid pose field the "
                "composed archive could ship as-is.",
        "single_xfer_mean": float(single.mean()) if n else None,
        "two_xfer_mean_tail": float(np.mean([cache[p]["d_two_xfer"] for p in tail]))
        if tail else None,
        "composed_min_mean": float(comp_arr.mean()) if n else None,
        "composed_contribution": contribution(float(comp_arr.mean())) if n else None,
        "gate_stale_d_pose": GATE_D_POSE, "ref_d_pose": REF_D_POSE,
        "n_tail_two": len(tail),
    }
    (OUT / f"ck1_transfer_receipt_n{n}.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def run_solve(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    d2, order = _tail_order()
    tail = set(order[:112])
    # tail-first ordering: decisive tail pairs get solved before the cruise bulk
    seq = order if args.tail_first else list(range(600))
    seq = seq[: int(args.k)] if args.k else seq
    oracle = build_kneeA_oracle(s_r=1.0)
    two_solve = _two_plane_solver(oracle)
    jl = OUT / "ck1_solve.partial.jsonl"
    cache = _load_jl(jl)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    for pidx in seq:
        if pidx in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[ck1-solve] --max-seconds; {len(cache)} done; re-run --resume",
                  flush=True)
            break
        s_t = float(d2[pidx]["s_t"])
        tp = oracle.targets64[pidx].copy()
        # SINGLE-plane 6-DOF GN on the Knee-A base (RUNG P0 on this base)
        single = d2m.solve_pair_gn(oracle, pidx, s_t, relins=RELINS)
        rec = {"pair": int(pidx), "s_t": s_t, "in_tail": bool(pidx in tail),
               "d_pose_warp_fullbase": float(d2[pidx]["d_pose_warp"]),
               "d_single_fullbase": float(d2[pidx]["d_pose_solved"]),
               "d_single_kneeA": float(single["d_pose_solved"]),
               "p_single_kneeA": single["p_star"],
               "n_fwd_single": int(single["n_forwards"])}
        best_d = rec["d_single_kneeA"]
        best_p = rec["p_single_kneeA"]
        best_kind = "single"
        # TWO-plane GN only where it can matter (tail)
        if pidx in tail:
            two = two_solve(pidx, s_t, tp, np.asarray(d2[pidx]["p_star"]))
            rec.update({k: two[k] for k in (
                "d_two_at_pstar", "d_two_from_p0", "d_two_from_pstar",
                "d_two_solved", "p_two_star", "n_fwd_two")})
            if two["d_two_solved"] < best_d:
                best_d, best_p, best_kind = two["d_two_solved"], two["p_two_star"], "two"
        rec["d_best_kneeA"] = float(best_d)
        rec["p_best_kneeA"] = [float(v) for v in best_p]
        rec["best_kind"] = best_kind
        fj.write(json.dumps(rec) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[pidx] = rec
        done = len(cache)
        tail_done = [p for p in cache if cache[p].get("in_tail")]
        if done % 5 == 0 or pidx == seq[-1]:
            bt = np.asarray([cache[p]["d_best_kneeA"] for p in tail_done]) \
                if tail_done else np.asarray([0.0])
            print(f"[ck1-solve {done:3d}/{len(seq)}] pair {pidx} "
                  f"warp_full {rec['d_pose_warp_fullbase']:.4f} "
                  f"single_kneeA {rec['d_single_kneeA']:.4f} "
                  f"best {best_d:.4f} ({best_kind}) | tail_best_mean {bt.mean():.4f} "
                  f"({len(tail_done)} tail) {time.time()-t0:.0f}s", flush=True)
    fj.close()
    _summarize_solve(cache, d2, order)


def _summarize_solve(cache: dict[int, dict], d2: dict[int, dict],
                     order: list[int]) -> None:
    n = len(cache)
    tail = [p for p in order[:112] if p in cache]
    # composed d_pose field over solved pairs: min(single_kneeA, two if tail).
    # for any UNSOLVED pair, fall back to the full-base best (upper bound proxy).
    comp = {}
    for i in range(600):
        if i in cache:
            comp[i] = cache[i]["d_best_kneeA"]
        else:
            comp[i] = d2[i]["d_pose_solved"]  # full-base proxy (unsolved-on-kneeA)
    comp_arr = np.asarray([comp[i] for i in range(600)])
    tail_two_win = [p for p in tail if cache[p].get("best_kind") == "two"]
    receipt = {
        "schema": "ddm_ck1_solve.v1", "utc": _utc(), "n_solved": int(n),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "gate_stale_d_pose": GATE_D_POSE, "ref_d_pose": REF_D_POSE,
        "n_tail_solved": len(tail),
        "tail_best_mean_kneeA": float(np.mean([cache[p]["d_best_kneeA"] for p in tail]))
        if tail else None,
        "tail_two_wins": len(tail_two_win),
        "composed_d_pose_mean_600": float(comp_arr.mean()),
        "composed_contribution_600": contribution(float(comp_arr.mean())),
        "composed_note": "unsolved-on-kneeA pairs use full-base d_pose_solved as a "
                         "PROXY (upper bound: kneeA re-solve can only match/beat once "
                         "run); a FULLY-solved field replaces the proxy.",
        "single_only_mean_solved": float(np.mean(
            [cache[p]["d_single_kneeA"] for p in cache])) if n else None,
    }
    (OUT / "ck1_solve_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("control", "transfer", "solve"), required=True)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--k", type=int, default=0, help="solve: limit to first K in order")
    ap.add_argument("--tail-first", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.mode == "control":
        run_control(args)
    elif args.mode == "transfer":
        run_transfer(args)
    else:
        run_solve(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
