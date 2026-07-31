#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ps1 — THE POSE STAGE on the B-control parent (task #791, fu1 rank-1).

Receiver-realized pose ladder on ONE frozen parent: the B-control checkpoint
export (compile_archive_from_checkpoint, cell-mask-aware), archive sha 438bc022,
d_seg 0.005114 realized (zb1 S1 dress rehearsal, deploy parity 1.86e-7). Its
pose stream is the zeros-frame0 inert stub. This harness measures, all through
the frozen CPU-torch PoseNet + banked GT targets (rows[i].center, MSE surface):

  S0  stub        f0 = zeros                       (the inert executable meaning)
  S1  warp base   f0 = warp(f1, H(p0; s_t))        (p0 = tp-translation, rot=0;
                                                     the ax1 §4b carried-ξ warp)
  S2  terminal    f0 = warp(f1, H(p*; s_t))         (p* = damped-GN 6-DOF solve)

REUSE (no shared-tool edits): the PROVEN pfs1 WarpPoseOracle + solve_pair_gn are
imported and the parent packet is swapped to B-control. Same p3v2 frozen PoseNet
authority, same vendored pfs1_warp_receiver warp primitives (byte-identical to
the engine per the D1 positive control), same D1 per-pair s_t grid init.

Axis: [macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. score_claim=false;
research_only. Pointer 0.1910828242 [contest-CPU] UNMOVED. No PoseNet ran until
this harness is invoked; it consumes the scorer slot.

Modes:
  stub   batched S0 d_pose (f0=zeros) over all pairs -> summary json (fast).
  solve  serial S1+S2 damped-GN per pair -> resumable JSONL {pair, d_pose_stub,
         d_pose_warp(S1), d_pose_solved(S2), p_star, s_t, n_forwards}.
  agg    aggregate the JSONL -> n600 ladder means + custody summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

B_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_zb1_s1_dress_rehearsal_20260730/"
    "B_compiled_archive.bin"
)
B_ARCHIVE_SHA = "438bc022fcd835ab68c3d12d7fa6f8e212600653478b8edc4da7d3ad7b4f5336"
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_ps1_20260730")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bind_oracle_to_b_control(s_r: float):
    """Construct the proven pfs1 oracle, then swap the parent to B-control.

    The oracle's __init__ hardcodes the pb1 p2c seg archive purely to render f1;
    swapping self.packet to the B-control packet redirects ALL rendering to the
    B-control parent (render_frame1_camera_uint8 reads self.packet). The frozen
    PoseNet, vendored warp primitives, and banked GT targets are parent-agnostic.
    """
    import ddm_pfs1_ep_warp_pose_solve as pf

    from tac.optimization import ddm_tr1_runtime as rt

    if not B_ARCHIVE.is_file():
        raise SystemExit(f"B-control archive absent: {B_ARCHIVE}")
    got = _sha256_file(B_ARCHIVE)
    if got != B_ARCHIVE_SHA:
        raise SystemExit(f"B-control archive sha drift: {got} != {B_ARCHIVE_SHA}")

    oracle = pf.WarpPoseOracle(s_r=float(s_r))
    parsed = rt.parse_archive(B_ARCHIVE.read_bytes())
    oracle.packet = rt.parse_packet(rt.reemit_packet(parsed.packet))
    return pf, oracle


def run_stub(args: argparse.Namespace) -> None:
    """S0: batched d_pose with f0=zeros over all pairs (the inert stub)."""
    import torch

    torch.set_num_threads(int(args.threads))
    from train_witness_realized_through_R_mlx import cpu_verdict_d_pose_batch

    pf, oracle = _bind_oracle_to_b_control(args.s_r)
    n = int(args.n_pairs)
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dposes = np.zeros(n, dtype=np.float64)
    zeros = None
    for c0 in range(0, n, int(args.chunk)):
        c1 = min(c0 + int(args.chunk), n)
        idxs = list(range(c0, c1))
        frames = [oracle.f1(i) for i in idxs]
        if zeros is None:
            zeros = np.zeros_like(frames[0])
        targets = [oracle.targets64[i] for i in idxs]
        for b0 in range(0, len(idxs), int(args.pose_batch)):
            b1 = min(b0 + int(args.pose_batch), len(idxs))
            dp = cpu_verdict_d_pose_batch(
                oracle.posenet,
                [zeros] * (b1 - b0),
                frames[b0:b1],
                targets[b0:b1],
            )
            dposes[c0 + b0 : c0 + b1] = dp
        print(
            f"[stub {c0}:{c1}] running d_pose_mean {dposes[:c1].mean():.6f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    summary = {
        "schema": "ddm_ps1_stub.v1",
        "stage": "S0_stub_zeros_frame0",
        "evidence_axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False,
        "research_only": True,
        "parent_archive": str(B_ARCHIVE),
        "parent_archive_sha256": B_ARCHIVE_SHA,
        "n_pairs": n,
        "d_pose_stub_mean": float(dposes.mean()),
        "d_pose_stub_max": float(dposes.max()),
        "d_pose_stub_median": float(np.median(dposes)),
        "pose_term_contribution": float(np.sqrt(10.0 * dposes.mean())),
        "wall_seconds": time.time() - t0,
    }
    out = OUT / "ps1_S0_stub_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    np.save(OUT / "ps1_S0_stub_dposes.npy", dposes)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def run_solve(args: argparse.Namespace) -> None:
    """S1+S2: per-pair damped GN. d_pose_warp=S1 (p0), d_pose_solved=S2 (p*)."""
    import torch

    torch.set_num_threads(int(args.threads))
    pf, oracle = _bind_oracle_to_b_control(args.s_r)
    n = int(args.n_pairs)
    st_idx = pf.load_solved_st(n)
    OUT.mkdir(parents=True, exist_ok=True)
    jl = OUT / "ps1_ladder.partial.jsonl"

    cache: dict[int, dict] = {}
    if jl.exists() and args.resume:
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        print(f"[solve] resume: {len(cache)} cached", flush=True)

    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    for i in range(int(args.start), min(int(args.end), n)):
        if i in cache:
            continue
        if args.max_seconds and (time.time() - t0) > float(args.max_seconds):
            print(
                f"[solve] --max-seconds at pair {i}; {len(cache)} done; "
                "re-run --resume",
                flush=True,
            )
            fj.close()
            raise SystemExit(2)
        row = pf.solve_pair_gn(
            oracle, i, pf.ST_GRID[int(st_idx[i])], relins=int(args.relins)
        )
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[i] = row
        if i % 10 == 0 or i == n - 1:
            done = len(cache)
            elapsed = time.time() - t0
            rate = elapsed / max(done, 1)
            print(
                f"[solve pair {i}] warp {row['d_pose_warp']:.5f} -> "
                f"solved {row['d_pose_solved']:.5f} | {done} done "
                f"{elapsed:.0f}s ({rate:.1f}s/pair)",
                flush=True,
            )
    fj.close()
    print(f"[solve] complete: {len(cache)} pairs", flush=True)


def run_agg(args: argparse.Namespace) -> None:
    """Aggregate the ladder JSONL -> n600 means + custody summary."""
    jl = OUT / "ps1_ladder.partial.jsonl"
    if not jl.is_file():
        raise SystemExit(f"no ladder jsonl: {jl}")
    rows: dict[int, dict] = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    n = int(args.n_pairs)
    have = sorted(rows)
    warp = np.array([rows[i]["d_pose_warp"] for i in have], dtype=np.float64)
    solved = np.array([rows[i]["d_pose_solved"] for i in have], dtype=np.float64)
    nfwd = np.array([rows[i]["n_forwards"] for i in have], dtype=np.float64)

    stub_path = OUT / "ps1_S0_stub_dposes.npy"
    stub_mean = None
    if stub_path.is_file():
        stub = np.load(stub_path)
        stub_have = stub[have] if len(stub) >= n else stub
        stub_mean = float(stub_have.mean())

    def term(mean):
        return float(np.sqrt(10.0 * mean)) if mean is not None else None

    summary = {
        "schema": "ddm_ps1_agg.v1",
        "evidence_axis": "[macOS-CPU frozen-PoseNet advisory]",
        "score_claim": False,
        "research_only": True,
        "parent_archive_sha256": B_ARCHIVE_SHA,
        "n_pairs_target": n,
        "n_pairs_solved": len(have),
        "complete": len(have) == n,
        "S0_stub": {"d_pose_mean": stub_mean, "pose_term": term(stub_mean)},
        "S1_warp_base": {
            "d_pose_mean": float(warp.mean()),
            "d_pose_median": float(np.median(warp)),
            "pose_term": term(float(warp.mean())),
        },
        "S2_terminal_solve": {
            "d_pose_mean": float(solved.mean()),
            "d_pose_median": float(np.median(solved)),
            "d_pose_max": float(solved.max()),
            "pose_term": term(float(solved.mean())),
            "wins_vs_warp": int((solved < warp).sum()),
        },
        "total_forwards": int(nfwd.sum()),
    }
    out = OUT / "ps1_ladder_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("stub")
    ps.add_argument("--n-pairs", type=int, default=600)
    ps.add_argument("--chunk", type=int, default=120)
    ps.add_argument("--pose-batch", type=int, default=32)
    ps.add_argument("--threads", type=int, default=4)
    ps.add_argument("--s-r", type=float, default=1.0)
    ps.set_defaults(func=run_stub)

    pv = sub.add_parser("solve")
    pv.add_argument("--n-pairs", type=int, default=600)
    pv.add_argument("--start", type=int, default=0)
    pv.add_argument("--end", type=int, default=600)
    pv.add_argument("--relins", type=int, default=3)
    pv.add_argument("--threads", type=int, default=4)
    pv.add_argument("--s-r", type=float, default=1.0)
    pv.add_argument("--resume", action="store_true")
    pv.add_argument("--max-seconds", type=float, default=0.0)
    pv.set_defaults(func=run_solve)

    pa = sub.add_parser("agg")
    pa.add_argument("--n-pairs", type=int, default=600)
    pa.set_defaults(func=run_agg)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
