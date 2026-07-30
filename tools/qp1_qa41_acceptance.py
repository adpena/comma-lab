#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_qp1 QA41 — acceptance-ledger RE-PRICE of the pb1 P2a/P4b candidates at the
warp-base operating row (pfs1 D1).

WHY (gc8 memo §1-B row B; ledger QA41): the pb1 P2a/P4b accept/reject rows were
measured on the pose-EXPLODE base (d_pose ~78.2, sqrt-term sensitivity
5/sqrt(10*78.2)=0.1788/unit). The current operating row is the warp base
(pfs1 D1, d_pose 0.22144216, sensitivity 5/sqrt(10*0.22144216)=3.360/unit) —
~19x different acceptance geometry. AND at the warp base frame_0 = warp(f1), so a
token edit that changes f1 changes BOTH f1 and f0=warp(f1): the raw pose response
of a token edit at the warp base is a DIFFERENT object than at the zeros-frame0
base and CANNOT transfer analytically. It is re-MEASURED realized here.

WHAT transfers vs re-measured, per candidate:
  * d_seg   : re-MEASURED realized at the pfs1 D1 seg base (p2c_aimed, b9a7983b),
              through render+SegNet argmax (frame_1-only law).
  * d_pose  : re-MEASURED realized at the warp base — apply the token delta,
              re-render the affected pair's f1, re-warp f0 = warp(f1_edited; shipped
              t_p, s_t, s_r=0), re-run frozen CPU-torch PoseNet6 vs the banked
              target. Only affected pairs move (token edits are local) => bounded.
  * rate    : ADVISORY only. The old tr1-reencode byte delta is carried as an
              advisory rate term; the TRUE shipping price is the r7 SMEVR coder
              (xi1 handoff). The pfs1 D1 tokens are SMEVR-coded (557KB) not
              tr1-reencoded, so the old byte deltas do NOT transfer. Acceptance is
              reported BOTH with the advisory rate AND on seg+pose only.

The knee additivity LAW (0.088; P4b) is mechanism-backed (Brotli-context
non-additivity) and is NOT re-derived here — only the per-candidate accept/reject
FLIPS are re-measured. Lever-D (1.22 B/flip vs 1.2731 water) is pose-base-
INDEPENDENT seg/rate arithmetic and is treated analytically (no realized run).

Axis: [macOS-CPU advisory]; score_claim=false; promotion_eligible=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SEG_PX = 196608          # 512*384 SegNet internal resolution
NPAIRS = 600
GRID = (600, 24, 32, 4)
LEVELS = 16
N_UNCOMP = 37_545_489.0
BYTE_PRICE = 25.0 / N_UNCOMP
# pfs1 D1 MEASURED operating row (evaluate.py rc=0 full n600, archive 624ffe57):
D1_DPOSE = 0.22144216     # Average PoseNet Distortion (evaluator)
D1_DSEG = 0.00389011      # Average SegNet Distortion (evaluator)
OLD_BASE_BYTES = 768689   # p2a base_action archive_bytes (tr1 reencode)
OLD_BASE_DPOSE = 78.19802405770572
OLD_BASE_DSEG = 0.0038887702094184034
KNEE_ADDITIVITY = 0.08836456763313423  # P4b measured LAW (do NOT re-derive)

P2A_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2a_qdbs_receipt.json")
P4B_RECEIPT = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p4b_composed_receipt.json")
SEG_BASE = Path("/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip")
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def pose_term(d_pose_mean: float) -> float:
    return math.sqrt(10.0 * float(d_pose_mean))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_candidates() -> list[dict]:
    """48 candidates: identity, deltas[(index,delta)], old (d_seg,d_pose,bytes,
    delta_vs_base, strict_accept, proposal_class)."""
    d = json.loads(P2A_RECEIPT.read_text())
    sch = d["result"]["schedule"]
    id2deltas: dict[str, list] = {}
    for src in ("scorer_proposals", "random_controls"):
        for prop in sch[src]:
            id2deltas[prop["identity"]] = [(int(x["index"]), int(x["delta"]))
                                           for x in prop["deltas"]]
    p4 = json.loads(P4B_RECEIPT.read_text())
    p4b_set = set(p4["applied_candidates"])
    cands = []
    for t in d["result"]["traces"]:
        idy = t["identity"]
        a = t["action"]
        cands.append({
            "identity": idy,
            "deltas": id2deltas[idy],
            "proposal_class": t["proposal_class"],
            "old_d_seg": float(a["d_seg"]),
            "old_d_pose": float(a["d_pose"]),
            "old_bytes": int(a["archive_bytes"]),
            "old_delta_vs_base": float(t["delta_vs_base"]),
            "old_accept": bool(t["strict_realized_improvement"]),
            "in_p4b": idy in p4b_set,
        })
    return cands


class QA41Runtime:
    """One writable p2c_aimed token grid feeding BOTH the SegNet argmax path and
    the warp+PoseNet pose path (s_r=0 == the D1 shipped receiver)."""

    def __init__(self) -> None:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        sys.path.insert(0, str(REPO / "experiments"))
        sys.path.insert(0, str(REPO / "upstream"))
        import torch
        torch.set_num_threads(1)  # good citizen: ck1 also runs PoseNet
        import ddm_pfs1_ep_warp_pose_solve as ep
        from train_witness_realized_through_R_mlx import cpu_verdict_d_seg_batch

        from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
        from tac.boundary_math.seg_core import load_real_segnet

        self.ep = ep
        self.orc = ep.WarpPoseOracle(s_r=0.0)   # D1 shipped receiver == s_r=0
        # rebind a writable token grid so edits are seen by render+warp+seg.
        codes = np.asarray(self.orc.packet.token_codes)
        self.codes = codes.astype(codes.dtype, copy=True)
        self.codes.setflags(write=True)
        object.__setattr__(self.orc.packet, "token_codes", self.codes)
        self.st_idx = ep.load_solved_st(NPAIRS)
        self.tp = np.asarray([self.orc.targets64[i] for i in range(NPAIRS)])
        # seg path
        self._verdict = cpu_verdict_d_seg_batch
        self.seg = load_real_segnet("cpu")
        self.lstars = open_stored_npy_memmap(GT_CACHE, "lstars")
        self._base_flips: dict[int, int] = {}
        self._base_dpose: dict[int, float] = {}

    def _render(self, p: int) -> np.ndarray:
        return self.orc.rt.render_frame1_camera_uint8(self.orc.packet, p)

    def _flips(self, p: int, f1) -> int:
        gt = np.asarray(self.lstars[p], dtype=np.int64)
        return round(float(self._verdict(self.seg, [f1], [gt])[0]) * SEG_PX)

    def _dpose(self, p: int, f1) -> float:
        st = self.ep.ST_GRID[int(self.st_idx[p])]
        return float(self.orc.d_pose_shipped(p, f1, self.tp[p], st))

    def base(self, p: int) -> tuple[int, float]:
        if p not in self._base_flips:
            f1 = self._render(p)
            self._base_flips[p] = self._flips(p, f1)
            self._base_dpose[p] = self._dpose(p, f1)
        return self._base_flips[p], self._base_dpose[p]

    def measure_candidate(self, deltas: list) -> dict:
        """Realized (Δflips, Δd_pose_sum) over affected pairs, keeping shipped t_p."""
        pairs = sorted({idx // 3072 for idx, _ in deltas})
        for p in pairs:
            self.base(p)  # warm cache at un-edited grid
        # apply
        saved = []
        for idx, dl in deltas:
            pp, gy, gx, ch = np.unravel_index(int(idx), GRID)
            cur = int(self.codes[pp, gy, gx, ch])
            nc = min(LEVELS - 1, max(0, cur + int(dl)))
            saved.append((pp, gy, gx, ch, cur))
            self.codes[pp, gy, gx, ch] = nc
        d_flips = 0
        d_dpose_sum = 0.0
        per_pair = {}
        for p in pairs:
            f1 = self._render(p)
            ef = self._flips(p, f1)
            ed = self._dpose(p, f1)
            d_flips += ef - self._base_flips[p]
            d_dpose_sum += ed - self._base_dpose[p]
            per_pair[p] = {"base_flips": self._base_flips[p], "ed_flips": ef,
                           "base_dpose": self._base_dpose[p], "ed_dpose": ed}
        # restore
        for pp, gy, gx, ch, cur in saved:
            self.codes[pp, gy, gx, ch] = cur
        return {"affected_pairs": pairs, "d_flips": d_flips,
                "d_dpose_sum": d_dpose_sum, "per_pair": per_pair}


def reprice(row_old: dict, meas: dict) -> dict:
    d_seg_mean = meas["d_flips"] / (NPAIRS * SEG_PX)
    dS_seg = 100.0 * d_seg_mean
    d_pose_mean_delta = meas["d_dpose_sum"] / NPAIRS
    new_dpose = D1_DPOSE + d_pose_mean_delta
    dS_pose = pose_term(new_dpose) - pose_term(D1_DPOSE)
    d_bytes_adv = row_old["old_bytes"] - OLD_BASE_BYTES
    dS_rate_adv = BYTE_PRICE * d_bytes_adv
    dS_full = dS_seg + dS_pose + dS_rate_adv
    dS_segpose = dS_seg + dS_pose
    return {
        "identity": row_old["identity"],
        "proposal_class": row_old["proposal_class"],
        "in_p4b": row_old["in_p4b"],
        "affected_pairs": meas["affected_pairs"],
        "d_flips": meas["d_flips"], "d_seg_mean": d_seg_mean, "dS_seg": dS_seg,
        "d_pose_mean_delta": d_pose_mean_delta, "new_dpose_mean": new_dpose,
        "dS_pose_new": dS_pose,
        "d_bytes_adv": d_bytes_adv, "dS_rate_adv": dS_rate_adv,
        "dS_new_full": dS_full, "dS_new_segpose": dS_segpose,
        "old_delta_vs_base": row_old["old_delta_vs_base"],
        "old_accept": row_old["old_accept"],
        "new_accept_full": dS_full < 0.0,
        "new_accept_segpose": dS_segpose < 0.0,
        "flip_full": (dS_full < 0.0) != row_old["old_accept"],
        "per_pair": meas["per_pair"],
    }


def run(args) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "qa41_rows.jsonl"
    cands = load_candidates()
    if getattr(args, "limit", 0):
        cands = cands[: args.limit]
    done = {}
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["identity"]] = r
        print(f"[resume] {len(done)} rows", flush=True)
    rt = QA41Runtime() if len(done) < len(cands) else None
    t0 = time.time()
    for c in cands:
        if c["identity"] in done:
            continue
        meas = rt.measure_candidate(c["deltas"])
        row = reprice(c, meas)
        with jsonl.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        done[row["identity"]] = row
        print(f"[{row['identity']:34s}] old_acc={int(row['old_accept'])} "
              f"new_acc={int(row['new_accept_full'])} flip={int(row['flip_full'])} "
              f"dS_new={row['dS_new_full']:+.3e} (seg {row['dS_seg']:+.2e} "
              f"pose {row['dS_pose_new']:+.2e} rate_adv {row['dS_rate_adv']:+.2e}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    rows = [done[c["identity"]] for c in cands]
    # ---- corrected acceptance ledger summary ----
    old_acc = [r for r in rows if r["old_accept"]]
    new_acc_full = [r for r in rows if r["new_accept_full"]]
    new_acc_sp = [r for r in rows if r["new_accept_segpose"]]
    rej_to_acc = [r["identity"] for r in rows if r["new_accept_full"] and not r["old_accept"]]
    acc_to_rej = [r["identity"] for r in rows if r["old_accept"] and not r["new_accept_full"]]
    # recoverable dS at the new base (naive additive over disjoint pairs for seg+pose;
    # rate part knee-scaled by the measured additivity LAW 0.088).
    sum_segpose = sum(r["dS_new_segpose"] for r in new_acc_full if r["dS_new_segpose"] < 0)
    sum_rate = sum(r["dS_rate_adv"] for r in new_acc_full)
    composed_recoverable = sum_segpose + KNEE_ADDITIVITY * sum_rate
    receipt = {
        "schema": "ddm_qp1_qa41_acceptance_reprice.v1",
        "item": "QA41 acceptance-ledger re-price at the warp-base operating row (pfs1 D1)",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "operating_row": {"d_pose": D1_DPOSE, "d_seg": D1_DSEG,
                          "sensitivity_new": 5.0 / math.sqrt(10.0 * D1_DPOSE),
                          "sensitivity_old": 5.0 / math.sqrt(10.0 * OLD_BASE_DPOSE)},
        "n_candidates": len(rows),
        "n_old_accept": len(old_acc),
        "n_new_accept_full": len(new_acc_full),
        "n_new_accept_segpose": len(new_acc_sp),
        "reject_to_accept_full": rej_to_acc,
        "accept_to_reject_full": acc_to_rej,
        "n_reject_to_accept": len(rej_to_acc),
        "n_accept_to_reject": len(acc_to_rej),
        "composed_recoverable_dS_full": composed_recoverable,
        "composed_recoverable_note": ("seg+pose additive over disjoint pairs; rate part "
                                      "knee-scaled by measured additivity 0.088; rate ADVISORY "
                                      "(tr1-reencode; true=SMEVR r7 xi1 handoff)"),
        "knee_additivity_law": KNEE_ADDITIVITY,
        "lever_d": ("pose-base-INDEPENDENT: 1.22 B/flip vs 1.2731 water is pure seg/rate "
                    "arithmetic (no pose term); verdict TRANSFERS unchanged to the warp base"),
        "verdict_scope": "INSTANCE (these 48 candidates, this warp-base endpoint, this scorer)",
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/qp1_qa41_acceptance.py",
    }
    _atomic_write(out / "qa41_receipt.json", json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[QA41 done] old_acc={len(old_acc)} new_acc_full={len(new_acc_full)} "
          f"rej->acc={len(rej_to_acc)} acc->rej={len(acc_to_rej)} "
          f"recoverable_dS={composed_recoverable:+.4e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N candidates")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
