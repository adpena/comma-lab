#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P2c — ru1-atlas-AIMED single-quantum token-cell edits on the frozen endpoint.

Consumes ru1's typed flip atlas (per-flip rows: pair,y,x,classes,m_def,margins;
verified bit-identical to the pb1 P1 verdict) to aim edits at the measured
hotspot (pair,cell) instances — the concentration law says the effective solve
surface is ~100 cells, not 600 pairs.

Per instance (top-K by flip count): try the 8 single-quantum edits
(4 channels x +/-1, lattice-legal) of that cell's token in that pair; accept
the best strictly-flip-reducing edit on the FULL-PAIR realized d_seg (never
cell-local — ERF collateral crosses cell boundaries; ru1 measured blind edits
median -1 flip / 65% net-negative vs aimed best-of-8 positive in 17/18).
Then compose ALL accepted edits, compile ONE composed archive, and gate the
composition on the full-population joint action including real Brotli bytes
(the P4b knee lesson: byte effects do not compose additively).

Aim staleness note: the atlas maps the P1 BASE flips; the endpoint differs by
<=38 token coords (+ renderer mods if P2b won). Aiming is advisory; every
acceptance is measured on the CURRENT endpoint bytes.

[macOS-CPU advisory]; score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pb1_p2c_aimed_cell_edits_receipt.v1"
LEVELS = 16
SHAPE = (600, 24, 32, 4)
CELL = 16
BYTE_PRICE = 25.0 / 37_545_489.0


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path,
                    help="frozen current-best endpoint archive")
    ap.add_argument("--atlas", required=True, type=Path)
    ap.add_argument("--endpoint-chunks", required=True, type=Path,
                    help="p1_chunks dir of the ENDPOINT per-pair verdict")
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--pose-targets", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=24)
    ap.add_argument("--frame0-policy", choices=("zeros", "copy"),
                    default="zeros",
                    help="frame0 base used for the pose leg of the verdicts")
    return ap.parse_args()


def _load_chunks(chunks_dir: Path):
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    idxs, dsegs, dposes = [], [], []
    for p in paths:
        z = np.load(p)
        idxs.append(z["idxs"])
        dsegs.append(z["dsegs"])
        dposes.append(z["dposes"])
    idx = np.concatenate(idxs)
    order = np.argsort(idx)
    if not np.array_equal(idx[order], np.arange(SHAPE[0])):
        raise SystemExit("endpoint chunk cache incomplete")
    ds = np.concatenate(dsegs)[order]
    dp = np.concatenate(dposes)[order]
    if not np.isfinite(dp).all():
        raise SystemExit("endpoint cache lacks pose rows")
    return ds, dp


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path
    from train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_batch,
    )

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization import ddm_tr1_runtime as rt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_archive = args.archive.read_bytes()
    parsed = rt.parse_archive(base_archive)
    base_codes = np.asarray(parsed.packet.token_codes, dtype=np.int64)
    base_dsegs, base_dposes = _load_chunks(args.endpoint_chunks)

    atlas = np.load(args.atlas)
    pair = atlas["pair"].astype(np.int64)
    cy = (atlas["y"].astype(np.int64)) // CELL
    cx = (atlas["x"].astype(np.int64)) // CELL
    key = (pair * 24 + cy) * 32 + cx
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    instances = [(int(uniq[o] // (24 * 32)),
                  int((uniq[o] // 32) % 24),
                  int(uniq[o] % 32),
                  int(counts[o])) for o in order[:args.top_k]]
    print(f"[aim] top-{args.top_k} instances; flip counts "
          f"{[c for *_, c in instances][:8]}...", flush=True)

    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    rows = json.loads(args.pose_targets.read_text())["rows"]
    targets = [np.asarray(r["center"], dtype=np.float64) for r in rows]
    seg_cpu = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p_ in posenet_cpu.parameters():
        p_.requires_grad = False

    def render_pair_dseg_dpose(packet, p: int) -> tuple[float, float]:
        f1 = rt.render_frame1_camera_uint8(packet, p)
        gt = np.asarray(lstars[p], dtype=np.int64)
        d_seg = cpu_verdict_d_seg_batch(seg_cpu, [f1], [gt])[0]
        f0 = np.zeros_like(f1) if args.frame0_policy == "zeros" else f1.copy()
        d_pose = cpu_verdict_d_pose_batch(
            posenet_cpu, [f0], [f1], [targets[p]])[0]
        return float(d_seg), float(d_pose)

    def packet_for(codes4: np.ndarray):
        token_payload = rt._encode_tokens(codes4.astype(np.uint8))
        payloads = {
            "tokens": token_payload,
            "lotto_renderer": parsed.packet.section_payloads[1],
            "selector": parsed.packet.section_payloads[2],
            "pose_stub": parsed.packet.section_payloads[3],
        }
        packet = rt.build_packet(parsed.packet.metadata, payloads)
        manifest = rt._archive_manifest(packet)
        archive = rt._deterministic_stored_zip({
            "manifest.json": rt._canonical_json(dict(manifest)),
            "state/tr1.ddt1": packet,
        })
        return rt.parse_archive(archive), archive

    t0 = time.time()
    accepted = []
    trials = []
    work_codes = base_codes.copy()
    for (p, gy, gx, nflips) in instances:
        cand_rows = []
        for ch in range(4):
            code = int(work_codes[p, gy, gx, ch])
            for sign in (-1, 1):
                nc = code + sign
                if nc < 0 or nc >= LEVELS:
                    continue
                trial_codes = work_codes.copy()
                trial_codes[p, gy, gx, ch] = nc
                cpk, _ = packet_for(trial_codes)
                d_seg, d_pose = render_pair_dseg_dpose(cpk.packet, p)
                cand_rows.append((d_seg, d_pose, ch, sign))
        if not cand_rows:
            continue
        cand_rows.sort()
        best = cand_rows[0]
        d_seg_best, d_pose_best, ch, sign = best
        delta_flips = round((d_seg_best - base_dsegs[p]) * 196608)
        row = {
            "pair": p, "cell": [gy, gx], "atlas_flips": nflips,
            "pair_dseg_before": float(base_dsegs[p]),
            "pair_dseg_after": float(d_seg_best),
            "delta_flips": int(delta_flips),
            "channel": ch, "sign": sign,
            "accepted": bool(d_seg_best < base_dsegs[p]),
        }
        trials.append(row)
        if d_seg_best < base_dsegs[p]:
            work_codes[p, gy, gx, ch] += sign
            base_dsegs[p] = d_seg_best
            base_dposes[p] = d_pose_best
            accepted.append(row)
        print(f"[cell p{p} ({gy},{gx}) flips {nflips}] "
              f"dflips {delta_flips:+d} accepted={row['accepted']}",
              flush=True)

    # composed gate: ONE archive, full joint action vs the endpoint
    cparsed, carchive = packet_for(work_codes)
    ds0, dp0 = _load_chunks(args.endpoint_chunks)
    joint0 = (100.0 * ds0.mean() + float(np.sqrt(10.0 * dp0.mean()))
              + BYTE_PRICE * len(base_archive))
    joint1 = (100.0 * base_dsegs.mean()
              + float(np.sqrt(10.0 * base_dposes.mean()))
              + BYTE_PRICE * len(carchive))
    improved = joint1 < joint0
    if improved and accepted:
        (args.out_dir / "p2c_aimed_archive.zip").write_bytes(carchive)

    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "aim_source": ("ru1 atlas_flat.npz (verified bit-identical to the"
                       " pb1 P1 verdict - positive control); aim advisory,"
                       " acceptance measured on current endpoint bytes"),
        "endpoint_archive_sha256": _sha(base_archive),
        "top_k": args.top_k,
        "trials": trials,
        "n_accepted": len(accepted),
        "accepted_delta_flips_total": int(sum(r["delta_flips"]
                                              for r in accepted)),
        "composed": {
            "archive_bytes": len(carchive),
            "archive_sha256": _sha(carchive),
            "d_seg": float(base_dsegs.mean()),
            "d_pose": float(base_dposes.mean()),
            "joint_action": joint1,
        },
        "endpoint_joint_action": joint0,
        "composed_delta_vs_endpoint": joint1 - joint0,
        "composed_improves_joint": bool(improved),
        "wall_seconds": time.time() - t0,
        "generated_by": "tools/pb1_aimed_cell_edits.py",
    }
    out = args.out_dir / "p2c_aimed_receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[done] accepted {len(accepted)}/{len(trials)} "
          f"dflips {receipt['accepted_delta_flips_total']:+d} "
          f"joint {joint1:.6f} (delta {joint1-joint0:+.3e}) receipt {out}",
          flush=True)


if __name__ == "__main__":
    main()
