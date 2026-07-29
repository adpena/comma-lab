#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P4b — compose the strictly-improving QDBS edits and verdict ONCE.

Takes the P2a QDBS receipt, selects every strictly-improving candidate in
rank order (best delta first), applies its deltas to the base token lattice
unless any of its coordinates was already edited (first-wins; avoids sign
conflicts), compiles ONE composed archive via the committed runtime framing,
and measures the realized full-population joint action with a fresh
non-incremental verdict on the touched pairs + P1 cache for untouched pairs
(exact: d_seg/d_pose means are per-pair averages; tokens are per-pair).

This is the first knee row of the correction-granularity ladder race on this
vehicle: measured composed gain vs the sum of single-edit gains (Brotli
context interactions make additivity an empirical question, never an
assumption).  [macOS-CPU advisory]; score_claim=false.
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
SCHEMA = "ddm_pb1_p4b_composed_qdbs_edits_receipt.v1"
LEVELS = 16
SHAPE = (600, 24, 32, 4)
BYTE_PRICE = 25.0 / 37_545_489.0


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdbs-receipt", required=True, type=Path)
    ap.add_argument("--archive", required=True, type=Path,
                    help="the QDBS BASE archive the receipt deltas refer to")
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--pose-targets", required=True, type=Path)
    ap.add_argument("--p1-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seg-batch", type=int, default=12)
    ap.add_argument("--pose-batch", type=int, default=6)
    return ap.parse_args()


def _load_p1_arrays(p1_dir: Path):
    paths = sorted((p1_dir / "p1_chunks").glob("chunk_*.npz"))
    idxs, dsegs, dposes = [], [], []
    for p in paths:
        z = np.load(p)
        idxs.append(z["idxs"])
        dsegs.append(z["dsegs"])
        dposes.append(z["dposes"])
    idx = np.concatenate(idxs)
    order = np.argsort(idx)
    if not np.array_equal(idx[order], np.arange(SHAPE[0])):
        raise SystemExit("P1 cache incomplete")
    return (np.concatenate(dsegs)[order], np.concatenate(dposes)[order])


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
    receipt_in = json.loads(args.qdbs_receipt.read_text())
    result = receipt_in["result"]
    sched = result["schedule"]
    schedule = {c["identity"]: c
                for c in (list(sched["scorer_proposals"])
                          + list(sched["random_controls"]))}
    improving = sorted(
        (t for t in result["traces"] if t["strict_realized_improvement"]),
        key=lambda t: t["delta_vs_base"])

    base_archive = args.archive.read_bytes()
    if _sha(base_archive) != receipt_in["base_archive_sha256"]:
        raise SystemExit("base archive does not match the QDBS receipt")
    parsed = rt.parse_archive(base_archive)
    base_codes = np.asarray(parsed.packet.token_codes, dtype=np.int64)
    codes = base_codes.copy().reshape(-1)

    applied, skipped = [], []
    edited: set[int] = set()
    sum_single_deltas = 0.0
    for t in improving:
        cand = schedule[t["identity"]]
        idxs = [d["index"] for d in cand["deltas"]]
        if any(i in edited for i in idxs):
            skipped.append(t["identity"])
            continue
        for d in cand["deltas"]:
            codes[d["index"]] += d["delta"]
        edited.update(idxs)
        applied.append(t["identity"])
        sum_single_deltas += t["delta_vs_base"]
    if codes.min() < 0 or codes.max() >= LEVELS:
        raise SystemExit("composed codes escaped the lattice")

    codes4 = codes.reshape(SHAPE)
    token_payload = rt._encode_tokens(codes4.astype(np.uint8))
    payloads = {
        "tokens": token_payload,
        "lotto_renderer": parsed.packet.section_payloads[1],
        "selector": parsed.packet.section_payloads[2],
        "pose_stub": parsed.packet.section_payloads[3],
    }
    packet = rt.build_packet(parsed.packet.metadata, payloads)
    manifest = rt._archive_manifest(packet)
    composed = rt._deterministic_stored_zip({
        "manifest.json": rt._canonical_json(dict(manifest)),
        "state/tr1.ddt1": packet,
    })
    cparsed = rt.parse_archive(composed)

    touched = [int(p) for p in np.nonzero(
        (codes4 != base_codes).any(axis=(1, 2, 3)))[0]]
    print(f"[compose] applied {len(applied)} candidates "
          f"({len(edited)} coords, {len(touched)} pairs), "
          f"skipped {len(skipped)} on coord overlap", flush=True)

    base_dsegs, base_dposes = _load_p1_arrays(args.p1_dir)
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    rows = json.loads(args.pose_targets.read_text())["rows"]
    targets = [np.asarray(r["center"], dtype=np.float64) for r in rows]
    seg_cpu = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False

    t0 = time.time()
    frames = [rt.render_frame1_camera_uint8(cparsed.packet, p)
              for p in touched]
    dsegs = base_dsegs.copy()
    dposes = base_dposes.copy()
    for b0 in range(0, len(touched), args.seg_batch):
        b1 = min(b0 + args.seg_batch, len(touched))
        gts = [np.asarray(lstars[p], dtype=np.int64)
               for p in touched[b0:b1]]
        ds = cpu_verdict_d_seg_batch(seg_cpu, frames[b0:b1], gts)
        for j, p in enumerate(touched[b0:b1]):
            dsegs[p] = ds[j]
    zeros = np.zeros_like(frames[0])
    for b0 in range(0, len(touched), args.pose_batch):
        b1 = min(b0 + args.pose_batch, len(touched))
        dp = cpu_verdict_d_pose_batch(
            posenet_cpu, [zeros] * (b1 - b0), frames[b0:b1],
            [targets[p] for p in touched[b0:b1]])
        for j, p in enumerate(touched[b0:b1]):
            dposes[p] = dp[j]
    wall = time.time() - t0

    d_seg = float(dsegs.mean())
    d_pose = float(dposes.mean())
    action = (100.0 * d_seg + float(np.sqrt(10.0 * d_pose))
              + BYTE_PRICE * len(composed))
    base_action = (100.0 * float(base_dsegs.mean())
                   + float(np.sqrt(10.0 * float(base_dposes.mean())))
                   + BYTE_PRICE * len(base_archive))
    delta = action - base_action

    out_zip = args.out_dir / "p4b_composed_archive.zip"
    out_zip.write_bytes(composed)
    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "base_archive_sha256": _sha(base_archive),
        "composed_archive_sha256": _sha(composed),
        "composed_archive_bytes": len(composed),
        "base_archive_bytes": len(base_archive),
        "applied_candidates": applied,
        "skipped_on_overlap": skipped,
        "edited_coords": len(edited),
        "touched_pairs": touched,
        "composed": {"d_seg": d_seg, "d_pose": d_pose,
                     "joint_action": action},
        "base_joint_action": base_action,
        "composed_delta_vs_base": delta,
        "sum_of_single_deltas": sum_single_deltas,
        "additivity_ratio": (delta / sum_single_deltas
                             if sum_single_deltas != 0 else None),
        "verdict_wall_seconds": wall,
        "note": ("first knee row of the ladder race: in-place token edits"
                 " (rungs 2/3 collapse to rung-5 on this vehicle - every"
                 " candidate carrier is already a counted stream)"),
        "generated_by": "tools/pb1_compose_qdbs_edits.py",
    }
    out = args.out_dir / "p4b_composed_receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[done] composed action {action:.6f} (delta {delta:+.3e}; "
          f"sum singles {sum_single_deltas:+.3e}) receipt {out}", flush=True)


if __name__ == "__main__":
    main()
