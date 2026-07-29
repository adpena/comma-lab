#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P2b — #400 ``mc_finisher`` DIAGONAL mode on the tr1 renderer stream.

Realizes the queued rv1-R1 measurement (verbatim: "#400 mc_finisher diagonal
adapted from witness head/palette tensors to the tr1 token/renderer tensors
(3,284 B lotto stream = ideal (1+1)-ES target)").  The #400 node in the engine
composition table is typed ``execution_enabled=false`` with no landed
executable — this driver is the bounded tr1 instantiation:

* target = the counted renderer modulation/bias values (fp16 ``g_``/``b_``
  per layer; the binary supermask is NOT touched),
* mechanism = (1+1)-ES with a DIAGONAL step-size vector (per-coordinate sigma
  proportional to |v| at fp16-ULP scale) and the 1/5th success rule,
* acceptance = strict decrease of the realized full-population joint action
  ``100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`` measured through the
  committed receiver + frozen CPU-torch SegNet/PoseNet on ALL 600 pairs
  (renderer edits touch every pair; no incremental shortcut exists).

Rung-5 semantics: refined values of EXISTING counted params; zero new stream
bytes (the fp16 re-encode may change Brotli size — that lands in the action).

Quote scope: the witness-vehicle 0.05-0.07 S scale for the #400 family is a
FOREIGN-PARENT prior, not a prediction here.  Whatever this measures IS the
same-parent quote E2 asked for.  [macOS-CPU advisory]; score_claim=false.
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
SCHEMA = "ddm_pb1_p2b_mc400_diagonal_receipt.v1"
BYTE_PRICE = 25.0 / 37_545_489.0


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path,
                    help="base archive (current best endpoint)")
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--pose-targets", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--budget", type=int, default=8,
                    help="candidate evaluations (each is a full n600 verdict)")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--sigma-rel", type=float, default=1.0 / 512.0,
                    help="initial per-coord sigma as fraction of |v| (~4 fp16 ULP)")
    ap.add_argument("--seg-batch", type=int, default=12)
    ap.add_argument("--pose-batch", type=int, default=6)
    return ap.parse_args()


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
    packet = parsed.packet
    masks = [np.asarray(m, dtype=np.uint8) for m in packet.masks]
    gains = [np.asarray(g, dtype=np.float32) for g in packet.gains]
    biases = [np.asarray(b, dtype=np.float32) for b in packet.biases]
    sizes = [(g.size, b.size) for g, b in zip(gains, biases, strict=True)]

    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    rows = json.loads(args.pose_targets.read_text())["rows"]
    targets = [np.asarray(r["center"], dtype=np.float64) for r in rows]
    seg_cpu = load_real_segnet("cpu")
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False

    n_pairs = int(packet.selector["num_pairs"])

    def rebuild(gs, bs) -> bytes:
        renderer_payload = rt._encode_renderer(masks, gs, bs)
        payloads = {
            "tokens": packet.section_payloads[0],
            "lotto_renderer": renderer_payload,
            "selector": packet.section_payloads[2],
            "pose_stub": packet.section_payloads[3],
        }
        new_packet = rt.build_packet(packet.metadata, payloads)
        manifest = rt._archive_manifest(new_packet)
        return rt._deterministic_stored_zip({
            "manifest.json": rt._canonical_json(dict(manifest)),
            "state/tr1.ddt1": new_packet,
        })

    def full_verdict(archive_bytes: bytes) -> tuple[float, float, float]:
        cand = rt.parse_archive(archive_bytes)
        dsegs = np.zeros(n_pairs)
        dposes = np.zeros(n_pairs)
        frames = [rt.render_frame1_camera_uint8(cand.packet, i)
                  for i in range(n_pairs)]
        for b0 in range(0, n_pairs, args.seg_batch):
            b1 = min(b0 + args.seg_batch, n_pairs)
            gts = [np.asarray(lstars[i], dtype=np.int64)
                   for i in range(b0, b1)]
            dsegs[b0:b1] = cpu_verdict_d_seg_batch(
                seg_cpu, frames[b0:b1], gts)
        zeros = np.zeros_like(frames[0])
        for b0 in range(0, n_pairs, args.pose_batch):
            b1 = min(b0 + args.pose_batch, n_pairs)
            dposes[b0:b1] = cpu_verdict_d_pose_batch(
                posenet_cpu, [zeros] * (b1 - b0), frames[b0:b1],
                targets[b0:b1])
        d_seg = float(dsegs.mean())
        d_pose = float(dposes.mean())
        action = (100.0 * d_seg + float(np.sqrt(10.0 * d_pose))
                  + BYTE_PRICE * len(archive_bytes))
        return d_seg, d_pose, action

    # verify rebuild determinism on the base
    rebuilt = rebuild(gains, biases)
    if rebuilt != base_archive:
        raise SystemExit("base renderer rebuild is NOT byte-identical")
    print("[determinism] base renderer rebuild byte-identical", flush=True)

    t0 = time.time()
    base_dseg, base_dpose, base_action = full_verdict(base_archive)
    print(f"[base] d_seg {base_dseg:.7f} d_pose {base_dpose:.5f} "
          f"action {base_action:.6f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    v = np.concatenate([a.reshape(-1) for pairi in
                        zip(gains, biases, strict=True) for a in pairi])
    sigma_scale = 1.0
    best_v = v.copy()
    best = (base_dseg, base_dpose, base_action)
    best_archive = base_archive
    trials = []

    def unpack(vec):
        gs, bs, off = [], [], 0
        for (gsz, bsz), g0, b0 in zip(sizes, gains, biases, strict=True):
            gs.append(vec[off:off + gsz].reshape(g0.shape).astype(np.float32))
            off += gsz
            bs.append(vec[off:off + bsz].reshape(b0.shape).astype(np.float32))
            off += bsz
        return gs, bs

    for k in range(args.budget):
        for _attempt in range(6):
            step = rng.standard_normal(best_v.size)
            sig = np.maximum(np.abs(best_v) * args.sigma_rel, 1e-4)
            cand_v = best_v + sigma_scale * sig * step
            gs, bs = unpack(cand_v)
            cand_archive = rebuild(gs, bs)
            if cand_archive != best_archive:
                break
            sigma_scale *= 2.0
        else:
            print(f"[trial {k}] mutation vanished at fp16; stopping", flush=True)
            break
        t1 = time.time()
        d_seg, d_pose, action = full_verdict(cand_archive)
        wall = time.time() - t1
        accepted = action < best[2]
        trials.append({
            "trial": k,
            "sigma_scale": sigma_scale,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": len(cand_archive),
            "joint_action": action,
            "delta_vs_best": action - best[2],
            "accepted": accepted,
            "wall_seconds": wall,
        })
        print(f"[trial {k}] action {action:.6f} (delta {action-best[2]:+.2e})"
              f" accepted={accepted} sigma={sigma_scale:.3f} {wall:.0f}s",
              flush=True)
        if accepted:
            best = (d_seg, d_pose, action)
            best_v = cand_v
            best_archive = cand_archive
            sigma_scale *= 1.5
        else:
            sigma_scale *= 0.6

    improved = best[2] < base_action
    if improved:
        (args.out_dir / "p2b_best_archive.zip").write_bytes(best_archive)
    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "design_source": ("rv1-R1 verbatim: #400 mc_finisher diagonal on the"
                          " tr1 renderer stream; typed node was"
                          " execution_enabled=false/unbuilt — this is the"
                          " bounded tr1 instantiation"),
        "base_archive_sha256": _sha(base_archive),
        "base": {"d_seg": base_dseg, "d_pose": base_dpose,
                 "archive_bytes": len(base_archive),
                 "joint_action": base_action},
        "budget": args.budget,
        "seed": args.seed,
        "sigma_rel_init": args.sigma_rel,
        "trials": trials,
        "strict_improvement_found": improved,
        "best": {"d_seg": best[0], "d_pose": best[1],
                 "archive_bytes": len(best_archive),
                 "joint_action": best[2],
                 "archive_sha256": _sha(best_archive),
                 "delta_vs_base": best[2] - base_action},
        "quote_for_e2": {
            "kind": "SEG_GN_FAMILY_DIAGONAL_ES",
            "gain": base_action - best[2],
            "wall_seconds": time.time() - t0,
            "verdict_scope": "INSTANCE (this budget, this sigma schedule)",
        },
        "generated_by": "tools/pb1_mc400_diagonal_renderer.py",
    }
    out = args.out_dir / "p2b_mc400_diagonal_receipt.json"
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"[done] improved={improved} best action {best[2]:.6f} "
          f"receipt {out}", flush=True)


if __name__ == "__main__":
    main()
