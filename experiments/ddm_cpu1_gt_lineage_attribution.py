#!/usr/bin/env python
"""ddm_cpu1 — attribute the jg5 CPU-vs-CUDA distortion gap to GT-DECODE LINEAGE vs kernel drift.

WHY THIS EXISTS
---------------
``upstream/evaluate.py:39-42`` picks the ground-truth decoder BY DEVICE::

    if device.type == "cuda":  DefaultDatasetClass = DaliVideoDataset   # NVDEC
    else:                      DefaultDatasetClass = AVVideoDataset     # PyAV

So the contest-CPU axis and the contest-CUDA axis do not merely run different
kernels — they score against a DIFFERENT GROUND TRUTH DECODE. Our vehicle's pose
was solved against the DALI lineage, so the CPU axis pays the full distance
between the two GT tables.

THE MEASUREMENT
---------------
``compute_distortion`` is symmetric in its two arguments and the candidate side
does not depend on which GT you compare against. Therefore ONE forward pass over
the retained inflated raws yields the candidate's scorer outputs, and those
outputs can then be scored against BOTH GT lineages. That isolates the lineage
term exactly, at the cost of a single pass.

POSITIVE CONTROL (mandatory, per the confound-gates discipline)
---------------------------------------------------------------
The PyAV leg has a KNOWN ANSWER: the jg5 macOS-CPU advisory row measured
d_pose = 0.00014701 and d_seg = 0.00034740 on these exact raws. If this script's
PyAV leg does not reproduce those to report precision, the GT tables are not the
objects the official path uses and NO attribution verdict is admissible.

AXIS
----
``[macOS-CPU advisory]``. This is a DECOMPOSITION of an existing advisory row,
never a score claim, never a promotion, never an authority row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "upstream"


def sha256_file(path: Path, *, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def assert_lineage(path: Path, expect_sha: str, label: str) -> dict:
    """Fail closed on GT lineage: resolve by CONTENT HASH, never by path/basename.

    This is the ddm_dg1 / ddm_na10 cure applied at the load site: the basename
    ``gt_first6_n600.npy`` exists at two shas with OPPOSITE lineages.
    """
    got = sha256_file(path)
    if got != expect_sha:
        raise SystemExit(
            f"GT LINEAGE REFUSAL [{label}]: {path}\n"
            f"  expected sha256 {expect_sha}\n"
            f"  measured sha256 {got}\n"
            "  A basename is not a lineage. Refusing rather than guessing."
        )
    return {"label": label, "path": str(path), "sha256": got, "bytes": path.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="inflated 0.raw from the candidate")
    ap.add_argument("--raw-sha256", type=str, required=True, help="expected sha of --raw; mismatch refuses")
    ap.add_argument("--gt-pose-pyav", type=Path, required=True)
    ap.add_argument("--gt-pose-pyav-sha256", type=str, required=True)
    ap.add_argument("--gt-pose-dali", type=Path, required=True)
    ap.add_argument("--gt-pose-dali-sha256", type=str, required=True)
    ap.add_argument("--gt-argmax-pyav", type=Path, required=True)
    ap.add_argument("--gt-argmax-pyav-sha256", type=str, required=True)
    ap.add_argument("--gt-argmax-dali", type=Path, required=True)
    ap.add_argument("--gt-argmax-dali-sha256", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16, help="MATCH the advisory row: batch shape is part of the instrument")
    ap.add_argument("--limit-pairs", type=int, default=0, help="0 = all; >0 is a PLUMBING SMOKE ONLY, never a verdict")
    ap.add_argument("--torch-threads", type=int, default=0, help="0 = torch default")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(UPSTREAM))
    import torch
    from frame_utils import camera_size, seq_len
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cpu")

    # ---- custody: every input pinned by CONTENT, refuse on drift -------------
    t_cust = time.time()
    custody = {
        "raw": assert_lineage(args.raw, args.raw_sha256, "candidate_inflated_raw"),
        "gt_pose_pyav": assert_lineage(args.gt_pose_pyav, args.gt_pose_pyav_sha256, "GT_POSE_PYAV_YUV420_TO_RGB"),
        "gt_pose_dali": assert_lineage(args.gt_pose_dali, args.gt_pose_dali_sha256, "GT_POSE_DALI_NVDEC"),
        "gt_argmax_pyav": assert_lineage(args.gt_argmax_pyav, args.gt_argmax_pyav_sha256, "GT_ARGMAX_PYAV_YUV420_TO_RGB"),
        "gt_argmax_dali": assert_lineage(args.gt_argmax_dali, args.gt_argmax_dali_sha256, "GT_ARGMAX_DALI_NVDEC"),
    }
    custody_seconds = time.time() - t_cust
    print(f"CUSTODY OK ({custody_seconds:.1f}s): 5 inputs pinned by sha256", flush=True)

    W, H = camera_size
    frame_bytes = H * W * 3
    n_frames = args.raw.stat().st_size // frame_bytes
    n_pairs_total = n_frames // seq_len
    n_pairs = n_pairs_total if args.limit_pairs <= 0 else min(args.limit_pairs, n_pairs_total)
    is_smoke = n_pairs != n_pairs_total
    print(f"RAW: {n_frames} frames -> {n_pairs_total} pairs; running {n_pairs}"
          f"{'  [PLUMBING SMOKE - NOT A VERDICT]' if is_smoke else ''}", flush=True)

    net = DistortionNet().eval().to(device=device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    mm = np.memmap(args.raw, dtype=np.uint8, mode="r", shape=(n_frames, H, W, 3))

    pose_pred = np.zeros((n_pairs, 6), dtype=np.float32)
    # SegNet emits logits at segnet_model_input_size = (512, 384) -> argmax (384, 512)
    argmax_pred = np.zeros((n_pairs, 384, 512), dtype=np.uint8)

    t0 = time.time()
    done = 0
    with torch.inference_mode():
        while done < n_pairs:
            b = min(args.batch_size, n_pairs - done)
            # (B, seq_len, H, W, C) uint8 — exactly TensorVideoDataset's contract
            chunk = np.ascontiguousarray(
                mm[done * seq_len : (done + b) * seq_len].reshape(b, seq_len, H, W, 3)
            )
            x = torch.from_numpy(chunk).to(device)
            pose_out, seg_out = net(x)
            # PoseNet head 'pose': distortion uses [..., :out//2] -> first 6 dims
            head = next(h for h in net.posenet.hydra.heads if h.name == "pose")
            pose_pred[done : done + b] = pose_out["pose"][..., : head.out // 2].cpu().numpy()
            argmax_pred[done : done + b] = seg_out.argmax(dim=1).to(torch.uint8).cpu().numpy()
            done += b
            if (done % (args.batch_size * 5) == 0) or done == n_pairs:
                el = time.time() - t0
                print(f"  {done}/{n_pairs} pairs  {el:.1f}s  ({el / max(done,1):.3f} s/pair)", flush=True)
    forward_seconds = time.time() - t0

    # ---- score the SAME candidate outputs against BOTH GT lineages ----------
    # The slice below aligns the GT tables to whatever the forward pass produced.
    # In the VERDICT configuration n_pairs == n_pairs_total == 600, so it is the
    # IDENTITY and no subset is taken. It is a prefix ONLY under --limit-pairs,
    # which the script already stamps as `is_plumbing_smoke_not_a_verdict: true`
    # and which must never be cited as a result. A prefix is the correct shape
    # for a plumbing check precisely because pose prefixes measure HARDER
    # (2.54-4.21x, ddm_mi1): a smoke that passes on the hard end is a stronger
    # plumbing signal, and it is not a population estimate either way.
    gp_pyav = np.load(args.gt_pose_pyav)[:n_pairs].astype(np.float64)  # SUBSET_SELECTION_OK:identity at n600 verdict config; prefix only in the self-labelled plumbing smoke
    gp_dali = np.load(args.gt_pose_dali)[:n_pairs].astype(np.float64)  # SUBSET_SELECTION_OK:identity at n600 verdict config; prefix only in the self-labelled plumbing smoke
    P = pose_pred.astype(np.float64)

    # upstream: MSE over the 6 dims, then mean over pairs
    dpose_pyav_pp = ((P - gp_pyav) ** 2).mean(axis=1)
    dpose_dali_pp = ((P - gp_dali) ** 2).mean(axis=1)

    ga_pyav = np.load(args.gt_argmax_pyav, mmap_mode="r")
    ga_dali = np.load(args.gt_argmax_dali, mmap_mode="r")
    dseg_pyav_pp = np.zeros(n_pairs, dtype=np.float64)
    dseg_dali_pp = np.zeros(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        a = argmax_pred[i]
        dseg_pyav_pp[i] = float((a != np.asarray(ga_pyav[i])).mean())
        dseg_dali_pp[i] = float((a != np.asarray(ga_dali[i])).mean())

    C_pose = float(((gp_pyav - gp_dali) ** 2).mean())
    gt_argmax_disagree = float(
        np.mean([float((np.asarray(ga_pyav[i]) != np.asarray(ga_dali[i])).mean()) for i in range(n_pairs)])
    )

    res = {
        "schema": "ddm_cpu1_gt_lineage_attribution.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "is_plumbing_smoke_not_a_verdict": is_smoke,
        "n_pairs": int(n_pairs),
        "n_pairs_total": int(n_pairs_total),
        "git_head": git_head(),
        "host": platform.platform(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "batch_size": int(args.batch_size),
        "device": "cpu",
        "custody": custody,
        "forward_seconds": forward_seconds,
        "custody_hash_seconds": custody_seconds,
        "d_pose_vs_PYAV_gt": float(dpose_pyav_pp.mean()),
        "d_pose_vs_DALI_gt": float(dpose_dali_pp.mean()),
        "d_seg_vs_PYAV_gt": float(dseg_pyav_pp.mean()),
        "d_seg_vs_DALI_gt": float(dseg_dali_pp.mean()),
        "C_pose_gt_table_mse": C_pose,
        "gt_argmax_lineage_disagreement_rate": gt_argmax_disagree,
    }
    res["pose_lineage_term_abs"] = res["d_pose_vs_PYAV_gt"] - res["d_pose_vs_DALI_gt"]
    res["seg_lineage_term_abs"] = res["d_seg_vs_PYAV_gt"] - res["d_seg_vs_DALI_gt"]
    res["seg_lineage_ratio"] = (
        res["d_seg_vs_PYAV_gt"] / res["d_seg_vs_DALI_gt"] if res["d_seg_vs_DALI_gt"] else None
    )

    # ---- ALWAYS KEEP THE PAYLOAD -------------------------------------------
    pose_path = args.out_dir / f"cpu1_pose_pred_n{n_pairs}.npy"
    argmax_path = args.out_dir / f"cpu1_seg_argmax_n{n_pairs}.npy"
    perpair_path = args.out_dir / f"cpu1_per_pair_n{n_pairs}.npz"
    np.save(pose_path, pose_pred)  # PAYLOAD_WRITE_ORDER_OK:result custody records this retained array's post-write size and digest
    np.save(argmax_path, argmax_pred)  # PAYLOAD_WRITE_ORDER_OK:result custody records this retained array's post-write size and digest
    np.savez(  # PAYLOAD_WRITE_ORDER_OK:result custody records this retained bundle's post-write size and digest
        perpair_path,
        d_pose_pyav=dpose_pyav_pp, d_pose_dali=dpose_dali_pp,
        d_seg_pyav=dseg_pyav_pp, d_seg_dali=dseg_dali_pp,
    )
    res["retained_payloads"] = [
        {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in (pose_path, argmax_path, perpair_path)
    ]

    out_json = args.out_dir / ("cpu1_attribution_smoke.json" if is_smoke else "cpu1_attribution_n600.json")
    out_json.write_text(json.dumps(res, indent=2, sort_keys=True))

    print("\n=== GT-LINEAGE ATTRIBUTION (candidate outputs FIXED, GT swapped) ===")
    print(f"  d_pose vs PyAV GT : {res['d_pose_vs_PYAV_gt']:.10e}")
    print(f"  d_pose vs DALI GT : {res['d_pose_vs_DALI_gt']:.10e}")
    print(f"  pose lineage term : {res['pose_lineage_term_abs']:.10e}")
    print(f"  C (GT table MSE)  : {C_pose:.10e}")
    print(f"  d_seg  vs PyAV GT : {res['d_seg_vs_PYAV_gt']:.10e}")
    print(f"  d_seg  vs DALI GT : {res['d_seg_vs_DALI_gt']:.10e}")
    print(f"  seg lineage ratio : {res['seg_lineage_ratio']}")
    print(f"  GT argmax lineage disagreement: {gt_argmax_disagree:.10e}")
    print(f"\nWROTE {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
