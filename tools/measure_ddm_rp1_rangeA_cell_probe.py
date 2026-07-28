"""ddm_rp1 — range(A)-cell realized probe (project -> uint8 -> real SegNet/PoseNet).

THE QUESTION (gc5 M2-Q2): does a range(A)-projected, generically ker-filled, uint8-rounded
version of solved frames still land inside the same SegNet argmax cells through the real decode?

Substrate note (custody, verified by rp1): the 1.52e-4 exact-solve object (q1) is a MEASURED
scorer control with ZERO materialized frame records on disk (only q4/q8 box-solve chunks exist,
d_seg 1.16e-3). Per the charter honest-boundary clause this probe runs on the largest custodied
real solved substrate available. `--substrate gt` uses the GT frames (gt_f1 SegNet argmax == the
lstars cells; the highest-margin / OPTIMISTIC-bound operating point — a BREAK here is decisive,
a HOLD is necessary-not-sufficient). `--substrate boxsolve` runs the actual box-solve frames.

Conditions (per pair, camera-space frame X, uint8):
  C0 (control): X as realized -> real SegNet argmax vs lstars. (GT: 0 flips by construction.)
  C1a (THE probe): Y = round(clip(project_range(X),0,255)) -> uint8. project_range(X)=Q_h X Q_w is
      the exact min-norm (zero-ker) camera preimage of A(X); rounding it to uint8 is #532's naive
      range-carrier lift (generic zero-ker fill, decoder-derivable from A(X) for free). Score Y.
  C2: round(project_range(X)+project_kernel(X)) = round(X) = X == C0. Degenerate BY CONSTRUCTION
      because the solve is over uint8 (exact_binary_solve): projection round-trip of an integer
      frame is float-exact, so all realization damage lives in C1. Reported, not recomputed.

All results [macOS-CPU frozen-scorer advisory], score_claim=false. Pointer 0.1910828242 UNMOVED.
Reuses upstream/modules.py (authority forward) + tac.optimization.resize_full_kernel #580 projector
+ tac.optimization.resize_null_preimage (A parity), never reinvented.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = Path("/Users/adpena/Projects/pact")


def _load_scorers():
    sys.path.insert(0, str(MAIN / "upstream"))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(MAIN / "src"))
    import modules as M
    import torch
    from safetensors.torch import load_file

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    seg = M.SegNet().eval()
    seg.load_state_dict(load_file(str(MAIN / "upstream/models/segnet.safetensors"), device="cpu"))
    pose = M.PoseNet().eval()
    pose.load_state_dict(load_file(str(MAIN / "upstream/models/posenet.safetensors"), device="cpu"))
    return torch, seg, pose


def _seg_argmax_and_margin(torch, seg, frames_hwc):
    """frames_hwc: (N,874,1164,3) uint8/float -> argmax (N,384,512) int64, margin (N,384,512) f32."""
    x5 = torch.from_numpy(np.ascontiguousarray(frames_hwc).astype(np.float32)).permute(0, 3, 1, 2).unsqueeze(1)
    logits = seg(seg.preprocess_input(x5))  # (N,5,384,512)
    top2 = torch.topk(logits, 2, dim=1).values  # (N,2,384,512)
    margin = (top2[:, 0] - top2[:, 1]).numpy().astype(np.float32)
    am = logits.argmax(dim=1).numpy().astype(np.int64)
    return am, margin


def _pose6(torch, pose, f0_hwc, f1_hwc):
    pair = torch.from_numpy(np.stack([f0_hwc, f1_hwc], axis=1).astype(np.float32)).permute(0, 1, 4, 2, 3)
    out = pose(pose.preprocess_input(pair))
    return out["pose"][:, :6].numpy().astype(np.float64)  # (N,6)


def run_chunk(substrate: str, start: int, end: int, out_dir: Path, boxsolve_dir: Path | None):
    from tac.optimization.resize_full_kernel import FullResizeKernel

    torch, seg, pose = _load_scorers()
    K = FullResizeKernel.build()
    gt = np.load(str(MAIN / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    lstars = gt["lstars"]
    gt_margins = gt["margins"]
    gt_poses = gt["gt_poses"]

    if substrate == "gt":
        f0_all, f1_all = gt["gt_f0"], gt["gt_f1"]
    elif substrate == "boxsolve":
        f0_all, f1_all = _load_boxsolve_frames(boxsolve_dir, start, end)
    else:
        raise ValueError(substrate)

    n_sites = 384 * 512
    per_pair = []
    t0 = time.time()
    for i in range(start, end):
        f1 = f1_all[i if substrate == "gt" else i - start]
        f0 = f0_all[i if substrate == "gt" else i - start]
        ls = lstars[i]
        gm = gt_margins[i]

        # C1a: range-carrier zero-ker lift of both frames.
        f1r = K.project_range(f1.astype(np.float64))
        f0r = K.project_range(f0.astype(np.float64))
        y1 = np.clip(np.round(f1r), 0, 255).astype(np.uint8)
        y0 = np.clip(np.round(f0r), 0, 255).astype(np.uint8)

        # exactness diagnostics (camera-space): how far the round pushed off range(A).
        # residual of A on the rounded frame vs on X (float32-consistent with SegNet resize).
        # C0 seg = lstars (custody-verified elsewhere); compute C1a seg forward.
        am1, m1 = _seg_argmax_and_margin(torch, seg, y1[None])
        am1 = am1[0]
        m1 = m1[0]
        flip = (am1 != ls)
        nflip = int(flip.sum())

        # per-class flip mass keyed by GT class (lstars) and by predicted-wrong class.
        per_class_gt = {int(c): int(((ls == c) & flip).sum()) for c in range(5)}
        class_sites = {int(c): int((ls == c).sum()) for c in range(5)}

        # margin-erosion telemetry: GT (pre-round) margin at flipped vs held sites.
        held = ~flip
        pre_flip = gm[flip]
        pre_held = gm[held]
        post_flip = m1[flip]
        post_held = m1[held]

        # pose
        p1 = _pose6(torch, pose, y0[None], y1[None])[0]
        dpose = float(np.mean((p1 - gt_poses[i]) ** 2))

        per_pair.append({
            "pair": i,
            "c1a_flips": nflip,
            "c1a_dseg": nflip / n_sites,
            "c1a_dpose": dpose,
            "per_class_gt_flips": per_class_gt,
            "class_sites": class_sites,
            "pre_margin_flipped_mean": float(pre_flip.mean()) if pre_flip.size else None,
            "pre_margin_held_mean": float(pre_held.mean()) if pre_held.size else None,
            "pre_margin_flipped_p90": float(np.percentile(pre_flip, 90)) if pre_flip.size else None,
            "post_margin_flipped_mean": float(post_flip.mean()) if post_flip.size else None,
            "post_margin_held_mean": float(post_held.mean()) if post_held.size else None,
            "range_round_maxabs_camera": float(np.max(np.abs(np.clip(np.round(f1r), 0, 255) - f1r))),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": "ddm_rp1_rangeA_cell_probe_chunk.v1",
        "substrate": substrate,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 UNMOVED",
        "start": start,
        "end": end,
        "n_sites_per_pair": n_sites,
        "elapsed_s": time.time() - t0,
        "per_pair": per_pair,
    }
    out_path = out_dir / f"chunk_{substrate}_{start:04d}_{end:04d}.json"
    out_path.write_text(json.dumps(rec, indent=2))
    tot_flip = sum(p["c1a_flips"] for p in per_pair)
    print(f"[rp1] {substrate} [{start}:{end}] pairs={len(per_pair)} "
          f"C1a total_flips={tot_flip} mean_dseg={tot_flip/(len(per_pair)*n_sites):.6e} "
          f"mean_dpose={np.mean([p['c1a_dpose'] for p in per_pair]):.6e} "
          f"elapsed={rec['elapsed_s']:.1f}s -> {out_path}")
    return out_path


def _load_boxsolve_frames(boxsolve_dir: Path, start: int, end: int):
    """Inflate box-solve archive receiver for pairs [start,end). Returns (f0,f1) subset arrays."""
    raise NotImplementedError(
        "boxsolve inflate wired only when GT probe HOLDS; see charter adaptive plan"
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=["gt", "boxsolve"], default="gt")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out-dir", default="/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/chunks")
    ap.add_argument("--boxsolve-dir", default=None)
    a = ap.parse_args(argv)
    run_chunk(a.substrate, a.start, a.end, Path(a.out_dir),
              Path(a.boxsolve_dir) if a.boxsolve_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
