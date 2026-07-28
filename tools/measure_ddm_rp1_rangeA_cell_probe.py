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
        stage_root = _ensure_boxsolve_inflated(
            boxsolve_dir if boxsolve_dir is not None else Path(BOXSOLVE_DIR_DEFAULT),
            Path(BOXSOLVE_WORK_DEFAULT),
        )
        f0_all, f1_all = _load_boxsolve_frames(stage_root, start, end)
    else:
        raise ValueError(substrate)

    n_sites = 384 * 512
    per_pair = []
    t0 = time.time()
    for i in range(start, end):
        f1 = f1_all[i if substrate == "gt" else i - start]
        f0 = f0_all[i if substrate == "gt" else i - start]
        ls = lstars[i]

        c0_row: dict | None = None
        if substrate == "gt":
            # C0 = GT frames = the cells themselves (custody-verified 0 flips in rp1).
            gm = gt_margins[i]
            am0 = ls
        else:
            # C0 (box-solve baseline): score the AS-REALIZED receiver frames. Its
            # margins are THE pre-round reference at this operating point, and its
            # argmax is the cell assignment the lift must hold.
            am0, m0 = _seg_argmax_and_margin(torch, seg, f1[None])
            am0 = am0[0]
            gm = m0[0]
            c0_flip = (am0 != ls)
            p0 = _pose6(torch, pose, f0[None], f1[None])[0]
            c0_row = {
                "c0_flips_vs_lstars": int(c0_flip.sum()),
                "c0_dseg_vs_lstars": int(c0_flip.sum()) / n_sites,
                "c0_dpose": float(np.mean((p0 - gt_poses[i]) ** 2)),
                "c0_margin_mean": float(gm.mean()),
            }

        # C1a: range-carrier zero-ker lift of both frames.
        f1r = K.project_range(f1.astype(np.float64))
        f0r = K.project_range(f0.astype(np.float64))
        y1 = np.clip(np.round(f1r), 0, 255).astype(np.uint8)
        y0 = np.clip(np.round(f0r), 0, 255).astype(np.uint8)

        # exactness diagnostics (camera-space): how far the round pushed off range(A).
        # residual of A on the rounded frame vs on X (float32-consistent with SegNet resize).
        # C0 seg = lstars (gt) or the box-solve forward above; compute C1a seg forward.
        am1, m1 = _seg_argmax_and_margin(torch, seg, y1[None])
        am1 = am1[0]
        m1 = m1[0]
        flip = (am1 != ls)
        nflip = int(flip.sum())
        cell_hold_flip = (am1 != am0)

        # per-class flip mass keyed by GT class (lstars) and by predicted-wrong class.
        per_class_gt = {int(c): int(((ls == c) & flip).sum()) for c in range(5)}
        class_sites = {int(c): int((ls == c).sum()) for c in range(5)}

        # margin-erosion telemetry: pre-round margin (GT margins for gt substrate;
        # the box-solve C0 forward margins for boxsolve) at CELL-HOLD flip sites
        # (C1 vs C0 argmax; identical to lstars-flips on the gt substrate).
        held = ~cell_hold_flip
        pre_flip = gm[cell_hold_flip]
        pre_held = gm[held]
        post_flip = m1[cell_hold_flip]
        post_held = m1[held]

        # pose
        p1 = _pose6(torch, pose, y0[None], y1[None])[0]
        dpose = float(np.mean((p1 - gt_poses[i]) ** 2))

        row = {
            "pair": i,
            "c1a_flips": nflip,
            "c1a_dseg": nflip / n_sites,
            "c1a_dpose": dpose,
            "c1a_cell_hold_flips_vs_c0": int(cell_hold_flip.sum()),
            "per_class_gt_flips": per_class_gt,
            "class_sites": class_sites,
            "pre_margin_flipped_mean": float(pre_flip.mean()) if pre_flip.size else None,
            "pre_margin_held_mean": float(pre_held.mean()) if pre_held.size else None,
            "pre_margin_flipped_p90": float(np.percentile(pre_flip, 90)) if pre_flip.size else None,
            "post_margin_flipped_mean": float(post_flip.mean()) if post_flip.size else None,
            "post_margin_held_mean": float(post_held.mean()) if post_held.size else None,
            "range_round_maxabs_camera": float(np.max(np.abs(np.clip(np.round(f1r), 0, 255) - f1r))),
        }
        if c0_row is not None:
            row.update(c0_row)
        per_pair.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": (
            "ddm_rp1_rangeA_cell_probe_chunk.v1"
            if substrate == "gt"
            else "ddm_rp1_rangeA_cell_probe_chunk.v2"
        ),
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


BOXSOLVE_ARCHIVE_SHA256 = "e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e"
BOXSOLVE_DIR_DEFAULT = (
    "/Volumes/VertigoDataTier/pact/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/"
    "stage_checkpoints/04_candidate"
)
BOXSOLVE_WORK_DEFAULT = "/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/boxsolve_inflate"
# r6cal 07-27 receiver-closed inflate custody: 1200 frames @ 874x1164x3 (memo
# r6cal_solved_object_byteclose_eval_20260727.md table row "inflated 0.raw").
BOXSOLVE_RAW_SHA256 = "32a773a23a79c036ca39352b9ca9a048e20c089dc45beaa4c847689083641558"
_FRAME_BYTES = 874 * 1164 * 3


def _ensure_boxsolve_inflated(boxsolve_dir: Path, work_dir: Path) -> Path:
    """Run the real v10 production receiver inflate once (write-once resumable).

    Returns the stage directory holding pair-NNNNNN.bin files (frame0+frame1
    camera raw per pair). Custody: archive sha bound to the ms2r_r3 box-solve
    candidate receipt; assembled raw sha cross-checked against r6cal's
    receiver-closed inflate custody (fails closed on drift).
    """
    from tac.witness_dsl.v10_production_receiver import inflate_archive

    archive_path = boxsolve_dir / "archive.zip"
    import hashlib

    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != BOXSOLVE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"box-solve archive custody drift: {digest.hexdigest()} != {BOXSOLVE_ARCHIVE_SHA256}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    stage_root = work_dir / ".v10-production-receiver" / "0"
    manifest_path = stage_root / "inflate-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest.get("archive_sha256") == BOXSOLVE_ARCHIVE_SHA256
            and manifest.get("raw_sha256") == BOXSOLVE_RAW_SHA256
            and manifest.get("pair_count") == 600
        ):
            return stage_root
        raise RuntimeError("existing box-solve inflate manifest custody drift")
    result = inflate_archive(
        boxsolve_dir,
        work_dir,
        MAIN / "upstream/public_test_video_names.txt",
    )
    if not result.completed:
        raise RuntimeError("box-solve inflate did not complete")
    if result.raw_sha256 != BOXSOLVE_RAW_SHA256:
        raise RuntimeError(
            f"box-solve inflate raw sha drift vs r6cal custody: "
            f"{result.raw_sha256} != {BOXSOLVE_RAW_SHA256}"
        )
    stage_root = work_dir / ".v10-production-receiver" / "0"
    if not stage_root.is_dir():
        raise RuntimeError(f"box-solve stage dir missing: {stage_root}")
    return stage_root


def _load_boxsolve_frames(stage_root: Path, start: int, end: int):
    """Read receiver-preserved pair stages for [start,end). Returns (f0,f1) uint8 arrays."""
    count = end - start
    f0 = np.empty((count, 874, 1164, 3), dtype=np.uint8)
    f1 = np.empty((count, 874, 1164, 3), dtype=np.uint8)
    for offset, pair_index in enumerate(range(start, end)):
        payload = (stage_root / f"pair-{pair_index:06d}.bin").read_bytes()
        if len(payload) != 2 * _FRAME_BYTES:
            raise RuntimeError(f"pair stage {pair_index} byte-count drift")
        f0[offset] = np.frombuffer(payload[:_FRAME_BYTES], dtype=np.uint8).reshape(874, 1164, 3)
        f1[offset] = np.frombuffer(payload[_FRAME_BYTES:], dtype=np.uint8).reshape(874, 1164, 3)
    return f0, f1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", choices=["gt", "boxsolve"], default="gt")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out-dir", default="/Volumes/VertigoDataTier/pact/ddm_rp1_20260728/chunks")
    ap.add_argument("--boxsolve-dir", default=BOXSOLVE_DIR_DEFAULT)
    ap.add_argument("--chunk", type=int, default=120,
                    help="write a resumable receipt every N pairs (single decode per process)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    for chunk_start in range(a.start, a.end, a.chunk):
        chunk_end = min(chunk_start + a.chunk, a.end)
        out_path = out_dir / f"chunk_{a.substrate}_{chunk_start:04d}_{chunk_end:04d}.json"
        if out_path.is_file():
            print(f"[rp1] resume-skip existing {out_path}")
            continue
        run_chunk(a.substrate, chunk_start, chunk_end, out_dir,
                  Path(a.boxsolve_dir) if a.boxsolve_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
