#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pu2 F1 for ddm_hg1 — cache cx1's per-pixel SegNet argmax and produce the DIRECTED flip split.

WHY.  ``ddm_hg1`` reports that no per-pixel ``cx1`` argmax is cached anywhere, so its
≈15.2%-of-gap Road<->Lane figure is a HYPOTHESIS that mixes vehicles.  This produces
the missing measurement on cx1's OWN n600 argmax and caches the argmax planes so no
future arm has to re-run a scorer pass to ask a per-class question.

DIRECTED, NEVER POOLED.  ``hg1`` measured that Lane<->Road is asymmetric in EXTENT
(Lane shell 75.04% vs <=9.73%, mean depth 1.134 px -- the only truncating side) while
Road<->Movable (1.03x) and Road<->Undrivable (1.05x) are symmetric.  A pooled or
undirected count destroys exactly that signal, and ``hg1`` further measured three
pathologies of pooling: sign-cancellation, MASS DILUTION (12.17x on MyCar, needing no
sign flip at all), and Simpson reversal.  So this emits the full 5x5 DIRECTED
confusion matrix C[gt_class][rendered_class], per pair and summed -- never a scalar.

EXACT SCORER PATH (upstream/modules.py, source-inspected):
  SegNet.preprocess_input : x[:, -1, ...]  then bilinear -> (384, 512)
  SegNet.compute_distortion : (argmax(out_gt) != argmax(out_rendered)).float().mean()
So d_seg per pair = disagreement fraction over 384*512 = 196,608 scorer pixels, and
frame_0 is structurally invisible to SegNet.

POSITIVE CONTROL (fail-closed): the summed disagreement rate MUST reproduce cx1's
evaluator row d_seg = 0.00431179.  If it does not, the argmax planes are wrong and
every directed count built on them is void -- the run refuses rather than reporting.

CLASS ORDER (comma10k canonical, MEASURED -- never re-derive by luma-sort):
  0=Road  1=Lane markings  2=Undrivable(incl sky)  3=Movable(cars)  4=MyCar(ego hood)

Axis: [macOS-CPU frozen-SegNet advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.  Chunked <=120 pairs per the slot rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

UPSTREAM = Path("/Volumes/VertigoDataTier/pact/molab_witness_machine_upstream_20260709")
CX1_SUB = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
               "submissions/v4d_cx1_pj2ix2")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
N_PAIRS = 600
SEG_H, SEG_W = 384, 512
SEG_PX = SEG_H * SEG_W
CX1_D_SEG = 0.00431179          # cx1 report.txt, the fail-closed target
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_segnet():
    """Frozen upstream SegNet on the custodied snapshot (never a re-implementation)."""
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import modules
    import torch
    from safetensors.torch import load_file
    if Path(modules.__file__).resolve() != (UPSTREAM / "modules.py").resolve():
        raise RuntimeError("imported non-custodied upstream modules.py")
    torch.set_num_threads(4)
    seg = modules.SegNet().eval().cpu()
    seg.load_state_dict(load_file(str(UPSTREAM / "models/segnet.safetensors"), device="cpu"))
    for p in seg.parameters():
        p.requires_grad = False
    return seg, modules


def argmax_of(seg, frame_u8: np.ndarray) -> np.ndarray:
    """(874,1164,3) uint8 -> (384,512) uint8 argmax, through the EXACT scorer preprocess."""
    import torch
    x = torch.from_numpy(frame_u8[None, None]).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        out = seg(seg.preprocess_input(x))
    return out.argmax(dim=1)[0].to(torch.uint8).numpy()


def _gt_frame1_stream(lo: int, hi: int):
    """Yield (pair_idx, GT frame_1) for pair_idx in [lo, hi), streamed from 0.mkv.

    GT CUSTODY, and this is the bug this helper exists to fix.  The first draft read GT
    from ``p3v2.load_pair`` -- which loads a COMPOSED-vehicle reconstruction, not ground
    truth (MEASURED: meanabs 12.5-27.9 vs the real decode, maxabs 253).  Using it put the
    per-pair d_seg at 0.042 against cx1's evaluator row of 0.00431179, ~10x high, and the
    fail-closed positive control refused it.

    The evaluator batches seq_len=2 NON-OVERLAPPING, so pair i is video frames (2i, 2i+1)
    and SegNet reads the last one, 2i+1.  Decode is via ``frame_utils.yuv420_to_rgb`` --
    never PyAV rgb24, which manufactures ~100x phantom pose (CLAUDE.md).
    """
    import av
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from frame_utils import yuv420_to_rgb
    container = av.open(str(UPSTREAM / "videos/0.mkv"))
    try:
        for fidx, frame in enumerate(container.decode(video=0)):
            pair, slot = divmod(fidx, 2)
            if slot != 1 or pair < lo:
                continue
            if pair >= hi:
                break
            yield pair, yuv420_to_rgb(frame).numpy()
    finally:
        container.close()


def run(args) -> None:
    import torch
    torch.set_num_threads(4)
    seg, _ = load_segnet()
    if str(CX1_SUB) not in sys.path:
        sys.path.insert(0, str(CX1_SUB))
    import inflate_runner as ir
    dec = ir.Decoder(CX1_SUB / "archive")

    OUT.mkdir(parents=True, exist_ok=True)
    gt_mm = np.lib.format.open_memmap(OUT / "gt_argmax_n600.npy", mode="w+" if not
                                      (OUT / "gt_argmax_n600.npy").exists() or not args.resume
                                      else "r+", dtype=np.uint8,
                                      shape=(N_PAIRS, SEG_H, SEG_W))
    cx_mm = np.lib.format.open_memmap(OUT / "cx1_argmax_n600.npy", mode="w+" if not
                                      (OUT / "cx1_argmax_n600.npy").exists() or not args.resume
                                      else "r+", dtype=np.uint8,
                                      shape=(N_PAIRS, SEG_H, SEG_W))
    jl = OUT / "per_pair_directed.jsonl"
    done: dict[int, dict] = {}
    if jl.exists() and args.resume:
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                done[int(rr["pair"])] = rr
        print(f"[argmax] resume: {len(done)} pairs cached", flush=True)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    lo, hi = args.start, min(args.start + args.chunk, N_PAIRS)
    for i, gt_f1 in _gt_frame1_stream(lo, hi):
        if i in done:
            continue
        a_gt = argmax_of(seg, gt_f1)         # SegNet reads frame_1 only
        a_cx = argmax_of(seg, dec.f1(i))
        gt_mm[i] = a_gt
        cx_mm[i] = a_cx
        C = np.zeros((5, 5), np.int64)
        np.add.at(C, (a_gt.ravel().astype(np.int64), a_cx.ravel().astype(np.int64)), 1)
        flips = int(SEG_PX - np.trace(C))
        fj.write(json.dumps({"pair": i, "flips": flips,
                             "d_seg": flips / SEG_PX, "C": C.tolist()}) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        done[i] = {"pair": i, "flips": flips}
        if i % 20 == 0 or i == hi - 1:
            print(f"  [argmax {i:3d}] flips={flips} d_seg={flips/SEG_PX:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fj.close()
    gt_mm.flush()
    cx_mm.flush()
    print(f"[argmax] chunk [{lo},{hi}) done; {len(done)}/{N_PAIRS} total", flush=True)
    if len(done) == N_PAIRS:
        summarize()


def summarize() -> None:
    jl = OUT / "per_pair_directed.jsonl"
    rows: dict[int, dict] = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            rr = json.loads(ln)
            rows[int(rr["pair"])] = rr
    if len(rows) != N_PAIRS:
        raise SystemExit(f"incomplete: {len(rows)}/{N_PAIRS}")
    C = np.zeros((5, 5), np.int64)
    per_pair = np.zeros(N_PAIRS)
    for i in range(N_PAIRS):
        C += np.asarray(rows[i]["C"], np.int64)
        per_pair[i] = rows[i]["d_seg"]
    total_flips = int(C.sum() - np.trace(C))
    d_seg = float(per_pair.mean())
    rel = abs(d_seg - CX1_D_SEG) / CX1_D_SEG
    # FAIL-CLOSED positive control: the argmax planes must reproduce the evaluator row.
    verdict = "ARGMAX_VERIFIED" if rel < 2e-5 else "ARGMAX_REFUTED_DO_NOT_USE"
    off = C.copy()
    np.fill_diagonal(off, 0)
    pairs_directed = []
    for a in range(5):
        for b in range(5):
            if a == b or off[a, b] == 0:
                continue
            pairs_directed.append({
                "gt_class": CLASSES[a], "rendered_class": CLASSES[b],
                "gt_idx": a, "rendered_idx": b, "flips": int(off[a, b]),
                "pct_of_all_flips": 100.0 * off[a, b] / total_flips,
            })
    pairs_directed.sort(key=lambda r: -r["flips"])
    # undirected edge totals + the asymmetry that pooling would destroy
    edges = []
    for a in range(5):
        for b in range(a + 1, 5):
            fwd, bwd = int(off[a, b]), int(off[b, a])
            if fwd + bwd == 0:
                continue
            edges.append({
                "edge": f"{CLASSES[a]}<->{CLASSES[b]}",
                "gt_{}_to_{}".format(CLASSES[a], CLASSES[b]): fwd,
                "gt_{}_to_{}".format(CLASSES[b], CLASSES[a]): bwd,
                "total": fwd + bwd,
                "pct_of_all_flips": 100.0 * (fwd + bwd) / total_flips,
                "asymmetry_ratio": (max(fwd, bwd) / min(fwd, bwd)) if min(fwd, bwd) else None,
                "dominant_direction": (f"{CLASSES[a]}->{CLASSES[b]}" if fwd >= bwd
                                       else f"{CLASSES[b]}->{CLASSES[a]}"),
            })
    edges.sort(key=lambda r: -r["total"])
    receipt = {
        "schema": "ddm_pu2_cx1_directed_flips.v1", "utc": _utc(), "git": _git(),
        "axis": "[macOS-CPU frozen-SegNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False, "pointer_moved": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "base_vehicle": "v4d_cx1_pj2ix2 (cx1's OWN argmax; no vehicle mixing)",
        "positive_control": {
            "d_seg_measured": d_seg, "d_seg_cx1_report": CX1_D_SEG,
            "rel_err": rel, "verdict": verdict,
            "note": "summed argmax disagreement must reproduce the evaluator row",
        },
        "scorer_pixels_per_pair": SEG_PX, "n_pairs": N_PAIRS,
        "total_flips": total_flips,
        "flips_per_pair_mean": total_flips / N_PAIRS,
        "class_order": list(CLASSES),
        "confusion_matrix_gt_by_rendered": C.tolist(),
        "directed_flips": pairs_directed,
        "undirected_edges_with_asymmetry": edges,
    }
    (OUT / "cx1_directed_flip_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "confusion_matrix_gt_by_rendered"}, indent=1), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100,
                    help="pairs per invocation (slot rule: <=120)")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.chunk > 120:
        raise SystemExit("chunk must be <=120 per the scorer-slot rule")
    if args.summarize_only:
        summarize()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
