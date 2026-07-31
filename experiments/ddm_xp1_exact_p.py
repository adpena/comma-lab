# SPDX-License-Identifier: MIT
"""ddm_xp1 — the MAIN-owed EXACT-P measurement on the rung-1 birth endpoint (task #806).

The rung-1 endpoint manifest
(`/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/rung1_endpoint_manifest.json`)
records `above_nucleus_erased_estimate = 474` as a **DERIVED-ESTIMATE** (tr1 4-conn
betti0 x QA91 8-conn area frac) and names the exact obligation verbatim:

    "QA92 base-pass method on this endpoint render (experiments/ddm_qa92_carrier_discriminator.py)
     -> P in S units"

This is a **small driver** that REUSES QA92's exact base/P computation path — the
same `erased_super_nucleus_mask` (scipy 8-conn on GT Lane, super-nucleus >5px, a
component erased iff <50% of its GT-Lane pixels are classified Lane in the base
pass) and the same P formula (`P = 100 * sum(base_flip within target) / total_px`,
QA92 aggregate() line 353) — but on the **ep641 birth endpoint** checkpoint
(`.../ddm_r1c_20260731/window_01/checkpoints/stage_seg_trunk_tau_final.npz`,
sha 40553db8..., n600 d_seg 0.00426407708) instead of QA92's ep499 control_tail
(sha a2dc86b8..., d_seg 0.0049411).  NO paint tiers (those were QA92's DISCRIMINATOR;
MAIN owes only the base-pass P).  It ADDS the realized-Lane super-nucleus component
count (8-conn) the base-pass P does not itself yield, for the birth-progress line.

METHOD (n600; frozen CPU-torch SegNet = authority, NEVER MPS; base pass ONLY):
  BASE : render ep641 endpoint frame (model.render_frame -> MLX-cpu = trainer
         realized_gate) -> deploy R (_torch_R_to_camera_uint8: bicubic up to
         874x1164 -> uint8) -> frozen SegNet (cpu_verdict_d_seg_argmax_batch,
         chunk 120 / seg_batch 12) -> base realized argmax + base d_seg
         (validation target: 0.00426407708 == manifest n600_d_seg).
  P    : per pair, erased super-nucleus target T (QA92 erased_super_nucleus_mask,
         8-conn) -> base_flip_T = ((base_realized != gt) & T).sum();
         P = 100 * sum(base_flip_T) / (n600 * 384 * 512)  [S-units].
  betti0: realized Lane super-nucleus count (8-conn label of base_realized==Lane,
         keep >5px) vs GT super-nucleus count (from erased_super_nucleus_mask
         n_super) — scorer basis, distinct from the birth-gate telemetry basis.

OUTPUTS (all [macOS-CPU advisory], score_claim=false; additive-S sole authority):
  P_pool_S_units, n_erased/n_super super-nucleus (8-conn), realized_super_nucleus,
  base_dseg_mean, base_per_class_S_units, base_lane_S_units, target_px_total.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Seeded + resumable + atomic.
[no-triality] [p0-ledger-ok]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import numpy.lib.format as _npfmt

REPO = Path("/Users/adpena/Projects/pact")
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))

# REUSE QA92's exact base/P computation surface (do NOT re-derive).
from ddm_qa92_carrier_discriminator import (
    CLASS_ORDER,
    LANE_CLASS,
    N_CLASSES,
    NUCLEUS_PX,
    SEG_H,
    SEG_W,
    _atomic_save_npz,
    _atomic_write_bytes,
    _per_class_flip_counts,
    _sha256_file,
    erased_super_nucleus_mask,
    load_module,
)

DEFAULT_CKPT = (
    "/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/window_01/"
    "checkpoints/stage_seg_trunk_tau_final.npz"
)
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_xp1_20260731"


# -------------------------------------------------------------------- gt loader
def load_lstars(gt_cache: Path, num_pairs: int) -> np.ndarray:
    """Return lstars (P,384,512) int64 from the gt cache.  Base-pass P needs ONLY
    the GT SegNet argmax labels — no gt_f1 (no paint tier), so this is memory-light."""
    import zipfile

    z = zipfile.ZipFile(gt_cache)
    with z.open("lstars.npy") as f:
        lst = _npfmt.read_array(f)[:num_pairs].astype(np.int64)
    return np.ascontiguousarray(lst)


# -------------------------------------------------------------- per-chunk worker
def run_chunk(args, c0: int, c1: int, model, seg, lstars, structure) -> dict:
    """Render base 384 -> R -> uint8 -> frozen SegNet base pass over pairs [c0,c1).
    Per pair: erased super-nucleus target (QA92 8-conn), base_flip_T, per-class flip
    counts, realized Lane super-nucleus component count.  Base pass ONLY (no paint)."""
    import mlx.core as mx
    import scipy.ndimage as ndi
    from train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )

    idxs = list(range(c0, c1))
    npc = len(idxs)

    # ---- render base 384 (deploy stream = mlx cpu, matches trainer realized_gate) ----
    base384: list[np.ndarray] = []
    with mx.stream(mx.cpu):
        for i in idxs:
            rgb = model.render_frame(int(i))
            mx.eval(rgb)
            base384.append(np.asarray(rgb, dtype=np.float32)[0])  # (384,512,3)
    base_cam = [_torch_R_to_camera_uint8(f) for f in base384]
    gts = [lstars[i] for i in idxs]

    # ---- BASE SegNet pass (batched, seg_batch) ----
    base_realized = np.zeros((npc, SEG_H, SEG_W), dtype=np.int64)
    base_dseg = np.zeros(npc)
    for b0 in range(0, npc, args.seg_batch):
        b1 = min(b0 + args.seg_batch, npc)
        ds, rz = cpu_verdict_d_seg_argmax_batch(seg, base_cam[b0:b1], gts[b0:b1])
        base_dseg[b0:b1] = ds
        base_realized[b0:b1] = np.asarray(rz)

    # ---- per-pair accumulators (base pass only) ----
    cls_gt = np.zeros((npc, N_CLASSES), dtype=np.int64)
    cls_base = np.zeros((npc, N_CLASSES), dtype=np.int64)
    base_flip_T = np.zeros(npc, dtype=np.int64)     # flips inside erased super-nucleus target
    base_flip_off = np.zeros(npc, dtype=np.int64)   # flips outside T (bookkeeping)
    n_super = np.zeros(npc, dtype=np.int64)         # GT Lane super-nucleus components (8-conn)
    n_erased = np.zeros(npc, dtype=np.int64)        # erased GT super-nucleus components
    target_px = np.zeros(npc, dtype=np.int64)       # union support px of erased comps (pre-rim)
    realized_super = np.zeros(npc, dtype=np.int64)  # realized Lane super-nucleus components (8-conn)
    for k in range(npc):
        g = gts[k]
        for c in range(N_CLASSES):
            cls_gt[k, c] = int((g == c).sum())
        cls_base[k] = _per_class_flip_counts(base_realized[k], g)
        lane_gt = g == LANE_CLASS
        tmask, ns, ne, tpx = erased_super_nucleus_mask(lane_gt, base_realized[k], structure)
        n_super[k] = ns
        n_erased[k] = ne
        target_px[k] = tpx
        base_flip = base_realized[k] != g
        base_flip_T[k] = int((base_flip & tmask).sum())
        base_flip_off[k] = int((base_flip & (~tmask)).sum())
        # realized Lane super-nucleus betti0 (8-conn label of realized==Lane, keep >5px)
        rlab, rn = ndi.label(base_realized[k] == LANE_CLASS, structure=structure)
        if rn:
            rsizes = np.bincount(rlab.ravel(), minlength=rn + 1)
            realized_super[k] = int((rsizes[1:] > NUCLEUS_PX).sum())

    return {
        "idxs": np.asarray(idxs, dtype=np.int64),
        "base_dseg": base_dseg,
        "cls_gt": cls_gt, "cls_base": cls_base,
        "base_flip_T": base_flip_T, "base_flip_off": base_flip_off,
        "n_super": n_super, "n_erased": n_erased, "target_px": target_px,
        "realized_super": realized_super,
    }


# ---------------------------------------------------------------- aggregate/verdict
def aggregate(out_dir: Path, num_pairs: int, ckpt: str, ckpt_sha: str) -> dict:
    accs = []
    c0 = 0
    while c0 < num_pairs:
        cand = sorted(out_dir.glob(f"chunk_{c0:04d}_*.npz"))
        if not cand:
            raise RuntimeError(f"missing chunk starting at {c0}")
        z = np.load(cand[0])
        accs.append({k: z[k] for k in z.files})
        c0 = int(z["idxs"][-1]) + 1
    cat = {k: np.concatenate([a[k] for a in accs], 0) for k in accs[0]}

    total_px = float(num_pairs * SEG_H * SEG_W)
    sum_base_T = float(cat["base_flip_T"].sum())
    P = 100.0 * sum_base_T / total_px  # erased super-nucleus Lane pool remaining (S-units) — QA92 L353

    base_total = float(cat["base_dseg"].sum())
    base_dseg_mean = base_total / num_pairs
    base_per_class = [100.0 * float(cat["cls_base"][:, c].sum()) / total_px
                      for c in range(N_CLASSES)]

    return {
        "schema": "ddm_xp1_exact_p.v1",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False, "research_only": True,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": num_pairs, "ckpt": ckpt, "ckpt_sha256": ckpt_sha,
        "class_order": CLASS_ORDER, "lane_class_index": LANE_CLASS,
        "nucleus_px_threshold": NUCLEUS_PX, "connectivity": "8 (structure 3x3 ones)",
        "method_note": "QA92 base-pass reuse (erased_super_nucleus_mask + P L353) on ep641 endpoint",
        # base state (validation: base_dseg_mean should == manifest n600_d_seg 0.00426407708)
        "base_dseg_mean": round(base_dseg_mean, 9),
        "base_per_class_S_units": [round(x, 5) for x in base_per_class],
        "base_lane_S_units": round(base_per_class[LANE_CLASS], 5),
        # THE owed answer: exact P
        "P_pool_S_units": round(P, 5),
        "n_erased_super_nucleus_total": int(cat["n_erased"].sum()),
        "n_super_nucleus_total": int(cat["n_super"].sum()),
        "realized_super_nucleus_total": int(cat["realized_super"].sum()),
        "target_px_total": int(cat["target_px"].sum()),
        "base_flip_off_total": int(cat["base_flip_off"].sum()),
    }


# --------------------------------------------------------------------- driver
def cmd_run(args) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity (QA92 identical)
    ckpt_sha = _sha256_file(args.checkpoint)
    print(f"[xp1] ckpt {Path(args.checkpoint).name} sha256 {ckpt_sha[:16]}...  "
          f"n_pairs {args.num_pairs}", flush=True)

    lstars = load_lstars(Path(DEFAULT_GT_CACHE), args.num_pairs)
    print(f"[xp1] gt loaded lstars {lstars.shape}", flush=True)
    _cfg, model = load_module(args.checkpoint)
    from tac.boundary_math.seg_core import load_real_segnet
    seg = load_real_segnet("cpu")

    t0 = time.time()
    for c0 in range(0, args.num_pairs, args.chunk):
        c1 = min(c0 + args.chunk, args.num_pairs)
        cpath = out_dir / f"chunk_{c0:04d}_{c1:04d}.npz"
        if cpath.exists():
            print(f"[skip] {cpath.name}", flush=True)
            continue
        acc = run_chunk(args, c0, c1, model, seg, lstars, structure)
        _atomic_save_npz(cpath, acc)
        print(f"[{c0}:{c1}] base_dseg {float(acc['base_dseg'].mean()):.7f}  "
              f"erased_super {int(acc['n_erased'].sum())}  "
              f"realized_super {int(acc['realized_super'].sum())}  "
              f"{time.time()-t0:.0f}s", flush=True)

    verdict = aggregate(out_dir, args.num_pairs, str(args.checkpoint), ckpt_sha)
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_bytes(out_dir / "xp1_verdict.json",
                        (json.dumps(verdict, indent=1, sort_keys=True) + "\n").encode())
    print("\n=== XP1 VERDICT ===")
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0


def cmd_aggregate(args) -> int:
    ckpt_sha = _sha256_file(args.checkpoint)
    verdict = aggregate(args.out_dir, args.num_pairs, str(args.checkpoint), ckpt_sha)
    _atomic_write_bytes(args.out_dir / "xp1_verdict.json",
                        (json.dumps(verdict, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "aggregate"])
    ap.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--seg-batch", type=int, default=12)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
