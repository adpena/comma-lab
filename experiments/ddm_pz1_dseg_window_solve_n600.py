#!/usr/bin/env python
"""ddm_pz1 — d_seg under the ddm_ll1 window solve, n600, frozen SegNet authority.

WHY THIS IS CHEAP AND EXACT
---------------------------
``SegNet.preprocess_input`` (``upstream/modules.py:107-109``) is::

    x = x[:, -1, ...]                 # LAST frame only  => frame_1
    return F.interpolate(x, (384, 512), 'bilinear')

so ``d_seg`` depends on **frame_1 alone**.  The window solve acts directly on
frame_1, and the GT argmax is already cached for all 600 pairs.  The seg side of
the trade is therefore an n600 measurement, not a bracket -- no warp, no GT
decode, no evaluator slot.

This replaces ``sg2``/``ra1``'s 1.98-3.27%-of-gap RANGE (an n=3 smoke against
IDEAL targets, transferred and then rms-extrapolated) with the real number on the
real base, and it directly answers ``ra1`` §5.4's pre-registered seg falsifier:

    retire the rung if  Delta d_seg * 100 > -0.010 S

POSITIVE CONTROL (the whole point of doing it this way)
-------------------------------------------------------
``d_seg_base * 100`` must reproduce the live base's KNOWN n600 seg term
(cx1: 0.4311790).  If it does, every link -- packet parse, render, camera raster,
D, frozen SegNet, GT argmax cache -- is validated end to end on the same data the
verdict is drawn from.  If it does not, no seg number here is admissible.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

#: cx1 n600 real-evaluator seg term (= 100 * d_seg)
CX1_SEG_TERM: float = 0.4311790
_LSTARS = Path("experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz")


def _install_paths(sub: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for p in (str(sub), str(root / "upstream"), str(root / "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_segnet():
    import modules
    import torch
    from safetensors.torch import load_file

    root = Path(__file__).resolve().parents[1]
    if Path(modules.__file__).resolve() != (root / "upstream" / "modules.py").resolve():
        raise RuntimeError("imported non-custodied upstream modules.py")
    torch.set_num_threads(6)
    seg = modules.SegNet().eval().cpu()
    seg.load_state_dict(load_file(str(root / "upstream/models/segnet.safetensors"), device="cpu"))
    for p in seg.parameters():
        p.requires_grad = False
    return seg


def _argmax_of(seg, f1_u8: np.ndarray) -> np.ndarray:
    """Frozen SegNet argmax through the canonical preprocess for one frame_1."""
    import torch

    # SegNet.preprocess_input takes (b,t,c,h,w) and slices the LAST t.  Feed a
    # single-frame sequence so the slice is a no-op and the operator is exact.
    x = torch.from_numpy(f1_u8[None, None]).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        out = seg(seg.preprocess_input(x))
    return out.argmax(dim=1)[0].numpy().astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", required=True, type=Path)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    _install_paths(sub)

    from inflate_runner import Decoder  # noqa: E402
    from tac.optimization import ddm_tr1_runtime as repo_tr1  # noqa: E402

    seg = _load_segnet()
    lstars = np.load(_LSTARS)["lstars"]
    dec = Decoder(sub / "archive")
    n = int(dec.n_pairs)
    idx = np.unique(np.linspace(0, n - 1, args.pairs).round().astype(int))
    px = float(lstars.shape[1] * lstars.shape[2])
    print(f"[pz1] base={sub.name} pairs={len(idx)} lstars={lstars.shape} px/frame={px:.0f}",
          flush=True)

    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        i = int(i)
        gt = lstars[i].astype(np.uint8)
        f1b = repo_tr1.render_frame1_camera_uint8(dec.packet, i, window_solve=False)
        f1s = repo_tr1.render_frame1_camera_uint8(dec.packet, i, window_solve=True)
        ab = _argmax_of(seg, f1b)
        as_ = _argmax_of(seg, f1s)
        db = float(np.mean(ab != gt))
        ds = float(np.mean(as_ != gt))
        rows.append({"pair": i, "d_seg_base": db, "d_seg_solved": ds,
                     "delta": ds - db,
                     "flips_changed": int(np.count_nonzero(ab != as_))})
        if k % 25 == 0 or k == len(idx) - 1:
            bm = np.mean([r["d_seg_base"] for r in rows])
            sm = np.mean([r["d_seg_solved"] for r in rows])
            print(f"[pz1] {k + 1:3d}/{len(idx)} pair {i:4d} | running d_seg "
                  f"base {bm:.8f} solved {sm:.8f} | dS {100 * (sm - bm):+.5f} "
                  f"| {time.time() - t0:6.1f}s", flush=True)

    base = np.array([r["d_seg_base"] for r in rows])
    sol = np.array([r["d_seg_solved"] for r in rows])
    seg_term_base = 100.0 * float(base.mean())
    seg_term_solved = 100.0 * float(sol.mean())
    control_ratio = seg_term_base / CX1_SEG_TERM

    summary = {
        "base": sub.name,
        "n_pairs": len(idx),
        "positive_control": {
            "measured_seg_term_base": seg_term_base,
            "known_n600_seg_term": CX1_SEG_TERM,
            "ratio": control_ratio,
            "passes": bool(0.97 <= control_ratio <= 1.03),
            "note": ("must reproduce the live base's known n600 seg term; if it does "
                     "not, no seg number in this file is admissible"),
        },
        "d_seg_base_mean": float(base.mean()),
        "d_seg_solved_mean": float(sol.mean()),
        "seg_term_base": seg_term_base,
        "seg_term_solved": seg_term_solved,
        "delta_S_seg": seg_term_solved - seg_term_base,
        "frac_pairs_improved": float(np.mean(sol < base)),
        "frac_pairs_worse": float(np.mean(sol > base)),
        "ra1_seg_falsifier": {
            "threshold": "retire if delta_S_seg > -0.010",
            "delta_S_seg": seg_term_solved - seg_term_base,
            "retired": bool((seg_term_solved - seg_term_base) > -0.010),
        },
        "rows": rows,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print("\n[pz1] ===== d_seg n600 =====")
    print(f"[pz1] POSITIVE CONTROL seg term base {seg_term_base:.7f} vs known "
          f"{CX1_SEG_TERM:.7f} = {control_ratio:.5f}x  "
          f"PASS={summary['positive_control']['passes']}")
    print(f"[pz1] d_seg  base {base.mean():.8f} -> solved {sol.mean():.8f}")
    print(f"[pz1] seg term {seg_term_base:.7f} -> {seg_term_solved:.7f}")
    print(f"[pz1] Delta S(seg) = {summary['delta_S_seg']:+.6f}")
    print(f"[pz1] pairs improved {100 * summary['frac_pairs_improved']:.1f}% / "
          f"worse {100 * summary['frac_pairs_worse']:.1f}%")
    print(f"[pz1] ra1 seg falsifier retired={summary['ra1_seg_falsifier']['retired']}")
    print(f"[pz1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
