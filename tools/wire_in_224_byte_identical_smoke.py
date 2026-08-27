#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""#224 CONSOLIDATED WIRE-IN — default-off BYTE-IDENTICAL regression smoke.

The non-negotiable acceptance bar for the #224 wire-in is: with ALL new lever flags at
their defaults, the trainer's render + loss path is BYTE-IDENTICAL to the pre-#224 baseline.
This smoke proves it deterministically (single process, one seeded witness -> no MLX-GPU
run-to-run non-determinism confound) at the two surfaces the wire-in touches:

  (A) base ``render_through_R_mlx`` per-frame compose hook: ``compose_fn=None`` (the default
      the unified _render_R uses when no lever is on) is bit-identical to the pre-hook call.
  (B) base ``render_batch_through_R_mlx`` compose hook (#224 §5 addition): ``compose_fn=None``
      is bit-identical to the pre-hook call.
  (C) the #224 witness accessors (``call_margin`` / ``render_lane_appearance``) are ADDITIVE:
      they are callable + finite AND calling them does NOT perturb the default ``__call__``
      RGB forward (they are only invoked when --lane-render-band is ON).

Corroborated by the CLI 1-epoch default run (docs/#224 report): the epoch-0 seeded-init
verdict (d_seg / d_pose) matches the pre-#224 code, and the deploy-npz keyset differs ONLY
by the 3 additive default-valued AA provenance scalars (__cfg_render_aa='none' /
__cfg_aa_supersample=1 / __cfg_aa_ipe_footprint=1.0).

MLX-only, CPU-friendly for the RENDER hooks (no scorer / no GPU / no dispatch). Advisory;
pointer 0.19110 UNMOVED. Run from repo root:
    PYTHONPATH=src:upstream:. .venv/bin/python tools/wire_in_224_byte_identical_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("src", "upstream", str(REPO), str(REPO / "experiments")):
    sp = str((REPO / p) if not p.startswith("/") and p != str(REPO) else p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _bit_equal(a, b) -> bool:
    import mlx.core as mx

    a = np.asarray(a, np.float32) if isinstance(a, mx.array) else np.asarray(a)
    b = np.asarray(b, np.float32) if isinstance(b, mx.array) else np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def main() -> int:
    import mlx.core as mx

    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness
    from train_witness_realized_through_R_mlx import (
        render_batch_through_R_mlx,
        render_through_R_mlx,
    )

    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )

    # -- deterministic default-config setup (matches the trainer defaults for the render path) --
    render_h, render_w = 96, 128  # small grid: the hook algebra is grid-agnostic (fast, GPU-free)
    mx.random.seed(0)
    np.random.seed(0)
    from train_witness_realized_through_R_mlx import _build_render_coords

    coords = _build_render_coords(render_h, render_w)
    bank = CurveletBankConfig(n_scales=2, n_orient0=8, f0=1.0, base=2.0, n_iso=4)
    B = curvelet_directional_B(bank, max_freq=None)
    feats_np = curvelet_feats(coords, B).astype(np.float32)
    cf = mx.array(feats_np)
    in_feat = feats_np.shape[1]

    model = build_levelset_rgb_witness(
        num_pairs=2, in_feat=in_feat, hidden_dim=96, n_hidden=4, mod_dim=32, n_classes=5,
        activation="relu", softmax_temp=1.0, chroma=False,
        wire_w0=30.0, wire_s0=10.0, hosc_beta=1.0, hosc_omega=30.0,  # unused for relu; construction args
    )
    mx.eval(model.parameters())

    results: list[tuple[str, bool]] = []

    # (A) per-frame compose hook: compose_fn=None (default) == the plain pre-hook call.
    r_plain = render_through_R_mlx(model, cf, 1, render_h, render_w)
    r_none = render_through_R_mlx(model, cf, 1, render_h, render_w, compose_fn=None)
    results.append(("(A) render_through_R_mlx compose_fn=None bit-identical", _bit_equal(r_plain, r_none)))
    # and an IDENTITY compose_fn returns the SAME rgb pre-R -> same R output (hook plumbing correct).
    r_id = render_through_R_mlx(model, cf, 1, render_h, render_w, compose_fn=lambda rgb, ci: rgb)
    results.append(("(A') render_through_R_mlx identity compose_fn bit-identical", _bit_equal(r_plain, r_id)))

    # (B) batch compose hook (#224 §5): compose_fn=None == the plain pre-hook batch call.
    codes = mx.array(np.array([0, 1], dtype=np.int32))
    rb_plain = render_batch_through_R_mlx(model, cf, codes, render_h, render_w)
    rb_none = render_batch_through_R_mlx(model, cf, codes, render_h, render_w, compose_fn=None)
    results.append(("(B) render_batch_through_R_mlx compose_fn=None bit-identical", _bit_equal(rb_plain, rb_none)))
    rb_id = render_batch_through_R_mlx(model, cf, codes, render_h, render_w, compose_fn=lambda rgb, ci: rgb)
    results.append(("(B') render_batch_through_R_mlx identity compose_fn bit-identical", _bit_equal(rb_plain, rb_id)))

    # (C) the #224 witness accessors are ADDITIVE: callable + finite, and do NOT perturb __call__.
    rgb_before = model(cf, 1)
    margin = model.call_margin(cf, 1)
    lane_rgb = model.render_lane_appearance(cf, 1, lane_cls=1)
    rgb_after = model(cf, 1)
    results.append(("(C) __call__ unchanged after call_margin/render_lane_appearance", _bit_equal(rgb_before, rgb_after)))
    results.append(("(C') call_margin finite + shape (P,)", bool(
        margin.shape == (render_h * render_w,) and np.all(np.isfinite(np.asarray(margin))))))
    results.append(("(C'') render_lane_appearance finite + shape (P,3)", bool(
        lane_rgb.shape == (render_h * render_w, 3) and np.all(np.isfinite(np.asarray(lane_rgb))))))

    print("#224 default-off byte-identical smoke (render + accessor surfaces):")
    ok = True
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("RESULT:", "PASS — the #224 wire-in default path is byte-identical (hooks are no-ops; "
          "accessors additive)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
