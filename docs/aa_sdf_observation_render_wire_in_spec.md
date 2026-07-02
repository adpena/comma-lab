# AA-SDF observation-map render — wire-in spec (task #220)

**Module:** `src/tac/boundary_math/aa_sdf_observation_render.py` (committed 03771a08a).
**Equation:** `aa_sdf_observation_footprint_render_dseg_v1`.
**Lever class:** representation, **~0-rate** (decode-time deterministic op; witness `archive.zip`
bytes UNCHANGED). Sister of `analytic_lane_render_band` (render-side) + the loss-side inline levers.

## What it is

The MEASURED #1 representation lever (committed gate `tools/levelset_gate_discriminators_n600.py`,
DAG FEED-ly): POINT-sampling the witness render grid ERASES finest-scale lane structure through the
contest R; FOOTPRINT-INTEGRATED (anti-aliased) rendering RECOVERS it. Two AA modes over the WHOLE
softmax-of-SDF partition, composited BEFORE R:

1. **SUPERSAMPLE → BOX** (ground-truth footprint integration): render the witness at `(ss*H, ss*W)`,
   AREA/box-downsample (integer ss ⇒ exact ss×ss block mean, bit-identical to `torch area`) to
   `(H, W)`. General; works on any render field. Cost: `ss²` × the witness forward.
2. **mip-NeRF IPE CONE** (analytical, ~0 extra compute; Barron 2103.13415): attenuate each curvelet
   Fourier column by `exp(-2π²(Bx²σx² + By²σy²))` (`σ` = per-pixel footprint std). Anti-aliases the
   BASIS with no supersampling. Analytical approximation to (1) — (1) is the authority.

The **render-grid** knob (`--render-h/--render-w`, default 384×512) is the base grid R sees:
the achievable-through-R floor drops as `(H,W)` → camera (874×1164). AA is complementary (recovers
the finest scale WITHIN a fixed grid).

## Public API (drop-in for the trainer render path)

```python
from tac.boundary_math.aa_sdf_observation_render import (
    build_supersampled_coords,      # coords at (ss*H, ss*W); ss=1 == _build_render_coords(H,W)
    box_downsample_mlx,             # (M, ss*H, ss*W, C) -> (M, H, W, C) block-mean (MLX)
    box_downsample_np,              # numpy authority twin
    ipe_footprint_sigma,            # (H, W, scale) -> (sx, sy)
    ipe_curvelet_attenuation,       # (B, sx, sy) -> per-column att (cols,)
    apply_ipe_attenuation,          # (curv_feats, att) -> attenuated [sin|cos] feats
    render_aa_batch_through_R_mlx,  # drop-in for render_batch_through_R_mlx (ss=1 byte-identical)
    render_aa_through_R_mlx,        # per-frame twin of render_through_R_mlx
)
```

## Trainer wire-in (SPEC — parent consolidates; do NOT hand-edit the launch-path trainer here)

Target: `experiments/train_LEVELSET_witness_realized_through_R_mlx.py`.

**New argparse flags (add near `--render-h/--render-w`, ~line 2424):**
```python
ap.add_argument("--render-aa", choices=["none", "supersample", "ipe"], default="none",
                help="AA observation-map render mode (default none = byte-identical point-sample).")
ap.add_argument("--aa-supersample", type=int, default=1,
                help="supersample factor ss for --render-aa supersample (render at ss*grid, box-down).")
ap.add_argument("--aa-ipe-footprint", type=float, default=1.0,
                help="footprint std scale for --render-aa ipe (1.0 = one-pixel box).")
```

**Coord/feature build (SUPERSAMPLE mode — where `coords_np`/`coord_feats_mx` are built, ~line 925/968):**
```python
if args.render_aa == "supersample" and args.aa_supersample > 1:
    ss = args.aa_supersample
    coords_np = build_supersampled_coords(render_h, render_w, ss)          # (ss²·P, 2)
    curv_feats_np = curvelet_feats(coords_np, B).astype(np.float32)         # fine-grid feats
    # self-orient dir feats (if used) recompute at the fine grid via the SAME public helper,
    # NN-upsampling the witness argmax to (ss*render_h, ss*render_w) before the tangent EDT.
elif args.render_aa == "ipe":
    att = ipe_curvelet_attenuation(B, *ipe_footprint_sigma(render_h, render_w, args.aa_ipe_footprint))
    curv_feats_np = apply_ipe_attenuation(curvelet_feats(coords_np, B), att).astype(np.float32)
coord_feats_mx = mx.array(curv_feats_np)
```

**Render call (both the train-loss and verdict render sites that call `render_batch_through_R_mlx`):**
```python
if args.render_aa == "supersample" and args.aa_supersample > 1:
    r = render_aa_batch_through_R_mlx(model, coord_feats_mx, code_indices,
                                      render_h, render_w, args.aa_supersample)
else:  # none | ipe (ipe changes feats only; render grid unchanged)
    r = render_batch_through_R_mlx(model, coord_feats_mx, code_indices, render_h, render_w)
```

**Byte-close / decode:** the IPE attenuation vector is a deterministic function of `(B, render_h,
render_w, footprint_scale)` — all already in the checkpoint config (`__bank_*`, `__render_hw`), so it
is FREE at inflate (rule 118). Supersample renders at a finer grid deterministically at decode (no
extra archive bytes). Add `__cfg_render_aa` / `__cfg_aa_supersample` / `__cfg_aa_ipe_footprint`
scalars to the checkpoint flat dict + the `levelset_byte_close_and_eval` (#202) render path so the
exact-eval decode uses the same AA mode.

## DSL gauge SPEC (parent adds to `tac.witness_dsl.gauge`; do NOT edit gauge.py here)

```
GaugeComponent.RENDER_AA
RenderAAGauge = {NONE, SUPERSAMPLE_2X, SUPERSAMPLE_3X, IPE}
  NONE           -> --render-aa none                              (byte-identical default)
  SUPERSAMPLE_2X -> --render-aa supersample --aa-supersample 2    (measured #1 lever)
  SUPERSAMPLE_3X -> --render-aa supersample --aa-supersample 3
  IPE            -> --render-aa ipe --aa-ipe-footprint 1.0        (analytical ~0-compute proxy)
build_render_aa_lever() -> RENDER_AA_TRAINER_FLAG_DEFAULTS (3 flags)
```
Composes with `RenderGrid.{384,512}` (the render-grid knob) and the AnalyticLaneRenderBand /
IslandProtection / TopologyLoss gauges. Fail-closed until the trainer flags land (never-invent-flags).

## Compute leg (co-equal facet)

The box-downsample hot path is MLX-first + `mx.compile`-friendly: n600-shaped batch (M=12, fine
768×1024) `box_downsample_mlx` ~8 ms/call vs numpy 210 ms (~26×); numpy is the bit-identical
authority. **#212 Metal-kernel candidate:** a FUSED `aa_render_through_R` kernel that folds
supersampled-render → box-downsample → bicubic-up-to-camera into one strided pass (avoids
materializing the `(M, ss·H, ss·W, 3)` intermediate). Signature:
`aa_box_downsample(rgb_fine[M,fh,fw,3], ss) -> rgb[M,fh/ss,fw/ss,3]`, `METAL_KERNEL_FLAG`.
Composes with `TAC_MLX_CUSTOM_GROUPED_BACKWARD` + `apply_contest_faithful_roundtrip_nhwc` (fused-R);
does NOT touch the launch-gate perf path.

## Verify

`tools/aa_sdf_observation_render_verify_n600.py` — $0 n600, frozen CPU-torch SegNet through the
ACTUAL contest R (never MPS), resumable. Real-frame (confound-free achievable-through-R) +
partition-proxy signals, point vs AA(ss=2), render-grid curve {192,256,384}. Reproduces the gate's
+0.38 lane-recall lift / floor-toward-0.00091. Output:
`reports/aa_sdf_observation_render_verify_n600_20260701.json`.

## HARD GATE

Pointer 0.19110. All AA numbers are `[macOS-CPU advisory]`/realized-through-R — NONE is a score until
a composed θ* byte-closed #202 exact row (CPU/CUDA, MPS never). CONTAINMENT: build + $0-CPU/local-MLX
only, no autonomous heavy-GPU launch.
