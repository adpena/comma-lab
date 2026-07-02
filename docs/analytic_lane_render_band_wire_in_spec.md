# Analytic-lane render-band — WIRE-IN SPEC (FEED-dv; tasks #203/#213/#215)

**Component:** `src/tac/boundary_math/analytic_lane_render_band.py` (the non-naive
analytic-lane render-band: AA-SDF coverage × range-dependent dash gate ×
witness-uncertainty mask). **The PARENT sequences this wire-in;** this doc is the
exact contract. Companion equation `analytic_lane_render_band_fp_reduction_v1`;
DSL lever `tac.witness_dsl.AnalyticLaneRenderBand`; DAG block FEED-dv.

**PRIMARY TARGET = the LIVE levelset trainer** `experiments/train_levelset_witness_realized_through_R_mlx.py`
(the softmax-of-SDF K=5 witness; 208K). It IMPORTS `make_loss_fn` + `render_through_R_mlx`
from the base `train_witness_realized_through_R_mlx.py` (150K) — so the base's `compose_fn`
hook (`render_through_R_mlx:355`, `compose_fn(rgb_nhwc, code_idx) -> rgb_nhwc`) is the
wire-in point, and the levelset trainer already routes every realized render through
`_render_R` (line 1174/1201) → `make_loss_fn(..., render_fn=_render_R)` (line 1216/1220).
Default `compose_fn=None` is byte-identical → the wire-in is ADDITIVE.

**COMPOSE with the inline levers, do NOT duplicate.** The levelset trainer already carries
default-off inline levers: `self-orient` directional basis, `--lane-edge-*` (LEVER-3),
`--margin-saliency-*` (LEVER-4, the #141 margin lever), `--lane-thin-*` (LEVER-B),
`--eikonal-*`, `--length-weight`. These are LOSS-SIDE (they weight the SegNet realized
top1-top2 margin on the COMPOSED render, POST-R; trainer line ~1323–1362). My band is
RENDER-SIDE (analytic class-1 authority composited PRE-R). They are COMPLEMENTARY: the band
supplies the lane structurally where the witness erases it; LEVER-4 then applies margin
pressure on the composed render. My uncertainty gate RIDES THE SAME #141 top1-top2 margin
QUANTITY as LEVER-4 — computed on the witness's PRE-R softmax at compose time (the levelset
witness already computes `soft`), not a new heuristic.

## 0. Measured readiness ($0 n600 through R, frozen CPU-torch SegNet argmax)

`tools/levelset_analytic_lane_band_dseg_n600.py` (l7-best levelset ckpt, all 600 pairs,
`[macOS-CPU advisory] NON-PROMOTABLE`). Ablation ladder — see
`reports/levelset_analytic_lane_band_dseg_n600_20260701.json` for the authoritative row:

| condition | lever added | Δ d_seg vs witness-alone |
|---|---|---|
| `c1_default` | — (witness alone, sanity ≈ 0.00333) | 0.0 |
| `c_naive` | full-coverage band, no gate | **+0.00082 (HURT)** — reproduces sizing c3 |
| `c_range` | + range-dependent dash gate (#215) | ~+0.00082 |
| `c_full_gt` | + uncertainty from FROZEN GT SegNet margin | intermediate |
| `c_full_wit` | + uncertainty from WITNESS softmax margin | **~break-even (FP killed ~98%)** |

**VERDICT (honest):** the three levers DO kill the FP that made the naive band hurt
(+0.00082 → ~break-even). But POST-HOC on a render-overfit ckpt the band is NEUTRAL,
not net-negative — the witness overfit its exact render pipeline (sizing VERDICT). **The
NET-NEGATIVE d_seg win is realized by TRAINING WITH the band active** (the witness
re-adapts its boundaries and reallocates capacity off the lane long-tail). That is what
THIS wire-in enables. The component is the readiness-passing render-band; the exact-row
win comes from the trainer engaging it.

## 1. Trainer argparse additions (the `--lane-band-*` flags)

Add to the trainer's argparse (both trainers, same names — the DSL lever emits these):

```python
p.add_argument("--lane-render-band", action="store_true",
               help="FEED-dv: composite the analytic-lane render-band via compose_fn")
p.add_argument("--lane-band-softness", type=float, default=1.0,
               help="AA-SDF coverage ramp width (px) on the band lateral edge")
p.add_argument("--lane-band-dash-forward-max-m", type=float, default=55.0,
               help="#215 SegNet-Nyquist: dash-gate ONLY where forward < this (m); continuous beyond")
p.add_argument("--lane-band-uncertainty-source", type=str, default="witness",
               choices=["witness", "gt", "none"],
               help="uncertainty margin source for the FP-killer gate")
p.add_argument("--lane-band-tau", type=float, default=0.85,
               help="uncertainty threshold (witness margin = PROB [0,1]; gt margin = LOGIT ~[0,13])")
p.add_argument("--lane-band-eps", type=float, default=0.35, help="uncertainty ramp width")
p.add_argument("--lane-band-weight", type=float, default=1.0, help="band strength (curriculum ramp)")
p.add_argument("--lane-band-start-epoch", type=int, default=300, help="engage the band at this epoch")
```

## 2. Per-code prior precompute (ONCE, at trainer setup, from GT lstars)

The band geometry is a FIXED deterministic prior fit to the frozen GT SegNet class-1
mask (cached `GTData.lstars`). Precompute one `LaneBandPrior` per pair/code:

```python
from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior
# code_idx -> pair index: the trainer uses interleaved f0/f1 codes (2*pi, 2*pi+1);
# the lane is scored on frame1 -> key the prior by the SEG frame's code_idx.
lane_priors = {}
for pi in range(gt.n_pairs):
    prior = build_analytic_lane_band_prior(
        gt.lstars[pi], lane_cls=1, softness=args.lane_band_softness,
        dash_gate=True, dash_forward_max_m=args.lane_band_dash_forward_max_m)
    lane_priors[2 * pi + 1] = prior   # frame1 code_idx (the SegNet-scored frame)
    lane_priors[2 * pi]     = prior   # frame0 is seg-free but keep symmetric
```

Cost: ~0.34 ms/frame numpy (n600 ≈ 0.2 s total, once). NO GPU.

## 3. The witness margin (the uncertainty signal, rides #141)

`--lane-band-uncertainty-source witness` (recommended, the measured winner): the
softmax-of-SDF witness's OWN top1-top2 decision margin (PROB scale). Expose it from the
witness at compose time. The levelset witness (`train_levelset_witness...:470-475`) already
computes `phi -> soft`; add a margin accessor:

```python
# in the witness class (levelset), alongside the RGB forward:
def call_margin(self, coord_feats, code_idx):
    soft = mx.softmax(self.out_sdf(self._trunk(coord_feats, code_idx)) / self.softmax_temp, axis=-1)
    top2 = mx.topk(soft, 2, axis=-1)            # (...,2) — or sort; MLX topk values
    return top2[..., 0] - top2[..., 1]          # (P,) margin, reshape to (H,W)
```

`--lane-band-uncertainty-source gt` (cheaper, no live accessor): the FROZEN GT SegNet
top1-top2 margin — ALREADY precomputed in `GTData.margins` / the cache (`margins`),
the #141 quantity. Use LOGIT-scale `--lane-band-tau` (~2.0). Measured slightly weaker
than `witness` post-hoc but zero extra forward and is the boundary-annulus prior.

## 4. Build the compose_fn and pass it as `render_fn`

```python
import functools
from tac.boundary_math.analytic_lane_render_band import make_lane_band_compose_fn
from experiments.train_witness_realized_through_R_mlx import render_through_R_mlx  # or levelset sibling

# lane appearance = the witness's OWN per-pixel lane color (self-consistent, byte-free);
# for the levelset witness this is sigmoid(palette[1] + tex)*255 rendered per code.
def lane_rgb_provider(code_idx):
    return witness.render_lane_appearance(coord_feats, code_idx)  # (H,W,3) or (3,)

# margin provider per --lane-band-uncertainty-source
if args.lane_band_uncertainty_source == "witness":
    def margin_provider(code_idx):
        return witness.call_margin(coord_feats, code_idx).reshape(render_h, render_w)
elif args.lane_band_uncertainty_source == "gt":
    margin_provider = {c: gt.margins[c // 2] for c in lane_priors}
else:
    margin_provider = None

band_compose_fn = make_lane_band_compose_fn(
    lane_priors, lane_rgb_provider=lane_rgb_provider, margin_provider=margin_provider,
    tau=args.lane_band_tau, eps=args.lane_band_eps, weight=args.lane_band_weight, use_mlx=True)
```

**Wire into the levelset trainer's `_render_R` (line 1174/1201) — chain with the existing
`_compose_mx` if `residual_mode` is also on** (order: bulk residual first, THEN lane band on
top), else the band is the sole compose:

```python
# in the levelset trainer, extend the _render_R construction (near line 1174-1202):
_band_active = args.lane_render_band  # gate the epoch ramp inside band_compose_fn via --lane-band-start-epoch
if residual_mode or _band_active:
    def _compose_chain(rgb_nhwc, code_idx):
        if residual_mode:
            rgb_nhwc = _compose_mx(rgb_nhwc, code_idx)     # existing v2 bulk residual first
        if _band_active:
            rgb_nhwc = band_compose_fn(rgb_nhwc, code_idx)  # analytic lane band on top
        return rgb_nhwc
    def _render_R(witness, coord_feats, code_idx, rh, rw):  # noqa: F811
        return render_through_R_mlx(witness, coord_feats, code_idx, rh, rw, compose_fn=_compose_chain)
# make_loss_fn(..., render_fn=(_render_R if (residual_mode or _band_active) else None))  # line 1216/1220
```

The seg + pose loss (and LEVER-3/4/B, eikonal, length — which read the COMPOSED render) are
then realized on the band-composited witness through R — the d_seg gradient backprops into
the witness (coverage/u_mask are stop-grad constants; gradient flows through `rgb` and
`lane_rgb`, both witness-derived). The `--lane-band-start-epoch` ramp lives inside
`band_compose_fn` (multiply `weight` by the ramp; a no-op weight=0 leaves the render byte-identical).

## 5. Batch render path (`render_batch_through_R_mlx`, line ~364 — NO compose hook)

`render_batch_through_R_mlx(witness, coord_feats, code_indices, render_h, render_w)` has
NO compose hook. Add one, mirroring the per-frame path:

```python
def render_batch_through_R_mlx(witness, coord_feats, code_indices, render_h, render_w, compose_fn=None):
    rgb_flat = witness.call_batch(coord_feats, code_indices)      # (M, P_px, 3)
    rgb = mx.reshape(rgb_flat, (-1, render_h, render_w, 3))       # (M,H,W,3)
    if compose_fn is not None:
        # compose_fn must accept the batch: gather per-code coverage/lane/margin into
        # (M,H,W) / (M,H,W,3) stacks and call composite_lane_band_mlx once (vectorized).
        rgb = compose_fn(rgb, code_indices)                       # batched compose
    return apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=(SEG_H, SEG_W), ste_round=True)
```

The batched `compose_fn` variant: stack `coverage`/`u_mask` to `(M,H,W)` and `lane_rgb`
to `(M,H,W,3)` (or broadcast a constant), then ONE `composite_lane_band_mlx` call — the
64× MLX-GPU speedup is measured at exactly this batched shape.

## 6. COMPUTE facet (benchmark + Metal-kernel flag)

- **Hot path = the composite** (`composite_lane_band_mlx`): elementwise, `mx.compile`-friendly,
  BATCHED over M frames. Measured n600: numpy per-frame 842 ms → **MLX-GPU batched+compiled
  13.1 ms = 64.1×**, BIT-IDENTICAL to numpy (corr 1.0, maxabs 0.0). Composes cleanly with
  `TAC_MLX_CUSTOM_GROUPED_BACKWARD` (~17× grouped-backward) + `apply_contest_faithful_roundtrip_nhwc`
  (fused-R, bit-identical). Does NOT touch the launch-gate perf-env throughput path.
- **Metal-kernel candidate** = the AA-SDF coverage raster (`METAL_KERNEL_FLAG` in the module):
  numpy vectorized 0.34 ms/frame (precompute today; #203 makes it in-loop). Kernel signature:
  `aa_sdf_lane_coverage(u_center[L,H], hw[L,H], gate[L,H], col_grid[W], softness, out coverage[H,W])`
  — per (row v, col u): `s = hw - |col - u_center|; cov = max_l clip(s/soft+0.5,0,1)*gate`.
  MLX parity reference: `rasterize_lane_coverage_mlx`. The parent builds+wires per the #212 suite
  (sister of `metal_fused_r_operator` / `metal_grouped_conv_backward`).

## 7. Per-class decomposition (GR-unified-action) — compose, don't duplicate

Witness owns SMOOTH classes (Road/Undrivable/hood/MyCar; sisters
`road_horizon_component`, `hood_static_component`); LANE (class 1, the finest-scale
erasure tail) = THIS render-band (render-time authority). Sister lane legs (same
manifold, NOT duplicated): `lane_headstart` (init-side head-start), `lane_sdf_component`
(phi_1 SDF injection into `lever_b_levelset_generator`), `margin_saliency_map` (#141, the
uncertainty signal this band rides).

## 8. Risks / open items

- **Dash-fit robustness (MEASURED weakness):** the borrowed `lane_sdf_component._fit_dash`
  matched-filter can spuriously fit a dash period to a short/truncated lane (its forward
  window is 6–50 m). The range-dependent gate mitigates (far continuous), but the parent
  should consider tightening `_fit_dash` (require the lane to span ≥1.5 periods) OR set
  `--lane-band-dash-forward-max-m` conservatively. Real-GT n600 dash-fit quality is in the
  measurement's `analytic_band_fit.band_vs_gt_lane_recall_mean`.
- **Net-negative needs training-in:** post-hoc is ~break-even (measured). The parent must
  sequence a fine-tune with `--lane-render-band` active (start-epoch ~300, weight ramp)
  and re-measure d_seg through R; that is the exact-row candidate.
- **Lane appearance:** the measurement uses the witness's OWN lane color (self-consistent).
  A fixed bright color is a byte-free fallback if a live accessor is undesirable; measured
  head-to-head is a cheap follow-up.
