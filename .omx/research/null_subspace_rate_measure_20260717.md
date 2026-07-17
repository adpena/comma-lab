# #519 STAGE-1 — score-invariant null subspaces in the frozen donor witness: magnitudes + rate cost (MEASURED)

Date: 2026-07-17 · Agent: p0_519_null_measure (Fable subagent) · $0, CPU, 1-thread, read-only on checkpoints
Axis: **[macOS-CPU advisory] NON-PROMOTABLE** — no score claims; pointer UNMOVED.

Donor: `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`
(mod32cap ep650 EMA BEST; hosc β=2.949, softmax_temp=0.3098, chroma=1, self_orient=1, render 384×512).

## STORES CONSULTED (proactive recall)

- `tools/graph_memory_recall.py "gauge identity ker(A) blind complement resize adjoint"` →
  FEED-blindcoord-401, FEED-scorerblind, `resize_exploit_flip_fix_frontier_v1`, FEED-v9-cgauge,
  FEED-resize-exploit (#391).
- `.omx/research/bregman_all_surfaces_504_DAG_FEED_20260715.md` (#504 Bregman gauge notes: CGauge
  categorical-Bregman Hessian, `GAUGE_IDENTITY_VERIFIED_NOT_MODEL_FACTORIZED`).
- `src/tac/through_r/blind_coordinate.py` (#391/#401 exact resize-kernel machinery: 22.6969%
  camera px exactly zero-weight; SCOPE note — pure-generator archive stores no camera px, direct
  blind saving 0).
- `tools/levelset_byte_close_and_eval.py` + `src/tac/boundary_math/lever_b_levelset_generator.py`
  (`quantize_levelset_blob`, `_int8_symmetric` = per-tensor symmetric int8, scale = max|a|/127;
  REUSED, not reimplemented).
- Established (not re-derived): SegNet head exact rank-4 linear (`segnet_head_rank4_linear_flipdist_v1`);
  shared resize A_seg ≡ A_pose → (384,512) (`frozen_scorer_exact_factorization_20260715.md`).

## The witness head form (derived first, as tasked)

The LVLS1 shared-head witness renders `RGB = sigmoid( softmax(phi/T) @ palette + tex ) * 255`
with `phi = out_sdf(h)` a plain K=5 linear head. Therefore:

- **Gauge 1 (head class-mean).** softmax is exactly invariant under `phi -> phi + c(p)·1`, so the
  class-constant component of `out_sdf.{weight,bias}` (`W = 1⊗w̄ + W⊥`) is a true gauge orbit of
  the WHOLE render (not just argmax; the self-orient fixed point consumes argmax(phi) and is also
  invariant). Exact in fp32 algebra; only int8 quantization can break it.
- **Gauge 2 (palette channel-mean ↔ out_tex.bias).** softmax rows sum to 1, so
  `palette -= 1⊗v`, `out_tex.bias += v` is exactly render-invariant for any per-channel v.
- **Blind (ker A).** Both scorers share the bilinear (874,1164)→(384,512) resize A (separable,
  align_corners=False, antialias=False). Every element of range(Aᵀ) is exactly zero on the 22.70%
  blind camera rows/cols; ker(A) = 80.674% of camera-space dimensions. ker(A) lives in IMAGE
  space: it does NOT pull back to a weight-space linear subspace through the nonlinear render →
  **no exact weight-space projection exists for it** (honest residual, leg 2b is linearized).

## Commands (exact)

```
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python -u tools/null_subspace_rate_measure.py \
  --ckpt-dir experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z \
  --npz-name levelset_witness_ema_BEST.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --pairs 32 --blind-frame-pairs 8 --jacobian-pairs 2 \
  --out reports/null_subspace_rate_measure_519_n32.json
```

## Leg 1 — gauge magnitudes (MEASURED, fp64 on the fp32 checkpoint tensors)

| tensor | fro-norm | gauge component | fraction of norm | fraction of energy |
|---|---|---|---|---|
| out_sdf.weight (5,96) | 16.7630 | 8.7765 | **52.36%** | 27.41% |
| out_sdf.bias (5,) | 2.1809 | 1.5122 | **69.34%** | 48.07% |
| palette (5,3) | 8.2371 | 8.0120 | **97.27%** | 94.61% |

Palette channel-mean = (−1.892, −2.085, −2.217) — the palette is almost entirely its channel-mean
(the night-video darkness), the classic gauge-carried DC.

## Leg 3 — rate delta through the REAL byte-close packing (MEASURED; brotli q11 on int8, the only admissible rate number)

`build_levelset_blob` (LVLS1: manifest + base int8 brotli + code int8 brotli), orig total = **84,126 B**:

| variant | total 0.bin B | Δ vs orig | base brotli B |
|---|---|---|---|
| orig | 84,126 | 0 | 61,842 |
| gauge_proj (head class-mean removed) | 84,137 | **+11** | 61,853 |
| palette_canon (palette mean → out_tex.bias) | 84,120 | **−6** | 61,838 |
| both_proj | 84,137 | **+11** | 61,855 |

**Rate verdict: NEUTRAL (±11 B on 84,126 B = ±0.013%).** Mechanism (derived, then verified by the
measured deltas): dense fixed-shape int8 coding prices ELEMENTS, not NORM — removing a rank-1/DC
component changes no element count, so the null components are (near-)rate-free to store. The gauge
is NOT a rate lever on this grammar.

## Derived corollary — the gauge is a PRECISION lever, not a rate lever (MEASURED)

Per-tensor scale = max|a|/127, and the gauge component inflates max|W|:

- out_sdf.weight scale 0.028416 → 0.022088 after projection (**22.3% finer** at identical bytes);
  int8 error on the score-relevant deviation component `dev(Ŵ) − W⊥`: 0.15764 → 0.11819
  (**25.0% lower**).
- out_sdf.bias dev err 0.011207 → 0.005203; palette dev err 0.021094 → 0.006351 (3.3× lower);
  cost: out_tex.bias abs err 0.000473 → 0.008938 (it now carries the palette mean at coarser scale).

So the same 84 KB buys a strictly more precise W⊥ when the gauge is projected — whether that moves
d_seg through the real decode is leg 4.

## Leg 2 — blind (ker A) energy (MEASURED)

Projector self-test (exact): max |A(X−P X)| = **1.65e-15** · seen-space element blind fraction =
**−2.2e-16** · blind-row-supported image blind fraction = **1.0** · exact-zero-weight rows/cols =
**106 / 140** (matches `blind_coordinate.py` 106/874 rows, 140/1164 cols → 230,904 px = 22.6969%).
ker(A) dimension = 80.674% of camera space. (numpy/Accelerate emits spurious FPE RuntimeWarnings
inside f64 matmul on this host; the self-test proves the values exact.)

- **(2a) decoded output frames (16 frames = 8 pairs × f0,f1 × 3 channels, 48 fields):** energy
  fraction of the byte-closed decoded uint8 camera frames lying in ker(A):
  **raw 52.42%** [52.35, 52.58] · **mean-removed 52.88%** [52.59, 53.18]. Over HALF the rendered
  camera-frame energy is structurally invisible to both scorers — remarkably stable across frames.
- **(2b) output-layer weight energy → ker(A) via the Jacobian adjoint (linearized past uint8;
  2 pairs {0, 16}):** weight-energy-weighted blind fraction of the pushforward
  (σ′·(∂pre/∂W_dir)·h_j → bicubic-up U → projected against ker(A)):
  **out_tex 52.76%** (pairs agree to 4e-5) · **out_sdf 50.13%** (per-pair 45.2% / 55.0% —
  partition directions vary more with scene content). Unweighted means: 52.80% / 50.79%.
  ≈ half of every marginal output-layer weight-direction's rendered effect lands in ker(A).
- **Rate lever: 0 bytes by structure** — the witness blob stores no camera-resolution section, so
  ker(A) has nothing to delete here (matches `blind_coordinate.py` SCOPE: the 22.7%/20.55% saving
  applies only to camera-res payload sections). The blind numbers are a CAPACITY/WASTE observation
  (render+quantization effort spent where no scorer looks), not a rate projection.

## Leg 4 — score invariance through the REAL decode (frozen CPU-torch scorer; [macOS-CPU advisory])

Frame bit-identity (n32, 64 frames/variant): NO variant decodes bit-identically — int8
re-quantization under the changed scale re-rounds every tensor (mean 2.0-4.1e5 changed subpixels
of 3.05M/frame, ±1-level shifts). **palette_canon keeps the render argmax identical on 32/32
frames** (palette/out_tex.bias never feed phi — the partition is structurally untouched);
gauge_proj changes the render argmax on 32/32 (scale change re-rounds W_perp). The exact fp32
invariance does NOT survive the int8 re-pack verbatim; the measured question is the scorer delta.

Frozen CPU-torch scorer, first 32 pairs (contiguous), gt_n600 cache lstars/gt_poses:

| variant | d_seg | Δd_seg vs orig | d_pose | Δd_pose | Δbytes |
|---|---|---|---|---|---|
| orig | 0.00305780 | — | 143.658 | — | 0 |
| gauge_proj | 0.00305303 | **−4.77e-6** | 143.616 | −0.042 | +11 |
| palette_canon | 0.00305144 | **−6.36e-6** | 143.418 | −0.240 | −6 |
| both_proj | 0.00304508 | **−1.27e-5** | 143.403 | −0.256 | +11 |

(d_pose ≈ 143.7 is the pose-BLIND-by-design baseline — this donor trained with w_pose=0; the
comparison across variants is still valid. Δd_seg −1.27e-5 ≈ −0.42% of d_seg ≈ −0.0013 score
units at the 100× weight — small, favorable, and NOT established at n600.)

**n600 extension gate (per the task charter): NOT met** — the gate required |Δd_seg| < 1e-6 AND
bytes drop; every variant has |Δd_seg| ≥ 4.8e-6 (and the byte deltas are ±11 B noise). These are
measured TRADEOFF rows, not "free" invariance; no n600 run fired.

## Verdicts (verdict-scope: this checkpoint, this int8+brotli grammar — INSTANCE level)

| component | rate | score (n32 advisory) | verdict |
|---|---|---|---|
| Gauge 1 (out_sdf class-mean; 52.4% of head norm) | +11 B (NEUTRAL) | Δd_seg −4.8e-6 (slightly favorable) | **NEUTRAL on rate; small measured precision GAIN — a PRECISION lever, not a rate lever** |
| Gauge 2 (palette channel-mean; 97.3% of palette norm) | −6 B (NEUTRAL) | Δd_seg −6.4e-6, argmax-preserving | **NEUTRAL on rate; small measured gain; structurally safest (partition untouched)** |
| Blind ker(A) | 0 B by structure | n/a (no weight-space projection exists) | **COSTS capacity, not rate: ~52% of rendered output energy + ~50-53% of marginal output-layer weight effect is scorer-invisible** |

The mechanism law (derived, then confirmed by the measured rows): dense fixed-shape int8+brotli
coding prices ELEMENTS, not norm — projecting a score-invariant component out changes no element
count, so its rate cost is ~0; its REAL cost is quantization precision (the gauge inflates
max|W| → 22.3% coarser scale → 25.0% higher int8 error on the score-relevant deviation W_perp).
The byte-budget-free win is to canonicalize the gauge away BEFORE quantization.
`# FORMALIZATION_PENDING: single-checkpoint (mod32cap ep650) + n32 advisory scorer subset —
register as a canonical equation only after an n600 confirm and/or a second vehicle reproduces
the sign; the derived mechanism is general but the measured deltas are INSTANCE-scope.`

## Residuals (honest)

- ker(A) has no exact weight-space projection (nonlinear render); leg 2b is first-order.
- so_tau/so_iters are not persisted by the trainer; decode used the tool defaults (tau=4, iters=4)
  with the PERSISTED freq_across=32 / freq_along=8 — identical for all variants, so Δ-comparisons
  are unconfounded.
- Leg 4 subset is the first 32 pairs (contiguous), not strided; n600 is the full-evidence tier.
