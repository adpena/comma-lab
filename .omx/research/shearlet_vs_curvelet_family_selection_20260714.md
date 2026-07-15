# #502 shearlet half + curvelet-vs-shearlet family selection (MEASURED, $0)

**Date:** 2026-07-14 · **Scope:** completes task #502 (the compact-shearlet half) +
the directional-frame family-selection capacity measurement.
**Authority tag:** `[macOS advisory / not score authority]`. Pointer UNMOVED
(0.19108 submittable / 0.18804 borrowed bank). This is a linear-dictionary N-term
UPPER BOUND, NOT a d_seg row. The realized through-R d_seg is OWED (operator-GO /
CONTAINMENT).

## What landed
- `src/tac/boundary_math/compact_shearlet_frame.py` — GENUINELY compact cone-adapted
  shearlet frame: parabolic scaling A_j + SHEARING S_k + translation, two cones
  (horizontal + vertical anchor). numpy fp64 authority + MLX mirror (parity
  5.96e-8 ≤ 1e-4). Sister of the curvelet half `windowed_curvelet_frame.py`
  (both parabolic; they differ ONLY in ROTATION vs SHEAR steering).
- `src/tac/boundary_math/tests/test_compact_shearlet_frame.py` — 14 tests incl. the
  two swap-tests + the #351 rotation-swap flip test.
- `experiments/shearlet_vs_curvelet_vs_fourier_capacity_probe.py` — 3-way OMP
  N-term capacity probe reusing the curvelet probe's harness verbatim.

## The shearlet certificate (NO-FAKE, catalog #351)
`shearlet_certificate(CompactShearletConfig())` PASSES with two GENUINE gates a
rotation/Fourier basis structurally FAILS:

1. **Localization** (Fourier fails): shearlet paired-envelope span **1.0**,
   Fourier span **1.5e-7** (constant envelope), ratio > 1e5; energy
   concentration@10% **0.91**.
2. **Shear-selectivity** (rotation fails): a shear S_k FIXES the anchor line
   pointwise (`S_k·(x,0)=(x,0)`), so shear atoms sampled ALONG the anchor axis are
   invariant — `shear_anchor_dispersion` **1.08e-34** (~0). A ROTATION-steered
   family covering the SAME normal directions is NOT anchor-invariant —
   `rotation_anchor_dispersion` **0.006**, discrimination ratio **inf**. Plus
   integer-lattice preservation (S_1 integer/det-1 maps Z²; matched rotation does
   not) and parabolic-scaling monotonicity.

**Swap-proof (the #351-tight guard):** the shear-selectivity gate reads the REAL
`compact_shearlet_feats` forward. Monkeypatching the internal `_atom_xi_eta` to a
ROTATION flips `passes` True→**False** (shear_disp jumps 1e-34→0.006, disc_ratio
inf→1.00). A rotation/Fourier basis wearing a shearlet label CANNOT falsely pass
(covered by `test_rotation_swap_flips_certificate_to_false`).

## The MEASURED head-to-head (family selection)
Real cached frozen-SegNet boundary-annulus patches (32×32), OMP N-term (not
thresholding), matched **160-col** dictionaries for curvelet & shearlet (both
< Fourier 236), BOTH at finer-orientation OPTIMAL FORM, seeds 0/1/2, gt_n600 + gt_n96.

MEAN rel-error at fixed coefficient budget K (lower = better):

| cache | K | FOURIER | CURVELET | SHEARLET | best |
|---|---|---|---|---|---|
| n600 | 8  | 0.1289 | **0.1132** | 0.1207 | C |
| n600 | 16 | 0.0831 | **0.0677** | 0.0719 | C |
| n600 | 24 | 0.0697 | **0.0536** | 0.0565 | C |
| n96  | 8  | 0.1457 | **0.1201** | 0.1281 | C |
| n96  | 16 | 0.0978 | **0.0722** | 0.0773 | C |
| n96  | 24 | 0.0829 | **0.0569** | 0.0608 | C |

Coefficient budget to reach rel-err ≤ 0.10 (MEAN): n600 FOURIER K=12, CURVELET
K=10, SHEARLET K=10 (K_F/K_C=1.20×, K_C/K_S≈1.00 but curvelet strictly lower
rel-err at every K); n96 CURVELET/SHEARLET both edge Fourier ~1.20×, curvelet
strictly ahead at every K.

**Curvelet has strictly lower rel-err than shearlet at EVERY K on BOTH caches;
both directional frames beat Fourier by ~1.20×.**

## The NO-FAKE catch (why this is trustworthy)
An early ASYMMETRIC run compared a finer-orientation shearlet (5 shears/cone)
against the sibling's COARSER committed curvelet (6 orientations, 192 cols) and
showed shearlet ~1.08–1.33× AHEAD — a tuning-asymmetry artifact, NOT a real
shearlet advantage. Applying optimal-form discipline symmetrically (giving
curvelet the SAME finer-orientation treatment, n_orient0=10, at MATCHED 160 cols)
REVERSED it: curvelet is ahead. This is exactly the trap the optimal-form
non-negotiable exists to catch; the committed probe encodes the symmetric
comparison so the honest verdict reproduces.

**Mechanism (honest):** boundary-annulus patches are locally near-straight edge
segments. A ROTATED atom is a rigid rotation of the anisotropic Gaussian — it
aligns to the local tangent WITHOUT distorting the envelope. A SHEARED atom
distorts the envelope into a parallelogram (shear skews/stretches the tangent
extent), a poorer match to a clean edge at equal coefficient. Consistent with the
shearlet literature: shearlets match curvelets' ASYMPTOTIC N-term RATE but the
constant is slightly worse; shearlets are preferred for FAITHFUL DIGITAL
implementation (integer lattice), NOT for a better approximation constant.

## VERDICT: CURVELET for the boundary-annulus frame
- Curvelet (rotation) is the family-selection winner: strictly better N-term
  sparsity at matched/smaller budget on both caches, and it is already the wired,
  canonical incumbent.
- Shearlet is a GENUINE, certified, near-equal (~5–8%) alternative whose only edge
  is cheaper/integer-lattice decode — not enough to displace curvelet, which is
  also the better approximator here.
- **Build the OWED through-R d_seg harness for the CURVELET** (winner). Keep the
  shearlet as a certified, decode-cheaper fallback to A/B ONLY if a through-R
  measurement later makes decode cost binding (they behave nearly identically).

## OWED (serialized follow-ups — NOT done here, by boundary)
- **Through-R d_seg receipt** for the curvelet frame (operator-GO / CONTAINMENT):
  wire into the trainer forward + generated inflate.py op-parity + exact n600
  byte-closed d_seg. The OMP number is an UPPER BOUND, never a score.
- **Shearlet canonical-equation registration** in `tac.canonical_equations`
  (the island-birth sibling arm owns `canonical_equations/`; NOT touched here) —
  register `compact_shearlet_parabolic_capacity_v1` (mirrors
  `windowed_curvelet_parabolic_capacity_v1`) with the measured anchors above.
- **DSL lever** for the shearlet frame (default-off, byte-identical) mirroring the
  curvelet `windowed_curvelet_basis_lever` — OWED to the `witness_dsl`-owning arm.

## Reproduce
```
PYTHONPATH=src:upstream .venv/bin/python \
  experiments/shearlet_vs_curvelet_vs_fourier_capacity_probe.py \
  --npz experiments/results/mlx_fleet_gt_cache/gt_n600.npz --seeds 0,1,2
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/boundary_math/tests/test_compact_shearlet_frame.py -q
```
