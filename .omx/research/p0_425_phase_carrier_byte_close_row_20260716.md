# p0_425 — phase-carrier byte-close measured row ($0, cached witness)

**Date:** 2026-07-16 · **Arm:** #515 $0 P0 burn-down · **Pointer 0.19108 UNMOVED** (means).
`[macOS-CPU advisory]` NON-PROMOTABLE. Exact-eval (`upstream/evaluate.py`) stays operator-GO.

## What ran ($0)

`tools/levelset_byte_close_and_eval.py --phase-carrier --skip-parity` on the CACHED
`v9_cgauge_432_coherent_arm_20260711` final EMA (copied OUT of the run dir; live run untouched).
The #359 phase-residual carrier (`src/tac/boundary_math/phase_residual_carrier.py`) STORES the
per-pair sub-pixel boundary-phase residual `r = t_wit_actual − t_wit_ξ_predicted` (ξ-transport
amortized) for the GROUND classes, entropy-coded.

## Measured phase-carrier row

```
[phase-carrier] ACTIVE (GROUND classes [0,1,2] = Road,Lane,Undrivable):
  section          = 10682 B   (13222 residuals, scheme=zlib9)
  rate_term       += 0.007113
  ξ_amort          = 1.041
  rmse_px          = 0.06568   (sub-pixel phase fidelity)
  bit_identical    = True
  recovered_d_seg  = OWED_through_R_n600_AB   (intrinsic bytes MEASURED; d_seg NOT claimed — NO-FAKE)
```

Composed with the seg-carrier archive (final EMA: 0.bin=66514 B / archive.zip=65797 B /
rate_term 0.0438, per p0_444), the phase-carrier adds a **10682 B** section →
combined rate_term ≈ **0.0438 + 0.0071 = 0.0509**. The homography is valid on the ground plane, so
the carrier is scoped to classes 0/1/2 (Road/Lane/Undrivable); Movable/MyCar are DEFERRED (homography
wrong there).

## Honest scope (NO-FAKE)

The BYTES are measured and deterministic. The `recovered_d_seg` (the d_seg the stored phase residual
buys back through R) is **explicitly OWED** — it requires the through-R n600 A/B (phase-carrier ON vs
OFF), the same >10-min CPU parity path that is harness-compute-pending (see p0_444). The row does NOT
claim a d_seg/d_pose delta; it banks the intrinsic carrier cost (10682 B for 13222 sub-pixel residuals
at 0.0657 px RMSE, bit-identical) so the costate ranker has the real rate side of the phase endgame
(MEMORY L86 APPEARANCE-PHASE ENDGAME · #425 ξ-residual codec).

## Provenance

Tool `tools/levelset_byte_close_and_eval.py` (imports the carrier at line ~3017); cached checkpoint
`v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz`; scheme=zlib9 (auto-selected best of
varint/zlib9/rice); q-step default 1/64; band = flip-prone straddle set.
