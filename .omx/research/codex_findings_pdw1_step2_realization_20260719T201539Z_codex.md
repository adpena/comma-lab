# Codex Findings — PDW1 STEP-2 realization

Date: 2026-07-19T20:15:39Z  
Lane: `pdw1_dB_attack_step2_20260719T192712Z`  
Axis: `[macOS-CPU advisory]`  
Authority: `research_only=true`, `score_claim=false`, `promotion_eligible=false`  
Pointer: `0.1910828242 [contest-CPU Linux x86_64] UNMOVED`

## Verdict

`NAMED_EV_RANKED_HARD_ORACLE_RESIDUAL_AFTER_RATE_WATERLINE`, scoped to
`FORMULATION x SUBSET`. STEP-2 did **not** produce an in-box point. The
EV-ranked sparse camera-value repair loses to counted rate at its first measured
prefix, and the counted PDW1P base section is already above the 264,320-byte
gate. This falsifies this sparse-patch formulation, not the realization family.

## Measured rows

| subset | d_A | baseline d_B | selected d_B | residual px | selected k | projected base bytes | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| n24 | 0 | 0.00806956821017795 | 0.00806956821017795 | 38,077 | 0 | 496,067 | 6,211,452,928 |
| n48 | 0 | 0.008124139573838975 | 0.008124139573838975 | 76,669 | 0 | 497,079 | 6,128,975,872 |

The n24 first measured prefix changed 83 camera values for 8 ranked cells,
fixed 15 net hard-oracle pixels, and projected to 6,499 patch bytes. Its
marginal Seg score per projected byte was `4.89139004157513e-8`, below the
counted rate waterline `6.658589531221714e-7`.

The n48 confirmation changed 83 camera values for 8 ranked cells, fixed 3 net
hard-oracle pixels, and projected to 3,374 patch bytes. Its marginal was
`9.421797237727529e-9`, again below the same waterline. Strict patch parse-back
was identical in both measurements.

## Named residual

The exact n24 residual coordinate list is in the n24 receipt. Counts are:

- 24,138 `FULL_REFERENCE_PROBE_NOT_HARD_ACCEPT_OR_UINT8_DEAD`;
- 3,562 `INTERIOR_OR_NONRESIZE_NECESSITY_ZERO_149_WALL`;
- 10,377 `STOPPED_AT_MEASURED_RATE_WATERLINE`.

The n48 confirmation repeats the classification with 48,733, 6,098, and
21,838 cells respectively. Its compact receipt binds the canonical residual
coordinate JSON by SHA-256
`c2ebcfc2d181add08cc8c6d2e5f904714add15c81bae56ac6460e19a839fe1eb`.

No exhaustive joint affine-plus-SegNet search was performed, so these residuals
must not be relabeled as Diophantine-infeasible. The separate pair-0 exact-target
affine canary found 0 infeasible channel blocks out of 10,893, while leaving all
3,631 hard-oracle mismatches unchanged. That falsifies the delegated claim that
the existing affine solver itself supplies SegNet `HARD_ACCEPT`; it does not.

## Byte and authority boundaries

- n24 receipt SHA-256:
  `0ea81f25f0a0f2771c8ba183318af150a3b1e689e9da9a86f9b17f9a8fb15114`.
- n48 confirmation SHA-256:
  `fd71ea8cf3fb0c511729c13c6157b0ab465d168dca78d2d2d18082d98b39063b`.
- The PDW2 133-byte margin packet was not counted as a spatial base because its
  sealed receipt says `spatial_receiver_present=false`.
- Pose delta, an exact n600 archive total, and contest-CPU/CUDA replay remain
  unmeasured. No score or promotion claim follows from these subset rows.

## Exact next action

The receiver owner must first provide a counted spatial receiver for the PDW2
packet. Then re-run the receiver-closed candidate through exact n600 Seg and Pose
and contest CPU/CUDA. Any residual carrier must use the registered
curvelet/shearlet basis; the diagnostic sparse camera-value patch is not
admitted as that carrier.

## Stores consulted

- `.omx/research/pdw1_fp32_realization_receipt_20260719.json`
- `.omx/research/pdw2_gauge_packet_probe_20260719_receipt.json`
- `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
- `/Volumes/VertigoDataTier/pact/experiments/results/v10_power_diagram_byteclose_20260718/n600_rank4_features/quotient_features.f32.npy`
- registered equations named in the accompanying DAG FEED

