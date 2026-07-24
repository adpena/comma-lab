# DDM WS1 Seg-lexicographic warm-start DAG — 2026-07-24

`research_only=true` · `[macOS-CPU frozen-scorer advisory]` · `score_claim=false` · pointer `0.1910828242 [contest-CPU]` unmoved.

## Executable dependency graph

```text
V19B settled strict-Seg prefix (10 moves)
  + V19C exact n600 decision chain (104 joint admissions)
  └─ strict Seg filter: incremental Seg term < 0, Pose ignored
       ├─ keep 96 V19C moves
       └─ reject 8 non-strict moves
            ↓ exact source replay and receiver compile
       Seglex96 V19C receiver
       137,827 B · sha256 4fbba057...
            ├─ base fresh n600 scorer replay
            ├─ temporal affine 204 B, unmasked
            └─ temporal affine 204 B, decoder-hood-masked
                    ↓ exact uint8/R/frozen scorer readback
               W_seg
               d_seg 0.024124510023328993
               d_pose 146.3649324958955
               bytes 138,031
               MyCar 37,619

Original SHA-bound V19C receiver
  + MENU1 local-statistics/hard-analytic 974 B
       ↓ fresh composition hash + first-batch scorer replay
     W_joint
     d_seg 0.07051923116048177
     d_pose 36.6181847780574
     bytes 138,801
     MyCar 4,072,489

W_seg + W_joint measured task terms
  └─ R* = extra Pose debt / opening Seg advantage
       = 4.1215446777965665
       ↓ preregistered J5 four-step smoke from both starts
     metric authority:
       Seg margin-Fisher/rank-4 class-pair hyperplanes
       Pose exact low-rank quadratic through official YUV6/R
       Hessian/SPD normal coordinates from step 1 where measured
       Euclidean identity-L2 = side-by-side CONTROL only
       ↓ no launch in this landing
     adopt W_seg only if ratio >= R*, Pose progresses, Seg does not regress;
     otherwise keep W_joint
```

## Triality and FEED

- DSL: `.omx/research/configs/ddm_ws1_seglex96_filtered_warmstart_20260724.json` and `.omx/research/configs/ddm_ws1_j5_slope_falsifier_20260724.json`.
- DAG/FEED: this file; final candidates also land in `.omx/research/ddm_train_decision_table_warm_start_rows_20260724.json` and the J5/#366 ticket’s `warm_start_candidates`.
- Equation: `ddm_ws1_warm_start_slope_falsifier_v1`, callable at `tac.optimization.ddm_warm_start_slope_falsifier:critical_pose_to_seg_slope_ratio`, registered through the locked canonical-equation helper.

The measurement runner checkpoints each 16-pair batch on `/Volumes/VertigoDataTier/pact`. Training, paid dispatch, contest exact eval, and frontier mutation are all false.
