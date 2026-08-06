# DDM ET3 Solve-Within + DK1 CVP N32 Receipt

Status: `HELD_POSE_BOUND_FAIL_NO_FIRE_ORDER_2`

Axis: `[macOS-CPU frozen-scorer advisory]`  
Score claim: `false`  
Promotion eligible: `false`  
Archive/evaluate.py row: none

## Result Row

| n | pair set | eta | bar | eta/bar | pose min | pose median | pose max | decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 32 | fixed ET1/ET2 n32 same-set comparability | 0.3562364031907179 | 0.1710048742006269 | 2.0831944402518787 | 0.8127642334869045 | 1.00314130363039 | 1.128389479902771 | `NOT_FIRED_POSE_BOUND_FAIL` |

The segmentation eta clears the ET2 priced bar, but the pose max fails both the script gate (`1.04`) and the stricter ET2 Arm-E reference max (`1.0356058119`). ET3 therefore does not fire the full n600 byte-close order. This is not a family fold on eta; it is a pose-held formulation result.

## Measured Components

| component | value |
|---|---:|
| flips before subset | 27029 |
| CVP flips after subset | 23099 |
| net flip reduction subset | 3930 |
| label ceiling net fixed subset | 11032 |
| subset seg delta S, no rate | -0.003331502278645833 |
| subset pose delta S against parent | 0.00001676523791938056 |
| subset joint delta S, no rate | -0.0033147370407264525 |

Worst pose-ratio rows:

| pair | eta | pose ratio | flips before | flips after | label ceiling net fixed |
|---:|---:|---:|---:|---:|---:|
| 485 | 0.4014778325123153 | 1.128389479902771 | 819 | 656 | 406 |
| 521 | 0.37181996086105673 | 1.108214370139264 | 1432 | 1242 | 511 |
| 471 | 0.3115727002967359 | 1.0998840750479255 | 919 | 814 | 337 |
| 48 | 0.3402061855670103 | 1.0442225143202255 | 708 | 609 | 291 |

## N4 To N32 Stability

| metric | value |
|---|---:|
| SW1 n4 solve-within eta | 0.31140716069941715 |
| ET3 n32 solve-within + CVP eta | 0.3562364031907179 |
| retention factor n32/n4 | 1.1439570059680542 |
| shrink factor n4/n32 | 0.8741587269302723 |

Unlike ET2 projected-static, which shrank about 2.3x from n4 to n32, ET3 retained and slightly exceeded the SW1 n4 eta on this fixed n32 set. The blocker moved from eta to pose max.

## Method And Boundaries

Parent: `tq1c`, archive SHA-256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`, 357837 B, parent `S=0.7534578126155775`, `d_seg=0.004305419922`, `d_pose=0.000716508925`.

Solver: SW1 solve-within null-basis, per-2x2 `c in R6`, `delta=N@c`, no post-solve projection, cap ladder `[15]`, lr `2.0`, eval every `5`.

Realizer: DK1 private-support CVP/Babai, `tap_radius=0`, `max_channel_candidates=9`, `max_pixel_candidates=16`, `max_combinations=250000`, exact within declared finite tap-radius scope, not a global integer optimum claim. The DK1 helper now clips an out-of-bounds continuous center to the nearest uint8-feasible bound before constructing the declared finite candidate set, and records that clipping in channel diagnostics.

Metric source: diagonal scorer-grid margin saliency from `tac.margin_saliency_map.compute_margin_saliency_map`, `lambda_saliency=1.0`, `outside_weight=0.02`, `saliency_clip=20.0`; no full MS4D row-Gram claim.

Pair selection: fixed ET1/ET2 n32 same-set comparability list, not random, not stratified, not contiguous prefix. Denominator is 32 measured pairs out of the 600-pair population.

Artifacts:

| artifact | SHA-256 |
|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_rows.jsonl` | `4634e5ceb62822a7ed0de2678c14da1c662485b96f85c03fd304e5057b1fe83d` |
| `/Volumes/VertigoDataTier/pact/ddm_et3_20260806/et3_solve_within_cvp_summary.json` | `3e74d83a6c78677bfd4776db92dda4fa0526665ce696465e1ca76797302e2d67` |
| `.omx/research/ddm_et3_20260806/et3_solve_within_cvp_summary.json` | `3e74d83a6c78677bfd4776db92dda4fa0526665ce696465e1ca76797302e2d67` |

Own-vehicle frontier was not moved. No `archive.zip` was built, no `upstream/evaluate.py` run was consumed, and no contest-CPU/CUDA claim is made.
