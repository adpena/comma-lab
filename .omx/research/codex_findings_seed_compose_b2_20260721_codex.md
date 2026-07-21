# Codex findings — seed_compose_b2 real n600 constraint seed

## Verdict

`REAL_N600_DESCRIPTION_SEED_MEASURED_RECEIVER_REALIZATION_BLOCKED`.

The single PPCS object is real, canonical, and measured through frozen CPU-Torch cache replay at n16, n64, and n600. At n600 it is cell-exact at all 3,188 declared sites, all 600 banked PoseNet targets remain inside their declared tubes, `d_pose=0`, and every double decode is byte-identical. It is **not** camera-RGB receiver-closed: `uint8_factor2_exact=false` for all 600 pairs. This is `[macOS-CPU advisory]`, not `upstream/evaluate.py`, not a contest score, and the frontier pointer remains unchanged.

## D1/D5 curve and KKT verdict

| point | constraints | PPCS bytes | zlib-9 | d_seg description | advisory objective |
|---|---:|---:|---:|---:|---:|
| loose | 3,188 | 884,872 | 78,969 | 0.343497721354 | 34.938972078984 |
| knee | 6,375 | 1,515,520 | 99,885 | 0.343470704820 | 35.356193042678 |
| tight | 12,749 | 2,809,838 | 125,159 | 0.343416671753 | 36.212622964416 |

Both measured tightening marginals are below `25/37,545,489`; the preregistered minimum is therefore a **boundary KKT result**, not an interior knee. A symmetric looser/knee/tighter triple is infeasible at the nonnegative/full-pair tube floor; the three rows above provide one-sided curvature.

## D2 hard-oracle ladder

| scope | d_seg | d_pose | cells exact | Pose tubes | uint8 factor-2 |
|---|---:|---:|---|---|---|
| n16 | 0.347817420959 | 0.000000000000 | True | True | False |
| n64 | 0.347797314326 | 0.000000000000 | True | True | False |
| n600 | 0.343497721354 | 0.000000000000 | True | True | False |

## D3 predictor satisfaction by target class

| class | satisfied | total | fraction |
|---:|---:|---:|---:|
| 0 | 13,936,579 | 27,407,046 | 0.508503507 |
| 1 | 57,156 | 690,639 | 0.082758141 |
| 2 | 36,036,893 | 58,413,281 | 0.616929787 |
| 3 | 1,457,041 | 1,460,325 | 0.997751186 |
| 4 | 25,953,303 | 29,993,509 | 0.865297321 |

Lane is the failure center. Its whole-field satisfaction is low because Lane never wins the temporal mode; its compatibility site is derived from full n600 occupancy instead. The loose constraint prefix is correspondingly concentrated on Road-Lane boundaries. This negative is scoped to the five-site compatibility raster. The native Morse-Smale family remains open under exact blocker `MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED`.

## B5 and terminal handoff

The single object is 884,872 raw bytes versus 235,952,891 for equal per-frame represented fields; zlib-9 is 78,969 versus 996,148. Rung 1 (camera-RGB realization) blocks the diagonal finisher, MC finisher, gauge quotient, adaptive-statistics strip, JRD, #557 entropy pack, composition closure, and R6. Those later rungs were not faked or bypassed.

## STORES CONSULTED

- `/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/receipt.json`
- `/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/hard_oracle_n16/receipt.json`
- `/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/hard_oracle_n64/receipt.json`
- `/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/hard_oracle_n600/receipt.json` and its 600 preserved pair stages
- `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json` through LawRef
- `.omx/research/s2_compose_full_partition_20260721T041640Z.json` and the exact S2 packet

Machine-readable measurement: `.omx/research/seed_compose_b2_measurements_20260721.json` (`5108ae6ab4febf0c1d8f22c5f978224a803d25b332159c48a7e2e74130509205`).
