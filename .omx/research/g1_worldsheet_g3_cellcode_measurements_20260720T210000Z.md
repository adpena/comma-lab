# G1 worldsheet transport + G3 cell-code floor — real n600 measurement

- UTC: `2026-07-20T21:00:00Z`
- lane: `lane_g1g3_successor_measurements_20260720`
- authority: `[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`
- borrowed-bank pointer: `0.18804 UNMOVED`
- receipt: `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json`
- receipt SHA-256: `38b1f5d5475037e360ce13f5aed7ae114d9e3c4834e7bffe388f0fb748fc5089`
- tool SHA-256: `6e8f4be210108b96e6c7329a8a9be8e70191256f604039e75a7f79a13de6a4a3`
- MAIN landing review: **REQUIRED**; this isolated-branch result carries no authority until reviewed and merged.

## Two-sentence intent

Measure whether all 1,199 sequential SegNet-boundary transitions are a low-residual worldsheet under the banked ego screw, while keeping within-pair and cross-pair custody separate. Independently price the exact 17,926 live flip targets as causal argmax-cell identities and compare the ideal identity stream with equal-site raw coordinates, the canonical 1,852-byte realized break-even, and the 2,114-byte receiver fixture.

## Landed verdicts

**G1: `SPARSE_EVENT_GO` for post-row elevation of #574, with a narrow formulation scope.** The median of per-transition finite residual medians is `0.2792 px` within-pair and `0.2798 px` cross-pair. Residuals above `4 px` are `8.3977%` and `8.2497%`, respectively, so the global boundary object is near-zero-centered with a sparse heavy tail under the explicit analysis convention `median <= 1 px && E_4 <= 10%`. This is a GO for the **worldsheet object plus an event grammar**; it is not a claim that one global ground-plane homography is a sufficient witness.

**G3: `CELL_ID_PAYS_VS_RAW_COORDINATES_BUT_CHEAP_PRIORS_MISS_LIVE_BYTE_GATES`.** The best measured causal prior, spatial+temporal Laplace, costs `2,724.8733` ideal bytes for all 17,926 sites and `2,474.8396` bytes for the 16,319 moderate-band sites. Those are only `4.3431%` and `4.3330%` of equal-site 28-bit coordinates, but remain `+610.8733 / +360.8396 B` above the 2,114-byte fixture and `+872.7820 / +622.7483 B` above the canonical 1,852.0913-byte realized gate. Cell identity is therefore the right rate coordinate, but this known-site cheap-prior realization does not yet close r1b5 GAP-3.

No score was computed. No archive was promoted. No critical-path r1b5 surface was edited.

## G1 — actual adjacency and transport custody

The cache is pair-structured:

- 600 exact within-pair transitions: `(2k -> 2k+1)`, using banked `gt_poses[k]` from the frozen 600x6 PoseNet target surface described by `src/tac/scorer_targets.py`.
- 599 cross-pair transitions: `(2k+1 -> 2k+2)`. The bank has no PoseNet target for these pairs, so the measurement uses `gt_poses[k+1]` as the nearest-target-pair proxy. Cross rows are advisory under that proxy, not exact cross-pair ego-motion evidence.
- Total: 1,199 sequential transitions, 10 unordered class-pair strata per transition, 11,990 durable pair/stratum rows.

The first realization extracts both pixels of every unlike 4-neighbor adjacency, applies one scorer-grid ground-plane homography, and computes symmetric nearest-edge Chamfer residuals. The SE(3) transform is produced by `tac.lie._se3_numpy.exp_se3` in translation-first `(rho, omega)` convention, with `rho=s_t*[pose2,pose1,pose0]`, `omega=s_r*pose[3:6]`, EON intrinsics `(fx,fy,cx,cy)=(910,910,582,437)` scaled from `1164x874`, height `1.22 m`, and the already-settled n200 label calibration `s_t=-0.00143, s_r=0, pitch=-0.05`. Applying that settled calibration avoids silently refitting the answer on n600; it also means this realization is translation-dominated and not a universal screw test.

### Aggregate transport

| cadence | transitions | finite Chamfer mean (px) | median of transition medians (px) | `E_1` | `E_2` | `E_4` |
|---|---:|---:|---:|---:|---:|---:|
| within pair, exact pose target | 600 | 2.1455 | 0.2792 | 23.1913% | 12.1776% | 8.3977% |
| cross pair, pose proxy | 599 | 2.0931 | 0.2798 | 22.8681% | 11.9873% | 8.2497% |

The mean/median split is the decisive structure: the center is subpixel, while sparse topology changes and a few tiny, spatially disjoint boundary sets produce a long tail. Similar within/cross numbers do not validate the cross proxy; they only show no gross proxy penalty in this clip.

### Where the worldsheet breaks

| stratum, both cadences | finite Chamfer mean (px) | median of transition medians (px) | `E_4` | births / deaths | interpretation |
|---|---:|---:|---:|---:|---|
| Lane–MyCar | 71.2320 | 0.6729 | 44.7181% | 234 / 234 | tiny intermittent lane/hood contacts; rare rows reach ~314 px, so topology/event state is mandatory |
| Lane–Undrivable | 11.3337 | 0.5045 | 35.2499% | 89 / 88 | lane contact appears/disappears against the undrivable/horizon partition |
| Lane–Movable | 7.5610 | 0.9822 | 17.9003% | 98 / 98 | moving-island birth/death plus independent object motion |
| Road–Lane | 3.2413 | 0.2256 | 16.3803% | 0 / 0 | always-present separatrix is centered, but dash/jitter events carry the tail |
| Undrivable–Movable | 1.9866 | 0.2925 | 4.3869% | 0 / 0 | mostly transported, with modest object/horizon residual |
| Road–Movable | 0.9628 | 0.1118 | 1.9327% | 0 / 0 | strongest nontrivial transported stratum |

The 30 worst rows are almost entirely tiny Lane–MyCar sets with `E_4=1`; examples include within transition 473 `(946->947)` at `313.76 px` mean and cross transition 259 `(519->520)` at `313.68 px`. These do not kill the worldsheet family because their support is small and explicitly topological. They do kill the formulation “one ground homography, no event channel.”

`verdict_scope=single global ground-plane-homography realization using exact within-pair poses and nearest-target-pair proxy cross poses; not the worldsheet object/family`

## G3 — exact live flip inventory and cell entropy

Inventory was re-derived from the 38 preserved live batch-16 stage files under `/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/baseline_stages_a7192f938785_31d77be9ab9f_107a7d3a179d`, not copied from a prose claim. Tree SHA-256 is `9287ba63fdb3eaf8d0ca58189487ac02fe8995c131daef021bf220255dffe5fc`. It contains 17,926 unique sorted flips, including 16,319 with margin `[1e-3,1)`. The live batch-16 target differs from the batch-32 cache at one inventoried cell; the stage receipts record three cache-label mismatches overall.

At each known flip site the receiver baseline class cannot be the target, leaving four target cells. Priors use only raster-causal left/up decoded target cells and/or the previous pair-index target at the same site, with add-one smoothing. The 5-ary row is the unconstrained reference. These are ideal arithmetic lengths: site locations, candidate-set transport, headers, receiver bytes, and realized-flip efficacy are excluded.

| prior | all 17,926 (B) | moderate 16,319 (B) | all / raw-coordinate |
|---|---:|---:|---:|
| uniform 5-ary | 5,202.8604 | 4,736.4431 | 8.2926% |
| uniform 4-ary, exclude baseline | 4,481.5000 | 4,079.7500 | 7.1429% |
| spatial Potts + Laplace | 3,148.9370 | 2,861.0064 | 5.0189% |
| temporal same-site + Laplace | 3,709.0241 | 3,373.7996 | 5.9116% |
| spatial + temporal + Laplace | **2,724.8733** | **2,474.8396** | **4.3431%** |
| fixed-width raw coordinate `(pair,row,col)=10+9+9 bits` | 62,741.0000 | 57,116.5000 | 100% |

Spatial context is more predictive than temporal context alone; the joint context is best. The stream is strongly edge concentrated:

- Road–Lane edge: 5,193 sites, `810.0446 B` under the joint prior — individually below both byte comparators.
- Other edge: 12,468 sites, `1,884.7952 B` — below 2,114 B but `32.7039 B` above the empirical 1,852.0913 B gate.
- Nonedge: 265 sites, `30.0335 B`.

This decomposition gives #572/r1b5 a concrete reverse-waterfill order: Road–Lane cell identity first, then only the highest-EV subset of other edges. Summing all strata still fails the live gate, and the omitted site/coder/receiver costs make this an optimistic floor.

The 1,852 comparator was consumed by ID `realization_breakeven_bytes_v1`: empirical anchor `1,852.091296 B`, callable replay `1,852.091427 B`, absolute registry-anchor residual `0.000130 B`. The 2,114 comparator is the deterministic production-fixture carrier delta from commit `1e574f44e1`; it is not mislabeled as a byte-closed live n600 candidate.

`verdict_scope=known-site ideal cell-identity stream under the measured uniform, local Potts, and same-site temporal priors; excludes site-location, candidate-set, coder-header, receiver, and realized-flip costs`

## Custody, resumability, and kernel drift

- GT cache: `5,078,017,610 B`, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`; every member was accessed as a ZIP_STORED memmap, never dense-loaded.
- Derived f0 labels: `/Volumes/VertigoDataTier/pact/evidence/g1g3_successor_measurements_20260720T203750Z/lstars_f0_n600_batch32.npy`, `117,964,928 B`, SHA-256 `cda80013ef79258e1102bc30fd4528ef1a461041fb35ab6e2f975b56c563dfc6`; 19 preserved batch stages, stage-tree SHA-256 `a4c819fdd9d670c86751401dbf7323975afde630855cfe9dbc5a414ffde84761`.
- Local batch-32 scorer vs cached f1 labels: exactly 3 mismatches over `600*384*512=117,964,800` cells, recorded as `MEASURED_KERNEL_GEOMETRY_DRIFT`; not coerced to exact.
- G1: 600 preserved pair stages, stage-tree SHA-256 `cd2f0cd8758a4dfe3766f3dbec6d397409dba0b93e141fadfb23b35a0301e1b6`.
- Governed launch history: the 4 GiB cap stopped at 4,424 MiB after 8.09 s; the 8 GiB cap stopped at 8,552 MiB after 199.29 s with 12/19 stages preserved; the admitted 12 GiB resume completed exit 0 in 119.03 s with a 6,874 MiB peak. No completed stage was lost.
- Run log: `/Volumes/VertigoDataTier/pact/evidence/g1g3_successor_measurements_20260720T203750Z/run.log`, SHA-256 `fc31b6783d4c3d9a822d2c32d6065133bfccf3278b088e3371752abfc5674053`.
- Cleanup: the derived 113 MiB sidecar is certified rebuildable but preserved for MAIN review; delete/move remains blocked unless its manifest and stage tree accompany the action. The auto-started memory blackbox was stopped by process group after harvest; no orphan remains.

## Triality and system intelligence

- EQUATIONS: `worldsheet_transport_residual_event_rate_v1` and `argmax_cell_identity_ideal_bytes_v1` have executable callables and n600 anchors in `src/tac/canonical_equations/g1g3_successor_measurements_20260720.py`.
- DAG: `.omx/research/g1_worldsheet_g3_cellcode_DAG_FEED_20260720T210000Z.md` routes G1 to post-row #574 and G3 to #572/r1b5 GAP-3 with formulation-scoped negatives.
- DSL: `research_only=true`; this pass adds no runtime lever and does not authorize a launch. A future consumer must compile an explicit event grammar/cell coder through typed DSL before actuation.

## Required next actions

1. #574 may elevate the worldsheet object post-row, but must add explicit birth/death and large-offset event state; never compile the global homography alone as the witness.
2. Build an exact cross-pair PoseNet target bank before treating cross-cadence transport as physical evidence.
3. #572/r1b5 should charge a receiver-closed site grammar plus cell identities, reverse-waterfilled Road–Lane first, and stop at `realization_breakeven_bytes_v1`; the known-site floor alone is non-authorizing.
4. Preserve the pointer and contest axes until a byte-closed exact CPU/CUDA archive exists.

## STORES CONSULTED

- `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md` addenda W and X2
- `.omx/research/time_traveler_doctrine_gaps_20260720.md` G1 and G3
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` including §8
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `src/tac/scorer_targets.py`, `src/tac/lie/_se3_numpy.py`, `tools/measure_pose_warp_dseg.py`
- `.omx/state/canonical_equations_registry.jsonl` by canonical helper, including `realization_breakeven_bytes_v1`
- `.omx/state/lane_registry.json`, current frontier scan, council anchors, probe blockers, latest sister findings/session/design/council memos
- live flip-stage bytes and n600 cache bytes at the SHA-pinned paths above
