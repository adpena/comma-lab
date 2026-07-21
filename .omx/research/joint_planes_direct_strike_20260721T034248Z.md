# Joint planes direct strike — from-scratch task-space level-set spine audit

**UTC:** 2026-07-21T03:42:48Z  
**Lane:** `joint_planes_direct_strike`  
**Authority:** `[macOS-CPU advisory]`, `score_claim=false`, `promotion_eligible=false`  
**Pointer:** `0.1910828242 [contest-CPU] UNMOVED`  
**Receipt:** `.omx/research/joint_planes_direct_strike_20260721T034248Z.json`

## Outcome first

The source-only S0/S1 spine is real, current code reproduces the n600 lane-chart rate, and the
Rust AA-SDF/xi/range decoder parity gates pass. The from-scratch S2→S4 composition is **not yet a
receiver-closed candidate**: the only finite structured byte row is Lane-only, G3 is an ideal
cell-identity lower bound without site locations or finite-coder overhead, no counted pose/xi
stream is bound to the same full partition, and S3 has not realized the combined description.

Therefore this landing records
`PARTIAL_CONSTRUCTIVE_SPINE_MEASURED_S2_AND_S3_NOT_COMPOSED_S4_ABSENT`. This is a custody verdict
on the current composition, not a negative verdict on the task-space level-set witness family.

## What was built

`tools/audit_task_space_levelset_spine.py` is a fail-closed composition auditor. It:

1. hashes the source video, frozen upstream modules/weights, n600 target cache, and every consumed
   receipt;
2. rejects cross-cache M2/G1/G3 evidence, non-exact support fill, missing lane losslessness, or
   receipts that cross the score-authority firewall;
3. reruns the complete `tac-levelset-inflate` Rust suite and requires AA-SDF positive parity,
   its coefficient-bit-flip negative control, xi-column parity, and range-decoder parity;
4. emits stagewise S0–S4 admission state without converting partial component rates into a score.

Three regression tests cover the honest partial admission and two fail-closed custody cases.

## Stagewise result

| Stage | Status | Measured/verified fact | Exact scope |
|---|---|---|---|
| S0 true targets | `VERIFIED_EXISTING_N600_SOURCE_DERIVATION` | cache `cf8d8360…b8cd6`, 600 pairs | Existing canonical cache was re-hashed and cross-bound. Its historical build log does not itself bind source/module hashes, so this pass does not claim a fresh n600 rebuild. |
| S1 support fill | `MEASURED_TWO_DISTINCT_EXACTNESS_ROWS` | canonical support fill has `0 / 117,964,800` fp32 errors against rounded uint8 plane `Y` | The tie receipt's verdict prose says “NOT fp32-exact” but contradicts its numeric fields; the numeric array result was re-derived and is the authority. |
| S1 direct source target | same stage | `d_seg=0`, `d_pose=0`, `1,717,172,741 B` archive | Official macOS-CPU advisory M2 row; exact but direct per-camera realization, not compact structured description. |
| S2 Lane chart | `PARTIAL_MEASURED_NOT_PARTITION_POSE_COMPLETE` | current-code n600 replay: coherent-slot lossless-to-fit `41,303 B`, rate term `0.02750197234`; 2,967 fitted lines | Fitted band recall is only `0.5474557766`; lossless-to-fit is not lossless-to-Lane-mask or full-partition fidelity. |
| S2 xi/worldsheet | same | within/cross median transport `0.279212 / 0.279849 px`; >4px event fractions `0.083977 / 0.082497` | One global ground-plane homography. Cross-pair pose is a nearest-target-pair proxy; this formulation does not kill the richer worldsheet family. |
| S2 Morse–Smale cell prior | same | 17,926 flip identities, spatial-temporal ideal floor `2,724.873 B` | Site locations, headers, and finite-coder overhead are excluded. This is not a packet size. |
| S2 AA-SDF | same | n600 grid-384 AA row `d_seg=0.0008598582`; Rust golden parity + negative control pass | The n600 row measures the renderer on real frames; Rust bit parity uses real n96 lane coefficients. Neither is a full n600 partition receiver. |
| S2 range/blind geometry | same | full linear nullity `0.8067423152`; implemented exact blind fraction `0.2269692609`; `230,904` blind camera pixels/frame | Generic blind fill is free only where a camera-resolution payload exists; it saves zero direct bytes in a pure generator. |
| S3 integer-aware realization | `COMPONENTS_PRESENT_NOT_COMPOSED_WITH_S2` | lattice/support-fill modules exist; r1b7's adjudicated bounded n16 prefix produced zero new hard crossings | r1b7 is consumed as a fixed-magnitude/no-sub-step law only. Its inherited base archive and every payload byte are excluded. |
| S4 strict archive | `NOT_BUILT_FOR_THIS_FROM_SCRATCH_COMPOSITION` | no archive, per-class row, `d_pose`, or score | No score claim and no pointer movement. |

## Task-space level-set witness: the actual S2 object

The five-class partition is described by level-set functions `phi_c`; a class boundary is a
separatrix where two class fields tie. The Lane polynomial is the zero level-set of the Lane SDF,
and the AA-SDF receiver renders that chart. Per-class strata are the Morse–Smale cells. A single
chart is advected by the Chasles screw `xi`, with explicit residual symbols at measured
birth/death and large-transport events. Where polynomial/ground charts leave residual, the queued
representation is a curvelet/shearlet boundary chart, not Fourier.

The division of labor is binding:

- **Shape:** the task-space level-set witness describes the chart/partition.
- **Values:** the inverse solve pins scorer-bearing preimage values in the selected cells.
- **Realization:** the integer-aware solve preserves both chart and values through uint8/R.

Precision must be waterfilled on the measured Fisher/margin field. Bits are spent near the
separatrix and saddle cells where the argmax constraints bind; interiors do not receive precision
by default. The registered equations and their exact scope are listed in
`.omx/research/joint_planes_direct_strike_equations_20260721T034248Z.md`.

## Borrowed-substrate accounting and pedigree

`borrowed_candidate_archives=[]`, `borrowed_candidate_payloads=[]`,
`inherited_bytes_in_candidate=0`. The **lineage is submission-eligible** because every
candidate-bearing datum is source-derived/our-solve. There is no current submission candidate
because S4 is absent.

| Artifact | SHA-256 | Content lineage | Use |
|---|---|---|---|
| `upstream/videos/0.mkv` | `2611f5f3…2fa9` | source video | Sole video input to S0 |
| `upstream/modules.py` + frozen weights | `065961ba…49aa`, `68956e32…91b6`, `0f3a0874…6576` | upstream frozen scorer | S0 authority, not candidate bytes |
| `gt_n600.npz` | `cf8d8360…b8cd6` | source-video-derived, our build | S0 target custody |
| `reports/tie_aware_preimage_ab_receipt_n600_fidelity.json` | `02be4f4f…e724` | our support-fill solve measured on the exact-plane receiver | Law-only S1 exactness; no receiver/archive bytes consumed |
| `.omx/research/m2_live_target_selection_20260720T1548Z.json` | `513d2fe5…e574` | source-video-derived, our solve | S1 direct exact source-target row |
| SSD `lane_tracking_n600_current.json` | `c37d02b2…4d30` | source-video-derived, our solve | S2 current-code lane rate |
| `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json` | `38b1f5d5…5089` | source-video-derived, our measurement | S2 transport/cell priors |
| `reports/aa_sdf_observation_render_verify_n600_20260701.json` | `40eb2c58…3d76` | source-video-derived, our measurement | S2 AA fidelity curve |
| `.omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json` | `61f3d039…d03c` | mixed inherited-base experiment | Law-only S3 lesson; no bytes/content consumed |

The prompt's earlier #549/HNeRV-adjacent archive premise was superseded before construction. No
plane set, archive, decoder state, or residual payload from that lineage entered this candidate
spine.

## Exact missing closure, ordered by dependency

1. Emit one finite, parse-back-complete five-class chart grammar: Lane polynomial/ground chart,
   Movable Hungarian slots, Road/Undrivable/MyCar closures, and topology events.
2. Bind a counted pose/xi stream to the same n600 chart and measure its pose fidelity.
3. Add a genuine curvelet/shearlet residual chart only on measured chart residuals.
4. Compose that descriptor with the value preimage, `range(A)`, generic blind fill, and the
   integer lattice solver; require fixed-magnitude hard-oracle admissions.
5. Build the strict archive, parse it back in the receiver, and run the n600 hard oracle to obtain
   bytes, per-class `d_seg`, `d_pose`, and only then `S`.

Until those five edges land, `41,303 B` is a Lane-chart component row—not “partition+pose bytes”—
and `2,724.873 B` is a Shannon ideal identity floor—not a counted section.

## Verification

- `ruff check`: pass
- focused Python tests: `3 passed`
- Python byte-compile: pass
- current-code n600 lane replay: pass, `41,303 B`
- full `cargo test -p tac-levelset-inflate -- --nocapture`: pass, including AA-SDF parity,
  non-vacuous bit-flip control, xi-column parity, and range-decode parity
- `git diff --check`: pass

## MAIN landing review required

MAIN must independently inspect the S1 rounded-`Y` versus unrounded-source distinction, confirm
that r1b7 remains law-only, rerun the focused Python and Rust gates, and reject any downstream
consumer that treats the Lane-only/G3-lower-bound fields as a full S2 admission. MAIN should land
the branch only after that review; this isolated worktree does not alter the frontier pointer.
