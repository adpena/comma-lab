# ddm_mf1 — DX2's manufactured Seg debt is a boundary-context defect, but both tested repairs lose jointly

**Disposition:** `MEASURED-N600-LOCALIZATION / FORMULATION-CLOSED-N32-X2 / RENDERER-FAMILY-OPEN`  
**verdict_scope:** `FORMULATION:{frame-local-boundary-pull, counted-native-oracle-mask-pull} × alpha{0.25,0.50,1.00} × SEEDED-STRATIFIED-RANDOM-N32`  
**axes:** localization support `[contest-CUDA T4 component field + macOS-CPU advisory intermediate logits, n600]`; repair response `[macOS-CPU advisory, seeded stratified-random n32]`  
**score claim:** false; no candidate archive, full-n600 candidate scorer row, contest score, or pointer move.

## Result first

The breaking has a precise address. Of the **16,917** final manufactured pixels first wrong at the
native render/head observation, **16,774 (99.1547%)** lie on the exact four-neighbor boundary of the
correct decoded token field; all **16,917** lie within Chebyshev radius 9, the nominal context radius
of the current renderer. The exact boundary is only **2.1665%** of the full 117,964,800-pixel body, so
the native failures are enriched there by **45.77x**. They are not an interior-capacity cloud.

The address is not yet a winning repair:

- A zero-counted-byte developmental pull on every decoded-token boundary was decisively harmful.
  Its mildest row fixed 309 errors but introduced 2,574 and had joint `delta S = +0.77834455`; pose
  supplied 95.37% of the loss. This result is retained, but is not load-bearing because the exact
  v2 instrument source bytes were not separately copied before the instrument evolved; that
  provenance incident is recorded below.
- The load-bearing oracle upper bound edits **only** the exact 16,917-pixel native-stage support.
  At alpha 0.25 it genuinely improves Seg on the n32 sample: **72 fixed, 46 introduced, net 26 fewer
  errors**, `delta d_seg = -4.1325887e-6`, `delta S_seg = -0.00041325887`. Pose nonetheless worsens
  from `5.4316097e-6` to `7.2015297e-5`, adding `+0.01946572 S`; distortion alone is therefore
  **+0.01905246 S worse**.
- Perfect addressing is not free. The real full-n600 native address mask is 14,745,600 raw packbit
  bytes and **35,969 B under Brotli q11**, deterministically repeated and decoded byte-identically.
  That payload adds at least `+0.023950282371 S` before container overhead. The best oracle row is
  therefore **joint `delta S >= +0.04300274393`**. The 35,969 B address alone is **14,431.8 B / 1.670x
  larger** than the native cluster's 21,537.2 B perfect-repair ceiling, so this address
  representation is rate-dominated even under impossible 100% repair with zero pose harm.

The charter's prior prediction is therefore **CONFIRMED at the stated formulation scope**: none of
the six measured rows has negative joint delta-S. This does not close jointly retraining the counted
renderer weights so that the repair is internalized without a shipped mask; it closes the two
post-render prototype-pull formulations tested here.

## Exact localization and currency

Every denominator below is the exact `600*384*512 = 117,964,800` body. `deep/moderate/hairline`
means signed target margin `<=-1 / (-1,-0.25] / (-0.25,0]` at the earliest observed wrong stage.
All 21,493 manufactured pixels are **NEW_FROM_CORRECT_TRANSMITTED_LABEL** by construction
`(A != G) and (L == G)`; amplified-existing count is **0**. The separate **6,918** representation
errors corrected by realization are beneficial flow and are not included in manufactured support.

| earliest observed stage | pixels | exact token boundary | within r9 | deep / moderate / hairline | mechanism status | measured repair / byte / pose / joint fields |
|---|---:|---:|---:|---:|---|---|
| native render + frozen head, unseparated | **16,917** | **16,774 (99.1547%)** | **16,917 (100%)** | 254 / 5,151 / 11,512 | **CONFIRMED-AT-SOURCE:** contextual CNN boundary mixing; literal painter and fixed palette **REFUTED-AT-SOURCE** | Oracle n32 alpha 0.25: `delta d_seg=-4.1325887e-6`; mask payload 35,969 B lower bound; `delta d_pose=+6.6583687e-5`; joint `delta S >= +0.04300274`. |
| float R + frozen head | **4,030** | **3,999 (99.2308%)** | **4,030 (100%)** | 0 / 28 / 4,002 | **CONFIRMED-AT-SOURCE operator:** bilinear lift/downsample; causal subpixel phase remains **UNDETERMINED** | **UNMEASURED:** no directed candidate admitted; R globally repairs 11,821 while breaking 4,841, and the full scorer lane was unavailable. Bytes/pose/joint likewise UNMEASURED. |
| uint8 + frozen head | **544** | **539 (99.0809%)** | **544 (100%)** | 0 / 0 / 544 | **CONFIRMED-AT-SOURCE operator:** round-to-uint8 amplitude floor | **UNMEASURED:** no directed candidate admitted; uint8 globally repairs 1,624 while breaking 853. Bytes/pose/joint UNMEASURED. |
| CPU-to-CUDA terminal, unseparated | **2** | 2 | 2 | CUDA margins absent | device/head cause **UNDETERMINED and directly unreachable** | **UNMEASURED:** retained CUDA logits absent; no candidate. |

Using only TX1 section 0's pinned `6.658590e-07 S/B` rate price, not a re-derived denominator:

| perfect oracle cure | `d_seg` recovered | Seg score recovered | byte-equivalent ceiling | share of 42,382 B demand | demand still open |
|---|---:|---:|---:|---:|---:|
| all current Seg errors, 23,757 | 0.00020139058431 | 0.020139058431 | **30,245.23 B** | 71.3634% | **12,136.77 B** |
| all manufactured, 21,493 | 0.000182198418511 | 0.018219841851 | **27,362.91 B** | 64.5626% | **15,019.09 B** |
| native manufactured, 16,917 | 0.000143407185872 | 0.014340718587 | **21,537.17 B** | 50.8168% | **20,844.83 B** |
| float-R manufactured, 4,030 | 0.000034162733290 | 0.003416273329 | **5,130.63 B** | 12.1057% | **37,251.37 B** |
| uint8 manufactured, 544 | 0.000004611545139 | 0.000461154514 | **692.57 B** | 1.6341% | **41,689.43 B** |
| terminal manufactured, 2 | 0.000000016954210 | 0.000001695421 | **2.55 B** | 0.0060% | **42,379.45 B** |

The charter's “30,248 B” is a headline rounding; the current exact 23,757-pixel field and TX1 price
give **30,245.23 B**. Even perfect Seg leaves 12,136.77 B of the fixed-distortion rate demand; even
zero total distortion still leaves the already-established **150 B** rate cut. MF1 did not alter
either campaign requirement.

## Source adjudication

The retained source is not the v14 fixed-palette painter. `SemanticTokenRenderer` performs one
simultaneous learned forward: token embedding + coordinate mix + frame FiLM, then four depthwise
3x3 blocks with dilations `(1,1,2,4)`, a learned 3x3 RGB head, and sigmoid. The cumulative spatial
radius is 9. There is no paint loop, overdraw order, fixed RGB palette, or literal antialias switch.

- **Paint ordering: REFUTED-AT-SOURCE.** No painter exists in the current receiver.
- **Fixed prototype color as the current cause: REFUTED-AT-SOURCE.** Colors are learned,
  coordinate-conditioned, frame-conditioned CNN outputs. Frame-local interior means were tested as
  a repair rule, not asserted as the current renderer mechanism.
- **Effective soft boundary/context spill: CONFIRMED-AT-SOURCE and by field join.** Every native
  manufactured pixel lies within the renderer's context radius, and 99.15% lies on the exact token
  boundary. The frozen head remains inseparable from the RGB observation, so MF1 does not claim a
  causal renderer-only/head-only split.
- **Pre-R subpixel placement: operator CONFIRMED, causal cure UNDETERMINED.** Bilinear lift/downsample
  is the only float operator between the native and preuint8 observations. Its final manufactured
  tail is 99.31% hairline, but R is a net repairer, so a generic “undo R” direction is invalid.
- **uint8 amplitude floor: operator CONFIRMED.** All 544 earliest uint8 errors are hairline. The
  stage is too small and too repair-positive globally to justify an undirected dither arm.

## Typed cluster localization rows

These are the complete nonempty `{stage,class,vertical band}` rows. `boundary` is exact decoded-token
four-neighbor boundary support. `D/M/H` is the margin split above. The stage table supplies the
mechanism, candidate bytes, pose, and joint response joined to these rows. Per-row **n600** recovery
is `UNMEASURED` because MF1 did not own the full scorer lane; the measured n32 native response follows.

| stage | GT class | spatial band | pixels | boundary | D/M/H | dominant wrong class |
|---|---|---|---:|---:|---:|---|
| native | Road | horizon_96_191 | 2,622 | 2,589 | 17/681/1,924 | Lane |
| native | Road | roadfield_192_287 | 3,296 | 3,277 | 22/851/2,423 | Lane |
| native | Road | nearfield_288_383 | 278 | 273 | 10/117/151 | MyCar |
| native | Lane | horizon_96_191 | 1,540 | 1,540 | 31/502/1,007 | Road |
| native | Lane | roadfield_192_287 | 2,356 | 2,356 | 56/621/1,679 | Road |
| native | Lane | nearfield_288_383 | 48 | 48 | 4/24/20 | MyCar |
| native | Undrivable | horizon_96_191 | 2,961 | 2,956 | 38/1,060/1,863 | Movable |
| native | Undrivable | roadfield_192_287 | 825 | 825 | 4/251/570 | Road |
| native | Movable | horizon_96_191 | 1,851 | 1,845 | 28/581/1,242 | Undrivable |
| native | Movable | roadfield_192_287 | 476 | 449 | 10/134/332 | Road |
| native | Movable | nearfield_288_383 | 9 | 9 | 0/8/1 | MyCar |
| native | MyCar | roadfield_192_287 | 494 | 447 | 29/240/225 | Road |
| native | MyCar | nearfield_288_383 | 161 | 160 | 5/81/75 | Road |
| float-R | Road | horizon_96_191 | 837 | 832 | 0/0/837 | Undrivable |
| float-R | Road | roadfield_192_287 | 961 | 956 | 0/4/957 | Lane |
| float-R | Road | nearfield_288_383 | 54 | 54 | 0/0/54 | MyCar |
| float-R | Lane | horizon_96_191 | 399 | 399 | 0/6/393 | Road |
| float-R | Lane | roadfield_192_287 | 875 | 875 | 0/10/865 | Road |
| float-R | Lane | nearfield_288_383 | 5 | 5 | 0/0/5 | MyCar |
| float-R | Undrivable | horizon_96_191 | 271 | 270 | 0/0/271 | Movable |
| float-R | Undrivable | roadfield_192_287 | 29 | 29 | 0/0/29 | Road |
| float-R | Movable | horizon_96_191 | 425 | 424 | 0/5/420 | Undrivable |
| float-R | Movable | roadfield_192_287 | 105 | 98 | 0/2/103 | Road |
| float-R | MyCar | roadfield_192_287 | 49 | 37 | 0/1/48 | Road |
| float-R | MyCar | nearfield_288_383 | 20 | 20 | 0/0/20 | Road |
| uint8 | Road | horizon_96_191 | 91 | 90 | 0/0/91 | Undrivable |
| uint8 | Road | roadfield_192_287 | 221 | 221 | 0/0/221 | Undrivable |
| uint8 | Road | nearfield_288_383 | 24 | 24 | 0/0/24 | MyCar |
| uint8 | Lane | horizon_96_191 | 11 | 11 | 0/0/11 | Road |
| uint8 | Lane | roadfield_192_287 | 50 | 50 | 0/0/50 | Road |
| uint8 | Undrivable | horizon_96_191 | 28 | 28 | 0/0/28 | Movable |
| uint8 | Undrivable | roadfield_192_287 | 3 | 3 | 0/0/3 | Movable |
| uint8 | Movable | horizon_96_191 | 58 | 58 | 0/0/58 | Undrivable |
| uint8 | Movable | roadfield_192_287 | 28 | 28 | 0/0/28 | Road |
| uint8 | MyCar | roadfield_192_287 | 26 | 22 | 0/0/26 | Road |
| uint8 | MyCar | nearfield_288_383 | 4 | 4 | 0/0/4 | Road |
| terminal | Road | horizon_96_191 | 1 | 1 | CUDA margin absent | — |
| terminal | Lane | roadfield_192_287 | 1 | 1 | CUDA margin absent | — |

## Measured repair responses

Selection was seeded (`20260823`) and stratified-random, not a prefix: two pairs from each
time-quartile x native-burden-rank-quartile cell, 32 unique pairs. The denominator is
`32*384*512 = 6,291,456` pixels. It intentionally over-samples burden strata and is not presented as
an unweighted n600 population estimate.

The matched baseline reuses v2's verified batch-1 CPU fields: `d_seg=0.0002063115438` (1,298 errors),
`d_pose=5.431609679e-6`, and 924/924 sampled native-manufactured pixels still wrong on CPU.

| candidate | Seg fixed / introduced / net | `delta d_seg` | `delta S_seg` | `d_pose` | `delta S_pose` | distortion `delta S` | exact address payload | joint `delta S` lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native oracle alpha 0.25 | 72 / 46 / **-26 errors** | **-4.132588704e-6** | **-0.0004132588704** | 0.00007201529661 | +0.01946572043 | **+0.01905246155** | 35,969 B | **>= +0.04300274393** |
| native oracle alpha 0.50 | 141 / 111 / **-30 errors** | **-4.768371582e-6** | **-0.0004768371582** | 0.0002609688381 | +0.04371516368 | **+0.04323832653** | 35,969 B | **>= +0.06718860890** |
| native oracle alpha 1.00 | 215 / 287 / **+72 errors** | +1.144409180e-5 | +0.001144409180 | 0.0009965241188 | +0.09245610915 | **+0.09360051833** | 35,969 B | **>= +0.11755080070** |

The best joint row is alpha 0.25, not the row with the largest gross native repair. Pose is not a
small correction: it is larger than the entire distortion loss because the Seg improvement partly
offsets it.

At alpha 0.25, the measured native-cluster response is:

| GT class | sampled support pixels | baseline errors | candidate errors | fixed | introduced | net errors |
|---|---:|---:|---:|---:|---:|---:|
| Road | 1,442,815 | 500 | 483 | 35 | 18 | **-17** |
| Lane | 36,175 | 307 | 307 | 16 | 16 | **0** |
| Undrivable | 3,117,392 | 213 | 205 | 13 | 5 | **-8** |
| Movable | 95,231 | 253 | 251 | 8 | 6 | **-2** |
| MyCar | 1,599,843 | 25 | 26 | 0 | 1 | **+1** |

Spatially, Road roadfield supplies 10 of the net 26-error improvement and Undrivable roadfield 7;
Lane horizon **loses 5** while Lane roadfield gains 5, netting to zero. This is why neither the 56
repaired native-manufactured pixels nor the all-class aggregate can be called a Lane-safe cure.

## Direct LD1 confrontation

MF1 is **not an LD1 rung in a new costume**:

- the decoded semantic token field remains byte-identical at
  `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`;
- no Lane token is dropped, coarsened, re-indexed, or re-encoded;
- the tested actuator is after token decode, in native RGB, and the oracle address is separately
  counted rather than charged to a fictitious token saving.

That distinction does not save the candidate. LD1's six Lane rungs all made the archive larger;
MF1's exact address also makes it larger by **at least 35,969 B**, and its mild row is only Lane-net
neutral on n32 while damaging pose. Lane is therefore defended on both surfaces in the evidence we
have: lossy token motion loses rate, and prototype RGB motion loses joint distortion/rate.

## Prior-law adjudication and RJ1 correction

**CONFIRMED at FORMULATION x2 / seeded-stratified n32.** The registered falsifier required a measured
negative joint delta-S at zero or near-zero bytes. The best load-bearing row is instead
`delta S >= +0.04300274393`; the developmental zero-byte family was worse at `+0.77834455`.

The broad statement “renderer-side improvement is impossible” remains **UNDETERMINED**. MF1 directly
measured a small Seg improvement under perfect addressing, so the RGB surface is controllable. What
failed was the tested prototype direction, its pose coupling, and the mask representation.

The charter's attribution “RJ1 refused renderer re-representation 3.51x, with pose 97.7%” is not
supported by the current RJ1 memo. The source-verified RJ1 disposition is
`MECHANISM-INCOMPLETE-WITHHELD`; all three distortion and joint columns are UNMEASURED. MF1 therefore
does not repeat the 3.51x/97.7% statement as an RJ1 fact. The underlying warning is independently
validated here: pose supplies **95.37%** of the mild zero-byte boundary-pull loss and more than 100%
of the oracle alpha-0.25 distortion loss after the Seg credit.

## Custody, verification, and authority boundary

The load-bearing stores are source-closed to the current committed candidates:

- v3 localization: 15 scientific files / **91,952,298 B**. `LOCALIZATION.json` is 17,672 B,
  SHA-256 `b06fcfe52861120308a7a03df9fabefee5edda80a69be4fbc93fff2a824c753c`;
  `CLUSTERS.json` is 35,363 B, SHA-256
  `220b63cff679752f90fbf283aea8aec64557c46b6f985df224b477f5e2058779`.
- oracle probe: 101 scientific files / **1,795,013,093 B**. `ORACLE_RESULT.json` is 40,713 B,
  SHA-256 `4017c93089c0c570f468bf355a0df17e99bf6f8f5df2d535a21dada0aa89fb47`;
  `ORACLE_MANIFEST.json` is 8,593 B, SHA-256
  `d8fa2a7307e7400cbb83e42d2b56ab21164d9fbf4286ff078d15a87df63953e0`;
  verification SHA-256 `186780f613d579166a96264ab451214629ea95ae4c6f4e098fef0bf65c8056ed`.
- each oracle variant retains 32 complete per-pair NPZ payloads: 598,313,360 B / 598,310,783 B /
  598,306,506 B. Verification rehashed all 96 candidate payloads and all 32 reused baseline payloads.
- the main v3 source hash is `89ac31c2...67cbb`; the oracle source hash is
  `46f13212...88c0`, and both matched the result's source facts after measurement.

The first v1 probe stopped before baseline admission when its exact-logit diagnostic mistook the
expected batch-1 versus batch-16 CPU float envelope (`max_abs=4.529953e-5`) for semantic drift.
Exact RGB and argmax were equal. v1's completed localization remains retained. v2 corrected only
that diagnostic, retained the broad-boundary family, passed completed verification, and repeated
`PROBE_RESULT.json` byte-identically at SHA-256
`4b6f6e4e7de75d42175c76d614430613bfaf3a6bae7b0446a96e0fec87b5b8b0`.

**Provenance incident:** before extending to the oracle formulation, the v2 instrument source was
edited in place without first copying its exact source bytes to the retained store. v2 retains the
source hash `c3a4dfe7...b63ee5`, all scientific fields, result, checkpoint, and manifest, but not an
exact separate source copy. It is therefore developmental corroboration only, not load-bearing.
The v3/oracle sources are the live repository files and will be serializer-committed.

The oracle mask compression has an exact primary/repeat identity. A second oracle result assembly
was not run because concurrent APDataStore consumption left only about 0.75 GB, below the instrument's
2.0 GB fail-closed resume preflight after all payloads were retained. Completed verification did run.
No bytes were deleted to manufacture headroom.

No Modal or Metal action ran. `upstream/` was read-only. MF1 did not touch JF1 receipts or launch a
full-n600 scorer. The current candidates are component measurements, not archives and not score rows.

**STORES CONSULTED:** `.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/` (all 38 retained chunks, masks, G/L/A, renderer source) · `/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/inflated/0.raw` · `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy` · `/Volumes/APDataStore/pact/ddm_mf1_manufactured_seg_repair/{measurement_v1,measurement_v2,measurement_v3,oracle_native_v1}` · `.omx/state/canonical_frontier_pointer.json` · JF1 scorer receipts: none loaded or modified · Modal: none · Metal: none · upstream writes: none.

## RECALL EVIDENCE

Recall searched the full `.omx/research/` corpus by content, not only charter filenames; the canonical
equation listing; `CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*` FEED blocks; current focus/hot/task/lane
state; and the actual DX2 renderer/evaluator sources. Queries included `paint ordering`, `overdraw`,
`fixed palette`, `prototype`, `semantic renderer`, `anti-alias`, `subpixel`, `camera placement`,
`pre-R`, `uint8`, `manufactured native`, `realization fidelity`, `margin`, `argmax`, `renderer
re-representation`, and the exact archive/token hashes.

Beyond the charter seeds:

- v14's fixed-palette/paint ordering lineage was found and rejected as a mechanism transfer because
  DX2 is a learned simultaneous CNN, not a painter. That changed the candidate from a paint-order
  swap to a decoder-derived class-color test.
- RT2/RN1's generic blur/deblur/AA/dither negatives and MST1's gross repair flows prevented a rerun
  of the undirected camera ladder. That preserved R rather than treating fidelity as the objective.
- WJ1 and the live lane state showed that full scorer ownership remained elsewhere, so MF1 used the
  legal seeded-random n32 scope reduction and emitted no n600 fire.
- GV1 corrected the stronger “joint moves uniquely win” narrative; MF1 therefore treats jointness as
  a measurement obligation, not routing privilege.
- The current RJ1 memo contradicted the charter's claimed measured 3.51x/97.7% row: RJ1 withheld all
  distortion. That changed the memo from citing the number to recording the citation correction.
- LD1's exact six archive-growing Lane rows made an oracle address sidecar admissible only if counted.
  MF1 therefore retained and priced the real 35,969 B mask instead of calling perfect addressing free.

## Verification

- Python compilation, Ruff check/format, strict targeted payload-retention census, and two genuine
  review-tracker passes passed for both Python instruments.
- Localization checkpointed atomically every 16 pairs; each probe checkpointed after every retained
  pair. Explicit `--resume-from` paths are source-bound and restricted to APDataStore.
- The unchanged v2 CPU baseline had exact raw RGB and argmax identity against MST1; batch-geometry
  logits stayed inside the recorded `1e-4` diagnostic envelope.
- All scientific candidate fields were retained before scalar metrics: native RGB, camera preuint8,
  camera uint8, evaluator resize, SegNet logits/argmax, PoseNet YUV6 input/output, edit mask, and
  per-frame prototype/support metadata.
- The full-n600 raw/compressed/repeat address mask passed exact decode and deterministic repeat.
- The oracle completion verifier rehashed 96 candidate payloads, 32 reused baseline payloads, result,
  manifest, and checkpoint. The shared Git index was empty before landing work.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — fold MF1's boundary/margin map into the already-queued RJ1 exact-object joint renderer solve, not into a shipped mask.** Owner: MAIN-designated RJ1 successor. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/joint_r1/`. Fire trigger: JF1's terminal fits are harvested, MAIN confirms the work is not a duplicate active lane, a resumable trainer keeps the renderer packet at the same or smaller real archive size, and pose is in-loop from the first accepted update. Use MF1 masks only as training weights; any video-derived runtime address remains counted.

## LIVE-HYPOTHESES

- A same-length joint renderer-weight update may internalize the boundary correction without the
  35,969 B mask. This remains plausible because perfect addressing produced a real Seg improvement;
  the missing ingredient is a direction that protects PoseNet rather than a stronger prototype pull.
- A low-dimensional frame/global subpixel phase control may address part of the 4,030 float-R tail
  more cheaply than a pixel mask. It is plausible because 99.31% of that tail is hairline and on the
  boundary, but it must preserve R's much larger gross repair flow.
- A tangent-aware or learned local correction may outperform frame-mean prototype colors. The present
  pull changes boundary chroma/luma normal to the class mean; the source CNN's learned boundary chart
  can carry directions that a class centroid erases.

## DEAD-ENDS

- Literal paint-order repair is closed at source for DX2: there is no painter or overdraw order.
- Fixed-palette diagnosis is closed at source for DX2: the renderer has learned token/frame/context
  mappings, not a fixed class palette.
- Uniform frame-local prototype pulls over every decoded-token boundary are closed for alpha
  0.25/0.50/1.00 on seeded-random n32: all lose Seg collateral and pose; the mildest is
  `delta S=+0.77834455` before any receiver-integration caveat.
- Shipping the exact native-stage repair mask is closed as a representation: 35,969 B exceeds its
  21,537.2 B perfect Seg ceiling before pose or container overhead.
- The counted oracle prototype pulls are closed for alpha 0.25/0.50/1.00 on seeded-random n32: the
  best joint lower bound is `+0.04300274393`.
- Generic blur/deblur/AA/dither is not reopened: the searched RT2/RN1 ancestor ladder is negative and
  current-object R/uint8 are net repairers, so an undirected rerun has no current mechanism warrant.
- The charter's RJ1 3.51x/97.7% sentence is closed as a measured citation; the current RJ1 memo has
  no distortion measurement.
- A full-n600 scorer fire for either tested MF1 family is folded: both are already positive joint
  losses on n32, and the counted mask is rate-dominated even under perfect repair.

Own-vehicle frontier: **DX2 — S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; MF1 did not move it.
