| earliest observed stage | errors vs contest-CUDA DALI `G` / 117,964,800 px | earliest final manufactured vs `G` | share of 21,493 | earliest final repaired vs `G` | gross right->wrong / wrong->right vs `G` | net gross error change | byte-equivalent ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| native render + frozen SegNet head, unseparated | 31,503 | **16,917** | **78.7093%** | **6,154** | 28,602 / 6,281 | +22,321 | **21,537.2 B** |
| bilinear camera lift + evaluator downsample, before uint8, + frozen head | 24,523 | **4,030** | **18.7503%** | **695** | 4,841 / 11,821 | **-6,980** | **5,130.6 B** |
| uint8 camera quantization + evaluator downsample + frozen head | 23,752 | **544** | **2.5311%** | **69** | 853 / 1,624 | **-771** | **692.6 B** |
| CPU-to-CUDA terminal device/head transfer, unseparated | 23,757 | **2** | **0.0093%** | **0** | 7 / 2 | +5 | **2.5 B** |

# ddm_mst1 — DX2 manufactures 78.71% of its final downstream Seg error at the native-render observation, not in R

**Disposition:** `MEASURED-N600-STAGE-SPLIT / COMPLETE`,
`verdict_scope=INSTANCE:DX2_T4_n600_WITH_MACOS_CPU_INTERMEDIATE_OBSERVATIONS`.
The first table is the result. Every count in it compares against the exact contest-CUDA DALI GT
field `G`; the first three decision fields were observed with frozen CPU-torch SegNet on
`[macOS-CPU advisory]`, while final support and the last row come from MS9's exact
`[contest-CUDA T4 component-only exact field replay]`.

The registered prior predicted that the bilinear round trip would carry at least 50% of the
manufactured support. It carries **18.7503%**, so that prediction is falsified. Its registered
distributed-loss falsifier, “no single stage carries more than 30%,” is also false: the native
render + frozen-head observation carries **78.7093%**. The result is a third, more actionable case:
the majority localizes before R, on a surface controllable only through what DX2 renders.

This is not a score row. No archive changed and the pointer did not move.

## Exact gate reproduction and definitions

The analysis reused, rather than re-derived, MS9's exact fields:

- `G`: contest-CUDA DALI GT argmax;
- `L`: decoded DX2 semantic labels;
- `A`: terminal contest-CUDA argmax.

Before stage attribution it reproduced the required gates exactly over
`600*384*512 = 117,964,800` pixels:

| exact field-replay gate, contest-CUDA DALI `G` | count |
|---|---:|
| transmitted representation errors, `L != G` | **9,182** |
| final DX2 errors, `A != G` | **23,757** |
| final manufactured errors, `(A != G) and (L == G)` | **21,493** |
| representation errors still wrong, `(A != G) and (L != G)` | **2,264** |
| representation errors corrected at terminal, `(A == G) and (L != G)` | **6,918** |

Any disagreement would have stopped the run before final masks were admitted. There was none.

The observed sequence is:

```text
L: decoded semantic labels
  -> B1: frozen SegNet(native 384x512 float RGB)
  -> B2: frozen SegNet(D_bilinear(U_bilinear(native))), before uint8
  -> B3: frozen SegNet(D_bilinear(uint8(round(U_bilinear(native))))) on CPU
  -> B4: A, the retained contest-CUDA terminal argmax
```

At each RGB intermediate, “wrong” means
`argmax(frozen upstream SegNet(intermediate)) != G`. This is faithful because it uses the exact
upstream `SegNet`, weights, preprocessing call, batch geometry, and frozen decision rule on each
reachable RGB state. It is not a claim that RGB itself has a class label. The SegNet forward and
argmax are deliberately named **unseparated** from every RGB observation; forcing a fictitious
render-only/head-only split would be unsupported.

For each final manufactured pixel, the charged stage is the first `Bi` at which it is wrong. For
each of the 6,918 beneficial final corrections, the charged stage is the first `Bi` at which it is
right. The gross columns count every adjacent right-to-wrong and wrong-to-right transition, whether
or not it survives to terminal. That distinction is load-bearing: the path oscillates and repairs
real errors.

## What the path is doing in both directions

The state-error trajectory against contest-CUDA DALI `G` is:

```text
L 9,182 -> native-head 31,503 -> preuint8-roundtrip-head 24,523
          -> uint8-roundtrip-head 23,752 -> CUDA terminal 23,757
```

The native observation adds 22,321 errors net. The float round trip then removes **6,980** net and
uint8 removes another **771** net. In gross terms, the float round trip repairs **11,821** pixels
while breaking **4,841**; uint8 repairs **1,624** while breaking **853**. Therefore “undo R” or
“make realization more faithful” is not an admissible objective by itself. Even though 4,030 final
manufactured pixels first become wrong at the float round trip, a cure must preserve its much larger
gross repair flow and the terminal 6,918 representation corrections.

The CPU terminal observation and the retained CUDA terminal field disagree at only **9** pixels:
7 right-to-wrong and 2 wrong-to-right on transfer, net +5 CUDA errors. Only 2 of the 21,493 final
manufactured pixels are first charged there. This is why the device/head boundary is left merged.

## Per-class stage split

Every count and per-area rate below uses the pixel's contest-CUDA DALI GT class. Rates are
manufactured pixels per million pixels of that GT class; stage cells are
`manufactured / repaired / manufactured-per-million`.

| contest-CUDA DALI GT class | GT pixels / area | native render `m/r/rate` | float round trip `m/r/rate` | uint8 `m/r/rate` | CPU->CUDA merged `m/r/rate` | total manufactured / per-million | final repairs attributed across stages |
|---|---:|---:|---:|---:|---:|---:|---:|
| Road | 27,407,372 / 23.2335% | 6,196 / 2,838 / 226.071 | 1,852 / 246 / 67.573 | 336 / 4 / 12.259 | 1 / 0 / 0.036 | **8,385 / 305.940** | 3,088 |
| **Lane** | **690,754 / 0.5856%** | **3,944 / 1,239 / 5,709.703** | **1,279 / 89 / 1,851.600** | **61 / 8 / 88.309** | **1 / 0 / 1.448** | **5,285 / 7,651.060** | **1,336** |
| Undrivable | 58,413,067 / 49.5174% | 3,786 / 1,024 / 64.814 | 300 / 230 / 5.136 | 31 / 38 / 0.531 | 0 / 0 / 0 | **4,117 / 70.481** | 1,292 |
| Movable | 1,460,386 / 1.2380% | 2,336 / 830 / 1,599.577 | 530 / 47 / 362.918 | 86 / 6 / 58.889 | 0 / 0 / 0 | **2,952 / 2,021.383** | 883 |
| MyCar | 29,993,221 / 25.4256% | 655 / 223 / 21.838 | 69 / 83 / 2.301 | 30 / 13 / 1.000 | 0 / 0 / 0 | **754 / 25.139** | 319 |

The Lane sub-prediction holds. Body-wide manufacture is 182.198 pixels per million; Lane is
7,651.060 per million, **41.99x** denser. At the dominant native stage, Lane is 5,709.703 per
million versus 143.407 body-wide, **39.81x** denser. Lane is therefore a location of concentrated
debt, not a licence for a Lane-only repaint: LQ1 measured that such targeting dies on collateral.

## Reachability adjudication

`upstream/`, R, uint8, SegNet, argmax, and the CPU/CUDA implementations remain frozen. “Reachable”
below means controllable only by changing the legal rendered input before the frozen operation.

| stage | reachability | cheapest concrete candidate lever and counted-byte condition | collateral/admission boundary |
|---|---|---|---|
| native render + frozen head | **ADDRESSABLE THROUGH RENDER; head unseparated and frozen** | Re-optimize the existing semantic-renderer/token values against the frozen SegNet with the exact round trip in-loop. DX2's ZIP contains one 180,268-byte **Stored** member, so a same-length replacement targets **0 B archive delta**; this is a constraint for a future byte-closed build, not a measured saving. | Must preserve pose, the 6,154 final corrections first realized here, all downstream corrections, and non-target classes. Exact archive bytes decide whether the 0 B target was met. |
| float bilinear round trip | **CONDITIONALLY ADDRESSABLE PRE-R; R itself is unreachable/frozen** | Joint train-through-R native-field optimization / camera sub-pixel placement, constrained to the same packed payload length: **0 B target**, currently unmeasured. | Gross flow is strongly beneficial: 11,821 repairs vs 4,841 breaks. Generic deblur, blur, sharpen, and antialias ladders are closed on the RT2 ancestor and are not a justified DX2 rerun. |
| uint8 | **CONDITIONALLY ADDRESSABLE PRE-QUANTIZATION; uint8 itself is unreachable/frozen** | Quantization-aware optimization of the same renderer values, folded into the joint native/R solve at a **0 B same-length target**. | Only 544 final manufactured pixels / 692.6 B ceiling. Preserve 1,624 gross repairs. Undirected dither is not a directed cure and is closed on ancestor evidence. |
| CPU->CUDA terminal merged stage | **MANUFACTURED-BUT-DIRECTLY-UNREACHABLE** | No admissible direct lever: device arithmetic, frozen head, and argmax cannot change. A renderer margin increase may indirectly remove sensitivity but is not a stage-specific cure. | Two earliest final pixels / 2.5 B ceiling cannot justify a separate arm. Fold into validation of any renderer candidate. |

The 0 B entries are **candidate constraints**, not accomplished mechanisms. Video-derived optimized
values remain counted inside `archive.zip`; keeping the payload member length fixed avoids an
increment, it does not make those values free or prove a score gain.

## Ceiling economics — not a plan

At the exact `1.273108215332031 B/flip` exchange rate:

- native stage: 16,917 flips -> **21,537.2 B ceiling**, 50.82% of the 42,382 B campaign demand;
- float round trip: 4,030 -> **5,130.6 B ceiling**, 12.11% of demand;
- uint8: 544 -> **692.6 B ceiling**, 1.63% of demand;
- CPU/CUDA merged terminal: 2 -> **2.5 B ceiling**, 0.006% of demand;
- all manufactured support: 21,493 -> **27,362.9 B ceiling**, 64.56% of demand.

These are oracle-equivalent ceilings that assume every targeted flip is removed while pose, rate,
and all beneficial corrections are preserved. They are not predicted archive savings. No measured
mechanism currently realizes any one of them.

## Authority, controls, and custody

The source pins were:

| object | bytes | SHA-256 | authority/use |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | exact body |
| decoded labels `L` | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | exact retained token field |
| DALI GT `G` | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | contest-CUDA DALI GT |
| CUDA terminal `A` | 117,964,928 | `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34` | contest-CUDA component field |
| frozen SegNet weights | 38,502,892 | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` | exact upstream mirror |
| capture source | 40,117 | `2f42f4cd602c98aed43291fc1af3db219bc3f0ed1046316eb9e152797ee64382` | copied into instrumented runtime byte-identically |

The sole n600 launch used `tools/fire_local_advisory.py`, launch counter 422, and exited 0 after
3,039.4 s. Its score tail is explicitly `[env-mismatch advisory]` with PyAV GT and is **not used**
in any MST1 count. No Modal or Metal job ran.

The instrumented CPU decode retained a 3,662,409,600-byte raw with SHA-256
`7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7`. The earlier uninstrumented
DX2 CPU decode reports the same bytes and SHA in
`/Volumes/APDataStore/pact/ddm_dx2/r7/decode_r1/launch/run.log`; the observer did not change the CPU
render. This is a byte-identity positive control, not cross-axis score authority. The terminal CUDA
field still comes only from MS9's retained component authority.

The current charter's explicit-opt-in tier is **local disk** because both SSDs were measured at
100% (Vertigo 8.4 GiB free; APDataStore 11 GiB free). The durable store is
`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local/`.
Its storage preflight recorded 535,660,544,000 B free before copying. No `/Volumes/*` write occurred.

The stage fields came from the already-complete canonical-fire capture that predated the storage
reroute. The migration read it without mutation, verified every source hash, and copied each payload
atomically with a per-chunk local checkpoint. `MIGRATION_COMPLETE.json` is 14,501 B, SHA-256
`b57500333b0a18cfe50fe1f57b1162cf1903a2247ebed63f3a369294c6a442d2`, and records
`no_volume_writes=true`. Local `CAPTURE_COMPLETE.json` is 18,522 B, SHA-256
`4bb011f3a620d50086a8aae18c5987a3ff0f456d1c74c5df02404ab2510b8489`; it binds 418 primary
payloads totaling **20,834,592,704 B** across 38 chunks. Every chunk retains native float RGB,
preuint8 camera float RGB, camera uint8 RGB, both evaluator resizes, three five-class logit fields,
and three argmax fields.

The analysis was recomputed from those local copies. Local `MST1_RESULT.json` has SHA-256
`9ff459805e6bc880cd94d0fed0ecc724e2fa4e585571eb3c675d175b891ed6f8`.
`ATTRIBUTION_MASK_MANIFEST.json` has SHA-256
`70e1dce36feb561bfaac9c261d748fa14b9bc946319b0133ffe4e5368e9aefc3` and binds 33 packed
n600 support, state, survival, earliest-stage, and gross-transition masks totaling 486,604,800 B
(14,745,600 B each). The exact archive, frozen SegNet, upstream evaluator/modules, capture source,
and instrumented renderer source were also copied into the local provenance-source store with
byte-identical hashes. The pre-reroute SSD originals remain intact; nothing was moved or deleted.

## RECALL EVIDENCE

Before building or adjudicating, the recall searched the full `.omx/research` corpus by content,
the canonical research index, the `sub015_DAG_*` FEED blocks, current task/status stores, and the
canonical equation catalog. Queries included `manufactured Seg`, `round-trip`, `render`, `R_surv`,
`resize`, `uint8`, `subpixel`, `camera placement`, `deblur`, `antialias`, `SegNet logits`, and the
DX2 archive/token hashes. `tools/list_canonical_equations.py --json` was searched with the same
mechanism terms.

Beyond the charter's named MS9/RT1 seed, the search recovered RT2's exact current-family operator
ordering and kernel, RN1's receiver-blind symmetry result, MP1/RA1's camera-grid decomposition, and
the current RI1/NI1/CB2 terminal handoffs. That changed the plan materially:

1. The instrument captured the actual `384x512 native -> bilinear camera lift -> uint8 -> evaluator
   bilinear downsample` ordering instead of the charter's simplified one-resize shorthand.
2. It retained native/preuint8/uint8 RGB, logits, and argmax fields rather than transferring RT1 or
   RT2 stage fractions across vehicles.
3. It did not rerun the ancestor's closed generic deblur, blur, sharpen, antialias, or undirected
   dither ladders. Those are mechanism negatives, not DX2 stage magnitudes.
4. It kept the nine-pixel CPU/CUDA difference as an explicitly unseparated terminal stage rather
   than pretending CPU and CUDA values were identical.

The bounded index/DAG/task-ledger search did not find an existing current-DX2 progressive n600 field
set. MS9's queued fire order was therefore consumed, not duplicated.

## Verification and boundaries

- `ruff check`, `python -m py_compile`, `git diff --check`, two genuine
  `tools/review_tracker.py mark-file` passes, and the targeted
  `check_no_measure_and_discard_payload` audit passed.
- Capture checkpointing admitted at most 16 pairs at a time, after all eleven payloads were fsynced,
  hashed, and manifested. Incomplete directories are preserved, never deleted.
- The migration rehashed all 20.8 GB of primary fields at source and destination, verified every NPY
  shape/dtype, and checkpoint-admitted all 38 chunks locally. The analysis then copied G/L/A
  byte-identically into the local store and admitted all masks as one atomic directory.
- A completed replay rehashed the 418 local primary payloads, all copied source fields, and all 33
  masks, then rechecked the gate, stage, class, mask-count, and gross-transition identities. Its
  `COMPLETED_VERIFICATION.json` is SHA-256
  `283cfbe02de02046d3560965c4471ea09cc4c5a59a65cbd81ef987f609801ccd` and binds analysis source
  SHA-256 `abd4382053bf8db6a935afc19c7d76fab51d854f5096e3f089869ea3ca2a297d`.
- Measured: the exact gates, n600 stage/class charges, gross transitions, CPU/CUDA terminal
  disagreement, and byte-equivalent ceilings.
- Not measured: any candidate cure, pose/rate response to a cure, any changed archive, exact
  contest-CPU/CUDA score for a new object, or a causal split inside the frozen SegNet/head observer.
- No upstream file, shipped DX2 runtime, jo1 r9 directory, Modal resource, Metal resource, or staged
  index was modified.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN / the next DX2 Seg-cure builder; consumer store:
  `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/renderer_joint_cure_r1/`;
  fire trigger: the sole n600 scorer lane is free **and** a retained byte-closed same-length candidate
  exists. Optimize the existing renderer/token values jointly through native, R, and uint8 with the
  frozen head in-loop; require an exact 180,368-byte archive or report its measured byte delta; preserve
  pose and the 6,918 beneficial final corrections; validate the nine-pixel CPU/CUDA merge on the
  contest-CUDA component axis. Do not reopen a generic camera-operator ladder.

## LIVE-HYPOTHESES

- A same-length joint renderer solve can trade some of the native stage's 28,602 gross breaks against
  its 6,281 gross repairs and lower terminal error without adding archive bytes. This is plausible
  because 78.7093% of final manufacture first appears at the controllable native-render observation.
- Lane-aware admission inside that joint solve may pay if it uses frozen-head collateral constraints,
  not a Lane-only repaint. Lane manufacture is 41.99x denser than the body average, but LQ1 proves
  unconstrained targeting spills catastrophically into other classes.
- A margin objective shared across CPU and CUDA may absorb the nine-pixel terminal disagreement. This
  is plausible as a validation fold, not a separate arm, because only two final manufactured pixels are
  first charged there.

## DEAD-ENDS

- The registered claim that the float bilinear round trip manufactures at least 50% is closed on this
  DX2 instance: it carries 18.7503%; the native-render-plus-frozen-head observation carries 78.7093%.
- The registered distributed-loss falsifier is also closed: one stage exceeds 30% by a wide margin.
- Generic deblur, blur, sharpen, antialias, or undirected-dither ladders remain closed as direct
  follow-ons: RT2/RN1 measured them dead on the ancestor, and MST1 found that R and uint8 are net repair
  stages on DX2. A current-body joint constrained solve is a different formulation.
- A direct CPU/CUDA terminal cure is closed as its own arm: frozen device/head/argmax behavior is
  inadmissible to edit and the ceiling is only 2.5 B.

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600], archive SHA-256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674; MST1 made no score claim.
