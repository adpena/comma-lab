# EB2 — high cell agreement does not make the temporal residual cheap

`[no-triality] [p0-ledger-ok]` · owner `ddm_eb2` · `research_only=true` · `score_claim=false`.

**The tested pose-warp residual costs 542,248 B; persistence costs 536,997 B;
the same coder describes the partition from scratch in 357,123 B.** All are real
retained n600 payloads with identical twin encodes and exact conditional parse-back.
The pose-warp formulation does not earn a build. The exact frontier did not move.

**Literal charter status: BLOCKED_ON_RECEIVER_PREMISE.** Move 40 does not transmit
six pose scalars, and its receiver does not hold the scored argmax partitions.
The numerical work below grants cached scored partitions and separately charges a
historical pose/calibration payload. It is a conditional diagnostic, not an
implementation of the charter's claimed receiver. The generator door remains
undecided; no temporal family is closed.

New measurements use **[macOS-CPU advisory; cached n600 partitions; scorer-free
exact bytes]**. Bounds use **[closed-form; exact real-count reference classes]**.
No neural scorer, training, video decode, GPU, remote evaluation or contest run occurred.

Let `W = /Volumes/APDataStore/pact/ddm_eb2_conditional_bound` and
`E = .omx/research/ddm_eb2_20260910`. Code and memo are in `W/worktree`;
bulk and complete per-frame checkpoints are in W. All byte units below are B,
not KiB or projected population prices.

## Measured coder comparison

One fixed codec: five binary residual masks, alternating row-major run lengths
as unsigned LEB128, then zlib level 9 across all 600 frames of each class.
Each stream has an 11-byte ERL1 header carrying class, frame count and geometry;
zlib framing is counted too. A mask identifies the positions whose predictor
is wrong and whose true class is c. Five disjoint masks reconstruct the complete
five-class target; a binary disagreement mask alone would omit replacement labels.
The first frame uses an all-Road predictor and is charged in every stream.

| Conditioning / complete encoded object | Real B, n600 | B / frame | Difference from 119,754-B tail |
|---|---:|---:|---:|
| Pose-warped previous cached scored partition | 542,248 | 903.747 | +422,494 |
| Same, plus extra pose and calibration | 549,472 | 915.787 | +429,718 |
| Unwarped previous cached scored partition | 536,997 | 894.995 | +417,243 |
| Partition from scratch, same codec | 357,123 | 595.205 | +237,369 |
| Incumbent move-40 tail, parsed from its archive | 119,754 | 199.590 | 0 |

Sources: `E/SUMMARY.json:8,171,334,499`; `E/SOURCE_AUDIT.json`.
Encoder/decoder: `experiments/ddm_eb2_conditional_bound.py:178-217,315-345`.
These are **achieved D=0 instance code lengths conditional on the named inputs**,
not minima, not population worst-case ceilings, and not replacement archives.
The incumbent tail also depends on its paid HPAC/renderer/carrier models and codes
a different object, the token field. The comparison is a scale benchmark, not
an apples-to-apples marginal-tail entropy or total archive comparison.

| True class of correction | Warped residual cells | Warped residual B | Persistence B | From-scratch B | C / N minimax lower B at D=12,540, single-class suborbit |
|---|---:|---:|---:|---:|---:|
| Road | 667,209 | 249,176 | 247,003 | 35 | 0 / 0 |
| Lane | 416,103 | 174,473 | 172,062 | 193,903 | 42.125 / 42.125 |
| Undrivable | 268,956 | 58,350 | 57,763 | 81,525 | 0 / 0 |
| Movable | 114,298 | 42,647 | 42,567 | 60,408 | 0 / 0 |
| MyCar | 154,789 | 17,602 | 17,602 | 21,252 | 0 / 0 |
| **Total real coder** | **1,621,355** | **542,248** | **536,997** | **357,123** | **Do not sum class floors** |

Sources: `E/SUMMARY.json:384-491,571-682`; corresponding persistence/from-scratch
sections at `:58-145,221-308`. Each single-class suborbit varies only groups whose
residual alphabet is {match,c}; every row is granted the entire global D budget.
They are nested subpopulations, not an additive allocation of a shared error budget.
The all-class floor below includes all allowed groups. Road is the implicit class
in the from-scratch control, so its empty mask costs only framing/compressed zeros.
Do not read these codec-dependent class costs as intrinsic class entropies.

## Spatial counts and the receiver-information question

Population: pair indices 0..599, 600 x 384 x 512 = **117,964,800 cells**;
`selection_mode=all_scored_pairs_0_through_599`. All bootstrap-inclusive counts use
that denominator. Adjacent scored-partition transitions number **599**, with
117,768,192 cells; their statistics exclude frame 0 rather than inventing a predecessor.

| Predictor | Wrong cells, all 600 | Wrong cells, 599 transitions | Agreement, 599 transitions | Residual boundary edges | Residual edges / GT edges |
|---|---:|---:|---:|---:|---:|
| Persist | 1,620,365 | 1,467,894 | 98.753573% | 2,153,408 | 132.9387% |
| Historical calibrated pose warp | 1,621,355 | 1,468,884 | 98.752733% | 2,171,934 | 134.0824% |
| GT partition reference | — | — | — | 1,619,850 | 100% |

Sources: `E/SUMMARY.json:168-220,309-380,472-478`; counts and all 25 directed
predictor-class/GT-class histogram entries are in those sections and in
`E/FRAME_COUNTS.jsonl:1-600`. Each line points by hash to `W/frames/<id>.json`.
All 600 original scored maps differ from GT in **12,540 cells**, whereas the
actual move-40 stored tokens differ in **17,631** (`SUMMARY.json:497-498`).
The 12,540 count is now verified on the cached **macOS instrument**; it does not
replace the distinct rounded T4 receipt or establish a new T4 integer count.
The shard report's inherited CL2 token table is not the move-40 token census.

Residual symbol E is 0 on a match and c+1 on a correction to c. Its perimeter
counts each unequal horizontal/vertical E adjacency once, without exterior or
time edges. Binary support perimeter and E's labeled perimeter are different
objects; the table uses the labeled residual perimeter throughout. Components
use four-connectivity, separately for each nonzero residual class.
The warped residual has **142,432 nonzero components**: Road 65,221; Lane 44,047;
Undrivable 17,270; Movable 10,400; MyCar 5,494. Its matching background has 4,202
components, reported separately (`SUMMARY.json:336-343`).

For the warped residual, **1,523,051 / 1,621,355 = 93.9369%** of disagreements
lie in rows 128..319. A geometric GT-boundary band of Manhattan radius 3 contains
**1,173,190 = 72.3586%**; its complement contains **448,165 = 27.6414%**.
The undilated GT boundary contains **811,849** mismatches. Thus the row band is
not the same object as an actual boundary neighborhood (`SUMMARY.json:331-383,475-476`;
producer `:114-122,268-286`). The class-pair histogram includes all off-diagonal
pairs, not only Road/Lane. Lane+Movable account for 530,401 residual cells, not
most of the residual; Road alone accounts for 667,209. The bootstrap is included
in these totals and retained individually as frame 000.

**What fraction of the description does the receiver already hold? UNKNOWN.**
The diagnostic predictor matches 98.752733% of transition cells, but this is a
cell-accuracy fraction, not a fraction of information. Under this one actual
codec, conditioning increases bytes by **51.8379%** relative to its own
from-scratch control; even persistence increases them by 50.3675%. High uniform
area agreement leaves a fragmented, expensive residual. No source distribution
or optimal conditional code is identified by these measurements, so neither
mutual information nor an intrinsic receiver-held fraction is reported.

## Receiver custody and geometry boundary

The archive is move 40, SHA-256
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`, 180,233 B.
Its single ZIP member has 180,133 B: header 14; HPAC 11,911; semantic 29,862;
carrier 18,592; tail 119,754. Parsing the current archive resolves the staging
receipt's older 18,595-B carrier (`E/SOURCE_AUDIT.json`).

The retained runtime's `runtime/f26_inflate.py:472-480` materializes a carrier
and calls `unpack_semantic_pose`; `cpr1/inflate.py:28-29,245-250,335-339` constructs
**12 image-basis coefficients per pair**, with 12 x 3 x 24 x 32 basis values,
and contracts coefficients against that basis to produce frame 0. The name
`semantic_pose` does not mean it returns six PoseNet target scalars. No six-value
pose extraction exists in this consumed path. Likewise, the receiver decodes
tokens and renders RGB; the cached SegNet argmax is an evaluator output. Using it
as already-available receiver state would require an additional legal mechanism.
Scorer weights are not supplied by this diagnostic or licensed as free decoder code.

The actual parsed raw was cold-stored during this arm. Its `.MOVED.json` resolves to
`/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_compose39_price/parseback/0.raw`.
Both **3,662,409,600 B** and SHA-256
`c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5`
were independently verified (`E/MOVED_RESOLUTION.json`). All six scorer-cache
shards bind that raw hash and the DALI GT cache hash. The argmax NPY hash is
`50abe278279af3909baf1307389037b5a102250a2d58580e60c9f4135a927667`.
The GT NPY is the EB1 retained copy bound to DALI cache SHA
`a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994`.
`W/INPUTS.json` hashes these exact inputs; `SOURCE_AUDIT` retains the complete
source/receipt chain. No GT video conversion was performed.

To make the requested diagnostic concrete, the warp reuses XI1/XI2's historical
class composite with their retained **7,200-B fp16 pose table + 24-B fp64 global
calibration**. They are copied into W/inputs, hashed, and charged separately;
they are not silently imported into move 40. No new calibration or per-frame
fitting occurred. Calibration is approximately `(s_t=-0.00322471, s_r=0, pitch=-0.01)`;
its exact bytes control the run. The zero rotation scale means this tested
formulation is translation based. It is not a general six-DOF motion test.

The generic NumPy homography uses source-independent camera constants
fx=fy=910 at 1164x874, rescaled to 512x384, centered principal point and camera
height 1.22. With camera-frame t = s_t*(pose[2],pose[1],pose[0]),
R = exp(s_r*pose[3:6]), and n=(0,-cos(pitch),-sin(pitch)),
H = K*(R-t*n^T/1.22)*K^-1. The XI1 composite warps classes 0/1/3 by the ground
homography, class 2 by rotation only, and keeps source MyCar fixed. Out-of-bounds
samples retain the colocated predecessor; they are counted, never excluded from
the denominator. Nearest-neighbor ties use NumPy rint (`producer:77-111`).

PoseNet output units/axes and the association of a within-pair pose with motion
between consecutive scored last frames remain hypotheses. The evaluator consumes
nonoverlapping two-frame sequences (`upstream/frame_utils.py:9,118-147`); a
within-pair vector is not automatically the motion across the two-frame scored
spacing. The historical calibration absorbs this approximation but does not
prove it. These limitations prohibit an actual-receiver or physical-motion verdict.

## Conditional lower bounds, with both reference classes defined

Fix the diagnostic predictor sequence Z and a residual skeleton outside eligible
sites. This extra skeleton is a **mathematical grant**, not free archive content.
Residual alphabets at each site exclude predictor_class+1, since that would be a
noncanonical replacement by the already-predicted class. Mapping allowed E to P
is one-to-one and preserves Hamming distance for fixed Z.

Eligible sites are two cells inside the image, on `(y+2*x) mod 5 = 0`; distinct
sites have Manhattan distance at least three. Following EB1, group sites by
frame, 64-row band, four-neighbor label histogram, unsaturated-neighbor histogram,
and simple-label set in a fixed 3x3 ring. **Also condition on the predictor label.**
Only groups containing multiple observed simple labels are mutable. This
preserves residual area per class, boundary-cell union, each class-pair edge
count and each class's component count for every permutation; disjoint affected
neighborhoods make the local invariants additive. Predictor strata preserve the
allowed residual alphabet. It does not assert invariant GT geometry or physical
video realizability (`producer:125-175`; full n600 alternative recount at `:287-305`).

- **C, conservative:** greedily pair distinct labels inside each group, leaving
  other sites fixed. Every disjoint pair is an independent two-cell swap.
  K = **28,994** and |C| = 2^K.
- **N, local-profile class:** allow every permutation of the observed label
  multiset in each group. |N| = product_g m_g! / product_c n_gc!.
  There are **12,640 groups**, **104,811 mutable cells**, q_max = **2**,
  and log2|N| = **61,369.74037131264...**. C is a subset of N.
- **F_Z, full same-residual-profile class:** all allowed residual sequences
  sharing those per-frame geometry/stratum statistics, with predictor Z fixed.
  C subset N subset F_Z. Its exact size and physical source law are unknown.
  The N minimax lower bound transfers by covering-number monotonicity, not by
  pretending a ball restricted to N bounds the larger F_Z ball.

For a stated finite A and **every** source member within hard total radius D,
R_wc(A,D) >= log2|A| - log2 max_y |A intersect B_H(y,D)|.
The maximum ranges over all reconstruction centers, including off-class centers.
For N, B_2(M,D) is a valid ball upper bound. For C, the product Chernoff bound
V_C(D) <= z^-D*(1+z^2)^K includes midpoint centers via 2z <= 1+z^2.
The exact EB1 integer/rational bound engine is reused at
`experiments/ddm_eb1_entropy_bound.py:222-299`; the new group-count consumer is
`experiments/ddm_eb2_conditional_bound.py:220-229`. Certified floors round downward.

| Total hard D, cells | C minimax lower B | N raw counting lower B | N minimax lower B, strengthened by C | EB1 C / N minimax B |
|---:|---:|---:|---:|---:|
| 6,270 | 1,833.000 | 3,391.375 | 3,391.375 | 2,467.625 / 5,006.000 |
| 12,540 | 894.125 | 750.250 | 894.125 | 1,404.500 / 2,100.250 |
| 25,080 | 47.750 | 0 | 47.750 | 255.250 / 255.250 |

Source: `E/SUMMARY.json:503-570`; EB1's memo and pinned profile result.
N's raw converse may also lower-bound expected prefix length under a uniform N
source and hard per-source distortion. The inherited C minimax floor **does not**
transfer to that uniform-N expected-length claim. The reference classes changed
from EB1's target geometry to residual geometry; the ratio is not a measured
conditional-information saving. EB1 already granted a source-specific skeleton,
so calling its result wholly unconditional is inaccurate.

At D=12,540, the small minimax floor and the 542,248-B achieved instance code are
**not a proved [floor,ceiling] interval for the incumbent or the natural-video
optimum**: their quantifiers, conditioning, distortion contracts and objects differ.
The only universally justified pointwise lower bound supplied here is zero bits;
that does not claim a legal zero-byte generator exists. A weak floor proves
neither that savings exist nor that the current tail is near optimal.

## Verdict and prior-law falsification

**INSTANCE:** this historical calibrated warp adds 990 disagreements and
5,251 actual residual bytes to persistence. It does not improve the tested
predictor/coder pair on the current cached output partitions.
**FORMULATION:** this RLE+zlib residual channel exceeds the 119,754-B stop threshold;
no follow-on build is fired from it. The <=40,000-B build trigger does not fire.
The numerical stop is recorded even though the literal actual-receiver test is
blocked. No family-level or general six-DOF warp closure follows.

The >=90% cell agreement prediction holds, even for persistence. The predicted
10–25% residual edge ratio fails: it is 134.08%. The 20–60 KB simple-coder
prediction fails: it is 542,248 B. The 200–800-B natural-floor prediction holds
only for the weaker raw N converse (750.25 B); the stronger valid N minimax floor
is 894.125 B. None of these tiny floors decides the generator door.
The instruction to infer that the generator must describe boundaries from scratch
from one losing simple coder is a bound-direction error, so that inference is
explicitly refused rather than banked as a negative result.

## RECALL EVIDENCE

Read the complete charter/common contract, PROGRAM, matching AGENTS/CLAUDE,
operating handoff, live hot state/pointer, nearby operator directives and actual
runtime/evaluator source. Predecessor checkpoint lookup found no ddm_eb2 record;
no ddm_eb2 conflict appeared in lane/task searches. No scorer job was requested.
The common contract's old frontier paragraph is superseded by the live pointer.

`E/RECALL_SEARCHES.json` preserves exact queries, counts, hashes and W/recall
outputs: research memos/arm receipts by content (1,372 lines); index/DAG FEEDs
(808); design/config/SPEC surface (33); task/P0/lane ledgers (5). Query families:
`conditional.*partition|previous.partition|pose.warp|ground.plane.*homography|temporal.*transport|side.information`;
index/DAG also `partition|side.information|screw|entropy`. The canonical-equations
CLI export contained **483 entries** and was searched for the same mechanisms.

Beyond the charter seeds:

1. XI1/XI2 already implement the class-composite pose warp and explicitly price
   pose/calibration payloads. The XI2 build memo says READY_TO_FIRE, but original
   ledger recall led to `/Volumes/APDataStore/pact/ddm_xi2_20260812/FULL_SCALE_RESULT.json`:
   the actual full-scale result is **116,860 vs 116,716 B, +144 B**, formulation
   closed on its older token vehicle. This prevented treating the build memo
   as live work or claiming this generic warp as a new invention. Its learned
   token result does not substitute for the current scored-partition measurement.
2. `tools/measure_pose_warp_dseg.py:22-39,125-138` labels pose axes, units and
   adjacent-pair approximation as assumptions and fits three global scalars.
   This changed the diagnostic to count the calibration rather than hide it in
   free geometry code. The existing s_r=0 is disclosed.
3. `partition_temporal_transport_amortization_jitter_bound_v1` and the earlier
   transport/jitter memos separate predictable area from costly boundary changes.
   This required a real from-scratch control with the same coder and prevents
   interpreting 98% cell agreement as 98% description already received.
4. The current runtime and six scorer-cache shards exposed the actual receiver
   mismatch and the inherited CL2 token-table provenance. These changed the
   verdict to a granted-information diagnostic with an explicit receiver blocker.
5. GS3 Correction 3 already rejects achieved upper bounds promoted to entropy
   floors. Its later EB1 addendum nevertheless brackets the tail by a population
   floor and instance tail size. This memo preserves the quantifier/conditioning
   mismatch instead of copying that bracket into a new generator verdict.

The prior memory preference for real serialized coder prices and explicit
population denominators was applied; no old n32 price was transferred. Scoped
negative: these searches did not find a current-archive pose6 extraction,
receiver-visible scored-argmax mechanism, or a verified conditional natural-video
source law. This is not a claim of global nonexistence.

## Verification, retention, registration and handoff

Final Python source passed **two visible review-tracker passes and Ruff**.
Real-map functional checks cover zero-motion identity and RLE inversion; all
600 measured frames verify both C/N geometry invariants and full five-mask joint
reconstruction for all three predictors. All fifteen primary/repeat stream pairs
match exactly. A complete resume revalidated saved payload hashes, skipped all
600 completed count stages, and reproduced the same result and code bytes.
An independent exact-factorial recount verifies all 12,640 multinomial factors;
independent floating arithmetic agrees with every certified floor, while the
rational engine remains the verdict authority. See `E/VERIFICATION.json`.

The initial verification glob encountered macOS AppleDouble metadata; the audit
was corrected to explicit 000..599 enumeration. The measurement and resume
already used those explicit IDs. No payload was deleted or repaired. This was
an audit-enumeration issue, not a population or payload corruption finding.

One numerical process, native thread caps of one, APDataStore storage admission
and a 16-GiB reserve were used. `W/LAUNCH.json`, `INPUTS.json`, per-frame JSON,
retained predictors/residuals/alternate partitions, primary/repeat `.erl` files,
raw RLE streams and logs preserve the run. Atomic temporary files are promoted
only after complete writes; every partial stage is restartable. All bulk is
required retained evidence; cleanup is KEEP, with no orphan raw decode or
candidate sweep. The isolated source checkout is reproducible from its pinned
HEAD plus the landing patch and is certified as such in `W/CHECKOUT.json`.
To replay after landing, use that original checkout HEAD plus the patch; INPUTS
intentionally checks recorded source and Git identity. No live-tree/index,
upstream, protected-file, publication or frontier mutation was performed.

`conditional_partition_description_bound_v1` was registered through the canonical
API and query-verified **in isolation**. Its callable is the already-proved EB1
counting engine actually consumed by this measurement; EB2 adds the fixed-predictor
residual class construction and real conditional codec. The narrow registration
event is `E/equation_event.jsonl`. No new mathematical theorem or novel codec is
claimed. `E/task_events.jsonl` and `E/FIRE_ORDERS.json` carry the typed blocker and
harvest/rebind actions. Live stores are not silently claimed updated.

Solver hooks: sensitivity, bit allocation, autopilot dispatch and a numerical
scorer-posterior update are N/A because this arm builds no actuator/candidate and
measures no score. No production Pareto floor is installed. The research law and
explicit receiver/quantifier distinctions are the consumer output and probe
disambiguation; MAIN's GS3/task/equation harvest is the declared integration point.

**Completed:** all-n600 diagnostic warp, actual residual geometry and class-pair
counts, explicit C/N classes and exact bound constants, one real coder with twins,
full parse-back/resume verification, source/moved-file custody and isolated law
registration. **Not completed or claimed:** the literal current-receiver conditional
bound, its information fraction, a natural-video entropy law, an actual receiver
construction, a universal temporal closure, a new archive or any new score.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer stores** `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`, `.omx/state/canonical_equations_registry.jsonl`, `.omx/state/canonical_task_status.jsonl`; **fire trigger:** harvest the verified isolated landing. Import this memo/code and its narrow equation/task events, preserving the receiver blocker and formulation-only refusal.
- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store** `/Volumes/APDataStore/pact/ddm_eb2_conditional_bound/RECEIVER_SIDE_INFORMATION.json`; **fire trigger:** harvest this source audit. Bind actual decoder-visible state and a charged or byte-derived pose mapping across adjacent scored frames before issuing a corrected generator-door charter. Do not rerun this unchanged oracle diagnostic.

## LIVE-HYPOTHESES

- A better conditional representation may compress the residual: the measured
  codec fragments temporal changes into class masks, and its losing length is
  only an achieved upper bound. No applicable optimality certificate exists.
- The existing paid carrier may expose useful motion information through an
  explicit legal mapping. It shapes PoseNet outputs, but its 12 coefficients
  have not been proved equivalent to a six-value inter-frame pose statistic.
- A useful natural-video conditional floor may exist beyond the frozen-skeleton
  C/N orbits; these classes deliberately retain very few degrees of freedom.
  Its source law, actual receiver information and physical reachability remain owed.

## DEAD-ENDS

- This exact historical calibrated class warp plus RLE/zlib residual channel:
  +990 wrong cells and +5,251 B versus persistence; 542,248 B exceeds the tail.
- Treating move40's 12 image-carrier coefficients as six transmitted pose scalars,
  or scorer-cache argmax as decoder state: the consumed runtime does neither.
- Converting high cell agreement into a fraction of description already received:
  98.752733% transition agreement coexists with 51.8379% more bytes in this codec.
- Promoting one achieved code length into a lower bound, or a population floor
  into a pointwise/marginal-tail floor: the directions and quantifiers do not match.
- Calling old XI2 READY_TO_FIRE: its later n600 result already records +144 B
  and a formulation closure on its old vehicle. Do not re-fire the stale build memo.

Own-vehicle frontier, **unchanged by EB2**: **S 0.13763861019288715 @ 180,233 B
[contest-CUDA T4 n600]**, archive SHA-256
`986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`.
