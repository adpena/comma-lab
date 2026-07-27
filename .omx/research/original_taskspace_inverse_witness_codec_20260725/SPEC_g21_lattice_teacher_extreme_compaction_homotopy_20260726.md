# G21 executable design — lattice-teacher to selected-solution extreme-compaction homotopy

Date: 2026-07-26  
Lane: `lane_g21_lattice_teacher_extreme_compaction_spec_20260726`  
Status: append-only research specification; `research_only=true`; no run, scorer call, archive mutation, dispatch, score, promotion, or pointer claim  
Ownership: G21 owns only this specification. It does not edit G17/G19/G20/root-owned files.

## 0. Result and non-claim

This is not a new codec paradigm. It is the executable composition of the
already-built lattice, quotient, factorization, receiver, population, entropy,
and whole-object pricing machinery around one corrected objective:

> Compile the cheapest complete description of one evaluator-equivalent
> **solution**. Do not store the inverse problem by default, do not stop at the
> first feasible lattice member, and do not optimize raw residual size when the
> charged object is the final coded archive.

The historical lattice objects are our own high-information encoder teachers.
They may expose existence, strata, recurring factors, action proposals, and
falsifiers. Their bytes, frames, tensors, factors, selections, scorer products,
and target-dependent state never become candidate payload or decoder state.
Every candidate description is newly compiled from the current original
candidate lineage and remeasured as a complete object.

The exact control that anchors this design is DDM MS2R R2:

```text
C1 receiver-closed exact control     409,526,925 bytes
MS2R R2 q4/q8 receiver-closed object 291,205,400 bytes
physical saving                      118,321,525 bytes
fractional saving                    28.8922456%
d_seg                                0.001159998575846354
d_pose                               0.01663315449034709
```

Its receipt literally types the 291,205,400-byte archive as
`headline.stored_problem`, but its physical payload is a complete selected
q4/q8 quotient-plane solution consumed by the V10 production receiver. That
object is a teacher/control, not the representation target: the homotopy must
describe the selected solution more cheaply than storing those dense planes.

Pointer delta from this specification: **zero**. No fixed per-axis distortion
or byte threshold is an admission rule. Admission uses the complete nonlinear
score against the live G17 production envelope and current effective frontier.

## 1. Frozen evidence identities and authority

A future implementation MUST rehash these inputs before reading derived fields.
Path identity alone is insufficient. Missing SSD bytes or SHA drift refuses the
stage; it does not substitute a local copy, old worktree, or approximate row.

### 1.1 MS2R R2 — exact receiver-closed capped solution/control

| object | exact identity | authority and use |
|---|---|---|
| canonical receipt | `.omx/research/ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/receipt.json`, 92,594 bytes, SHA-256 `03cd9aabc1275c49c983631dd547e7497f8fe95804a9bfd7a24c5d61e9a81d25` | `[macOS-CPU frozen-scorer advisory]`; research only; pointer unmoved |
| C1 control archive | `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/archive.zip`, 409,526,925 bytes, SHA-256 `e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42` | exact receiver-closed q1 control |
| selected MS2R archive | `/Volumes/VertigoDataTier/pact/ddm_ms2r_tolerance_capped_solve_r2_20260724T181428Z/stage_checkpoints/04_candidate/archive.zip`, 291,205,400 bytes, SHA-256 `e3d0581ff4a3f475057e77e530374dad444b640a049b058cd66b37563534773e` | exact stored-member archive; teacher/control only |
| selected receiver packet | same stage directory, `0.bin`, 291,205,292 bytes, SHA-256 `daf1e1db6314e8cdbf63347afa35899e9891e3068428d42dc5a2fca235bb5295` | actual V10 receiver input |
| typed config | `.omx/research/configs/ddm_ms2r_tolerance_capped_solve_r2_20260724.json`, SHA-256 `04d496a79ae10486e4ee5f2edc0e2ec0a0269523fbb01a830069e4d767339486` | finite-family selection contract |
| solver module | `src/tac/optimization/ddm_ms2r_tolerance_capped_solve_r2.py`, SHA-256 `97bc41e37ee64f805ddbd40620bcd023065335b5268cb6723516ee2565a117fe` | exact q4/q8 dynamic program |
| materializer | `tools/run_ddm_ms2r_tolerance_capped_solve_r2.py`, SHA-256 `9ec93cf99c1de2e263adb80459dc9131284311b00fcc6fbded1d46b5a04fcee7` | record selection and production archive construction |

The finite dynamic program chose 208 q4 and 392 q8 pair records with exactly
136,839 Seg errors under its pre-registered cap. It is exact only within that
binary per-pair family. Its measured flags are quotient coordinates active,
scorer metric active, and Pose tube active; typed subproblem alternation, typed
block atlas, and per-dimension effective quanta are false. RAW won all 50
isolated stream races, and the 291,205,320-byte sum of separately framed stream
rows is not a counted alternative because no corresponding one-object receiver
container was materialized. Those limits are part of the control.

### 1.2 MS1 — low-distortion lattice existence and SENSE teacher

| object | exact identity | authority and use |
|---|---|---|
| immutable receipt | `.omx/research/ddm_ms1_min_description_lattice_solve_20260723T233549Z/receipt.json`, 8,624 bytes, SHA-256 `546a7fddb0225edb15b2254ab73e362758b7b0f244e4ff39cb7bfef25f779098` | immutable historical authority |
| current ingest | `.omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json`, current SHA-256 `1b7063a44574b0839ede08c807f348ad417be0492ac32d68634b124b9c2b1e97` | repository bridge; must still bind immutable receipt |
| dense teacher raw | `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/capstone_submission/inflated/0.raw`, 3,662,409,600 bytes, SHA-256 `31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b` | encoder-only source; never candidate/runtime payload |
| target/scorer cache | `experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz`, 2,064,464 bytes, SHA-256 `e41718a047b77b9072828e72f8cbffa0f5ef7ddf462c7ef4329d997ace89de50` | encoder-only target evidence; forbidden payload |
| pair SENSE rows | `/Volumes/VertigoDataTier/pact/evidence/ddm_ms1_min_description_lattice_solve_20260723_final/sense/pair_rows.jsonl`, 1,170,365 bytes, SHA-256 `276dde04cc0d6f4f4df1bfb1c7544f997800da189d49e789d00f87e699073803` | 600 historical pair telemetry rows |
| SENSE factorization | same root, `sense/factorization.json`, 5,232 bytes, SHA-256 `1c798be26b6e8aeb4b259d9e56beedd0cd99f5e5d6b5c2c6ba59f1a0ee03b450` | proposal surface only; zero factors were representation-distilled |
| measurement tool | `tools/measure_ddm_ms1_min_description_lattice_solve.py`, historical/current SHA-256 `b29b8f53aed9bff917054d45fe0678bbb11cd1ba4a3da941a2954b2bb7e50a41` | exact historical implementation identity |

MS1 scored the unchanged canonical member at
`d_seg=0.0001519690619574653`, `d_pose=0.00010184327939026322`, with
600/600 argmax and Pose6 identity between canonical and selected outputs. Its
best previous-frame conditional count was 731,622,325 bytes, a 12,986,636-byte
(1.744088%) improvement over the zero-origin residual. It is **not** a
materialized archive, not a global lattice optimum, not an MDL optimum, not
receiver closed, not Pose-active during selection, and not a candidate. The
589,824-vector saturated integer kernel and 1,200 local-CVP proposals per
conditioning expansion produced zero changed pairs. This falsifies only that
local formulation and confirms the existence of a low-distortion solution
surface.

The historical factorization has eight SVD rows and six factors above the
one-byte numerical coder floor, but the active-set fields are nearly degenerate
and all representation roles remain blocked without a measured per-stratum
SKELETON-versus-FIBER coder race. Historical singular values and loadings can
rank proposal families; they cannot create payload, factors, or byte credit.

### 1.3 V7 receipts — bounded factor hierarchy signal, not the n600 control

- n64 receipt SHA-256 `8db93c4ef90e6d7f29943b4334a0d441f5cfbe226bf68fceb7ed03e59730970b`:
  exact-all 43,112,153 bytes; q4 27,479,944 bytes.
- n256 receipt SHA-256 `d68f1d9ead9401173160b8cc4ec7fb9d49753a6bb0f298af23de293bc28d4274`:
  exact-all 171,332,654 bytes; q4 108,637,789 bytes.
- Cross-receipt `.omx/research/ddm_v7_solved_plane_tolerance_waterfill_603_613_20260722T102423Z.receipt.json`, SHA-256 `64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb`.

V7 never measured full n600. It contributes only the recurring exact-stream
hierarchy (Undrivable, MyCar, Road dominate before Boundary, Movable, Lane) and
the formulation-scoped exact-residual rate wall. It must not be relabeled as
the 28.8922456% endpoint or as population evidence.

### 1.4 Frozen evaluator identities

- `upstream/evaluate.py`: SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`.
- `upstream/modules.py`: SHA-256 `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`.
- SegNet weights: SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.
- PoseNet weights: SHA-256 `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.

Frozen component scorer measurements are advisory. Only the exact contest
oracle on exact archive bytes and proper contest hardware may create an exact
score row, and no such invocation is authorized by this specification.

## 2. Selected-solution and active-inflate objective

Let `z` be the irreducible video-specific counted statistic, `D` the generic
deterministic decoder/inflate program, and `Y=D(z)` the complete received uint8
population. Let

```text
Phi_seg(Y)  = argmax SegNet(R(Y.frame1))
Phi_pose(Y) = PoseNet(R(Y.frame0), R(Y.frame1))[:6]
B(z,D)      = exact bytes of the physically scored archive.zip
```

If the packaging surface does not charge `D`, generic decoder code contributes
zero to `B` but remains hash-, determinism-, runtime-, and portability-bound. If
any decoder executable, table, dependency, or generated code blob physically
ships inside `archive.zip`, its exact compressed bytes are charged. The encoder
must minimize shipped decoder-code bytes whenever they exist; calling a blob
"code" does not make it free.

The candidate problem is

```text
choose (z,D,Y) with D(z)=Y
minimize S(Y,z,D) = 100*d_seg(Y) + sqrt(10*d_pose(Y))
                    + 25*B(z,D)/37_545_489
subject to source lineage, exact parse-back, determinism, runtime, and custody.
```

The decoder is active, not a passive parser. Subject to the contest source
boundary and runtime wall, generic video-independent code may deterministically:

- generate fixed bases and analytic dictionaries;
- expand tensor, time, population, and semantic factors;
- entropy-decode and invert transforms;
- synthesize, solve, repair, project, iterate, and bounded-search;
- select a canonical parameter gauge or fill exact receiver-null coordinates;
- run inverse realization and chronological pose construction; and
- verify its own parse, checksum, and stage identities.

One lawful implementation form is a generic task-space domain VM/constraint
solver. Its video-independent opcode semantics, reconstruction algebra,
projection, superposition, factor expansion, gauge canonicalization, and
inverse-solve loop live on the free inflate surface. The video-specific program
is counted: topology/event bytecode, factor operands, fitted seeds, selected
inequalities, tolerances, breakpoints, codebooks, and residual escapes all live
in `archive.zip`. Decode may search only the finite domain determined by those
counted operands and generic rules. It cannot consult SegNet, PoseNet, targets,
teacher/oracle state, or encoder evidence, and it cannot hide a video-derived
solution table in opcode implementations.

The following remain counted because their values are video-derived: weights,
latents, atoms, fitted dictionaries, events, temporal breakpoints, exceptions,
target-selected seeds or branches, fitted entropy contexts, gauge selectors,
and any parameters needed to reproduce the selected solution. Scorer weights,
GT/argmax tables, teacher frames, oracle tensors, VJPs, margins, and solve traces
are encoder evidence and are forbidden from both archive and runtime assets.

The compiler-placement invariant is therefore:

```text
generic algorithmic expansion/solve  -> free active inflate surface
irreducible video-specific statistic -> counted archive section
encoder oracle/teacher evidence       -> evidence store only
physically shipped executable/table   -> counted exact archive bytes
```

Every field has exactly one physical home. No field is both implicit in a
video-specific decoder branch and counted in a latent; no deletion receives
credit until the bytes and dead decoder dependencies are physically absent.

## 3. Quotient and gauge mathematics

### 3.1 Evaluator quotient

For a complete received population `Y`, define the evaluator coordinate

```text
Q(Y) = (Phi_seg(Y), Phi_pose(Y)).
```

The exact semantic fiber of a Seg state `s` is

```text
F_seg(s) = {Y : Phi_seg(Y)=s}.
```

Pose and rate vary inside this fiber. A usable gauge orbit is therefore not
"same RGB" and not "same d_seg"; it is the registered, receiver-realizable set
of exact Seg-equivalent representatives carried through chronological pose and
whole-object coding. Representatives are selected by complete score.

No fixed `d_seg`, `d_pose`, or byte gate prices an action. A continuation state
retains its actual triple `(d_seg,d_pose,B)` and remains nondominated if no
measured complete state is no worse in all three coordinates and strictly
better in one.

### 3.2 Resize/preimage gauge

For a locally linear disjoint resize map `A`, an integer kernel element
`k in ker_Z(A)` is an exact preimage gauge only after the actual uint8/rounding
path proves `R(x+k)=R(x)`. For the general receiver, define the exact fiber
directly:

```text
G_R(q) = {x in uint8 camera space : R(x)=q exactly}.
```

The compiler chooses within `G_R(q)` for Pose and final code length. It does
not minimize Euclidean norm. The MS1 affine-origin gauge and saturated local
CVP are one historical proposal policy, not the canonical gauge selector.

`GAUGE` bytes in `ddm_min_description_contract.py` are required to be zero. A
video-independent canonical fill rule may therefore be free gauge machinery.
A target- or video-selected gauge choice is not a zero-byte gauge stream: its
selector or parameter is counted under the earliest consumed SKELETON,
CONNECTION, FIBER, or RESIDUAL home unless it is deterministically derivable
from already-counted `z` by generic code.

### 3.3 Representation gauge

Factor models have exact aliases: sign flips, scale exchange, permutation,
basis rotation inside repeated singular subspaces, tensor transposes, code
chronology, dictionary order, context order, and equivalent ZIP spellings. The
encoder canonicalizes or races these representations against actual final ZIP
bytes. A selected video-specific spelling flag is counted unless the decoder
can infer it from a sealed canonical rule and existing counted bytes.

G20 is the exact control: receipt
`.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_lossless_xcodec_recode_20260726/receipt.json`,
SHA-256 `02ccb8a6209c79651b64fa93b15aa1ed6155b03d9709f5f18b4ff98edfe25c8c`,
proved a same-quantized-state rewrite from 83,838 to 81,027 archive bytes by
jointly choosing two tensor transposes, frame-delta modulo-256 code spelling,
and the outer DEFLATE profile over 10,240 exact points. It remains
`.not_a_candidate.zip` because full-n600 receiver replay is owed. Its -2,811
bytes are indivisible whole-object evidence, not a reusable additive credit.

## 4. Seven coupled factorization levels

The factor graph is the existing G17 seven-axis object, made executable against
the two exact teacher controls:

```text
F = W x C x T x N x Q x E x R

W: tensor/decoder weights, bases, shared atoms
C: per-frame/per-pair codes, amplitudes, coordinates, exceptions
T: temporal predictors, screw trajectories, events, persistence, resets
N: pair-population dictionaries, clusters, prototypes, sparse escapes
Q: semantic -> realization -> pose chronology and evaluator gauge
E: entropy contexts, symbol ordering, transforms, coder and ZIP spelling
R: analytic/generated structure versus irreducible learned residual
```

### 4.1 Tensor

Factor bases, per-pair codes, channel/class blocks, low-rank decompositions,
Kronecker/separable planes, shared dictionaries, tensor transposes, and sparse
residuals. Rank is not selected by Frobenius error. Each retained rank/factor
must survive receiver reconstruction and win a whole-object score comparison.

### 4.2 Code

Split generic interpreter instructions from video-derived operands; remove dead
branches; common-subexpression-eliminate repeated receiver graphs; merge shared
operators; canonicalize instruction order; and account for every shipped
executable/table. Explicitly race a compact counted task-space bytecode/parameter
description against tensor descriptions: generic opcodes may express topology
events, factor superposition, constraints/inequalities, projection, gauge fill,
and inverse realization, while every video-selected instruction and operand is
counted. A decoder algorithm can be free while every fitted operand it consumes
remains counted.

### 4.3 Time

Race within-pair prediction, previous-pair same-slot prediction, previous
emitted frame, chronological delta, screw transport, event/persistence streams,
piecewise-stationary segments, keyframes, and reset schedules. The pair order is
the canonical 0..599 identity. Permuted-time controls must demonstrate that a
gain comes from temporal structure rather than incidental byte order.

### 4.4 Population

Race independent pair factors against shared atoms plus references, class- or
stratum-conditioned prototypes, global dictionaries, mixture assignments,
population low rank, and local escapes. The 600 historical SENSE rows can
propose population axes; a current exact coder race is required for admission.

### 4.5 Semantic-realization-pose

Separate SKELETON (partition/topology), CONNECTION (chronology/transport),
FIBER (same-cell realization), exact zero-byte GAUGE fill, and RESIDUAL
(scorer-visible escape). Preserve semantic representatives through same-class
realization and chronological A. Frame 0 remains Seg-free but Pose-coupled; it
is not globally free. Pose is active in every lossy member-selection cycle.

### 4.6 Entropy

Price exact symbols under actual contexts, resets, transforms, dictionaries,
record boundaries, headers, manifests, CRCs, and outer ZIP interaction. Isolated
stream sums, estimated entropy, raw record bytes, and nominal parameter counts
are proposal metadata only. Every surviving row has a physically rebuilt
archive.

### 4.7 Analytic versus learned

Move video-independent geometry, basis generation, projection, factor
expansion, and solve logic into active inflate. Count analytic parameters when
they are fitted to this video. Admit a learned atom/residual only after a
matched analytic control and deletion test prove its exact complete-object
benefit. Learned escapes are residual owners of last resort, never a dense
teacher copy under a new name.

These levels interact. No level is optimized once and frozen before the others:
the scheduler retains exact cross-level branches and measures their interaction
through the same receiver and whole archive.

### 4.8 Cross-level R10 prosody/feature relay

The V10 history exposed a constraint layer that a pixel-only or low-rank-only
distillation loses: amplitude, frequency, phase, contrast, per-channel energy,
texture, and multiple-shooting continuity through Pose features. These are not
an eighth independent byte stream. They are cross-level constraint/feature
coordinates coupling W/C/T/Q/R and the chronological Pose map.

The teacher index therefore records encoder-only, source-bound observations for
those coordinates and their actual receiver/scorer foreign keys. The compiler
may express a selected solution using generic relay/prosody opcodes plus counted
video-specific amplitudes, phases, frequencies, contrasts, channel-energy or
texture parameters, shooting knots, and escapes. Generic feature reconstruction
and multiple-shooting solve logic may live in active inflate; all fitted values
are counted. The standalone decoder cannot evaluate PoseNet or infer a missing
constraint from teacher state. Each relayed constraint must terminate in a
receiver-computable operation, with Pose effects admitted only by current
complete-object measurement.

The required controls are: pixel/tensor factorization alone; feature relay
alone; their joint branch; phase/time permutation; channel-energy preserving
shuffle; texture deletion; and multiple-shooting knot deletion/merge. Missing
controls leave the R10 relay unpriced rather than zero-valued.

## 5. Closed action algebra

Each action has a unique ID, exact parent object SHA, exact target section set,
one or more factor axes, a physical mutation, an inverse/receiver contract, and
an evidence scope.

1. `DELETE`: remove a factor, atom, event, exception, context, selector, or
   decoder dependency. Bytes must be absent; a zeroed tensor is not deleted.
2. `MERGE_SHARE`: replace repeated values/graphs with one shared owner plus
   counted references and escapes.
3. `MIGRATE`: move one degree of freedom between base, pair code, temporal
   stream, population grammar, semantic/realization/A, residual, or entropy
   metadata. Ownership conservation must show it is counted once.
4. `GAUGE_SELECT`: choose another exact evaluator-equivalent or Seg-equivalent
   representative, then carry it through Pose and coding.
5. `REQUANTIZE`: jointly change precision, scale, codebook, deadzone, and
   entropy context; never apply an output-histogram step without an actual
   actuator/receiver foreign key.
6. `RECODE_IDENTITY`: change only representation spelling while proving exact
   decoded-population equality. This is the G20-style control.
7. `ANALYTICIZE`: replace learned/video-specific structure with generic
   deterministic decoder work plus the smaller fitted statistic.
8. `LEARNED_ESCAPE`: add a counted residual only for a receiver-measured debt
   left by analytic/shared factors.
9. `BUNDLE`: apply a preregistered coherent set of the above atomically when
   independent moves are known to interact.

Every action owes a no-op/base control, a physical deletion receipt, a
matched-byte reallocation control when distortion changes, and a same-output
repack control when representable. One failed action kills only that action and
tested context, never its factor family.

## 6. Rate-distortion continuation

Use

```text
D_Q(x)      = 100*d_seg(x) + sqrt(10*d_pose(x))
J_lambda(x) = D_Q(x) + lambda*B(x)
```

only to schedule neighboring actions. `lambda` is a continuation coordinate,
not an acceptance threshold and not an imputed shadow price. At every knot the
scheduler retains the measured Pareto set over `(d_seg,d_pose,B)`, immediate
neighbors, branch points, and coherent bundles. Final selection always uses the
complete contest functional on exact archive bytes and the live production
envelope.

The path is discontinuous at uint8 rounding, resize fibers, argmax boundaries,
Pose-tube activity, factor deletion, entropy-context changes, and ZIP spellings.
Therefore a single greedy trace is forbidden. Each branch has immutable parent
foreign keys and no-replace stage artifacts.

### H0 — custody and exact teacher census

- Rehash every object in §1.
- Strict-parse all 600 MS2R records with the actual V10 receiver parser.
- Join pair order, selected q4/q8 step, exact record bytes, MS1 SENSE rows, and
  V7 stratum hints into an encoder-only dense-free index.
- Preserve the C0B lifecycle joins and the R10 amplitude/frequency/phase,
  contrast, channel-energy, texture, and Pose multiple-shooting feature relay.
- Emit no candidate payload and no factor role.

### H1 — same-solution code/time/population/entropy contraction

- Decode the complete MS2R selected solution once on SSD-backed stages.
- Race exact population-global predictor/residual spellings: current per-pair
  Brotli control; previous-pair same-slot; previous emitted frame; modulo-256
  temporal deltas; deterministic segment resets; tensor/channel transposes;
  jointly coded global streams; and exact outer ZIP profiles.
- Reconstruct the identical 600-pair quotient-plane population, build one exact
  archive, double-replay, and compare final bytes. This requires no scorer
  inference because decoded equality is bit-exact.
- The 50 RAW isolated races remain controls; they do not veto this unmeasured
  population-global container.

### H2 — current-solution quotient/gauge contraction

- Recompute fresh current-G17 encoder observations; historical frames may only
  nominate the gauge/action family.
- Enumerate receiver-realizable exact Seg-equivalent representatives.
- Select them jointly through chronological A, Pose, and final archive bytes.
- Keep explicit same-Seg/different-Pose and same-output/recode controls.

### H3 — typed SKELETON/CONNECTION/FIBER distillation

- Build the measured typed atlas `stratum x scorer visibility x G4 temporal
  class` with all ten class-pair boundaries covered.
- Activate the canonical alternation: argmax-cell selection, within-cell
  lattice solve, real-coder price.
- Route a factor only after a strict current-object SKELETON-versus-FIBER coder
  race. Historical SENSE loadings seed proposals; they cannot choose roles.
- Race tensor/pixel factors against R10 feature/prosody constraints and their
  joint branch; do not assume low-rank pixels subsume phase, texture, channel
  energy, or Pose multiple-shooting structure.

### H4 — tensor and population substitution

- Factor shared atoms and per-pair codes across W/C/T/N.
- Execute DELETE/MERGE/MIGRATE/REQUANTIZE actions against the exact current
  candidate partition, including ep725 base/code, G17 semantic/realization/A,
  residual, and entropy homes.
- Physically rebuild every complete survivor and preserve ownership
  conservation.

### H5 — analyticization and learned residual

- Move generic expansion, projection, basis, factor, solve, and gauge work into
  the active decoder.
- Compare analytic, learned, and analytic-plus-learned variants with all fitted
  state charged.
- Admit the smallest learned escape only when deletion increases complete
  score and the matched analytic reallocation does not recover the debt.

### H6 — final whole-object closure

- Compile one versioned current-G17 candidate object.
- Strict parse, canonical re-encode, double replay, full n600 runtime, exact
  output order, frozen-scorer advisory measurement, exact STORE/DEFLATE archive
  construction, and complete G7 row.
- Only nondominated complete survivors may be offered to governed exact-eval
  custody. This specification grants no dispatch authority.

## 7. Telemetry, costates, and interactions

### 7.1 Required action row

Each `LatticeTeacherHomotopyActionReceiptV1` contains:

- schema, lane, run/stage/action IDs, parent and candidate SHA-256;
- MS1, MS2R, V7, current-G17, receiver, scorer, config, code, and archive-codec
  foreign keys actually consulted;
- factor axes, action algebra member, target sections, and physical byte
  ownership before/after;
- exact section vector, packet/member/archive bytes, and exact delta bytes;
- exact full-n600 `d_seg`, `d_pose`, score coordinates for lossy actions, or a
  bit-exact full-population equality proof for `RECODE_IDENTITY`;
- per-pair, class, inter-class edge, cell/edge/saddle, scorer-visibility, and G4
  temporal breakdowns when measured;
- pair chronology, reset/context/dictionary identities, entropy/coder rows, and
  final ZIP spelling;
- decoder runtime, peak storage, deterministic replay roots, and cleanup
  certificate;
- retained/excluded reason, verdict scope, pointer delta, and blocker list.

Dense target/scorer arrays remain encoder-side ephemera; the receipt stores
hashes and aggregates only.

### 7.2 Honest costates

A byte/distortion slope is finite only from a matched complete-object edge:

```text
lambda_B,D(a|x) = (B(x+a)-B(x)) / (D_Q(x)-D_Q(x+a))
```

when the denominator is positive and both endpoints are exact measured rows.
Otherwise it is JSON null with a reason. KKT duals are recorded only when the
actual solver exposes them. Margins, SVD amplitudes, endpoint accounting homes,
or output histograms never fabricate duals.

Per-dimension effective quantum is

```text
q_eff_i = uint8_step_i * measured_scorer_sensitivity_i
```

and is active only when dimension `i` has a concrete receiver actuator,
parse-back-stable perturbation, and current-object scorer secant. The prior R3
wall—accounting homes without composable actuator streams—remains a hard guard.

### 7.3 Interactions

For two actions from the same parent, measure

```text
I_S(a,b|x) = S(x+a+b) - S(x+a) - S(x+b) + S(x)
I_B(a,b|x) = B(x+a+b) - B(x+a) - B(x+b) + B(x).
```

All four objects must be physically built and measured under the same receiver,
scorer, and outer codec. Missing corners make the interaction null, never zero.
Retain interactions for tensor x entropy, time x population, semantic x
realization, realization x pose, gauge x entropy, analytic x learned, and each
action x outer ZIP spelling. G18 n2 feedback may nominate a corner; it cannot
populate an n600 interaction.

## 8. Real code map — reuse, refactor, build

| surface | disposition | exact role / required change |
|---|---|---|
| `src/tac/witness_dsl/v10_production_receiver.py` SHA `84d4ce09...` | reuse frozen | strict q4/q8 packet parse, archive build, factor-2 realization, storage preflight, inflate; do not mutate V1 |
| `src/tac/codec/v10_predictor_residual.py` SHA `fec4d269...` | reuse as exact H1 control | parse current 600 records and compare V2 global recode; preserve exact decoder tests |
| `src/tac/witness_dsl/ep725_lossless_xcodec_recode.py` SHA `7a54d13f...` | reuse search/accounting pattern | finite same-state transform and outer-ZIP race; do not reuse its ep725-specific wire as MS2R payload |
| `src/tac/optimization/ddm_lattice_costate_sense.py` SHA `7a634285...` | refactor by new adapter, do not relabel | historical producer is MS1-specific; new current-action telemetry must use a new schema/producer and measured coder races |
| `src/tac/optimization/ddm_typed_quotient_solve.py` SHA `fa7af7ec...` | reuse math validators | measured non-Euclidean geometry, visible coordinates, typed atlas, alternation, effective quanta, exact metric sieve; add action/receiver foreign keys in a new adapter |
| `src/tac/optimization/ddm_min_description_contract.py` SHA `6a12543f...` | reuse stream types; extend in new version | current headline is stored-problem-centric and GAUGE is zero-byte; add selected-solution/compiler-placement receipt rather than changing old semantics |
| `src/tac/optimization/ddm_description_vocabulary.py` SHA `b25a244a...` | proposal library only | persistent level sets, boundary splines, turning curves, joint vocabulary and real coding; every use must win current receiver races |
| `src/tac/optimization/tensor_factorize_receiver.py` SHA `5217795f...` | reuse reconstruction/custody primitives | adapt to a new current-G17 packet/version; legacy candidate schema is not G17 authority |
| `src/tac/witness_dsl/factorized_v9_predictor.py` SHA `325ec698...` | reuse semantic-factor interface | source-bound five-role semantic decode and counted temporal state; no historical V9 payload import |
| `src/tac/witness_control/factorized_features.py` SHA `3c720fc6...` | advisory proposal telemetry | exact resize support and margin snapshots; subset defaults are never decision evidence |
| `src/tac/witness_control/factorized_adjoint.py` SHA `4de3b6cd...` | advisory scheduling | rank-4 class operator and explicit gauge null; no byte/distortion admission |
| `src/tac/witness_control/factorized_duty_ranking.py` SHA `e85e0184...` | advisory scheduling | first-order lever proposals only; exact whole-object follow-up mandatory |
| `src/tac/witness_dsl/pair_population_envelope.py` SHA `cdb74fb9...` | reuse payload firewall | teacher/oracle/explicit-preimage artifacts forbidden from counted payload |
| `src/tac/witness_dsl/taskspace_whole_archive_allocator.py` SHA `6e491faf...` | reuse final authority adapter | exact whole-object STORE/DEFLATE, receiver and scorer callbacks, nonlinear score transition |
| G17 spec | compose, never edit here | population grammar, substitutive allocation, compiler placement, gauge selection, and solution homotopy interfaces |

New implementation surfaces, only after root review:

1. `src/tac/optimization/ddm_lattice_teacher_solution_index.py` — streaming
   hash-verifying H0 join and dense-free corpus.
2. `src/tac/codec/v10_population_predictor_residual_v2.py` — exact H1
   population-global same-solution codec; new magic/version, never a silent V1
   mutation.
3. `src/tac/witness_dsl/taskspace_selected_solution_compiler.py` — placement,
   ownership, action, gauge, and factor manifests for the current G17 child.
4. `src/tac/optimization/ddm_lattice_teacher_compaction_homotopy.py` — immutable
   Pareto/continuation scheduler that accepts only actual callback rows.
5. `tools/run_ddm_lattice_teacher_compaction_homotopy.py` — governed,
   storage-preflighted, resumable runner whose default is refusal/print command.

None of these names implies success. Each implementation remains blocked until
its real full-object gates pass.

### 8.1 Historical missing-layer audit against G17 and G19

The old lattice campaign lacked necessary types at several levels: quotient
constraints, selected representative/gauge, physical factor ownership,
population sharing, temporal/Pose chronology, entropy context, analytic versus
learned residual ownership, and the three-way decoder/payload/oracle placement.
G17 now names all of those layers architecturally. G19 now types their
finite-action/control coordinates and preserves nonadditive observations.
Neither currently supplies the remaining executable bridge:

1. **physical actuator/ownership IR** — a bijection from each typed factor or
   constraint operand to actual receiver-consumed bytes, the generic operation
   that consumes it, the decoded coordinates it moves, and the exact archive
   section that owns it;
2. **decoder-computable constraint program** — validated task-space bytecode and
   operands that the standalone decoder can solve without scorer, target, or
   teacher access; G19 placement evidence is hash-preserved but deliberately not
   dereferenced and G17's placement manifest remains owed;
3. **full-population membership/receive proof** — G19 binds an external n600
   manifest but proves no membership, while G17's P-once population ABI is not
   implemented; and
4. **mechanism-verified action semantics** — G19's DELETE/MERGE/FACTOR/MIGRATE/
   GAUGE labels are declared-only until exact bytes, equivalence proof, receiver
   callbacks, and whole-object rows exist.

The selected-solution compiler also freezes the five prior C0B joins that
cannot be replaced by caller-attested hashes:

1. reopened-object-derived digests, recomputed from the actual reopened bytes;
2. explicit `complete` or `sparse-owned` ObligationIR coverage, with every debt
   assigned once and no uncovered/duplicate coordinate;
3. a hashed `PairPopulation` coordinate map binding canonical pair IDs, order,
   source coordinates, and population membership;
4. exclusive ownership between the V9 Pose6 coordinate and any frame-0 residual
   representation; and
5. retention of the actual explicit-preimage bytes in encoder evidence when a
   join or falsifier requires them—not merely a hash/size attestation. Those
   evidence bytes remain forbidden candidate payload unless a new lawful
   current-lineage description is independently compiled and counted.

The lifecycle is closed and ordered:

```text
SourceTruth
  -> ObligationIR
  -> RealizedPair(Y0,Y1)
  -> ArchiveArtifact
  -> DecodeReceipt
  -> ScoreReceipt(axis)
```

Every transition derives its downstream identity from the actual upstream
object. A `ScoreReceipt` cannot bind an archive by a free-form digest, and a
`DecodeReceipt` cannot stand in for an `ArchiveArtifact`. The five logical
representation types remain exactly SKELETON, CONNECTION, FIBER, GAUGE, and
RESIDUAL with earliest homes L1_program, L2_chart, L3_raster,
L4_scorer_feature, and L5_verdict. Logical typing assigns ownership and
recursion; it does **not** require five physically independent entropy files.
The entropy compiler may interleave or jointly code logical types if its
manifest preserves exact ownership and the receiver consumes them unambiguously.

G17/G19 also do not yet implement the historical R10 prosody/feature relay in
§4.8. G19 can index a factor coordinate and preserve a declared action, but it
does not carry the amplitude/frequency/phase/contrast/channel-energy/texture or
Pose multiple-shooting constraint through a real decoder. G17's seven-axis
factor graph permits that relay but does not freeze its executable IR. Both
gaps belong in the one selected-solution compiler rather than a new sidecar.

Thus the missing layer is no longer another conceptual factor axis. It is the
typed, receiver-executable **selected-solution compiler IR** joining constraint
bytecode, factor ownership, full-population reconstruction, and exact archive
pricing. `taskspace_selected_solution_compiler.py` in the new-surface list is
the single owner; duplicating parallel schemas in the homotopy scheduler or
G19 controller is forbidden.

## 9. Exact gates and resumability

### Gate C0 — custody

All §1 hashes, file sizes, schemas, pair order, scorer identities, receiver
source closure, and current G17 parent hash match. Any drift refuses.

### Gate C1 — placement and lineage

Every field is classified as generic decoder, counted video statistic,
encoder-only evidence, or physically charged executable/table. Mixed blobs,
teacher bytes, scorer products, public/donor payload, and target-selected code
constants refuse.

### Gate C2 — packet and archive closure

Strict parse, exact EOF, CRC/hash validation, canonical parse/re-encode, one
complete population object, exact archive member map, and deterministic archive
construction x2.

### Gate C3 — receiver closure

Double replay all 600 pairs, exact pair order, byte-identical outputs for
identity actions, exact received arrays for lossy actions, and full runtime
below the claimed contest wall. n2/n24/n64/n256 are wiring only, never evidence.

### Gate C4 — scorer closure

For any changed decoded output: actual uint8/R, batch-appropriate frozen CPU
scorers over all n600, per-axis/per-stratum telemetry, Pose active, and no proxy
substitution. This is still advisory, not an exact contest row.

### Gate C5 — complete-object economics

Actual packet, member, final ZIP bytes, complete score transition, matched
controls, and interaction corners. No isolated coder sum or nominal saving.

### Gate C6 — promotion custody

Only a new own-lineage nondominated archive with standalone runtime, exact
archive SHA/bytes, full provenance, governed lane claim, and explicit dispatch
authority may enter contest CPU/CUDA evaluation. This spec cannot satisfy C6.

Every run stage is atomic and no-replace:

```text
00_preflight
01_teacher_index
02_identity_population_recode
03_gauge_candidates
04_typed_factor_atlas
05_substitutive_actions
06_analytic_learned_race
07_complete_objects
08_candidate_closure
```

Each stage writes a complete byte-close-loadable checkpoint plus manifest,
config, RNG seed/state, code hashes, input hashes, parent action roots, and
cleanup certificate. Resume uses `--resume-from`; completed scorer calls are
never repeated unless their complete foreign-key set changed. SSD waterfall is
VertigoDataTier, then APDataStore, then explicit local opt-in. Scratch is
context-managed; evidence is never cited from `/tmp`.

## 10. Smallest real next cut

The smallest non-naive executable cut is **H0 plus one H1 complete-object
identity action**, not a scorer probe and not another memo:

1. Implement the streaming `DDMLatticeTeacherSolutionIndexV1` loader over all
   600 MS2R records and 600 MS1 SENSE rows. Rehash the immutable receipts,
   selected archive/packet, pair rows, and factorization; emit one dense-free
   content root.
2. Implement one new versioned population-global exact codec branch using the
   actual selected MS2R quotient planes. Its finite menu is:
   - current V1 per-pair predictor-residual control;
   - previous-pair same-slot and previous-emitted-frame predictors;
   - exact modulo-256 deltas;
   - reset intervals `{none,8,16,32,64}`;
   - canonical channel/tensor order and its sealed transpose alternatives;
   - actual supported content coders and final ZIP profiles.
3. Build every finite point as one complete archive, strict-decode the complete
   600-pair population, require exact decoded equality to MS2R, build the winner
   twice, and record the exact final archive delta. Put the generic inverse and
   predictor logic on the free decoder surface; count all headers, modes,
   resets, dictionaries, and video-derived streams physically present.
4. If no point strictly beats 291,205,400 bytes, retain the original object and
   scope the negative to this exact lossless population-predictor menu. Do not
   repeat isolated coder races and do not infer that tensor/population/semantic
   factorization is dead.
5. If a point wins, label it a receiver-closed same-solution research control,
   not a candidate or score row. Feed its measured byte structure into H2/H3;
   do not claim goal progress while the pointer is unchanged.

This cut is full-n600, real-code, same-object, and immediately falsifiable. It
closes the missing counted population-global container behind the R2 50-stream
diagnostics and yields the first exact code/time/population/entropy knot for
the deeper selected-solution homotopy.

## 11. Ranked eurekas

1. **MS2R already proves the first large deletion is real.** A finite q4/q8
   selected solution physically removes 118,321,525 bytes (28.8922456%) while
   remaining inside its registered distortion cap. The endpoint was optimized
   for additive per-pair record bytes, not final factorability or shortest
   selected-solution program, so it is a starting oracle rather than a wall.
2. **Solution versus problem is the main byte-home correction.** The dense
   quotient plane is a valid solution but a catastrophic description. Active
   inflate can generate/factor/project/repair the same solution from a much
   smaller statistic; generic work is free, fitted choices are counted.
3. **MS1 and MS2R are complementary, not competing.** MS1 supplies the
   low-distortion existence/factor proposal surface; MS2R supplies exact
   receiver-closed full-object deletion evidence. Conflating the 731 MB
   diagnostic count with the 291 MB archive destroys both signals.
4. **The missing first measurement is population-global coding.** R2 priced
   pair records and isolated streams; G20 proved whole-state transform and ZIP
   spelling interact. A complete global same-solution recode is a lawful,
   exact, scorer-free first knot.
5. **Gauge is a rate actuator only when selected jointly.** Exact Seg
   equivalence leaves large realization freedom, but Pose and entropy vary
   inside the cell. Min-norm or d_seg-only paint suppresses the cheapest
   representative.
6. **SVD is a proposal, not a factor.** The MS1 six-above-floor rows acquire
   semantic roles only through current SKELETON/FIBER coder races and complete
   receiver replay.
7. **Deletion and migration must be physical.** Zeroed bytes, additive
   sidecars, and nominal parameter reductions cannot release the 84,536-byte P
   allocation or the 291 MB plane allocation. Substitution is the operative
   action.

## 12. Ranked blockers

1. `CURRENT_G17_SELECTED_SOLUTION_TEACHER_OWED`: historical teacher state may
   nominate actions but cannot initialize or populate the current candidate.
2. `POPULATION_GLOBAL_SAME_SOLUTION_CODEC_OWED`: no counted full-n600 container
   yet prices cross-pair prediction/factorization for the MS2R object.
3. `CURRENT_RECEIVER_ACTUATOR_FOREIGN_KEYS_OWED`: the 162 accounting homes and
   old SENSE dimensions are not composable physical actuator streams.
4. `TYPED_ATLAS_AND_PER_DIMENSION_QUANTA_OWED`: current R2 lacks the complete
   stratum x visibility x G4 atlas and actual effective quanta.
5. `EVALUATOR_EQUIVALENT_GAUGE_SCHEDULER_OWED`: Seg-equivalent representatives
   are not yet carried through Pose and whole-object rate.
6. `SUBSTITUTIVE_OWNERSHIP_COMPILER_OWED`: current original G17 V1 is additive
   and cannot yet delete/merge/migrate P/base/code/grammar/A/residual homes.
7. `COMPILER_PLACEMENT_MANIFEST_OWED`: generic decoder work and video-derived
   state are not yet exhaustively classified field-by-field.
8. `FULL_N600_RUNTIME_AND_COMPLETE_OBJECT_CUSTODY_OWED`: G20 remains bounded
   replay, G17 production population receive is unimplemented, and no new G21
   object exists.
9. `EXACT_CONTEST_ROW_OWED`: no candidate, governed dispatch, or pointer change
   follows from this research design.

## 13. Triality and G17 bridge

- DSL leg: future typed placement, action, factor, and receiver schemas plus the
  sealed finite H1 codec menu.
- DAG leg: H0 through H6 and C0 through C6, with immutable parent/action roots
  and stage checkpoints.
- Equation leg: selected-solution program length, exact evaluator quotient and
  gauge fibers, seven-axis factor graph, nonlinear score, continuation
  scheduler, matched costates, and measured interactions.

This file refines G17's
`G17_OVERCOMPLETE_TEACHER_SOLUTION_HOMOTOPY_IMPLEMENTATION_OWED`,
`G17_POPULATION_GLOBAL_GRAMMAR_IMPLEMENTATION_OWED`,
`G17_SUBSTITUTIVE_RATE_REALLOCATION_IMPLEMENTATION_OWED`,
`G17_SOLUTION_DESCRIPTION_COMPILER_PLACEMENT_IMPLEMENTATION_OWED`, and
`G17_EVALUATOR_EQUIVALENT_GAUGE_SELECTION_IMPLEMENTATION_OWED` edges. It does
not alter G17 ABI, reorder its semantic -> realization -> chronological A
causality, or authorize implementation.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md` SHA-256
  `47d4ac3a38f91a8b8e7dc3061131717d8122bd48ffb204ffb914eb58e687f0c9`;
- `PROGRAM.md` SHA-256
  `a6d5f79f3241ca1ae17b2587afd9940e1a4ea598804fd9efa152f2330e15db82`;
- current lane/subagent registries and research-only G21 checkpoint;
- exact MS1, MS2R R2, V7, R3 refusal, and G20 receipts described above;
- current G17 unified production envelope and G7/G8/G10/G12/G13/G14/G15/G16/G18/G20 specs;
- current V9/V10, SENSE, typed quotient, minimum-description, vocabulary,
  tensor-factor, population-envelope, factorized-control, and whole-object
  allocator code surfaces enumerated in §8.

HISTORICAL_PROVENANCE: first G21 specification that distinguishes the MS1
731,622,325-byte diagnostic count from the exact MS2R 291,205,400-byte
receiver-closed stored solution/control, and turns both into an encoder-only
teacher for a selected-solution, active-inflate, complete-object compaction
homotopy without importing historical payload.

## 14. G100 H0 implementation and bounded H1 freeze — 2026-07-27

Lane `lane_g100_lattice_teacher_compaction_takeoff_20260727` implemented the
smallest honest takeoff surface without launching a candidate, scorer, or
291 MB recode menu.

### 14.1 Real H0 result

`src/tac/optimization/ddm_lattice_teacher_solution_index.py` now implements the
streaming `DDMLatticeTeacherSolutionIndexV1` mechanics. It:

- rehashes the immutable MS2R/MS1 receipts, selected packet, SENSE rows,
  factorization, selected archive, C1 control, dense raw teacher, target cache,
  typed config, solver, and materializer before admitting their evidence;
- strict-parses the V10 production envelope and reconstructs each of the 600
  predictor records independently;
- performs two one-pair-at-a-time passes so the exact declared plane-major
  decoded root is verified without retaining the population;
- joins the exact 208 q4 / 392 q8 decisions, the 600 selected-record
  byte/step bindings, all 600 SENSE rows, and the six-factor proposal receipt;
  and
- persists hashes/accounting only. Teacher pixels and historical payload are
  neither retained nor made payload-eligible.

The governed run completed on the Vertigo SSD tier:

```text
config SHA-256          184b91b8e736a635ef57ff24c66d3ede40575e613d17f4eed87c9156aa35b8ee
preflight receipt       885 bytes
preflight SHA-256       7e23246b4e53198b342255d5e9e5b4abc9675ab3a85adbc4b003854619e35c95
solution index          877,374 bytes
solution-index SHA-256  b0838bdada728c4537ca95b4552ff284da7fe1296d0774c5219a13db8e6d3445
content root SHA-256    073d079ed3cc30abe7b7e13bfdfa96debb9f7cc06eb5482d07dc55a229981c6c
packet decoded root     3494c0cf81a6df32512ffb524e1f75186f3213cf050d55111d793ec3d6351338
pair population         600
q4 / q8                 208 / 392
peak population rows    1
dense teacher persisted 0 bytes
candidate payload       none
historical payload reuse false
pointer delta           NONE
```

Durable SSD checkpoints:

- `/Volumes/VertigoDataTier/pact/g100_lattice_teacher_compaction_takeoff_20260727/stage_checkpoints/00_storage_preflight.json`
- `/Volumes/VertigoDataTier/pact/g100_lattice_teacher_compaction_takeoff_20260727/stage_checkpoints/01_solution_index.json`

The repository receipt is
`.omx/research/original_taskspace_inverse_witness_codec_20260725/g100_lattice_teacher_h0_takeoff_receipt_20260727.json`.

### 14.2 Executable surfaces and authority boundary

- `src/tac/codec/v10_population_predictor_residual_v2.py` is one bounded H1
  proposal, not a menu result: exact previous-pair same-slot prediction,
  modulo-256 deltas, explicit periodic resets, Brotli-Q11 streams, per-component
  and per-frame hashes, atomic immutable output, and one-pair streaming decode.
- `src/tac/optimization/ddm_lattice_teacher_compaction_homotopy.py` admits only
  physical archive rows. Rate is exact whole-archive bytes. Lossy rows require
  a full-n600 scorer callback. Identity recodes inherit distortion only after
  exact receiver-root equality, and every row requires equal first/decode-replay
  roots. Missing interaction corners remain JSON null.
- `tools/run_ddm_lattice_teacher_compaction_homotopy.py` implements immutable
  SSD preflight and H0 stage checkpoints. Its default is refusal/status. H1 is
  deliberately absent from the runner while
  `h1_materialization_authorized=false`.
- The tracked canonical
  `src/tac/witness_dsl/taskspace_selected_solution_compiler.py` remains
  read-only; G100 did not fork or mutate its compiler IR.

Focused mechanics validation: 7 tests passed across the H0 index, V2 exact
codec, and whole-object homotopy ledger. This is production mechanics evidence,
not evaluator or score evidence.

### 14.3 Exact continuation blocker

H0 closes `LATTICE_TEACHER_SOLUTION_INDEX_OWED`. It does not close the
candidate-facing compiler/receiver bridge. The next bounded measurement remains
`POPULATION_GLOBAL_SAME_SOLUTION_CODEC_MEASUREMENT_OWED`, but materializing the
291 MB selected packet through V2 is frozen pending root review. If authorized,
the one finite proposal must be built twice, wrapped as a complete archive,
strict-decoded over all 600 pairs, and proven equal to every source frame in an
explicit common ordering. The H0 root `073d079e...` is the index/custody root,
not a decoded-population root; the historical packet root `3494c0cf...` is
plane-major, while V2 is pair-major, so neither may be compared across orderings.
The materializer must derive and bind the source pair-major root while streaming,
then prove it equal to V2's pair-major root and per-frame hashes. Price only the
actual final archive bytes. No scorer call is needed for an exact identity
recode; no pointer or promotion claim follows.
