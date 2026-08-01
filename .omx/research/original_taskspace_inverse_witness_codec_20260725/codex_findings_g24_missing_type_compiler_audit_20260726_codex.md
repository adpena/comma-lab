# G24 adversarial findings — missing-type selected-solution compiler audit

UTC: 2026-07-26

Lane: `lane_g24_missing_type_compiler_audit_20260726`

Mode: `research_only=true`; local read and design audit only; no training,
heavy run, scorer dispatch, exact evaluation, candidate claim, score claim,
promotion, pointer mutation, commit, or push

Pointer policy: always read
`.omx/state/canonical_frontier_pointer.json#effective_frontier` dynamically.
No dated score literal in this memo is an executable gate. At this audit
snapshot the dynamic target is the upstream official-display `0.172` row;
that external display is a competitive target, not our archive custody or an
exact local score. The executable success condition is one custody-complete
own-lineage archive whose authority-tier exact score is strictly below the
freshly refreshed `effective_frontier`. Sub-`0.15` remains a program stretch
aim after/beyond that moving admission boundary, never a substitute for it.
This memo moves the pointer by `0` and is not score progress.

## Verdict

`THE_MISSING_LAYER_IS_A_PRODUCT_TYPE_AND_RECEIVER_EXECUTABLE_SELECTED_SOLUTION_COMPILER_NOT_ANOTHER_CODEC_FAMILY`

The historical finding that the codec was missing necessary types and layers
is correct, but the sharp form is more specific than “add the five types.” The
five semantic roles already exist. Seven scientific roles already exist. The
G17 packet chronology, G19 interaction controller, G21 homotopy design, C0B
joins, V9/V10 math, and R10 feature-relay vocabulary also exist. Signal is
still suppressed because several *orthogonal* coordinates are represented by
one overloaded string, one digest, or one section label:

1. what scientific object a datum describes;
2. what semantic representation role it plays;
3. where it lives in evaluator recursion;
4. whether it is free decoder code, counted video state, encoder evidence, or
   a charged executable/table;
5. which exact post-compression bytes physically own it and which decoder
   consumer reads those bytes;
6. which receiver function it denotes versus how its parameters are spelled;
7. which lifecycle object and parent identity it belongs to;
8. what evidence authority is permitted to interpret it;
9. whether its measured effect is an endpoint transition or an indivisible
   higher-order interaction; and
10. which external controls a proof actually depends on, versus immutable
   artifact identity and separately rebasable admission state; and
11. which cross-level R10 constraints survive into executable decode.

These coordinates are related but are not aliases. The capstone compiler must
therefore be a tagged product/refinement type, not another flat enum and not a
directory of independently optimized streams:

```text
SelectedSolutionAtom =
  LifecyclePhase
  x ScientificRole
  x SemanticRole
  x RecursionCoordinate(namespace, stage)
  x PlacementClass
  x LogicalOwnership
  x PhysicalCodingGroup
  x FunctionalQuotientIdentity
  x ParameterSpellingIdentity
  x AuthorityEvidence
  x ProofDependencySet
  x EffectObservation
```

Not every variant inhabits every coordinate. Encoder-only teacher evidence,
for example, has no candidate byte range. A zero-byte receiver-null gauge has
no counted operand. Those absences must be typed variants, not null fields that
can later be mistaken for missing custody.

The highest-severity present omissions at the refreshed concurrent snapshot
are:

- the supposed explicit-preimage join still reopens only a hash-only receipt;
  the actual V10 receiver packet bytes are not retained or parsed by the join;
- authority remains a free-form string on `ActionEffect`, G19, and applied
  action receipts despite the already-derived rule that authority is a type;
- the current single-stream `StreamHomeClaim` cannot honestly model several
  logical owners jointly entropy-coded into one physical range;
- nominal functional-quotient and parameter-spelling identity classes now
  exist, but they are not attached to placement, lifecycle, exact byte
  ownership, or action receipts, so lossless repacks, prune/merge actions, and
  functional mutations still cannot share one honest action algebra;
- R10 amplitude/frequency/phase/contrast/channel-energy/texture and Pose
  multiple-shooting constraints have names and opaque payload slots but no
  receiver-executable typed semantics or proof of consumption;
- lifecycle dataclasses now exist, but their constructors do not prove archive
  membership, actual receiver execution, or scorer execution; an axis enum can
  currently be attached to research-only supplied forward arrays, so axis
  tagging can launder authority;
- exact artifact identity, semantic-control identity, and proof validity are
  conflated, allowing irrelevant pointer refreshes to revoke byte-equality
  proofs while stale admission comparisons can remain attached; and
- several newly named axes (`ScientificRole`, recursion namespace, proof
  dependency, authority, effect kind) are enums only: they are not yet fields
  of one validated selected-solution product or its receipts.

G23's specification is directionally the closest current closure. Five
`src/tac/witness_dsl/taskspace_g17_*.py` implementation files materialized
concurrently during this audit, after the first snapshot. They implement
several real parsers, packets, receipts, and a deterministic byte VM; they do
not yet close the product type or executable authority chain described here.
No matching `test_taskspace_g17_*.py` file existed at the refreshed snapshot.
After this audit reported the G21/G23 owner conflict, G23 moved the definitions
to G21's canonical `taskspace_selected_solution_compiler.py` and reduced the
former G17 module to a nine-line identity-preserving re-export. That specific
ownership conflict is closed; the semantic/executable gaps below remain.

## Evidence snapshot and scope

Repository HEAD at preflight:
`0058123af31779d83d1fc10a728389b0ce7823ec`.

Contract custody:

- `CLAUDE.md` and `AGENTS.md` are byte-identical, SHA-256
  `47d4ac3a38f91a8b8e7dc3061131717d8122bd48ffb204ffb914eb58e687f0c9`;
- G17 frozen specification SHA-256
  `f315c8c0ad3708394e96cbbf40de9bb6af7d6072989bb28ea38a226f5354953b`;
- G19 findings SHA-256
  `633d61172529d93444ff6be32b3af4dc9686faaf87356eb359893eb622dccf3f`;
- G19 implementation SHA-256
  `52f3a8f54fe2ac9c2f56e415f337e8d5d83db52645b1a621448879b2aab8c7a6`;
- G21 design SHA-256
  `a608b25ae18e07636a6794fa13a3e49bd1a80a2d63b8dc73e3d8162ebc792130`;
- emerging G23 spec snapshot SHA-256
  `b71d1a29c9ae419b15fbb5be30be7c33789634796e416ddc79d67870baf4a9a6`;
- concurrent G23 compiler-placement implementation snapshot SHA-256
  `e8dd891daccabaaa479ad4a754d27e7d0ecc1a1e0d97f7ec619e897e7a8d8dae`;
- concurrent G23 forward-observation snapshot SHA-256
  `98af25ae72bd947c7ae06f1a3612686372f59f63dc70e8c3111cfe48cb2f83e7`;
- concurrent G23 G-descriptor-custody snapshot SHA-256
  `4900efbff6691a0d1ec77f21394ef825a62cbcb434fd679f5415f590a40d9eda`;
- concurrent G23 generalized-XIP2-A snapshot SHA-256
  `074b8988a3473e5c6ea41ee6371d1870044e316afa39de95980726eaefed252a`;
- concurrent G23 production-envelope snapshot SHA-256
  `a51a6d4b8772ba6d89db6e8b69e335e8becd0709dc262c616f6b016e924a2dee`;
- post-finding canonical selected-solution owner snapshot SHA-256
  `97205c9c0071c654311091288def73a7a22a256ae68b34b97892f6280a006cd3`;
- post-finding G17 compatibility adapter snapshot SHA-256
  `136466d1dfe8ecc4aae1ea6bcf924133834da1b24da29d2bee4d3121cb3218b2`;
- TS1 minimum-description type source SHA-256
  `6a12543f1359bf447222ddccf14ec800b3aa60eb24712ca70ecc0a96d94f2019`;
- SN1 addendum source SHA-256
  `79ec7f27a2a73e748eba8bd053d09cc56b7cb46c78dd0aa5cb17314a4c2ef29d`;
- LP1 layer-pricing source SHA-256
  `813ab277475da132817190b9141ca7639e602797c84e44cc1500b8e81e36292a`;
- current C0B pair/join surface snapshot SHA-256
  `cdb74fb97a0a5d80e2d4a93e7d0ec81e5943bdeb1899ddf5f40a83190f039f00`;
- explicit-obligation/preimage source SHA-256
  `db7b765bacc24d7f5b9c6278417ccde1f9c50f5be6e32208363f4c5c30c6f422`;
- seven-role scientific-state source SHA-256
  `5b04de3ae51fb78cb31cee97bd62820f5da0131cf76c4d0630a17b18ae7121f5`;
- `ActionEffect` source SHA-256
  `4018bf5c8c23c2240391bcc739c2d2381bec230702c5932576747d2f66ef2d67`;
- applied-action bridge source SHA-256
  `59480ae83b473f78eabfededa82ceecd229d19133636f5d0d80d75840bb2dd00`.

The shared tree is dirty and concurrent. These hashes are audit snapshots,
not claims that another owner has finished or frozen an implementation. G24
does not edit G17/G19/G21/G23-owned source or specifications.

### Concurrent G23 implementation audit

What is now real at the snapshot above:

- fourteen explicit nonaliasing logical byte classes, the five semantic-role enum,
  seven-scientific-role enum, namespaced recursion and placement enums,
  lifecycle/proof-dependency/authority/effect-kind enums, and
  false-free/decoder-dead placement checks;
- strict G/A population packet directories with contiguous exact payload
  slices, parent bindings, hashes/CRCs, canonical shard windows, and terminal
  P/G/A binding;
- frozen-parser custody for active G descriptors;
- an actual generalized-XIP2-A packet parser and deterministic double decoder;
- dense-retaining target/candidate forward-observation evidence with strict
  canonical receipts and double-forward equality; and
- a deterministic generic byte VM whose admitted byte operations execute
  twice and compare exactly, while unknown opcodes refuse.

The canonical-owner repair is also real: an import-identity audit over all 39
public exports found zero missing and zero nonidentical objects between the
canonical module and the compatibility adapter. This prevents a second schema;
it does not by itself wire the new enums into the product/lifecycle.

Those are valuable receiver/compiler pieces. They do not establish closure:

1. `G17CompilerPlacementRecordV1` assigns one semantic role and one placement
   to each logical byte object. It has no separate scientific role, no
   many-to-many logical-to-physical incidence, and no exact archive member,
   offset, length, range hash, coder owner, container owner, or receiver
   consumer. `receiver_consumed: bool` is an attestation, not a consumption
   edge.
2. `G17PairPopulationV1` binds source/V9 and one zero-based IR/V10 tuple, not
   the distinct global/source, V9, PBR, IR, and V10 coordinate map already
   required by C0B. `G17ObligationCoverageV1` covers pair IDs, not the actual
   obligation/cell universe or one live receiver owner per sparse debt.
3. `G17PosePreimageOwnershipV1` accepts opaque
   `explicit_preimage_evidence_bytes` without a frozen parser or a link to the
   generalized-XIP2 packet. Thus the new strict A packet is real, but the C0B
   explicit-preimage ownership join still does not require it.
4. `G17R10ProsodyFeatureRelayV1` stores one opaque byte string per constraint;
   `require_receiver_consumption(...)` proves only that a caller repeated the
   full enum list. It does not parse values, bind support/tolerance/block or
   chronology, identify counted operand spans, or execute a receiver action.
   The current VM has byte-copy/concat/slice/XOR/repeat/hash operations, not
   topology, constraint, feature-relay, or multiple-shooting semantics.
5. The lifecycle constructors retain typed parent Python objects, but
   `C0BArchiveArtifactV1` accepts arbitrary member/archive bytes without
   proving containment or strict parse; `C0BDecodeReceiptV1` accepts arbitrary
   receiver-receipt bytes and equates supplied decoded bytes only to supplied
   `RealizedPair` bytes; and `C0BScoreReceiptV1` can attach any
   `C0BScoreAxisV1`, including contest CPU/CUDA, to a research-only forward
   observation. `taskspace_g17_forward_observation.py` explicitly does not run
   a scorer. This is typed ancestry, not authority-bearing execution.
6. Names now exist for scientific role, lifecycle phase, recursion namespace,
   proof dependency domains, authority class, effect kind, functional
   quotient identity, and parameter spelling identity. But no
   `ProofDependencySet`, sealed authority evidence, general interaction
   hyperedge, or exact joint-coding group exists, and the new axes/identities
   are not fields of `G17CompilerPlacementRecordV1` or the lifecycle receipts.
   The placement validator also still admits only the original ten logical
   classes, rejecting its four newly declared VM/executable logical classes.
7. The production-envelope module owns strict G/A/E syntax but, at this
   snapshot, exports packet build/parse surfaces only; imported whole-archive,
   receiver, and measurement interfaces are not connected into an n600
   standalone receiver/evaluator chain.

Two constructor-only adversarial fixtures confirm the two sharpest gaps; these
are structural ABI counterexamples, not scientific measurements:

```text
C0BScoreReceiptV1(
  archive_bytes=b"not-a-zip",
  member_bytes=b"not-contained-member",
  receiver_receipt_bytes=b"not-a-receiver-receipt",
  axis=CONTEST_CPU,
  observation=<research-only supplied arrays>
) -> CONSTRUCTED

G17R10ProsodyFeatureRelayV1(
  every_constraint_name -> b"opaque-unparsed-junk"
).require_receiver_consumption(every_constraint_name) -> ACCEPTED
```

The first observation receipt itself truthfully says
`encoder_evidence_only=True` and `candidate_payload_allowed=False`; the
lifecycle wrapper nevertheless accepts the contest axis. The second fixture
executes no payload. These are direct proof that axis/name coverage is not
authority/receiver consumption.

Therefore G23 status at this snapshot is
`SUBSTANTIAL_EXECUTABLE_COMPONENTS_PRESENT / SELECTED_SOLUTION_PRODUCT_AND_AUTHORITY_CHAIN_OPEN`.

## 1. Exact crosswalk: independent namespaces, not one enum

| Axis | Exact question answered | Canonical values / source | Forbidden alias |
|---|---|---|---|
| Lifecycle | Which identity-bearing object exists now? | `SourceTruth -> ObligationIR -> RealizedPair(Y0,Y1) -> ArchiveArtifact -> DecodeReceipt -> ScoreReceipt(axis)` | A digest, receipt, or archive may not stand in for another stage. |
| Scientific role | Which part of the contest-specific world model is described? | seven `ScientificStreamRole` values in `coupled_witness_state.py` | Not a byte section and not one of the five semantic types. |
| Semantic role | What mathematical information kind does this atom carry? | `SKELETON`, `CONNECTION`, `FIBER`, `GAUGE`, `RESIDUAL` from TS1 | Not an LP1 layer, physical archive member, or scientific object name. |
| TS1 information home | At what minimum-description layer is the information first owned? | `L1_program`, `L2_chart`, `L3_raster`, `L4_scorer_feature`, `L5_verdict` | Not SN1's identically numbered measurement stack. |
| SN1 artifact layer | At what recursive evaluator artifact surface was evidence observed? | `L1_PROGRAM`, `L2_RECEIVER_R`, `L3_SCORER_FEATURE`, `L4_SCORER_DECISION`, `L5_VERDICT` | `L2`/`L3` integers cannot be cast to TS1 or LP1 by position. |
| SN1 derivation recursion | Which evaluator derivation generated the type? | `L0_SCORE_SIGNATURE`, `L1_TERM_NATIVE_GEOMETRY`, `L2_TEMPORAL_COMPOSITION` | Not an archive home or pipeline stage. |
| LP1 pricing stratum | What deepest information/pricing surface was proven for a measured object? | `L1_PROGRAM`, `L2_CHART`, `L3_RGB`, `L4_SCORER_FEATURE`, with LP1's `PROGRAM/CONTEXT/FIBER/RESIDUAL/GAUGE` accounting labels | LP1's local accounting vocabulary is not the five-type semantic SoT. |
| Placement | Is the artifact free decoder mechanism, counted video state, encoder-only evidence, or charged packaged executable/table? | `GENERIC_DECODER_FREE`, `COUNTED_VIDEO_STATISTIC`, `ENCODER_ONLY_EVIDENCE`, `COUNTED_PACKAGED_EXECUTABLE` from G23 design | “Executable” does not imply free; “analytic” does not imply free. |
| Physical coding | Which exact bytes are charged, by which coder, and consumed where? | archive SHA, member, offset, length, range hash, coding-group ID, coder owner, receiver consumer, container owner | A logical owner or estimated entropy cannot stand in for exact ZIP bytes. |
| Function | What receiver transformation/evaluator quotient action is implemented? | canonical instruction graph + decoder contract + input/output type + functional-equivalence receipt | Not parameter bytes, source-code hash alone, or output digest alone. |
| Spelling | How is one function represented? | exact opcodes, operands, quantization, ordering, transforms, contexts, dictionaries, resets | A same-function repack is not a semantic mutation. |
| Authority | What conclusions may this evidence support? | a sealed authority variant with axis, sample scope, runtime, scorer, archive/decode custody, and promotion capability | A string such as `macos_advisory` or `contest_cpu` is not authority by itself. |
| Proof dependency | Which identities can make this proposition stale? | typed proposition plus exact dependency set, invariant artifact identities, and separately rebasable semantic controls | Whole-object dataclass equality and “hash everything nearby” are not validity rules. |
| Effect | Is this an endpoint transition or a higher-order nonadditive observation? | `TransitionEffect` or `InteractionHyperedge(arity>=2)` | Interaction residuals may not be spread across atomic action credits. |
| R10 constraint | Which nonlinear feature property must survive? | amplitude, frequency, phase, contrast, channel energy, texture, feature relay, Pose multiple-shooting | Not an eighth byte stream and not optional telemetry. |

### 1.1 The numbered layer collision is a real bug source

Three historical surfaces use `L1` through `L5` for related but non-identical
objects:

```text
TS1: L1 program -> L2 chart -> L3 raster -> L4 scorer feature -> L5 verdict
SN1: L1 program -> L2 receiver/R -> L3 scorer feature -> L4 decision -> L5 verdict
LP1: L1 program -> L2 chart grammar -> L3 RGB realization -> L4 scorer feature
```

The compiler must carry
`RecursionCoordinate(namespace, stage)` where `namespace` is at least
`TS1_INFORMATION_HOME`, `SN1_ARTIFACT_LAYER`,
`SN1_DERIVATION_RECURSION`, or `LP1_PRICING_STRATUM`. Crosswalks are explicit
proof objects. Numeric equality is not compatibility. This is how we retain
SN1 evidence without silently moving it one layer when adapting it to TS1 or
LP1.

### 1.2 The five semantic roles are necessary but not sufficient

The five roles are derived from evaluator recursion:

- `SKELETON`: partition/interface/topology obligations;
- `CONNECTION`: chronology, transport, prediction, and event linkage;
- `FIBER`: within-cell/scorer-visible realization coordinates;
- `GAUGE`: exactly receiver/scorer-null freedom, with zero counted bytes only
  when the representative is deterministically derivable; and
- `RESIDUAL`: remaining scorer-visible debt not rehomed into the other types.

They do not say which class/world-model object owns an atom, where a physical
byte is stored, or how strong the evidence is. In particular, `GAUGE` is not a
blanket “palette/gauge is free” label. A video-selected palette index,
representative selector, fitted nullspace coefficient, or gauge-switch bit is
counted under the earliest non-null role that consumes it unless the decoder
derives it generically. Only the null freedom itself is zero-byte.

### 1.3 Seven scientific roles map many-to-many into five semantic roles

This table is an admissible crosswalk, not a preassigned allocation. Each
populated atom must carry its actual semantic role.

| Scientific role | Admissible semantic roles | Required dependency meaning |
|---|---|---|
| topology/worldsheet | `SKELETON`, `CONNECTION` | cells/interfaces plus temporal worldsheet incidence |
| Road/Undrivable bulk boundary | `SKELETON`, `CONNECTION`, `FIBER` | boundary topology, transport, and boundary realization parameters |
| Lane chart and phase | `SKELETON`, `CONNECTION`, `FIBER` | lane support, dash/phase chronology, chart values |
| Movable islands + MyCar closure | `SKELETON`, `CONNECTION`, `FIBER` | birth/death/incidence, tracks, shapes/appearance |
| cell/value + two-plane preimage | `FIBER`, exact `GAUGE`, `RESIDUAL` | cell representative, free fill, and irreducible realization escape |
| xi/frame0/prosody/Pose feature relay | `CONNECTION`, `FIBER`, exact `GAUGE`, `RESIDUAL` | chronological coupling, fitted photometry, free representatives, last-resort escape |
| irreducible quotient | `RESIDUAL` | terminal debt only after all prior dependency routes are actually consumed |

The entropy coder may jointly encode atoms from any of these rows. That does
not merge their logical ownership.

## 2. Lifecycle as a sealed type chain

The lifecycle must be executable as six different types. Each constructor
reopens and derives the exact parent identity; none accepts a parent digest as
sufficient proof.

### 2.1 `SourceTruth`

Required retained evidence:

- exact source video/content identity;
- frozen evaluator source, weights, runtime, resize/R, and pair-order identity;
- canonical source pair population;
- target observations as encoder-only evidence; and
- originality/payload-lineage declaration.

It is not a candidate payload and cannot be decoded from the archive.

### 2.2 `ObligationIR`

Required foreign keys:

- exact `SourceTruth` identity;
- exact predictor state and freshly decoded predictor semantics;
- one `PairPopulation` map;
- semantic/scientific role ownership;
- complete or sparse-owned obligation coverage;
- conditional frame-0 Pose obligations bound to exact frame-1 obligations;
- R10 constraint rows where needed; and
- encoder-only oracle/teacher evidence typed as forbidden candidate payload.

### 2.3 `RealizedPair`

Required values:

- actual retained `uint8 Y0/Y1` or an exact bounded batch iterator whose bytes
  are rehashed while consumed;
- source/local pair coordinate;
- obligation coverage receipt;
- actual explicit-preimage packet bytes and strict parse receipt;
- exact R/resize proof; and
- frame-0/Pose ownership receipt.

A `RealizedPair` is not an archive. It has no rate authority.

### 2.4 `ArchiveArtifact`

Required values:

- actual immutable `archive.zip` bytes/path, SHA-256, and exact size;
- deterministic member map and canonical parse/re-encode proof;
- compiler-placement manifest;
- `PhysicalCodingGroup` records covering every byte exactly once, including
  headers and container overhead;
- decoder program/runtime identity; and
- no dead counted section.

### 2.5 `DecodeReceipt`

Required values:

- freshly reopened exact `ArchiveArtifact`, not its caller-supplied digest;
- exact decoder/runtime/asset identity;
- every physical coding group's named receiver consumer;
- canonical pair order and full population closure;
- raw/native output hashes and realized `Y0/Y1` identities; and
- deterministic double-decode equality.

### 2.6 `ScoreReceipt(axis)`

Required values:

- exact `DecodeReceipt` foreign key;
- sealed `AuthorityEvidence` variant;
- frozen scorer/R/runtime/hardware identities for that authority axis;
- exact `d_seg`, `d_pose`, archive bytes, component terms, sample count, and
  nonlinear total.

CPU and CUDA are distinct variants. A local/advisory receipt cannot be cast to
either. A public leaderboard display is a target observation, not our
`ScoreReceipt`. The dynamic pointer belongs only to a separate rebasable
`AdmissionReceipt(score_receipt, pointer_snapshot)`, never to score validity.

### 2.7 Proof validity is dependency-scoped, not whole-object equality

Three identities must remain distinct:

1. `ArtifactIdentity`: immutable bytes and parser/type identity;
2. `SemanticControlIdentity`: current pointer, target sublevel, selection
   policy, or another mutable control used to decide what to do; and
3. `ProofValidity`: one proposition plus the exact identities on which that
   proposition logically depends.

The current failures are concrete. G14 metadata pointer refresh aborted through
dataclass equality even though not every G14 proposition depended on that
pointer. G22 replay initially would have invalidated a decoded-byte equality
receipt merely because the pointer changed later. Conversely, an admission
claim that really compares against the frontier must be rebased whenever the
current pointer changes.

Required model:

```text
ProofReceipt[P]:
  proposition_type: P
  subject_artifact_identities
  dependency_set: exact typed identities used to prove P
  semantic_controls_observed: controls used only by downstream decisions
  proof_body/result
```

For `DecodedBytesEqual`, the dependency set contains the two exact artifacts,
decoder/runtime/assets, pair order, and comparison algorithm. It does not
contain the competitive pointer. A pointer refresh cannot revoke the proof.
For `BeatsCurrentFrontier`, the dependency set contains the exact score receipt
and the current pointer manifest; changing that manifest makes only the
admission comparison stale and triggers deterministic rebase. The underlying
archive, decode, and score receipts remain valid.

Proof dependency is semantic, not temporal: a later timestamp does not make a
proof stale unless a declared dependency changed. Every proof constructor must
declare a closed dependency set; unknown implicit dependencies block use.

## 3. Authority is a type, not a string

The June quotient audit already derived this rule, but current implementation
still violates its strong form:

- `tac.analysis.action_effect.ScoreAuthority` exists, yet
  `ActionEffect.authority` is a nonempty `str` and unknown strings are retained;
- `AppliedActionReceipt.authority_axis` is a `str` and only equality-compared
  with the `ActionEffect` string;
- G19 `FeedbackExactBaseBindingV1.axis` and replay axes are strings; and
- several `truth` mappings contain booleans that can drift independently.

The compiler requires a sealed sum type, for example:

```text
AuthorityEvidence =
  StructuralMechanism(sample_scope, synthetic_or_real, receipt)
  | AdvisoryLocal(axis, sample_scope, scorer, runtime, decode_receipt)
  | ReceiverClosedLocalN600(axis, scorer, runtime, archive, decode_receipt)
  | ExactContestCPU(archive, decode_receipt, evaluator_receipt, hardware)
  | ExactContestCUDA(archive, decode_receipt, evaluator_receipt, hardware)
```

The exact names may follow the canonical custody package, but the invariants
are non-negotiable:

1. construction, not a string comparison, proves membership;
2. sample scope and authority axis are separate fields;
3. no implicit or lossy conversion points upward;
4. CPU/CUDA never infer each other;
5. partial interaction evidence never becomes a score receipt; and
6. promotion capability is a property of the variant, never a mutable boolean.

Authority and proof dependency are orthogonal. Authority says what a valid
proof may support. `ProofDependencySet` says whether that proof remains valid
after external state changes.

## 4. Logical ownership, physical coding, and exact bytes

### 4.1 One logical owner does not imply one physical file

The correct representation is bipartite:

```text
LogicalOwnershipAtom --(symbol/extraction incidence)--> PhysicalCodingGroup
PhysicalCodingGroup --(decoder consumer)-------------> realized coordinates
```

`LogicalOwnershipAtom` owns one function/constraint/degree of freedom exactly
once. `PhysicalCodingGroup` owns one exact charged archive range exactly once.
A group may jointly code many logical atoms; an atom may be reconstructed from
several ranges. Exact rate is the union of physical groups, not the sum of
logical attributions.

Each physical group must carry:

- archive SHA and member path;
- exact offset, length, and range SHA;
- container/coder profile and owner;
- receiver consumer ID and parse rule;
- ordered logical-incidence selectors;
- generic decoder operation IDs and counted operand spans;
- counterfactual/deletion receipt if a marginal byte value is claimed; and
- an explicit shared/joint coding flag.

### 4.2 Current `StreamHomeClaim` is too narrow

`AppliedActionReceipt.StreamHomeClaim` binds one `StreamType`, one
`LayerHome`, one byte-home ID, one coder, and requires
`ActionEffect.delta_bytes == stream_home.delta_bytes`. That works only for an
independently accounted stream. It cannot faithfully represent a merge/share,
factor, dictionary, context, or outer-ZIP action whose byte movement is jointly
owned by several logical atoms. Forcing one role onto such an action either
double-counts bytes or suppresses cross-role entropy signal.

The successor must bind one or more logical ownership atoms to one or more
physical coding groups and retain exact whole-archive delta separately. A
per-role marginal is null unless a matched physical counterfactual identifies
it. This preserves the EV2 seven-home lesson and avoids fabricating 162
cellwise byte owners from jointly coded bytes.

### 4.3 Functional quotient identity is separate from spelling

The HOPE cross-check is useful here: functional/operator identity must be
separate from parameter spelling, and prune/merge/macro eviction belongs in
one action algebra. Its static-payoff/item-independence assumption is not
portable to this contest because outer ZIP, receiver nonlinearity, and action
order interact.

The compiler therefore needs:

- `FunctionalQuotientIdentity`: canonical receiver instruction graph,
  operation contracts, input/output types, evaluator-obligation action, and
  equivalence relation;
- `ParameterSpellingIdentity`: exact video-derived opcodes/operands,
  quantization, ordering, transform, dictionary/context/reset choices; and
- `PhysicalObjectIdentity`: exact archive ranges and whole ZIP identity.

A `REQUANTIZE_STORAGE`, reorder, transpose, or entropy repack may change
spelling and physical identity while preserving functional identity and
decoded output. A `PRUNE_DELETE` may preserve function only if its deleted
atom is proven dead. A `REPLACE`, `MIGRATE`, or realization action may change
function and must be remeasured. All of them rebuild and reprice the exact
whole archive. Static item payoffs are proposal metadata only.

## 5. The five C0B hidden joins: exact invariants and present status

### J1 — reopened-object-derived identities

Invariant:

```text
every downstream digest = H(canonical bytes freshly reopened by its typed parser)
```

Caller-provided hashes may be comparison hints only. The constructor must
reopen state, predictor, IR, preimage, archive, and receipts; derive identities;
and reject a hint that differs.

Present status: `ReopenedObjectJoin` reopens `CoupledWitnessState`, the V9
predictor program, `EvaluatorObligationIR`, and
`ExplicitV10PreimageCompileResult`. This is useful. It is not the complete
lifecycle and does not reopen the explicit V10 packet itself.

### J2 — complete or sparse-owned obligation coverage

Invariant for universe `U` of IR obligations:

```text
U = predictor_owned disjoint_union residual_owned
predictor_owned intersect residual_owned = empty
```

`COMPLETE` means the candidate receiver realizes all obligations directly.
`SPARSE_OWNED` means every unfulfilled coordinate has exactly one named,
receiver-consumed counted owner. Missing, duplicate, dead, or encoder-only
owners refuse.

Present status: `IRCoveragePolicy`, `SparseObligationOwnership`, and
`IRCoverageReceipt` exist in `pair_population_envelope.py`. The selected-
solution compiler still must carry that receipt through archive and decode.
G23's new `G17ObligationCoverageV1` is only a pair-ID coverage declaration; it
does not supersede obligation-level coverage or prove a live receiver owner for
each sparse debt.

### J3 — hashed `PairPopulation` coordinate map

Invariant: one canonical map binds global/source IDs to V9, PBR, IR, and V10
local coordinates. It is bijective over the declared population, preserves
order where required, and is hash-bound into every downstream stage.

Present status: `PairPopulation`, `PairDomainIndex`, and `PairCoordinateRow`
exist and reconstruct serialized rows from domain indexes. G17/G19 do not yet
consume this as their sole population identity; G19 merely binds an external
manifest plus two source IDs in its n2 context. G23's new
`G17PairPopulationV1` preserves source/V9 order and a zero-based IR/V10 local
tuple, but collapses the distinct PBR, IR, and V10 mappings required by this
join.

### J4 — exclusive V9 Pose6 versus frame-0 residual ownership

Invariant: each Pose/frame-0 degree of freedom is in exactly one of:

```text
V9_predictor_pose6
frame0_residual_beyond_V9_conditioned_on_exact_Y1
reverse_causal_frame0_from_exact_Y1
```

No packet stores absolute Pose6 while claiming residual ownership. A coupled
interaction may reference several owners but may not duplicate their values.

Present status: `CompactGeneratorDecode` refuses absolute Pose6 and requires
`RESIDUAL_BEYOND_V9_POSE6`; the current pair-population surface also names a
separate reverse-causal `FRAME0_FROM_EXACT_Y1` grammar. These grammars are not
yet unified under one whole-archive ownership proof across all G17 A branches.
G23's `G17PosePreimageOwnershipV1` chooses V9 Pose6 or frame-zero residual per
pair, but neither includes the reverse-causal variant nor binds the selected
owner to exact A packet spans and receiver operations.

### J5 — actual explicit-preimage packet bytes

Invariant: if a join, falsifier, or teacher claim relies on an explicit V10
packet, the exact packet bytes are retained as encoder evidence, reopened by
the real strict parser, and consumed to derive the receipt. Hash and size alone
are insufficient. Those dense evidence bytes remain forbidden candidate
payload unless a new current-lineage counted description independently owns
them.

Present status: **OPEN/P0**. `ExplicitV10PreimageCompileResult` explicitly calls
itself a “Hash-only receipt” and stores only `receiver_packet_sha256` and
`receiver_packet_bytes`. `ReopenedObjectJoin.reopen(...)` accepts
`explicit_preimage_result_bytes` but no `receiver_packet_bytes`; its identity
dictionary repeats the attested hash and size. The compact-program seal proves
that a separate counted program freshly generates matching Y identities, which
is valuable, but that does not reopen the explicit packet claimed by the
encoder-evidence join. G23 now contains a real strict generalized-XIP2-A packet
and decoder, but `G17PosePreimageOwnershipV1` accepts its evidence field as
opaque bytes and never calls that parser. The product compiler must require the
actual packet bytes plus strict parser/consumer receipt, or explicitly remove
every claim that depends on them.

## 6. R10 is a cross-level constraint language, not a sidecar

R10 preserves information lost by pixel-only and low-rank-only descriptions:

- amplitude;
- frequency;
- phase;
- contrast;
- per-channel energy;
- texture;
- frozen-feature relay constraints; and
- Pose multiple-shooting knots/continuity through nonlinear blocks.

These coordinates couple tensor, code, time, semantic/realization/Pose, and
analytic-versus-learned axes. They are not an eighth independent entropy file.
Generic relay, projection, shooting, interpolation, and solve operations may
live in `inflate.py`; every video-fitted value, selected constraint, knot,
parameter, exception, and bytecode operand is counted.

Required executable types:

```text
R10Constraint =
  Amplitude(target_or_interval, support, tolerance)
  | Frequency(band_or_mode, support, tolerance)
  | Phase(reference, chronology, tolerance)
  | Contrast(statistic, support, tolerance)
  | ChannelEnergy(channel, statistic, support, tolerance)
  | Texture(statistic_or_code, support, tolerance)
  | FeatureRelay(block_id, feature_statistic, shooting_node, tolerance)
  | PoseMultipleShooting(ordered_knots, continuity_constraints, terminal_pose6)
```

Every row foreign-keys `SourceTruth`, `PairPopulation`, exact frame role,
scientific role, semantic role, frozen feature/block identity, and a generic
decoder operation plus counted operands. The standalone decoder never runs
PoseNet and never infers a missing teacher constraint. Unsupported constraints
must produce `G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED`, not silently
be omitted.

G23 now has the correct closed R10 constraint-name enum and the named blocker,
but its relay payloads are unparsed byte strings and its consumption check is
only equality with the enum list. That is not the typed executable form above:
the caller can say every name was consumed without any payload affecting a
receiver operation. The byte VM's current operations cannot implement these
physics. Keep the blocker live until mutation of each counted R10 operand is
observed at its named receiver consumer or refuses deterministically.

Required matched controls remain:

- pixel/tensor factorization alone;
- R10 relay alone;
- their joint branch;
- phase/time permutation;
- channel-energy-preserving shuffle;
- texture deletion; and
- shooting-knot delete/merge.

Missing controls make R10 value unpriced, not zero.

## 7. Higher-order effects and the live G14 partial signal

G19 correctly preserves G8-by-A four-cell residuals as indivisible effects and
refuses reconstruction from component residuals or local score derivatives.
The current live G14 matched-control signal sharpens the required type:
negative/nonlinear interaction is primarily Pose-side, with zero Seg
interaction and near-zero byte interaction in the observed partial. Those
zeros are coordinates of one coupled observation. They do not authorize
splitting out an independent “Pose credit,” and the partial is not a family or
n600 verdict.

The general successor is not a fixed binary commutator row. It is:

```text
InteractionHyperedge:
  interaction_id
  ordered_action_ids[arity >= 2]
  exact_parent_object_identity
  measured_corner_lattice_or_partial_support
  archive/decode/scorer identity per measured corner
  delta_d_seg_residual
  delta_d_pose_residual
  delta_archive_bytes_residual
  independently_computed_joint_score_residual
  authority: AuthorityEvidence
  sample_scope
  terminal_effect_consumer
  causal_owner_set
  completeness: PARTIAL | COMPLETE
  verdict_scope
  credit_decomposition: FORBIDDEN | INDEPENDENTLY_MEASURED
```

For the G14/G18 case, `terminal_effect_consumer` may be the Pose relay/A path,
but `causal_owner_set` must retain both G8 and A. The negative Pose-side
interaction belongs to the hyperedge. It is not divided between the atomic
actions. A higher-arity macro retains its exact ordered parent set and is
recomputed after any parent archive changes.

## 8. Fail-closed acceptance suite

These are production-ABI structural tests. Small fixtures may test parser
mechanics only; no fixture supports a scientific verdict. Real scientific
admission remains complete n600 and exact same-object measurement.

### A. Namespace and type separation

1. Construct one atom with all independent axes and prove canonical
   parse/re-emit identity.
2. Pass a TS1 layer where an SN1/LP1 recursion coordinate is expected; reject
   even when both serialize as `L2` or `L3`.
3. Pass a semantic `SKELETON` value as a scientific role or physical home;
   reject rather than coerce.
4. Encode two scientific roles and three semantic roles into one joint coding
   group; accept while preserving all logical ownership rows and charging the
   physical range once.
5. Duplicate one logical ownership atom across two claimed owners; reject.
6. Classify a fitted palette/gauge selector as zero-byte `GAUGE`; reject unless
   a deterministic derivation/equivalence proof establishes no counted operand.

### B. Lifecycle and authority

7. Attempt to construct `ArchiveArtifact` from a hash without archive bytes;
   reject.
8. Attempt to construct `DecodeReceipt` from an archive digest without fresh
   parse/inflate; reject.
9. Attempt to construct `ScoreReceipt` from a decode digest or local score
   mapping without a sealed authority variant; reject. In particular, attaching
   `C0BScoreAxisV1.CONTEST_CPU` or `CONTEST_CUDA` to a research-only supplied
   forward observation must not mint contest authority.
10. Attempt macOS/advisory -> contest CPU, CPU -> CUDA, subset -> n600, or
    partial interaction -> score-receipt conversion; no conversion exists.
11. Mutate scorer/runtime/hardware/axis/archive/decode custody inside a sealed
    authority object; reject on reconstruct.
12. Recompute the exact nonlinear score and component terms from the retained
    values; reject any stored arithmetic drift or fixed independent gate.
13. Refresh only the dynamic pointer after a decode-equality proof; preserve the
    decode proof byte-for-byte while marking only frontier admission for
    rebase.
14. Refresh the pointer after `BeatsCurrentFrontier`; require a new comparison
    receipt against the new pointer without rerunning decode/score whose own
    dependencies are unchanged.
15. Mutate archive, receiver, runtime, pair order, or equality algorithm; revoke
    the decode-equality proof even if the pointer is unchanged.
16. Add an undeclared external control read to a proof producer; fail the
    dependency-closure test rather than hashing the entire process context.

### C. The five C0B joins

17. Supply a correct caller digest with different reopened state/predictor/IR
    bytes; reject based on reopened identity.
18. Create complete coverage with one missing obligation, one duplicate, or an
    encoder-only/dead owner; reject.
19. Create sparse-owned coverage with exactly one live counted receiver owner
    per debt; accept; deleting/mutating that section must change output or
    produce a typed receiver refusal.
20. Permute, duplicate, omit, or cross-map one V9/PBR/IR/V10 pair coordinate;
    reject and prove the serialized population hash changes.
21. Store absolute Pose6 plus a frame-0 residual, or claim the same frame-0
    coordinate in the legacy and reverse-causal grammars; reject.
22. Supply an explicit-preimage receipt with correct packet hash/size but no
    packet bytes; reject. Supply bytes with one mutation; the strict parser or
    derived receipt must reject. A hash-only fake must never pass.

### D. Physical coder and functional identity

23. Require exact archive member/range coverage with no gap, overlap, dead
    bytes, or double-counted container overhead.
24. Mutate each physical coding group; the named receiver consumer must either
   change its typed output/receipt or fail closed. A no-op counted group is a
   blocker; a caller-provided `receiver_consumed=True` is not evidence.
25. Repack/reorder/requantize an object while claiming functional equivalence;
    require identical decoded output and functional identity but distinct
    spelling/physical identities, plus exact whole-ZIP repricing.
26. Delete/merge/factor a logical atom; require physical deletion of its bytes
    and dead dependencies, ownership conservation, fresh archive, and fresh
    decode. Zeroing bytes is not deletion.
27. Joint-code several atoms and verify that no action receipt fabricates a
    per-role byte delta; only measured counterfactuals may populate marginals.

### E. R10 and interactions

28. Round-trip every R10 constraint variant and require a concrete generic
   decoder operation plus counted operand span.
29. Remove an R10 consumer, shooting knot, block foreign key, or fitted
   operand; reject executable closure rather than defaulting to zero effect.
   Passing the complete constraint-name enum tuple without executing payloads
   must also reject.
30. Verify the decoder has no scorer/teacher/target access and still executes
    every admitted R10 operation deterministically.
31. Preserve a complete four-cell G8-by-A observation as one hyperedge; verify
    that no serializer/controller emits two atomic credits from it.
32. Preserve a partial Pose-dominant interaction with zero dseg and near-zero
    byte residual as `PARTIAL`, nonpromotional, and nondecomposable.
33. For higher arity, require the declared corner support or mark it partial;
    missing corners are absent evidence, never numeric zero.
34. Change action order, parent archive, outer codec, or one corner identity;
    invalidate/reprice the interaction rather than reusing it.

### F. Payload placement and originality

35. Enumerate every instruction, operand, table, seed, selector, constraint,
    context, executable, and evidence object exactly once in placement.
36. Reject target-selected constants hidden as generic code, mixed generic and
    video-derived blobs, scorer/GT/oracle/teacher payload, decoder-dead counted
    bytes, and public/donor candidate payload.
37. Accept arbitrarily sophisticated generic decoder algorithms only when all
    video-specific operands are counted and full deterministic runtime remains
    within the contest wall.
38. Build archive twice and decode twice; require byte-identical archive and
    outputs with exact full-population order.

## 9. G17/G19/G21/G23 punch-list

### G17 — frozen architecture and packet chronology

Preserve:

- one coupled exact score surface, never independent Seg/Pose/rate gates;
- P-once population framing and exact `P -> G -> A -> T` chronology;
- complete canonical shard coverage and pair order;
- forward-observation custody and fresh post-topology/post-G8/A identities;
- substitutive action algebra and physical deletion/ownership conservation;
- solution-description primacy and generic-decoder/counted-operand boundary;
- seven-factor homotopy, exact whole-object pricing, and gauge alternatives
  surviving through Pose/rate selection.

Omissions to close:

1. G23 now supplies `G17CompilerPlacementManifestV1` and lifecycle dataclasses,
   but they are nominative types, not actual archive/receiver/scorer
   constructors. Close the executable joins and authority boundary.
2. G17 packet sections have physical syntax but no complete semantic-role x
   scientific-role x physical-coding ownership incidence.
3. P-once population framing does not itself supply the hashed C0B
   `PairPopulation` object to every stage.
4. R10 is architectural only.
5. The actual explicit-preimage packet-byte join is absent.
6. G17 forward and score authority must use a sealed authority variant, not a
   digest/boolean/string bundle.
7. proof receipts must bind only their real dependencies: forward/decode
   invariants survive irrelevant pointer refresh, while admission comparisons
   rebase to the current pointer.

### G19 — interaction costate/control bridge

Preserve:

- transition and interaction as separate, indivisible control observations;
- exact-base archive/population binding;
- nonlinear interaction score residual independent of component residuals;
- action semantics beyond ADD;
- factor axes, generic decoder versus counted fields, and no authority upgrade;
- blocked handoffs when n2 advisory evidence lacks consumer authority.

Omissions to close:

1. `axis`/authority is string-typed.
2. factorization coordinates and action semantics are controller declarations,
   not mechanism-verified ownership/receiver edges.
3. `DecoderPlacementV1` hashes evidence but deliberately does not dereference
   program operands or physical archive ranges.
4. no lifecycle types, full `PairPopulation` membership proof, five semantic
   roles, seven scientific roles, or exact joint coding groups are carried.
5. no R10 constraint values or Pose multiple-shooting receiver consumer.
6. interactions are fixed to the G8-by-A four-cell shape; the selected-solution
   IR needs general ordered hyperedges and partial-support typing.
7. Pose-dominant interaction effects need terminal Pose-relay consumer plus the
   complete causal owner set, never credit reassignment to A alone.
8. controller state must not use whole-dataclass equality as proof validity;
   observation dependencies and rebasable pointer controls need separate
   identities.

### G21 — complete design, canonical owner resolved during audit

Preserve:

- the seven coupled factorization levels;
- physical actuator/ownership IR;
- decoder-computable constraint program;
- full-population membership/receive proof;
- mechanism-verified action semantics;
- all five C0B joins;
- lifecycle chain;
- logical ownership compatible with joint entropy coding; and
- R10 as a cross-level relay with required matched controls.

Closed during this audit:

1. G23 moved all definitions into
   `src/tac/witness_dsl/taskspace_selected_solution_compiler.py`; the old
   `taskspace_g17_compiler_placement.py` is a thin re-export, and a 39-export
   identity audit found no duplicate objects.

Omissions to close:

1. The concurrent implementation has lifecycle wrappers but still lacks an
   executable lifecycle constructor, physical coding group, authority type,
   R10 consumer, and general interaction edge.
2. Its prose saying logical typing assigns “ownership and recursion” must not
   collapse semantic role into TS1/SN1/LP1 recursion namespaces.
3. Wire typed proof dependency cuts so homotopy evidence is not invalidated by
   unrelated control refresh and current-frontier decisions never go stale.

### G23 — emerging vertical bundle

Preserve from the spec:

- explicit semantic, recursion, and placement axes;
- nonaliasing logical roles;
- lifecycle ordering;
- C0B PairPopulation/coverage/Pose/preimage requirements;
- R10 typed object and blocker;
- generic instruction semantics separated from counted operands;
- actual VM reconstruction, unsupported-operation refusal, full-population
  packet chronology, exact G7 callbacks, immutable resume, and false-authority
  truth.

Must add/tighten before implementation acceptance (based on the exact
concurrent hashes in the evidence snapshot):

1. Preserve the now-resolved single canonical owner and add an explicit
   adapter-identity regression test.
2. Validate stages against their versioned recursion namespaces; the new
   coordinate currently accepts an arbitrary ASCII `stage` string.
3. Represent exact `PhysicalCodingGroup` ranges and many-to-many logical
   incidence, not only four placement classes and a receiver-consumed boolean;
4. add physical coder owner, receiver consumer, and exact-byte owner, then wire
   the new functional-identity and parameter-spelling types separately through
   placement, lifecycle, and actions;
5. replace authority strings/booleans with sealed authority variants;
6. bind actual V10/generalized-XIP2 explicit-preimage packet bytes through its
   strict parser into Pose ownership, not the current hash-only/opaque evidence
   path;
7. carry obligation-level coverage, the full V9/PBR/IR/V10 pair map, Pose
   ownership, R10, placement, and byte groups through
   `ArchiveArtifact -> DecodeReceipt`, not only compiler-time validation;
8. represent general interaction hyperedges and the current partial
   Pose-dominant G14 signal without decomposed credit;
9. enforce the exact GAUGE rule so video-selected representatives/operands are
   not accidentally made free;
10. make joint entropy coding first-class and prevent per-role byte-sum
    accounting; and
11. keep the seven scientific roles separate from the five semantic roles.
12. carry `ProofDependencySet` separately from artifact hashes and pointer
    observations; decoded equality is invariant to pointer refresh, candidate
    admission is not.
13. replace caller assertions (`receiver_consumed`, full consumed-enum tuple,
    supplied forward arrays) with receipts derived by actual receiver/scorer
    execution; an axis enum must never upgrade research evidence.

At the refreshed audit snapshot, five implementation files exist and no
matching G17 test file exists. The honest status is
`SUBSTANTIAL_EXECUTABLE_COMPONENTS_PRESENT / SELECTED_SOLUTION_PRODUCT_AND_AUTHORITY_CHAIN_OPEN`.

## 10. Ordered closure roadmap

This is not a new research detour. It is the minimum coherent implementation
order required to stop losing signal while G23 closes the full vertical.

1. **Preserve the now-resolved canonical ownership.** The selected-solution
   compiler is the definitions owner and the G17 path is an identity-preserving
   adapter. Add the regression test so the rediscovery/orphan failure cannot
   recur.
2. **Complete the orthogonal core types.** Reuse the landed logical/semantic,
   placement, and lifecycle definitions; add scientific role, versioned
   recursion namespace, logical owner, physical coding group, function,
   spelling, sealed authority, and proof-dependency types with structural
   parse/mutation tests.
3. **Close the five joins on actual objects.** Reuse current
   `PairPopulation`/coverage work, add actual explicit-preimage packet reopen,
   and carry the joins across archive/decode.
4. **Land the receiver-executable instruction/operand IR.** Generic operations
   belong in inflate; fitted instructions and operands are counted. Mutation
   tests prove each physical group reaches its consumer.
5. **Add R10 constraints and Pose ownership.** Do not score or price R10 until
   each constraint has a receiver consumer and matched controls.
6. **Generalize effect custody.** Adapt G19 transitions and interactions into
   typed endpoint/hyperedge observations with physical groups, functional
   identity, authority, and no decomposed interaction credit.
7. **Compile the coarse complete n600 vertical.** It may be bad. It must be one
   full object with all joins, exact ZIP, fresh decode, same-object local score,
   per-class/per-pair debt, and 25 IS1 rows. Only then do costates have one
   coherent object to optimize.
8. **Optimize recursively across all axes.** Alternate semantic topology,
   realization/R10, conditional Pose, factorization, prune/merge/migrate,
   entropy spelling, and outer coding. Every survivor is a rebuilt complete
   object; exact finite score and whole ZIP admit it.
9. **Branch complete eligible bytes to governed authority.** CPU and CUDA are
   separate receipts. Bank any break of the freshly refreshed dynamic
   `effective_frontier`; thereafter continue toward the subordinate sub-`0.15`
   stretch aim.

## 11. What this audit rules out

- It does not call for a sixth semantic type or an eighth scientific stream.
- It does not require five or seven independent archive files.
- It does not reopen fixed independent Seg/Pose/rate gates.
- It does not authorize using a historical lattice, public archive, scorer,
  teacher, target, or explicit dense preimage as candidate payload.
- It does not make analytic/video-fitted values free because their decoder
  operation is generic.
- It does not treat a hash as custody of absent bytes.
- It does not turn n2/partial interaction evidence into an n600 or authority
  verdict.
- It does not add another controller, ontology, or compiler in parallel.

The forest-level conclusion is that the codec already has most of its
mathematical vocabulary. The missing capstone is a single executable type
system that conserves identity, ownership, authority, and higher-order signal
from source obligation through exact scored bytes. Once those joins are real,
the existing V9/V10/lattice/factorization/costate machinery can optimize one
coherent object instead of repeatedly rediscovering incompatible fragments.

## Stores consulted

- full `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, and operating manual;
- canonical pointer/lane/checkpoint state and current shared-tree ownership;
- original codec synthesis, C0B gestalt/composition, consumption audit, and
  session summary;
- TS1 five-type directive/findings/source, SN1 addendum/findings/source, and
  LP1 findings/source;
- G17 frozen spec, G18 interaction spec, G19 implementation/findings, G21
  homotopy spec, and emerging G23 spec;
- `coupled_witness_state.py`, `evaluator_obligation_ir.py`,
  `pair_population_envelope.py`, `action_effect.py`, and
  `applied_action_receipt.py`;
- June score-program/compiler quotient validation and the HOPE mechanism
  cross-check supplied by root.

HISTORICAL_PROVENANCE: append-only G24 adversarial audit of the missing
type/layer selected-solution compiler contract. No prior spec, result, pointer,
candidate, or owner file was rewritten.
