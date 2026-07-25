# Codex findings — original task-space inverse witness codec

UTC: 2026-07-25

Lane: `lane_codex_original_taskspace_inverse_codec_20260725`

Mode: `research_only=true`; local read, verify, and build only; no paid
dispatch, no contest evaluation, and no promotion

Score claim: `false`

Competitive pointer: dynamically read from
`.omx/state/canonical_frontier_pointer.json#effective_frontier`; refreshed
snapshot `0.172 [official-leaderboard display]`, external target only.

Mission gate: authoritative exact score strictly below `0.15`. A first score
below the live competitive pointer is a milestone, not the campaign stop.

## Outcome

Verdict:
`PROPOSED_ORIGINAL_CODEC_ARCHITECTURE_SYNTHESIZED / C0_IDENTITY_SCAFFOLD_LANDED / COMPLETE_N600_ARCHIVE_STILL_OWED`.

The project is indeed creating a new codec, in the same broad sense that MP4 or
JPEG define a representation, syntax, reconstruction process, and entropy
layer. This codec is specialized to one frozen contest information space. It
does not minimize human-visible reconstruction error. It minimizes

\[
S(A)=100d_{seg}(D(A))+\sqrt{10d_{pose}(D(A))}
  +25|A|/37{,}545{,}489
\]

over legal archives `A`, where `D` is the deterministic contest receiver and
the two distortions are computed only by the frozen evaluator.

The counted state should be described as a *candidate sufficient statistic*
for those evaluator obligations. Global minimality is not proven. The correct
object is a content-addressed scientific state compiled into a real archive,
not a bag of deltas, packets, models, or receipts borrowed from unrelated
archive identities.

Pointer delta: `0`. This pass produced no scored n600 candidate and relabelled
no historical result as progress.

## The gestalt

Let `F0,F1` be the two native `874x1164` uint8 camera frames in a pair, and let
the frozen torch-f32 spatial operator produce

`Y0=R(F0)` and `Y1=R(F1)` at `384x512`.

SegNet consumes RGB `Y1`. PoseNet consumes the coupled YUV6 transform of both
`Y0` and `Y1` and then a nonlinear feature network. The codec therefore seeks
the least-description pair of native frames whose two evaluator outputs lie in
the required Seg argmax cells and Pose tube. It need not reproduce the source
pixels outside those obligations.

The proposed architecture has two co-designed halves:

1. **V9 is the predictive scientific syntax.** It describes five-class level
   sets, Morse-Smale cells, topology events, per-class heterogeneous carriers,
   worldsheet/xi transport, merge-diff-correct reconciliation, and AA-SDF
   rasterization.
2. **V10 is the evaluator-native realization compiler.** It selects cell and
   margin obligations; constructs two temporally distinct but coupled planes;
   solves bounded uint8/native-f32/tie-aware preimages; chooses palette, gauge,
   and nullspace representatives; and solves frame 0 conditionally on the
   realized last frame.
3. **The real missing V10 layer is R10 prosody/feature relay.** Pose is not just
   a low-rank additive patch. The inverse needs amplitude, frequency, phase,
   contrast, channel-energy, and texture coordinates, plus feature-relay or
   multiple-shooting constraints through frozen nonlinear blocks. Batch norm,
   squeeze/excitation, and other nonlocal couplings make this part of the
   representation, not optional polish.
4. **The entropy grammar participates from the first archive.** A description
   that is compact before real coding can be expensive after headers,
   quantization, contexts, and global recompression. Syntax and realization
   must be co-designed with the coder.

This is not “finish V9, then patch with V10.” The encoder alternates:

`V9 obligation proposal -> V10 Y0/Y1 realization -> real coder -> exact decode
and score -> costate debt back to V9/V10 -> atomic accept or rollback`.

The state lifecycle is:

`SourceTruth -> ObligationIR -> RealizedPair(Y0,Y1) -> ArchiveArtifact ->
DecodeReceipt -> ScoreReceipt(axis)`.

Every edge needs exact parent/child hashes. Archive sections and ledger rows are
compiler outputs; they are not the scientific state.

## Why fixed Seg, Pose, and rate targets failed

The feasible set is a coupled sublevel surface, not three independent gates.
For target `T`, the strict conditional byte ceiling is

\[
B_{max}(d_s,d_p;T)=\left\lceil\frac{37{,}545{,}489}{25}
(T-100d_s-\sqrt{10d_p})\right\rceil-1,
\]

when the remaining slack is positive. Pose's local price is state-dependent:
`dS/dd_pose=5/sqrt(10*d_pose)`.

Smooth KKT values, Fisher maps, and costates may rank proposals. Admission is
an exact, serialized, same-object before/after decision. A useful
noncommutative bundle may contain micro-steps that are individually uphill, so
the required invariant is negative exact `delta S` for the atomic bundle, not
for every artificial prefix.

## What was directly under our noses

The repeatedly missing composition was not another thin adapter. It was the
translation between compact task-space geometry and the nonlinear evaluator
feature state, while charging the real code length. V9 descriptions were often
judged through an approximate paint renderer; V10 exact-plane work often proved
feasibility with rate-dead direct storage. Their seam must itself be the codec.

The current V10 structural receiver cannot simply be promoted. It still says
`launch_ready=false`, and several compiler handlers are structural
XOR/digest/cyclic-pixel placeholders rather than the physics of the frozen
evaluator. The reusable parts are strict parsing, ownership, checkpointing,
integer feasibility, and receiver custody.

Likewise, the historical seven-home allocator and WTNV2 packet linker do not
compose: their accepted physical-home vocabularies have an empty intersection
and their base objects differ. They remain useful guard and grammar lessons.
The original codec needs a native compiler edge from its scientific state.

## V9 evidence and priorities

The fresh source-only spine at
`.omx/research/original_taskspace_inverse_witness_codec_20260725/spine_refresh.json`
binds the current dynamic pointer and preserves zero borrowed candidate bytes.
It establishes only a partial constructive spine:

- S0/S1 source and scorer target custody are real.
- A finite S2 seed stores `17,926` topology/cell events in `39,836` bytes with
  parse-back, but it is not a full partition.
- The current coherent Lane chart is `41,303` bytes and approximately `0.57`
  F1 against the true mask.
- The n600 AA-SDF primitive reaches
  `d_seg=0.0008598581949869792`; it is a renderer primitive, not a witness.
- S2 is incomplete, S3 is not composed with it, and S4 is absent.

On the measured historical n256 V9 carrier bridge, error mass is:

| Class | Errors | Share |
|---|---:|---:|
| Road | 957,226 | 47.35% |
| Movable | 802,005 | 39.67% |
| Lane | 134,317 | 6.64% |
| Undrivable | 114,090 | 5.64% |
| MyCar | 14,145 | 0.70% |

Road plus Movable account for about 87% of total current error. Lane and
Movable have the worst conditional rates. The implication is not to train one
larger universal renderer; it is to improve the joint bulk-boundary and island
syntax while pricing their V10 realization debt.

V9 mechanisms retained:

- Road/Undrivable joint bulk boundary;
- Lane ground chart, width, dash phase, and sparse events;
- Movable island birth/death/shape events;
- MyCar static closure;
- worldsheet and xi transport;
- merge-diff-correct and receiver-visible AA-SDF;
- event-native continuation and typed provenance.

Formulation-scoped negatives remain scoped. The old “curvelet/shearlet” label
was attached to global Fourier plane waves; that is a polar directional Fourier
control, not a localized residual basis. Event markers that do not mutate
decoded bytes are not actuation. More epochs cannot repair a wrong
description-to-receiver projection.

## V10 evidence and priorities

Exact inverse work proves very low distortion is possible on the integer
lattice, but direct planes and frames are rate-dead. The useful V10 pieces are:

- scorer cell, margin, and tie geometry;
- bounded uint8 preimage and actual resize-numerator constraints;
- the two-plane `Y0,Y1` factorization;
- conditional frame-0/Pose solving;
- palette/gauge/nullspace reconciliation;
- exact discrete actions and hard-oracle checks;
- terminal quotient ownership.

The n6 two-plane feasibility row establishes a scoped existence result, not
n600 compactness. The failed low-rank additive frame-0 actuator closes only
that formulation. It redirects work toward amplitude-structured xi-advected
photometry and frozen-feature relay.

The 25 exact source-derived IS1 demand rows in
`.omx/research/ddm_is1_rg3_solution_demand_25_20260724.json` should become an
interface acceptance suite for the original codec: 16 SKELETON/interface rows
and 9 FIBER/within-cell rows. Their historical byte prices are null. They must
be rematerialized through the new receiver and real coder; no historical
archive or actuator is inherited.

## Seven logical roles, not seven independent files

The proposed scientific coordinates are:

1. topology/worldsheet;
2. joint Road/Undrivable bulk boundary;
3. Lane chart and phase;
4. Movable islands plus MyCar closure;
5. cell/value and two-plane preimage;
6. xi, frame0, prosody, and Pose feature relay;
7. irreducible quotient.

Each role needs provenance, dependencies, and mutation ownership. Physical
entropy coding may still be joint or monolithic when that gives fewer exact
archive bytes. Logical ownership is not an assumption of byte independence.
For populated streams, a caller-attested `ContentAddress` is not enough:
the C0B loader must hash the exact content and provenance snapshots it parsed
and emit a parent-state-bound construction receipt before the state is trusted.

Palette/gauge reconciliation occurs before the terminal quotient. Otherwise T
pays to relearn a free or lower-description representative.

## Training policy

Default: `DERIVE, SOLVE, OR CODE`.

Training is not categorically reserved for T. It may be the numerical method
used to fit a declared compact typed function when the scorer and real coder
are in the loop and all learned variables, byte costs, seeds, and resume state
are explicit. What is forbidden is using training as an untyped substitute for
the world model or inverse solve.

An *additional residual T* is terminal. It is allowed only for measured debt
unreachable by geometry, topology, lattice, palette/gauge, prosody, and
conditional Pose routes. It must beat matched-byte analytic and dictionary
controls and pass T-off, deletion, no-relearning, quantization, parse-back,
gradient-ownership, deterministic seed, and per-stage-resume tests.

## The first decision-quality artifact

The next artifact is not n24. Small fixtures remain parser, unit, and mutation
tests only. They may not choose a scientific branch or support a score verdict.

The next artifact is one coarse but complete n600 vertical slice:

- all 600 pairs and all five classes internal to one source-derived state;
- explicit Y0 and Y1 obligations;
- contest-native `874x1164` uint8 realization;
- a finite grammar and real coder charged from the start;
- deterministic archive and inflate entrypoint;
- decoded-output hashes and parse-back receipt;
- same-object `d_seg`, `d_pose`, bytes, and coupled score on a clearly labelled
  local axis;
- per-class/per-pair debt and the 25-row acceptance readout.

It may score badly. Completeness is the point: its measured debt decides whether
the next unit of work belongs to V9 geometry, V10 R10 realization, conditional
Pose, or coding. This prevents months of optimizing components against an
unpriced or non-production seam.

The corrected machine-readable DAG is
`.omx/research/original_taskspace_inverse_witness_codec_20260725/roadmap.json`.
After the vertical slice, V9 geometry, V10 prosody/relay, conditional Pose, and
the coder advance in parallel on the same object. Authority may branch from any
complete n600 archive; C5/C6 polish is not a prerequisite. A pointer break is
banked, then the campaign continues until `<0.15`.

## Concrete landings in this pass

### C0 scientific identity

`src/tac/witness_dsl/coupled_witness_state.py` now defines:

- content-addressed frozen source/evaluator identity;
- seed-separated scientific state;
- seven provisional logical roles with exact dependency hashes;
- parent-bound state patches and transition receipts;
- a separately content-addressed compile policy;
- logical-to-physical stream policies that permit joint entropy sections;
- a deliberately false-authority C0 codec-object envelope.

`tools/build_coupled_witness_scaffold.py` hashes the live source, evaluator,
weights, runtime lock, and receiver components, strict-parses one spec snapshot,
roundtrips the canonical envelopes, and writes a receipt last. The landed C0
receipt records:

- frozen-space SHA-256
  `8ba5b6f9d98e2b2a5f9f363abccc811f137699c1bb9230d62a716498de744c57`;
- state SHA-256
  `d2e02be67208d789cdbd9b68dcc0fd0741546dff6dcfc50d9e6e2ceb0f8c57a6`;
- compile-config SHA-256
  `5a3ac8f198b539f5f64a57822f035e26cd4bc6f68103c45904663c604e53d6e3`;
- codec-object SHA-256
  `95ab705f13b8521a2f3bf86c3c71d1785ef58e9aca3c572910ff8432947f7bf2`;
- `archive_emitted=false`, `decoded_output_emitted=false`, and
  `score_measured=false`.

This is metadata identity, not a receiver roundtrip and not a candidate.

### Composition and custody bug classes closed

- The task-space spine audit now reads the competitive target dynamically and
  binds the same pointer bytes it parsed. Every parsed receipt now likewise
  reports the identity of that exact snapshot, and source video, GT cache, and
  evaluator artifacts use one-pass identities rather than reopened files.
- C0 state patches and transition receipts now have canonical envelopes and a
  deterministic replay validator that checks patch, parent, child, index,
  frozen-space, generation, and exact changed-role semantics.
- C0 scaffold and WTNV2 linker outputs are staged as whole immutable bundles
  under exclusive publication locks. Receipt-last publication prevents a
  concurrent producer or crash from mixing candidate bytes and identities.
- J8F adapters no longer pretend individual application archives form the
  cumulative chain: 11/12 SHA mismatches and 1/12 byte mismatch remain named
  blockers.
- The seven-home allocator now consumes the canonical adapter validator,
  refuses mixed objects/axes, and cannot compare asserted transition scores to
  the frontier without an exact-eval custody foreign key.
- The WTNV2 linker explicitly refuses the incompatible seven-home plan, emits
  complete receiver files only on a real link, hashes and parses one manifest
  snapshot, and binds both `inflate.py` and the authoritative `inflate.sh` in
  its receiver identity.
- The WTNV2 reference oracle now consumes stored per-class warp regimes just as
  the generated receiver does; canonical, all-identity, and learned/fallback
  maps have end-to-end parity tests.
- Generated/reference receiver validation now agrees on invalid bank-frequency
  fields.
- Residual serialization refuses parameters the decoder never consumes,
  eliminating counted dead payload and incomplete optional parameter families.
- The allocator and scaffold now hash the exact JSON snapshots they parse.
- Residual trainer argument-guard tests now explicitly mock only heavy-run
  admission, so the production governor stays first while invalid config guards
  remain executable in the local test suite.

The corrected real historical manifests still produce zero applied receipts.
That is the honest state. These mechanisms are guards and lessons for the new
compiler, not a candidate composition path.

Artifact-hygiene disposition: the canonical linker blocker is
`codex_applied_action_packet_link_20260725/attempt/link_attempt.json`, which
names the stronger object/grammar incompatibility. An older ignored derived
receipt at `.../output/link_attempt.json` (SHA-256
`9f39f9de5178415b9bad0dd16fd0a5fc201c0a49da28a4d358d39ce19b7230a8`)
only reported that the allocation had no selection, bound the same input
manifest SHA-256
`bf117807eee9f3875f0050a3af0c8620d8abee684c921cbc8c2d7f427ec33289`,
emitted no candidate bytes, and was removed as superseded derived output.

## Bounded P2 debt carried into C0B

- Whole-bundle publication is race-safe for all cooperating builders through
  an exclusive sibling lock. POSIX `os.rename` still has a last-instruction
  no-replace gap against a noncooperating process that creates an empty target
  directory; use `renameat2(RENAME_NOREPLACE)`, `renamex_np(RENAME_EXCL)`, or an
  equivalent portable primitive when this becomes a production publisher.
- A nonempty `ScientificStream` cannot become trusted merely by deserializing
  caller-supplied addresses. C0B owes an exact-byte content/provenance loader
  and parent-bound construction receipt; `StatePatch.provenance_ref` is only a
  locator until that receipt binds its bytes.
- Historical WTNV2 `reach_kstar` is counted and parsed but decode-inert after
  explicit keyframe indices are read. Do not inherit that alias into the native
  grammar: derive and validate it from the index schedule or remove the field.

These are explicit C0B/native-grammar gates, not C0A score or archive claims.

## Public PR harvest — lessons only

A fresh inventory covered 125 public PRs and the ranked leaderboard families.
The family-level lessons are:

| Family | Retained lesson |
|---|---|
| Classical codec, ROI, resize, denoise | scorer-aligned resolution and region allocation matter; ordinary fidelity plateaus |
| Masks, quantized Pose, sparse actions | evaluator obligations are low-dimensional and frame roles differ |
| Repack, Brotli, range/arithmetic coding | headers, contexts, concatenation, and final recompression change real rate |
| Joint task renderers | Seg and Pose need distinct but coupled carriers |
| INR/HNeRV-style work | compact shared functions, QAT, curriculum, and discrete polish are useful methods |
| Exact-grid/click/greedy work | receiver coordinates and cumulative exact remeasurement beat smooth proxies |
| Current public frontier | an external existence proof and dynamic target only |
| Loophole attempts | decoder legality and counted-data boundaries are part of the codec |
| Cool-Chic/per-instance models | a possible matched-byte terminal competitor |

No public archive, code, weight, checkpoint, latent, click, selector, token, or
sidecar becomes a parent of this candidate.

## The instruction concern

There is no global Codex instruction requiring the “smallest missing slice.”
The repository contract says the opposite: `Frontier Velocity And
Anti-Conservatism`, `Long-Burn Campaign Default`, and `Execution Accountability`
all require broad high-upside campaigns that produce concrete artifacts.

The conservative micro-slice pattern was an execution-strategy failure: local
proof obligations became the strategy instead of protecting the archive
boundary of a larger strategy. Exact custody and no-fake guards remain
necessary, but they belong around the complete codec pipeline. The corrected
roadmap starts with a coarse full n600 object and uses local proofs to keep that
object honest.

## Triality and system intelligence

- **DSL/state:** `CoupledWitnessState`, typed dependencies, state patches, and
  compile policy.
- **DAG:** C0A metadata scaffold -> C0B complete n600 vertical slice -> parallel
  V9/V10/Pose/coder refinement -> factor10 bundle compiler -> conditional T ->
  authority branch -> `<0.15`.
- **Equations:** exact nonlinear score, conditional byte ceiling,
  state-dependent Pose price, V9 level-set action, and V10 two-plane/R10
  realization.
- **Sensitivity:** exact decoder debt becomes a state-bound costate map back to
  chart, topology, prosody, and coder coordinates.
- **Allocator:** exact atomic bundle deltas after global recompression; no
  foreign or additive marginals.
- **Autopilot:** expected negative exact `delta S / wall-clock`, tempered by
  uncertainty, blocker depth, and information gain.
- **Continual learning:** scoped negatives and receiver mutations attach to the
  same scientific-state lineage; foreign rows remain priors only.

## Verification receipt

The final integrated local suite passed `420` tests, including all `24`
residual-compose tests; a last changed-surface replay passed `128` tests. Ruff,
Python compilation, Rust receiver parity, JSON validation, review-policy
checks, and `git diff --check` were clean. The final swarm disposition was
`APPROVE_WITH_P2_DEBT`: no P0/P1 finding remained, and the bounded P2 debt is
recorded above rather than hidden or promoted away.

## STORES CONSULTED

Full `CLAUDE.md` and `AGENTS.md`; current canonical frontier, lane, subagent,
cost-band, continual-learning, task-status, and gradient-anchor surfaces;
latest Codex and Claude findings/design/council memos; v7.5 operating contract;
v8 per-class specification; V9 master action, cohesive package, fake-audit,
carrier-compose, worldsheet, signal-loss, and campaign receipts; V10 reconciled
integer-plane, compiler/receiver, inverse-completeness, production-receiver,
power-diagram, joint-inverse, and quotient work; source-only S0-S4 spine inputs;
DDM J8F/J12/PF3/IS1/MS2/P1/V19/CC3/E5A evidence; V9 and V10 code/tests; all
current public PR metadata and representative ranked mechanism-family bodies
and file inventories; fresh Rust receiver parity; swarm adversarial reviews.

HISTORICAL_PROVENANCE: append-only Codex findings for the 2026-07-25 original
task-space/inverse witness codec synthesis. No prior result ledger was
rewritten.
