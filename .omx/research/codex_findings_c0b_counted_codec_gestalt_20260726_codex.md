# Codex findings — C0B counted-codec gestalt and vertical composition

UTC: 2026-07-26

Lane: `lane_codex_original_taskspace_inverse_codec_20260725`

Scope: original-work-only local build and adversarial review. No remote or GPU
dispatch, official evaluation, promotion, score claim, or pointer movement.

## TIER-0 verdict

The comprehensive pass found a concrete missing composition boundary, not a
missing optimizer. The project already has most of the scientific organs:
V9 task-space factors, an identity-bound residual, coupled state, exact uint8
lattice realization, a V10 two-plane receiver, frozen hard-oracle machinery,
exact coupled score geometry, and real archive grammars. What was still being
carried incorrectly was the distinction between three information classes:

1. encoder-only truth and evidence (`SourceTruth`, target labels, margins,
   Pose targets, hard-oracle observations);
2. counted decoder sufficient statistics (predictive geometry, a compact
   generative correction program, and a compact inverse-realization program);
3. generic deterministic receiver code plus non-payload proof receipts.

The capstone is therefore not “V9 plus an exact target residual plus dense
explicit Y.” It is one non-duplicative sufficient statistic whose decoder
expands

`P(V9) + G(generative task-space correction) + A(preimage controls)
-> distinct uint8 Y0,Y1 -> native raw`.

`EvaluatorObligationIR` and hard-oracle evidence admit `A` at encode time but
do not ship. Exact PBR1/PBR2 target reconstruction and explicit dense `Y0/Y1`
are teacher/control representations. PBR event/span bytes are a compressed
GT-argmax table, so counting them does not make them legal candidate payload.
`G` must instead be a compact generative approximation in the original
task-space grammar and is allowed to leave score-priced debt. This
information-lineage rule is the bridge from the micro implementation to the
macro codec gestalt and the most important finding of this pass.

## What landed in this pass

### 1. Scientific-state replacement is now closure-aware

`CoupledWitnessState` can atomically replace one scientific stream together
with every transitive dependent against the complete prospective state view.
It refuses an incomplete replacement closure whose downstream dependency still
names stale content. This is the state transition needed for repeated V9/V10
alternation on one object; unrelated child states are no longer the only safe
update mechanism.

### 2. Exact factorized V9 predictor identity exists

`factorized_v9_predictor.py` receives exact V9 bytes only, fresh-opens the
decoder, paints all five canonical classes, binds the ordered source-pair
window and declared interpreter source set, hashes decoded semantics and Pose6,
and can build a PBR1 against that exact predictor. V10 descendants, mutable
cached-receiver substitutions, malformed programs, pair-window drift, and
large-read truncation are refused.

A historical original V9 n64 program was accepted under this ABI at 51,668
bytes, pair window `448:512`, with exact program SHA-256
`56b563f2f9fb442508134bfb144eb1dc67a07675c93e0c56ec3a569f649bac9a`.
That is predictor-receive evidence, not a good-score or production-n600 claim.

### 3. Evaluator obligations now reach the real V10 receiver

`evaluator_obligation_ir.py` defines typed five-class cell/margin ownership and
conditional frame-zero Pose-fibre obligations. Given explicit candidate
scorer planes, it uses the real factor-two solver, requires a caller-supplied
hard-oracle decision, independently re-verifies lattice proofs and camera-frame
hashes, and parse-backs distinct `Y0/Y1` through the existing V10 production
receiver. Its result binds frozen evaluator artifacts, coupled state,
predictor identities, solver/receiver sources, pair receipts, logits, Pose6,
and the receiver packet.

The API intentionally does not synthesize the explicit planes. Tests can use
synthetic oracle evidence for structural checking; that is not frozen-scorer
authentication or score authority. This preserves the exact remaining debt
instead of laundering a callback into proof.

### 4. The first counted receiver ABI is causal and crash-resumable

`c0b_counted_receiver_codec.py` counts and consumes exact V9 program bytes,
PBR1 bytes, source-pair order, pair/plane/class palettes, and sparse RGB
overrides. It performs exact factor-two parse-back and pair-streamed raw output.

The initial red-team reproduced a builder bug: a valid n1 predictor plus a
valid incompatible PBR could be zipped successfully and fail only during
decode. The public builder now performs the complete fresh cross-section
receive before returning bytes. A pair-ID swap and predictor/PBR mismatch are
refused during build.

The initial writer also had two durability bugs. A crash after raw `fsync` but
before state replacement left a complete pair that continuation refused, and
prefix replay rewrote `completed_pairs` downward before finalization. Durable
state is now authoritative: uncheckpointed suffix bytes are truncated and
replayed, checkpointed prefixes are hash-verified, existing prefixes never
regress state, state files and parent directories are fsynced, and an
exclusive advisory output lock prevents concurrent writers. Injected failure
after raw `fsync` resumes to the identical raw hash.

PBR accounting is now honest. The header and receipt expose exact
target-derived section bytes, nested residual bytes, event count, and exact
event-density numerator/denominator. “No separate dense target-table section”
is distinct from “no target-derived bytes.” Current target-derived residuals
are never promotion-admitted.

The ABI labels itself `codec_role=abi_causality_ablation`,
`capstone_eligible=false`, and
`shared_semantic_partition_across_planes=true`. It also records all four
missing capstone bindings as false: obligation IR, coupled state, hard-oracle
admission, and independent frame-zero preimage. This makes future accidental
promotion structurally harder.

## Exact n1 structural receipt

The post-hardening deterministic fixture is recorded at
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_counted_receiver_n1_structural_receipt_v2.json`.

- archive: 11,328 bytes,
  SHA-256 `e1b1281bafc07c93e7d71fd67c72f5d41332fdf02a659cca4ac33dc764c10ae8`;
- packet: 11,220 bytes; V9 7,865, PBR1 1,465, RGB preimage 44, explicit
  header/framing 1,846;
- PBR1: one target-derived event over 196,608 cells; all 1,465 PBR bytes are
  declared target-derived and 516 bytes are the nested residual; exact target
  reconstruction makes the whole ABI candidate-inadmissible;
- output: 6,104,016 raw bytes,
  SHA-256 `77b7c78bcad0f0cc0cdadcbe0db755186fca18be0a567eea53816ac02d8c928c`;
- exact factor-two proof: 1,179,648 scorer values and 1,179,648 numerator
  values verified;
- authority: research-only n1 structural evidence, no standalone inflate,
  n600, scorer, score, promotion, or pointer claim.

The extra header bytes are deliberate truth, not a rate win: they expose the
target-derived and capstone-debt fields that the smaller pre-red-team packet
omitted.

## Progressive geometry residual probe

The real V9 n64 window has 489,519 predictor/target mismatches over 12,582,912
cells. Current exact PBR1 is 93,896 bytes, SHA-256
`f598ff93c46a04da87683b19c375bd822e79e59352e02eb80029aad32abd722c`.
Current exact PBR2 is 78,665 bytes, SHA-256
`3372eee1d989012fb3293c7abe08eac233c874bf485e5ea15c5bd26d7306f0a1`:
15,231 bytes (16.22%) below PBR1. Its exact staged error ledger is
`489519 -> 95968 -> 5551 -> 0`; its target-derived section payload is 75,354
bytes split into a 32,844-byte temporal block-context stratum, 33,254-byte
component-island stratum, and 9,256-byte singleton tail.

Two false claims were fixed during independent review. `auto` had minimized
only the temporal subsection and could choose a larger complete packet; it now
prices mode-dependent header bytes and carries the exact 2,634-vs-2,630-byte
counterexample as a regression. `max_strata` had been described as physical
prefix decoding even though length, CRC, and all suffixes were validated first;
the API and accounting now state that this is staged application from a complete
packet, and truncated/suffix-corrupted inputs are tested to refuse.

The 51,668-byte predictor plus PBR2 total is 130,333 conditional representation
bytes. A fresh-process regression now derives predictor semantics solely from
the counted V9 program before applying PBR2, so the arithmetic is receiver-
closed inside the repository. But this is deliberately **not candidate rate**:
the packet losslessly reconstructs all 12,582,912 bytes of frozen GT argmax.
Its header, receiver, materializer, and receipt say research-only,
candidate-inadmissible, exact-target-derived, and no-score. It is an entropy
bound and a teacher for `G`, never a candidate section. Exact packet binaries,
the 5,078,017,610-byte GT-cache hash, implementation hashes, runtime, and argv
are recorded in
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_pbr2_progressive_geometry_n64.json`.

## The exact capstone edge still owed

The final line audit found five hidden joins that the type must close. Current
IR state/predictor digests are accepted strings rather than derived from
reopened objects; IR cell rows need an explicit complete-or-sparse-owned
coverage relation to recovered PBR semantics; V9 source-pair coordinates and
IR/V10 local coordinates need one hashed `PairPopulation`; V9 Pose6 and the
frame-zero Pose stream need exclusive predictor/residual ownership; and the
explicit-preimage result currently retains only the dense V10 packet hash and
size, not the packet bytes. Leaving any of these implicit recreates orphaned
signal at an object boundary.

The next production type is not another archive wrapper. It is a
`CoupledPreimageProgram` (working name) with this contract:

Inputs, encoder-only:

- exact `CoupledWitnessState` and frozen-space identity;
- exact factorized V9 predictor `P`, exact PBR teacher/debt measurements, and
  the candidate-admissible generative correction program `G` being tested;
- `EvaluatorObligationIR` with cell margins, collateral owners, and conditional
  Pose fibres;
- an authenticated frozen hard oracle.

Counted output:

- predictor program `P` and only compact generative task-space correction
  parameters `G`; never a lossless PBR target table or exhaustive event/span
  reconstruction;
- a compact preimage-control program `A` that deterministically expands to
  distinct scorer planes `Y0` and `Y1`;
- exact ordered pair IDs plus required compact palettes/gauge/phase/feature
  relay/override parameters;
- no target table, exact PBR packet, scorer, weights, oracle observations,
  dense duplicate semantic field, or dense terminal `Y` when those values are
  already implied by `P+G+A`.

Admission:

- fresh decode `P`; expand `G`; compare its decoded semantics against exact
  PBR teacher debt at encode time without packaging PBR bytes;
- expand `A` to independent `Y0/Y1`, realize camera frames, and hard-oracle the
  exact realized uint8 bytes against the exact IR;
- bind state -> predictor -> teacher debt -> generative correction -> IR ->
  oracle -> preimage program ->
  archive -> receiver -> raw with exact hashes and pair order at every edge;
- require deletion/no-op/mutation tests for every counted section and refuse
  any section whose valid removal does not change output or an admitted debt;
- price the whole same-object archive with exact finite joint score geometry.

Decoder:

- imports only generic, source-custodied free code;
- consumes every counted section;
- expands `P+G+A` pair-at-a-time to distinct native frames;
- needs no source video, scorer, target labels, oracle evidence, or external
  repository checkout;
- supports per-pair write-once stages, restart, fresh-root double inflate, and
  deterministic final assembly.

## Autonomous execution order

1. Treat the now-sealed PBR2 result as an encoder-only entropy bound and
   structured teacher. Candidate builders must reject both exact PBR formats.
2. Define the compact `CoupledPreimageProgram` interface and foreign-key
   envelope. Keep two defensible preimage modes callable: palette/gauge plus
   sparse overrides, and typed analytic/feature-relay controls. Exact
   receiver-closed bytes arbitrate them.
3. Compile a bounded `G` from PBR2's temporal/island/tail teacher strata back
   into V9 geometry/topology primitives; stop before exhaustive exact-target
   reconstruction and admit atoms by exact coupled score value per byte.
4. Implement independent frame-zero synthesis conditional on realized frame
   one. The current shared-partition counted ABI remains the control arm.
5. Move actual frozen-oracle invocation behind one sealed runner that returns
   the typed evidence expected by `compile_explicit_v10_preimages`; synthetic
   evidence remains test-only.
6. Compose one n64 source-only object through state, `P/G/A`, archive, standalone
   receiver bundle, clean-root double inflate, and local frozen-scorer joint
   debt. This is the first branch-decision object.
7. Scale the winning exact grammar to n600 with resumable stages and the storage
   waterfall. Emit per-class, per-pair, and section value-per-byte rows plus all
   25 IS1 interface obligations.
8. Route every alternative through
   `score_transition_audit`: no independent Seg, Pose, or byte threshold may
   admit or reject it. Train only a compact typed remainder that survives
   matched-byte analytic/dictionary controls.
9. After standalone custody and a dispatch lane claim, evaluate the exact same
   archive separately on contest CPU and CUDA. Bank any result below the live
   competitive pointer and continue until an authoritative exact score is
   strictly below 0.15.

## Coupled score and pointer discipline

The only admission law remains

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489`.

The effective competitive pointer at this pass is the dynamically refreshed
official-leaderboard display `0.172` (PR130), external target only. At PR130's
displayed components, holding distortions fixed would require only 213 bytes
of rate reduction to cross strictly below 0.172, while reaching 0.15 would
require a conditional archive ceiling of 157,799 bytes. These are points on an
equal-score surface, not fixed gates. Any Seg or Pose movement changes the byte
budget immediately through `tac.score_geometry`.

## External mechanism lessons, original payload boundary

MPEG-4 binary-shape coding reinforces mode-separated transparent, opaque, and
boundary blocks plus motion-compensated temporal prediction. AV1 reinforces
palette and intra-block-copy style reuse. The official PR130 result reinforces
joint semantic/pose design and representation-specific entropy coding. These
are mechanism priors and falsifiers only. No public archive, source, learned
payload, weights, latents, token stream, sidecar, or selector enters this
candidate lineage.

## Triality and no-orphan wire-in

DSL: exact predictor/generative-correction/preimage sections and their
cross-object foreign keys; exact PBR packets are teacher-only types.

DAG: V9 predictive state -> exact teacher debt -> bounded generative correction
-> obligation IR -> oracle-admitted compact preimage program -> standalone
archive/receiver -> raw/eval.

Equations: exact semantic apply-back, bounded integer resize feasibility,
hard-oracle winner/Pose acceptance, and exact finite coupled score transition.

Six hooks:

1. sensitivity map rows are keyed to predictor, IR, preimage, archive, pair,
   class, and axis identities;
2. Pareto admission uses exact complete-object `score_transition_audit`;
3. bit allocation prices mutually exclusive representation bundles, prevents
   double payment, and hard-rejects exact-target residual packets;
4. cathedral/autopilot schedules the first missing typed edge, not another
   orphan optimizer;
5. empirical PBR2 updates the canonical probe as `BLOCKED_FOR_CANDIDATE` but
   `USE_AS_TEACHER`, while first legal receiver-closed rows update the promotion
   posterior;
6. ambiguous preimage/coder modes remain callable and are resolved by exact
   receiver-closed probes.

## Honest terminal state

- Pointer delta: 0.
- Complete standalone n600 candidate: absent.
- Authoritative score: absent.
- Counted ABI: locally structural and explicitly non-capstone.
- Exact next blocker: compact generative correction `G` plus a decoder-consumed,
  oracle-admitted independent `Y0/Y1` preimage program, with full
  state/P/G/IR/archive/raw foreign keys and no exact PBR payload.
- Original candidate lineage: preserved.
- Unrelated operator config artifact: untouched and excluded.

HISTORICAL_PROVENANCE: append-only Codex adversarial findings and execution
anchor for the C0B counted-codec composition pass.
