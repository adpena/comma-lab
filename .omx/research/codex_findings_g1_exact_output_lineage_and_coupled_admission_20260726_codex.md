# Codex findings — G1 exact-output lineage and coupled admission

UTC: 2026-07-26

Lanes: `lane_g1_bounded_taskspace_correction_20260726`,
`lane_g1_pair_population_envelope_20260726`,
`lane_g1_teacher_atom_census_20260726`, and
`lane_g1_prior_signal_harvest_20260726`.

Scope: local research build, exact teacher measurement, and adversarial review.
No remote/GPU dispatch, official evaluation, candidate archive, score claim,
promotion, pointer movement, or originality proof.

## TIER-0 correction to the prior policy

The prior C0B memo correctly forbade exact encoder-only PBR1/PBR2 packets,
target labels, obligation IR, hard-oracle observations, and explicit teacher
preimages from candidate bytes. It overgeneralized that rule to all dense Y and
camera-preimage representations, even when a representation is counted,
own-lineage, parser-closed, and rate-priced. It then made a second
over-conservative inference: it said a lawful `G` must remain non-exhaustive and
stop before exact target reconstruction.

That inference is false and is superseded by this memo.

Exact decoder output is not the same thing as target-derived payload lineage.
A tiny finite generator that happens to reconstruct every target semantic cell
is the desired legal exploit. A one-cell counted hard-pixel correction can also
be legal. A typed counted dense/preimage representation is not categorically
forbidden; it must survive byte pricing, lineage accounting, and production
parse-back. What is forbidden is carrying an exact encoder-only target table,
PBR stream, oracle transcript, explicit teacher preimage, or equivalent hidden
data in candidate bytes or decoder source. Candidate legality is decided by
payload lineage, receiver closure, and exact byte custody; usefulness is
decided only by the complete coupled score.

This distinction removes a real signal-suppression bug. The first G1 compiler
had rejected exact target output, one-atom-per-cell output, and any proposal
whose Seg debt did not improve independently. Those gates would discard a
compact exact generator and any Seg-harming move whose Pose/rate gain wins the
actual contest objective.

## One conditional objective, not three independent gates

The only admission equation is

`delta_S = 100*delta_d_seg + delta(sqrt(10*d_pose)) + 25*delta_bytes/37545489`.

A same-object candidate is admitted iff `delta_S < 0` after exact recomposition
from measured `d_seg`, `d_pose`, and archive bytes. Archive-byte delta is signed:
a replacement representation may contain a new G section while shrinking the
whole archive. No positive-added-byte assumption is valid.

The local differential geometry is therefore conditional:

- `partial S / partial d_seg = 100`;
- `partial S / partial d_pose = 5/sqrt(10*d_pose)` for positive `d_pose`;
- `partial S / partial byte = 25/37545489`.

For a live competitive pointer `S_star`, the strict archive ceiling at a given
distortion point is the largest integer B satisfying

`B < (37545489/25) * (S_star - 100*d_seg - sqrt(10*d_pose))`.

This is a moving level-set constraint, not a fixed Seg, Pose, or rate target.
`S_star` must be read from the canonical effective-pointer selection rule
`min(local contest-CPU, local contest-CUDA, upstream official)` at decision
time. The refreshed snapshot during this pass was upstream official `0.172`;
that number remains a transient external pointer, not a hardcoded compiler
constant.

The decision surface itself is now fail-closed. The effective score is
recomputed from the local CPU, local CUDA, and every valid official constituent
row rather than trusted from either a cached winner or an inconsistent
`best_entry`; a pointer with no constituents cannot fall back to its display
cache. The official snapshot has its own freshness gate, and a failed refresh
preserves the last successful official timestamp, so a fresh local wrapper or
failed-attempt timestamp cannot mask stale upstream state. The
C1 n600 distortion coordinate is accepted only as the immutable historical v2
file at its canonical path/SHA and remains macOS-CPU advisory. Finally, the
provisional integer byte ceiling is ratcheted against the fully recomposed score,
eliminating a floating-point case where `ceil(boundary)-1` landed exactly on the
forbidden equality boundary. At the current advisory C1 distortion coordinate
and live `0.172` target, the conditional maximum is 187,563 bytes:
`S(187563)=0.17199948562979062`, while
`S(187564)=0.17200015148874376`. This is prediction geometry, not a candidate
archive or score claim.

## Reverse-causal factorization: the representation bug under our noses

The frozen evaluator makes Y1 the semantic anchor: SegNet consumes only the
last frame, while PoseNet consumes the pair Y0/Y1. The inherited V10 direction
stored or reconstructed Y0 first and predicted Y1 from it. That is the wrong
conditional for this task-space codec. The shorter and more controllable MDL
factorization is:

`P,G -> semantic/reference Y1 -> exact uint8 Y1 hash -> generic Y0 fibre | exact Y1 -> chronological (Y0,Y1)`.

The decoder may materialize Y1 before Y0 while emitting frames in chronological
order. The new A packet makes that order executable and exposes two modes for
exact coupled arbitration rather than selection by taste:

1. `FRAME1_ANCHORED_Y0_FIBRE`: realize semantic Y1, then transport and
   photometrically correct Y0 conditioned on the exact Y1 object;
2. `JOINT_SHARED_SKELETON_TWO_FIBRE`: share the semantic skeleton, realize Y1,
   then apply a distinct frame0 fibre over that skeleton.

Neither mode uses Pose6 as a causal/serialized A input or claims Pose usefulness;
the aggregate predictor object is accepted for custody only. Exact n600
whole-archive score decides whether either Y0 fibre helps PoseNet.

The v1 control domain is the exact coordinatewise behavior quotient forced by
the realized frame ABI: vertical shifts are `[-383,383]`, horizontal shifts are
`[-511,511]`, and each uint8 RGB delta is `[-255,255]`. Larger int16 values are
universal aliases under edge-clamped translation or uint8 clipping and are
rejected, preventing redundant packet states from splitting acquisition and
entropy-coding signal. This is geometry-derived canonicalization, not an
empirical score threshold. The anchored per-pair parameter space is
`767*1023*511^3`; the joint-skeleton space is `767*1023*511^15`. Both behavior
maps can still be noninjective for a particular input and remain deliberately
expressivity-incomplete.

## Concrete landings

### 1. Canonical G packet with an explicit lineage boundary

`src/tac/witness_dsl/generative_taskspace_correction.py` composes the existing
original V9/V19c receiver-closed finite primitives:

- boundary coefficient deltas;
- topology birth/death events;
- compact boundary shearlets;
- island-shape atoms;
- movable worldsheet tracks and knots;
- one shared finite realization palette.

The packet is canonical, CRC protected, pair-population and predictor-state
bound, mutation refusing, and freshly decoded. PBR1/PBR2, target labels,
obligation IR, oracle evidence, dense Y, and explicit preimages have no packet
field. Encoder-only target labels are used only to measure debt in the compile
receipt and are rehashed from immutable bytes. Typed own-lineage dense Y or
camera-preimage payloads remain a lawful A/T fallback if they are fully counted;
their absence from G is not a universal ban.

Exact target reconstruction and a one-cell sparse correction are tested to
remain structurally eligible. Atom lifetimes are charged on every pair they can
affect, including worldsheet knot interpolation and the shared palette. Resource
counts describe exact packet contents; there is no arbitrary family, pair, or
changed-cell cap below the wire ABI.

The adversarial consumption audit found a sharper false-authority bug: canonical
G bytes equal to one unique decompressed candidate ZIP member prove presence,
not receiver consumption. The existing synthetic fixture's `inflate.py` was
inert and its raw outputs were produced independently. Exact G admission now
deliberately raises `receiver_consumption_custody_absent` after reopening all
available custody instead of emitting a value-per-byte receipt. An accepting
path requires a runtime-emitted, provenance-bound chain from archive member to
strict G parse/apply to decoded labels/profile to the evaluated raw aggregate,
plus a matched G-only counterfactual with fixed runtime and non-G members whose
decoded state and raw output change. Caller-authored receiver receipts and
literal/wrapped G claims carry no authority.

The canonical exact-eval producer now emits the recomputed full upstream-tree
SHA-256. The tree hash rejects every symlink before applying exclusion rules,
binds executable bytecode in ordinary digests, and makes authority producers
and consumers fail closed on `.pyc`/`.pyo` rather than letting sourceless or
cache bytecode alter evaluator behavior outside custody. The compiler is
reconstructed from the packet and every semantic receipt field must agree.
Tolerance-accepted display drift can never turn a mathematical tie into an
admission.

### 2. PairPopulation and compact-program join

`src/tac/witness_dsl/pair_population_envelope.py` defines one hashed global
source-pair population with explicit V9/PBR/IR/V10 local indexes. It fresh-opens
the coupled state, V9 program, obligation IR, explicit V10 result, PBR2 teacher,
and typed V19c pair config instead of accepting caller-attested hashes.

In its explicitly non-transitive Python reference mode, it binds:

- recovered PBR2 semantics to every scoped IR winner obligation;
- the exact counted G section to the candidate semantic decode and coverage;
- every unmatched sparse obligation to an existing counted frame1-preimage or
  terminal-quotient section hash;
- generated Y0/Y1 identities to the encoder-only explicit V10 admission;
- a reference-receiver declaration that absolute Pose6 belongs to V9 and a
  distinct frame0 section is conditioned on generated frame1; production
  parser/runtime verification of that declaration remains absent;
- receiver state to reopened objects, receiver source, program bytes, and the
  exact section manifest.

Identical program bytes must replay deterministically. Each counted section has
one valid whole-program same-role mutation counterfactual with non-target bytes
fixed; deletion causality is not claimed. A G section cannot claim ownership of
an obligation its own semantic decode left unmatched.

This envelope remains `candidate_payload_eligible=false`. Receiver source/code
binding intentionally does not bind closure cells, defaults, globals, transitive
imports, callable classes, native runtimes, or a standalone archive runtime.
Payload-presence and provenance fields are retained as unproven
producer/reference-receiver declarations, not verified absence or originality.
The landed reverse-causal A grammar still needs an adapter into the Pair counted
sections, production archive parsers, complete runtime closure, and exact replay.

### 3. Exact prior-signal harvest

`g1_prior_signal_harvest_v1.json` strict-reopens the exact PBR2 packet and joins
its complete packet accounting to the materialization receipt. It also joins
the same 5.078 GB frozen target-cache identity across PBR2, the n600 partition
census, V10 lattice evidence, V13, and V14. V19c is explicitly correction-only
because its source receipt does not declare that cache.

The harvest preserves source advisory axes, rejects nested authority laundering
and JSON boolean/float type confusion, binds producer source/git/argv, and is
crash-atomically write-once. It records the corrected composition rule:
compact exact output and sparse counted corrections are lawful; independent
component thresholds are not.

### 4. Bounded n64 teacher timing/acquisition diagnostic — non-promotable

The following n64 row is not a finding, fit-order verdict, branch-decision
surface, candidate row, or score precursor. It is retained only to exercise
exact teacher acquisition, estimate bounded producer/runtime behavior, and
shape an n600 measurement plan. **All decisions are n600-only.**

The real exact PBR2 teacher debt over source pairs `448:512` contains 489,519
mismatch cells. Exclusive staged ownership is:

- same-coordinate temporal repeats: 393,551 cells, 80.3954494105%, 17,845
  packet atoms, 32,844 teacher payload bytes;
- connected-island row spans: 90,417 cells, 18.4705803043%, 7,629 components,
  19,292 spans, 33,254 teacher payload bytes;
- singleton sparse tail: 5,551 cells, 1.1339702851%, 5,551 events, 9,256
  teacher payload bytes.

This is an encoder-side n64 timing/acquisition diagnostic, not candidate bytes,
score value, or an admissible fit-order result. Candidate-admissible coverage
remains zero. Its temporal/geometry/tail ordering is only a prior to remeasure
on n600; it cannot select or reject an acquisition family. It does not
authorize copying teacher atoms. Only a lineage-clean n600 G fit that survives
receiver realization and exact same-object scoring can inform a decision.

### 5. Exact output, payload lineage, and originality are separate predicates

Exact output is desirable but proves neither legal payload lineage nor
originality. A compact counted generator may lawfully reconstruct every target
cell; a lossless PBR/target table remains forbidden even when counted. The
complete candidate must close every shipped byte to P/G/A plus optional T
ownership. T is the counted irreducible terminal quotient and is admitted only
after measured same-object matched-byte P/G/A controls fail to improve the total
score. Encoder-only teacher/evaluator truth is E and never ships.

The executable stack receipt now carries current-stack
`borrowed_substrate_accounting` for the named inputs: HNeRV/PR130 is
research/mechanism-only with archive/source/weights/latents/selectors candidate
bytes all zero; the quarantined 149f donor is signal-only with candidate bytes
zero; V9/V10/C2 are our internal original code/mechanisms with inherited
candidate bytes zero; and PBR/GT/oracle/dense-teacher objects are E with
candidate bytes zero. No complete candidate archive exists, so the inventory
cannot yet account every byte of a shippable object. Therefore this pass makes
**no originality proof or originality claim**. Exact output, named-input
accounting, complete-archive payload lineage, and originality remain separate
predicates.

## Landed representation edge and remaining production chain

The project is no longer missing a vague composition direction. The two-mode
reverse-causal A reference grammar now provides the next representation edge:

1. a frame1 reference realization that consumes the generated semantic field
   and expands to exact uint8 Y1 on the reference scorer grid; camera/resize and
   production parse-back closure remain absent;
2. a frame0 Y0 fibre conditioned on the exact generated frame1 object, with no
   causal or serialized Pose6 semantic input or claim.

Both modes have canonical strict parsers, byte-identical re-emission, exact Y1
conditioning, direct-module-file source binding explicitly scoped as
nontransitive, and packet-bound expected Y0/chronological identities. Lineage
and forbidden-payload classifications are producer declarations with
`candidate_lineage_proven=false` and `originality_proven=false`; a closed field
set cannot prove how numeric controls were derived. Declared-reference removal
variants test parser completeness only, while separate value interventions
prove that active controls, G/Y1, and the joint skeleton actually change output.
This is still L0 structural work. It does not provide a source-custodied
canonical A instance/compile receipt, complete typed control derivation,
standalone runtime custody, Pair adapter, n600 P/G/A join, complete archive
accounting, double-inflate proof, exact CPU/CUDA receipts, or the missing
canonical DAG/index/autopilot consumers. Dense V10 results and the Python
reference receiver remain encoder-side/structural existence proofs until that
chain closes.

## Autonomous execution order

1. Produce one source-custodied canonical A instance, then adapt both landed A
   modes into PairPopulation counted sections and a
   standalone receiver without collapsing their distinction. A receiver-closed
   n600 matched-byte exact-score measurement arbitrates them.
2. Reopen the canonical teacher census and prior-signal harvest on all 600
   pairs, then fit G using only source-closed finite V9/V19c primitives.
   Encoder-side teacher truth may guide acquisition, but only compact primitive
   parameters enter counted bytes.
3. Compose one real n600 PairPopulation through P, G, obligation coverage,
   typed A, optional measured terminal T, distinct Y0/Y1, camera realization,
   and a standalone archive.
4. Use n64 only when a bounded producer timing/acquisition diagnostic is useful;
   it is optional, never a prerequisite, and may not choose a grammar, fit
   order, branch, or candidate.
5. Emit one exact component balance sheet for every candidate: raw delta
   d_seg, nonlinear Pose-score delta, signed byte/rate delta, and total delta S.
   Admit only the total sign.
6. Run local frozen-scorer evidence as explicitly advisory, then claim the
   dispatch lane and replay the exact same archive separately on contest CPU
   and CUDA. No pointer movement precedes those receipts.
7. Preserve per-stage resumable checkpoints and storage preflight throughout
   the n600 build. Every window shares the same global PairPopulation and
   content-addressed section identities.
8. Train only the typed terminal remainder that remains after matched-byte
   analytic, dictionary, sparse, and entropy controls. Learned payload must
   beat those controls on exact same-object delta S.

## Triality and no-orphan wire-in

DSL: finite G packet; PairPopulation; complete-or-sparse-owned IR coverage;
typed counted-section manifest; explicit absence of a Pose claim in A; exact
coupled score observation.

DAG: counted P -> lineage-clean G -> typed A -> optional measured terminal T ->
distinct uint8 Y0/Y1 -> native raw -> exact evaluator -> signed component
balance sheet. Encoder-only E guides acquisition but is not a decoder owner.

Equations: candidate legality is payload lineage; realization is exact integer
preimage/parse-back; admission is only the coupled level-set inequality above.

Sensitivity and bit allocation must consume per-section exact component deltas.
Pareto constraints must use the complete score transition, not isolated
component ceilings. Cathedral/autopilot may dispatch only after the A adapter
and standalone archive blockers close. These consumers are not wired yet: this
pass only reopens their sources, marker-scans them, and records explicit
integration blockers. Every future empirical anchor must update the prior
harvest or a typed result ledger; the two A modes remain separate until their
matched-byte probe decides.

HISTORICAL_PROVENANCE: append-only correction and execution anchor. It
supersedes only the prior requirement that a lawful G must remain
non-exhaustive; all prior teacher-lineage and no-score boundaries remain in
force.
