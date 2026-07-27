# G23 implementation spec — G17 capstone vertical bundle V1

Date: 2026-07-26

Authority: local `BUILD + local-verify ONLY`, `research_only=true`. This spec is
subordinate to `CLAUDE.md`/`AGENTS.md`, `PROGRAM.md`, the operating manual, and
the frozen G17 production-envelope spec whose SHA-256 is
`f315c8c0ad3708394e96cbbf40de9bb6af7d6072989bb28ea38a226f5354953b`.

No scorer dispatch, exact evaluation, candidate claim, score claim, promotion,
pointer update, commit, or push is authorized. Synthetic tests are structural
mechanism checks only. A real-n2 command must not be advertised unless actual
ep725 P/runtime/R/scorer custody is executable through this bundle.

## Ownership and non-touch boundary

G23 owns only new files matching:

- `src/tac/witness_dsl/taskspace_g17_*.py`;
- focused `src/tac/witness_dsl/tests/test_taskspace_g17_*.py`;
- `tools/run_taskspace_g17_composition.py` and its focused test;
- G23 research/checkpoint/lane records.

Do not edit frozen G7/G8/G10/G12/G13/G14/G15/G16/G18 modules or tests, the
current G14 run, G19/G20/G21/G22-owned files, shared package exports, unrelated
dirty paths, or current candidate/pointer artifacts. Reuse frozen public
builders, parsers, geometry, archive codecs, and G7 callback types through
adapters.

## Required implementation surfaces

The coherent vertical comprises these G17-owned modules:

1. `taskspace_g17_forward_observation.py`
   - Exact in-memory target and candidate forward-observation objects.
   - Object identities are derived from retained arrays/bytes, never accepted
     as caller strings.
   - Canonical dense-free strict receipts use duplicate-key refusal, exact
     field sets, finite arithmetic, parse/re-emit identity, source-pair order,
     exact-R projected RGB/numerator/denominator identities, Seg labels, Pose6,
     scorer/runtime custody, and double-forward equality.
   - Candidate observations additionally bind archive/member/receiver/decoded
     identities and derive per-pair plus aggregate `d_seg`/`d_pose` from the
     retained target/candidate values.
   - Target/scorer arrays and receipts are encoder evidence and cannot enter
     candidate packets/runtime.

2. `taskspace_g17_g_descriptor_custody.py`
   - Strict canonical `G17GDescriptorAcquisitionCustodyV1` with exactly one row
     per nonempty G descriptor and exact empty-PASS index coverage.
   - Closed acquisition classes are `PASS_G8_V1`,
     `SELECTIVE_G15_ROW3_V1`, and `EXACT_TARGET_DIAGNOSTIC_V1`.
   - Referenced receipt bytes are retained and reopened through frozen public
     strict parsers; descriptor bytes/window/family/mode/outer-inner hashes are
     recomputed.
   - Diagnostic relabel, stale/wrong-window evidence, inner/outer swap,
     missing/duplicate/omitted row, or stored eligibility boolean drift fails.
   - Custody is encoder-only and never serialized in P/G/A/T.

3. `taskspace_g17_generalized_xip2_a.py`
   - Exact `TACX2A4` header/footer (`>8sBBBBBHHHHHHf32s32sII`, 101 bytes;
     `>I`, 4 bytes), strict EOF/CRC/enum/dimension/fp32/source closure, exact
     PASS/global-copy 105-byte encodings, and active nested `SE3XiTransportV2`
     parsing through the frozen public transport type.
   - A typed `G17ConditionalY1SurfaceV1` retains predictor P0 and actual final
     conditional Y1 arrays. Its source binding is derived from reopened
     P/G/post-topology/post-G8 objects and exact receiver receipts; no arbitrary
     digest join and no scorer field.
   - PASS returns P0; global copy sets `Y0 := final Y1`; XIP2 uses the same
     public geometry/transport primitives and NumPy reference semantics as
     G13. Y1 must remain byte-identical in every mode.
   - Encoder-only guidance custody may bind scorer/forward evidence but never
     enters runtime except for its nonzero digest in an active counted packet.

4. `taskspace_g17_compiler_placement.py`
   - Keep these axes distinct and explicit:
     semantic stream role `{SKELETON, CONNECTION, FIBER, GAUGE, RESIDUAL}`;
     evaluator-recursion stage `{L1_program,L2_chart,L3_raster,
     L4_scorer_feature,L5_verdict}`; physical byte home
     `{GENERIC_DECODER_FREE,COUNTED_VIDEO_STATISTIC,ENCODER_ONLY_EVIDENCE,
     COUNTED_PACKAGED_EXECUTABLE}`.
   - Explicit non-aliasing logical types exist for semantic topology,
     realization/gauge, chronological Pose preimage, population sharing,
     entropy/context, analytic residual ownership, learned residual ownership,
     encoder-only teacher/oracle evidence, forward observation, and terminal
     envelope. A generic blob cannot stand for two roles.
   - Every identity is derived from retained exact bytes or a reopened typed
     object. No caller-attested content digest can replace bytes.
   - Freeze the lifecycle as distinct exact types/edges:
     `SourceTruth -> ObligationIR -> RealizedPair(Y0,Y1) -> ArchiveArtifact ->
     DecodeReceipt -> ScoreReceipt(axis)`. A later type cannot be constructed
     from a free-form earlier hash.
   - Bind one hashed `PairPopulation` mapping canonical V9/source coordinates
     to IR/V10 local coordinates; bind complete-or-sparse-owned obligation
     coverage; enforce exclusive V9 Pose6 versus frame-zero residual ownership;
     retain exact encoder-only explicit-preimage bytes where evidence requires
     them.
   - R10 has a separate typed prosody/feature-relay object covering amplitude,
     frequency, phase, contrast, channel energy, texture, multiple-shooting,
     and frozen-feature constraints. Unless an exact receiver instruction
     consumes every declared coordinate/constraint, executable closure is
     blocked with a typed `G17_R10_PROSODY_FEATURE_RELAY_IMPLEMENTATION_OWED`.
   - A compiler-placement manifest covers every instruction, field, payload,
     table, seed, selector, branch, and asset exactly once. It refuses mixed or
     unclassified values, generic target-selected constants, teacher/oracle/
     scorer/public payload, decoder-dead counted bytes, and packaged code
     falsely claimed free.
   - Generic decoder instruction semantics are explicit and separate from
     counted operands. Land a small deterministic stack/section VM whose
     interpreter/operations are generic receiver code while its video-specific
     bytecode, parameters, selectors, constraints, contexts, and exceptions are
     counted operands. It must actually reconstruct bytes for its supported
     opcodes and remain representation-neutral/extensible; unsupported
     topology/constraint/R10 operations fail closed rather than masquerading as
     implemented.

5. `taskspace_g17_production_envelope.py`
   - Exact `TACG17G1` and `TACG17A1` structs:
     header `>8sBBHHHII32s32s` (88 bytes), descriptor
     `>HHBBII32sI` (50 bytes), footer `>I`; strict CRC/hash/EOF/re-encode.
   - Exact magic/version, G/A family/mode matrices, A layout distinction, A
     parent digest, empty PASS canonicalization, and no P bytes inside G/A.
   - Canonical unique width-four shard layouts: n2=1, n24=6, n600=150; reject
     gaps, overlaps, duplicates, alternate partitions, noncontiguous payload
     offsets, zero-entry sections, and source-window drift.
   - Exact order leaves/roots, descriptor-window roots, and pair-order root.
   - Exact `TACG17E1` `>8sBBBBBBBHH32s32s32s32sI` 151-byte packet, derived
     semantic/G8/A summaries, population binding, zero flags, CRC with its
     field zeroed, exact EOF, and parse/re-encode identity.
   - Strict four-section `TACPGA3` build/parse/receive through frozen outer
     builders/parsers. P occurs exactly once and is decoded exactly once before
     bounded immutable shard views; all pair IDs emit once in order.
   - Cycle-free canonical post-topology and optional post-G8 receipts are
     derived from exact P/G bytes and retained received arrays, never accepted
     by hash. Stale H/Y1, pre-topology G8, pre-G8 A, scorer/target fields, or
     receipt cycles refuse.
   - Typed production receiver runtime custody binds the exact implementation
     and receiver assets. Missing runtime/receiver custody blocks a G7 adapter.
   - Expose exact frozen G7 `TaskspaceReceiverCallback` and
     `TaskspaceMeasurementCallback` adapters. Mutually exclusive complete
     branches are evaluated as singleton whole-object proposals; the G17 API
     must not relabel G7 greedy prefix order as a global optimizer.
   - STORE and DEFLATE are both exact built objects and both receiver-replayed;
     selected encoding follows the frozen outer-codec rule. No raw/packet byte
     estimate is an archive measurement.
   - Truth objects keep `research_only=true`, `candidate_claim=false`,
     `score_claim=false`, `pointer_moved=false`, and preserve blockers for
     population-global grammar, substitutive rate reallocation,
     evaluator-equivalent gauge scheduling, overcomplete-teacher homotopy,
     and V1 candidate-lattice sufficiency.

6. `taskspace_g17_selective_g8_adapter.py`
   - Versioned selective `TACG1C` post-forward base, with exact retained fresh
     H'/target/semantic arrays and typed observation receipt closure.
   - Revalidate every selected repair cell as `Z' == T and H' != T`; reject
     baseline H/mask/program reuse and exact-control relabelling.
   - Use frozen public G8 program/compiler/receiver types. If the frozen public
     API cannot lawfully compile the selected base without copied private
     implementation, return a typed blocker; do not invent a frozen enum.

7. `tools/run_taskspace_g17_composition.py`
   - One production ABI at n2/n24/n600. No toy wire path.
   - Mandatory `--resume-from`; immutable stage names and ordering exactly
     `000_custody`, `010_p`, `020_baseline`, `030_topology`,
     `040_post_topology_forward`, `050_g8_acquisition`,
     `060_post_g8_receive`, `065_population_grammar_proposals`,
     `070_a_acquisition`, `080_whole_object_rows`,
     `085_solution_homotopy_rows`, `090_selection`,
     `100_candidate_closure`, `110_exact_eval`.
   - Write-once same-directory temporary publication, file fsync, no replace,
     directory fsync; unequal existing stage, drift, gap, foreign parent, or
     changed implementation/config blocks resume. Same checkpoint ABI through
     n600. Dense arrays never become checkpoint payload.
   - Before long n24/n600 work, use the SSD waterfall and fail closed if
     unavailable. This local landing launches nothing.
   - Structural mode may build/reopen synthetic protocol objects, but reports
     `real_n2_execution_supported=false` until actual ep725 P decode, runtime,
     R, scorer, target, and callback custody are wired. It emits a typed blocker
     instead of a fake reviewed command.

## Tests and acceptance

Focused tests must use the production ABI and cover at least:

- exact G/A/E round trips and single-byte mutation refusal;
- n2/n24/n600 canonical windows and pair order;
- global-vs-sharded A non-aliasing at n2;
- G/A gap, overlap, offset, CRC, family/mode, parent, and trailing-byte refusal;
- P exactly once in one four-section n600 member;
- strict terminal summary derivation and diagnostic hard exclusion;
- target/candidate observation binding and arbitrary/swapped hash refusal;
- compiler-placement complete coverage, distinct semantic/layer/home axes,
  forbidden payload classes, exact retained evidence bytes, explicit decoder
  instruction/operand placement, actual VM reconstruction, and R10 blocker;
- pair-population/coverage/Pose ownership/preimage-byte lifecycle closure;
- cycle-free post-topology/post-G8 receipts and stale Y1 refusal;
- G7 receiver and measurement callback exact types on structural fixtures;
- immutable resume/no-replace/drift/gap behavior and identical stage ABI for
  2/24/600;
- no candidate/score/evaluation/pointer authority claim.

Run focused pytest, relevant composed frozen-module tests, Ruff check/format,
`py_compile`, and `git diff --check`. Preserve unrelated dirty work.

## Completion truth

This bundle is complete only as a local implementation/structural-verification
artifact. It does not move the exact pointer and is not mission success. If the
real ep725 P decoder, current forward/scorer custody, or standalone runtime
cannot be connected without touching frozen/owned files or performing a heavy
launch, record the exact typed blocker and do not advertise a real-n2 command.
