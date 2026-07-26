# Codex findings — C0B exact residual composition and codec reference identity

UTC: 2026-07-26

Lane: `lane_codex_original_taskspace_inverse_codec_20260725`

Mode: `research_only=true`; original-work local build and verification only;
no paid dispatch, official evaluation, promotion, or pointer movement

Score claim: `false`

Competitive pointer: dynamically validated from
`.omx/state/canonical_frontier_pointer.json#effective_frontier`; current
snapshot `0.172 [official-leaderboard display]`, external target only.

Mission: a custody-complete authoritative exact archive strictly below `0.15`.

## TIER-0 outcome

Verdict:
`REFERENCE_FRAME_BUG_MEASURED_AND_EXTINCTED_AT_THE_ABI / COUPLED_SCORE_ROUTING_HARDENED / V9_TO_V10_VERTICAL_COMPILER_STILL_OWED`.

The comprehensive sweep found two composition failures that had been obscuring
signal and one larger missing executable edge:

1. The exact coupled score geometry already existed in `tac.score_geometry`,
   but active planning and terminal-coder reasoning did not consistently route
   through it. That allowed conditional coordinates to masquerade as arbitrary
   independent Seg, Pose, and rate gates.
2. The exact S2 residual is defined in the coordinate system of the C1 decoded
   baseline, while the proposed tiny W_seg semantic base produces a different
   partition. The two were individually real but not composable. This is the
   semantic analog of applying an inter-frame residual to the wrong reference
   frame.
3. There is still no production consumer that carries one
   `CoupledWitnessState` through a typed V9 evaluator-obligation IR, explicit
   coupled V10 `Y0/Y1` preimages, a real archive, fresh inflate, and a
   state/archive/raw-bound score receipt. The architecture is coherent; this
   vertical compiler edge is the remaining execution crux.

Pointer delta: `0`. No candidate archive or authoritative score was produced.

## Exact n600 reference-frame falsification

The resumable batch-16 C0B-BJ1 pass regenerated the live M2 target from exact
custodied raw bytes with the frozen CPU SegNet, reconstructed the complete C1
baseline by reversing every geometry-bound S2 event, decoded W_seg's semantic
cells through its real receiver, and compared all `600 * 384 * 512` cells.

Receipt:
`.omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_bj1_wseg_c1_baseline_join.json`

- 38 preserved stage receipts; 600 pairs; 117,964,800 sites.
- S2 apply-back to its actual C1 baseline recovers the live target exactly.
- The known live-target/cache discrepancy count of 3 is reproduced.
- W_seg differs from C1 on 59,814,423 sites: `0.5070531463623047`.
- Of 17,926 S2 event sites, only 7,039 have the W_seg class expected by the
  C1-referenced event; 10,887 do not.
- The non-event W_seg-to-C1 residual is 59,803,536 sites.
- By C1 baseline class, mismatch fractions are Road `0.0417991`, Lane
  `0.4189164`, Undrivable `0.9974865`, Movable `0.0515516`, and MyCar
  `0.00127266`.
- The final receipt rehashes all 14 bound inputs after the last stage and
  validates exact stage/aggregate/final reuse.

Scoped conclusion: `WSEG_AS_THE_COMPACT_C1_BASELINE` is falsified. W_seg remains
a separately identified control/predictor and V9 semantic factorization remains
open. The S2 event family is not killed: the packet remains an exact C1 syndrome,
teacher, and checksum. What is forbidden is silently transplanting that
syndrome to a different decoded predictor.

## Bug-class extinction: predictor-bound residual ABI

`src/tac/witness_dsl/predictor_bound_residual.py` now provides a deterministic
counted `PBR1` envelope around the finite S2 event grammar. Every residual binds:

- exact predictor-program bytes and SHA-256;
- predictor renderer contract ID and renderer-source SHA-256;
- exact decoded predictor semantic-stream SHA-256 and geometry;
- nested residual bytes, schema, SHA-256, and event count; and
- exact recovered target semantic-stream SHA-256.

The receiver verifies all identities before event application and verifies the
target stream afterward. Program swap, renderer swap, semantic-stream swap,
corruption, trailing bytes, unbound legacy S2 input, and malformed identity all
refuse. The target digest is a counted integrity checksum, not a stored target
table. This ABI is generic foundation; it does not yet supply the production V9
predictor it must bind.

## Exact coupled score correction

The only admission surface is

`S = 100*d_seg + sqrt(10*d_pose) + 25*B/37_545_489`.

For a finite same-object transition:

`delta_S = 100*delta_d_seg + sqrt(10*d_pose_after) - sqrt(10*d_pose_before) + 25*delta_B/37_545_489`.

`tools/audit_coupled_score_surface.py` now consumes a strictly validated dynamic
pointer, binds the exact manifest snapshot, emits a body hash, and writes once.
Independent component thresholds are explicitly non-admission metadata.

The corrected S2 terminal-coder receipt proves why a scalar byte cap was wrong:

- live C1 baseline mean `d_pose = 0.00010184347386600314`;
- unchanged-Pose strict Seg-only break-even: 22,821 total bytes;
- full Pose elimination strict break-even: 70,749 total bytes;
- the 39,836-byte current packet requires
  `d_pose_after < 0.00004236938742218667` for a strict joint improvement;
- the 34,218-byte Brotli payload lower bound requires
  `d_pose_after < 0.00005916871939646534`.

Therefore both are rate-negative when Pose is unchanged, but neither is
unconditionally dead before receiver-closed joint measurement. This supersedes
the earlier zero-Pose-slice interpretation.

## The measured V9 source grammar

The full frozen-target census is content-addressed at
`.omx/research/original_taskspace_inverse_witness_codec_20260725/v9_target_partition_grammar_census_n600.json`.
It reads the 5,078,017,610-byte cache through a read-only stored-NPY memmap and
requires equal start/end cache custody:

- 117,964,800 target sites;
- 570,049 row runs, mean `950.0817` per pair-end field;
- mean horizontal and vertical boundary counts `566.0817` and `2133.78`;
- 1,466,965 changed sites across 599 successive pair-end transitions,
  fraction `0.0124563770`;
- 8-connected component totals by class: Road 1,088; Lane 14,323;
  Undrivable 631; Movable 2,197; MyCar 600.

This supports a predictive heterogeneous V9 grammar. It does not authorize
shipping the diagnostic target table or a lossless restatement of it.

## Gestalt architecture after falsification

The candidate codec is a predictive semantic source model plus inverse
evaluator synthesis, not a stack of independently optimized artifacts:

`SourceTruth -> V9 Predictor P -> predictor-bound syndrome E -> obligation IR`

`-> V10 coupled preimage(Y0,Y1) -> real coder/archive -> fresh inflate`

`-> exact same-object score receipt(axis)`.

The invariant is `decode(P) + decode(E)`, under the declared semantic operation,
recovers the intended obligations and both `P` and `E` bind each other's exact
identity. The original C1 S2 packet can supervise the V9 factor compiler, but a
new syndrome must be computed against the exact decoded production predictor.

V9 owns compact source structure: Road/Undrivable bulk, Lane chart and phase,
Movable topology events, MyCar closure, worldsheet/xi transport, and an explicit
irreducible quotient. V10 owns cell margins, palette/gauge, factor-two and
native factor-ten uint8 preimages, collateral accounting, and frame-zero Pose
conditional on realized frame one. The entropy coder prices the whole object
from the first build.

The exact missing executable interface is:

`EvaluatorObligationIR(classes, margins, conditional Pose fibre)`

`-> CoupledPreimageState(explicit uint8 Y0,Y1)`

`-> V10 production archive -> fresh double inflate -> ScoreReceipt(axis)`.

`predict_project_receiver` correctly refuses to invent RGB from class IDs;
`v10_production_receiver` can take explicit planes once this compiler produces
them. Structural digest/XOR/cyclic-pixel handlers remain quarantined.

## Autonomous build DAG

1. `C0B-RB1`: instantiate a complete five-class V9 predictor under the landed
   predictor-bound residual ABI. Recompute the syndrome against its exact
   decoded semantic stream. Falsifier: identity-bound apply-back fails.
2. `C0B-SF1`: fit the measured target partition grammar with typed V9 factors
   and expose the exact irreducible remainder. Falsifier: real coded bytes plus
   remainder are not better than the matched control at the same receiver debt.
3. `C0B-IR1`: compile the same obligations to explicit Y1 cells/margins and Y0
   conditional Pose preimages using DM4, exact lattice feasibility, VJP custody,
   and the frozen hard oracle. Falsifier: receiver parse-back violates declared
   obligations; this scopes the realization formulation, not V9.
4. Compile one deterministic original archive sibling through the production
   two-plane receiver, double inflate from fresh roots, and bind state, config,
   archive, and native raw hashes.
5. Measure one complete n600 local joint row with resumable batch-16 stages and
   route it through exact coupled score geometry plus all 25 IS1 interface rows.
6. Add closure-aware atomic replacement of upstream scientific roles and their
   transitive dependents, then alternate V9, V10, conditional Pose, and coder
   bundles on the same object.
7. Only after a complete archive and lane claim, branch the exact bytes to
   separate governed contest CPU and CUDA evaluation. Bank a real pointer break
   and continue to `<0.15`.

## Triality and no-orphan wire-in

DSL:

- `PBR1` makes predictor/residual reference identity part of the counted syntax.
- `CoupledWitnessState` remains the scientific parent; production streams must
  be reopened and hashed, not caller-attested.

DAG:

- `roadmap_v3.json` changes the next node from W_seg-plus-S2 composition to an
  identity-bound V9 predictor, then the obligation-to-preimage compiler.
- `.omx/state/next_experiments.md` carries the same execution order.

Equations:

- residual validity is `H(program)`, `H(renderer)`, `H(decode(program))`, and
  `H(apply(decode(program), residual))` equality, not merely residual CRC.
- proposal admission is exact finite `delta_S`, not three scalar gates.

Six hooks:

1. Sensitivity map: emit per-pair/per-class/event Seg and Pose effects keyed to
   the exact predictor and archive identities.
2. Pareto constraint: route all complete points and transitions through
   `tac.score_geometry` exact coupled audits.
3. Bit allocator: price predictor factors and residual sections by measured
   same-object score value per exact recompressed byte.
4. Cathedral/autopilot: schedule only typed DAG nodes with explicit blockers;
   reject unbound residuals and conceptual-only IR edges.
5. Continual learning: append BJ1 as a premise falsification and use the first
   receiver-closed row to update the next-action posterior.
6. Probe disambiguator: keep defensible coder/realizer modes callable and let
   receiver-closed joint measurements arbitrate.

## Apparatus hardening in the same pass

- strict dynamic frontier-pointer semantic validation, classified literal
  inventory, immutable/body-hashed audit output, and stale public-cache refusal;
- scorer source/weight/runtime custody compatible with decorated Torch
  callables and a real pinned scorer smoke;
- raw-debt exact target/contest archive pinning, legacy-fabrication refusal,
  preserved-stage/final-state reuse validation, and end-of-run input barriers;
- single-snapshot C1/S2 bridge reads and path-swap regression coverage;
- immutable joint S2 terminal-coder economics; and
- full V9 cache start/end custody plus exact spatial/temporal closure checks.

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md` in full;
- `.omx/state/canonical_frontier_pointer.json`;
- `.omx/state/lane_registry.json` and
  `.omx/state/subagent_progress.jsonl`;
- `reports/latest.md`;
- current master-gradient, cost-band, continual-learning, probe-outcome,
  council, design, Codex findings, and Codex session-summary surfaces;
- V9/V10, S2, C0B, inverse-solver, score-geometry, receiver, archive, and
  exact-debt source/receipt surfaces named above.

## Honest terminal state

- Pointer delta: `0`.
- Complete receiver-closed n600 candidate: absent.
- Official score claim: absent.
- Remote/GPU/authority dispatch: not attempted.
- Config task: already closed and untouched.
- Original-only candidate lineage: preserved.
- Next exact blocker: instantiate the first full-five-class V9 predictor under
  the predictor-bound residual ABI, then implement the obligation-to-coupled-
  preimage vertical compiler.

HISTORICAL_PROVENANCE: append-only Codex adversarial findings and execution
anchor for the 2026-07-26 C0B exact residual composition pass.
