# G3 ep725 predictor and transport seam specification

## Verdict and objective

The next missing composition edge is a truthful decoder-owned predictor state,
not another optimizer or archive sidecar.  Generalize the existing V9-only
`PredictorSemanticStateV1` boundary so the original ep725 level-set receiver can
provide counted `P` bytes and decoder-derived semantics without inventing
`Pose6`.  Then make the already-counted `G` and `A` programs consume that state
under one reverse-causal contract.

This lane is `research_only=true`.  It may prove bounded receiver causality and
source custody.  It must not claim a candidate, score, frontier movement,
promotion eligibility, complete originality lineage, or standalone n600
closure.

## Do not add a competing container

The G2 `G/A[/T]` fragment grammar is the only new outer section-directory seed.
The ep725 adapter must produce a typed counted `P` section for the next version
of that same grammar.  It must not introduce another ZIP member, nested ZIP,
JSON sidecar, Pair-only serialization, or unrelated packet hierarchy.

The future monolithic member order is exactly:

`PREDICTOR, GENERATIVE_CORRECTION, COUPLED_PREIMAGE[, TERMINAL_QUOTIENT]`.

Each inner section retains its own strict typed parser.  The outer directory
owns ordering, byte ranges, lengths, and hashes.  `PairPopulationEnvelope`
remains the scientific foreign-key/provenance view; it is not a second archive
encoding.

## Frozen source object

- Original ep725 source archive:
  `/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/archive.zip`.
- Archive bytes: 83,838; SHA-256
  `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`.
- Canonical sole member: `0.bin`, 84,536 bytes; SHA-256
  `f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c`.
- Shipped generic decoder: `inflate.py`, SHA-256
  `4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224`.
- Manifest: 1,906 bytes; SHA-256
  `1b8ff55fd7c21395fe7d596558406e2608421522bb0eb3eb11d1e3b0cf047088`;
  `n_pairs=600`, scorer grid `384x512`, camera grid `874x1164`, five
  classes, `has_pose_sidecar=false`.

These bytes are a frozen original-project source substrate.  Reopen and verify
them; do not copy the checkpoint, decoded dense frames, or labels into a new
uncounted dependency.

## Typed predictor state

Add a new state contract while preserving V1 byte/read compatibility:

`TaskspacePredictorStateV2`

- exact predictor-program, renderer, source archive, and source-runtime hashes;
- exact ordered contiguous source pair IDs;
- decoder-derived `uint8 [pair,384,512]` five-class semantic labels;
- a discriminated temporal transport coordinate;
- no target labels, target PoseNet outputs, PBR rows, scorer weights, oracle
  observations, or dense source frames.

The transport discriminant is closed and typed:

1. `NONE`: valid only for label-local/static `G` primitives;
2. `V9_POSE6`: legacy adapter over exact counted V9 Pose6 state;
3. `SE3_XI`: exact counted XIP2/xi trajectory with source-pair and decoder
   binding, explicitly a transport statistic rather than a PoseNet solution.

The binding hash includes the transport kind and its exact counted-content hash
or a canonical absent marker.  Zero-filled Pose6 is forbidden as an absence
encoding.  An INR latent/code row must never be relabeled as Pose6 or SE(3) xi.

## P decode and custody contract

1. Open archive, member, and runtime as stable regular non-symlink files using
   descriptor-bound identities; reject mutation during reopen.
2. Require the exact canonical one-member ZIP and exact LVLS1 stream
   consumption.  Trailing bytes, alternate member metadata, nested ZIP, and
   unknown optional blocks fail closed.
3. Treat the exact LVLS1 member, not the source ZIP wrapper, as counted `P`
   section bytes in the future composed packet.  Preserve the source ZIP hash
   separately as custody evidence.  Measure the newly composed outer ZIP bytes;
   never add the old 83,838-byte rate as if compression were additive.
4. Derive RGB frames and internal semantic labels from the same counted P
   bytes and the bound generic receiver.  Labels come from the decoder's own
   final frame-one `phi.argmax`; they are not SegNet outputs and are not a
   target table.
5. On the bounded gate, compare the in-repository NumPy reference and the exact
   shipped receiver over the same source pairs.  Raw bytes must match exactly;
   semantic identity must be deterministic over two fresh decodes.
6. Any materialization tool writes pair checkpoints atomically, binds the
   completed raw prefix hash, resumes without state regression, runs storage
   preflight, and preserves stage outputs.  A transient `/tmp` path is not an
   evidence path.

## G transport admission

The existing G wire packet remains usable because its predictor foreign key is
a 32-byte binding hash.  Generalize its live-state API without weakening strict
parse-back.

- Boundary coefficients and boundary shearlets are admitted with transport
  `NONE`; they depend only on source-pair IDs and P labels.
- Topology events, island lifetimes, worldsheets, and knots are refused when
  transport is `NONE`.
- Legacy V9 behavior remains byte-identical through a `V9_POSE6` adapter.
- `SE3_XI` remains refused until an explicit event/island/worldsheet mask parity
  adapter exists.  `xi_pose_coder` may provide quantize/serialize/decode and
  homography mechanics; its output is never evidence that PoseNet debt is
  solved.
- The G receipt records which primitive families required transport and which
  transport contract was actually consumed.

## A admission

`CoupledPreimageProgramV1` already binds only predictor program, renderer,
ordered IDs, and predictor labels; its materializer does not consume Pose6.
Generalize its live-state type check to the common predictor-state protocol
without changing its packet schema or causal meaning.

The decoder must still execute:

`decoded G labels -> exact realized uint8 Y1 -> verify Y1 hash -> counted A -> Y0 | Y1`.

An A-only counterfactual changes `Y0` while preserving `Y1`, P, G, optional T,
runtime, pair order, and every non-A byte.

## Acceptance gates

### Structural unit gate

- V1 legacy vectors remain byte-identical and all existing G/A/Pair tests pass.
- V2 `NONE` state accepts a boundary-local G program and refuses every
  transport-dependent family before applying any output mutation.
- V2 source binding rejects labels, pair-order, program, renderer, runtime,
  archive, member, and transport mutations.
- A compiles and decodes against both V1 and V2 state when their visible
  semantic foreign keys are identical; no Pose6 value is serialized or read.
- Outer role mapping proves one-to-one correspondence between the fragment
  directory and the Pair scientific manifest; duplicate ownership is refused.

### Bounded real-source gate

- Use exact ep725 bytes, preregistered pair prefix n2 only for the first proof.
- Double decode is raw-byte-identical and label-identical.
- A counted local/static G mutation changes Y1; deletion/corruption fail closed.
- A counted A mutation changes Y0 with exact Y1 fixed.
- Receipt records exact P/G/A section bytes, outer packet/archive bytes,
  runtime/source hashes, pair IDs, raw hashes, and all remaining false closure
  fields.
- No scorer invocation or score estimate is required for this gate.  If local
  frozen scoring is subsequently run, admit transitions only through
  `tac.score_geometry.score_transition_audit` against a freshly recomputed
  canonical pointer.

## Triality and no-orphan wire-in

- DSL: `TaskspacePredictorStateV2`, discriminated transport, and one outer
  `P/G/A[/T]` section directory.
- DAG: counted LVLS1 P -> decoder-owned labels/frames -> transport-qualified G
  -> exact Y1 -> pose-independent A -> chronological raw.
- Equations: G family dependency predicates, exact section conservation,
  deterministic realization, and the coupled finite score transition law.

Six hooks:

1. sensitivity rows key P/G/A coordinates to exact pair/class/section hashes;
2. Pareto admission consumes complete-object `score_transition_audit` only;
3. bit allocation prices the recompressed monolithic archive, not a sum of
   independently compressed section prices;
4. autopilot dispatches the first missing typed edge in this DAG;
5. every real bounded result updates the continual-learning/probe ledger;
6. `NONE`, `V9_POSE6`, and `SE3_XI` are explicit modes, with parity probes
   arbitrating transport-capable families.

## Explicit blockers after this landing

- real n600 G and A programs remain to be inverse-solved from encoder-only
  obligations;
- complete standalone composed runtime and data-in-code audit remain owed;
- n24/n600 exact score evidence remains absent;
- full original payload derivation lineage remains to be certified;
- exact contest CPU/CUDA replay and pointer movement remain forbidden until
  all preceding custody gates close.

## Stores consulted

- `CLAUDE.md` and `AGENTS.md`;
- `SPEC_g2_receiver_pair_composition_20260726.md`;
- `SPEC_g3_dynamic_levelset_inverse_candidate_20260726.md`;
- `codex_findings_c0b_counted_codec_gestalt_20260726_codex.md`;
- ep725 archive/member/manifest/runtime bytes and the canonical byte-close
  receiver source;
- current G2 receiver, A2 Pair adapter, generative-correction, coupled-preimage,
  PairPopulation, XIP2/xi, and exact score-geometry implementations;
- canonical frontier pointer, lane registry, and subagent progress surfaces.

HISTORICAL_PROVENANCE: append-only executable specification for the first
truthful ep725 P-to-G/A composition seam.
