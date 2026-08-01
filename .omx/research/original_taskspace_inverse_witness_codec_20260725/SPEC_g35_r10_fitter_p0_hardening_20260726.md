# G35 specification — R10 bounded-inverse fitter P0 hardening

Date: 2026-07-26
Lane: `lane_g35_r10_fitter_p0_hardening_20260726`
Reviewed predecessor: G32 fitter plus G34 adversarial memo
Authority: local structural and bounded real-input verification only; `research_only=true`
Execution: no n600 launch, scorer, public mux, candidate, pointer mutation, commit, or push

## 1. Objective and honest name

Harden the existing G32 implementation into a deterministic, crash-recoverable
**bounded R10 inverse fitter**.  The implementation searches one explicitly
finite source-RGB action family.  It is not a maximum inverse solver, does not
exhaust the receiver action universe, and does not prove that training is the
remaining irreducible option.  Historical G32 path names remain only as stable
filesystem compatibility; public Python types, functions, receipts, solver
labels, and current findings use `bounded_inverse` terminology.

## 2. Frozen file boundary

G35 may edit only:

- `src/tac/witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py`;
- `tools/fit_taskspace_r10_n600_maximum_inverse.py`;
- their two focused test files; and
- this new G35 spec and a new G35 findings memo.

G29/G31 files and every root-owned costate/controller/materialization file are
frozen.  Preserve the root's single linear ffmpeg source stream and its tests.

## 3. Checkpoint cryptographic contract

Every stage and range checkpoint must reopen through all of these independent
checks:

1. exact JSON key set and schema;
2. exact binding SHA-256;
3. exact stage/kind and requested range coordinates;
4. filename suffix equals SHA-256 of the retained file bytes;
5. payload SHA-256 equals canonical payload bytes;
6. predecessor record SHA-256 equals the preceding retained checkpoint;
7. range byte offset/length and retained raw range SHA-256 match; and
8. one checkpoint per coordinate, with no gap or conflicting retained state.

Mutation of a record plus its internal digest while retaining its filename must
refuse.  Stage publication must be append-only and exactly next-in-chain.

## 4. Crash recovery

- A legal crash after `030_geometry` must resume XIP2 from the retained pitch;
  it must neither reject the prefix nor delete/rewrite geometry.
- Selected-base production must call frozen `_setup()` exactly once per process,
  start at the first missing contiguous range, fsync each completed range, and
  publish its range checkpoint before proceeding.
- Long per-pair fit stages expose deterministic range callbacks and consume a
  contiguous, binding-validated range-state prefix.  At minimum XIP2,
  BASE_FEATURE, TEXTURE, SHOOTING_KNOT, STRATIFIED_FLOW, and each joint-refit
  block preserve enough canonical records to resume without re-solving a
  completed range.
- A changed implementation/config/input binding must refuse all inherited
  stage and range state.

## 5. Exact geometry operation

Geometry/XIP2 candidate evaluation must construct the frozen G27 geometry on
the native output dimensions, execute the full-native frozen warp, and only
then sample its integer residual when `sample_stride > 1`.  Decimating first and
constructing a new geometry on the decimated dimensions is forbidden.  Preserve
a full-resolution zero-versus-fitted control per pair after XIP2 quantization;
the bounded fitter may retain the fitted coordinate only when that exact
full-resolution control does not worsen source RGB SSE.

This closes the noncommuting proxy bug.  It does not turn source RGB into
scorer authority.

## 6. Governed n600 admission

Remove the caller-only `--confirm-full-n600` and
`--confirm-no-live-heavy-owner` authority.  Exactly n600 must require all of:

- `TAC_GOVERNED_ADMISSION=1` in the child environment;
- a current nonterminal canonical claim in
  `.omx/state/active_lane_dispatch_claims.md` for the fixed G32 lane and exact
  hash-bound job/platform values;
- claim age at most 24 hours;
- the current claim record and its canonical SHA-256 embedded in the stable
  launch binding, while the live ledger SHA-256 is recorded on each
  revalidation observation (binding the whole mutable ledger would brick an
  otherwise valid resume whenever an unrelated lane appends a row); and
- revalidation before materialization and every resumed heavy fit range/stage.

Bounded n1/n2 mechanism verification remains local and non-authoritative.

## 7. Claim, telemetry, and remaining blocker precision

- `nontraining_options_exhausted_on_bounded_source_objective` is removed.
  Replace it with a field that is false and names the unenumerated action
  families.
- Solver labels state the exact bounded family; no `maximum`, `complete`, or
  `exhausted` authority survives in current receipts.
- Positive int16 saturation at +32767 and negative saturation at -32768 are
  both counted.
- Per-section values remain null where not separately measured; do not copy a
  whole-packet objective/runtime into every section as if marginal.
- The action-universe/global score selection, public G29/G23 endpoint, exact
  scorer, complete ZIP pricing, control-bisimulation identity, semantic
  code-as-data audit, and terminal joint descent remain blockers unless this
  landing actually closes them.

## 8. Acceptance

1. Focused tests inject checkpoint-record drift, predecessor drift, coordinate
   drift, and legal geometry-only crash/resume.
2. Focused tests prove one selected-runtime setup and contiguous tail restart.
3. Focused tests prove fit-range checkpoint reopen and mutation refusal.
4. Focused tests prove direct n600 booleans cannot authorize execution and a
   governed live claim is required.
5. Geometry tests distinguish full-warp-then-sample from the forbidden
   sample-then-regeometry operation.
6. n1/n2 deterministic fixtures or a bounded real smoke are mechanism evidence
   only and never scientific/score evidence.
7. Focused pytest, Ruff lint, Ruff format, pycompile, lane validation, and an
   untracked-file-safe whitespace check pass.
8. A new dated G35 findings memo records exact fixes, tests, hashes, unresolved
   blockers, NO-FIRE status, and pointer delta false.

## Stores consulted

- byte-identical `CLAUDE.md` / `AGENTS.md`, `PROGRAM.md`, arbitrage skill, top
  project MEMORY, live lane/progress state, and last-24h directives;
- G32 spec/findings, exact n1 receipt, and current four-file implementation;
- G34 adversarial review and its executable attacks;
- canonical `tools/claim_lane_dispatch.py` and the existing governed-execution
  pattern in `tools/probe_einstein_kolmogorov_xi_bridge.py`.
