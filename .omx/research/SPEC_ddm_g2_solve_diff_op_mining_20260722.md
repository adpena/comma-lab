# DDM G2 solve-minus-predict differential operator mining — frozen implementation spec

Date: 2026-07-22
Lane: `ddm_g2_solve_diff_op_mining`
Delegation checkpoint: `codex_delegate:ddm_g2_solve_diff_op_mining:20260722T182405Z`
Authority: `ddm_g2_solve_diff_op_mining_20260722T182405Z.wrapped.prompt.txt`, SHA-256 `51f1de05508ff8c25cce0c71e9d638edf089bfb7f6be86435e25dd8530341119`

## Objective

Build a reusable, scorer-free primary-path instrument that compares the existing
full-n600 C1 exact solved scorer planes with the receiver-closed V12 add-zero
predictor and emits derivative-producing, typed JSONL telemetry for the next
`g2g2` Newton/KKT rung.  The instrument must measure, per pair, per target class,
per topology stratum, per margin stratum, and per fixed temporal window:

1. the exact real-linear resize `range(A)` and `ker(A)` energy split on the
   camera-space solve-minus-predict field;
2. the exact bounded-uint8 reachable canonical-basis lower bound from #532/#580;
3. the rank-four SegNet head coordinates, Fisher/margin flip-distance strata,
   and class-pair costate coordinates;
4. the fraction explained by SE(3) `xi` transport versus birth/death/event
   innovations, using the NumPy reference in `tac.lie`;
5. chart expressibility and irreducible residual sizes under real byte coders,
   with a governed compact parabolic shearlet/curvelet-family spatial chart and
   no Fourier residual basis;
6. the exact-resize-adjoint pullback of the rank-four class-coordinate costate
   into camera space, summarized in coefficient-space rows suitable for a
   standing costate-organ consumer;
7. sensitivity to both endpoints: START deltas from the current V12 obligation
   receipt/budget ladder and END tolerance rungs made by deterministically
   relaxing solve-plane deltas, not only the exact solved-plane endpoint.

The primary pass is analysis/instrumentation only.  It may read existing archive
bytes and frozen scorer-cache products.  It must not call a frozen scorer,
generate an archive, mutate a run, dispatch work, claim a score, or move the
frontier pointer.

## Frozen endpoints and custody

- High-rate endpoint selected because it is already built and is cheaper than a
  rebuild: `.omx/research/ddm_full_precision_target_planes_603_20260722T010130Z.json`,
  schema `direct_description_full_precision_target_planes.v1`, 600 uint8 scorer
  planes per frame, source role `existing_c1_solved_pair_scorer_planes`.  Read
  the 50 immutable 12-pair chunks on `/Volumes/VertigoDataTier/pact`; verify each
  selected chunk against the receipt SHA-256 before consuming it.  Realize each
  plane transiently with
  `tac.optimization.uint8_lattice_feasibility.realize_factor2_uint8_scorer_plane`.
  Do not persist full camera frames.
- Cheap START endpoint:
  `.omx/research/ddm_v12_obligation_n600_20260722T161517Z/ddm_v12_obligation_n600_add0.not_a_candidate.zip.receipt-bytes`,
  exact 102105 bytes and SHA-256 from its receipt.  Strictly parse with
  `receive_carrier_compose_archive`, then render bounded pair batches.
- Frozen target labels, margins, poses, and source frames:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
  Open stored NPY members by mmap; do not read the 5 GB cache into RAM.
- Current measured START bridge is the V12 receipt.  Preserve its exact baseline
  values and strata as external measured custody; do not re-score them.
- Evidence axis is exactly `[macOS-CPU frozen-scorer advisory]`.  Every receipt
  carries `research_only=true`, `execution_allowed=false`, `score_claim=false`,
  `promotion_eligible=false`, `archive_emitted=false`, `pointer_moved=false`,
  and `pointer="0.1910828242 [contest-CPU]"`.

## New implementation surfaces

Create only these code surfaces (names may differ only if collision forces it):

- `src/tac/optimization/solve_diff_operator_mining.py`
- `tools/measure_ddm_solve_diff_operator.py`
- `src/tac/optimization/tests/test_solve_diff_operator_mining.py`
- `.omx/research/configs/ddm_g2_solve_diff_op_mining_n600_20260722.json`

Generated outputs belong under a new dated
`.omx/research/ddm_g2_solve_diff_op_mining_*` directory.  The main session will
write final narrative/DAG/equation/self-review artifacts after reviewing the
implementation and measurement.  Do not edit the active sibling hot surfaces
`direct_description_carrier_compose.py`, `run_ddm_v9_carrier_compose.py`, G1
grammar files, V13 predictor files, scorer modules, evaluator files, or existing
receipts/configs.

## Required typed API

Implement fail-closed frozen dataclasses or Pydantic models for:

- `SolveDiffMiningConfigV1`
- `SolveDiffPairRowV1`
- `SolveDiffStratumRowV1`
- `SolveDiffWindowRowV1`
- `SolveDiffCostateRowV1`
- `SolveDiffSummaryV1`

Expose pure/testable helpers for:

- canonical JSON/JSONL serialization with no NaN/Infinity;
- SHA-256 checked chunk loading;
- exact solve-camera realization;
- `range(A)`/`ker(A)` energy accounting using `FullResizeKernel`;
- bounded uint8 reachability lower-bound accounting using
  `FullResizeKernel.uint8_reachability`;
- canonical class, topology, and margin masks;
- `xi` features and transport prediction using `tac.lie._se3_numpy` with the
  translation-first convention;
- rank-four class operator/head-coordinate and flip-distance accounting using
  the canonical equations and `factorized_adjoint` surfaces;
- exact resize adjoint pullback using
  `tac.analysis.hprc_synthesis_adjoint.bilinear_resize_adjoint`;
- real coded byte counts.  Measure deterministic zlib level 1 and LZMA preset 0
  payload lengths (fast full-P pricing surfaces, explicitly not entropy
  estimates) and
  select by a declared deterministic policy.  Entropy estimates are forbidden;
- governed compact parabolic spatial chart coefficients.  The name and output
  must say `compact_parabolic_shearlet` or `windowed_curvelet_family`, not
  `genuine_curvelet`; do not make a Candes–Donoho frame claim;
- an iterator/loader for costate rows so the existing costate organ can consume
  the typed JSONL without importing the CLI.

All public helpers must validate shape, dtype, class order, pair coverage,
finite values, and custody.  Never silently coerce missing metadata.

## Mathematical contract

For each realized solve camera frame `x_s` and predictor frame `x_p`, set
`Delta = float64(x_s) - float64(x_p)`.  Use the already-settled exact projector:

`Delta_range = P_range Delta`, `Delta_ker = Delta - Delta_range`.

Report squared-L2 fractions with a named zero-energy policy.  Prove numerically
in tests that the split reconstructs Delta and is orthogonal within an explicit
fp64 tolerance; never identify real-linear kernel dimension with bounded uint8
reachability.  Report the latter separately and explicitly as the exact lower
bound for the chosen canonical primitive basis.

Use cached target cells/margins only for stratification/Fisher ranking.  The
rank-four head operator comes from the canonical exact class operator and
pair-normal constants; it is not a claim that head coordinates alone are an RGB
encoder Jacobian.  Costate rows must therefore distinguish:

- `head_linearization`: exact in the 5-class quotient / rank-four subspace;
- `resize_adjoint`: exact for the bilinear resize operator;
- `inner_encoder_jacobian`: absent from the scorer-free primary path.

The emitted camera costate is a declared factorized surrogate built from the
class-coordinate field and exact resize adjoint.  It must not be labeled an
exact frozen-SegNet input gradient.  Include the blocker/status field
`INNER_ENCODER_JACOBIAN_NOT_MEASURED_PRIMARY_PATH`; optional scorer validation is
out of this implementation pass unless the main session explicitly invokes it.

For temporal transport, convert pose deltas to SE(3) with
`tac.lie._se3_numpy.exp_se3`/`log_se3` and use the resulting translation-first
twist/adjoint features in a deterministic leave-one-window-out ridge or least
squares model.  Report held-out explained squared energy.  Do not call plain
lag autocorrelation `xi transport`.  Birth/death/event mass is the remaining
support that cannot be matched by the fitted transport within the declared
support threshold.  Report Movable predictable-after-birth separately from
birth-frame innovation and per-frame residual.

For END sensitivity, deterministic tolerance rungs may shrink/clip the exact
solve-minus-predict scorer-space field according to explicit retained-energy or
flip-distance thresholds.  Label these `DERIVED_TOLERANCE_LADDER`, not evaluator
measurements.  For START sensitivity, consume exact measured V12 budget-ladder
rows and attribute which stratum columns shrink where the receipt provides the
necessary per-stratum data; otherwise emit an explicit `NOT_IDENTIFIABLE_FROM_RECEIPT`
field rather than inventing an attribution.

For candidate operator ranking, compute measured bytes and derived reachable
Seg debt for combinations of at least: `xi_transport`, `rank4_head_chart`,
`compact_parabolic_shearlet`, and `irreducible_residual`.  Apply the registered
rate stop law `marginal_delta_S_per_byte > 25/37545489` where the numerator is
supported; otherwise mark the KKT admission `BLOCKED_NO_RECEIVER_DELTA_DSEG`.
No candidate archive is built.

## Resource, resumability, and output contract

- Default batch/chunk size: at most 12 pairs.  Peak working set must remain
  bounded; never materialize all camera frames or all dense deltas.
- Run a storage waterfall/preflight even though outputs are small.  Bulk scratch
  must use `/Volumes/VertigoDataTier/pact` first, then `/Volumes/APDataStore/pact`.
  Durable operator-facing outputs live in `.omx/research`; never cite `/tmp`.
- Every completed chunk writes a distinct atomic stage checkpoint containing
  source hashes, pair range, row count, and output JSONL digest.  `--resume`
  revalidates and skips exact completed stages.  Never overwrite an unequal
  stage checkpoint.
- A finalizer-only code correction may consume already-complete immutable
  stages only with explicit `--resume-stage-module-sha256 <sha256>`.  The tool
  must refuse a missing stage (no mixed-producer run), validate every stage
  against that declared producer SHA, and record separate stage-producer and
  finalizer module hashes in the receipt.
- Write JSONL incrementally and atomically by stage.  Combine only after all
  requested stages verify.  PNG/HTML charts are bounded summaries; no raw frame
  dump.  Produce: energy-by-hyperplane, xi held-out/persistence, coded-byte
  waterfall, flip-distance histogram, and 3–5 hard-pair panels.
- Auto-clean only certified success scratch.  Emit a machine-readable cleanup
  manifest with path, bytes, SHA-256/tree hash, rebuild command/config hashes,
  and reason; otherwise retain bytes and report a blocker.

## CLI contract

The thin CLI accepts only typed config plus explicit output/resume controls,
e.g.:

```bash
python3 tools/measure_ddm_solve_diff_operator.py \
  --config .omx/research/configs/ddm_g2_solve_diff_op_mining_n600_20260722.json \
  --output-root .omx/research/ddm_g2_solve_diff_op_mining_<UTC> \
  --resume
```

For a finalizer-only replay over a complete older stage set, add
`--resume-stage-module-sha256 <checkpoint producer SHA-256>`.  This option is
invalid without `--resume` and cannot create a missing stage.

Also provide a bounded `--pair-limit` or a dedicated fixture config for local
smoke without weakening production config validation.  The receipt must persist
the exact argv and hashes of tool, module, config, all read receipts/archives,
git SHA, host/axis, and output members.

## Acceptance tests

At minimum, the new test file must prove:

1. strict config/schema rejection of unknown keys, wrong class order, score or
   execution authority, non-12-or-smaller chunks, and malformed SHA/path data;
2. JSONL canonical round trip and rejection of nonfinite values;
3. exact `range + ker == delta`, orthogonality, and separate uint8-reachability
   lower-bound semantics on a small geometry fixture;
4. exact resize-adjoint dot product against the forward operator;
5. rank-four class operator gauge-null/rank properties and honest inner-Jacobian
   blocker labeling;
6. deterministic real-coder byte accounting and no Fourier candidate name;
7. deterministic `tac.lie` xi feature production and held-out transport split;
8. correct Movable birth-frame versus post-birth versus per-frame partition on a
   synthetic sequence;
9. deterministic tolerance-ladder monotonicity with derived-not-measured labels;
10. chunk SHA refusal, write-once checkpoint behavior, resume skip/revalidation,
    bounded pair coverage, and no archive member/output;
11. smoke CLI receipt contains all false-authority/pointer fields and emits
    pair/stratum/window/costate JSONL plus chart paths;
12. importing the module and running unit tests does not import/load SegNet,
    PoseNet, MLX, or create files outside the supplied temporary/output roots.

Run targeted tests, a small fixture smoke, `ruff`/format checks applicable to
new Python, and compile checks.  Do not commit; the main session owns review,
serializer landing, and commit.

## Explicitly out of scope / do not touch

- No scorer evaluation, GPU/Modal/CUDA/MLX dispatch, paid service, daemon, or
  lane claim beyond the already-registered local research lane.
- No archive creation, mutation, deletion, candidate promotion, score claim,
  pointer edit, or edit to `reports/latest.md`.
- No re-derivation of settled resize nullity, head rank, class order, or existing
  V12 score values.
- No Fourier residual basis and no claim that a compact spatial atom is a
  genuine curvelet frame.
- No edits to pinned upstream, evaluator/scorer weights/code, live run dirs,
  G1 grammar induction, V13 worldsheet predictor, or existing result receipts.
- No MAIN landing.  This branch must end with a commit that is explicitly
  `REQUIRES MAIN LANDING REVIEW`.

## Round-1 implementation review corrections (binding)

The first delegated implementation pass produced a skeleton but did not pass
review.  The next pass must correct these concrete defects before declaring the
spec complete:

1. Empirical probe of the exact SHA-bound V12 archive supersedes the initial
   review assumption: `receive_carrier_compose_archive(...).render_pairs(ids)`
   returns `(B,2,384,512,3)` uint8 planes for this artifact.  Treat those as the
   predictor's receiver-closed scorer-plane representation.  Apply the same
   exact factor-2 realization separately to solved and predictor planes, then
   form `delta_camera = solved_camera - predictor_camera` and apply the exact
   resize operator to that camera-space difference.  Refuse any unexpected
   runtime shape instead of silently assuming camera-native output.
2. Production must render and load no more than `chunk_size <= 12` pairs at a
   time.  It must not return or retain an n600 camera tensor.  Release each
   chunk before loading the next.
3. Temporal/xi fitting must consume a bounded compact chart/stratum feature
   sequence.  Never accumulate all dense n600 scorer deltas in memory.
   Persist enough compact per-stage features that a valid `--resume` can
   revalidate and skip a completed stage without recomputing it.
4. A stage is one bounded pair chunk.  Its write-once checkpoint must cover all
   typed stage JSONL/intermediate members needed for deterministic combine;
   resume must validate their hashes and load the compact rows/features.
5. Persist the tolerance-ladder rows, the measured start-receipt rows (while
   keeping per-stratum attribution explicitly unidentifiable where custody does
   not exist), per-stratum/window information fractions and real coder bytes,
   and both rank-4-head and compact-parabolic-shearlet costate families.
6. The adjoint field must be derived from cached target labels/margins and the
   endpoint residual.  Do not manufacture class coordinates from coefficient
   index alone.  Keep the absence of the frozen SegNet inner Jacobian explicit;
   no exact-gradient claim is allowed.
7. Produce actual bounded PNG and HTML summaries, rank hard pairs by measured
   endpoint residual energy, and include 3--5 hard-pair panels when pair count
   permits.
8. Read and SHA-check the predictor start receipt and persist its measured
   global ladder values.  Attribute no nonexistent per-stratum predictor
   telemetry.
9. Add the specified targeted tests.  Use
   `/Users/adpena/Projects/pact/.venv/bin/python` for pytest/ruff/compile/smoke;
   the worktree-local `.venv` does not exist.
10. Resolve all `ruff` findings in the touched Python files.  Correct the
    production loader return annotation and remove unused imports.
11. A completed n600 stage set may be re-finalized after a warning-only
    aggregation correction only through the explicit stage-producer override
    above.  The finalizer must leave all 50 stage checkpoints byte-identical,
    refuse a wrong producer SHA, and expose both module hashes in the receipt.

## Late operator addenda disposition

The 2026-07-22 19:16--19:26Z Lane phase/jitter/BEV/openpilot and Movable
projective/template directives arrived after the frozen primary pass began.
The aggregate ledger must report any signal available from the typed stage
rows and fail closed on the rest.  In particular, cached target labels and
receiver RGB planes do not identify the SegNet 16-channel stride-2 skip
activation, track identity, a receiver-closed projective reconstruction, or the
receiver delta-d-seg of a proposed phase symbol.  These missing-custody states
are implementation blockers, not negative family verdicts; they must not be
filled with label-proxy claims.
