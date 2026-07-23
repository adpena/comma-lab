# DDM AT1x atlas materialization implementation spec

Date: 2026-07-23
Authority: delegated `ddm_at1x_atlas_materialize`; `research_only=true`;
`score_claim=false`; local macOS-CPU advisory execution only; no provider
dispatch and no training. MAIN landing review is required.

## Recall boundary

Do not recompute the settled AT1 inventory or the complete VJP campaign. Consume:

- `src/tac/optimization/scorer_analytic_atlas.py`;
- `src/tac/optimization/scorer_module_inventory.py`;
- `.omx/research/ddm_at1_scorer_analytic_atlas_20260723T194312Z/`;
- `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json`,
  whose settled status is `COMPLETE_N600` with all 600 pair sidecars;
- the SHA-pinned E2 archive
  `/Volumes/VertigoDataTier/pact/evidence/ddm_e2_pose_stream_and_doctrine_export_20260723/upstream_harness/submission/archive.zip`
  (`8891012e4019e474d1e8ae7578104d74f27c25838c7b68a3798af35853469819`).

The immutable upstream input root is `/Users/adpena/Projects/pact/upstream`.
It may be read but must not be modified.

## Owned implementation

Build one cohesive, testable materialization surface, preferably:

- `src/tac/optimization/scorer_atlas_materialization.py`;
- `tools/materialize_ddm_at1x_atlas.py`;
- focused tests under `src/tac/tests/`.

Minimal compatible extensions to `scorer_analytic_atlas.py` are allowed.
Do not edit upstream, scorer weights, settled AT1 receipts, the VJP campaign,
frontier pointers, or shared live state.

### 1. Exact-lock environment and inventory contract

Provide deterministic helpers/CLI stages that:

- require an explicit SSD environment path below
  `/Volumes/VertigoDataTier/pact`;
- record the exact `uv sync --locked --group cpu` command, Python selection,
  `UV_PROJECT_ENVIRONMENT`, `UV_LINK_MODE=copy`, storage preflight, uv version,
  lock/project bytes and SHA-256, and all selected/observed package versions;
- certify environment bytes and a deterministic tree digest over
  `(relative path, bytes, sha256)` rows;
- define the environment as reproducibly rebuildable but never delete it;
- rebuild `build_scorer_module_inventory` under that interpreter and require
  zero `version_drift` plus
  `PASS_LOCKED_LIBRARY_SOURCES_MATERIALIZED`;
- preserve exact per-package failure rows if the lock cannot materialize.

The runner must be resumable by stage, publish atomic stage receipts, and never
overwrite a non-byte-identical prior receipt.

### 2. Locked closed forms

Under the verified locked interpreter, load the frozen PoseNet and SegNet
checkpoints and emit source-bound factors for:

- every eval BatchNorm affine table;
- every actual SE gate, preserving its real activation (`ReLU` for FastViT
  `SEModule`, `SiLU` for EfficientNet `SqueezeExcite`) rather than inventing
  a common formula;
- convolution-kernel native 2-D DFT magnitude and phase;
- BN followed by actual SiLU compositions only.

Each factor must carry `FIRST-RUNG=true`, network/layer identity, checkpoint and
locked-source hashes, source file with line-start/line-stop citations, package
version-set hash, freshness rule, consumer status, and a canonical content hash.
Large factor shards belong on the SSD evidence tier; the tracked receipt is a
small immutable index with shard bytes/SHA/path and deterministic rebuild
command. No global DFT dead band may be admitted: point-4/#580 remains an empty
exact-dead-band set and `REFUSE_ZERO_BYTE_TRUNCATION`.

### 3. n600 gaze and contraction atlas

Validate and index, without recomputation, every pair sidecar from the settled
VJP campaign. Require:

- exact pair coverage `0..599`, no duplicates/refusals;
- six nonzero fp32 Pose rows per pair at scorer plane and camera-input relay;
- the existing Seg winner-rival/head-normal pullback, explicitly bound to
  rank-4 `segnet_head_rank4_linear_flipdist_v1`;
- hash/shape/dtype/source custody for every referenced tensor;
- a version stamp identifier and version-set SHA on every tensor index row.

Compute deterministic float64 contraction spectra from the stored fp32 tensors
at the measured candidate relay depths:

- Pose: eigenvalues of the six-by-six row Gram at `scorer_plane_y` and
  `camera_input_x`;
- Seg: contracted singular-energy row at both depths, plus head-pair norm
  distribution summaries.

Emit per-pair rows plus n600 aggregates, immutable stage checkpoints, and an
atlas manifest whose closed-form/gaze/Jacobian counts are nonzero. Preserve the
honest scope: these are the two measured relay depths, not unmeasured internal
network layers.

The new atlas coverage must account for all 600 gaze pairs. The prior V19
receiver-closed lambda producer still has only eight exact V19 joins; do not
rename the remaining 592 as V19-measured. Instead emit an explicit n600
gaze-lambda/index coverage table that classifies 8 rows as
`V19_EXACT_JOIN_AVAILABLE` and 592 as
`GAZE_MEASURED_V19_JOIN_OWED_COUNTED_INERT`, with a named consumer waiting on
receiver-closed V19 evidence.

### 4. Version-drift calibration

Provide a parser/receipt builder for one official upstream `evaluate.sh` run in
the exact locked environment on the exact E2 archive. Record exact argv,
environment, archive/runtime/upstream hashes, stdout/stderr/report custody,
wallclock, and parsed `archive_bytes`, `d_seg`, `d_pose`, and formula total.
Compare against:

- observed-environment E2 total `43.411509751432`,
  `d_seg=0.02861482`, `d_pose=162.58094788`,
  `archive_bytes=343466`;
- the measured frozen-scorer realization row
  `d_seg=0.027470296224`, whose delta to upstream was
  `+0.001144523776`.

Report signed per-axis and score-term drift. This remains
`[macOS-CPU locked-env upstream frozen-harness advisory]`,
`score_claim=false`, never contest authority.

### 5. Shared contracts and receipt

All generated receipts must include:

- `first_rung`, `research_only`, `score_claim=false`, evidence axis, and
  scoped verdict;
- amplitude figures only with an explicit through-R/uint8 survival row; if no
  amplitude factor is produced, state zero and why;
- explicit non-additive pools; do not sum overlapping factors;
- freshness stamps and directive-consumption rows for operator directives
  `2026-07-19T19:42:07Z` and `2026-07-19T19:48:01Z`;
- storage/certify-or-block and reconstruction commands;
- pointer unchanged, triality, and MAIN landing-review requirement.

Add focused unit tests for lock drift refusal, real-activation SE factors,
tree/factor receipt custody, exact n600/no-duplicate coverage, six-row Pose
spectra, Seg rank-4 binding, 8+592 classification, version-stamp requirement,
and calibration arithmetic. Tests must use tiny fixtures and never invoke the
real scorer or mutate SSD evidence.

The implementation worker must not commit. It returns the exact changed-file
list, commands run, and any remaining execution blocker to the parent. The
parent may commit only on the isolated Codex worktree branch; MAIN landing
review remains mandatory.
