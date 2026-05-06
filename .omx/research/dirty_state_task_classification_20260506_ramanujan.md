# Dirty-state task classification

Date: 2026-05-06
Operator: codex
Scope: dirty-state preservation only
Repo: `/Users/adpena/Projects/pact`
Branch: `main`

## Safety contract

- No code was cleaned, reverted, staged, or modified.
- No GPU, remote, exact-eval, Lightning, Modal, Vast.ai, or other dispatch job
  was run.
- This ledger is a preservation and routing artifact only.

## Root state

- `git status --short --branch --ignore-submodules=none` reports
  `main...origin/main [ahead 1]`.
- Local ahead commit observed: `4d4a25c1 Harden research-backed field planning`.
- Dirty root code scope: 9 modified `src/tac` files, 152 insertions and 24
  deletions by `git diff --stat`.
- Nested gitlink-like intake directories are dirty. `git submodule status
  --recursive` failed with `fatal: no submodule mapping found in .gitmodules`
  for the public PR intake paths, so these are best treated as detached nested
  repos/custody clones, not normal submodules.

## Modified `src/tac` files

### `src/tac/balle_sensitivity_weighted.py`

- Likely workstream: sensitivity-weighted Balle/hyperprior correctness.
- Observed change: `aggregate_pixel_sensitivity_to_blocks()` now handles
  `aggregate="sum"` explicitly and fixes mean aggregation for a padded tail
  block by dividing by real element count instead of `block_size`.
- Blockers:
  - Needs targeted CPU regression tests for mean/sum/max with non-divisible
    tail lengths.
  - Existing padding path still appears to create padding tensors without an
    explicit device argument; if this function is expected to run on CUDA
    tensors, the owner should test device preservation before promotion.
- Safe ownership recommendation: sensitivity/Balle lane owner should keep this
  patch isolated and add focused aggregation tests before broad preflight.
- Next step: run a small local pytest for block aggregation semantics only; do
  not dispatch from this patch alone.

### `src/tac/component_sensitivity_artifact.py`

- Likely workstream: component-sensitivity custody and certification hardening.
- Observed changes:
  - `file_metadata()` now rejects `/tmp`, `/private/tmp`, and `/var/tmp`
    evidence paths.
  - Certification validation now requires `review_unresolved_blockers` to be
    present and exactly an empty list, rather than accepting missing/null.
- Blockers:
  - Tests or local fixtures using pytest `tmp_path` as persisted evidence will
    likely fail until moved to durable experiment/result paths or explicitly
    scoped as non-persisted.
  - Existing certification manifests that omit `review_unresolved_blockers` or
    set it to null now fail closed.
- Safe ownership recommendation: custody/preflight owner should pair this with
  fixture migration and manifest-schema tests before enabling as a durable
  gate.
- Next step: inventory existing certification artifacts for missing/null
  `review_unresolved_blockers`; update fixtures deliberately, not by blanket
  search/replace.

### `src/tac/hnerv_wavelet_apply_transform.py`

- Likely workstream: WR01/HNeRV wavelet apply-transform hardening.
- Observed changes:
  - `strength_numerator` now must be strictly positive.
  - `wavelet_archive`, `output_dir`, and optional `source_archive` now call a
    `/tmp` path guard.
  - Section constants are now explicit strings checked against
    `REPACKABLE_SECTIONS`.
- Blockers:
  - `_guard_no_tmp_path()` is called but no definition was found in this file;
    `build_wavelet_apply_transform_candidate()` will raise `NameError` before
    archive reading once invoked.
  - The section membership checks use `assert`, which can be stripped under
    optimized Python execution. If this is meant as a fail-closed guard, use an
    explicit exception in the owning patch.
  - Rejecting `/tmp` output paths may break local tests that use pytest
    `tmp_path`; the owner should distinguish persisted-manifest paths from
    ephemeral test output.
- Safe ownership recommendation: WR01/HNeRV wavelet owner should define the
  missing helper, add focused tests for zero strength, `/tmp` rejection, and
  section mapping, then rerun only local tests.
- Next step: fix the missing helper before any use of
  `build_wavelet_apply_transform_candidate`; no exact eval or dispatch should
  be launched from the current dirty state.

### `src/tac/mask_entropy_coder.py`

- Likely workstream: mask entropy decoder fail-closed behavior.
- Observed change: decompressed-size mismatch now raises `ValueError` with a
  deterministic corruption/version message instead of using an `assert`.
- Blockers:
  - Tests expecting `AssertionError` need deliberate update.
  - Needs malformed-size regression coverage so optimized Python cannot skip
    the guard.
- Safe ownership recommendation: mask entropy owner can likely promote this as
  a narrow decoder hardening patch after a focused corrupt-payload test.
- Next step: add/run local decode failure tests only.

### `src/tac/mdl_bayesian_codec.py`

- Likely workstream: lane maturity / research-scaffolding classification for
  an MDL Bayesian codec advisor.
- Observed change: top-of-file comment marks the module as
  `RESEARCH_SCAFFOLDING_NOT_WIRED` and says it is an analysis/ranking advisor,
  not a byte-producing codec.
- Blockers:
  - Comment references `lane_mdl_bayesian_advisor`, but no matching lane id was
    found in `.omx/state/lane_registry.json` during read-only inspection.
  - No behavior changed; this is not a functional guard by itself.
- Safe ownership recommendation: lane-maturity owner should either add the lane
  through `tools/lane_maturity.py` or revise the comment to match current
  registry state.
- Next step: run registry validation only after the owning patch reconciles the
  referenced lane id.

### `src/tac/neural_weight_codec_sensitivity.py`

- Likely workstream: NWCS neural-weight codec decode safety.
- Observed change: bucket-size mismatch error is expanded to explain that the
  codec should be rebuilt from the archive `codec_checkpoint_blob`.
- Blockers:
  - Needs regression coverage for mismatched codebook ladders.
  - Needs confirmation that all valid archive/container paths preserve and use
    the matching codec checkpoint blob.
- Safe ownership recommendation: NWCS owner should keep this with container
  decode tests, not with unrelated sensitivity-map changes.
- Next step: run focused NWCS encode/decode mismatch tests locally.

### `src/tac/self_compressing_nn.py`

- Likely workstream: self-compressing NN lane maturity / deferred-scaffolding
  classification.
- Observed change: top-of-file comment marks the module as
  `RESEARCH_SCAFFOLDING_NOT_WIRED` with reactivation criteria.
- Blockers:
  - Comment references `lane_scnn`, but no matching lane id was found in
    `.omx/state/lane_registry.json` during read-only inspection.
  - No runtime behavior changed; this is documentation/classification only.
- Safe ownership recommendation: lane-maturity owner should reconcile the
  registry reference or keep this as a ledger-only note.
- Next step: if promoted, register the lane through the lane maturity tool and
  attach evidence paths.

### `src/tac/sensitivity_map.py`

- Likely workstream: certified sensitivity-map metadata hardening.
- Observed change: certification metadata now requires
  `review_unresolved_blockers` to be present and exactly an empty list.
- Blockers:
  - Existing metadata fixtures/manifests with missing or null blockers now fail
    closed.
  - Needs a focused validation test for missing, null, non-list, nonempty, and
    empty-list cases.
- Safe ownership recommendation: sensitivity-map owner should coordinate this
  with `component_sensitivity_artifact.py` because the same schema tightening
  appears in both files.
- Next step: run focused metadata-validation tests after fixture inventory.

### `src/tac/tto.py`

- Likely workstream: TTO/inflate-time scorer-load safety classification.
- Observed change: docstring now states `INFLATE_TTO=0` gated status and says
  `test_time_optimize()` must not run at inflate time unless explicitly enabled.
- Blockers:
  - This is documentation only; no enforcement was added in `tto.py`.
  - Comment references `lane_tto_inflate_gated`, but no matching lane id was
    found in `.omx/state/lane_registry.json` during read-only inspection.
  - Needs preflight or inflate-path enforcement if the gate is intended to be
    operational.
- Safe ownership recommendation: TTO/preflight owner should add an executable
  guard and registry evidence before treating this as a closed safety issue.
- Next step: trace actual inflate call sites for TTO and add a local non-GPU
  guard test if any call path exists.

## Dirty nested repos and gitlink-like paths

### Public PR intake clones

Paths:

- `experiments/results/public_pr100_intake_20260504_codex/source`
- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source`
- `experiments/results/public_pr103_intake_20260504_codex/source`
- `experiments/results/public_pr105_kitchen_sink_intake_20260504_codex/source`
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source`
- `experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/repo`
- `experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/repo`
- `experiments/results/public_pr91_intake_20260504_worker/pr91_src/repo`

Observed nested branches:

- PR100: `submission/hnerv_lc_v2...origin/submission/hnerv_lc_v2`
- PR101: `hnerv-ft-microcodec...origin/hnerv-ft-microcodec`
- PR103: `add-hnerv_lc_ac...origin/add-hnerv_lc_ac`
- PR105: `submission/kitchen_sink...origin/submission/kitchen_sink`
- PR106: `submission/belt_and_suspenders...origin/submission/belt_and_suspenders`
- PR81: `qzs3-range-mask...origin/qzs3-range-mask`
- PR82: `henosis_frontier_submit...origin/henosis_frontier_submit`
- PR91: `hpac-coder-hybrid...origin/hpac-coder-hybrid`

Observed nested changes:

- PR100/101/103/105/106/81 each modify five public-submission files:
  `submissions/fp4_mask_gen/compress.py`,
  `submissions/neural_inflate/train_ren.py`,
  `submissions/ph4ntom_drv/compress.py`,
  `submissions/quantizr/compress.py`, and
  `submissions/svtav1_dilated_ren/svtav1_dilated_ren_training.py`.
- PR82/91 modify the same set except `submissions/ph4ntom_drv/compress.py`.
- The inspected PR101 diff shows comment-only `KL_BATCHMEAN_OK` annotations on
  `F.kl_div(..., reduction="batchmean")` call sites.

Likely workstream: public PR intake / external-code quality-debt waiver for KL
`batchmean` scanners.

Blockers:

- These are dirty nested repos without `.gitmodules` mappings; root status can
  show them as modified gitlinks, but root-level submodule tooling is not
  reliable for custody.
- Changes are inside public/external submission clones and should not become
  implicit source of truth for mainline `tac`.
- Comment-only waivers may be appropriate for scanner suppression, but each
  clone should retain provenance that this is external intake code.

Safe ownership recommendation:

- Public-frontier intake owner should preserve these nested edits until the KL
  scanner/waiver decision is recorded in a dated ledger or reverse-engineering
  note.
- Do not stage root gitlink changes casually. If a patch must be preserved,
  record it as a custody artifact or patch file linked from a ledger.

Next step:

- Confirm whether the KL scanner expects these exact `KL_BATCHMEAN_OK` tokens.
  If yes, preserve a patch manifest per PR intake clone; if no, leave them
  untouched until the owner decides whether to discard or rework.

### Kaggle ingest nested repo

Path:

- `reports/raw/kaggle_ingest/kaggle-dilated-h64-long1000-retry-v6-20260410T234220Z/comma_video_compression_challenge`

Observed nested state:

- Branch: `master...origin/master`
- Untracked files:
  - `submissions/gt_passthrough/inflate.py`
  - `submissions/gt_passthrough/inflate.sh`
  - `submissions/gt_passthrough/report_pyav.txt`

Likely workstream: Kaggle ingest / ground-truth passthrough or baseline
submission forensic fixture.

Blockers:

- This lives under `reports/raw`, which is normally private/raw custody rather
  than public source.
- Files are untracked inside a nested repo, so they are easy to lose if the
  nested checkout is cleaned by mistake.

Safe ownership recommendation:

- Kaggle-ingest owner should either manifest these files in a custody ledger or
  move a sanitized summary to `.omx/research` while leaving raw files private.
- Do not clean the nested repo without confirming whether `gt_passthrough` is
  an intentional baseline artifact.

Next step:

- Record byte sizes and SHA-256s in a follow-up custody manifest if the owner
  wants this baseline preserved.

## Cross-cutting ownership recommendations

- Treat the current dirty tree as multiple independent workstreams:
  sensitivity/certification hardening, HNeRV WR01 hardening, entropy/NWCS
  decoder fail-closed behavior, lane-maturity comments, public PR intake
  waivers, and Kaggle raw-custody artifacts.
- Do not combine these into one broad commit without owner review. Several
  changes are comment-only, while others alter fail-closed validation behavior.
- Highest immediate blocker is the missing `_guard_no_tmp_path()` helper in
  `src/tac/hnerv_wavelet_apply_transform.py`.
- Second-order blocker is lane-registry drift for comments referencing
  `lane_mdl_bayesian_advisor`, `lane_scnn`, and `lane_tto_inflate_gated`.
- Schema-tightening changes in `component_sensitivity_artifact.py` and
  `sensitivity_map.py` should be tested together because they enforce the same
  explicit-empty-blocker convention.

## Read-only commands used

- `git status --short --branch --ignore-submodules=none`
- `git diff --stat -- <named src/tac files>`
- `git diff -- <named src/tac files>`
- `git diff --numstat -- <named src/tac files>`
- `git submodule status --recursive`
- `git diff --submodule=log -- <nested intake paths>`
- `git -C <nested repo> status --short --branch`
- `git -C <nested repo> diff --stat`
- `git -C <nested repo> diff -- <representative public PR files>`
- `find <kaggle nested repo> -maxdepth 2 -type d -name .git -print`
- `jq` / `rg` read-only checks against `.omx/state/lane_registry.json`

No cleanup, staging, code editing, GPU jobs, or remote jobs were performed.
