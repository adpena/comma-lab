# Cross-Paradigm Static Surface And B3 Custody Hardening

Date: `2026-05-08`
Owner: `codex`
Scope: local score-lowering packet closure plus custody guard hardening.
Dispatch performed: `false`
Score claim: `false`
Promotion/rank/kill claim: `false`

## Context Reviewed

Fresh local review covered the newest 2026-05-08 `.omx/research` control
ledgers around CPU/GPU dual-axis semantics, DALI/PyAV loader drift, A1/A2/A4,
ChARM, Track 1, and cross-paradigm packet closure. I also reviewed the latest
repo-local Claude memory snapshot:

- `.omx/research/codex_30k_strategy_review_cpu_gpu_loader_drift_20260508.md`
- `.omx/research/true_strategy_cpu_gpu_loader_drift_optimal_floor_20260508_xhigh.md`
- `.omx/research/a2_sensitivity_weighted_pr101_packet_ladder_20260508_codex.md`
- `.omx/research/codex_finding_charm_high_a_b_recursive_review_20260508.md`
- `.omx/research/track1_a1_pr101_archive_state_loader_20260508_worker_a.md`
- `.omx/research/loader_drift_discriminator_hardening_20260508_worker_b.md`
- `.omx/research/next_score_lowering_actions_20260508_worker_d.md`
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_grand_council_pcc4_kill_memory_review_enforcement_20260430.md`

The actionable conclusion was to avoid duplicating active claimed GPU lanes and
instead close packet-custody gaps on the already byte-closed
cross-paradigm ADMM/K plus Op1 candidate.

## Packet Closure

Rebuilt the cross-paradigm candidate with the current static-release-surface
builder:

```bash
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py \
  --output-root experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508
```

Newest artifact:

- build dir:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/`
- archive:
  `experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/archive.zip`
- archive bytes: `153513`
- archive SHA-256:
  `7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897`
- ZIP member: `x`
- Op1 inner blob: `137348` bytes
- CPLX decoder section: `137419` bytes
- PR101 latent blob: `15387` bytes
- PR101 sidecar blob: `607` bytes
- local smoke:
  - `rel_err_vs_lossy_substrate=0.005149`
  - `rel_err_vs_orig_fp32=0.037245`
  - `max_per_tensor_lossy=0.012669`
  - `n_tensors=28`
  - `n_latent_pairs=600`

The builder now stages the static submission surface:

- `submission_dir/archive.zip`
- `submission_dir/archive_manifest.json`
- `submission_dir/report.txt`

It does not write `contest_auth_eval.json` and does not mark the packet
dispatch-ready.

## Non-Final Compliance

Command:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/submission_dir \
  --archive experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/archive.zip \
  --archive-manifest-json experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/submission_dir/archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 7bbba307b1432d8d885e22533fdda9ab5cc87a6025510b2d5098084895284897 \
  --expected-archive-size-bytes 153513 \
  --json-out experiments/results/cross_paradigm_admm_x_op1_static_surface_codex_20260508/cross_paradigm_admm_x_op1_finalizer_20260508T185236Z/pre_submission_compliance.nonfinal.json \
  --strict
```

Result:

- `passed=true`
- required failures: none
- non-final optional missing: `auth_eval_optional_missing`

This is `[CPU-build]` packet readiness only. It is not `[contest-CUDA]`,
`[contest-CPU]`, a score, a promotion, or a retirement.

## B3 Custody Hardening

The live B3 scanner found older ignored `experiments/results/**/build_manifest.json`
rows with `archive_relpath` plus `archive_sha256` but no custody disposition.
Permanent source fix: the following CPU-build generators now emit
`custody_status="transient-allowed"` and an explicit custody reason:

- `experiments/lossy_coarsening_lightning_cuda_test.py`
- `tools/build_admm_x_lossy_coarsening_path_b_step6.py`
- `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py`
- `tools/build_pr106_uniward_runtime_packet.py`
- `tools/build_cross_paradigm_admm_x_op1_finalizer.py`

Focused tests were updated so future regressions fail before new artifacts are
generated.

Historical ignored local manifests were annotated in place with the same
transient custody disposition. These edits are intentionally local custody
state under ignored `experiments/results/`; the durable signal is this ledger
plus the generator/test changes.

Verification:

```bash
.venv/bin/python tools/check_build_manifest_archive_custody_clean.py --repo-root . --strict
```

Result:

```text
[B3-build-manifest-custody] OK
```

The untracked-source disposition manifest was also updated for this turn's
local generated runtime packet files:

- `experiments/results/` runtime-source baseline:
  `count=10527`,
  SHA-256 `94a472acd44166552c9987ba7e4ba35768bc1e8a7de8654b1a4930f6e298d445`
- tracked disposition added for this ledger.

## Protocol Corrections From Latest Memory Review

The xhigh read-only review also found two stale protocol statements in
`CLAUDE.md`.

1. `check_artifact_lifecycle_compliance` was described as warn-only / pending
   strict flip even though `src/tac/preflight.py` now runs it with
   `strict=True`. `CLAUDE.md` now describes this gate as strict.
2. The `/tmp` rule was internally inconsistent with heartbeat/Codex examples
   that still used `/tmp`. The protocol now routes live scratch examples to
   `.omx/tmp/` and clarifies that `/tmp` may appear in historical transcripts
  only as scratch-only, non-evidence context, never as durable custody.

The first full preflight rerun after staging the artifact registry surfaced one
more guard interaction: `tools/audit_release_index_split.py` treated every
staged `.omx/state/` file as provider/runtime state except a small historical
allowlist. That blocked the artifact lifecycle registry itself even though
`CLAUDE.md` and `src/tac/artifact_lifecycle.py` require it as a committed
control-plane contract. The release split guard now narrowly allows the durable
control files:

- `.omx/state/artifact_kind_registry.yaml`
- `.omx/state/artifact_classification_allowlist.json`
- `.omx/state/next_catalog_number.txt`

Provider/runtime state remains blocked; the regression test
`test_durable_omx_control_files_are_stageable` covers the exception.

## Concurrent ChARM Hardening Carried Forward

The shared worktree also contained a ChARM trainer hardening diff. I did not
originate it, but verified and carried it forward because it is relevant to the
A4/entropy-coding workstream:

- `CharmContextNet` no longer advertises an unused constructor argument;
- the CARM2/CHRC wire-contract comment now says the embedded range payload is
  a self-contained CHRC-framed blob, not only raw range-coded bits;
- `build_archive()` now writes deterministic ZIP entries with pinned timestamp
  and permissions before computing archive SHA-256.

Focused verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_charm_range_coder.py \
  src/tac/tests/test_train_charm_50k_toy_rate_math.py -q
```

Result: `40 passed, 3 warnings`.

## Remaining Blockers

The cross-paradigm candidate still requires all of the following before any
score claim or dispatch promotion:

1. explicit Level-2 lane claim immediately before exact eval dispatch;
2. exact `[contest-CUDA]` auth eval on the exact archive/runtime tree;
3. exact `[contest-CPU]` Linux x86_64 auth eval for public-axis statements;
4. `contest_auth_eval.json` copied into the submission surface only after the
   real eval exists;
5. adversarial result review and terminal dispatch-claim row.

Until then, the artifact remains a local byte-closed score-lowering candidate
with `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
