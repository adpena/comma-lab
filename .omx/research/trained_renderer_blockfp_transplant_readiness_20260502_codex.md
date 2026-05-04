# Trained Renderer Block-FP Transplant Readiness - 2026-05-02 Codex

## Scope

This ledger covers the trained renderer self-compression / Block-FP transplant
readiness lane only. It does not cover SJ-KL, Lane12, or mask-topology planner
work.

## Local Infrastructure Added

Implemented `experiments/preflight_trained_renderer_transplant.py` as a
fail-closed local preflight for trained JointFrameGenerator renderer exports.

The preflight:

- requires `--renderer-export` unless the operator explicitly passes
  `--allow-source-renderer-surrogate`;
- accepts only pickle-free renderer export magics `QZS3`, `MQZ1`, and `QBF1`;
- validates the decoded state dict against the Quantizr-faithful
  JointFrameGenerator template, including tensor count, keys, shapes, and
  finite floating values;
- repacks the renderer into deterministic QBF1 Block-FP candidates across a
  supplied block-size list;
- replaces only logical `renderer.bin` while preserving source `masks.mkv` and
  `optimized_poses.bin`;
- emits single-member packed archive candidates with score-affecting bytes
  charged inside `archive.zip`;
- verifies the runtime payload unpack path and CPU QBF1 load path;
- emits claim and Lightning exact-eval dry-run/submit command shapes; and
- records `score_claim=false`, `promotion_eligible=false`, and
  `remote_gpu_dispatch_performed=false`.

## Artifact

Generated local surrogate preflight artifact:

`experiments/results/trained_renderer_blockfp_preflight_20260502_codex/trained_renderer_blockfp_preflight.json`

Source archive:

`experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`

Source archive SHA-256:

`226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`

Surrogate mode:

`source_renderer_surrogate`

Best local QBF1 surrogate candidate by archive bytes:

- candidate: `trained_qbf1_b1024`
- archive:
  `experiments/results/trained_renderer_blockfp_preflight_20260502_codex/trained_qbf1_b1024/archive.zip`
- bytes: `283869`
- SHA-256:
  `6d331e479d961df22a2baa8b3f09722394ece7d0c194821c80c6aa354cb1449b`
- delta vs source archive: `+7655` bytes
- dispatchable: `false`

Interpretation:

The surrogate proves the archive/runtime plumbing but is intentionally not an
H100-ready trained transplant because the renderer bytes are the source
renderer. It is empirical infrastructure evidence only, not score evidence and
not a candidate to dispatch.

## Lightning Dry-Run

Validated the emitted H100 exact-eval command shape locally with:

`scripts/launch_lightning_batch_job.py exact-eval --dry-run`

Isolated dry-run state:

`experiments/results/trained_renderer_blockfp_preflight_20260502_codex/dry_run_lightning_state.json`

The dry-run inferred the expected archive identity for the surrogate best
candidate:

- expected bytes: `283869`
- expected SHA-256:
  `6d331e479d961df22a2baa8b3f09722394ece7d0c194821c80c6aa354cb1449b`
- machine: `H100`
- adjudication: enabled
- component trace: enabled
- remote dispatch: not performed

## Focused Tests

Passed:

- `.venv/bin/python -m pytest src/tac/tests/test_preflight_trained_renderer_transplant.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_build_blockfp_c067_archive.py src/tac/tests/test_plan_c067_renderer_self_compression_v2.py src/tac/tests/test_qbf1_renderer_codec.py src/tac/tests/test_quantizr_torch_fp4_codec.py -q`

## H100 Dispatch Gate

Do not dispatch from this surrogate artifact. H100 spend becomes warranted only
after a trained renderer export path is supplied and this preflight reports:

- `renderer_export.mode == "trained_renderer_export"`
- `renderer_export.same_as_source_renderer == false`
- `h100_lightning_readiness.ready == true`
- a current active dispatch claim exists for
  `c067_trained_renderer_self_compression_blockfp`
- a Lightning source manifest includes the candidate archive and inflate
  runtime closure

Command shape after a trained export passes the preflight:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id c067_trained_renderer_self_compression_blockfp \
  --platform lightning \
  --instance-job-id exact_eval_<candidate_id>_h100diag_<timestamp> \
  --agent codex:gpt-5 \
  --predicted-eta-utc <UTC_ETA> \
  --status eval \
  --notes "h100_diagnostic candidate=<candidate_id> archive_sha256=<archive_sha256>"

.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --job-name exact_eval_<candidate_id>_h100diag_<timestamp> \
  --archive <candidate_archive_inside_repo> \
  --repo-dir /teamspace/studios/this_studio/pact \
  --upstream-dir /teamspace/studios/this_studio/pact/upstream \
  --machine H100 \
  --adjudicate \
  --baseline-score 0.31561703078448233 \
  --baseline-archive-bytes 276214 \
  --predicted-band 0.0 10.0 \
  --regression-threshold 10.0 \
  --infer-expected-archive \
  --dispatch-lane-id c067_trained_renderer_self_compression_blockfp \
  --queue-metadata lane_id=c067_trained_renderer_self_compression_blockfp \
  --queue-metadata candidate_id=<candidate_id> \
  --queue-metadata source_archive_sha256=<source_archive_sha256> \
  --queue-metadata trained_renderer_sha256=<trained_renderer_sha256> \
  --queue-metadata purpose=trained_renderer_blockfp_h100_diagnostic \
  --component-trace \
  --component-trace-top-k 80 \
  --max-sane-score 10.0 \
  --studio "${LIGHTNING_STUDIO}" \
  --source-manifest "${LIGHTNING_SOURCE_MANIFEST_JSON}" \
  --remote-preflight-ssh-target "${LIGHTNING_PREFLIGHT_SSH_TARGET}"
```

## Export-Unlock Planner - 2026-05-02

Added local-only readiness planner:

`experiments/plan_trained_renderer_export_unlock.py`

Artifact:

`experiments/results/trained_renderer_export_unlock_20260502_codex/trained_renderer_export_unlock_plan.json`

Verdict:

`blocked_no_h100_dispatch`

The planner scanned the known trained-renderer, Block-FP, QBF, and
self-compression result directories and found no non-surrogate QZS3/MQZ1/QBF1
trained renderer export that passed
`experiments/preflight_trained_renderer_transplant.py`. It therefore emitted no
H100/Lightning submit command shape and performed no remote GPU dispatch.

Current blockers:

- existing preflight summaries are `source_renderer_surrogate`;
- best surrogate archive remains `283869` bytes, above C067's `276214` bytes;
- discovered checkpoints still require a pickle-free renderer export step;
- discovered legacy renderer bins use ASYM/FP4A, not the accepted transplant
  magics `QZS3`, `MQZ1`, or `QBF1`.

Byte targets from the C067 frontier formula:

- sub-0.30 with unchanged components requires archive bytes at or below
  `252760`, a `23454` byte saving vs C067;
- sub-0.24 with unchanged components requires archive bytes at or below
  `162650`, a `113564` byte saving vs C067;
- the current surrogate best at `283869` bytes still needs `31109` additional
  bytes for sub-0.30 or `121219` additional bytes for sub-0.24 if components
  are unchanged.

Focused verification:

- `.venv/bin/python -m py_compile experiments/plan_trained_renderer_export_unlock.py`
- `.venv/bin/python -m pytest src/tac/tests/test_plan_trained_renderer_export_unlock.py -q`
