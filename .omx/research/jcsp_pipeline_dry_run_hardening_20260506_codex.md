# JCSP Pipeline Dry-Run Hardening - 2026-05-06 Codex

Evidence grade: empirical local guard/test
Score claim: false
Dispatch attempted: false
Remote/GPU jobs dispatched: false

## Scope

Focused hardening stayed inside the JCSP pipeline/runtime path:

- `experiments/pipeline.py`
- `src/tac/jcsp_stream_builder.py`
- focused JCSP tests

No archive bytes were written, no lane was claimed, no exact eval was run, and
no dispatchable JCSP archive path was implemented.

## Change

- `cfg.use_joint_codec_stack=True` now changes the effective
  weight-compression cache mode to `joint_codec_stack_dry_run`, so a stale
  `.done_compress_weights` marker from `fp4` cannot skip the JCSP fail-closed
  branch.
- Added a deterministic JSON score-marginal loader with duplicate-key
  rejection. Supported dry-run inputs are direct stream-name mappings,
  `score_marginals` wrappers, and JCSP stream-manifest rows.
- Added `jcsp_stream_source_dry_run_metadata(...)`, which loads score
  marginals, decomposes the model into JCSP stream specs, quantizes tensor
  metadata, and returns JSON-ready StreamSource metadata only.
- The pipeline branch with a present `jcsp_score_marginals_path` now runs that
  metadata-only dry run and still raises `NotImplementedError` before any ADMM,
  container build, archive write, lane claim, GPU dispatch, remote dispatch, or
  exact eval.

## Evidence

- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_jcsp_stream_builder.py
  src/tac/tests/test_pipeline_jcsp_gate.py -q` passed with 21 tests.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py
  src/tac/tests/test_jcsp_stream_builder.py
  src/tac/tests/test_pipeline_jcsp_gate.py -q` passed with 40 tests.
- [empirical:syntax] `.venv/bin/python -m py_compile
  experiments/pipeline.py src/tac/jcsp_stream_builder.py
  src/tac/tests/test_pipeline_jcsp_gate.py` passed.
- [empirical:local-lint] `.venv/bin/python -m ruff check
  src/tac/jcsp_stream_builder.py src/tac/tests/test_jcsp_stream_builder.py
  src/tac/tests/test_pipeline_jcsp_gate.py` passed.
- [empirical:diff-check] `git diff --check` passed.

## Ruff Note

Full-file ruff on `experiments/pipeline.py` still reports unrelated pre-existing
issues outside the JCSP diff, including old ambiguous alpha/gamma comments,
unused imports, and older broad style findings. Those were not bulk-fixed to
avoid unrelated pipeline churn.

## Remaining Blockers

- JCSP is still not dispatchable.
- No byte-closed `jcsp.bin` archive member is produced by the pipeline.
- No canonical `submissions/robust_current` inflate/runtime path consumes a
  JCSP member.
- The ADMM/sequential codec dispatch loop is intentionally not wired in
  `pipeline.py`.
- Exact CUDA auth eval is still missing for any stacked JCSP archive.
