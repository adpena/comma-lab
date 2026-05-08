# Source-manifest byte/SHA closure hardening - 2026-05-08

Scope: exact-eval custody and promotion hardening only. No GPU dispatch was
launched and no score artifact was changed.

Adversarial review found that the hardened PR104 replay staged the public
runtime dependency root, but promotion still lacked an explicit
source-manifest byte/SHA closure hash in the queue/adjudication surfaces. This
patch makes that closure machine-readable.

## Behavior added

- `scripts/launch_lightning_batch_job.py exact-eval` now validates that every
  required archive, runtime, external dependency-root, and relevant metadata
  artifact has integer `bytes` and hex `sha256` in the staged source manifest.
- Queue metadata now records `source_manifest_sha256`,
  `source_manifest_file_sha256`, and
  `source_manifest_runtime_closure_sha256` when a non-dry-run Studio exact eval
  uses a source manifest.
- `scripts/lightning_exact_eval_repro.py` now rejects stale
  `manifest_sha256` during staged-manifest consistency checks.
- `scripts/adjudicate_contest_auth_eval.py` now compares the harvested
  `inflate_runtime_manifest` against the recorded source manifest when
  available and gates promotion on byte/SHA closure mismatches.
- The adjudication provenance records the source-manifest identity, runtime
  closure hash, checked runtime files, external dependency roots, and closure
  violations.

## Evidence semantics

This hardening does not alter PR104's score. It changes how future harvested
exact-eval artifacts prove that the evaluated runtime files were byte/SHA-bound
to the submitted source manifest. A closure failure is a custody failure, not a
method result.

## Live PR104 check

After the patch, the hardened adjudicator was rerun over:

```text
experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.json
```

Result:

- `SOURCE_MANIFEST_CLOSURE_GATE_TRIGGERED=0`
- `PROMOTION_ELIGIBLE=1`
- `source_manifest_sha256`:
  `a284ea6c4977c532b5c912df0f95728ff5f974419515c2f7207514cd98bd7537`
- `source_manifest_runtime_closure_sha256`:
  `c87462131696b11ad966abf503794c4280c522ebf995efb9f7237aaf67ec85c3`
- Checked runtime files: `22`
- Checked external dependency roots: `1`

## Verification

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_public_replay_exact_eval_hardening.py \
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/python -m py_compile \
  scripts/adjudicate_contest_auth_eval.py \
  scripts/launch_lightning_batch_job.py \
  scripts/lightning_exact_eval_repro.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_public_replay_exact_eval_hardening.py \
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/ruff check \
  scripts/adjudicate_contest_auth_eval.py \
  scripts/launch_lightning_batch_job.py \
  scripts/lightning_exact_eval_repro.py \
  src/tac/tests/test_remote_auth_eval_hardening.py \
  src/tac/tests/test_public_replay_exact_eval_hardening.py \
  src/tac/tests/test_lightning_batch_jobs.py
```

Focused test result: `175 passed in 3.35s`; ruff and py_compile passed.

During review, the auth-eval hardening suite exposed a pre-existing
environment-coupled Vast.ai test that required `.venv/bin/vastai` even though
the test monkeypatched subprocess output. The test now points the launcher at a
temporary fake CLI path, preserving the intended label-matching assertion
without depending on local Vast.ai installation state.
