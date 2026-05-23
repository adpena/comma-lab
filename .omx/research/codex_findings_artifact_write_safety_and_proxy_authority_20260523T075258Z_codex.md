# Codex Findings - Artifact Write Safety And Proxy Authority

timestamp_utc: 2026-05-23T07:52:58Z
agent: codex
lane_id: lane_codex_artifact_write_safety_proxy_authority_20260523
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Scope

Adversarial hardening pass following the MLX queue normalized-objective guard
landing. This pass focused on two bug classes:

1. Large local queue/materialization artifacts could still clobber or partially
   overwrite prior state without an expected-hash contract.
2. Shared proxy/planning rows could preserve `score_claim_valid=True` unless a
   local consumer remembered to clear it.

## Landed Fixes

- Added guarded artifact writers in `tac.repo_io`:
  - `write_bytes_artifact`
  - `write_text_artifact`
  - `write_json_artifact`
  - `artifact_dir_transaction`
  - `tree_sha256`

  File writes are no-clobber by default, use same-directory temporary files,
  fsync data and parent metadata, return write metadata, and support a free-space
  floor. Overwrites require an expected existing file SHA. Directory writes stage
  into a `.partial` sibling and only replace an existing directory when an
  expected tree SHA matches immediately before commit.

- Wired guarded writes into:
  - DQS1 local-first harvest queue reroute writes.
  - DQS1 local-first queue builder `--write` output.
  - DQS1 autopilot append-only JSON artifact writes.
  - Staircase DAG and dispatch-plan outputs.
  - Decoder-q selective runtime materialization output directories.

- Removed the default `--force` from generated DQS1 local-first materialization
  steps. Existing materialization directories now fail closed unless a caller
  supplies an explicit expected output tree hash.

- Added `score_claim_valid: false` to the shared proxy evidence boundary so
  proxy/local/planning rows cannot preserve a truthy score-validity field while
  being normalized for spend triage or queue planning.

- Updated `.gitignore` for interrupted guarded directory transactions:
  `**/.*.partial.*` and `**/.*.backup.*`.

- Updated the checked-in DQS1 queue false-authority postcondition order to match
  the canonical eureka false-authority field order after `score_claim_valid`
  became a first-class proxy false-authority field.

## Verification

Focused pytest:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_repo_io.py \
  src/tac/tests/test_dqs1_local_first_queue_builder.py \
  src/tac/tests/test_staircase_dag.py \
  src/tac/tests/test_decoder_q_selective_runtime_materializer.py \
  src/tac/tests/test_optimizer_guided_candidate_generation.py \
  src/tac/tests/test_optimizer_candidate_queue.py
```

Result: `79 passed in 1.25s`.

Lint:

```bash
.venv/bin/ruff check <touched implementation, tool, and test files>
```

Result: `All checks passed!`.

Whitespace:

```bash
git diff --check
```

Result: clean.

CLI smokes:

```bash
.venv/bin/python tools/plan_staircase_dag.py machine-presets
.venv/bin/python tools/build_dqs1_local_first_queue.py --candidate-limit 1
```

Both completed successfully with output redirected to `/tmp`.

## Sidecar Audit Intake

Bacon artifact-safety audit confirmed the next directory-scale extensions:

- DQS1 locality-control raw work directories.
- Local CPU advisory auth-eval work directories.
- MLX scorer-input memmap/cache materialization.
- Experiment queue log paths and ledger-before-reroute ordering.

Kuhn local-signal guard audit confirmed the next normalized-objective consumers:

- OOF scorer-response training and validation gates.
- Cathedral distilled-scorer consumer.
- PACT-NeRV distilled scorer duplicate validator.
- Byte-shaving signal scorer-response references.
- PacketIR and cathedral macOS advisory loaders still need
  `score_claim_valid` refusal.

## Remaining Work

1. Apply `artifact_dir_transaction` to locality-control work dirs, auth-eval
   work dirs, and MLX scorer-input cache directories with explicit expected
   byte reservations.

2. Refactor scorer-response OOF/distillation consumers to use a single
   planning-value accessor that rejects MLX rows without normalized full-video
   objective fields and selects projected full-video delta for MLX rows.

3. Add consumer-specific tests for cathedral distilled scorer, PACT-NeRV
   distilled scorer, byte-shaving signal refs, PacketIR, and macOS advisory
   loaders.

4. Promote queue-reroute ordering to ledger-before-swap once the existing
   harvest/autopilot handoff is ready for a small transactional API.
