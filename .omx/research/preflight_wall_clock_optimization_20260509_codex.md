# Preflight Wall-Clock Optimization - 2026-05-09

<!-- generated_at: 2026-05-10T00:55:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

## Scope

Codex optimized the bounded developer preflight surface without changing
preflight semantics, dispatch gates, or exact-eval custody rules.

No remote job, GPU job, exact eval, or lane claim was launched.

## Negative Control: Naive Parallelism

A direct attempt to run the developer checks concurrently through the existing
`_ParallelPreflightRunner` was measured and rejected before promotion.

Command:

```bash
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  .venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_parallel_cold_codex_20260510T0040Z.json \
  --top 20 \
  --fail-on-surface-failure
```

Observed: `22.049s` wall-clock, worse than the serial cold profile. The broad
markdown/state scanners contended on the shared source index and file cache,
so this implementation was not landed.

## Landed Optimization

The landed change reduces repeated file opens instead of adding thread
contention:

- `check_codebase_drift` now reuses the process-local `SourceIndex` for Python
  and shell file discovery/text reads when the normal preflight context
  provides one.
- `check_authoritative_tag_requires_custody_metadata` now uses the shared
  source index to prefilter likely candidate files by custody-tag substrings.
- `check_continual_learning_writes_use_lock` now uses the shared source index
  to prefilter `save_posterior` candidates.
- `check_no_tag_only_custody_validation` now uses the shared source index to
  prefilter tag/grade predicate candidates.
- `src/tac/source_index.py` bumped the persistent text-facts schema to v7 and
  added hot custody/concurrency needles needed by these checks.

## Measurements

Baseline before the landed indexing patch:

- normal developer profile: `9.721s`;
- warm clean-cache developer profile: `0.568s`;
- incremental-cache-disabled serial profile: `10.977s`.

After the landed indexing patch:

- normal first run after source edit: `9.780s`;
- normal first run after this ledger edit: `9.309s`;
- warm clean-cache developer profile: `0.554s`;
- incremental-cache-disabled profile: `10.158s` / `10.019s` / `11.000s`
  across repeated local runs.

Hot-check movement from the comparable cache-disabled serial profiles:

- `check_authoritative_tag_requires_custody_metadata`: `1.144s` to `0.558s`;
- `check_continual_learning_writes_use_lock`: `0.400s` to `0.020s`;
- `check_no_tag_only_custody_validation`: `0.191s` to `0.156-0.198s`
  after source-index prefiltering;
- total cold wall-clock remains about `10s`, which is below the 30s DX crash
  budget but still leaves the next optimization target.

## Verification

```bash
.venv/bin/python -m py_compile src/tac/preflight.py src/tac/source_index.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_preflight_cli_timeout.py \
  src/tac/tests/test_preflight_meta_bugs.py -q
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  .venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_indexed_cold3_codex_20260510T0052Z.json \
  --top 20 \
  --fail-on-surface-failure
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_indexed_warm2_codex_20260510T0055Z.json \
  --top 20 \
  --fail-on-surface-failure
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_post_ledger_codex_20260510T0058Z.json \
  --top 10 \
  --fail-on-surface-failure
```

Observed:

- `333 passed in 12.47s`;
- cache-disabled developer profile passed at `10.158s`;
- warm developer profile passed at `0.554s`.
- post-ledger normal developer profile passed at `9.309s`.

## Remaining Work

The next preflight speedups should target:

- `check_public_pr_intake_clones_pristine`: cache clone roots and status
  inputs instead of recursively rediscovering public PR clones each run.
- `preflight_arity`: move Python launcher arity scans onto the same
  single-pass source-index query model.
- Per-check clean caching: avoid invalidating every developer check when a
  source edit cannot affect that check's candidate set.
- Native backend option: keep the Python `SourceIndex` contract stable so a
  Rust/Rayon or Zig scanner can replace only the inventory/facts backend and
  prove parity against the same Python checks.

## Codex Follow-Up: Shell-Lane Arity Target Filtering

<!-- generated_at: 2026-05-10T00:43:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex removed a cold-path waste class in `preflight_shell_lane_arity`: the
shell-lane checker now parses argparse signatures only for targets actually
invoked by `scripts/remote_lane_*.sh`, instead of walking every possible
`experiments/`, `scripts/`, and `src/tac/experiments` target. This preserves
the same fail-closed validation for each shell invocation while avoiding
unrelated target-parser churn.

Verification:

```bash
.venv/bin/python -m py_compile src/tac/preflight.py
.venv/bin/python -m pytest src/tac/tests/test_preflight_shell_lane_arity.py -q
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_arity.py \
  src/tac/tests/test_preflight_shell_lane_arity.py \
  src/tac/tests/test_profile_preflight_latency.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_shellarity_targetfilter_codex_20260510T0042Z.json \
  --top 10 \
  --fail-on-surface-failure
```

Observed:

- shell-lane focused suite: `14 passed in 0.93s`;
- combined arity/profile tests: `78 passed in 7.29s`;
- developer preflight profile: `8.718s` wall-clock;
- `preflight_shell_lane_arity` moved from `1.596s` in the immediately prior
  profile to `0.227s`;
- default developer preflight remains below the 30s crash budget.

Next target from the new profile:

- `check_public_pr_intake_clones_pristine` is now the top hot check
  (`1.452s`) and should cache clone-root discovery/status inputs without
  weakening dirty-clone fail-closed behavior.

## Codex Follow-Up: Python And Remote-Lane Arity Target Filtering

<!-- generated_at: 2026-05-10T00:58:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex extended the same invoked-target filter to the Python launcher arity path
and the older remote-lane argparse-arity check:

- `preflight_arity()` now scans launchers first, records invoked targets, and
  parses argparse signatures only for those targets on cache misses.
- `check_remote_lane_argparse_arity()` now reads each `remote_lane_*.sh` once,
  records target-path candidates, and parses only those target signatures
  before validating each invocation.

This preserves Rule A/B validation on every observed callsite while removing
unrelated argparse parsing from cold paths.

Verification:

```bash
.venv/bin/python -m py_compile src/tac/preflight.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_arity.py \
  src/tac/tests/test_preflight_shell_lane_arity.py -q
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_arity.py \
  src/tac/tests/test_preflight_shell_lane_arity.py \
  src/tac/tests/test_profile_preflight_latency.py \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_invoked_target_filters_codex_20260510T0050Z.json \
  --top 12 \
  --fail-on-surface-failure
```

Observed:

- arity focused suite: `69 passed in 7.99s`;
- combined focused suite: `108 passed in 7.56s`;
- developer preflight profile: `8.556s` wall-clock;
- `preflight_shell_lane_arity` stayed low at `0.237s`;
- default developer preflight remains below the 30s crash budget.

Remaining top hot checks from the new profile:

- `check_no_bare_writes_to_shared_state`: `1.333s`;
- `check_public_pr_intake_clones_pristine`: `1.324s`;
- `check_no_mps_fallback_default`: `1.209s`.

## Codex Follow-Up: Shared-State Marker Prefilter

<!-- generated_at: 2026-05-10T01:20:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex moved `check_no_bare_writes_to_shared_state()` onto the shared
`SourceIndex` substring candidate path when a source-index context is active.
The gate still runs the same line/AST logic on every candidate, and the raw
recursive scan remains the fallback when no source index exists.

Implementation:

- bumped `src/tac/source_index.py` text-facts schema to `v8`;
- added the shared-state marker strings to the text-facts needle set;
- used `SourceIndex.files_containing_substrings(..., require_all=False)` to
  select candidate Python files for catalog #131;
- added a regression test proving the gate uses the source-index substring
  index while still catching a bare shared-state write.

Verification:

```bash
.venv/bin/python -m py_compile src/tac/preflight.py src/tac/source_index.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_source_index.py -q
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_preflight_arity.py \
  src/tac/tests/test_preflight_shell_lane_arity.py \
  src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_shared_state_sourceindex_codex_20260510T0118Z.json \
  --top 12 \
  --fail-on-surface-failure
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_shared_state_sourceindex_warm_codex_20260510T0120Z.json \
  --top 12 \
  --fail-on-surface-failure
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  .venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out experiments/results/preflight_dev_profile_shared_state_sourceindex_cold_codex_20260510T0120Z.json \
  --top 12 \
  --fail-on-surface-failure
```

Observed:

- focused custody/source-index tests: `51 passed in 3.66s`;
- broader focused suite: `148 passed in 10.31s`;
- normal developer profile: `9.918s`;
- warm clean-cache developer profile: `0.601s`;
- cache-disabled developer profile: `10.480s`;
- target check improved from `1.333s` in the previous profile to `0.688s`
  normal / `0.582s` cache-disabled;
- total cold wall-clock remains around `10s`, still below the 30s crash
  budget, with current hot checks shifting to codebase drift, locked-write
  deletion preservation, dispatch shell hazards, and public PR clone status.

Next target per read-only red-team:

- add a conservative clean-status cache for
  `check_public_pr_intake_clones_pristine()`, caching only empty
  `git status --short` results and failing open to live `git status` on any
  fingerprint uncertainty.

## Codex Follow-Up: Public-PR Pristine Clean Cache + Whole-Cache Guard

<!-- generated_at: 2026-05-10T01:55:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex implemented the conservative Check 109 cache, but the first red-team pass
found the important recursive bug class: the whole `preflight_all` /
`preflight_developer` clean cache could skip Check 109 entirely because its
fingerprint intentionally excluded `experiments/results/**`.

Permanent fix:

- discover only public-PR intake clone roots under `experiments/results`;
- include those clone metadata paths in the whole-preflight clean-cache
  fingerprint, without admitting arbitrary result payloads into the source
  fingerprint;
- add a scanner-local cache for empty `git status --short` results only;
- invalidate scanner-local cache rows on tracked edits, untracked files,
  deletions, `.git` metadata changes, cache schema mismatch, or
  `PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1`;
- never cache dirty/nonzero/error outcomes; those continue down the existing
  LFS/diff classification path.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_all_clean_cache.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_all_clean_cache.py -q
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_preflight_arity.py \
  src/tac/tests/test_preflight_shell_lane_arity.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_public_pr_intake_clones_pristine \
  --json-out .omx/research/artifacts/preflight_public_pr_pristine_profile_20260509_after_cache_warm.json \
  --top 20
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260509_after_public_pr_cache.json \
  --top 20
```

Observed:

- public-PR/cache focused tests: `23 passed in 0.79s`;
- broader focused preflight tests: `108 passed in 9.75s`;
- `check_public_pr_intake_clones_pristine` in developer profile improved from
  `1.289s` to `0.800s` on the cached path;
- standalone public-PR check after cache: `1.465s` wall-clock;
- developer profile remains below the 30s crash budget, but cold total stayed
  noisy at `11.593s` because `check_codebase_drift`,
  `check_locked_writes_preserve_deletions`, and
  `check_no_mps_fallback_default` dominated the run.

Next perf targets:

- `check_locked_writes_preserve_deletions`: SourceIndex substring prefilter for
  `.unlink` / `.replace` / shared-state markers;
- `check_no_mps_fallback_default`: source-facts candidate narrowing before AST;
- `check_codebase_drift`: split cheap file inventory from full AST scanners so
  repeated developer preflight can use one index per opened file.

## Codex Follow-Up: Locked-Write Deletion-Preservation SourceIndex Path

<!-- generated_at: 2026-05-10T02:12:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex moved `check_locked_writes_preserve_deletions()` onto the shared
`SourceIndex` substring candidate path for the deletion-merge anti-pattern
markers. This preserves the existing precise line/window logic, but avoids a
fresh recursive read of every `.py` / `.sh` file when the developer preflight
already has a source index active.

Implementation:

- bumped `src/tac/source_index.py` text-facts schema to `v9`;
- added deletion-merge markers (`existing.update(`, `loaded.update(`, etc.) to
  the one-pass text-facts needle set;
- used `SourceIndex.files_containing_substrings(..., require_all=False)` for
  Check 132 candidate discovery;
- kept raw recursive scan as the no-source-index fallback;
- added a regression test proving Check 132 works inside a source-index
  context and builds substring index entries.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py src/tac/source_index.py \
  src/tac/tests/test_codex_round3_check_132_locked_writes_preserve_deletions.py
.venv/bin/python -m pytest \
  src/tac/tests/test_codex_round3_check_132_locked_writes_preserve_deletions.py \
  src/tac/tests/test_source_index.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_locked_writes_preserve_deletions \
  --json-out .omx/research/artifacts/preflight_locked_writes_profile_20260509_sourceindex.json \
  --top 20
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260509_after_locked_sourceindex.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused tests: `23 passed in 2.32s`;
- standalone cold Check 132: `2.218s` because it has to populate the v9
  text-facts cache;
- developer preflight Check 132: `0.151s`, down from `1.438s` in the previous
  profile;
- total developer preflight: `10.778s`, still under the 30s crash budget;
- new dominant checks: `check_authoritative_tag_requires_custody_metadata`
  (`2.152s`), `check_custody_gate_accept_tokens_concrete_only` (`1.457s`),
  and `check_no_mps_fallback_default` (`1.402s`).
