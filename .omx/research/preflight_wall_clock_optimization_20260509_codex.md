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
