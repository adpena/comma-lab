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

## Codex Follow-Up: Custody Accept-Token SourceIndex Path

<!-- generated_at: 2026-05-10T02:35:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex moved `check_custody_gate_accept_tokens_concrete_only()` onto the shared
`SourceIndex` candidate path. The precise AST validation is unchanged; only
candidate discovery changed from "read every Python file" to "files containing
validator/accept-list constant-name markers".

Implementation:

- bumped `src/tac/source_index.py` text-facts schema to `v10`;
- added accept-list marker substrings (`VALIDATOR_TOKENS`,
  `VALIDATOR_PATTERNS`, `ACCEPT_TOKENS`, `CUSTODY_TOKENS`, etc.) to the
  one-pass text-facts needle set;
- used `SourceIndex.files_containing_substrings(..., require_all=False)` for
  Check 136 candidate discovery;
- retained the raw recursive fallback when no source index is active;
- added a regression test for the source-index path.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py src/tac/source_index.py \
  src/tac/tests/test_check_136_custody_gate_accept_tokens_concrete_only.py
.venv/bin/python -m pytest \
  src/tac/tests/test_check_136_custody_gate_accept_tokens_concrete_only.py \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_source_index.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_custody_gate_accept_tokens_concrete_only \
  --json-out .omx/research/artifacts/preflight_check136_profile_20260509_sourceindex.json \
  --top 20
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260509_after_check136_sourceindex.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused tests: `70 passed in 4.56s`;
- standalone cold Check 136: `2.556s` because the v10 text-facts cache was
  populated;
- developer preflight Check 136: `0.353s`, down from `1.457s`;
- total developer preflight: `10.036s`, still below the 30s crash budget;
- remaining dominant checks: `check_authoritative_tag_requires_custody_metadata`
  (`2.209s`) and `check_no_mps_fallback_default` (`1.264s`).

## Codex Follow-Up: MPS Fallback Structural AST Check

<!-- generated_at: 2026-05-10T03:05:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no lane claim -->

Codex removed the remaining `ast.unparse()` dependency from
`_scan_python_for_mps_fallback()` and replaced it with structural AST detection
of `torch.cuda.is_available()` / `cuda.is_available()`. This keeps the same
forbidden pattern while avoiding repeated source reconstruction inside the AST
walk.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py src/tac/tests/test_preflight_meta_bugs.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_source_index.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_no_mps_fallback_default \
  --json-out .omx/research/artifacts/preflight_mps_profile_20260510_ast_structural.json \
  --top 20
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  .venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260510_mps_ast_cache_disabled.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused tests: `292 passed in 12.18s`;
- standalone cold MPS check: `1.627s`;
- cache-disabled developer preflight: `10.843s`;
- cache-disabled MPS check: `1.102s`;
- warm full developer preflight clean-cache path: `2.230s`.

The remaining cold-profile hotspots are now mostly broad shell/eval-roundtrip
scanners plus public-PR status when cache is disabled. The normal developer
loop should prefer the clean-cache path unless debugging cache invalidation.

## Codex Follow-Up: MPS Evidence Boundary

<!-- generated_at: 2026-05-10T03:20:00Z -->
<!-- evidence_grade: protocol_guardrail; no dispatch; no score claim -->

Operator reminder, made explicit for future preflight and score-lowering work:
MPS is advisory-only. It is useful for cheap sweeps, curve-shape measurement,
optimal-config identification, training-start sanity, and local implementation
triage. It is never auth-eval evidence, never promotion evidence, and never a
substitute for `[contest-CUDA]` or carefully labeled `[contest-CPU]` custody.

This reinforces the `check_no_mps_fallback_default` guard: MPS fallback is a
DX hazard when it silently replaces CUDA, and MPS-derived sidecar/training
signals must remain proxy/advisory until reproduced through the exact
archive/runtime path on the correct evidence axis.

## Codex Follow-Up: Public-PR Custody Fail-Closed + Check 127 Token Hardening

<!-- generated_at: 2026-05-10T03:35:00Z -->
<!-- evidence_grade: local_dev_correctness_and_performance; no dispatch; no score claim -->

Adversarial review found two preflight correctness hazards while profiling the
next hotspots:

- Check 109 treated `git status --short` timeout/nonzero results for discovered
  public-PR git clones as skipped/non-git. This is now fail-closed: a real git
  clone with unavailable status is a custody violation until git/status access
  is healthy.
- Check 109 clone discovery no longer hard-codes `pr91_src/repo`; it covers
  generic `pr*_src/repo` intake layouts so newer public frontier clones stay
  under the pristine-custody guard.
- Check 127's adjacent validator check used raw text, so comments/strings that
  merely mentioned `posterior_update` or `validate_custody` could whitelist a
  tag-only authoritative predicate. The check now scans executable tokens in
  the bounded code window, while preserving valid nearby calls even when the
  bounded window ends inside an open expression.

Verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_preflight_meta_bugs.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_public_pr_intake_clones_pristine \
  --preflight-check check_authoritative_tag_requires_custody_metadata \
  --json-out .omx/research/artifacts/preflight_check109_127_profile_20260510_failclosed_fix1.json \
  --top 20
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260510_failclosed_check109_127.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused public-PR/custody tests: `61 passed in 3.56s`;
- public-PR layout/fail-closed slice: `21 passed in 0.92s`;
- broader changed-check slice: `341 passed in 15.76s`;
- focused Check 109 + Check 127 profile: `2.948s PASSED`;
- Check 109 step: `1.592s`;
- Check 127 step: `0.642s`.
- full developer preflight after the fail-closed patch: `8.014s PASSED`;
  largest steps were MPS fallback (`1.255s`), public-PR pristine (`0.819s`),
  authoritative-tag custody (`0.651s`), bare-writes (`0.648s`), dispatch
  hazards (`0.580s`), and eval-roundtrip false (`0.412s`).

## Codex Follow-Up: MPS Alias Soundness + Fact-Filter Candidate Narrowing

<!-- generated_at: 2026-05-10T02:26:42Z -->
<!-- evidence_grade: local_dev_correctness_and_performance; no dispatch; no score claim -->

Fresh read-only review found a correctness gap in the hot MPS fallback check:
the candidate prefilter required the exact text `cuda.is_available`, so
`from torch import cuda as torch_cuda` and `getattr(torch, "cuda").is_available()`
forms could evade the AST detector. Codex fixed the detector by:

- adding alias binding for `from torch import cuda [as name]`;
- detecting `getattr(torch, "cuda").is_available()`;
- narrowing candidates with one shared text-facts pass over files containing
  `mps` plus one of `cuda.is_available`, `from torch import cuda`, or
  `getattr(torch`;
- bumping the persistent text-facts schema to `v12` so the new lexical facts
  are not read from stale cache rows.

Verification:

```bash
.venv/bin/python -m py_compile \
  tools/build_a1_per_pair_latent_correction_sidecar.py \
  src/tac/preflight.py src/tac/source_index.py \
  src/tac/tests/test_a1_sidecar_builder_hardening.py \
  src/tac/tests/test_preflight_meta_bugs.py
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_a1_sidecar_builder_hardening.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-checks \
  --preflight-check check_no_mps_fallback_default \
  --json-out .omx/research/artifacts/preflight_mps_profile_20260510_alias_factfilter.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused MPS/source-index/sidecar tests: `29 passed in 1.04s`;
- MPS check after alias/fact-filter patch: `1.482s` step time in the focused
  profile (`2.114s` surface).
- full developer preflight profile after the sidecar/MPS hardening patches:
  `7.728s PASSED`, with the largest steps now MPS fallback (`1.158s`),
  public-PR pristine (`0.715s`), authoritative-tag custody (`0.630s`),
  bare-writes (`0.585s`), dispatch hazards (`0.537s`), and eval-roundtrip
  false (`0.349s`).

This is slightly slower than the earlier exact-string candidate path, but it
removes a real false-negative class while staying far below the 30s crash
budget. Next speed target is to move dispatch-shell hazards onto the same
shared source-index/text-provider contract instead of importing and rescanning
files through the helper.

## Codex Follow-Up: Dispatch Shell Hazards Shared Source Index

<!-- generated_at: 2026-05-10T02:45:00Z -->
<!-- evidence_grade: local_dev_performance; no dispatch; no score claim -->

Codex wired `tools/check_dispatch_cli_shell_hazards.py` to accept an optional
`source_index` and updated `tac.preflight.check_dispatch_cli_shell_hazards()` to
pass the active shared source index. The helper still scans the same default
paths and exclusions; the change replaces private `os.walk` + `Path.read_text`
work with the shared file inventory/text cache when preflight is already inside
`source_index_context`.

Verification:

```bash
.venv/bin/python -m py_compile \
  tools/check_dispatch_cli_shell_hazards.py \
  src/tac/preflight.py \
  src/tac/tests/test_dispatch_cli_shell_hazards.py
.venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_cli_shell_hazards.py \
  src/tac/tests/test_preflight_meta_bugs.py::TestNoMpsFallbackDefault \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_a1_sidecar_builder_hardening.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260510_dispatch_hazards_sourceindex.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- focused dispatch/MPS/source-index/sidecar tests: `45 passed in 1.06s`;
- full developer preflight profile: `7.325s PASSED`;
- dispatch-shell hazards step in that full profile: `0.578s`.

The helper-specific profile is not a useful comparison because it now pays the
source-index setup cost by itself. The full developer surface is the relevant
DX target and improved from the prior `7.728s` profile to `7.325s` while
keeping all hazard semantics covered by the existing fixture set.

## Codex Follow-Up: Source-Facts Persistent Cache Identity Hardening

<!-- generated_at: 2026-05-10T03:05:00Z -->
<!-- evidence_grade: local_dev_correctness_and_performance; no dispatch; no score claim -->

Codex hardened the persistent `SourceIndex` text-facts cache against same-size
same-mtime rewrites by storing and validating `ctime_ns`, inode, and device in
addition to path, relpath, suffix, size, and mtime. This closes the stale-cache
false-negative class without changing scanner semantics. The text-facts schema
was bumped to `v13`, so the first profile after this patch rebuilds the cache.

Verification:

```bash
.venv/bin/python -m py_compile src/tac/source_index.py src/tac/tests/test_source_index.py
.venv/bin/python -m pytest src/tac/tests/test_source_index.py -q
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260510_sourceindex_v13_identity.json \
  --top 20 --fail-on-surface-failure
.venv/bin/python tools/profile_preflight_latency.py \
  --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_dev_profile_20260510_sourceindex_v13_warm.json \
  --top 20 --fail-on-surface-failure
```

Observed:

- source-index focused tests: `13 passed in 0.76s`;
- cold post-schema-bump developer preflight profile: `10.169s PASSED`;
- warm developer preflight clean-cache path: `1.984s PASSED`.

The cold one-time schema rebuild is still under the 30s operator crash budget.
The normal repeat developer loop now returns in about two seconds while using
stronger cache invalidation.

## Codex Follow-Up: SourceIndex Directory-Key Canonicalization

<!-- generated_at: 2026-05-10T03:42:00Z -->
<!-- evidence_grade: local_dev_performance_correctness; no dispatch; no score claim -->

Fresh read-only performance review found that `SourceIndex` keyed absolute
scan roots and repo-relative scan roots differently even when they named the
same directory. Hot preflight checks mix both styles, splitting file-list,
fact-group, and substring-index caches.

Codex changed `SourceIndex._dir_key()` so absolute paths under the index root
canonicalize to the same repo-relative key as relative paths. External absolute
paths remain absolute. Returned file paths are unchanged.

Verification:

```bash
.venv/bin/python -m py_compile src/tac/source_index.py src/tac/tests/test_source_index.py
.venv/bin/python -m pytest src/tac/tests/test_source_index.py -q
```

Observed: `15 passed in 0.77s`.

Follow-up profile should be run after the active A1 sidecar CPU builder exits,
so wall-clock numbers are not confounded by the decoder/MSE search.
