# Preflight performance: measured parallelization negative (2026-05-10)

## Summary

The developer preflight now passes under the operator's 30s crash threshold.
Normal cached developer preflight is ~1.8s; forced no-incremental-cache
developer preflight is ~8.6-9.9s on this macOS workstation.

A direct attempt to run the developer-scope checks through the existing
`_ParallelPreflightRunner` was measured and reverted. It was slower here:

- sequential forced cache miss: `real 9.82s`;
- parallel forced cache miss with source-index prewarm: `real 12.88s`;
- parallel forced cache miss without source-index prewarm: `real 13.05s`.

The result is a real negative, not a policy decision against parallelism. The
current checks share filesystem and source-index caches; broad thread-level
parallelism contends on the same source tree and increases user/sys time.

## Current perf state

- `python -m tac.preflight --scope dev --timeout-s 30`: passes.
- Hot clean developer cache profile: ~`1.788s`.
- Forced no-incremental-cache developer preflight after Check #137 fix:
  `real 8.64s`.
- Check #137 itself is cheap (~10ms direct); the earlier ~8s failure latency was
  sequencing before reaching the check, not the check's implementation cost.

## 2026-05-10 post-runtime-hardening dev-gate measurement

After the Phase 1 runtime/compiler hardening edits, the fast developer gate
remained below the operator's 30s crash threshold:

```bash
/usr/bin/time -p .venv/bin/python -m tac.preflight --scope dev --timeout-s 30
# PREFLIGHT PASSED
# real 11.20
# user 8.56
# sys 3.97
```

The current wall-clock bottleneck remains broad source scanning, not the new
runtime/compiler checks. Next optimization tranche should migrate the hottest
remaining broad scans to `SourceIndex` candidate filtering with false-negative
fixtures before considering a native Rust/Zig fact-index helper.

## 2026-05-10 timings-json landing

`python -m tac.preflight` now supports `--timings-json <path>` for both dev and
all/release scopes. The timing profile records wall time, serial check time,
per-check rows, hot steps, timeout budget, and failure details when a run fails.

Post-integration verification:

```bash
/usr/bin/time -p .venv/bin/python -m tac.preflight \
  --scope dev --timeout-s 30 \
  --timings-json experiments/results/preflight_dev_timing_20260510_codex.json
# PREFLIGHT PASSED
# real 8.22
```

Timing JSON summary:

- status: `passed`
- measured wall in JSON: `7.778647s`
- step count: `23`
- slow step count (>=0.5s): `5`
- current hottest checks:
  `check_public_pr_intake_clones_pristine`,
  `check_no_eval_roundtrip_false`,
  `check_no_bare_writes_to_shared_state`,
  `check_dispatch_cli_shell_hazards`,
  `check_authoritative_tag_requires_custody_metadata`.

## 2026-05-10 additional hardening: pipefail + tee + PIPESTATUS

Codex red-team round found a second shell result-loss class adjacent to the
older `pipefail + grep -q` trap: remote scripts using `cmd | tee log` and then
capturing `${PIPESTATUS[0]}` under `set -euo pipefail` can exit at the pipeline
before the capture/report block runs. That loses training/eval return codes,
logs, and score-custody manifests.

Permanent guard added:

- `tac.preflight.check_no_pipefail_tee_pipestatus_loss(strict=True)`.
- Wired into `preflight_all()` and `preflight_developer()`.
- Scanner masks heredocs and accepts explicit `set +e` or `set +o pipefail`
  guards before the result-capturing pipeline.

Backfill:

- Initial live scan found 231 violations across 100 shell dispatch scripts.
- All 231 were mechanically remediated by inserting `set +e` before the
  `tee`/`PIPESTATUS` pipeline and `set -e` immediately after the capture.
- `bash -n` passed across every modified `.sh` file.
- The guard now reports 0 violations across 179 shell files.

Adversarial review then found one unsafe remediation in
`scripts/remote_lane_mae_v.sh`: direct `${PIPESTATUS[0]}` use inside an `if`
condition consumed the volatile array and left `errexit` disabled on the
success path. The guard was tightened accordingly:

- A guarded `tee` pipeline must assign `${PIPESTATUS[...]}` to a variable
  immediately after the pipeline, before any `if`, `echo`, or other command.
- The next significant command must restore `set -e` (or `set -o pipefail`
  when that was the disabled mode).
- Split-pipe forms such as `cmd |\n  tee log` are scanned.
- `tee | tail` logging pipelines are allowed only when the full
  `${PIPESTATUS[@]}` array is captured immediately.

The stricter scanner found and fixed the remaining live issues in
`remote_lane_mae_v.sh` and `remote_lane_imp_c067_bridge.sh`; final live count:
0 violations across 179 shell files.

Related Check #126 DX fix: worktree scanning now checks only changed new-file
lines for tracked modified files. That preserves the before-work-starts rule for
new WIP lane IDs while avoiding false failures when a mechanical safety edit
touches a legacy script that already contained old unregistered lane strings.

This is DX and custody work with direct score-lowering relevance: failed train,
build, or auth-eval commands now reach the manifest/reporting blocks instead
of disappearing as opaque provider failures.

## Next correct optimization targets

1. Shrink clean-cache fingerprint scope. The hot cache still fingerprints a
   large public-PR/result surface; split source WIP fingerprints from public PR
   clone status so a cache hit can approach sub-second without hiding clone
   mutations.
2. Move `check_codebase_drift` candidate selection into `SourceIndex` so it does
   not open every launch-surface Python file before applying text predicates.
3. Port the SourceIndex fact extractor to a native Rust/Zig helper only after
   the Python policy boundary is stable. Native code should emit file facts and
   conformance vectors, not own contest policy.
4. Keep thread/process parallelism evidence-gated. If a future patch claims a
   wall-clock win, it must report sequential vs parallel forced-cache-miss
   timings before landing.

## Claim discipline

No score claim or custody claim is made here. This is DX velocity evidence only.

## 2026-05-10 Gate #25 Modal image audit hotspot reduction

`tools/audit_modal_image_build_order.py` was reduced from broad AST parsing of
every tracked Python file to a two-phase source-index-compatible scan:

1. keep the `git ls-files` tracked-file boundary;
2. read Python source once and prefilter for literal Modal mount methods
   `add_local_dir` / `add_local_file`;
3. AST-parse only the Modal candidate files that can affect the policy.

Verification:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_modal_image_build_order_audit.py \
  tests/test_preflight_source_index_equivalence.py
# 8 passed

.venv/bin/python tools/audit_modal_image_build_order.py --strict
# modal image build order: PASS (20 Modal candidate files; 2520 tracked Python files checked)
```

Worker timing:

- before: `3 loops, best of 3: 4.1 sec per loop`;
- after: `10 loops, best of 5: 179 msec per loop`.

Residual risk: the candidate filter is intentionally tied to the literal mount
method names used by the AST policy surface. Dynamic wrappers that hide those
method names remain outside this check, matching the previous practical policy
boundary rather than expanding it silently.

## 2026-05-10 SourceIndex reuse for dev preflight hot scans

Targeted next-cost scans from forced dev timing:

```bash
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python -m tac.preflight \
  --scope dev --timeout-s 30 --timings-json /tmp/pact_preflight_dev_before.json
# PREFLIGHT PASSED
# timing wall_elapsed_s=7.395472
# hot: check_codebase_drift=1.120s, check_public_pr_intake_clones_pristine=1.103s,
#      check_no_mps_fallback_default=0.832s, check_dispatch_cli_shell_hazards=0.770s
```

Implemented a safe Python SourceIndex speedup rather than native Rust/Zig:

- `check_codebase_drift` now reuses SourceTextFacts for its Python candidate
  prefilter instead of launching a redundant ripgrep regex scan inside an
  already-indexed preflight context.
- The fact prefilter mirrors `_python_forbidden_scan_may_match` and remains
  conservative around `/tmp/`; the precise AST/text scanner still owns policy.
- `check_training_scripts_use_real_data_in_nonsmoke_mode` now asks SourceIndex
  for `train_*.py` files containing `make_synthetic*` or `make_smoke*` before
  parsing, avoiding the large renderer trainers when they cannot affect the
  synthetic-data policy.
- Source facts schema bumped to v18 because the fact needle set changed.

Focused policy checks:

```bash
.venv/bin/python -m pytest -q tests/test_preflight_source_index_equivalence.py
# 8 passed in 0.76s
```

Forced dev preflight after the one-time v18 fact-cache rebuild:

```bash
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python -m tac.preflight \
  --scope dev --timeout-s 30 --timings-json /tmp/pact_preflight_dev_after_7.json
# PREFLIGHT PASSED
# timing wall_elapsed_s=7.466809
# codebase_drift scanned 172 source Python files, down from 372
# hot: check_public_pr_intake_clones_pristine=1.130s,
#      check_codebase_drift=1.046s,
#      check_no_mps_fallback_default=0.823s,
#      check_dispatch_cli_shell_hazards=0.779s
```

Normal cached dev preflight after storing the clean developer cache:

```bash
.venv/bin/python -m tac.preflight --scope dev --timeout-s 30 \
  --timings-json /tmp/pact_preflight_dev_cached_after_2.json
# PREFLIGHT PASSED
# cached clean developer preflight
# shell wall ~=1.72s
```

Incidental fail-closed hardening: expanding SourceIndex facts surfaced an
existing tokenization edge in `check_authoritative_tag_requires_custody_metadata`.
Bounded source windows that start mid-block can raise `IndentationError`; the
helper now retries a left-stripped token window rather than crashing or trusting
comments/strings. The same scan was narrowed to ignore `eval_axis ==
"[contest-CUDA]"` classification branches because they are not authoritative
evidence-tag promotion predicates.

Residual perf targets:

1. `check_public_pr_intake_clones_pristine` is the current hottest forced-cache
   dev check (~1.1s). The next safe target is a cheaper clone fingerprint that
   avoids walking every public PR worktree when `.git/index` and untracked
   indicators are unchanged.
2. `check_dispatch_cli_shell_hazards` still opens every docs/reports/tools
   text file. A future patch should move its tool-local `scan_paths` loop onto
   SourceIndex fact candidate filtering with fixture parity before changing the
   policy surface.
3. `check_no_mps_fallback_default` is already fused with eval-roundtrip checks;
   further speedup needs AST parse-count reduction or broader reuse of parsed
   candidate ASTs, not more thread-level parallelism.

Wire-in declaration:

- Sensitivity-map contribution: N/A - preflight DX/cache infrastructure only,
  no codec sensitivity or scored tensor changed.
- Pareto constraint: N/A - no archive candidate or rate/distortion bound
  changed.
- Bit-allocator hook: N/A - no per-tensor importance changed.
- Cathedral autopilot dispatch hook: N/A - this only reduces local dev-gate
  latency; no deployable candidate was produced.
- Continual-learning posterior update: N/A - no empirical score anchor.
- Probe-disambiguator: N/A - single policy-preserving interpretation; parity
  tests compare SourceIndex and no-index behavior.

## 2026-05-10 Public-PR and dispatch-hazard hot scan reduction

<!-- generated_at: 2026-05-10T13:45:00Z -->
<!-- evidence_grade: local_dev_correctness_and_performance; no dispatch; no score claim -->

Scope: extreme preflight wall-clock optimization with semantics preserved.

Changes:

- `tools/check_dispatch_cli_shell_hazards.py` now uses SourceIndex text facts
  as a conservative candidate filter when an active SourceIndex is available.
  The existing `scan_text(...)` validator still owns every policy decision.
- `src/tac/source_index.py` text facts schema bumped to v19 with the exact
  dispatch-hazard tokens needed by that candidate filter.
- `src/tac/preflight.py::_public_pr_intake_clone_roots` now discovers the
  documented public-PR intake layouts directly instead of recursively crawling
  every `experiments/results/**/_intake_*` directory. Live parity check against
  the previous recursive discovery found the same 62 clone roots.
- `tools/check_pr101_proxy_promotion_blocker.py` now routes its A1
  `[contest-CUDA]` anchor acceptance through
  `tac.continual_learning.ContestResult.validate_custody()` before treating the
  anchor as available. This fixed a strict Catalog #127 failure surfaced by the
  forced dev profile; no waiver was added.

Timing evidence:

```bash
PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python \
  tools/profile_preflight_latency.py --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_sourceindex_20260510_before_dev.json \
  --top 25 --fail-on-surface-failure
# before: 7.713s; public_pr=1.083s; dispatch_hazards=0.781s

PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 .venv/bin/python \
  tools/profile_preflight_latency.py --surface preflight-dev-cli \
  --json-out .omx/research/artifacts/preflight_sourceindex_20260510_after_dev_forced_retry2.json \
  --top 25 --fail-on-surface-failure
# after: 7.100s; public_pr=0.496s; dispatch_hazards=0.693s

.venv/bin/python tools/profile_preflight_latency.py --surface preflight-checks \
  --preflight-check check_public_pr_intake_clones_pristine \
  --preflight-check check_dispatch_cli_shell_hazards \
  --json-out .omx/research/artifacts/preflight_sourceindex_20260510_before_named_checks.json \
  --top 20 --fail-on-surface-failure
# before named checks: 2.348s; public_pr=1.369s; dispatch_hazards=0.620s

.venv/bin/python tools/profile_preflight_latency.py --surface preflight-checks \
  --preflight-check check_public_pr_intake_clones_pristine \
  --preflight-check check_dispatch_cli_shell_hazards \
  --json-out .omx/research/artifacts/preflight_sourceindex_20260510_after_named_checks_publicpr_fastdisc.json \
  --top 20 --fail-on-surface-failure
# after named checks: 1.672s; public_pr=0.743s; dispatch_hazards=0.573s

PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 /usr/bin/time -p \
  .venv/bin/python -m tac.preflight --scope dev --timeout-s 30 \
  --timings-json .omx/research/artifacts/preflight_sourceindex_20260510_after_cli_forced.json
# PREFLIGHT PASSED; real 7.84s
```

Focused verification:

```bash
.venv/bin/python -m py_compile \
  src/tac/preflight.py src/tac/source_index.py \
  tools/check_dispatch_cli_shell_hazards.py \
  tools/check_pr101_proxy_promotion_blocker.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_dispatch_cli_shell_hazards.py \
  src/tac/tests/test_source_index.py

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py \
  src/tac/tests/test_dispatch_cli_shell_hazards.py \
  src/tac/tests/test_source_index.py \
  tests/test_preflight_source_index_equivalence.py -q
# 62 passed in 0.91s

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_custody_validator_and_locked_writes.py \
  src/tac/tests/test_preflight_harden_2026_05_08_checks.py -q
# 63 passed in 2.88s
```

Residual top bottleneck after this landing:

1. `check_codebase_drift` remains the top forced dev step at about 1.04s.
2. `check_no_mps_fallback_default` remains second at about 0.80s.
3. `check_dispatch_cli_shell_hazards` is still a broad text-file surface at
   about 0.69s in full dev preflight, despite the candidate filter.

Wire-in declaration:

- Sensitivity-map contribution: N/A - preflight DX/cache infrastructure only,
  no codec sensitivity or scored tensor changed.
- Pareto constraint: N/A - no archive candidate or rate/distortion bound
  changed.
- Bit-allocator hook: N/A - no per-tensor importance changed.
- Cathedral autopilot dispatch hook: N/A - this only reduces local dev-gate
  latency; no deployable candidate was produced.
- Continual-learning posterior update: N/A - no empirical score anchor.
- Probe-disambiguator: N/A - single policy-preserving interpretation; parity
  tests compare SourceIndex and no-index behavior.
