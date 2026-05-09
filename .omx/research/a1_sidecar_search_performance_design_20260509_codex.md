# A1 sidecar search performance design - 2026-05-09

<!-- generated_at: 2026-05-09T23:28:13Z, from_state_hash: local_a1_sidecar_profile_codex -->

research_only=true

## Scope

Owned files for this landing:

- `tools/build_a1_per_pair_latent_correction_sidecar.py`
- `src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py`
- `.omx/research/a1_sidecar_search_performance_design_20260509_codex.md`

No dispatch, no lane claim, no remote/GPU/GHA/eval work.

## Bottleneck inspection

The current proxy-MSE A1 sidecar search evaluates one base decode plus
`28 * 16 = 448` candidate latent perturbation decodes per pair. Local profile
on A1 pair 0, CPU, exact A1 archive SHA prefix `87ec7ca5f2f328a8`:

| candidate batch size | elapsed seconds | selected dim | selected delta_idx | scalar-match |
|---:|---:|---:|---:|---|
| 1 | 15.589 | 11 | 15 | true |
| 4 | 12.168 | 11 | 15 | true |

Earlier local sweep in the same session showed larger chunks were not reliably
faster on this CPU path: batch 16 ~23.7s, 64 ~22.6s, 128 ~21.7s, 449 ~18.6s
for the same pair. Blindly raising the default would risk slower runs and
would make exact search semantics harder to audit.

## Patch

Added `profile_pair_proxy_mse_candidate_batches(...)`:

- always inserts scalar `candidate_batch_size=1` as the semantic reference;
- profiles requested candidate batch sizes on one pair;
- marks a profiled size selectable only when best dim, best delta index, base
  MSE, and best MSE match the scalar reference within a tight numeric bound;
- selects the fastest semantic match, otherwise leaves scalar selected.

Added CLI controls:

- `--profile-candidate-batches N ...` records the profile in `search_meta`;
- `--auto-candidate-batch-size` uses the fastest scalar-matching profiled
  size, defaulting to `[1, 2, 4, 8, 16, 32, 64]` when no explicit list is
  provided.

Default behavior remains scalar `candidate_batch_size=1`. This preserves the
old search path unless the operator opts into profiling or auto-selection.

## Fail-Closed Readiness

This is not a score claim and not an exact-eval-ready result. It changes only
local proxy search ergonomics and metadata. Exact-eval dispatch remains blocked
by the existing manifest readiness gates: lane claim required before dispatch,
runtime smoke evidence required, full non-smoke sidecar coverage required,
member-section proof required, local runtime custody required, sidecar choice
state required, and no-op detector required.

Unified solver hook status:

- sensitivity-map contribution: N/A, profiler-only local tooling
- Pareto constraint: N/A, no new empirical anchor
- bit-allocator hook: N/A, no changed allocation policy
- cathedral autopilot dispatch hook: N/A, no dispatch and no default actuator
- continual-learning posterior update: N/A, no authoritative result
- probe-disambiguator: N/A, scalar-reference profiler is the arbitration probe

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
```

Result: `28 passed in 0.53s`.

Local profile command used direct module import only; it did not create a
submission packet or dispatch any job. Result:

```text
selected=4 reason=fastest_semantic_match
batch=1 elapsed=15.589 dim=11 delta_idx=15 match=true
batch=4 elapsed=12.168 dim=11 delta_idx=15 match=true
```

## Parent Adversarial Review

Codex parent reviewed the worker patch before promotion and removed the
timing-dependent unit-test assumption. The `fastest_semantic_match` test now
uses a deterministic `perf_counter` schedule plus a monkey-patched semantic
oracle, so the behavior under test is the selector logic rather than host
scheduler timing.

Final parent verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py -q
.venv/bin/python -m py_compile tools/build_a1_per_pair_latent_correction_sidecar.py
git diff --check -- tools/build_a1_per_pair_latent_correction_sidecar.py src/tac/tests/test_build_a1_per_pair_latent_correction_sidecar.py .omx/research/a1_sidecar_search_performance_design_20260509_codex.md .omx/research/a1_sidecar_resumable_search_state_20260509_codex.md reports/lane_maturity.md .omx/research/lane_registry_status_reconciliation_20260509_codex.md
```

Observed:

- `28 passed in 0.56s`
- `py_compile` emitted no findings
- `git diff --check` emitted no findings
