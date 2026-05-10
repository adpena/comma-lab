# Preflight Check 126 dedup + runtime-proof red-team closure

Generated: 2026-05-10

## Scope

This ledger records a hardening pass after the main source-of-truth push.
No score claim and no dispatch were performed.

```yaml
score_claim: false
dispatch_attempted: false
target: developer_velocity_and_exact_eval_custody
```

## Red-team findings reviewed

Existing agent review reported that PR101 proxy runtime packets must not become
exact-eval-ready unless the runtime-consumption proof exists and proves the
patched runtime consumes the score-affecting constants. Current `main` already
contains that closure:

- `src/tac/optimizer/exact_readiness.py` validates
  `runtime_consumption_proof_required`, proof existence, schema, provenance,
  archive SHA, route proof, and false-authority fields.
- `tools/build_pr101_kaggle_proxy_runtime_packet.py` refuses hidden runtime
  sidecars and refuses `--force` unless the destination is a self-authored
  prior packet directory with only manifest-listed rebuildable files.

Focused proof:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_optimizer_exact_readiness.py \
  tests/test_build_pr101_kaggle_proxy_runtime_packet.py

25 passed in 0.28s
```

## New fix

Developer preflight then failed on Check #126:

```text
check_lane_pre_registered_before_work_starts:
  src/tac/optimizer/exact_readiness.py referenced unregistered lane_id
  'lane_id_missing' and
  'lane_dispatch_claim_required_before_gpu_or_remote_eval'
```

Adversarial classification: scanner false positive. These strings are exact
readiness blocker labels, not actual lane IDs. Registering fake lanes would
hide the bug and pollute the lane registry.

Patch:

- add blocker labels to `_LANE_ID_REFERENCE_BLOCKLIST`;
- scan each current file once even when multiple recent commits touched it;
- preserve all source commit labels in the violation message rather than
  rereading the same current file per commit source.

This keeps Check #126 strict while reducing repeated file reads and duplicate
violations.

## Verification

Focused Check #126/coherence regression suite:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_check_lane_pre_registered_before_work_starts.py \
  src/tac/tests/test_preflight_subagent_coherence.py

133 passed in 15.20s
```

Developer preflight with the normal 30-second DX budget active:

```text
/usr/bin/time -p .venv/bin/python -m tac.preflight \
  --timings-json .omx/research/artifacts/preflight_dev_timing_after_check126_dedup_20260510_codex.json

PREFLIGHT PASSED
real 8.08
user 6.33
sys 3.68
```

The timing profile reports:

```yaml
wall_elapsed_s: 7.625399
serial_elapsed_s: 4.063812
step_count: 23
failed_step_count: 0
timeout_s: 30.0
```

## Remaining optimization targets

The routine developer preflight is under budget but still has a visible
pre-step source-index prewarm cost. Future work should continue toward a
single indexed pass across source files, then vectorized/fanout queries over
that index:

1. expose `_prewarm_preflight_source_index` timing as an explicit recorded
   preflight step;
2. collapse remaining direct `Path.read_text()` broad scans onto
   `SourceIndex.files_containing_substrings`;
3. make cold/warm cache behavior explicit in `tools/profile_preflight_latency.py`;
4. only move to Rust/Zig once the Python oracle contract and golden query
   vectors are fixed, so native SIMD/Rayon ports can prove byte-for-byte
   equivalent scanner decisions.
