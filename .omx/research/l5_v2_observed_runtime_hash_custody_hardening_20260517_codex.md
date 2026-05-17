# L5 v2 Observed Runtime Hash Custody Hardening

Date: 2026-05-17

## Finding

The L5/L5-v2 exact-eval intake path could treat planned runtime hashes as if
they were observed custody. `extract_runtime_tree_sha256()` intentionally keeps
legacy support for `expected_runtime_tree_sha256`, but L5 probe intake reused it
when populating evidence rows. TT5L side-info effect-curve intake also accepted
`expected_runtime_content_tree_sha256` for runtime-content pairing.

That made a source artifact with only an expected/planned runtime hash capable
of filling the observed `runtime_tree_sha256` or `runtime_content_tree_sha256`
slots before validation. This is not a score claim by itself, but it weakens
architecture-lock and paired-axis evidence discipline.

## Fix

- Added observed-only runtime hash extractors in `src/tac/exact_eval_custody.py`:
  - `extract_observed_runtime_tree_sha256`
  - `extract_observed_runtime_content_tree_sha256`
- Made the default `extract_runtime_tree_sha256()` custody-safe by delegating
  to observed-only extraction.
- Preserved the old broad behavior under the explicit
  `extract_runtime_tree_sha256_allow_expected()` name for non-promotional
  legacy audit surfaces.
- Added `extract_expected_runtime_tree_sha256()` so L5 can compare observed
  versus planned runtime hashes without treating planned hashes as custody.
  The expected extractor also reads `--expected-runtime-tree-sha256` from
  auth-eval command text because current Modal exact-eval artifacts often carry
  the planned hash there rather than in a structured field.
- Switched L5 probe intake to observed-only runtime tree extraction.
- Switched TT5L side-info effect-curve intake to observed-only runtime-content
  extraction.
- Tightened L5 probe `byte_closed_archive`: it now requires every selected
  exact axis to have a valid observed runtime hash.

## Regression Coverage

- `src/tac/tests/test_exact_eval_custody.py`
  - expected-only runtime tree/content hashes are rejected by observed-only
    extractors.
  - expected runtime hashes embedded in auth-eval command text are parsed only
    for mismatch checks.
- `src/tac/tests/test_l5_v2_probe_intake.py`
  - a TT5L exact-eval source with only `expected_runtime_tree_sha256` is
    recognized but fails custody with `runtime_tree_sha_invalid`.
  - a TT5L exact-eval source with observed/expected runtime mismatch fails
    custody with `runtime_tree_sha_mismatch`.
  - expected-only runtime evidence cannot set `byte_closed_archive=true`.
- `src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
  - a trained TT5L cell with only `expected_runtime_content_tree_sha256` fails
    runtime identity pairing.
  - a trained TT5L cell with observed/expected runtime-tree mismatch keeps an
    `exact_eval_runtime_tree_sha_mismatch` blocker.

## Validation

```text
.venv/bin/python -m pytest src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py -q
44 passed in 0.49s

.venv/bin/python -m py_compile src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py
pass

.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py -q
175 passed in 2.91s

.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py src/tac/tests/test_packetir_exact_closure.py -q
64 passed in 1.55s

.venv/bin/python -m ruff check src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py
All checks passed!

.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md
wrote intake artifacts; exited nonzero as expected because architecture lock remains blocked

.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md
architecture_lock_allowed=false; blockers=[requires_all_l5_v2_gate_evidence_valid, requires_c1_z5_tt5l_probe_gate_evidence, requires_paired_cpu_cuda_sideinfo_effect_curve]

.venv/bin/python tools/audit_research_state_tracking.py --repo-root . | rg -n "l5_v2_observed_runtime_hash_custody_hardening" -C 3
disposition=track_in_git; git_status=untracked; reason=Small research ledgers and structured summaries are durable lab knowledge.
```

## Current Status

This hardens custody interpretation only. It does not unblock architecture
lock, does not create a score claim, and does not mark any TT5L result
promotion-ready. The packet remains correctly fail-closed until paired
CPU/CUDA side-info effect-curve evidence and C1/Z5/TT5L probe gate evidence are
complete.
