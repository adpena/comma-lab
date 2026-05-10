# Phase 1 runtime custody + preflight speed greenup - 2026-05-10

<!-- generated_at: 2026-05-10T14:27:00Z -->
<!-- evidence_grade: local_dev_correctness; no dispatch; no score claim -->

## Verdict

The integrated patch is CLEAN for commit after focused tests, all-lanes
preflight, direct Catalog #146 runtime guard, py_compile, review policy, and
diff hygiene.

## Clean Files

- tools/all_lanes_preflight.py CLEAN
- tools/audit_untracked_source_artifacts.py CLEAN
- experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py CLEAN
- src/tac/phase1_packet_compiler.py CLEAN
- src/tac/preflight.py CLEAN
- src/tac/tests/test_all_lanes_preflight_timing_profile.py CLEAN
- src/tac/tests/test_audit_untracked_source_artifacts.py CLEAN
- src/tac/tests/test_build_phase1_packet_compiler.py CLEAN
- src/tac/tests/test_preflight_phase1_runtime_guard.py CLEAN
- tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py CLEAN

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_phase1_runtime_guard.py \
  src/tac/tests/test_build_phase1_packet_compiler.py \
  tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py
# 126 passed, 3 warnings

.venv/bin/python -m pytest \
  src/tac/tests/test_audit_untracked_source_artifacts.py \
  src/tac/tests/test_build_phase1_packet_compiler.py \
  tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py
# 144 passed, 3 warnings

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_phase1_runtime_guard.py \
  src/tac/tests/test_audit_untracked_source_artifacts.py \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py
# 26 passed

.venv/bin/python -m py_compile \
  tools/all_lanes_preflight.py \
  tools/audit_untracked_source_artifacts.py \
  experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py \
  src/tac/phase1_packet_compiler.py \
  src/tac/preflight.py
# passed

.venv/bin/python tools/all_lanes_preflight.py --timings
# ALL 29 PREFLIGHT CHECKS PASSED; wall=1.65s; real 1.68

.venv/bin/python - <<'PY'
from tac.preflight import check_phase1_trainer_runtime_emits_contest_compliant_inflate
print(check_phase1_trainer_runtime_emits_contest_compliant_inflate(strict=True, verbose=False))
PY
# []

git diff --check
# passed
```

## Score-Lowering Relevance

This is not a score claim. It removes a Phase 1/T1 promotion blocker by
ensuring future patched Modal/Vast/Kaggle/AWS/Azure/GCP runs consume exactly the
archive bytes supplied by contest auth eval, and keeps all-lanes preflight under
the operator's 30s wall-clock budget.
