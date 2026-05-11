# Kaggle check tempdir cleanup hardening (2026-05-11)

## Scope

This hardening is score-lowering infrastructure, not score evidence. It fixes a
false-failure class in the free Kaggle proxy status path so completed proxy
sweeps can be harvested without masking useful optimizer signal.

## Bug class

`scripts/kaggle_check.py` downloaded kernel logs into a temporary directory.
When Kaggle output fetching hit timeout/error paths, the Kaggle CLI could leave
nested output behind while `TemporaryDirectory` cleanup was unwinding. Python
then raised `OSError: [Errno 66] Directory not empty`, which converted a useful
status/log harvest into a script failure.

## Fix

`get_kernel_log()` now uses `TemporaryDirectory(ignore_cleanup_errors=True)`.
The actionable signal is the kernel status/log content; failed scratch cleanup
must not be promoted to a Kaggle run failure.

## Score-lowering boundary

- Kaggle remains proxy/config-search only.
- MPS remains proxy/config-search only.
- Exact score movement still requires a byte-closed packet, runtime-consumption
  proof, dispatch claim, and contest auth eval on the correct CPU/CUDA axis.
- Existing PR101 Kaggle/CMA-ES exact CUDA evidence remains negative relative to
  the active CUDA floor and must not be redispatched as a score-lowering
  candidate without a new charged-byte transform or score-affecting runtime
  identity.

## Verification

- `.venv/bin/python -m pytest -q tests/test_kaggle_check.py`
- `.venv/bin/python scripts/kaggle_check.py --only-kernel adpena/pr101-bias-refine --only-kernel adpena/pr101-proxy-sweep --status-timeout-s 10 --log-timeout-s 10`
