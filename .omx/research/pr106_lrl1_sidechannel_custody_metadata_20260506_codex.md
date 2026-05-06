# PR106 LRL1 Sidechannel Custody Metadata - 2026-05-06

This tranche aligns the PR106 LRL1 sister sidechannel with the y-shift custody
standard. The LRL1 zero-mode builder already round-tripped deterministically,
but its build metadata lacked archive SHA, source archive SHA, dispatch flags,
and explicit dispatch blockers.

## What Changed

`experiments/build_pr106_lrl1_sidechannel.py` now records:

- `manifest_schema=pr106_lrl1_sidechannel_build_metadata_v2`
- built archive SHA-256
- source archive path and SHA-256
- `score_claim=false`
- `dispatch_attempted=false`
- `remote_jobs_dispatched=false`
- `ready_for_exact_eval_dispatch=false`
- explicit blockers for real CUDA search, exact CUDA auth eval, runtime tree
  provenance, and lane claim requirements

## Evidence

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr106_lrl1_sidechannel.py \
  src/tac/tests/test_dispatch_dryrun_pr106_sidechannels.py
```

Result: `36 passed`.

```bash
.venv/bin/python tools/all_lanes_preflight.py --timings
```

Result: `ALL 23 PREFLIGHT CHECKS PASSED`.

## Status

- no score claim
- no dispatch attempted
- no remote provider state touched
- LRL1 remains a local wire-format/custody surface until real CUDA search and
  exact auth eval exist

This is a DX/custody hardening step for the PR106 sidechannel stack. It does
not claim a score improvement.
