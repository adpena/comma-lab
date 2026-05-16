# L5 v2 TT5L Side-Info Proof Registry

Date: 2026-05-16
Author: Codex
Axis: TT5L / L5 v2 control-plane discovery
Evidence grade: source-and-test hardening; no score claim

## Finding

The L5 v2 readiness surface had a discoverability gap: the next TT5L action
asks the operator to materialize:

`experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_consumption_proof.json`

but `l5_v2_canonical_sideinfo_gate_evidence()` only looked at the older
`.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json` path. That
older artifact is a toy/local parser proof and is now semantically rejected by
the full-frame side-info gate.

Without this patch, producing the stronger contest full-frame proof would not
automatically update the L5 v2 readiness surface.

## Change

`src/tac/optimization/l5_staircase_v2.py` now treats side-info proof discovery
as a small ordered registry:

1. `experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_consumption_proof.json`
2. legacy `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json`

The contest full-frame artifact is preferred, uses its computed SHA-256, and
must pass the existing semantic blocker stack:

- contest full-frame proof scope;
- 600 pairs / 1200 frames;
- raw frame byte shape;
- file-list SHA;
- archive SHA pair;
- runtime tree SHA;
- inflate SHA pair;
- inflated output aggregate manifest;
- canonical `inflate.sh archive_dir output_dir file_list` command.

`l5_v2_dispatch_readiness()` now auto-loads this canonical side-info evidence
when no explicit gate evidence is supplied.

## Guard

`src/tac/tests/test_l5_staircase_v2.py` adds:

- direct discovery test for the contest full-frame artifact path;
- dispatch-readiness test proving Dykstra + auto-discovered side-info evidence
  sets `first_anchor_timing_smoke_allowed=true`.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
  - `66 passed`
- `ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py`
  - `All checks passed`

## No Score Claim

This is a control-plane discovery patch only. It does not create the proof,
does not run a timing smoke, and does not promote TT5L.
