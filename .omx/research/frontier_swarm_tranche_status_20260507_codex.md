---
title: Frontier swarm tranche status
date: 2026-05-07
author: codex
status: CONTROL LEDGER — no score claim, no GPU dispatch
---

# Frontier swarm tranche status

## Concrete progress this tranche

- q10 HNeRV low-level Brotli remains the selected shortest-wall-clock exact
  candidate:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/hnerv_lowlevel_exact_eval_packet.json`.
- Added a narrow `hnerv_rate_only_raw_equivalent_kkt_proof_v1` to the q10
  packet and manifest. It is valid only for static-ready, byte-negative,
  raw-equivalent, single-payload Brotli repacks with zero expected SegNet and
  PoseNet deltas.
- Rebuilt field-meta selection:
  `pr106_q10_151byte_brotli` now has `kkt_ready_for_field_planning=true`.
- The remaining q10 exact-dispatch blocker is the active Level-2 lane claim.
  Do not create that claim until the submit path can immediately run, because
  the current shell lacks the required Lightning environment.
- Preserved the Wave-Ω and PARADIGM-δεζ architecture blueprints as tracked
  control-plane state, then corrected over-strong score/byte arithmetic so they
  remain prediction-only until exact CUDA evidence exists.

## Current dispatch state

Selected candidate:

```text
candidate_id: pr106_q10_151byte_brotli
byte_delta: -151
expected_total_score_delta_rate_only: -0.00010054470192144788
selection_decision: needs_active_lane_claim_before_dispatch
kkt_ready_for_field_planning: true
ready_for_exact_eval_dispatch: false
blockers:
  - missing_active_lane_dispatch_claim
  - claim:active_dispatch_claim_missing
```

Lightning submit is not currently runnable from this shell. Missing env:

```text
LIGHTNING_SSH_TARGET
LIGHTNING_REMOTE_PACT
LIGHTNING_UPSTREAM_DIR
LIGHTNING_TEAMSPACE
LIGHTNING_STUDIO
LIGHTNING_SDK_USER
```

`gws` is installed, but it is Google Workspace CLI and is not a Lightning/GPU
submitter for this repo.

## Active swarm

- Worker Omega-1/SJ-KL owns `src/tac/sjkl_basis.py`,
  `experiments/build_sjkl_residual.py`, and focused SJ-KL tests.
- Research/adversarial council owns a dated `.omx/research` ledger comparing
  Wave-Ω, PARADIGM-δεζ, LA-pose, telescopic foveation, Fridrich/entropy coding,
  and categorical/openpilot/selfcomp priors against papers and code.

## Next comprehensive tranche

1. If Lightning env becomes available, create the q10 lane claim and submit the
   exact CUDA eval immediately; harvest with expected archive SHA
   `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7` and
   bytes `186088`.
2. Integrate Worker Omega-1/SJ-KL if green. The next real Wave-Ω blocker is
   removing scorer/Fisher stubs without scorer-at-inflate leakage.
3. Integrate the research adversarial ledger and convert top findings into
   implementable tasks, not more roadmap prose.
4. Move from q10's tiny rate-only exact eval to the larger frontier path:
   Omega-1 measurable archive, Omega-2 NeRV mask runtime branch/training
   harness, then δεζ CPU scaffolding only after exact q10/Wave-Ω evidence
   updates the priors.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_lowlevel_exact_eval_packet.py src/tac/tests/test_build_field_meta_dispatch_selection.py -q`
  passed: `29 passed`.
- `.venv/bin/python -m ruff check tools/build_hnerv_lowlevel_exact_eval_packet.py src/tac/tests/test_hnerv_lowlevel_exact_eval_packet.py`
  passed.
- `.venv/bin/python tools/all_lanes_preflight.py` passed all 23 checks.
