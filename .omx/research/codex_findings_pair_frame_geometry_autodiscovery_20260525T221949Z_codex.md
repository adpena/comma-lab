<!-- SPDX-License-Identifier: MIT -->
---
schema: codex_findings_v1
topic: pair_frame_geometry_autodiscovery
created_at_utc: 2026-05-25T22:19:49Z
author: codex
lane_id: lane_codex_pair_frame_geometry_autodiscovery_20260525
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
research_only: false
---

# Pair-Frame Geometry Autodiscovery

## Finding

Typed pair-frame geometry lattices existed in `.omx/research`, but the default
frontier feedback cycle only discovered them when a caller manually supplied a
`--pair-frame-geometry-lattice` path or explicit frontier artifact root. Fresh
cycle artifacts therefore showed `pair_frame_geometry_discovery` at zero while
queue-executable low-impact drop-many starts were already available.

## Landing

`discover_pair_frame_geometry_queue_requests` now uses `.omx/research` as its
bounded default discovery root, matching the DQS1 observation discovery pattern.
Explicit paths and explicit roots are preserved. Missing default roots are
treated as empty discovery so temp repos and early bootstraps remain usable.

## Empirical Artifact

The refreshed cycle at
`.omx/research/codex_frontier_rate_attack_pair_frame_autodiscovery_20260525T221949Z/`
proves the bridge is live:

- scanned pair-frame geometry lattice paths: 2
- discovered queue-executable lattice count: 2
- queue-executable pair-frame requests: 12
- selected candidates: 4 geometry low-impact drop-many starts
- generated queue: 4 experiments, 28 steps, valid
- drop-many greedy verdict model: still active

Selected candidates:

- `pairset_geometry_lowimpact_k003_h9dfec80f80`
- `pairset_geometry_lowimpact_k004_h3d2fc811a7`
- `pairset_geometry_lowimpact_k006_h7ce3073a1a`
- `pairset_geometry_lowimpact_k008_hc4a555f7ab`

## Authority Boundary

The autodiscovery only forwards typed
`pair_frame_geometry_queue_executable_drop_request.v1` rows with false-authority
fields. It does not claim score, promotion, rank/kill, or exact-eval dispatch
readiness. The selected rows remain local DQS1 materialization/control starts
that must earn authority through the normal advisory, exact-readiness, and
contest CPU/CUDA gates.

## Verification

- `ruff` on touched frontier feedback files: clean
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 14 passed
- `tools/experiment_queue.py --queue .omx/research/codex_frontier_rate_attack_pair_frame_autodiscovery_20260525T221949Z/initial_refresh/dqs1_followup_queue.json validate`: valid, 4 experiments, 28 steps
