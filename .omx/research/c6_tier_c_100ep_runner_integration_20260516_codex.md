# C6 100ep Tier C Runner Integration - 2026-05-16

## Purpose

Make the harvested C6 100ep IBPS1 archive discoverable from the canonical
real-scorer Tier C runner. This is class-discriminating evidence plumbing, not
a contest score claim.

## Patch

- Added `ibps1_c6_100ep_a10g_advisory` to
  `tools/run_tier_c_with_real_scorer.py::DEFAULT_CANDIDATES`.
- Archive path:
  `experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260515T100257Z__smoke__100ep_modal/harvested_artifacts/archive.zip`
- Grammar: `ibps1`
- Role: `ib_bottleneck_control_100ep_a10g_advisory`

## Evidence

Plan-only runner manifest at `/tmp/pact-tier-c-plan-c6-100ep/` confirmed:

- `archive_exists=true`
- `archive_bytes=224857`
- `archive_sha256=d6fa790cc1aa10315831cedb387b6274a941ce45cdb13c49f5766ab0ad69a492`
- `pair_capacity=600`
- `score_claim=false`
- `promotion_eligible=false`

Verification:

```bash
.venv/bin/ruff check tools/run_tier_c_with_real_scorer.py src/tac/tests/test_run_tier_c_with_real_scorer.py
PYTHONPATH=src:upstream .venv/bin/python -m pytest \
  src/tac/tests/test_run_tier_c_with_real_scorer.py \
  src/tac/tests/test_mdl_ablation_tier_c_ibps1.py \
  src/tac/tests/test_cathedral_autopilot_tier_c_and_composition.py \
  -q
```

Result: `89 passed`.

## Status

This closes the immediate discoverability gap where the default Tier C runner
only covered the stale C6 5ep control. The next unblocked action is executing
the real-scorer Tier C runner for the 100ep archive and then classifying the
result under the usual no-score-claim, pair-sampled CPU evidence axis.
