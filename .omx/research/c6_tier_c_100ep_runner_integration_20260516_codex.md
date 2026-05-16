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

## 64-Pair Real-Scorer Probe

Executed locally after the runner integration:

```bash
PYTHONPATH=src:upstream .venv/bin/python tools/run_tier_c_with_real_scorer.py \
  --execute \
  --output-dir experiments/results/tier_c_real_scorer_c6_100ep_codex_20260516T140014Z \
  --pair-samples 64 \
  --scorer-batch-size 4 \
  --archive ibps1_c6_100ep_a10g_advisory=experiments/results/lane_substrate_c6_e4_mdl_ibps_modal_t4_dispatch_20260515T100257Z__smoke__100ep_modal/harvested_artifacts/archive.zip,grammar=ibps1,role=ib_bottleneck_control_100ep_a10g_advisory
```

Result artifact directory:
`experiments/results/tier_c_real_scorer_c6_100ep_codex_20260516T140014Z/`

Key fields:

- evidence_axis: `[real-scorer CPU Tier-C delta curves; pair-sampled; no score claim]`
- hardware_axis: `[macOS-CPU advisory only]`
- score_claim: `false`
- promotion_eligible: `false`
- pair_samples: `64`
- elapsed_seconds: `1364.653979063034`
- baseline_seg: `0.2509330874308944`
- baseline_pose: `0.16412035573739558`
- mdl_tier_c_density_estimate: `0.6107912711463205`
- mdl_tier_c_substrate_class_verdict: `indeterminate`
- mdl_tier_c_curve_knee_signal: `1.2657731618153776`
- mdl_tier_c_latent_sigma1_delta: `0.6153786550257436`

Classification:

- This does not promote C6, retire C6, or claim a contest score.
- The 100ep C6 archive is now real-scorer-probed, but the Tier C density remains
  too high for a clean across-class verdict under this 64-pair macOS CPU advisory
  probe.
- The measured runtime is about `21.32s/pair` for this local CPU path, so a full
  600-pair local sweep should be budgeted at roughly `3.55h` for one archive on
  this machine before overhead.

Next action:

- Run the same 100ep C6 archive under the contest-compliant Linux CPU or CUDA
  axis before using the result to rank, kill, promote, or schedule paid full
  training.
