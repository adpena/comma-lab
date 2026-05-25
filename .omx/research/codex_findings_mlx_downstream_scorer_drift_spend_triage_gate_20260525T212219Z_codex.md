# Codex Findings - MLX Downstream Scorer Drift Spend-Triage Gate

- Created: 2026-05-25T21:22:19Z
- Agent: Codex
- Lane: `lane_mlx_downstream_scorer_drift_spend_triage_gate_20260525`
- Scope: PR95/HNeRV MLX full-decoder downstream scorer-drift signal, MLX production contract, scorer-response spend triage.

## Finding

The PR95 MLX full-decoder downstream scorer-drift measurement had landed as a useful
artifact, but the strict MLX scorer-response production contract and effective spend
triage gate did not yet require it. That left a structural gap: an MLX scorer-response
row could clear spend triage with Conv2d accumulation evidence while the full decoder
and contest-uint8 scorer boundary remained a side memo.

## Landing

This pass promotes downstream scorer drift into the production contract:

- `tac.local_acceleration.mlx_production_contract` now accepts an optional
  `downstream_scorer_drift` manifest and can require it via
  `require_downstream_scorer_drift`.
- The gate is fail-closed on schema, verdict, contest-uint8 input mode, MLX axis tag,
  finite drift units, positive pair count, no dispatch/GPU launch, and explicit
  false-authority boundary blockers.
- Post-audit hardening additionally binds the drift proof to the response archive
  SHA-256, PoseNet SHA-256, SegNet SHA-256, trained-archive checkpoint mode,
  nonnegative drift units, and a minimum 100-pair coverage window that must cover
  the scorer-response pair window.
- The production contract schema/gate-set version was bumped to prevent stale
  no-downstream-gate contracts from remaining green inside bundle artifacts.
- `tools/check_mlx_scorer_production_contract.py` exposes
  `--downstream-scorer-drift` and `--require-downstream-scorer-drift`.
- `tac.optimization.scorer_response_dataset` now requires the downstream gate for
  strict effective MLX spend triage and routes missing evidence to a dedicated next
  probe before CPU-harvest follow-up.
- `tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py` now emits
  the canonical `schema_version` field expected by optional production gates.

## Authority Boundary

This is still `[macOS-MLX research-signal]` evidence. It can select local follow-up
and exact-eval spend candidates only after the full production contract and
calibration gates pass. It cannot claim score, promote, rank/kill, or skip paired
contest CPU/CUDA auth evaluation.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_production_contract.py tools/check_mlx_scorer_production_contract.py tools/plan_ll_scorer_response_next.py tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_production_contract.py -q`
  - `43 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py -q`
  - `106 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_scorer_torch_parity.py -q`
  - `36 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_scorer_torch_parity.py -q`
  - `185 passed`
- `.venv/bin/python tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py --help`
- `.venv/bin/python tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py --n-pairs 1 --start-pair 0 --scorer-input-mode contest_uint8 --output-json .omx/tmp/mlx_downstream_gate_fresh/results.json`
  - Fresh one-pair artifact emits canonical evidence grade/tag and is correctly
    refused as production coverage by the downstream gate.

## Remaining Work

- Run the downstream drift tool with at least 100 covered pairs, or a full-video
  600-pair sweep, on the current PR95/HNeRV reproduction artifact once the next
  timing/parity smoke refreshes the source archive/runtime pair.
- Feed the resulting strict production contract into the queue-owned grouped search
  surface so MLX can accelerate candidate selection while exact CPU/CUDA remains the
  authority boundary.
- Extend the same downstream-consumption gate shape to non-HNeRV materializers where
  full-frame receiver/scorer parity is the decisive proof rather than payload-parser
  parity.
