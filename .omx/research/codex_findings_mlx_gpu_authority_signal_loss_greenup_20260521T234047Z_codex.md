# Codex Findings: MLX GPU authority and signal-loss greenup

## Scope

Recursive adversarial review of the landed MLX port of the canonical auth
upstream contest scorer, with special focus on false authority, generated
artifact custody, `.gitignore`, and Markdown signal preservation.

## Findings and fixes

### HIGH: MLX GPU scorer-response path was too easy to invoke

Prior state: `build_mlx_scorer_response_payload(..., device_type="gpu")`,
`tools/run_mlx_scorer_response_cache.py --device gpu`, and
`tools/profile_mlx_scorer_response_cache.py --devices gpu` accepted GPU
scorer-response execution without an explicit research-only acknowledgement.
This contradicted the FEC6 GPU-profile finding: GPU rows were internally
repeat-stable but not CPU-transfer-stable against the MLX CPU baseline.

Fix landed in this pass:

- Added `GPU_RESEARCH_SIGNAL_BLOCKER` in
  `src/tac/local_acceleration/mlx_scorer_response.py`.
- `device_type="gpu"` now fails closed unless
  `allow_gpu_research_signal=True`.
- CLI GPU runs now require `--allow-gpu-research-signal`.
- Profile payloads record `gpu_research_signal_allowed`.
- Tests cover direct API rejection, CLI rejection, profiler rejection, and
  explicit profiler allowance propagation.

Historical note: older memos such as
`codex_findings_mlx_gpu_profile_stability_fec6_20260521T234700Z_codex.md`
contain now-stale rerun commands with `--devices gpu` and no explicit
allowance flag. They are preserved append-only as historical provenance; future
reruns must add `--allow-gpu-research-signal`.

### MEDIUM: generated MLX cache/profile roots needed explicit ignore coverage

Current state at review start: `git ls-files --others --exclude-standard`
returned no paths, so there was no live untracked signal to rescue.

Fix landed in this pass:

- `.gitignore` now explicitly covers ad hoc root/report MLX scorer cache,
  response, and profile directories:
  - `mlx_scorer_input_cache*/`
  - `mlx_scorer_response*/`
  - `mlx_scorer_response_profile*/`
  - `reports/mlx_scorer_input_cache*/`
  - `reports/mlx_scorer_response*/`
  - `reports/mlx_scorer_response_profile*/`

Durable signal policy remains: raw tensor caches and profile outputs stay local
unless intentionally curated; compact findings belong in `.omx/research/` or
tracked manifests.

### LOW: unrelated dirty partner work remains untouched

Unrelated dirty files observed and intentionally left unstaged:

- `.omx/state/modal_call_id_ledger.jsonl`
- `experiments/results/_modal_harvest_summary.json`
- `reports/cathedral_autopilot_evidence.jsonl`
- `tools/build_hfv1_sparse_sidecar_candidate.py`

No attempt was made to revert, overwrite, or bulk-stage those files.

## Verification

Narrow greenup:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_profile_stability.py -q
```

Result: `16 passed in 3.27s`.

Recursive MLX local-acceleration suite:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
```

Result: `83 passed in 16.78s`.

## Authority status

All MLX scorer-response and profile outputs remain local research signal only.
They are not auth-eval scores, not contest-axis results, not rank/kill evidence,
and not promotion evidence without paired contest CPU/CUDA evaluator custody.

