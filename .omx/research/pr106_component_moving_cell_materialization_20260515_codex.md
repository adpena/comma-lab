# PR106 Component-Moving Cell Materialization - 2026-05-15

This is a byte-closed materialization ledger, not a score claim.

## Tooling Landed

- `tools/materialize_pr106_component_moving_cell_candidates.py`
- `src/tac/tests/test_materialize_pr106_component_moving_cell_candidates.py`

The tool consumes `.omx/research/pr106_component_moving_cells_20260515_codex.json`
and emits PR106 format-0x01 sidecar candidate archives for top single-cell and
prefix-bundle probes.  It validates source archive custody against the plan's
local PR106 payload SHA before writing archives.

## Materialized Artifacts

Command:

```bash
PYTHONPATH=src .venv/bin/python tools/materialize_pr106_component_moving_cell_candidates.py \
  --plan-json .omx/research/pr106_component_moving_cells_20260515_codex.json \
  --output-dir experiments/results/pr106_component_moving_cell_candidates_20260515_codex \
  --singles 3 \
  --prefixes 1,4,16
```

Output summary:

- `experiments/results/pr106_component_moving_cell_candidates_20260515_codex/materialization_summary.json`
- source archive: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- source archive SHA-256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- source member: `0.bin`
- source member SHA-256: `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`

Candidate archives:

| candidate | archive bytes | delta vs source | archive SHA-256 | proxy plan delta |
| --- | ---: | ---: | --- | ---: |
| `latent_sidecar_row545_candidate100` | 186272 | +33 | `8e437adf51825bedcfed51c9e4deeab9c20f75829d5c65e3afef4966dbc2f22d` | -0.007331343441071264 |
| `latent_sidecar_row518_candidate21` | 186275 | +36 | `c0114963de0d8881b8afaf8f5ff526df697593d31e36e507b68ef0c3d08d62bf` | -0.007073732166352026 |
| `latent_sidecar_row513_candidate65` | 186275 | +36 | `878997d44b097080af3eb5d8372b6375e28561c65f7699e14749b385e6e2997d` | -0.006888190357746832 |
| `prefix_top_1` | 186272 | +33 | `8e437adf51825bedcfed51c9e4deeab9c20f75829d5c65e3afef4966dbc2f22d` | -0.007331343441071264 |
| `prefix_top_4` | 186289 | +50 | `2e15b5d755c0faa9157aa8277bcc7401703296256504ca001879a04162cc3e3a` | -0.02815317391897103 |
| `prefix_top_16` | 186320 | +81 | `3432e83fc04a8d8354048d637b9ac6f97e445f2d9e0e2cc1c90f6fe97ab498f6` | -0.10675982503274524 |

`prefix_top_16` requests 16 ranked cells and applies 13 unique-pair
corrections; duplicate pair cells are recorded in the candidate manifest.

## Authority Boundary

All candidates keep:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Required next step before frontier language:

1. claim one selected lane with `tools/claim_lane_dispatch.py`;
2. run paired exact `[contest-CUDA]` and `[contest-CPU]` auth eval on the same
   archive/runtime;
3. recompute component fields and write a result-review packet.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_materialize_pr106_component_moving_cell_candidates.py \
  src/tac/tests/test_plan_pr106_component_moving_cells.py
# 7 passed

.venv/bin/ruff check \
  tools/materialize_pr106_component_moving_cell_candidates.py \
  src/tac/tests/test_materialize_pr106_component_moving_cell_candidates.py
# All checks passed
```
