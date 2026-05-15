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

## Runtime-Supported PR101 Grammar Recode - 2026-05-15

Fresh review found a materialization bug: `_archive_for_arrays()` always
emitted format `0x01` brotli sidecars even when the existing
runtime-supported PR101 grammar (`0x02`) was smaller for the sparse
component-moving corrections. This is a byte-closed compiler improvement, not a
score claim.

Code change:

- `tools/materialize_pr106_component_moving_cell_candidates.py` now evaluates
  `lossless_pr106_sidecar_recode_candidates(...)`, keeps only
  `runtime_decoder_implemented=true` candidates with concrete format IDs, and
  emits the smallest actual packet payload.
- The candidate manifest records the selected recode and all runtime-supported
  alternatives.
- `src/tac/tests/test_materialize_pr106_component_moving_cell_candidates.py`
  now asserts that sparse cell materialization selects format `0x02`.

Regenerated artifacts:

```bash
PYTHONPATH=src .venv/bin/python tools/materialize_pr106_component_moving_cell_candidates.py \
  --plan-json .omx/research/pr106_component_moving_cells_20260515_codex.json \
  --output-dir experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex \
  --singles 3 \
  --prefixes 1,4,16
```

Byte movement against the earlier format-0x01 candidates:

| candidate | old bytes | new bytes | byte delta | new archive SHA-256 |
| --- | ---: | ---: | ---: | --- |
| `latent_sidecar_row545_candidate100` | 186272 | 186258 | -14 | `ff90ed06afaa164b8fa838bfb2d4e21e520e4e6e605caf91876522bf0de922e5` |
| `prefix_top_4` | 186289 | 186263 | -26 | `63df794c0f06136c46415155fc9638bbc83950a793cf81b31171a6970b466ccd` |
| `prefix_top_16` | 186320 | 186278 | -42 | `4e9a10339cb6474ad1ca332cb1ddbd255d1577a18197ce541df4b8c189c12365` |

Runtime-consumption proof for the current strongest proxy candidate:

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_16/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --output-json experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_16/runtime_consumption.json
```

Result: `format_id=0x02`, `blockers=[]`,
`runtime_sidecar_decode_consumption_claim=true`,
`runtime_sidecar_apply_consumption_claim=true`, and
`runtime_all_score_affecting_sections_consumed=true`. This is local runtime
decode evidence only; exact paired `[contest-CUDA]` and `[contest-CPU]` evals
remain required before any score language.

## Paired Modal Auth Eval - 2026-05-15

The strongest local proxy candidate (`prefix_top_16`, archive SHA
`4e9a10339cb6474ad1ca332cb1ddbd255d1577a18197ce541df4b8c189c12365`,
`186278` bytes) was dispatched through paired Modal auth eval under pair group
`pr106_component_prefix16_pr101grammar_pair_20260515T191614Z`.

Runtime-tree custody nuance: Modal CUDA and Modal CPU wrappers compute
different uploaded-runtime tree hashes for the same `submission_dir` because
the wrappers mount different auxiliary runtime assets. CUDA accepted
`ae5cfc4d1ea1f7cd920623f44209e18147825c46f6698e6b3acad8325358864a`; CPU
accepted `b6b453a5c288c42d3069494f7cbea67e07219e2fb4b2f0bac36426ca861b22c4`.
Both use runtime content tree
`128604ad742deb46008fc312424801ac8a2e607c924266bdedaa763c059aaf72` and the
same archive SHA.

Recovered artifacts:

```text
[contest-CUDA]
call_id=fc-01KRPH3B57X94Q3Y024WYYEBN3
artifact=experiments/results/modal_auth_eval/pr106_component_prefix16_pr101grammar_paired_20260515T191614Z_cuda/contest_auth_eval.json
score_recomputed_from_components=0.20934621522786373
avg_segnet_dist=0.00067022
avg_posenet_dist=0.00003345
archive_size_bytes=186278
score_claim=true

[contest-CPU]
call_id=fc-01KRPH5KWP3WHXZHWAE6CFRP47
artifact=experiments/results/modal_auth_eval_cpu/pr106_component_prefix16_pr101grammar_paired_20260515T191614Z_cpu/contest_auth_eval.json
score_recomputed_from_components=0.23010997767094082
avg_segnet_dist=0.00065535
avg_posenet_dist=0.00016435
archive_size_bytes=186278
score_claim=false
```

Classification: legitimate paired negative for this component-moving candidate.
The runtime consumed the sidecar, but the component move damages the scorer more
than the byte savings help. This does not falsify PacketIR/format0C; it
falsifies this `prefix_top_16` sparse component-moving configuration as a
frontier candidate.
