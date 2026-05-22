# Codex Findings: Decoder-Q Selective Window Bridge

Timestamp: 2026-05-22T13:33:41Z

## Scope

This pass turns the strict full-600 MLX decoder-q window selector into a
byte-closed work-order surface without granting MLX score, rank, promotion,
kill, or dispatch authority.

The packet-builder audit found that the current PR101 frame-exploit selector
grammar can select FES/FEC frame transforms per pair, but it cannot encode a
decoder-q tensor mutation arm such as `rgb_1.weight q_offset=0 delta=+1` only
for selected windows. The safe bridge is therefore a fail-closed runtime work
order, not an exact-eval dispatch packet.

## Code Landed

- `tac.optimization.decoder_q_selective_window_bridge` builds a strict bridge
  plan from `mlx_effective_spend_triage_candidate_selection.v1` plus a
  materialized `fec6_decoder_q_materialized_candidate_v1`.
- `tools/build_decoder_q_selective_window_bridge_plan.py` exposes the bridge
  as an operator CLI.
- `src/tac/tests/test_decoder_q_selective_window_bridge.py` covers false
  authority preservation, candidate archive SHA enforcement, source-window
  custody, run coalescing, and authority-leak rejection.

## Artifact Generated

Bridge artifact:

- JSON:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.json`
- JSON SHA-256:
  `f98e3017e8568bf36b304b5c053b6c7b4afc0e7ece8ef6d3f927849114765202`
- Markdown:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.md`
- Markdown SHA-256:
  `702ff84497bae42f8b4355b48e656e0a9509aed937a87c76ac1ad3862c93f338`

Command:

```bash
.venv/bin/python tools/build_decoder_q_selective_window_bridge_plan.py \
  --selection experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_effective_spend_triage_observed_window_selection_top32.json \
  --candidate-manifest experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_selected_candidates_20260520_codex/d1f1e56e042692f2/mutation_manifest.json \
  --lane-id lane_codex_decoderq_selective_window_bridge_20260522 \
  --max-windows 32 \
  --coalesce-gap 0 \
  --json-out experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.json \
  --md-out experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_window_bridge_plan_top32.md
```

## Empirical Inputs

- Strict MLX selector:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_effective_spend_triage_observed_window_selection_top32.json`
- Materialized decoder-q candidate:
  `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_selected_candidates_20260520_codex/d1f1e56e042692f2/mutation_manifest.json`
- Candidate archive SHA-256:
  `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`
- Mutation:
  `rgb_1.weight`, `q_offset=0`, `delta=+1`, `q_before=70`, `q_after=71`
- Source decoder SHA-256:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
- Mutated decoder SHA-256:
  `2b43984c32d5eef1d76e6f2b1051cb69c20781dec72afc8b5d3074c5f6ff00d6`

## Bridge Result

- Schema: `decoder_q_selective_window_bridge_plan.v1`
- Bridge status:
  `blocked_missing_decoder_q_selective_runtime_grammar`
- Selected windows: `32`
- Coalesced runs with `--coalesce-gap 0`: `32`
- Top window: `[501, 502]`
- Top observed MLX gain:
  `0.0020326847010743165 [macOS-MLX research-signal]`
- Top byte-budget margin: `3052.725643386161`
- OOF prediction agreement across selected units: `0/32`

## Authority

All bridge outputs remain `[macOS-MLX research-signal]` candidate-generation
inputs only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

Dispatch blockers recorded in the bridge plan:

- missing byte-closed selective decoder-q runtime grammar
- official `inflate.sh` raw-output locality controls not run for a selective
  packet
- claimed contest CPU/CUDA auth eval not run for a selective packet
- MLX window gains are candidate-generation signal only

## Verification

```bash
ruff check src/tac/optimization/decoder_q_selective_window_bridge.py \
  tools/build_decoder_q_selective_window_bridge_plan.py \
  src/tac/tests/test_decoder_q_selective_window_bridge.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_window_bridge.py -q
```

Result: `4 passed`.

## Next Engineering Step

Implement a byte-closed selective decoder-q runtime grammar that can apply the
materialized tensor-domain mutation only for selected pair windows, then
materialize singleton and small-run archives, run official inflate/raw-output
controls, and only then claim exact CPU/CUDA auth-eval dispatch.
