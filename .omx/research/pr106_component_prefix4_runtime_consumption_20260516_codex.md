# PR106 component-moving prefix_top_4 runtime consumption (2026-05-16)

## Candidate

- archive:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/archive.zip`
- bytes: `186263`
- sha256:
  `63df794c0f06136c46415155fc9638bbc83950a793cf81b31171a6970b466ccd`
- runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`

## Proof

Generated ignored artifact:

`experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/runtime_consumption.json`

Command:

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --expected-archive-sha256 63df794c0f06136c46415155fc9638bbc83950a793cf81b31171a6970b466ccd \
  --output-json experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/runtime_consumption.json
```

Result summary:

- blockers: `[]`
- format_id: `0x02`
- runtime all score-affecting sections consumed: `true`
- runtime corrected-latents digest changed: `true`
- PacketIR consumed-byte accounting passed: `true`
- sidecar payload SHA changed under mutation: `true`
- framing meta mutation was runtime-consumed by rejection with invalid
  combinadic rank, which proves the runtime reads that framing field
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Boundary

This is local runtime sidecar-decode evidence only. It is not full-frame parity
and not a contest score. Because `prefix_top_16` was already exact-negative,
this row should remain a bounded A/B behind format0D and prefix_top_1 unless
paired-eval capacity is deliberately allocated to component-moving cells.
