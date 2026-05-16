# PR106 component-moving prefix_top_1 runtime consumption (2026-05-16)

## Candidate

- archive:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/archive.zip`
- bytes: `186258`
- sha256:
  `ff90ed06afaa164b8fa838bfb2d4e21e520e4e6e605caf91876522bf0de922e5`
- runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`

## Proof

Generated ignored artifact:

`experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/runtime_consumption.json`

Command:

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --expected-archive-sha256 ff90ed06afaa164b8fa838bfb2d4e21e520e4e6e605caf91876522bf0de922e5 \
  --output-json experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/runtime_consumption.json
```

Result summary:

- blockers: `[]`
- format_id: `0x02`
- runtime all score-affecting sections consumed: `true`
- runtime corrected-latents digest changed: `true`
- PacketIR consumed-byte accounting passed: `true`
- sidecar payload SHA changed under mutation: `true`
- framing meta consumption probe changed runtime digest: `true`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Boundary

This proof shows the actual submission runtime decodes and applies the PR106
sidecar bytes; it is not full-frame parity and not a contest score. The next
score-bearing step is the paired CPU/CUDA exact eval plan, after comparing it
against the format0D and format0C queue priority.
