# PR106 component-moving paired Modal plans (2026-05-16)

## Scope

Generated plan-only paired Modal auth-eval manifests for the two component-moving
PR106 candidates whose local runtime-consumption proofs are green:

- `prefix_top_1`
- `prefix_top_4`

No paid eval was launched by this step.

## prefix_top_1 plan

- archive:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/archive.zip`
- bytes: `186258`
- sha256:
  `ff90ed06afaa164b8fa838bfb2d4e21e520e4e6e605caf91876522bf0de922e5`
- plan:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/paired_auth_eval_plan.json`
- pair group: `pair_pr106_component_prefix1_pr101grammar_20260516`
- CUDA lane: `lane_pr106_component_prefix1_pr101grammar_20260516_contest_cuda`
- CPU lane: `lane_pr106_component_prefix1_pr101grammar_20260516_contest_cpu`
- expected runtime tree sha256:
  `5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c`
- existing anchors reused: none
- score_claim: `false`
- promotion_eligible: `false`

Command:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh \
  --label pr106_component_prefix1_pr101grammar \
  --pair-group-id pair_pr106_component_prefix1_pr101grammar_20260516 \
  --lane-id-base lane_pr106_component_prefix1_pr101grammar_20260516 \
  --claim-agent codex:gpt-5.5 \
  --claim-notes "PR106 component-moving prefix_top_1; runtime-consumption proof green; prefix16 exact paired negative; paired axes required" \
  --expected-runtime-tree-sha256 5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c \
  --json-out experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_1/paired_auth_eval_plan.json
```

## prefix_top_4 plan

- archive:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/archive.zip`
- bytes: `186263`
- sha256:
  `63df794c0f06136c46415155fc9638bbc83950a793cf81b31171a6970b466ccd`
- plan:
  `experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/paired_auth_eval_plan.json`
- pair group: `pair_pr106_component_prefix4_pr101grammar_20260516`
- CUDA lane: `lane_pr106_component_prefix4_pr101grammar_20260516_contest_cuda`
- CPU lane: `lane_pr106_component_prefix4_pr101grammar_20260516_contest_cpu`
- expected runtime tree sha256:
  `5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c`
- existing anchors reused: none
- score_claim: `false`
- promotion_eligible: `false`

Command:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh \
  --label pr106_component_prefix4_pr101grammar \
  --pair-group-id pair_pr106_component_prefix4_pr101grammar_20260516 \
  --lane-id-base lane_pr106_component_prefix4_pr101grammar_20260516 \
  --claim-agent codex:gpt-5.5 \
  --claim-notes "PR106 component-moving prefix_top_4; runtime-consumption proof green; prefix16 exact paired negative; bounded A/B; paired axes required" \
  --expected-runtime-tree-sha256 5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c \
  --json-out experiments/results/pr106_component_moving_cell_candidates_pr101grammar_20260515_codex/prefix_top_4/paired_auth_eval_plan.json
```

## Priority

Execution priority remains:

1. format0D latent score-table materialized candidate;
2. `prefix_top_1` component-moving cell;
3. `prefix_top_4` only as a bounded A/B, because `prefix_top_16` already had a
   paired exact-negative.
