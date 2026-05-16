# PR106 Format0D runtime consumption and paired plan (2026-05-16)

## Candidate

- archive: `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`
- bytes: `186876`
- sha256: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`

## Runtime-Consumption Proof

Generated:

`experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/runtime_consumption_format0d.json`

Command:

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --expected-archive-sha256 9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4 \
  --output-json experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/runtime_consumption_format0d.json
```

Result summary:

- blockers: `[]`
- format_id: `0x0D`
- parser/PacketIR consumed-byte accounting: `true`
- runtime all score-affecting sections consumed: `true`
- runtime corrected-latents digest changed: `true`
- runtime apply order:
  - `base_format0c_corrections`
  - `extra_pr101_ranked_no_op_corrections`
- runtime content tree sha256:
  `5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c`
- runtime source tree sha256:
  `373f19a1a892cf21c432d4949312cc788f4d4d23c02f2c1ca0cb3e666fc5c4bc`

This is not full-frame parity and not a score claim. The proof scope is
`actual_submission_inflate_py_sidecar_decode_and_apply_not_full_frame`.

## Paired Modal Plan

Generated plan only:

`experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_auth_eval_plan.json`

Command:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh \
  --label pr106_format0d_latent_score_table \
  --pair-group-id pair_pr106_format0d_latent_score_table_20260516 \
  --lane-id-base lane_pr106_format0d_latent_score_table_20260516 \
  --claim-agent codex:gpt-5.5 \
  --claim-notes "PR106 format0D byte-closed candidate; runtime-consumption proof green; proxy score-table boundary; paired axes required" \
  --expected-runtime-tree-sha256 5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c \
  --json-out experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_auth_eval_plan.json
```

Plan summary:

- schema: `modal_paired_auth_eval_dispatch_plan_v2`
- pair_group_id: `pair_pr106_format0d_latent_score_table_20260516`
- contest CUDA lane: `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`
- contest CPU lane: `lane_pr106_format0d_latent_score_table_20260516_contest_cpu`
- existing anchors reused: none
- score_claim: `false`
- promotion_eligible: `false`

## Next Decision

Format0D is now past the runtime-consumption blocker. Because it intentionally
changes corrected latents, same-runtime full-frame parity is not expected to be
the promotion proof. The next score-bearing action is the paired Modal exact eval
from the generated plan, after operator review of the plan or an explicit
autonomous spend decision.
