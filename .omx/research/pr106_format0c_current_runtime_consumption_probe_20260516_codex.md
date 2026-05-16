# PR106 Format 0x0C Current-Runtime Consumption Probe - 2026-05-16

## Context

The PR106 PacketIR candidate matrix is currently fail-closed because exact
CPU/CUDA artifacts and runtime-consumption artifacts bind older runtime content
hashes, while the current Modal-uploaded runtime content hash is:

`8790ec81e5153a8fe3cb250e82b522763ae82b052b48655556be94acb05d5d51`

Before spending on paired exact eval, I ran the local parser/decoder
consumption proof against the current
`submissions/pr106_latent_sidecar_r2_pr101_grammar` runtime for the strongest
current PacketIR candidate, `format_0x0c_exact_radix`.

## Command

```bash
.venv/bin/python tools/prove_pr106_sidecar_runtime_consumption.py \
  --archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --runtime-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --expected-archive-sha256 56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7 \
  --output-json experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/runtime_consumption_format0c_current_runtime_20260516.json
```

## Result

Artifact path:
`experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/runtime_consumption_format0c_current_runtime_20260516.json`

Summary:

```json
{
  "blockers": [],
  "packet_ir_consumed_byte_accounting_passed": true,
  "runtime_all_score_affecting_sections_consumed": true,
  "runtime_sidecar_decode_consumption_claim": true,
  "runtime_sidecar_apply_consumption_claim": true,
  "sidecar_kind": "pr101_ranked_no_op_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided"
}
```

Current local runtime source/content hashes recorded by the proof:

- `runtime_source_tree_sha256`:
  `373f19a1a892cf21c432d4949312cc788f4d4d23c02f2c1ca0cb3e666fc5c4bc`
- `runtime_content_tree_sha256`:
  `5e66742426af649623c6dce7144914b8fc5993183ce053e58fe6646e8b81e48c`

## Classification

This is a real positive local runtime-consumption proof for the current
runtime, but it does **not** reopen the PR106 PacketIR matrix for L5-v2 stack
selection by itself.

Reason: paired exact CPU/CUDA artifacts for `format_0x0c_exact_radix` still bind
the older Modal runtime-content hash
`128604ad742deb46008fc312424801ac8a2e607c924266bdedaa763c059aaf72`.
The matrix correctly remains `runtime_consumption_blocked` until paired exact
eval is rerun or backfilled against the same current runtime/upload content
that dispatch will use.

## Next

Use this probe as the local preflight for a paired CPU/CUDA exact rerun of
`format_0x0c_exact_radix` once active dispatch claims clear. The paired rerun
must pass:

- `--expected-archive-sha256 56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`;
- current-runtime Modal upload custody;
- CPU/CUDA axis separation;
- terminal dispatch-claim closure.
