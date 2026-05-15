# PR106 format 0x0C exact-radix candidate

Date: 2026-05-15

## Artifact

Source archive:
`experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip`

Source SHA-256: `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
Source bytes: `186341`

Candidate archive:
`experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`

Candidate SHA-256: `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
Candidate bytes: `186327`
Delta vs source: `-14` bytes
Pure-rate score delta if components are unchanged: `-0.000009322025343710398`

Profile artifacts:

- `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/profile.json`
- `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/profile.md`

## Status

`score_claim=false`; candidate is byte-closed but not yet exact-eval scored.

Open blockers from the profile:

- `runtime_decode_apply_proof_required_for_new_candidate_archive`
- `full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing`
- `exact_cuda_auth_eval_missing`
- `contest_auth_eval_adjudication_missing`

## Next Action

Dispatch exact Modal T4 CUDA auth eval with the PR106 R2 runtime:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval.py \
  --archive experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip \
  --submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval/pr106_hdm12_hlm3_fmt0c_t4_20260515T000000Z \
  --detach --provider-detach-ack \
  --lane-id lane_pr106_hdm12_hlm3_magicless_exact_radix_format0c_20260515 \
  --instance-job-id pr106_hdm12_hlm3_fmt0c_t4_20260515T000000Z \
  --claim-agent codex:modal_auth_eval \
  --claim-notes "PR106 format0C exact-radix byte-closed CUDA eval; archive_sha=56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7; bytes=186327; expected pure-rate delta=-0.000009322025343710398"
```

## Supersession - paired-by-default exact eval landed

The single-axis Modal command above is superseded by the paired
`[contest-CUDA]` + `[contest-CPU]` run recorded in
`.omx/research/pr106_format0c_paired_cpu_cuda_auth_eval_20260515_codex.md`
and closed by
`.omx/research/pr106_format0c_packetir_exact_closure_20260515_codex.md`.

Current canonical score-lowering route:

- `[contest-CUDA]`: `0.2063163866158099`
- `[contest-CPU]`: `0.22776488386973992`
- archive SHA-256:
  `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- submission/runtime:
  `submissions/pr106_latent_sidecar_r2_pr101_grammar`

Do not dispatch a successor from this memo's stale single-axis command. Any
score-bearing successor must use paired CPU/CUDA custody or carry an explicit
waiver in its lane claim and result review.
