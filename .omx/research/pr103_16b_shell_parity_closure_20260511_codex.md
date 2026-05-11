# PR103 -16B shell-level inflate parity closure (2026-05-11)

## Summary

The PR103 clean-runtime mid32+latent-hi `-16B` packet now has exact
source-vs-candidate shell-level inflate output parity on local CPU.

Proof artifact:

`.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/inflate_shell_output_parity_full_cpu.json`

Command class:

```bash
.venv/bin/python tools/probe_inflate_shell_output_parity.py \
  --source-archive experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip \
  --source-inflate-sh experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/source/submissions/hnerv_lc_ac/inflate.sh \
  --candidate-archive .omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/packet/archive.zip \
  --candidate-inflate-sh .omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/packet/inflate.sh \
  --video-name 0.mkv \
  --timeout-s 1200 \
  --python-bin .venv/bin/python \
  --json-out .omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_mid32_plus_latent_hi_candidate/inflate_shell_output_parity_full_cpu.json
```

## Result

- `passed=true`
- parity method: `exact_inflate_sh_archive_dir_output_dir_file_list`
- `score_claim=false`
- `dispatch_attempted=false`
- source archive SHA-256:
  `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- candidate archive SHA-256:
  `8460014d70855ce9226285f80513d6d743ed23723870a6a38b009cfca40f423e`
- source return code: `0`, elapsed `35.29661933402531s`
- candidate return code: `0`, elapsed `35.5800535841845s`
- output file: `0.raw`
- output bytes: `3662409600`
- source output SHA-256:
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`
- candidate output SHA-256:
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`
- output mismatches: `[]`

## Environment note

Both public PR103 and the candidate shell scripts invoke `python`. On this
macOS host, `python` is not on PATH, so the parity probe used a temporary PATH
shim pointing to `.venv/bin/python`. The shim is recorded in the JSON proof.
The packet/runtime files were not modified.

This is a local CPU shell-contract parity proof, not an auth-eval score claim.
The existing exact Modal T4 `[contest-CUDA]` rate-only result remains the score
evidence.

## Score-lowering implication

Together with strict pre-submission compliance, this upgrades the PR103 `-16B`
candidate from "byte-closed but parity-scope blocked" to "strict-compliance and
shell-parity closed." It remains a small rate-only CUDA-positive observation,
not a new representation breakthrough. Remaining decision surface is paired
CPU/adjudication policy and whether to use PR103 `-16B` as a release/supplement
anchor while focusing future score-lowering on larger training-time substrate
work.
