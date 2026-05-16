# PR106 format0D Paired Modal Eval - 2026-05-16

## Candidate

- archive: `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip`
- archive bytes: `186876`
- archive sha256: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
- pair group: `pair_pr106_format0d_latent_score_table_20260516`
- dispatch plan: `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_auth_eval_plan_execute_autohash_20260516.json`
- dispatch commit: `81cea44616483b7f60e00a7145c0cf4145e5f447`

## Result

CUDA:

- call id: `fc-01KRQT9KAWAS19EA3XBJ1QBTEC`
- artifact dir: `experiments/results/modal_auth_eval/pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cuda`
- evidence grade: `[contest-CUDA]`
- score: `0.20533002902019143`
- avg seg dist: `0.00063042`
- avg pose dist: `0.00003188`
- archive bytes: `186876`

CPU:

- call id: `fc-01KRQTA49BQF871350BABQRZ9P`
- artifact dir: `experiments/results/modal_auth_eval_cpu/pr106_format0d_latent_score_table_paired_modal_auth_20260516T071622Z_cpu`
- evidence grade: `[contest-CPU]`
- score: `0.22712591739832488`
- avg seg dist: `0.00062212`
- avg pose dist: `0.00016387`
- archive bytes: `186876`

## Pair Analysis

Pair drift artifacts:

- `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_cpu_cuda_drift_contest_json_20260516.json`
- `experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/paired_cpu_cuda_drift_contest_json_20260516.md`

Drift verdict:

- valid same-archive axis score pair: `true`
- same archive sha: `true`
- same runtime content tree: `true`
- same inflated output aggregate sha: `false`
- mechanism class: `different_raw_outputs_runtime_or_inflate_drift`
- CUDA minus CPU score gap: `-0.021795888378133454`

Inflated-output aggregate SHA:

- CUDA: `67ca511b07307f88991b1dd2e3f7617103e5c4206fb8db3740c4a71b8f166d33`
- CPU: `fc6147747aa99bba4212cf356540eb48fe34e9ee318f0c1d17dd407ff47cea64`

## Classification

This is a legitimate paired exact-eval measurement of this archive/runtime, but
it is a measured-configuration regression versus the current sub-0.192 target.

It is not a PR106/PacketIR falsification:

- format0D runtime-consumption proof stayed green before dispatch;
- paired exact eval ran after the Modal runtime-hash contract was fixed;
- both axes used the same archive SHA and shared runtime content tree;
- CUDA and CPU inflated raw outputs differ, so this result updates the
  runtime/inflate-device trust region rather than killing the representation
  or sidecar grammar.

Next useful work:

- do not promote or submit format0D as-is;
- localize the CPU/CUDA raw-output drift through inflate-device and scorer xray
  probes before drawing algorithmic conclusions;
- keep PR106 PacketIR as an archive/runtime-consumption substrate for smaller
  component mutations and stack-of-stacks experiments.

