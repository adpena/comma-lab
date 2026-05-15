# PR101 CPU Axis Submission Gate + PR106 Runtimefix Dispatch - 2026-05-15

schema: `contest_axis_gate_and_runtimefix_dispatch_v1`
score_claim: `false`
promotion_eligible: `false`

## Operator Gate

The active submission cutoff is exact score `<= 0.192000000000` on the
selected contest axis. Scores that only round to `0.192` or sit at
`0.1920513168811056` are not submission-eligible under this operator gate.

Landed enforcement:

- commit: `49cbb1490`
- file: `scripts/pre_submission_compliance_check.py`
- default: `--max-submission-score 0.192`
- selectable axis: `--submission-score-axis contest_cuda|contest_cpu`
- CPU-axis final packets still require explicit `[contest-CPU]` custody and
  paired runtime/archive consistency; they are not promoted through CUDA
  shorthand.

Focused proof:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_submission_compliance_check.py -q
.venv/bin/ruff check scripts/pre_submission_compliance_check.py src/tac/tests/test_pre_submission_compliance_check.py
.venv/bin/python -m py_compile scripts/pre_submission_compliance_check.py
```

Result:

- `39 passed`
- `All checks passed!`
- py_compile passed

## PR101 FEC6 Classification

PR101/FEC6 fixed-Huffman K16 is a real CPU-axis result, but it is below neither
the operator cutoff nor the CUDA frontier:

- archive sha256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- archive bytes: `178517`
- `[contest-CPU]`: `0.1920513168811056`
- `[contest-CUDA/T4]`: `0.22621002169349796`
- CPU gap to operator cutoff: `0.00005131688110560084`
- unchanged-component byte equivalent: about `78` bytes

Conclusion: CPU-axis submission is contest-eligible in principle, but this
specific packet is not submission-eligible because it misses the operator's
`<=0.192` cutoff. Simply submitting CPU is not novel versus PR101; PR101's
public leaderboard position already exploits the CPU-favorable axis.

## PR106 topk8 Runtimefix Dispatch

The first PR106 latent score-table topk8 paired dispatch failed before scoring
because it used `submissions/robust_current/inflate.sh` against a single-member
PR106 `0.bin` archive. That runtime expects `masks.mkv` / `.amrc` siblings, so
the failure is classified as `archive/runtime_mismatch`, not a method result.

Corrected paired dispatch uses `submissions/pr106_latent_sidecar_r2_pr101_grammar`
with fresh lane IDs:

- pair group: `pair_pr106_latent_topk8_20260515_codex_runtimefix1`
- archive:
  `experiments/results/pr106_latent_score_table_topk8_20260515_codex/sidecar_archive.zip`
- archive sha256:
  `582c6ff6ea427c03f7bc241c85830a30d95302bd1a0bf03da7891530bb20ed96`
- archive bytes: `186302`
- CUDA lane: `lane_pr106_latent_topk8_cuda_eval_20260515_runtimefix1`
- CUDA job: `pr106_latent_topk8_cuda_modal_t4_20260515_runtimefix1`
- CUDA call id: `fc-01KRNKPQ63W9GSHX3GYJ9Q37VA`
- CUDA output:
  `experiments/results/modal_auth_eval/pr106_latent_score_table_topk8_20260515_cuda_runtimefix1`
- CPU lane: `lane_pr106_latent_topk8_cpu_eval_20260515_runtimefix1`
- CPU job: `pr106_latent_topk8_cpu_modal_20260515_runtimefix1`
- CPU call id: `fc-01KRNKPQ891QF8JG5PME0X8C2S`
- CPU output:
  `experiments/results/modal_auth_eval_cpu/pr106_latent_score_table_topk8_20260515_cpu_runtimefix1`

## Runtimefix Harvest Result

Both runtimefix calls recovered cleanly through canonical Modal auth-eval
recovery. The runtime mismatch was fixed, but the method result is negative.

CUDA:

- output:
  `experiments/results/modal_auth_eval/pr106_latent_score_table_topk8_20260515_cuda_runtimefix1`
- evidence grade: `[contest-CUDA]`
- final score (full precision recompute):
  `0.20941739550293906`
- rounded report score: `0.21`
- avg SegNet distortion: `0.00067069`
- avg PoseNet distortion: `0.00003348`
- archive bytes: `186302`
- scorer device: `cuda`
- GPU: `Tesla T4`
- n samples: `600`
- result: `passed=true`, `promotion_eligible=false`

CPU:

- output:
  `experiments/results/modal_auth_eval_cpu/pr106_latent_score_table_topk8_20260515_cpu_runtimefix1`
- evidence grade: `[contest-CPU]`
- final score (full precision recompute):
  `0.23016349151751497`
- rounded report score: `0.23`
- avg SegNet distortion: `0.00065575`
- avg PoseNet distortion: `0.00016433`
- archive bytes: `186302`
- scorer device: `cpu`
- platform: `Linux x86_64`
- n samples: `600`
- result: `passed=true`, `promotion_eligible=false`

Classification:

- `archive/runtime_mismatch` was fixed by using
  `submissions/pr106_latent_sidecar_r2_pr101_grammar`.
- The topk8 latent sidecar candidate is not a sub-0.192 path. It is worse
  than PR106 format0C on both axes and much worse than PR101/FEC6 CPU.
- Do not fan out topk16/32/64 under the same method without a new xray finding
  showing component movement rather than sidecar byte churn.
