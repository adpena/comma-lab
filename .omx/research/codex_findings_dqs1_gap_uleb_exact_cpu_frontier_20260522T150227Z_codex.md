# Codex Findings: DQS1 Gap-ULEB Exact CPU Frontier

utc: 2026-05-22T15:02:27Z
lane: lane_dqs1_top32_gap_uleb_selective_decoderq_exact_cpu_20260522
status: LANDED
score_claim: true
promotion_eligible: false

## Summary

The compact DQS1 `sorted_gap_uleb` top32 selective decoder-q archive is the new
scanner-derived `[contest-CPU]` frontier.

- Archive:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/submission_dir/archive.zip`
- Archive SHA-256:
  `e12f5cfe93f9dbf624549466cda62d00a01e10bee8d1e0ea8a635af69247908a`
- Archive bytes: `178560`
- DQS1 payload bytes: `43`
- Exact `[contest-CPU]` score: `0.19202894881608987`
- Prior `[contest-CPU]` frontier: `0.1920513168811056`
- CPU improvement: `0.000022368065015737626`

The paired exact `[contest-CUDA T4]` result for the same archive is
`0.22619043540195719`, so this is a CPU-axis frontier move only. It is not a
CUDA promotion candidate.

## Exact Artifacts

- CPU result:
  `experiments/results/modal_auth_eval_cpu/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cpu/modal_cpu_auth_eval_result.json`
- CPU raw evaluator:
  `experiments/results/modal_auth_eval_cpu/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cpu/contest_auth_eval.json`
- CUDA result:
  `experiments/results/modal_auth_eval/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cuda/modal_cuda_auth_eval_result.json`
- CUDA raw evaluator:
  `experiments/results/modal_auth_eval/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cuda/contest_auth_eval.json`
- Locality controls:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/locality_controls_top32_gap_uleb.json`

## Findings

1. Gap-ULEB was the needed rate fix.

   The raw-u16 DQS1 top32 packet had `75` payload bytes and exact
   `[contest-CPU]` score `0.1920502563025898`. Gap-ULEB reduced the payload to
   `43` bytes and exact `[contest-CPU]` to `0.19202894881608987`.

2. The score movement is rate-dominated on CPU.

   CPU SegNet and PoseNet terms match the raw-u16 packet at the displayed
   precision: SegNet `0.00055978`, PoseNet `0.00002943`. The useful movement
   came from shrinking the consumed DQS1 trailer while preserving the same
   selected/unselected frame locality.

3. CUDA rejects this decoder-q direction for now.

   The same archive is `0.22619043540195719 [contest-CUDA T4]`, with SegNet
   `0.00066252` and PoseNet `0.00016845`, far worse than the current CUDA
   frontier. CPU and CUDA stay separate axes.

4. The earlier local feedback artifact is stale for compact packets.

   The old local feedback files for archive
   `3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`
   describe the raw-u16 `75` byte packet. They must not be cited as evidence
   for the gap-ULEB archive
   `e12f5cfe93f9dbf624549466cda62d00a01e10bee8d1e0ea8a635af69247908a`.

5. Feedback authority hardening landed after adversarial review.

   `tac.optimization.decoder_q_selective_runtime_feedback` now treats macOS
   advisory data as local calibration only. It cannot rank, kill, promote, or
   suppress exact replay. The feedback payload also preserves advisory raw SHA
   and materialization/locality custody so stale artifacts are easier to catch.

6. Signed-calibrated waterbucket follow-up is operational but not measured.

   The top32-plus-singleton sign labels were fed back into
   `tools/plan_decoder_q_signed_waterbucket.py`; the resulting five unique
   fixed-length candidates all pass official inflate visibility controls and
   change 600 frames. They still have no advisory component response and no
   exact-eval authority.

## Verification

```bash
shasum -a 256 experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/submission_dir/archive.zip
wc -c experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_gap_uleb_top32/submission_dir/archive.zip
.venv/bin/python tools/recover_modal_auth_eval.py --output-dir experiments/results/modal_auth_eval_cpu/dqs1_top32_gap_uleb_selective_decoderq_paired_20260522T145356Z_cpu --timeout-s 300
.venv/bin/python tools/scan_best_anchor_per_axis.py --format json
```

Focused tests were rerun after feedback hardening:

```bash
ruff check src/tac/optimization/decoder_q_selective_runtime_feedback.py \
  tools/build_decoder_q_selective_runtime_feedback.py \
  src/tac/tests/test_decoder_q_selective_runtime_feedback.py

.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_runtime_feedback.py -q
```

## Next Action

Keep the gap-ULEB DQS1 archive as the current `[contest-CPU]` exact frontier,
but do not route it toward CUDA promotion. Next frontier work should search for
selective/runtime packet variants that preserve CPU rate wins while avoiding
the CUDA scorer-input shift, or move the same DQS1 grammar to a CPU-targeted
submission packet with explicit axis labeling.
