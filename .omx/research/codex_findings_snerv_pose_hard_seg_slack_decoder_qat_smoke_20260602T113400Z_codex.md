# Codex Findings: SNeRV Pose-Hard Seg-Slack Decoder-QAT Smoke

Created: 2026-06-02T11:34:00Z

Axis: `[macOS-CPU advisory]` local smoke only. This is not a score claim, promotion claim, rank claim, kill claim, or exact-eval result.

## What Changed

`src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py` now exposes `seg_slack` for local scorer-loop decoder-QAT acceptance.

Default behavior remains strict: `seg_slack=0.0` still rejects SegNet no-worse violations. The new path is explicit and opt-in, so SNeRV can run the user-requested PoseNet-hard, total-score-primary local continuation without silently discarding score-improving directions that slightly worsen SegNet.

## What Ran

Both smokes used the real upstream video:

- upstream dir: `/Users/adpena/Projects/pact/upstream`
- video: `/Users/adpena/Projects/pact/upstream/videos/0.mkv`
- mode: `nes_pair_robust`
- pairs: `1`
- levels: `3`
- target bits per LF coeff: `2.0`
- max trials: `1`
- perturb scale: `0.01`
- pair score-improved fraction guard: `1.0`
- pair PoseNet-worsened fraction guard: `0.0`

Artifacts:

- strict SegNet gate: `.omx/research/snerv_scorer_loop_decoder_qat_smoke_nes1pair_20260602T112900Z.json`
- explicit SegNet slack: `.omx/research/snerv_scorer_loop_decoder_qat_smoke_nes1pair_segslack_20260602T113400Z.json`

## Evidence

Strict `seg_slack=0.0`:

- evaluations: `4`
- baseline score_linf: `1.1817133779961255`
- best accepted score_linf: `1.1817133779961255`
- accepted improvement: `false`
- blocker: `no_quantized_decoder_trial_improved_score_under_pose_guard`
- best rejected direction lowered score but failed `seg_gate_failed`

Explicit `seg_slack=0.00005`:

- evaluations: `4`
- baseline score_linf: `1.1817133779961255`
- best score_linf: `1.1718591965053646`
- score delta: `-0.009854181490760894`
- d_seg_linf delta: `+0.000025430694222450256`
- d_pose_linf delta: `-0.0005317204631865025`
- accepted improvement: `true`
- ready_for_pose_guard_gate: `true`
- ready_for_exact_eval_dispatch: `false`

## Verdict

GO for SNeRV local continuation with explicit PoseNet-hard, score-primary `seg_slack` sweeps on stratified 2-pair or 4-pair windows.

NO-GO for promotion, rank, kill, full-video, exact, or CUDA. This is a 1-pair local advisory smoke and still carries:

- `local_smoke_only_not_full_600_pairs`
- `paired_contest_cpu_cuda_pass_missing`
- `mixed_precision_decoder_payload_grammar_not_byte_optimized`

## Roadmap / Blockers

Immediate: PR101 CPU remains pending through canonical recovery, so no new exact/full-video/CUDA launches.

SNeRV: continue with bounded stratified 2-pair or 4-pair `nes_pair_robust` sweeps using PoseNet hard guard and explicit SegNet slack grid. Preserve per-pair deltas and receiver-decoded byte accounting. If the accepted delta collapses across windows, pivot to learned/nonlinear decoder QAT rather than coordinate perturbation.

Rate: current decoder QAT still emits fp32 receiver payload after fake quant; the next byte work is mixed-precision decoder grammar or decoder-delta packing.

HiNeRV: continue parallel full-stack work with real-teacher SegNet/PoseNet training, coder-aware QAT, joint P18/P19 weighting, and dense decoder VJP L-infinity allocator under the same archive/runtime/eval-axis discipline.

PR95: use upstream PR95 `hnerv_muon` as same-axis control before any beat claim.

Promotion: blocked until full-600 byte-closed receiver proof plus paired contest CPU/CUDA pass.
