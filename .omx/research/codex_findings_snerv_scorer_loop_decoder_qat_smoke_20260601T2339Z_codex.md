# Codex Findings: SNeRV Scorer-Loop Decoder/QAT Smoke

UTC: 2026-06-01T23:39:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Authority: false-authority; no score claim; no promotion; no exact-eval readiness

## What Landed

Added a bounded local SNeRV scorer-loop decoder/QAT smoke:

- `src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py`
- `tools/run_snerv_scorer_loop_decoder_qat_smoke.py`
- `src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py`

The smoke uses real frames from `upstream/videos/0.mkv`, fits the existing
least-squares SNeRV HF decoder, quantizes decoder weights during evaluation, and
tests small decoder-weight perturbations against receiver-replayed SNAR1 packets
with the real SegNet/PoseNet mirror. A trial can only replace the current best
if receiver replay passes, PoseNet is no worse, SegNet improves, and advisory
score improves.

This is a genuine local scorer loop, not a placeholder trainer. It remains a
tiny smoke: 1 pair, 3 scorer-loop evaluations, CPU only.

## Smoke Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --n-pairs 1 --levels 2 --max-trials 1 --qat-bits 8 --step-map-bins 16 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_smoke_20260601T2338Z.json
```

Artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_smoke_20260601T2338Z.json`
- SHA-256: `78c6d2b05b22b2998fb56bea217b8e1ab1f737a7df40f8f2bfe08a5d4dcb6592`
- Receiver contract: satisfied
- Scorer-loop evaluations: `3`
- Accepted improvement under full pose+seg+score guard: `false`
- Exact-eval readiness: `false`

Rows:

- `least_squares_qat_baseline`: `d_seg=0.002471923828125`,
  `d_pose=0.002065342850983143`, `score=0.6145014925781653`.
- `trial_1_plus`: score improves to `0.6096631306465231` and pose improves to
  `0.0019145157421007752`, but SegNet worsens slightly to
  `0.0024770100135356188`; rejected by `seg_gate_failed`.
- `trial_1_minus`: SegNet improves to `0.0024058024864643812`, but PoseNet
  worsens to `0.0063820164650678635` and score worsens to
  `0.7168028829248886`; rejected by `pose_guard_failed` and `score_gate_failed`.

## Pose-Gate Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_smoke_20260601T2338Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_pose_gate_20260601T2339Z.json
```

Artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_pose_gate_20260601T2339Z.json`
- SHA-256: `9a7efbd3a72f1c15824282795e353dd7927d740d73531a78e932c3c70df7aafe`
- Verdict: `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`
- Accepted rows: `0`
- Blocker: `no_candidate_passes_pose_guarded_local_continuation_gate`

## Verification

```text
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  tools/run_snerv_scorer_loop_decoder_qat_smoke.py
All checks passed!

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/substrates/snerv_inverse_steg_carrier/tests/test_scorer_loop_decoder_qat.py \
  src/tac/tests/test_snerv_scorer_loop_decoder_qat_contract.py \
  src/tac/tests/test_snerv_pose_guarded_decoder_gate.py \
  src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py \
  src/tac/tests/test_snerv_rate_adjudication.py \
  src/tac/tests/test_snerv_step_map_coder.py -q
38 passed in 1.35s
```

## Fresh-Eyes Review

The xhigh fresh-eyes explorer independently identified the same highest-risk
blocker: score-faithful distortion closure, not receiver packaging. It
recommended the SNeRV scorer-loop decoder/QAT smoke as the next concrete
artifact and made no file edits.

## Engineering Consequence

The first true scorer-loop smoke did not pass the continuation gate. It did,
however, expose the two-objective tension cleanly: one local direction helps
PoseNet/score while barely hurting SegNet; the opposite helps SegNet while
hurting PoseNet hard. Next SNeRV work should therefore implement a stronger
multi-objective optimizer/nonlinear decoder with explicit SegNet and PoseNet
constraints, not a scalar random perturbation loop and not exact eval.
