# Codex Findings: SNeRV Pose-Guarded Decoder Gate

UTC: 2026-06-01T23:24:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Authority: false-authority; no score claim; no exact-eval readiness

## What Landed

Added a reusable SNeRV decoder-fit gate:

- `src/tac/analysis/snerv_pose_guarded_decoder_gate.py`
- `tools/build_snerv_pose_guarded_decoder_gate.py`
- `src/tac/tests/test_snerv_pose_guarded_decoder_gate.py`

The gate consumes existing SNeRV advisory/sweep JSON rows through the canonical
`iter_snerv_candidate_rows(...)` helper, so row ingestion is not duplicated. It
selects the least-squares waterfill control and refuses candidate continuation
unless all of the following hold:

- receiver archive replay is verified;
- archive bytes are within the configured slack;
- `d_pose_linf` is no worse than the control;
- `d_seg_linf` is below the SegNet ceiling and better than the control;
- advisory score improves versus the control.

Passing this gate still does not grant promotion or exact-eval authority. It
only allows bounded local continuation.

## Verification

```text
/Users/adpena/Projects/pact/.venv/bin/ruff check src/tac/analysis/snerv_pose_guarded_decoder_gate.py src/tac/tests/test_snerv_pose_guarded_decoder_gate.py tools/build_snerv_pose_guarded_decoder_gate.py
All checks passed!

/Users/adpena/Projects/pact/.venv/bin/python -m pytest src/tac/tests/test_snerv_pose_guarded_decoder_gate.py -q
3 passed in 0.23s
```

## Gate Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_score_weighted_decoder_fit_gain_sweep_20260601T223158Z.json \
  .omx/research/snerv_hf_decoder_saliency_component_sweep_20260601T223741Z.json \
  .omx/research/snerv_hf_decoder_pose_saliency_gain_sweep_20260601T223927Z.json \
  --out .omx/research/snerv_pose_guarded_decoder_gate_20260601T2324Z.json
```

Artifact:

- Path: `.omx/research/snerv_pose_guarded_decoder_gate_20260601T2324Z.json`
- SHA-256: `25f577bce214d08499e56bf4d7a39d8117aa57a8a654b91a8dd15d2fc211e0c5`
- Baseline: `least_squares_baseline_existing`
- Baseline `d_seg_linf`: `0.02264404296875`
- Baseline `d_pose_linf`: `2.1390697956085205`
- Baseline `score_linf`: `6.911887587116307`
- Rows evaluated: `16`
- Accepted rows: `0`

Verdict:

- `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`
- `closed_form_scalar_weighting_no_go=true`
- Blockers:
  - `no_candidate_passes_pose_guarded_local_continuation_gate`
  - `closed_form_scalar_component_weighting_no_go`

Representative failures:

- `score_weighted_gain_0.25`: fails pose, seg, and score gates.
- `combined_gain_0.25`: fails pose, seg, and score gates.
- `pose_gain_0.01`: improves SegNet but fails pose and score gates.
- `pose_gain_0.25_existing`: improves SegNet but fails pose and score gates.

## Engineering Consequence

Closed-form scalar/component HF residual weighting is now machine-gated as a
NO-GO family for promotion/exact-eval routing. Future SNeRV candidate rows must
beat the least-squares waterfill control under the PoseNet hard guard before
even local continuation is allowed.

The next implementation should be scorer-loop or nonlinear/QAT decoder training,
not another scalar DWT-residual gain sweep.
