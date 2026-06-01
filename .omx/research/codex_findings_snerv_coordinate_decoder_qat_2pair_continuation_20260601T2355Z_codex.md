# Codex Findings: SNeRV Coordinate Decoder/QAT 2-Pair Continuation

UTC: 2026-06-01T23:55:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Authority: false-authority; local continuation only; no score claim; no promotion;
no exact-eval readiness

## What Ran

Ran the first bounded non-exact continuation after the 1-pair coordinate
decoder/QAT gate. The run used the same receiver-replayed SNAR1 path and the same
strict pose+seg+score acceptance gate, but widened from 1 pair to 2 pairs.

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --n-pairs 2 --levels 2 --max-trials 4 --search-mode top_weight_coordinate \
  --qat-bits 8 --step-map-bins 16 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_smoke_20260601T2353Z.json
```

Smoke artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_smoke_20260601T2353Z.json`
- SHA-256: `de4d028d94d38385ab55abee6710f7d7b88712ff3a6cd9dea17feb420e285537`
- Search mode: `top_weight_coordinate`
- Scorer-loop evaluations: `9`
- Receiver contract: satisfied
- Accepted improvement: `true`
- Ready for pose-guard gate: `true`
- Ready for exact eval: `false`

Baseline:

- `d_seg=0.0025355020770803094`
- `d_pose=0.0017251845565624535`
- `score=0.823265396480944`

Best local candidate:

- Label: `coord_040_plus`
- `d_seg=0.002532958984375`
- `d_pose=0.001720065949484706`
- `score=0.8228160908469069`
- `score_delta=-0.00044930563403711155`
- `d_pose_delta=-0.000005118607077747583`
- `d_seg_delta=-0.000002543092705309391`

## Pose-Gate Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_smoke_20260601T2353Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_pose_gate_20260601T2355Z.json
```

Gate artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_coordinate_2pair_pose_gate_20260601T2355Z.json`
- SHA-256: `2ae63f19601d3de3c8c8f50bcd01f898c1ff51d2cbb17992b358ae91fcaae5bd`
- Verdict: `GO_LOCAL_CONTINUATION_ONLY`
- Accepted rows: `3`
- Blockers: `[]`
- Still false-authority:
  - `score_claim=false`
  - `promotion_eligible=false`
  - `ready_for_exact_eval_dispatch=false`

Accepted labels:

- `coord_040_plus`
- `coord_039_minus`
- `coord_028_minus`

## Interpretation

The 1-pair coordinate signal survived a 2-pair local continuation, but the effect
contracted sharply: score delta moved from about `-0.02484` at 1 pair to about
`-0.000449` at 2 pairs. That is still a strict local continuation GO, but it is
not promotion evidence and not enough for exact/full-600 dispatch.

Next step is another bounded local continuation with a broader pair sample and
the same gate, while preserving per-pair component deltas. If the delta keeps
contracting toward zero, the optimizer should pivot from single-coordinate
perturbations to learned/nonlinear decoder QAT or pair-conditional decoder
sections before any receiver grammar promotion work.
