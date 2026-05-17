# PR101 vs A1 MacOS CPU Advisory Score-Response Probe

Authority:
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false

Verdict: `RATE_ONLY_IMPROVEMENT`
Mode: `candidate`

Axis note: exact contest axes and advisory axes are kept distinct. A `macos_cpu_advisory` report is not a `[contest-CPU]` promotion claim.

## Source Artifacts

- baseline: `experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/per_archive/a1_baseline/contest_auth_eval.json`
- candidate: `experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/per_archive/pr101_hnerv_ft_microcodec/contest_auth_eval.json`
- generated_json: `experiments/results/rule6_score_response_pr101_vs_a1_20260517_codex/score_response_probe.json`
- generated_json_sha256: `ca30e75d5b29ee94432ceaad5c3a77915ea284ab009301fd6b37b1ed5233f579`
- command: `.venv/bin/python tools/probe_substrate_score_response.py --baseline-json experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/per_archive/a1_baseline/contest_auth_eval.json --candidate-json experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z/per_archive/pr101_hnerv_ft_microcodec/contest_auth_eval.json --mode candidate --relaxed-custody --min-total-improvement 0.000001 --min-scorer-term-improvement 0.000001 --title 'PR101 vs A1 MacOS CPU Advisory Score-Response Probe' --output-json experiments/results/rule6_score_response_pr101_vs_a1_20260517_codex/score_response_probe.json --output-md .omx/research/pr101_vs_a1_macos_cpu_advisory_score_response_20260517_codex.md`

## Thresholds

- min_total_improvement: `1e-06`
- min_scorer_term_improvement: `1e-06`

## Evidence

| side | axis | score | seg_term | pose_term | rate_term | bytes | runtime_tree_sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | macos_cpu_advisory | 0.192863676 | 0.056039000 | 0.018127327 | 0.118697349 | 178262 | `880d78659405` |
| candidate | macos_cpu_advisory | 0.192861013 | 0.056039000 | 0.018127327 | 0.118694685 | 178258 | `8b59f1cae8e1` |

## Deltas

Negative deltas improve the contest score.

- total_delta: `-2.663435812455539e-06`
- scorer_term_delta: `0.0`
- seg_term_delta: `0.0`
- pose_term_delta: `0.0`
- rate_term_delta: `-2.6634358124832946e-06`

## Blockers

- none

## Interpretation

This probe tests score response, not byte liveness. A positive result requires official-score component movement under matched controls; a rate-only improvement is not evidence that the substrate's distinguishing feature is scorer-visible.

## Operational Consequence

This is useful signal but not a submission claim. In this local macOS advisory
pair, PR101 vs A1 improves by `0.000002663435812455539` entirely through a
4-byte rate reduction. The SegNet and PoseNet terms are bit-identical at the
reported precision. Treat this as evidence that PR101-style microcodec
discipline can still harvest bytes on a verified substrate, but not as evidence
of scorer-response architecture movement.

Next score-lowering action: run the same scorer-response probe on a
contest-CPU or contest-CUDA exact replay pair when available, and use
`ablation` mode for any same-runtime feature toggle. If a candidate only
reports `RATE_ONLY_IMPROVEMENT`, optimize and celebrate it as a byte/grammar
win; do not promote it as a substrate class-shift.
