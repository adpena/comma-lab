# HDM8 Headerless Format07 vs FES1 All-None Contest CUDA Score-Response Probe

Authority:
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false

Verdict: `RATE_ONLY_IMPROVEMENT`
Mode: `candidate`

Axis note: exact contest axes and advisory axes are kept distinct. A `macos_cpu_advisory` report is not a `[contest-CPU]` promotion claim.

## Source Artifacts

- baseline: `.omx/research/frame_exploit_fes1_all_none_control_exact_cuda_result_review_20260515_codex.json`
- candidate: `.omx/research/pr106_hdm8_headerless_format07_exact_cuda_result_review_20260515_codex.json`
- generated_json: `experiments/results/rule6_score_response_hdm8_headerless_vs_fes1_20260517_codex/score_response_probe.json`
- generated_json_sha256: `f23dddb79d3b85404bf3a97f6a314f3d48d501778062952467fed715e2960769`
- command: `.venv/bin/python tools/probe_substrate_score_response.py --baseline-json .omx/research/frame_exploit_fes1_all_none_control_exact_cuda_result_review_20260515_codex.json --candidate-json .omx/research/pr106_hdm8_headerless_format07_exact_cuda_result_review_20260515_codex.json --axis contest_cuda --mode candidate --relaxed-custody --min-total-improvement 0.000001 --min-scorer-term-improvement 0.000001 --title 'HDM8 Headerless Format07 vs FES1 All-None Contest CUDA Score-Response Probe' --output-json experiments/results/rule6_score_response_hdm8_headerless_vs_fes1_20260517_codex/score_response_probe.json --output-md .omx/research/hdm8_headerless_vs_fes1_exact_cuda_score_response_20260517_codex.md`

## Thresholds

- min_total_improvement: `1e-06`
- min_scorer_term_improvement: `1e-06`

## Evidence

| side | axis | score | seg_term | pose_term | rate_term | bytes | runtime_tree_sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | contest_cuda | 0.206903674 | 0.064260000 | 0.017988885 | 0.124654789 | 187209 | `2832a8922905` |
| candidate | contest_cuda | 0.206351677 | 0.064260000 | 0.017988885 | 0.124102792 | 186380 | `f1890b6c9121` |

## Deltas

Negative deltas improve the contest score.

- total_delta: `-0.0005519970721382939`
- scorer_term_delta: `0.0`
- seg_term_delta: `0.0`
- pose_term_delta: `0.0`
- rate_term_delta: `-0.00055199707213828`

## Blockers

- none

## Interpretation

This probe tests score response, not byte liveness. A positive result requires official-score component movement under matched controls; a rate-only improvement is not evidence that the substrate's distinguishing feature is scorer-visible.

## Operational Consequence

This exact-CUDA comparison is a byte/grammar win, not a scorer-response win.
HDM8 headerless format07 improves total score by `0.0005519970721382939`
entirely through the rate term: `187209 -> 186380` bytes. SegNet and PoseNet
terms are unchanged at the reported precision.

This is still useful frontier pressure because it proves the packet/format
path can harvest score without changing decoded scorer behavior. It should be
routed as a Rule #6 rate-axis bolt-on candidate, not as evidence for a new
substrate class. The next engineering action is to stack more header/payload
elision or entropy coding on this exact packet family, then re-run the same
score-response probe to keep scorer-response and rate-only movement separated.
