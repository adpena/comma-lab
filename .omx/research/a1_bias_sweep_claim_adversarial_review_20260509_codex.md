# A1 bias sweep claim adversarial review - 2026-05-09

<!-- generated_at: 2026-05-09T11:35:00Z -->
<!-- evidence_grade: adversarial_review; no new eval dispatch; no score promotion -->

## Claim Reviewed

> a3c89347 LANDED - 11 variants, hypothesis FALSIFIED, A1 0.19284758 ROUNDS
> TO PR101 gold 0.19 (medal-band proximity on CPU axis).

## Verdict

Partly correct, but the wording needs tightening.

Correct:

- The 11-row A1 inflate-time bias sweep completed on GHA Linux x86_64 CPU.
- The best row is the inherited PR101-style baseline:
  `v1_pr101_baseline = 0.19284757743677347`.
- No tested alternate bias-coordinate row beats the baseline.
- The closest non-baseline row is `v7_pr101_stack_pr102_red =
  0.19293013742414405`, worse by `+0.00008255998737058401`.
- Removing the inherited bias (`v0_control_no_bias`) is materially worse:
  `0.19521398680571203`, `+0.00236640936893856`.
- The public rounded report field for the baseline is `0.19`.

Unsafe or over-broad:

- "hypothesis FALSIFIED" is too broad unless the hypothesis is scoped to:
  "one of the 10 tested inflate-time bias variants beats A1's inherited
  PR101-style bias on the GHA CPU axis." The broader bias-correction hypothesis
  is not falsified; the no-bias control shows the inherited bias is
  load-bearing.
- "A1 rounds to PR101 gold 0.19" is true only at the public two-decimal display
  precision. It should not be used as an exact equivalence claim. Safer
  wording: "A1's GHA CPU baseline is `0.1928475774`, which displays as `0.19`
  under the same public two-decimal rounding used in the GHA report, placing it
  in the PR101 medal band on the CPU/public axis."
- The result is `[contest-CPU GHA Linux x86_64]`, not `[contest-CUDA]`.
  Promotion or retirement decisions still require paired exact CUDA custody on
  the same archive/runtime packet.

## Custody Re-Harvest

The original aggregate result file lacked the newer exact-report custody
fields (`exact_report_custody`, `report_sha256`, `report_submission_name`,
`expected_submission_name`, `archive_bytes_from_report`,
`promotion_blockers`). I re-harvested the already-completed GHA runs with the
fixed exact-identity harvester. This did not dispatch any new eval.

Command:

```bash
.venv/bin/python tools/harvest_a1_bias_correction_sweep.py \
  --rollup experiments/results/a1_bias_correction_sweep_rollup_20260509T053000Z.json \
  --output experiments/results/a1_bias_correction_sweep_results_20260509T053000Z_reharvested_custody.json
```

Re-harvested aggregate:

- Path: `experiments/results/a1_bias_correction_sweep_results_20260509T053000Z_reharvested_custody.json`
- SHA-256: `f75075dedc5a7da60aa5f8c9fd15c7536fdccff63f76a6d2618c31014f85bc89`
- `n_completed`: 11
- `custody_missing`: 0
- `score_promotion_policy`: "GHA Linux x86_64 CPU rows are public-axis
  evidence only; internal score promotion requires paired exact contest-CUDA
  custody on the same archive/runtime packet."

Best row:

```text
variant_id: v1_pr101_baseline
score: 0.19284757743677347
tag: [contest-CPU GHA Linux x86_64]
report_sha256: ea29e4ee5131b7c42da0cfbb292c4a621fb2eaa8abc45a5bd1ad9e5555f46f9a
archive_bytes_from_report: 178262
exact_report_custody: true
promotion_eligible: false
promotion_blockers: [missing_paired_contest_cuda]
```

## Score Table

| variant | score | delta vs A1 baseline |
|---|---:|---:|
| v1_pr101_baseline | 0.19284757743677347 | 0.0 |
| v7_pr101_stack_pr102_red | 0.19293013742414405 | +0.00008255998737058401 |
| v9_frame1_only | 0.19343922433087615 | +0.000591646894102682 |
| v2_half_magnitude | 0.19429575568975346 | +0.0014481782529799925 |
| v3_one_point_five_x | 0.19478427736834286 | +0.0019366999315693911 |
| v6_pr102_pattern | 0.19516575568975345 | +0.0023181782529799744 |
| v0_control_no_bias | 0.19521398680571203 | +0.00236640936893856 |
| v10_red_channel_only | 0.19573907579177027 | +0.002891498354996802 |
| v4_two_x | 0.19601098680571202 | +0.0031634093689385523 |
| v8_frame0_only | 0.19614983211177614 | +0.0033022546750026693 |
| v5_opposite_sign | 0.19838118614011174 | +0.005533608703338272 |

## Safe Replacement Wording

"A1 bias sweep landed and re-harvested with exact report custody: 11/11 GHA
CPU rows completed; no tested alternate inflate-time bias beats the inherited
PR101-style baseline. A1 baseline is `0.1928475774` `[contest-CPU GHA Linux
x86_64]`, displayed as `0.19` in the public rounded report, and remains
promotion-blocked by missing paired exact CUDA custody. The measured
coordinate-sweep improvement hypothesis is falsified; the broader bias
correction mechanism is not falsified because removing the inherited bias
regresses by `+0.0023664`."
