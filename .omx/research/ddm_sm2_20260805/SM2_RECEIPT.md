# SM2 Receipt -- #865 Missing Third Arm

## Answer First

SM2 does not promote the entropy+SMEVR SUM surrogate as the guidance default.

On the banked RG5 delta rows, the affine SUM model is the best linear fit, but only slightly: RMSE
41,187.6 B versus 41,866.2 B for entropy and 42,799.4 B for SMEVR. On the live QO1/FZ1 `sub_final`
token stream, SMEVR-only remains the best per-cell ranker, and the combined affine arm is not a
material winner.

The #862 blind-subspace wording needs narrowing. Live pair permutations are marginal-entropy blind
(`max |d_entropy| = 1.38e-14 bits`) while real SMEVR bytes move by 13-13,466 B. SMEVR and SUM do see
the random-permutation class through the temporal-delta surrogate, but the RG5-fit affine predictors
are not byte-calibrated on these rows. If #862 means "marginal entropy is blind," it is confirmed; if
it means "both single surrogates are blind," it is over-scoped.

No scorer was run. No `upstream/evaluate.py` call was made. No archive was mutated. This is a
rate-only, scorer-free guidance result.

## Recall Evidence

Stores consulted before building:

| store | used fact |
|---|---|
| `.omx/tmp/codex_runs/sm2_prompt.md:1-54` | SM2 asks for the missing third arm, live token stream race, blind-subspace rows, and queued migration only. |
| `.omx/tmp/codex_runs/_common_contract.md` | No scorer/evaluate, no launches, review passes for `.py`, serializer commit with expected hashes. |
| `.omx/state/main_hot_state.md` | Own-vehicle pointer is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved. |
| `.omx/research/ddm_rg5_rate_gradient_sign_20260801.md` | RG5 found the original "wrong way" gradient claim false, but left the missing SUM arm and permutation-blind residual. |
| `.omx/research/ddm_rg5_rows_20260801.jsonl` | 152 measured RG5 rows used for the affine and quadratic fits. |
| `.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md` | Prior rank-fidelity context: entropy can anti-correlate in rearrangement regimes, SMEVR proxy is usually closer to real bytes. |
| `.omx/research/ddm_qo1_repair_stream_optimal_form_20260804.md` | QO1/FZ1 `sub_final` lineage and live 357,836 B row. |
| `/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_subfinal_tokens/sub_final_tokens_extract_receipt.json` | Live token stream shape/hash: `[600,24,32,4]`, uint8, sha256 `d4eacbf619d09aeda1c15a5015b0cd45ab2d3de33d349c881b7e0f59dc803a56`. |
| `.omx/research/ddm_pa2_20260805/PA2_RECEIPT.md` | PA2 says shared context helps modestly, but does not change this rate-surrogate routing. |

## Artifacts

| artifact | sha256 |
|---|---|
| `experiments/ddm_sm2_sum_surrogate_race.py` | `f2c086b1007b9d9162679c854adcfe7c106b0b4c54dc4a05b790e5d9accbe766` |
| `src/tac/tests/test_ddm_sm2_sum_surrogate_race.py` | `861d5206f839bdd9f3a8a2233009271d6dc451658928f86b0945fbf25d3c0c5e` |
| `.omx/research/ddm_sm2_20260805/SM2_RESULTS.json` | `44ae282bbfae1d30e030cb4b59244bfd2c510a1041022f92dba3d50e28ce4f82` |
| `.omx/research/ddm_sm2_20260805/SM2_ROWS.jsonl` | `4918d9c7d2402ef96488891dad9064534e7c1a78acd52f77b12504c231de0847` |

Command:

```bash
.venv/bin/python experiments/ddm_sm2_sum_surrogate_race.py --out-dir .omx/research/ddm_sm2_20260805
```

Authority surface:

| item | value |
|---|---|
| scorer forwards | 0 |
| `upstream/evaluate.py` calls | 0 |
| archive mutations | 0 |
| byte truth | `experiments.ddm_r7_token_coder.encode_token_codes(codec="smevr")` |
| live token denominator | 600 pairs x 24 x 32 cells x 4 channels |
| RG5 fit denominator | 152 measured rows from `.omx/research/ddm_rg5_rows_20260801.jsonl` |
| live race denominator | 14 full-stream rows, 768 per-cell rows, 48 spatial-tile rows |

## RG5 Fit

Delta fits target `d_bytes` from RG5 rows:

| arm | predictors | RMSE B | MAE B | R2 |
|---|---:|---:|---:|---:|
| entropy | `d_surr_entropy` | 41,866.2 | 25,361.1 | 0.0663 |
| SMEVR | `d_surr_smevr` | 42,799.4 | 25,824.7 | 0.0242 |
| fixed SUM | `d_surr_entropy + d_surr_smevr` | 42,195.1 | 25,631.3 | 0.0516 |
| affine SUM | `248937.2266*d_entropy + 48051.0758*d_smevr + 16551.4648` | 41,187.6 | 25,277.9 | 0.0963 |

Absolute fits target `smevr_bytes` from RG5 rows:

| arm | RMSE B | MAE B | R2 |
|---|---:|---:|---:|
| entropy | 44,986.3 | 32,716.0 | 0.8785 |
| SMEVR | 125,432.6 | 113,904.2 | 0.0555 |
| fixed SUM | 73,977.3 | 61,047.0 | 0.6715 |
| affine SUM | 43,001.3 | 30,676.6 | 0.8890 |

Residual form check: a quadratic delta model reduces RMSE from 41,187.6 B to 39,059.1 B
(5.17%). That is enough to log as residual curvature, but not enough to justify a nonlinear trainer
surrogate before a linear arm proves live usefulness.

## Live Race

Full-stream pair permutations:

| metric | value |
|---|---:|
| base SMEVR bytes | 346,478 |
| base entropy bits | 3.457817 |
| base SMEVR-surrogate bits | 1.975895 |
| non-identity rows | 13 |
| real `d_bytes` range | 13 to 13,466 B |
| `max |d_entropy|` | 1.38e-14 bits |
| `d_smevr` range | 0 to 0.206809 bits |

Full-stream delta rank/fit:

| arm | Pearson | Spearman | MAE B | note |
|---|---:|---:|---:|---|
| entropy | 0.6098 | 0.5526 | 7,763.2 | Apparent rank comes from numerical noise plus intercept; entropy itself is blind. |
| SMEVR | 0.999956 | 0.8163 | 12,511.7 | Sees random permutations, not calibrated in magnitude. |
| fixed SUM | 0.999956 | 0.8207 | 15,379.1 | Same signal as SMEVR plus blind entropy. |
| affine SUM | 0.999956 | 0.8207 | 14,510.8 | Not a material win over SMEVR. |

Per-cell absolute-byte ranking, 768 rows:

| arm | Pearson | Spearman | MAE B |
|---|---:|---:|---:|
| entropy | 0.9392 | 0.8586 | 546,972.6 |
| SMEVR | 0.9903 | 0.9762 | 277,640.3 |
| fixed SUM | 0.9763 | 0.9074 | 579,754.6 |
| affine SUM | 0.9485 | 0.8707 | 588,116.9 |

The absolute MAE is not used as an admitted byte estimator on cells because the RG5 fit is full-field
calibrated while these rows are independent one-cell codings with their own headers. Rank is the
decision signal here, and SMEVR-only wins.

Spatial-tile absolute-byte ranking, 48 rows:

| arm | Pearson | Spearman | MAE B |
|---|---:|---:|---:|
| entropy | 0.9382 | 0.8774 | 374,790.0 |
| SMEVR | 0.9874 | 0.9497 | 280,119.6 |
| fixed SUM | 0.9819 | 0.9597 | 384,099.1 |
| affine SUM | 0.9536 | 0.9059 | 366,702.5 |

Fixed SUM has a small tile Spearman edge over SMEVR, but loses Pearson and calibration. That is a
diagnostic nibble, not a migration trigger.

## Verdict

| question | answer |
|---|---|
| Does the SUM arm win materially? | No. It improves RG5 linear delta fit slightly, but does not dominate live per-cell/tile ranking. |
| Does SUM close the #862 blind subspace? | No. Entropy remains blind; SUM only inherits SMEVR's temporal-delta visibility and does not calibrate the byte magnitude. |
| Is nonlinear warranted now? | No. Quadratic residual improvement is 5.17% on RG5 fit rows, but live linear SUM has not earned trainer complexity. |
| Guidance default today | Keep SMEVR-only as the rate-guidance default for token drop/waterfill planning. |
| Scoped regrade | #862 should say "marginal entropy blind subspace"; "both singles blind" is over-scoped under this measurement. |

## Queued Routing

No hot-swap in this arm. If a future migration wants to expose SUM as a third race arm, wire it as
`C_sum_affine` behind the existing surfaces:

| consumer | current surface | queued change |
|---|---|---|
| `experiments/train_tr1_partition_renderer_mlx.py:2172` | `--rate-model` choices are only `entropy` and `smevr_surrogate`. | Add a third explicit choice only after registering coefficients/provenance; likely `sum_affine`. |
| `experiments/train_tr1_partition_renderer_mlx.py:2993` | `token_rate_term` branches on SMEVR else entropy. | Compute both soft histograms and combine with registered coefficients; do not include the fitted intercept in a loss term. |
| `src/tac/witness_dsl/spec_tr1_renderer_20260728.py:275` | `lever_rate_in_loss` validates only the two current choices. | Add the third choice with SM2 receipt provenance and scoped no-promotion status. |
| `src/tac/witness_dsl/spec_tr1_burn2_20260731.py:96` | QA86 returns `A_entropy` and `B_smevr_surrogate`. | Add `C_sum_affine` as a queued race arm, not as default. |
| `src/tac/tests/test_ddm_b2b_burn2_composition.py:213` | Test asserts exactly two QA86 arms. | Update only when the DSL migration lands. |
| `experiments/ddm_tw1_token_waterfill_state_dependence.py:3` | #869 measures state-dependent real coder prices, scorer-free. | Feed SMEVR-only as current ranker; SUM can be logged as an auxiliary column, not an allocator default. |
| `experiments/ddm_tz1_token_sweep_rate_attack.py:20` | #869 adaptive per-cell L surface already structures token-by-token waterfill. | Keep measured coder bytes authoritative; if SUM is added, use it only to order probes before real-byte reprice. |
| `tools/measure_s2_terminal_coder_break_even.py:3` | #866 computes terminal-coder break-even over real payload bytes. | Do not replace break-even arithmetic with a surrogate; SUM may only be a prefilter before real payload bytes. |

## Next If Resumed

```json
{
  "resume_id": "ddm_sm2_20260805",
  "status": "complete_scored_free_rate_only",
  "default_guidance": "smevr_surrogate",
  "queued_optional_arm": "C_sum_affine",
  "do_not_hot_swap": true,
  "next_candidate_unit": "wire C_sum_affine in QA86 only if a governed race wants an explicit third trainer arm; otherwise feed SM2's regrade into #869/#866 docs",
  "hard_boundaries": {
    "scorer_forwards": 0,
    "evaluate_py_calls": 0,
    "archive_mutations": 0
  },
  "artifacts": {
    "results": ".omx/research/ddm_sm2_20260805/SM2_RESULTS.json",
    "rows": ".omx/research/ddm_sm2_20260805/SM2_ROWS.jsonl",
    "script": "experiments/ddm_sm2_sum_surrogate_race.py",
    "test": "src/tac/tests/test_ddm_sm2_sum_surrogate_race.py"
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
