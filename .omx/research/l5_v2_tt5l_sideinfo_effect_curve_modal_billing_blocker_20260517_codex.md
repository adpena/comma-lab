# L5 v2 TT5L side-info effect-curve Modal billing blocker

- schema: `l5_v2_tt5l_sideinfo_effect_curve_provider_blocker_v1`
- recorded_at_utc: `2026-05-17T05:50:18Z`
- plan_artifact: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
- plan_id: `l5_v2_tt5l_sideinfo_effect_curve_dispatch_0902d4c91d23e972`
- provider: `modal`
- attempt_classification: `provider_app_creation_failed_before_lane_claim`
- failure_stage: `modal_app_creation_before_remote_wrapper_start`
- failure_message: `App creation failed: workspace billing cycle spend limit reached`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- method_verdict: `indeterminate_provider_blocked`

## Attempted Cells

All five byte-closed TT5L side-info effect-curve variants from the current
full-shape manifest were fired through the canonical paired Modal dispatcher.
Each returned rc=`1` before app creation because the Modal workspace billing
cycle spend limit was reached.

| Variant | Archive bytes | Archive SHA-256 | Pair group | Result |
| --- | ---: | --- | --- | --- |
| `zero` | 34373 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102` | provider blocked |
| `random_lsb` | 38681 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190` | provider blocked |
| `shuffled` | 43284 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4` | provider blocked |
| `trained` | 43323 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779` | provider blocked |
| `ablated` | 42419 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998` | provider blocked |

## Custody Observations

- Required axes: `contest_cpu`, `contest_cuda`
- Runtime content tree SHA-256: `bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32`
- Modal uploaded runtime tree SHA-256, `contest_cpu`: `025123f140784b92b0d6145a05df6e97aedace7cf3e620a10551454cfa8057a2`
- Modal uploaded runtime tree SHA-256, `contest_cuda`: `83f5f760000802b69abb187e8239403a0d8fd9ed21ec8d127fa0542ca91001aa`
- `rg -n "lane_l5_v2_tt5l_sideinfo_effect_curve|tt5l_sideinfo_effect_curve" .omx/state/active_lane_dispatch_claims.md` returned no rows.
- `find experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve -maxdepth 4 -type f` returned no files.

## Adversarial Classification

This is not score evidence, not a method negative, and not side-info usefulness
evidence. Modal failed before app creation, before remote wrapper start, before
per-axis lane claims, before call IDs, and before any recoverable auth-eval
artifact.

The five variants remain byte-closed and ready for paired measurement once a
contest-compliant provider route is available. Do not classify TT5L side-info
as dead or useful from this blocker. The next valid actions are:

- raise the Modal billing limit and re-fire the same current archive/runtime SHA set through the paired dispatcher;
- regenerate a contest-compliant alternate-provider plan against the five current archive SHA-256 values;
- refuse any stale provider plan that names older TT5L side-info archive SHAs.

