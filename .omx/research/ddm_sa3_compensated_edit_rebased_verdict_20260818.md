# sa3 ADMITTED — the eighth pointer move, and the family's own ceiling in the same arithmetic

`verdict_scope`: **INSTANCE** for the admitted row (archive
`d2ad58ee28b84388a262bd5c8b11611a163dcc2694ad3c29a1283605a206b992` @ 179,140 B, contest-CUDA
T4, n=600). The exchange ratios are INSTANCE; the shape (rate/seg linear, pose concave,
retention rising with mass) is DERIVED-EXACT and travels.

Receipts: `/Volumes/APDataStore/pact/ddm_sa3/t4_row_cuda/` — `MODAL_REMOTE_RESULT.json` plus the
eight embedded artifacts now persisted separately (`artifact_contest_auth_eval.json`,
`artifact_provenance.json`, `artifact_inflated_outputs_manifest.json`, …) per P0 ALWAYS KEEP THE
PAYLOAD. Call `fc-…`, wall 1,190 s, ~$0.16.

## 1. The row

| | value |
|---|---|
| S | **0.15765851477950737** |
| archive | 179,140 B, sha `d2ad58ee…` |
| d_seg | 0.00029815 |
| d_pose | 7.33e-06 |
| axis | `contest-CUDA`, Tesla T4, `gpu_t4_match=True`, n=600 |
| harness | `passed=True`, `rc=0`, `validation_errors=[]` |

Against the sz1 pointer (0.15771357797660338 @ 179,930 B): **net ΔS −5.506320e-05, ADMITTED.**
Eighth micro-campaign pointer move. Gap to 0.15: 0.00771 → **0.00766** (0.71% closed).

## 2. The leg split — exact, and it sums

| leg | Δ | share of the rate credit |
|---|---:|---:|
| rate | **−5.260286e-04** (−790 B) | 1.000 |
| pose | +2.669654e-04 (d_pose 6.880e-06 → 7.330e-06) | 0.508 |
| seg | +2.040000e-04 (d_seg +2.04e-06 = +241 flips) | 0.388 |
| **net** | **−5.506320e-05** | **0.105 retained** |

The three legs sum to the harness's own `score_recomputed_from_components` to 17 digits. This is
a decomposition, not a reconciliation.

**Quantization custody.** `canonical_score_source = report_8dp_components_plus_exact_archive_bytes`
— the score is built from 8-dp components, and the harness publishes
`report_8dp_score_worst_case_abs_error_bound = 3.4205e-06` (pose ±2.92e-06, seg ±5e-07). The net
is **16.1× that bound**. The sign is determinate by a stated margin. That is the standing cure for
the `#1032` genus, where a −4e-06 "result" turned out to be one pose ULP wearing a verdict's
clothes: **always divide the delta by the bound before believing it.**

## 3. The four sealed falsifiers — all PASS

| | bar | measured | |
|---|---|---|---|
| F1 net | < −3.5e-06 | −5.506e-05 (15.7×) | PASS |
| F2 seg | ≤ 2× the predicted +1.72e-06 | +2.04e-06 = 1.19× | PASS |
| F3 pose | < +5.226e-04 | +2.670e-04 (51% of credit) | PASS |
| F4 clean | `passed`, no validation errors | both | PASS |

F2 is the one worth pausing on: the seg damage was **predicted before the fire** and landed at
1.19× the prediction. The edit's collateral is modelable, which is what makes the family
composable rather than a coin flip.

## 4. The counter-intuitive part — retention RISES with mass

Rate credit and seg damage are linear in edit mass. Pose damage is not: `√(10·d_pose)` is
concave, and this family pays pose in the **upward** direction, where the marginal
`5/√(10·d_pose)` is falling. Extrapolating `d_pose` linearly (licensed by sa1's measured 0.91×-of-
linear mass law) and then re-taking the square root:

| mass | rate credit | pose damage | seg damage | net | retention |
|---:|---:|---:|---:|---:|---:|
| ×1 | −5.260e-04 | +2.670e-04 | +2.040e-04 | −5.506e-05 | 10.47% |
| ×2 | −1.052e-03 | +5.259e-04 | +4.080e-04 | −1.182e-04 | 11.24% |
| ×4 | −2.104e-03 | +1.022e-03 | +8.160e-04 | −2.660e-04 | 12.64% |

Everywhere else on this campaign concavity has been the enemy — `ddm_asym1` measures a 2×
worsening costing 1.41× what a 2× improvement buys. Here it is the friend, for exactly the same
reason with the sign flipped. **The single easiest way to get this wrong is to scale the pose
S-leg directly**; that assumes a linearity the score function does not have and under-states the
family at every mass above 1.

## 5. The ceiling — and it is the routing verdict

Solve the same arithmetic for the mass whose net alone closes the 0.00765851 gap:
**52.1× mass, requiring 41,160 B of rate credit.** The entire `semantic_blob` is **34,243 B**.
The demand is **1.20× the whole section** — the family is arithmetically incapable of closing the
gap by itself at any mass, even in the physically impossible limit where the section goes to zero.

**It is a CONTRIBUTOR, not a route.** Fire it for its net, compose it, and take the gap elsewhere.
Falsifier: a higher-mass rung whose measured retention beats the concavity prediction by more than
the bound would move this ceiling.

## 6. What this settles about the family

sa1 refused the UNCOMPENSATED version 3/3, with pose damage 68–512× the rate credit. sa3's
compensated version pays 0.508×. **The in-compile Schur compensation is the whole difference** —
two-plus orders of magnitude — and qs4 proved the converse: a compensation carried from a
*different* object cost +2.396e-04 (cross-regime constant transfer in miniature). The compensation
must be re-solved against the edited object, in the compile, asserted in code.

## 7. Routing

- Registered as `compensated_semantic_edit_exchange_v1` in `tac.canonical_equations`;
  `project_at_mass()` is what a successor rung calls before sealing, and
  `family_cannot_close_alone()` is why no successor should be aimed at the gap.
- The rate axis remains the linear, additive, honest one (`ddm_asym1`), but per
  `ddm_bp1` its **coding** half is measured shut (−5 B total). Rate progress is REPRESENTATION.
- Pose still owns ~108% of the gap and needs a 203.8× reduction to close it alone — the integral
  and the work are different statements, and both belong in the next routing decision.
