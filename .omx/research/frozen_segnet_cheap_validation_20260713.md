# Task 454: frozen-SegNet cheap validation mechanism and measured disposition

Date: 2026-07-13 UTC  
Lane: `lane_454_segnet_cheap_validation_20260713`  
Authority: `[macOS-CPU advisory]`; `score_claim=false`; `promotion_eligible=false`  
Review status at write: `recovery-written-UNREVIEWED`. Any upgrade must be carried by fresh-context review receipts bound to this file's exact SHA-256; the measurement receipt retains its at-measurement `unreviewed_fix_round_2` tag.

STORES CONSULTED: `research(5715)`, `equations(622)`, `memory(1893)`, `dag(505)`, `council(277)`, `tasks(96)`, and `docs(92)` through one `tools/corpus_query.py` retrieval; `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; the v7.5 and v8 specifications; the final YOPO provider, receipt, DSL, equation, and probe; the frozen-SegNet alternatives memo; the goldmine ledger; current canonical lane, task, probe-outcome, and subagent state; sealed pair-0 early, boundary, and late renderer checkpoints; the exact frozen CPU SegNet and PoseNet paths. Deliberately not consulted or actuated: paid/cloud state, the live trainer, the protected V9 run, `upstream/evaluate.py`, and score-pointer mutation.

## Outcome

**VERDICT: NO-GO, recovery-written-UNREVIEWED.** `verdict_scope: formulation` — pair 0; sealed early, boundary, and late regimes; the landed `blocks[0]` YOPO split; seed `20260712`; the registered `1e-2 * 0.5^j` ladder; macOS CPU advisory execution; this empirical-bound and strict feature-ball formulation only.

**MEASURED:** the final receipt is `experiments/results/segnet_validation_certificate_20260713T015633Z/receipt.json`, SHA-256 `60fe88fa1a5058d018170005890ef0720f01b31762b5e7ef0b5c7d6dc19a7d60`. It admits 3 of 58 disjoint holdout candidates and rejects 55, a rejection rate of `0.9482758620689655`. There are zero exact-`d_seg` unsafe accepts among the three proxy accepts. Two accepted boundary candidates worsen exact Pose; one also worsens teacher cross-entropy. Thus the SegNet-only margin predicate has `false_negative_dseg=0/3` on a very small admitted set, while joint held CE/`d_seg`/`d_pose` descent is `NO-GO` with `unsafe_accepts_any=2/3`. Across-seed variance is **UNKNOWN**.

**DERIVED from content-bound measured components:** the median cheap validation cost is `0.006529812060762197 s`. For `K=2`, `t_exact=1.036083978950046 s`, `t_approx=0.07453912496566772 s`, and rejection-weighted full-teacher fallback is `0.9824934283146988 s`; the formula gives `0.9869128501255486x`. For `K=4`, `t_exact=1.0148602295084856 s`, `t_approx=0.07091529201716185 s`, and fallback is `0.9623674590166674 s`; the formula gives `0.9818936607306264x`. Both are below parity and below the pre-registered `1.3x` gate.

**BOUNDARY:** those speedups are component economics, not a sequence-integrated whole-step measurement. The parallel master gate was not duplicated. This does not weaken the scoped NO-GO: the rigorous construction lacks the bound needed to exist, the empirical construction fails joint held descent, and even its rejection-weighted component arithmetic is below `1x`. A throughput `GO` is not claimed.

The defensive `[contest-CPU]` pointer `0.1880443979880752` is **UNMOVED (MEANS)**. No exact archive was evaluated and no score is claimed.

## Trust-region derivation

Let `h0=f(x0)` be the frozen first-block feature at an anchor. Let `a_p=argmax_c z_{p,c}(h0)` and

`m_p = z_{p,a_p}(h0) - max_{c != a_p} z_{p,c}(h0)`.

Protect the pixels whose anchor prediction is the fixed target label. Suppose an actual neighborhood upper bound satisfies, for every protected pixel and competing class,

`|(z_{p,a_p}-z_{p,c})(h) - (z_{p,a_p}-z_{p,c})(h0)| <= L_p ||h-h0||_inf`.

Then the triangle inequality gives the sufficient strict feature radius

`r_h = min_{p correct at anchor} m_p/L_p`.

If `||h-h0||_inf < r_h`, every anchor-correct pixel remains correct. Pixels already wrong cannot add another error, so exact `d_seg` cannot worsen. If a separately proven prefix bound `||f(x)-f(x0)||_inf <= L_f ||x-x0||_inf` exists, `r_x=r_h/L_f` is sufficient in input units. The landed control law checks the feature displacement directly and immediately selects `full_teacher_and_refresh` on ball exit, nonfinite data, shape/custody mismatch, or empirical rejection. This is an event-conditioned tested predicate with a completion guarantee on every step: reuse or refresh; no `TBD` state exists.

**RIGOROUS / DERIVED:** this implication uses only the triangle inequality; no external theorem is imported. A local Jacobian value is not a neighborhood Lipschitz upper bound. The first-block Jacobian also does not bound the downstream suffix. No actual suffix pairwise-logit upper-bound artifact exists in this probe. Therefore the rigorous mechanism's verdict is `NO-GO`, scoped to this certificate construction; trust regions remain intact as a family.

**BOUNDED HEURISTIC / MEASURED ON HOLDOUT:** the empirical proxy calibrates

`Lhat_p = max_j |d_p(h_j)-d_p(h0)| / max(||h_j-h0||_inf, eps)`

on the first two registered candidates and tests the remaining ladder candidates with exact same-frame SegNet and PoseNet controls. Its measured feature radii are `0.0817609896644987` for early, `0.15851609885329931` for boundary, and `0.02289550955017739` for late. These values are empirical local slopes, not certificates. The accepted set is one early candidate, two boundary candidates, and zero late candidates. The observed exact-`d_seg` false-negative rate is `0/3`; the joint unsafe-accept rate is `2/3`. A single-seed zero does not establish a population false-negative bound.

The costate-reuse law is attributed at its point of use to Dinghuai Zhang, Tianyuan Zhang, Yiping Lu, Zhanxing Zhu, and Bin Dong (2019), *You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle*, arXiv:1905.00877. The arXiv abstract identifier was resolved to that exact paper before this derivation. No other literature result or theorem is imported.

## Measurement custody and controls

**MEASURED:** source custody in the receipt binds `tools/probe_segnet_validation_certificate.py` to SHA-256 `000cac7a849ef28ae92a075907e751fe9b26865a0a669f690c4312d04b029636` and `src/tac/boundary_math/segnet_validation_certificate.py` to SHA-256 `2c860f63dda9cda38e5778e76a21bacda439c2ca2e514bbb8bc238a6d54beba1`. The inherited YOPO receipt is bound to SHA-256 `a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3`. The positive known-linear-map control, outside-ball negative control, confusion-meter positive/negative control, calibration-content mutation control, renderer parity check, and prefix/full-forward feature parity checks all pass.

**MEASURED:** exact safety comparisons use the identical receiver-realized camera `uint8` frame for the proxy candidate and its exact control. Median exact frozen-SegNet CE/`d_seg` safety measurement is `0.446324291464407 s`; median PoseNet safety measurement is `0.1841492290259339 s`. Those safety-control timings are excluded from the cheap action cost. Rejections are instead charged for the DSL's actual `full_teacher_and_refresh` fallback using the inherited exact teacher forward/backward median for each cadence.

**NOISE FLOOR:** identical deterministic within-run inputs define a zero functional comparison floor for exact `d_seg` and `d_pose`; timing uses matched receipt medians but no separate timing-floor canary was registered for this mechanism. Timing deltas are therefore component economics, not promotion-grade whole-step evidence. Across-seed variance and contest-CPU/CUDA transfer are **UNKNOWN**.

## Triality and ownership

DSL: `tac.witness_dsl.segnet_validation_certificate_policy`, with typed rigorous and empirical authority, no loose trainer flag, and `full_teacher_and_refresh` fallback.

Equation: `segnet_margin_trust_region_v1` in `tac.canonical_equations.segnet_margin_trust_region_20260713`.

DAG: the matching terse FEED in `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.

The mechanism is isolated and `research_only=true`. The live trainer, shared YOPO provider, protected run, evaluator, and sibling-owned dirty files were not edited.
