# REPLACE round 4 — rank the exact-costate support

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round4_support_ranking_20260713`  
**Authority:** `[macOS-CPU advisory; NumPy-fp64 convex fit; frozen CPU SegNet exact costates]`  
**Status:** `NO_GO_SHALLOW_CHEAP_FEATURE_CONVEX_LOCALIZATION`; `research_only=true`; `score_claim=false`; `promotion_eligible=false`  
**Pointer delta:** `NONE`

## Executive verdict

**MEASURED heldout retained mass is `20.172451295048283%` at realized `4.7017415364583336%` area, versus the preregistered `47%` bar.** The winning fixed rung is `pairwise-rank-pair-block-44`; it misses the gate by `26.827548704951715` percentage points. Its uplift over uniform area is `4.290421142597201x`, and its conditional masked-exact costate cosine is `0.44913752120089323`.

Calibration is not the failure: winner heldout ECE is `0.003073753177168275`, below the preregistered live-refusal bar `0.05`, with zero selected calibration fallbacks. The exact same-area oracle remains `52.78150212253758%`, above the gate by `5.7815021225375795` percentage points. Therefore useful support exists, but this shallow cheap-feature convex ranking family cannot identify enough of it.

The exact-optimum verdict covers all three preregistered convex classes: one 84-column global weighted-top-k head, twenty 44-column ordered-pair weighted-top-k heads, and twenty 44-column implicit all-positive-negative RankRLS heads. Every populated fit passes its rank-truncated float64 Moore-Penrose optimum certificate. The family-level negative is narrowly scoped to this first-pre-SE chart and fixed n600 replay; deeper, nonlinear, dense-label, transition-complete FORE/on-policy, other-replay, and evaluator-equivalent witness families remain open.

## Settled inputs held read-only

Round 3, committed at `87dc484ee7`, is not rederived. It settled:

- direct costate regression is dead across its linear and fixed-RFF exact convex formulations;
- source-margin localization retained `16.34677541848741%` mass;
- RFF log-mass regression retained `2.4426459564827255%`, worse than uniform;
- the same-area exact oracle retained `52.78150212253758%`;
- the local pre-SE prefix costs a DERIVED `0.5714118050141177%` of full-teacher convolution FLOPs;
- the class-source sensitivity input spans `32 / 0.26 = 123.07692307692308x` between Lane and Undrivable, with Road `2.2`, Lane `32`, Undrivable `0.26`, MyCar approximately zero, and Movable explicitly ASSUMED neutral `1.0` because no committed coefficient was supplied.

Round-2/3 modules, policies, receipts, and memos remained read-only.

## Preregistered decision rule

Before the first round-4 teacher call, `experiments/results/replace_round4_support_ranking_20260713/preregistration.json` sealed the full decision surface. It is `8816` bytes with SHA-256 `ec90adf96b0ec8f239409971f55bb9f5d3f8e442365df772a2ce983d9521c8ff`.

- Population: fixed V9 n600 replay, seed455, checkpoints ep150/ep251/ep275, deterministic 480 train / 120 heldout split.
- Selection: exactly `2311` of `49152` prefix cells per heldout state, realized fraction `0.047017415364583336`; tie-break `(-calibrated_score, flat_cell_index)`.
- Primary metric: aggregate exact input-costate L2-square mass retained across all 120 heldout states.
- Admission: PASS iff the maximum of the three fixed rungs is `>=0.47`.
- Winner: maximum retained mass; exact ties follow preregistered rung order.
- Calibration diagnostic: 10-bin ECE, Brier score, reliability rows, and support prevalence; live REFUSE if winner ECE `>0.05`, a selected block uses fallback calibration, FORE is inadmissible, or custody drifts.
- Stop: measure all three fixed rungs and add no fourth rung after seeing heldout results.

The three target/feature combinations were:

1. `weighted-topk-global-84`: state-balanced weighted squared binary top-k support error on the global chart.
2. `weighted-topk-pair-block-44`: the same objective in twenty ordered source-to-competitor block heads.
3. `pairwise-rank-pair-block-44`: implicit all-positive-negative squared RankRLS within each state and ordered-pair block.

The 84-column global chart adds ordered-pair one-hot and pair-times-margin channels to the 42-column round-3 base, plus source-sensitivity and sensitivity-times-margin. The 44-column block chart keeps the base and two sensitivity channels while preventing cross-pair cancellation through independent heads.

## Exact convex custody

For each sufficient-statistic pair `(G,r)`, the preregistered numerical problem is the symmetric-eigendecomposition Moore-Penrose minimum-norm solution after discarding eigenvalues at or below

`eps_float64 * feature_width * lambda_max(G)`.

The fit certificate tests `V_retained^T(Gw-r)` because discarded directions define the numerical nullspace. It also records the full gradient, discarded-space RHS mass, objective, numerical rank, eigenvalue bounds, and weight SHA. The full residual is diagnostic, not a first-order condition for the declared rank-truncated problem.

Fit artifact: `stage_fit_complete.json`, `46527` bytes, SHA-256 `b431f9389aebf6e85657d8689359f782a4387c30d22e8e38628d4425374ea8bf`. All populated heads have `normal_equation_optimum_certified=true`. No width, regularization, threshold, seed, or rung sweep occurred.

## MEASURED decision table

| Rung | Exact retained L2-square mass | Uplift / uniform | Conditional cosine | Heldout ECE | Decision |
|---|---:|---:|---:|---:|---|
| weighted-topk-global-84 | `0.19865776607447305` | `4.225195377798571x` | `0.4457104060648271` | `0.003096012008190578` | FAIL mass gate |
| weighted-topk-pair-block-44 | `0.19771315378268864` | `4.205104688328304x` | `0.44464947293647905` | `0.0028016677593879187` | FAIL mass gate |
| pairwise-rank-pair-block-44 | `0.20172451295048283` | `4.290421142597201x` | `0.44913752120089323` | `0.003073753177168275` | winner; FAIL mass gate |
| exact same-area oracle | `0.5278150212253758` | diagnostic | `0.72650878950318` | N/A | feasibility witness only |

The global weighted head actually beats the weighted pair-block head by `0.00094461229178441` absolute mass fraction. Pairwise block ranking recovers only `0.00306674687600978` above the global weighted rung and `0.00401135916779419` above weighted blocks. Explicit class-pair structure therefore does not close the measured gap in this shallow convex chart.

The winner retains `0.00025184753555350115` of `0.001248472641573917` aggregate exact costate L2-square mass, while the oracle retains `0.000658962613811638`. The remaining oracle-to-winner headroom is `0.32609050827489297` mass fraction.

## DIG-S1 query composition and FORE admission

The fixed-replay research policy composes localization with `DIG-S1-QUERY-REAL-CALIBRATION`:

- `trust`: winner selects the cell, at least two of the three fixed selectors agree, and no calibration fallback is used;
- `query`: winner selects the cell but selector disagreement or calibration fallback remains;
- `refuse`: FORE support is inadmissible, ECE guard fails, or source/feature custody drifts.

Current status is `REFUSE_LIVE__RESEARCH_ONLY_FIXED_REPLAY`. The causal-manifest FORE checker returns `NOT_IDENTIFIED`, `weights_applied=false`, with zero decisions, transitions, observed rewards, run manifests, or support evidence. Its blockers include missing treatment manifest, state-action-reward-successor transitions, coverage receipt, and executed decision rows. No occupancy weight, live admission, or teacher-call credit is fabricated.

## Honest economics

The closed conditional law is

`C_teacher = A + c_label * D`, with `c_label = p + (1-p)q`.

Using DERIVED prefix fraction `p=0.005714118050141177` and realized selected area `q=0.047017415364583336` gives

`c_label=0.05246287035291876`, or a CONDITIONAL variable-cost ratio `19.061099655298698x`.

This is not a measured wall-clock speedup. The current exact EfficientNet-B2 teacher has global squeeze-excite dependence and no sparse exact-kernel receipt, so current exact-teacher wall cost remains dense. Because the localizer fails the primary gate, it is not admitted even conditionally.

Round-4 exact-teacher custody is `600` unique batch-size-1 forward-plus-input-backward state calls: `480` train and `120` heldout. Started calls equal completed calls; retries charged are `0`. The ledger has `2280` rows and SHA-256 `ab91f7e130d9ac4cc88d9df7151bb9ef389d3ea36b7a772c335b2eedeba71249`. Round3 plus round4 campaign calls total `1200`; no round-4 retry is hidden.

## Resumability and source-amendment custody

Training preserved a compact support target per state, one atomic sufficient-statistic accumulator after every state, and three distinct stage snapshots:

- ep150 after 160 train states, SHA-256 `e683f98a0c54d711cf2a25f97d1baffbca8257a7d6089c3ba317fab2ad1ae290`;
- ep251 after 320 states, SHA-256 `85b92cb263dfce37bd84afc53edfd0824eb5867631581fd9e4bf025c9a797621`;
- ep275 after 480 states, SHA-256 `6beb0c180eab68c322f27c593a190715894a0acfd8baed75ef3bcce939247b72`.

The first fit attempt stopped before calibration or heldout because its verifier incorrectly demanded a full-gradient residual in numerical-rank directions the preregistration explicitly discarded. All 480 support labels were already sealed. Before any heldout call, amendment `rank-truncated-mp-certificate-v1` preserved original/new source bundles, the 480-target tree, accumulator SHA, and train-only teacher-event tree; it recomputed zero teacher calls. Its receipt SHA is `96aea52a9bf7070c846c7f72a82168fbf281ab2652cb11c912e502d209e2d779`.

A second pre-fit resume guard compared the populated `completed_pairs` vector to an empty template shape. Amendment `resume-accumulator-schema-v2` changed only that resume validator, again before heldout and with zero teacher recomputation. Its receipt SHA is `b5027d60723fb22f4f7b3068aa0f10cf70ab1fd72382d5c178bed3c9b1a66884`.

Both amendments are implementation/verifier scope only. Labels, features, targets, objectives, rank threshold, rung ladder, calibration, heldout decision, and gate did not change. These failed attempts carry no hidden formulation retry because no alternative fit or heldout result was observed.

## Receipt custody and disk hygiene

- Primary receipt: `receipt.json`, `237850` bytes, SHA-256 `6ccbf0e10691dc39c94b77aaefdfe7d9ac3a38b32962bfa5eefcb1107f627222`.
- Heldout stage: `stage_heldout_complete.json`, `26122` bytes, SHA-256 `b540579580fc55d2525606d2485418973646777a4b9532f50aad0aa694275dab`.
- Calibration stage: `stage_calibration_complete.json`, `43424` bytes, SHA-256 `3935a4d2f2631c4106cd051bdb182dc2b72f9df023faa4a471971da6e33c8cd2`.
- Decision stage: `stage_decision_complete.json`, `943` bytes, SHA-256 `17e995d21189b62b2e3414429e46e93ab941013c1fcb97983c8bcbade8312aae`.
- Cleanup manifest: `cleanup_manifest.json`, `3438` bytes, SHA-256 `ad816ab5ffdfa294913ec3b7f0dbd796911bee337018a6cdc946d7bae5216d1e`; blockers `[]`.
- Total durable round-4 directory: approximately `24 MiB`; raw frames and costates were process-local and never persisted.
- Storage preflight failed closed on unavailable SSD mounts and used the explicitly permitted local tier with more than the requested `512 MiB` free. No paid, remote, GPU, MPS, evaluator, or archive run occurred.

## Verdict scope and reformulation queue

**Verdict scope: `FAMILY x FIXED REPLAY`.** The family-level negative covers shallow first-pre-SE cheap-feature convex support localizers using the registered global/pair-block weighted-top-k and pairwise RankRLS objectives on V9 n600 seed455. The exact-optimum citation is `stage_fit_complete.json` SHA-256 `b431f9389aebf6e85657d8689359f782a4387c30d22e8e38628d4425374ea8bf`.

It does not kill localization: the same-area oracle still passes. EV-ordered reformulation queue:

4. **Deeper scorer features with new cost/global-state custody.** The current cut is immediately before the first squeeze-excite. Any deeper chart loses the inherited tileable claim unless it explicitly carries the required global state.
5. **Transition-complete FORE and disagreement-audited on-policy queries.** Collect treatment manifest, `(s,a,r,s')`, positive propensities, coverage receipt, and executed decision rows before applying occupancy weights.
6. **Dense-label or nonlinear support learner under the same 47%-at-matched-area gate.** This is a new family and owes stability, teacher-call, and heldout preregistration.

Other replay distributions and seeds remain open, but a seed sweep of the killed shallow convex family is not justified by this result.

## Triality and unified-system wire-in

- DSL: `tac.witness_dsl.replace_round4_support_ranking_policy`; typed fixed instance, `research_only=true`, no live trainer argv.
- Equation: `tac.canonical_equations.replace_round4_support_ranking_20260713`; support-retention and conditional sparse-teacher laws plus the family-scoped empirical anchor.
- DAG: `.omx/research/replace_round4_support_ranking_DAG_FEED_20260713.md`; shared hot DAG deferred to main review.
- Sensitivity map: per-pair cell count, selected count, total/selected exact mass, source class sensitivity channels.
- Pareto: retained exact mass × area × conditional compute; archive bytes and evaluator score are unmeasured.
- Bit allocator: no direct score actuator; future passing support can become a compute-allocation mask.
- Cathedral/autopilot: `REFUSE`; no live, paid, or trainer dispatch.
- Continual learning: advisory family-scoped `KILL` probe row plus `reformulation-queue` curriculum-pool row.
- Probe disambiguator: finite three-rung modes in `tools/probe_replace_round4_support_ranking.py`.

Lane maturity is L1 on `impl_complete` only. Real-archive, contest axes, STRICT preflight, three-clean review, memory, and deploy-runbook gates remain unclaimed. Shared canonical-equation registration remains deferred to main review because the operator explicitly requested an uncommitted landing.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`.
- v7.5 and v8 canonical specifications.
- `reports/latest.md`; lane registry; subagent progress; master gradient anchors; Modal call ledger; cost-band and continual-learning posteriors; probe outcomes.
- latest Codex findings/session summary, council T3, V9 design, and last-24-hour directives.
- committed round-2/3 memos, modules, DSL policies, DAG feeds, receipts, and exact replay custody.
- causal-manifest FORE checker and `DIG-S1-QUERY-REAL-CALIBRATION` crosswalk ticket.
- curriculum candidate pool status vocabulary and canonical locked record helper.
