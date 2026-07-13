# REPLACE round-5 — deeper features and nonlinear localization

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round5_deeper_nonlinear_20260713`  
**Checkpoint id:** `replace_round5`  
**Authority:** `[macOS-CPU advisory; NumPy-fp64 convex; CPU-torch exact SegNet costates]`  
**Status:** `KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE`; `research_only=true`; `$0 local only`; uncommitted  
**Score authority:** `false`; **promotion eligible:** `false`; **pointer delta:** `NONE`

## Executive result

**MEASURED:** the exact-optimum convex-deeper pair-block rung retained
`0.13046753525944724` exact input-costate L2-square mass at `4.701742%` area. The
three-seed nonlinear ensemble retained `0.29462633883840517`. Both fail the preregistered `0.47`
bar; the respective shortfalls are `0.33953246474055276` and `0.1753736611615948`.

The nonlinear seed masses are `{0.2773190155117359, 0.27591107023490496,
0.2796787058394356}` with population standard deviation `0.0015544032936474037`, so the stability
gate passes. All 600 unique exact-teacher calls completed with zero retries, so the campaign
economics gate also passes. Neither secondary gate can rescue a failed primary retained-mass gate.

**FAMILY VERDICT:** `KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE`, scoped exactly to
`FAMILY x FEATURE-SOURCE x FIXED REPLAY`. The same-area oracle remains `0.5278150212253758`, so
the result closes the registered cheap feature/head ladder, not the existence of localized support.

## Authority boundary and preregistration

Rounds 2–4 are settled read-only inputs. Round 5 does not rerun their direct-regression or shallow
support-ranking ladders. The standing round-4 comparison is:

- **MEASURED:** best shallow pair-block convex ranker retained
  `0.20172451295048283` exact input-costate L2-square mass at
  `0.047017415364583336` area.
- **PREREGISTERED BAR:** `0.47` retained mass.
- **MEASURED DIAGNOSTIC:** the same-area exact oracle retained
  `0.5278150212253758`; useful localized support exists, but the shallow features did not identify it.

Round 5 sealed its rules in
`experiments/results/replace_round5_deeper_nonlinear_20260713/preregistration.json` before its first
exact-teacher call. SHA-256:
`65d86b4f1cebf8ba0d31810c307f0311012a098beab7bbcc2aaac55dac0c3ad4`.

The sealed order is convex-deeper first, then the fixed nonlinear family; there is no post-heldout
third rung. The replay is the fixed V9 n600 seed455 set with 480 train and 120 heldout states. The
nonlinear dev split is train-only. Its deterministic MLP seeds are `{455,456,457}`, with max 60
epochs, patience 8, hidden width 32, and a heldout seed-population-standard-deviation gate `<=0.03`.
Both primary rungs must meet retained mass `>=0.47` at the exact matched area. The campaign-honest
teacher gate charges every started call and requires `<=600` starts plus all 600 unique completions.

## First derivation duty: honest post-SE cut cost

The receipt measures real runtime tensor shapes, enumerates every executed convolution, and derives
the following FLOP fractions. One multiply-add is one MAC and two FLOPs; forward plus input VJP is
charged symmetrically. Global pooling is charged as its additions/division plus a matching VJP pass.

| Cut | DERIVED cumulative cut/full teacher conv FLOPs | DERIVED global-pool share of cut conv+pool | DERIVED SE-MLP/cut forward conv MACs | Tileability verdict |
|---|---:|---:|---:|---|
| block2-post-SE | `0.04214211147013728` = `4.214211%` | `0.00840326428886221` = `0.840326%` | `0.000011647502438312681` | `not-independently-tileable-after-first-se` |
| block3-post-SE | `0.07129461126470672` = `7.129461%` | `0.00653169427790066` = `0.653169%` | `0.000028898113134391717` | `not-independently-tileable-after-first-se` |

The global-pooling arithmetic is small. Tileability is still lost because downstream values depend
on full-frame global means and SE gates at every included SE boundary. Exact tiled recovery requires
computing and broadcasting those globals first; local tiles cannot be evaluated independently.

This is a convolution-plus-pooling FLOP model, not wall time. Batch normalization, pointwise
activation/sigmoid, decoder interpolation, loss/argmax, localizer matmuls, and autograd bookkeeping
are explicitly omitted. Receipt:
`experiments/results/replace_round5_deeper_nonlinear_20260713/deep_cut_cost_model.json`.

## Formulations and gates

### Convex-deeper rung

The first rung retains the winning round-4 ordered class-pair block structure while replacing the
shallow source with block2/block3 post-SE channels. Twenty pair-specific RankRLS heads use an
implicit all-positive/all-negative objective and the same deterministic exact-support target. Each
head is solved by symmetric-eigendecomposition and a rank-truncated Moore-Penrose inverse; the
receipt carries exact optimum certificates for this registered convex class.

### Nonlinear rung

The new family is a deterministic pair-gated MLP ensemble over the same 116-column deeper
class-pair chart and the same support target. It uses train-derived standardization, a train-only
core/dev split, deterministic batching, epoch-end atomic checkpoints, best-state preservation, and
early stopping. There is no exact-optimum claim. The registered substitutes are the early-stop
trace, three seed-specific heldout masses, their population standard deviation, and the
campaign-honest exact-teacher ledger.

### Query/refuse calibration

The nonlinear ensemble's epistemic disagreement receives a `4%` targeted query budget plus a `1%`
randomized audit budget sampled only outside the targeted set. The audit therefore has positive
propensity. Admission requires both high/low disagreement absolute-error ratio `>=1.25` and positive
Spearman correlation between disagreement and absolute error. This calibration cannot authorize
live use on the fixed replay; it only adjudicates the rescoped research formulation.

## Measurement result

| Rung | MEASURED retained mass at 4.70% area | Gate | Other registered evidence |
|---|---:|---|---|
| convex-deeper-pair-block-mp | `0.13046753525944724` | FAIL by `0.33953246474055276` | all 20 exact optimum certificates pass |
| nonlinear-pair-gated-mlp-ensemble | `0.29462633883840517` | FAIL by `0.1753736611615948` | stability PASS; economics PASS; primary FAIL |
| exact same-area oracle | `0.5278150212253758` | diagnostic | useful support remains present |

The nonlinear ensemble's conditional masked-exact cosine is `0.5427949325835726`, compared with
`0.36120290040287223` for convex-deeper and `0.72650878950318` for the oracle. The nonlinear
family is materially better than this deeper convex formulation, but the preregistered decision
rule is absolute, not a relative-improvement rule.

## Conditional economics

For cut fraction `p`, selected area `q`, exact-teacher acquisition calls `A`, and future steps `D`:

`C_teacher = A + c_label D`, with `c_label = p + (1-p)q`.

Round 5 binds the economics claim to the deepest used feature source, block3-post-SE. **DERIVED:**
with `p=0.07129461126470672`, `q=0.047017415364583336`, and campaign-honest **MEASURED**
`A=600`, `c_label=0.11495993827820083`, the conditional variable-cost ratio is
`8.698682471279858x` and break-even is `D=677.9354132656225` future steps. This remains a
conditional FLOP claim. No sparse exact kernel or wall-clock speedup exists, and because both
localizers fail, `pay-only-on-support` is **NOT ADMITTED**.

## Rescoped ticket folds

### `DIG-S1-BRANCH-AUDIT-HORIZON`

The design is folded as equal exact-call arms `h={0,1,2,4}`, with `h=0` as the mandatory baseline.
Every horizon receives the same exact-call budget, randomized positive-propensity audit rows, and
terminal full-facet evaluation. Any `h>0` advances only if it beats `h=0` on that audited surface.

**IDENTIFICATION VERDICT:** `BLOCKED-NOT-IDENTIFIED` on this fixed replay. It contains no complete
`(Z,A,R,Z')` transition, behavior/audit propensity, or terminal reward/facet custody. Therefore no
MBPO/MVE/STEVE horizon value and no FORE occupancy weight is estimated. The next live run must write
those boundary rows through `src/tac/causal_manifest.py`; missing coverage must fail closed.

### `DIG-S1-QUERY-REAL-CALIBRATION`

The design and finite heldout audit are folded. **MEASURED:** high-disagreement cells have
`189.8129248528991x` the mean absolute error of low-disagreement cells; disagreement/error
Spearman is `0.8656102517385542`; all states pass the registered rank gate; the randomized audit
has positive propensity `0.01042704249231747`; realized total query fraction is
`0.05002848307291666`. Thus the preregistered disagreement error-ranking gate passes and the receipt
records `research-only-calibrated`.

That is not a live trust calibration: ensemble ECE is `0.18620396272803974`, the primary localizer
fails, and the replay is off-policy/transition-incomplete. Live status remains
`refuse-live-research-only-fixed-replay`; an on-policy transition-complete controller with preserved
query and randomized-audit propensities plus explicit probability calibration is still required.

## Verdict scope and reformulation queue

Both rungs fail, so the preregistered escalation is now active:
`FAMILY x FEATURE-SOURCE x FIXED REPLAY`. Shallow pre-SE and registered block2/block3 post-SE
cheap feature sources, across the registered convex and small nonlinear head families, failed to
identify enough localized costate support. This verdict does **not** kill dense labels,
transition-complete FORE, on-policy query/refuse control, a new replay distribution/seed, sparse
exact kernels, or evaluator-equivalent witness successors.

The preregistered queue is:

1. dense-label localizer on the same exact support target;
2. transition-complete FORE successor with `(Z,A,R,Z')`, behavior policy, audit propensity, and
   coverage custody;
3. on-policy query/refuse controller with randomized audit propensities;
4. evaluator-equivalent witness successor if this localization family closes.

## Triality and apparatus wire-in

- **DSL/policy:** `src/tac/witness_dsl/replace_round5_deeper_nonlinear_policy.py` — typed,
  research-only, live-off, finite rung order, gates sealed before measurement.
- **Equation:** `src/tac/canonical_equations/replace_round5_deeper_nonlinear_20260713.py` — retained
  mass, post-SE sparse-teacher economics, equal-call horizons, and positive-propensity query budget.
- **DAG:** `.omx/research/replace_round5_deeper_nonlinear_DAG_FEED_20260713.md`.
- **Formulation:** `src/tac/scorer_surrogate/replace_round5_deeper_nonlinear.py` plus resumable probe
  `tools/probe_replace_round5_deeper_nonlinear.py`.
- **Sensitivity map:** ordered source/competitor blocks, post-SE channels, and exact per-cell costate
  mass are the reusable localization surface.
- **Pareto/bit allocator:** retained mass × area × feature fraction is recorded; bytes/evaluator
  score are unmeasured, so the bit allocator remains non-binding.
- **Cathedral/autopilot:** `REFUSE`; no live/paid/trainer/score dispatch.
- **Continual learning:** final receipt, canonical equation, scoped probe-outcome row, and candidate
  pool updates; shared equation/DAG registry registration remains deferred to main review because
  this landing is explicitly uncommitted.

## Resumability, custody, and storage

The tool saves a complete checkpoint at each renderer-stage boundary, per-epoch nonlinear fit state,
best/final seed state, and atomic aggregate artifacts. Resume verifies the source bundle and sealed
policy. The storage waterfall tried `/Volumes/VertigoDataTier/pact` first, then
`/Volumes/APDataStore/pact`; permission/mount blockers caused the preregistered explicit local
fallback. The run required only 512 MiB and local preflight had sufficient free space. No artifact is
deleted; the completed cleanup manifest preserves per-file bytes and SHA-256 custody.

- Final receipt: `experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json`,
  `88403` bytes, SHA-256
  `38033922bd39cb48f72a154ddd622c41b18be0f137ede56fe4c76873e7bfe98f`.
- Completion marker: `experiments/results/replace_round5_deeper_nonlinear_20260713/complete.json`,
  `292` bytes, SHA-256
  `1c71597d32ec2f62d18ec936a0789bbd4edfdb88b06c1e0e879336ec1a51b4c9`.
- Teacher call ledger: 600 starts, 600 completions, 600 unique states, zero retries; SHA-256
  `80d40f8f37395cece8ebf794859c1582be42eb35da4a197c5705a22f849be02b`.
- Cleanup manifest SHA-256:
  `a4d7057ba8667e75cbebdb511ea6e8946079122afdadbcc49aa3fab694e36744`;
  blockers `[]`.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`.
- `reports/latest.md`, `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`,
  `.omx/state/master_gradient_anchors.jsonl`, `.omx/state/modal_call_id_ledger.jsonl`,
  `.omx/state/cost_band_posterior.jsonl`, `.omx/state/continual_learning_posterior.jsonl`, and
  `.omx/state/probe_outcomes.jsonl`.
- `.omx/research/replace_round3_fidelity_wall_20260713.md`,
  `.omx/research/replace_round4_support_ranking_20260713.md`, and their DAG FEEDs.
- `.omx/research/spinningup_keypapers_crosswalk_20260713.md` ticket charters and every
  `.omx/research/*_directive_*` file dated within the prior 24 hours.
- Latest sister/council/design anchors read at preflight:
  `.omx/research/codex_findings_sfess_oss_reconciliation_20260713T020000Z_codex.md`,
  `.omx/research/codex_session_summary_20260713T1935Z_codex.md`,
  `.omx/research/council_t3_symposium_islands_treatment_arm_20260706.md`, and
  `.omx/research/v9_cgauge_truly_optimal_design_20260712.md`.

## Pointer delta honesty

`NONE`. This is a local research localization measurement, not a byte-closed archive score, contest
CPU/CUDA evaluation, or frontier promotion.
