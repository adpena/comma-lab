# Standalone DAG FEED — REPLACE round-5 deeper/nonlinear localization

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round5_deeper_nonlinear_20260713`  
**Node:** `FEED-95kill-fleet/replace-round5-deeper-nonlinear`  
**Status:** `KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE`; `research_only=true`; shared-DAG append `DEFERRED_MAIN`  
**Pointer delta:** `NONE`

## Settled input boundary

Rounds 2–4 are read-only inputs. Round 4 measured `0.20172451295048283` retained exact
input-costate L2-square mass at `0.047017415364583336` prefix area for its best shallow
pair-block convex ranker, versus a preregistered `0.47` gate and a `0.5278150212253758`
same-area exact oracle. Round 5 neither re-fits nor re-measures those rungs. It opens the
preregistered post-SE feature-source and nonlinear-head rung on the identical fixed replay.

## Preregistered edge

```text
sealed V9 n600 replay, checkpoints {ep150, ep251, ep275}, seed455
  -> fixed 480 train / 120 heldout split; nonlinear dev is train-only (60 states)
  -> exact CPU SegNet input-costate support label once per unique state
  -> exact top-2311 prefix cells by input-costate L2-square mass
  -> block2-post-SE + block3-post-SE features, source margin, ordered class-pair chart
  -> RUNG 1: twenty exact convex pair-block Moore-Penrose RankRLS heads
  -> RUNG 2: twenty deterministic pair-gated MLP heads, seeds {455,456,457}
  -> nonlinear early stop on train-only dev, max 60 epochs, patience 8
  -> deterministic heldout top-2311 selection for both rungs
  -> primary PASS iff aggregate retained mass >= 0.47 at matched area
  -> nonlinear stability PASS iff heldout seed population std <= 0.03
  -> campaign-honest teacher gate iff started exact calls <= 600 and all n600 complete
  -> query calibration: 4% disagreement-targeted + 1% randomized positive-propensity audit
  -> finite-ladder stop; no post-heldout third feature/head family
```

The decision rule was sealed before the first round-5 teacher call in
`experiments/results/replace_round5_deeper_nonlinear_20260713/preregistration.json`.
SHA-256: `65d86b4f1cebf8ba0d31810c307f0311012a098beab7bbcc2aaac55dac0c3ad4`.

## Deeper-cut cost and tileability edge

The FLOP convention charges one multiply-add as two FLOPs. For a cut with cumulative
forward convolution MACs `M_cut` and full scorer forward MACs `M_full`, the forward-plus-input-VJP
convolution fraction is the same ratio `p_cut=M_cut/M_full`; both numerator and denominator
are charged four FLOPs per forward MAC. Global mean pooling is charged as one forward
reduction plus one matching VJP pass. Pointwise activations, batch normalization, interpolation,
loss/argmax, localizer matmuls, and autograd bookkeeping are deliberately outside this receipt.

| Cut | DERIVED cut fraction of full teacher conv FLOPs | DERIVED global-pool share of cut conv+pool | SE MLP share of cut forward conv MACs | Tileability verdict |
|---|---:|---:|---:|---|
| block2-post-SE | `0.04214211147013728` (`4.214211%`) | `0.00840326428886221` (`0.840326%`) | `0.000011647502438312681` | `not-independently-tileable-after-first-se` |
| block3-post-SE | `0.07129461126470672` (`7.129461%`) | `0.00653169427790066` (`0.653169%`) | `0.000028898113134391717` | `not-independently-tileable-after-first-se` |

The arithmetic overhead of global pooling is small, but the dependency is structural: exact tile
recovery must first compute and broadcast full-frame global means and SE gates at every included
SE boundary. Therefore post-SE tiles cannot be evaluated independently. The durable receipt is
`experiments/results/replace_round5_deeper_nonlinear_20260713/deep_cut_cost_model.json`.

## Closed equations

For score vector `s`, exact cell masses `m_i=||lambda_i||_2^2`, and deterministic
`S_k(s)=TopK(s,k)`, retained mass is

`rho_k = sum_{i in S_k(s)} m_i / sum_i m_i`, and
`cos(lambda,M_{S_k}lambda)=sqrt(rho_k)`.

Conditional deeper sparse-teacher economics use

`C_teacher = A + c_label D`, `c_label = p_cut + (1-p_cut)q`,

where `A` counts every exact-teacher call started during localizer acquisition and `q` is the
selected area. This is a conditional FLOP composition, not a realized sparse-kernel or wall-clock
claim. The post-SE global-state dependency remains explicit.

For equal-call branch comparison, each horizon `h in {0,1,2,4}` receives the same exact-call
budget. Horizon `h>0` may advance only if its audited full-facet error beats `h=0`; otherwise the
result is formulation-scoped and the next horizon remains refused. Query/refuse admission requires
disagreement to rank absolute error plus a randomized positive-propensity audit leg; targeted-only
queries never identify calibration.

## Measurement append

Receipt: `experiments/results/replace_round5_deeper_nonlinear_20260713/receipt.json`, `88403`
bytes, SHA-256 `38033922bd39cb48f72a154ddd622c41b18be0f137ede56fe4c76873e7bfe98f`.

| Rung | MEASURED retained mass at 4.70% area | Registered secondary gates | Verdict |
|---|---:|---|---|
| convex-deeper-pair-block-mp | `0.13046753525944724` | 20/20 exact optimum certificates | FAIL 47% gate |
| nonlinear-pair-gated-mlp-ensemble | `0.29462633883840517` | seed std `0.0015544032936474037` PASS; 600-call economics PASS | FAIL 47% gate |
| exact same-area oracle | `0.5278150212253758` | diagnostic | useful support remains |

The registered nonlinear seeds retain `{0.2773190155117359, 0.27591107023490496,
0.2796787058394356}`. All 600 unique exact-teacher calls completed with zero retries. The final
verdict is `KILL-CHEAP-LOCALIZATION-FAMILY-BY-FEATURE-SOURCE`, scoped to
`FAMILY x FEATURE-SOURCE x FIXED REPLAY`.

For block3-post-SE, `p=0.07129461126470672` and `q=0.047017415364583336` yield
`c_label=0.11495993827820083`, conditional ratio `8.698682471279858x`, and campaign break-even
`D=677.9354132656225` after **MEASURED** `A=600` started calls. Because localization fails,
pay-only-on-support is not admitted; this is not a wall-clock claim.

## Ticket folds and successor edges

- `DIG-S1-BRANCH-AUDIT-HORIZON`: design folded, measurement `BLOCKED-NOT-IDENTIFIED` on the fixed
  replay because it has no `(Z,A,R,Z')`, behavior/audit propensities, or terminal full-facet rows.
  The next run must source those rows from `src/tac/causal_manifest.py`; no FORE weights are applied.
- `DIG-S1-QUERY-REAL-CALIBRATION`: finite disagreement/query/audit design folded. The registered
  error-ranking gate passes: high/low error ratio `189.8129248528991`, disagreement/error Spearman
  `0.8656102517385542`, positive random-audit propensity `0.01042704249231747`, and state pass
  fraction `1.0`. Ensemble ECE is `0.18620396272803974`, the localizer fails, and transitions are
  absent, so this is research-only signal; live use remains refused.

## Wire-in

- Sensitivity map: ordered source/competitor blocks, block2/block3 post-SE channels, and exact
  per-cell costate mass.
- Pareto constraint: retained exact mass × selected area × post-SE feature fraction; archive bytes
  and evaluator score remain unmeasured.
- Bit allocator: non-binding; only a passing localizer can expose support as a compute-allocation mask.
- Cathedral/autopilot: `REFUSE`; no live, paid, trainer, or score dispatch.
- Continual learning: completed receipt, canonical equation, scoped probe outcome, and candidate-pool rows.
- Probe disambiguator: finite convex-deeper then nonlinear ladder in
  `tools/probe_replace_round5_deeper_nonlinear.py`.

Canonical equation: `tac.canonical_equations.replace_round5_deeper_nonlinear_20260713`.
Shared equation/DAG registry writes remain deferred to main review because this landing is explicitly
uncommitted.
