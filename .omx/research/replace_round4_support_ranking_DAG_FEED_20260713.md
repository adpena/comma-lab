# Standalone DAG FEED — REPLACE round-4 support ranking

**Date:** 2026-07-13 UTC  
**Lane:** `lane_replace_round4_support_ranking_20260713`  
**Node:** `FEED-95kill-fleet/replace-round4-support-ranking`  
**Status:** `NO_GO_SHALLOW_CHEAP_FEATURE_CONVEX_LOCALIZATION`; `research_only=true`; shared-DAG append `DEFERRED_MAIN`  
**Pointer delta:** `NONE`

## Settled input boundary

Round 3 is a read-only input. It killed direct costate regression across two exact convex formulations and measured a positive same-area oracle: `52.781502%` exact input-costate L2-square mass at realized `4.701742%` prefix area. Round 4 does not rederive that result. It tests the queued ranking, class-pair, and block-head formulation under the same deterministic n600 replay.

## Preregistered edge

```text
sealed V9 n600 replay, checkpoints {ep150, ep251, ep275}, seed455
  -> fixed 480 train / 120 heldout split
  -> exact CPU SegNet input-costate support label once per unique state
  -> exact top-2311 prefix cells by input-costate L2-square mass
  -> cheap local pre-SE prefix + source margin + ordered class-pair chart
  -> RUNG 1: one exact convex weighted-top-k global 84-column head
  -> RUNG 2: twenty exact convex weighted-top-k pair-block 44-column heads
  -> RUNG 3: twenty exact convex implicit all-pairs RankRLS block heads
  -> train-only 16-bin Jeffreys + PAV isotonic calibration
  -> deterministic heldout top-2311 selection for every rung
  -> PASS iff best aggregate retained mass >= 0.47 at matched area
  -> live calibration REFUSE iff winner ECE > 0.05 or selected fallback exists
  -> finite-ladder stop; no post-heldout fourth rung
```

The formulation was sealed in `preregistration.json` before any round-4 exact-teacher call. Its SHA-256 is `ec90adf96b0ec8f239409971f55bb9f5d3f8e442365df772a2ce983d9521c8ff`.

## Closed laws

For score vector `s`, exact costate-cell masses `m_i = ||lambda_i||_2^2`, and deterministic `S_k(s)=TopK(s,k)`:

`rho_k = sum_{i in S_k(s)} m_i / sum_i m_i`.

For the corresponding orthogonal mask `M`, `cos(lambda, M lambda)=sqrt(rho_k)`. Calibration estimates support membership; the primary gate remains exact retained mass.

If a future exact sparse teacher can consume only selected support, the conditional variable-cost coefficient is

`c_label = p + (1-p)q`, with `p=0.005714118050141177` and `q=0.047017415364583336`, hence `c_label=0.05246287035291876` and conditional ratio `19.061099655298698x`.

This is not a realized wall-clock claim. The current EfficientNet-B2 teacher contains global squeeze-excite dependence and has no sparse exact-kernel receipt, so its current exact wall cost remains dense.

## Measurement append

Receipt: `experiments/results/replace_round4_support_ranking_20260713/receipt.json`, `237850` bytes, SHA-256 `6ccbf0e10691dc39c94b77aaefdfe7d9ac3a38b32962bfa5eefcb1107f627222`.

| Rung | MEASURED retained mass | Uplift over uniform | Heldout ECE | Verdict |
|---|---:|---:|---:|---|
| weighted-topk-global-84 | `0.19865776607447305` | `4.225195377798571x` | `0.003096012008190578` | FAIL primary gate |
| weighted-topk-pair-block-44 | `0.19771315378268864` | `4.205104688328304x` | `0.0028016677593879187` | FAIL primary gate |
| pairwise-rank-pair-block-44 | `0.20172451295048283` | `4.290421142597201x` | `0.003073753177168275` | winner; FAIL primary gate |
| exact same-area oracle | `0.5278150212253758` | diagnostic | N/A | useful support exists |

The winner misses the `0.47` bar by `0.26827548704951715`. Calibration passes, so calibration is not the failure mechanism. All populated global and block heads carry certified exact optima for their preregistered rank-truncated float64 Moore-Penrose convex classes.

## Verdict scope and successor edges

Verdict: `FAMILY x FIXED REPLAY` negative for shallow first-pre-SE cheap-feature convex support localizers using this global/pair-block weighted-top-k or pairwise RankRLS ladder on the V9 n600 seed455 replay.

The verdict does not cover:

1. deeper features with separately measured cost and global-state custody;
2. nonlinear or dense-label support learners under the same heldout gate;
3. transition-complete FORE/on-policy disagreement query policies;
4. other replay distributions or seeds;
5. the evaluator-equivalent witness family.

FORE remains `NOT_IDENTIFIED`: zero transitions, decisions, observed rewards, run manifests, or coverage receipts exist for these replay states. No occupancy weights are applied. The composed DIG-S1 policy is `REFUSE_LIVE__RESEARCH_ONLY_FIXED_REPLAY` despite the ECE pass.

## Wire-in

- Sensitivity map: ordered source/competitor blocks, per-pair selected/total exact mass, and class-sensitivity channels.
- Pareto constraint: retained exact mass × selected area × prefix fraction; bytes and evaluator score are unmeasured.
- Bit allocator: non-binding; a future passing localizer can expose selected support as a compute allocation mask.
- Cathedral/autopilot: `REFUSE`; no live, paid, or trainer dispatch.
- Continual learning: byte-closed receipt, canonical equation, scoped advisory probe outcome, and candidate-pool reformulation row.
- Probe disambiguator: the finite three-rung ladder in `tools/probe_replace_round4_support_ranking.py`.

Canonical equation: `tac.canonical_equations.replace_round4_support_ranking_20260713`. Shared equation/DAG registry writes remain deferred to main review because this landing is explicitly uncommitted.
