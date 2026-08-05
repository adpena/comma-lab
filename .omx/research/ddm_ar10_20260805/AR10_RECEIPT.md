---
arm: ddm_ar10
title: "arXiv 2410.14326 Jeffreys-centroid proxy crosswalk"
utc: 2026-08-05
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[paper-crosswalk scorer-free]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# AR10 Receipt - arXiv 2410.14326 Crosswalk

## Answer First

Verdict for Pact: **ADOPT Nielsen's Jeffreys-Fisher-Rao categorical center as a head-space
representative rule, not as an RGB prototype or score claim.** Softmax outputs are categoricals, so the
paper gives an immediately computable representative for any set of per-pixel or per-cell frozen-head
probability vectors. The transfer boundary is binding: our scorer head is frozen and is steered only
through input RGB plus the receiver/R path, so a JFR softmax center is a target for a direct RGB solve or
codebook ranking probe, never a shippable pixel value by itself.

No scorer, no launch, no archive mutation, no `upstream/evaluate.py`, and no frontier number moved.
AR9's sister receipt is absent, so the mean-vs-minimax reconciliation is explicitly owed.

## Paper Identification

Sources fetched:

- `https://arxiv.org/abs/2410.14326`
- `https://arxiv.org/pdf/2410.14326`

| field | observed value |
|---|---|
| arXiv id | `2410.14326v1` |
| title | Fast proxy centers for Jeffreys centroids: The Jeffreys-Fisher-Rao and the inductive Gauss-Bregman centers |
| author | Frank Nielsen |
| submitted | 2024-10-18 |
| subjects | Information Theory; Computer Vision and Pattern Recognition; Machine Learning |
| journal reference | Entropy 2024, 26(12), 1008 |
| paper claim | Jeffreys centroids for categorical/normal families lack general closed form; JFR is a Fisher-Rao midpoint of the two sided KL/Bregman centroids and has categorical and normal closed forms; Gauss-Bregman gives a convergent inductive proxy. |

## Deep Read

The paper is about **average-case distribution representatives** under symmetric KL/Jeffreys geometry.
It defines the Jeffreys-Fisher-Rao center as the Fisher-Rao geodesic midpoint between the two sided KL
centroids. For categorical distributions, those sided centroids are the arithmetic probability mean
and the normalized geometric probability mean. The JFR center therefore supplies a closed-form
interpolation between the m-mean and e-mean in the simplex.

The paper also gives a Gauss-Bregman inductive center: initialize with the arithmetic and normalized
geometric means, then iterate an arithmetic update and a normalized geometric update until the two
sequences meet. For categorical sets this is cheap enough to be a fallback or cross-check when the
closed-form JFR center is numerically awkward or support zeros need guarded handling.

The dually-flat framing is the useful join to our existing apparatus. Pact already treats SegNet
softmax space as categorical information geometry: logits/natural coordinates, probabilities/dual
coordinates, categorical Fisher curvature, Bregman/KL divergence, and margin as a Fisher proxy. AR10
adds a concrete representative point between the two sided Bregman centroids. That is a candidate
selection rule inside head space; it does not dissolve receiver realization.

## Ranked Crosswalk

| rank | disposition | paper element | Pact surface | named consumer | falsifier / stop rule | cost |
|---:|---|---|---|---|---|---|
| 1 | ADOPT | Categorical JFR center of softmax distributions. | One real per-stratum codebook/prototype target: RG3/HB1 Fisher-margin per-stratum codebooks, v14/FP1 prototype-color successors, and #869 cell risk representatives. | `hope_bn_capacity_per_stratum_codebook_v1`, RG3 residual-family rows, direct RGB solve/prototype readers. | Fold if a JFR-targeted prototype/RGB solve does not reduce realized frozen-head flips against the current representative on the same cached stratum, or if it cannot survive R/uint8 parse-back. | $0 design now; future probe only on cached head tensors or scorer-owned subset. |
| 2 | ADOPT | JFR = Fisher-Rao midpoint between sided KL centroids. | m65/dual-metric law: never one metric alone; arithmetic probability mean and normalized geometric mean are the two Bregman legs. | `bregman_dual_metric_squared_hessian_v1`, `optimal_metric_unification_v1`, `bregman_dual_metric_guard.py`. | Fold any no-solve dual-Euclidean shortcut; Fisher-natural cotangent still requires typed `H^-1` solve. If JFR collapses to an existing chosen representative in a stratum, mark ALREADY-EMBODIED. | $0 equation/readback hygiene. |
| 3 | ADOPT as fallback | Gauss-Bregman inductive categorical center. | Robust center when closed-form JFR is inconvenient, or as a T-iteration quality knob for a future codebook representative. | Future codebook/prototype selector; #504 centroid/sigma duty ledger. | Fold if T-iteration GB center is indistinguishable from JFR or current centroid on realized flip outcome; refuse if it becomes an unmeasured trainer lever. | $0 CPU on cached distributions. |
| 4 | ALREADY-EMBODIED / sharpen | Categorical Fisher/Bregman geometry. | Existing Fisher-Rao surrogate, Fisher-margin curvature/margin law, and #504 Bregman equations. | `src/tac/losses/core.py::segnet_fisher_rao_per_pixel`, `frozen_scorer_fisher_curvature_margin_colocation_v1`, #504 DAG feed. | Do not replace exact frozen-head margin/prototype custody with paper analogy. Reopen only where a named consumer lacks a representative rule. | Already active. |
| 5 | ADOPT with precondition tags | Mean-vs-minimax distinction. | d_seg is flip-count and can be worst-case inside a cell; JFR is average-case. | Sampled verdict readers, TP1/SL1/GR1 style stratum reads, future AR9 join. | Use JFR for high-margin/interior/soft-codebook strata; require minimax or risk-capped representative for low-margin boundary strata. If AR9 lands a contrary minimax rule, reconcile by consumer surface, not globally. | $0 policy in receipt; AR9 join owed. |
| 6 | N-A / FOLDED as direct RGB solution | A probability simplex center is not an RGB paint value. | v14 and FP1 showed fixed prototype/flat class-field receiver losses through R and frozen SegNet. | None as a shipping row. Use only as target for direct RGB optimization. | FP1 receiver floor `d_seg 0.008305` and v14 projection wall block any claim that a head-space center directly ships. | Closed unless paired with receiver-realized RGB solve. |
| 7 | N-A for bytes | The paper is not a coder, map-price law, or archive format. | #869 adaptive maps, TW1/TZ1/TD1 token-drop and coder surfaces. | `experiments/ddm_tw1_token_waterfill_state_dependence.py`, `experiments/ddm_tz1_token_sweep_rate_attack.py`, TD1 receipt readers. | Any byte claim must come from real coder bytes and same-object d_seg/pose scoring. JFR may rank risk but cannot price tokens. | No action. |

## Probe Design: AR10-JFR-STRATUM

Goal: compare a JFR-center representative against the current representative on **one real stratum
codebook**, without claiming a score.

Fire order:

1. Select one cached stratum with frozen-head probability vectors and existing representative custody:
   prefer an RG3/HB1 Fisher-margin per-stratum codebook or a v14/FP1 prototype-paint stratum with
   saved head outputs. Record pair ids, class-pair/stratum id, selection mode, source hashes, and
   denominator.
2. Compute the arithmetic probability mean and normalized geometric probability mean over that stratum.
   Compute the categorical Fisher-Rao midpoint between them as the JFR target. Compute the GB center as a
   T-iteration cross-check.
3. Compare three head-space representatives: current choice, arithmetic/geometric baseline, and JFR.
   Metrics: within-stratum Jeffreys objective, Fisher wall distance to winner/rival, and predicted
   argmax stability. These are diagnostic only.
4. Realization gate: if a cached RGB/prototype realization surface exists, solve or read the smallest RGB
   parameterization that targets the JFR distribution through the frozen head and R path. Otherwise queue
   the scorer-owned subset rather than running it here.
5. Admit only realized flip improvement on the same stratum and same receiver path. If head-space JFR
   improves the soft objective but does not improve realized flips, disposition is FORMULATION-negative:
   "JFR target not transferable through this realization."

Falsifier: current representative has equal or lower realized flips, or the JFR target requires RGB
changes that exceed the stratum's byte/risk budget. A softmax-only win is not a Pact win.

## Mean-vs-Minimax Reconciliation

AR10 is the **average-case** twin: it chooses a central distribution. Pact's d_seg consumer is often
not average-case at the point of actuation. One misrepresented low-margin boundary pixel can be one hard
flip, and a cell representative that minimizes average KL can still lose the tail that determines score.

Operating rule until AR9 lands:

| consumer surface | representative profile | AR10 use |
|---|---|---|
| high-margin interior paint/codebook | average error matters; flips are unlikely | JFR is admissible as first representative rule. |
| boundary annulus, Lane/Movable birth, low-margin class-pair strata | tail/worst pixel matters; one local miss can score | Use minimax/risk-capped rule first; JFR only as a baseline. |
| #869 adaptive maps | byte state and d_seg risk both bind | JFR can summarize head risk but cannot replace real coder pricing or scorer break-even. |
| direct RGB receiver solve | input-space transfer is the wall | JFR is a target, not a verdict; receiver-realized flips decide. |

AR9 join is owed because `.omx/research/ddm_ar9_20260805/AR9_RECEIPT.md` did not exist at AR10
landing time.

## RECALL EVIDENCE

| source searched | query or lookup | found | impact |
|---|---|---|---|
| AR10 charter prompt and common contract | direct read | AR10 is scorer-free, no launches; receipt path fixed; serializer/tags required; final frontier line fixed. | Kept scope to one markdown receipt and no scorer work. |
| `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | required governing reads | Own-vehicle line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved; AR10 owns no scorer slot. | No score language or promotion claim. |
| arXiv paper | `2410.14326`, abs/pdf | JFR center is the Fisher-Rao midpoint of sided KL centroids; categorical closed form; GB inductive center; dually-flat interpretation. | Classified as representative-rule paper, not coder/renderer. |
| AR9 twin | `test -f .omx/research/ddm_ar9_20260805/AR9_RECEIPT.md` | File absent. | Mean-vs-minimax join recorded as owed. |
| local corpus | `Jeffreys`, `Fisher-Rao`, `Bregman`, `dual-metric`, `rg3`, `Fisher-margin`, `margin-optimal prototype`, `#869`, `#504`, `m65`, `m44` over `.omx/research`, `.omx/state`, docs, src | Found #504 Bregman equations, dual-metric guard, Fisher-margin per-stratum codebook registry, v14/FP1 prototype evidence, #869 token surfaces; no prior AR10 receipt. | Reframed JFR as head-space target with receiver gate; no new equation landed. |
| #504 Bregman receipts | `bregman_all_surfaces_504_DAG_FEED_20260715.md`; `codex_session_summary_20260715_bregman_all_surfaces_504_codex.md` | Bregman categorical metric/covariance and centroid invariants exist; DSL/trainer-consumed Bregman lever is still OWED. | AR10 sharpens #504 but does not claim an implemented lever. |
| dual metric guard | `src/tac/witness_dsl/bregman_dual_metric_guard.py`; canonical equation registry filtered for `bregman|fisher|dual|centroid|codebook|prototype|metric` | Raw dual Euclidean is squared-Hessian only; Fisher-natural cotangent requires typed `H^-1`; relevant equations already registered. | Prevented a no-solve JFR/dual shortcut. |
| v14/FP1 receiver evidence | `ddm_v14_realization_fidelity...`; `ddm_fp1_class_field_projection_20260731.md` | v14 fixed prototype/context projection remains a receiver/R/SegNet projection loss; FP1 perfect GT class-field flat paint has receiver floor `d_seg 0.008305`. | Blocked direct "JFR center as RGB prototype" adoption. |
| #869 token/adaptive-map evidence | `ddm_tw1_token_waterfill_state_dependence_20260801.md`; `ddm_tz1_token_sweep_rate_attack_20260804.md`; `ddm_td1_20260804/td1_verdict.md` | Real token byte prices are state-dependent; adaptive maps have byte legs but scorer-gated/negative projection evidence; TD1 maps are formulation/instance negatives under rt1-calibrated projection. | JFR cannot price bytes; at most it can rank head-space risk before real coder/scorer tests. |
| information-geometry docs | `docs/paper/information_geometric_foundations.md`; `docs/paper/novel_contributions_and_originality_accounting.md` | Nielsen/Amari/Fisher/Bregman tools are already treated as gated design principles; codebook centroid tools not yet built-and-measured. | Classified AR10 as adoptable design principle plus probe, not a contribution or result. |
| AR4 receipt pattern | `.omx/research/ddm_ar4_20260805/AR4_RECEIPT.md` | Recent paper-crosswalk receipt format uses ranked dispositions, explicit N-A rows, recall evidence, and queued fire orders. | Matched format and precondition-tag discipline. |

## What AR10 Did Not Measure

- No SegNet or PoseNet forward pass.
- No `upstream/evaluate.py`.
- No candidate archive, byte-closed row, or receiver packet.
- No n32/n120/n600 Pact measurement.
- No trainer launch, GPU/MLX job, paid dispatch, or lane claim.
- No canonical equation or DSL lever was added.
- No protected file edit.

## Follow-On Disposition

| item | disposition | fire order |
|---|---|---|
| AR10-JFR-STRATUM probe | QUEUED-WITH-FIRE-ORDER | Fire only on one cached real stratum with frozen-head probability custody; if realization needs scorer forwards, hand to a scorer-owning arm. |
| #504 JFR equation candidate | QUEUED-WITH-FIRE-ORDER | Register only after a consumer needs the categorical JFR representative and the probe reports a realized benefit or exact redundancy. |
| AR9 mean-vs-minimax join | QUEUED-WITH-FIRE-ORDER | When `.omx/research/ddm_ar9_20260805/AR9_RECEIPT.md` exists, add a reconciliation paragraph/table mapping average-case JFR vs minimax representative by consumer surface. |
| JFR as RGB prototype or score row | FOLDED | Receiver-realized RGB and exact scorer evidence are required; FP1/v14 already block direct transfer. |
| JFR as #869 byte/coder law | FOLDED | Real coder bytes and scorer break-even remain the only authority. |

## NEXT_IF_RESUMED

```json
{
  "schema": "ddm_ar10_next_if_resumed.v1",
  "status": "PAPER_CROSSWALK_COMPLETE_NO_SCORE",
  "paper": "arXiv:2410.14326 Jeffreys centroid proxies",
  "adopt": [
    "categorical_jfr_center_as_head_space_representative_rule",
    "gauss_bregman_center_as_cached_distribution_fallback",
    "jfr_midpoint_as_em_duality_sharpening_for_504"
  ],
  "already_embodied": [
    "categorical_fisher_margin_geometry",
    "dual_metric_no_shortcut_guard",
    "bregman_centroid_equation_duty_without_trainer_lever"
  ],
  "refuse_as_solution_for": [
    "direct_rgb_prototype_without_receiver_realization",
    "archive_byte_or_token_price_claim",
    "exact_score_or_frontier_move"
  ],
  "owed_join": "read AR9 receipt when present and reconcile average-case JFR against minimax representative by consumer surface",
  "scorer_forwards": 0,
  "evaluate_py": false,
  "archive_mutation": false,
  "pointer_moved": false,
  "follow_on_disposition": "QUEUED-WITH-FIRE-ORDER"
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
