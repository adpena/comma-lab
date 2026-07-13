# DAG FEED — VR-GHAL theorem-body custody and solver-locus re-adjudication

**Date:** 2026-07-13  
**Mode:** `MEANS`; `research_only=true`  
**Shared DAG append:** `DEFERRED_MAIN`  
**Canonical equation append:** `REGISTERED` as `vrghal_high_probability_fixed_operator_law_v2`
through the locked append helper  
**Deepens:** `.omx/research/vrghal_95kill_fixedpoint_DAG_FEED_20260713.md`  
**Source memo:** `.omx/research/vrghal_theorem_deepen_20260713.md`  
**Pointer delta:** `NONE`

## Node updates

```yaml
- node_id: PAPER_2607_09097_THEOREMS
  prior_status: BLOCKED/UNKNOWN
  new_status: MEASURED
  evidence:
    - arXiv:2607.09097v1 full PDF, Algorithm 3
    - equations (2), (7)-(10), (15)-(17), (20), (29), (33), (36), (38), (39)
  exact_delta:
    - clipping radius is bar_gamma * norm(x-y), with no free c
    - recursive estimator uses MoME epoch anchors plus n_k copies of clipped m_kj minibatch differences
    - anytime bound exposes A_0, A_1, A_2, A_5/2 and beta^k polynomial residual envelope
  custody_limit:
    - epoch threshold retains unspecified O_beta(1)
    - corollary complexities use Otilde and do not expose numeric leading constants

- node_id: PRE_SE_FIXED_REPLAY_CONVEX_SOLVER
  status: MEASURED
  evidence:
    - src/tac/scorer_surrogate/pre_se_locus_20260713.py
    - src/tac/scorer_surrogate/replace_round4_support_ranking.py
    - experiments/results/pre_se_locus_20260713/receipt.json
  facts:
    - fixed n600 replay = 480 cached train targets + 120 fresh heldout targets
    - 20 pair-specific heads at each of two loci
    - 40/40 rank-truncated Moore-Penrose normal-equation optimum certificates
    - no stochastic oracle and no iterative solve

- node_id: VRGHAL_PRE_SE_SOLVER_SELECTION
  status: NO-GO
  verdict_scope: FORMULATION x CURRENT-FIXED-N600-PRE_SE-CONVEX-RUNG x SOLVER-SELECTION
  reason: deterministic sufficient statistics plus certified one-shot MP solve strictly dominate an iterative stochastic fixed-point wrapper
  req_R:
    - direct sufficient statistics/factorization exceed a recorded resource budget
    - deterministic direct or Krylov alternatives fail the same objective certificate
    - only an unbiased stochastic oracle is affordable
    - fixed gamma-Lipschitz map and native-norm variance constants are custodied
    - matched replay shows a wall/query benefit at no worse objective and certificate

- node_id: WITNESS_SGD_FROZEN_STAGE_VRGHAL_CANDIDATE
  status: CONDITIONAL/OPEN
  verdict_scope: METHOD-FAMILY x FROZEN-STAGE-FROZEN-REPLAY-FIXED-LOSS-WITNESS-SGD
  reason: only audited solve that is forced to iterate, but current nonconvex update lacks fixed-map nonexpansivity and oracle-premise proof
  req_R:
    - freeze replay, stage, loss, optimizer geometry, and teacher semantics
    - define one update operator T
    - prove gamma <= 1 on a registered trust region
    - custody unbiasedness, sigma, kappa_E, and bar_gamma in the native norm
    - compare at matched realized-through-R downstream debt

- node_id: HEAVY_TAIL_VRGHAL_COMPLEMENT
  status: COMPOSED/NO-COLLISION
  estimator_leg: .omx/research/heavy_tail_interp_fold_20260713.md
  iteration_leg: .omx/research/vrghal_theorem_deepen_20260713.md
  composition: estimator rare-error tail and iterative residual receive separate failure budgets
```

## Edge updates

```yaml
- from: PAPER_2607_09097_THEOREMS
  to: VRGHAL_FIXED_OPERATOR_ASSUMPTION_GATE
  relation: MEASURED_UNBLOCKS

- from: PRE_SE_FIXED_REPLAY_CONVEX_SOLVER
  to: VRGHAL_PRE_SE_SOLVER_SELECTION
  relation: DIRECT_SOLVE_DOMINATES

- from: TASK_455_MOVING_DISTRIBUTION_MAP
  to: VRGHAL_FIXED_OPERATOR_ASSUMPTION_GATE
  relation: STILL_FAILS

- from: WITNESS_SGD_FROZEN_STAGE_VRGHAL_CANDIDATE
  to: VRGHAL_FIXED_OPERATOR_ASSUMPTION_GATE
  relation: MUST_PROVE_BEFORE_ACTUATION

- from: HEAVY_TAIL_ESTIMATOR_RELIABILITY
  to: HEAVY_TAIL_VRGHAL_COMPLEMENT
  relation: ESTIMATOR_LEG

- from: VRGHAL_ANYTIME_RESIDUAL_BOUND
  to: HEAVY_TAIL_VRGHAL_COMPLEMENT
  relation: ITERATION_LEG
```

## No-stray / triality routing

- **Equation:** `vrghal_high_probability_fixed_operator_law_v2`; supersedes the generic #462
  paper-recursion reconstruction only.
- **DAG:** this FEED is the isolated append-only review artifact; main owns shared DAG ingestion.
- **DSL:** deliberately no edge. This is theorem custody and solver selection, not an actuator.
- **Continual learning:** the direct-solve dominance and theorem-admission gates are machine-readable
  in the canonical equation domain and this FEED.
- **Bit allocator / sensitivity / autopilot:** non-binding because no byte, score, or candidate
  archive delta exists. Consumers must not dispatch VR-GHAL from this node.
