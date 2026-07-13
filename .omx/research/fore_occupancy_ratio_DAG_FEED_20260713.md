# Standalone DAG FEED — FORE occupancy-ratio drift bridge

- Date: 2026-07-13
- Lane: `lane_fore_occupancy_ratio_dig_20260713`
- Node: `FEED-FORE-occupancy-ratio-drift-bridge`
- Status: `DESIGN_ANALYSIS`, `research_only=true`
- Shared-DAG append: `DEFERRED_MAIN`
- Pointer delta: `NONE`

## Parent edges consumed

```text
#455 on-policy nonlinear surrogate
  -- FORMULATION NO-GO: operator/target drift under live reuse --> this node

#462 VR-GHAL fixed-point audit
  -- family-open condition: explicit contraction + admissible hypotheses --> this node

round2 frozen-replay convex head
  -- fixed-distribution GO; 600 cached exact labels; no on-policy claim --> this node

#426/#431/#436 costate organ
  -- recorded-run telemetry, new-run deployment, walk-forward gate --> this node

#468 marked conditional chain rule
  -- temporal conditional-density object --> this node
```

## Derived edge and current refusal

```text
declare full optimizer state Z + schedule action A + transition P
  -> freeze target program pi_j and target time law gamma
  -> fixed replay law nu_R over full (Z,A,Z') transitions
  -> require d0 << nu_R and nu_R P_pi_j << nu_R
  -> fit normalized log-ratio omega_hat with single-level FORE KL objective
  -> KL contraction survives deterministic P when gamma < 1
  -> freeze omega_hat
  -> weighted convex-head fit on cached exact labels
  -> re-derive weighted Hessian contraction
  -> independent target-occupancy exact-label holdout
  -> evaluator-cell gate remains separate
```

Current edge status:

```text
three isolated cold render checkpoints
  -X-> full Markov optimizer state
  -X-> logged state/action/next-state transitions
  -X-> target-policy one-step successor moment
  -X-> target/replay absolute-continuity receipt

therefore:
  NO-GO = current round2 cache as an identified on-policy FORE bridge
  verdict_scope = FORMULATION x CURRENT ROUND-2 INSTANCE
  family status = OPEN after transition/support repair
```

## Canonical math disposition

Candidate identity:

```text
L_pi,gamma(beta)
  = E_{d_pi,gamma}[ell_beta]
  = E_nu[omega_pi,gamma * ell_beta]

D_nu(B_gamma^pi omega || B_gamma^pi omega_tilde)
  <= gamma D_nu(omega || omega_tilde), gamma < 1
```

The deterministic transition is not the fracture: data processing applies to deterministic Markov
kernels, and the strict factor comes from the common discounted reset mixture. At `gamma=1`, a
deterministic injective map is generally only KL-nonexpansive.

Equation registration is `FORMALIZATION_PENDING_NOT_CANONICAL`. Main review must not register the
candidate until a transition-sufficiency and coverage receipt exists.

## Round-2 addressed successor ticket

```text
Arm A: sealed unweighted frozen-replay convex head
Arm B: same head family + fixed FORE occupancy weights

invariants:
  same cached-label bytes where rows overlap
  target program/time law frozen before fit
  full transition and target-action support receipt
  ratio weights frozen before head fit
  weighted Hessian certificate re-derived
  independent target-occupancy exact-label holdout
  teacher calls reconcile to C_teacher = A + c_label*D
```

Inherited same-cache custody is `A=600`, `D=7200`, `c_label=0`, hence 600 calls. Every new target
support/successor/validation teacher state adds to `A_new`; its count is `UNKNOWN` until the ticket
fixes the transition design. Ratio fitting alone does not call the SegNet costate teacher.

## FEED-costate edge

```text
logged organ transition (Z_t, A_t, R_t, Z_t+1)
  -> target schedule pi_sched
  -> omega_pi = d_pi,gamma / d_nu_log
  -> V_hat_FORE(pi) = mean_i omega_hat_pi(Z_i,A_i) R_i
  -> optional doubly robust Q correction
  -> schedule-policy backtest gate
```

Verdict: `CONDITIONAL ADOPT AS ADDITIONAL GATE`; `NO-GO AS CURRENT-GATE REPLACEMENT`.
Only-ratio-realizability removes ratio-class Bellman completeness and critic richness. It does not
remove Markov sufficiency, stable conditional dynamics/rewards, action positivity, initial/one-step
coverage, or cross-run support. Present deterministic single-trajectory logs do not identify unseen
schedule arms.

## #468 composition edge

```text
#468 inner conditional chain surprisal:
  ell_chain = -log p(X|C) -log p(E|X,C)
              -log p(Phi|E,X,C) -log p(Delta^E|Phi,E,X,C)

FORE outer transport:
  R_temporal^pi = E_nu[omega_pi(Z,A) * ell_chain]
```

The ratio is an outer change of measure, not a fifth entropy term and not an archive section. For
the receiver's decoded witness bitstream, optimizer occupancy slots nowhere.

## Triality and wire-in

- DAG: this collision-free standalone feed; shared append deferred.
- Equation: candidate only, registration refused pending transition/support receipts.
- DSL: no live flag or trainer argv; future typed record fields are enumerated in the parent memo.
- Sensitivity: prospective occupancy-weighted costate/renderer-gradient risk.
- Pareto: teacher calls, coverage/ESS/weight debt, transition storage, and evaluator-cell debt.
- Bit allocator: non-binding; possible future outer weighting of marked codelength observations.
- Cathedral/autopilot: no dispatch hook.
- Continual learning: parent memo and this feed preserve the deterministic-contraction correction
  and the current support failure.
- Probe disambiguator: `full_state_action_mdp` versus `fixed_optimizer_actionless_mrp`, decided by
  a transition-sufficiency/coverage receipt.

## Verdict scope and reformulation queue

- Current bridge negative: `FORMULATION x CURRENT ROUND-2 INSTANCE` only.
- Undiscounted deterministic negative: `FORMULATION x gamma=1 x NO-MIXING` only.
- Current organ causal-OPE negative: `FORMULATION x CURRENT SINGLE-TRAJECTORY LOGS` only.
- Reformulate with full resumable transition state, declared target time law, positive action/support
  logging, stage-frozen target program, weighted-head recertification, and independent target holdout.

## STORES CONSULTED

See the parent memo's full list. Primary source: van der Laan and Kallus,
arXiv:2607.05375v1. No live run, scorer, evaluator, archive, provider, GPU, or sibling deliverable was
mutated or launched.
