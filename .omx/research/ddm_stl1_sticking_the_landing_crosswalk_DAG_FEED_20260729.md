# DDM STL1 — Sticking-the-Landing crosswalk DAG / FEED — 2026-07-29

`research_only=true` · `score_claim=false` · pointer 0.1910828242 [contest-CPU] UNMOVED.
Arm: ddm_stl1 ($0 paper crosswalk; NO launches, NO scorer jobs). [no-triality] [p0-ledger-ok].

## FEED — three typed outputs

```text
Roeder/Wu/Duvenaud 2017 (arXiv:1703.09194)  +  DReG 1810.04152 · Geffner-Domke 2007.14634
        │  (recall: score-vs-pathwise dichotomy ALREADY in policy_gradient_variance_reduction_survey
        │   line 140/155; STL variance-at-optimum theorem is the NEW axis)
        ▼
[D2] retro-type P2b MC400-diagonal ES  (receipt on SSD; joint_action −0.0411 = 100% pose √-term,
        d_pose 78.196→77.965 = 0.30% rel; d_seg +2.6e-6 WORSE; 8 trials / 2023 s)
        → LAW CANDIDATE: score-class search variance-dominated & dominated by aimed pathwise
          edits at low-|gradient| plateaus
        → FALSIFIER: an ES/MC row that BEATS aimed pathwise at matched evals at this endpoint
        → consumers: E2 boundary · P2c round-2 budget split · canonical-equations leg (2nd instance)
        ▼
[D3] exact ∂rate/∂Δ decomposition for OUR term −log₂ p((d+u·Δ)/Δ):
        (d+u·Δ)/Δ = d/Δ + u  ⇒ noise Δ-INDEPENDENT ⇒ NO stochastic score term in uniform proxy
        · fixed Δ → pure pathwise (STL no-op)
        · learned Δ, uniform → score term is DETERMINISTIC Jacobian (var 0) → STL biases, don't
        · genuine stochastic score term only under Gaussian entropy-model OR scale-hyperprior
        → DSL lever stub `rate_stl_path_only` DEFAULT OFF (derived), ON only for Gaussian/hyperprior;
          DReG guard if K>1; ordering: waterfill first (Hotz)
        → consumers: gc6 row 8 raced arm · v10 SPEC rate section
        ▼
[D3-route] N1=NO (P3 d_pose 38.06, photometric wall) routes pose-in-burn to v10 re-burn SPEC →
        rate-in-loss arm becomes NATIVE to the re-burn (more likely to fire than E2-extension-only)
        ▼
[D4] N-A sweep: STL N/A for deterministic seg loss · terminal pose GN · costate organ (already
        pathwise) · gradient-free search (algorithm N/A, theorem applies) · combinatorial coding
```

## Consumers (named)
- **E2 boundary** — D2 types the P2b ~0-seg/negligible-pose booking; D3-route amends the N4 fold list.
- **gc6 §4 row 8** (Ballé rate-in-loss raced arm) — D3 lever stub + derived default + DReG guard.
- **v10 SPEC rate section** — the re-burn is now the primary habitat for the row-8 arm (N1=NO).
- **canonical-equations leg / co9 SENSE** — the §2.1 LAW, IF a 2nd plateau instance confirms.
- **P2c round-2 budget** (pb1) — deprioritize blind ES vs atlas-aimed singles at matched Q.

## Anti-pursuits preserved
- Do NOT wire `rate_stl_path_only` ON for the uniform proxy (it BIASES; §3.1) — inert/harmful flag.
- Do NOT re-open the STL crosswalk on deterministic/adjoint surfaces (§4).
- Do NOT claim ES=likelihood-ratio as novel (already corpus; §0).
