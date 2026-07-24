# DAG FEED — DDM FR1 Fisher actuator base curves

`FEED-ddm-fr1-fisher-actuator-base-curves-20260724` ·
`research_only=true` · `[macOS-CPU frozen-scorer advisory]` ·
`score_claim=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Nodes

```text
N0  #583 Brotli ordering, SHA 765457d4..., 38,077 rows
N1  rank 1 = pair22/cell(225,45), Road->Lane, Fisher/margin VJP row
N2  M1 inner-Jacobian status, SHA 53cca3a5...
N3  realized secants ABSENT; QP receiver closure ABSENT; formalization PENDING
N4  ddm_runtime_sensitivity chart/semantic perturbation API, SHA 0b8cbe97...
N5  rank1 RGB-cell -> runtime-coordinate typed bridge ABSENT
N6  V19C materialized endpoint, 2,923,991 errors, 137,827 B
N7  WS1 W_seg endpoint, 2,845,843 errors, 138,031 B, endpoint != state
N8  WS2 launchable materialized W_seg receipt count = 0
N9  strict FR1 preflight receipt, SHA 877b3d9f...
N10 heavy scorer phase REFUSED; no deltas; base-dependence NO VERDICT
```

## Typed edges

```text
N0 -> N1
N0 -> N2 -> N3 -> N9
N1 -> N5 -> N9
N4 -> N5
N6 -> N9
N7 -> N9
N8 -[fallback only]-> N7
N9 -[execution_allowed=false]-> N10
```

## Stop hooks

- Ordering rank is not execution authority. A VJP pullback without fresh
  candidate-state receiver-closed secant and QP custody stops at N3.
- G2e n16 openpilot rows cannot satisfy N3 for n600 V19C/WS1: base, pair
  population, and trust-region disposition differ.
- A `(pair,row,col,RGB)` row cannot be passed to chart anchors, gradients,
  residuals, or semantic labels without an explicit typed bridge.
- V19C must remain the 2,923,991-error base. The Menu1 joined 8,318,787-error
  row is a different curve and cannot substitute.
- WS1 `W_seg` is an endpoint row until an archive path and hash bind a
  materialized endpoint state.
- No E-line byte delta may enter V19C or WS1 arithmetic.
- Missing deltas remain `NOT_MEASURED`, never numeric zero.

## Reactivation predicate

N10 may reopen only when one receipt binds:

1. the exact rank-1 ordering row;
2. current-base paired signed realized secants;
3. a nonempty class/margin trust region;
4. deterministic QP coefficients;
5. parse-back-identical receiver application with positive hard-oracle margin;
6. exact incremental counted bytes; and
7. the typed bridge consumed by `DDMRuntimePerturbationV1` or its reviewed
   successor.

## Wire-in

- Sensitivity map: rank-1 first-order row remains input-only and inadmissible.
- Pareto/rate: no marginal byte row, so no allocation.
- Bit allocator: reverse-waterfill is stopped before candidate admission.
- Cathedral/autopilot: heavy measurement dispatch false.
- Continual learning: strict preflight plus regression tests preserve the
  missing-custody signal.
- Probe disambiguator: paired signed secants arbitrate amplitude locality versus
  sign failure before QP admission.
