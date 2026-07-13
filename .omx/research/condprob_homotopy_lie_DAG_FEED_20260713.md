# DAG FEED — conditional temporal transport over topology, Weyl strata, and SE(3)

**Date:** 2026-07-13

**Feed id:** `FEED-condprob-homotopy-lie-temporal-rate-20260713`

**Status:** DESIGN / ANALYSIS; no launch authority

**Verdict:** repaired temporal rate law DERIVED; topology-only event law rejected at formulation
scope; class-conditional flip sufficiency requires one decisive codelength probe

**Pointer:** UNMOVED

## Claim carried by this feed

The temporal source is a marked innovation process conditioned on decoded history, quantized SE(3)
transport, the current refined Weyl stratum, and receiver phase. Homotopy changes are one marked
event family. Any rate or pointer claim must pass conditional model, receiver, exact packed-byte,
and scorer-survival gates.

## Dependency graph

```text
[T0 frozen source/evaluator/R/axis + legal receiver context C_t]
        |
        v
[T1 decoded state W_t + quantized xi source/custody]
        |
        +--> [charge H(Qxi|C) unless receiver-derived/already paid]
        |
        v
[T2 refined stratum sigma=(kappa, orbit/stabilizer, activation, R-phase)]
        |
        +----------------------+-----------------------+
        |                      |                       |
        v                      v                       v
[T3 topology test]      [T4 chart/stabilizer]    [T5 receiver-phase cell]
 discriminant/persist.    admissible arrows        sampled crossing
        |                      |                       |
        +----------------------+-----------------------+
                               |
                               v
                 [T6 preregistered marked event E_pred]
                               |
                +--------------+----------------+
                |                               |
                v                               v
       [T7 regular branch]               [T8 event branch]
       phase + residual                   full mark + geometry
                |                               |
                +---------------+---------------+
                                |
                                v
                 [T9 exact chain-rule/codelength receipt]
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
[T10 flip class-independence probe]      [T11 transported contour coder]
 q0 vs q1 held-out bits                   model+table+container bytes
             |                                     |
             +------------------+------------------+
                                |
                                v
            [T12 receiver parse-back + surviving-flip ledger]
                                |
                     +----------+----------+
                     |                     |
                     v                     v
            [REFUSE if >=0.65 B/flip]  [conditional GO if <bar]
                                             |
                                             v
                         [exact archive/scorer axis -> pointer review]
```

## Node contracts

### T0 — apparatus custody

Bind evaluator/runtime/source hashes, `R` chain, scored-frame cadence, class order, topology complex,
hardware axis, and receiver grammar. Digital and continuous topology must have distinct identifiers.

### T1 — Lie datum custody

Record how `xi` is obtained, quantized, predicted, and made available to the receiver. A Noether
charge or an `SE(3)` symbol does not grant zero bits. Zero rate requires public/already-coded
derivability.

### T2 — common-refinement stratum

Use `sigma=(kappa,omega,a,r)`:

- `kappa`: labeled topology/Morse-Smale signature;
- `omega`: orbit/stabilizer/admissible-arrow type;
- `a`: scorer activation/clamp/tie chart;
- `r`: receiver phase cell.

Never use `kappa` as a complete Weyl-stratum id.

### T3 — topology test

Topology is constant only off the critical discriminant with regular zero sets and transverse
junctions. Track persistence and critical values along preserved tau-rung checkpoints. Tau alone is
not a birth clock because fixed-theta argmax is temperature invariant.

### T4 — chart/stabilizer test

Record atlas changes with topology held fixed: occlusion, nonrigid residual, stabilizer change,
clamp/ReLU/argmax-cell transition, and lost receiver-valid arrow.

### T5 — receiver-phase test

Record lattice/uint8/resize cell crossings separately. L85's `0.005318` spike channel belongs here
unless the declared latent topology test independently fires.

### T6 — event preregistration

Freeze the event predicate before entropy measurement. A binary event must be accompanied by a full
mark grammar. Do not define “event” post hoc as every residual; that makes phase-only sufficiency
tautological.

### T7/T8 — branch coordinates

Prove phase/residual/event coordinates reconstruct the next receiver state. Otherwise report model
cross entropy as an upper-bound design metric, not the exact source entropy.

### T9 — codelength receipt

Report `H(Qxi|C)`, marked-event bits, regular phase bits, regular residual bits, event-branch bits,
model/table overhead, and finite-coder redundancy. Mixture weights are mandatory.

### T10 — single decisive class test

Cross-fit:

```text
q00 = P(flip | margin, xi)
q01 = P(flip | margin, xi, directed_class_pair)
q10 = P(flip | margin, xi, phase)
q11 = P(flip | margin, xi, phase, directed_class_pair)
```

The `q00:q01` contrast answers the stated conditional-independence question; `q10:q11` answers the
receiver-coder question after phase is available. Report held-out log-loss bits, bootstrap
uncertainty, calibration by class pair, and an ablation adding normalized Jacobian/normal velocity.
Do not infer conditional independence from the 120x unconditional class-density spread.

### T11 — transported contour grammar

Condition normal-displacement symbols on transported previous contour and `xi`; reuse existing
anchors; spend new anchors only for marked events; retain literal pixel escape. Compare against
the #307 `0.8201 B/flip` measured row.

### T12 — receiver/economics gate

Count actual archive bytes and only flips surviving parse-back without collateral. Keep both:

- exact physical break-even `1.27311 B/flip`;
- registered engineering admission `0.65 B/flip`.

The current arm uses the stricter registered bar.

## SE(3) coadjoint side branch

```text
[translation-first tac.lie adjoint]
        |
        v
[dual coadjoint action (p,L)->(Rp,RL+t x Rp)]
        |
        v
[Casimirs ||p||^2 and p.L; generic orbit dimension 4]
        |
        +--> [rate-distortion metric / canonical momentum coordinates]
        |
        +--> [NO automatic reduction of 6-DOF pose]
        |
        v
[separate pure-translation/rotation charts + spline/event controls]
```

Orbit invariants admit a coordinate design; they do not determine the empirical `k` in `6+k`.

## Six-hook wire-in

1. **Sensitivity map:** conditional surprise plus normalized class-pair facet scale.
2. **Pareto constraint:** receiver survival, component score value, and exact bytes remain separate.
3. **Bit allocator:** split budget across xi, regular phase, marked events, and escape residual using
   measured conditional marginal bits per score unit.
4. **Cathedral/autopilot:** no actuation until event ontology, topology scale, coordinate chart,
   codelength overhead, receiver section, and storage preflight are complete.
5. **Continual learning:** append class-aware codelength and event-family byte anchors; negatives keep
   their formulation/topology-scale scope.
6. **Probe disambiguator:** class-blind vs class-aware; latent-continuous vs receiver-digital
   topology; phase-only vs phase-plus-residual.

## Triality handoff

- **Equation:** `marked_temporal_transport_rate_law_v1` in the companion equation feed.
- **DAG:** this file.
- **DSL:** proposed temporal-event accounting record in the main memo; no shared DSL file edited.

## Verdict and queue

**DERIVED AFTER REPAIR:** conditional marked temporal rate law.

**PARTIAL:** homotopy is one factor of the Weyl stratification.

**UNKNOWN EMPIRICALLY / NO UNIVERSAL THEOREM:** class independence given margin and xi.

**SINGLE NEXT ACTION:** held-out n600 conditional-codelength probe, then conditional contour bytes
only if the information gain clears its model/table overhead.

**FORMULATION NO-GOS:** unweighted three-term equality; topology-events-equal-all-events; tau-only
birth clock; coadjoint-orbit-implies-`6+k` rate reduction.

Queue: freeze ontology -> fit/cross-fit -> account overhead -> conditional contour model -> exact
receiver pack/parse-back -> registered byte gate -> declared-axis pointer review.
