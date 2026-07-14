# Per-class convergence A/B — DAG FEED — 2026-07-14

`research_only=true` · `score_claim=false` · pointer unchanged.

## Feed node

`FEED-PERCLASS-CONVERGENCE-AB-20260714`

Inputs are the four typed real-n600, 150-epoch tickets, one frozen seed/order/EMA/curriculum substrate,
and the existing `ordinal_perclass_convergence_trajectory.v1` analyzer. The decisive edge is A→B:
arithmetic CE versus the exact zero-margin winner/rival hinge. C (step-native HOSC/FINER partition
indicator limit) and D (fp32 M+Adam) are sibling mechanism probes, never substitutes for the matched A/B.

## Triality

DSL:

- `ZeroMarginWinnerRivalHinge` emits only `--seg-loss margin_hinge --margin-target-end 0.0`.
- `StepNativeActivation(basis="step_basis")` emits the repo-native trainable partition-indicator route:
  HOSC beta 1→8 with FINER initialization.
- `MPlusAdam` emits only the additive/multiplicative optimizer discriminator, eta-m, and tau.
- `spec_perclass_convergence_ab_20260714.py` seals four fresh, true-single-stage, real-n600/150 tickets.

DAG:

```text
real n600 cache + frozen seed/order/EMA/curriculum
                 |
          +------+------+------+------+
          | A:CE | B:hinge | C:step | D:M+ |
          +------+------+------+------+
                 | analyzer-ready class x stratum trajectories
                 v
       robust update-axis and wall-axis Theil-Sen rates
                 |
                 +--> rare/common gap closure
                 +--> Lane/Movable hard/easy attribution
                 +--> temporal-spike/non-spike/end-point partition
                 +--> bounded excess above 0.005318 label floor
```

Equations (canonical candidate, becomes measured only after main runs the receipts):

```text
alpha[c,s,a] = -median_{i<j} (d[c,s,a,j] - d[c,s,a,i]) / (u[j] - u[i])
G[a]         = mean_{c in common} alpha[c,all,a] - mean_{c in rare} alpha[c,all,a]
closure[a]   = 1 - G[a] / G[CE]
d_excess[u]  = max(d_seg[u] - 0.005318, 0)
```

`rare={Lane,Movable}` and `common={Road,Undrivable,MyCar}`. The update axis is authoritative for the
matched INSTANCE verdict; the wall axis is the measured cost companion.

## Geometry/representation separation

The receipt exposes two orthogonal surfaces rather than inventing a causal scalar:

- class × all/hard/easy rates isolate the Lane/Movable representation/convergence debt where MCF
  thin-feature erasure can appear;
- the exact GT temporal single-frame-spike/non-spike/end-point contributions and the measured
  `0.005318` smooth-label oracle floor bound what a label-smooth loss/basis/optimizer can recover.

`max(d_seg-0.005318,0)` is an upper bound on curable representation excess, not proof that every pixel
above it is optimizer-curable. There is no measured scalar that causally attributes MCF erasure, so this
feed deliberately does not fake one. A rate win that stalls at the floor is bounded; crossing it requires
an independently demonstrated appearance-phase mechanism.

## Gates and consumers

- Strict memory projection: 24.48 GiB per arm <= 89.6 GiB (0.70×128 GiB).
- Two concurrent arms plus the observed system baseline exceed the adaptive ceiling; sequence A, then B,
  then C/D.
- This builder's dry-starts are `BLOCKED_HOST_NO_METAL`; main owes a green two-pass Metal dry-start before
  the 150-epoch fire.
- Consumer: `tools/probe_ordinal_perclass_convergence.py` for A/B. C/D use the same receipt schema and
  `derive_rates`, but do not get laundered into the analyzer's strict CE-versus-margin verdict.
- The fleet repoint freezes this experiment at exactly A/B/C/D: SPS ep275 is an uninformative
  disengaged instance, not evidence for a fifth arm. Any future SPS admission requires real screw/phase
  engagement plus the converged #121 taper. Adam-beta2/reference-semantics and M+Adam/Muon remain named
  optimizer reformulations; they do not silently expand this matched four-arm harness.
- No bit allocator or autopilot promotion hook is armed before a measured analyzer receipt and exact
  byte-close. The result feeds those consumers only after the existing promotion gates.

## Verdict ladder

Any negative is `INSTANCE` first: this seed, n600 cache, exact treatment, fp32 implementation, and
150-epoch horizon. A naive/first-cut result cannot close even the formulation: it must name the untested
optimal form and leave that reformulation queued. Only an engagement-gated real-n600 optimal-form test
may falsify its exact formulation at this horizon; no result here kills a loss, optimizer, basis family,
or witness paradigm.
