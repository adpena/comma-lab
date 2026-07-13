# DAG FEED — polar-chart Manifold Muon / MuonH disposition — 2026-07-13

**FEED ID:** `FEED-muon-manifold-20260713`  
**Lane:** `lane_muonh_manifold_muon_dig_20260713`  
**Status:** research/design feed; no launch authority  
**Pointer:** unchanged

## Verdict routing

```text
primary-source identity
    ├── Manifold Muon = Stiefel tangent + spectral norm
    │      └── actual V9 module inventory
    │             ├── trunk matrices -> current Muon already matches
    │             ├── film W0 -> exact polar chart W0=Q0 H0
    │             │      └── BUILD exact Manifold Muon on Q only
    │             └── codes/pose/heads/bias/palette -> semantic non-spectral norms
    └── MuonH = Muon Hyperball, not Hessian
           ├── current witness scale-invariance premise absent -> NO-GO formulation
           └── radial/gain diagnostic -> possible future reactivation only
```

## Nodes and gates

| Node | State | Artifact / requirement | Refusal condition |
|---|---|---|---|
| `MM-S0-source` | **COMPLETE** | primary sources and OSS identified in memo §1 | conflicting source identity |
| `MM-S1-inventory` | **COMPLETE** | 12-row V9 module inventory; 87,575 trainable; 59,136 current Muon | shape/count mismatch |
| `MM-S2-static-geometry` | **COMPLETE** | `muonh_manifold_muon_static_probe_20260713.json` | missing checkpoint/hash or nonfinite SVD |
| `MM-E1-equation` | **COMPLETE, UNANCHORED** | `witness_modular_norm_assignment_v1` | missing `s_q` prevents LR budgeting |
| `MM-B0-polar-reference` | **OWED / P0** | deterministic NumPy-fp32 rectangular polar `W0=Q0H0`, reconstruction hash, fold-back | common-boundary tensor/function parity fails |
| `MM-B1-tangent-LMO` | **OWED / P0** | exact Bernstein dual solver, or separately named SPEL approximation | approximate path labelled “exact” |
| `MM-B2-momentum` | **OWED / P0** | tangent momentum projection/transport compatible with tuned warm-start | cold-reset hidden inside treatment |
| `MM-B3-resume` | **OWED / P0** | atomic `Q,H0,dual,momentum,EMA,RNG,stage,epoch` state; per-stage + intra-stage checkpoints | any state process-local or prior stage overwritten |
| `MM-B4-MLX-parity` | **OWED / P0** | MLX vs NumPy parity for polar, tangent residual, LMO, retraction, fold-back | parity below current canonical threshold or missing exact reference |
| `MM-B5-EMA-deploy` | **OWED / P0** | fold `Q_shadow H0` without mutating resume shadow; byte-close load proof | deploy tensor differs without custody |
| `MM-D1-DSL` | **OWED / P0** | typed default-OFF scheduled polar-chart lever; compiler-only argv | invented raw flag or half-wired resume |
| `MM-P0-local-smoke` | **BLOCKED ON B0-B5,D1** | deterministic no-training algebra/unit smoke; optimizer one-step synthetic smoke only | any training dataset/run launch |
| `MM-A1-n600-ticket` | **WIRING_NEEDED / UNMEASURED** | `MANIFOLD_MUON_AB_TICKET`; tuned control vs film-only exact polar-chart treatment | build gates incomplete or governor refusal |
| `MH-P0-radial-gain` | **OWED READ-ONLY** | per-block radial-gradient fraction, stable-rank ratio, explicit function-preserving gain/gauge | no gain or radial signal above measured floor |
| `MH-A1-hyperball-ticket` | **NOT ADMITTED** | only create after `MH-P0` passes and optimal radius/angular schedule is preregistered | generic all-matrix Hyperball |

## Required build order

1. `MM-B0`: NumPy reference and exact common-boundary polar/fold proof.
2. `MM-B1`: exact tangent LMO; keep SPEL under a distinct formulation token if implemented.
3. `MM-B2`: tuned warm-momentum transport.
4. `MM-B3` + `MM-B5`: complete resume and EMA/deploy custody.
5. `MM-B4`: MLX parity against the NumPy authority.
6. `MM-D1`: typed default-OFF DSL compile and resume guard.
7. synthetic/local one-step verification only.
8. separate review/seal.
9. governed n600 A/B only after explicit launch authority and storage preflight.

## A/B edge

```text
cold seed-0 n600 vehicle
    -> exact common pre-Muon checkpoint
       -> common n600 through-R d_seg_start
          ├── CONTROL: tuned ambient Muon on trunk+film
          └── TREATMENT: tuned Muon on trunk + exact tangent Manifold Muon on Q, H0 frozen
                -> first exact crossing d_seg <= 0.98*d_seg_start
                -> right-censor at 250 finisher epochs
                -> compare accepted updates + direct elapsed + d_pose + rate + topology
```

The factor `0.98` is `ASSUMED`; the resulting numeric threshold is `DERIVED` after `d_seg_start`. Missing crossings are `None`. MLX is advisory; the crossing surface is deterministic NumPy-fp32 through actual R plus frozen CPU-torch on all 600 states.

## Composition edges

- `exact Manifold Muon -> #216 signature -> #217 leap-residual` is sequential and permitted after each independent gate.
- `exact Manifold Muon -> TerminalSolve/#423-family` is permitted only if the terminal mutation is accepted in the folded `W=QH0` chart or excludes FiLM; post-solve projection is forbidden as silent mutation.
- `MuonH -> #423` has no curvature edge; MuonH is radial control.
- Newton-Muon has a possible input-covariance edge but owns no ticket in this feed.

## Triality

- **Equation:** `src/tac/canonical_equations/witness_modular_norm_assignment_20260713.py`
- **DAG:** this file
- **DSL:** `WIRING_NEEDED`; existing `--film-stiefel` remains a separately named approximation

## Verdict scopes

- Direct unit projection is `NO-GO` only for finishing retrofit on the named V9 checkpoint.
- Generic MuonH is `NO-GO` only for current unnormalized raw matrix blocks.
- Exact polar-chart Manifold Muon is `GO-BUILD`, not empirically `GO-FIRE`.
- No negative closes gain-decoupled spheres, cold co-designed Stiefel vehicles, SPEL, or future normalized architectures.

## Anti-collision

Own files only. No edits to `condprob_homotopy_lie_dig`, `replace_round2_convexhead`, or `witness_rate_bitalloc_336_respawn` deliverables. No training launch. No shared frontier-pointer mutation.

