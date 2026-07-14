# DAG FEED — V9 empirical gauge refinement and receiver descent

Date: 2026-07-14  
Feed id: `FEED-v9-cgauge-gauge-refinement-d37-d38-20260714`  
Lane: `warmstart_gauge_symmetry_homotopy`  
Pointer: unchanged; `score_claim=false`; `research_only=true` until exclusive-owner integration.

## Dependency graph

```text
[V9 EMA-best selector + checkpoint]
        + [GT n600 cache]
        + [launch taper + IPE + historical thread law]
        + [canonical NumPy render -> R -> frozen CPU SegNet]
                              |
                              v
             [exact all-pixel selector equality]
                 PASS: 4,107,576 / 117,964,800
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
 [D37 pair-blocked codelength]       [D38 finite restrictions]
 L(F|M,Qxi)-L(F|M,Qxi,C)             pair/triple overlap checks
             |                                 |
   net CI > 0: refinement needed       exact restrictions glue
             |                         but global rate untyped
             v                                 v
 [receiver-derived or charged C?]    [typed cover/restrictions/
       NO -> block bytes              isotropy/filler exists?]
       YES -> pack/parse-back A/B       NO -> block rate
             |                          YES -> charge Theta_U
             +----------------+----------------+
                              |
                              v
       [exclusive-owner default-OFF DSL observation + LawRef]
                              |
              [whole-V9 source closure green?]
                   NO -> integration blocked
                   YES -> verdict-forward consumer only
```

## Node contracts

### G0 — authority custody: PASS

The source-closed r2 run binds selector, checkpoint, GT cache, launch, tool, NumPy forward, R,
SegNet loader/architecture, taper, IPE and thread law by SHA-256. Six-thread/batch-32 execution
reproduces the selector exactly. One-thread near-matches are preserved as refusals.

### G1 — D37 empirical refinement: PASS, scoped

`2,551,382` boundary pixels; `1,066,627` flips. Base `(M,Qxi)` versus refined `(M,Qxi,C)`:
gross `467,373.9089 bits`; table `10,342 B`; net `384,637.9089 bits`; pair-bootstrap 95% interval
`[373,674.7586,395,236.5487]`. Verdict:
`RESIDUAL_NON_GAUGE_STRUCTURE_DETECTED__M_NOT_SUFFICIENT`, scope
`FORMULATION x V9_EMA_BEST_N600_EMPIRICAL_SURFACE`.

### G2 — receiver context custody: BLOCKED

`C` is an assumed directed unlike-neighbour GT class pair. The table is charged; the context
sequence is not. A codec consumer must prove receiver derivation or jointly code `C` before using
the conditional gain.

### G3 — D38 exact local gluing: PASS only at instance

Four exact quadrant restrictions agree on `19,660,800` overlap points across 600 maps. This is
`INSTANCE x EXACT_ARRAY_RESTRICTION`, not a global-section or zero-rate theorem.

### G4 — Bousfield/descent typing: BLOCKED

Owed: a site/cover, edge-tube and junction charts, restriction functors, changing-isotropy band,
finite Cech nerve and receiver-closed filler. Empty `G_U(x)` means infeasible; only nonempty filler
classes can be charged as `Theta_U`.

### G5 — Noether/covariance action: BLOCKED

The discrete continuity helper is built and tested, but the observed flip-mass/zero-flux proxy is
not a Noether charge. Owed: typed affine-Legendre transform pair, executable action momentum,
event/source terms and pre/post action/divergence equality under `(R,xi)` custody.

### G6 — V9 DSL integration: HELD

Exclusive owner request:

- `GaugeSymmetryHomotopyProbePolicy.from_receipt(path, sha256)`;
- default-OFF observational lever;
- held LawRef `v9_empirical_gauge_refinement_d37_v1`;
- consumer is asynchronous/local frozen-SegNet verdict-forward plus provenance bijection;
- no backward/training-loss mutation;
- accept only exact n600 receipt plus explicit context custody and whole-V9 green source closure.

Current status: `V9_INTEGRATION_BLOCKED_EXCLUSIVE_PROVENANCE_OWNER` because the fleet's latest strict
whole-V9 declaration seal is red. Do not weaken that gate.

## Triality

- **Code:** `src/tac/boundary_math/gauge_symmetry_homotopy_20260714.py` plus the resumable n600
  probe and focused tests.
- **DAG:** this feed, with D37 context custody and D38 site/filler typing as hard prerequisites.
- **Equations held, not landed:** `v9_empirical_gauge_refinement_d37_v1`,
  `v9_gauge_covariance_pair_receipt_v1`, and `v9_receiver_descent_section_cost_v1`.
- **DSL held, not landed:** one observation-only receipt lever under the exclusive provenance owner.

## Six-hook disposition

1. Sensitivity map: D37 identifies class-edge-conditioned flip debt, but no pixel actuator is
   admitted until `C` custody closes.
2. Pareto: exact selector equality, context bytes, d_seg/d_pose and archive bytes remain separate.
3. Bit allocator: may consume only receiver-derived/jointly-charged conditional gain, never gross MI.
4. Autopilot: default OFF; no dispatch or training action from this research-only receipt.
5. Continual learning: the positive D37 and the two near-match refusals are durable receipt signal.
6. Probe disambiguator: base, phase-aware and normalized-velocity formulations remain separate; the
   first two reject sufficiency, the overparameterized diagnostic does not after its rate charge.

## Pointer delta

None. This feed adds structural evidence and blockers, not a contest row.

## HISTORICAL_PROVENANCE

The epoch-50 D37 and prior D38 split remain historical inputs. This feed records the authority-
compatible V9 successor without overwriting prior ledgers or hot DAG/equation registries.
