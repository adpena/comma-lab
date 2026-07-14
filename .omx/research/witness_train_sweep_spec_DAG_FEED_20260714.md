# DAG FEED — V9 CGauge witness-training sweep — 2026-07-14

**FEED:** `FEED-V9-WITNESS-TRAIN-SWEEP-20260714`
**Lane:** `witness_train_sweep_spec`
**Status:** `DESIGN_COMPLETE / COMPILE_HELD / NO_LAUNCH`
**Pointer:** submit-ready `0.1910828242 [contest-CPU Linux x86_64]` unchanged; borrowed
`0.1880443980` remains non-submission.

## Nodes

| Node | Evidence/state | Output authority |
|---|---|---|
| `Q_ACTIVATION` | 81 registered duties; top Taper/Horizon/Step = 78.9/47.3/34.2% | ranking signal only |
| `Q_CURRICULUM` | 60 total, 45 owed, 20 built-never-fired | readiness inventory only |
| `DSL_GLOBAL` | 301/407 mapped, 106 unmapped, 0 stale | never-invent-flags gate |
| `V9_BASE_432` | pure compile succeeds; 378 argv tokens, 14 expected levers | scientific control constructor |
| `C0` | fresh immutable #432 control; n600, seed/order frozen, verdict batch32 | exact-row candidate + reference trajectory |
| `V1_TAPER_ISO` | remove existing DsegAwareTaper from C0 | isolated structural contrast |
| `V2_HORIZON_ISO` | measured-band HWM with stage-boundary LawRef weight | isolated support-loss contrast |
| `V3_STEP_ISO` | fresh beta 1→8 activation-basin treatment | isolated activation contrast |
| `V4_AA_SUPER2` | replace IPE with exact 2x supersample coverage | observation-operator contrast |
| `V5_ETF_HEAD` | frozen simplex ETF head | rare-class/head contrast |
| `V6_POLAR_FINISH` | W=QH0 FiLM MCSD/SPEL at typed Muon boundary | terminal optimizer contrast |
| `V7_HORIZON_X_STEP` | conditional stack after V2 and V3 both pass | non-additivity contrast |
| `GOVERNOR_GATE` | three-pass-sealed fix is not landed; serializer `git add` failed `rc128` | launch refuse |
| `TREE_GATE` | shared contested tree not reproducibly serialized | compile/launch refuse |
| `PROVENANCE_GATE` | direct treatment append refuses missing/conflicting LawRefs | compile refuse |
| `OPERATOR_GO` | absent | launch refuse |
| `ORGAN_INGEST` | one immutable treatment-labeled record per real run | predictive multi-run corpus, not causal inflation |
| `BYTE_CLOSE` | EMA/live/Polyak exact receiver candidates | candidate archive/SHA |
| `AUTH_EVAL` | upstream evaluator on exact bytes, CPU/CUDA kept separate | only score authority |
| `POINTER` | submit-ready and borrowed-bank pointers kept separate | moves only on admissible exact row |

## Edges

```text
Q_ACTIVATION + Q_CURRICULUM + current V9 evidence
                      |
                      v
                ranked V1..V7
                      |
DSL_GLOBAL --> exclusive-owner typed variant compile --> PROVENANCE_GATE
                      |                                      |
                      | pass                                 | refuse
                      v                                      v
                    C0/V_i ------------------------------> HELD
                      |
        GOVERNOR_GATE + TREE_GATE + SSD + resume seal + OPERATOR_GO
                      |
                      v
              governed claimed launch
                 /             \
                v               v
      chronological trajectory  terminal checkpoints
                |               |
                v               v
          ORGAN_INGEST       BYTE_CLOSE
                                |
                                v
                            AUTH_EVAL
                                |
                    exact total S and facets
                                |
                +---------------+----------------+
                |                                |
        better admissible row             no improvement
                |                                |
                v                                v
             POINTER                  scoped negative + posterior
```

## Ranked edges and falsifiers

| Rank | Treatment edge | Derived screening band | Exact/mechanism falsifier | Organ regime added |
|---:|---|---|---|---|
| 1 | C0 → TAPER-OFF | `B=[0,0.030]` | taper-ON not better than OFF | spectral saliency allocation |
| 2 | C0 → HORIZON | `B=[0,0.024]` | `B≤0` or no higher-margin survivor shift | reducible horizon/noise split |
| 3 | C0 → STEP | `B=[0,0.013]` | `B≤0`, ring survival unchanged, or saturation | activation stiffness/topology |
| 4 | C0 → AA-SUPER2 | `B_seg=[0,0.2225]` | no Lane lift, `B≤0`, receiver/decode failure | aliasing/coverage |
| 5 | C0 → ETF | `B_seg=[0,0.1395]` | rare classes or exact S fail to improve | minority norm/head geometry |
| 6 | C0 suffix → POLAR | `B_seg=[0,3.5597]` loose ceiling | exact S, tangent, or split-resume fails | terminal manifold geometry |
| 7 | C0 → HORIZON×STEP | `B=[0,0.037]` additive ceiling | fails to beat better single | cross-regime non-additivity |

`B=S_control-S_treatment`; positive is better. Component ceilings never authorize a total-score claim.

## Control and custody law

1. Structural arms V1, V3, V4, and V5 start fresh from the same sealed seed/order/config family.
2. V2 and V6 may fork an immutable stage checkpoint only if the typed resume registry proves complete
   model/optimizer/RNG/EMA/controller identity and the sole delta is the treatment.
3. V7 is unreachable until both V2 and V3 produce positive exact rows and pass their mechanism gates.
4. Every run preserves all stage checkpoints, periodic checkpoints, exact treatment manifest, and
   trajectory record. Pairs/epochs do not masquerade as independent runs.
5. Every terminal candidate receives receiver parse-back, deterministic inflate, exact bytes/SHA, and
   separate contest-CPU/CUDA evaluation. MPS/MLX remains research signal only.

## Held/deferred edges

- `HardnessOversample`: held behind the known consumed-visit truncation repair and fixed-oversample
  weighted-versus-uniform verification.
- `MarginCompandedGroundChart`: held behind V9 IPE compatibility plus counted receiver closure; no hidden
  AA co-treatment.
- D18 truncation and mod19/mod32: terminal byte-close edges on each compatible candidate, not training runs.
- #445: execution/CUDA ticket only until a canonical typed scientific config object exists.

## Triality

- **DSL leg:** held pseudo-DSL variants in the sibling spec; real factories exist, but V9 LawRef/provenance
  declarations and expected-lever manifests must be compiled by the exclusive owner on a clean tree.
- **DAG leg:** this feed.
- **Equation leg:** existing named mechanism laws only; no new effect equation and no contested source edit.
- **Consumers:** costate trajectory ledger after real runs, curriculum/activation measurement ledgers after
  byte-closed verdicts, and exact archive selection after auth evaluation.

## Launch stop

As of this feed, governor, shared-tree, per-variant provenance, storage, deterministic resume seal, and
operator-GO gates are not all green. Therefore every launch edge terminates at `HELD`. Pointer delta is zero.
