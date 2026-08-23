HG1 is complete and landed in commit `1eb31298ecde3e2d3d9a1ce6fe15086a62ddb69c`.

Measured `[macOS-CPU advisory / scorer-free exact byte measurement]`:

| Row | Generators | Residual | Total | Zero-distortion excess |
|---|---:|---:|---:|---:|
| Exact categorical field | 47,667 B | 359,280 B | **460,408 B** | +280,190 B |
| BL1+MS9 protected bracket | 47,667 B | 319,518 B | **420,646 B** | +240,428 B |

The prediction is **CONFIRMED** at `verdict_scope=FORMULATION`. Heterogeneous generators save 75,353 B versus ET1, but both rows remain byte-dead. The protected row’s `d_seg` is **UNMEASURED**; no scorer, Metal, Modal, or upstream write occurred. The own frontier remains DX2 at `S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600]`.

Artifacts:

- [Audit memo](/Users/adpena/Projects/pact/.omx/research/ddm_hg1_heterogeneous_analytic_generator_gate_20260823.md)
- [Resumable runner](/Users/adpena/Projects/pact/experiments/ddm_hg1_heterogeneous_analytic_generator_gate.py)
- [Retained manifest](/Volumes/APDataStore/pact/ddm_hg1_heterogeneous_analytic_generator_gate/manifest.json), SHA-256 `85606aa340ace57c056215ec41d9fa78c263cf0baeefa4916f66680fb621bec6`

Validation covered 60 real-coder rows and deterministic repeats, all 440 retained payloads, exact direct/archive receiver equality, malformed-packet rejection, true checkpoint reuse, two Python review passes, Ruff, and byte compilation.

## NEXT_IF_RESUMED

- **FOLDED** — owner: `MAIN`; consumer store: `.omx/state/main_hot_state.md`; action: consume HG1 as a formulation-scoped current-object closure; fire trigger: commit `1eb31298ec` landed.
- **HOLD** — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_hg1_heterogeneous_analytic_generator_gate/scorer_fire_order/`; action: keep the scorer unfired; fire trigger: a different object-changing receiver first measures below 137,986 B with a distortion-measurement plan.

## LIVE-HYPOTHESES

- A genuine object change may make a closed representation leg useful, consistent with the pinned SY2 composition law.
- A learned implicit evaluator-cell carrier may eliminate substantially more of the dominant 359,280 B residual.
- Curve-relative residual coding may help, but it would need an extraordinary 280,190 B saving before becoming scorer-worthy.

## DEAD-ENDS

- Exact current-DX2 encoding with this heterogeneous roster: 280,190 B over the zero-distortion ceiling.
- BL1-top1 union MS9-error protection: 240,428 B over that ceiling before scoring.
- Coder-only or simple ordering rescue: closed after eight orderings and three real coders.
- Treating BL1 concentration or MS9 error location as measured Seg sensitivity: invalid.
- Firing a scorer on HG1: closed by the byte gate and ownership rules.