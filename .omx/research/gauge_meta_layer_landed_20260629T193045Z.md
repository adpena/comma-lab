# GAUGE meta-layer LANDED — the equations→GAUGE→DSL→DAG bridge (DAG FEED-ji)

**UTC:** 2026-06-29T19:30:45Z · **Lane:** extends task #189 (`tac.witness_dsl`) ·
**Cost:** $0 (pure code + tests + wire-in; NO GPU, NO dispatch, NO training) ·
**means≠ends:** this is observability + decision INFRASTRUCTURE; it is NOT a score claim.
The exact contest-CPU pointer **0.19110 is UNMOVED** — only a byte-closed exact eval moves it.
The gauge layer makes the next byte-closed candidate cheaper to FIND and CERTIFY.

## What landed

The GAUGE meta-layer — the layer the DAG↔DSL↔equations triality was missing — built as a
clean $0 extension of `tac.witness_dsl`, parallel to the `Lever`/`with_lever` campaign layer.
It is the OPERATIONALIZATION of the level-set/fiber QUOTIENT codec (task #155): the witness is
a gauge-INVARIANT object (the scorer-equivalence class — all witnesses with the same SegNet
argmax + PoseNet output); equivalent expressions are GAUGES (charts/fiber representatives) with
gauge-DEPENDENT cost; coding the witness = picking the cheapest legal fiber representative.

The 4-layer stack is now complete: **equations (E0-E12, invariant math) → GAUGE (chart choice
+ cost + selection, the bridge) → DSL (program, given a fixed gauge) → DAG (trajectory).**

### Files
- `src/tac/witness_dsl/gauge.py` (NEW) — the meta-layer.
- `src/tac/witness_dsl/curriculum_dsl.py` (EDIT) — added the `gauge` field + `with_gauge`
  method to `WitnessProgram` (pure composition, lazy import to break the cycle; does NOT
  affect `flag_dict()` — the gauge is ABOVE the trainer flags).
- `src/tac/witness_dsl/__init__.py` (EDIT) — exports the gauge surface (50 names total).
- `src/tac/tests/test_witness_gauge_layer.py` (NEW) — **30 tests**; full witness_dsl suite
  (gauge + curriculum + campaign) = **76 green**.

### The API (composes with the existing DSL)
```python
from tac.witness_dsl import BASELINE, CarrierGauge, ResidualGauge, fix_gauge, GaugeComponent
v = fix_gauge(GaugeComponent.CARRIER)         # → GaugeVerdict; v.explain() reads the rule chain
prog = (BASELINE
        .with_gauge(carrier=CarrierGauge.SINGLE_SDF)   # gauge-FIXING step (pure; baseline unmutated)
        .with_lever(...))                              # then the A/B campaign layer
```

## Seeded MEASURED cells vs PENDING cells (NO-FAKE: every number cited; pending = None+provenance)

`default_cost_table()` is the probe-fed ledger. Each cell carries
`{counted_bytes, d_seg_through_R, conditioning, compliant(HARD), deterministic(HARD),
measured, provenance}`. **NO-FAKE invariant (enforced in `GaugeCost.__post_init__`):** a
PENDING (`measured=False`) cell MUST have None numeric fields — a fabricated number cannot
hide in a pending cell.

| component | chart | measured? | seeded cost | provenance |
|---|---|---|---|---|
| carrier | SINGLE_SDF | ✅ | d_seg_R 0.0319 | F1 FEED-iw 0e44b4e8d |
| carrier | HARD_BITMAP | ✅ | d_seg_R 0.166 | F1 0e44b4e8d |
| carrier | MSDF | ⏳ | None | probe a1d5682964 running |
| warp | SCREW_TWIST | ✅ (MEASURED-WIN) | ~0 marginal bytes; d_seg_R None (PRE-R 0.01074/0.01538 advisory) | a513372a FEED-jj 0bbc147b8/0b3e4b1be |
| warp | PER_CLASS_HOMOGRAPHY | ✅ | ≈6600 params; d_seg_R None (PRE-R 0.00950/0.01372, ~85% overfit) | grok-test 2f83e0b9e/FEED-ja + a513372a FEED-jj |
| warp | LEARNED | ⏳ | None | high-cost fallback, unmeasured |
| residual | CONDITIONAL_ON_LANE_PRIOR | ✅ (base) | base d_seg_R 0.00214 + ~5KB coeffs | a99f41f0 389f84f6f + a5b83c730 (Wyner-Ziv head-start; residual-on-top PENDING-GPU) |
| residual | DIRECT_LEARNED | ⏳ | None | GPU run pending (the binding move) |
| residual | ALARD_LUPTON | ⏳ | None | a1da84c b052ab09d (poly-base floor 0.00214 = context) |
| residual | PERSISTENCE_EVENTS | ⏳ | None | event-coded, unmeasured |
| pose | RANGE_DELTA | ✅ | 875 bytes | F4 095ed3e1a + a99f41f0 389f84f6f |
| pose | SCALAR_STORE | ✅ | 4800 bytes | F4 (~5KB sidecar) |
| pose | LOW_RANK | ⏳ | None | task #140 (not byte-closed) |
| movables | STORE | ✅ | 2700 bytes; d_seg 0.0 | F3 FEED-je 930b6d348 |
| movables | WARP_PREDICT | ✅ | d_seg_R 0.00082 | F3 930b6d348 |
| generation | DETERMINISTIC_FREE | ✅ | 0 bytes | rule-118 (README.md:118) |
| generation | LEARNED_COUNTED | ⏳ (det=False) | None | rule-118 counted; deterministic UNTIL certified bit-identical |

Module constant `POLY_BASE_DSEG_FLOOR = 0.00214` (context: the deterministic centerline/poly
base floor; a99f41f0 389f84f6f) — NOT a gauge cell.

## How `fix_gauge` selects (the rule chain, GOSDT-style `.explain()`)
1. drop charts failing a HARD gate (non-compliant OR non-deterministic) → `hard_gate_dropped`;
2. drop PENDING/unmeasured (and measured-but-unrankable) charts → `pending` (so the caller
   knows a probe is needed — can't select what's unmeasured);
3. among the rest minimize the component's S-contribution
   (`S = 100·d_seg_through_R + 25·counted_bytes/37_545_489`; pose is byte-only — the
   √(10·d_pose) term is equal across pose charts so it cancels, bytes decide);
4. deterministic synergy/composition tiebreak by enum-declaration order.

Selected on the seeded table: warp→SCREW_TWIST (~0 bytes beats the 6,600-param overfit
homography; a513372a FEED-jj), carrier→SINGLE_SDF,
residual→CONDITIONAL_ON_LANE_PRIOR (the head-start base; DIRECT_LEARNED is the pending binding
move), pose→RANGE_DELTA, movables→STORE, generation→DETERMINISTIC_FREE. These are
`CANONICAL_GAUGE`. `GaugeChoice.validate()` REJECTS a non-compliant/non-deterministic chart BY
CONSTRUCTION (raises `GaugeViolation`), mirroring the DSL's preserve/contain/authority clauses.

## 6-pillar canonical-helper wire-in
1. **Wired + integrated** — `with_gauge` on `WitnessProgram` (a production type) + standalone
   `compose_gauged_program`; exported from the package `__init__`. (Decision-infra: the
   "production caller" is the campaign-planning surface that composes gauges with levers.)
2. **Tested** — 30 dedicated tests (enum membership, NO-FAKE provenance + pending guards,
   seeded-cell regression guards, BY-CONSTRUCTION rejection, fix_gauge selection/pending/
   hard-gate/tiebreak/none, pure composition, rule-chain readback). Full suite 76 green.
3. **Provenance-routed** — every `GaugeCost` carries a REQUIRED non-empty `provenance` (the
   axis_tag/evidence-grade analogue); pending cells are None+provenance (NO-FAKE).
4. **Memo-anchored** — this memo (+ the 6 unified-Lagrangian hooks below).
5. **Lane-registered** — extends task #189 (`tac.witness_dsl`); no new lane id.
6. **Retroactive-swept** — N/A: this landing introduces NO STRICT preflight gate.

## 6 unified-Lagrangian hooks (this is decision infra — most N/A-with-rationale)
1. **sensitivity-map** — N/A directly, but the GaugeCostTable IS a per-chart score-sensitivity
   ledger (Δd_seg / Δbytes per chart); future wire-in can feed the cathedral sensitivity map.
2. **Pareto constraint** — ACTIVE in spirit: `fix_gauge` minimizes the per-component
   S-contribution under the HARD compliance/determinism constraints (the gauge is the chart
   that lands on the rate/distortion Pareto frontier for its component).
3. **bit-allocator** — N/A (the gauge selects the representation; the DSL levers + trainer do
   the bit allocation). The counted_bytes cells inform a future allocator.
4. **cathedral autopilot dispatch** — N/A (no archive-deployable artifact; emits decisions).
5. **continual-learning posterior** — ACTIVE pattern: the running $0 gauge-probes FILL the
   GaugeCostTable cells (screw/twist a513372a → warp; MSDF a1d5682964 → carrier; openpilot
   a99f41f0/a5b83c730 → residual/pose). Each future probe lands a cell → the ledger compounds.
6. **probe-disambiguator** — ACTIVE: `fix_gauge`'s `pending` list IS the probe-disambiguator —
   it names exactly which charts need a measurement before they can be selected.

## Integration nuances resolved
- **Import cycle** — `gauge.py` imports `curriculum_dsl` at module load; `curriculum_dsl`
  imports `gauge` ONLY inside the `with_gauge` method body (lazy) → no cycle. Verified.
- **Frozen-dataclass field add** — `gauge: object | None = None` is a trailing default field;
  annotated `object` (lazy) so `curriculum_dsl` never imports the gauge module at load. Does
  NOT touch `flag_dict()` → BASELINE round-trip + byte-identity guards stay green (a dedicated
  test asserts `with_gauge` does not leak into trainer flags).
- **Wyner-Ziv head-start (coordinator addition)** — added `ResidualGauge.CONDITIONAL_ON_LANE_PRIOR`
  with an HONEST base-measured/residual-pending cell: the deterministic centerline base is
  measured (0.00214, ~64% lane d_seg recovered, ~5KB coeffs), the LEARNED residual-on-top
  (target ≤1.23e-3) stays `measured=False` PENDING-GPU. The d_seg in the cell is the BASE
  (an upper bound the trained residual lowers) — NOT a fabricated full-chart number.
- **`measured` semantics** — `measured=True` is a LANDED probe (incl. rule-118 definitional
  facts like DETERMINISTIC_FREE=0 bytes); `measured=False` = a running probe / pending GPU
  move. `LEARNED_COUNTED` is seeded `deterministic=False` (fail-closed until certified
  bit-identical) → it is HARD-gate-dropped by `fix_gauge`, a real seeded drop-case.
