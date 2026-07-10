# Eightfold SEAL standing checks — reusable crucible template (2026-07-09)

STORES CONSULTED: design_philosophies_eightfold_20260709 memory (P1-P9 incl. the P9 no-proxy capstone) · operator_no_duplicate_data_archive_geometry_first_20260709 (clauses A/B) · verdict_scope_ladder memory · crucible-2 P6 seal-round history (tac.review_counter crucible2_v752, the 9-finding class record these checks generalize) · DAG FEED-eightfold-philosophies + FEED-p9-no-proxy · t5_crucible3 ORCHESTRATION_LEDGER standing failure-mode checks.

**Operator GO 2026-07-09 "Encode all".** The eight design philosophies
(`~/.claude/.../design_philosophies_eightfold_20260709.md`) become STRUCTURAL:
P1 + P4 are automatable → warn-only preflight gates (`tac.confound_gates`,
`EIGHTFOLD_GATES`); **P2/P5/P6/P7/P8 are fuzzy-by-nature** → they live HERE as
crucible SEAL standing checks. Every future convening (crucible-3 `_v8` onward)
cites this file and runs these five before declaring SEAL. This is a template,
not a one-off — copy the checklist into each convening's seal section.

Twins map (why these five are SEAL-checks not gates): P2/P8 = measurement
honesty · P5 = instrument honesty · P6/P7 = design honesty. None reduces to a
static repo scan; each is a per-claim / per-design judgement a reviewer applies
at seal time.

---

## The five SEAL standing checks (apply to EVERY seal candidate)

**P2 — EVERY COMPARED Δ CARRIES ITS NOISE FLOOR.** For every measured Δ the seal
leans on (a lever's ΔS, an A/B gap, a per-class delta): state its composed noise
floor, OR mark the claim INSTANCE-scoped per the verdict-scope ladder. A Δ below
the instrument's floor is NOT a verdict. SEED-VARIANCE honesty: our deterministic
single-seed spine means across-seed variance is UNKNOWN → a small single-seed Δ
is INSTANCE-level until the floor is measured or bounded.
- **SEAL fails** if any load-bearing Δ has no floor AND no INSTANCE-scope tag.
- Receipts of the absence: OT n600 (0.00314 vs 0.00331, floor unmeasured), R7
  δ_mask ad-hoc, #141 label-noise floor, δ_R=0.0196.
- **Routing:** the noise-floor column requirement is carried into the #385 brief
  spec (the doc) — a noise-floor column beside the dedup + min-dim columns, so
  every future comparison row budgets its floor up front.

**P5 — NO ARM WITHOUT ITS IN-RUN CONTROL.** Every A/B in the seal names BOTH arms
run under identical conditions (matched-compute where capacity matters). Borrowed
baselines are REFUSED — an ancestor-vehicle or prior-run number is a HYPOTHESIS
here, never a control arm.
- **SEAL fails** if any A/B cites a baseline not produced in the same run/matched
  conditions.
- Generalizes L18 (ancestor numbers don't transfer) + R8 toy-isolation +
  P3b-F1's matched-compute control arm.

**P6 — THE SEQUENCE IS THE OBJECT (temporal first-class).** The seal's design has
a TEMPORAL section, OR an explicit temporal-N/A derivation stating why per-frame
is sufficient for this increment. Slot-churn / tracking coherence / GOP-keyframe
structure / dash-phase=ego-distance get an owning design principle, not
facet-by-facet fixes.
- **SEAL fails** if the design is silent on temporal structure (the crucible-3 P3
  blind-spot: all six seats designed per-frame).
- First application owed: the v8 temporal section (SPEC_v8.1 seal check).

**P7 — FALSIFIER BEFORE BUILD.** Every build item in the seal carries a
PRE-REGISTERED kill criterion + a threshold set against a MEASURED baseline
(not a hoped value), named BEFORE the build starts.
- **SEAL fails** if any build item lacks a named falsifier + measured-baseline
  threshold.
- Receipts of the absence: increment-1a got a falsifier only after a red-team
  forced it; the babysat wrong-vehicle runs are the cost of its absence.

**P8 — FLOOR-FIRST.** Every optimized term in the seal states its derived/measured
FLOOR + the gap-to-floor; only the gap is optimized. A term already at floor is
CLOSED to further polish (say so — do not spend the seal polishing it).
- **SEAL fails** if any optimized term lacks a floor + gap statement.
- Receipts: S_floor 0.118, label-noise floor, counted-seed floor — derived when
  cornered, not required up front.

---

## Copy-paste seal checklist

```
[ ] P2  every load-bearing Δ: noise floor stated OR INSTANCE-scoped
[ ] P5  every A/B: both arms in-run/matched; no borrowed baseline
[ ] P6  temporal section present OR explicit temporal-N/A derivation
[ ] P7  every build item: pre-registered falsifier + measured-baseline threshold
[ ] P8  every optimized term: floor + gap-to-floor stated; at-floor terms CLOSED
```

## Sibling apparatus (the automatable two)
- **P1** `check_significance_keys_canonical` (warn-only preflight, `tac.confound_gates`)
  — every relative-significance store key resolves through
  `canonicalize_significance_keys` to a held DSL factory; unresolvable = orphan.
- **P4** `check_witness_control_meters_have_canaries` (warn-only preflight) — every
  witness_control measurement/detector class ships a canary/positive-control;
  heuristic-uncertain classes get an opt-in fmtools advisory (#259 firewall),
  never sole authority.
- Landing memo: `.omx/research/eightfold_apparatus_build_20260709.md`.
