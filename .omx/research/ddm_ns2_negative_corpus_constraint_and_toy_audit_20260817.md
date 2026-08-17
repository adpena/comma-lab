---
arm: ddm_ns2
title: "Campaign-wide audit of the negative/mixed corpus for exploitable constraint signal and naive/toy form. Half the corpus cannot constrain today's design (30.1% apparatus checkboxes, 20.4% dead-vehicle numbers). Two live-lineage rows carry real value: r1b7's uint8-survival refusal rests on a treatment that moved d_seg and flip_count by EXACTLY ZERO (a vacuous falsifier), and the UNIWARD cost-map family ratcheted monotonically THROUGH its own pre-registered 0.5 gate to 0.2597 while every row stayed labelled PARTIAL."
utc: 2026-08-17
axis: "[$0 local ledger + retained-receipt arithmetic] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "stated inline per row; this unit issues no new FAMILY verdict"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ns2 — the negative corpus: what it constrains, and what was measured on a toy

**Operator directive 2026-08-17:** *"Audit all negative and mixed results for signal to optimize
against also for naive and toy."*

## Result first

**Half the negative corpus cannot constrain today's design.** Of 216 non-PROCEED probes,
**65 (30.1%) are apparatus checkboxes or degenerate gauges**, and **44 (20.4%) are dead-vehicle
numbers**. Only **107 (49.5%)** are live-lineage experimental negatives. The corpus reads as more
explored than it is.

**Two live-lineage rows hide real value:**

1. **`r1b7` uint8-survival carrier — the falsifier was VACUOUS.** The refusal rests on a treatment
   that moved `d_seg` by **exactly zero** (0.0031795501708984375 → 0.0031795501708984375) and
   `flip_count` by **exactly zero** (10002 → 10002) across **498 selected road/lane sites**. The
   test could not have failed *or* succeeded. Its own reactivation criterion points at the
   **non-binding dial**.
2. **UNIWARD cost-map family — it CROSSED its gate and nobody advanced the verdict.** The family
   ratcheted monotonically 1.1249 → 0.6259 → 0.5673 → 0.5233 → 0.3915 → 0.3599 → **0.2597**
   against a pre-registered gate of **0.5**. It cleared the gate by **1.93×**. Every row is still
   labelled `PARTIAL`, and the stated consequence of clearing (authorize the next stage) never
   fired.

Pointer unmoved. No launch, no dispatch, no spend. This unit is MEANS, not goal progress.

## Coverage, de-dupe, and the denominator

Five prior audits cover this corpus. I read their scope first and went where they did not.

| audit | date | scope it covered | overlap with me |
|---|---|---|---|
| `na7` | 08-14 | MC36 negative corpus; JS8/RX2 routing correction | none |
| `na8` | 08-16 | ledger census (728 raw rows); expiry-as-amnesty pattern | denominator only |
| `ns1` | 08-16 | post-na7 window 08-14 → 08-16; pose anisotropy | none |
| `nx1` | 08-16 | every negative/mixed in the 08-15/16 window (75 files) | none |
| `ra1` | 07-24 | pending-task blocker dissolution (5 reactivate / 9 event-gated / 4 blocked / 30 superseded) | none |

**De-dupe is MEASURED, not asserted.** I grepped all five for `r1b7`, `pc2`, `uniward`,
`wavelet_subband`, `uint8_survival` — **zero hits in any file**. I grepped all five for
gap-to-threshold or vacuous-gauge analysis — **zero hits**. Neither instrument below has been run
on this corpus before.

**Denominators, stated so my negative-existence claims are bounded:**

- `.omx/state/probe_outcomes.jsonl`: **728 rows → 436 distinct probes** (latest-row-wins).
- Non-PROCEED: **216 probes** (126 DEFER · 55 PARTIAL · 20 KILL · 9 INDEPENDENT · 3
  OPERATOR_REVIEW · 2 INFRASTRUCTURE_FAILURE · 1 FALSIFIED_AT_MEASUREMENT_PROTOCOL).
- By month: **May 145 (67.1%)** · Jun 17 · Jul 26 · Aug 28. The recent audits worked the August
  tail; **two thirds of the corpus is May** and had not been re-graded.
- Expired: **172 of 216 (79.6%)**.
- `.omx/research/`: 7,315 `.md`; **358** carry a negative/mixed verdict token in the filename.

**What I did NOT sweep:** the 358-memo research corpus row-by-row (I sampled it by grep only), the
task ledger, `charters/`, and `arm_final_messages/`. I make no claim about those.

## D1 — the CONSTRAINT table (Lens 1: what each wall carves)

New instrument: **gap-to-threshold**. The ledger carries `metric_value` and `threshold`, so the
distance from each wall is computable. A negative that missed by 5% and one that missed by 100× are
different objects and had never been separated.

**137 of 216 (63.4%) carry a computable gap. 79 do not — their shape is unknown and they cannot be
ranked.** That gap-less 36.6% is itself a finding: over a third of our negatives record a verdict
without recording how far from passing it landed.

| band | n | what it means |
|---|---:|---|
| NEAR-MISS (within 1.26×) | 22 | the dial, not the family, is binding |
| CLOSE (1.26–2×) | 11 | one mechanism change plausibly crosses |
| FAR (2–10×) | 28 | needs a different formulation |
| **CHASM (>10×)** | **76** | **honest closure — the family is not the answer** |

**55% of gap-computable negatives are CHASM.** That is a healthy corpus: most of our walls are real
walls. The exploitable population is the **33 rows within 2×**.

### C1 — the UNIWARD cost-map family: a MONOTONE bar that crossed its gate

Sharpening the cost map drives `textured_avg_weight` down monotonically. Measured, `[macOS-CPU
advisory]`, all rows `PARTIAL`:

| formulation | value | gate | verdict |
|---|---:|---:|---|
| HILL filter sister | 1.1249 | 0.5 | DEFER |
| wavelet-subband sharper inversion | 0.6259 | 0.5 | PARTIAL |
| per-class explicit SegNet | 0.5673 | 0.5 | PARTIAL |
| per-segment label SegNet | 0.5233 | 0.5 | PARTIAL |
| per-instance multi-scale (100-pair) | 0.3915 | 0.4412 | PARTIAL |
| per-level wavelet basis selection | 0.3599 | 0.3915 | PARTIAL |
| **per-instance multi-scale wavelet** | **0.2597** | **0.5** | **PARTIAL** |

**SHAPE: monotone descending, 4.33× total, and it cleared the 0.5 gate by 1.93×.** The gate's own
stated consequence — `Tier-2 paid dispatch ... REMAINS GATED on FULL POSITIVE_SIGNAL_SHARPER` — had
its precondition met at 0.2597 and never fired.

**DESIGN IMPLICATION.** `textured_avg_weight` is a **proxy** (a steg cost-map property), not a
score, so crossing it is **not** a ΔS claim and I make none. What it establishes is narrower and
still useful: the *sharpening direction* is real and monotone over 7 formulations, and the live
witness lever `msal_uni` (UNIWARD margin-saliency, in-tree, **default-off**) is the consumer of
exactly this cost map. The proxy→S link is **UNMEASURED**.

### C2 — generic recompression of archive members is AT FLOOR (ceiling constraint)

4 rows, 4 different PR archives, one answer: `best_compression_ratio_on_archive_member = 1.00003`
against a gate of 0.99, `classification=AT_FLOOR`, `deliverable_score_savings_estimate=0.0`
(dominant member 178,417 raw B, best codec brotli).

**SHAPE: flat across four independent archives.** This is a property of the OBJECT (already
entropy-coded members), not of the four PRs.

**DESIGN IMPLICATION — and it is already honored.** Generic coders cannot pay on an entropy-floor
member; only a **learned** model of the stream can. The live RX2/HPAC line is precisely that
response. Per [[m18]], the numbers are dead-vehicle but the **mechanism transfers**, and it is
already consumed. **No action owed.**

### C3 — 498 boundary-site integer edits produce ZERO argmax flips

From `r1b7` (below): 498 selected road/lane sites at fixed magnitude, `flip_count` 10002 → 10002.

**DESIGN IMPLICATION: the per-site MAGNITUDE is binding, not the site COUNT.** 498 sites bought
zero flips for 184 archive bytes. The row's own reactivation criterion asks for *"receiver-composed
per-site collateral"* — a **site-selection** refinement. That is the dial the measurement just
showed is **not** binding. This is the sharpest Lens-1 result in the sweep: **a negative whose
stated cure points at the wrong dial.**

## D2 — the NAIVE/TOY re-grade (Lens 2)

**59 of 216 (27.3%)** negative probes were measured on a reduced instrument (prefix / subset /
smoke / synthetic / single-seed / n≤120). Axis mix: 20 RATE · 13 POSE+RATE · 11 untagged · 5 POSE ·
5 SEG · 3 SEG+RATE · 2 POSE+SEG+RATE.

**20 touch the POSE axis on a reduced instrument.** Per [[m96]], pose prefixes measure **2.54–4.21×
HARDER** than the population while seg measures ≈0.96×. A pose NO-GO drawn on a prefix is exactly
the false-negative shape.

**The dating is the finding.** The prefix-bias law was measured **2026-08-03** (`ddm_na2`).
**15 of those 20 rows predate it** (05-17 → 07-25) and were drawn blind to a bias that runs in the
false-negative direction on their own axis. **The remaining 5 are correctly scoped** — the 08-09
`na5` rows explicitly BLOCK on obtaining a representative n120 rather than concluding. `na5` got
this right and earns no re-grade.

### The re-grade table

| row | reduction | class | re-scoped verdict | settling measurement |
|---|---|---|---|---|
| `r1b7_uint8_survival_carrier_n16` | n16 **and** zero-perturbation treatment | **MECHANISM** | **INSTANCE** — vacuous falsifier; carries no information about uint8 survival | re-run at a magnitude that produces a non-zero flip delta; $0 local |
| UNIWARD family (7 rows) | 100-pair / smoke | **SCOPE** (legal) | **FORMULATION** stands; gate-crossing row is mis-labelled | advance the ledger verdict; then price proxy→S |
| 13 pre-08-03 pose-on-prefix rows | prefix, bias-blind | **SCOPE** with a known-sign bias | **INSTANCE** — not FORMULATION | seeded random n≥120 per [[m96]] |
| `na5` ×5 (08-09) | n120 blocked | none | **correct as written** | already queued |

### The apparatus-checkbox pollution (new pattern, not previously named)

**25 of 137 gap-computable negatives (18.2%) run a DEGENERATE gauge** — `metric_value == threshold`,
or a boolean 0/1 against a 0/1 gate. Examples: `canonical_l0_scaffold_completeness` 1.0 vs 1.0 ·
`apparatus_maintenance_landings_cap` 1.0 vs 1.0 · `design_audit_satisfies_contrarian` 1.0 vs 1.0 ·
`catalog_325_symposium_completion` 0.0 vs 1.0.

**These gauges cannot discriminate.** The verdict did not come from the metric. **23 of 25 are
May 2026** apparatus-era rows. They are compliance checkboxes recorded as experimental negatives,
and they inflate the denominator of "things we tried."

This is the [[VACUITY==PASS]] law read from the other side: a gauge that cannot fail also cannot
*inform*, and a corpus padded with them makes the search space look exhausted.

**Same-shape inflation, measured:** the 4 `option_b` rows are one measurement; the 4 `na5` rows are
one block; the 16 `ddm_vh2_vp1_rank_*` rows (08-10) are one sweep. Counting them as distinct
negatives is the [[same_defect_negatives_masquerade_as_family_convergence]] genus at the ledger
surface.

**Classifier honesty.** The 30.1/20.4/49.5 split is a MEASURED count under a **named regex
heuristic** (apparatus tokens + degenerate gauge; banned-lineage vehicle tokens), not a
hand-adjudicated classification. Treat the split as accurate to a few points, not exact.

## D3 — the ranked REACTIVATION queue

Short by design. Three rows, not thirty.

**R1 — `r1b7` uint8-survival carrier. Re-run at a flip-producing magnitude. $0 local.**
The refusal is INSTANCE-scoped and correct as written, but its evidence is a measured no-op:

```
baseline   d_seg 0.0031795501708984375   flip_count 10002   d_pose 131.15494709357975
treatment  d_seg 0.0031795501708984375   flip_count 10002   d_pose 131.15495020105763
```

`d_seg` identical to the last bit; `flip_count` identical; `d_pose` differs by 3.107e-06 out of
131.15 (**2.37e-08 relative**). Cost: **184 archive bytes**. Reported metric: **−4.29e-07** against
a 0.0 gate — a number indistinguishable from arithmetic noise, recorded as a negative.
Three independent reasons this row cannot speak for the family: n16 scope, a treatment that did not
perturb the scored object, and a metric at noise. Per C3, sweep **magnitude**, not site selection.
Note the baseline `d_pose` of 131.15 — the object was pose-blind, so any pose reading here is
uninformative by construction.

**R2 — advance the UNIWARD ledger verdict, then price proxy→S. $0 then cheap.**
The gate was met at 0.2597 vs 0.5. Two actions, in order: (a) correct the `PARTIAL` label on the
gate-crossing row at source — headline *and* body per [[stale headlines]]; (b) measure what the
cost map is worth in S by firing the in-tree `msal_uni` lever, which is **default-off** and is the
built consumer of this exact map. Step (b) is the one that could move the pointer; step (a) is the
signal-preservation debt.

**R3 — re-grade the 13 pre-08-03 pose-on-prefix rows to INSTANCE. $0, apparatus.**
They were drawn blind to a 2.54–4.21× false-negative bias on their own axis. They should not be
cited as FORMULATION or FAMILY evidence. Most sit on banned-lineage vehicles, so the correct
outcome is usually a scope correction, not a re-run.

## D4 — the honest non-reactivations (mandatory)

An audit that reactivates everything has no discriminating power. These are correctly dead.

- **76 CHASM rows (>10× from their own threshold).** The dial cannot close that distance. This is
  the majority of the gap-computable corpus and it is genuinely closed.
- **C2, archive-member recompression.** AT_FLOOR across four independent archives, savings estimate
  0.0. The constraint is real, and the live RX2/HPAC line already honors it. **No action owed.**
- **The 5 `na5` pose rows (08-09).** Correctly BLOCKED on representativeness rather than concluding.
  `na5` applied the prefix law properly. Nothing to re-grade.
- **44 banned-lineage rows.** Per [[m18]], the numbers are dead. Mechanisms may transfer, and C2 is
  the one that did; the rest carry no live mechanism I could identify.
- **65 apparatus rows.** Not experimental negatives at all. They should be filtered from the
  negative denominator, not reactivated.

## Pre-registered fork — resolution

I pre-registered three outcomes. Two fired.

- **CONSTRAINTS FOUND — YES.** C3 is the strongest: 498 boundary-site edits buy zero flips, so
  magnitude binds and site selection does not. Consumer: any future integer/uint8 carrier design.
  C1 gives a monotone 7-point sharpening trajectory whose consumer (`msal_uni`) is built and off.
- **TOYS FOUND — YES.** `r1b7` is mechanism-reduced (a zero-perturbation treatment), 27.3% of the
  corpus is scope-reduced, and 18.2% of gap-computable rows run a gauge that cannot discriminate.
- **CORPUS CLEAN — NO.** Correctly not fired.

## What this unit is worth

Two instruments that did not exist (gap-to-threshold; degenerate-gauge detection), one corrected
denominator (half the corpus cannot constrain today's design), one vacuous falsifier caught on the
live lineage, and one gate-crossing that was never labelled. **The pointer did not move.** The only
row here that can move it is **R2(b)** — firing `msal_uni` and measuring what the sharpened cost map
is worth in S.

## NEXT_IF_RESUMED

1. **R2(b) first — it is the only pointer-capable row.** Fire the in-tree `msal_uni` lever with the
   per-instance multi-scale wavelet cost map (the 0.2597 formulation) and measure ΔS byte-closed.
   Everything else here is apparatus.
2. **R1** — re-run the r1b7 carrier as a magnitude sweep, $0 local, gated on producing a non-zero
   `flip_count` delta before any verdict is recorded.
3. **R3** — apply the INSTANCE re-grade to the 13 pre-08-03 pose-on-prefix rows, at source, headline
   and body.
4. **Apparatus debt worth landing:** the ledger accepts a `metric_value == threshold` boolean as a
   probe outcome. A writer-side refusal on degenerate gauges would have stopped 25 rows from
   entering the corpus. That is a structural cure, not a procedural one.
5. **Unswept, named so the boundary is honest:** the 358 filename-marked negative memos row-by-row,
   `charters/`, the task ledger, and `arm_final_messages/`.
