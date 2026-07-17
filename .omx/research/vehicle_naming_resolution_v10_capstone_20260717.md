# Vehicle naming resolution + the v10 capstone declaration (operator directive 2026-07-17)

**Operator verbatim:** "I caught you referencing v7.5 and v8 again when the current run is a restarted
v9 cgauge I believe and you started using a new random c1 and c2 notation yesterday. We should advance
to v10 as our capstone frontier and clear up naming resolutions and be very clear that after the
current run and any outstanding A/B we are doing cold start on a fully seeded Kolmogorov optimal
program."

## The canonical version ledger (SoT for vehicle names from this commit forward)

| Name | What it IS | Status |
|---|---|---|
| **v7.5.x** | Optimal-single-trunk SPEC line (SPEC_v75; crucible-2 output v7.5.2/.3). The mod32cap donor checkpoint is v7.5-lineage. | HISTORICAL — design absorbed into v9 |
| **v8** | Per-class-carrier decomposition SPEC (SPEC_v8/v8.1, edge-centric tropical-argmax carriers). | DESIGN LINE — absorbed into v9 ("v9·CGauge is v8 with even more optimal carriers per class") |
| **v9·CGauge** | The composed VEHICLE line: v8 per-class-carrier physics + CGauge/optimal-metric + phase stack + sealed-law composition. | LIVE line |
| **v9c1** (was "c1", #507) | First composed v9 config, FIRED 2026-07-15. | superseded by c2 |
| **v9c2** (was "c2_surgical_warm", #515) | The LIVE run `levelset_n600_witness_20260717T113932Z` — v9·CGauge config WARM-STARTED on the v7.5-lineage mod32cap ep650 trunk (weights-only, fresh AdamW, derived schedule). | **LIVE** (ep~722/1400) |
| **v10** | **THE CAPSTONE FRONTIER: cold start on a fully seeded, Kolmogorov-optimal program.** Fresh-init vehicle, projection-native from birth (see below). | SPEC ASSEMBLY OPEN (task #521) |

**Naming rules (binding):** run configs are tagged `v<vehicle>c<N>` (never bare cN); SPEC docs keep their
historical filenames but every NEW doc/config/task references the canonical ledger name; "the current
run" = v9c2. This file is the naming SoT; sister MEMORY.md line + P0 ledger row
`p0_v10_capstone_cold_start_seeded_20260717`.

## The v10 commitment (operator-binding sequencing)

**After v9c2 completes + the outstanding A/Bs (curvelet matched-bytes p0_497; #518 8-vs-27 warm-up A/B
as a short arm), the next full launch is v10: COLD START, FULLY SEEDED, KOLMOGOROV-OPTIMAL.** No more
warm-starting off ancestor trunks — v9c2's measured results become LAWS and SEEDS, not weights.

v10 = train-least at full force. Everything that a solve/seed/projection produces is NOT trained:
1. **Seeded static classes** — hood/sky born from the measured masks (IoU 0.993/0.976), lane from
   openpilot polynomial priors + per-dash anchors; structured init as the default, not an option.
2. **Head born SOLVED + gauge-fixed** — rank-4 exact head solve at init (#518 ForkHeadSolve machinery),
   sum-zero gauge canonicalization from birth (#519: precision lever, 22.3% finer int8 scale).
3. **range(A)-restricted render targets** (#520) — do not spend capacity on the measured ~52%
   scorer-invisible complement; render in the scorer's sigma-algebra.
4. **Content-priced coder** — the #519-exposed Kolmogorov violation (dense int8 prices shape not
   content) fixed via #336 sensitivity bit-alloc / #461 cross-tensor structure / entropy coding, so
   rate = |program| + |seed| genuinely.
5. **Boundary laws ON from birth** — #518 set (beta2-derived warm-up, engage-boundary registration,
   w_pose ramp, EMA clearance, state persistence) as defaults, not levers.
6. **Store-nothing pose restored** (#314 operator decision, fresh-arm boundary) + joint descent per the
   v9c2 pose-window outcome.
7. **Per-class carriers as built** (v8 increment-1 kit #386) with the basis-cure per the p0_497 A/B verdict.
8. **Seeds from v9c2's terminal state where Kolmogorov-optimal** — e.g. converged self-orient field,
   per-dash anchors, phase carriers — as COUNTED seeds/sections, never as warm weights.

Every v10 constant arrives via LawRef (constants-are-poison); the config is DSL-compiled; the SPEC
carries the completeness table (4th leg) + the twelve-philosophy conformance pass at every scale.
Pointer 0.19108 UNMOVED — v9c2's byte-close/exact-eval remains the near-term pointer attempt; v10 is
the capstone that follows it.
