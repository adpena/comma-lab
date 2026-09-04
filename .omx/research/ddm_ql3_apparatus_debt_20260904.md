---
title: "The C2 lever's three flags were never drift — the registry had no way to state their trainer; and the census found the retired n96 m_safe still live, unlabelled, in two uint8-feasibility harnesses"
arm: ddm_ql3
charter: .omx/research/charters/ddm_ql3_apparatus_debt_20260904.md
charter_commit: a0dba60f4
utc: 2026-09-04
verdict_scope: "[apparatus . no scorer . no Metal . no Modal . NON-PROMOTABLE . moves no pointer]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_ql3 — apparatus debt: the C2 lever binding + the prefix-constant census

## The findings, first

**Item 1.** `completeness().stale` was not reporting drift. It was reporting a **binding the registry
could not express**. The three `--integer-plane-emitter-*` flags are live and owned by the dedicated C2
parser; the lever that emits them simply lived in a module bound to a different trainer. Giving the lever
its own module with a declared `TRAINER_RELPATH` takes `stale` to `[]` — the test passes untouched.

**Item 2.** The census found **two live harnesses still carrying the retired n96-prefix `m_safe`** as an
argparse default, with **no provenance comment at all** — which is exactly why no grep for `n96` had ever
found them. One of them uses it as a **feasibility verdict**, and the error direction is the unsafe one.

Both are landed and self-protected. The exact pointer is untouched by either; this is apparatus.

---

## Item 1 — the C2 `IntegerPlaneEmitter` lever

### The red state (MEASURED at HEAD `a0dba60f4`, `.venv/bin/python -c "…completeness()"`)

```
stale            = ['--integer-plane-emitter-basis', '--integer-plane-emitter-mode',
                    '--integer-plane-emitter-policy-sha256']
coverage_frac    = 0.8194130925507901   (363 of 443)
vehicle_label    = [RETIRED vehicle: train_levelset_witness_realized_through_R_mlx.py]
```

`src/tac/tests/test_lever_registry.py:138` asserts `c.stale == []`, so the file was red.

### What the report actually meant (VERIFIED at source)

`stale = dsl_emitted_flags() − trainer_flags`, and both sides are module-scoped:

- `lever_registry.dsl_emitted_flags` (`src/tac/witness_dsl/lever_registry.py:534-546`) ASTs **one file**,
  `curriculum_dsl.py`, via `_module_source()` (`:101-102`).
- `completeness()` (`:576-604`) grades those emissions against `real_trainer_flags`, i.e. the level-set
  entry point plus the base it imports its primitives from — `curriculum_dsl.py:58-61`
  `TRAINER_RELPATHS`.
- The factory `IntegerPlaneEmitter` lived at `curriculum_dsl.py:2112` and emitted three flags that the
  **C2 band trainer** declares: `src/tac/boundary_math/integer_plane_banded_trainer.py:1619-1621`
  (`build_parser`, three `add_argument("--integer-plane-emitter-…")` lines). `git log -S` finds **0**
  commits in which either level-set trainer carried them.

So the flags were real, the trainer was real, and the grade was a **false FAIL** — the same trade the
registry's own repair note (`lever_registry.py:118-125`) refuses in the other direction.

The registry already has the mechanism to fix this: `module_trainer_paths` (`:176-200`) honours a
module-level `TRAINER_RELPATH` / `TRAINER_RELPATHS`, and `module_declares_trainer` (`:163-175`) exists
precisely so a **declared** binding is distinguishable from a defaulted one. `curriculum_dsl` cannot
declare `TRAINER_RELPATH` for one of its levers — it legitimately needs the plural form for the rest.

### The cure (design-consistent; no side registry)

New module `src/tac/witness_dsl/integer_plane_emitter_lever.py` declaring

```python
TRAINER_RELPATH = "src/tac/boundary_math/integer_plane_banded_trainer.py"
```

with the factory moved verbatim (no behaviour change: same overrides, notes, receipts, policy contract).
`curriculum_dsl` re-exports the name, so every historical import path still works —
`tools/materialize_c2_integer_plane_emitter_fire.py:33`,
`src/tac/witness_dsl/tests/test_integer_plane_emitter_policy.py`,
`src/tac/boundary_math/tests/test_integer_plane_banded_glue.py:41`. Import direction is one-way on
purpose (the lever module never imports `curriculum_dsl` at module scope), so there is no cycle and no
`__getattr__` magic. This is the registry's **own** binding mechanism read by the **same** AST scan every
other lever module goes through — not a hand-typed exception list beside the DSL.

### MEASURED after

```
stale         = []
coverage_frac = 0.8194130925507901   (unchanged — these flags were never on this trainer)
package_lever_factories() row for IntegerPlaneEmitter:
  module='integer_plane_emitter_lever.py'
  trainer='src/tac/boundary_math/integer_plane_banded_trainer.py'
  missing_flags=()  is_stub=False  trainer_declared=True  label_drift=False
re-export identity: tac.witness_dsl.curriculum_dsl.IntegerPlaneEmitter is <the canonical factory> → True
```

`test_332_coverage_rose_from_deorphaning` was **not** touched or loosened.

### `describes_live_vehicle` remains **False** — stated, not fixed

`completeness()` still scopes to `train_levelset_witness_realized_through_R_mlx.py`, and
`Completeness.LIVE_TRAINER_BASENAME` is `train_tr1_partition_renderer_mlx.py`. The default coverage
number (81.94%) therefore describes a **RETIRED** vehicle. Per the charter, this unit does not retarget
`LIVE_TRAINER_BASENAME`; the label is honest and the gap is real, and moving it is a separate decision
because it would change every coverage reading at once.

### Test changes, and why they are not a loosening

`test_registry_and_activation_surfaces_track_required_policy_factory` asserted
`"IntegerPlaneEmitter" in lever_factories()`. That assertion is now **inverted and strengthened**: the
factory must appear in `package_lever_factories()` under `integer_plane_emitter_lever.py` with exactly
those three flags, and must **not** appear in `lever_factories()` — because that curriculum_dsl-scoped
membership *is* the defect. Two new tests pin the cure at both ends:

- `test_c2_lever_module_declares_the_trainer_that_owns_its_flags` — the binding is DECLARED
  (`module_declares_trainer` is True), resolves to exactly one existing trainer, that trainer's argparse
  really declares all three flags, and the factory grades `missing_flags == ()` / not a stub.
- `test_c2_lever_reexport_from_curriculum_dsl_is_the_same_object` — both import paths yield the same
  object, and `completeness().stale == []`.

---

## Item 2 — census of prefix-measured, restriction-scoped constants in live consumers

Law: `tac.canonical_equations` **`annulus_restricted_prefix_bias_detector_v1`**
(`.omx/state/canonical_equations_registry.jsonl:939`; callable
`tac.canonical_equations.annulus_restricted_prefix_bias_detector_20260904:global_check_is_blind`).
dr1 MEASURED: the n96 contiguous prefix biased the **annulus** statistic **+11.698%** while the
**all-pixel** statistic moved **+0.451%** — a **25.94×** amplification, with a bit-identical prefix
positive control, so the deviation is 100% cohort and 0% instrument. A global sanity check has no power
against this. `[macOS-CPU advisory]`, non-promotable.

### Method (reproducible)

Two instruments, because the first has a blind spot the second closes:

1. **Provenance scan** (AST + a ±6-line source window, live `src/tac` + `tools`, excluding `tests/`,
   `test_*`, `results/`): every module-level assignment binding a **measured-looking float** whose window
   names a contiguous-prefix cohort (`n96` / `n24` / `n8` / `gt_n96` / `n_frames: 96` / `96 frames` /
   `contiguous prefix`), then flagged by whether the window also names a **restricted** population
   (annulus / boundary / band / edge / margin / ring / per-class / island / flip / lane).
   **43 hits — 28 restricted, 15 global.**
2. **Value-fingerprint sweep** — grep live code for the retired dr1 literals themselves
   (`0.019590163`, `0.039180326`, `0.03712034`, `0.025631957`, `0.038173675`, `0.01356075`).

**Instrument (1) missed the worst debt, and that is itself the finding.** The two harnesses below carry
the retired value with **no provenance comment**, so their window names no cohort and no restriction —
they are invisible to every provenance grep by construction. Only the value sweep found them.
**An unlabelled constant is not a low-provenance constant; it is an unfindable one.**

### The table

| constant | file:line | population | restricted? | live consumer | verdict |
|---|---|---|---|---|---|
| `DEFAULT_M_SAFE = 0.039180326461791926` | `tools/constructive_inverse_solve_harness.py:48` | n96 contiguous prefix (derived: `headroom × δ_R`) | **YES** — annulus p95 | `--m-safe` default → `_winner_metrics` **feasibility** verdict (`:553-554`) | **SUSPECT → CURED** |
| `DEFAULT_M_SAFE = 0.039180326461791926` | `tools/measure_uint8_lattice_feasibility.py:66` | n96 contiguous prefix | **YES** — annulus p95 | `--m-safe` default → `fragility = mean(margin < m_safe)` (`:371`) → pair selection | **SUSPECT → CURED** |
| `FLIP_MASS_FRACTION_IN_BAND = 0.721` + `BAND_RENDER_ROWS = (160, 240)` | `src/tac/canonical_equations/ddm_b2b_rowband_flip_mass_20260731.py:37,39` | `gt_n96.npz` | **YES** — render-row band [160,240] | **LIVE**: `spec_tr1_renderer_20260728.py:199-218` cites the anchor as the TR1 foveation-band provenance; `qa84_rowband_grammar_20260731.py:34` restates the same rows | **SUSPECT — open** |
| `ANNULUS_THRESHOLD_MARGIN = 2.013` | `src/tac/canonical_equations/evasion_ceiling_fisher_null_20260715.py:62` | n96 margin field | **YES** — "4.7%-area percentile" (an area percentile of a restricted set: the exact statistic dr1 moved) | equation module + `tools/adversarial_evasion_fisher_null_probe.py` | **SUSPECT — open** |
| `G_GAIN = 0.0606` | same file `:60` | n96 field | **YES** — median at a Road–Lane **boundary** anchor | `tools/adversarial_evasion_fisher_null_probe.py:56` (re-derives the same value from the same anchors), `experiments/ddm_mp1_…:93` | **SUSPECT — open** |
| `REALIZATION_FLOOR_DSEG = (0.000465, 0.000929)` | same file `:64` | n96 | **YES** — Lane-dominated (class-restricted) | equation module only | **SUSPECT — low priority** |
| `CHROMA_FLIPS_IN_MARGIN_LT1_ANNULUS = 0.934`, `CHROMA_FLIPS_IN_MARGIN_LT025 = 0.337`, `CONSTANT_LUMA_DESAT_ANNULUS_FLIP = 0.031`, `MARGIN_GRAD_ENERGY_LUMA_FRACTION = 0.788`, `CHROMA_REMOVAL_*_FLIP` | `src/tac/canonical_equations/lane_dash_residual_root_cause_findings_20260703.py:100-105` | n96 | **YES** — margin-annulus / per-class flip | cross-equation: `chroma_boundary_match_20260709.py` | **SUSPECT — open** |
| `CE_TERM_MEAN_EP100`, `BD_RAW_MEAN_EP100`, `BULK_BOUNDARY_GRAD_SHARE_BY_W_EP100` | `src/tac/canonical_equations/boundary_distance_calibration_20260705.py:50-53` | `gt_n24` prefix | **YES** — boundary-distance weighted | equation module only | **SUSPECT — low priority** |
| `ISLAND_WEIGHT_SHARE_BY_GAMMA_EP50`, `ISLAND_GRAD_SHARE_EP50` | `src/tac/canonical_equations/focal_gradient_concentration_20260705.py:52,55` | `gt_n24` prefix | **YES** — island-restricted | equation module only | **SUSPECT — low priority** |
| `FOURIER_ENVELOPE_SPAN`, `CURVELET_ENERGY_CONCENTRATION` | `src/tac/canonical_equations/windowed_curvelet_parabolic_capacity_20260714.py:70-71` | n96 | **YES** — boundary-aligned energy | equation module only | **SUSPECT — low priority** |
| `DPOSE_A1T_BEST_STRATIFIED_TEXTURED = 2.608`, `DPOSE_HPLAN_REAL_SELFFIT = 0.878` | `src/tac/canonical_equations/morse_smale_stratified_parallax_dpose_20260708.py:63,74` | `n24` prefix | **YES** — stratified (textured/planar) | equation module only | **SUSPECT — and prefix bias on POSE is 2.54–4.21× ([[m96]])** |
| `WASTED_GRADIENT_SHARE_AT_TRAINER_DEFAULT = 0.9765` | `src/tac/witness_dsl/hg1_ring0_margin_hinge_levers_20260816.py:59` | **n=96 SEEDED RANDOM** | yes (ring-0 hinge support) | live lever notes | **NOT this class** — a random draw is the law's stated *cure*, not the disease. Sister genus `seed_ensemble_falsifier_band_v1` still applies. |
| `DELTA_R_PROXY_RETIRED = 0.019590163230895963` | `src/tac/inc1a_harness/decoupling_screen.py:48` | n96 | yes | — | **correctly quarantined**: named RETIRED and actively refused by `witness_autoconfig.py:3723-3728` |
| `DELTA_R_N96`, `ALL_PIXEL_P95_N96`, `ANNULUS_AREA_FRAC_N96`, … | `src/tac/canonical_equations/annulus_restricted_prefix_bias_detector_20260904.py:93-99` | n96 | yes | the law itself | **by design** — the "before" side of the comparison |
| `_D_POSE_SIDECAR = 3.4e-05` | `tools/compose_witness_archive.py:80` | n96 window mention | **no** — a global mean over pairs | live composer | **not this class** — but it is the ANCESTOR-vehicle d_pose already flagged in CLAUDE.md; a different, known defect |
| `m_safe` at `subset_selection.py:432`, `curriculum_dsl.py:5221-5222`, `hg1_ring0_…:21`, `margin_band_satisficing_threshold_20260712.py:14,50` | — | n600 | yes | live | **already re-measured (c)** — carry `δ_R = 0.021881818771362305`, `m_safe = 0.04376363754272461` |

Doc-drift, not a value defect: `src/tac/witness_dsl/spec_v9_cgauge.py:1130` still says
`(headroom*delta_R=0.03918)` in a comment while the value is resolved live through the LawRef. Stale
headline over a corrected body ([[m106]]).

### What was CURED, and why here rather than as a fire-condition

No re-measurement was needed: **dr1 already measured the replacement.** The law module, the DSL,
`tac.subset_selection` and the hg1 ring-0 levers all moved to the n600 value on 2026-09-04; these two
harnesses did not, because the literal named no cohort. The remaining work cost 0 minutes of compute, so
writing it down as a "fire-condition" would have been deferral, not deferral-discipline.

The cure is **structural, not a value patch** — both harnesses now resolve the constant instead of
restating it:

```python
DEFAULT_M_SAFE = resolve_margin_band_threshold().m_safe     # → 0.04376363754272461 (MEASURED, verified)
```

`resolve_margin_band_threshold` falls back to the same MEASURED n600 constant when
`reports/delta_R_noise_floor_n600.json` is absent, so it never fails open. A staleness of this class is
now impossible in these two files: there is no number to go stale.

**Why the direction matters.** `m_safe` is a satisficing **TARGET**: push a boundary pixel's margin up to
it, then stop. A target 11.70% too low is **anti-conservative, not merely stale** — in
`constructive_inverse_solve_harness.py` it decides `feasible = gaps.amin(dim=1) >= m_safe` (`:554`), so
the old value declared candidates R-**safe** that the real uint8 noise can still flip. That is a
feasibility verdict, not a summary statistic.

### Self-protection (two gates, positive-controlled)

Landed in `src/tac/canonical_equations/tests/test_margin_band_satisficing_threshold_20260712.py`:

- `test_retired_n96_m_safe_literal_has_no_live_home` — **AST**, not text: the retired literals are
  refused anywhere in live `src/tac` / `tools` / `experiments` as a real numeric constant. Comments and
  docstrings that name the value as history are exempt *because that is what good provenance looks like*.
  The one legal way to keep it live is to **say so in the name** (`*_N96*` / `*RETIRED*` / `*HISTORICAL*`)
  — which is precisely what the two harnesses did not do.
- `test_both_uint8_harnesses_derive_m_safe_from_the_law` — both harnesses must equal
  `resolve_margin_band_threshold().m_safe` exactly.

**POSITIVE CONTROL (MEASURED, not asserted):** reverting `constructive_inverse_solve_harness.py` to the
old literal makes **both** gates fail (`assert 0.039180326461791926 == 0.04376363754272461`); restoring
the cure makes all 40 tests pass. The gates are not vacuous.

### Fire-conditions for the SUSPECTs that remain open

None are re-measured here — each needs a scorer pass over `gt_n600`, which this unit's `$0` /
no-scorer / no-Metal constraint forbids, and the QBR1 chain owns the Metal.

1. **`FLIP_MASS_FRACTION_IN_BAND` + `BAND_RENDER_ROWS` — highest priority, live TR1 vehicle.** FIRE when
   any arm has a `gt_n600` flip-mass pass open. The quantity at risk is not only the 0.721 fraction but
   the **band location**: dr1's mechanism is that the prefix has a different *boundary* population, so a
   flip-mass-vs-row profile from 96 frames can put rows [160,240] in the wrong place, and the TR1
   foveation grammar is built on those rows. Falsifier: recompute the per-row flip-mass profile at n600;
   the band is confirmed if the argmax-mass window moves by ≤ 8 render rows and the in-band fraction
   stays > `FOVEATION_GATE_CRITERION = 0.50`.
2. **`ANNULUS_THRESHOLD_MARGIN = 2.013`.** FIRE with any n600 margin-field pass. It is an **area
   percentile of a restricted set** — the same statistic class dr1 moved (annulus area fraction itself
   moved +4.17%). Falsifier: ±10%, i.e. [1.812, 2.214].
3. **`G_GAIN = 0.0606`.** FIRE together with (2) — same field, same pass. Boundary-anchored median.
4. **The `lane_dash_residual` chroma-annulus family.** FIRE when a chroma arm next opens; it is
   cross-consumed by `chroma_boundary_match_20260709`, so the two must move together.
5. **`morse_smale_stratified_parallax_dpose` n24 pose constants.** Lowest cost/benefit as a re-measure,
   but the **largest** expected bias: prefix bias on the pose axis measures **2.54–4.21× harder** than
   the population ([[m96]]), and n24 is a quarter of the cohort dr1 already falsified. Treat every number
   in that module as order-of-magnitude only until re-measured.
6. **The `gt_n24` boundary/island families** (`boundary_distance_calibration`,
   `focal_gradient_concentration`). No consumer outside their own equation modules today; fire only if a
   lever starts reading them.

### A third straggler, found by the suite: the guard for the cure was RED and asserting the old value

The detached `witness_dsl` suite surfaced two failures in
`src/tac/witness_dsl/tests/test_hg1_ring0_margin_hinge_levers.py`. They are the **same class again**, one
layer up:

```
:231  assert lever_hg1_ring0_margin_hinge().overrides["--margin-target"] == approx(sister.m_safe)
E     assert 0.04376363754272461 == 0.039180326461791926 ± 3.9e-08
:115  assert moved == approx(baseline * 2.0)   "the target did not track the MEASURED artifact"
E     assert 0.07836065292358385 == 0.08752727508544922
```

The lever is **correct** — `hg1_ring0_margin_hinge_levers_20260816.py:48` reads
`reports/delta_R_noise_floor_n600.json`. Three test sites still read
`reports/delta_R_noise_floor.json`, the n96 artifact the law module itself names
`DELTA_R_ARTIFACT_N96_HISTORICAL`. So the test was comparing the live n600 lever against an n96
"sister", and the doubling fixture built its expectation from the n96 file.

**A red guard is an ignored guard — and this one guards the very cure that made it red.** Fixed by
importing `DELTA_R_ARTIFACT` from the lever module instead of restating the filename, at all three
sites. `15 passed` (was 13 passed / 2 failed). Same structural principle as the harnesses: stop writing
the identifier down, resolve it from the owner.

### Suite results (detached, `tools/launch_detached_process.py`)

| run | result |
|---|---|
| `src/tac/witness_dsl/tests` (838.6 s) | 8 failed · 1338 passed · 20 xfailed · 17 errors |
| the 4 failing files, **with** ql3 changes vs **stashed at HEAD** | **8 failed / 41 passed / 17 errors — IDENTICAL both ways** |
| the 4 files after the hg1 artifact fix | **6 failed** / 43 passed / 17 errors |
| 12 sister `src/tac` lever-consumer files (38.3 s) | 2 failed · 239 passed |
| `test_integer_plane_emitter_policy` + `test_lever_registry` + `test_build_completeness_grades` + `test_integer_plane_banded_glue` | 153 passed |
| `test_margin_band_satisficing_threshold_20260712` | 40 passed |

The A/B is the load-bearing row: **every** suite failure reproduces bit-identically with all ql3 changes
stashed, so none is caused by this unit.

**The 6 + 17 that remain have ONE root cause**, and it is not a code defect:

```
DynamicFrontierTargetError: last_refreshed_utc is stale under the canonical 24-hour policy
TaskspaceInverseStackReceiptError: canonical frontier pointer is stale; refresh before admission planning
```

`.omx/state/canonical_frontier_pointer.json` is >24 h old and every taskspace surface fails **closed** on
it, as designed. `tools/refresh_canonical_frontier.py` clears it, but that fetches the upstream
leaderboard and rewrites the pointer — custody this unit must not touch while the QBR1 chain is live.
**Routed to MAIN, not fixed here.**

### The reusable lesson

The census's own first instrument could not see its worst finding. A constant with a rich provenance
comment is *discoverable* and therefore *fixable*; a bare literal is neither. **When a measured constant
is written down without its cohort, the cohort is not merely undocumented — it is unauditable.** The
durable cure is not a better grep; it is to stop writing the number down at all and resolve it from the
law, which is what both harnesses now do.

---

## Scope, honesty, and what this unit did NOT do

- The exact pointer is **unmoved** and cannot be moved by this work. Both items are apparatus.
- **Two more pre-existing failures, NOT caused by this unit** (verified by stashing every change and
  re-running at HEAD — both fail identically there):
  `src/tac/tests/test_eightfold_gates.py::test_p1_repo_live_count_bounded` and
  `src/tac/tests/test_fresh_frequency_shift_dsl.py::test_default_v9_program_does_not_silently_enable_fresh`
  (the latter raises a `CURRICULUM EPOCH-BUDGET FEASIBILITY` violation at `witness_autoconfig.py:2755`
  with `epochs=2`). Reported, not adopted.
- **Scope note, stated plainly.** The charter scoped item 2 to a census plus fire-conditions. Three fixes
  were landed beyond that line — the two harnesses and the hg1 test fixtures. Each cost **zero** compute
  and needed **no** new measurement, because dr1 had already measured the replacement and the code simply
  had not been told. Each is now resolved from its owner rather than restated, so the class cannot
  recur in those files. Everything that would have required a scorer pass is left as a fire-condition,
  exactly as the charter directed.
- `completeness().describes_live_vehicle` is still `False`. Stated above; not retargeted here.
- No re-measurement was performed. The one value that changed was already MEASURED by dr1 and is
  resolved, not restated.
