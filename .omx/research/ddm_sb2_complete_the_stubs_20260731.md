# ddm_sb2 (#819) — build-closure of the optimal configs: the four grades, the vacuous detector, and the reset hook that did not exist

**Scorer-FREE (0 SegNet/PoseNet forwards — window_03 owns the slot; Arm B′ keeps first claim). $0.
Pointer `0.1910828242` [contest-CPU] UNMOVED — this is APPARATUS, and apparatus debt is what made a
strategy argument wrong.** All local rows `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false`.

---

## §0 HEADLINE (answer first)

Three things, in order of how much they change what happens next.

**1. The named NOT-EVEN-DESIGNED instance is real, it is bigger than one hook, and it is now BUILT.**
MAIN predicted arm D± needed an optimizer hook that does not exist. Verified from source, and the gap
is wider than predicted: `experiments/train_tr1_partition_renderer_mlx.py:1543` is
`optimizer = optim.Adam(learning_rate=cfg.lr)` — one line, no betas, no `bias_correction`, no
persistence, no injection point — and **all six `save_checkpoint` call sites pass
`opt_state_flat={}`**, so the moments are never written and never restored. A 64-flag census returns
**zero** matches for `adam|beta|bias|moment|restart|precond|warmup`. Of gc15 §7's five pre-registered
reset arms, **exactly one (arm B, the accidental incumbent) was reachable**; A, B′, C and D± had no
Lever, no flag, and no mechanism anywhere. `tac.optimization.reset_operator` now closes A/B/B′/C
outright and supplies D±'s injection contract — **one module unblocks the whole reset race**, which is
the BOTH-BRANCHES rule satisfied for the campaign's biggest live decision.

**2. The detector that was supposed to find this class was scanning 0.6% of its domain.**
`lever_registry._module_source()` read ONE file. Repaired to a package-wide scan with **per-module
trainer resolution** — and that qualifier is load-bearing: a naive widening reports every tr1 flag as
"stale" against the levelset trainer, i.e. it swaps a vacuous PASS for a false FAIL. Delta:

| | before | after |
|---|---|---|
| modules in scope | 1 of 171 (0.6%) | 170 globbed, **16 with factories** |
| lever factories visible | 116 | **177** (+61, **+52.6%**) |
| DESIGNED-STUBs detectable | **0 — structurally impossible** | **10** |
| of which SILENT (no marker at all) | — | **2** |

**3. The closure audit found a FIFTH failure mode that neither the mandate nor my gate anticipated,
and it is the dominant one.** The four-grade taxonomy assumes debt is *missing*. The real Path-A/B
debt is **BUILT-ELSEWHERE-UNWIRED-HERE**: eight components that are fully built, tested and fired on
the levelset or torch vehicle and have **zero** tr1 wiring. A stub sweep marks every one of them
GREEN, because a mechanism does exist — somewhere. My own gate cannot see them either, by construction
(it resolves each module against its own trainer). Naming it is this arm's most useful finding.

---

## §1 PROVENANCE

| item | value |
|---|---|
| venv | `/Users/adpena/Projects/pact/src/tac/__init__.py` — hijack check **CLEAN** |
| commit | `c44c7565af` (6 files, +1669/−5) |
| scorer jobs | **0** |
| tests | 52 new, all passing; ruff `--select F` clean |
| pointer | **`0.1910828242` [contest-CPU] UNMOVED** |

---

## §2 THE FIVE GRADES (the taxonomy, corrected)

The mandate specified four. The closure audit forced a fifth, which sits *between* built and stub and
is the one that actually bites:

| grade | detectable by | count |
|---|---|---|
| BUILT-AND-FIRED | registry + ledger | 2 |
| BUILT-NEVER-FIRED | registry + ledger | 165 |
| **BUILT-ELSEWHERE-UNWIRED-HERE** | **nothing automated — cross-vehicle** | **8** |
| DESIGNED-STUB | registry (after repair) | 10 |
| NOT-EVEN-DESIGNED | declaration only | 12 declared |

### 2a. DESIGNED-STUB — 10 (auto-derived, `build_completeness()`)

| module | factory | missing flag | marker? |
|---|---|---|---|
| `fh1_adapted_force_levers_20260731` | TieLocusEdgeWeighted | `--tie-locus-edge-weight` | marked |
| " | MarginSatisficeCap | `--margin-weight-fn-satisfice-cap` | marked |
| " | XiAdvectedTokenBase | `--token-temporal-mode-xi-advected` | marked |
| " | BirthPlateauKneeConjunct | `--knee-requires-birth-plateau` | marked |
| " | ErfBirthContextCoadapt | `--erf-birth-context-weight` | marked |
| `ph3_s10_frontloaded_levers_20260731` | Qa80MarginBoundedPhotometric | `--photometric-margin-budget-weight` | marked |
| " | Qa81LaneCarrierComposite | `--lane-carrier-composite` | marked |
| `ax1_derived_levers_20260730` | Ax1Frame0CarriedWarp | `--frame0-carried-warp` | marked |
| `constants_telemetry_build_wave_20260715` | **WeightNormTelemetryRow** | `--weight-norm-telemetry` | **SILENT** |
| `curriculum_dsl` | **IntegerPlaneEmitter** | 3 `--integer-plane-emitter-*` | **SILENT** |

The two SILENT rows are the worst grade in the table: they present as fully built, with nothing
anywhere saying otherwise. `WeightNormTelemetryRow` is score-neutral OBSERVABILITY, which per
CLAUDE.md's "'Off' is a tracked queue" should default ON and is not gate-able at all — it is a pure
orphan with no safety argument. `IntegerPlaneEmitter` is also the source of the 3 `stale` flags the
levelset `completeness()` was correctly reporting all along.

### 2b. BUILT-ELSEWHERE-UNWIRED-HERE — 8 (the grade nothing detects)

| component | built + fired at | tr1 status |
|---|---|---|
| KD warm start (#74/#129) | `torch_vehicle/kd_warm_start.py:52,104` ← `driver.py:1693` | **no path exists** |
| gradient surgery (lg1 leg a) | `island_protection.py:594` ← levelset trainer `:8426` | zero refs |
| σ_cc′ length/tension | `length_sigma.py:71-124`, `--length-sigma-matrix` | absent by construction |
| birth-seeding Lever | `curriculum_dsl.py:4262` (levelset DSL) | no equivalent |
| rank-4 head quantities | `segnet_head_rank4_flipdist_20260715.py` + 6 consumers | reaches tr1 only as **hardcoded float literals** (`lane_guard.py:64-65`) |
| ms4d metric bundle | `ddm_ms4d_direct_completion.py:617` + 6 consumers | no tr1 consumer |
| #725 BN capacity | `tools/run_ddm_hb1_hope_bn_capacity.py:37` (sole importer) | no tr1 consumer |
| #425 phase carrier | `dash_phase_carrier.py` (codec side) | compress-time only |

**Correction to a Path-A premise:** tr1's `--distill-*` is NOT a from-checkpoint KD warm start. It
mmaps a **precomputed teacher-logit cache** (`trainer:1444-1456`), never loads a teacher checkpoint and
never initializes student weights. Anything that counted `--distill-*` as satisfying Path A's
`kd_warm_start_dir` was counting the wrong mechanism.

---

## §3 WHAT WAS BUILT TO REAL ADMISSION

### `src/tac/optimization/reset_operator.py` (+ 32 behaviour tests)

gc15 §7's three knobs plus the bias-correction switch, as one typed operator:

- **`effective_lr_multiplier(t)` = (1−β₁ᵗ)/√(1−β₂ᵗ)** — verified **against the real
  `mlx.optimizers.Adam`** across 40 steps (measured uncorrected/corrected displacement ratio, rel 2e-3),
  not against a table. η(1)=3.1623, peak **η(12)=6.5685**.
- **`cumulative_excess_sign_steps` → asymptote 1212.57**, independently reproducing gc15's quoted
  **1,212.6 to four significant figures** = **16.17 epochs/boundary** at 75 steps/epoch, **82% inside
  the first 13 epochs**. A refinement worth recording: convergence is SLOW (a 2000-step sum reaches only
  94% of the asymptote; η is still 2.6% high at t=3000), so any window arithmetic that truncates at a
  few thousand steps under-prices the impulse.
- **`solve_norm_match_scalar`** — monotone bisection on real gradient magnitudes, so each D-arm's kick
  matches the zero-reset arm's first-N-step displacement within tolerance. The test feeds the returned
  scalar back through the simulation and checks the round trip; a canned scalar cannot pass.
- **Fail-closed everywhere:** `to='prior'` without a prior file REFUSES at construction; arms A/C
  refuse when no state was persisted, rather than silently degrading to arm B — **which would be a reset
  race quietly measuring the same arm five times.**
- **Byte-identity:** the default IS arm B; `apply_reset` returns `{}` and `requires_persistence` is
  False, so checkpoints keep `opt_state_flat={}`. Verified by test.

**Tests verify behaviour, not constants** (NO-FAKE class #2). An explicit `test_mutation_guard_*` in
each suite asserts what a marker-returning body could not do — η varies with its argument, the scalar
varies with its input, different arms produce different states, and the gate quotes the REAL missing
flag string.

### Detector repair + 4-grade schema + refusal gate

- `lever_registry.package_lever_factories/build_completeness` — package-wide, per-module trainer,
  **`lru_cache` on an (size, mtime) fingerprint: cold 1.03s → warm 0.9ms, 1096×**. A gate that costs
  seconds per call is a gate that gets turned off — which is how the vacuous scan survived. An edit
  invalidates the entry, so it can never serve a stale grade.
- `activation_ledger` gains the **BUILD axis** the old schema could not express: `{default,
  ever_fired, last_verdict, state}` cannot tell "off but real" from "off and HOLLOW" — both read
  `never-fired` to every consumer. Plus `record_required_component` / `not_even_designed`: grade 5 is
  invisible to any AST sweep, so it must be DECLARED, and the declaration refuses hollow charters
  (owner / missing_mechanism / consumer / fire_order all mandatory-by-refusal).
- `check_no_stub_lever_factories` — **LIVE COUNT 10, WARN-ONLY.** Structural, not label-based: a
  factory that forgot to say "DESIGNED-STUB" is still caught (both SILENT rows are), and one that says
  so while its flags exist is reported as label drift.

**Strict-flip decision: NOT flipped, deliberately.** Live count is 10 and all ten are chartered builds
owned by other arms (fh1 → burn-4 owner, ph3_s10, ax1). Flipping now would refuse the tree for debt
this arm does not own — the opposite of the atomicity rule's intent. **Flip condition: live count 0**,
i.e. each stub either gains its trainer wiring or a `# DESIGNED_STUB_OK:<rationale>` waiver.

---

## §4 CHARTERS — 12 registered, machine-readable

In `.omx/state/required_component_ledger.jsonl` (fcntl-locked, append-only, latest-row-wins). Every row
carries `{grade, owner, missing_mechanism, fire_order, consumer, needed_by}`. **The registry drains this
queue, not a human editing a memo:** a declared component whose factory later lands drops off
automatically, so the only way to clear a real row is to BUILD it.

| fire | component | owner | why it blocks |
|---|---|---|---|
| 1 | **TR1ResetOperatorWiring** | ddm_sb2 | hook built+tested; trainer flags + opt-state save/load + Lever not yet wired. Gates the entire #815 reset race |
| 1 | **TR1KDWarmStart** | burn-4 owner | gates Path A *and* both deferred fresh cells |
| 1 | TR1RowbandD8Config | burn-5 owner | **fully built, never fired** — needs only a spec `.json` + `--grid-downsample 8` |
| 2 | ResetDiagonalPriorProvider | ddm_sb2 successor | D± cannot fire without it; contract exists, producer does not |
| 2 | TR1Hb1CapacityConsumer | ddm_sb2 successor | #725 has one non-test importer, zero tr1 wiring |
| 3 | TR1PerComponentRank4Hinge | lg1 successor | helper built+tested, hinge loss term deferred |
| 3 | TR1GradientSurgery | lg1 successor | levelset-only call site; ~1.8× step tax |
| 3 | Cg1PerClassGuardLedger | cg1 #809 | **zero code artifacts** — memo prose + two `routes_to` strings |
| 4 | TR1SigmaCCPrime / TR1BirthSeeding / FlickerSidecar535 / W1CohLossTerm | unassigned | named, off the critical path |

---

## §5 CROSS-FINDINGS

**To cn3 (#818).** (a) Two pre-existing test failures, confirmed by stash — NOT regressions from this
work: `test_332_coverage_rose_from_deorphaning` and `test_p1_repo_live_count_bounded` (significance
live count 5 > bound 4). Both are themselves orphan-signal instances (a coverage assertion and a
live-count bound that drifted and nobody noticed). (b) The BUILT-ELSEWHERE-UNWIRED-HERE grade is a
consumption-surface problem, which is cn3's axis: 8 components produce signal no tr1 consumer reads.

**To gd1 (#817).** Consumed, not duplicated — no A1 gate-bias work here.

**Owed, not done (honest):** the two remaining anti-orphan gate repairs MAIN routed — the
`check_codex_findings_memos_consumed` 3-day scan window (0 of 1,260 files in scope) and the Catalog
#396 strict-flip (433 live / 108 in-window). Both are real, both are unstarted, and neither is in this
commit. Naming them as debt with no owner assigned would be the exact failure this arm exists to kill,
so: **they need an owner in the next dispatch.**

---

## §6 POINTER HONESTY

**`0.1910828242` [contest-CPU] UNMOVED.** Nothing here lowered the exact score. This is MEANS. The
justification for spending an arm on it is specific and measurable: a strategy argument for a fresh
from-birth run rested on "the full protection/force stack has never run from ep0", and 5 of those 6
forces were stubs — wiring was the blocker, not birth. Apparatus that reports hollow levers as built
does not merely lose signal; it produces **wrong decisions**, and it produced one this week.
