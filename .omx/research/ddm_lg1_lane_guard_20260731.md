# ddm_lg1 (#808) — the CONSTRAIN-AND-PROTECT layer for burn-4 (landed)

**Date:** 2026-07-31 · **Actor:** ddm_lg1 · **Task:** #808 · **Commits:** c009a2e123 (layer) +
c66acf4d79 (warm-λ amend) · **Custody:** `/Volumes/VertigoDataTier/pact/ddm_lg1_20260731/`
**Authority:** `[macOS-CPU advisory]`; `research_only=true`; `score_claim=false`;
**pointer 0.1910828242 [contest-CPU] UNMOVED** (this is apparatus for burn-4, MEANS not END).

**Operator directives (2026-07-31 ×2):** *"We can use sophisticated techniques to constrain and
protect"* + *"Remember the upstream channels and hyperplane and basis and such too."* Context: xp1
(cc55db90d5) measured rung-1's UNPROTECTED continuation eroding Lane (+0.00151 S, +187 erased
components, pool 0.04189→0.04401 S) while bulk classes descended. This layer makes that trade
infeasible-or-punished in burn-4.

## What landed (wired) vs deferred, per piece

| piece | state | surface |
|---|---|---|
| 1. λ_Lane primal-dual constraint | **WIRED** | `tac.optimization.lane_guard.dual_ascent`/`gate_update` + trainer gate-block; realized g from the a1 gate's EXISTING argmax (zero new scorer passes); λ enters the loss as +λ per GT-Lane pixel via the existing `seg_pixel_w` hook |
| 2. Born-lane protection mask | **WIRED** | `born_lane_support_mask` (gt==Lane & realized==Lane, refreshed per gate) × `LANE_HEAD_SENSITIVITY_RATIO` 1.19607 (measured rank-4 head normals) as a `seg_pixel_w` addend |
| 3. Margin floor per born component (head-hyperplane metric) | **WIRED (pixel form) + helper LANDED (component form), loss wiring = the ONE deferred piece** | pixel form: hinge `relu(1 − m/floor)` on GT-Lane, floor = per-run p10 of the Lane-restricted QA80 margin field (derived at first gate). Component form: `per_component_min_flip_distance(margin, born_mask, ‖Δw‖=4.007)` = the exact rank-4 closed form `d=|m|/‖Δw‖` per born component — landed + tested; its per-component hinge loss term is deferred (insertion point named below) |
| 4. Lane-vs-bulk gradient surgery (optional) | **DEFERRED** | 2-backward split + Fisher-metric projection; see deferred table |
| Warm-λ relaunch (b4s rollback path) | **WIRED** (amend c66acf4d79) | `--lane-guard-lambda-init`, clamped [0, λ_max] |
| 3 DSL Lever factories | **WIRED** | `spec_tr1_renderer_20260728.lever_lane_guard_{lambda,born,margin_floor}`; budget custodied via `dsl_custodied_scalar_identity_v1` LawRef (resolves 0.12589, no fallback) |

All default-OFF, flag-gated (`--lane-guard` master + per-piece weights), additive `seg_pixel_w`
addends — ONE hook, composing with `class_weight_lane`.

## Derived values (constants-are-poison; every input measured/principled)

- **Budget = 0.12589 S-units** — MEASURED: xp1 ep641 endpoint `base_per_class_S_units[1]`
  (`ddm_xp1_20260731/xp1_verdict.json`; ckpt sha `40553db8…70db`, ep641). Error definition matches
  qa92 `_per_class_flip_counts` + P formula exactly (`100·flips/(n·384·512)`), so realized-vs-budget
  is apples-to-apples.
- **η_λ = 66.2252 = λ_target / (n_gates·erosion_s) = 1.0/(10·0.00151)** — λ_target 1.0 = one unit of
  per-Lane-pixel weight (the sn1 `class_weight_lane` natural scale); n_gates 10 = deliberate slow
  engage (react to persistent drift, not single-gate noise); erosion_s 0.00151 = the xp1-MEASURED
  unprotected Lane erosion. Self-consistent: at steady erosion one gate's step η·g ≈ the cap.
- **λ step cap = 0.1/gate = λ_target/n_gates** (caps-law: one gate cannot thrash); **λ_max = 5.0**
  (5× the natural unit; above that one class dominates the primal → refuse further ascent).
- **Sensitivity ratio = 1.19607 = mean(4 Lane-pair ‖Δw‖)/mean(all-10)** — MEASURED rank-4 head
  normals (4.007/3.953/3.862/3.748 vs all-pair mean 3.2544), fractal memo §2. The four Lane pairs
  are the four LARGEST — the frozen net amplifies Lane; protection on it is proportionally worth more.
- **Margin floor = per-run p10 of QA80 margin | gt==Lane**, derived at first gate — pct=10 from the
  MEASURED bottom-decile flip law (sg1 §1.3: 100% of realized flips in the bottom GT-margin decile).
  QA80 n600 field anchors (zb1 custody): q05 med 0.4302, q50 med 1.8181. Smoke (n=4) derived 0.10734.
- **Fail-closed:** `resolved()` raises on η<0 / cap<0 / λ_max≤0 (a negative η silently inverts the
  dual into a reward).

## Caps-law reconciliation (mandatory paragraph)

The v9 rule "loss weights change at STAGE boundaries only" governs the PRIMAL loss weights. λ_lane is
NOT a primal weight — it is a KKT dual multiplier, and primal-dual theory places its update at the
constraint-EVALUATION cadence: the a1 GATE, the only point the realized constraint g is measured.
Between gates λ is CONSTANT, so the primal sees a per-interval-fixed weight — no per-step thrash. The
per-gate step cap (0.1) + λ_max (5.0) bound the worst case so one noisy gate cannot dominate. Dual
state is NOT persisted across relaunches (mirrors the Adam-moment re-anchor law #517/#518);
`--lane-guard-lambda-init` is the supervisor's warm-start for the rollback+raise path.

## Byte-identity receipt (the tp1 OFF-vs-OFF pattern)

Custody `smoke_{pre_off,post_off,post_off2,cpu_pre_off,cpu_pre_off2,cpu_post_off,cpu_post_off2,post_on,post_on2,smoke_lambda_init}`:

- **Structural:** every new code path is behind `cfg.lane_guard` (default False); trainer diff is
  pure insertions; OFF preserves `seg_pixel_w=None` semantics exactly; no import, state, or RNG when off.
- **MEASURED noise floor (the honest empirical bound):** the tr1 vehicle is rerun-NONDETERMINISTIC
  with identical code+argv on BOTH devices (Metal ep1 d_seg 0.4476 vs 0.4413; forced-CPU 0.4544 vs
  0.4813 post / 0.5061 vs 0.5217 pre; counted bytes vary ±2 even at ep0) — consistent with the
  standing law "bit-id = DECODE only". Pre-vs-post OFF differences are inside the same-code rerun
  scatter; **ep0 realized gate d_seg is bit-equal (0.5078303019205729) across ALL 8 OFF/ON runs**
  (uint8-R + argmax absorbs sub-LSB float noise). No detectable OFF-path effect.
- **ON smoke (n=4, ep2):** λ 0→0.1→0.2 (capped steps under g=0.5065 violation), floor 0.10734
  derived once, 4 born masks, complementarity λ·g telemetry (#549 KKTDiagnostics-aligned), OFF arm
  emits zero lane_guard rows.

## ENGAGEMENT SPEC for ddm_b4s (one page)

**Flags (all argparse-verified on `experiments/train_tr1_partition_renderer_mlx.py`):**
`--lane-guard` (master, store_true) · `--lane-guard-budget-s 0.0` (0.0⇒0.12589 xp1 ep641) ·
`--lane-guard-eta 0.0` (0.0⇒66.2252 derived) · `--lane-guard-lambda-step-cap 0.0` (0.0⇒0.1) ·
`--lane-guard-lambda-max 5.0` · `--lane-guard-born-weight W` (0.0=off; suggest race W∈{0.25,0.5}) ·
`--lane-guard-margin-floor-weight V` (0.0=off; suggest race V∈{0.5,1.0}) ·
`--lane-guard-lambda-init L` (rollback+raise relaunch; clamped [0,λ_max]).
DSL: `lever_lane_guard_lambda()` + `lever_lane_guard_born(W)` + `lever_lane_guard_margin_floor(V)`
in `spec_tr1_renderer_20260728` — fold via the ticket's normal lever composition.

**Gate-cadence rules:** the dual updates once per a1 gate, AFTER the a1 alarm block (skipped on a
refuse-exit gate — the run stops anyway). Telemetry row `event=lane_guard`:
`realized_lane_s_units · budget_s_units · g_s_units · lambda_lane · complementarity ·
margin_floor · born_mask_pairs · epoch · stage · gate_basis` — one JSON row per gate in
`telemetry.jsonl`, machine-readable for the supervisor.

**Supervisor-side erosion-key parameters consistent with this dual:**
- b4s's LANE-EROSION GUARD (P2-inverted betti0 slope key) is the TOPOLOGY-trend constraint; my dual
  is the LEVEL constraint on the SAME gate cadence — complementary, no double-counting: the guard
  STOPS a window; the dual PRICES the erosion inside a window.
- Supervisor level-check (optional, from my rows): alarm when `realized_lane_s_units >
  budget_s + t_crit·max(SE_ols, SE_quant)` over the last-5-gate window (the lp2 ε template;
  SE from the rows themselves — the fd2 gate SUBSET estimate at n600 has unmeasured absolute-level
  noise, so a window-SE-derived slack, never a hand constant).
- **Rollback+raise-λ path (`LG1_DUAL_ENGAGED` flip):** on LANE_EROSION ALARM → rollback to the
  pre-window checkpoint → relaunch with `--lane-guard-lambda-init = (last lambda_lane) +
  lambda_step_cap` (one capped step, honoring the caps-law) and the same derived η/cap. λ_max 5.0
  bounds the escalation ladder to ≤50 raises — practically the guard should escalate to operator
  well before (suggest ALARM-ESCALATE at λ ≥ 1.0 = the natural full-unit).
- n600 honesty: gate rows measure Lane-S on the fd2 gate subset (~36 pairs) — an estimate of the
  full-n600 level the budget was measured on. The dual reacts to persistent drift regardless;
  full-n600 Lane-S re-measure at window ends stays the xp1-method authority.

## Deferred table

| item | reason | cost estimate | insertion point |
|---|---|---|---|
| Per-component margin-floor hinge (piece-3 loss form) | pixel-form wired first (lean); component form needs per-pair labels refreshed per gate + an mx-side gather | ~40 LOC + ~0.2s/gate (scipy label on gate pairs) | `pair_loss`: replace the `deficit·is_lane` term with a per-component weight map built in `gate_update` from `per_component_min_flip_distance` (helper LANDED + tested); floor per component = t_crit·SE per the lp2 template |
| 4. Lane-vs-bulk gradient surgery (2-backward Fisher projection) | optional per charter; 2× backward cost; ms3/ms4 row-Gram custody needs a per-batch metric assembly | ~120 LOC; ~1.8× step wall-clock (second `value_and_grad` on L_lane) + Gram lookup; report BOTH Euclid + Fisher cosines (dual-metric law) | trainer step: split `loss_fn` into L_lane (GT-Lane px) / L_bulk, `mx.value_and_grad` each, project g_bulk ⟂ g_lane when Fisher cosine < 0; reuse `contain_protected_grad_mx` (`tac.boundary_math.island_protection` L594) as the projection primitive — do NOT fork |
| #725 per-channel Fisher anchor (born protection refinement) | render params do not expose scorer channels; the honest render-side projection is the scalar head-sensitivity ratio | needs a channel-to-pixel pullback study first | weight the born addend by a pullback of the #725 per-stratum table (`ddm_hb1…/hope_per_stratum_capacity_table.json`: Lane strata top-3 pre-head channels (2,9,6) = 70.7%; ch9 alone 30% of Lane-Undrivable) |
| Dual-state persistence in checkpoints | additive-persistence change to `save_checkpoint`/resume; warm-λ flag covers the b4s path now | ~20 LOC | `save_checkpoint(..., lane_guard_state=…)` + resume block |

## STORES CONSULTED

CLAUDE.md · AGENTS.md · docs/operating_manual_craft_handoff.md ·
`.omx/research/segnet_recursive_fractal_factorization_20260715.md` (rank-4 head, flip dist, Lane
normals, 77% skip-limited) · xp1 custody (`ddm_xp1_20260731/xp1_verdict.json` + manifest: budget +
error definition) · fp1 custody (`ddm_fp1_20260731/qa91_erased_lane.json`: birth curve, erasure) ·
`experiments/ddm_qa92_carrier_discriminator.py` (per-class flip counts + P formula) ·
`experiments/train_tr1_partition_renderer_mlx.py` (a1 gate, topology telemetry, seg_pixel_w hook) ·
`.omx/research/ddm_hb1_hope_bn_capacity_findings_20260727.md` (#725 per-channel Lane capacities) ·
`src/tac/optimization/coupled_margin_levelset.py` (#549 KKT: batch working-set QP — reused the
diagnostics SEMANTICS (multipliers/complementarity), not forked; an online gate-cadence dual is a
different shape than a batch QP) · lp2 ε-derivation template (`ddm_lp2_ladder_prep_20260731.md` /
DAG L26102-26108) · `src/tac/boundary_math/island_protection.py` (#208: containment projection +
birth terms — named reuse surface for deferred piece 4) · QA80 field custody
(`ddm_zb1_qa80_field_20260730`: q05 0.4302, q50 1.8181, bottom-decile flip law) ·
`src/tac/witness_dsl/{curriculum_dsl,lawref,spec_tr1_renderer_20260728}.py` (Lever pattern +
`dsl_custodied_scalar_identity_v1`) · memories: dual_metric_readback · constants_are_poison ·
verdict_scope ladder · b4s FEED-b4s-amend (DAG tail).

## Honest negatives / limitations

- Byte-identity is proven structurally + noise-floor-bounded, NOT bit-exact-run-compared: the
  vehicle itself is rerun-nondeterministic (MEASURED both devices, custody above). `tr1_config.json`
  + checkpoint-embedded cfg gain the new default-off fields (the sanctioned additive-persistence
  pattern).
- Constraint pressure is a loss-WEIGHT dual (soft enforcement priced at λ), not a hard projection —
  if burn-4 still erodes Lane at λ_max, the constraint form is falsified on this vehicle and the
  #208 containment projection (deferred piece 4) is the successor. Pre-registered falsifier in
  `lever_lane_guard_lambda`.
- Pre-existing test failure `test_ddm_tb1_tr1_renderer.py::test_counted_ledger_keys…` fails at HEAD
  with my edits stashed (smevr ledger keys outran the test) — NOT an lg1 surface, not touched.
- Gate-subset estimate honesty per the engagement spec (fd2 ~36 pairs at n600).
