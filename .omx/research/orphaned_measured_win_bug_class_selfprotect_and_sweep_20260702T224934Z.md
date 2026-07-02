# Orphaned-measured-win bug class — self-protect landing + retroactive sweep (#225)

**Date:** 2026-07-02T22:49:34Z · **Status:** LANDED (structural self-protect + audit tool + sweep).
**Advisory / apparatus-hardening; pointer 0.19110 UNMOVED** (this landing moves no exact row — it is the
META fix that stops measured wins from rotting un-integrated). `score_claim=false`, `promotable=false`.
`# FORMALIZATION_PENDING:this-is-the-self-protect-META-landing-for-the-canonical-equations-surface-itself-not-a-new-empirical-finding`
`# ORPHAN_WIN_WAIVED:this-ledger-IS-the-self-protect-landing-that-defines-the-orphan-discipline-not-an-orphaned-mechanism-win`

Operator 2026-07-02: *"not wiring Track B's store-nothing win was CLASSIC ORPHANED SIGNAL and a related
meta-bug ... but that was a meta bug too."* Sister memory:
`orphaned_measured_win_not_wired_into_vehicle_and_triality_bug_class_20260702.md`.

---

## 0. ROOT-CAUSE DIAGNOSIS — WHY the existing apparatus DIDN'T FIRE (the actual meta-bug)

The operator's sharpening is the load-bearing insight: the discipline we ALREADY WROTE to prevent this
existed, was STRICT, and STILL missed it. That is **apparatus blindness**, not a missing rule.

**Proven root cause (measured, not asserted):**

1. The **"Canonical equations + models registry"** non-negotiable says verbatim: *"FORBIDDEN: introducing
   a new empirical-finding memo without ALSO registering the underlying canonical equation."* Its
   enforcement gate **Catalog #344** (`check_empirical_finding_memo_references_canonical_equation`) is
   **STRICT** (strict-flipped 2026-05-19).
2. Yet `keyframe_rate_minimization_builds_20260702.md` (Track B store-nothing) landed with **NO registered
   canonical equation** and nothing fired. WHY: #344's trigger-token set is
   `{"empirical anchor", "predicted vs measured", "ratified", "falsified", "posterior update",
   "bayesian update", "empirical reduction", "empirical residual", "prediction-vs-empirical"}` — none of
   which is how a **MEASURED MECHANISM WIN** memo is actually written ("MEASURED n600 ... WINS ...
   BETTER than ... −N%"). **Measured fact: the store-nothing memo contains 0 of #344's 10 trigger
   tokens.** So strict-#344 was *structurally blind* to the single highest-value class it should protect.
3. **#219** (triality-maintenance: DSL gauge + `canonical_equation` + DAG) is a **TASK, never a gate**.
   "Results must become system intelligence" + "Subagent coherence-by-default" (Mandatory wire-in) are
   **doctrine without a findings-memo firewall**. So the INTEGRATION axis had NO enforcement at all.
4. Net: a measured win becomes a ledger row and drifts as ORPHANED SIGNAL while the launch config
   silently ships an inferior un-integrated mechanism (`sealed_205` kept the naive warp-real-luma carrier
   while the measured-optimal store-nothing sat un-wired). The recursive review is blind because it
   reviews the CONFIG, not "is this the best MEASURED mechanism available?" (the config-mechanism
   blindness sister, `review_seals_borrowed_numbers_and_unrun_configs...`).

**Adjacent finding surfaced by this diagnosis (OPERATOR-ROUTABLE, OUT OF SCOPE here):** Catalog #344 is
STRICT but its **live count has silently drifted 0 → 480** (memos dated 2026-05-20 .. 2026-07-02 that
carry a #344 trigger token with no `canonical_equation` ref and no `FORMALIZATION_PENDING` waiver). This
is the SAME meta-class one layer up — a strict gate whose backlog decayed without anyone noticing — and it
means `preflight_all(strict=True)` is currently red on #344. Fixing it is a ~480-memo backfill (add
`FORMALIZATION_PENDING` waivers or register equations) + a live-count-drift alarm; it is a separate
landing, not this task. Flagged here so it is not re-forgotten.

---

## 1. THE FIX — consolidation, NOT pure-additive (gate-consolidation discipline)

Per the coordinator + gate-consolidation discipline (a new gate must subsume ≥3 sister cases or replace
one, never pure-add), the fix ENFORCES/CONSOLIDATES the existing surface rather than bolting on an
unrelated gate.

**Catalog #396 `check_measured_win_findings_are_wired_or_research_only`** (claimed via
`tools/claim_catalog_number.py`; catalog max was 395). It **consolidates FOUR previously-unenforced
doctrinal surfaces into ONE binding gate**:

1. the canonical-equations non-negotiable **for the measured-win memo class that #344's narrow trigger
   structurally misses** (FORMALIZATION axis);
2. **#219 triality-maintenance** lifted from TASK → GATE (INTEGRATION axis: DSL gauge + canonical_equation
   + DAG);
3. **"Results must become system intelligence"** at the findings-memo surface;
4. **"Subagent coherence-by-default"** Mandatory wire-in, for measured findings.

**Why #344 is left UNMUTATED (deliberate):** widening #344's STRICT trigger to catch measured-win phrasing
would immediately strict-break the live orphan backlog (store-nothing, wave-f) and the 480-drift above,
violating the **Strict-flip atomicity rule**. Instead: #396 covers the missed class and is **WARN-ONLY**,
so it surfaces the backlog without breaking the build; a cross-ref note was added to #344's header
documenting the trigger-gap + the #396 sister.

**The contract.** A `.omx/research/*.md` memo dated >= **2026-07-02** (the naming day; forward-looking)
whose body co-mentions a MEASURED-evidence token (`measured` / `n600` / `byte-closed` /
`realized-through-R` / `gt_n600`) AND a mechanism-win token (`wins` / `beats` / `better than` /
`outperforms` / `measured-optimal` / `lowers S` / `strongest single` / `net win` / `huge win`, OR a
`−N%` reduction) is a **measured-win memo** and MUST satisfy ONE of:

- **(a) WIRED** — a canonical_equation reference (reuses #344's token set) **AND** a launch-config/DSL/
  wiring pointer (`witness_autoconfig` / `proven_base` / `sealed_205` / `witness_dsl` / `trainer mode` /
  `wiring task` / ...). Formalized AND integrated.
- **(b) RESEARCH_ONLY** — `research_only` AND `reactivation` criteria pinned.
- **(c) WAIVER** — same-line `# ORPHAN_WIN_WAIVED:<rationale>` (placeholder `<rationale>`/`<reason>` +
  <4-char rationales rejected per Catalog #287).

else **ORPHAN** → violation. The ONE canonical classifier
`tac.preflight.classify_findings_memo_orphan_status` (verdicts `NOT_A_WIN` / `WIRED` / `RESEARCH_ONLY` /
`WAIVED` / `ORPHAN`) is shared by the gate AND the audit tool so they never drift.

**WARN-ONLY at landing** (Strict-flip atomicity): the live backlog is real — **15 measured-win memos
dated >= 2026-07-02 flag ORPHAN at landing**. Strict-flip after the sister agents wire/tag them.

**AGENT-BEHAVIOR default (encoded in the gate docstring + memory):** a measured win is **NOT done at the
ledger**. Proactive integration (wire into vehicle + config + all 3 triality legs, OR tag
`research_only` + `reactivation_criteria`) is a **DEFAULT agent step**, not operator-prompted. Before
calling a measured finding complete, run `tools/audit_orphaned_measured_wins.py`.

---

## 2. RETROACTIVE ORPHAN SWEEP (#225) — ranked by |measured ΔS| ÷ ease-of-wire

Full machine-readable list: `tools/audit_orphaned_measured_wins.py --json` (default window >= 2026-06-01
= 340 measured-win memos: 320 ORPHAN / 12 RESEARCH_ONLY / 8 WIRED; `--since 20260702` = 18 / 15-ORPHAN,
matching the gate). The heuristic tool-ranking is coarse; the CURATED, load-bearing prioritization below
is my manual read of the current-vehicle families (sweep sub-agent + direct reads). Honest no-fake note:
a DERIVE-only or physics-refuted item that is correctly `research_only` is NOT an orphan.

| # | Family / memo | MEASURED win (magnitude) | Verdict | Ease-of-wire | Priority |
|---|---|---|---|---|---|
| 1 | **Wave-F Stage-1 LBND2** `wave_f_lane_band_rd_code_LANDED_stage1_measured_20260702.md` | **3.76× rate** (n600 0.1041 → 0.02765), decode-consistent, **default-ON** | ORPHAN (no registered eqn, no config-pointer; net-S #205-gated) | **TRIVIAL** — landed default-on plumbing; needs only eqn registration + config pointer | **HIGH** |
| 2 | **Store-nothing keyframe** `keyframe_rate_minimization_builds_20260702.md` + `warp_keyframe_payload_rate_minimization_20260702.md` | **d_pose 1.12 < full-keyframe 1.37** at ~0 marginal rate; Pareto-dominates codec | ORPHAN ("Did NOT touch #205 trainer/launch config"; no eqn) | **MEDIUM** — tools landed; needs #205 ckpt + `compose_witness_archive` wiring (in flight, sister task #241) | **HIGH** |
| 3 | **Analytic lane band** `lane_render_band_decode_consistency_landed_20260702.md` | decode-consistent (bit-exact); naive serialize +0.147 rate WALL → RD codec supersedes | ORPHAN (landed but **default-OFF** in `levelset_byte_close_and_eval.py`; eqn FORMALIZATION_PENDING) | **LOW** — plumbing landed; needs #205 trained-in d_seg A/B | MED |
| 4 | **MBO / curvature** `sdf_levelset_dynamical_topology_opt_research_20260702T214342Z.md` (commit `7c3d27e75`) | curvature↔SegNet-margin **10–40× separation** (m_flip 0.13→0.55 vs m_keep ~5.6); σ=1.0 removes 11.4% boundary length @ d_seg 0.00087 | ORPHAN (**CANDIDATE** eqn `curvature_ranks_segnet_margin_v1` + CANDIDATE DSL lever `mbo_decode_regularizer`, neither registered) | **LOW** — new class-pair-weighted MBO pass inside inflate.py; through-R + byte-closed A/B | MED |
| 5 | **Wave-F unified-ξ** `wave_f_unified_xi_build_measured_20260702.md` (source-smoothing win15 **−42% rate**) + **lane-tracking** `wave_f_lane_tracking_coherent_fit_measured_20260702.md` (lossless 0.5%) | −42% rate; ego-predictive coding MEASURED NEGATIVE | **RESEARCH_ONLY** (self-tagged, 6-hook `research_only` + #205 reactivation) — COMPLIANT, register eqn when net-S lands | TRIVIAL (ships as existing LBND2 bytes) | (compliant) |
| — | **Sig-proc levers** `signal_processing_filter_levers_derived_20260701T014119Z.md` (#204/#207 era) | headline is a MEASURED NEGATIVE (R near all-pass |H|≥0.842, deconv ≤+1.25dB "not a d_seg lever"); L3/L4 levers DERIVED-only | **NOT-A-WIN** (measured-negative + derived-only) — honest, no action | — | (n/a) |

**The two clean fresh ORPHANs with genuine measured mechanism wins: (2) store-nothing and (4) MBO
curvature.** (1) LBND2 is landed default-on plumbing missing only its equation + config pointer — the
lowest-effort highest-value burndown. Sig-proc is correctly a measured-negative (NOT an orphan) — the
audit is honest, not a numbers-chase.

---

## 3. WHAT LANDED (3 surfaces + this ledger)

1. **Gate** `src/tac/preflight.py::check_measured_win_findings_are_wired_or_research_only` (Catalog #396,
   WARN-ONLY in `preflight_all`) + the shared classifier `classify_findings_memo_orphan_status` +
   verdict constants + a cross-ref note added to #344's header (the diagnosed trigger-gap).
2. **Audit tool** `tools/audit_orphaned_measured_wins.py` (imports the ONE canonical classifier; `--since`
   / `--json` / `--orphans-only`; heuristic |ΔS|÷ease ranking, explicitly labeled non-measured).
3. **Tests** `src/tac/tests/test_check_396_measured_win_findings_wired_or_research_only.py` (29 tests, all
   pass: classifier 5-verdict coverage, measured+win co-occurrence, percent-reduction win, WIRED requires
   eqn AND pointer, RESEARCH_ONLY requires reactivation, waiver placeholder/short rejection, date-filter,
   not-a-win exemption, strict-raise + warn-no-raise, string repo_root, missing-dir silent, non-md/undated
   ignored, multi-memo aggregation, live-repo known-orphan flagged + warn-only-never-raises, orchestrator
   warn-only wire-in guard).
4. **Catalog doc row** appended to `docs/meta_bug_class_catalog.md` (#396), phrased WARN-ONLY to match the
   `strict=False` callsite (#159 clean) and NOT claiming "STRICT @ 0" (#185 clean).

## 4. 6-hook wire-in declaration (Catalog #125)

- **#1 sensitivity-map** = N/A (a discipline gate, not a score-axis contributor).
- **#2 Pareto constraint** = N/A.
- **#3 bit-allocator** = N/A.
- **#4 cathedral autopilot dispatch** = **ACTIVE** — the gate + audit tool prevent measured wins from
  silently orphaning; the audit tool is the operator-facing burndown surface the autopilot can consult.
- **#5 continual-learning posterior** = N/A (no new posterior; it enforces that findings BECOME system
  intelligence via the equation/DSL/DAG legs).
- **#6 probe-disambiguator** = **ACTIVE** — `classify_findings_memo_orphan_status` IS the disambiguator
  between WIRED / RESEARCH_ONLY / WAIVED / ORPHAN for any findings memo.

Mission contribution (Catalog #300): `frontier_protecting` — extincts the orphaned-signal bug class that
let the launch config ship an inferior mechanism (the store-nothing instance); structural protection so
measured wins reach the vehicle instead of rotting in ledgers.

## 5. Verification

- `ruff check --select F821,F401` clean on both new/edited files.
- 29/29 #396 tests pass; `#118`/`#159` catalog-integrity gates clean on the addition; the 3 `#185` +
  480 `#344` violations are PRE-EXISTING (dates 2026-05-20+, none created by this landing; #344 detection
  tokens unchanged — my #344 edit is comment-only) → operator-routable, out of scope.
- `check_measured_win_findings_are_wired_or_research_only(strict=False)` on the live tree: 15 ORPHANs
  flagged (store-nothing + wave-f + 13 others), warn-only, never raises.
