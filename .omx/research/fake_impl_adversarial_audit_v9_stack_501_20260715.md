# Adversarial Fake-Implementation Audit — Live V9·CGauge Witness Stack (task #501, NO-FAKE #1)

- **Date:** 2026-07-14 (fresh-eyes read-only audit)
- **Scope:** live V9·CGauge witness surfaces — the level-set trainer, the witness DSL
  (`src/tac/witness_dsl/`), the optimal-basis surface, the costate organ, and the
  value/score-claim path. READ-ONLY. No code edited, nothing launched, pointer untouched.
- **Method:** grep + graph-memory recall + direct read of each named surface; every claim
  labeled MEASURED / DERIVED / INFERRED. Paradigm-vs-implementation separated per Catalog #307.
- **The 8 forbidden classes** (CLAUDE.md §NO FAKE IMPLEMENTATIONS): (1) markers-without-work,
  (2) tests-verify-constants-not-behavior, (3) synthetic-fixture-instead-of-real-input,
  (4) placeholder-in-data-field, (5) enum-padding-without-distinct-impls,
  (6) search-masquerading-as-solver, (7) borrowed-substrate-passed-as-original,
  (8) surrogate-optimized-but-not-exact-authority-verified.

## VERDICT (headline)

**The live V9·CGauge stack is NO-FAKE-CLEAN on its score-affecting surfaces.** No fake was
found that must be fixed before a score claim. The stack is, on the surfaces most at risk,
an *exemplar* of anti-fake discipline: the basis surface reclassifies "curvelet"→Fourier by
what the code computes and fails closed on unimplemented frames; the costate organ is
advisory-only with no execute path; every score row I found is honestly axis-tagged
(macOS-CPU advisory / archive-bytes-unmeasured); inert levers are SURFACED and excluded, not
hidden.

**One real finding is a GOVERNANCE gap, not a live-score fake:** the automated V9 anti-fake
*enforcement* checks that CLAUDE.md's 2026-07-14 catalog amendments declare "STRICT complete"
(#332 `check_config_flag_provenance_bijection_complete`, #351 `check_v9_fake_claim_guards` /
`check_evidence_authority_claims_are_custodied`) **do not exist in the tree.** The substance
they would enforce is present at the data layer by construction, so this is medium (regression
risk), not a BLOCKER — but it should be reconciled so anti-fake protection is structural rather
than human-review-dependent.

---

## FINDINGS (ranked, most-severe first)

### F1 — GOVERNANCE: CLAUDE.md declares V9 anti-fake STRICT checks that are ABSENT from code
- **Class:** 1 (markers-without-work) at the *governance/documentation* surface; NO-FAKE sister
  "Memos must be implemented — a landed memo describing canonical work that was actually
  placeholder-emission is itself a fake-implementation incident at the documentation surface."
- **Severity:** REAL / MEDIUM (not BLOCKER — see mitigant).
- **Evidence (MEASURED):**
  - CLAUDE.md §"2026-07-14 catalog amendments" states, as **"STRICT completion"**, that
    `check_config_flag_provenance_bijection_complete` "compiles every live V9 factory and
    refuses unless … every source-hashed runtime consumer actually reads its value," and that
    `check_v9_fake_claim_guards` + `check_evidence_authority_claims_are_custodied` refuse
    unlocalized-Fourier-labeled-curvelet, inert selected-pose markers, advisory-as-authority, etc.
  - `src/tac/preflight.py` has **426** `def check_` functions; **none** match those names
    (`grep 'def check_config_flag_provenance_bijection\|def check_v9_fake_claim\|def check_evidence_authority' → no match`).
    Whole-repo `grep -rl 'check_config_flag_provenance_bijection_complete' src tools` → **no match**.
  - `src/tac/witness_dsl/tests/test_config_provenance.py` **source is deleted** — only the
    compiled `test_config_provenance.cpython-313-pytest-9.1.1.pyc` remains in
    `src/tac/witness_dsl/tests/__pycache__/`. A test that no longer has source cannot run in CI.
- **Failure scenario:** a future V9 config that (a) parses a flag no trainer code reads
  (parsed-but-inert), (b) labels an unlocalized Fourier bank "curvelet/shearlet active," or
  (c) quotes an advisory MLX/macOS number as authority — would **not be refused by any gate**.
  The CLAUDE.md-promised structural protection is currently human-review-only.
- **Mitigant (why MEDIUM not BLOCKER):** the *substance* those checks would enforce is present
  at the data layer **by construction** — see CLEAN-1/2/3 below (rung-tagged scalars,
  fail-closed basis families, surfaced inert rows). No live score is corrupted today.
- **Fix-direction:** either implement the three named checks (wire into `preflight_all`) OR
  amend the CLAUDE.md amendment to mark them PLANNED / pointer-correct their status. Restore or
  delete-with-note the orphaned `test_config_provenance.pyc`. Do NOT leave CLAUDE.md asserting
  "STRICT completion" of absent code.

### F2 — BASIS: "curvelet" alias persists in the generator (latent mislabel) — VERIFIED HONEST
- **Class:** 5-adjacent (naming/label) — the #502 "real-not-Fourier-in-disguise" concern.
- **Severity:** MINOR / FALSE-ALARM-VERIFIED-HONEST.
- **Evidence (MEASURED):**
  - `src/tac/boundary_math/lever_b_levelset_generator.py:187-188`:
    `curvelet_directional_B = polar_directional_fourier_B` / `curvelet_feats = polar_directional_fourier_feats`
    — the "curvelet" names survive as **byte-compat aliases** with explicit comments
    ("curvelet-*inspired sampling only*", "not a localized curvelet or shearlet frame", :24-25,157).
  - The DSL layer that could make a *claim* off these is CLEAN:
    `src/tac/witness_dsl/optimal_basis_20260714.py` reclassifies the family to
    `BasisFamily.POLAR_DIRECTIONAL_FOURIER` "by what the code actually computes"; and
    `audit_legacy_polar_bank` (:262-295) **re-derives** that the paired-feature envelope is
    `sin²+cos²=1` (constant → `has_spatial_window=False`, `has_translation_index=False`),
    proving the atoms are global plane waves, not localized curvelets.
  - Genuinely-different frames (`WINDOWED_CURVELET`, `COMPACT_SHEARLET`, `STEERABLE_GABOR`, …)
    **fail closed** via `UnsupportedBasisFamily` (:298, :377-381) — they cannot compile a lever
    or claim a score until train+inflate op-parity + an equal-budget real-n600 through-R receipt exist.
- **Failure scenario:** none live — the alias cannot produce a curvelet CLAIM; any such claim is
  gated. The only residual is cosmetic (an ancestral name in the generator).
- **Fix-direction:** optional future cleanup — rename the generator alias to shed the ancestral
  "curvelet" token. No action required for NO-FAKE compliance.

### F3 — COSTATE ORGAN: λ = ∂S/∂x, advisory-only — VERIFIED HONEST (watch item)
- **Class checked:** 8 (surrogate/wrong-metric score claim), 1 (marker-without-work).
- **Severity:** FALSE-ALARM-VERIFIED-HONEST (with a named watch item).
- **Evidence (MEASURED / DERIVED):**
  - `src/tac/witness_dsl/costate_agent_dsl.py:1-40`: the compiled organ's methods are "the REAL
    wiring (sense → adjoint → decide → act), not a paper spec"; λ = ∂S/∂x "binds to
    `cgauge_master_action_v1` + `costate_lambda_marginal_ds_v1` via EquationBinding, resolved
    fail-closed against the canonical registry"; the organ is "exercised by the test suite
    against the live run directory's telemetry."
  - CONTAINMENT is a typed field: SAFE actions pass as **advisory artifacts**; HEAVY actions
    **RETURN `OperatorGoTickets` structurally — there is no execute path.** The organ makes
    recommendations, not autonomous heavy actions and not score claims.
- **Watch item (INFERRED, not a live fake):** per the just-measured organ-ceiling result
  (`n1_organ_capacity_ceiling_shrinkage_physics_residual_measured_20260714`), the learned-U
  organ wins *aggregate* d_seg but LOSES *per-class*. That nuance is already SURFACED as a
  MEASURED caveat, not hidden — so it is not a code fake today. It **would** become a class-8
  fake if any organ arm ever emits a "d_seg win / promote" verdict computed off the aggregate
  metric while per-class regresses. Recommend a per-class-facet gate on any future organ
  score-verdict (holistic-facet discipline, memory `watch-items-are-facets-never-lineage-scoped`).

---

## CLEAN SURFACES (stated plainly — a clean audit is a valid result)

- **CLEAN-1 — Score rows are honestly axis-tagged (class 8 clean).**
  `optimal_basis_20260714.py:220-233`: the only numeric d_seg rows (0.004244 polar,
  0.004259 self-oriented) carry `evidence=MEASURED_THROUGH_R_N600_FORMULATION` **and** the
  caveats "bounded warm-start ep675, seed0, **macOS-CPU advisory; archive bytes unmeasured**."
  No advisory number is dressed as authority; the sub-0.15 need is stated as a *need*, never a claim.

- **CLEAN-2 — Inert levers are SURFACED, not hidden (anti-fake handling of the inert class).**
  `experiments/train_levelset_witness_realized_through_R_mlx.py` tracks `term_inert_rows`
  (:159) and annotates inert flags per-config throughout (:873, :1131, :1211, :1252, :1314,
  :1322, :1332, :1363, :2133, :2478) — e.g. "inert for the training trajectory → persisted for
  provenance but NOT in [the loss]." This is the "off is a tracked queue, never a forgotten
  default" discipline in force: parsed-but-inert flags are excluded from the trajectory AND
  recorded, so the receiver-consumption-bijection (#417) class is handled honestly.

- **CLEAN-3 — Value-provenance rungs on every live scalar.**
  `spec_v9_cgauge.py` tags each constant with a rung (`measured_anchor` / `derived_at_config` /
  `derived_live`) per the value-provenance ladder — no bare constants. The 0.005318 flicker
  floor is cited as PROVED; the sub-0.15 need (0.00077–0.00118) is labeled a need "4.5–7× below" it.

- **CLEAN-4 — lever_registry is honest coverage-only.**
  `src/tac/witness_dsl/lever_registry.py` derives factory↔flag mapping by static AST (no
  hand-typed registry) and **explicitly declines** to claim any lever is "on" (:11-16: flag
  presence cannot decide activation; `PoseDecouple` ON iff `--w-pose==0`). It reports UNMAPPED
  gaps + STALE drift only — it does not overstate what it knows.

## Paradigm vs implementation (Catalog #307)
All findings are IMPLEMENTATION/GOVERNANCE-level. The V9·CGauge paradigm (covariant single
trunk, task-space witness, value-provenance ladder, advisory costate) is intact and, on the
audited surfaces, faithfully implemented. F1 is a missing *enforcement gate*, not a corrupted
mechanism.

## Bottom line for a score claim
Nothing on the live score-affecting path is a fake that blocks a score claim. Before the next
score claim, F1 should be reconciled (implement the named checks OR correct CLAUDE.md's
"STRICT completion" language) so the anti-fake protection is structural, and the F3 per-class
watch item should be enforced on any future organ verdict.
