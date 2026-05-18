---
council_tier: T2
council_attendees: [Yousfi, Fridrich, Wyner, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: FINDINGS_REQUIRES_FIX_WAVE
council_dissent:
  - member: Yousfi
    verbatim: "C6 IBPS landing memo claim 'PROCEED-unconditional 6-of-6' is overstated. The frontmatter is honest (council_verdict: PROCEED), but the landing memo's prose escalates to 'unconditional 6-of-6' despite TWO verbatim dissents from Contrarian + Assumption-Adversary. Per CLAUDE.md 'Apples-to-apples evidence discipline': the language must match the evidence. Recommend the landing memo's TL;DR + table row 4 be re-worded to 'PROCEED 6-of-6 with 2 verbatim dissents on language' to preserve maximum signal per Catalog #300 v2 maximum-signal preservation rule."
  - member: Wyner
    verbatim: "Cooperative-receiver canonical lens: the Z6 landing memo filename 'feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md' contains 'proceed_unconditional_unlock' but the actual verdict was PROCEED_WITH_REVISIONS_v2 (unchanged from 2026-05-16 with #4 new revision added). The filename is a sister-instance of Catalog #249 'phantom-score directory trap' applied at the landing-memo filename layer: the filename claims outcome the body does not deliver. Per the operator's directive to honor maximum-signal preservation: this is a structural recurrence of the misleading-filename class that should be either renamed OR flagged as a structural protection extension target."
  - member: Contrarian
    verbatim: "Catalog #315 has a structural blind spot for C6 IBPS: the council deliberation row's deferred_substrate_id field is null, so the gate cannot join the PROCEED verdict back to the C6 substrate lane. The family-token fallback list (nscs01-06 / z3-z8 / atw_codec / wunderkind_g1 / tishby_ib_pure / rudin / darts_supernet) does NOT include 'c6' or 'time_traveler' families. The C6 IBPS landing memo claims Catalog #315 is satisfied, but the gate STRUCTURALLY cannot see C6 — the satisfaction is asserted, not enforced. This is the same META class as Catalog #185 META-meta drift but at the gate-join-key level: 'gate claims to be enforcing X but cannot see X structurally'. Recommend either (a) C6 council posterior row be re-emitted with deferred_substrate_id='c6_e4_mdl_ibps_substrate' OR equivalent canonical surface id, OR (b) Catalog #315 family-token list be extended to include 'c6' + 'time_traveler'."
  - member: Assumption-Adversary
    verbatim: "Per the MANDATORY assumption-challenge axis (Catalog #291 item #8): the SHARED ASSUMPTION operating across this entire wave is *'class-shift substrates (C6 IBPS / Z6 / Z7 / NSCS06) will produce empirical anchors that beat the 0.193 contest-CPU plateau when dispatched at their predicted bands [0.05, 0.16].'* I classify this CARGO-CULTED. The HARD-EARNED evidence we have is: Tier-C density 2.67e-5 confirms C6 IBPS is structurally ACROSS-CLASS per Catalog #227 boundary semantics. The CARGO-CULTED part: NO across-class substrate has produced a paired CPU/CUDA empirical anchor below 0.193 yet. The 7+ historical Modal smokes on C6 IBPS demonstrate trainer-dispatch readiness, NOT score-band achievability. The Dykstra polytope [0.099879, 0.531541] CONTAINS the predicted band [0.113, 0.163] but it also contains every score up to 0.531 — feasibility is necessary, not sufficient. My assumption-violation hypothesis: 'if we dispatched C6 IBPS at $0.76 right now and the score landed at 0.220 (within Dykstra polytope but well outside predicted band), the wave's planning artifacts would be structurally consistent with the failure but the predicted-band CARGO-CULT would not be visible to autopilot.' Recommend the dispatch be wired with a fail-closed Tier-C re-confirmation per Assumption-Adversary's own CARGO-CULTED classification in the C6 council itself."
council_assumption_adversary_verdict:
  - assumption: "Wave's PROCEED verdicts (C6 IBPS PROCEED + Z6 PROCEED_WITH_REVISIONS_v2) are structurally enforced by Catalog #315"
    classification: CARGO-CULTED
    rationale: "C6 council posterior row has deferred_substrate_id=None; Catalog #315 family-token fallback does not include 'c6' or 'time_traveler' families. The gate is structurally blind to both substrates. The PROCEED verdicts are honestly recorded in the council memos but NOT structurally enforced."
  - assumption: "Class-shift substrates will produce empirical anchors below 0.193 plateau"
    classification: CARGO-CULTED
    rationale: "Zero across-class substrate has produced a paired CPU/CUDA empirical anchor below 0.193. Tier-C density evidence confirms class-shift potential but does NOT confirm band achievability. Per Contrarian's verbatim dissent in the C6 council: 'signing off on feasibility-consistency, NOT band-achievability'."
  - assumption: "Catalog #287/#249/#319/#321/#322/#323 phantom-score class is fully extincted"
    classification: HARD-EARNED-IN-PRINCIPLE
    rationale: "Catalog #321 + #322 + #323 cover the persistent-artifact + autopilot-consumer + canonical-Provenance surfaces. Catalog #323 currently warn-only at 544 baseline (3 new C6 IBPS state JSONs would inherit warn-only). The META-class IS structurally protected but not yet strict-enforced; one new phantom artifact CAN still ship under warn-only."
  - assumption: "Bare /commit pattern does not absorb sister-subagent files"
    classification: CARGO-CULTED
    rationale: "Catalog #314 reports 15 absorption-pattern violations from THIS wave (commits 42ab329 / ecc3f87 / cb92c90 / etc). The operator's /commit slash command bypasses tools/subagent_commit_serializer.py and absorbs in-flight subagent files. Same bug class as 2026-05-15 WAVE-D 2c957c31e forensic finding."
  - assumption: "Z6 landing memo filename matches the actual verdict outcome"
    classification: CARGO-CULTED
    rationale: "Filename 'feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md' contains 'proceed_unconditional_unlock' but the actual verdict was PROCEED_WITH_REVISIONS_v2 (no unlock occurred). Sister of Catalog #249 phantom-score directory at landing-memo filename layer."
  - assumption: "Wave's empirical claims all carry [evidence:<path>] or canonical Provenance tags per Catalog #287"
    classification: HARD-EARNED-WITH-EXCEPTION
    rationale: "Most landing memos correctly tag predicted bands as [prediction] and empirical anchors with archive sha citations. Exception: 3 C6 IBPS fix JSONs (dykstra_feasibility / composition_alpha / tier_c_density) lack canonical Provenance dataclass embed (covered by Catalog #323 warn-only state); they DO carry score_claim=False so are semantically non-promotable."
council_decisions_recorded:
  - "VERDICT: FINDINGS_REQUIRES_FIX_WAVE — 4 HIGH + 3 MEDIUM + 2 LOW findings; clean-pass counter resets to 0; FIX-WAVE-R1 spec created"
  - "TaskCreate spec: FIX-WAVE-R1 to address F1-F4 HIGH findings before R2 re-fires"
  - "Operator-routable: clarify language in C6 IBPS landing memo + rename Z6 landing memo filename + add C6/time_traveler tokens to Catalog #315 family list OR backfill C6 council deferred_substrate_id"
  - "Per CLAUDE.md 'Forbidden premature KILL' + 'Apples-to-apples evidence discipline': the substrate work itself is sound (Z6 honest defer + C6 IBPS honest PROCEED); the findings target META-class protection gaps + landing-artifact discipline, not the underlying substrate decisions"
  - "Continual-learning anchor appended via tac.council_continual_learning.append_council_anchor per Catalog #300"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_protecting
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
related_deliberation_ids:
  - council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517
  - council_z6_phase_2_sextet_proceed_unconditional_unlock_20260517
  - feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517
  - feedback_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_landed_20260517
  - feedback_asymptotic_pursuit_substrate_class_shift_top_priority_landed_20260517
originSessionId: lane_recursive_adversarial_review_r1_post_redo_council_rotation_a_20260517
---

# R1 Recursive Adversarial Review — Post-PROVENANCE/Z6/C6 wave 2026-05-17

**Lane**: `lane_recursive_adversarial_review_r1_post_redo_council_rotation_a_20260517` (L1: impl_complete + three_clean_review_R1 attempted + memory_entry pending)
**Task**: #829
**Tier**: T2 (Inner-Skunkworks sextet, rotation A)
**Council attendees**: Yousfi + Fridrich + Wyner + Contrarian + Assumption-Adversary (5 of 6 sextet seats; Shannon+Dykstra recused as authors of sister-wave landings)
**Verdict**: **FINDINGS_REQUIRES_FIX_WAVE**
**Clean-pass counter**: **0/3** (RESET; FIX-WAVE-R1 must land before R2 fires)
**$0 GPU. ~2.5h editor. NO commits per operator NON-NEGOTIABLE.**

## Mandatory Assumption-Challenge Axis (Catalog #291 item #8)

**The SHARED ASSUMPTION operating across this entire wave**:

> *"Class-shift substrates (C6 IBPS / Z6 / Z7 / NSCS06) will produce empirical anchors that beat the 0.193 contest-CPU plateau when dispatched at their predicted bands [0.05, 0.16]."*

**Classification**: CARGO-CULTED

**Rationale**:
- HARD-EARNED: Tier-C density 2.67e-5 confirms C6 IBPS structurally ACROSS-CLASS per Catalog #227 boundary semantics.
- CARGO-CULTED: NO across-class substrate has produced a paired CPU/CUDA empirical anchor below 0.193 yet. 7+ historical Modal smokes on C6 IBPS demonstrate trainer-dispatch readiness, NOT score-band achievability. The Dykstra polytope [0.099879, 0.531541] CONTAINS predicted band [0.113, 0.163] but also contains every score up to 0.531 — feasibility is necessary, not sufficient.

**Assumption-violation hypothesis**: If we dispatched C6 IBPS at $0.76 right now and the score landed at 0.220 (within Dykstra polytope but well outside predicted band), the wave's planning artifacts would be structurally consistent with the failure but the predicted-band CARGO-CULT would not be visible to autopilot.

**Required action**: Wire fail-closed Tier-C re-confirmation per Assumption-Adversary's own CARGO-CULTED classification in the C6 council itself (which already flagged "$0.76 paired CPU+CUDA dispatch is sufficient" as CARGO-CULTED).

## Findings Table

| # | Severity | Surface | Finding | Proposed fix | Blocker for R2 |
|---|---|---|---|---|---|
| F1 | HIGH | `.omx/state/council_deliberation_posterior.jsonl` row for `council_c6_ibps_phase_2_sextet_for_dispatch_unlock_20260517` + `src/tac/preflight.py` Catalog #315 family-token list | C6 IBPS council posterior row has `deferred_substrate_id=None`. Catalog #315's family-token fallback list does NOT include 'c6' or 'time_traveler' families. Gate is STRUCTURALLY BLIND to C6 IBPS. The landing memo claim "Catalog #315 PROCEED-unconditional satisfied" is asserted, not enforced. Same META class as Catalog #185 drift but at gate-join-key level. | (a) Re-emit C6 council posterior row with `deferred_substrate_id='c6_e4_mdl_ibps_substrate'`; OR (b) extend `_CHECK_315_SUBSTRATE_FAMILY_TOKENS` to include 'c6_e4_mdl_ibps' + 'time_traveler' + sister candidates ('atw_codec_v2', etc); recommended: BOTH for defense-in-depth. | YES — gate enforcement gap must be closed before R2 re-fires on next dispatch wave |
| F2 | HIGH | CLAUDE.md Catalog #131 row + `tools/asymptotic_pursuit_candidate_readiness_assessment.py:704` | Catalog #131 claims "Live count: 0 -> STRICT" but the gate currently returns 1 violation: `tools/asymptotic_pursuit_candidate_readiness_assessment.py:704` bare write to `.omx/state/asymptotic_pursuit/readiness_assessment_*.json` without fcntl-lock or canonical helper. Catalog #185 META-meta drift detection FIRES on this. The asymptotic_pursuit tool was added by THIS wave. | Wrap the `path.write_text(...)` in `tools/asymptotic_pursuit_candidate_readiness_assessment.py:704` with `fcntl.flock(LOCK_EX)` OR route through canonical helper. Alternative: same-line `# BARE_WRITE_OK:<rationale>` waiver if the file is genuinely single-writer per session. | YES — Catalog #185 META-meta drift is forbidden by structural protection rule |
| F3 | HIGH | `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md` filename | Filename contains 'proceed_unconditional_unlock' but actual verdict was PROCEED_WITH_REVISIONS_v2 (unchanged from 2026-05-16, with #4 new revision added). Sister-instance of Catalog #249 'phantom-score directory trap' applied at landing-memo filename layer. Future agents reading the memory index will mis-classify the wave's outcome by filename alone. | Rename memo to `feedback_z6_phase_2_sextet_council_proceed_with_revisions_v2_landed_20260517.md` per CLAUDE.md "Forbidden misleading-directory-name (the phantom-score directory trap)" extended to landing-memo filename layer. Alternative: add prefix `feedback_z6_phase_2_sextet_council_REFUSED_unconditional_unlock_landed_*.md` to reflect actual outcome. | YES — misleading filename can mis-classify the wave's outcome at index-scan time |
| F4 | HIGH | `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517.md` TL;DR + row 4 of 4-fix closure table | Landing memo TL;DR claims "Sextet PROCEED-unconditional 6-of-6" and 4-fix closure table row 4 says "PROCEED-unconditional 6-of-6". BUT the council frontmatter says `council_verdict: PROCEED` (not PROCEED_UNCONDITIONAL) AND records TWO verbatim dissents (Contrarian + Assumption-Adversary) who both explicitly note PROCEED is on outcome, NOT on language. Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #300 maximum-signal preservation: the prose escalates beyond the evidence. | Re-word TL;DR + table row 4 to "PROCEED 6-of-6 with 2 verbatim dissents on language" OR "PROCEED with caveat dissents" to preserve maximum signal. The fix is editorial; no structural change required. | YES — landing memo's prose must be apples-to-apples with the council frontmatter |
| F5 | MEDIUM | Catalog #314 absorption pattern + bare-commit `/commit` slash command | Catalog #314 reports 15 absorption-pattern violations from THIS wave. Sample commits: `42ab329`, `ecc3f87`, `cb92c90`, `a0b5da2d7`. Bare `/commit` slash command (per `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/commit-commands/commands/commit.md`) bypasses `tools/subagent_commit_serializer.py` and absorbs in-flight subagent files. Same bug class as 2026-05-15 WAVE-D 2c957c31e forensic finding. | (a) Operator-routable: switch `/commit` plugin to use canonical serializer; OR (b) operator audits the 15 absorption commits to confirm intent. NB: Catalog #314 is WARN-ONLY by design per CLAUDE.md text; the warn-only state IS the intent until a backfill window proves the pattern is extincted. | NO — Catalog #314 is warn-only by design; flagged for operator awareness |
| F6 | MEDIUM | `feedback_provenance_canonical_fix_meta_class_extinction_landed_20260517.md` | Landing memo has NO YAML frontmatter at all. The wave's other 4 landing memos (Z6/C6/REDO/ASYMPTOTIC) all carry full Catalog #300 v2 frontmatter. Provenance landing is NOT a council deliberation per se, so Catalog #300 does not strictly apply, but the inconsistency reduces cross-memo machine-readability. | Add minimal YAML frontmatter to PROVENANCE landing memo (council_tier: N/A_meta_class_protection_landing, horizon_class: frontier_protecting, council_predicted_mission_contribution: frontier_protecting). | NO — convention not contract; flagged for cross-wave hygiene |
| F7 | MEDIUM | `feedback_redo_pivot_fix_all_phantom_score_substrate_class_shift_q4_budget_redirect_landed_20260517.md` frontmatter | REDO+PIVOT landing memo frontmatter lacks `horizon_class` field. The other 4 wave memos have it. Catalog #309 requires horizon_class declaration only for design memos with predicted-band sections (this memo does not have one). | Add `horizon_class: frontier_protecting` (since this is a structural fix wave, not a substrate scaffold). | NO — convention not contract; flagged for cross-wave hygiene |
| F8 | LOW | `.omx/state/{dykstra_feasibility,composition_alpha,tier_c_density_reconciliation}_c6_e4_mdl_ibps*.json` | 3 C6 IBPS fix JSONs do NOT carry canonical `provenance` dataclass embed per Catalog #323. Two of them carry `score_claim=False` so are semantically non-promotable, but they would still be flagged by Catalog #323 strict-mode (which is currently warn-only at 544 baseline). | When Catalog #323 strict-flips (op-routable #5 in PROVENANCE landing), backfill these 3 JSONs with `provenance: Provenance` dataclass embed via `build_provenance_for_research_sidecar(...)` per the canonical adapter pattern. | NO — Catalog #323 is warn-only; these 3 JSONs are forward-looking backfill candidates |
| F9 | LOW | All wave landing memos: Catalog #316 frontier section in reports/latest.md | The 5 wave landing memos do NOT cite the canonical frontier (0.19205 [contest-CPU] / 0.20533 [contest-CUDA] per Catalog #316 + `tools/scan_best_anchor_per_axis.py`). Per CLAUDE.md "Frontier target — non-negotiable": every score-lowering wave should anchor against current frontier. C6 IBPS predicted band [0.113, 0.163] is BELOW 0.19205 (good); Z6 predicted band [0.13, 0.16] is BELOW 0.19205 (good). | Cross-reference current frontier in landing memos' TL;DR sections so future agents see the apples-to-apples comparison without re-deriving it. | NO — convention not contract; flagged for cross-wave hygiene |

## Per-axis audit findings

### Axis (a) Phantom-score class recurrence

CLEAN at META-class layer:
- Catalog #321 (research-sidecar phantom): 0 violations
- Catalog #322 (autopilot adjustment from phantom composition): 0 violations
- Catalog #319 (Wyner-Ziv phantom-savings from research sidecar; renamed `check_no_phantom_wyner_ziv_savings_from_research_sidecar`): 0 violations

Catalog #323 reports 544 violations (warn-only baseline). 3 new C6 IBPS state JSONs would inherit warn-only state per F8.

Catalog #249 (misleading device-named output dirs): 44 violations (pre-existing warn-only baseline).
Catalog #287 (docstring overstatement without evidence tag): 20 violations (pre-existing warn-only baseline).

**Finding**: F8 (LOW) — 3 wave-emitted JSONs lack canonical Provenance but carry `score_claim=False`.

### Axis (b) Cargo-cult assumptions

The wave's own Assumption-Adversary verdicts honestly classify cargo-cults:
- Z6: classified 2 of 6 as CARGO-CULTED ("Council re-deliberation can advance verdict absent revisions" + "Z6 paradigm IS empirically falsified by probes")
- C6 IBPS: classified 1 of 6 as CARGO-CULTED ("$0.76 sufficient is only UPPER bound on disconfirmation")
- ASYMPTOTIC PURSUIT: classified 1 of 2 as CARGO-CULTED ("All 6 candidates equally credible")

The MISSING META-cargo-cult that the wave's own councils did NOT surface (per the mandatory Catalog #291 item #8 axis): **"Class-shift substrates will produce empirical anchors below 0.193 plateau"** — this is the shared backdrop the wave operates within. Surfaced explicitly above.

### Axis (c) Dead-flag wiring per Catalog #12

CLEAN per my inspection of:
- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2` (Q3 v2 cascade): cascade 1/2/3 paths all reachable; load_deliverability_proof_for_archive imported correctly with ImportError fallback to 1.0 passthrough.
- `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_venn_classification_v2`: thin wrapper delegates correctly.
- C6 IBPS recipe `predicted_band` field: consumed via `tools/scan_best_anchor_per_axis.py` + Catalog #316 frontier scan (sister of Catalog #240).
- Catalog #322/#323 STRICT gate stubs: wired into `preflight_all()` strict=False per CLAUDE.md text; live count matches CLAUDE.md text.

**No findings** in this axis.

### Axis (d) Call-site trace correctness

CLEAN per inspection. The cathedral_autopilot v2 cascade IS loading DeliverabilityProof per `_venn_deliverability_reward_factor_for_archive` at `tools/cathedral_autopilot_autonomous_loop.py:1062-1112`. The Provenance backward-compat adapters (`tac.provenance.adapters`) are not yet wired into existing dataclasses (per the PROVENANCE landing memo's "backward-compat" intentional design choice; consumers can call adapter(legacy_obj) on read).

**No findings** in this axis.

### Axis (e) MPS/CPU/CUDA axis discipline per Catalog #127/#192

CLEAN. C6 IBPS recipe declares `target_modes: [contest_one_video_replay, contest_generalized, research_substrate]`. Z6 council memo correctly uses `[prediction]` axis tag (NOT `[contest-CUDA]` or `[contest-CPU]`). PROVENANCE adapters enforce axis-hardware-substrate triple via `_grade_axis_hardware_invariant` per `tac.provenance.contract.Provenance.__post_init__`.

**No findings** in this axis.

### Axis (f) HNeRV parity L1-L13 compliance

The C6 IBPS substrate is `independent_substrate` per Catalog #173 (architecturally distinct from HNeRV-family). The 24-dim latent + β-sweep [0.001, 1.0] probe-disambiguator is canonical Catalog #125 hook #6 wire-in. C6 trainer's `_full_main` is implemented (not NotImplementedError per Catalog #240).

The Z6-v1 single-layer FiLM predictor is correctly diagnosed by Rao (grand council seat) as lacking capacity to express semantic ego-conditioning at scorer's stride-2 stem — this is HNeRV parity L5 (full renderer not single-component slot) re-cast for predictive coding.

**No findings** in this axis.

### Axis (g) Every empirical claim has [evidence:path] tag per Catalog #287

Wave's empirical claims correctly tagged:
- Tier-C density 2.67e-5 → cited archive sha + `.omx/state/...mdl_ablation.json`
- Dykstra polytope [0.099879, 0.531541] → cited `.omx/state/dykstra_feasibility_c6_e4_mdl_ibps.json`
- composition_alpha=1.0 → cited `.omx/state/composition_alpha_c6_e4_mdl_ibps_x_wyner_ziv.json`
- Z6 probe corpus → cited 3 codex artifacts at `.omx/research/l5_v2_z6_*_20260517_codex.md`
- Predicted bands tagged `[prediction]` correctly

**No findings** in this axis.

### Axis (h) Every new STRICT gate has CLAUDE.md row + matches text + live count truth

CLEAN at the new-gate layer (Catalog #319/#320/#321/#322/#323 all have CLAUDE.md rows). **Catalog #185 META-meta drift detection FIRES on Catalog #131** (live count 0 → STRICT in CLAUDE.md vs 1 violation in gate) — this is F2.

### Axis (i) MANDATORY ASSUMPTION-CHALLENGE AXIS per Catalog #291

Addressed in the dedicated section above.

### Axis (j) Cross-wave consistency

- REDO+PIVOT correctly marked 3 lanes DEFERRED (super_additive / q6_preprobe / batched_815). C6 IBPS sextet acknowledges DEFERRED state on Z6 path (cost-adjusted redirect to C6 IBPS at $0.76).
- PROVENANCE backward-compat adapters preserve sister-subagent autonomy. No conflict observed.
- Catalog #322 vs Catalog #323 boundary: #322 covers autopilot-consumer surface (substrate_composition_matrix.json + pairwise_alpha_*.json); #323 covers persisted-artifact-row level umbrella. Clear, no overlap.

**No additional findings** in this axis.

### Axis (k) Provenance audit on wave's own artifacts

`tools/audit_provenance_compliance.py --summary` reports 196 violations across all .omx/state + experiments/results (warn-only). 3 wave-emitted C6 IBPS fix JSONs lack Provenance embed (F8). All other wave artifacts (council memos, posterior anchors) are either accepted-by-scope (council memos are markdown not JSON) or canonical-helper-routed (canonical posterior write via `tac.council_continual_learning.append_council_anchor`).

**Finding**: F8 (LOW) — flagged above.

## Sister-subagent ownership map honored

This R1 review declared scope to `.omx/research/` + memory + tasks (via TaskCreate) ONLY. No `src/tac/` or `tools/` edits. Sister-subagents in flight (2 in last 6 hours):
- `c6_ibps_first_asymptotic_dispa*` — touches `.omx/state/lane_registry.json` + own premise verifier; DISJOINT
- `z6_v2_redesign_cargo_cult_unwi*` — touches `.omx/state/lane_registry.json`; DISJOINT

Catalog #302 sister-subagent scope overlap: 0 violations (gate confirms no collision).

## Premise verification per Catalog #229

`.omx/tmp/recursive_adversarial_review_r1_premise_verifier.txt` — 15 PVs all VERIFIED before any output.

## Checkpoint discipline per Catalog #206

- Step 1: pre-flight reads complete (CLAUDE.md / AGENTS.md / state tracker / all 5 wave memos + their council sister memos + 3 Z6 probe codex artifacts)
- Step 2: premise verifier written; 15 PVs VERIFIED
- Step 3 (this commit): council memo + landing memo written; canonical posterior appended; lane marks landed

## 9-dimension success checklist evidence

1. **UNIQUENESS**: First R1 review covering the PROVENANCE + Z6 + C6 IBPS wave; first to surface the C6 Catalog #315 join-blind-spot
2. **BEAUTY + ELEGANCE**: 9 findings ranked HIGH→LOW; each maps to specific surface + fix; reviewable in 30 sec via table
3. **DISTINCTNESS**: Distinct from R2 (rotation B) + R3 (rotation C) per the protocol; council rotation A is Yousfi+Fridrich+Wyner+Contrarian+Assumption-Adversary
4. **RIGOR**: 15 PVs + 5-of-6 sextet quorum + 4 substantive dissents recorded verbatim + 6 Assumption-Adversary classifications + MANDATORY axis #8 surfaced
5. **OPTIMIZATION PER TECHNIQUE**: N/A — this is a review, not a substrate
6. **STACK-OF-STACKS COMPOSABILITY**: F1+F2+F3+F4 fixes compose orthogonally; each is independent
7. **DETERMINISTIC REPRODUCIBILITY**: Premise verifier + canonical posterior anchor + this memo all byte-stable
8. **EXTREME OPTIMIZATION + PERFORMANCE**: $0 GPU; ~2.5h editor
9. **OPTIMAL MINIMAL CONTEST SCORE**: Frontier-protecting (prevents wave's META-class protection gaps from masking real contest-score regressions)

## Observability surface

- **Inspectable per layer**: 9 findings + 5 dissent verbatims + 6 Assumption-Adversary classifications
- **Decomposable per signal**: findings ranked HIGH→LOW with severity + surface + proposed fix
- **Diff-able across runs**: canonical posterior anchor in `.omx/state/council_deliberation_posterior.jsonl`; future R1 sister reviews can `query_anchors_by_topic("recursive_adversarial_review_r1")` to track evolution
- **Queryable post-hoc**: structured frontmatter per Catalog #300 v2
- **Cite-able**: 5 related_deliberation_ids + 15 PVs + this memo + landing memo
- **Counterfactual-able**: F1's family-token-list-fix is the counterfactual — without it, future C6 IBPS dispatches will silently bypass Catalog #315

## Cargo-cult audit per assumption

| # | Assumption | Classification | Rationale |
|---|---|---|---|
| 1 | Wave's PROCEED verdicts are structurally enforced by Catalog #315 | CARGO-CULTED (F1) | C6 council posterior row deferred_substrate_id=None + family-token list lacks 'c6' |
| 2 | Catalog #131 live count is 0 as CLAUDE.md claims | CARGO-CULTED (F2) | Gate returns 1 violation in asymptotic_pursuit tool |
| 3 | Z6 landing memo filename matches verdict | CARGO-CULTED (F3) | Filename says 'proceed_unconditional_unlock' but verdict is PROCEED_WITH_REVISIONS_v2 |
| 4 | C6 IBPS landing memo prose matches frontmatter | CARGO-CULTED (F4) | TL;DR escalates to "PROCEED-unconditional 6-of-6" despite 2 verbatim dissents |
| 5 | Class-shift substrates will produce empirical anchors below 0.193 | CARGO-CULTED (assumption-challenge axis #8) | No across-class substrate has produced paired anchor below 0.193 |
| 6 | Bare /commit pattern is benign | CARGO-CULTED (F5) | Catalog #314 reports 15 absorption violations from this wave |
| 7 | Wave's empirical claims all carry canonical evidence tags | HARD-EARNED-WITH-EXCEPTION (F8) | 3 C6 IBPS fix JSONs lack Provenance embed but carry score_claim=False |
| 8 | All landing memos carry Catalog #300 v2 frontmatter | HARD-EARNED-WITH-EXCEPTIONS (F6+F7) | PROVENANCE lacks frontmatter; REDO+PIVOT lacks horizon_class |

## Predicted ΔS band

NOT APPLICABLE — this is a recursive adversarial review, not a substrate scaffold. No score band predicted.

## Cross-stack wire-in declaration per Catalog #125 (6 hooks)

| # | Hook | Status |
|---|---|---|
| 1 | Sensitivity-map contribution (`tac.sensitivity_map.*`) | N/A — review only; no score-axis weight contribution |
| 2 | Pareto constraint (`tac.pareto_*`) | N/A — review only; no Pareto constraint added |
| 3 | Bit-allocator hook | N/A — review only |
| 4 | Cathedral autopilot dispatch hook | ACTIVE — findings F1+F2 directly inform autopilot consumer behavior (Catalog #315 + #185 sister gates) |
| 5 | Continual-learning posterior update | ACTIVE — council deliberation anchor appended via `tac.council_continual_learning.append_council_anchor` per Catalog #300 |
| 6 | Probe-disambiguator | ACTIVE — the assumption-challenge axis hypothesis IS the canonical probe-disambiguator: "if C6 IBPS dispatched at $0.76 lands outside predicted band, does autopilot have visible CARGO-CULT classification?" |

## Op-routables for R2 (#830) dispatch

1. **PREREQUISITE: FIX-WAVE-R1 must land** addressing F1-F4 HIGH findings before R2 fires. R2 rotation B (Boyd + Atick + Tishby) re-fires AFTER fix wave.
2. **R2 scope**: rotation B (Boyd + Atick + Tishby + Contrarian + Assumption-Adversary) covers the SAME wave but from a different adversarial perspective (optimization-feasibility + cooperative-receiver + IB framework lenses).
3. **R2 MUST honor item #8 assumption-challenge axis**: independently surface its own shared-assumption hypothesis (different from R1's "class-shift substrates beat 0.193").
4. **3-clean-pass protocol**: R1 returned FINDINGS, counter resets to 0/3. R2 re-fires AFTER FIX-WAVE-R1 lands.
5. **HOLD on new dispatches**: per operator-agreed sequence, no new feature dispatches until 3-clean-pass SEAL achieved.

## FIX-WAVE-R1 spec (for sister subagent dispatch via TaskCreate)

**Lane**: `lane_fix_wave_r1_post_provenance_z6_c6_wave_20260517`
**Task**: TBD (will be created by parent)
**Subagent ownership scope**:
- READ: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md` + `feedback_c6_ibps_4_recipe_fixes_dispatch_unlock_landed_20260517.md`
- EDIT: `src/tac/preflight.py` (extend `_CHECK_315_SUBSTRATE_FAMILY_TOKENS` per F1)
- EDIT: `tools/asymptotic_pursuit_candidate_readiness_assessment.py:704` (add fcntl-lock per F2)
- EDIT (rename): `feedback_z6_phase_2_sextet_council_proceed_unconditional_unlock_landed_20260517.md` → `feedback_z6_phase_2_sextet_council_refused_proceed_unconditional_unlock_landed_20260517.md` per F3
- EDIT: C6 IBPS landing memo TL;DR + table row 4 per F4
- EDIT: `.omx/state/council_deliberation_posterior.jsonl` (append NEW row with `deferred_substrate_id='c6_e4_mdl_ibps_substrate'` per F1 alt-path)
- EDIT: CLAUDE.md Catalog #131 row to acknowledge new live-count baseline OR backfill the asymptotic_pursuit tool fix
- TESTS: extend `src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py` with c6/time_traveler family-token regression test
- Premise verification per Catalog #229
- Checkpoint discipline per Catalog #206
- Sister-subagent ownership map per Catalog #230

**Acceptance criteria for FIX-WAVE-R1**:
- Catalog #131 returns 0 violations (or CLAUDE.md text updated to match)
- Catalog #315 family-token list includes 'c6_e4_mdl_ibps' + 'time_traveler' OR C6 council posterior row backfilled with deferred_substrate_id
- Z6 landing memo filename renamed to reflect actual PROCEED_WITH_REVISIONS_v2 outcome
- C6 IBPS landing memo TL;DR + table row 4 re-worded apples-to-apples with council frontmatter
- All 5 META-meta CLAUDE.md gates (#118/#159/#176/#185/#299) return 0 violations

## Cross-references

- Catalog #185 META-meta drift detection (the gate that surfaced F2)
- Catalog #249 phantom-score directory trap (sister at filename layer; F3)
- Catalog #287 docstring-overstatement (sister at empirical-claim layer)
- Catalog #291 META-ASSUMPTION cadence (mandatory axis #8 satisfied by Assumption-Adversary verdict above)
- Catalog #292 per-deliberation assumption surfacing (satisfied by Assumption-Adversary frontmatter field)
- Catalog #294 9-dimension success checklist (satisfied via dedicated section above)
- Catalog #296 Dykstra-feasibility (cited in C6 IBPS landing memo correctly)
- Catalog #300 v2 council deliberation frontmatter (satisfied by this memo's frontmatter)
- Catalog #303 cargo-cult audit per assumption (satisfied via dedicated section above)
- Catalog #305 observability surface (satisfied via dedicated section above)
- Catalog #313 probe-outcomes ledger (sister of F1 — C6 IBPS post-smoke probe-outcomes registration is the runtime sister)
- Catalog #314 absorption pattern (surfaced F5)
- Catalog #315 OPTIMAL FORM gate (F1 is the structural gap)
- Catalog #316 frontier scan + frontier signal loss prevention (sister of F9)
- Catalog #319/#321/#322/#323 phantom-score class meta-protection (clean at META layer; F8 is forward-looking)
- CLAUDE.md "Recursive adversarial review protocol" (item #8 mandatory assumption-challenge satisfied)
- CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" (F1 is the structural enforcement gap)
- CLAUDE.md "Apples-to-apples evidence discipline" (F4 is the prose-vs-evidence-axis mismatch)
- CLAUDE.md "Forbidden misleading-directory-name (the phantom-score directory trap)" (F3 is the filename-layer extension)
- CLAUDE.md "Subagent coherence-by-default" (Catalog #314 absorption pattern persistence is the operational sister)

---

**STATUS**: R1 RECURSIVE ADVERSARIAL REVIEW LANDED 2026-05-17. Verdict: FINDINGS_REQUIRES_FIX_WAVE. Clean-pass counter: 0/3 (RESET). FIX-WAVE-R1 specified above. R2 (#830) HOLDS until FIX-WAVE-R1 lands.
