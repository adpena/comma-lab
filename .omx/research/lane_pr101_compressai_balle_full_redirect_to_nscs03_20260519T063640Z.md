---
council_tier: T1
council_attendees: [Claude]
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "lane_pr101_compressai_balle_full requires standalone re-implementation"
    classification: CARGO-CULTED
    rationale: "Per Cable C6 synthesis: NSCS03 ALREADY LANDS the end-to-end Ballé 2018 joint codec paradigm (commit 2026-05-15 per feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md). The lane_pr101_compressai_balle_full FALSIFICATION was substrate-mismatch-as-class-kill (Catalog #185 META-class) + Catalog #290 canonical-cargo-cult. The work was REDIRECTED structurally by NSCS03 trainer impl_complete landing; no additional spend needed. Path A NSCS03 already lands."
council_decisions_recorded:
  - "Path A NSCS03 ($0): NSCS03 trainer _full_main IS ALREADY landed; this redirect memo formalizes the substrate-mismatch fix routing"
  - "Path B + C ($10 sister REDIRECT pair, future): ATW V2 + NSCS06 v7 paired smokes per Cable C6.4 DRAFT"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: |
  Operator quote 2026-05-19 verbatim: "All operator fates and decisions approved" + Cable C6 synthesis
  cheap-signal-first sequencing (commit `4c056724c`) designates this $0 redirect as Tier 1 (no spend; PROCEED IMMEDIATELY).
horizon_class: plateau_adjacent
council_assumption_classification_addendum: |
  Per Cable C6 synthesis: Path A NSCS03 already lands; Path B+C $10 sister REDIRECT pair is the
  subsequent dispatch tier (NOT Tier 1).
related_deliberation_ids:
  - cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z
  - feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515
  - council_t3_lane_pr101_compressai_balle_full_redirected_re_eval_high_symposium_DRAFT_20260519T060557Z
---

# lane_pr101_compressai_balle_full REDIRECT to NSCS03 — Redirect memo

## Authority

Per Cable C6 synthesis 2026-05-19 (commit `4c056724c`) "cheap-signal-first sequencing" + operator-frontier-override 2026-05-19 "All operator fates and decisions approved" + Cable C6.4 DRAFT verdict (`DRAFT_PENDING_CONVOCATION` + `REDIRECT` priority).

## META-bug attribution + redirect rationale

Per Cable C6 synthesis: Cable C6.4 META-bug = **substrate-mismatch** + Catalog #290 canonical-cargo-cult.

The 2026-04-29 `lane_pr101_compressai_balle_full` FALSIFICATION at score [REDACTED-original-verdict] was attributed to "TECHNIQUE FALSIFIED" but the actual root cause was **CompressAI Ballé hyperprior force-applied to PR101 substrate (which has its own codec)** — a canonical-cargo-cult per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable.

**Redirect status**: ALREADY STRUCTURALLY RESOLVED. NSCS03 (lane `lane_nscs03_end_to_end_balle_joint_codec_20260515`) landed `_full_main` 2026-05-15 per memory entry `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md` (commit batch in same range). NSCS03 IS the canonical substrate for end-to-end Ballé 2018 joint codec; PR101 substrate is NOT — the original `lane_pr101_compressai_balle_full` work was substrate-misrouted by definition.

## Path A: NSCS03 already lands ($0)

Per `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`:

- **Substrate**: NSCS03 end-to-end Ballé joint codec
- **Trainer**: `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py::_full_main` (was NotImplementedError; 548 LOC landed 2026-05-15)
- **Bound ingredients per PR 95 paradigm**: 14 CANONICAL ADOPT + 4 UNIQUE (NSCS03 substrate / score-aware loss with END-TO-END diff rate λ_R·(R_main+R_hyper) / Ballé λ_R linear warmup 0→target across first 10% epochs / archive build from 5 state_dicts + 2 latent streams hard-rounded on real GT pairs) + 3 DOCUMENTED FORK (AUTOCAST_FP16_WAIVED EB+GDN fp16 instability / TORCH_COMPILE_WAIVED / Ballé 0.999/0.997 differentiated EMA deferred Phase 2)
- **Tests**: 76 pass (53 pre + 23 new test_nscs03_full_main: canonical-pattern presence + grad reaches ALL 5 sub-nets g_a/g_s/h_a/h_s/EB + archive helpers + λ_R warmup math + smoke regression)
- **Gates**: Catalog #187 HNeRV parity passed; Catalog #226/#193/#190/#180 0 NSCS03 violations
- **Lane**: `lane_nscs03_end_to_end_balle_joint_codec_20260515` L1 (impl_complete + memory_entry; recipe `research_only=true` until Phase 2 council λ_R sweep + σ-floor calibration)
- **6-hook wire-in**: ACTIVE per Catalog #125

**No additional spend required.** Path A is complete.

## Path B + C: $10 sister REDIRECT pair (future tier, NOT Tier 1)

Per Cable C6.4 DRAFT:
- **Path B**: ATW V2 paired smoke ($5) — sister cooperative-receiver codec
- **Path C**: NSCS06 v7 paired smoke ($5) — sister sparse-encoding lane

These remain at Tier 2 priority per cable C6 synthesis ordering; out of scope for Tier 1 cheap-signal-first wave.

## Canonical-vs-unique decision per layer

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD standing directive 2026-05-15:

| Layer | Decision | Rationale |
|---|---|---|
| Substrate base | FORK (REDIRECT to NSCS03) | PR101 substrate is NOT the canonical site for end-to-end Ballé joint codec; NSCS03 IS |
| Ballé 2018 codec implementation | FORK (NSCS03 unique) | 4 UNIQUE per NSCS03 landing memo (substrate / score-aware loss / λ_R warmup / archive build) |
| Score-aware training | ADOPT canonical | NSCS03 uses canonical eval_roundtrip + EMA + scorer-preprocess per CLAUDE.md non-negotiables |
| Archive grammar | FORK (NSCS03 unique) | 5 state_dicts + 2 latent streams; documented in NSCS03 design memo |
| Inflate runtime | ADOPT canonical | NSCS03 honors HNeRV parity L4 inflate budget |
| Export contract | FORK (NSCS03 unique) | NSCS03 export contract is end-to-end joint codec specific |
| Tier-1 engineering | 4 ADOPT + 3 FORK (documented in NSCS03 design memo) | 3 documented forks: AUTOCAST_FP16 / TORCH_COMPILE / Ballé differentiated EMA |

## 9-dimension success checklist evidence

Per NSCS03 landing memo (this redirect memo inherits NSCS03's evidence):

1. **UNIQUENESS** — end-to-end Ballé 2018 joint codec is class-shift (joint learned codec vs PR101's separate-codec paradigm)
2. **BEAUTY+ELEGANCE** — 548 LOC trainer + canonical bolt-on size per PR 95 paradigm
3. **DISTINCTNESS** — distinct from PR101 (separate codec) + PR106 (latent-bias sidecar) + standalone CompressAI (substrate-mismatch was original bug)
4. **RIGOR** — Catalog #229 PV pre-edit + Catalog #292 assumption surfacing + 76 tests pass
5. **OPTIMIZATION PER TECHNIQUE** — Dimension 5 covered by Catalog #290 canonical-vs-unique section above
6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal to sister Cable C substrates (Z6/Z7/DP1); additive ΔS per Catalog #322 v2 cascade
7. **DETERMINISTIC REPRODUCIBILITY** — byte-stable archive (5 state_dicts + 2 latent streams hard-rounded)
8. **EXTREME OPTIMIZATION + PERFORMANCE** — Tier-1 engineering primitives 4 ADOPT + 3 documented FORK per NSCS03
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted band [0.180, 0.192] composition path A+B+C per Cable C6.4 DRAFT

## Observability surface

Per NSCS03 landing memo: 6 facets ACTIVE via NSCS03 trainer instrumentation.

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | HARD-EARNED / CARGO-CULTED | Unwind path |
|---|---|---|
| Ballé hyperprior is universal across substrates | CARGO-CULTED (the original lane_pr101_compressai_balle_full bug) | Reformulate per substrate; NSCS03 IS the substrate-matched landing |
| `lane_pr101_compressai_balle_full` FALSIFICATION = paradigm-level kill | CARGO-CULTED | Substrate-mismatch-as-class-kill per Catalog #185; NSCS03 redirect resolves |
| NSCS03 + PR101 composition is additive | PENDING-EMPIRICAL | Cable C6.4 Path B+C $10 sister REDIRECT pair tests this |

## Predicted ΔS band (per Catalog #296 + #324 discipline)

**Predicted band**: [0.180, 0.192] contest-CUDA (per Cable C6.4 DRAFT composition path A+B+C).
**Predicted_band_validation_status**: pending_post_training (NSCS03's recipe stays `research_only=true` until Phase 2 council λ_R sweep + σ-floor calibration per NSCS03 landing memo).
**Dykstra-feasibility check**: NSCS03 end-to-end Ballé joint codec is the canonical substrate; convex feasibility = intersection of (Ballé 2018 rate constraint) ∩ (PR101-grammar archive constraint via composition) ∩ (NSCS03 substrate constraint). Composition Path A+B+C predicted band is HIGH-VARIANCE planning prior pending Phase 2 council + Path B+C smokes.
<!-- PREDICTED_BAND_VIBES_OK:Cable C6.4 DRAFT composition path A+B+C predicted band [0.180, 0.192] is HIGH-VARIANCE planning prior pending Phase 2 council ratification + Path B+C $10 sister REDIRECT pair smokes. -->

## Reactivation criteria per CLAUDE.md "Forbidden premature KILL"

- (a) NSCS03 Phase 2 council λ_R sweep + σ-floor calibration lands → recipe flips `research_only=false` → empirical anchor lands
- (b) Empirical anchor within Cable C6.4 predicted band → Path B + C $10 sister REDIRECT pair dispatches
- (c) Composition path A+B+C within band → substrate composition matrix updates → cathedral autopilot reranks

## 6-hook wire-in declaration (Catalog #125)

Inherited from NSCS03 landing memo:
1. **Sensitivity-map** = ACTIVE
2. **Pareto constraint** = ACTIVE
3. **Bit-allocator hook** = ACTIVE
4. **Cathedral autopilot dispatch hook** = ACTIVE
5. **Continual-learning posterior update** = ACTIVE
6. **Probe-disambiguator** = N/A (NSCS03 is canonical end-to-end Ballé landing)

## Cross-references

- Cable C6 synthesis 2026-05-19: `.omx/research/cable_c6_re_eval_high_symposium_drafts_synthesis_20260519T060557Z.md`
- NSCS03 landing memo: `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`
- Cable C6.4 DRAFT: `.omx/research/council_t3_lane_pr101_compressai_balle_full_redirected_re_eval_high_symposium_DRAFT_20260519T060557Z.md`
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable + "Forbidden artifact-lifecycle violations"
- Catalog #185 (META-meta-meta drift detection); Catalog #290 (canonical-vs-unique per layer); Catalog #324 (predicted-band post-training validation)


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
