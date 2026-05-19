<!--
SPDX-License-Identifier: MIT
council_tier: T1
council_attendees: [Operator]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "approve_all_operator_routable_items_for_hf_jobs_segnet_surrogate_distillation"
  - "flip_recipe_research_only_true_to_false"
  - "flip_recipe_dispatch_enabled_false_to_true"
  - "declare_predicted_band_validation_status_pending_post_training_per_catalog_324"
  - "register_operator_frontier_override_ratification"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "all operator routable items approved"
related_deliberation_ids:
  - "council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519"
  - "hf_dataset_plus_hf_jobs_implementation_surface_landed_20260519"
deferred_substrate_id: hf_jobs_segnet_surrogate_distillation
-->

# Operator-approval capture — HF Jobs SegNet surrogate distillation (Items 1+2)

**UTC:** 2026-05-19T07:00:00Z
**Operator quote (verbatim):** *"all operator routable items approved"*
**Source operator-routable inventory:** `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_hf_dataset_plus_hf_jobs_implementation_surface_landed_20260519.md`
**Symposium memo this approval ratifies:** `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (T1 PROCEED_WITH_REVISIONS 5/6)

## Scope of approval

The operator-routable inventory in slot 7's landing memo enumerated 5 items:

1. **HF dataset push (first upload to Hub)** — Item 1 (this approval ratifies)
2. **T1 symposium revisions integration + recipe `dispatch_enabled: true` flip** — Item 2 (this approval ratifies)
3. Phase 2 per-pixel mIoU sister lane — slot 12 (separate operator-routable, parallel-dispatched)
4. `tools/operator_authorize.py` HF Jobs wire-in — slot 13 (separate operator-routable, parallel-dispatched)
5. `src/tac/preflight.py` canonical helper extension — slot 13 (separate operator-routable, parallel-dispatched)

The verbatim operator quote *"all operator routable items approved"* covers ALL FIVE per the briefing scope, with slots 7/12/13 handling items 1+2 / 3 / 4+5 respectively in parallel. This memo specifically captures items 1+2.

## Catalog #325 6-step contract compliance (per symposium memo §"Catalog #325 6-Step Contract Evidence")

| Step | Status | Memo section |
|------|--------|--------------|
| 1. Cargo-cult audit per Catalog #303 | ✓ COMPLETE | §1 Cargo-cult audit per assumption (10 assumptions classified) |
| 2. 9-dim checklist evidence per Catalog #294 | ✓ COMPLETE | §2 9-dimension success checklist evidence |
| 3. Observability surface per Catalog #305 | ✓ COMPLETE | §3 Observability surface |
| 4. Sextet pact deliberation | ✓ COMPLETE | §4 — Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary all participated |
| 5. Per-substrate reactivation criteria pinned | ✓ COMPLETE | §5 — 4 reactivation criteria enumerated |
| 6. Catalog #324 post-training Tier-C validation | ✓ COMPLETE | Recipe declares `predicted_band_validation_status: pending_post_training` per this approval |

## Recipe state transition (2026-05-19 ratified)

| Field | Pre-approval | Post-approval |
|-------|--------------|---------------|
| `research_only` | `true` | `false` |
| `dispatch_enabled` | `false` | `true` |
| `predicted_band_validation_status` | (already present) | `pending_post_training` per Catalog #324 |
| `predicted_band_reactivation_criteria` | (already present) | Tier-C density re-measurement criterion |
| `operator_approval_rationale` | (absent) | verbatim operator quote |
| `operator_approval_memo` | (absent) | reference to canonical capture memo (THIS file) |
| 3 prior `dispatch_blockers` | active | CLEARED |

Recipe file path: `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml`

## Sister-coordination disjoint-scope manifest per Catalog #230

This subagent's edit scope (Items 1+2 only):

- `.omx/operator_authorize_recipes/substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml` (recipe state flip)
- `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md` (append `## Operator-approval ratification` section only)
- `.omx/research/operator_authorizations/hf_jobs_segnet_surrogate_distillation_operator_approval_20260519T070000Z.md` (this NEW memo)
- `.omx/state/lane_registry.json` (lane gates marked)
- `.omx/state/council_deliberation_posterior.jsonl` (canonical posterior anchor written)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_hf_dataset_upload_plus_recipe_enable_landed_20260519.md` (landing memo)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md` (prepend)

NO touches to:

- `tools/operator_authorize.py` (slot 13 owns)
- `src/tac/preflight.py` (slot 13 owns)
- `CLAUDE.md` (slot 13 owns)
- Per-pixel mIoU sister lane recipe / files (slot 12 owns)

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution** — N/A (recipe-state flip; no signal contribution)
- **Hook #2 Pareto constraint** — N/A (no Pareto-relevant signal)
- **Hook #3 bit-allocator hook** — N/A (no bit-allocator signal)
- **Hook #4 cathedral autopilot dispatch hook** — **ACTIVE** (HF Jobs now becomes a routing target candidate via Catalog #335 auto-discovery; cathedral_consumers/ entry may follow in a future iteration)
- **Hook #5 continual-learning posterior update** — **ACTIVE** (this memo's frontmatter + canonical anchor written via `tac.council_continual_learning.append_council_anchor`)
- **Hook #6 probe-disambiguator** — N/A (no probe disambiguator needed; cheap-canary pattern documented for first-dispatch decision-making)

## Cite-chain

- T1 symposium memo: `.omx/research/council_t1_hf_jobs_segnet_surrogate_distillation_symposium_20260519.md`
- Slot 7 landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_hf_dataset_plus_hf_jobs_implementation_surface_landed_20260519.md`
- Catalog #325 standing rule: CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
- Catalog #324 standing rule: CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density (the phantom-predicted-band trap)"
- Catalog #300 v2 frontmatter standing rule: CLAUDE.md "Council hierarchy: 4-tier protocol"
- CLAUDE.md "Mission alignment — non-negotiable" Consequence 1 (operator-frontier-override discipline)
- HF dataset Hub URL: https://huggingface.co/datasets/adpena/comma-video-segnet-image-level-600pairs
- HF dataset manifest_sha256: `b0ba07e71640d110df6f59badb6cd831cedd529d1fdda942d83f743e44df4034`
- HF dataset Hub commit sha (post-card-update): `52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a`
- HF dataset Hub commit sha (initial upload): `613e9a2e5cd77716ec0b8c95846eef21df44dc73`
- HF dataset scorer_sha (canonical SegNet CPU contest_cpu_authoritative): `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
- Lane gates: `lane_hf_dataset_upload_and_recipe_enable_20260519` L2 (impl_complete + real_archive_empirical)
