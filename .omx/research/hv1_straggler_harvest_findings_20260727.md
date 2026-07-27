---
title: hv1 straggler harvest — independent-approver review + merge-prep of the 7 unmerged non-pricing branches
date_utc: 2026-07-27
reviewer: hv1-independent-approver
charter: .omx/tmp/codex_prompts/ddm_hv1_straggler_harvest.md
integration_branch: hv1/straggler_integration_20260727
integration_base: main@29632337f3 (main has since advanced to dcd1bba9cd; base is its ancestor)
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
verdict: ALL_SEVEN_MERGED_ON_INTEGRATION_BRANCH_REGISTRY_ROWS_CARRIED_IN_PATCH
---

# hv1 straggler harvest — findings

Fresh independent adversarial review (`hv1-independent-approver`, replacing the quota-killed
codex hv1 arm 1:1) of the 7 unmerged non-pricing straggler branches, followed by merge-prep
onto ONE integration branch. Landing to main is DEFERRED to MAIN at a quiet boundary, per
charter. Pointer `0.1910828242 [contest-CPU]` UNMOVED by any of this — every branch is
research-only custody/measurement work; none claims a score.

## Per-branch verdict table

| # | branch | commits | verdict | review basis (adversarial) | conflict resolution on integration branch |
|---|---|---|---|---|---|
| b5 | closed_scorer_variational_de_20260721T172114Z | 1 | **MERGE-WORTHY** | Math re-derived and verified: Laguerre power-cell identity 2q·s+ω−|s|² = a·q+b (s=a/2, ω=b+|s|²) is exactly the affine argmax; Bregman debt logsumexp(z)−z_c correct; pose √(10·d) chain-rule scale √10/(2√d) correct. Receipt has real per-tile rows (20 tiles / 20,480 px, 0.0 disagreement — expected for an exact head identity), axis `[macOS-CPU advisory ... NON-PROMOTABLE]`. Pending anchors honestly residual=1.0 / NOT_MEASURED. Tests verify behavior (randomized equivalence), not constants. No NO-FAKE class present. | lane_registry → main (rows in patch); canonical_equations_registry.jsonl → union (+6 rows, verified absent from main, JSONL re-validated) |
| b3 | einstein_kolmogorov_ultra_20260721T150001Z | 3 | **MERGE-WORTHY** | "Compiler" = custody compiler, explicitly does NOT invoke an evaluator and cannot mint a score (honest naming). Emits NO_FEASIBLE_CANDIDATE at all 4 tolerances as an explicit custody/search-scope negative, not an impossibility claim. C1 409.5MB reference correctly kept historical/non-admitted. Tests literally guard NO-FAKE classes (`test_marker_only_fake_measurement_is_never_admitted`, `test_fake_u3_markers_cannot_create_ready_tuple`, external-attestation-never-authority). | lane_registry → main (rows in patch); all code files new, no overlap |
| b7 | direct_description_minimizer_builder_20260721T221054Z | 2 | **MERGE-WORTHY (custody only)** | .omx-only records. Memo states plainly "not a minimizer and not launch readiness"; launch-readiness FALSIFIED by three reviews; task 603 rows pending→in_progress→blocked. Does NOT touch main's `src/tac/optimization/direct_description_minimizer.py` (charter caution checked — no overlap). | add/add on owner_bundle_603 JSON resolved to MAIN's LATER state; branch delta recorded: `launch_readiness` BLOCKED_RECEIVER_AND_RUNNER_NOT_IMPLEMENTED → (main) BLOCKED_PRIMARY_EXECUTION_FALSE_AND_AUTHORITY_GATES_RED; `dsl_compile_hash` b52deb67→46f41597. canonical_task_status.jsonl → union. lane_registry → main (rows in patch) |
| b6 | bev_staticity_developability_probe_20260721T165801Z | 1 | **MERGE-WORTHY (superseded by b1 for code; kept for custody)** | Honest NO_VERDICT_C1_HOOD_CONTROL_FAILED at n600: positive control failed → correctly refused to read Road/Lane residuals as evidence, D3 emitted nothing. v1 tool bytes are IDENTICAL to b1's v1-import commit (verified by content diff). | lane_registry → main (rows in patch); probe_outcomes.jsonl → union |
| b1 | bev_staticity_v2_absolute_trajectory_20260721T174021Z | 5 | **MERGE-WORTHY** | v2 repaired the v1 control (bottom-connected hood, singleton f0 sidecar, direct PoseNet cross-transitions replacing the v1 nearest-target proxy). D0 PASS n64+n600 (0/600 label mismatches, closure ≤3.6e-15 m); D1/D2 MEASURED NEGATIVE (Road p50 39.02 px, Lane 47.12 px — not static in the absolute chart); D3 correctly blocked, no coefficients/bytes/score emitted. All rows axis-tagged `[macOS-CPU advisory]`. Cross-dep verified: its import `load_g1_worldsheet_motion` exists on main. | add/add on tool+test resolved to b1's v2 (strict successor of v1); probe_outcomes union DEDUPED vs b6 (3 branch rows → 1 new); lane_registry → main (rows in patch) |
| b2 | g2g2_joint_multichart_solve_20260721T163037Z | 3 | **MERGE-WORTHY** | MEASURED_G2G2_RATE_BREAK_EVEN_STOP_FAMILY_OPEN: 0/6 pairs admitted; every prefix priced in ACTUAL counted bytes (20 B k=1, 8 B incremental); hard-oracle (real receiver + factor-2 uint8 + frozen scorer) held admission authority, model never did. Search honestly named `projected_greedy_with_swap_search` (NO-FAKE #6 clean). Verdict scoped to the measured search path; family left open. Parallel-divergence fear vs main's Task-578 edits did NOT materialize: git merge-tree showed the only conflict is lane_registry; code auto-merged; ruff-F + ast clean post-merge. | lane_registry → main (rows in patch); code auto-merged |
| b4 | ddm_ms1_min_description_lattice_solve_20260723T233549Z | 2 | **ALREADY-HARVESTED — merged as ancestry record** | Content already on main via cherry-picks a35cd0de28 + 72e49cd924 (patch-id match on the 2nd; ms1 SourceSpec present at main ddm_costate_organ.py:128; measure tool, lattice sense module, receipts byte-identical on main). Main's `ddm_min_description_contract.py` + test are LATER E4/E5 evolutions — verified strict superset of every branch top-level symbol. Its lane row is the only one already registered on main. | contract+test add/add resolved to MAIN (later evolution); lane_registry → main; net merge delta = 3 audit-log rows |

**No branch required a fix-in-branch; none is not-merge-worthy.** The only content
superseded rather than taken was (a) b7's owner-bundle readiness state and (b) b4's
contract/test versions — in both cases main already held the later state, resolved per the
charter's resolve-to-main-and-report-the-delta rule.

## Integration branch contents (7 merges + 1 cherry-pick, base main@29632337f3)

1. `1f0be50826` merge b5 · 2. `00c45e4348` merge b3 · 3. `6c87abc1e0` merge b7 ·
4. `14ba1d5200` merge b6 · 5. `f54316648d` merge b1 · 6. `8181d87632` merge b2 ·
7. `0f98002ea1` merge b4 · 8. `5e52f0424c` cherry-pick e71e99dab1 (mr2 3-file review package).

Verification on the integration tip: `ast.parse` + `ruff check --select F` clean on all
merged/auto-merged .py files; canonical_equations_registry.jsonl and probe_outcomes.jsonl
re-validated as JSONL after unions; probe rows deduped (no duplicate v1 rows from b6/b1).
Review gate: every merged .py was content-reviewed and marked via
`tools/review_tracker.py mark-file <f> --status reviewed`; NO REVIEW_GATE_OVERRIDE used.

## Harvests

- **CAE crosswalk memo (dc8077a8fc, worktree-agent-a44abbaddc2e68d0e):** ALREADY ON MAIN —
  landed byte-identical as `0df88ce48c` (`.omx/research/mpeg4_shape_coding_intake_and_crosswalk_20260719.md`).
  Cherry-pick was a verified no-op; nothing owed. That worktree's unique content is fully harvested.
- **mr2 review package (e71e99dab1):** cherry-picked onto the integration branch — 3 files
  (`codex_findings_ddm_mr2_pricing_wave_merge_20260726_codex.md`, DAG_FEED, conflict receipt).
  mr2's verdict `CONTENT_CLEAN_SERIAL_MERGE_BLOCKED_AT_PF3B_NONTRIVIAL_CONFLICT` stands:
  PF3B/WF7/CB1 remain unmerged (their worktrees are in the NOT-prune-eligible list below).

## Registry patch

`.omx/research/hv1_straggler_registry_rows_patch_20260727.md` — 6 lane rows owed to main
(b4's already registered). Every lane_registry.json conflict resolved to main's version;
the driver `tools/merge_lane_registry.py` refused these merges with schema-validation errors
(110 on b3's), so MAIN should re-add rows via `tools/lane_maturity.py add-lane`/`mark`,
not raw-diff application.

## Prune-eligible worktrees (NAMED LIST ONLY — nothing deleted; deletion is MAIN's call)

Survey of all 36 worktrees (`git worktree list` + tip-ancestor-of-main + porcelain-dirty),
receipt: this memo; raw survey reproduced by the one-liner in §Method.

**Prune-eligible NOW (tip merged to main + clean tree):**
- `.omx/tmp/codex_worktrees/ddm_hv1_straggler_harvest_20260727T215117Z` (the quota-killed codex hv1 arm this session replaced 1:1)
- `.claude/worktrees/agent-a897a47c0c3ed74dc`
- `.omx/tmp/claude_cli_worktrees/smoke_opus5_20260727T220611Z` (smoke)

**Merged+clean but spawned TODAY — possibly LIVE arms, HOLD until their sessions close:**
- `.omx/tmp/codex_worktrees/ddm_hb1_hope_bn_capacity_family_20260727T215134Z`
- `.omx/tmp/codex_worktrees/ddm_j11r_366_opening_proposal_decomposition_20260727T215152Z` (j11r harvested at main tip dcd1bba9cd)
- `.omx/tmp/codex_worktrees/ddm_sh1_compose_and_local_exact_20260727T215059Z` (sh1 continues in `.claude/worktrees/agent-ae029aa6d20642139`)

**Prune-eligible AFTER `hv1/straggler_integration_20260727` lands to main** (their unmerged
content is exactly what the integration branch carries):
- `.omx/tmp/codex_worktrees/bev_staticity_developability_probe_20260721T165801Z`
- `.omx/tmp/codex_worktrees/bev_staticity_v2_absolute_trajectory_20260721T174021Z`
- `.omx/tmp/codex_worktrees/closed_scorer_variational_de_20260721T172114Z`
- `.omx/tmp/codex_worktrees/ddm_ms1_min_description_lattice_solve_20260723T233549Z`
- `.omx/tmp/codex_worktrees/direct_description_minimizer_builder_20260721T221054Z`
- `.omx/tmp/codex_worktrees/einstein_kolmogorov_ultra_20260721T150001Z`
- `.omx/tmp/codex_worktrees/g2g2_joint_multichart_solve_20260721T163037Z`
- `.omx/tmp/codex_worktrees/ddm_mr2_pricing_wave_merge_20260726T142758Z` (its package cherry-picked here)
- `.claude/worktrees/agent-a44abbaddc2e68d0e` (CAE memo already byte-identical on main as 0df88ce48c)

**NOT prune-eligible:**
- 13 MERGED-but-DIRTY worktrees (uncommitted state; certify-or-block applies), incl.
  `ddm_measurement_ladder_rungs123`, `ddm_target_receipt_pose_rung0`, `ddm_v13/v15/v18b/v4/v5/v7/v8`,
  `einstein_kolmogorov_crux_20260719`, `mdl_member_solve_v3_entropy`, `pose_attach_audit_prep`,
  `.claude/worktrees/agent-a1f080a4426f446f9`, `.claude/worktrees/agent-a4ba21155753780e9`
- UNMERGED work: `ddm_cb1_perclass_carrier_byteclose`, `ddm_pf3b_52probe_joint_improving_hunt`,
  `ddm_wf7_seven_home_stream_waterfill` (the mr2-blocked pricing wave), `agent-a6bf225d05c1d2059`,
  `agent-ae029aa6d20642139` (live sh1 integration, dirty)

## Method

Per-branch: `git log/diff merge-base..branch` content review at exact commit bytes → NO-FAKE
8-class screen (markers-without-work, constants-only tests, placeholder data fields,
search-as-solver naming, surrogate-as-authority claims) → custody claims cross-checked
against receipts (schemas, SHAs, axis tags) → `git merge --no-ff` → charter conflict policy →
ast/ruff-F verification → review-gate marks → merge commit. Worktree survey one-liner:
`git worktree list --porcelain` + `git merge-base --is-ancestor <tip> main` + `git -C <wt> status --porcelain`.

## STORES CONSULTED

- `CLAUDE.md` (NO-FAKE 8 classes; review-gate; lane-registry discipline; append-only JSONL policy)
- charter `.omx/tmp/codex_prompts/ddm_hv1_straggler_harvest.md`
- `MEMORY.md` index (stale-base clobber class; disposition≠consumption; serializer/no-co-author rules)
- `.omx/state/lane_registry.json` @ main (lane-row presence checks, all 7 lanes)
- `.omx/state/canonical_equations_registry.jsonl` @ main (b5 equation-id absence check)
- main git history: cherry/patch-id vs b4 (a35cd0de28, 72e49cd924), Task-578 commits (5d595efacf, 9b49d7bc20, c3f0dba5d4), Task-603 (4e4edb550b), CAE re-land (0df88ce48c)
- harvested mr2 package (e71e99dab1) — PF3B/WF7/CB1 disposition context
- branch receipts/memos at exact commit bytes for all 7 branches
