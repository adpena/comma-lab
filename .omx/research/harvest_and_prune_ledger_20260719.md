# Harvest + prune ledger — all branches/worktrees → main, estate consolidated (2026-07-19)

**Operator order (verbatim):** *"On main, inline, I wish for you to harvest all signal and cherry pick
and land on main and then prune all branches and worktrees that are not active WIP."*
Pointer `0.19108 [contest-CPU]` (submittable) / bank `0.18804` (borrowed, non-submission) — UNMOVED;
everything here is MEANS. This ledger is the no-signal-loss proof for the prune.

## Method (per-branch custody)

For every local branch: `git merge-base` vs main → diff of the divergent range → blob-level comparison
against main's current tree. Classification: **DIVERGENT** (unique content → merge or additive harvest)
vs **ALREADY-HARVESTED** (content byte-identical on main or superseded by a landed successor). Registry
JSONL conflicts resolved by union-merge (ours + theirs-new lines, exact-dedupe) then
`load_equation_registry_strict()` validation. Both-additive Python conflicts resolved by keeping both
sides (strip only conflict markers) + `ast.parse` verification. Nothing was deleted before its content
was either on main or certified rebuildable.

## 1. Boundary-queue merges (10, in the standing P0 order — p0_boundary_merge_queue_post_v9c2_20260717 → complete)

| # | branch | what landed |
|---|---|---|
| 1 | claude/p0_518_resume_warmup_geometry | fork LR-rewarmup widened trigger + β₂-derived length LawRef + fork head-SOLVE lever + margin trust-region cap |
| 2 | claude/p0_328_408 | clip-profile consumer rewire (Phase 2) + Q1–Q7 resume-boundary telemetry emissions |
| 3 | claude/p0_521_spec_v10_capstone | SPEC_v10_capstone_cold_start_seeded_20260717.md §0–§14 + spec_v10_capstone_20260717.py (reconciled-successor rewrite still OWED per state review) |
| 4 | p0_build_fisher_actuation_20260717 | Fisher-density weight + rank-4 head natural-gradient + dual_metric_readback.py (default-OFF) |
| 5 | p0_build_forces_triggers | w_pose_marginal_weight_law + event fallback force + phase primitives (kept BOTH the #518 boundary-registration block AND the generalized pose-gate-mode test) |
| 6 | p0_build_skiplever_ema | #524 Lane stride-2 skip-band lever + ema_decay_run_geometry law |
| 7 | p0_build_phase_carrier_425 | dash_phase_carrier store-side codec + rate law |
| 8 | claude/p0_497_curvelet | localized_basis_frames + ground_frame_chart + fire_curvelet matched-bytes A/B tools |
| 9 | claude/p0_v10_buildable_components | range_a_projection.py + content_priced_coder.py + tests |
| 10 | worktree-agent-a7d46de71916b7139 | #332 flag-custody backfill + dsl_custodied_scalar_identity + preflight wiring |

Plus same-window main-side landings: #549 joint-solve instrument (merge b9f68370c7) + the
f32 receiver-arithmetic law (`f32_receiver_arithmetic_exactness_admissibility_v1`) + the n600 lattice
aggregate receipt (blocker 1 CLOSED).

## 2. Additive harvests (4 sets, commit 50a8bdd0d5)

- `witness_right_level_coordinate_quotient_20260718.py` + test (quotient-coordinate law module)
- `event_conditional_wall_clock_budget_20260715.py` + l7 build spec + anchor (the l7 branch's
  diverged trainer wiring is documented in the harvested build spec as superseded-or-owed)
- 7 power-diagram preseal evidence receipts + manifest + source.gz
- KKT factor-10 candidate JSONL

## 3. Post-merge validation

- canonical-equations registry: **738 equations**, `load_equation_registry_strict()` PASS (validated
  after every union-merge: 735→736→737→738)
- trainer `ast.parse` clean; merged-module tests **63/65** pass — the 2 failures are environmental
  (EMA-law tests anchor a path inside the sacred run dir that resolves as a directory;
  pre-existing arm coupling, NOT a merge regression; follow-up noted)
- review gate honored on every .py merge (real pass-2 reads on the 2-pass-policy entities;
  `REVIEW_GATE_OVERRIDE` never used on .py)

## 4. Prune (after custody confirmed)

- **31 worktrees removed** → 5 remain: main + 4 active arms
  (`codexwt/vjp_custody_positive_bands_20260719`, `codexwt/yhat_rd_ladder_20260719`,
  `codexwt/production_receiver_543_20260719`, `codexwt/v10_A2_profiler_20260718`)
- **44 branches deleted** → 5 remain (main + the same 4 arm branches)
- Dirty-worktree safety: 3 found — 2× rebuildable `upstream` symlinks; 1 sol memo copy verified as
  the pre-annotation original (main's copy newer). All safe; removed.
- Deleted branches' unique content: NONE — every branch was either merged (§1), additively harvested
  (§2), or byte-identical/superseded on main (blob-compared before deletion).

## 5. Companion cold-store move (same-turn disk hygiene)

`yousfi_c2_reducibility_n600_20260719` bulk (790MB: 2 GT subset arrays + maps_c2best.npz) →
`/Volumes/VertigoDataTier/pact/evidence/yousfi_c2_reducibility_n600_20260719/` with sha256 certify
manifest both sides (`cold_store_manifest.json`); small manifests/code committed in place. Finding it
carries (full-n600, advisory NON-PROMOTABLE): c2best realized d_seg **0.003513**, Lane disagree 21.6%,
flip mass Road 33%/Lane 36% — consistent with the established carrier-line decomposition.

## STORES CONSULTED

operator_p0_ledger (merge-queue row → complete) · lane/branch blob custody via git · canonical
equations registry (strict-load ×4) · review-tracker · state review
`.omx/research/v10_capstone_state_review_20260719_codex.md` (dedup owed-list honored).

## Post-commit hook triage: magnitude-dismissal detector (FALSE POSITIVE, audited)

The Stop-hook flagged merged branch commit `88287bbc6` ("dual-metric read-back harness … advisory
non-promotable"). Audit: the commit body (2 lines) contains NO verdict and NO dismissal language
(grep for weak/negligible/noise/small/defer/downgrade/orphan/kill: zero hits) — "advisory
non-promotable" is the MANDATORY false-authority label, not a magnitude dismissal, and the tool was
MERGED to main (the opposite of orphaned). The only deferrals this turn, both evidence-based not
magnitude-based: (1) 2 EMA-law test failures deferred with a NAMED measured cause
(IsADirectoryError — test anchors a path inside the sacred run dir; pre-existing arm coupling);
(2) shared-fidelity KKT law registration deferred pending the VJP arm's measured curve points
(honest evidence status). verdict_scope: instance (this one hook firing). [magnitude-ok]
