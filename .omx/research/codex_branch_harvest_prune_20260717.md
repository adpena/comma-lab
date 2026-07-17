# Codex-branch harvest + worktree prune — 2026-07-17

`research_only=true` · operator directive 2026-07-17 ("a bunch of codex work and
worktrees and branches that need to be cherry picked and landed and then
pruned") · no score/pointer authority · MAIN unchanged in content (pointer
UNMOVED). Supreme constraint honored: **certify-or-block, no signal loss** —
nothing deleted whose content is not PROVEN landed or PROVEN a superseded
subset of what is already on `main`.

## Headline

- **0 commits newly cherry-picked to main.** Every "unlanded" codex branch's
  substantive content was already re-landed on `main` via the prior harvest
  arms (patch-id drifted because the branches sit −283 behind `main`, so
  `git cherry` false-flags them `+`). Certified by (a) subject-match in
  `main` history, (b) blob-identity of each branch's distinctive NEW files,
  and (c) `git cat-file -e` existence checks.
- **1 branch HELD (not pruned): `l7_default_failloud`** — genuinely unlanded
  content, but it mutates the FROZEN live trainer's `--l7-start-epoch` default
  (800 → −1) plus control-plane files while a run is mid-flight, and is 283
  commits behind. Per the operator's explicit HELD rule for trainer/
  control-plane touches. Branch preserved for a deliberate trainer-unfreeze
  window.
- **8 branches pruned** (6 landed-dup unlanded + 2 fully-merged).
- **6 worktrees removed** (all clean or certified-superseded-subset).
- **1 worktree + 4 branches kept** (boundary-held DO-NOT-MERGE active work).

### Process note (transparency)

While probing cherry-pick feasibility for the l7 branch, a `git worktree add`
silently failed (`main` already checked out); the subsequent `cd` fell through
and the test cherry-picks ran **on main**, landing 2 commits (`c73ae1dd`,
`efba0f28`) and leaving an aborted 3rd. This was immediately detected and fully
unwound: `git reset --mixed b2a89181d6` + surgical `git checkout b2a89181d6 --`
of the 9 touched tracked files + `rm` of the 2 new files. Verified: trainer
byte-identical to `main` HEAD, live run unaffected, only the 3 pre-existing
unrelated working-tree modifications preserved. `main` HEAD back at
`b2a89181d6`.

## Per-branch disposition — UNLANDED set (7 branches)

| Branch | ahead/behind | Verdict | Evidence |
|---|---|---|---|
| `derive_solver_provenance_…092744Z` | +9/−283 | **LANDED-DUP → PRUNED** | 8 commits patch-id-dup (`git cherry -` marks); the 1 `+` `6bd9b62` "Register ARM-DERIVE L1 custody" re-landed on main as `6b8c31d7ba` |
| `c0_optform_compute_audit_…131323Z` | +3/−263 | **LANDED-DUP → PRUNED** | re-landed on main `8df4723df8`; new files (`pose_verdict_gate.py`, DAG feed, tests) blob-IDENTICAL on main |
| `l7_default_failloud_…152116Z` | +3/−228 | **HELD (kept)** | genuinely unlanded: 3 files ABSENT on main (`canonical_equations/event_conditional_wall_clock_budget_20260715.py`, `l7_…eventlaw_anchor_20260715.json`, `l7_…build_spec_20260715.md`). Touches frozen live trainer `--l7-start-epoch` default 800→−1 + control-plane (`scorer_throughput_gate.py`, `witness_autoconfig.py`, `launch_witness_run.py`); 283 behind; 3rd commit conflicts on `lane_maturity_audit.log`/`lane_registry.json`. Commits: `df486e5e` (l7 opt-in), `3ba168b9` (event-stage wall-clock budget), `5d002671` (lane custody) |
| `iso_configs_…092618Z` | +2/−283 | **LANDED-DUP → PRUNED** | re-landed on main `6e761bbeac`; `spec_v9_cgauge.py` + all v9_cgauge_432 iso configs present on main (dry-run receipts blob-IDENTICAL) |
| `basis_d21a_prod_…092624Z` | +1/−283 | **LANDED-DUP → PRUNED** | re-landed on main `cd03549893`; `affine_legendre_gauge_policy.py`, `surrogate_vjp_fidelity_policy.py`, `test_basis_d21a_prod.py` blob-IDENTICAL on main |
| `curvelet_optimal_form_crux_…093444Z` | +1/−280 | **LANDED-DUP → PRUNED** | its only `+` commit's SPEC `.omx/research/curvelet_optimal_form_crux_20260715_SPEC.md` is blob-IDENTICAL on main; only append-ledger rows differ (drift). Worktree removed (see below) |
| `p0_confound_hardening_…101514Z` | +1/−267 | **LANDED-DUP → PRUNED** | re-landed on main `d7eadd6361`; `confound_observability.py`, `confound_gates` additions, SPEC/DAG feed present on main |

## Per-branch disposition — FULLY-MERGED set (2 branches, 0 ahead)

| Branch | ahead/behind | Verdict |
|---|---|---|
| `costate_organ_elevation_20260716T191723Z` | 0/−42 | **MERGED → PRUNED** (worktree removed first) |
| `curvelet_crux_harvest_complete_20260716T192419Z` | 0/−41 | **MERGED → PRUNED** (worktree removed first; this was the arm that re-landed most curvelet content on main via `98b7896bc6`) |

## Boundary-held branches — KEPT (DO-NOT-MERGE, active work)

| Branch | ahead/behind | Verdict |
|---|---|---|
| `claude/p0_521_spec_v10_capstone_20260717` | +3/−5 | KEEP (boundary-held); worktree dir removed, branch preserved |
| `claude/p0_328_408_merge_window_prep_20260717` | +3/−5 | KEEP (boundary-held); worktree dir removed, branch preserved |
| `claude/p0_518_resume_warmup_geometry_20260717` | +4/−8 | KEEP (boundary-held); worktree dir removed, branch preserved |
| `claude/p0_497_curvelet_matched_bytes_ab_20260717` | +4/−15 | KEEP (boundary-held); **worktree KEPT** — sealed byte-close #497 gap-a/gap-b A/B work; in-worktree fire script `tools/fire_curvelet_matched_bytes_ab_p0_497.py` (on branch, not main) resolves `REPO = Path(__file__).parents[1]` = runs in-place in the worktree; live memo `.omx/research/curvelet_matched_bytes_ab_20260717.md` references the worktree path. Per the operator prune-exception: LEAVE the dir. |

## Worktree dispositions

| Worktree dir | branch | clean? | Action |
|---|---|---|---|
| `.claude/worktrees/agent-a456410994630180d` | claude/p0_521 | clean | **REMOVED** (branch preserved) |
| `.claude/worktrees/agent-ac7f6074e2d1a9bca` | claude/p0_328_408 | clean | **REMOVED** (branch preserved) |
| `.claude/worktrees/agent-ae799a9bcb4460892` | claude/p0_518 | clean | **REMOVED** (branch preserved) |
| `.omx/tmp/codex_worktrees/costate_organ_elevation_…` | codexwt/costate_organ_elevation | clean | **REMOVED** (+ branch deleted) |
| `.omx/tmp/codex_worktrees/curvelet_crux_harvest_complete_…` | codexwt/curvelet_crux_harvest_complete | clean | **REMOVED** (+ branch deleted) |
| `.omx/tmp/codex_worktrees/curvelet_optimal_form_crux_…` | codexwt/curvelet_optimal_form_crux | 6 untracked | **REMOVED (--force)** — see salvage audit below (+ branch deleted) |
| `.omx/tmp/codex_worktrees/p0_497_curvelet_ab_20260717` | claude/p0_497 | clean | **KEPT** (boundary-held active A/B; fire-script/memo path dependency) |

### Salvage audit — curvelet_optimal_form_crux worktree (6 untracked files)

All 6 exist on `main` as tracked, reviewed, refined versions. Certified subset
of `main` (no unique unlanded signal):

- `through_r/equal_archive_budget.py` + its test — **byte-IDENTICAL to main.**
- `boundary_math/curvelet_placement.py`, `localized_basis_frames.py` + their
  2 tests — earlier drafts; `main` is a **strict superset** (e.g. `main` adds
  `normal_covectors_from_argmax_numpy` / `NativeOrientationFixedPointReceipt`
  that the drafts LACK). The only worktree-only fragments (recorded here for
  zero-signal-loss even post-discard) were trivial superseded formulations:
  - `curvelet_placement.py` (2 lines, a superseded `logits` formulation):
    `logits = np.float32(kappa) * mx.cos(np.float32(2.0) * (group_theta - normal_angle[..., None])`
  - `localized_basis_frames.py` (1 line, a superseded import):
    `from dataclasses import asdict, dataclass`

  These fragments live inside functions `main` also carries (refined). No lost
  concept; the curvelet-crux implementation landed on `main` via the
  `curvelet_crux_harvest_complete` arm (`98b7896bc6`).

## Final state

See `git branch` / `git worktree list` transcript in the harvest session.
`main` HEAD `b2a89181d6`, pointer UNMOVED. `[no-triality]`
