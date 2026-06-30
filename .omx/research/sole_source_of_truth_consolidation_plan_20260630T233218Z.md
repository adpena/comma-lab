# Sole-source-of-truth consolidation plan — branches / stashes / worktrees

**Date:** 2026-06-30T233218Z · **Repo:** `/Users/adpena/Projects/pact` · `main` @ `0a688e8e0`
**Mode:** STRICTLY READ-ONLY AUDIT. No merge/rebase/branch-d/stash-drop/commit(except this memo)/push/worktree-remove
performed. This is a PLAN; operator approves before any action.
**Operator directive:** "audit + report a plan" so `main` becomes the SOLE source of truth.

## TL;DR
- **17 local branches** besides `main`, **19 stashes** (`stash@{0..18}`), **11 non-main worktrees**.
- **Only ONE branch carries unique frontier-canonical work:** `worktree-agent-abc89d4fb64ae03a7`
  (θ* TIER-2 levers, task #184). Everything else is already-on-main, a stale/deprioritized
  scaffold, or an old safety snapshot.
- **Conflict reality:** the θ* branch's ONLY conflicting file is the LIVE n600 trainer
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (228 commits of main drift since
  branch point). Its levers are additive + default-off + bit-identical-when-off (26 tests), so once
  ported the live run behavior is unchanged unless a new flag is passed — but the port itself MUST be
  done at a safe checkpoint boundary, not during an active n600 resume.
- **9 branches are already-on-main** (0 unique commits) → safe prune.
- **3 NeRV-family branches are unique-but-stale/superseded** (NeRV deprioritized post witness-pivot;
  faithful-MPS superseded by main's canonical `patch_scorer_for_mps`) → prune (optional cherry-pick if
  NeRV reactivated).
- **4 `safety/stash-recovered-*` branches** = 2-month-old WIP snapshots → KEEP-AS-SAFETY (operator
  discretion) or prune after confirm.
- All 11 worktrees except `agent-abc89d4f` (the keep) sit on already-on-main / detached-already-on-main
  commits and are clean, EXCEPT `agent-a76feb23` (LOCKED, 5 uncommitted files) and
  `.omx/tmp/wt_fcommit` (1 untracked `upstream/`).

---

## 1. Branch classification table

| Branch | Class | unique (cherry +) | ahead/behind main | Action | Risk |
|---|---|---|---|---|---|
| `worktree-agent-abc89d4fb64ae03a7` | **UNIQUE_CANONICAL** | 1 | 1 / 228 | **CHERRY-PICK** then prune | MED — conflicts on LIVE n600 trainer file |
| `worktree-agent-a5fd9a3dd5d5b0a7c` | UNIQUE_STALE_SUPERSEDED | 2 | 3 / 2291 | PRUNE (opt cherry-pick if NeRV reactivated) | LOW — touches `src/tac/scorer.py` (small) |
| `worktree-agent-a499f0d20eb747e36` | STALE (subset of a5fd9a3dd) | 1 | 1 / 2291 | PRUNE (redundant) | none |
| `worktree-agent-aa7fdf14d19e7ead5` | STALE (subset of a5fd9a3dd) | 1 | 1 / 2291 | PRUNE (redundant) | none |
| `codex/hinerv-execute-gate-reconcile-20260602` | ALREADY_ON_MAIN | 0 | 0 / 2210 | PRUNE | none |
| `codex/pr101-recovery-reconcile-20260602` | ALREADY_ON_MAIN | 0 | 0 / 2291 | PRUNE | none |
| `lane-inverse-steganalysis-linf-vs-l2-gate-20260601` | ALREADY_ON_MAIN | 0 | 0 / 2404 | PRUNE | none |
| `task57-score-native-pose-carrier` | ALREADY_ON_MAIN | 0 | 0 / 1112 | PRUNE | none |
| `worktree-agent-a07343bc1a92ea2f5` | ALREADY_ON_MAIN (no worktree) | 0 | 0 / 2398 | PRUNE (direct) | none |
| `worktree-agent-a9d422cdf471406b2` | ALREADY_ON_MAIN (no worktree) | 0 | 0 / 2404 | PRUNE (direct) | none |
| `worktree-agent-a76feb23917b9bd07` | ALREADY_ON_MAIN | 0 | 0 / 5428 | PRUNE — but resolve LOCKED+DIRTY worktree first | LOW — 5 uncommitted files in worktree |
| `safety/snapshot-20260504T223259Z-pre-rigor-pass` | ALREADY_ON_MAIN | 0 | 0 / 7745 | PRUNE | none |
| `safety/snapshot-pre-filter-repo-20260505T144000Z` | ALREADY_ON_MAIN | 0 | 0 / 7659 | PRUNE | none |
| `safety/stash-recovered-...stash0` | SAFETY_SNAPSHOT | 1 (untracked layer) | 3 / 7745 | KEEP-AS-SAFETY / prune after confirm | none |
| `safety/stash-recovered-...stash1` | SAFETY_SNAPSHOT (preflight 82/83 landed) | 1 | 3 / 7894 | KEEP-AS-SAFETY / prune after confirm | none |
| `safety/stash-recovered-...stash2` | SAFETY_SNAPSHOT (council fixes landed) | 1 | 2 / 8184 | KEEP-AS-SAFETY / prune after confirm | none |
| `safety/stash-recovered-...stash3` | SAFETY_SNAPSHOT (DEN-V2 partial) | 1 | 2 / 8187 | KEEP-AS-SAFETY / prune after confirm | none |

**Counts:** UNIQUE_CANONICAL = 1 · UNIQUE_STALE_SUPERSEDED = 3 · ALREADY_ON_MAIN = 9 · SAFETY_SNAPSHOT = 4.
**Prune candidates (branches):** 9 already-on-main + 3 stale-NeRV = **12 immediate**; +4 safety-snapshot after
confirm = **16**; +1 (θ* abc89d4f) after its cherry-pick lands = **17 total** (→ main sole SoT).

### Notes on the unique commits
- **abc89d4f** — `6b4c0b962` "levelset witness theta* TIER-2: 4 additive default-off levers
  (tau-anneal-shape{cosine/geometric/cosine_hold}, nuclear-norm low-rank code penalty, SWA wider-finisher
  EMA, junction-aware eikonal relax) + 26 bit-identical-when-off tests." Touches exactly 2 files:
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (+268, the conflict file) and a NEW test
  `src/tac/tests/test_levelset_theta_star_tier2_levers.py` (+319, clean add). Verified additive:
  all new argparse flags default to neutral (`--code-nuclear-weight 0.0`, `--eikonal-junction-relax 0.0`,
  `--ema-decay-finisher None`, `--tau-anneal-shape cosine`). This is on the CURRENT frontier (the θ*
  lever campaign in MEMORY). **THE keep.**
- **a5fd9a3dd** — 2 commits: `d1e870d14` "scorer: additive faithful-MPS opt-in in
  load_differentiable_scorers" + `e966311ce` "scaffold NVRC entropy-pipeline + FFNeRV-flow pose-channel
  enhancer ... L1; 23 CPU tests; advisory non-promotable." Adds new modules
  `src/tac/analysis/{nvrc_entropy_pipeline,ffnerv_flow_pose_channel}.py` (+ tests) and a registry lane
  `lane_nerv_family_enhancers_scaffold_20260602`. SUPERSESSION: (a) main's canonical MPS path is
  `tac.torch_mps_compat.patch_scorer_for_mps()` wired into `score_aware_loop/targets.py` +
  `torch_vehicle/driver.py` (post-2026-06-12), which supersedes this branch's earlier `scorer.py`
  faithful-MPS approach; (b) NeRV-family enhancers are deprioritized by the witness-capstone pivot.
  The lane is L1 advisory-non-promotable. Main's `ffnerv_as_renderer` registry entries are a DIFFERENT,
  already-landed FFNeRV lane (2026-05-11) — not these modules.
- **a499f0d2 / aa7fdf14** — each holds a strict SUBSET of a5fd9a3dd's two commits (faithful-MPS only,
  NVRC/FFNeRV only respectively). Redundant; prune regardless.
- **safety/stash-recovered-*** — the unique "+" cherry is the `untracked files on main: <hash> ...`
  layer of recovered May-4/5 stashes (the `index on main:` layer is already-equivalent in main). These
  are explicit safety snapshots, not active work.

---

## 2. Stash classification (`stash@{0..18}`)

All stashes are signal-preservation snapshots; none is a ref that competes with `main` as SoT (so they
do not block the goal). Most cluster around the 2026-06-02 "converge to main" reconciliation event;
several are pre-2026-05-06. **Recommendation: KEEP-AS-SAFETY until the operator confirms the
witness-capstone pivot rendered them obsolete, then drop.** Flag the oversized ones for SSD-spill
hygiene — they almost certainly captured generated artifacts/reports that should never live in a stash.

| Stash | Subject (truncated) | Scope | Recommendation |
|---|---|---|---|
| `@{0}` | preserve canonical upstream fallback runner WIP 20260602 | 1 file (+35) `run_compact_renderer_mlx_spine_runner.py` | KEEP-AS-SAFETY (overlaps current working-tree M) |
| `@{1}` | preserve selector hi_nerv launch gate WIP | 16 files **+40585** (artifact-laden) | KEEP-AS-SAFETY; FLAG: huge, likely generated bytes |
| `@{2}` | preserve snerv WIP + pre-fix hinerv report | 4 files (+68) snerv ladder | KEEP-AS-SAFETY (overlaps working tree) |
| `@{3}` | preserve snerv trained ladder WIP | 2 files (+18) | KEEP-AS-SAFETY |
| `@{4}` | preserve pre-ff WIP before hinerv saliency replay | 6 files (+559) | KEEP-AS-SAFETY |
| `@{5}` | autostash | 9 files (+378) nerv/spine-runner tests | KEEP-AS-SAFETY (overlaps working tree) |
| `@{6}` | pre-convergence-state-files RECOVERABLE (on a5fd9a3dd) | 2 files (+44) registry/audit | PRUNE-after-confirm (state-only) |
| `@{7}` | pre-main-checkout stopped-agent WIP RECOVERABLE | empty diff | PRUNE-after-confirm (no content) |
| `@{8}` | lane-wip-pre-resync RECOVERABLE (on lane-inverse-steg) | 261 files **+98930** | KEEP-AS-SAFETY; FLAG: huge, artifact-laden |
| `@{9}` | WIP z7-mamba2 static control + custody | 12 files (+1243) | KEEP-AS-SAFETY (old; z7) |
| `@{10}` | WIP orphan-signal audit + 8 op-routables | 72 files (+2247) | KEEP-AS-SAFETY (old) |
| `@{11}` | WIP z3 `_full_main` Phase 2 | **3315 files** (+8298) | KEEP-AS-SAFETY; FLAG: huge, artifact-laden |
| `@{12}` | WIP composition cell registry | 42 files (+2159) | KEEP-AS-SAFETY (old) |
| `@{13}` | WIP dispatch consolidation | 30 files (+1515) | KEEP-AS-SAFETY (old) |
| `@{14}` | pre-integration signal preservation 20260506 | 19 files (+2046) | PRUNE-after-confirm (pre-pivot) |
| `@{15}` | pre-rigor-pass safety stash 20260504 | 121 files (+35373/-26543) | PRUNE-after-confirm (pre-pivot, huge) |
| `@{16}` | WIP preflight checks 82/83 (already landed) | 6 files (+335) | PRUNE-after-confirm (landed) |
| `@{17}` | yousfi_3_5_pending_greenup | 19 files (+3021) | PRUNE-after-confirm (pre-pivot) |
| `@{18}` | DEN-V2 partial arch-drift | 3 files (+371/-314) | PRUNE-after-confirm (pre-pivot) |

Cross-ref: stashes `@{0},{2},{3},{4},{5}` overlap the files currently showing `M` in the session-start
`git status` (`run_compact_renderer_mlx_spine_runner.py`, `test_compact_renderer_mlx_spine_runner.py`,
`support_codec_router.py`, etc.) — i.e. that WIP partially re-surfaced in the working tree. Confirm the
working tree carries what's needed before dropping those.

---

## 3. Worktree classification (11 non-main)

| Worktree path | Branch / HEAD | Dirty | On-main? | Action | Risk |
|---|---|---|---|---|---|
| `.claude/worktrees/agent-abc89d4fb64ae03a7` | `worktree-agent-abc89d4f` | 0 | unique | **KEEP until θ* cherry-pick lands**, then `worktree remove` + prune branch | the keep |
| `.claude/worktrees/agent-a5fd9a3dd5d5b0a7c` | `worktree-agent-a5fd9a3dd` | 0 | unique-stale | `worktree remove` then prune branch | none |
| `.claude/worktrees/agent-a499f0d20eb747e36` | `worktree-agent-a499f0d2` | 0 | stale-subset | `worktree remove` then prune | none |
| `.claude/worktrees/agent-aa7fdf14d19e7ead5` | `worktree-agent-aa7fdf14` | 0 | stale-subset | `worktree remove` then prune | none |
| `.claude/worktrees/agent-a76feb23917b9bd07` | `worktree-agent-a76feb23` | **5** | yes (0 unique) | INVESTIGATE 5 dirty files → unlock → `worktree remove` → prune | LOW |
| `.omx/tmp/wt_fcommit` | detached `2ce14c114` | 1 (`?? upstream/`) | yes (0 unique) | `worktree remove` (ephemeral tmp; `upstream/` is the pinned-snapshot dir) | none |
| `/Volumes/VertigoDataTier/pact/codex_hinerv_execute_gate_...` | `codex/hinerv-...` | 0 | yes | `worktree remove` then prune branch | none |
| `/Volumes/VertigoDataTier/.../pact-main-guardrails-20260602` | detached `58230d48c` | 0 | yes | `worktree remove` | none |
| `/Volumes/VertigoDataTier/.../pact-main-nerv-seam-20260602T0340Z` | detached `58230d48c` | 0 | yes | `worktree remove` | none |
| `/Volumes/VertigoDataTier/.../pact-main-pr101-recovery-poll-...` | `codex/pr101-...` | 0 | yes | `worktree remove` then prune branch | none |
| `/Volumes/VertigoDataTier/.../pact-main-snerv-stratified-smoke-...` | detached `b9de3c93d` | 0 | yes (0 unique) | `worktree remove` | none |

`agent-a76feb23` dirty files: `M CLAUDE.md`, `M src/tac/preflight.py`,
`M tools/cathedral_autopilot_autonomous_loop.py`, `M tools/operator_authorize.py`,
`?? src/tac/preflight_rudin_daubechies/` — old (2026-05-15) uncommitted edits; verify none is unique
signal vs current main before removal (almost certainly superseded, but confirm to honor no-signal-loss).

---

## 4. Merge order for the UNIQUE_CANONICAL set

There is effectively **one** canonical merge (plus one optional, low-priority NeRV reconcile):

### (1) `worktree-agent-abc89d4fb64ae03a7` — θ* TIER-2 levers (task #184) — HIGHEST CARE
- **Why first/only:** the sole branch with unique, on-frontier, keep-grade work.
- **Conflict surface:** `experiments/train_levelset_witness_realized_through_R_mlx.py` — THE LIVE n600
  trainer. 228 commits of main drift (main has since landed v2 residual-only mode `a2460b14e`,
  EMA/checkpoint hardening `3da9a6b10`, review batch-fix `6c9adc243` — all on this file). A clean
  `git merge` WILL conflict here. The test file `test_levelset_theta_star_tier2_levers.py` is a clean add.
- **Does it disturb the live n600 resume path?** YES, it edits that file — so:
  - Do the port **at a safe checkpoint boundary**, NOT during an active resume/launch.
  - Recommended method: **cherry-pick-by-hand / 3-way port**, not a blind merge — re-apply the 4
    additive default-off levers onto current `main`'s trainer (preserve `main`'s v2-residual + EMA-best +
    review-fix logic), then run the 26 bit-identical-when-off tests + the existing 102-test witness suite
    to prove default behavior is unchanged. The levers' neutral defaults mean a resumed run is byte-faithful
    until a flag is passed (preserves the per-stage-checkpoint + deterministic-reproducibility
    non-negotiables).
  - Land via `tools/subagent_commit_serializer.py` with the review gate (touches `.py`).
- **After it lands:** `git worktree remove .claude/worktrees/agent-abc89d4fb64ae03a7` → `git branch -d
  worktree-agent-abc89d4fb64ae03a7`.

### (2) OPTIONAL — `worktree-agent-a5fd9a3dd5d5b0a7c` — NeRV-family enhancers scaffold + faithful-MPS
- **Only if** NeRV-family is reactivated. Otherwise PRUNE.
- **Conflict surface:** `src/tac/scorer.py` (+49, possible small conflict) + clean new analysis modules +
  `.omx/state/lane_registry.json` (+43) / `lane_maturity_audit.log` (+3 — append-only, will need a clean
  re-apply per artifact-lifecycle discipline). The faithful-MPS portion is likely **superseded** by
  `patch_scorer_for_mps`; cherry-pick at most the NVRC/FFNeRV scaffold modules + their registry lane.
- **Does NOT touch the live trainer → zero n600 disturbance.** Safe to do anytime, independent of (1).
- Cherry-picking (2) makes `a499f0d2` and `aa7fdf14` fully redundant (already their superset).

No ordering dependency between (1) and (2); they touch disjoint files. Do (1) first (frontier value);
(2) is discretionary.

---

## 5. What becomes the sole source of truth after this plan executes

After the operator approves and the actions run (in this order):

1. **Cherry-pick/port θ* TIER-2 levers (abc89d4f) onto `main`** at a safe n600 checkpoint boundary
   (additive, default-off, tests green) — the only unique frontier work absorbed.
2. *(optional)* Cherry-pick the NVRC/FFNeRV scaffold from `a5fd9a3dd` if NeRV reactivated; else skip.
3. **Remove all 11 non-main worktrees** (after confirming `a76feb23`'s 5 dirty files + `wt_fcommit`'s
   untracked `upstream/` carry no unique signal). Frees the `.claude/worktrees/*` and
   `/Volumes/VertigoDataTier/pact/*` copies (SSD-tier hygiene win).
4. **Prune all 17 non-main branches** — 9 already-on-main + 3 stale-NeRV + abc89d4f (post-cherry-pick) +
   the 4 `safety/stash-recovered-*` (operator discretion: keep-as-safety or prune).
5. **Drop stashes** the operator confirms obsolete; keep the rest as labeled safety until the
   witness-capstone pivot is confirmed to have superseded them. (Stashes don't block SoT either way.)

**End state:** `main` @ (current `0a688e8e0` + θ* levers) is the SOLE branch and SOLE source of truth.
Zero competing branches, zero stray worktrees, no unmerged unique frontier work, no signal lost — every
pruned item was either already-in-main, a strict subset, a superseded/deprioritized scaffold, or an
explicitly-preserved safety snapshot the operator chose to retain.

**Guardrails honored:** read-only audit; the live n600 run + its trainer file are untouched; the one
risky merge is flagged as checkpoint-boundary-only + hand-ported (not blind-merged) so deterministic
reproducibility and the per-stage-checkpoint resume path are preserved; no destructive action taken in
this audit.
