# #237 — Reconcile worktree/agent branches → main (2026-07-03)

**Blocker before the #205 launch. Repo:** `/Users/adpena/Projects/pact`
**Main HEAD at start AND end:** `23824507c` (140 ahead of origin — NOT pushed; separate operator decision, out of scope for #237).
**Verdict: MERGE NOTHING. Main already contains all #205 / witness / MLX-relevant work.** No merge was performed because every unmerged branch is a stale snapshot that would REGRESS main, and every high-value branch is already an ancestor of main. Conservative outcome — a deferred merge beats a broken main.

## Headline for the #205 sister subagent

- **The θ* TIER-2 witness levers ARE ON MAIN** (all 4), so if the config designer flagged one as "missing," it is NOT missing — it is present in the LEVELSET trainer on main. They were **hand-ported** into main by commit `8ac690876` (2026-06-30) rather than by merging `worktree-agent-abc89d4…` directly. Confirmed in `experiments/train_levelset_witness_realized_through_R_mlx.py`:
  - **MUST-1** `--tau-anneal-shape` (cosine / geometric / cosine_hold) — line ~1012/1028
  - **MUST-2** `--code-nuclear-weight` nuclear-norm low-rank code penalty (`_nuclear_norm_smooth_mlx`) — line ~758/1889/2229
  - **MUST-3** SWA / wider-finisher EMA `--ema-decay-finisher` — line ~1603/3143/3565/4276
  - **MUST-4** `--eikonal-junction-relax` junction-aware eikonal relax — line ~3443
  - Tests: `experiments/tests/test_levelset_theta_star_tier2_levers.py` → **22 passed** on main (bit-identical-when-off).
- **The fused-R metal transpose VJP (`p2b_metal_vjp`, #212/#251/#252) IS ON MAIN** (ancestor `ef243aeef`).
- **The #205 compute-facet plan (`worktree-agent-ad39…`) IS ON MAIN** (ancestor `16cb540f0`).

## Per-branch verdict table

| Branch | tip | category | merged? | rationale |
|---|---|---|---|---|
| `p2b_metal_vjp` | `ef243aeef` | ALREADY-IN-MAIN | n/a (ancestor) | Fused-R metal VJP, bit-identical to numpy authority. `merge-base --is-ancestor` = true; 0 ahead. Nothing to do. |
| `reconcile_205_verify` | `1d9294a49` | ALREADY-IN-MAIN | n/a (ancestor) | 0 commits ahead of main; the earlier #205 reconcile branch is fully superseded by main. (`diff main..branch` shows only deletions of work main has since added.) |
| `worktree-agent-ad39a6023f16cd1b0` | `16cb540f0` | ALREADY-IN-MAIN | n/a (ancestor + LOCKED worktree) | #205 compute-facet MLX/Metal plan. Ancestor of main; locked worktree left untouched. |
| `worktree-agent-abc89d4fb64ae03a7` | `6b4c0b962` | **SKIP-STALE** | NO | θ* TIER-2 levers — **content already hand-ported into main via `8ac690876`** (all 4 levers + tests verified present). Branch is a 2026-06-27 snapshot; `diff main..abc89d4` = 446 files, **116,716 deletions** vs 2,008 insertions → a direct merge would regress ~114K lines of main's 2026-06-27→07-03 work. DO NOT merge. |
| `worktree-agent-a499f0d20eb747e36` | `d1e870d14` | **DEFER (stale)** | NO | "scorer: additive faithful-MPS opt-in". Merge-base **2026-06-02** (month-stale); `diff` = 3,495 files, **6,798,234 deletions** vs main. Not #205-relevant (#205 uses MLX-GPU train + CPU authority, not MPS). Direct merge = catastrophic regression. If the operator ever wants the single scorer change, cherry-pick the one commit's `load_differentiable_scorers` hunk in isolation — do NOT merge the branch. |
| `worktree-agent-a5fd9a3dd5d5b0a7c` | `43234b72d` | **DEFER (stale scaffold)** | NO | Merge of `aa7fd` — NVRC/FFNeRV nerv-family enhancer SCAFFOLD, **L1 advisory non-promotable**. Merge-base 2026-06-02; `diff` = 3,499 files, **6,798,234 deletions**. Not #205-relevant. Direct merge = catastrophic regression. |
| `worktree-agent-aa7fdf14d19e7ead5` | `e966311ce` | **DEFER (stale scaffold)** | NO | Same NVRC/FFNeRV scaffold, L1 advisory non-promotable, 2026-06-02, 6.8M-line regression. Not #205-relevant. |
| `worktree-agent-a76feb23917b9bd07` | `668a06595` | **DEFER (locked)** | NO | Dispatch feasibility umbrella. Worktree is **locked** — left untouched per instruction (do not touch a locked worktree). Not #205-relevant. |
| `safety/stash-recovered-20260505T052046Z-stash{0,1,2,3}` | (2026-05-05) | **SKIP-STALE** | NO | Safety archive snapshots from 2026-05-05. Not merge candidates by design. |
| `safety/snapshot-20260504…` / `safety/snapshot-pre-filter-20260505…` | (2026-05-04/05) | **SKIP-STALE** | NO | Safety archive snapshots (already ancestors / archival). Not merge candidates. |
| `codex/hinerv-execute-gate-reconcile-20260602` | `515355f35` | ALREADY-IN-MAIN | n/a (ancestor) | VertigoDataTier worktree; `merge-base --is-ancestor` = true. Nothing to do. |
| `codex/pr101-recovery-reconcile-20260602` | `58230d48c` | ALREADY-IN-MAIN | n/a (ancestor) | VertigoDataTier worktree; ancestor of main. Nothing to do. |
| `lane-inverse-steganalysis-linf-vs-l2-gate-20260601` | `ab5da9fb7` | ALREADY-IN-MAIN | n/a (ancestor) | Ancestor of main. |
| `task57-score-native-pose-carrier` | `5b34022d7` | ALREADY-IN-MAIN | n/a (ancestor) | Ancestor of main. |
| `worktree-agent-a07343bc1a92ea2f5` | `cb44ff387` | ALREADY-IN-MAIN | n/a (ancestor) | Ancestor of main. |
| `worktree-agent-a9d422cdf471406b2` | `a295aa322` | ALREADY-IN-MAIN | n/a (ancestor) | Ancestor of main. |

Authoritative unmerged set (`git branch --no-merged main`): the 4 `safety/stash-recovered-*` + `worktree-agent-{a499…, a5fd…, aa7fd…, abc89d4…}`. Everything else named above is already an ancestor of main.

## What was MERGED

**Nothing.** No `git merge` was run. Rationale: the only unmerged branches are (a) 2026-05-05 safety archives, (b) 2026-06-02 month-stale snapshots that would regress main by ~6.8M lines, and (c) `abc89d4` whose valuable content (θ* TIER-2 levers) is already on main via the `8ac690876` hand-port. Merging any of them would break/regress main — the exact failure mode #237 exists to avoid.

## What was DEFERRED and why

- `worktree-agent-a499f0d20eb747e36` (faithful-MPS scorer opt-in) — stale 2026-06-02; not #205-relevant; operator-only cherry-pick if ever wanted.
- `worktree-agent-a5fd9a3dd5d5b0a7c` + `worktree-agent-aa7fdf14d19e7ead5` (NVRC/FFNeRV nerv-family enhancer scaffold, L1 advisory non-promotable) — stale 2026-06-02; scaffold; not #205-relevant.
- `worktree-agent-a76feb23917b9bd07` (dispatch feasibility umbrella) — LOCKED worktree, left untouched.
- `safety/*` — archival snapshots, never merge candidates.

No branch or worktree was deleted, forced, or clobbered. No real conflict was resolved by clobbering (no merge attempted). No push to origin (main is 140-ahead of origin — a separate operator decision, NOT part of #237 and NOT blocking #205, which runs on local main).

## Main-is-green confirmation (post-reconcile == pre-reconcile; HEAD unchanged `23824507c`)

- `import tac` → **OK** (`src/tac/__init__.py`).
- `experiments/train_levelset_witness_realized_through_R_mlx.py` → **parses OK** (ast).
- `tools/lane_maturity.py validate` (Check 90) → **OK — 1787 lane(s) validated cleanly.**
- `pytest experiments/tests/test_levelset_theta_star_tier2_levers.py` → **22 passed** (θ* TIER-2 levers bit-identical-when-off, functional on main).

**Launch-readiness for #205: GREEN.** Main is consistent, imports, and carries all witness/θ*/MLX levers. No reconcile merge is a blocker. #237 resolves as "already reconciled by prior hand-port + merges; remaining branches are stale and correctly deferred."
