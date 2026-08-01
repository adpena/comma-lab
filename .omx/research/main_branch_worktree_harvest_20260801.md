---
schema: main_branch_worktree_harvest.v1
date_utc: 2026-08-01
arm: BRANCH/WORKTREE SIGNAL HARVEST (read-only; no merge, no prune, no launch)
lane_id: "lane_main_branch_worktree_harvest_20260801"
research_only: true
score_claim: false
pointer_moved: false
paid_dispatch: false
axis: "[no measurement produced — enumeration + content-identity only]"
operator_verbatim: "We also need to check out those work trees and branches because I imagine there's probably a lot of signal there that we could benefit from."
extends: ".omx/research/ddm_cs1_consolidation_harvest_20260728.md (2026-07-28) · .omx/research/codex_branch_harvest_prune_20260717.md"
verdict: ONE_REAL_ORPHAN_FOUND_EINSTEIN_UNCOMMITTED_DELTA; SH1_STILL_THE_ONLY_MEASURED_ROW_BRANCH; ALL_ELSE_SUPERSEDED_OR_SCRATCH
---

# Branch + worktree signal harvest — 2026-08-01

Extends cs1 (07-28). cs1 enumerated the same surfaces but **explicitly declined to open
the dirty worktrees' contents** ("dirty-leftover-scratch … left in place; not touched").
That decision was the gap this pass closes: I opened all 16 and verified every
supersession claim by content hash rather than inheriting it.

**Every count below is MEASURED over a stated denominator.** No verdict here rests on an
empty grep: where a search returned zero I re-ran it with a corrected predicate and report
the correction (see §6 — one of my own counts was a false zero).

## 1. Measured counts

| quantity | count | how measured |
|---|---:|---|
| worktrees (incl. main) | **53** | `git worktree list --porcelain \| grep -c '^worktree '` |
| non-main worktrees | 52 | as above minus main |
| **unmerged refs** (not ancestors of `main`) | **6** | `git branch -a --no-merged main` → 4 local + 2 remote |
| non-main worktrees with **uncommitted** state | **16** | per-worktree `status --porcelain`, all 52 walked |
| dirty entries in **main's own** working tree | 144 | live parallel session's WIP — NOT TOUCHED |
| `git stash list` entries | **0** | — |
| **dangling commits** | **551** | `git fsck --no-reflogs` |
| ↳ deliberately-named dangling stashes (07-31) | 5 | dropped stash refs; all inspected |
| origin remote heads | 9 | `git ls-remote --heads origin`; 1 unmerged |

Note the 16-dirty figure: **`git status --porcelain` hides ignored files.** I re-walked all
52 with `--ignored=matching` (§6, miss-class 3).

## 2. Per-item disposition — the 6 unmerged refs

| ref | ahead | files / lines | tip | class | what it is | merge-safe? | disposition |
|---|---:|---|---|---|---|---|---|
| `ddm/sh1_integration_20260727` | **10** | 42 f / +7,916 | 07-27 | **MEASURED-RESULT** | The campaign's first end-to-end original-line row through full 600-sample `upstream/evaluate.py`: S=23.913488 (6.190 seg + 17.636 pose + 0.088 rate), archive 131,620 B sha `5e144118…`. Subsumes cb1+pf3b+wf7. | **NO — 1 hazard** | **HIGHEST-VALUE. Merge when unblocked.** |
| `codexwt/ddm_cb1_perclass_carrier_byteclose_…` | 1 | 17 f / +4,930 | 07-25 | BUILT-CODE | per-class carrier byte-close at n600; MyCar carrier ADMIT +319 B | subsumed | **contained in sh1** — merge sh1, not this |
| `codexwt/ddm_pf3b_52probe_joint_improving_hunt_…` | 2 | 24 f / +3,614 | 07-25 | BUILT-CODE | 52-probe joint-improving hunt; finds real joint gain, E4 rate-dominated | subsumed | **contained in sh1** |
| `codexwt/ddm_wf7_seven_home_stream_waterfill_…` | 2 | 9 f / +1,197 | 07-25 | BUILT-CODE | seven-home lossless rate waterfill, −1,776 B | subsumed | **contained in sh1** |
| `origin/codexwt/l7_default_failloud_budget_eventlaw_…` | 3 | 14 f / +745 | 07-15 | APPARATUS | l7 fail-loud opt-in + event-derived wall-clock budget | **NO** — touches `launch_witness_run.py`, `witness_autoconfig.py`; **2,329 behind** | **KEEP HELD** (already held since 07-17 prune) |
| `held/dsl_hash_enforcement_20260715` | 1 | 17 f / +2,131 | 07-15 | APPARATUS | DSL compile-hash enforced at launch admission | **NO** — `launch_witness_run.py`, `operator_authorize.py`; **2,392 behind** | **KEEP HELD** |

`merge-base --is-ancestor` confirms cb1/pf3b/wf7 are all ancestors of sh1. Merging sh1
lands all four; merging them individually is redundant work.

### sh1 merge hazard — measured, exactly one file

Intersecting sh1's 42 changed files against main's 144 dirty entries:

```
src/tac/optimization/direct_description_carrier_compose.py     <-- the only overlap
```

**Overlap count = 1.** This is precisely the blocker task #729 recorded, still live 5 days
later. sh1 is otherwise a clean 41-file merge. Charter-forbidden file; not touched.

## 3. THE FIND — einstein worktree's 2,485-line uncommitted delta (grade-5 orphan)

`.omx/tmp/codex_worktrees/einstein_kolmogorov_crux_20260719T212159Z`, branch
`codexwt/einstein_kolmogorov_crux_20260719T212159Z`. Branch tip `6a78ee8209` **is** an
ancestor of main — so the *committed* work landed and every branch-level tool reports it
merged. The signal is in the **uncommitted working-tree delta** that no branch points at:

```
12 files changed, 2485 insertions(+), 288 deletions(-)
  src/tac/preflight.py                                  1702 +++++++++---
  src/tac/canonical_equations/registry.py                144 +-
  test_check_344_anchor_roundtrip_scope_extension.py     355 +++-
  test_check_351_canonical_producer_identity_...py       236 ++-
  test_check_154_manifestless_cleanup_identity_...py     150 +-
  CLAUDE.md (+9) · docs/meta_bug_class_catalog.md (+10) · 3 research memos · uv.lock
```

**Decisive test — 27 new helper `def`s introduced by the delta, checked against main:**

```
PRESENT_ON_MAIN = 5      ABSENT_FROM_MAIN = 22
```

absent: `_check_351_ast_structure`, `_check_351_provenance_bindings`,
`_check_351_exact_sha_guard_pairs`, `_check_351_module_imports_canonical_builder`,
`_check_344_rebinds_empirical_anchor`, `_check_344_serialized_field_values`,
`_check_154_manifestless_branches`, `_check_154_suite_always_exits`, … (22 total).

This is the **AST-structural hardening** of Catalog #154/#344/#351 written during the
07-19/20 recursive adversarial review — the "2026-07-20 scope extension" that CLAUDE.md
on main *documents in prose*. It is the largest single unlanded artifact in the repo.

**Honest boundary (do not overstate this):** main **does** carry the base gates —
`check_351` 18 token matches, `check_154` 23, `check_344` 12, `SOURCE_` 25 in main's
`preflight.py`. So the finding is **name-level, not behaviour-level**: 22 named helpers are
absent, but I did **not** prove main lacks the same *guarantee* under different structure.
Classify as APPARATUS-REFINEMENT-OWED, not "main is unguarded."

**Merge-safe? NO.** A 1,702-line delta to `preflight.py` (the single hottest shared
surface, ~89.5k lines on main) computed against a **12-day-old base**. Do not merge.
Correct disposition: **re-derive the 22 helpers against today's `preflight.py`** as a
scoped landing with the two-landing rule, or file it as a typed debt row. Its 5 untracked
`einstein_kolmogorov_xi_bridge_*.json` blocker receipts should be read first — they may
record why it was never committed.

## 4. Already in main by content — do NOT re-merge (verified, not assumed)

| item | verification | verdict |
|---|---|---|
| `agent-a4ba…` pdw1 ×5 untracked `.py` | 4 of 5 **sha-identical** to main; `pdw1_fp32_realization_first_inbox_point.py` wt **539 L** vs main **1,389 L** | worktree holds the STALE earlier draft — committing would REGRESS. cs1 correct. |
| `agent-a1f08…` v10_ratecrush ×3 | `rank_donor_coders.py` sha-identical; `rank_streams.py` 240 vs 253; `v10_jxl_plane_codec.py` 286 vs **335** (main has the imagecodecs backend) | stale drafts. cs1 correct. |
| 10 receipt-dir worktrees (v4, v5, v7, v8, v13, v15, v18b, mdl, measurement_ladder, target_receipt) | **all 0 commits ahead of main**; findings memos + canonical timestamped receipt dirs present on main (e.g. `ddm_entropy_priced_member_…044916Z_artifacts/candidate_receipts/*.json`, 591 v18b receipt files) | untracked dirs are `_REDERIVE` / duplicate-timestamp re-runs. **Scratch. No signal owed.** |
| 5 named dangling stashes 07-31 (`fmt`, `v2`, `gh1-verify`, `tmp2`, `bp1tmp`) | **0 added symbols absent from main**; main strictly larger in every file (`confound_gates.py` 3,631 vs 3,405; `train_tr1_…mlx.py` 2,540 vs 2,533; `launch_tr1_run.py` 297 vs 297) | main **absorbed** them and continued past. **Superseded.** |
| `?? upstream` in `agent-a356e…`, `agent-a8874…` | `ls -ld` → **symlink** (`lrwxr-xr-x`), 16 entries | worktree plumbing, not a mutation of the pinned snapshot. No violation, no signal. |

## 5. Ranked — what MAIN should pull in

1. **`ddm/sh1_integration_20260727`** — the only branch carrying a *measured* end-to-end
   row plus 3 subsumed built-code branches (7,916 lines, 42 files). One-file blocker.
   Unblocking `direct_description_carrier_compose.py` is the single highest-leverage
   unblock available; it releases four branches at once.
2. **einstein uncommitted delta** — 22 absent AST-structural gate helpers (§3). Not a
   merge; a scoped **re-derivation** against today's `preflight.py`. Apparatus, not score.
3. **Nothing else.** Everything remaining is verified-superseded, scratch, or
   deliberately held old-lineage.

## 6. Adversarial completeness — what would this method MISS?

I asked the question and went looking. Four miss-classes found and closed, two residual.

- **MISS-1 dropped stashes / dangling commits.** `git branch --no-merged` and
  `git stash list` both show nothing. `git stash list` = 0 yet `git fsck` = **551 dangling
  commits**, including 5 deliberately-named 07-31 stashes. CLOSED — all 5 superseded (§4).
  *Residual:* I sampled only tips ≥ 07-25; the other ~546 (mostly `WIP on main:` auto-stash)
  were not individually opened.
- **MISS-2 server-only branches.** Local refs can lag the remote. `git ls-remote --heads
  origin` = 9; 8 merged, 1 unmerged and already known. CLOSED.
- **MISS-3 ignored-path receipts.** `status --porcelain` **hides ignored files** — a
  receipt under `.omx/state/` or `experiments/results/` is invisible. Re-walked all 52 with
  `--ignored=matching`: content is `__pycache__` (141/140 per tree), `.omx/state/*` locks +
  `review_tracker`, `.claude/settings.local.json`. The one substantive hit is einstein's 72
  entries = `candidate.pdw1p.bin` + `checkpoints/` — real byte-closed candidates, correctly
  gitignored as rebuildable bulk per the disk-hygiene rule. CLOSED (custody, not merge).
- **MISS-4 worktree nested inside a worktree.** `agent-a8874…/.omx/tmp/codex_worktrees/
  ddm_r7_token_coder_race_…` — caught only because I drove the loop off
  `worktree list --porcelain` rather than a directory glob. Clean + merged.
- **RESIDUAL-A: non-git signal.** A detached daemon writing to `experiments/results/` or an
  SSD tier leaves **no git trace at all** — no branch, no worktree, no dangling object.
  Nothing in this method can enumerate it. Bounding that needs a filesystem+ledger sweep
  (`.omx/state/` dispatch ledgers vs on-disk run dirs), not a git sweep.
- **RESIDUAL-B: branches deleted with expired reflog** are unrecoverable by construction.

## 7. What I could NOT determine

1. **Whether main's existing #154/#344/#351 already deliver the einstein guarantee.** I
   compared *symbol names*, not behaviour. "22 absent" is name-level evidence only.
2. **Whether the einstein delta passes tests on today's main.** Running its tests in its own
   worktree tests a 12-day-old base and would not answer the question; running them against
   main requires the re-derivation that is itself the recommended action. Not run.
3. **Main's own 144-file dirty WIP.** Charter-forbidden (live parallel session). Not
   audited — it is the largest single uncommitted pile in the repo and is outside this
   harvest's scope by construction.
4. **The ~546 unsampled dangling commits.**
5. **Why the einstein delta was never committed.** Its `*_blocker_20260719.json` receipts
   likely say; not opened in this pass.

## 8. Actions taken

**None mutating.** No merge, rebase, cherry-pick, push, branch delete, worktree removal, or
`checkout`. No scorer job, training, launch, or paid dispatch. The forbidden paths
(`direct_description_carrier_compose.py`, `ddm_qa43_two_plane_parallax_probe.py`,
`burn_out*`) were never opened for write. Pointer UNMOVED; no score claim.
