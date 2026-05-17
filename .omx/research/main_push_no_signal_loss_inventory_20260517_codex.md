# Main Push No-Signal-Loss Inventory - 2026-05-17

## Scope

Operator request: commit and push all to `origin/main`, ensure `main` is the
source of truth, and avoid signal loss.

This ledger records the pre-push state that is intentionally discoverable from
`main`. It is not a result ledger and it does not promote any score claim.

## Current source-of-truth state

- Repository: `/Users/adpena/Projects/pact`
- Branch: `main`
- Remote: `origin git@github.com:adpena/comma-lab.git`
- Starting commit: `ea39614c29c91bcce49f2e82c656cbda3cbc7bf4`
- Starting decoration: `HEAD -> main, origin/main, origin/HEAD`
- `git rev-list --left-right --count main...origin/main`: `0 0`
- `git status --short --branch --untracked-files=all`: clean, `## main...origin/main`
- `git ls-files --others --exclude-standard`: no non-ignored untracked files
- `.omx/state/lane_registry.json`: JSON-valid

## Local-only signal inventory

The normal worktree has no uncommitted tracked or non-ignored untracked files.
The remaining local-only signal surfaces are historical stash and safety refs.
They are preserved as refs and indexed here so future agents can find them from
`main` without treating them as production code.

### Stashes

1. `stash@{2026-05-15 10:16:47 -0500}` - WIP on `main`: `67360c3a7 research: orphan-signal audit + 8 op-routables for score-lowering wire-ins (Catalog #247 reservation)`; 72 files, 2247 insertions, 164 deletions.
2. `stash@{2026-05-14 11:25:59 -0500}` - WIP on `main`: `1aaa258dc z3: implement _full_main per Phase 2 council approval (6/6 PROCEED unanimous)`; broad experiment/preflight touch set.
3. `stash@{2026-05-12 13:57:47 -0500}` - WIP on `main`: `76872c1b src/tac/composition: substrate x primitive x order composition cell registry`; 42 files, 2159 insertions, 234 deletions.
4. `stash@{2026-05-11 15:55:19 -0500}` - WIP on `main`: `38b0c2f0 v: dispatch consolidation - 3 family CUDA + 2 paired CPU + D4 DALI probe`; 30 files, 1515 insertions, 135 deletions.
5. `stash@{2026-05-06 16:55:07 -0500}` - pre-integration signal preservation `20260506T215507Z`; 19 files, 2046 insertions, 84 deletions.
6. `stash@{2026-05-04 17:33:00 -0500}` - `pre-rigor-pass safety stash 20260504T223300Z`; includes old `.omx/state` and provider/job state. A recovered branch also exists.
7. `stash@{2026-04-29 18:13:44 -0500}` - WIP on `main`: `45d808ae 2 new STRICT preflight checks (82, 83) + CLAUDE.md FORBIDDEN extension`; a recovered branch also exists.
8. `stash@{2026-04-26 10:17:03 -0500}` - `On master: yousfi_3_5_pending_greenup`; a recovered branch also exists.
9. `stash@{2026-04-26 07:26:12 -0500}` - WIP on `main`: `c5214993 DEN-V2 partial: 4 layers of arch-drift fixed`; a recovered branch also exists.

### Branch refs

Unmerged historical safety refs:

- `safety/stash-recovered-20260505T052046Z-stash0`
- `safety/stash-recovered-20260505T052046Z-stash1`
- `safety/stash-recovered-20260505T052046Z-stash2`
- `safety/stash-recovered-20260505T052046Z-stash3`

Merged refs:

- `safety/snapshot-20260504T223259Z-pre-rigor-pass`
- `safety/snapshot-pre-filter-repo-20260505T144000Z`
- `worktree-agent-a76feb23917b9bd07`

The `worktree-agent-a76feb23917b9bd07` branch has no commits ahead of `main`
(`git log main..worktree-agent-a76feb23917b9bd07` returned empty) and is listed
as merged into `main`; its worktree remains locked by the agent metadata.

## Merge decision

No stale stash or stash-recovery branch was auto-applied or auto-merged during
this push. These refs are broad historical WIP snapshots, several include raw
`.omx/state` provider/job state, and blindly merging them would risk replacing
current production code with obsolete scaffolding. The no-signal-loss action is
to keep the refs intact, index them from this tracked ledger, and require future
salvage to cherry-pick or reimplement specific live signal deliberately.

## Reproduction commands

```bash
git status --short --branch --untracked-files=all
git rev-list --left-right --count main...origin/main
git branch --no-merged main
git branch --merged main
git stash list --date=iso
git stash show --stat 'stash@{0}'
git worktree list --porcelain
python3 -m json.tool .omx/state/lane_registry.json
```
