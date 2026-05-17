# Main source-of-truth push audit - 2026-05-17

## Current verification addendum - post L5-v2 freshness landing

This addendum refreshes the source-of-truth custody after the later L5-v2
architecture-lock freshness landing.

- Local branch: `main`
- Local HEAD before this addendum: `e554c498e2c7c3c2af09bf534e84926a330a7696`
- Local `origin/main` before this addendum:
  `e554c498e2c7c3c2af09bf534e84926a330a7696`
- Remote `refs/heads/main` before this addendum:
  `e554c498e2c7c3c2af09bf534e84926a330a7696`
- Worktree status before this addendum: clean
- Network refresh command: `git fetch --prune origin main`
- Remote URL: `git@github.com:adpena/comma-lab.git`

Current local branches not merged to `main` are historical safety refs only:

| Branch | Status |
| --- | --- |
| `safety/stash-recovered-20260505T052046Z-stash0` | Preserved historical WIP; do not blindly merge. |
| `safety/stash-recovered-20260505T052046Z-stash1` | Preserved historical WIP; do not blindly merge. |
| `safety/stash-recovered-20260505T052046Z-stash2` | Preserved historical WIP; do not blindly merge. |
| `safety/stash-recovered-20260505T052046Z-stash3` | Preserved historical WIP; do not blindly merge. |

These refs and the stash stack remain preserved signal, not production source
of truth. Any future promotion from them must happen through an isolated
worktree, focused patch extraction, tests, review-tracker evidence, and a dated
research ledger. They are intentionally not bulk-merged because doing so would
mix old recovery/WIP surfaces into the current contest-custody tree without
review.

## Verdict

`main` is the production source of truth and is pushed to `origin/main`.

- Local branch: `main`
- Local HEAD: `0ef3b72e46ab8c80ae2fa4849ca409db79df4085`
- Local `origin/main`: `0ef3b72e46ab8c80ae2fa4849ca409db79df4085`
- Remote `refs/heads/main`: `0ef3b72e46ab8c80ae2fa4849ca409db79df4085`
- Push result before this audit landed: `Everything up-to-date`
- Worktree status before this audit landed: clean

## No-signal-loss stash inventory

Local stashes exist and are treated as preserved historical WIP, not as
production code. They were not replayed into `main` during this audit because
several entries are older than current HEAD, span broad state surfaces, or touch
raw/transient `.omx/state` and provider artifacts. Blindly applying them would
risk corrupting the current source-of-truth state instead of producing a safe
merge.

Any future promotion from these stashes should happen in an isolated worktree,
extracting focused patches into `main` with tests, review-tracker evidence, and
dated `.omx/research` ledgers.

| Stash | Commit | Date | Subject | Stat summary |
| --- | --- | --- | --- | --- |
| `stash@{0}` | `22f5f70e1c5534b2cea035844f160532fef93f57` | 2026-05-15 10:16:47 -0500 | orphan-signal audit + op-routables | 72 files, 2247 insertions, 164 deletions |
| `stash@{1}` | `ccc49e7da96dd0335b4ab574427210d5cd78464f` | 2026-05-14 11:25:59 -0500 | Z3 `_full_main` Phase 2 WIP | 3315 files, 8298 insertions, 867 deletions |
| `stash@{2}` | `c6acf03f5b47d85e164a7cc23489732eb0dfbccc` | 2026-05-12 13:57:47 -0500 | substrate composition registry | 42 files, 2159 insertions, 234 deletions |
| `stash@{3}` | `6cead9cef7a754ac65c46789e2eff8a668903ad6` | 2026-05-11 15:55:19 -0500 | dispatch consolidation | 30 files, 1515 insertions, 135 deletions |
| `stash@{4}` | `b2fef9d41a74a814044a80464a0978a6c9462df4` | 2026-05-06 16:55:07 -0500 | pre-integration signal preservation | 19 files, 2046 insertions, 84 deletions |
| `stash@{5}` | `1d9e73294a72679dc1dbf4e18771504dbeabc7b1` | 2026-05-04 17:33:00 -0500 | pre-rigor-pass safety stash | 121 files, 35373 insertions, 26543 deletions |
| `stash@{6}` | `710bd3a23b73e9aa18983d0f3994ce00168fcf7d` | 2026-04-29 18:13:44 -0500 | strict preflight checks 82/83 WIP | 6 files, 335 insertions, 59 deletions |
| `stash@{7}` | `e8ca384e3fac569c4f06df74ba0de9a466e85047` | 2026-04-26 10:17:03 -0500 | `yousfi_3_5_pending_greenup` | 19 files, 3021 insertions, 552 deletions |
| `stash@{8}` | `c242e5dc5fbd1b2874051dd074fa74016e4db10d` | 2026-04-26 07:26:12 -0500 | DEN-V2 partial | 3 files, 371 insertions, 314 deletions |

## Recovery commands

Inspect without mutating the shared worktree:

```bash
git stash show --stat stash@{N}
git stash show --patch stash@{N} -- path/to/file
git worktree add ../pact-stash-N main
cd ../pact-stash-N
git stash apply --index stash@{N}
```

If a stash contains durable work, promote only the focused, current-compatible
slice back to `/Users/adpena/Projects/pact` on `main`.
