# Main Source Of Truth Safety-Stash Audit - 2026-05-17

## Scope

Operator asked to commit and push all current work to `origin/main` with no
signal loss. Current working branch was `main` and the worktree was clean before
this audit.

This audit records the only local refs not merged into `main`: recovered safety
stash branches from the 2026-05-05 recovery sweep. They are preserved locally
and on `origin`; they are not promoted as source-of-truth code by blind merge.

## Current main

- Local `main`: `bbf14fdfebe5408b66eda6c11b92c639331d6ea9`
- Remote `origin/main`: `bbf14fdfebe5408b66eda6c11b92c639331d6ea9`
- Remote: `git@github.com:adpena/comma-lab.git`
- Pre-audit worktree: clean

## Preserved quarantine refs

| Branch | Head | Commit date | Diff size versus current main | Disposition |
| --- | --- | --- | --- | --- |
| `safety/stash-recovered-20260505T052046Z-stash0` | `1d9e73294a72679dc1dbf4e18771504dbeabc7b1` | `2026-05-04T17:33:00-05:00` | 121 files, 35,373 insertions, 26,543 deletions | Preserved on `origin`; not blind-merged because it carries broad stale state, submission archive, runtime, docs, and provider-history changes that predate current L5 control-plane state. |
| `safety/stash-recovered-20260505T052046Z-stash1` | `710bd3a23b73e9aa18983d0f3994ce00168fcf7d` | `2026-04-29T18:13:44-05:00` | 6 files, 335 insertions, 59 deletions | Preserved on `origin`; candidate for targeted forensic intake if old STC/AV1 or multipass optimizer signal is needed. |
| `safety/stash-recovered-20260505T052046Z-stash2` | `e8ca384e3fac569c4f06df74ba0de9a466e85047` | `2026-04-26T10:17:03-05:00` | 19 files, 3,021 insertions, 552 deletions | Preserved on `origin`; candidate for targeted forensic intake if old Yousfi/Fridrich correction signal is needed. |
| `safety/stash-recovered-20260505T052046Z-stash3` | `c242e5dc5fbd1b2874051dd074fa74016e4db10d` | `2026-04-26T07:26:12-05:00` | 3 files, 371 insertions, 314 deletions | Preserved on `origin`; candidate for targeted forensic intake if old DEN-V2 arch-drift signal is needed. |

## No-signal-loss conclusion

The recovered stash branches are not lost: each has a local branch and matching
remote branch under `origin/safety/...`. The exact commit IDs above are enough
to recover every byte, diff, and file if a future targeted intake is warranted.

They were not merged directly into `main` because AGENTS.md treats recovered or
quarantined trees as forensic inputs until explicitly reviewed. A blind merge
would risk overwriting current contest custody, L5/L5 v2 control-plane state,
and newer production hardening with stale state from April/early May.

## Recommended future intake path

If any safety branch becomes relevant, do a targeted cherry-pick or port from
the named commit into a new `main` commit with:

1. current-file diff review,
2. lane/provenance ledger entry,
3. focused tests,
4. exact axis labels for any score artifacts,
5. no mutation of stale `.omx/state` or submission archives unless explicitly
   promoted by a current evidence packet.
