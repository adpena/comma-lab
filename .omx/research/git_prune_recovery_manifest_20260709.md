# Git prune recovery manifest — 2026-07-09

Before pruning stashes + stray branches (consolidation into main, pushed to origin/main 76401e87f),
ALL of them were archived into ONE verified git bundle. Nothing is lost — every stash/branch is
fully recoverable from the bundle. This manifest is the durable pointer (the 112MB bundle lives on SSD).

- **bundle (SSD cold-store):** `/Volumes/VertigoDataTier/pact/git_archive/pre_prune_stashes_branches_20260709.bundle`
- **sha256:** `244532a501da12f49065812aca89d3925c299a14b6a41d962c834e99228ebe45`
- **verify:** `git bundle verify <bundle>` → 'is okay', 23 refs
- **recover a ref:** `git fetch <bundle> archive/stash-<N>-20260709`  then `git stash apply FETCH_HEAD` (or checkout)

## Archived stashes (18)
- `archive/stash-0-20260709` = `13f793aceaabf4f40a17fc486881f8e5019d0538` — On main: preserve canonical upstream fallback runner WIP before saliency push 20260602
- `archive/stash-1-20260709` = `519df04bdad75e729a719afe01abb69ce224ab39` — On (no branch): preserve selector hi_nerv launch gate WIP during saliency rebase 20260602
- `archive/stash-2-20260709` = `ea88ea4004fca5dce6012aa2a66f3a3d097af098` — On main: preserve snerv WIP and pre-fix hinerv bad report before saliency rebase 20260602
- `archive/stash-3-20260709` = `1a5973da86999899a95f5c2344da24e00c089c24` — On main: preserve snerv trained ladder WIP before main fast-forward 20260602
- `archive/stash-4-20260709` = `8afeb2d562eefa3e887e762558657bfd5ada4257` — On main: preserve pre-ff WIP before hinerv saliency replay 20260602
- `archive/stash-5-20260709` = `bb6acadd9c20c9530f5f894879f853bd00bf8b92` — On main: autostash
- `archive/stash-6-20260709` = `1e4b152ebe0f7f2b7efa7800ac9edb28dfe69aaa` — On worktree-agent-a5fd9a3dd5d5b0a7c: pre-convergence-state-files-20260602 RECOVERABLE
- `archive/stash-7-20260709` = `aea615dc4edcc4688b870a336d00afa512ff7b62` — On lane-carrier-fits-on-converged-main-20260602: pre-main-checkout-stopped-agent-wip-20260602 RECOVERABLE
- `archive/stash-8-20260709` = `dc54ec273bea4569b400e323a9a0b247424589ee` — On lane-inverse-steganalysis-linf-vs-l2-gate-20260601: lane-wip-pre-resync-to-converged-main-20260602 RECOVERABLE: git stash apply
- `archive/stash-9-20260709` = `314f442f73022d3e0c72b94268c8bb0d22712d23` — WIP on main: a10f5e097 z7-mamba2: harden static control and remote custody
- `archive/stash-10-20260709` = `22f5f70e1c5534b2cea035844f160532fef93f57` — WIP on main: 67360c3a7 research: orphan-signal audit + 8 op-routables for score-lowering wire-ins (Catalog #247 reservation)
- `archive/stash-11-20260709` = `ccc49e7da96dd0335b4ab574427210d5cd78464f` — WIP on main: 1aaa258dc z3: implement _full_main per Phase 2 council approval (6/6 PROCEED unanimous)
- `archive/stash-12-20260709` = `c6acf03f5b47d85e164a7cc23489732eb0dfbccc` — WIP on main: 76872c1b src/tac/composition: substrate × primitive × order composition cell registry
- `archive/stash-13-20260709` = `6cead9cef7a754ac65c46789e2eff8a668903ad6` — WIP on main: 38b0c2f0 v: dispatch consolidation — 3 family CUDA + 2 paired CPU + D4 DALI probe
- `archive/stash-14-20260709` = `b2fef9d41a74a814044a80464a0978a6c9462df4` — On master: pre-integration signal preservation 20260506T215507Z
- `archive/stash-15-20260709` = `1d9e73294a72679dc1dbf4e18771504dbeabc7b1` — On main: pre-rigor-pass safety stash 20260504T223300Z
- `archive/stash-16-20260709` = `710bd3a23b73e9aa18983d0f3994ce00168fcf7d` — WIP on main: 4b32d89c 2 new STRICT preflight checks (82, 83) + CLAUDE.md FORBIDDEN extension — Round 3 council prescription
- `archive/stash-17-20260709` = `e8ca384e3fac569c4f06df74ba0de9a466e85047` — On master: yousfi_3_5_pending_greenup
- `archive/stash-18-20260709` = `c242e5dc5fbd1b2874051dd074fa74016e4db10d` — WIP on main: 7099f023 DEN-V2 partial: 4 layers of arch-drift fixed, pipeline downstream still AsymmetricPairGenerator-only

## Archived stray branches
- `safety/stash-recovered-20260505T052046Z-stash0` = `1d9e73294a72679dc1dbf4e18771504dbeabc7b1`
- `safety/stash-recovered-20260505T052046Z-stash1` = `710bd3a23b73e9aa18983d0f3994ce00168fcf7d`
- `safety/stash-recovered-20260505T052046Z-stash2` = `e8ca384e3fac569c4f06df74ba0de9a466e85047`
- `safety/stash-recovered-20260505T052046Z-stash3` = `c242e5dc5fbd1b2874051dd074fa74016e4db10d`
